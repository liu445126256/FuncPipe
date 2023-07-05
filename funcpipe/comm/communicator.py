'''Communicator takes charge of the communication tasks'''
from abc import ABCMeta, abstractmethod
from collections import deque
from typing import Dict, List, Deque
import threading
import time
import numpy as np
import gc

import torch
from torch import nn, Tensor
from torch.autograd import Variable

from funcpipe.comm.collectives import P2P, One2Many, AllReduce
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline
from funcpipe.scheduler import Task
from funcpipe.monitor import Monitor
from funcpipe.platforms import Platform

class Communicator:
    __metaclass__ = ABCMeta

    def __init__(self, func_pipe):
        self.pipeline = func_pipe
        self.task_queue: Deque = deque()
        self.comm_thread = None
        self.to_stop = False
        self.in_execution = False

    def start(self):
        self.comm_thread = threading.Thread(target=self.run)
        self.comm_thread.start()
        Logger.debug("Communicator thread started.")

    def run(self):
        while True:
            if len(self.task_queue) == 0:
                self.in_execution = False
                time.sleep(0.001)
                if self.to_stop: break # exit only when no task in queue
                continue
            self.in_execution = True
            task = self.task_queue.popleft()
            self.execute(task)

    def stop(self):
        self.to_stop = True

    def join(self):
        while self.in_execution or len(self.task_queue) > 0:
            time.sleep(0.01)

    def enqueue(self, task: Task):
        self.task_queue.append(task)

    @abstractmethod
    def execute(self, task: Task):
        pass

    # todo: the decision of the comm group might be better to move to the scheduler
    def get_comm_group(self, partition_id, comm_type, pipeline_arch: Dict) -> List[int]:
        '''
        :return: a list of rank_ids that participate in the communication
        '''
        rank_ids = []

        return rank_ids


class Upload_Communicator(Communicator):

    def execute(self, task: Task):
        '''Perform a communication task'''
        comm_type = task.type
        comm_rank_ids = task.comm_task_info
        micro_batch_id = task.micro_batch_id

        if comm_type == Task.SEND_OUTPUT or comm_type == Task.SEND_GRAD:
            if comm_type == Task.SEND_OUTPUT:
                Timeline.start("send activation")
                Monitor.print_my_processs_mem("Before sending act")
                if isinstance(self.pipeline.output_tensors[micro_batch_id], tuple):
                    data_to_send = []
                    for tensor in self.pipeline.output_tensors[micro_batch_id]:
                        data_to_send.append(tensor.data.numpy())
                else:
                    data_to_send = self.pipeline.output_tensors[micro_batch_id].data.numpy()
                Monitor.print_my_processs_mem("sending act data prepared")
            else:
                Timeline.start("send gradient")
                if isinstance(self.pipeline.input_activations[micro_batch_id], tuple):
                    data_to_send = []
                    for i, tensor in enumerate(self.pipeline.input_activations[micro_batch_id]):
                        try:
                            data_to_send.append(tensor.grad.data.numpy())
                        except:
                            Logger.info("id:{}".format(i))# there may be some inputs that doe not involve gradient computation, like masks
                            data_to_send.append(None)
                else:
                    data_to_send = self.pipeline.input_activations[micro_batch_id].grad.data.numpy()
                #Logger.info(str(data_to_send))
            src_rank = comm_rank_ids[0]
            dst_ranks = comm_rank_ids[1]
            assert src_rank == self.pipeline.my_rank
            Monitor.print_my_processs_mem("Before p2p")
            if len(dst_ranks) > 1:
                One2Many.send(src_rank, dst_ranks, data_to_send, str(micro_batch_id))
            else:
                P2P.send(src_rank, dst_ranks[0], data_to_send, str(micro_batch_id))
            gc.collect()
            Monitor.print_my_processs_mem("After p2p")
            # end timeline
            if comm_type == Task.SEND_OUTPUT: Timeline.end("send activation")
            else: Timeline.end("send gradient")


        elif comm_type == Task.SYNC_WEIGHTS:
            Timeline.start("sync weights")
            Monitor.print_my_processs_mem("sync weights start")
            parameter_shape = []
            parameter_length = []
            # serialize
            Timeline.start("serialization")
            # todo: a BUG happens here! when some of the nodes are not assigned any micro batch
            data_type = None
            for param in self.pipeline.my_partition.parameters():
                grad = param.grad
                if grad is None: continue
                param_np = grad.data.numpy()
                parameter_shape.append(param_np.shape)
                tmp_length = 1
                for s in param.grad.data.numpy().shape:
                    tmp_length *= s
                parameter_length.append(tmp_length)
                data_type = param_np.dtype
            Logger.debug("allreduce size:" + str(sum(parameter_length)))
            param_w = np.zeros((sum(parameter_length)), dtype=data_type)
            Monitor.print_my_processs_mem("data room allocated")
            offset = 0
            for i, param in enumerate(self.pipeline.my_partition.parameters()):
                grad = param.grad
                if grad is None: continue
                param_np = param.grad.data.numpy().flatten()
                if i == 0: Logger.debug("param type:{}".format(param_np.dtype))
                param_w[offset: offset + parameter_length[i]] = param_np
                offset += parameter_length[i]
            Timeline.end("serialization")
            # comm
            gc.collect()
            Logger.debug("serialized type:{}".format(param_w.dtype))
            Monitor.print_my_processs_mem("data room1")
            data_to_send = param_w
            Monitor.print_my_processs_mem("data room2")

            # get stage id
            stage_id = self.pipeline.pipeline_arch.rank_id_to_arch_ids(self.pipeline.my_rank)[0]
            sync_bucket_name = "funcpipe-stage{}".format(stage_id)
            Logger.debug("Using bucket {}".format(sync_bucket_name))
            Platform.set_bucket_name(sync_bucket_name)
            reduced_data = AllReduce.latency_opt(self.pipeline.my_rank, comm_rank_ids, data_to_send)
            #reduced_data = AllReduce.bandwidth_opt(self.pipeline.my_rank, comm_rank_ids, data_to_send)
            Platform.set_bucket_name() #reset bk name
            # deserialize
            pos = 0
            for layer_index, param in enumerate(self.pipeline.my_partition.parameters()):
                grad = param.grad
                if grad is None: continue
                param.grad.data = Variable(torch.from_numpy(
                    np.asarray(reduced_data[pos:pos + parameter_length[layer_index]], dtype=np.float32)
                        .reshape(parameter_shape[layer_index])))
                pos += parameter_length[layer_index]
            Timeline.end("sync weights")
            Monitor.print_my_processs_mem("sync weights finished")
        # inform task manager of the pipeline of the finished task
        self.pipeline.task_manager.add_executed(task.id)


class Download_Communicator(Communicator):

    def execute(self, task: Task):
        '''Perform a communication task'''
        comm_type = task.type
        comm_rank_ids = task.comm_task_info
        micro_batch_id = task.micro_batch_id

        if comm_type == Task.RECV_GRAD or comm_type == Task.RECV_INPUT:
            if comm_type == Task.RECV_GRAD: Timeline.start("recv gradient")
            else: Timeline.start("recv input")
            src_ranks = comm_rank_ids[0]
            dst_rank = comm_rank_ids[1]
            assert dst_rank == self.pipeline.my_rank
            if len(src_ranks) > 1:
                recv_data = One2Many.recv(dst_rank, src_ranks, str(micro_batch_id))
            else:
                recv_data = P2P.recv(dst_rank, src_ranks[0], str(micro_batch_id))
            if comm_type == Task.RECV_GRAD:
                Timeline.end("recv gradient")
                if isinstance(recv_data, list):
                    grad_list = []
                    for data in recv_data:
                        if data is not None: grad_list.append(torch.from_numpy(data))
                        else: grad_list.append(None)
                    self.pipeline.input_grads[micro_batch_id] = tuple(grad_list)
                    #Logger.debug(":::::::" + str(recv_data))
                else:
                    self.pipeline.input_grads[micro_batch_id] = torch.from_numpy(recv_data)
            else:
                Timeline.end("recv input")
                if isinstance(recv_data, list):
                    inputs = []
                    for i, data in enumerate(recv_data):
                        req_grad = False
                        if data.dtype in [np.float, np.float32, np.float64, np.float16]: req_grad = True
                        inputs.append(Variable(torch.from_numpy(data),requires_grad=req_grad))
                        #Logger.info("id:{} req_grad:{} data.dtype:{}".format(i, req_grad, data.dtype))
                    self.pipeline.input_activations[micro_batch_id] = tuple(inputs)
                else:
                    self.pipeline.input_activations[micro_batch_id] = Variable(torch.from_numpy(recv_data),
                                                                                requires_grad=True)

        # inform task manager of the pipeline of the finished task
        self.pipeline.task_manager.add_executed(task.id)