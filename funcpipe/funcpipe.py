"""
FuncPipe: pipeline parallelism in serverless environment
"""
from typing import Dict, Deque, Union, Tuple, List
import time
import numpy as np
import gc

import torch
from torch import Tensor, nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from funcpipe.planner import Planner, PipelineArch, Profiler
from funcpipe.scheduler import Scheduler, Task, TaskManager, ScheduleInfo
from funcpipe.comm import Communicator, Upload_Communicator, Download_Communicator
from funcpipe.func_manager import FunctionManager
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline
from funcpipe.monitor import Monitor
import funcpipe.microbatch as Microbatch
from funcpipe.platforms import Platform

class FuncPipe:#(nn.Module):

    def __init__(self, model, batch_size = 256, loss_func = F.cross_entropy, optim_class = optim.SGD, learning_rate = 0.1):
        #super().__init__()
        # pipeline info
        self.model = model
        # todo: skip connection
        #self.my_partition: List[nn.Sequential] = None #using list to deal with skip connections
        self.my_partition: nn.Sequential = None
        self.learning_rate = learning_rate
        self.optimizer_class = optim_class
        self.loss_func = loss_func
        self.optimizer = None
        self.pipeline_arch: PipelineArch = None
        self.batch_size = batch_size
        self.micro_batchsize= -1
        self.skip_links = []

        # distributed training
        self.is_init_worker = False
        self.my_rank = -1
        self.my_stage_id = -1
        self.my_tensor_id = -1
        self.my_data_id = -1

        # model partition
        self.input_grads: Dict = {} # recved gradients in BP
        self.input_activations: Dict[int,Union[Variable, Tuple[Variable, ...]]] = {} # grad should be computed for this input (zero)
        self.output_tensors: Dict[int, Union[Tensor, Tuple[Tensor, ...]]] = {} # output in FWD. Tuple[0]: normal output Tuple[1]: skip connection
        #todo: skip connection
        #self.output_tensors: Dict[int, Union[Tensor, List[Tensor]]] = {}

        Logger.debug("Initing modules ...")
        # key modules
        self.planner = Planner()
        self.profiler = Profiler()
        self.scheduler = Scheduler()
        self.task_manager = TaskManager()
        # self.communicator = Communicator(self)
        # self.communicator.start()
        self.upload_comm = Upload_Communicator(self)
        self.download_comm = Download_Communicator(self)
        self.upload_comm.start()
        self.download_comm.start()
        self.monitor = Monitor()
        self.monitor.start()
        self.func_manager = FunctionManager()

    def set_skip_links(self, skip_links: List):
        ''' Specifies the skip connections within the model
        Params:: skip_links [(layer_id, output_id, skip_to_layer_id)]
        '''
        self.skip_links = skip_links

    def init(self, launch_info, profiling_sample = None):
        """Initialize the whole pipeline
        """
        Logger.debug("Initing pipeline ...")
        self.my_rank = int(launch_info["rank"])
        if "is_init_worker" in launch_info.keys() and int(launch_info["is_init_worker"]) > 0:
            self.is_init_worker = True
        assert  not self.is_init_worker or self.is_init_worker and self.my_rank == 0

        # model profiling is done only on the first rank
        # will be later sent to other ranks
        # todo: profiler now is running as a independent component, wiil be integrated later
        if self.my_rank == -1:
            model_info = self.profiler.profile(self.model, profiling_sample)
            launch_info["model_info"] = model_info
        else:
            model_info = None#launch_info["model_info"]

        #'''Pipeline design'''
        # todo: consider saving the planning for ranks other than rank0, if the planning is of high complexity
        self.my_partition, self.pipeline_arch, self.micro_batchsize = self.planner.plan(self.model, model_info, self.my_rank, self.skip_links)
        self.optimizer = self.optimizer_class(self.my_partition.parameters(), lr = self.learning_rate) #optim.SGD(self.my_partition.parameters(), lr=0.1)
        self.my_stage_id, self.my_tensor_id, self.my_data_id = self.pipeline_arch.rank_id_to_arch_ids(self.my_rank)
        Logger.debug("Pipeline arch: " + str(self.pipeline_arch))
        Logger.debug("my: stage_id %d   tensor_id %d   data_id %d" % (self.my_stage_id, self.my_tensor_id, self.my_data_id))

        #'''Launch other nodes'''
        if Platform.platform_type == 'local':
            if self.my_rank == 0:
                self.func_manager.start_pipeline(self.pipeline_arch, launch_info)
        else:
            if self.my_rank == 0:
                if self.is_init_worker:
                    self.func_manager.start_pipeline(self.pipeline_arch, launch_info)
                    Logger.debug("Init done, init worker exiting ..." + str(self.pipeline_arch))
                    self.end()
                    exit()

        # only keeps one partition
        del self.model
        gc.collect()

    def pipeline_train(self, inputs: Union[Tensor, Tuple[Tensor, ...]], targets: Tensor):
        Timeline.start('iter')
        #'''Micro batch partition'''
        micro_batches, micro_targets = Microbatch.split(inputs, targets, self.batch_size, self.micro_batchsize)

        # '''Pipeline schedule'''
        schedule_info = self.fill_schedule_info()
        pipeline_task_queue: Deque[Task] = self.scheduler.schedule(schedule_info)
        Logger.debug("%d tasks scheduled in total." % len(pipeline_task_queue))

        # '''Pipeline run'''
        self.my_partition.train()
        while len(pipeline_task_queue) > 0:
            task_to_run = pipeline_task_queue.popleft()
            Logger.debug(str(task_to_run))

            # check dependency
            while not self.task_manager.check_dependency(task_to_run.dependencies):
                # Logger.debug("Wait on dependency ...")
                time.sleep(0.001) # sleep for 1 ms

            # execute tasks
            micro_batch_id = task_to_run.micro_batch_id
            if task_to_run.is_comp_task():
                if task_to_run.type == Task.FORWARD:
                    Timeline.start("forward")
                    if self.my_stage_id == 0:
                        micro_batch = micro_batches[micro_batch_id]
                    else:
                        micro_batch = self.input_activations[micro_batch_id]
                    self.forward(micro_batch, micro_batch_id)
                    Timeline.end("forward")
                elif task_to_run.type == Task.BACKWARD:
                    Timeline.start("backward")
                    if self.my_stage_id == self.pipeline_arch.get_stage_n() - 1:
                        #Logger.debug(str(self.output_tensors[micro_batch_id]))
                        #Logger.debug(str(micro_targets[micro_batch_id]))
                        loss = self.loss_func(self.output_tensors[micro_batch_id], micro_targets[micro_batch_id])
                        loss.backward()
                    else:
                        bp_gradient = self.input_grads[micro_batch_id]
                        self.backward(bp_gradient, micro_batch_id)
                    Timeline.end("backward")
                elif task_to_run.type == Task.WEIGHT_UPDATE:
                    Timeline.start("weightUpdate")
                    self.weight_update()
                    Timeline.end("weightUpdate")
                # inform manager of the finished task
                # for communication task, this operation is done by the communicator
                # executed asynchronously
                self.task_manager.add_executed(task_to_run.id)

            # elif task_to_run.is_comm_task():
            #     # enqueue the comm task
            #     #self.communicator.enqueue(task_to_run)
            #     self.communicator.execute(task_to_run)
            elif task_to_run.is_upload_task():
                self.upload_comm.enqueue(task_to_run)
            elif task_to_run.is_download_task():
                self.download_comm.enqueue(task_to_run)


        # need to sync with coomunicator here?
        self.upload_comm.join()
        self.download_comm.join()
        # clear manager for next batch
        self.task_manager.clear()
        # delete tensors
        for key in list(self.input_grads):
            del self.input_grads[key]
        for key in list(self.input_activations):
            del self.input_activations[key]
        for key in list(self.output_tensors):
            del self.output_tensors[key]

        gc.collect()
        Timeline.end('iter')

    def forward(self, input_activation: Union[Tensor, Tuple[Tensor, ...]], batch_id: int):
        Monitor.print_my_processs_mem("Before fwd")
        self.output_tensors[batch_id] = self.my_partition(input_activation)
        # Logger.debug(str(type(self.output_tensors[batch_id])))
        Monitor.print_my_processs_mem("After fwd")

    def backward(self, input_gradient: Union[Tensor, Tuple[Tensor, ...]], batch_id: int):
        Monitor.print_my_processs_mem("Before bp")
        if isinstance(input_gradient, tuple):
            output = self.output_tensors[batch_id] # a tuple of tensors
            #Logger.debug(str(input_gradient))

            '''
            for i, tensor in enumerate(output):
                if i >= len(input_gradient) or input_gradient[i] is None: continue
                retain = False
                if i != (len(input_gradient) - 1): retain = True
                #Logger.debug("backward:" + str(i))
                Timeline.start("backward tuple {}".format(i))
                tensor.backward(input_gradient[i], retain_graph = retain)
                Timeline.end("backward tuple {}".format(i))
            '''
            pass
            # version two
            #tensor_for_grad = []
            output_tensors = []
            grad_recv = []
            #if isinstance(self.input_activations[batch_id], tuple):
            #    for i, tensor in enumerate(self.input_activations[batch_id]): tensor_for_grad.append(tensor)
            for i, tensor in enumerate(output):
                if i >= len(input_gradient) or input_gradient[i] is None: continue
                output_tensors.append(tensor)
                grad_recv.append(input_gradient[i])
            torch.autograd.backward(output_tensors, grad_recv)
        else:
            self.output_tensors[batch_id].backward(input_gradient)
        Monitor.print_my_processs_mem("After bp")

    def weight_update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # this call cleans the background thread so that the program can exit
    def end(self):
        self.upload_comm.stop()
        self.download_comm.stop()
        self.monitor.stop()
        Logger.debug("Training done.")
        Timeline.report()
        self.monitor.report()
        Logger.finalize()

    def evaluate(self, inputs, targets):
        pass

    def fill_schedule_info(self):
        return ScheduleInfo(pipeline_arch = self.pipeline_arch, my_stage_id = self.my_stage_id, my_tensor_id = self.my_tensor_id,
                            my_data_id = self.my_data_id, my_rank = self.my_rank, microbatch_num = int(self.batch_size / self.micro_batchsize),
                            skip_links = self.skip_links)