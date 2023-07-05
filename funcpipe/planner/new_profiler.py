'''The profiler requires a function uploaded to the cloud'''
# todo: to be automated

from typing import Dict, List, Tuple, Union
import time
import numpy as np
import pickle

import torch
from torch import nn
from torch.autograd import Variable

from funcpipe.debugger import Logger
from funcpipe.utils import get_mem_usage, linefit, get_random_seq
from funcpipe.platforms import Platform
from funcpipe.configs import Config

class Profiler:
    #Profiles the detailed information of each layer
    TIMEOUT = 30 # OOM happens if time out
    ELEMENT_BYTES = 4 #using float32 by default

    def __init__(self):
        self.prof_seq = None

    def profile(self, model: nn.Sequential, resource_option: List, input_sample=None, batch_size=1) -> Dict:
        Logger.info("Model profiling started.")
        if input_sample is None:
            x = torch.rand(1, 3, 224, 224)  # use a synthetic img as default
        else:
            x = input_sample[:1]

        # random task sequence number
        task_seq = get_random_seq(10)
        layer_id = 0
        model_files = []
        for n, l in model.named_children():
            if isinstance(x, tuple):
                input_shape = []
                for i, tensor in enumerate(x): input_shape.append(
                    (tensor.data.numpy().shape, tensor.data.numpy().dtype))
            else:
                input_shape = (x.data.numpy().shape, x.data.numpy().dtype)
            layer_entry = [n, l, input_shape]

            # forward one batch to get the input size of each layer
            with torch.no_grad():
                x = l(x)

            if isinstance(x, tuple):
                inputs = []
                for i, tensor in enumerate(x):
                    data = tensor.data.numpy()
                    inputs.append(Variable(torch.from_numpy(data)))
                x = tuple(inputs)
            else:
                data = x.data.numpy()
                x = Variable(torch.from_numpy(data))

            layer_data = pickle.dumps(layer_entry)
            file_name = "{}_l{}".format(task_seq, layer_id)
            model_files.append(file_name)
            layer_id += 1
            Logger.info('uploading layer {}'.format(layer_id))
            Platform.upload_to_storage(file_name, layer_data)

        # launch a sequence of functions for profiling
        # invoke profiler function
        profiler_info = Platform.get_profiler_info()
        layer_num = len(model)
        launch_info = {}
        launch_info["service_name"] = profiler_info["service_name"]
        launch_info["seq"] = task_seq
        launch_info["layer_num"] = layer_num
        launch_info["layer_id"] = 0  # starting from layer 0
        launch_info["batch_size"] = batch_size
        launch_info["profile_round"] = Config.getvalue("common", "profile_round")
        launch_info["profile_start_round"] = Config.getvalue("common", "profile_start_round")
        for mem in resource_option:
            launch_info["function_name"] = profiler_info["function_name"].format(mem)
            launch_info["memory"] = mem
            Logger.info("Invoking {}".format(launch_info["function_name"]))
            Platform.invoke(launch_info, asynchronous=True)

        # get results
        prof_result = {}
        result_files = []
        for mem in resource_option:
            model_info = {}
            for layer_id in range(layer_num):
                file_name = "{}_res_{}_l{}".format(task_seq, mem, layer_id)
                result_files.append(file_name)
                data = Platform.download_from_storage(file_name, timeout=self.TIMEOUT)
                if data is None:  # OOM happened, relaunch function for next layer
                    Logger.info("{} OOM happend, layer cannot fit into the function.".format(file_name))
                    layer_info = [0, mem, 0, 0, 0, 0]
                    launch_info["layer_id"] = layer_id + 1
                    launch_info["function_name"] = profiler_info["function_name"].format(mem)
                    launch_info["memory"] = mem
                    Platform.invoke(launch_info, asynchronous=True)
                else:
                    layer_info = pickle.loads(data)
                model_info[layer_id] = layer_info
            prof_result[mem] = model_info

        # clean intermediate data
        for file_name in model_files: Platform.delete_from_storage(file_name)
        for file_name in result_files: Platform.delete_from_storage(file_name)

        return prof_result

    def profile_layer(self, layer_data: List, batch_size=1) -> List:
        '''
        :param layer_data: [layer_name, layer, input_info] input_info: List[tuple] or tuple: (shape, dtype)
        :param batch_size:
        :return: [layer_size(MB), activation_size, fwd_time(ms), output_size, bp_time(ms), grad_size]
        '''
        # layer_name = layer_data[0]
        layer = layer_data[1]
        input_info = layer_data[2]
        result = []

        # get layer size
        layer_size = 0
        for params in layer.parameters():
            tmp_size = 1
            for s in params.data.numpy().shape:
                tmp_size *= s
            layer_size += tmp_size
        result.append(layer_size * self.ELEMENT_BYTES / 1024 / 1024)  # todo: distinguish data type - we use 4 bytes for now

        # get activation size and fwd time
        if isinstance(input_info, list):
            inputs = []
            for i, entry in enumerate(input_info):
                shape = list(entry[0])
                shape[0] = batch_size
                data_type = entry[1]
                data = np.ones(shape, dtype=data_type)
                req_grad = False
                if data.dtype in [np.float, np.float32, np.float64, np.float16]: req_grad = True
                inputs.append(Variable(torch.from_numpy(data), requires_grad=req_grad))
            x_input = tuple(inputs)
        else:
            shape = list(input_info[0])
            shape[0] = batch_size
            data_type = input_info[1]
            data = np.ones(shape, dtype=data_type)
            req_grad = False
            if data.dtype in [np.float, np.float32, np.float64, np.float16]: req_grad = True
            x_input = Variable(torch.from_numpy(data), requires_grad=req_grad)
        mem_usage = get_mem_usage()
        '''
        with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as prof:
            x = layer(x_input)
        computation_time = prof.self_cpu_time_total / 1000.0
        '''
        start_t = time.time()
        x = layer(x_input)
        end_t = time.time()
        computation_time = (end_t - start_t) * 1000.0
        mem_cost = get_mem_usage() - mem_usage
        result.append(mem_cost)
        result.append(computation_time)

        # output size
        output_size = 0
        if isinstance(x, tuple):
            for t in x:
                tmp_size = 1
                for s in t.shape:
                    tmp_size *= s
                output_size += tmp_size
        else:
            output_size = 1
            for s in x.shape:
                output_size *= s
        result.append(output_size * self.ELEMENT_BYTES / 1024 / 1024)

        # bp
        if isinstance(x, tuple):
            '''
            with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as prof:
                for i, l in enumerate(x):
                    if l is not None:
                        gradient = torch.ones_like(l)
                        l.backward(gradient)
            bp_time = prof.self_cpu_time_total / 1000.0
            '''
            start_t = time.time()
            for i, l in enumerate(x):
                if l is not None:
                    gradient = torch.ones_like(l)
                    try:
                        l.backward(gradient)
                    except:
                        pass
            end_t = time.time()
            bp_time = (end_t - start_t) * 1000.0

            result.append(bp_time)
        else:
            gradient = torch.ones_like(x)
            '''
            with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as prof:
                x.backward(gradient)
            bp_time = prof.self_cpu_time_total / 1000.0
            '''
            start_t = time.time()
            x.backward(gradient)
            end_t = time.time()
            bp_time = (end_t - start_t) * 1000.0

            result.append(bp_time)

        # grad sze
        gradient_size = 0
        if isinstance(x_input, tuple):
            gradient = []
            for l in x_input:
                try:
                    gradient.append(l.grad.data)
                    tmp_size = 1
                    for s in l.grad.data.shape:
                        tmp_size *= s
                    gradient_size += tmp_size
                except:
                    gradient.append(None)
        else:
            gradient = x_input.grad.data
            tmp_size = 1
            for s in gradient.shape:
                tmp_size *= s
            gradient_size = tmp_size
        result.append(gradient_size * self.ELEMENT_BYTES / 1024 / 1024)

        return result

    def profile_layer_fwd(self, layer, input_info, batch_size=1):
        # get activation size and fwd time
        if isinstance(input_info, list):
            inputs = []
            for i, entry in enumerate(input_info):
                shape = list(entry[0])
                shape[0] = batch_size
                data_type = entry[1]
                data = np.ones(shape, dtype=data_type)
                req_grad = False
                inputs.append(Variable(torch.from_numpy(data), requires_grad=req_grad))
            x_input = tuple(inputs)
        else:
            shape = list(input_info[0])
            shape[0] = batch_size
            data_type = input_info[1]
            data = np.ones(shape, dtype=data_type)
            req_grad = False
            x_input = Variable(torch.from_numpy(data), requires_grad=req_grad)
        mem_usage = get_mem_usage()
        start_t = time.time()
        x = layer(x_input)
        end_t = time.time()
        computation_time = (end_t - start_t) * 1000.0
        act_mem = get_mem_usage() - mem_usage

        return x, act_mem, computation_time

    def profile_layer_bp(self, layer, input_info, batch_size=1):
        if isinstance(input_info, list):
            inputs = []
            for i, entry in enumerate(input_info):
                shape = list(entry[0])
                shape[0] = batch_size
                data_type = entry[1]
                data = np.ones(shape, dtype=data_type)
                req_grad = False
                if data.dtype in [np.float, np.float32, np.float64, np.float16]: req_grad = True
                inputs.append(Variable(torch.from_numpy(data), requires_grad=req_grad))
            x_input = tuple(inputs)
        else:
            shape = list(input_info[0])
            shape[0] = batch_size
            data_type = input_info[1]
            data = np.ones(shape, dtype=data_type)
            req_grad = False
            if data.dtype in [np.float, np.float32, np.float64, np.float16]: req_grad = True
            x_input = Variable(torch.from_numpy(data), requires_grad=req_grad)
        x = layer(x_input)

        # bp
        if isinstance(x, tuple):
            start_t = time.time()
            for i, l in enumerate(x):
                if l is not None:
                    gradient = torch.ones_like(l)
                    l.backward(gradient)
            end_t = time.time()
            bp_time = (end_t - start_t) * 1000.0
        else:
            gradient = torch.ones_like(x)
            start_t = time.time()
            x.backward(gradient)
            end_t = time.time()
            bp_time = (end_t - start_t) * 1000.0

        # grad size
        gradient_size = 0
        if isinstance(x_input, tuple):
            gradient = []
            for l in x_input:
                try:
                    gradient.append(l.grad.data)
                    tmp_size = 1
                    for s in l.grad.data.shape:
                        tmp_size *= s
                    gradient_size += tmp_size
                except:
                    gradient.append(None)
        else:
            gradient = x_input.grad.data
            tmp_size = 1
            for s in gradient.shape:
                tmp_size *= s
            gradient_size = tmp_size
        grad_size = gradient_size * self.ELEMENT_BYTES / 1024 / 1024

        return bp_time, grad_size

    def profile_layer_v2(self, layer_data: List, batch_size=1, test_round=1, start_round=0) -> List:
        layer = layer_data[1]
        input_info = layer_data[2]
        result = []

        # get layer size
        layer_size = 0
        for params in layer.parameters():
            tmp_size = 1
            for s in params.data.numpy().shape:
                tmp_size *= s
            layer_size += tmp_size
        result.append(layer_size * self.ELEMENT_BYTES / 1024 / 1024)  # todo: distinguish data type - we use 4 bytes for now

        # fwd
        fwd_time = []
        act_size = 0
        output = None
        for rid in range(test_round):
            output, act_mem, comp_time = self.profile_layer_fwd(layer, input_info, batch_size)
            if rid == 0: act_size = act_mem
            if rid >= start_round: fwd_time.append(comp_time)
            Logger.info("fwd round {}, mem usage {}MB".format(rid, get_mem_usage()))
        result.append(act_size)
        result.append(np.mean(fwd_time))

        # output size
        output_size = 0
        if isinstance(output, tuple):
            for t in output:
                tmp_size = 1
                for s in t.shape:
                    tmp_size *= s
                output_size += tmp_size
        else:
            output_size = 1
            for s in output.shape:
                output_size *= s
        result.append(output_size * self.ELEMENT_BYTES / 1024 / 1024)

        # bp
        bp_time = []
        grad_size = 0
        for rid in range(test_round):
            comp_time, grad_size = self.profile_layer_bp(layer, input_info, batch_size)
            if rid >= start_round: bp_time.append(comp_time)
            Logger.info("bp round {}, mem usage {}MB".format(rid, get_mem_usage()))
        result.append(np.mean(bp_time))
        result.append(grad_size)

        return result
