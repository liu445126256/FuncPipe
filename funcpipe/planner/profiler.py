"""model profiler"""
from typing import Dict, List, Tuple, Union
import time
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from funcpipe.debugger import Logger
from funcpipe.utils import get_mem_usage, linefit


class Profiler:
    #Designed as a static class for now
    #
    def __init__(self):
        pass

    def profile(self, model: nn.Sequential, input_sample = None)-> Dict:
        return self.profile_n_sample(model, input_sample)
        #
        #:param self:
        #:param model:
        #:param input sample: one input sample is required for profiling (using a 3 * 224 * 224 img by default)
        #:return: Dict - layer_index: [model_size, activation_size, output_size]
        #
        if input_sample is None:
            self.input_sample = torch.rand(1, 3, 224, 224)  # use a synthetic img as default
        else:
            self.input_sample = input_sample

        model_infos = []
        #using batch size 1, 2, 4 for line fit
        sample_num = 3
        sample = self.input_sample
        for i in range(sample_num):
            model_infos.append(self.profile_n_sample(model, sample))
            if isinstance(sample, tuple):
                mul_sample = []
                for t in sample:
                    mul_sample.append(torch.cat([t, t], dim=0))
                sample = tuple(mul_sample)
            else:
                sample = torch.cat([sample, sample], dim=0)

        # line fit
        model_info = model_infos[0].copy()
        for lid in range(len(model_infos[0])):
            #fwd
            x = [2 ** i for i in range(sample_num)]
            y = [model_infos[i][lid][3] for i in range(sample_num)]
            if sum(y) == 0: alpha = beta = 0
            else: alpha, beta, _ = linefit(x , y)
            model_info[lid][3] = (alpha, max(beta, 0), model_info[lid][3])
            #bp
            x = [2 ** i for i in range(sample_num)]
            y = [model_infos[i][lid][4] for i in range(sample_num)]
            if sum(y) == 0: alpha = beta = 0
            else: alpha, beta, _ = linefit(x, y)
            model_info[lid][4] = (alpha, max(beta, 0), model_info[lid][4])
        return model_info

    def profile_n_sample(self, model: nn.Sequential, input_sample = None)-> Dict:
        if input_sample is None:
            self.input_sample = torch.rand(4, 3, 224, 224)  # use a synthetic img as default
        else:
            self.input_sample = input_sample

        # get layer size
        model_info = {}
        layer_num = 0
        for n, l in model.named_children():
            layer_size = 0
            for params in l.parameters():
                tmp_size = 1
                for s in params.data.numpy().shape:
                    tmp_size *= s
                layer_size += tmp_size
            #Logger.debug("layer{}:\nsize:{}MB\n{}".format(layer_num, layer_size * 4 / 1024 / 1024, n))
            model_info[layer_num] = [layer_size * 4 / 1024 / 1024] #todo: distinguish data type - we use 4 bytes for now
            layer_num += 1

        # get activation size
        x = self.input_sample
        mem_usage = get_mem_usage()
        layer_num = 0
        output_store = []
        input_store = []
        for n, l in model.named_children():
            #print(layer_num)
            start_t = time.time()
            with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as prof:
                x = l(x)
                #print(x.shape)
                #print(x.data.numpy().shape)
                #print(torch.from_numpy(x.data.numpy()).shape)
            end_t = time.time()
            computation_time = prof.self_cpu_time_total / 1000.0
            #print("profiler:{}  time:{}".format(computation_time, (end_t - start_t) * 1000))
            output_store.append((n, x))
            if isinstance(x, tuple):
                inputs = []
                for i, tensor in enumerate(x):
                    data = tensor.data.numpy()
                    req_grad = False
                    if data.dtype in [np.float, np.float32, np.float64, np.float16]: req_grad = True
                    inputs.append(Variable(torch.from_numpy(data), requires_grad=req_grad))
                x = tuple(inputs)
            else:
                data = x.data.numpy()
                req_grad = False
                if data.dtype in [np.float, np.float32, np.float64, np.float16]: req_grad = True
                x = Variable(torch.from_numpy(data), requires_grad=req_grad)
            input_store.append((n, x))
            last_mem_usage = mem_usage
            mem_usage = get_mem_usage()
            mem_cost = mem_usage - last_mem_usage
            while mem_cost < 0: mem_cost +=1
            model_info[layer_num].append(mem_cost)
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
            model_info[layer_num].append(output_size * 4 / 1024 / 1024) #todo: distinguish data type - we use 4 bytes for now
            model_info[layer_num].append(computation_time)
            #Logger.info("layer{}: size{}MB  activation:{}MB output_size:{}MB computation_time:{}ms".format(layer_num, str(model_info[layer_num][0]),
            #                                                                         str(model_info[layer_num][1]),
            #                                                                         str(model_info[layer_num][2]), str(model_info[layer_num][3])))
            layer_num += 1


        gradient = torch.ones_like(output_store[-1][1])
        for layer_num in range(len(output_store) - 1, -1, -1):
            #print(layer_num)
            output = output_store[layer_num][1]
            #print(type(output))
            #print(type(gradient))
            if isinstance(output, tuple):
                assert type(gradient) == list
                with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as prof:
                    for i,l in enumerate(output):
                        if gradient[i] is not None:
                            l.backward(gradient[i])
                bp_time = prof.self_cpu_time_total / 1000.0
                model_info[layer_num].append(bp_time)
            else:
                with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as prof:
                    #print(gradient.shape)
                    output.backward(gradient)
                bp_time = prof.self_cpu_time_total / 1000.0
                model_info[layer_num].append(bp_time)

            gradient_size = 0
            if layer_num > 0:
                input = input_store[layer_num - 1][1]
                #print(type(input))
                if isinstance(input, tuple):
                    gradient = []
                    for l in input:
                        try:
                            gradient.append(l.grad.data)
                            tmp_size = 1
                            for s in l.grad.data.shape:
                                tmp_size *= s
                            gradient_size += tmp_size
                        except:
                            gradient.append(None)
                else:
                    gradient = input.grad.data
                    tmp_size = 1
                    for s in gradient.shape:
                        tmp_size *= s
                    gradient_size = tmp_size
            model_info[layer_num].append(gradient_size * 4 / 1024 / 1024)

        return model_info


'''
class Profiler:
    #Designed as a static class for now
    def __init__(self, input_specs: Union[List[Tuple], Tuple, None] = None, input_types = torch.FloatTensor):
        self.input_specs = input_specs
        self.input_types = input_types

    def profile(self, model: nn.Sequential)-> Dict:

        # generate a synthetic sample
        if self.input_specs is None:
            # using 3 * 224 * 224 img
            input_sample = torch.rand(1, 3, 224, 224)
        elif isinstance(self.input_specs, list):
            input_sample = []
            for i, spec in enumerate(self.input_specs):
                total_num =1
                for s in spec: total_num *= s
                sample = self.input_types[i](total_num)
                sample = torch.reshape(sample, [1]+ list(spec))
                input_sample.append(sample)
                print(sample.shape)
            input_sample = tuple(input_sample)
        else:
            total_num = 1
            for i in self.input_specs: total_num *= i
            sample = self.input_types(total_num)
            input_sample = torch.reshape(sample, [1]+ list(self.input_specs))

        # get layer size
        model_info = {}
        layer_num = 0
        for n, l in model.named_children():
            layer_size = 0
            for params in l.parameters():
                tmp_size = 1
                for s in params.data.numpy().shape:
                    tmp_size *= s
                layer_size += tmp_size
            Logger.debug("layer{}:\nsize:{}MB\n{}".format(layer_num, layer_size * 4 / 1024 / 1024, n)) #todo: distinguish data type - we use 4 bytes for now
            model_info[layer_num] = [layer_size * 4 / 1024 / 1024]
            layer_num += 1

        # get activation size
        x = input_sample
        mem_usage = get_mem_usage()
        layer_num = 0
        for n, l in model.named_children():
            x = l(x)
            last_mem_usage = mem_usage
            mem_usage = get_mem_usage()
            mem_cost = mem_usage - last_mem_usage
            model_info[layer_num].append(mem_cost)
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
            model_info[layer_num].append(output_size * 4 / 1024 / 1024) #todo: distinguish data type - we use 4 bytes for now
            layer_num += 1

        return model_info

#'''


"""BACKUP"""
'''
"""model profiler"""
from typing import Dict, List, Tuple, Union
import gc

import torch
from torch import nn

from funcpipe.debugger import Logger
from funcpipe.utils import get_mem_usage


class Profiler:

    def __init__(self):
        pass

    def profile(self, model: nn.Sequential, input_sample = None)-> Dict:

        if input_sample is None: self.input_sample = torch.rand(1, 3, 224, 224) # use a synthetic img as default
        else: self.input_sample = input_sample
        # get layer size
        model_info = {}
        layer_num = 0
        for n, l in model.named_children():
            layer_size = 0
            for params in l.parameters():
                tmp_size = 1
                for s in params.data.numpy().shape:
                    tmp_size *= s
                layer_size += tmp_size
            #Logger.debug("layer{}:\nsize:{}MB\n{}".format(layer_num, layer_size * 4 / 1024 / 1024, n))
            model_info[layer_num] = [layer_size * 4 / 1024 / 1024] #todo: distinguish data type - we use 4 bytes for now
            layer_num += 1

        # get activation size
        x = self.input_sample
        mem_usage = get_mem_usage()
        layer_num = 0
        output_store = []
        for n, l in model.named_children():
            with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as prof:
                x = l(x)
            output_store.append((n, x))
            computation_time = prof.self_cpu_time_total / 1000.0
            last_mem_usage = mem_usage
            mem_usage = get_mem_usage()
            mem_cost = mem_usage - last_mem_usage
            while mem_cost < 0: mem_cost +=1
            model_info[layer_num].append(mem_cost)
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
            model_info[layer_num].append(output_size * 4 / 1024 / 1024) #todo: distinguish data type - we use 4 bytes for now
            model_info[layer_num].append(computation_time)
            #Logger.info("layer{}: size{}MB  activation:{}MB output_size:{}MB computation_time:{}ms".format(layer_num, str(model_info[layer_num][0]),
            #                                                                         str(model_info[layer_num][1]),
            #                                                                         str(model_info[layer_num][2]), str(model_info[layer_num][3])))
            layer_num += 1

        return model_info
'''