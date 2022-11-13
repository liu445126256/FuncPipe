'''test for Ali coud serverless apis'''
import numpy as np
import pickle
import time
import threading
import gc

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline
from funcpipe.monitor import Monitor

from pipetests.models.resnet import resnet101
from pipetests.models.amoebanet import amoebanetd


def profile_layer(layer_data, batch_size=1):
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
    result.append(layer_size * 4 / 1024 / 1024)  # todo: distinguish data type - we use 4 bytes for now

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

    start_t = time.time()
    x = layer(x_input)
    end_t = time.time()
    computation_time = (end_t - start_t) * 1000.0

    return computation_time


# model fwd
def fwd_model(model, input_x):
    '''
    x = input_x
    for n, l in model.named_children():
        x = l(x)
    '''
    fwd_time = 0
    bp_time = 0

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    start_tt = time.time()
    x = model(input_x)
    end_tt = time.time()
    fwd_time += (end_tt - start_tt) * 1000

    start_tt = time.time()
    #x.backward(torch.ones_like(x))
    end_tt = time.time()
    bp_time += (end_tt - start_tt) * 1000

    optimizer.step()
    optimizer.zero_grad()

    Logger.info("round fwd time: {}".format(fwd_time))
    Logger.info("round bp time: {}".format(bp_time))
    return x


def fwd_model2(model, input_x):
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    x = input_x
    fwd_output = []

    fwd_time = 0
    bp_time = 0
    layer_id = 0
    for n, l in model.named_children():
        gc.collect()
        # Logger.info(str(n))
        if isinstance(x, tuple):
            inputs = []
            for i, tensor in enumerate(x):
                shape = tensor.data.numpy().shape
                data_type = tensor.data.numpy().dtype
                data = np.ones(shape, dtype=data_type)
                req_grad = False
                if data.dtype in [np.float, np.float32, np.float64, np.float16]: req_grad = True
                inputs.append(Variable(torch.from_numpy(data), requires_grad=req_grad))
            x = tuple(inputs)
        else:
            shape = x.data.numpy().shape
            data_type = x.data.numpy().dtype
            data = np.ones(shape, dtype=data_type)
            req_grad = False
            if data.dtype in [np.float, np.float32, np.float64, np.float16]: req_grad = True
            x = Variable(torch.from_numpy(data), requires_grad=req_grad)

        start_tt = time.time()
        x = l(x)
        end_tt = time.time()
        fwd_time += (end_tt - start_tt) * 1000
        Logger.info("{} fwd:{}ms".format(n, (end_tt - start_tt)*1000))
        fwd_output.append(x)

        '''
        # get output for generating bp gradient
        if True:
            if isinstance(x, tuple):
                pass
                start_t = time.time()
                for i, k in enumerate(x):
                    if k is not None:
                        data_type = k.data.numpy().dtype
                        if data_type in [np.float, np.float32, np.float64, np.float16]:
                            gradient = torch.ones_like(k)
                            k.backward(gradient, retain_graph = True)
                end_t = time.time()
                bp_time += (end_t - start_t) * 1000.0
            else:
                gradient = torch.ones_like(x)
                start_t = time.time()
                x.backward(gradient, retain_graph = True)
                end_t = time.time()
                bp_time += (end_t - start_t) * 1000.0

            optimizer.step()
            optimizer.zero_grad()
        # '''
        layer_id += 1

    '''
    layer_id = 0
    for n, l in model.named_children():
        x = fwd_output[layer_id]
        if isinstance(x, tuple):
            for i, k in enumerate(x):
                if k is not None:
                    gradient = torch.zeros_like(k)
                    start_t = time.time()
                    k.backward(gradient)
                    end_t = time.time()
                    bp_time += (end_t - start_t) * 1000.0
        else:
            gradient = torch.zeros_like(x)
            start_t = time.time()
            x.backward(gradient)
            end_t = time.time()
            bp_time += (end_t - start_t) * 1000.0
        layer_id += 1
    #'''

    #'''
    layer_id = 0
    for n, l in model.named_children():
        x = fwd_output[layer_id]
        output_tensors = []
        recv_grads = []
        if isinstance(x, tuple):
            for i, k in enumerate(x):
                if k is not None:
                    gradient = torch.zeros_like(k)
                    output_tensors.append(k)
                    recv_grads.append(gradient)
        else:
            gradient = torch.zeros_like(x)
            output_tensors.append(x)
            recv_grads.append(gradient)
        start_t = time.time()
        torch.autograd.backward(output_tensors,recv_grads)
        end_t = time.time()
        bp_time += (end_t - start_t) * 1000.0
        layer_id += 1
    #'''


    model.zero_grad()
    Logger.info("round fwd time: {}".format(fwd_time))
    Logger.info("round bp time: {}".format(bp_time))
    return x


layer_num = 24
task_seq = "2325212833"


def download_layers():
    layers = []
    for layer_id in range(layer_num):
        Logger.info("Download {}".format(layer_id))
        file_name = "{}_l{}".format(task_seq, layer_id)
        layer_data = pickle.loads(Platform.download_from_storage(file_name))
        layers.append(layer_data)
    return layers


def fwd_model3(model, input_x, layers):
    # Timeline.start("fwd")
    sum_t = 0
    for layer_id in range(layer_num):
        _ = profile_layer(layers[layer_id], batch_size=4)
        sum_t += _
    Logger.info("round time: {}".format(sum_t))
    # Timeline.end("fwd")


# entrance function
#def handler(event, context=None):
if __name__ == "__main__":
    Platform.use("local")
    Logger.use_logger(Logger.NATIVE, Logger.INFO, "platform_test")
    data = np.array([4, 2, 3, 4, 5, 6])
    data = pickle.dumps(data)
    filename = "ali_platform_test.npy"
    mon = Monitor()
    mon.start()

    stop = False
    test_rounds = 30

    Logger.info("Cal started")
    input_sample = torch.rand(4, 3, 224, 224)
    # model = resnet101()
    model = amoebanetd(num_classes=1000, num_layers=18, num_filters=256)
    Logger.info("Model built")

    time_stats = []

    # layers = download_layers()

    for rid in range(test_rounds):
        Timeline.start("fwd")
        start_t = time.time()

        # x = model(input_sample)
        # x = fwd_model3(model, input_sample, layers)
        x = fwd_model2(model, input_sample)
        #x = fwd_model(model, input_sample)

        Timeline.end("fwd")
        end_t = time.time()
        # del x
        # gc.collect()
        time_stats.append((end_t - start_t) * 1000.0)
    Logger.info("mean: {}ms".format(np.mean(time_stats)))
    Logger.info("cal stopped")

    Timeline.report()
    mon.report()