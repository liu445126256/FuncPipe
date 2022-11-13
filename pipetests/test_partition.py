"""
This is a test for FuncPipe module
"""
import time

import torch
import torch.optim as optim
import torch.nn.functional as F

from funcpipe import FuncPipe
from funcpipe.debugger import Logger

"""
User-defined model
"""
from pipetests.models.resnet import resnet101

if __name__ == "__main__":
    Logger.use_logger(Logger.NATIVE, Logger.DEBUG)
    Logger.debug("Start")

    # training configuration
    batch_size = 4
    epoches = 5
    learning_rate = 0.001
    loss_func = F.cross_entropy
    optimizer = optim.SGD

    # generate input data
    dataset_size = 100
    input = torch.rand(batch_size, 3, 224, 224)
    target = torch.randint(1000, (batch_size,))
    data = [(input, target)] * (dataset_size // batch_size)
    if dataset_size % batch_size != 0:
        last_input = input[:dataset_size % batch_size]
        last_target = target[:dataset_size % batch_size]
        data.append((last_input, last_target))

    # FuncPipe training
    my_rank = 0
    role_info = {"rank": my_rank}

    time.sleep(100)
    model = resnet101()  # numclasses = 1000
    time.sleep(100)
    '''
    layer_count = 0
    for n, l in model.named_children():
        print(n, "l")
        layer_count += 1
    print(layer_count)
    exit()
    '''
    '''
    print(data[0][0].chunk(4)[0].shape)
    exit()
    '''

    Logger.debug("Model built")
    model = FuncPipe(model, loss_func = loss_func, optim_class = optimizer, learning_rate = learning_rate, batch_size = batch_size)
    model.init(role_info)
