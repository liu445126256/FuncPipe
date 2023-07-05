"""
This is a test for FuncPipe module
"""
from typing import cast

import torch
import torch.optim as optim
import torch.nn as nn
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
    model = resnet101()  # numclasses = 1000
    Logger.debug("Model built")
    model = FuncPipe(model, loss_func = loss_func, optim_class = optimizer, learning_rate = learning_rate, batch_size = batch_size)

    # partition plan
    # we manually specify the partition for test
    model.planner.partition_plan = [304]
    model.planner.tensor_parallelism = [1]
    model.planner.data_parallelism = [1]
    model.planner.micro_batchsize = 1

    model.init(role_info)

    for epoch_id in range(epoches):
        for batch_id, (inputs, targets) in enumerate(data):
            model.pipeline_train(inputs, targets)