import torch.nn as nn
from pipetests.analyzers.outputsize_analyzer import *
from functools import reduce

import sys
sys.path.append("..")
from pipetests.models.resnet.bottleneck import Gutter, Twin, Residual


def compute_memory(module, inp, out):
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_memory(module, inp, out)
    elif isinstance(module, nn.PReLU):
        return compute_PReLU_memory(module, inp, out)
    elif isinstance(module, nn.Conv2d):
        return compute_Conv2d_memory(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_memory(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_memory(module, inp, out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_memory(module, inp, out)
    else:
        print(f"[Memory]: {type(module).__name__} is not supported!")
        return (0, 0)
    pass


def num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_ReLU_memory(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU))
    batch_size = inp[0]
    mread = batch_size * inp[1:].numel()
    mwrite = batch_size * inp[1:].numel()

    return (mread, mwrite)


def compute_PReLU_memory(module, inp, out):
    assert isinstance(module, (nn.PReLU))
    batch_size = inp[0]
    mread = batch_size * (inp[1:].numel() + num_params(module))
    mwrite = batch_size * inp[1:].numel()

    return (mread, mwrite)


def compute_Conv2d_memory(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp) == 4 and len(inp) == len(out)

    batch_size = inp[0]
    in_c = inp[1]
    out_c, out_h, out_w = out[1:]

    # This includes weighs with bias if the module contains it.
    mread = batch_size * (inp[1:].numel() + num_params(module))
    mwrite = batch_size * out_c * out_h * out_w
    return (mread, mwrite)


def compute_BatchNorm2d_memory(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp) == 4 and len(inp) == len(out)
    batch_size, in_c, in_h, in_w = inp

    mread = batch_size * (inp[1:].numel() + 2 * in_c)
    mwrite = inp.numel()
    return (mread, mwrite)


def compute_Linear_memory(module, inp, out):
    assert isinstance(module, nn.Linear)
    assert len(inp) == 2 and len(out) == 2
    batch_size = inp[0]
    mread = batch_size * (inp[1:].numel() + num_params(module))
    mwrite = reduce(lambda x,y:x*y,out)

    return (mread, mwrite)


def compute_Pool2d_memory(module, inp, out):
    assert isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))
    assert len(inp) == 4 and len(inp) == len(out)
    batch_size = inp[0]
    mread = batch_size * inp[1:].numel()
    mwrite = batch_size * out[1:].numel()
    return (mread, mwrite)


def compute_memory_wrapper_old(module, inp, out):
    if isinstance(module, Gutter):
        return compute_memory(module.module, inp, out)
    elif isinstance(module, Twin):
        return 0
    elif isinstance(module, Residual):
        downsample = module.downsample
        total_memory = 0
        for layer in downsample:
            cur_output = compute_size(layer, inp)
            total_memory += compute_memory(layer, inp, cur_output)
        return total_memory
    else:
        return compute_memory(module, inp, out)


def compute_memory_wrapper(module, inp, out):
    if isinstance(module, Gutter):
        return reduce(lambda x,y:x*y,out) * 4 * 1.2 / (1024 ** 2)
    elif isinstance(module, Twin):
        return 0
    elif isinstance(module, Residual):
        downsample = module.downsample
        total_memory = 0
        if downsample == None:
            return 0
        for layer in downsample:
            cur_output = compute_size(layer, inp)
            total_memory += reduce(lambda x,y:x*y, cur_output) * 4 * 1.2 / (1024 ** 2)
        return total_memory
    else:
        if isinstance(out, tuple):
            return reduce(lambda x,y:x*y,out) * 4 * 1.2 / (1024 ** 2)
        else:
            return out