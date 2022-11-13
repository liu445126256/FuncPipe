import torch.nn as nn
import math
from functools import reduce

import sys
sys.path.append("..")
from pipetests.models.resnet.bottleneck import Gutter, Twin, Residual
from pipetests.models.resnet import Flatten


# input: B C H W

def compute_Conv2d_size(module, inp):
    assert isinstance(module, nn.Conv2d)

    k_h, k_w = module.kernel_size
    s_h, s_w = module.stride
    p_h, p_w = module.padding

    o_h = math.floor((inp[2] - k_h + 2 * p_h) / s_h + 1)
    o_w = math.floor((inp[3] - k_w + 2 * p_w) / s_w + 1)
    return inp[0], module.out_channels, o_h, o_w


def compute_ConvTranspose2d_size(module, inp):
    assert isinstance(module, nn.ConvTranspose2d)
    print("ConvTranspose2d Not support yet!")
    pass


def compute_BatchNorm2d_size(module, inp):
    assert isinstance(module, nn.BatchNorm2d)
    return inp


def compute_MaxPool2d_size(module, inp):
    assert isinstance(module, nn.MaxPool2d)

    i_h, i_w = inp[2:]
    # print(module)
    # print(module.padding)
    p_h = p_w = module.padding
    d_h = d_w = module.dilation
    k_h = k_w = module.kernel_size
    s_h = s_w = module.stride

    o_h = math.floor((i_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1)
    o_w = math.floor((i_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1)

    return inp[0], inp[1], o_h, o_w


def compute_AvgPool2d_size(module, inp):
    assert isinstance(module, nn.AvgPool2d)

    i_h, i_w = inp[2:]
    p_h, p_w = module.padding
    k_h, k_w = module.kernel_size
    s_h, s_w = module.stride

    o_h = math.floor((i_h + 2 * p_h - k_h) / s_h + 1)
    o_w = math.floor((i_w + 2 * p_w - k_w) / s_w + 1)

    return inp[0], inp[1], o_h, o_w


def compute_ReLU_size(module, inp):
    assert isinstance(module, (nn.ReLU, nn.ReLU6))

    return inp


def compute_Softmax_size(module, inp):
    assert isinstance(module, nn.Softmax)
    assert len(inp) > 1

    return inp[0] * inp[1] * inp[2] * inp[3]


def compute_Linear_size(module, inp):
    assert isinstance(module, nn.Linear)

    o_features = module.out_features
    return inp[0], o_features


def compute_Bilinear_size(module, inp1, inp2):
    assert isinstance(module, nn.Bilinear)
    print("Bilinear not supported yet!")
    pass


def compute_AdaptiveAvgool2d_size(module, inp):
    assert isinstance(module, nn.AdaptiveAvgPool2d)
    o_h, o_w = module.output_size
    return inp[0], inp[1], o_h, o_w


def compute_flatten_size(module, inp):
    assert isinstance(module, Flatten)
    return inp[0], reduce(lambda x, y: x * y, inp[1:])


def compute_twin_size(module, inp):
    assert isinstance(module, Twin)
    return inp


def compute_gutter_size(module, inp):
    assert isinstance(module, Gutter)
    return compute_size(module.module, inp)

def compute_size(module, inp):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_size(module, inp)
    elif isinstance(module, nn.ConvTranspose2d):
        return compute_ConvTranspose2d_size(module, inp)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_size(module, inp)
    elif isinstance(module, nn.MaxPool2d):
        return compute_MaxPool2d_size(module, inp)
    elif isinstance(module, nn.AvgPool2d):
        return compute_AvgPool2d_size(module, inp)
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        return compute_ReLU_size(module, inp)
    elif isinstance(module, nn.Softmax):
        return compute_Softmax_size(module, inp)
    elif isinstance(module, nn.Linear):
        return compute_Linear_size(module, inp)
    elif isinstance(module, nn.Bilinear):
        return compute_Bilinear_size(module, inp[0], inp[1])
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return compute_AdaptiveAvgool2d_size(module, inp)
    elif isinstance(module, Flatten):
        return compute_flatten_size(module, inp)
    else:
        print(f"[size]: {type(module).__name__} is not supported!")
        return 0


def compute_size_wrapper(module, inp):
    if isinstance(module, Gutter):
        return compute_size(module.module, inp), 0
    elif isinstance(module, Twin):
        ## here begins a residual connection
        # return next layer and start_residual_input
        return inp, inp
    elif isinstance(module, Residual):
        ## here ends a residual connection
        # return next layer
        return inp, 0
    else:
        return compute_size(module, inp), 0
