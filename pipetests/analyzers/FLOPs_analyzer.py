import torch.nn as nn
from pipetests.analyzers.outputsize_analyzer import *

import sys
sys.path.append("..")
from pipetests.models.resnet.bottleneck import Gutter, Twin, Residual

def compute_Conv2d_flops(module, inp, out):
    assert isinstance(module, nn.Conv2d)
    assert len(inp) == 4 and len(inp) == len(out)

    in_c = inp[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out[1:]
    groups = module.groups

    # ops per output element
    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
    kernel_add_group = kernel_add * out_h * out_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return total_mul + total_add


def compute_ConvTranspose2d_flops(module, inp, out):
    assert isinstance(module, nn.ConvTranspose2d)
    assert len(inp) == 4 and len(inp) == len(out)

    in_c, in_h, in_w = inp[1:]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out[1:]
    groups = module.groups

    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * in_h * in_w * (out_c // groups)
    kernel_add_group = kernel_add * in_h * in_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return total_mul + total_add


def compute_BatchNorm2d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp) == 4 and len(inp) == len(out)

    in_c, in_h, in_w = inp[1:]

    # 1. sub mean
    # 2. div standard deviation
    # 3. mul alpha
    # 4. add beta
    return 4 * in_c * in_h * in_w


def compute_MaxPool2d_flops(module, inp, out):
    assert isinstance(module, nn.MaxPool2d)
    assert len(inp) == 4 and len(inp) == len(out)

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out[1:]

    return (k_h * k_w - 1) * out_h * out_w * out_c


def compute_AvgPool2d_flops(module, inp, out):
    assert isinstance(module, nn.AvgPool2d)
    assert len(inp) == 4 and len(inp) == len(out)

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out[1:]

    kernel_add = k_h * k_w - 1
    kernel_avg = 1

    return (kernel_add + kernel_avg) * (out_h * out_w) * out_c


def compute_ReLU_flops(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6))

    count = 1
    for i in inp[1:]:
        count *= i
    return count


def compute_Softmax_flops(module, inp, out):
    assert isinstance(module, nn.Softmax)
    assert len(inp) > 1

    count = 1
    for s in inp[1:]:
        count *= s
    exp = count
    add = count - 1
    div = count
    return exp + add + div


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    assert len(inp) == 2 and len(out) == 2

    num_in_features = inp[1]
    num_out_features = out[1]

    mul = num_in_features
    add = num_in_features - 1
    return num_out_features * (mul + add)


def compute_Bilinear_flops(module, inp1, inp2, out):
    assert isinstance(module, nn.Bilinear)
    assert len(inp1) == 2 and len(inp2) == 2 and len(out) == 2

    num_in_features_1 = inp1[1]
    num_in_features_2 = inp2[1]
    num_out_features = out[1]

    mul = num_in_features_1 * num_in_features_2 + num_in_features_2
    add = num_in_features_1 * num_in_features_2 + num_in_features_2 - 1
    return num_out_features * (mul + add)


def compute_flops(module, inp, out):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp, out)
    elif isinstance(module, nn.ConvTranspose2d):
        return compute_ConvTranspose2d_flops(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_flops(module, inp, out)
    elif isinstance(module, nn.MaxPool2d):
        return compute_MaxPool2d_flops(module, inp, out)
    elif isinstance(module, nn.AvgPool2d):
        return compute_AvgPool2d_flops(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        return compute_ReLU_flops(module, inp, out)
    elif isinstance(module, nn.Softmax):
        return compute_Softmax_flops(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp, out)
    elif isinstance(module, nn.Bilinear):
        return compute_Bilinear_flops(module, inp[0], inp[1], out)
    else:
        print(f"[flops]: {type(module).__name__} is not supported!")
        return 0


def compute_flops_wrapper(module, inp, out):
    if isinstance(module, Gutter):
        return compute_flops(module.module, inp, out)
    elif isinstance(module, Twin):
        # return current layer flops and total flops
        return 0
    elif isinstance(module, Residual):
        downsample = module.downsample
        total_flops = 0
        if downsample == None:
            return 0
        for layer in downsample:
            cur_output = compute_size(layer, inp)
            total_flops += compute_flops(layer, inp, cur_output)
        return total_flops
    else:
        return compute_flops(module, inp, out)