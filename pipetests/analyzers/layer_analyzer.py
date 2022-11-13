import torch
import torch.nn as nn
from functools import reduce
from pipetests.analyzers.FLOPs_analyzer import *
from pipetests.analyzers.memory_analyzer import *
from pipetests.analyzers.outputsize_analyzer import *

def analysis_model(model, input_size):
    """
    analysis each layer of model: get layers' outputsize, FLOPs and inference memory usage
    :param input_model:(nn.Sequential) , input_size
    :return: FLOPs, inf_memory
    """
    Flops_dict = {}
    memory_dict = {}
    start_id = -1
    end_id = -1
    in_block = False
    next_input = input_size
    start_res_input = None
    for id, layer in enumerate(model):
        if isinstance(layer, Twin):
            start_id = id
            total_flops = 0
            total_memory = 0
            in_block = True
            cur_output, start_res_input = compute_size_wrapper(layer, next_input)
        elif isinstance(layer, Residual):
            end_id = id
            cur_output, _ = compute_size_wrapper(layer, next_input)
            cur_flops = compute_flops_wrapper(layer, next_input, cur_output)
            total_flops += cur_flops
            cur_memory = compute_memory_wrapper(layer, next_input, cur_output)
            total_memory += cur_memory
            next_input = cur_output
            in_block = False
            cur_block_id = str(start_id) + "--" + str(end_id)
            Flops_dict[cur_block_id] = total_flops
            memory_dict[cur_block_id] = total_memory
        else:
            cur_output, _ = compute_size_wrapper(layer, next_input)
            if in_block:
                cur_flops = compute_flops_wrapper(layer, next_input, cur_output)
                total_flops += cur_flops
                cur_memory = compute_memory_wrapper(layer, next_input, cur_output)
                total_memory += cur_memory
                next_input = cur_output
            else:
                cur_flops = compute_flops_wrapper(layer, next_input, cur_output)
                cur_memory = compute_memory_wrapper(layer, next_input, cur_output)
                cur_id = str(id)
                Flops_dict[cur_id] = cur_flops
                memory_dict[cur_id] = cur_memory
                next_input = cur_output
    return Flops_dict, memory_dict

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from pipetests.models.resnet import resnet101, resnet50
    model = resnet101()
    print(model)
    Flops_dict, memory_dict = analysis_model(model, (1, 3, 224, 224))
    print(Flops_dict)
    print(memory_dict)
    total_mem = 0
    for value in memory_dict.values():
        total_mem += value
    print(total_mem)