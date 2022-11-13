'''Micro batch related operation'''
from typing import Tuple, List, Union

import torch
from torch import Tensor

def split(input: Union[None, Tensor, Tuple[Tensor, ...]], targets: Tensor, batchsize, micro_batchsize) -> Tuple[List[Union[Tensor, Tuple[Tensor, ...]]], ...]:
    microbatches = []
    microtargets = []
    chunk_num = batchsize // micro_batchsize

    # we use a trick to deal with the data loading problem for now
    # todo: redesign the data loading
    if input is not None:
        if isinstance(input, tuple):
            tmp_batches = []
            for tensors in input:
                tmp_microbatches = []
                for tensor in tensors.chunk(chunk_num):
                    tmp_microbatches.append(tensor)
                tmp_batches.append(tmp_microbatches)
            for i in range(len(tmp_batches[0])):
                l_microbatch = []
                for j in range(len(tmp_batches)):
                    l_microbatch.append(tmp_batches[j][i])
                microbatches.append(tuple(l_microbatch))
        else:
            for tensor in input.chunk(chunk_num):
                microbatches.append(tensor)
        for tensor in targets.chunk(chunk_num):
            microtargets.append(tensor)
    else:
        for tensor in range(chunk_num):
            inputt = torch.rand(micro_batchsize, 3, 224, 224)
            microbatches.append(inputt)
        for tensor in range(chunk_num):
            targett = torch.randint(10, (micro_batchsize,))  # num classes = 10 - cifar10
            microtargets.append(targett)

    return microbatches, microtargets
