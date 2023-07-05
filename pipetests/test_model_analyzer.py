import sys
sys.path.append("../")

from funcpipe.debugger import Logger
from pipetests.analyzers.layer_analyzer import analysis_model
from pipetests.models.bert.model.language_model import make_bert_nsp

from collections import OrderedDict
from typing import Iterator, Tuple
from torch import nn


def flatten_sequential(module: nn.Sequential):
    """flatten_sequentials a nested sequential module."""
    if not isinstance(module, nn.Sequential):
        raise TypeError('not sequential')

    return nn.Sequential(OrderedDict(_flatten_sequential(module)))


def _flatten_sequential(module: nn.Sequential, res):
    for name, child in module.named_children():
        # flatten_sequential child sequential layers only.
        if len(list(child.named_children())) > 1:
            _flatten_sequential(child, res)
            #for sub_name, sub_child in _flatten_sequential(child):
                #print(f'{name}_{sub_name}', sub_child)
                #return f'{name}_{sub_name}', sub_child
        else:
            print(name, child)
            res.append(child)
            #return name, child

if __name__ == "__main__":
    Logger.use_logger(Logger.NATIVE, Logger.DEBUG, "analyzer_test")

    layers = []
    _flatten_sequential(make_bert_nsp(14), layers)
    model = layers
    print(len(model))

    Flops_dict, memory_dict = analysis_model(model, (1, 3, 224, 224))
    Logger.info(str(Flops_dict))
    Logger.info(str(memory_dict))
    total_mem = 0
    for value in memory_dict.values():
        total_mem += value
    Logger.info(str(total_mem))