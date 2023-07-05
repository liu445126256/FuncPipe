import torch
from torch import nn
import pickle

from funcpipe.planner.new_profiler import Profiler
from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.utils import get_mem_usage

"""
User-defined model
"""
from pipetests.models.resnet import resnet101
from pipetests.models.amoebanet import amoebanetd
from pipetests.models.bert.model.language_model import make_bert_nsp
from pipetests.models.bert.dataset import BERTDataset, WordVocab


if __name__ == "__main__":
    Platform.use("local")
    Logger.use_logger(Logger.NATIVE, Logger.INFO, "new_profiler_test")

    model = amoebanetd(num_classes=1000, num_layers=18, num_filters=256)

    profiler = Profiler()
    Logger.info("Starting mem usage: {}MB".format(get_mem_usage()))
    res = profiler.profile(model, [1024], batch_size=1)

    Logger.info(str(res))
    # upload to storage
    res_data = pickle.dumps(res)
    Platform.upload_to_storage("prof_resnet101", res_data)