import pickle
import gc
import numpy as np

import torch
from torch import nn

from funcpipe.planner.new_profiler import Profiler
from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline
from funcpipe.utils import get_mem_usage



#entrance function
def handler(event, context=None):
    # settings
    #event = eval(event)
    Platform.use("local")
    Logger.use_logger(Logger.NATIVE, Logger.DEBUG, "profiler_{}MB".format(int(event["memory"])))
    PROF_ROUND = 3
    START_ROUND = 1


    task_seq = event["seq"]
    batch_size = event["batch_size"]
    mem = event["memory"]
    layer_num = event["layer_num"]

    Logger.debug("Profiling start")
    profiler = Profiler()
    model_info = {}
    # profiling at layer level
    for round_id in range(PROF_ROUND):
        for layer_id in range(layer_num):
            Logger.info("profiling round {} layer {}, mem usage {}MB".format(round_id, layer_id, get_mem_usage()))
            file_name = "{}_l{}".format(task_seq, layer_id)
            layer_data = pickle.loads(Platform.download_from_storage(file_name))
            layer_info = profiler.profile_layer(layer_data, batch_size = batch_size)
            if layer_id not in model_info.keys(): model_info[layer_id] = []
            model_info[layer_id].append(layer_info)
            del layer_data
            gc.collect()
    # averaging the final results
    for layer_id in range(layer_num):
            model_info[layer_id] = np.mean(model_info[layer_id], axis = 0).tolist()
    # upload the result
    file_name = "{}_res{}".format(task_seq, mem)
    result_data = pickle.dumps(model_info)
    Platform.upload_to_storage(file_name, result_data)

    Logger.info("Profiler_{}MB done.".format(mem))