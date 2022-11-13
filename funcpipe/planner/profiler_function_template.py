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

def prof_layer(profiler, batch_size, layer_id, task_seq):
    file_name = "{}_l{}".format(task_seq, layer_id)
    layer_data = pickle.loads(Platform.download_from_storage(file_name))
    layer_info = profiler.profile_layer(layer_data, batch_size=batch_size)
    for l in layer_data: del l
    del layer_data
    gc.collect()
    return layer_info

#entrance function
def handler(event, context=None):
    # settings
    #event = eval(event)
    Platform.use("local")
    Logger.use_logger(Logger.NATIVE, Logger.DEBUG, "profiler_{}MB".format(int(event["memory"])))

    task_seq = event["seq"]
    batch_size = int(event["batch_size"])
    mem = int(event["memory"])
    layer_num = int(event["layer_num"])
    prof_round = int(event["profile_round"])
    start_round = int(event["profile_start_round"])

    Logger.debug("Profiling start")
    profiler = Profiler()
    model_info = {}
    # get activation memory usage at the first round
    act_mem = []
    # profiling at layer level
    for round_id in range(prof_round):
        for layer_id in range(layer_num):
            Logger.info("profiling round {} layer {}, mem usage {}MB".format(round_id, layer_id, get_mem_usage()))

            file_name = "{}_l{}".format(task_seq, layer_id)
            layer_data = pickle.loads(Platform.download_from_storage(file_name))
            layer_info = profiler.profile_layer(layer_data, batch_size = batch_size)

            if round_id == 0: act_mem.append(layer_info[1])
            if layer_id not in model_info.keys(): model_info[layer_id] = []
            if round_id >= start_round: model_info[layer_id].append(layer_info)

            del layer_data
            gc.collect()
    # averaging the final results
    for layer_id in range(layer_num):
        model_info[layer_id] = np.mean(model_info[layer_id], axis = 0).tolist()
        model_info[layer_id][1] = act_mem[layer_id]

    # upload the result
    file_name = "{}_res{}".format(task_seq, mem)
    result_data = pickle.dumps(model_info)
    Platform.upload_to_storage(file_name, result_data)

    Logger.info("Profiler_{}MB done.".format(mem))