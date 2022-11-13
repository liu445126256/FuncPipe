try:
    import unzip_requirements
except ImportError:
    print("Import unzip failed!")

import pickle
import gc
import numpy as np
import time

import torch
from torch import nn

from funcpipe.planner.new_profiler import Profiler
from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline
from funcpipe.utils import get_mem_usage


# entrance function
def handler(event, context=None):
    # settings
    # print(event)
    # event = eval(event)
    Platform.use("aws")
    Logger.use_logger(Logger.HTTP, Logger.DEBUG, "profiler_{}MB".format(int(event["memory"])))
    Logger.info("Profiling start")
    # return {}
    task_seq = event["seq"]
    batch_size = int(event["batch_size"])
    mem = int(event["memory"])
    layer_num = int(event["layer_num"])
    prof_round = int(event["profile_round"])
    start_round = int(event["profile_start_round"])
    layer_id = int(event["layer_id"])

    Logger.debug("Profiling start")
    profiler = Profiler()
    layer_info = []  # info of one layer
    # get activation memory usage at the first round
    act_mem = -1

    # trial run to get through the resource allocation anomaly
    Logger.info("trial run util resource allocation is stable")
    if layer_id == 0:
        stable_time = 10
        file_name = "{}_l{}".format(task_seq, 0)
        data = Platform.download_from_storage(file_name)
        layer_data = pickle.loads(data)
        del data
        gc.collect()
        trial_start_t = time.time()
        while True:
            _ = profiler.profile_layer(layer_data, batch_size=batch_size)
            if time.time() - trial_start_t > stable_time: break
        del layer_data
        gc.collect()

    # profiling layer
    Logger.info("profiling layer {}, mem usage {}MB".format(layer_id, get_mem_usage()))
    file_name = "{}_l{}".format(task_seq, layer_id)
    # layer_data = layers[layer_id]
    data = Platform.download_from_storage(file_name)
    layer_data = pickle.loads(data)
    del data
    gc.collect()

    # prof version 1
    # '''
    for round_id in range(prof_round):
        info = profiler.profile_layer(layer_data, batch_size=batch_size)
        if round_id == 0: act_mem = info[1]
        if round_id >= start_round: layer_info.append(info)
        Logger.info("round {}, mem usage {}MB".format(round_id, get_mem_usage()))
    # Logger.info(str(layer_info))
    layer_info = np.mean(layer_info, axis=0).tolist()
    layer_info[1] = act_mem
    # '''

    # prof version 2
    # layer_info = profiler.profile_layer_v2(layer_data, batch_size, prof_round, start_round)

    # upload the result
    file_name = "{}_res_{}_l{}".format(task_seq, mem, layer_id)
    result_data = pickle.dumps(layer_info)
    Platform.upload_to_storage(file_name, result_data)

    if layer_id < layer_num - 1:
        # invoke the worker for profiling next layer
        event["layer_id"] = layer_id + 1
        Platform.invoke(event, asynchronous=True)
    else:
        Logger.info("Profiler_{}MB done.".format(mem))

    return {}