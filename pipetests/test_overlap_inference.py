'''test for Ali coud serverless apis'''
import numpy as np
import pickle
import time
import threading

import torch

from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline
from funcpipe.monitor import Monitor

from pipetests.models.resnet import resnet101

# entrance function
def handler(event, context=None):
    Platform.use("ali")
    Logger.use_logger(Logger.HTTP, Logger.INFO, "platform_test")
    data = np.array([4, 2, 3, 4, 5, 6])
    data = pickle.dumps(data)
    filename = "ali_platform_test.npy"
    mon = Monitor()
    mon.start()

    stop = False
    '''
    Platform.upload_to_storage(filename, data)
    recv_data = pickle.loads(Platform.download_from_storage(filename))
    Logger.info(str(recv_data))
    Platform.delete_from_storage(filename)
    '''
    data = np.array([1.0 for i in range(100000000)])
    data = pickle.dumps(data)
    def cal():
        Logger.info("cal started")
        input_sample = torch.rand(1, 3, 224, 224)
        model = resnet101()
        Logger.info("Model built")
        while True:
            Timeline.start("fwd")
            x = model(input_sample)
            Timeline.end("fwd")
        Logger.info("cal stopped")

    cal_thd = threading.Thread(target=cal)
    cal_thd.start()

    # bandwidth test
    # 400MB data = 400M/ 4 = 100M float
    for k in range(3):
        time.sleep(10)
        continue
        filename = "ali_platform_bw_test.npy"
        Timeline.start("bandwidth-test")
        start_t = time.time()
        data_size = len(data)
        Platform.upload_to_storage(filename, data)
        end_t = time.time()
        bw = data_size / (end_t - start_t)
        Timeline.end("bandwidth-test")
        Logger.info("datasize:{} bandwidth: {:.4f} byte/s".format(data_size, bw))
        Platform.delete_from_storage(filename)
    stop = True
    '''
    # parallel bandwidth
    # todo
    def _bw_test(rank: int):
        filename = "ali_platform_bw_test_{}.npy".format(rank)
        data = np.array([1.0 for i in range(100000000)])
        data = pickle.dumps(data)
        Timeline.start("bandwidth-test-{}".format(rank))
        start_t = time.time()
        data_size = len(data)
        Platform.upload_to_storage(filename, data)
        end_t = time.time()
        bw = data_size / (end_t - start_t)
        Timeline.end("bandwidth-test-{}".format(rank))
        Logger.info("bandwidth-{}: {:.4f} bit/s".format(rank, bw))
        Platform.delete_from_storage(filename)


    # latency test
    for k in range(20):
        filename = "ali_platform_lat_test.npy"
        Timeline.start("latency-test-upload")
        Platform.upload_to_storage(filename, bytes(1))
        Timeline.end("latency-test-upload")

        Timeline.start("latency-test-download")
        _ = Platform.download_from_storage(filename)
        Timeline.end("latency-test-download")

        Timeline.start("latency-test-del")
        Platform.delete_from_storage(filename)
        Timeline.end("latency-test-del")
    '''
    Timeline.report()
    mon.report()