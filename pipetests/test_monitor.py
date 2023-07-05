'''test for resource monitor'''
import numpy as np
import pickle
import time

from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline
from funcpipe.monitor import Monitor

if __name__ == "__main__":
    Platform.use("ali")
    Logger.use_logger(Logger.HTTP, Logger.INFO, "platform_test")
    mon = Monitor()
    mon.start()

    # bandwidth test
    # 400MB data = 400M/ 4 = 100M float
    for k in range(10):
        filename = "ali_platform_bw_test.npy"
        data = np.array([1.0 for i in range(10000000)])
        data = pickle.dumps(data)
        Timeline.start("bandwidth-test")
        start_t = time.time()
        data_size = len(data)
        Platform.upload_to_storage(filename, data)
        end_t = time.time()
        bw = data_size / (end_t - start_t)
        Timeline.end("bandwidth-test")
        Logger.info("bandwidth: {:.4f} bit/s".format(bw))
        Platform.delete_from_storage(filename)

    mon.stop()
    mon.report()