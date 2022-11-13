import numpy as np

from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline

if __name__ == "__main__":
    Logger.use_logger(Logger.NATIVE, Logger.INFO, "serialization_test")

    Timeline.start("using concatenate")
    param_w = np.zeros((1))
    param_np = np.array([1 for i in range(100000)])
    for i in range(100):
        print(param_np.flatten().shape)
        param_w = np.concatenate((param_w, param_np.flatten()))
    Timeline.end("using concatenate")

    Timeline.start("not using concatenate")
    param_np = np.array([1 for i in range(100000)])
    param_w = np.zeros((1 * param_np.shape[0] * 100))
    for i in range(100):
        param_w[i * 100000 : i * 100000 + 100000] = param_np
    Timeline.end("not using concatenate")