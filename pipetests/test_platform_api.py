'''test for Ali coud serverless apis'''
import numpy as np
import pickle
import time
import threading
import json

from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.timeline import Timeline

if __name__ == "__main__":

    '''
    import boto3
    # Let's use Amazon S3
    s3 = boto3.client('s3')
    bucket = "funcpipe"
    # Print out bucket names
    #for bucket in s3.list_buckets():
    #    print(bucket.name)

    data = json.dumps({"test": 1})
    res = s3.list_objects_v2(Bucket = bucket)
    print(res["Contents"])
    exit()
    
    res = s3.put_object(Body = data, Bucket = bucket, Key = "test-obj")
    res = s3.get_object(Bucket = bucket, Key = "test-obj")
    recv_data = res["Body"].read()
    print(type(recv_data))
    print(json.loads(recv_data))
    exit()
    '''

    Platform.use("aws")
    Logger.use_logger(Logger.NATIVE, Logger.INFO, "api_test")
    worker_num = 64
    launch_info = {"function_name":"hybrid_trigger"}
    for i in range(worker_num):
        Logger.info("Launch {}".format(i))
        Platform.invoke(launch_info, asynchronous=True)

    exit()
    s = Platform.get_profiler_info()
    print(s)
    exit()
    Logger.use_logger(Logger.NATIVE, Logger.INFO, "platform_test")
    data = np.array([4, 2, 3, 4, 5, 6])
    data = pickle.dumps(data)
    filename = "ali_platform_test.npy"

    Platform.upload_to_storage(filename, data)
    #Platform.delete_from_storage(filename)
    #exit()
    recv_data = pickle.loads(Platform.download_from_storage(filename))
    Logger.info(str(recv_data))
    Platform.delete_from_storage(filename)

    exit()
    # bandwidth test
    # 400MB data = 400M/ 4 = 100M float
    for k in range(10):
        filename = "ali_platform_bw_test.npy"
        data = np.array([1.0 for i in range(100000000)])
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
    for k in range(100):
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
    Timeline.report()