'''
Ali cloud serverless apis
'''
import time
import json

import fc2
import oss2

from funcpipe.configs import Config

class AliPlatform:
    def __init__(self):
        #Aliyun account
        try:
            self.endpoint = Config.getvalue("platform-ali", "endpoint") #'https://1333298343766298.cn-hangzhou.fc.aliyuncs.com'
            self.accessKeyID = Config.getvalue("platform-ali", "accessKeyID") #'LTAI5tRs9g2i4XPHJ15dCZhD'
            self.accessKeySecret = Config.getvalue("platform-ali", "accessKeySecret") #'hJqoO5yvGJKws7sHxQ3n3w2t17UKAv'

            #oss infos
            self.oss_endPoint = Config.getvalue("platform-ali", "ossEndPoint") #'http://oss-cn-hangzhou-internal.aliyuncs.com'
            #self.oss_endPoint = "oss-cn-hangzhou.aliyuncs.com"
            self.bucketName =  Config.getvalue("platform-ali", "bucketName") #'funcpipe'
            self.auth = oss2.Auth(self.accessKeyID, self.accessKeySecret)
            self.bucket = oss2.Bucket(self.auth, self.oss_endPoint, self.bucketName)
            self.bucket_path = Config.getvalue("platform-ali", "bucketPath") #"./"
        except:
            raise Exception("Ali platform config file loading failed!")
        # serverless
        self.serverless_client = fc2.Client(
            endpoint = self.endpoint,
            accessKeyID = self.accessKeyID,
            accessKeySecret = self.accessKeySecret)

    # invoke a local function
    def invoke(self, launch_info, asynchronous = False) -> None:
        # service name need to be specified for ali cloud
        self.service_name = launch_info["service_name"]
        self.function_name = launch_info["function_name"]
        if asynchronous:
            self.serverless_client.invoke_function(self.service_name,
                                               self.function_name,
                                               payload=json.dumps(launch_info),
                                               headers={'x-fc-invocation-type': 'Async'})
        else:
            self.serverless_client.invoke_function(self.service_name,
                               self.function_name,
                               payload =json.dumps(launch_info))

    def storage_put(self, filename, data: bytes) -> None:
        file_path = self.bucket_path + filename
        result = self.bucket.put_object(file_path, bytes(data))
        #print(result.resp.status)

    def storage_get(self, filename, timeout = -1) -> bytes:
        file_path = self.bucket_path + filename
        start_t = time.time()
        while not self.bucket.object_exists(file_path):
            time.sleep(0.001)
            if timeout > 0:
                if time.time() - start_t > timeout: return None
        return self.bucket.get_object(file_path).read()

    def storage_list(self):
        return [sbj.key for sbj in self.bucket.list_objects_v2().object_list]

    # check if a file exists
    def file_exists(self, filename):
        file_path = self.bucket_path + filename
        return self.bucket.object_exists(file_path)

    def storage_del(self, filename):
        file_path = self.bucket_path + filename
        if self.bucket.object_exists(file_path):
            self.bucket.delete_object(file_path)

    # return profiler function info
    def get_profiler_info(self):
        info = {}
        info["service_name"] = Config.getvalue("platform-ali", "profiler_service_name")
        info["function_name"] = Config.getvalue("platform-ali", "profiler_function_name_fmt")
        return info