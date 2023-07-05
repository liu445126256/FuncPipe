'''
AWS lambda serverless apis
'''
import boto3
import json
import time

from funcpipe.configs import Config 

class AWSPlatform: 
    def __init__(self):
        try:
            self.serverless_client = boto3.client('lambda')
            self.s3_client = boto3.client('s3')
            self.bucket_name =  Config.getvalue("platform-aws", "bucketName") #'funcpipe'
            self.default_bucket_name = self.bucket_name
        except:
            raise Exception("AWS platform setup failed!")

    # invoke a local function
    def invoke(self, launch_info, asynchronous=False) -> None:
        # service name need to be specified for ali cloud
        self.function_name = launch_info["function_name"]
        if asynchronous:
            self.serverless_client.invoke(FunctionName=self.function_name,
                             InvocationType='Event',
                             Payload=json.dumps(launch_info))
        else:
            self.serverless_client.invoke(FunctionName=self.function_name,
                                          InvocationType='RequestResponse',
                                          Payload=json.dumps(launch_info))

    def storage_put(self, filename, data: bytes) -> None:
        _ = self.s3_client.put_object(Body = data, Bucket = self.bucket_name, Key = filename)
        # print(result.resp.status)

    def storage_get(self, filename, timeout = -1) -> bytes:
        start_t = time.time()
        while not self.file_exists(filename):
            time.sleep(0.1)
            if timeout > 0:
                if time.time() - start_t > timeout: return None
        res = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
        recv_data = res["Body"].read()
        return recv_data

    def storage_list(self):
        res = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
        keys = []
        for item in res["Contents"]: keys.append(item['Key'])
        return keys

    # check if a file exists
    def file_exists(self, filename):
        res = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
        keys = []
        for item in res["Contents"]: keys.append(item['Key'])
        return filename in keys

    def storage_del(self, filename):
        if self.file_exists(filename):
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)

    # return profiler function info
    def get_profiler_info(self):
        info = {}
        info["service_name"] = None#Config.getvalue("platform-aws", "profiler_service_name")
        info["function_name"] = Config.getvalue("platform-aws", "profiler_function_name_fmt")
        return info

    # use another bucket
    def set_bucket_name(self, name = None):
        if name is None:
            self.bucket_name = self.default_bucket_name
        else:
            self.bucket_name = name