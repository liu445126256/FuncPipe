'''
Local environment apis - for test only
'''
import os
import multiprocessing
import time

from funcpipe.debugger import Logger
from funcpipe.configs import Config

class LocalPlatform:


    def __init__(self):
        self.subprocess = []

        # todo: add platform settings to a config file
        self.storage_path = "./local_storage/"

    # invoke a local function - always asynchronous
    def invoke(self, launch_info, asynchronous = True) -> None:
        ''' method 1:
        func_path = launch_info["function_name"]
        import_path = '.'.join(func_path.split('.')[:-1])
        func_name = '.'.join(func_path.split('.')[1:])
        module = __import__(import_path)
        eval("module.%s(launch_info)" % func_name)
        '''

        # method 2: use getattr
        func_path = launch_info["function_name"]
        import_path = '.'.join(func_path.split('.')[:-1])
        submodules = func_path.split('.')[:-1][1:]
        func_name = func_path.split('.')[-1]
        module = __import__(import_path, fromlist=submodules)
        func = getattr(module, func_name)

        #func(launch_info)
        Logger.debug("Local invoke: %s" % func_path)
        process = multiprocessing.Process(target=func, args=(launch_info, ))
        self.subprocess.append(process)
        process.start()

    # upload data structure to storage
    def storage_put(self, filename, data: bytes) -> None:
        file_path = self.storage_path + filename
        self._acquire_filelock(file_path)
        with open(file_path, "wb") as f:
            f.write(data)
            f.flush()
        self._release_filelock(file_path)

    # get data structure from storage
    def storage_get(self, filename, timeout = -1) -> bytes:
        file_path = self.storage_path + filename
        start_t = time.time()
        while not os.path.exists(file_path):
            time.sleep(0.001)
            if timeout > 0:
                if time.time() - start_t > timeout: return None
        self._wait_filelock(file_path)
        while True:
            with open(file_path, "rb") as f:
                data = f.read()
            data_len = len(data)
            if data_len == 0:
                Logger.debug("Local platform: read error, retry ...")
                time.sleep(0.001)
                continue
            break
        return data

    # delete file from storage
    def storage_del(self, filename):
        file_path = self.storage_path + filename
        if os.path.exists(file_path):
            self._acquire_filelock(file_path)
            os.remove(file_path)
            self._release_filelock(file_path)

    # check if a file exists
    def file_exists(self, filename):
        file_path = self.storage_path + filename
        return os.path.exists(file_path)

    # return profiler function info
    def get_profiler_info(self):
        info = {}
        info["service_name"] = Config.getvalue("platform-local", "profiler_service_name")
        info["function_name"] = Config.getvalue("platform-local", "profiler_function_name_fmt")
        return info

    # create our own simple file lock since we may debug in Windows environment
    def _acquire_filelock(self, file_path):
        lock_path = file_path + ".lock"
        while os.path.exists(lock_path):
            time.sleep(0.001)
        with open(lock_path, "wb") as f:
            f.write(bytes(1))

    def _release_filelock(self, file_path):
        lock_path = file_path + ".lock"
        if os.path.exists(lock_path):
            os.remove(lock_path)

    def _wait_filelock(self, file_path):
        lock_path = file_path + ".lock"
        while os.path.exists(lock_path):
            time.sleep(0.001)