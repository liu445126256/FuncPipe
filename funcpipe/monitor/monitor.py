'''A monitor on resources consumption'''
import threading
import time
import numpy as np
import psutil
import os

from funcpipe.debugger import Logger

class Monitor:
    ''''''
    def __init__(self, interval = 0.1):
        self.mon_thread = None
        self.to_stop = False
        self.interval = interval #todo: allow different intervals for different resources
        self.data_log = {"cpu_util(%)": [], "mem_util(MB)":[]}

    def start(self):
        self.mon_thread = threading.Thread(target=self.run)
        self.mon_thread.start()
        Logger.debug("Monitor thread started.")

    def run(self):
        while not self.to_stop:
            # get cpu utilization
            cpu = psutil.cpu_percent()
            self.data_log['cpu_util(%)'].append(cpu)
            # get memory utilization
            mem = psutil.virtual_memory()
            mem_in_MB = mem.used / 1024 / 1024
            self.data_log['mem_util(MB)'].append(mem_in_MB)
            time.sleep(self.interval)

    def stop(self):
        self.to_stop = True

    def report(self) -> None:
        '''Report the statistics of each type of resources'''
        # cpu, memory
        for rs in self.data_log.keys():
            data = self.data_log[rs]
            avg = np.mean(data)
            max = np.max(data)
            s = "Monitor: |{}| avg:{:.4f} | peak:{:.4f} |".format(rs, avg, max)
            Logger.info(s)
        #Logger.info(self.data_log)

    @staticmethod
    def print_my_processs_mem(key):
        Logger.info("{}:{} MB".format(key,str(psutil.Process(os.getpid()).memory_info().rss / 1024.0 / 1024.0)))