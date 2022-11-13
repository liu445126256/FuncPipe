import time
import numpy as np

from funcpipe.debugger import Logger

# todo: consider using standard chrome tracing format
class Timeline:
    '''A timeline logger
    the timeline logger reports at the INFO level of the global logger
    thus the global logger has to be initialized before any timeline call'''

    log_data = {}
    def __init__(self) -> None:
        raise Exception("This is a static class and should not have any instance.")

    @staticmethod
    def start(uid: str, silent = False):
        Timeline.check_logger()
        if uid not in Timeline.log_data.keys():
            Timeline.log_data[uid] = {'s': -1, 'history':[]}
        Timeline.log_data[uid]['s'] = time.time()
        if not silent: Logger.info("Timeline: {} start here.".format(uid))

    @staticmethod
    def end(uid: str, silent = False):
        Timeline.check_logger()
        if uid not in Timeline.log_data.keys() or Timeline.log_data[uid]['s'] == -1:
            Logger.info("Timeline: invalid end point!")
            return
        consumed_time = time.time() - Timeline.log_data[uid]['s']
        Timeline.log_data[uid]['s'] = -1
        Timeline.log_data[uid]['history'].append(consumed_time)
        if not silent: Logger.info("Timeline: {} - {:.4f}s".format(uid, consumed_time))

    @staticmethod
    def report():
        '''Report the time statistics of each uid'''
        # count, avg, std
        for uid in Timeline.log_data.keys():
            entries = Timeline.log_data[uid]["history"]
            count = len(entries)
            avg = np.mean(entries)
            std = np.std(entries)
            maxx = np.max(entries)
            minn = np.min(entries)
            s = "Timeline: |{}| count:{} | avg:{:.4f}s | std:{:.4f}s | max:{:.4f}s | min:{:.4f}s |".format(uid, count,
                                                                                                           avg, std,
                                                                                                           maxx, minn)
            Logger.info(s)

    @staticmethod
    def clear():
        Timeline.log_data = {}

    @staticmethod
    def check_logger():
        if Logger.log == None:
            raise Exception("Logger has to be initialized before any timeline call.")