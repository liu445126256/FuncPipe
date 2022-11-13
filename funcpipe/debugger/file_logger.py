'''reporting log info to http server'''
import time

from funcpipe.platforms import Platform

class FileLogger:
    # log level
    DEBUG = 0
    INFO = 1
    LOG_LEVELS = [DEBUG, INFO]

    def __init__(self, log_level, log_mark):
        self.log_level = log_level
        self.log_mark = log_mark
        self.info_log = ""

    def debug(self, s: str):
        if self.log_level <= self.DEBUG:
            msg = "DEBUG:" + s
            self.report(msg)

    def info(self, s: str):
        if self.log_level <= self.INFO:
            msg = "INFO:" + s
            self.report(msg)

    def report(self, msg: str):
        log_t = time.time()
        log_msg = "{:.2f}   -   {}  -   {}\n".format(log_t, self.log_mark, msg)
        self.info_log += log_msg

    def finalize(self):
        file_name = "{}_log".format(self.log_mark)
        Platform.upload_to_storage(file_name, self.info_log.encode())

