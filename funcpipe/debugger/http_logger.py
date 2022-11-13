'''reporting log info to http server'''
import requests

from funcpipe.configs import Config

class HttpLogger:
    # log level
    DEBUG = 0
    INFO = 1
    LOG_LEVELS = [DEBUG, INFO]

    def __init__(self, log_level, log_mark):
        self.log_level = log_level
        self.log_mark = log_mark
        self.url = Config.getvalue("logger-http", "url")

    def debug(self, s: str):
        if self.log_level <= self.DEBUG:
            msg = "DEBUG:" + s
            self.report(msg)

    def info(self, s: str):
        if self.log_level <= self.INFO:
            msg = "INFO:" + s
            self.report(msg)

    def report(self, msg: str):
        data = {'mark': self.log_mark, 'msg': msg}
        requests.post(self.url, data)

    def finalize(self):
        pass

