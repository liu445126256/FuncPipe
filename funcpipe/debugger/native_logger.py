import logging


class NativeLogger:
    # log level
    DEBUG = 0
    INFO = 1
    LOG_LEVELS = [DEBUG, INFO]

    def __init__(self, log_level, log_mark):
        if log_level == self.DEBUG:
            logging.basicConfig(level=logging.DEBUG)
        elif log_level == self.INFO:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(log_mark)

    def debug(self, s: str):
        self.logger.debug("\t\t" + s)

    def info(self, s: str):
        self.logger.info("\t\t" + s)

    def finalize(self):
        pass


