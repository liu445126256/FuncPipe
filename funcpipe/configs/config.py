'''global configs'''
import sys
import os
from configparser import ConfigParser

class Config:
    '''configs will be loaded from user config file
    at the first access'''
    config_data = None
    config_file = "funcpipe.conf"
    def __init__(self) -> None:
        raise Exception("This is a static class and should not have any instance.")

    @staticmethod
    def load_from_file():
        # search sys.path for config file
        file_path = None
        for path in sys.path:
            if os.path.exists(path + "/" + Config.config_file):
                file_path = path + "/" + Config.config_file
        if not file_path:
            raise Exception("funcpip.conf file not found, check your path setting!")
        configfile = ConfigParser()
        configfile.read(file_path)
        Config.config_data = configfile

    @staticmethod
    def getvalue(section, key):
        if not Config.config_data: Config.load_from_file()
        return Config.config_data[section][key]
