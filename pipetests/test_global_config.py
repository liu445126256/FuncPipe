import sys
sys.path.append('../')

from funcpipe.configs import Config


if __name__ == "__main__":
    print(Config.getvalue("logger-http", "url"))