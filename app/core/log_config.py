import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

LOG_FILE = 'scanimal-logstream.log'


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='W0',utc=True)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger


logger = get_logger(__name__)