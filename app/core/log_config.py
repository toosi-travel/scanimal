import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

LOG_FILE = 'scanimal-logstream.log'


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='W0', utc=True)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    
    # Set log level based on environment or default to INFO
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Add handlers
    logger.addHandler(get_file_handler())
    logger.addHandler(get_console_handler())
    
    # Don't propagate to avoid duplicate logs
    logger.propagate = False
    return logger


# Global logger instance
logger = get_logger(__name__)

# Set default logging level for third-party libraries
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('insightface').setLevel(logging.WARNING)
logging.getLogger('onnxruntime').setLevel(logging.WARNING)