import logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.DEBUG

LOGFORMAT = '%(log_color)s%(asctime)s%(name)s - %(log_color)s%(filename)s:%(lineno)d ' \
            '- %(log_color)s%(levelname)s - %(log_color)s%(message)s'

logging.root.setLevel(LOG_LEVEL)

formatter = ColoredFormatter(LOGFORMAT)

stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)

logger = logging.getLogger(' Scenario-based Evaluation Tool')
logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)