import logging
import logging.handlers
import sys
from pathlib import Path
from config.settings import LOG_FILE, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT, DEBUG

Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('ASSISTENTE_SAUDE_FEMININA')
logger.setLevel(getattr(logging, LOG_LEVEL))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

if DEBUG:
    logger.setLevel(logging.DEBUG)

def get_logger(name: str = None) -> logging.Logger:

        return logging.getLogger(name)
    return logger
