import logging
import sys
from src.utils.config import config

log_file_path = config.LOG_FILE

logging.root.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
