#!/usr/bin/env python
# coding: utf-8

import logging
import os
from typing import Optional
from functools import lru_cache
from src.utils.config import config

def setup_logging(log_file_path: Optional[str] = None, level: Optional[str] = None) -> None:
    log_level = getattr(logging, level or config.LOG_LEVEL)
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

@lru_cache(maxsize=128)
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name) 