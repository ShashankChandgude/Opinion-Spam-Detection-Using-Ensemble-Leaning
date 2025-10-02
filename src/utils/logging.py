import logging
from functools import lru_cache

@lru_cache(maxsize=128)
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance. Use setup_logging() to configure handlers."""
    return logging.getLogger(name)
