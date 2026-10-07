import logging
import sys
import os
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(
    name: str = "vision_inspection",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configures and returns a consistent logger instance for the application.

    Args:
        name: Name of the logger, typically module name.
        log_level: Standard logging level (INFO, DEBUG, WARNING, ERROR).
        log_file: Optional filepath to persist logs to disk.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler if specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Default root application logger
logger = setup_logger("vision_inspection")
