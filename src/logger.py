import logging
import os
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
FILE_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)d - %(message)s"


def _create_file_handler(name: str, level=logging.INFO) -> TimedRotatingFileHandler:
    log_file = LOG_DIR / f"{name}.log"
    handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(FILE_FORMAT))
    return handler


def _create_stream_handler(level=logging.INFO) -> logging.StreamHandler:
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    return sh


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Return a configured logger.
    Usage:
        logger = get_logger(__name__)
        logger.info("message")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(_create_stream_handler(level))
        logger.addHandler(_create_file_handler(name, level))
        logger.propagate = False

    return logger


__all__ = ["get_logger"]

if __name__ == "__main__":
    logger = get_logger("__app__")
    logger.info("Logging has started.")
    try:
        1 / 0
    except Exception:
        logger.exception("Test exception logged.")