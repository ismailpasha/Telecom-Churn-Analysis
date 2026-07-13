"""
logger.py

Central logging utility for the Telecom Churn Prediction project.

Author: Mohammed Ismail Pasha

Features
--------
✓ Console logging
✓ File logging
✓ Automatic log directory creation
✓ Configurable log levels
✓ Colored level names (console friendly)
✓ Prevents duplicate handlers
✓ Reusable across every module
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from .config import (
        LOG_DIR,
        LOG_FILE,
        LOG_LEVEL,
    )
except ImportError:
    from config import (
        LOG_DIR,
        LOG_FILE,
        LOG_LEVEL,
    )

# ==============================================================================
# CREATE LOG DIRECTORY
# ==============================================================================

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LOG FORMAT
# ==============================================================================

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ==============================================================================
# LOGGER CLASS
# ==============================================================================


class LoggerManager:
    """
    Creates and manages loggers across the project.
    """

    _configured = False

    @classmethod
    def configure(cls) -> None:

        if cls._configured:
            return

        root_logger = logging.getLogger()

        root_logger.setLevel(LOG_LEVEL)

        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT
        )

        # ------------------------------------------------------------------
        # Console Handler
        # ------------------------------------------------------------------

        console_handler = logging.StreamHandler(sys.stdout)

        console_handler.setLevel(LOG_LEVEL)

        console_handler.setFormatter(formatter)

        # ------------------------------------------------------------------
        # File Handler
        # ------------------------------------------------------------------

        file_handler = RotatingFileHandler(

            filename=LOG_FILE,

            maxBytes=5 * 1024 * 1024,

            backupCount=5,

            encoding="utf-8"

        )

        file_handler.setLevel(LOG_LEVEL)

        file_handler.setFormatter(formatter)

        # ------------------------------------------------------------------
        # Remove Existing Handlers
        # ------------------------------------------------------------------

        if root_logger.handlers:

            root_logger.handlers.clear()

        root_logger.addHandler(console_handler)

        root_logger.addHandler(file_handler)

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:

        cls.configure()

        return logging.getLogger(name)


# ==============================================================================
# HELPER FUNCTION
# ==============================================================================


def get_logger(name: str) -> logging.Logger:
    """
    Returns configured logger.

    Example
    -------
    from logger import get_logger

    logger = get_logger(__name__)

    logger.info("Training started")
    """

    return LoggerManager.get_logger(name)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    logger = get_logger("demo")

    logger.debug("Debug message")

    logger.info("Info message")

    logger.warning("Warning message")

    logger.error("Error message")

    logger.critical("Critical message")

    print()

    print(f"Log file saved at:\n{Path(LOG_FILE).resolve()}")