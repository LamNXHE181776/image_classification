"""Centralised logging setup for the classify package.

Usage::

    from classify.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Training started.")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and return a consistently formatted logger.

    Args:
        name: Logger name, typically ``__name__``.
        log_file: Optional absolute path for a file handler.
        level: Logging level (default: ``logging.INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the module is re-imported.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid double output.
    logger.propagate = False

    return logger
