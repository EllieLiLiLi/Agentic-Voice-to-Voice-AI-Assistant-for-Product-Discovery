"""Logging helpers for consistent structured logs."""
from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = "INFO", json: bool = False) -> logging.Logger:
    """Configure root logger with sensible defaults."""

    logger = logging.getLogger()
    logger.setLevel(level)
    handler: logging.Handler
    if json:
        # TODO: Add structured JSON formatter.
        handler = logging.StreamHandler()
    else:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
