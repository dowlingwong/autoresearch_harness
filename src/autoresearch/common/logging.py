"""Structured logging utilities for the autoresearch package.

All campaign and control-plane code should use ``get_logger(__name__)`` rather
than bare ``print`` statements.  The default formatter emits ISO-8601 timestamps
and the logger name so that log lines from different layers are distinguishable
in a single campaign run.

Usage::

    from autoresearch.common.logging import get_logger

    _log = get_logger(__name__)
    _log.info("trial %s: decision=%s", trial_id, decision)
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_configured: bool = False


def configure_logging(
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATE_FORMAT,
    stream=None,
) -> None:
    """Configure the root logger once.

    Safe to call multiple times; subsequent calls are no-ops so that library
    users who call ``configure_logging()`` from a script entry point do not
    interfere with embedding applications that configure logging themselves.
    """
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a named logger, configuring root logging on first call.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Optional level override for this specific logger.

    Returns:
        A ``logging.Logger`` instance.
    """
    configure_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


# Convenience re-exports so callers only need to import from this module.
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
