"""Centralised logging for ARISE.

Uses `rich` for readable colourised console output when available and always
mirrors to a rotating log file inside the run's output directory so every
reduction is fully reproducible/auditable.
"""
from __future__ import annotations

import logging
from pathlib import Path

try:  # rich is a listed dependency, but degrade gracefully if missing
    from rich.logging import RichHandler

    _HAVE_RICH = True
except Exception:  # pragma: no cover
    _HAVE_RICH = False

_LOGGER_NAME = "arise"
_CONFIGURED = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the ARISE logger (or a child of it)."""
    if name and name != _LOGGER_NAME:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure the root ARISE logger once. Safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    # Reset handlers so repeated calls (e.g. per-run) don't duplicate output.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if _HAVE_RICH:
        console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        console.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:  # pragma: no cover
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fileh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fileh.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(fileh)

    _CONFIGURED = True
    return logger
