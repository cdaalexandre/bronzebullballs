"""Centralized logging configuration for bronzebullballs.

Entrypoints call `setup_logging()` exactly once at startup. All other modules
use `get_logger(__name__)` to obtain a child logger that inherits from the
'bronzebullballs' root logger.

Log file lives at `~/.bronzebullballs/bronzebullballs.log` (outside the repo)
so it is never accidentally committed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_DIR = Path.home() / ".bronzebullballs"
LOG_FILE = LOG_DIR / "bronzebullballs.log"

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the 'bronzebullballs' root logger.

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        A `logging.Logger` instance. No handlers are attached here; handler
        configuration is done once by `setup_logging` on the root logger.
    """
    return logging.getLogger(name)


def setup_logging(
    *,
    level: str = "INFO",
    log_to_file: bool = True,
    quiet: bool = False,
) -> None:
    """Configure the 'bronzebullballs' root logger. Call ONCE from entrypoints.

    Args:
        level: Minimum level for the root logger (e.g. "DEBUG", "INFO").
        log_to_file: If True, attach a FileHandler writing to LOG_FILE.
        quiet: If True, console handler is restricted to WARNING and above.

    Notes:
        - Idempotent: clears any previously attached handlers before adding new
          ones, so calling twice does not duplicate output.
        - File handler always runs at DEBUG, independent of `level` and
          `quiet`. The console is the user-facing surface; the file is the
          forensic record.
    """
    root = logging.getLogger("bronzebullballs")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING if quiet else root.level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_to_file:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except OSError:
            root.warning("Could not create log file at %s -- console only", LOG_FILE)
