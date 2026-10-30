"""Stdlib logging configuration for Phase 0/1 scripts."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once. Safe to call from multiple entry points."""
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(numeric_level)
        return

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
