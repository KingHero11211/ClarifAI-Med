"""Reproducible RNG for ingestion, chunking, and test-question sampling."""

from __future__ import annotations

import random


def set_global_seeds(seed: int = 42) -> None:
    """Seed the stdlib RNG and NumPy. Does not import torch (Phase 0/1 only)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
