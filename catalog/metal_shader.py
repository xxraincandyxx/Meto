"""Compile each lesson's Metal source once per process, when it first runs."""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=None)
def _library(path):
    return torch.mps.compile_shader(path.read_text(encoding="utf-8"))


def kernel(module_file, entry="solve"):
    """Return a callable MSL entry point from the lesson's adjacent shader."""
    path = Path(module_file).with_suffix(".metal")
    if not path.exists():
        path = path.with_name("solution.metal")
    return getattr(_library(path), entry)
