"""Deterministic seed helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, cudnn_benchmark: bool = False) -> None:
    """Seed random generators for reproducibility.

    Args:
        seed: RNG seed applied to python/numpy/torch.
        cudnn_benchmark: when True, let cuDNN autotune the fastest kernels for
            the (fixed) input size. This trades strict run-to-run determinism for
            speed and is the recommended setting for fixed-shape training on a T4.
            When False, the original deterministic behavior is preserved.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark
