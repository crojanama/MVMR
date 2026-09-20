"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(payload: Dict[str, Any], checkpoint_path: str) -> None:
    """Persist checkpoint dictionary to disk."""

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load checkpoint dictionary from disk."""

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    data = torch.load(path, map_location=map_location)
    if not isinstance(data, dict):
        raise ValueError("Checkpoint format is invalid.")
    return data
