"""I/O utilities for checkpoints, CSV export, and directory management.

This module provides lightweight wrappers around :mod:`torch` serialisation
and :mod:`pandas` CSV writing so that the rest of the codebase does not need
to repeat boilerplate path handling and device mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch

logger = logging.getLogger(__name__)


def save_checkpoint(model_state: dict[str, Any], path: str | Path) -> None:
    """Persist *model_state* to disk with :func:`torch.save`.

    Parent directories are created automatically if they do not exist.

    Args:
        model_state: A dictionary (typically from ``model.state_dict()``)
            containing all data to serialise.
        path: Destination file path (e.g. ``checkpoints/padim_bottle.pth``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, path)
    logger.info("Checkpoint saved to %s (%.2f MB)", path, path.stat().st_size / 1e6)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint previously saved with :func:`save_checkpoint`.

    Args:
        path: Path to the ``.pth`` file.
        device: Target device for ``map_location`` (default ``"cpu"``).

    Returns:
        The deserialised state dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    logger.info("Checkpoint loaded from %s", path)
    return state


def export_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of result dictionaries to a CSV file.

    Each dictionary becomes one row; keys become column headers.  Parent
    directories are created if needed.

    Args:
        results: A non-empty list of dictionaries with consistent keys.
        path: Destination CSV file path.

    Raises:
        ValueError: If *results* is empty.
    """
    if not results:
        raise ValueError("Cannot export an empty results list to CSV.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    logger.info("Exported %d rows to %s", len(df), path)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved :class:`~pathlib.Path` of the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
