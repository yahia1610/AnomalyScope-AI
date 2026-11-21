"""Abstract base class for all anomaly detection models.

Every model in AnomalyScope AI inherits from :class:`AnomalyModel` and
implements the four core operations:

* **fit** — learn the distribution of normal (defect-free) images.
* **predict** — score a single image and produce a pixel-level anomaly map.
* **save / load** — persist and restore model state for reproducibility.

This contract allows the training loop, evaluation pipeline, and dashboard
to work identically regardless of which detection algorithm is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader


class AnomalyModel(ABC):
    """Interface that every anomaly detection model must implement."""

    @abstractmethod
    def fit(self, dataloader: DataLoader) -> None:
        """Learn the normal data distribution from *dataloader*.

        For embedding-based methods (PaDiM, PatchCore) this typically
        involves a single forward pass over the training set.  For
        reconstruction-based methods (Autoencoder) this runs a full
        training loop.

        Args:
            dataloader: A :class:`~torch.utils.data.DataLoader` that yields
                batches of normal (defect-free) training images.
        """

    @abstractmethod
    def predict(self, image: torch.Tensor) -> tuple[float, np.ndarray]:
        """Compute the anomaly score and anomaly map for a single image.

        Args:
            image: A single preprocessed image tensor of shape
                ``(C, H, W)`` or ``(1, C, H, W)``.

        Returns:
            A tuple ``(anomaly_score, anomaly_map)`` where:

            * *anomaly_score* is a scalar float indicating the image-level
              anomaly severity (higher → more anomalous).
            * *anomaly_map* is a 2-D NumPy array of shape ``(H, W)`` with
              per-pixel anomaly scores.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the model's learned state to *path*.

        Args:
            path: Destination file path (e.g. ``checkpoints/padim_bottle.pth``).
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the model's learned state from *path*.

        Args:
            path: Path to a checkpoint previously written by :meth:`save`.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable model name (e.g. ``"PaDiM"``)."""
