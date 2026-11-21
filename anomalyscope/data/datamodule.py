"""DataModule that wires MVTecDataset to PyTorch DataLoaders.

The :class:`DataModule` reads all data-related settings from the project
configuration namespace (see :mod:`anomalyscope.utils.config`) and exposes
ready-to-iterate train and test :class:`~torch.utils.data.DataLoader`
instances.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from torch.utils.data import DataLoader

from anomalyscope.data.dataset import MVTecDataset
from anomalyscope.data.transforms import get_mask_transforms, get_transforms

logger = logging.getLogger(__name__)


class DataModule:
    """Convenience wrapper that creates datasets and dataloaders from config.

    Typical usage::

        cfg = load_config("configs/padim.yaml")
        dm = DataModule(cfg)
        dm.setup()
        for batch in dm.train_dataloader():
            ...

    Args:
        config: A namespace (or any object with attribute access) containing at
            least ``dataset_root``, ``category``, ``image_size``, ``batch_size``,
            and ``num_workers``.
    """

    def __init__(self, config: SimpleNamespace | Any) -> None:
        self.config = config
        self._train_dataset: MVTecDataset | None = None
        self._test_dataset: MVTecDataset | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Instantiate train and test :class:`MVTecDataset` objects.

        This method is separated from ``__init__`` so that the caller can
        control *when* the potentially expensive filesystem scan happens.
        """
        image_size: int = getattr(self.config, "image_size", 256)
        dataset_root: str = self.config.dataset_root
        category: str = self.config.category

        train_transform = get_transforms(image_size=image_size, is_train=True)
        test_transform = get_transforms(image_size=image_size, is_train=False)
        mask_transform = get_mask_transforms(image_size=image_size)

        self._train_dataset = MVTecDataset(
            root=dataset_root,
            category=category,
            split="train",
            transform=train_transform,
            mask_transform=mask_transform,
        )

        self._test_dataset = MVTecDataset(
            root=dataset_root,
            category=category,
            split="test",
            transform=test_transform,
            mask_transform=mask_transform,
        )

        logger.info(
            "DataModule setup complete — train: %d samples, test: %d samples",
            len(self._train_dataset),
            len(self._test_dataset),
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        """Return a :class:`~torch.utils.data.DataLoader` over the training set.

        Images are shuffled each epoch.  ``pin_memory`` is enabled when the
        configured device is not CPU to speed up host-to-device transfers.

        Raises:
            RuntimeError: If :meth:`setup` has not been called yet.
        """
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader.")

        return DataLoader(
            self._train_dataset,
            batch_size=getattr(self.config, "batch_size", 32),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 0),
            pin_memory=getattr(self.config, "device", "cpu") != "cpu",
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a :class:`~torch.utils.data.DataLoader` over the test set.

        Images are **not** shuffled so that evaluation order is deterministic.

        Raises:
            RuntimeError: If :meth:`setup` has not been called yet.
        """
        if self._test_dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader.")

        return DataLoader(
            self._test_dataset,
            batch_size=getattr(self.config, "batch_size", 32),
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 0),
            pin_memory=getattr(self.config, "device", "cpu") != "cpu",
            drop_last=False,
        )
