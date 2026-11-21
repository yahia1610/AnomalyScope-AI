"""PyTorch Dataset for the MVTec Anomaly Detection (AD) benchmark.

The MVTec AD dataset is organised as::

    <root>/<category>/train/good/        — defect-free training images
    <root>/<category>/test/good/         — defect-free test images
    <root>/<category>/test/<defect>/     — defective test images
    <root>/<category>/ground_truth/<defect>/  — pixel-level masks

This module provides :class:`MVTecDataset`, a standard
:class:`torch.utils.data.Dataset` that follows this layout and returns a
dictionary per sample with the image, its anomaly mask (or a zero tensor for
normal samples), the binary label, the source file path, and the defect type
string.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

logger = logging.getLogger(__name__)

# Image extensions recognised when scanning directories.
_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class MVTecDataset(Dataset):
    """Dataset wrapper for a single MVTec AD category and split.

    Args:
        root: Path to the top-level MVTec AD directory (the folder that
            contains the per-category sub-directories).
        category: Category name (e.g. ``"bottle"``, ``"hazelnut"``).
        split: ``"train"`` or ``"test"``.
        transform: Torchvision transform pipeline applied to RGB images.
        mask_transform: Torchvision transform pipeline applied to
            ground-truth masks.  If ``None`` the raw mask is not loaded
            and a zero tensor is returned for every sample.

    Raises:
        FileNotFoundError: If the category directory or the requested split
            directory does not exist.
        ValueError: If *split* is not ``"train"`` or ``"test"``.
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        transform: T.Compose | None = None,
        mask_transform: T.Compose | None = None,
    ) -> None:
        super().__init__()

        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        category_dir = self.root / category
        if not category_dir.is_dir():
            raise FileNotFoundError(
                f"Category directory not found: {category_dir}. "
                f"Available categories: "
                f"{sorted(p.name for p in self.root.iterdir() if p.is_dir())}"
            )

        split_dir = category_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Build the sample list: (image_path, mask_path | None, label, defect_type)
        self._samples: list[tuple[Path, Path | None, int, str]] = []
        self._build_samples(split_dir, category_dir)

        logger.info(
            "MVTecDataset | category=%s  split=%s  samples=%d",
            category,
            split,
            len(self._samples),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_samples(self, split_dir: Path, category_dir: Path) -> None:
        """Scan the split directory and populate ``self._samples``."""
        gt_root = category_dir / "ground_truth"

        # Pre-index ground-truth masks by leading numeric index per defect.
        # This supports non-standard filenames where the image and its mask
        # share a leading index but differ in trailing suffix, e.g.
        # ``000-94.png`` (image) vs ``000_mask-67.png`` (mask).
        gt_index: dict[str, dict[str, Path]] = {}
        if gt_root.is_dir():
            for gt_defect_dir in gt_root.iterdir():
                if not gt_defect_dir.is_dir():
                    continue
                index_map: dict[str, Path] = {}
                for f in gt_defect_dir.iterdir():
                    if not f.is_file() or "_mask" not in f.name:
                        continue
                    m = re.match(r"^(\d+)", f.name)
                    if m:
                        index_map.setdefault(m.group(1), f)
                gt_index[gt_defect_dir.name] = index_map

        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            is_normal = defect_type == "good"

            for img_path in sorted(defect_dir.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue

                label = 0 if is_normal else 1

                mask_path: Path | None = None
                if not is_normal and gt_root.is_dir():
                    gt_defect_dir = gt_root / defect_type
                    # 1. Standard MVTec naming: <stem>_mask.<ext>.
                    candidate = gt_defect_dir / f"{img_path.stem}_mask{img_path.suffix}"
                    if candidate.exists():
                        mask_path = candidate
                    else:
                        # 2. Try common alternative extensions.
                        for ext in _IMAGE_EXTENSIONS:
                            alt = gt_defect_dir / f"{img_path.stem}_mask{ext}"
                            if alt.exists():
                                mask_path = alt
                                break
                        else:
                            # 3. Leading-index fallback for non-standard names.
                            m = re.match(r"^(\d+)", img_path.stem)
                            if m and gt_defect_dir.is_dir():
                                mask_path = gt_index.get(defect_type, {}).get(
                                    m.group(1)
                                )

                self._samples.append((img_path, mask_path, label, defect_type))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in this dataset split."""
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and return a single sample.

        Returns:
            A dictionary with the following keys:

            * ``"image"`` — the transformed RGB image tensor.
            * ``"mask"`` — the transformed binary mask tensor (C×H×W) or a
              zero tensor of the same spatial size for normal samples.
            * ``"label"`` — ``0`` for normal, ``1`` for anomalous.
            * ``"path"`` — the original image file path as a string.
            * ``"defect_type"`` — defect sub-type string (``"good"`` for
              normal samples).
        """
        img_path, mask_path, label, defect_type = self._samples[index]

        # Load image.
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Load or synthesise mask.
        if mask_path is not None and self.mask_transform is not None:
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
        else:
            # Zero mask — shape must match the (transformed) image spatial dims.
            if isinstance(image, torch.Tensor):
                h, w = image.shape[-2], image.shape[-1]
            else:
                h, w = image.size[1], image.size[0]
            mask = torch.zeros(1, h, w, dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "path": str(img_path),
            "defect_type": defect_type,
        }
