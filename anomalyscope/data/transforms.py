"""Image and mask transform pipelines for the MVTec AD dataset.

Anomaly detection models typically use the same deterministic preprocessing
for both training and test images (no random augmentation) so that the
learned normal-distribution statistics are not distorted by augmentation
artefacts.

All images are resized and centre-cropped to a fixed square resolution, then
normalised with ImageNet channel statistics.
"""

from __future__ import annotations

import torch
from torchvision import transforms

# ImageNet channel-wise statistics used by all ``torchvision`` pretrained models.
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 256, is_train: bool = True) -> transforms.Compose:
    """Return the image transform pipeline.

    Both train and test pipelines are identical for anomaly detection:
    ``Resize → CenterCrop → ToTensor → Normalize``.

    Args:
        image_size: Target spatial resolution (height = width).
        is_train: Included for API consistency; currently both branches
            return the same transforms because anomaly detection trains on
            defect-free images without augmentation.

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class _BinarizeMask:
    """Callable transform that binarises a single-channel tensor at a threshold."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a binary mask: 1 where *tensor* ≥ threshold, else 0."""
        return (tensor >= self.threshold).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"


def get_mask_transforms(image_size: int = 256) -> transforms.Compose:
    """Return the ground-truth mask transform pipeline.

    Masks are resized and centre-cropped to match the image resolution, then
    converted to a single-channel float tensor and binarised at 0.5 so that
    interpolation artefacts do not produce fractional labels.

    Args:
        image_size: Target spatial resolution (must match the image pipeline).

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            _BinarizeMask(threshold=0.5),
        ]
    )
