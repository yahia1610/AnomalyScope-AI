"""Heatmap generation and overlay visualisation for anomaly detection.

This module provides utilities to:

* Normalise raw anomaly maps into a ``[0, 1]`` colour-mapped heatmap.
* Overlay a heatmap on top of the original RGB image with adjustable opacity.
* Draw anomaly contours using OpenCV.
* Assemble side-by-side comparison panels (original / heatmap / overlay).

All functions accept either NumPy arrays or :class:`torch.Tensor` inputs
and gracefully handle NaN / Inf values and shape mismatches.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# OpenCV colormap applied when no explicit colormap is requested.
_DEFAULT_COLORMAP = cv2.COLORMAP_JET


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(data: Any) -> np.ndarray:
    """Convert *data* to a NumPy array, detaching from any autograd graph."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _sanitize(data: np.ndarray) -> np.ndarray:
    """Replace NaN / Inf with finite values and clip to a sane range."""
    arr = np.asarray(data, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _ensure_2d(data: np.ndarray) -> np.ndarray:
    """Squeeze *data* to a 2-D array ``(H, W)``.

    Raises:
        ValueError: If *data* is not 2-D after squeezing.
    """
    arr = np.asarray(data)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] > 1 and arr.shape[-1] > 1:
            # Multi-channel — take mean across the first axis.
            arr = arr.mean(axis=0)
        else:
            raise ValueError(f"Cannot squeeze 3-D array with shape {arr.shape}.")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D heatmap, got shape {arr.shape}.")
    return arr


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert an image array to ``uint8`` RGB/BGR for OpenCV.

    Accepts float arrays in ``[0, 1]``, ``[0, 255]``, or integer arrays.
    """
    arr = _sanitize(image)

    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        # HWC format — possibly single-channel.
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        # CHW format — convert to HWC.
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = arr[..., 0]

    arr = _sanitize(arr)

    if arr.max() <= 1.0 + 1e-6:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _match_spatial_dims(
    heatmap: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Resize *heatmap* to match the spatial dimensions of *reference*.

    Uses nearest-neighbour interpolation to avoid introducing artificial
    gradients.
    """
    if heatmap.shape == reference.shape:
        return heatmap

    target_h, target_w = reference.shape[:2]
    source_h, source_w = heatmap.shape

    if source_h == 0 or source_w == 0:
        raise ValueError("Heatmap has zero spatial extent.")

    # Use cv2.resize for efficiency; fall back to PIL if cv2 fails.
    resized = cv2.resize(
        heatmap.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_heatmap(anomaly_map: Any) -> np.ndarray:
    """Normalise a raw anomaly map to ``[0, 1]`` float32.

    Handles NaN, Inf, and negative values.  If the map is constant,
    all values are set to ``0.0``.

    Args:
        anomaly_map: 2-D (or squeeze-able) array of raw anomaly scores.

    Returns:
        A 2-D ``float32`` array with values in ``[0, 1]``.
    """
    raw = _to_numpy(anomaly_map)
    raw = _sanitize(raw)
    raw = _ensure_2d(raw)

    min_val = float(raw.min())
    max_val = float(raw.max())

    if max_val - min_val < 1e-8:
        # Constant map.
        return np.zeros_like(raw, dtype=np.float32)

    normalised = (raw - min_val) / (max_val - min_val)
    return normalised.astype(np.float32)


def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = _DEFAULT_COLORMAP,
) -> np.ndarray:
    """Apply an OpenCV colormap to a normalised heatmap.

    Args:
        heatmap: 2-D array with values in ``[0, 1]``.
        colormap: An OpenCV colormap identifier (e.g. ``cv2.COLORMAP_JET``).

    Returns:
        A 3-channel ``uint8`` BGR image ``(H, W, 3)``.
    """
    arr = _ensure_2d(_sanitize(_to_numpy(heatmap)))

    # Clip to [0, 1] in case of floating-point edge cases.
    arr = np.clip(arr, 0.0, 1.0)

    # OpenCV expects uint8 in [0, 255].
    uint8_map = (arr * 255.0).astype(np.uint8)
    coloured = cv2.applyColorMap(uint8_map, colormap)

    # cv2.applyColorMap returns BGR; ensure 3 channels.
    if coloured.ndim == 2:
        coloured = cv2.cvtColor(coloured, cv2.COLOR_GRAY2BGR)

    return coloured


def overlay_heatmap(
    image: Any,
    anomaly_map: Any,
    alpha: float = 0.5,
    colormap: int = _DEFAULT_COLORMAP,
    normalize: bool = True,
) -> np.ndarray:
    """Overlay an anomaly heatmap on the original image.

    Args:
        image: Original RGB or BGR image (HWC) as a NumPy array or tensor.
            Float arrays are assumed to be in ``[0, 1]`` if max ≤ 1.
        anomaly_map: 2-D anomaly score map.
        alpha: Blending factor in ``[0, 1]``.  ``0`` = original only,
            ``1`` = heatmap only.
        colormap: OpenCV colormap identifier.
        normalize: If ``True``, normalise the anomaly map before
            colour-mapping.

    Returns:
        A 3-channel ``uint8`` BGR image with the heatmap overlaid.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Convert image to uint8 BGR.
    img_bgr = _to_uint8_image(image)
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    # Prepare heatmap.
    hm = _to_numpy(anomaly_map)
    hm = _sanitize(hm)
    hm = _ensure_2d(hm)

    if normalize:
        hm = normalize_heatmap(hm)

    # Match heatmap resolution to image.
    hm = _match_spatial_dims(hm, img_bgr)

    # Apply colormap → BGR.
    coloured = apply_colormap(hm, colormap=colormap)

    # Blend.
    blended = cv2.addWeighted(coloured, alpha, img_bgr, 1.0 - alpha, 0.0)
    return blended


def draw_contours(
    image: Any,
    anomaly_map: Any,
    threshold: float = 0.5,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    normalize: bool = True,
) -> np.ndarray:
    """Draw anomaly contours on a copy of *image*.

    Args:
        image: Original RGB or BGR image.
        anomaly_map: 2-D anomaly score map.
        threshold: Binarisation threshold applied after normalisation
            (when *normalize* is ``True``) or directly on raw values.
        color: BGR colour for contour lines.
        thickness: Line thickness in pixels.
        normalize: Normalise the anomaly map before thresholding.

    Returns:
        A new ``uint8`` BGR image with contours drawn.
    """
    img_bgr = _to_uint8_image(image).copy()
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    hm = _to_numpy(anomaly_map)
    hm = _sanitize(hm)
    hm = _ensure_2d(hm)

    if normalize:
        hm = normalize_heatmap(hm)

    hm = _match_spatial_dims(hm, img_bgr)

    # Binarise.
    binary = (hm > threshold).astype(np.uint8) * 255

    # Find contours.
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(img_bgr, contours, -1, color, thickness)
    return img_bgr


def create_comparison_panel(
    image: Any,
    anomaly_map: Any,
    alpha: float = 0.5,
    colormap: int = _DEFAULT_COLORMAP,
    threshold: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """Create a three-panel comparison image.

    The returned image has three columns side by side:

    1. **Original** — the input image (no annotation).
    2. **Heatmap** — the colour-mapped anomaly map.
    3. **Overlay** — the heatmap blended over the original image.

    Optionally, anomaly contours are drawn on the overlay panel.

    Args:
        image: Original RGB or BGR image.
        anomaly_map: 2-D anomaly score map.
        alpha: Overlay opacity in ``[0, 1]``.
        colormap: OpenCV colormap identifier.
        threshold: Contour threshold (only used on the overlay panel).
        normalize: Normalise the anomaly map before colour-mapping.

    Returns:
        A single ``uint8`` BGR image of shape ``(H, W*3, 3)``.
    """
    # Convert original to BGR uint8.
    original = _to_uint8_image(image)
    if original.ndim == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Prepare heatmap.
    hm = _to_numpy(anomaly_map)
    hm = _sanitize(hm)
    hm = _ensure_2d(hm)

    if normalize:
        hm = normalize_heatmap(hm)

    hm = _match_spatial_dims(hm, original)

    # Heatmap panel.
    heatmap_panel = apply_colormap(hm, colormap=colormap)

    # Overlay panel (with contours).
    overlay_panel = overlay_heatmap(
        original, hm, alpha=alpha, colormap=colormap, normalize=False
    )
    overlay_panel = draw_contours(
        overlay_panel,
        hm,
        threshold=threshold,
        normalize=False,
    )

    # Concatenate horizontally.
    panel = np.hstack([original, heatmap_panel, overlay_panel])
    return panel


def save_heatmap(
    anomaly_map: Any,
    path: str,
    colormap: int = _DEFAULT_COLORMAP,
) -> None:
    """Save a colour-mapped heatmap to disk.

    Args:
        anomaly_map: 2-D anomaly score map.
        path: Destination file path.
        colormap: OpenCV colormap identifier.
    """
    hm = _to_numpy(anomaly_map)
    hm = _sanitize(hm)
    hm = _ensure_2d(hm)
    hm = normalize_heatmap(hm)

    coloured = apply_colormap(hm, colormap=colormap)
    cv2.imwrite(path, coloured)
    logger.info("Heatmap saved to %s", path)