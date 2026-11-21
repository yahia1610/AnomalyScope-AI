"""Evaluation metrics for anomaly detection on the MVTec AD benchmark.

Provides image-level AUROC, pixel-level AUROC, and the Per-Region Overlap
(PRO) metric with its integral variant AUPRO, following the standard MVTec
AD evaluation methodology.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from scipy.ndimage import label as _cc_label
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a NumPy array from either a tensor or array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sanitize_scores(scores: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with finite values and ensure float64."""
    arr = np.asarray(scores, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=-1.0, posinf=1e9, neginf=-1e9)
    return arr


def _collect_results(
    model: Any,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[list[float], list[np.ndarray], list[int], list[np.ndarray], list[str]]:
    """Run inference over *dataloader* and collect per-sample results.

    Returns
    -------
    scores, maps, labels, masks, defect_types
        Parallel lists with one entry per sample in iteration order.
    """
    scores: list[float] = []
    maps: list[np.ndarray] = []
    labels: list[int] = []
    masks: list[np.ndarray] = []
    defect_types: list[str] = []

    # Set model to eval mode if possible (best-effort, not all models expose this).
    for attr in ("net", "backbone"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "eval"):
            sub.eval()
            break

    for batch in dataloader:
        images: torch.Tensor = batch["image"]
        batch_labels: torch.Tensor = batch["label"]
        batch_masks: torch.Tensor = batch["mask"]
        batch_defect: list[str] = batch["defect_type"]  # type: ignore[assignment]

        for i in range(images.shape[0]):
            img = images[i]  # (C, H, W)
            score, amap = model.predict(img)
            scores.append(float(score))
            maps.append(np.asarray(amap, dtype=np.float32))
            labels.append(int(batch_labels[i].item()))
            masks.append(
                batch_masks[i].squeeze(0).numpy().astype(np.float32)
                if batch_masks.ndim == 4
                else batch_masks[i].numpy().astype(np.float32)
            )
            defect_types.append(str(batch_defect[i]))

    return scores, maps, labels, masks, defect_types


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_image_auroc(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> float:
    """Compute image-level AUROC.

    Args:
        scores: Per-sample anomaly scores (higher → more anomalous).
        labels: Binary ground-truth labels (0 = normal, 1 = anomalous).

    Returns:
        AUROC value in ``[0, 1]``.  Returns ``0.5`` if only one class
        is present.
    """
    s = _sanitize_scores(np.asarray(scores, dtype=np.float64))
    y = np.asarray(labels, dtype=np.int64)

    if len(np.unique(y)) < 2:
        logger.warning(
            "compute_image_auroc: only one class present, returning 0.5."
        )
        return 0.5

    return float(roc_auc_score(y, s))


def compute_pixel_auroc(
    anomaly_maps: list[np.ndarray] | np.ndarray,
    masks: list[np.ndarray] | np.ndarray,
) -> float:
    """Compute pixel-level AUROC.

    All pixels across all images are pooled into a single binary
    classification problem.

    Args:
        anomaly_maps: List of 2-D arrays ``(H, W)`` or a 3-D array
            ``(N, H, W)`` of per-pixel anomaly scores.
        masks: List of 2-D arrays ``(H, W)`` or a 3-D array ``(N, H, W)``
            of binary ground-truth masks (0 / 1).

    Returns:
        Pixel-level AUROC in ``[0, 1]``.  Returns ``0.5`` if only one
        class is present across all pixels.
    """
    # Stack into (N, H, W) arrays.
    if isinstance(anomaly_maps, list):
        amap_arr = np.stack([np.asarray(a, dtype=np.float64) for a in anomaly_maps])
    else:
        amap_arr = np.asarray(anomaly_maps, dtype=np.float64)

    if isinstance(masks, list):
        mask_arr = np.stack([np.asarray(m, dtype=np.float64) for m in masks])
    else:
        mask_arr = np.asarray(masks, dtype=np.float64)

    # Ensure shapes match.
    if amap_arr.shape != mask_arr.shape:
        # Resize masks to match anomaly maps (nearest neighbour).
        mask_arr = _resize_masks(mask_arr, amap_arr.shape[-2:])

    amap_flat = amap_arr.ravel()
    mask_flat = (mask_arr.ravel() > 0.5).astype(np.int64)

    # Replace non-finite scores.
    amap_flat = _sanitize_scores(amap_flat)

    if len(np.unique(mask_flat)) < 2:
        logger.warning(
            "compute_pixel_auroc: only one class present, returning 0.5."
        )
        return 0.5

    return float(roc_auc_score(mask_flat, amap_flat))


def _resize_masks(
    masks: np.ndarray, target_hw: tuple[int, int]
) -> np.ndarray:
    """Resize a stack of masks to *target_hw* using nearest-neighbour."""
    from PIL import Image

    n, h, w = masks.shape
    th, tw = target_hw
    out = np.empty((n, th, tw), dtype=np.float32)
    for i in range(n):
        img = Image.fromarray((masks[i] * 255).astype(np.uint8), mode="L")
        img = img.resize((tw, th), resample=Image.Resampling.NEAREST)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0
    return out


def compute_pro(
    anomaly_maps: list[np.ndarray],
    masks: list[np.ndarray],
    max_fpr: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """Compute the Per-Region Overlap (PRO) metric and its integral AUPRO.

    Following the MVTec AD evaluation methodology:

    1.  For a set of thresholds spanning the anomaly-score range, binarise
        each anomaly map.
    2.  For each ground-truth connected component (region), compute its
        recall (fraction of its pixels detected as anomalous).
    3.  The PRO value at a threshold is the average recall across all
        regions.
    4.  The false-positive rate is also computed at each threshold.
    5.  AUPRO is the area under the PRO-vs-FPR curve, integrated over
        ``[0, max_fpr]``.

    Args:
        anomaly_maps: List of 2-D anomaly maps ``(H, W)``.
        masks: List of 2-D binary masks ``(H, W)``.
        max_fpr: Upper bound of the FPR integration range (default 0.3).
        num_thresholds: Number of thresholds to evaluate.

    Returns:
        AUPRO value in ``[0, 1]``.  Returns ``0.0`` if no valid masks
        are provided.
    """
    if len(anomaly_maps) != len(masks):
        raise ValueError(
            f"anomaly_maps and masks length mismatch: "
            f"{len(anomaly_maps)} vs {len(masks)}."
        )

    # Filter to samples that have at least one anomalous pixel.
    valid_indices = [
        i for i, m in enumerate(masks) if np.asarray(m).max() > 0.5
    ]
    if not valid_indices:
        logger.warning("compute_pro: no anomalous masks found, returning 0.0.")
        return 0.0

    # Collect all connected components across all valid masks.
    # Each component is stored as a boolean array over its image.
    components: list[np.ndarray] = []  # list of boolean (H, W) arrays

    for idx in valid_indices:
        mask = np.asarray(masks[idx], dtype=np.float32)
        # Resize mask to match anomaly map if needed.
        amap = np.asarray(anomaly_maps[idx], dtype=np.float32)
        if mask.shape != amap.shape:
            mask = _resize_single_mask(mask, amap.shape[-2:])

        binary_mask = mask > 0.5
        labelled, num_features = _cc_label(binary_mask)
        for region_id in range(1, num_features + 1):
            component = labelled == region_id
            if component.any():
                components.append(component)

    if not components:
        logger.warning(
            "compute_pro: connected components extraction failed, returning 0.0."
        )
        return 0.0

    # Determine threshold range from all anomaly maps.
    all_scores = np.concatenate(
        [np.asarray(a).ravel() for a in anomaly_maps]
    )
    all_scores = _sanitize_scores(all_scores)
    min_val, max_val = float(all_scores.min()), float(all_scores.max())
    if max_val <= min_val:
        logger.warning(
            "compute_pro: degenerate score range [%.4f, %.4f], returning 0.0."
            % (min_val, max_val)
        )
        return 0.0

    thresholds = np.linspace(min_val, max_val, num_thresholds)

    pro_values: list[float] = []
    fpr_values: list[float] = []

    # Pre-compute total non-anomalous pixels across all images.
    total_normal_pixels = 0
    for idx in valid_indices:
        mask = np.asarray(masks[idx], dtype=np.float32)
        amap = np.asarray(anomaly_maps[idx], dtype=np.float32)
        if mask.shape != amap.shape:
            mask = _resize_single_mask(mask, amap.shape[-2:])
        total_normal_pixels += int((mask <= 0.5).sum())

    for thresh in thresholds:
        # Per-region recall.
        recalls: list[float] = []
        fp_count = 0

        for idx in valid_indices:
            amap = np.asarray(anomaly_maps[idx], dtype=np.float32)
            mask = np.asarray(masks[idx], dtype=np.float32)
            if mask.shape != amap.shape:
                mask = _resize_single_mask(mask, amap.shape[-2:])

            pred_binary = amap > thresh
            normal_mask = mask <= 0.5
            fp_count += int((pred_binary & normal_mask).sum())

            labelled, num_features = _cc_label(mask > 0.5)
            for region_id in range(1, num_features + 1):
                region = labelled == region_id
                region_size = int(region.sum())
                if region_size == 0:
                    continue
                hits = int((pred_binary & region).sum())
                recalls.append(hits / region_size)

        if recalls:
            pro_val = float(np.mean(recalls))
        else:
            pro_val = 0.0
        pro_values.append(pro_val)

        if total_normal_pixels > 0:
            fpr = fp_count / total_normal_pixels
        else:
            fpr = 0.0
        fpr_values.append(fpr)

    pro_arr = np.asarray(pro_values, dtype=np.float64)
    fpr_arr = np.asarray(fpr_values, dtype=np.float64)

    # Compute area under PRO-vs-FPR curve up to max_fpr.
    # If all FPR values are identical (e.g. perfect model → all zero),
    # the curve is a flat line and the area is pro_value * max_fpr.
    if np.all(fpr_arr == fpr_arr[0]):
        aupro = float(pro_arr.mean() * max_fpr)
    else:
        # np.interp requires the x-coordinates to be in ascending order.
        # As the threshold increases, FPR decreases, so fpr_arr is generally
        # descending — sort both arrays together before interpolating.
        order = np.argsort(fpr_arr, kind="stable")
        fpr_sorted = fpr_arr[order]
        pro_sorted = pro_arr[order]

        # Interpolate FPR to a regular grid for stable integration.
        fpr_grid = np.linspace(0.0, max_fpr, 100)
        pro_interp = np.interp(fpr_grid, fpr_sorted, pro_sorted)
        aupro = auc(fpr_grid, pro_interp)

    # Normalise to [0, 1] by dividing by the integration range.
    aupro /= max_fpr
    return float(min(aupro, 1.0))


def _resize_single_mask(
    mask: np.ndarray, target_hw: tuple[int, int]
) -> np.ndarray:
    """Resize a single 2-D mask to *target_hw* using nearest neighbour."""
    from PIL import Image

    th, tw = target_hw
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    img = img.resize((tw, th), resample=Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# High-level evaluation driver
# ---------------------------------------------------------------------------


def evaluate_model(
    model: Any,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Evaluate *model* on the test set yielded by *dataloader*.

    This function:
    1. Runs inference over the entire test set.
    2. Computes image-level AUROC, pixel-level AUROC, and AUPRO.
    3. Returns a structured result dictionary.

    Args:
        model: An :class:`~anomalyscope.models.base.AnomalyModel` instance
            that has already been fitted or loaded.
        dataloader: A ``DataLoader`` over the **test** split.  Each batch
            must contain the keys ``"image"``, ``"label"``, ``"mask"``,
            and ``"defect_type"`` (as provided by :class:`MVTecDataset`).
        device: Torch device string or :class:`torch.device`.

    Returns:
        A dictionary with the following keys:

        * ``"image_auroc"`` — image-level AUROC (float).
        * ``"pixel_auroc"`` — pixel-level AUROC (float).
        * ``"aupro"`` — AUPRO metric (float).
        * ``"per_defect_results"`` — per-defect-type breakdown
          (dict of defect_type → image AUROC).
        * ``"num_samples"`` — total number of test samples.
    """
    scores, maps, labels, masks, defect_types = _collect_results(
        model, dataloader, device=device
    )

    if not scores:
        logger.warning("evaluate_model: no samples collected, returning zeros.")
        return {
            "image_auroc": 0.0,
            "pixel_auroc": 0.0,
            "aupro": 0.0,
            "per_defect_results": {},
            "num_samples": 0,
        }

    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)

    image_auroc = compute_image_auroc(scores_arr, labels_arr)
    pixel_auroc = compute_pixel_auroc(maps, masks)
    aupro = compute_pro(maps, masks)

    # Per-defect breakdown (image-level AUROC per defect type).
    defect_set = sorted(set(defect_types) - {"good"})
    per_defect: dict[str, float] = {}
    for defect in defect_set:
        idx = [i for i, d in enumerate(defect_types) if d == defect]
        defect_scores = scores_arr[idx]
        defect_labels = labels_arr[idx]
        # Each defect type should have some normals mixed in for AUROC.
        # If only one class, skip.
        if len(np.unique(defect_labels)) < 2:
            per_defect[defect] = 0.5
        else:
            per_defect[defect] = float(
                roc_auc_score(defect_labels, defect_scores)
            )

    return {
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "aupro": aupro,
        "per_defect_results": per_defect,
        "num_samples": len(scores),
    }