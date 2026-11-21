"""Threshold calibration for anomaly detection.

This module provides utilities to calibrate a decision threshold from
**normal validation samples only** and apply it to score arrays.  The
central design constraint is **data-leakage prevention**:

* Calibration must use scores from normal (defect-free) samples.
* Test-set scores and labels must never be used for calibration.

Calibration workflow
--------------------

1.  Obtain anomaly scores for a set of **normal validation samples** by
    running the fitted model over those samples.
2.  Call :func:`calibrate_threshold` to compute a threshold.
3.  Call :func:`apply_threshold` to classify new scores as normal or
    anomalous.

Limitations
~~~~~~~~~~~

* **Percentile calibration is heuristic.**  The default 99.5th percentile
  assumes that normal samples rarely exceed this value.  If the model has
  poor normal-sample coverage or the validation set is small, the threshold
  may be too permissive or too aggressive.
* **No optimisation on the test set.**  Because calibration never touches
  test labels, the resulting threshold is not optimised for any particular
  evaluation metric (e.g. F1).  It is a principled but conservative choice.
* **Distribution shift.**  If the validation and test normal distributions
  differ (e.g. different camera, lighting, or preprocessing), the threshold
  may not transfer well.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------


def calibrate_threshold(
    normal_scores: list[float] | np.ndarray | torch.Tensor,
    percentile: float = 99.5,
) -> float:
    """Calibrate an anomaly threshold from normal-sample scores.

    The threshold is set to the *p*-th percentile of the normal-sample
    score distribution.  By construction, approximately ``(100 - p)`` percent
    of normal samples will score at or above the threshold.

    .. warning::
        This function must only be called with scores from **normal**
        (defect-free) validation samples.  Passing anomalous or test-set
        scores would constitute data leakage and invalidate evaluation
        results.

    Args:
        normal_scores: Anomaly scores from normal validation samples.
            Accepts a list of floats, a 1-D NumPy array, or a PyTorch tensor.
        percentile: The percentile to use for threshold selection.
            Must be in ``[0, 100]``.  The default of ``99.5`` means that
            99.5% of normal scores fall below the threshold.

    Returns:
        The calibrated threshold as a ``float``.

    Raises:
        ValueError: If *normal_scores* is empty, contains only NaN/Inf,
            or *percentile* is outside ``[0, 100]``.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(
            f"percentile must be in [0, 100], got {percentile}."
        )

    # Convert to a flat float64 array, sanitising NaN/Inf.
    raw = _to_raw_array(normal_scores)
    if raw.size == 0:
        raise ValueError("normal_scores is empty; cannot calibrate threshold.")

    # If all values are NaN/Inf, there is nothing to calibrate from.
    finite_mask = np.isfinite(raw)
    if not np.any(finite_mask):
        raise ValueError(
            "normal_scores contains no finite values (empty after NaN/Inf removal); "
            "cannot calibrate threshold."
        )

    arr = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    threshold = float(np.percentile(arr, percentile))
    logger.info(
        "Calibrated threshold at %.4f (p=%.2f, n=%d normal samples)",
        threshold,
        percentile,
        arr.size,
    )
    return threshold


# ---------------------------------------------------------------------------
# Threshold application
# ---------------------------------------------------------------------------


def apply_threshold(
    scores: float | list[float] | np.ndarray | torch.Tensor,
    threshold: float,
) -> np.ndarray | np.bool_:
    """Classify anomaly scores as normal (``False``) or anomalous (``True``).

    A sample is classified as **anomalous** when its score is **strictly
    greater than** *threshold*.  Scores equal to the threshold are
    classified as normal.

    Args:
        scores: A single score, a list of scores, a 1-D NumPy array, or
            a PyTorch tensor.
        threshold: The decision threshold (e.g. from
            :func:`calibrate_threshold`).

    Returns:
        - If *scores* is a scalar, a single ``numpy.bool_``.
        - If *scores* is a sequence or array, a 1-D ``numpy.ndarray``
          of ``bool`` with the same length.
    """
    threshold = float(threshold)
    arr = _sanitise_scores(scores)
    predictions = arr > threshold

    # Return a scalar bool for scalar input.
    if np.isscalar(scores) or (
        isinstance(scores, torch.Tensor) and scores.ndim == 0
    ):
        return np.bool_(predictions.item() if predictions.ndim > 0 else predictions)
    return predictions


# ---------------------------------------------------------------------------
# Threshold evaluation utilities
# ---------------------------------------------------------------------------


def compute_false_positive_rate(
    normal_scores: list[float] | np.ndarray | torch.Tensor,
    threshold: float,
) -> float:
    """Compute the false positive rate (FPR) on normal validation samples.

    The FPR is the fraction of normal samples whose score exceeds
    *threshold*.  An ideal calibration should yield an FPR close to
    ``(100 - percentile) / 100``.

    Args:
        normal_scores: Anomaly scores from normal validation samples.
        threshold: The decision threshold.

    Returns:
        FPR in ``[0, 1]``.  Returns ``0.0`` if *normal_scores* is empty.
    """
    arr = _sanitise_scores(normal_scores)
    if arr.size == 0:
        logger.warning(
            "compute_false_positive_rate: empty score array, returning 0.0."
        )
        return 0.0

    threshold = float(threshold)
    fpr = float(np.mean(arr > threshold))
    logger.info(
        "False positive rate: %.4f (n=%d, threshold=%.4f)",
        fpr,
        arr.size,
        threshold,
    )
    return fpr


def evaluate_threshold(
    normal_scores: list[float] | np.ndarray | torch.Tensor,
    anomalous_scores: list[float] | np.ndarray | torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Evaluate a threshold against normal and anomalous score sets.

    This is a **diagnostic** utility for threshold quality assessment.  It
    does *not* calibrate the threshold — it only measures how well a given
    threshold separates the two score distributions.

    Args:
        normal_scores: Scores from normal samples (label = 0).
        anomalous_scores: Scores from anomalous samples (label = 1).
        threshold: The decision threshold to evaluate.

    Returns:
        A dictionary with:

        * ``"fpr"`` — false positive rate on normal samples.
        * ``"fnr"`` — false negative rate on anomalous samples.
        * ``"tpr"`` — true positive rate (recall) on anomalous samples.
        * ``"precision"`` — precision at this threshold.
        * ``"f1"`` — F1 score at this threshold.
        * ``"accuracy"`` — overall accuracy.
    """
    normal_arr = _sanitise_scores(normal_scores)
    anomalous_arr = _sanitise_scores(anomalous_scores)
    threshold = float(threshold)

    if normal_arr.size == 0 and anomalous_arr.size == 0:
        raise ValueError("Both normal and anomalous score sets are empty.")

    # Combine into a single array with labels.
    all_scores = np.concatenate([normal_arr, anomalous_arr])
    all_labels = np.concatenate([
        np.zeros(normal_arr.size, dtype=np.int64),
        np.ones(anomalous_arr.size, dtype=np.int64),
    ])

    predictions = all_scores > threshold

    tp = int(np.sum(predictions & (all_labels == 1)))
    tn = int(np.sum(~predictions & (all_labels == 0)))
    fp = int(np.sum(predictions & (all_labels == 0)))
    fn = int(np.sum(~predictions & (all_labels == 1)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2.0 * precision * tpr / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    logger.info(
        "Threshold evaluation @ %.4f — FPR=%.4f  FNR=%.4f  TPR=%.4f  "
        "Precision=%.4f  F1=%.4f  Accuracy=%.4f",
        threshold,
        fpr,
        fnr,
        tpr,
        precision,
        f1,
        accuracy,
    )

    return {
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr": float(tpr),
        "precision": float(precision),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_raw_array(
    scores: list[float] | np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Convert *scores* to a flat float64 NumPy array **without** sanitising.

    Unlike :func:`_sanitise_scores`, this preserves NaN and Inf values so
    that callers can detect and handle them explicitly.
    """
    if isinstance(scores, torch.Tensor):
        arr = scores.detach().cpu().numpy()
    elif isinstance(scores, np.ndarray):
        arr = scores
    elif isinstance(scores, (list, tuple)):
        arr = np.asarray(scores)
    elif np.isscalar(scores):
        arr = np.asarray([scores])
    else:
        raise ValueError(
            f"Unsupported score type: {type(scores).__name__}."
        )

    return np.asarray(arr, dtype=np.float64).ravel()


def _sanitise_scores(
    scores: list[float] | np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Convert *scores* to a flat float64 NumPy array with NaN/Inf removed.

    Raises:
        ValueError: If the input cannot be converted to a numeric array.
    """
    if isinstance(scores, torch.Tensor):
        arr = scores.detach().cpu().numpy()
    elif isinstance(scores, np.ndarray):
        arr = scores
    elif isinstance(scores, (list, tuple)):
        arr = np.asarray(scores)
    elif np.isscalar(scores):
        arr = np.asarray([scores])
    else:
        raise ValueError(
            f"Unsupported score type: {type(scores).__name__}."
        )

    arr = np.asarray(arr, dtype=np.float64).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr