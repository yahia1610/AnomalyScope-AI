"""Evaluation metrics and utilities for AnomalyScope AI."""

from anomalyscope.evaluation.calibration import (
    apply_threshold,
    calibrate_threshold,
    compute_false_positive_rate,
    evaluate_threshold,
)
from anomalyscope.evaluation.heatmap import (
    apply_colormap,
    create_comparison_panel,
    draw_contours,
    normalize_heatmap,
    overlay_heatmap,
)
from anomalyscope.evaluation.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro,
    evaluate_model,
)
from anomalyscope.evaluation.runner import (
    SUPPORTED_CATEGORIES,
    ExperimentResult,
    ExperimentRunner,
    run_experiment,
)

__all__ = [
    "SUPPORTED_CATEGORIES",
    "ExperimentResult",
    "ExperimentRunner",
    "apply_colormap",
    "apply_threshold",
    "calibrate_threshold",
    "compute_false_positive_rate",
    "compute_image_auroc",
    "compute_pixel_auroc",
    "compute_pro",
    "create_comparison_panel",
    "draw_contours",
    "evaluate_model",
    "evaluate_threshold",
    "normalize_heatmap",
    "overlay_heatmap",
    "run_experiment",
]
