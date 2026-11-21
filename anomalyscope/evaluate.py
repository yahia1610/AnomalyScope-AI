"""Evaluation CLI for AnomalyScope AI.

Loads a previously trained checkpoint (and its calibrated threshold), runs
inference on the MVTec AD test split, and exports JSON + CSV results.

By default the model is **not** retrained.  Pass ``--retrain`` to explicitly
re-run the full training experiment instead.

The heavy lifting is delegated to
:class:`~anomalyscope.evaluation.runner.ExperimentRunner`; this module only
handles argument parsing, checkpoint/threshold resolution, and output.

Usage::

    # Evaluate a saved checkpoint (threshold read from saved results JSON)
    python -m anomalyscope.evaluate --model padim --category bottle \
        --dataset-root ./data/mvtec_anomaly_detection

    # Evaluate with an explicit threshold
    python -m anomalyscope.evaluate --model patchcore --category hazelnut \
        --threshold 1.25 --device cpu

    # Retrain then evaluate
    python -m anomalyscope.evaluate --model autoencoder --category bottle \
        --retrain
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from anomalyscope.evaluation.runner import (
    SUPPORTED_CATEGORIES,
    ExperimentResult,
    ExperimentRunner,
)
from anomalyscope.models import get_model
from anomalyscope.utils.config import load_config

logger = logging.getLogger(__name__)

MODEL_CHOICES: tuple[str, ...] = ("autoencoder", "padim", "patchcore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``evaluate`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="anomalyscope.evaluate",
        description="Evaluate a trained AnomalyScope AI model on the test set.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=SUPPORTED_CATEGORIES,
        help="MVTec AD category to evaluate on.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Path to the MVTec AD root directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 'auto', 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a model YAML config. Defaults to configs/<model>.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path. Defaults to "
        "<checkpoint-dir>/<model>_<category>.pth.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold. If omitted, read from saved results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for results and heatmaps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing model checkpoints.",
    )
    parser.add_argument(
        "--num-heatmaps",
        type=int,
        default=4,
        help="Number of sample heatmap panels to save.",
    )
    parser.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="Disable heatmap generation.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the model before evaluating (default: no).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Percentile for threshold calibration when --retrain is used.",
    )
    parser.add_argument(
        "--calibration-split",
        type=float,
        default=0.2,
        help="Fraction of normal training images held out for calibration "
        "(only used with --retrain).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (only used with --retrain).",
    )
    return parser


# ---------------------------------------------------------------------------
# Path / value helpers (testable, side-effect free)
# ---------------------------------------------------------------------------


def resolve_model_config_path(
    model: str,
    config: str | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Return the YAML config path for *model*."""
    if config is not None:
        return Path(config)
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    return Path(project_root) / "configs" / f"{model}.yaml"


def build_overrides(args: SimpleNamespace | Any) -> dict[str, Any]:
    """Build a config-override dict from parsed *args* (only set values)."""
    overrides: dict[str, Any] = {}
    if getattr(args, "dataset_root", None):
        overrides["dataset_root"] = args.dataset_root
    if getattr(args, "category", None):
        overrides["category"] = args.category
    if getattr(args, "device", None):
        overrides["device"] = args.device
    return overrides


def make_config(args: SimpleNamespace | Any) -> Any:
    """Assemble a frozen config namespace from *args*."""
    model_path = resolve_model_config_path(
        args.model, getattr(args, "config", None)
    )
    overrides = build_overrides(args)
    return load_config(
        model_config_path=str(model_path),
        overrides=overrides or None,
    )


def default_checkpoint_path(
    checkpoint_dir: str | Path,
    model: str,
    category: str,
) -> Path:
    """Return the default checkpoint path for *model* / *category*."""
    return Path(checkpoint_dir) / f"{model.lower()}_{category}.pth"


def load_threshold_from_results(
    output_dir: str | Path,
    model: str,
    category: str,
) -> float | None:
    """Read the calibrated threshold from a saved results JSON file.

    Returns ``None`` if the file does not exist or has no ``threshold`` key.
    """
    path = Path(output_dir) / "results" / f"{model.lower()}_{category}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read results JSON %s: %s", path, exc)
        return None
    value = data.get("threshold")
    return float(value) if value is not None else None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_evaluate(args: SimpleNamespace | Any) -> ExperimentResult:
    """Build a runner from *args* and evaluate (or retrain then evaluate)."""
    config = make_config(args)

    output_dir = getattr(args, "output_dir", None)
    checkpoint_dir = getattr(args, "checkpoint_dir", None) or getattr(
        config, "checkpoint_dir", "./checkpoints"
    )
    save_heatmaps = not getattr(args, "no_heatmaps", False)
    num_heatmaps = getattr(args, "num_heatmaps", 4)
    device = getattr(args, "device", None)

    # -- Retrain path: reuse ExperimentRunner.run() ---------------------
    if getattr(args, "retrain", False):
        logger.info("Retraining requested — running full training experiment.")
        runner = ExperimentRunner(
            config,
            percentile=getattr(args, "percentile", 99.5),
            save_heatmaps=save_heatmaps,
            num_heatmaps=num_heatmaps,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            device=device,
            calibration_split=getattr(args, "calibration_split", 0.2),
            seed=getattr(args, "seed", None),
        )
        return runner.run()

    # -- Evaluate-only path: load checkpoint + threshold ----------------
    ckpt = (
        Path(args.checkpoint)
        if getattr(args, "checkpoint", None)
        else default_checkpoint_path(checkpoint_dir, args.model, args.category)
    )
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. Train the model first or pass "
            f"--checkpoint."
        )

    model = get_model(config)
    model.load(str(ckpt))
    logger.info("Loaded checkpoint from %s", ckpt)

    threshold = getattr(args, "threshold", None)
    if threshold is None:
        out_for_results = output_dir or getattr(config, "output_dir", "./outputs")
        threshold = load_threshold_from_results(
            out_for_results, args.model, args.category
        )
        if threshold is None:
            logger.warning(
                "No threshold provided and none found in results JSON; "
                "defaulting to 0.0."
            )
            threshold = 0.0

    runner = ExperimentRunner(
        config,
        model=model,
        save_heatmaps=save_heatmaps,
        num_heatmaps=num_heatmaps,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    return runner.evaluate(threshold=threshold)


def _print_summary(result: ExperimentResult) -> None:
    """Print a concise human-readable summary of *result*."""
    print("=" * 60)
    print("  AnomalyScope AI — Evaluation complete")
    print("=" * 60)
    print(f"  Model:          {getattr(result, 'model_name', '?')}")
    print(f"  Category:       {getattr(result, 'category', '?')}")
    print(f"  Test samples:   {getattr(result, 'num_test_samples', 0)}")
    print(f"  Image AUROC:    {getattr(result, 'image_auroc', 0.0):.4f}")
    print(f"  Pixel AUROC:    {getattr(result, 'pixel_auroc', 0.0):.4f}")
    print(f"  AUPRO:          {getattr(result, 'aupro', 0.0):.4f}")
    print(f"  Threshold:      {getattr(result, 'threshold', 0.0):.4f}")
    print(f"  Precision:      {getattr(result, 'threshold_precision', 0.0):.4f}")
    print(f"  Recall:         {getattr(result, 'threshold_recall', 0.0):.4f}")
    print(f"  F1:             {getattr(result, 'threshold_f1', 0.0):.4f}")
    print(f"  Inference time: "
          f"{getattr(result, 'inference_time_sec', 0.0):.2f}s")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        result = run_evaluate(args)
    except Exception as exc:  # noqa: BLE001
        logger.error("Evaluation failed: %s", exc)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())