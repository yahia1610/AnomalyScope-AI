"""Training CLI for AnomalyScope AI.

Trains an anomaly-detection model (Autoencoder, PaDiM or PatchCore) on a
category of the MVTec AD dataset, calibrates a decision threshold on a
held-out subset of normal training images, evaluates on the test split, and
persists the checkpoint plus experiment metadata (JSON + CSV).

The heavy lifting is delegated to
:class:`~anomalyscope.evaluation.runner.ExperimentRunner`; this module only
handles argument parsing and configuration assembly.

Usage::

    python -m anomalyscope.train --model padim --category bottle \
        --dataset-root ./data/mvtec_anomaly_detection

    python -m anomalyscope.train --model autoencoder --category hazelnut \
        --device cpu --percentile 99.5 --calibration-split 0.2
"""

from __future__ import annotations

import argparse
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
from anomalyscope.utils.config import load_config

logger = logging.getLogger(__name__)

MODEL_CHOICES: tuple[str, ...] = ("autoencoder", "padim", "patchcore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``train`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="anomalyscope.train",
        description="Train an AnomalyScope AI anomaly-detection model.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=SUPPORTED_CATEGORIES,
        help="MVTec AD category to train on.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Path to the MVTec AD root directory "
        "(overrides dataset_root in config).",
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
        "--output-dir",
        default=None,
        help="Directory for results and heatmaps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Percentile for threshold calibration (default 99.5).",
    )
    parser.add_argument(
        "--calibration-split",
        type=float,
        default=0.2,
        help="Fraction of normal training images held out for calibration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the fit/calibration split.",
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
        "--resume",
        action="store_true",
        help="Resume from a recovery checkpoint if one exists "
        "(models that support periodic recovery checkpoints).",
    )
    return parser


# ---------------------------------------------------------------------------
# Configuration helpers (testable, side-effect free)
# ---------------------------------------------------------------------------


def resolve_model_config_path(
    model: str,
    config: str | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Return the YAML config path for *model*.

    If *config* is provided it is used verbatim; otherwise the default
    ``configs/<model>.yaml`` under *project_root* is returned.
    """
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


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_train(args: SimpleNamespace | Any) -> ExperimentResult:
    """Build a runner from *args* and execute a full training experiment."""
    config = make_config(args)
    runner = ExperimentRunner(
        config,
        percentile=getattr(args, "percentile", 99.5),
        save_heatmaps=not getattr(args, "no_heatmaps", False),
        num_heatmaps=getattr(args, "num_heatmaps", 4),
        output_dir=getattr(args, "output_dir", None),
        checkpoint_dir=getattr(args, "checkpoint_dir", None),
        device=getattr(args, "device", None),
        calibration_split=getattr(args, "calibration_split", 0.2),
        seed=getattr(args, "seed", None),
        resume=getattr(args, "resume", False),
    )
    return runner.run()


def _print_summary(result: ExperimentResult) -> None:
    """Print a concise human-readable summary of *result*."""
    print("=" * 60)
    print("  AnomalyScope AI — Training complete")
    print("=" * 60)
    print(f"  Model:          {getattr(result, 'model_name', '?')}")
    print(f"  Category:       {getattr(result, 'category', '?')}")
    print(f"  Fit / Calib:    "
          f"{getattr(result, 'num_fit_samples', 0)} / "
          f"{getattr(result, 'num_calibration_samples', 0)}")
    print(f"  Test samples:   {getattr(result, 'num_test_samples', 0)}")
    print(f"  Image AUROC:    {getattr(result, 'image_auroc', 0.0):.4f}")
    print(f"  Pixel AUROC:    {getattr(result, 'pixel_auroc', 0.0):.4f}")
    print(f"  AUPRO:          {getattr(result, 'aupro', 0.0):.4f}")
    print(f"  Threshold:      {getattr(result, 'threshold', 0.0):.4f}")
    print(f"  Train time:     {getattr(result, 'train_time_sec', 0.0):.2f}s")
    print(f"  Inference time: "
          f"{getattr(result, 'inference_time_sec', 0.0):.2f}s")
    ckpt = getattr(result, "checkpoint_path", None)
    if ckpt:
        print(f"  Checkpoint:     {ckpt}")
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
        result = run_train(args)
    except Exception as exc:  # noqa: BLE001
        logger.error("Training failed: %s", exc)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())