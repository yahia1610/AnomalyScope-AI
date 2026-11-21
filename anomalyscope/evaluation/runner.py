"""Experiment runner that connects models, data, and evaluation utilities.

The :class:`ExperimentRunner` ties together the whole pipeline for a single
(model, category) experiment:

1.  **Train** the model on the normal training split.
2.  **Save** a checkpoint to disk.
3.  **Calibrate** a decision threshold using **normal training samples only**
    (no test labels are ever used for calibration).
4.  **Evaluate** on the test split using the existing metrics
    (:func:`~anomalyscope.evaluation.metrics.compute_image_auroc`,
    :func:`~anomalyscope.evaluation.metrics.compute_pixel_auroc`,
    :func:`~anomalyscope.evaluation.metrics.compute_pro`).
5.  **Persist** the results to JSON and CSV.
6.  Optionally **save sample heatmap visualisations**.

Data-leakage note
-----------------

In the MVTec AD layout the training split contains **only** defect-free
(``good``) images.  Those training-normal scores are therefore used as the
"validation normal" reference for threshold calibration.  Test-split
scores and labels are **never** used to calibrate the threshold; any
threshold-dependent numbers computed on the test set are reported for
diagnostics only.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from anomalyscope.data.datamodule import DataModule
from anomalyscope.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from anomalyscope.evaluation.calibration import (
    apply_threshold,
    calibrate_threshold,
    compute_false_positive_rate,
)
from anomalyscope.evaluation.heatmap import create_comparison_panel
from anomalyscope.evaluation.metrics import (
    _collect_results,
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro,
)
from anomalyscope.models import AnomalyModel, get_model
from anomalyscope.utils.device import get_device
from anomalyscope.utils.io import export_csv

logger = logging.getLogger(__name__)

# Categories officially supported by the benchmark configs / CLI.
SUPPORTED_CATEGORIES: tuple[str, ...] = ("bottle", "hazelnut")


def split_normal_indices(
    n_samples: int,
    calibration_split: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Deterministically split ``n_samples`` into (fit, calibration) indices.

    The two returned index lists are **disjoint** and their union covers
    ``range(n_samples)``.  The split is fully reproducible for a given
    *seed*.

    Args:
        n_samples: Total number of normal samples to split.
        calibration_split: Fraction reserved for calibration (held out from
            fitting).  Must be in ``[0, 1)``.
        seed: Random seed controlling the permutation.

    Returns:
        A tuple ``(fit_indices, calibration_indices)`` of sorted index lists.
    """
    if n_samples <= 0:
        return [], []
    if not 0.0 <= calibration_split < 1.0:
        raise ValueError(
            f"calibration_split must be in [0, 1), got {calibration_split}."
        )

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)

    if n_samples == 1:
        # Cannot split a single sample; keep it for fitting.
        return [int(perm[0])], []

    calib_size = int(round(n_samples * calibration_split))
    # Guarantee at least one calibration sample and at least one fit sample.
    calib_size = max(1, min(calib_size, n_samples - 1))

    calib_idx = perm[:calib_size]
    fit_idx = perm[calib_size:]

    return sorted(int(i) for i in fit_idx), sorted(int(i) for i in calib_idx)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Structured outcome of a single experiment run."""

    model_name: str
    category: str
    image_auroc: float
    pixel_auroc: float
    aupro: float
    threshold: float
    threshold_fpr: float
    threshold_precision: float
    threshold_recall: float
    threshold_f1: float
    threshold_accuracy: float
    num_train_samples: int
    num_test_samples: int
    train_time_sec: float
    inference_time_sec: float
    total_time_sec: float
    per_defect_results: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    num_fit_samples: int = 0
    num_calibration_samples: int = 0
    calibration_split: float = 0.0
    calibration_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "model_name": self.model_name,
            "category": self.category,
            "image_auroc": float(self.image_auroc),
            "pixel_auroc": float(self.pixel_auroc),
            "aupro": float(self.aupro),
            "threshold": float(self.threshold),
            "threshold_fpr": float(self.threshold_fpr),
            "threshold_precision": float(self.threshold_precision),
            "threshold_recall": float(self.threshold_recall),
            "threshold_f1": float(self.threshold_f1),
            "threshold_accuracy": float(self.threshold_accuracy),
            "num_train_samples": int(self.num_train_samples),
            "num_test_samples": int(self.num_test_samples),
            "train_time_sec": float(self.train_time_sec),
            "inference_time_sec": float(self.inference_time_sec),
            "total_time_sec": float(self.total_time_sec),
            "per_defect_results": {
                str(k): float(v) for k, v in self.per_defect_results.items()
            },
            "checkpoint_path": self.checkpoint_path,
            "config": self.config,
            "num_fit_samples": int(self.num_fit_samples),
            "num_calibration_samples": int(self.num_calibration_samples),
            "calibration_split": float(self.calibration_split),
            "calibration_seed": int(self.calibration_seed),
        }

    def to_row(self) -> dict[str, Any]:
        """Return a flat single-row dictionary for CSV export."""
        row = self.to_dict()
        row.pop("per_defect_results", None)
        row.pop("config", None)
        return row


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Run a full train → calibrate → evaluate → persist experiment.

    Args:
        config: A configuration namespace (e.g. from
            :func:`~anomalyscope.utils.config.load_config`).  Must expose
            ``model.name``, ``category``, ``dataset_root``, ``image_size``,
            ``batch_size``, ``device``, ``checkpoint_dir``, ``output_dir``.
        datamodule: Optional pre-built :class:`DataModule`.  If ``None``,
            one is constructed from *config*.
        model: Optional pre-built model.  If ``None``, one is created via
            :func:`~anomalyscope.models.get_model`.
        percentile: Percentile used for threshold calibration (default 99.5).
        save_heatmaps: Whether to write sample heatmap panels.
        num_heatmaps: Maximum number of heatmap panels to write.
        output_dir: Override for the results directory.
        checkpoint_dir: Override for the checkpoint directory.
        device: Override for the compute device.
    """

    def __init__(
        self,
        config: SimpleNamespace | Any,
        *,
        datamodule: DataModule | None = None,
        model: AnomalyModel | None = None,
        percentile: float = 99.5,
        save_heatmaps: bool = True,
        num_heatmaps: int = 4,
        output_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        device: torch.device | str | None = None,
        calibration_split: float = 0.2,
        seed: int | None = None,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.percentile = float(percentile)
        self.save_heatmaps = save_heatmaps
        self.num_heatmaps = max(0, int(num_heatmaps))
        self.calibration_split = float(calibration_split)
        self.seed = int(seed) if seed is not None else int(
            getattr(config, "seed", 42)
        )
        self.resume = bool(resume)

        self.device = (
            torch.device(device)
            if device is not None
            else get_device(getattr(config, "device", "auto"))
        )

        self.datamodule = datamodule if datamodule is not None else DataModule(config)
        self.model = model if model is not None else get_model(config)

        self.checkpoint_dir = Path(
            checkpoint_dir
            if checkpoint_dir is not None
            else getattr(config, "checkpoint_dir", "./checkpoints")
        )
        self.output_dir = Path(
            output_dir
            if output_dir is not None
            else getattr(config, "output_dir", "./outputs")
        )

    # -- helpers ---------------------------------------------------------

    @property
    def model_key(self) -> str:
        """Short model identifier derived from config (e.g. ``"padim"``)."""
        model_cfg = getattr(self.config, "model", None)
        name = getattr(model_cfg, "name", None) if model_cfg is not None else None
        return str(name).lower() if name else self.model.name.lower()

    def _config_dict(self) -> dict[str, Any]:
        """Best-effort conversion of the config to a plain JSON-safe dict."""
        to_dict = getattr(self.config, "to_dict", None)
        raw: Any
        if callable(to_dict):
            try:
                raw = to_dict()
            except Exception:  # noqa: BLE001
                raw = None
        else:
            raw = None
        if raw is None:
            try:
                raw = dict(vars(self.config))
            except TypeError:
                raw = {}
        return _to_jsonable(raw)

    def _checkpoint_path(self, category: str) -> Path:
        return self.checkpoint_dir / f"{self.model_key}_{category}.pth"

    # -- pipeline steps --------------------------------------------------

    def _split_train_loaders(
        self, train_loader: Any
    ) -> tuple[Any, Any, int, int]:
        """Split the training loader into disjoint (fit, calibration) loaders.

        The underlying train dataset is assumed to contain only normal
        (``good``) samples (as in MVTec AD).  A deterministic split is made
        so that the model is fitted on one subset and the threshold is
        calibrated on a **held-out** subset.

        Returns:
            ``(fit_loader, calib_loader, num_fit, num_calib)``.  When a split
            is not possible (e.g. a single sample or ``calibration_split==0``),
            both loaders fall back to the original *train_loader*.
        """
        dataset = getattr(train_loader, "dataset", None)
        n = len(dataset) if dataset is not None else 0

        batch_size = getattr(train_loader, "batch_size", None) or getattr(
            self.config, "batch_size", 32
        )
        num_workers = getattr(
            train_loader, "num_workers", getattr(self.config, "num_workers", 0)
        )
        pin_memory = getattr(train_loader, "pin_memory", False)

        if dataset is None or n <= 1 or self.calibration_split <= 0.0:
            return train_loader, train_loader, n, 0

        fit_idx, calib_idx = split_normal_indices(
            n, calibration_split=self.calibration_split, seed=self.seed
        )
        if not fit_idx or not calib_idx:
            return train_loader, train_loader, n, 0

        from torch.utils.data import DataLoader, Subset

        fit_loader = DataLoader(
            Subset(dataset, fit_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        calib_loader = DataLoader(
            Subset(dataset, calib_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        return fit_loader, calib_loader, len(fit_idx), len(calib_idx)

    def _fit_model(self, fit_loader: Any, ckpt_path: Path) -> None:
        """Fit the model, forwarding recovery/resume kwargs when supported.

        Models whose ``fit`` accepts a ``recovery_path`` keyword (e.g. the
        Autoencoder) receive a recovery-checkpoint path derived from the final
        checkpoint path and the runner's ``resume`` flag.  Other models are
        fitted exactly as before.
        """
        import inspect

        try:
            fit_params = inspect.signature(self.model.fit).parameters
        except (TypeError, ValueError):  # pragma: no cover - defensive
            fit_params = {}

        if "recovery_path" in fit_params:
            recovery_path = str(ckpt_path.with_suffix(".recovery.pth"))
            kwargs: dict[str, Any] = {
                "recovery_path": recovery_path,
                "resume": bool(self.resume),
            }
            model_cfg = getattr(self.config, "model", None)
            checkpoint_every = getattr(model_cfg, "checkpoint_every", None)
            if checkpoint_every is not None:
                kwargs["checkpoint_every"] = int(checkpoint_every)
            self.model.fit(fit_loader, **kwargs)
        else:
            self.model.fit(fit_loader)

    def _collect_normal_scores(
        self, calib_loader: Any
    ) -> tuple[list[float], int]:
        """Collect anomaly scores for **held-out normal calibration** samples.

        The supplied loader must yield only normal samples (the calibration
        subset), never test data.
        """
        scores, _maps, _labels, _masks, _defects = _collect_results(
            self.model, calib_loader, device=self.device
        )
        return scores, len(scores)

    def _compute_test_metrics(
        self,
        scores: list[float],
        maps: list[np.ndarray],
        labels: list[int],
        masks: list[np.ndarray],
        defects: list[str],
        threshold: float,
    ) -> tuple[float, float, float, dict[str, float], dict[str, float]]:
        """Compute image/pixel AUROC, AUPRO, per-defect AUROC and threshold cls.

        Reuses the existing metric functions; shared by :meth:`run` and
        :meth:`evaluate`.
        """
        num_test = len(scores)
        if num_test == 0:
            return 0.0, 0.0, 0.0, {}, {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }

        labels_arr = np.asarray(labels, dtype=np.int64)
        scores_arr = np.asarray(scores, dtype=np.float64)

        image_auroc = compute_image_auroc(scores_arr, labels_arr)
        pixel_auroc = compute_pixel_auroc(maps, masks)
        aupro = compute_pro(maps, masks)

        per_defect: dict[str, float] = {}
        for defect in sorted(set(defects) - {"good"}):
            idx = [i for i, d in enumerate(defects) if d == defect]
            d_scores = scores_arr[idx]
            d_labels = labels_arr[idx]
            if len(np.unique(d_labels)) < 2:
                per_defect[defect] = 0.5
            else:
                per_defect[defect] = float(
                    compute_image_auroc(d_scores, d_labels)
                )

        cls = _classification_metrics(labels_arr, scores_arr, threshold)
        return image_auroc, pixel_auroc, aupro, per_defect, cls

    def run(self) -> ExperimentResult:
        """Execute the full experiment and return an :class:`ExperimentResult`."""
        total_start = time.perf_counter()

        category = str(getattr(self.config, "category", "unknown"))

        # 1. Data setup.
        self.datamodule.setup()
        train_loader = self.datamodule.train_dataloader()
        test_loader = self.datamodule.test_dataloader()

        # Split normal training data into disjoint fit / calibration subsets.
        fit_loader, calib_loader, num_fit, num_calib = self._split_train_loaders(
            train_loader
        )
        num_train_total = len(getattr(train_loader, "dataset", []) or [])

        # 2. Train on the FIT subset only (timed).
        ckpt_path = self._checkpoint_path(category)
        train_start = time.perf_counter()
        self._fit_model(fit_loader, ckpt_path)
        train_time = time.perf_counter() - train_start

        # 3. Save checkpoint.
        try:
            self.model.save(str(ckpt_path))
            checkpoint_str: str | None = str(ckpt_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Checkpoint save failed: %s", exc)
            checkpoint_str = None

        # 4. Calibrate threshold from held-out normal calibration scores ONLY.
        normal_scores, _ = self._collect_normal_scores(calib_loader)
        if normal_scores:
            threshold = calibrate_threshold(normal_scores, percentile=self.percentile)
        else:
            logger.warning(
                "No normal calibration scores available; defaulting threshold to 0.0."
            )
            threshold = 0.0

        # 5. Evaluate on test set (timed inference).
        infer_start = time.perf_counter()
        scores, maps, labels, masks, defects = _collect_results(
            self.model, test_loader, device=self.device
        )
        inference_time = time.perf_counter() - infer_start

        num_test = len(scores)

        # 6. Metrics (reusing existing metric functions).
        image_auroc, pixel_auroc, aupro, per_defect, cls = (
            self._compute_test_metrics(scores, maps, labels, masks, defects, threshold)
        )

        # Empirical FPR on the calibration (held-out normal) scores.
        threshold_fpr = (
            compute_false_positive_rate(normal_scores, threshold)
            if normal_scores
            else 0.0
        )

        total_time = time.perf_counter() - total_start

        result = ExperimentResult(
            model_name=self.model.name,
            category=category,
            image_auroc=image_auroc,
            pixel_auroc=pixel_auroc,
            aupro=aupro,
            threshold=threshold,
            threshold_fpr=threshold_fpr,
            threshold_precision=cls["precision"],
            threshold_recall=cls["recall"],
            threshold_f1=cls["f1"],
            threshold_accuracy=cls["accuracy"],
            num_train_samples=num_train_total,
            num_test_samples=num_test,
            train_time_sec=train_time,
            inference_time_sec=inference_time,
            total_time_sec=total_time,
            per_defect_results=per_defect,
            checkpoint_path=checkpoint_str,
            config=self._config_dict(),
            num_fit_samples=num_fit,
            num_calibration_samples=num_calib,
            calibration_split=self.calibration_split,
            calibration_seed=self.seed,
        )

        # 7. Persist + heatmaps.
        self._persist(result)
        if self.save_heatmaps and num_test > 0:
            self._save_heatmaps(maps, labels)

        return result

    def evaluate(self, threshold: float | None = None) -> ExperimentResult:
        """Evaluate an **already-loaded** model on the test set (no training).

        This is used by the evaluation CLI to score a saved checkpoint.  The
        model must already be fitted or loaded before calling this method.

        Args:
            threshold: Decision threshold to apply.  If ``None``, ``0.0`` is
                used (the caller should pass the calibrated threshold read
                from the saved experiment JSON).

        Returns:
            An :class:`ExperimentResult` with ``train_time_sec == 0`` and
            ``num_fit_samples == 0`` (no fitting was performed here).
        """
        total_start = time.perf_counter()
        category = str(getattr(self.config, "category", "unknown"))

        if threshold is None:
            threshold = 0.0
        threshold = float(threshold)

        # Data setup (test set only).
        self.datamodule.setup()
        test_loader = self.datamodule.test_dataloader()

        # Inference (timed).
        infer_start = time.perf_counter()
        scores, maps, labels, masks, defects = _collect_results(
            self.model, test_loader, device=self.device
        )
        inference_time = time.perf_counter() - infer_start

        num_test = len(scores)
        image_auroc, pixel_auroc, aupro, per_defect, cls = (
            self._compute_test_metrics(scores, maps, labels, masks, defects, threshold)
        )

        # Checkpoint is whatever the caller loaded; report its expected path.
        ckpt_path = self._checkpoint_path(category)

        total_time = time.perf_counter() - total_start

        result = ExperimentResult(
            model_name=self.model.name,
            category=category,
            image_auroc=image_auroc,
            pixel_auroc=pixel_auroc,
            aupro=aupro,
            threshold=threshold,
            threshold_fpr=0.0,
            threshold_precision=cls["precision"],
            threshold_recall=cls["recall"],
            threshold_f1=cls["f1"],
            threshold_accuracy=cls["accuracy"],
            num_train_samples=0,
            num_test_samples=num_test,
            train_time_sec=0.0,
            inference_time_sec=inference_time,
            total_time_sec=total_time,
            per_defect_results=per_defect,
            checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None,
            config=self._config_dict(),
            num_fit_samples=0,
            num_calibration_samples=0,
            calibration_split=self.calibration_split,
            calibration_seed=self.seed,
        )

        self._persist(result)
        if self.save_heatmaps and num_test > 0:
            self._save_heatmaps(maps, labels)

        return result

    # -- persistence -----------------------------------------------------

    def _persist(self, result: ExperimentResult) -> None:
        """Write JSON and CSV result files to the output directory."""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        base = f"{self.model_key}_{result.category}"

        json_path = results_dir / f"{base}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        logger.info("Results written to %s", json_path)

        csv_path = results_dir / "results.csv"
        try:
            export_csv([result.to_row()], csv_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CSV export failed: %s", exc)

    def _save_heatmaps(
        self, maps: list[np.ndarray], labels: list[int]
    ) -> None:
        """Save sample heatmap comparison panels for a few test samples."""
        if self.num_heatmaps == 0:
            return

        heatmap_dir = self.output_dir / "heatmaps" / f"{self.model_key}_{getattr(self.config, 'category', 'unknown')}"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        test_dataset = getattr(self.datamodule, "_test_dataset", None)
        if test_dataset is None:
            logger.warning("Test dataset unavailable; skipping heatmaps.")
            return

        # Prefer a mix of anomalous and normal samples.
        anomalous_idx = [i for i, lbl in enumerate(labels) if lbl == 1]
        normal_idx = [i for i, lbl in enumerate(labels) if lbl == 0]
        half = max(1, self.num_heatmaps // 2)
        chosen = (anomalous_idx[:half] + normal_idx[: self.num_heatmaps - half])[
            : self.num_heatmaps
        ]

        saved = 0
        for idx in chosen:
            try:
                sample = test_dataset[idx]
                image = _denormalize(sample["image"])
                panel = create_comparison_panel(image, maps[idx])
                out_path = heatmap_dir / (
                    f"sample_{idx:04d}_{'anom' if labels[idx] == 1 else 'norm'}.png"
                )
                cv2.imwrite(str(out_path), panel)
                saved += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save heatmap for sample %d: %s", idx, exc)

        logger.info("Saved %d heatmap panel(s) to %s", saved, heatmap_dir)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute threshold-based classification metrics on the test set.

    These are **diagnostic** values derived from test data.  They do not
    influence calibration.
    """
    preds = apply_threshold(scores, threshold)
    preds = np.asarray(preds, dtype=bool)

    tp = int(np.sum(preds & (labels == 1)))
    tn = int(np.sum(~preds & (labels == 0)))
    fp = int(np.sum(preds & (labels == 0)))
    fn = int(np.sum(~preds & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert *obj* into JSON-serialisable primitives."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, SimpleNamespace):
        return {k: _to_jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _denormalize(image: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor back to an HWC ``[0, 1]`` float array."""
    img = image.detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype).view(-1, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def run_experiment(
    config: SimpleNamespace | Any,
    *,
    datamodule: DataModule | None = None,
    model: AnomalyModel | None = None,
    percentile: float = 99.5,
    save_heatmaps: bool = True,
    num_heatmaps: int = 4,
    output_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    device: torch.device | str | None = None,
) -> ExperimentResult:
    """Convenience wrapper around :class:`ExperimentRunner`."""
    runner = ExperimentRunner(
        config,
        datamodule=datamodule,
        model=model,
        percentile=percentile,
        save_heatmaps=save_heatmaps,
        num_heatmaps=num_heatmaps,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    return runner.run()