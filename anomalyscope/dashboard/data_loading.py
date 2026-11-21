"""Data loading for the AnomalyScope AI dashboard.

This module contains **no Streamlit imports** so it can be unit-tested in
isolation.  It discovers experiment result files (JSON + CSV) written by
:class:`~anomalyscope.evaluation.runner.ExperimentRunner`, parses them into
typed records, and locates associated heatmap images.

All values are read from actual files on disk — nothing is hardcoded or
synthesised.  Missing or malformed data is handled gracefully (fields become
``None`` rather than raising).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# Numeric metric fields surfaced by the dashboard (per_defect_results is
# intentionally excluded — see project notes on per-defect AUROC validity).
_NUMERIC_FIELDS: tuple[str, ...] = (
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "threshold",
    "threshold_fpr",
    "threshold_precision",
    "threshold_recall",
    "threshold_f1",
    "threshold_accuracy",
)

# Timing / count fields.
_COUNT_FIELDS: tuple[str, ...] = (
    "num_train_samples",
    "num_test_samples",
    "num_fit_samples",
    "num_calibration_samples",
)
_TIME_FIELDS: tuple[str, ...] = (
    "train_time_sec",
    "inference_time_sec",
    "total_time_sec",
)


# ---------------------------------------------------------------------------
# Record container
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecord:
    """A single experiment parsed from a results JSON file."""

    model_name: str
    category: str
    json_path: Path
    output_dir: Path
    results_dir: Path
    mtime: float
    metrics: dict[str, float | None] = field(default_factory=dict)
    counts: dict[str, int | None] = field(default_factory=dict)
    times: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- derived helpers -------------------------------------------------

    @property
    def model_key(self) -> str:
        """Lowercase model identifier used in folder/file names."""
        return str(self.model_name).lower()

    @property
    def heatmap_dir(self) -> Path:
        """Directory expected to hold heatmaps for this experiment."""
        return self.output_dir / "heatmaps" / f"{self.model_key}_{self.category}"

    @property
    def label(self) -> str:
        """Human-readable unique label (model + category)."""
        return f"{self.model_name} · {self.category}"

    def metric(self, name: str) -> float | None:
        """Return a metric value or ``None`` if unavailable."""
        return self.metrics.get(name)

    def to_display_dict(self) -> dict[str, Any]:
        """Flat dict for tables/CSV.  Excludes ``per_defect_results``."""
        row: dict[str, Any] = {
            "model": self.model_name,
            "category": self.category,
        }
        row.update(self.metrics)
        row.update(self.counts)
        row.update(self.times)
        return row


# ---------------------------------------------------------------------------
# Discovery / parsing
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_result_json(path: str | Path) -> ExperimentRecord | None:
    """Parse a single results JSON file into an :class:`ExperimentRecord`.

    Returns ``None`` if the file cannot be read or lacks the minimum
    required keys (``model_name`` and ``category``).
    """
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read result JSON %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        return None

    model_name = data.get("model_name")
    category = data.get("category")
    if not model_name or not category:
        logger.warning("Result JSON %s missing model_name/category", path)
        return None

    results_dir = path.parent
    output_dir = results_dir.parent

    metrics = {k: _safe_float(data.get(k)) for k in _NUMERIC_FIELDS}
    counts = {k: _safe_int(data.get(k)) for k in _COUNT_FIELDS}
    times = {k: _safe_float(data.get(k)) for k in _TIME_FIELDS}

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0

    # Metadata: config + calibration info (per_defect_results deliberately
    # excluded from the record's display surface).
    metadata: dict[str, Any] = {
        "checkpoint_path": data.get("checkpoint_path"),
        "calibration_split": data.get("calibration_split"),
        "calibration_seed": data.get("calibration_seed"),
        "config": data.get("config") if isinstance(data.get("config"), dict) else {},
    }

    return ExperimentRecord(
        model_name=str(model_name),
        category=str(category),
        json_path=path,
        output_dir=output_dir,
        results_dir=results_dir,
        mtime=mtime,
        metrics=metrics,
        counts=counts,
        times=times,
        metadata=metadata,
    )


def discover_experiments(root: str | Path) -> list[ExperimentRecord]:
    """Recursively discover all experiment result JSONs under *root*.

    Searches for ``**/results/*.json`` files, parses each, and returns a list
    sorted by modification time (oldest first).  Unparseable files are
    skipped with a warning.
    """
    root = Path(root)
    if not root.exists():
        return []

    records: list[ExperimentRecord] = []
    for json_path in sorted(root.rglob("results/*.json")):
        record = parse_result_json(json_path)
        if record is not None:
            records.append(record)

    records.sort(key=lambda r: r.mtime)
    return records


def load_results_csv(root: str | Path) -> pd.DataFrame | None:
    """Load and concatenate every ``results/*.csv`` under *root*.

    Returns ``None`` if no CSV files are found.
    """
    root = Path(root)
    if not root.exists():
        return None

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(root.rglob("results/*.csv")):
        try:
            df = pd.read_csv(csv_path)
            df.insert(0, "source_csv", str(csv_path))
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read CSV %s: %s", csv_path, exc)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Filtering / grouping
# ---------------------------------------------------------------------------


def available_models(records: Iterable[ExperimentRecord]) -> list[str]:
    """Sorted unique model names present in *records*."""
    return sorted({r.model_name for r in records})


def available_categories(records: Iterable[ExperimentRecord]) -> list[str]:
    """Sorted unique categories present in *records*."""
    return sorted({r.category for r in records})


def filter_records(
    records: Iterable[ExperimentRecord],
    model: str | None = None,
    category: str | None = None,
) -> list[ExperimentRecord]:
    """Filter *records* by model and/or category (case-insensitive)."""
    out = []
    for r in records:
        if model and r.model_name.lower() != model.lower():
            continue
        if category and r.category.lower() != category.lower():
            continue
        out.append(r)
    return out


def latest_record(records: list[ExperimentRecord]) -> ExperimentRecord | None:
    """Return the most recently modified record, or ``None`` if empty."""
    if not records:
        return None
    return max(records, key=lambda r: r.mtime)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------


def _classify_heatmap_name(name: str) -> str:
    """Return ``'anomalous'``, ``'normal'`` or ``'unknown'`` from a filename."""
    lower = name.lower()
    if "anom" in lower:
        return "anomalous"
    if "norm" in lower or "good" in lower:
        return "normal"
    return "unknown"


@dataclass
class HeatmapImage:
    """A discovered heatmap image with a parsed classification label."""

    path: Path
    label: str  # 'anomalous' | 'normal' | 'unknown'

    @property
    def caption(self) -> str:
        return f"{self.label.capitalize()} — {self.path.name}"


def list_heatmaps(record: ExperimentRecord) -> list[HeatmapImage]:
    """Return heatmap images for *record* that actually exist on disk.

    Missing directories or empty folders yield an empty list (no error).
    """
    heatmap_dir = record.heatmap_dir
    if not heatmap_dir.is_dir():
        return []

    images: list[HeatmapImage] = []
    for path in sorted(heatmap_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            images.append(
                HeatmapImage(path=path, label=_classify_heatmap_name(path.name))
            )
    return images


# ---------------------------------------------------------------------------
# DataFrame helpers (for tables & charts)
# ---------------------------------------------------------------------------


def records_to_dataframe(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    """Build a display DataFrame from *records* (excludes per-defect data)."""
    rows = [r.to_display_dict() for r in records]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def comparison_frame(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    """Build a tidy DataFrame for model comparison charts.

    Only the primary, comparable metrics are included
    (``image_auroc``, ``pixel_auroc``, ``aupro``, ``threshold_f1``).
    """
    records = list(records)
    if not records:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "experiment": r.label,
                "model": r.model_name,
                "category": r.category,
                "image_auroc": r.metric("image_auroc"),
                "pixel_auroc": r.metric("pixel_auroc"),
                "aupro": r.metric("aupro"),
                "threshold_f1": r.metric("threshold_f1"),
            }
        )
    return pd.DataFrame(rows)


# Metrics shown as grouped bars in the comparison chart, with friendly labels.
_COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("image_auroc", "Image AUROC"),
    ("pixel_auroc", "Pixel AUROC"),
    ("aupro", "AUPRO"),
    ("threshold_f1", "F1"),
)


def comparison_long_frame(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    """Melt comparison metrics into long form for a *grouped* bar chart.

    Columns: ``experiment``, ``model``, ``category``, ``metric``, ``value``.
    Each (experiment, metric) pair is one row, so a grouped bar chart can
    place one bar per metric within each experiment group.
    """
    records = list(records)
    if not records:
        return pd.DataFrame(
            columns=["experiment", "model", "category", "metric", "value"]
        )

    rows: list[dict[str, Any]] = []
    for r in records:
        for key, label in _COMPARISON_METRICS:
            rows.append(
                {
                    "experiment": r.label,
                    "model": r.model_name,
                    "category": r.category,
                    "metric": label,
                    "value": r.metric(key),
                }
            )
    return pd.DataFrame(rows)


# Explorer table column spec: (key, friendly label, kind).
# kind drives numeric precision: rate/fixed→4dp, seconds→2dp, int→integer.
EXPLORER_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("model", "Model", "text"),
    ("category", "Category", "text"),
    ("image_auroc", "Image AUROC", "rate"),
    ("pixel_auroc", "Pixel AUROC", "rate"),
    ("aupro", "AUPRO", "rate"),
    ("threshold", "Threshold", "num"),
    ("threshold_precision", "Precision", "rate"),
    ("threshold_recall", "Recall", "rate"),
    ("threshold_f1", "F1", "rate"),
    ("num_test_samples", "Test samples", "int"),
    ("train_time_sec", "Train (s)", "sec"),
    ("inference_time_sec", "Inference (s)", "sec"),
)


def records_to_explorer_frame(
    records: Iterable[ExperimentRecord],
) -> pd.DataFrame:
    """Build a readable explorer table with friendly names and rounding."""
    rows: list[dict[str, Any]] = []
    for r in records:
        display = r.to_display_dict()
        row: dict[str, Any] = {}
        for key, label, kind in EXPLORER_COLUMNS:
            value = display.get(key)
            if value is None:
                row[label] = None
            elif kind in ("rate", "num"):
                row[label] = round(float(value), 4)
            elif kind == "sec":
                row[label] = round(float(value), 2)
            elif kind == "int":
                row[label] = int(value)
            else:
                row[label] = value
        rows.append(row)
    return pd.DataFrame(rows)