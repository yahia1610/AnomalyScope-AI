"""Reusable Streamlit rendering helpers for the AnomalyScope dashboard.

These functions are thin, side-effect-scoped wrappers around Streamlit
primitives.  They contain no data-loading logic (see
:mod:`anomalyscope.dashboard.data_loading`) so they can be swapped or reused
independently.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from anomalyscope.dashboard.data_loading import (
    EXPLORER_COLUMNS,
    ExperimentRecord,
    HeatmapImage,
    comparison_long_frame,
    records_to_explorer_frame,
)

# Metrics that are rates in [0, 1] and benefit from fixed formatting.
_RATE_METRICS: tuple[str, ...] = (
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "threshold_precision",
    "threshold_recall",
    "threshold_f1",
    "threshold_accuracy",
    "threshold_fpr",
)

# Friendly labels for display.
_LABELS: dict[str, str] = {
    "image_auroc": "Image AUROC",
    "pixel_auroc": "Pixel AUROC",
    "aupro": "PRO / AUPRO",
    "threshold": "Threshold",
    "threshold_fpr": "FPR (calibration)",
    "threshold_precision": "Precision",
    "threshold_recall": "Recall",
    "threshold_f1": "F1",
    "threshold_accuracy": "Accuracy",
    "num_train_samples": "Train samples",
    "num_test_samples": "Test samples",
    "num_fit_samples": "Fit samples",
    "num_calibration_samples": "Calibration samples",
    "train_time_sec": "Train time (s)",
    "inference_time_sec": "Inference time (s)",
    "total_time_sec": "Total time (s)",
}


def fmt_metric(name: str, value: Any) -> str:
    """Format a metric value for display, or return an em-dash if missing."""
    if value is None:
        return "—"
    if name in _RATE_METRICS:
        return f"{float(value):.4f}"
    if name.endswith("_sec"):
        return f"{float(value):.2f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def label_for(name: str) -> str:
    """Return the friendly display label for a metric key."""
    return _LABELS.get(name, name.replace("_", " ").title())


def _lookup(record: ExperimentRecord, name: str) -> Any:
    """Look up a metric/count/time value on *record* by key."""
    if name in record.metrics:
        return record.metrics.get(name)
    if name in record.counts:
        return record.counts.get(name)
    if name in record.times:
        return record.times.get(name)
    return None


# Ordered keys rendered in the details table.
_DETAIL_KEYS: tuple[str, ...] = (
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "threshold",
    "threshold_fpr",
    "threshold_precision",
    "threshold_recall",
    "threshold_f1",
    "threshold_accuracy",
    "num_train_samples",
    "num_test_samples",
    "num_fit_samples",
    "num_calibration_samples",
    "train_time_sec",
    "inference_time_sec",
    "total_time_sec",
)


def render_metric_table(record: ExperimentRecord) -> None:
    """Render a two-column metric/value table for a single experiment."""
    rows = [
        {"Metric": label_for(name), "Value": fmt_metric(name, _lookup(record, name))}
        for name in _DETAIL_KEYS
    ]
    st.table(pd.DataFrame(rows))


def render_kpis(record: ExperimentRecord) -> None:
    """Render headline KPI cards for an experiment."""
    cols = st.columns(4)
    kpis = [
        ("Image AUROC", record.metric("image_auroc")),
        ("Pixel AUROC", record.metric("pixel_auroc")),
        ("PRO / AUPRO", record.metric("aupro")),
        ("F1", record.metric("threshold_f1")),
    ]
    for col, (title, value) in zip(cols, kpis):
        col.metric(
            label=title,
            value=fmt_metric(title, value) if value is not None else "—",
        )


def _comparison_chart(long_df: pd.DataFrame) -> alt.Chart:
    """Build a grouped bar chart (one bar per metric within each experiment).

    The y-axis is fixed to ``[0, 1]`` so all rate metrics share a comparable
    scale and bar heights are never misleadingly stacked.
    """
    return (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("experiment:N", title="Experiment", sort=None),
            xOffset=alt.XOffset("metric:N", title="Metric"),
            y=alt.Y(
                "value:Q",
                title="Score",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".2f"),
            ),
            color=alt.Color("metric:N", title="Metric", sort=None),
            tooltip=[
                alt.Tooltip("experiment:N", title="Experiment"),
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Score", format=".4f"),
            ],
        )
        .properties(height=360)
    )


def render_comparison(
    records: list[ExperimentRecord], show_caption: bool = True
) -> None:
    """Render model-comparison table and a *grouped* bar chart."""
    if not records:
        st.info("No experiments available to compare.")
        return

    # Readable table (friendly names, rounded values).
    st.dataframe(
        records_to_explorer_frame(records),
        width="stretch",
        hide_index=True,
    )

    long_df = comparison_long_frame(records)
    if long_df.empty or long_df["value"].dropna().empty:
        st.info("No comparable metrics available to chart.")
    else:
        st.altair_chart(_comparison_chart(long_df), width="stretch")

    if len(records) < 2:
        st.info(
            "Only one experiment is available — this is a single result, "
            "not a meaningful cross-model comparison. Add more experiments "
            "to compare models."
        )

    if show_caption:
        st.caption(
            "Bars show Image AUROC, Pixel AUROC, AUPRO and F1 side by side "
            "for each experiment (model · category). Comparisons across "
            "different categories may not be directly equivalent, as dataset "
            "difficulty differs."
        )


def _heatmap_caption(image: HeatmapImage) -> str:
    """Build an explanatory caption distinguishing predictions from truth.

    The red contours are drawn from the model's predicted anomaly map
    (per-image thresholded) — they are model predictions, never ground-truth
    masks.
    """
    if image.label == "normal":
        return (
            f"Normal sample — {image.path.name}. Red contours mark "
            "model-predicted anomaly regions (relative hotspots). On a "
            "normal sample these are potential false positives, not "
            "confirmed defects. Ground-truth masks are not shown."
        )
    if image.label == "anomalous":
        return (
            f"Anomalous sample — {image.path.name}. Red contours mark "
            "model-predicted anomaly regions. Ground-truth masks are not "
            "shown here."
        )
    return (
        f"{image.path.name}. Red contours mark model-predicted anomaly "
        "regions (not ground truth)."
    )


def render_heatmap(image: HeatmapImage) -> None:
    """Render a single heatmap image with an explanatory caption."""
    try:
        st.image(str(image.path), caption=_heatmap_caption(image), width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not display {image.path.name}: {exc}")


def render_heatmap_legend() -> None:
    """Explain what the contours mean (predictions vs ground truth)."""
    st.markdown(
        """
**How to read these panels** (original · heatmap · overlay):

- **Contours** (coloured outlines) are drawn from the model's **predicted**
  anomaly map, thresholded per image — they are **not** ground-truth masks.
- On **normal** samples, contours indicate **predicted** anomaly regions,
  i.e. potential **false positives**, not confirmed defects.
- Colour intensity shows the (per-image normalised) anomaly score: warmer
  colours = higher predicted anomaly.
"""
    )


def render_record_table(records: list[ExperimentRecord]) -> None:
    """Render a readable table of all experiments (no per-defect data)."""
    if not records:
        st.info("No experiments found.")
        return

    frame = records_to_explorer_frame(records)

    # Per-column formatting: fixed precision for rates, integers for counts.
    column_config: dict[str, Any] = {}
    for _key, label, kind in EXPLORER_COLUMNS:
        if kind in ("rate", "num"):
            column_config[label] = st.column_config.NumberColumn(
                label, format="%.4f"
            )
        elif kind == "sec":
            column_config[label] = st.column_config.NumberColumn(
                label, format="%.2f"
            )
        elif kind == "int":
            column_config[label] = st.column_config.NumberColumn(
                label, format="%d"
            )

    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )