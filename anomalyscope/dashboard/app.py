"""AnomalyScope AI — Streamlit dashboard.

Run with::

    streamlit run anomalyscope/dashboard/app.py

The dashboard reads **actual** experiment results written by the training CLI
(``outputs/<run>/results/*.json`` + ``*.csv``) and any generated heatmaps
(``outputs/<run>/heatmaps/<model>_<category>/*.png``).  It never retrains
models, never writes to result/checkpoint directories, and never fabricates
metrics — missing information is shown explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is importable when launched via `streamlit run`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from anomalyscope.dashboard.components import ui  # noqa: E402
from anomalyscope.dashboard.data_loading import (  # noqa: E402
    available_categories,
    available_models,
    discover_experiments,
    filter_records,
    latest_record,
    list_heatmaps,
    load_results_csv,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AnomalyScope AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar_output_root() -> Path:
    """Resolve the output root directory from the sidebar."""
    st.sidebar.header("Data source")
    raw = st.sidebar.text_input(
        "Output directory",
        value=str(_DEFAULT_OUTPUT_ROOT),
        help="Root folder containing experiment output runs "
        "(each with results/ and heatmaps/).",
    )
    return Path(raw).expanduser()


def _sidebar_filters(records) -> tuple[str | None, str | None]:
    """Render model/category filters; return selected values or None."""
    models = available_models(records)
    categories = available_categories(records)

    model_choice = st.sidebar.selectbox(
        "Model", options=["All"] + models, index=0
    )
    category_choice = st.sidebar.selectbox(
        "Category", options=["All"] + categories, index=0
    )
    model = None if model_choice == "All" else model_choice
    category = None if category_choice == "All" else category_choice
    return model, category


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def render_overview(records) -> None:
    """Landing page: project intro + latest experiment snapshot."""
    st.title("AnomalyScope AI")
    st.caption(
        "Industrial visual anomaly detection — overview of saved experiments."
    )

    if not records:
        st.warning(
            "No experiment results found in the selected output directory. "
            "Run a training experiment first, or point the sidebar at the "
            "correct output folder."
        )
        return

    models = available_models(records)
    categories = available_categories(records)

    c1, c2, c3 = st.columns(3)
    c1.metric("Experiments", len(records))
    c2.metric("Models", ", ".join(models))
    c3.metric("Categories", ", ".join(categories))

    latest = latest_record(records)
    if latest is not None:
        st.subheader(f"Latest experiment — {latest.label}")
        st.caption(f"Source: `{latest.json_path}`")
        ui.render_kpis(latest)


def render_explorer(records) -> None:
    """Experiment explorer with filters and a metrics table."""
    st.header("Experiment explorer")

    model, category = _sidebar_filters(records)
    filtered = filter_records(records, model=model, category=category)

    if not filtered:
        st.info("No experiments match the current filters.")
        return

    st.caption(f"{len(filtered)} experiment(s) shown.")
    ui.render_record_table(filtered)


def render_comparison(records) -> None:
    """Model comparison with a separate view per dataset category."""
    st.header("Model comparison")

    if not records:
        st.info("No experiments match the current filters.")
        return

    categories = available_categories(records)
    for cat in categories:
        cat_records = filter_records(records, category=cat)
        st.subheader(f"{cat.capitalize()} — model comparison")
        ui.render_comparison(cat_records, show_caption=False)
        st.divider()

    st.caption(
        "Each section compares all models on one dataset category. "
        "Bars show Image AUROC, Pixel AUROC, AUPRO and F1 side by side; "
        "values are read directly from the saved experiment JSON files. "
        "Numbers across different categories are not directly comparable "
        "because dataset difficulty differs."
    )

    with st.expander("Reproducibility note — PatchCore implementation update"):
        st.markdown(
            "> PatchCore was implemented in two stages. The original Bottle "
            "experiment used shuffled data loading, while the later "
            "memory-efficient two-pass implementation introduced deterministic "
            "ordering to ensure consistent feature extraction across both "
            "passes. The core algorithm, backbone, image size, projection "
            "dimension, and coreset ratio were preserved. This implementation "
            "detail is documented for reproducibility when interpreting results."
        )


def render_methodology(records) -> None:
    """Static methodology, metric interpretation, and limitations."""
    st.header("Methodology & limitations")
    st.caption(
        "How the experiments were produced and how to read the metrics. "
        "This page contains no benchmark or state-of-the-art claims."
    )

    st.markdown(
        """
### Dataset (MVTec AD)
- **bottle** — 209 normal training images; 83 test images (20 good + 63 defective).
- **hazelnut** — 391 normal training images; 110 test images (40 good + 70 defective).
- Only defect-free (`good`) images are used for training.

### Train / calibration / test split
- The normal training set is split **80 / 20** into a **fit** subset and a
  **held-out calibration** subset (deterministic, seeded). The split is
  disjoint: the model never sees the calibration images during fitting.
- The **test set is never used for calibration or training.**
- The decision threshold is calibrated on the held-out normal calibration
  scores at the **99.5th percentile** (so ~0.5% of normal scores exceed it).

### Metrics
- **Image AUROC** — ROC-AUC of per-image anomaly scores vs. image labels.
- **Pixel AUROC** — ROC-AUC over all pixels (anomaly map vs. ground-truth mask).
- **PRO / AUPRO** — Per-Region Overlap, integrated over false-positive rate
  (up to 0.3) and normalised to [0, 1]. Higher is better.
- **Precision / Recall / F1 / Accuracy** — computed on the test set at the
  calibrated threshold. These are **diagnostic** (the threshold comes from
  calibration, not from maximising a test-set metric).
- **Threshold** — the calibrated cut-off applied to anomaly scores.

### Models
- **PaDiM** — density model (ResNet-18 features, random projection to 100
  dims). Memory-optimised: channels are projected **per batch before
  concatenation**, so the full 448-channel stack is never materialised.
- **PatchCore** — WideResNet-50-2 features with a **two-pass** coreset
  memory bank (projected features for selection, then a second pass to
  gather selected vectors), so the full high-dimensional bank is never held
  in memory. Coreset ratio 0.01, deterministic data ordering.
- **Autoencoder** — convolutional reconstruction model trained for 200
  epochs, with **periodic recovery checkpoints every 20 epochs** and
  `--resume` support.

### Methodological note — PatchCore implementation update

> PatchCore was implemented in two stages. The original Bottle experiment used
> shuffled data loading, while the later memory-efficient two-pass implementation
> introduced deterministic ordering to ensure consistent feature extraction
> across both passes. The core algorithm, backbone, image size, projection
> dimension, and coreset ratio were preserved. This implementation detail is
> documented for reproducibility when interpreting results.

- **PatchCore · Bottle** used the original (shuffled) data loading.
- **PatchCore · Hazelnut** used the later deterministic two-pass implementation.

### Limitations
- Results are on a **small** benchmark subset and are **not** claimed to be
  state-of-the-art or to match published numbers.
- The percentile threshold is a heuristic and may not be optimal for every
  metric; distribution shift between calibration and test data can affect it.
- The Autoencoder is a **weak reconstruction baseline** (especially on
  bottle); PaDiM/PatchCore are density/nearest-neighbour methods and assume
  representative normal training data.
- `per_defect_results` are **not shown** — the current per-defect AUROC
  implementation is not valid (single-class subsets), so it is excluded.
"""
    )


def render_gallery(records) -> None:
    """Heatmap gallery for a chosen experiment."""
    st.header("Heatmap gallery")

    if not records:
        st.info("No experiments available.")
        return

    labels = [r.label for r in records]
    choice = st.selectbox("Experiment", options=labels)
    record = next(r for r in records if r.label == choice)

    images = list_heatmaps(record)
    if not images:
        st.info(
            f"No heatmaps found for {record.label}. Expected under "
            f"`{record.heatmap_dir}`."
        )
        return

    st.caption(f"{len(images)} image(s) from `{record.heatmap_dir}`")
    with st.expander("How to read these panels (predictions vs ground truth)"):
        ui.render_heatmap_legend()
    # Two columns of images.
    for i in range(0, len(images), 2):
        cols = st.columns(2)
        for col, image in zip(cols, images[i : i + 2]):
            with col:
                ui.render_heatmap(image)


def render_details(records) -> None:
    """Full metadata and metric breakdown for a selected experiment."""
    st.header("Experiment details")

    if not records:
        st.info("No experiments available.")
        return

    labels = [r.label for r in records]
    choice = st.selectbox("Experiment", options=labels)
    record = next(r for r in records if r.label == choice)

    st.subheader(record.label)
    st.caption(f"Source: `{record.json_path}`")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Metrics**")
        ui.render_metric_table(record)

    with right:
        st.markdown("**Metadata**")
        cfg = record.metadata.get("config", {}) or {}
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        raw_meta = [
            ("Model", record.model_name),
            ("Category", record.category),
            ("Device", cfg.get("device", "—")),
            ("Image size", cfg.get("image_size", "—")),
            ("Backbone", model_cfg.get("backbone", "—")),
            ("Fit / Calibration split", record.metadata.get("calibration_split", "—")),
            ("Calibration seed", record.metadata.get("calibration_seed", "—")),
            ("Dataset root", cfg.get("dataset_root", "—")),
            ("Checkpoint", record.metadata.get("checkpoint_path") or "—"),
        ]
        # Coerce every value to string so the column is Arrow-serialisable.
        meta_rows = [
            {"Field": field, "Value": "—" if val is None else str(val)}
            for field, val in raw_meta
        ]
        st.table(pd.DataFrame(meta_rows))

    # CSV availability note.
    csv_df = load_results_csv(record.output_dir)
    st.markdown("**Raw results file**")
    if csv_df is not None:
        st.caption("A results CSV is also present in the run's results folder.")
    else:
        st.caption("No results CSV found for this run.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the dashboard."""
    output_root = _sidebar_output_root()

    page = st.sidebar.radio(
        "Page",
        options=[
            "Overview",
            "Experiment explorer",
            "Model comparison",
            "Heatmap gallery",
            "Experiment details",
            "Methodology & limitations",
        ],
    )

    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    records = discover_experiments(output_root)

    if not records:
        st.warning(
            f"No experiment results found under `{output_root}`. "
            "Train a model first, e.g.:\n\n"
            "`python -m anomalyscope.train --model padim --category bottle "
            "--device cpu --dataset-root <MVTec AD path>`"
        )

    if page == "Overview":
        render_overview(records)
    elif page == "Experiment explorer":
        render_explorer(records)
    elif page == "Model comparison":
        render_comparison(records)
    elif page == "Heatmap gallery":
        render_gallery(records)
    elif page == "Experiment details":
        render_details(records)
    elif page == "Methodology & limitations":
        render_methodology(records)


main()