"""Smoke tests that execute the Streamlit dashboard via AppTest.

These tests build a temporary experiment output tree, point the dashboard at
it, and assert the app runs without runtime exceptions across all pages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Import AppTest lazily inside tests so collection does not hard-require it.
from streamlit.testing.v1 import AppTest

_APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "anomalyscope"
    / "dashboard"
    / "app.py"
)


def _make_experiment(
    output_root: Path,
    run: str,
    model: str,
    category: str,
    with_heatmap: bool = True,
) -> None:
    results_dir = output_root / run / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "model_name": model,
        "category": category,
        "image_auroc": 0.99,
        "pixel_auroc": 0.98,
        "aupro": 0.93,
        "threshold": 15.0,
        "threshold_fpr": 0.02,
        "threshold_precision": 0.98,
        "threshold_recall": 0.94,
        "threshold_f1": 0.96,
        "threshold_accuracy": 0.94,
        "num_train_samples": 209,
        "num_test_samples": 83,
        "num_fit_samples": 167,
        "num_calibration_samples": 42,
        "train_time_sec": 86.9,
        "inference_time_sec": 37.5,
        "total_time_sec": 124.4,
        "checkpoint_path": f"ckpt/{model.lower()}_{category}.pth",
        "calibration_split": 0.2,
        "calibration_seed": 42,
        "config": {
            "dataset_root": "/data/mvtec",
            "category": category,
            "image_size": 256,
            "device": "cpu",
            "model": {"name": model.lower(), "backbone": "resnet18"},
        },
    }
    (results_dir / f"{model.lower()}_{category}.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    if with_heatmap:
        hm_dir = output_root / run / "heatmaps" / f"{model.lower()}_{category}"
        hm_dir.mkdir(parents=True, exist_ok=True)
        # 1x1 valid PNG bytes are enough for the gallery to discover files;
        # we do not rely on rendering them in this test.
        (hm_dir / "sample_0000_anom.png").write_bytes(b"\x89PNG\r\n")


class TestDashboardApp:
    def test_overview_runs_without_exception(self, tmp_path: Path) -> None:
        out = tmp_path / "outputs"
        _make_experiment(out, "smoke_test", "PaDiM", "bottle")
        at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
        # Point the sidebar output dir at our temp tree before running.
        at.run()
        at.sidebar.text_input[0].set_value(str(out))
        at.run()
        assert len(at.exception) == 0
        assert any(t.value == "AnomalyScope AI" for t in at.title)

    def test_all_pages_run_without_exception(self, tmp_path: Path) -> None:
        out = tmp_path / "outputs"
        _make_experiment(out, "smoke_test", "PaDiM", "bottle")
        _make_experiment(out, "run2", "PatchCore", "hazelnut")

        at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
        at.run()
        at.sidebar.text_input[0].set_value(str(out))
        at.run()

        for page in [
            "Experiment explorer",
            "Model comparison",
            "Heatmap gallery",
            "Experiment details",
            "Methodology & limitations",
        ]:
            at.sidebar.radio[0].set_value(page).run()
            assert len(at.exception) == 0, f"page {page} raised"

    def test_comparison_has_per_category_sections(self, tmp_path: Path) -> None:
        out = tmp_path / "outputs"
        _make_experiment(out, "r1", "PaDiM", "bottle")
        _make_experiment(out, "r2", "PatchCore", "bottle")
        _make_experiment(out, "r3", "PaDiM", "hazelnut")
        _make_experiment(out, "r4", "PatchCore", "hazelnut")

        at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
        at.run()
        at.sidebar.text_input[0].set_value(str(out))
        at.sidebar.radio[0].set_value("Model comparison").run()

        assert len(at.exception) == 0
        # One subheader per category plus the page header.
        subheaders = [s.value for s in at.subheader]
        joined = " ".join(subheaders).lower()
        assert "bottle" in joined and "hazelnut" in joined

    def test_empty_output_dir_handled(self, tmp_path: Path) -> None:
        out = tmp_path / "empty_outputs"
        out.mkdir()
        at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
        at.run()
        at.sidebar.text_input[0].set_value(str(out))
        at.run()
        assert len(at.exception) == 0