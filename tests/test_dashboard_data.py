"""Focused tests for dashboard data loading and result handling."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from anomalyscope.dashboard.data_loading import (
    ExperimentRecord,
    HeatmapImage,
    available_categories,
    available_models,
    comparison_frame,
    comparison_long_frame,
    discover_experiments,
    filter_records,
    latest_record,
    list_heatmaps,
    load_results_csv,
    parse_result_json,
    records_to_dataframe,
    records_to_explorer_frame,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_result(
    output_root: Path,
    run_name: str,
    model: str,
    category: str,
    *,
    image_auroc: float = 0.9,
    include_config: bool = True,
    per_defect: bool = True,
) -> Path:
    results_dir = output_root / run_name / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "model_name": model,
        "category": category,
        "image_auroc": image_auroc,
        "pixel_auroc": 0.8,
        "aupro": 0.7,
        "threshold": 1.5,
        "threshold_fpr": 0.02,
        "threshold_precision": 0.9,
        "threshold_recall": 0.85,
        "threshold_f1": 0.87,
        "threshold_accuracy": 0.9,
        "num_train_samples": 209,
        "num_test_samples": 83,
        "num_fit_samples": 167,
        "num_calibration_samples": 42,
        "train_time_sec": 10.0,
        "inference_time_sec": 3.0,
        "total_time_sec": 13.0,
        "checkpoint_path": "ckpt/padim_bottle.pth",
        "calibration_split": 0.2,
        "calibration_seed": 42,
    }
    if include_config:
        data["config"] = {
            "category": category,
            "device": "cpu",
            "seed": 42,
            "image_size": 256,
            "model": {"name": model.lower()},
        }
    if per_defect:
        data["per_defect_results"] = {"broken_large": 0.5}
    path = results_dir / f"{model.lower()}_{category}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_csv(output_root: Path, run_name: str) -> Path:
    results_dir = output_root / run_name / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "model_name": "PaDiM",
                "category": "bottle",
                "image_auroc": 0.9,
                "pixel_auroc": 0.8,
                "aupro": 0.7,
                "threshold": 1.5,
            }
        ]
    )
    path = results_dir / "results.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# parse_result_json
# ---------------------------------------------------------------------------


class TestParseResultJson:
    def test_parses_valid_json(self, tmp_path: Path) -> None:
        p = _write_result(tmp_path, "run1", "PaDiM", "bottle")
        rec = parse_result_json(p)
        assert rec is not None
        assert rec.model_name == "PaDiM"
        assert rec.category == "bottle"
        assert rec.metric("image_auroc") == 0.9
        assert rec.metric("aupro") == 0.7
        assert rec.counts["num_test_samples"] == 83
        assert rec.times["train_time_sec"] == 10.0

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert parse_result_json(tmp_path / "nope.json") is None

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        assert parse_result_json(p) is None

    def test_missing_required_keys_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "partial.json"
        p.write_text(json.dumps({"image_auroc": 0.9}), encoding="utf-8")
        assert parse_result_json(p) is None

    def test_missing_fields_become_none(self, tmp_path: Path) -> None:
        p = tmp_path / "min.json"
        p.write_text(
            json.dumps({"model_name": "PaDiM", "category": "bottle"}),
            encoding="utf-8",
        )
        rec = parse_result_json(p)
        assert rec is not None
        assert rec.metric("image_auroc") is None
        assert rec.metric("aupro") is None

    def test_output_and_results_dir_derived(self, tmp_path: Path) -> None:
        p = _write_result(tmp_path, "run1", "PaDiM", "bottle")
        rec = parse_result_json(p)
        assert rec.results_dir == (tmp_path / "run1" / "results")
        assert rec.output_dir == (tmp_path / "run1")


# ---------------------------------------------------------------------------
# discover_experiments
# ---------------------------------------------------------------------------


class TestDiscoverExperiments:
    def test_finds_multiple_runs(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        _write_result(tmp_path, "run2", "PatchCore", "hazelnut")
        recs = discover_experiments(tmp_path)
        assert len(recs) == 2
        names = {r.model_name for r in recs}
        assert names == {"PaDiM", "PatchCore"}

    def test_empty_root_returns_empty(self, tmp_path: Path) -> None:
        assert discover_experiments(tmp_path) == []

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        assert discover_experiments(tmp_path / "does_not_exist") == []

    def test_sorted_by_mtime(self, tmp_path: Path) -> None:
        p1 = _write_result(tmp_path, "run1", "PaDiM", "bottle")
        time.sleep(0.02)
        p2 = _write_result(tmp_path, "run2", "PaDiM", "bottle")
        recs = discover_experiments(tmp_path)
        assert [r.json_path for r in recs] == [p1, p2]

    def test_skips_invalid_files(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        bad = tmp_path / "run1" / "results" / "broken.json"
        bad.write_text("{oops", encoding="utf-8")
        recs = discover_experiments(tmp_path)
        assert len(recs) == 1


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


class TestLoadResultsCsv:
    def test_loads_csv(self, tmp_path: Path) -> None:
        _write_csv(tmp_path, "run1")
        df = load_results_csv(tmp_path)
        assert df is not None
        assert len(df) == 1
        assert "model_name" in df.columns
        assert "source_csv" in df.columns

    def test_no_csv_returns_none(self, tmp_path: Path) -> None:
        assert load_results_csv(tmp_path) is None

    def test_missing_root_returns_none(self, tmp_path: Path) -> None:
        assert load_results_csv(tmp_path / "nope") is None


# ---------------------------------------------------------------------------
# Filtering / grouping
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_available_models(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        _write_result(tmp_path, "run2", "PatchCore", "hazelnut")
        recs = discover_experiments(tmp_path)
        assert available_models(recs) == ["PaDiM", "PatchCore"]

    def test_available_categories(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        _write_result(tmp_path, "run2", "PaDiM", "hazelnut")
        recs = discover_experiments(tmp_path)
        assert available_categories(recs) == ["bottle", "hazelnut"]

    def test_filter_by_model(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        _write_result(tmp_path, "run2", "PatchCore", "bottle")
        recs = discover_experiments(tmp_path)
        filtered = filter_records(recs, model="padim")
        assert len(filtered) == 1
        assert filtered[0].model_name == "PaDiM"

    def test_filter_by_category(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        _write_result(tmp_path, "run2", "PaDiM", "hazelnut")
        recs = discover_experiments(tmp_path)
        filtered = filter_records(recs, category="bottle")
        assert len(filtered) == 1

    def test_filter_case_insensitive(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        recs = discover_experiments(tmp_path)
        assert len(filter_records(recs, model="PADIM", category="BOTTLE")) == 1

    def test_latest_record(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        time.sleep(0.02)
        p2 = _write_result(tmp_path, "run2", "PatchCore", "bottle")
        recs = discover_experiments(tmp_path)
        latest = latest_record(recs)
        assert latest is not None
        assert latest.json_path == p2

    def test_latest_empty(self) -> None:
        assert latest_record([]) is None


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------


class TestHeatmaps:
    def test_finds_heatmaps(self, tmp_path: Path) -> None:
        p = _write_result(tmp_path, "run1", "PaDiM", "bottle")
        rec = parse_result_json(p)
        hm_dir = rec.heatmap_dir
        hm_dir.mkdir(parents=True, exist_ok=True)
        (hm_dir / "sample_0000_anom.png").write_bytes(b"\x89PNG")
        (hm_dir / "sample_0063_norm.png").write_bytes(b"\x89PNG")

        images = list_heatmaps(rec)
        assert len(images) == 2
        labels = {i.label for i in images}
        assert labels == {"anomalous", "normal"}

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        p = _write_result(tmp_path, "run1", "PaDiM", "bottle")
        rec = parse_result_json(p)
        assert list_heatmaps(rec) == []

    def test_ignores_non_images(self, tmp_path: Path) -> None:
        p = _write_result(tmp_path, "run1", "PaDiM", "bottle")
        rec = parse_result_json(p)
        rec.heatmap_dir.mkdir(parents=True, exist_ok=True)
        (rec.heatmap_dir / "notes.txt").write_text("x")
        assert list_heatmaps(rec) == []


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


class TestDataFrames:
    def test_records_to_dataframe(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        recs = discover_experiments(tmp_path)
        df = records_to_dataframe(recs)
        assert len(df) == 1
        assert "image_auroc" in df.columns
        assert "per_defect_results" not in df.columns

    def test_records_to_dataframe_empty(self) -> None:
        df = records_to_dataframe([])
        assert df.empty

    def test_to_display_dict_excludes_per_defect(self, tmp_path: Path) -> None:
        p = _write_result(tmp_path, "run1", "PaDiM", "bottle", per_defect=True)
        rec = parse_result_json(p)
        d = rec.to_display_dict()
        assert "per_defect_results" not in d
        assert d["model"] == "PaDiM"

    def test_comparison_frame(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        _write_result(tmp_path, "run2", "PatchCore", "hazelnut")
        recs = discover_experiments(tmp_path)
        df = comparison_frame(recs)
        assert len(df) == 2
        for col in ("experiment", "model", "category", "image_auroc",
                    "pixel_auroc", "aupro", "threshold_f1"):
            assert col in df.columns

    def test_comparison_frame_empty(self) -> None:
        assert comparison_frame([]).empty


class TestComparisonLongFrame:
    def test_long_shape_one_row_per_metric(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        recs = discover_experiments(tmp_path)
        long_df = comparison_long_frame(recs)
        # 4 metrics per experiment.
        assert len(long_df) == 4
        assert set(long_df["metric"]) == {
            "Image AUROC",
            "Pixel AUROC",
            "AUPRO",
            "F1",
        }

    def test_multiple_experiments(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        _write_result(tmp_path, "run2", "PatchCore", "hazelnut")
        recs = discover_experiments(tmp_path)
        long_df = comparison_long_frame(recs)
        assert len(long_df) == 8
        assert set(long_df.columns) == {
            "experiment",
            "model",
            "category",
            "metric",
            "value",
        }

    def test_empty(self) -> None:
        df = comparison_long_frame([])
        assert df.empty
        assert "metric" in df.columns

    def test_values_within_range(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        recs = discover_experiments(tmp_path)
        long_df = comparison_long_frame(recs)
        vals = long_df["value"].dropna()
        assert ((vals >= 0) & (vals <= 1)).all()


class TestExplorerFrame:
    def test_friendly_columns_and_rounding(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle", image_auroc=0.996031746)
        recs = discover_experiments(tmp_path)
        df = records_to_explorer_frame(recs)
        # Friendly labels, not raw keys.
        assert "Image AUROC" in df.columns
        assert "AUPRO" in df.columns
        assert "image_auroc" not in df.columns
        # Rounded to 4 dp.
        assert df["Image AUROC"].iloc[0] == round(0.996031746, 4)

    def test_no_per_defect_column(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle", per_defect=True)
        recs = discover_experiments(tmp_path)
        df = records_to_explorer_frame(recs)
        assert "per_defect_results" not in df.columns

    def test_seconds_precision(self, tmp_path: Path) -> None:
        _write_result(tmp_path, "run1", "PaDiM", "bottle")
        recs = discover_experiments(tmp_path)
        df = records_to_explorer_frame(recs)
        # train_time_sec = 10.0 -> "Train (s)" rounded to 2 dp.
        assert df["Train (s)"].iloc[0] == round(10.0, 2)

    def test_empty(self) -> None:
        df = records_to_explorer_frame([])
        assert df.empty


class TestHeatmapCaption:
    def test_normal_labels_false_positive(self, tmp_path: Path) -> None:
        from anomalyscope.dashboard.components.ui import _heatmap_caption

        img = HeatmapImage(path=Path("sample_0063_norm.png"), label="normal")
        caption = _heatmap_caption(img)
        assert "Normal" in caption
        assert "false positive" in caption.lower()
        assert "not ground truth" in caption.lower() or "not" in caption.lower()

    def test_anomalous_labels_prediction(self, tmp_path: Path) -> None:
        from anomalyscope.dashboard.components.ui import _heatmap_caption

        img = HeatmapImage(path=Path("sample_0000_anom.png"), label="anomalous")
        caption = _heatmap_caption(img)
        assert "Anomalous" in caption
        assert "predicted" in caption.lower()

    def test_caption_does_not_claim_ground_truth(self) -> None:
        from anomalyscope.dashboard.components.ui import _heatmap_caption

        img = HeatmapImage(path=Path("x_norm.png"), label="normal")
        caption = _heatmap_caption(img).lower()
        # Must not present contours as confirmed ground truth.
        assert "ground-truth masks are not shown" in caption