"""Focused tests for the experiment runner using synthetic data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from anomalyscope.evaluation.runner import (
    ExperimentResult,
    ExperimentRunner,
    run_experiment,
    split_normal_indices,
)


# ---------------------------------------------------------------------------
# Synthetic data + model
# ---------------------------------------------------------------------------


class _SyntheticDataset(Dataset):
    """Dataset yielding MVTec-style dicts with a spatial anomaly square."""

    def __init__(
        self,
        num_normal: int,
        num_anomalous: int,
        h: int = 32,
        w: int = 32,
    ) -> None:
        super().__init__()
        self.records: list[dict] = []
        for _ in range(num_normal):
            self.records.append(self._make(h, w, label=0))
        for _ in range(num_anomalous):
            self.records.append(self._make(h, w, label=1))

    def _make(self, h: int, w: int, label: int) -> dict:
        image = torch.rand(3, h, w)
        mask = torch.zeros(1, h, w, dtype=torch.float32)
        defect = "good"
        if label == 1:
            ch, cw = h // 2, w // 2
            mask[0, ch - 4 : ch + 4, cw - 4 : cw + 4] = 1.0
            defect = "synthetic"
        return {
            "image": image,
            "mask": mask,
            "label": label,
            "path": f"/synthetic/{label}.png",
            "defect_type": defect,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


class _SyntheticDataModule:
    """Minimal stand-in for :class:`DataModule`."""

    def __init__(self, train: _SyntheticDataset, test: _SyntheticDataset) -> None:
        self._train = train
        self._test = test
        self._test_dataset = test
        self._train_dataset = train
        self._setup_called = False

    def setup(self) -> None:
        self._setup_called = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=2, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=2, shuffle=False)


class _SyntheticModel:
    """Mock anomaly model whose score depends on centre-region intensity."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._is_fitted = False
        self.net = torch.nn.Identity()  # so metrics _collect_results can call .eval()

    @property
    def name(self) -> str:
        return "Synthetic"

    def fit(self, dataloader: DataLoader) -> None:
        self._is_fitted = True

    def predict(self, image: torch.Tensor) -> tuple[float, np.ndarray]:
        h, w = image.shape[-2], image.shape[-1]
        amap = torch.rand(h, w).numpy().astype(np.float32)
        ch, cw = h // 2, w // 2
        amap[ch - 4 : ch + 4, cw - 4 : cw + 4] += 3.0
        return float(amap.max()), amap

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": self.name}, path)

    def load(self, path: str) -> None:
        self._is_fitted = True


def _make_config(tmp: Path) -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        seed=42,
        category="bottle",
        dataset_root=str(tmp),
        image_size=32,
        batch_size=2,
        num_workers=0,
        checkpoint_dir=str(tmp / "ckpt"),
        output_dir=str(tmp / "out"),
        model=SimpleNamespace(name="synthetic"),
    )


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def test_run_returns_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(4, 4), _SyntheticDataset(4, 4)
            )
            runner = ExperimentRunner(cfg, datamodule=dm, model=_SyntheticModel())
            result = runner.run()

            assert isinstance(result, ExperimentResult)
            assert result.model_name == "Synthetic"
            assert result.category == "bottle"
            assert result.num_test_samples == 8
            assert result.num_train_samples == 8

    def test_recovery_kwargs_forwarded_when_supported(self) -> None:
        captured: dict = {}

        class _RecoveryModel(_SyntheticModel):
            def fit(self, dataloader, recovery_path=None, resume=False,
                    checkpoint_every=None):
                captured["recovery_path"] = recovery_path
                captured["resume"] = resume
                captured["checkpoint_every"] = checkpoint_every
                self._is_fitted = True

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(4, 4), _SyntheticDataset(2, 2)
            )
            runner = ExperimentRunner(
                cfg, datamodule=dm, model=_RecoveryModel(), resume=True
            )
            runner.run()
            # recovery_path kwarg is forwarded and derived from the checkpoint.
            assert captured.get("recovery_path") is not None
            assert captured["recovery_path"].endswith(".recovery.pth")
            assert captured["resume"] is True

    def test_recovery_not_forwarded_when_unsupported(self) -> None:
        # _SyntheticModel.fit has no recovery_path kwarg; it must still work.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(2, 2), _SyntheticDataset(2, 2)
            )
            result = ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel()
            ).run()
            assert result.num_test_samples == 4

    def test_result_is_valid_json_serialisable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(4, 4), _SyntheticDataset(4, 4)
            )
            result = ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            # Must not raise.
            json.dumps(result.to_dict())

    def test_checkpoint_saved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(2, 2), _SyntheticDataset(2, 2)
            )
            result = ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            assert result.checkpoint_path is not None
            assert Path(result.checkpoint_path).exists()

    def test_results_json_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(3, 3), _SyntheticDataset(3, 3)
            )
            ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            json_path = tmp / "out" / "results" / "synthetic_bottle.json"
            assert json_path.exists()
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            assert "image_auroc" in data
            assert "pixel_auroc" in data
            assert "aupro" in data
            assert "threshold" in data

    def test_results_csv_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(3, 3), _SyntheticDataset(3, 3)
            )
            ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            csv_path = tmp / "out" / "results" / "results.csv"
            assert csv_path.exists()

    def test_threshold_calibrated_from_normal_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(5, 0), _SyntheticDataset(3, 3)
            )
            result = ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            # Threshold should be a finite float.
            assert isinstance(result.threshold, float)
            assert np.isfinite(result.threshold)

    def test_metrics_in_valid_range(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(4, 4), _SyntheticDataset(4, 4)
            )
            result = ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            assert 0.0 <= result.image_auroc <= 1.0
            assert 0.0 <= result.pixel_auroc <= 1.0
            assert 0.0 <= result.aupro <= 1.0

    def test_timing_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(2, 2), _SyntheticDataset(2, 2)
            )
            result = ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            assert result.train_time_sec >= 0.0
            assert result.inference_time_sec >= 0.0
            assert result.total_time_sec >= 0.0

    def test_heatmaps_saved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(2, 2), _SyntheticDataset(4, 4)
            )
            ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), num_heatmaps=2
            ).run()
            hm_dir = tmp / "out" / "heatmaps" / "synthetic_bottle"
            assert hm_dir.exists()
            assert any(hm_dir.glob("*.png"))

    def test_save_heatmaps_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(2, 2), _SyntheticDataset(2, 2)
            )
            ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            hm_dir = tmp / "out" / "heatmaps"
            assert not hm_dir.exists() or not any(hm_dir.rglob("*.png"))

    def test_per_defect_results_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(2, 2), _SyntheticDataset(2, 2)
            )
            result = ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            assert isinstance(result.per_defect_results, dict)

    def test_config_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(2, 2), _SyntheticDataset(2, 2)
            )
            result = ExperimentRunner(
                cfg, datamodule=dm, model=_SyntheticModel(), save_heatmaps=False
            ).run()
            assert isinstance(result.config, dict)
            assert result.config.get("category") == "bottle"


class TestRunExperiment:
    def test_convenience_function(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(3, 3), _SyntheticDataset(3, 3)
            )
            result = run_experiment(
                cfg,
                datamodule=dm,
                model=_SyntheticModel(),
                save_heatmaps=False,
            )
            assert isinstance(result, ExperimentResult)
            assert result.num_test_samples == 6


class TestExperimentResultRow:
    def test_to_row_is_flat(self) -> None:
        result = ExperimentResult(
            model_name="X",
            category="bottle",
            image_auroc=0.9,
            pixel_auroc=0.8,
            aupro=0.7,
            threshold=0.5,
            threshold_fpr=0.01,
            threshold_precision=0.9,
            threshold_recall=0.9,
            threshold_f1=0.9,
            threshold_accuracy=0.9,
            num_train_samples=10,
            num_test_samples=10,
            train_time_sec=1.0,
            inference_time_sec=0.5,
            total_time_sec=1.5,
            per_defect_results={"broken": 0.9},
            config={"a": 1},
        )
        row = result.to_row()
        assert "per_defect_results" not in row
        assert "config" not in row
        assert row["image_auroc"] == 0.9
        assert row["category"] == "bottle"


# ---------------------------------------------------------------------------
# Deterministic fit / calibration split
# ---------------------------------------------------------------------------


class TestSplitNormalIndices:
    def test_subsets_are_disjoint(self) -> None:
        fit, calib = split_normal_indices(100, calibration_split=0.2, seed=42)
        assert set(fit).isdisjoint(set(calib))

    def test_union_covers_all_indices(self) -> None:
        n = 100
        fit, calib = split_normal_indices(n, calibration_split=0.2, seed=42)
        assert set(fit) | set(calib) == set(range(n))

    def test_no_duplicates(self) -> None:
        fit, calib = split_normal_indices(50, calibration_split=0.3, seed=7)
        assert len(fit) == len(set(fit))
        assert len(calib) == len(set(calib))

    def test_calibration_size_approx_split(self) -> None:
        n = 200
        fit, calib = split_normal_indices(n, calibration_split=0.2, seed=42)
        assert len(calib) == 40
        assert len(fit) == 160

    def test_deterministic_for_same_seed(self) -> None:
        a = split_normal_indices(100, calibration_split=0.2, seed=123)
        b = split_normal_indices(100, calibration_split=0.2, seed=123)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = split_normal_indices(100, calibration_split=0.2, seed=1)
        b = split_normal_indices(100, calibration_split=0.2, seed=2)
        assert a != b

    def test_single_sample_all_fit(self) -> None:
        fit, calib = split_normal_indices(1, calibration_split=0.2, seed=42)
        assert fit == [0]
        assert calib == []

    def test_empty_input(self) -> None:
        fit, calib = split_normal_indices(0, calibration_split=0.2, seed=42)
        assert fit == []
        assert calib == []

    def test_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError):
            split_normal_indices(10, calibration_split=1.0, seed=42)

    def test_at_least_one_each(self) -> None:
        fit, calib = split_normal_indices(2, calibration_split=0.9, seed=42)
        assert len(fit) >= 1
        assert len(calib) >= 1


class TestRunnerUsesDisjointSplit:
    def test_fit_and_calibration_subsets_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            dm = _SyntheticDataModule(
                _SyntheticDataset(10, 0), _SyntheticDataset(2, 2)
            )
            result = ExperimentRunner(
                cfg,
                datamodule=dm,
                model=_SyntheticModel(),
                save_heatmaps=False,
                calibration_split=0.2,
                seed=42,
            ).run()
            assert result.num_calibration_samples > 0
            assert result.num_fit_samples > 0
            assert (
                result.num_fit_samples + result.num_calibration_samples
                == result.num_train_samples
            )
            assert result.calibration_split == 0.2
            assert result.calibration_seed == 42