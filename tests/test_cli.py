"""Focused tests for the training and evaluation CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import anomalyscope.evaluate as evaluate_cli
import anomalyscope.train as train_cli


# ---------------------------------------------------------------------------
# train.py — argument parsing & config assembly
# ---------------------------------------------------------------------------


class TestTrainArgParsing:
    def test_required_args(self) -> None:
        parser = train_cli.build_arg_parser()
        args = parser.parse_args(["--model", "padim", "--category", "bottle"])
        assert args.model == "padim"
        assert args.category == "bottle"

    def test_defaults(self) -> None:
        parser = train_cli.build_arg_parser()
        args = parser.parse_args(["--model", "autoencoder", "--category", "hazelnut"])
        assert args.percentile == 99.5
        assert args.calibration_split == 0.2
        assert args.no_heatmaps is False

    def test_invalid_model_rejected(self) -> None:
        parser = train_cli.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "vae", "--category", "bottle"])

    def test_invalid_category_rejected(self) -> None:
        parser = train_cli.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "padim", "--category", "grid"])


class TestTrainConfigHelpers:
    def test_resolve_model_config_path_default(self) -> None:
        p = train_cli.resolve_model_config_path("padim")
        assert p.name == "padim.yaml"
        assert p.parent.name == "configs"

    def test_resolve_model_config_path_explicit(self) -> None:
        p = train_cli.resolve_model_config_path("padim", config="custom/x.yaml")
        assert p == Path("custom/x.yaml")

    def test_build_overrides_only_set(self) -> None:
        args = SimpleNamespace(dataset_root="/data", category="bottle", device=None)
        ov = train_cli.build_overrides(args)
        assert ov == {"dataset_root": "/data", "category": "bottle"}

    def test_build_overrides_empty(self) -> None:
        args = SimpleNamespace(dataset_root=None, category=None, device=None)
        assert train_cli.build_overrides(args) == {}

    def test_make_config_builds_namespace(self) -> None:
        args = SimpleNamespace(
            model="padim",
            config=None,
            dataset_root="/tmp/dataset",
            category="bottle",
            device="cpu",
        )
        cfg = train_cli.make_config(args)
        assert cfg.category == "bottle"
        assert cfg.device == "cpu"
        assert cfg.model.name == "padim"


# ---------------------------------------------------------------------------
# train.py — end-to-end with a stubbed runner
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self) -> None:
        self.model_name = "Fake"
        self.category = "bottle"
        self.num_fit_samples = 8
        self.num_calibration_samples = 2
        self.num_test_samples = 8
        self.image_auroc = 0.9
        self.pixel_auroc = 0.8
        self.aupro = 0.7
        self.threshold = 0.5
        self.train_time_sec = 1.0
        self.inference_time_sec = 0.5
        self.checkpoint_path = "/tmp/fake.pth"


class _StubRunner:
    last_kwargs: dict = {}

    def __init__(self, config, **kwargs) -> None:
        _StubRunner.last_kwargs = {"config": config, **kwargs}

    def run(self) -> _FakeResult:
        return _FakeResult()


def test_train_main_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(train_cli, "ExperimentRunner", _StubRunner)
    rc = train_cli.main(
        [
            "--model",
            "padim",
            "--category",
            "bottle",
            "--dataset-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--no-heatmaps",
        ]
    )
    assert rc == 0
    assert _StubRunner.last_kwargs["config"].category == "bottle"


def test_train_main_failure_returns_one(monkeypatch) -> None:
    class _BoomRunner:
        def __init__(self, config, **kwargs) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(train_cli, "ExperimentRunner", _BoomRunner)
    rc = train_cli.main(["--model", "padim", "--category", "bottle"])
    assert rc == 1


# ---------------------------------------------------------------------------
# evaluate.py — argument parsing & helpers
# ---------------------------------------------------------------------------


class TestEvaluateArgParsing:
    def test_required_args(self) -> None:
        parser = evaluate_cli.build_arg_parser()
        args = parser.parse_args(["--model", "patchcore", "--category", "hazelnut"])
        assert args.model == "patchcore"
        assert args.category == "hazelnut"
        assert args.retrain is False

    def test_retrain_flag(self) -> None:
        parser = evaluate_cli.build_arg_parser()
        args = parser.parse_args(
            ["--model", "padim", "--category", "bottle", "--retrain"]
        )
        assert args.retrain is True


class TestEvaluateHelpers:
    def test_default_checkpoint_path(self) -> None:
        p = evaluate_cli.default_checkpoint_path("/ckpt", "PaDiM", "bottle")
        assert p == Path("/ckpt/padim_bottle.pth")

    def test_load_threshold_from_results_present(self, tmp_path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "padim_bottle.json").write_text(
            json.dumps({"threshold": 1.234})
        )
        val = evaluate_cli.load_threshold_from_results(tmp_path, "padim", "bottle")
        assert val == pytest.approx(1.234)

    def test_load_threshold_from_results_missing(self, tmp_path) -> None:
        assert (
            evaluate_cli.load_threshold_from_results(tmp_path, "padim", "bottle")
            is None
        )

    def test_load_threshold_from_results_no_key(self, tmp_path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "padim_bottle.json").write_text(json.dumps({"image_auroc": 0.9}))
        assert (
            evaluate_cli.load_threshold_from_results(tmp_path, "padim", "bottle")
            is None
        )


# ---------------------------------------------------------------------------
# evaluate.py — evaluate-only path with stubs
# ---------------------------------------------------------------------------


class _StubModel:
    def __init__(self) -> None:
        self.loaded_from = None

    def load(self, path: str) -> None:
        self.loaded_from = path


class _StubEvalRunner:
    last: dict = {}

    def __init__(self, config, **kwargs) -> None:
        _StubEvalRunner.last = {"config": config, **kwargs}

    def evaluate(self, threshold=None) -> _FakeResult:
        r = _FakeResult()
        r.threshold = threshold
        return r

    def run(self) -> _FakeResult:
        return _FakeResult()


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy")


def test_evaluate_loads_checkpoint_and_threshold(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "ckpt" / "padim_bottle.pth"
    _write_checkpoint(ckpt)
    model = _StubModel()

    monkeypatch.setattr(evaluate_cli, "get_model", lambda cfg: model)
    monkeypatch.setattr(evaluate_cli, "ExperimentRunner", _StubEvalRunner)

    rc = evaluate_cli.main(
        [
            "--model",
            "padim",
            "--category",
            "bottle",
            "--checkpoint",
            str(ckpt),
            "--threshold",
            "0.75",
        ]
    )
    assert rc == 0
    assert model.loaded_from == str(ckpt)
    assert _StubEvalRunner.last["config"] is not None


def test_evaluate_missing_checkpoint_returns_one(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(evaluate_cli, "get_model", lambda cfg: _StubModel())
    monkeypatch.setattr(evaluate_cli, "ExperimentRunner", _StubEvalRunner)
    rc = evaluate_cli.main(
        [
            "--model",
            "padim",
            "--category",
            "bottle",
            "--checkpoint",
            str(tmp_path / "nope.pth"),
            "--threshold",
            "0.5",
        ]
    )
    assert rc == 1


def test_evaluate_retrain_uses_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(evaluate_cli, "ExperimentRunner", _StubEvalRunner)
    rc = evaluate_cli.main(
        [
            "--model",
            "padim",
            "--category",
            "bottle",
            "--dataset-root",
            str(tmp_path),
            "--retrain",
        ]
    )
    assert rc == 0