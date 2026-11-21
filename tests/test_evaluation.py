"""Focused tests for evaluation metrics using synthetic data."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from anomalyscope.evaluation.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro,
    evaluate_model,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


class _SyntheticAnomalyDataset(Dataset):
    """Minimal dataset returning dicts compatible with MVTecDataset."""

    def __init__(
        self,
        num_normal: int = 4,
        num_anomalous: int = 4,
        height: int = 32,
        width: int = 32,
    ) -> None:
        super().__init__()
        self.samples: list[dict] = []
        for _ in range(num_normal):
            self.samples.append(self._make_sample(label=0, h=height, w=width))
        for _ in range(num_anomalous):
            self.samples.append(self._make_sample(label=1, h=height, w=width))

    def _make_sample(
        self, label: int, h: int, w: int
    ) -> dict:
        image = torch.randn(3, h, w)
        if label == 0:
            mask = torch.zeros(1, h, w, dtype=torch.float32)
            defect = "good"
        else:
            mask = torch.zeros(1, h, w, dtype=torch.float32)
            # Place a small square anomaly in the centre.
            c_h, c_w = h // 2, w // 2
            mask[0, c_h - 4 : c_h + 4, c_w - 4 : c_w + 4] = 1.0
            defect = "synthetic_defect"
        return {
            "image": image,
            "mask": mask,
            "label": label,
            "path": f"/synthetic/{label}/{len(self.samples)}.png",
            "defect_type": defect,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class _SyntheticModel:
    """Mock anomaly model for evaluate_model testing."""

    def __init__(self) -> None:
        self._is_fitted = True

    def predict(self, image: torch.Tensor) -> tuple[float, np.ndarray]:
        """Return deterministic scores based on image content."""
        # Simulate: normal images get low scores, anomalous get high.
        # The dataset doesn't carry labels into predict, so we use
        # image variance as a proxy (synthetic normals have lower variance).
        h, w = image.shape[-2], image.shape[-1]
        amap = torch.rand(h, w).numpy().astype(np.float32)
        # Add a hot region near centre for "anomalous" detection.
        c_h, c_w = h // 2, w // 2
        amap[c_h - 4 : c_h + 4, c_w - 4 : c_w + 4] += 5.0
        score = float(amap.max())
        return score, amap

    @property
    def name(self) -> str:
        return "SyntheticModel"


# ---------------------------------------------------------------------------
# Image-level AUROC
# ---------------------------------------------------------------------------


class TestImageAuroc:
    def test_perfect_separation(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
        labels = [0, 0, 0, 1, 1, 1]
        auroc = compute_image_auroc(scores, labels)
        assert auroc == 1.0

    def test_random_scores(self) -> None:
        scores = [0.5, 0.3, 0.8, 0.2, 0.7, 0.1]
        labels = [0, 1, 0, 1, 0, 1]
        auroc = compute_image_auroc(scores, labels)
        assert 0.0 <= auroc <= 1.0

    def test_single_class_returns_half(self) -> None:
        scores = [0.1, 0.2, 0.3]
        labels = [0, 0, 0]
        auroc = compute_image_auroc(scores, labels)
        assert auroc == 0.5

    def test_numpy_inputs(self) -> None:
        scores = np.array([0.1, 0.9])
        labels = np.array([0, 1])
        auroc = compute_image_auroc(scores, labels)
        assert auroc == 1.0

    def test_nan_scores_handled(self) -> None:
        scores = [0.1, float("nan"), 0.9]
        labels = [0, 1, 1]
        auroc = compute_image_auroc(scores, labels)
        assert 0.0 <= auroc <= 1.0


# ---------------------------------------------------------------------------
# Pixel-level AUROC
# ---------------------------------------------------------------------------


class TestPixelAuroc:
    def test_perfect_separation(self) -> None:
        # 2 images of 2x2, one normal (all 0) and one anomalous.
        maps = [
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.9, 0.9], [0.9, 0.9]]),
        ]
        masks = [
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
        ]
        auroc = compute_pixel_auroc(maps, masks)
        assert auroc == 1.0

    def test_single_class_returns_half(self) -> None:
        maps = [np.array([[0.0, 0.0], [0.0, 0.0]])]
        masks = [np.array([[0.0, 0.0], [0.0, 0.0]])]
        auroc = compute_pixel_auroc(maps, masks)
        assert auroc == 0.5

    def test_shape_mismatch_resized(self) -> None:
        maps = [np.zeros((4, 4), dtype=np.float32)]
        masks = [np.zeros((2, 2), dtype=np.float32)]
        # Should not raise, mask resized to match.
        auroc = compute_pixel_auroc(maps, masks)
        assert 0.0 <= auroc <= 1.0

    def test_tensor_inputs(self) -> None:
        maps = [torch.tensor([[0.0, 0.0], [0.9, 0.9]])]
        masks = [torch.tensor([[0.0, 0.0], [1.0, 1.0]])]
        auroc = compute_pixel_auroc(maps, masks)
        assert auroc == 1.0


# ---------------------------------------------------------------------------
# PRO / AUPRO
# ---------------------------------------------------------------------------


class TestPro:
    def test_perfect_detection(self) -> None:
        # Two images: one all-normal, one all-anomalous.
        maps = [
            np.zeros((8, 8), dtype=np.float32),
            np.ones((8, 8), dtype=np.float32) * 0.9,
        ]
        masks = [
            np.zeros((8, 8), dtype=np.float32),
            np.ones((8, 8), dtype=np.float32),
        ]
        aupro = compute_pro(maps, masks)
        # Should be high but not necessarily 1.0 due to FPR.
        assert aupro > 0.5

    def test_no_anomalous_masks_returns_zero(self) -> None:
        maps = [np.zeros((8, 8), dtype=np.float32)]
        masks = [np.zeros((8, 8), dtype=np.float32)]
        aupro = compute_pro(maps, masks)
        assert aupro == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            compute_pro(
                [np.zeros((4, 4))],
                [np.zeros((4, 4)), np.zeros((4, 4))],
            )

    def test_degenerate_scores_returns_zero(self) -> None:
        # All scores identical → no thresholding possible.
        maps = [np.ones((4, 4), dtype=np.float32) * 0.5]
        masks = [np.ones((4, 4), dtype=np.float32)]
        aupro = compute_pro(maps, masks)
        assert aupro == 0.0

    def test_partial_detection(self) -> None:
        # One large anomalous region, map only covers half of it.
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[2:6, 2:6] = 1.0
        amap = np.zeros((8, 8), dtype=np.float32)
        amap[2:4, 2:6] = 0.9  # top half detected
        aupro = compute_pro([amap], [mask])
        assert 0.0 < aupro < 1.0

    def test_well_separated_high_aupro(self) -> None:
        # Regression: FPR decreases as thresholds increase, so the raw
        # FPR array is descending. np.interp requires ascending x, so the
        # implementation must sort before interpolating. A well-separated
        # case must yield a high AUPRO (not a near-zero artefact).
        rng = np.random.RandomState(0)
        amap = rng.rand(16, 16).astype(np.float32) * 0.5
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[4:12, 4:12] = 1.0
        amap[4:12, 4:12] += 1.0  # anomalous region clearly hotter
        aupro = compute_pro([amap], [mask])
        assert aupro > 0.5


# ---------------------------------------------------------------------------
# evaluate_model integration
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_basic_evaluation(self) -> None:
        dataset = _SyntheticAnomalyDataset(
            num_normal=4, num_anomalous=4, height=32, width=32
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = _SyntheticModel()

        result = evaluate_model(model, loader, device="cpu")

        assert "image_auroc" in result
        assert "pixel_auroc" in result
        assert "aupro" in result
        assert "per_defect_results" in result
        assert "num_samples" in result
        assert result["num_samples"] == 8

    def test_returns_float_metrics(self) -> None:
        dataset = _SyntheticAnomalyDataset(
            num_normal=2, num_anomalous=2, height=16, width=16
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = _SyntheticModel()

        result = evaluate_model(model, loader, device="cpu")

        assert isinstance(result["image_auroc"], float)
        assert isinstance(result["pixel_auroc"], float)
        assert isinstance(result["aupro"], float)

    def test_metrics_in_valid_range(self) -> None:
        dataset = _SyntheticAnomalyDataset(
            num_normal=3, num_anomalous=3, height=16, width=16
        )
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        model = _SyntheticModel()

        result = evaluate_model(model, loader, device="cpu")

        assert 0.0 <= result["image_auroc"] <= 1.0
        assert 0.0 <= result["pixel_auroc"] <= 1.0
        assert 0.0 <= result["aupro"] <= 1.0

    def test_empty_dataloader(self) -> None:
        class _EmptyDataset(Dataset):
            def __len__(self) -> int:
                return 0

            def __getitem__(self, index: int) -> dict:
                raise IndexError

        loader = DataLoader(_EmptyDataset(), batch_size=1)
        model = _SyntheticModel()

        result = evaluate_model(model, loader, device="cpu")
        assert result["image_auroc"] == 0.0
        assert result["num_samples"] == 0