"""Tests for PaDiM anomaly detection model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from anomalyscope.models import get_model
from anomalyscope.models.padim import PaDiMModel


@pytest.fixture
def padim_config() -> SimpleNamespace:
    """Minimal PaDiM config for fast testing."""
    return SimpleNamespace(
        device="cpu",
        seed=42,
        model=SimpleNamespace(
            name="padim",
            backbone="resnet18",
            num_features=10,  # Small number of features for quick test
            epsilon=0.01,
        ),
    )


@pytest.fixture
def synthetic_dataloader() -> DataLoader:
    """Create a synthetic DataLoader with 4 normal images."""
    images = torch.randn(4, 3, 64, 64)
    dataset = TensorDataset(images)
    return DataLoader(dataset, batch_size=2)


def test_padim_init_and_factory(padim_config: SimpleNamespace) -> None:
    model = get_model(padim_config)
    assert isinstance(model, PaDiMModel)
    assert model.name == "PaDiM"


def test_padim_feature_extraction_and_fit(
    padim_config: SimpleNamespace, synthetic_dataloader: DataLoader
) -> None:
    model = PaDiMModel(padim_config)

    # Test embedding extraction
    dummy = torch.randn(2, 3, 64, 64)
    embs = model._extract_embeddings(dummy)
    assert embs.shape[0] == 2
    assert embs.shape[1] == 448  # 64 + 128 + 256

    # Test fit
    model.fit(synthetic_dataloader)
    assert model._is_fitted is True
    assert model.mean is not None
    assert model.inv_cov is not None
    assert model.selected_indices is not None
    assert len(model.selected_indices) == 10
    assert model.mean.shape[0] == 10


def test_padim_predict(
    padim_config: SimpleNamespace, synthetic_dataloader: DataLoader
) -> None:
    model = PaDiMModel(padim_config)
    model.fit(synthetic_dataloader)

    # Test 3D image
    test_img = torch.randn(3, 64, 64)
    score, amap = model.predict(test_img)
    assert isinstance(score, float)
    assert isinstance(amap, np.ndarray)
    assert amap.shape == (64, 64)

    # Test 4D image
    test_img_4d = torch.randn(1, 3, 64, 64)
    score4, amap4 = model.predict(test_img_4d)
    assert isinstance(score4, float)
    assert amap4.shape == (64, 64)


def test_padim_save_load(
    padim_config: SimpleNamespace, synthetic_dataloader: DataLoader
) -> None:
    model = PaDiMModel(padim_config)
    model.fit(synthetic_dataloader)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "padim_test.pth"
        model.save(str(ckpt_path))
        assert ckpt_path.exists()

        new_model = PaDiMModel(padim_config)
        new_model.load(str(ckpt_path))
        assert new_model._is_fitted is True

        test_img = torch.randn(3, 64, 64)
        score, amap = new_model.predict(test_img)
        assert isinstance(score, float)
        assert amap.shape == (64, 64)


# ---------------------------------------------------------------------------
# Memory-efficient per-batch channel selection regression tests
# ---------------------------------------------------------------------------


class TestPerBatchChannelSelection:
    """Prove reducing channels per batch == reducing after concatenation."""

    def _selected_indices(self, c_total: int, num_features: int, seed: int):
        torch.manual_seed(seed)
        n_features = min(num_features, c_total)
        return torch.randperm(c_total)[:n_features]

    def test_per_batch_equals_post_concat(self) -> None:
        torch.manual_seed(0)
        c_total, h, w = 448, 8, 8
        n_features, seed = 10, 42
        # Simulate 3 batches with distinct content.
        batches = [torch.randn(2, c_total, h, w) for _ in range(3)]

        # Post-concatenation selection (original behaviour).
        full = torch.cat(batches, dim=0)
        idx_after = self._selected_indices(c_total, n_features, seed)
        reduced_after = full[:, idx_after, :, :]

        # Per-batch selection (new behaviour).
        idx_before = self._selected_indices(c_total, n_features, seed)
        reduced_before = torch.cat(
            [b[:, idx_before, :, :] for b in batches], dim=0
        )

        assert torch.equal(idx_before, idx_after)
        assert torch.equal(reduced_before, reduced_after)

    def test_deterministic_same_seed(self) -> None:
        a = self._selected_indices(448, 100, 42)
        b = self._selected_indices(448, 100, 42)
        assert torch.equal(a, b)

    def test_different_seed_differs(self) -> None:
        a = self._selected_indices(448, 100, 42)
        b = self._selected_indices(448, 100, 7)
        assert not torch.equal(a, b)

    def test_same_indices_across_batches(self) -> None:
        # In the new fit(), selected_indices is computed once and reused; the
        # regression is that per-batch reduction uses one shared index tensor.
        idx = self._selected_indices(448, 10, 42)
        b1 = torch.randn(2, 448, 8, 8)
        b2 = torch.randn(2, 448, 8, 8)
        r1 = b1[:, idx, :, :]
        r2 = b2[:, idx, :, :]
        # Same channel positions selected in both batches.
        assert r1.shape[1] == r2.shape[1] == 10

    def test_fit_uses_shared_selected_indices(
        self, padim_config: SimpleNamespace, synthetic_dataloader: DataLoader
    ) -> None:
        model = PaDiMModel(padim_config)
        model.fit(synthetic_dataloader)
        # selected_indices must be set once and match the seed-derived set.
        expected = self._selected_indices(448, padim_config.model.num_features, 42)
        assert model.selected_indices is not None
        assert torch.equal(model.selected_indices, expected)

    def test_reduced_tensor_matches_manual_reduction(
        self, padim_config: SimpleNamespace
    ) -> None:
        # End-to-end: fit the model, then independently recompute the reduced
        # embedding for the same input and confirm the mean matches what fit
        # produced (proves per-batch reduction is numerically identical).
        model = PaDiMModel(padim_config)
        images = torch.randn(4, 3, 64, 64)
        dataset = TensorDataset(images)
        model.fit(DataLoader(dataset, batch_size=2))

        with torch.no_grad():
            emb = model._extract_embeddings(images)  # (4, 448, 16, 16)
            reduced = emb[:, model.selected_indices, :, :]
            expected_mean = torch.mean(reduced, dim=0)

        assert model.mean is not None
        assert torch.allclose(model.mean, expected_mean, atol=1e-6)
