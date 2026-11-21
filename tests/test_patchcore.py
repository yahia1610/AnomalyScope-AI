"""Tests for PatchCore anomaly detection model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from anomalyscope.models import get_model
from anomalyscope.models.patchcore import PatchCoreModel


@pytest.fixture
def patchcore_config() -> SimpleNamespace:
    """Minimal PatchCore config for fast testing using ResNet-18."""
    return SimpleNamespace(
        device="cpu",
        seed=42,
        model=SimpleNamespace(
            name="patchcore",
            backbone="resnet18",
            coreset_ratio=0.1,
            num_neighbors=3,
            pool_size=3,
        ),
    )


@pytest.fixture
def synthetic_dataloader() -> DataLoader:
    """Create a synthetic DataLoader with 4 normal images."""
    images = torch.randn(4, 3, 64, 64)
    dataset = TensorDataset(images)
    return DataLoader(dataset, batch_size=2)


def test_patchcore_init_and_factory(patchcore_config: SimpleNamespace) -> None:
    model = get_model(patchcore_config)
    assert isinstance(model, PatchCoreModel)
    assert model.name == "PatchCore"


def test_patchcore_feature_extraction_and_fit(
    patchcore_config: SimpleNamespace, synthetic_dataloader: DataLoader
) -> None:
    model = PatchCoreModel(patchcore_config)

    # Test patch feature extraction
    dummy = torch.randn(2, 3, 64, 64)
    patches = model._extract_patch_features(dummy)
    assert patches.shape[0] == 2

    # Test fit
    model.fit(synthetic_dataloader)
    assert model._is_fitted is True
    assert model.memory_bank is not None
    assert model.memory_bank.ndim == 2
    assert model.memory_bank.shape[0] > 0


def test_patchcore_predict(
    patchcore_config: SimpleNamespace, synthetic_dataloader: DataLoader
) -> None:
    model = PatchCoreModel(patchcore_config)
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


def test_patchcore_save_load(
    patchcore_config: SimpleNamespace, synthetic_dataloader: DataLoader
) -> None:
    model = PatchCoreModel(patchcore_config)
    model.fit(synthetic_dataloader)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "patchcore_test.pth"
        model.save(str(ckpt_path))
        assert ckpt_path.exists()

        new_model = PatchCoreModel(patchcore_config)
        new_model.load(str(ckpt_path))
        assert new_model._is_fitted is True
        assert new_model.memory_bank is not None

        test_img = torch.randn(3, 64, 64)
        score, amap = new_model.predict(test_img)
        assert isinstance(score, float)
        assert amap.shape == (64, 64)


# ---------------------------------------------------------------------------
# Two-pass memory-efficient coreset: equivalence regression tests
# ---------------------------------------------------------------------------


def _reference_fit(model: PatchCoreModel, dataloader: DataLoader):
    """Reproduce the ORIGINAL single-pass algorithm as a reference.

    Returns (selected_indices, coreset) using the full (N, D) bank and the
    original seeded greedy selection (verbatim logic from the previous
    implementation).
    """
    model.backbone.eval()
    all_patches = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"] if isinstance(batch, dict) else batch[0]
            patches = model._extract_patch_features(images)
            b, d, h, w = patches.shape
            all_patches.append(
                patches.permute(0, 2, 3, 1).contiguous().view(-1, d).cpu()
            )
    full = torch.cat(all_patches, dim=0)
    n, d = full.shape
    target_m = max(1, int(n * model.coreset_ratio))
    proj_dim = min(128, d)
    torch.manual_seed(model.seed)
    proj = torch.randn(d, proj_dim) / np.sqrt(proj_dim)
    fproj = torch.matmul(full, proj)
    sel = [int(torch.randint(0, n, (1,)).item())]
    mind = torch.full((n,), float("inf"), dtype=torch.float32)
    for step in range(1, target_m):
        last = fproj[sel[-1]].unsqueeze(0)
        for s in range(0, n, 5000):
            e = min(s + 5000, n)
            dd = torch.norm(fproj[s:e] - last, dim=1)
            mind[s:e] = torch.minimum(mind[s:e], dd)
        sel.append(int(torch.argmax(mind).item()))
    coreset = full[sel]
    return sel, coreset


class TestTwoPassCoresetEquivalence:
    def _config(self, ratio: float = 0.1):
        return SimpleNamespace(
            device="cpu",
            seed=42,
            model=SimpleNamespace(
                name="patchcore",
                backbone="resnet18",
                coreset_ratio=ratio,
                num_neighbors=3,
                pool_size=3,
            ),
        )

    def _loader(self, n: int = 6, batch_size: int = 2) -> DataLoader:
        torch.manual_seed(0)
        images = torch.randn(n, 3, 64, 64)
        return DataLoader(TensorDataset(images), batch_size=batch_size)

    def test_selected_indices_match_reference(self) -> None:
        model = PatchCoreModel(self._config())
        ref_sel, _ = _reference_fit(PatchCoreModel(self._config()), self._loader())
        model.fit(self._loader())
        assert model.selected_indices is not None
        assert model.selected_indices == ref_sel

    def test_memory_bank_matches_reference(self) -> None:
        model = PatchCoreModel(self._config())
        _, ref_coreset = _reference_fit(
            PatchCoreModel(self._config()), self._loader()
        )
        model.fit(self._loader())
        assert model.memory_bank is not None
        assert torch.equal(model.memory_bank, ref_coreset)

    def test_same_seed_same_coreset(self) -> None:
        m1 = PatchCoreModel(self._config())
        m2 = PatchCoreModel(self._config())
        m1.fit(self._loader())
        m2.fit(self._loader())
        assert m1.selected_indices == m2.selected_indices
        assert torch.equal(m1.memory_bank, m2.memory_bank)

    def test_batch_size_does_not_change_selection(self) -> None:
        m2 = PatchCoreModel(self._config())
        m3 = PatchCoreModel(self._config())
        m2.fit(self._loader(batch_size=2))
        m3.fit(self._loader(batch_size=3))
        assert m2.selected_indices == m3.selected_indices
        assert torch.equal(m2.memory_bank, m3.memory_bank)

    def test_predictions_match_reference(self) -> None:
        model = PatchCoreModel(self._config())
        model.fit(self._loader())
        # Reference model with identical memory bank.
        ref = PatchCoreModel(self._config())
        _, ref_coreset = _reference_fit(
            PatchCoreModel(self._config()), self._loader()
        )
        ref.memory_bank = ref_coreset
        ref._is_fitted = True

        torch.manual_seed(123)
        img = torch.randn(3, 64, 64)
        s1, m1 = model.predict(img)
        s2, m2 = ref.predict(img)
        assert abs(s1 - s2) < 1e-4
        assert np.allclose(m1, m2, atol=1e-4)

    def test_no_full_nxd_bank_retained(self) -> None:
        model = PatchCoreModel(self._config())
        loader = self._loader()
        model.fit(loader)
        # Compute N from the dataset.
        n_patches = 0
        with torch.no_grad():
            for batch in loader:
                images = batch[0]
                p = model._extract_patch_features(images)
                n_patches += p.shape[0] * p.shape[2] * p.shape[3]
        # No attribute should hold a full (N, D) tensor.
        for name, val in vars(model).items():
            if isinstance(val, torch.Tensor) and val.ndim == 2:
                assert val.shape[0] != n_patches or val.shape[1] != 1536, (
                    f"attribute {name} retains full N x D bank"
                )
        # The memory bank is the coreset (small), not the full bank.
        assert model.memory_bank.shape[0] < n_patches
