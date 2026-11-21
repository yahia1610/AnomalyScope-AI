"""Tests for Convolutional Autoencoder model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from anomalyscope.models import get_model
from anomalyscope.models.autoencoder import AutoencoderModel


@pytest.fixture
def ae_config() -> SimpleNamespace:
    """Minimal config for fast testing."""
    return SimpleNamespace(
        device="cpu",
        model=SimpleNamespace(
            name="autoencoder",
            encoder_channels=[16, 32],
            learning_rate=1e-3,
            weight_decay=0.0,
            epochs=2,
            scheduler_step=1,
            scheduler_gamma=0.5,
        ),
    )


@pytest.fixture
def synthetic_dataloader() -> DataLoader:
    """Create a synthetic DataLoader with 4 batches of size 2."""
    images = torch.randn(8, 3, 64, 64)
    dataset = TensorDataset(images)
    return DataLoader(dataset, batch_size=2)


def test_autoencoder_init_and_factory(ae_config: SimpleNamespace) -> None:
    model = get_model(ae_config)
    assert isinstance(model, AutoencoderModel)
    assert model.name == "Autoencoder"


def test_autoencoder_forward_and_fit(
    ae_config: SimpleNamespace, synthetic_dataloader: DataLoader
) -> None:
    model = AutoencoderModel(ae_config)
    # Test forward pass directly
    dummy_input = torch.randn(2, 3, 64, 64)
    out = model.net(dummy_input)
    assert out.shape == dummy_input.shape

    # Test fit
    model.fit(synthetic_dataloader)
    assert model._is_fitted is True


def test_autoencoder_predict(
    ae_config: SimpleNamespace, synthetic_dataloader: DataLoader
) -> None:
    model = AutoencoderModel(ae_config)
    model.fit(synthetic_dataloader)

    # Test single 3D image
    test_img_3d = torch.randn(3, 64, 64)
    score, amap = model.predict(test_img_3d)

    assert isinstance(score, float)
    assert isinstance(amap, np.ndarray)
    assert amap.shape == (64, 64)

    # Test 4D tensor (1, 3, 64, 64)
    test_img_4d = torch.randn(1, 3, 64, 64)
    score4, amap4 = model.predict(test_img_4d)
    assert isinstance(score4, float)
    assert amap4.shape == (64, 64)


def test_autoencoder_save_load(
    ae_config: SimpleNamespace, synthetic_dataloader: DataLoader
) -> None:
    model = AutoencoderModel(ae_config)
    model.fit(synthetic_dataloader)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ae_test.pth"
        model.save(str(ckpt_path))
        assert ckpt_path.exists()

        # Load into a new model
        new_model = AutoencoderModel(ae_config)
        new_model.load(str(ckpt_path))
        assert new_model._is_fitted is True

        # Test predict with loaded model
        test_img = torch.randn(3, 64, 64)
        score, amap = new_model.predict(test_img)
        assert isinstance(score, float)
        assert amap.shape == (64, 64)


# ---------------------------------------------------------------------------
# Periodic recovery checkpointing and resume
# ---------------------------------------------------------------------------


def _recovery_config(**overrides) -> SimpleNamespace:
    base = dict(
        device="cpu",
        model=SimpleNamespace(
            name="autoencoder",
            encoder_channels=[16, 32],
            learning_rate=1e-3,
            weight_decay=0.0,
            epochs=4,
            scheduler_step=1,
            scheduler_gamma=0.5,
            checkpoint_every=2,
        ),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _loader(n: int = 8, batch_size: int = 2) -> DataLoader:
    torch.manual_seed(0)
    images = torch.randn(n, 3, 64, 64)
    return DataLoader(TensorDataset(images), batch_size=batch_size)


class TestRecoveryCheckpointing:
    def test_periodic_checkpoint_created(self, tmp_path: Path) -> None:
        rec = tmp_path / "ae.recovery.pth"
        model = AutoencoderModel(_recovery_config())
        model.fit(_loader(), recovery_path=str(rec), checkpoint_every=2)
        assert rec.exists()

    def test_no_recovery_when_path_none(self, tmp_path: Path) -> None:
        model = AutoencoderModel(_recovery_config())
        # No recovery_path -> behaves like the original fit (no files written).
        model.fit(_loader())
        assert model._is_fitted is True
        assert list(tmp_path.iterdir()) == []

    def test_recovery_epoch_restored(self, tmp_path: Path) -> None:
        rec = tmp_path / "ae.recovery.pth"
        model = AutoencoderModel(_recovery_config())  # epochs=4, every=2
        model.fit(_loader(), recovery_path=str(rec), checkpoint_every=2)
        state = torch.load(rec, map_location="cpu", weights_only=False)
        # Last recovery saved at final epoch (4).
        assert state["epoch"] == 4

    def test_optimizer_and_scheduler_state_saved(self, tmp_path: Path) -> None:
        rec = tmp_path / "ae.recovery.pth"
        model = AutoencoderModel(_recovery_config())
        model.fit(_loader(), recovery_path=str(rec), checkpoint_every=2)
        state = torch.load(rec, map_location="cpu", weights_only=False)
        assert "optimizer_state_dict" in state
        assert "scheduler_state_dict" in state
        # Scheduler state exposes the number of completed steps.
        assert state["scheduler_state_dict"]["last_epoch"] == 4

    def test_resume_continues_from_next_epoch(self, tmp_path: Path) -> None:
        rec = tmp_path / "ae.recovery.pth"
        # Phase 1: train only 2 epochs.
        cfg = _recovery_config()
        cfg.model.epochs = 2
        m1 = AutoencoderModel(cfg)
        m1.fit(_loader(), recovery_path=str(rec), checkpoint_every=2)
        s1 = torch.load(rec, map_location="cpu", weights_only=False)
        assert s1["epoch"] == 2
        w_after_phase1 = m1.net.state_dict()
        sch_after_phase1 = s1["scheduler_state_dict"]["last_epoch"]

        # Phase 2: resume with total epochs=4 -> should run epochs 3 and 4.
        cfg2 = _recovery_config()  # epochs=4
        m2 = AutoencoderModel(cfg2)
        m2.fit(_loader(), recovery_path=str(rec), resume=True, checkpoint_every=2)
        s2 = torch.load(rec, map_location="cpu", weights_only=False)
        assert s2["epoch"] == 4
        # Scheduler advanced from 2 to 4 (two additional steps).
        assert sch_after_phase1 == 2
        assert s2["scheduler_state_dict"]["last_epoch"] == 4
        # Weights changed during resumed training (epochs 3-4 ran).
        changed = any(
            not torch.equal(w_after_phase1[k], m2.net.state_dict()[k])
            for k in w_after_phase1
        )
        assert changed

    def test_resume_incompatible_raises(self, tmp_path: Path) -> None:
        rec = tmp_path / "ae.recovery.pth"
        m1 = AutoencoderModel(_recovery_config())
        m1.fit(_loader(), recovery_path=str(rec), checkpoint_every=2)

        # Different architecture -> must raise, not silently resume.
        bad_cfg = _recovery_config()
        bad_cfg.model.encoder_channels = [8, 16]
        m2 = AutoencoderModel(bad_cfg)
        with pytest.raises(ValueError, match="Incompatible"):
            m2.fit(_loader(), recovery_path=str(rec), resume=True)

    def test_resume_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        m = AutoencoderModel(_recovery_config())
        with pytest.raises(Exception):
            m.fit(
                _loader(),
                recovery_path=str(tmp_path / "nope.pth"),
                resume=True,
            )

    def test_resume_without_path_raises(self) -> None:
        m = AutoencoderModel(_recovery_config())
        with pytest.raises(ValueError, match="requires a recovery_path"):
            m.fit(_loader(), resume=True)

    def test_resume_already_complete_is_noop(self, tmp_path: Path) -> None:
        rec = tmp_path / "ae.recovery.pth"
        m1 = AutoencoderModel(_recovery_config())  # epochs=4
        m1.fit(_loader(), recovery_path=str(rec), checkpoint_every=2)
        w_before = m1.net.state_dict()

        # Resume with the same total epochs (already complete) -> no training.
        m2 = AutoencoderModel(_recovery_config())
        m2.fit(_loader(), recovery_path=str(rec), resume=True)
        for k in w_before:
            assert torch.equal(w_before[k], m2.net.state_dict()[k])

    def test_final_checkpoint_compatible_with_load(
        self, tmp_path: Path
    ) -> None:
        rec = tmp_path / "ae.recovery.pth"
        final = tmp_path / "ae_final.pth"
        model = AutoencoderModel(_recovery_config())
        model.fit(_loader(), recovery_path=str(rec), checkpoint_every=2)
        model.save(str(final))
        # Recovery file must be separate from the final checkpoint.
        assert rec.exists() and final.exists() and rec != final

        # Final checkpoint loadable via existing load() + predict (evaluate CLI).
        loaded = AutoencoderModel(_recovery_config())
        loaded.load(str(final))
        assert loaded._is_fitted is True
        score, amap = loaded.predict(torch.randn(3, 64, 64))
        assert isinstance(score, float)
        assert amap.shape == (64, 64)
