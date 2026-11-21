"""Convolutional Autoencoder for industrial visual anomaly detection.

This module implements a reconstruction-based anomaly detection model.
The autoencoder is trained exclusively on defect-free (normal) images.
At inference time, defects produce high reconstruction errors because
the bottleneck representations cannot represent unseen anomalies.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from anomalyscope.models.base import AnomalyModel
from anomalyscope.utils.device import get_device
from anomalyscope.utils.io import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class ConvEncoder(nn.Module):
    """Convolutional encoder network."""

    def __init__(
        self,
        in_channels: int = 3,
        channel_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if channel_dims is None:
            channel_dims = [32, 64, 128, 256]

        layers: list[nn.Module] = []
        curr_channels = in_channels

        for out_c in channel_dims:
            layers.extend(
                [
                    nn.Conv2d(curr_channels, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            curr_channels = out_c

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        return self.network(x)


class ConvDecoder(nn.Module):
    """Convolutional decoder network."""

    def __init__(
        self,
        out_channels: int = 3,
        channel_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if channel_dims is None:
            channel_dims = [256, 128, 64, 32]

        layers: list[nn.Module] = []

        for i in range(len(channel_dims) - 1):
            in_c = channel_dims[i]
            out_c = channel_dims[i + 1]
            layers.extend(
                [
                    nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
            )

        # Final projection to output channels (linear activation to match ImageNet normalized space)
        layers.append(
            nn.ConvTranspose2d(
                channel_dims[-1],
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder."""
        return self.network(x)


class AutoencoderNet(nn.Module):
    """Complete Convolutional Autoencoder network."""

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]

        self.encoder = ConvEncoder(in_channels=in_channels, channel_dims=encoder_channels)
        decoder_channels = list(reversed(encoder_channels))
        self.decoder = ConvDecoder(out_channels=in_channels, channel_dims=decoder_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input image tensor."""
        latent = self.encoder(x)
        return self.decoder(latent)


class AutoencoderModel(AnomalyModel):
    """Convolutional Autoencoder anomaly detection model adhering to AnomalyModel.

    Args:
        config: Configuration namespace (loaded from YAML).
    """

    def __init__(self, config: SimpleNamespace | Any) -> None:
        self.config = config
        pref = getattr(config, "device", "auto")
        self.device = get_device(pref)

        # Model hyperparameters
        model_cfg = getattr(config, "model", None)
        if model_cfg is not None:
            self.encoder_channels = list(getattr(model_cfg, "encoder_channels", [32, 64, 128, 256]))
            self.lr = float(getattr(model_cfg, "learning_rate", 1e-4))
            self.weight_decay = float(getattr(model_cfg, "weight_decay", 1e-5))
            self.epochs = int(getattr(model_cfg, "epochs", 200))
            self.scheduler_step = int(getattr(model_cfg, "scheduler_step", 50))
            self.scheduler_gamma = float(getattr(model_cfg, "scheduler_gamma", 0.5))
        else:
            self.encoder_channels = [32, 64, 128, 256]
            self.lr = 1e-4
            self.weight_decay = 1e-5
            self.epochs = 200
            self.scheduler_step = 50
            self.scheduler_gamma = 0.5

        # Interval (epochs) for periodic recovery checkpoints during fit().
        self.checkpoint_every = (
            int(getattr(model_cfg, "checkpoint_every", 20))
            if model_cfg is not None
            else 20
        )

        self.net = AutoencoderNet(in_channels=3, encoder_channels=self.encoder_channels)
        self.net.to(self.device)
        self._is_fitted = False

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "Autoencoder"

    def _recovery_config(self) -> dict[str, Any]:
        """Return the hyperparameters that must match for a safe resume."""
        return {
            "encoder_channels": list(self.encoder_channels),
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "scheduler_step": self.scheduler_step,
            "scheduler_gamma": self.scheduler_gamma,
        }

    def _save_recovery(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
        epoch: int,
    ) -> None:
        """Persist a recovery checkpoint (weights + optimizer + scheduler)."""
        state = {
            "model_name": self.name,
            "encoder_channels": list(self.encoder_channels),
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "config": self._recovery_config(),
            "is_fitted": True,
        }
        save_checkpoint(state, path)
        logger.info("Recovery checkpoint saved (epoch %d) -> %s", epoch, path)

    def _load_recovery(self, path: str) -> dict[str, Any]:
        """Load and validate a recovery checkpoint for safe resuming.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the checkpoint is missing required state or is
                incompatible with the current configuration (architecture or
                hyperparameters), to avoid silently resuming an incompatible
                run.
        """
        state = load_checkpoint(path, device=self.device)
        required = (
            "net_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
        )
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(
                f"Recovery checkpoint {path} is missing required keys: {missing}."
            )

        # Compatibility checks against the current model configuration.
        mismatches: list[str] = []
        ckpt_cfg = state.get("config", {}) or {}

        if list(state.get("encoder_channels", [])) != list(self.encoder_channels):
            mismatches.append(
                f"encoder_channels {state.get('encoder_channels')} != "
                f"{list(self.encoder_channels)}"
            )

        float_keys = {
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "scheduler_gamma": self.scheduler_gamma,
        }
        for key, current in float_keys.items():
            if key in ckpt_cfg and not math.isclose(
                float(ckpt_cfg[key]), float(current), rel_tol=1e-9, abs_tol=0.0
            ):
                mismatches.append(f"{key} {ckpt_cfg[key]} != {current}")

        int_keys = {"scheduler_step": self.scheduler_step}
        for key, current in int_keys.items():
            if key in ckpt_cfg and int(ckpt_cfg[key]) != int(current):
                mismatches.append(f"{key} {ckpt_cfg[key]} != {current}")

        # NOTE: total ``epochs`` is intentionally NOT a hard incompatibility:
        # resuming to train longer (or finding the run already complete) is a
        # legitimate operation.  Architecture and optimizer/scheduler
        # hyperparameters above remain strictly validated.

        if mismatches:
            raise ValueError(
                "Incompatible recovery checkpoint "
                f"{path}: " + "; ".join(mismatches)
            )

        return state

    def fit(
        self,
        dataloader: DataLoader,
        recovery_path: str | None = None,
        resume: bool = False,
        checkpoint_every: int | None = None,
    ) -> None:
        """Train the autoencoder on normal (defect-free) images.

        Args:
            dataloader: DataLoader yielding batches of normal images.
            recovery_path: If provided, periodic recovery checkpoints are
                written here (separate from the final ``save()`` checkpoint),
                containing weights, optimizer/scheduler state, completed epoch,
                and configuration.
            resume: If ``True``, resume from *recovery_path* at the epoch after
                the last completed one.  Raises if the checkpoint is missing or
                incompatible (never silently restarts).
            checkpoint_every: Interval in epochs for recovery checkpoints.
                Defaults to the model's ``checkpoint_every`` (20).
        """
        self.net.train()
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma,
        )
        criterion = nn.MSELoss(reduction="mean")

        start_epoch = 1
        if resume:
            if not recovery_path:
                raise ValueError("resume=True requires a recovery_path.")
            state = self._load_recovery(recovery_path)
            self.net.load_state_dict(state["net_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch = int(state["epoch"]) + 1
            logger.info(
                "Resumed Autoencoder from epoch %d; continuing at %d/%d.",
                int(state["epoch"]),
                start_epoch,
                self.epochs,
            )

        self.net.train()

        if start_epoch > self.epochs:
            logger.info(
                "Recovery checkpoint already completed epoch %d (>= %d); "
                "no further training needed.",
                start_epoch - 1,
                self.epochs,
            )
            self._is_fitted = True
            self.net.eval()
            return

        every = (
            int(checkpoint_every)
            if checkpoint_every is not None
            else int(self.checkpoint_every)
        )

        logger.info("Training %s for %d epochs on %s", self.name, self.epochs, self.device)

        for epoch in range(start_epoch, self.epochs + 1):
            total_loss = 0.0
            batch_count = 0

            for batch in dataloader:
                if isinstance(batch, dict):
                    images = batch["image"]
                elif isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                optimizer.zero_grad()
                reconstructions = self.net(images)
                loss = criterion(reconstructions, images)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            scheduler.step()
            avg_loss = total_loss / max(1, batch_count)
            if epoch % 20 == 0 or epoch == 1 or epoch == self.epochs:
                logger.info("Epoch [%d/%d] - Loss: %.6f", epoch, self.epochs, avg_loss)

            if recovery_path and every > 0 and (epoch % every == 0 or epoch == self.epochs):
                self._save_recovery(recovery_path, optimizer, scheduler, epoch)

        self._is_fitted = True
        self.net.eval()
        logger.info("Autoencoder training complete.")

    def predict(self, image: torch.Tensor) -> tuple[float, np.ndarray]:
        """Compute the anomaly score and anomaly map for a single image.

        Args:
            image: Preprocessed image tensor of shape ``(C, H, W)`` or ``(1, C, H, W)``.

        Returns:
            Tuple of (anomaly_score, anomaly_map) where anomaly_map has shape ``(H, W)``.
        """
        self.net.eval()

        if image.ndim == 3:
            image_batch = image.unsqueeze(0)
        elif image.ndim == 4:
            image_batch = image
        else:
            raise ValueError(f"Expected 3D or 4D image tensor, got shape {image.shape}")

        orig_h, orig_w = image_batch.shape[-2], image_batch.shape[-1]
        image_batch = image_batch.to(self.device)

        with torch.no_grad():
            reconstruction = self.net(image_batch)
            # Pixel-wise squared error across color channels: (1, 1, H, W)
            diff = (image_batch - reconstruction) ** 2
            diff_map = torch.mean(diff, dim=1, keepdim=False).squeeze(0).cpu().numpy()

        # Gaussian smoothing
        smoothed_map = scipy.ndimage.gaussian_filter(diff_map, sigma=4.0)

        # Image-level anomaly score is the maximum or 95th percentile reconstruction error
        anomaly_score = float(np.max(smoothed_map))

        # Ensure spatial shape matches original image
        if smoothed_map.shape != (orig_h, orig_w):
            zoom_factors = (orig_h / smoothed_map.shape[0], orig_w / smoothed_map.shape[1])
            smoothed_map = scipy.ndimage.zoom(smoothed_map, zoom_factors, order=1)

        return anomaly_score, smoothed_map.astype(np.float32)

    def save(self, path: str) -> None:
        """Persist model state and hyperparameters to disk.

        Args:
            path: Destination file path.
        """
        state = {
            "model_name": self.name,
            "encoder_channels": self.encoder_channels,
            "net_state_dict": self.net.state_dict(),
            "is_fitted": self._is_fitted,
        }
        save_checkpoint(state, path)

    def load(self, path: str) -> None:
        """Restore model state from checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        state = load_checkpoint(path, device=self.device)
        self.encoder_channels = state.get("encoder_channels", self.encoder_channels)
        self.net = AutoencoderNet(in_channels=3, encoder_channels=self.encoder_channels)
        self.net.load_state_dict(state["net_state_dict"])
        self.net.to(self.device)
        self.net.eval()
        self._is_fitted = state.get("is_fitted", True)
        logger.info("Loaded Autoencoder checkpoint from %s", path)
