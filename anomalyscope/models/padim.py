"""PaDiM (Patch Distribution Modeling) anomaly detection model.

Reference:
    Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
    Anomaly Detection and Localization", ICPR 2021.

Extracts multi-scale patch embeddings from normal training images using a
frozen pretrained ResNet-18 backbone, applies random dimensionality reduction,
and models patch distributions as multivariate Gaussians (mean and covariance)
for each spatial location. Inference computes the Mahalanobis distance.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader

from anomalyscope.models.base import AnomalyModel
from anomalyscope.utils.device import get_device
from anomalyscope.utils.io import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class PaDiMModel(AnomalyModel):
    """PaDiM anomaly detection model.

    Args:
        config: Configuration namespace (loaded from YAML).
    """

    def __init__(self, config: SimpleNamespace | Any) -> None:
        self.config = config
        pref = getattr(config, "device", "auto")
        self.device = get_device(pref)

        model_cfg = getattr(config, "model", None)
        self.num_features = int(getattr(model_cfg, "num_features", 100))
        self.epsilon = float(getattr(model_cfg, "epsilon", 0.01))
        self.backbone_name = str(getattr(model_cfg, "backbone", "resnet18")).lower()
        self.seed = int(getattr(config, "seed", 42))

        # Build feature extractor
        self._init_backbone()

        # Learned distribution parameters
        self.mean: torch.Tensor | None = None  # (D, H, W)
        self.inv_cov: torch.Tensor | None = None  # (H, W, D, D)
        self.selected_indices: torch.Tensor | None = None
        self._is_fitted = False

    def _init_backbone(self) -> None:
        """Initialize frozen pretrained ResNet-18 backbone."""
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
            logger.info("Loaded pretrained ResNet-18 weights.")
        except Exception as exc:
            logger.warning(
                "Could not download pretrained ResNet-18 weights (%s). "
                "Falling back to uninitialized weights.",
                exc,
            )
            self.backbone = models.resnet18(weights=None)

        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Hook holders
        self._features: dict[str, torch.Tensor] = {}

        def _hook(name: str):
            def _fn(_m: nn.Module, _inp: Any, out: torch.Tensor):
                self._features[name] = out
            return _fn

        self.backbone.layer1.register_forward_hook(_hook("layer1"))
        self.backbone.layer2.register_forward_hook(_hook("layer2"))
        self.backbone.layer3.register_forward_hook(_hook("layer3"))

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "PaDiM"

    def _extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale patch embeddings and concatenate along channel dimension.

        Args:
            x: Input batch tensor (B, 3, H, W).

        Returns:
            Concatenated embedding tensor (B, C_total, H1, W1).
        """
        self._features.clear()
        _ = self.backbone(x)

        f1 = self._features["layer1"]  # (B, 64, H/4, W/4)
        f2 = self._features["layer2"]  # (B, 128, H/8, W/8)
        f3 = self._features["layer3"]  # (B, 256, H/16, W/16)

        target_h, target_w = f1.shape[-2], f1.shape[-1]

        # Upsample f2 and f3 to layer1 spatial dimensions
        f2_up = F.interpolate(f2, size=(target_h, target_w), mode="bilinear", align_corners=False)
        f3_up = F.interpolate(f3, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Concatenate: (B, 448, H1, W1)
        embeddings = torch.cat([f1, f2_up, f3_up], dim=1)
        return embeddings

    def fit(self, dataloader: DataLoader) -> None:
        """Learn per-position Gaussian distribution from normal images.

        Args:
            dataloader: DataLoader yielding batches of normal images.
        """
        self.backbone.eval()
        reduced_batches: list[torch.Tensor] = []
        selected_indices: torch.Tensor | None = None
        h_feat = w_feat = 0
        n_samples = 0

        logger.info("Extracting features for %s from training set...", self.name)
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    images = batch["image"]
                elif isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                emb = self._extract_embeddings(images)  # (B, C_total, H, W)

                # Select the deterministic channel subset once, using the same
                # seed and logic as before.  Feature extraction is deterministic
                # (eval mode, no dropout), so re-seeding here yields exactly the
                # same indices as selecting after concatenation.
                if selected_indices is None:
                    c_total = emb.shape[1]
                    h_feat, w_feat = emb.shape[-2], emb.shape[-1]
                    torch.manual_seed(self.seed)
                    n_features = min(self.num_features, c_total)
                    selected_indices = torch.randperm(c_total)[:n_features]
                    self.selected_indices = selected_indices

                # Reduce channels per batch BEFORE concatenation.  Selecting the
                # same column indices before vs. after torch.cat yields an
                # identical reduced tensor, but keeps peak memory at
                # (N, D, H, W) instead of (N, C_total, H, W).
                reduced_batches.append(emb[:, selected_indices, :, :].cpu())
                n_samples += emb.shape[0]

        if not reduced_batches or selected_indices is None:
            logger.warning("No training samples available; PaDiM fit skipped.")
            return

        # Concatenate reduced batches: (N, D, H, W)
        reduced = torch.cat(reduced_batches, dim=0)
        del reduced_batches  # free the per-batch list before covariance
        n_samples, d, h_feat, w_feat = reduced.shape
        logger.info(
            "Extracted %d patch embedding maps reduced to %d features "
            "of shape (%d, %d, %d)",
            n_samples,
            d,
            d,
            h_feat,
            w_feat,
        )

        logger.info(
            "Fitting multivariate Gaussians across %d spatial locations...", h_feat * w_feat
        )

        # Mean calculation per position: (D, H, W)
        self.mean = torch.mean(reduced, dim=0)  # (D, H, W)

        # Numerically stable per-position covariance calculation
        # Centered features: (N, D, H, W)
        centered = reduced - self.mean.unsqueeze(0)
        del reduced  # no longer needed; free before large permute/copy

        # Reshape to (H*W, D, N) for batched matrix multiplication
        # First permute to (H, W, D, N)
        centered_perm = centered.permute(2, 3, 1, 0).contiguous()
        del centered  # contiguous copy holds the data now
        p = h_feat * w_feat
        centered_flat = centered_perm.view(p, d, n_samples)  # (P, D, N)

        # Covariance: (P, D, D) = (centered_flat @ centered_flat.T) / (N - 1) + epsilon * I
        divisor = max(1, n_samples - 1)
        cov = torch.bmm(centered_flat, centered_flat.transpose(1, 2)) / divisor

        # Regularization: add epsilon * I
        eye = torch.eye(d).unsqueeze(0).expand(p, -1, -1)
        cov = cov + self.epsilon * eye

        # Invert covariance matrices using pseudo-inverse for guaranteed numerical stability
        inv_cov_flat = torch.linalg.pinv(cov)
        self.inv_cov = inv_cov_flat.view(h_feat, w_feat, d, d)

        self._is_fitted = True
        logger.info("PaDiM distribution fitting complete.")

    def predict(self, image: torch.Tensor) -> tuple[float, np.ndarray]:
        """Compute the anomaly score and anomaly map for a single image.

        Args:
            image: Preprocessed image tensor of shape ``(C, H, W)`` or ``(1, C, H, W)``.

        Returns:
            Tuple of (anomaly_score, anomaly_map) where anomaly_map has shape ``(H, W)``.
        """
        if (
            not self._is_fitted
            or self.mean is None
            or self.inv_cov is None
            or self.selected_indices is None
        ):
            raise RuntimeError("Model must be fitted before predict() is called.")

        self.backbone.eval()

        if image.ndim == 3:
            image_batch = image.unsqueeze(0)
        elif image.ndim == 4:
            image_batch = image
        else:
            raise ValueError(f"Expected 3D or 4D image tensor, got shape {image.shape}")

        orig_h, orig_w = image_batch.shape[-2], image_batch.shape[-1]
        image_batch = image_batch.to(self.device)

        with torch.no_grad():
            emb = self._extract_embeddings(image_batch)
            emb = emb[:, self.selected_indices, :, :].cpu()  # (1, D, H, W)

            mean = self.mean  # (D, H, W)
            inv_cov = self.inv_cov  # (H, W, D, D)

            # Center test embedding: (D, H, W)
            delta = (emb.squeeze(0) - mean).permute(1, 2, 0).unsqueeze(-1)  # (H, W, D, 1)

            # Mahalanobis distance squared: delta^T @ inv_cov @ delta -> (H, W, 1, 1)
            temp = torch.matmul(inv_cov, delta)  # (H, W, D, 1)
            dist_sq = torch.matmul(delta.transpose(-2, -1), temp).squeeze()  # (H, W)
            dist_map = (
                torch.sqrt(torch.clamp(dist_sq, min=0.0)).unsqueeze(0).unsqueeze(0)
            )  # (1, 1, H, W)

            # Upsample to original image resolution
            dist_map_up = F.interpolate(
                dist_map,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            raw_anomaly_map = dist_map_up.squeeze().numpy()

        # Gaussian smoothing
        smoothed_map = scipy.ndimage.gaussian_filter(raw_anomaly_map, sigma=4.0)

        # Image-level anomaly score is the maximum patch distance
        anomaly_score = float(np.max(smoothed_map))

        return anomaly_score, smoothed_map.astype(np.float32)

    def save(self, path: str) -> None:
        """Persist learned distribution parameters to disk.

        Args:
            path: Destination file path.
        """
        state = {
            "model_name": self.name,
            "backbone": self.backbone_name,
            "num_features": self.num_features,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "selected_indices": self.selected_indices,
            "mean": self.mean,
            "inv_cov": self.inv_cov,
            "is_fitted": self._is_fitted,
        }
        save_checkpoint(state, path)

    def load(self, path: str) -> None:
        """Restore learned distribution parameters from disk.

        Args:
            path: Path to checkpoint file.
        """
        state = load_checkpoint(path, device="cpu")
        self.num_features = state.get("num_features", self.num_features)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.seed = state.get("seed", self.seed)
        self.selected_indices = state.get("selected_indices")
        self.mean = state.get("mean")
        self.inv_cov = state.get("inv_cov")
        self._is_fitted = state.get("is_fitted", True)
        logger.info("Loaded PaDiM checkpoint from %s", path)
