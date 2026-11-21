"""PatchCore anomaly detection model with chunked coreset subsampling.

Reference:
    Roth et al., "Towards Total Recall in Industrial Anomaly Detection",
    CVPR 2022.

Builds a memory bank of locally aggregated patch features from normal training
images using a frozen pretrained backbone (WideResNet-50-2 or ResNet-18).
Applies greedy minimax coreset subsampling with chunked computation to reduce
memory usage while preserving coverage. At test time, nearest-neighbor search
with softmax re-weighting scores anomalous regions.
"""

from __future__ import annotations

import logging
from pathlib import Path
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


class PatchCoreModel(AnomalyModel):
    """PatchCore anomaly detection model adhering to AnomalyModel.

    Args:
        config: Configuration namespace (loaded from YAML).
    """

    def __init__(self, config: SimpleNamespace | Any) -> None:
        self.config = config
        pref = getattr(config, "device", "auto")
        self.device = get_device(pref)

        model_cfg = getattr(config, "model", None)
        self.backbone_name = str(getattr(model_cfg, "backbone", "wide_resnet50_2")).lower()
        self.coreset_ratio = float(getattr(model_cfg, "coreset_ratio", 0.01))
        self.num_neighbors = int(getattr(model_cfg, "num_neighbors", 9))
        self.pool_size = int(getattr(model_cfg, "pool_size", 3))
        self.seed = int(getattr(config, "seed", 42))

        # Memory bank (coreset patch vectors)
        self.memory_bank: torch.Tensor | None = None  # (M, D)
        # Global patch indices chosen by coreset selection (selection order).
        self.selected_indices: list[int] | None = None
        self._is_fitted = False

        # Initialize backbone
        self._init_backbone()

    def _init_backbone(self) -> None:
        """Initialize frozen backbone with local-aware average pooling."""
        try:
            if "wide_resnet" in self.backbone_name:
                weights = models.Wide_ResNet50_2_Weights.DEFAULT
                self.backbone = models.wide_resnet50_2(weights=weights)
                logger.info("Loaded pretrained WideResNet-50-2 weights.")
            else:
                weights = models.ResNet18_Weights.DEFAULT
                self.backbone = models.resnet18(weights=weights)
                logger.info("Loaded pretrained ResNet-18 weights.")
        except Exception as exc:
            logger.warning(
                "Could not download pretrained weights for %s (%s). "
                "Falling back to uninitialized weights.",
                self.backbone_name,
                exc,
            )
            if "wide_resnet" in self.backbone_name:
                self.backbone = models.wide_resnet50_2(weights=None)
            else:
                self.backbone = models.resnet18(weights=None)

        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Local-aware patch pooling
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.pool_size,
            stride=1,
            padding=self.pool_size // 2,
        )

        # Hook holders
        self._features: dict[str, torch.Tensor] = {}

        def _hook(name: str):
            def _fn(_m: nn.Module, _inp: Any, out: torch.Tensor):
                self._features[name] = out
            return _fn

        self.backbone.layer2.register_forward_hook(_hook("layer2"))
        self.backbone.layer3.register_forward_hook(_hook("layer3"))

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "PatchCore"

    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract local-aware multi-scale patch features.

        Args:
            x: Input batch tensor (B, 3, H, W).

        Returns:
            Aggregated patch tensor (B, D, H2, W2).
        """
        self._features.clear()
        _ = self.backbone(x)

        f2 = self._features["layer2"]
        f3 = self._features["layer3"]

        # Local pooling
        f2 = self.avg_pool(f2)
        f3 = self.avg_pool(f3)

        # Upsample f3 to f2 spatial resolution
        target_h, target_w = f2.shape[-2], f2.shape[-1]
        f3_up = F.interpolate(f3, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Concatenate patch features: (B, D, H2, W2)
        patch_features = torch.cat([f2, f3_up], dim=1)
        return patch_features

    def _greedy_coreset_subsampling(
        self,
        features: torch.Tensor,
        ratio: float,
        chunk_size: int = 5000,
    ) -> torch.Tensor:
        """Subsample patch memory bank using greedy minimax selection in chunks.

        Args:
            features: Full patch feature tensor of shape ``(N, D)``.
            ratio: Fraction of patches to retain in coreset.
            chunk_size: Candidate chunk size for distance computations.

        Returns:
            Coreset patch tensor of shape ``(M, D)``.
        """
        n_samples, d = features.shape
        target_m = max(1, int(n_samples * ratio))

        if target_m >= n_samples:
            logger.info("Target coreset size (%d) >= total samples (%d). Keeping all.", target_m, n_samples)
            return features

        logger.info(
            "Performing chunked greedy coreset subsampling: %d → %d patches (ratio=%.4f)",
            n_samples,
            target_m,
            ratio,
        )

        # Random projection for efficient coreset selection (Johnson-Lindenstrauss)
        proj_dim = min(128, d)
        torch.manual_seed(self.seed)
        proj_matrix = torch.randn(d, proj_dim) / np.sqrt(proj_dim)
        features_proj = torch.matmul(features, proj_matrix)  # (N, proj_dim)

        selected_indices: list[int] = []

        # Select initial point randomly
        first_idx = int(torch.randint(0, n_samples, (1,)).item())
        selected_indices.append(first_idx)

        # Track minimum distance of all candidate points to current coreset
        min_distances = torch.full((n_samples,), float("inf"), dtype=torch.float32)

        for step in range(1, target_m):
            last_selected = features_proj[selected_indices[-1]].unsqueeze(0)  # (1, proj_dim)

            # Chunked distance update to avoid high memory spikes
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                chunk = features_proj[start:end]  # (chunk_len, proj_dim)
                dists = torch.norm(chunk - last_selected, dim=1)
                min_distances[start:end] = torch.minimum(min_distances[start:end], dists)

            # Pick point farthest from current coreset
            next_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(next_idx)

            if step % max(1, target_m // 5) == 0 or step == target_m - 1:
                logger.info("Coreset subsampling progress: [%d/%d]", step + 1, target_m)

        coreset = features[selected_indices]
        return coreset

    @staticmethod
    def _batch_images(batch: Any) -> torch.Tensor:
        """Extract the image tensor from a batch (dict / tuple / tensor)."""
        if isinstance(batch, dict):
            return batch["image"]
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    def _deterministic_loader(self, dataloader: DataLoader) -> DataLoader:
        """Return a deterministic (shuffle-free) loader over the same dataset.

        Both passes of the two-pass coreset construction must observe exactly
        the same samples in exactly the same order, so any shuffling in the
        provided loader is disabled here.  A fixed order also makes the result
        reproducible for a given seed.
        """
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None:
            return dataloader
        batch_size = getattr(dataloader, "batch_size", None) or len(dataset)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    def _greedy_select_indices(
        self,
        features_proj: torch.Tensor,
        target_m: int,
        chunk_size: int = 5000,
    ) -> list[int]:
        """Greedy minimax coreset selection over projected features.

        This is the selection half of the original
        :meth:`_greedy_coreset_subsampling`, operating on ``(N, proj_dim)``
        projected features.  It returns the selected **global patch indices**
        in selection order.  The random projection and RNG seeding are
        performed by the caller (Pass 1) so that the random stream matches the
        original implementation exactly.
        """
        n_samples = features_proj.shape[0]
        selected_indices: list[int] = [
            int(torch.randint(0, n_samples, (1,)).item())
        ]
        min_distances = torch.full(
            (n_samples,), float("inf"), dtype=torch.float32
        )

        for step in range(1, target_m):
            last_selected = features_proj[selected_indices[-1]].unsqueeze(0)
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                chunk = features_proj[start:end]
                dists = torch.norm(chunk - last_selected, dim=1)
                min_distances[start:end] = torch.minimum(
                    min_distances[start:end], dists
                )
            next_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(next_idx)

            if step % max(1, target_m // 5) == 0 or step == target_m - 1:
                logger.info(
                    "Coreset subsampling progress: [%d/%d]",
                    step + 1,
                    target_m,
                )

        return selected_indices

    def _build_full_memory_bank(
        self, loader: DataLoader
    ) -> torch.Tensor:
        """Build the full ``(N, D)`` memory bank (fallback: keep-all case)."""
        all_patches: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                images = self._batch_images(batch)
                patches = self._extract_patch_features(images)
                b, d, h, w = patches.shape
                all_patches.append(
                    patches.permute(0, 2, 3, 1).contiguous().view(-1, d).cpu()
                )
        full = torch.cat(all_patches, dim=0)
        logger.info(
            "Full memory bank built: %d patches of dim %d",
            full.shape[0],
            full.shape[1],
        )
        return full

    def fit(self, dataloader: DataLoader) -> None:
        """Build coreset memory bank from normal images (two-pass, low memory).

        Pass 1 extracts patch features, applies the seeded random projection,
        and retains only the compact ``(N, proj_dim)`` projected features used
        for coreset selection.  Pass 2 re-extracts features deterministically
        and gathers only the selected full-dimensional vectors.  At no point is
        the full ``(N, D)`` feature bank materialised.

        Args:
            dataloader: DataLoader yielding normal training images.
        """
        self.backbone.eval()
        loader = self._deterministic_loader(dataloader)
        dataset_len = len(getattr(loader, "dataset", []))

        # ---- Pass 1: projected features for coreset selection ----
        logger.info("Extracting patch embeddings for %s...", self.name)
        proj_matrix: torch.Tensor | None = None
        d_total = 0
        proj_dim = 0
        features_proj: torch.Tensor | None = None
        offset = 0
        with torch.no_grad():
            for batch in loader:
                images = self._batch_images(batch).to(self.device)
                patches = self._extract_patch_features(images)  # (B, D, H, W)
                b, d, h, w = patches.shape
                if proj_matrix is None:
                    d_total = d
                    proj_dim = min(128, d_total)
                    # Same seed + projection draw as the original algorithm.
                    torch.manual_seed(self.seed)
                    proj_matrix = torch.randn(d_total, proj_dim) / np.sqrt(
                        proj_dim
                    )
                    patches_per_image = h * w
                    n_samples = max(1, dataset_len * patches_per_image)
                    features_proj = torch.empty(
                        (n_samples, proj_dim), dtype=torch.float32
                    )
                patches_flat = (
                    patches.permute(0, 2, 3, 1).contiguous().view(-1, d).cpu()
                )
                m = patches_flat.shape[0]
                features_proj[offset : offset + m] = torch.matmul(
                    patches_flat, proj_matrix
                )
                offset += m

        if features_proj is None or proj_matrix is None:
            logger.warning("No training samples available; PatchCore fit skipped.")
            return

        n_samples = offset
        features_proj = features_proj[:n_samples]  # (N, proj_dim)

        target_m = max(1, int(n_samples * self.coreset_ratio))

        if target_m >= n_samples:
            # Degenerate: keep all patches.  Build the full bank (rare).
            logger.info(
                "Target coreset size (%d) >= total samples (%d). Keeping all.",
                target_m,
                n_samples,
            )
            self.memory_bank = self._build_full_memory_bank(loader)
            self.selected_indices = list(range(self.memory_bank.shape[0]))
            self._is_fitted = True
            return

        # ---- Greedy coreset selection on projected features ----
        logger.info(
            "Performing chunked greedy coreset subsampling: %d → %d patches (ratio=%.4f)",
            n_samples,
            target_m,
            self.coreset_ratio,
        )
        selected_indices = self._greedy_select_indices(features_proj, target_m)
        del features_proj  # free projected features before Pass 2
        self.selected_indices = selected_indices

        # ---- Pass 2: gather selected full-dimensional vectors ----
        m_selected = len(selected_indices)
        coreset = torch.empty((m_selected, d_total), dtype=torch.float32)
        sel_tensor = torch.tensor(selected_indices, dtype=torch.long)
        # Map global patch index -> position in coreset (selection order).
        pos = torch.full((n_samples,), -1, dtype=torch.long)
        pos[sel_tensor] = torch.arange(m_selected)

        offset = 0
        with torch.no_grad():
            for batch in loader:
                images = self._batch_images(batch).to(self.device)
                patches = self._extract_patch_features(images)  # (B, D, H, W)
                b, d, h, w = patches.shape
                patches_flat = (
                    patches.permute(0, 2, 3, 1).contiguous().view(-1, d).cpu()
                )
                g0 = offset
                g1 = offset + patches_flat.shape[0]
                pos_slice = pos[g0:g1]
                mask = pos_slice >= 0
                if bool(mask.any()):
                    coreset[pos_slice[mask]] = patches_flat[mask]
                offset = g1

        self.memory_bank = coreset
        self._is_fitted = True
        logger.info(
            "PatchCore fitted successfully with %d coreset patches.",
            self.memory_bank.shape[0],
        )

    def predict(self, image: torch.Tensor) -> tuple[float, np.ndarray]:
        """Compute anomaly score and anomaly map for a single image.

        Args:
            image: Preprocessed image tensor of shape ``(C, H, W)`` or ``(1, C, H, W)``.

        Returns:
            Tuple of (anomaly_score, anomaly_map) where anomaly_map has shape ``(H, W)``.
        """
        if not self._is_fitted or self.memory_bank is None:
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
            patch_feats = self._extract_patch_features(image_batch)  # (1, D, H_feat, W_feat)
            _, d, h_feat, w_feat = patch_feats.shape

            # Reshape query patches: (N_query, D)
            query_patches = patch_feats.permute(0, 2, 3, 1).contiguous().view(-1, d).cpu()
            n_queries = query_patches.shape[0]

            # Compute pairwise distances to memory bank in chunks
            mem = self.memory_bank  # (M, D)
            min_dists = torch.empty(n_queries, dtype=torch.float32)

            chunk_size = 1000
            for i in range(0, n_queries, chunk_size):
                q_chunk = query_patches[i : i + chunk_size]
                dists_chunk = torch.cdist(q_chunk, mem, p=2.0)  # (chunk_len, M)
                min_dists[i : i + chunk_size] = torch.min(dists_chunk, dim=1).values

            # Reshape patch scores to 2D spatial grid: (1, 1, H_feat, W_feat)
            score_map = min_dists.view(1, 1, h_feat, w_feat)

            # Upsample score map to original image resolution
            score_map_up = F.interpolate(
                score_map,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            raw_anomaly_map = score_map_up.squeeze().numpy()

        # Gaussian smoothing
        smoothed_map = scipy.ndimage.gaussian_filter(raw_anomaly_map, sigma=4.0)

        # Image score with re-weighting
        max_idx = int(torch.argmax(min_dists).item())
        max_patch = query_patches[max_idx].unsqueeze(0)
        dists_to_mem = torch.cdist(max_patch, mem, p=2.0).squeeze(0)  # (M,)
        top_k_dists, _ = torch.topk(dists_to_mem, k=min(self.num_neighbors, len(dists_to_mem)), largest=False)

        # Softmax re-weighting (Roth et al. 2022)
        weights = F.softmax(top_k_dists, dim=0)
        reweighted_score = float((weights[0] * top_k_dists[0]).item())

        anomaly_score = float(np.max(smoothed_map))

        return anomaly_score, smoothed_map.astype(np.float32)

    def save(self, path: str) -> None:
        """Persist memory bank and configuration to disk.

        Args:
            path: Destination checkpoint file path.
        """
        state = {
            "model_name": self.name,
            "backbone": self.backbone_name,
            "coreset_ratio": self.coreset_ratio,
            "num_neighbors": self.num_neighbors,
            "pool_size": self.pool_size,
            "seed": self.seed,
            "memory_bank": self.memory_bank,
            "is_fitted": self._is_fitted,
        }
        save_checkpoint(state, path)

    def load(self, path: str) -> None:
        """Restore memory bank from checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        state = load_checkpoint(path, device="cpu")
        self.coreset_ratio = state.get("coreset_ratio", self.coreset_ratio)
        self.num_neighbors = state.get("num_neighbors", self.num_neighbors)
        self.pool_size = state.get("pool_size", self.pool_size)
        self.seed = state.get("seed", self.seed)
        self.memory_bank = state.get("memory_bank")
        self._is_fitted = state.get("is_fitted", True)
        logger.info("Loaded PatchCore checkpoint from %s with %d patches", path, len(self.memory_bank) if self.memory_bank is not None else 0)
