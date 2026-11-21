"""Focused tests for heatmap generation and visualisation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anomalyscope.evaluation.heatmap import (
    apply_colormap,
    create_comparison_panel,
    draw_contours,
    normalize_heatmap,
    overlay_heatmap,
)


# ---------------------------------------------------------------------------
# normalize_heatmap
# ---------------------------------------------------------------------------


class TestNormalizeHeatmap:
    def test_basic_normalization(self) -> None:
        raw = np.array([[0.0, 5.0], [10.0, 15.0]])
        out = normalize_heatmap(raw)
        assert out.dtype == np.float32
        assert out.min() == 0.0
        assert out.max() == 1.0
        assert out.shape == (2, 2)

    def test_constant_map_returns_zeros(self) -> None:
        raw = np.full((4, 4), 3.14)
        out = normalize_heatmap(raw)
        assert np.all(out == 0.0)

    def test_negative_values(self) -> None:
        raw = np.array([[-5.0, 0.0], [5.0, 10.0]])
        out = normalize_heatmap(raw)
        assert out.min() == 0.0
        assert out.max() == 1.0

    def test_nan_values_handled(self) -> None:
        raw = np.array([[0.0, np.nan], [10.0, 15.0]])
        out = normalize_heatmap(raw)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_inf_values_handled(self) -> None:
        raw = np.array([[0.0, np.inf], [10.0, -np.inf]])
        out = normalize_heatmap(raw)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_tensor_input(self) -> None:
        raw = torch.tensor([[0.0, 5.0], [10.0, 15.0]])
        out = normalize_heatmap(raw)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 2)

    def test_3d_tensor_squeezed(self) -> None:
        raw = torch.randn(1, 8, 8)
        out = normalize_heatmap(raw)
        assert out.shape == (8, 8)

    def test_output_shape_preserved(self) -> None:
        raw = np.random.rand(16, 16).astype(np.float32)
        out = normalize_heatmap(raw)
        assert out.shape == (16, 16)


# ---------------------------------------------------------------------------
# apply_colormap
# ---------------------------------------------------------------------------


class TestApplyColormap:
    def test_output_shape_and_dtype(self) -> None:
        hm = np.random.rand(8, 8).astype(np.float32)
        out = apply_colormap(hm)
        assert out.dtype == np.uint8
        assert out.shape == (8, 8, 3)

    def test_values_clipped(self) -> None:
        hm = np.array([[-1.0, 0.5], [2.0, 1.5]], dtype=np.float32)
        out = apply_colormap(hm)
        assert out.dtype == np.uint8
        assert out.shape == (2, 2, 3)

    def test_constant_zero(self) -> None:
        hm = np.zeros((4, 4), dtype=np.float32)
        out = apply_colormap(hm)
        assert out.shape == (4, 4, 3)


# ---------------------------------------------------------------------------
# overlay_heatmap
# ---------------------------------------------------------------------------


class TestOverlayHeatmap:
    def test_basic_overlay(self) -> None:
        image = np.random.rand(16, 16, 3).astype(np.float32)
        amap = np.random.rand(16, 16).astype(np.float32)
        out = overlay_heatmap(image, amap, alpha=0.5)
        assert out.shape == (16, 16, 3)
        assert out.dtype == np.uint8

    def test_alpha_zero_preserves_image(self) -> None:
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
        amap = np.ones((8, 8), dtype=np.float32)
        out = overlay_heatmap(image, amap, alpha=0.0)
        # Should be essentially the original image (allow for uint8 conversion).
        assert np.allclose(out, image, atol=1)

    def test_alpha_one_is_heatmap_only(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        amap = np.ones((8, 8), dtype=np.float32)
        out = overlay_heatmap(image, amap, alpha=1.0)
        # Should be the colormap output, not the black original.
        assert not np.all(out == 0)

    def test_size_mismatch_resized(self) -> None:
        image = np.random.rand(32, 32, 3).astype(np.float32)
        amap = np.random.rand(16, 16).astype(np.float32)
        out = overlay_heatmap(image, amap, alpha=0.5)
        assert out.shape == (32, 32, 3)

    def test_tensor_inputs(self) -> None:
        image = torch.rand(8, 8, 3)
        amap = torch.rand(8, 8)
        out = overlay_heatmap(image, amap)
        assert out.shape == (8, 8, 3)
        assert out.dtype == np.uint8

    def test_grayscale_image(self) -> None:
        image = np.random.rand(8, 8).astype(np.float32)
        amap = np.random.rand(8, 8).astype(np.float32)
        out = overlay_heatmap(image, amap)
        assert out.shape == (8, 8, 3)

    def test_nan_anomaly_map_handled(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        amap = np.full((8, 8), np.nan, dtype=np.float32)
        out = overlay_heatmap(image, amap)
        assert out.shape == (8, 8, 3)
        assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# draw_contours
# ---------------------------------------------------------------------------


class TestDrawContours:
    def test_contours_drawn_on_anomalous_region(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        amap = np.zeros((32, 32), dtype=np.float32)
        amap[10:22, 10:22] = 1.0
        out = draw_contours(image, amap, threshold=0.5, color=(0, 0, 255))
        assert out.shape == (32, 32, 3)
        # Some red pixels should have been drawn.
        assert np.any(out[..., 2] > 0)

    def test_no_contours_on_uniform_map(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        amap = np.ones((16, 16), dtype=np.float32) * 0.1
        out = draw_contours(image, amap, threshold=0.5, color=(0, 0, 255))
        # No red pixels.
        assert not np.any(out[..., 2] > 0)

    def test_output_is_copy(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        amap = np.ones((16, 16), dtype=np.float32)
        out = draw_contours(image, amap)
        # Original should be unchanged.
        assert np.all(image == 0)
        assert out is not image

    def test_size_mismatch_handled(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        amap = np.ones((16, 16), dtype=np.float32)
        out = draw_contours(image, amap, threshold=0.5)
        assert out.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# create_comparison_panel
# ---------------------------------------------------------------------------


class TestComparisonPanel:
    def test_panel_dimensions(self) -> None:
        image = np.random.rand(16, 16, 3).astype(np.float32)
        amap = np.random.rand(16, 16).astype(np.float32)
        panel = create_comparison_panel(image, amap)
        # Three panels side by side: (H, W*3, 3)
        assert panel.shape == (16, 48, 3)
        assert panel.dtype == np.uint8

    def test_panel_with_tensors(self) -> None:
        image = torch.rand(8, 8, 3)
        amap = torch.rand(8, 8)
        panel = create_comparison_panel(image, amap)
        assert panel.shape == (8, 24, 3)

    def test_panel_size_mismatch(self) -> None:
        image = np.random.rand(32, 32, 3).astype(np.float32)
        amap = np.random.rand(16, 16).astype(np.float32)
        panel = create_comparison_panel(image, amap)
        # Image is 32x32, heatmap resized to match.
        assert panel.shape == (32, 96, 3)

    def test_panel_with_nan_heatmap(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        amap = np.full((8, 8), np.nan, dtype=np.float32)
        panel = create_comparison_panel(image, amap)
        assert panel.shape == (8, 24, 3)

    def test_first_panel_matches_original(self) -> None:
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        amap = np.zeros((8, 8), dtype=np.float32)
        panel = create_comparison_panel(image, amap)
        # First 8 columns should be the original image.
        assert np.allclose(panel[:, :8], image, atol=1)