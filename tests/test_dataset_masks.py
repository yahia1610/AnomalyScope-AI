"""Regression tests for MVTecDataset mask matching."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from anomalyscope.data.dataset import MVTecDataset


def _make_image(path: Path, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=(10, 20, 30)).save(path)


def _make_mask(path: Path, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (size, size), color=0).save(path)


class TestMaskMatching:
    def test_standard_naming(self, tmp_path: Path) -> None:
        root = tmp_path / "mvtec"
        cat = root / "bottle"
        _make_image(cat / "test" / "broken" / "000.png")
        _make_mask(cat / "ground_truth" / "broken" / "000_mask.png")
        _make_image(cat / "train" / "good" / "000.png")

        ds = MVTecDataset(root=str(root), category="bottle", split="test")
        assert ds._samples[0][1] is not None

    def test_leading_index_fallback(self, tmp_path: Path) -> None:
        # Non-standard names: image '000-94.png', mask '000_mask-67.png'.
        root = tmp_path / "mvtec"
        cat = root / "bottle"
        _make_image(cat / "test" / "broken" / "000-94.png")
        _make_mask(cat / "ground_truth" / "broken" / "000_mask-67.png")
        _make_image(cat / "train" / "good" / "000.png")

        ds = MVTecDataset(root=str(root), category="bottle", split="test")
        assert ds._samples[0][1] is not None
        assert ds._samples[0][1].name == "000_mask-67.png"

    def test_missing_mask_is_none(self, tmp_path: Path) -> None:
        root = tmp_path / "mvtec"
        cat = root / "bottle"
        _make_image(cat / "test" / "broken" / "005.png")
        _make_image(cat / "train" / "good" / "000.png")

        ds = MVTecDataset(root=str(root), category="bottle", split="test")
        assert ds._samples[0][1] is None

    def test_good_has_no_mask(self, tmp_path: Path) -> None:
        root = tmp_path / "mvtec"
        cat = root / "bottle"
        _make_image(cat / "test" / "good" / "000.png")
        _make_image(cat / "train" / "good" / "000.png")

        ds = MVTecDataset(root=str(root), category="bottle", split="test")
        assert ds._samples[0][2] == 0
        assert ds._samples[0][1] is None