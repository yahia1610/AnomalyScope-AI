"""Focused tests for threshold calibration utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anomalyscope.evaluation.calibration import (
    apply_threshold,
    calibrate_threshold,
    compute_false_positive_rate,
    evaluate_threshold,
)


# ---------------------------------------------------------------------------
# calibrate_threshold
# ---------------------------------------------------------------------------


class TestCalibrateThreshold:
    def test_basic_calibration(self) -> None:
        scores = list(np.random.RandomState(42).rand(1000) * 10)
        thresh = calibrate_threshold(scores, percentile=99.5)
        assert isinstance(thresh, float)
        # 99.5th percentile of uniform [0, 10] should be near 10.
        assert 9.0 < thresh <= 10.0

    def test_percentile_zero(self) -> None:
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        thresh = calibrate_threshold(scores, percentile=0.0)
        assert thresh == 1.0

    def test_percentile_hundred(self) -> None:
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        thresh = calibrate_threshold(scores, percentile=100.0)
        assert thresh == 5.0

    def test_numpy_array_input(self) -> None:
        scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        thresh = calibrate_threshold(scores, percentile=90.0)
        assert isinstance(thresh, float)

    def test_tensor_input(self) -> None:
        scores = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7])
        thresh = calibrate_threshold(scores, percentile=90.0)
        assert isinstance(thresh, float)

    def test_empty_array_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            calibrate_threshold([], percentile=99.5)

    def test_empty_numpy_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            calibrate_threshold(np.array([]), percentile=99.5)

    def test_nan_values_sanitised(self) -> None:
        scores = [0.1, np.nan, 0.5, np.nan, 0.9]
        thresh = calibrate_threshold(scores, percentile=90.0)
        assert isinstance(thresh, float)
        assert not np.isnan(thresh)

    def test_inf_values_sanitised(self) -> None:
        scores = [0.1, np.inf, 0.5, -np.inf, 0.9]
        thresh = calibrate_threshold(scores, percentile=90.0)
        assert isinstance(thresh, float)
        assert not np.isnan(thresh)

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            calibrate_threshold([np.nan, np.nan], percentile=99.5)

    def test_invalid_percentile_negative(self) -> None:
        with pytest.raises(ValueError, match="percentile must be in"):
            calibrate_threshold([0.1, 0.2], percentile=-1.0)

    def test_invalid_percentile_over_100(self) -> None:
        with pytest.raises(ValueError, match="percentile must be in"):
            calibrate_threshold([0.1, 0.2], percentile=101.0)

    def test_scalar_score_input(self) -> None:
        thresh = calibrate_threshold(5.0, percentile=50.0)
        assert thresh == 5.0


# ---------------------------------------------------------------------------
# apply_threshold
# ---------------------------------------------------------------------------


class TestApplyThreshold:
    def test_scalar_input(self) -> None:
        result = apply_threshold(0.8, threshold=0.5)
        assert isinstance(result, (bool, np.bool_))
        assert bool(result) is True

    def test_scalar_below_threshold(self) -> None:
        result = apply_threshold(0.3, threshold=0.5)
        assert bool(result) is False

    def test_scalar_equal_to_threshold(self) -> None:
        result = apply_threshold(0.5, threshold=0.5)
        assert bool(result) is False

    def test_list_input(self) -> None:
        scores = [0.1, 0.5, 0.9]
        result = apply_threshold(scores, threshold=0.5)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, [False, False, True])

    def test_numpy_array_input(self) -> None:
        scores = np.array([0.1, 0.5, 0.9])
        result = apply_threshold(scores, threshold=0.5)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [False, False, True])

    def test_tensor_input(self) -> None:
        scores = torch.tensor([0.1, 0.5, 0.9])
        result = apply_threshold(scores, threshold=0.5)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [False, False, True])

    def test_nan_scores_sanitised(self) -> None:
        scores = [0.1, np.nan, 0.9]
        result = apply_threshold(scores, threshold=0.5)
        # NaN sanitised to 0.0, which is < 0.5 → False.
        np.testing.assert_array_equal(result, [False, False, True])


# ---------------------------------------------------------------------------
# compute_false_positive_rate
# ---------------------------------------------------------------------------


class TestFalsePositiveRate:
    def test_zero_fpr(self) -> None:
        # All scores below threshold → 0% FPR.
        scores = [0.1, 0.2, 0.3, 0.4]
        fpr = compute_false_positive_rate(scores, threshold=0.5)
        assert fpr == 0.0

    def test_full_fpr(self) -> None:
        # All scores above threshold → 100% FPR.
        scores = [0.6, 0.7, 0.8, 0.9]
        fpr = compute_false_positive_rate(scores, threshold=0.5)
        assert fpr == 1.0

    def test_partial_fpr(self) -> None:
        scores = [0.1, 0.3, 0.6, 0.9]
        fpr = compute_false_positive_rate(scores, threshold=0.5)
        assert fpr == 0.5

    def test_empty_returns_zero(self) -> None:
        fpr = compute_false_positive_rate([], threshold=0.5)
        assert fpr == 0.0

    def test_nan_scores_handled(self) -> None:
        scores = [0.1, np.nan, 0.9]
        fpr = compute_false_positive_rate(scores, threshold=0.5)
        assert 0.0 <= fpr <= 1.0

    def test_matches_calibration_percentile(self) -> None:
        rng = np.random.RandomState(42)
        scores = list(rng.rand(1000) * 10)
        thresh = calibrate_threshold(scores, percentile=95.0)
        fpr = compute_false_positive_rate(scores, threshold=thresh)
        # FPR should be approximately 5% (100 - 95).
        assert 0.03 < fpr < 0.07


# ---------------------------------------------------------------------------
# evaluate_threshold
# ---------------------------------------------------------------------------


class TestEvaluateThreshold:
    def test_perfect_separation(self) -> None:
        normals = [0.1, 0.2, 0.3, 0.4]
        abnormals = [0.8, 0.9, 1.0, 1.1]
        result = evaluate_threshold(normals, abnormals, threshold=0.5)
        assert result["fpr"] == 0.0
        assert result["tpr"] == 1.0
        assert result["precision"] == 1.0
        assert result["f1"] == 1.0
        assert result["accuracy"] == 1.0

    def test_partial_overlap(self) -> None:
        normals = [0.1, 0.4, 0.6]
        abnormals = [0.3, 0.5, 0.9]
        result = evaluate_threshold(normals, abnormals, threshold=0.5)
        assert 0.0 <= result["fpr"] <= 1.0
        assert 0.0 <= result["tpr"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_empty_both_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            evaluate_threshold([], [], threshold=0.5)

    def test_empty_normal_accepted(self) -> None:
        result = evaluate_threshold([], [0.9, 1.0], threshold=0.5)
        assert result["tpr"] == 1.0
        assert result["fpr"] == 0.0

    def test_return_keys(self) -> None:
        result = evaluate_threshold([0.1, 0.2], [0.8, 0.9], threshold=0.5)
        expected_keys = {"fpr", "fnr", "tpr", "precision", "f1", "accuracy"}
        assert set(result.keys()) == expected_keys

    def test_all_values_are_floats(self) -> None:
        result = evaluate_threshold([0.1], [0.9], threshold=0.5)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float: {type(val)}"