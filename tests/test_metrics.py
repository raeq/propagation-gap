"""Tests for CI-Bench evaluation metrics."""

import numpy as np
import pytest

from ci_bench.eval.metrics import (
    abstention_precision,
    abstention_recall,
    auroc,
    expected_calibration_error,
    reasoning_penalty,
)
from ci_bench.eval.bootstrap import bootstrap_ci


class TestECE:
    """Tests for expected_calibration_error."""

    def test_perfect_calibration(self):
        """A perfectly calibrated model has ECE = 0."""
        # 100 items, confidence = accuracy in each bin.
        rng = np.random.default_rng(42)
        confidences = rng.uniform(0, 1, size=200)
        # Make correctness match confidence (probabilistically).
        correctness = rng.random(200) < confidences
        # With enough data and matched probabilities, ECE should be small.
        ece = expected_calibration_error(confidences, correctness, n_bins=10)
        assert ece < 0.15  # Generous bound for stochastic test.

    def test_maximally_miscalibrated(self):
        """A model that is always confident but always wrong has high ECE."""
        confidences = np.ones(100)  # Always says 100% confident.
        correctness = np.zeros(100, dtype=bool)  # Always wrong.
        ece = expected_calibration_error(confidences, correctness)
        assert ece == pytest.approx(1.0)

    def test_empty_after_mask(self):
        """ECE of an empty set after masking is 0."""
        confidences = np.array([0.5, 0.6])
        correctness = np.array([True, False])
        mask = np.array([False, False])
        ece = expected_calibration_error(confidences, correctness, category_mask=mask)
        assert ece == 0.0

    def test_category_mask(self):
        """Mask selects a subset and ECE is computed correctly on it."""
        # Two items with confidence 0.9, both correct.
        # Bin [0.8, 0.9]: avg_conf=0.9, avg_acc=1.0, |diff|=0.1.
        confidences = np.array([0.9, 0.9, 0.1, 0.1])
        correctness = np.array([True, True, False, False])
        mask_first_half = np.array([True, True, False, False])
        ece = expected_calibration_error(
            confidences, correctness, category_mask=mask_first_half
        )
        assert ece == pytest.approx(0.1, abs=0.01)


class TestAbstentionPrecision:
    def test_perfect_precision(self):
        """Model abstains only when it should."""
        abstained = np.array([True, False, True, False])
        should = np.array([True, False, True, True])
        assert abstention_precision(abstained, should) == pytest.approx(1.0)

    def test_zero_precision(self):
        """Model abstains only when it shouldn't."""
        abstained = np.array([True, True, False, False])
        should = np.array([False, False, True, True])
        assert abstention_precision(abstained, should) == pytest.approx(0.0)

    def test_no_abstentions(self):
        """Model never abstains: precision is 0."""
        abstained = np.zeros(4, dtype=bool)
        should = np.ones(4, dtype=bool)
        assert abstention_precision(abstained, should) == 0.0

    def test_with_mask(self):
        """Mask filters to a subset."""
        abstained = np.array([True, True, True, True])
        should = np.array([True, False, True, False])
        mask = np.array([True, True, False, False])
        prec = abstention_precision(abstained, should, category_mask=mask)
        assert prec == pytest.approx(0.5)


class TestAbstentionRecall:
    def test_perfect_recall(self):
        """Model abstains on everything it should."""
        abstained = np.array([True, True, True, False])
        should = np.array([True, True, True, False])
        assert abstention_recall(abstained, should) == pytest.approx(1.0)

    def test_zero_recall(self):
        """Model never abstains when it should."""
        abstained = np.zeros(4, dtype=bool)
        should = np.array([True, True, False, False])
        assert abstention_recall(abstained, should) == pytest.approx(0.0)

    def test_nothing_should_abstain(self):
        """No items should be abstained on: recall is 0."""
        abstained = np.ones(4, dtype=bool)
        should = np.zeros(4, dtype=bool)
        assert abstention_recall(abstained, should) == 0.0


class TestAUROC:
    def test_perfect_discrimination(self):
        """Perfect scores yield AUROC = 1."""
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([True, True, False, False])
        assert auroc(scores, labels) == pytest.approx(1.0)

    def test_random_discrimination(self):
        """Random scores yield AUROC near 0.5."""
        rng = np.random.default_rng(42)
        scores = rng.random(1000)
        labels = rng.random(1000) > 0.5
        auc = auroc(scores, labels)
        assert 0.4 < auc < 0.6

    def test_single_class_raises(self):
        """AUROC is undefined with only one class."""
        scores = np.array([0.5, 0.6, 0.7])
        labels = np.array([True, True, True])
        with pytest.raises(ValueError, match="only one class"):
            auroc(scores, labels)

    def test_with_mask(self):
        """Mask works correctly."""
        scores = np.array([0.9, 0.8, 0.2, 0.1, 0.5])
        labels = np.array([True, True, False, False, True])
        mask = np.array([True, True, True, True, False])
        auc = auroc(scores, labels, category_mask=mask)
        assert auc == pytest.approx(1.0)


class TestReasoningPenalty:
    def test_penalty(self):
        """CoT hurts abstention: negative delta."""
        assert reasoning_penalty(0.8, 0.6) == pytest.approx(-0.2)

    def test_improvement(self):
        """CoT helps abstention: positive delta."""
        assert reasoning_penalty(0.5, 0.7) == pytest.approx(0.2)

    def test_no_change(self):
        assert reasoning_penalty(0.5, 0.5) == pytest.approx(0.0)


class TestBootstrapCI:
    def test_returns_three_values(self):
        """bootstrap_ci returns (point, lower, upper)."""
        confidences = np.random.default_rng(42).uniform(0, 1, 100)
        correctness = np.random.default_rng(42).random(100) > 0.5
        point, lo, hi = bootstrap_ci(
            expected_calibration_error,
            confidences, correctness,
            n_resamples=100,
            seed=42,
        )
        assert isinstance(point, float)
        assert lo <= point <= hi or np.isnan(lo)

    def test_ci_contains_point(self):
        """CI should generally contain the point estimate."""
        rng = np.random.default_rng(42)
        scores = np.concatenate([rng.normal(1, 0.5, 50), rng.normal(0, 0.5, 50)])
        labels = np.array([True] * 50 + [False] * 50)
        point, lo, hi = bootstrap_ci(
            auroc, scores, labels,
            n_resamples=500, seed=42,
        )
        assert lo <= point <= hi

    def test_narrow_ci_with_large_sample(self):
        """Large samples should produce narrow CIs."""
        rng = np.random.default_rng(42)
        confidences = np.full(1000, 0.7)
        correctness = rng.random(1000) < 0.7
        point, lo, hi = bootstrap_ci(
            expected_calibration_error,
            confidences, correctness,
            n_resamples=500, seed=42,
        )
        assert (hi - lo) < 0.1
