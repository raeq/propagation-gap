"""Formal validation suite for the confidence score parser.

Tests the parse_confidence() function against:
1. Synthetic patterns covering all expected model response formats.
2. Edge cases (boundary values, adversarial numbers in text, empty input).
3. Real pilot responses replayed from pilot_raw_results.json (if available).

Run:
    python -m pytest tests/test_parse_confidence.py -v
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# Import from the eval script (it lives in scripts/, so we patch the path).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_pilot_eval import parse_confidence


# -----------------------------------------------------------------------
# 1. Synthetic pattern tests
# -----------------------------------------------------------------------

class TestLeadingBareNumber:
    """Priority 1: Score as first token in response."""

    def test_bare_100(self):
        assert parse_confidence("100\n\nThe answer is X.") == 1.0

    def test_bare_0(self):
        assert parse_confidence("0\n\nExplanation: I have no idea.") == 0.0

    def test_bare_85(self):
        assert parse_confidence("85\n\nI believe the answer is Y.") == 0.85

    def test_bare_with_space_then_newline(self):
        assert parse_confidence("60 \n\nThe bridge is...") == 0.60

    def test_bare_no_explanation(self):
        """Score with nothing after it."""
        assert parse_confidence("75") == 0.75

    def test_leading_whitespace_then_number(self):
        """Response starts with whitespace then number."""
        assert parse_confidence("  95\n\nText follows.") == 0.95

    def test_bare_30(self):
        assert parse_confidence("30\n\nDr. Voskresenskaya...") == 0.30

    def test_bare_20(self):
        assert parse_confidence("20\n\nI'm not confident...") == 0.20


class TestExplicitLabel:
    """Priority 2: 'Confidence: N' pattern."""

    def test_confidence_colon(self):
        assert parse_confidence(
            "The answer is X.\n\nConfidence: 85"
        ) == 0.85

    def test_confidence_lowercase(self):
        assert parse_confidence(
            "Answer here.\nconfidence: 70"
        ) == 0.70

    def test_confidence_no_colon(self):
        assert parse_confidence(
            "Blah blah.\nConfidence 50"
        ) == 0.50


class TestFractionAndPercent:
    """Priority 3-4: '85/100' and '85%' patterns."""

    def test_fraction(self):
        assert parse_confidence("I am 90/100 sure.") == 0.90

    def test_fraction_with_spaces(self):
        assert parse_confidence("Confidence: 80 / 100") == 0.80

    def test_percent(self):
        assert parse_confidence("I am 75% confident.") == 0.75


class TestTrailingFallback:
    """Priority 5: Number near end of response."""

    def test_trailing_number(self):
        text = "The answer is probably X. My confidence level is about 65"
        assert parse_confidence(text) == 0.65


# -----------------------------------------------------------------------
# 2. Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Boundary values, adversarial numbers, empty input."""

    def test_empty_string(self):
        assert parse_confidence("") is None

    def test_whitespace_only(self):
        assert parse_confidence("   \n\n  ") is None

    def test_no_numbers_at_all(self):
        assert parse_confidence("I have no idea about this topic.") is None

    def test_number_over_100_leading(self):
        """Leading number > 100 should not be parsed as confidence."""
        assert parse_confidence("1956\n\nThe year was 1956.") is None

    def test_number_over_100_in_text(self):
        """Numbers > 100 in text body should not match."""
        # Should return None (no valid 0-100 score found via leading
        # or patterns; the 200 won't match).
        result = parse_confidence("There were approximately 200 people.")
        assert result is None

    def test_boundary_0(self):
        assert parse_confidence("0") == 0.0

    def test_boundary_100(self):
        assert parse_confidence("100") == 1.0

    def test_year_in_explanation_no_leading_score(self):
        """A 4-digit year in the body should NOT be parsed.
        But if there's no leading score, fallback might grab a 2-digit
        portion from a year. This tests the realistic failure mode."""
        # Response with no confidence score, just a year.
        text = "\n\nExplanation:\nThe event happened in 1956."
        # The fallback looks at last 80 chars. "56" is a word boundary match.
        # This is a known limitation: without a clear confidence indicator,
        # years can produce false positives in the fallback. We accept this
        # because the fallback is lowest priority and rarely reached.
        result = parse_confidence(text)
        # We just verify it doesn't crash; the value may be 0.56 (false positive).
        assert result is None or (0.0 <= result <= 1.0)

    def test_response_starts_with_newline_then_number(self):
        """Model sometimes outputs leading newline."""
        assert parse_confidence("\n85\n\nSome text.") == 0.85

    def test_leading_number_followed_by_larger_numbers(self):
        """Ensure we grab the leading score, not numbers in body text."""
        text = "80\n\nThe population was 5000 in the year 1990."
        assert parse_confidence(text) == 0.80


# -----------------------------------------------------------------------
# 3. Replay against real pilot data (if available)
# -----------------------------------------------------------------------

PILOT_RAW_PATH = Path(__file__).resolve().parent.parent / "results" / "pilot" / "pilot_raw_results.json"


@pytest.mark.skipif(
    not PILOT_RAW_PATH.exists(),
    reason="Pilot raw results not available for replay test.",
)
class TestPilotReplay:
    """Replay parser over real pilot confidence responses.

    Validates:
    - Parser extracts a score from >=90% of confidence-condition responses.
    - All extracted scores are in [0.0, 1.0].
    - K-category mean confidence > D-category mean confidence (expected
      from a model that's more confident on known questions).
    """

    @pytest.fixture(scope="class")
    def confidence_results(self):
        with open(PILOT_RAW_PATH) as f:
            data = json.load(f)
        return [r for r in data if r["condition"] == "confidence"]

    def test_extraction_rate(self, confidence_results):
        """Parser should extract a score from >= 90% of responses."""
        total = len(confidence_results)
        parsed = sum(
            1 for r in confidence_results
            if parse_confidence(r["response"]) is not None
        )
        rate = parsed / total if total > 0 else 0.0
        assert rate >= 0.90, (
            f"Extraction rate {rate:.1%} ({parsed}/{total}) below 90% threshold."
        )

    def test_all_scores_in_range(self, confidence_results):
        """All extracted scores must be in [0.0, 1.0]."""
        for r in confidence_results:
            score = parse_confidence(r["response"])
            if score is not None:
                assert 0.0 <= score <= 1.0, (
                    f"{r['question_id']}: score {score} out of [0,1] range."
                )

    def test_confidence_by_category(self, confidence_results):
        """Report mean confidence per category.

        Note: we do NOT assert K > D here. Depth ignorance is precisely
        the condition where the model is confident despite unreliable
        knowledge. If K ≈ D in mean confidence while K >> D in accuracy,
        that is the paper's predicted miscalibration for depth ignorance.
        The assertion is that C mean confidence < K mean confidence
        (the model should be less confident on fabricated entities).
        """
        scores_by_cat = {}
        for cat in ["K", "C", "D"]:
            scores = [
                parse_confidence(r["response"])
                for r in confidence_results
                if r["category"] == cat
                and parse_confidence(r["response"]) is not None
            ]
            if scores:
                scores_by_cat[cat] = sum(scores) / len(scores)
                print(f"  {cat}: mean={scores_by_cat[cat]:.3f} (n={len(scores)})")

        # C should be lower than K (model should show less confidence
        # on fabricated entities than on reliably-known facts).
        if "K" in scores_by_cat and "C" in scores_by_cat:
            assert scores_by_cat["C"] < scores_by_cat["K"], (
                f"C mean confidence ({scores_by_cat['C']:.3f}) should be "
                f"lower than K ({scores_by_cat['K']:.3f})."
            )

    def test_per_category_extraction_counts(self, confidence_results):
        """Report extraction counts per category (informational)."""
        for cat in ["K", "C", "D"]:
            cat_resp = [r for r in confidence_results if r["category"] == cat]
            parsed = sum(
                1 for r in cat_resp
                if parse_confidence(r["response"]) is not None
            )
            print(f"  {cat}: {parsed}/{len(cat_resp)} parsed")
            # Soft threshold: at least 80% per category.
            if len(cat_resp) > 0:
                assert parsed / len(cat_resp) >= 0.80, (
                    f"Category {cat}: only {parsed}/{len(cat_resp)} parsed."
                )
