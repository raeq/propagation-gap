"""Tests for the locked answer checker (ci_bench.eval.scorer).

Validates the authoritative scoring functions that all Phase 5+
screening must use. These tests encode the corrections and fixes
from Sessions 061-063.
"""

import pytest

from ci_bench.eval.scorer import (
    normalize,
    extract_answer,
    check_answer,
    detect_abstention,
    check_correct,
    parse_confidence,
)
from ci_bench.data.schema import Question, Category, SubCategory


class TestNormalize:
    """Test the normalize function."""

    def test_url_decode(self):
        assert normalize("Hello%20World") == "hello world"
        assert normalize("%22quoted%22") == "quoted"

    def test_quote_stripping(self):
        assert normalize('"hello"') == "hello"
        assert normalize("'hello'") == "hello"
        assert normalize("\u201chello\u201d") == "hello"
        assert normalize("`hello`") == "hello"

    def test_hyphen_to_space(self):
        assert normalize("well-known") == "well known"
        assert normalize("self-referential") == "self referential"

    def test_lowercase(self):
        assert normalize("HELLO") == "hello"

    def test_strip_articles(self):
        assert normalize("the cat") == "cat"
        assert normalize("a dog") == "dog"
        assert normalize("an apple") == "apple"

    def test_strip_punctuation(self):
        assert normalize("hello, world!") == "hello world"
        assert normalize("(parentheses)") == "parentheses"

    def test_trailing_period(self):
        assert normalize("answer.") == "answer"

    def test_collapse_whitespace(self):
        assert normalize("  hello   world  ") == "hello world"

    def test_combined(self):
        # URL-encoded quotes + hyphens + articles + punctuation.
        assert normalize('%22The Well-Known Cat%22.') == "well known cat"


class TestExtractAnswer:
    """Test the extract_answer function."""

    def test_explicit_label(self):
        text = "Answer: Paris\n\nExplanation follows."
        assert extract_answer(text) == "Paris"

    def test_confidence_cutoff(self):
        text = "Answer: Paris\nConfidence (0-100): 95"
        assert extract_answer(text) == "Paris"

    def test_correctness_cutoff(self):
        text = "Answer: Paris\nCorrectness confidence: 90"
        assert extract_answer(text) == "Paris"

    def test_no_label_first_line(self):
        text = "Paris is the capital of France.\n\nMore details here."
        assert extract_answer(text) == "Paris is the capital of France."

    def test_multiline_first_paragraph(self):
        text = "Line one\nLine two\n\nSecond paragraph."
        assert extract_answer(text) == "Line one"


class TestCheckAnswer:
    """Test the three-stage answer checker."""

    # Stage 1: bidirectional substring.
    def test_exact_match(self):
        assert check_answer("Paris", ["Paris"]) is True

    def test_substring_ref_in_answer(self):
        assert check_answer("The answer is Paris, France", ["Paris"]) is True

    def test_substring_answer_in_ref(self):
        assert check_answer("Paris", ["Paris, France"]) is True

    def test_no_match(self):
        assert check_answer("London", ["Paris"]) is False

    # Stage 2: n-gram matching.
    def test_embedded_number(self):
        assert check_answer("It was around 1500 when this happened", ["1500"]) is True

    def test_embedded_short_answer(self):
        assert check_answer("The Middle Ages ended around 1500", ["1500"]) is True

    # Stage 2 guard: short n-grams don't match longer refs.
    def test_ngram_guard_single_word_vs_multiword_ref(self):
        """Single-word n-gram 'red' should not match multi-word ref 'Woman in Red'."""
        long_answer = ("The answer involves many things including the color red "
                       "through various historical periods and contexts")
        assert check_answer(long_answer, ["Woman in Red"]) is False

    def test_multiword_ngram_matches_multiword_ref(self):
        """Multi-word n-gram should match multi-word ref."""
        assert check_answer("The Woman in Red", ["Woman in Red"]) is True

    # Stage 3: clause-level matching.
    def test_clause_match(self):
        long = "The thing was discovered by Edward Gibbon, through his historical research into decline"
        assert check_answer(long, ["Edward Gibbon"]) is True

    def test_clause_guard_single_word(self):
        """Single-word clause should not match multi-word ref."""
        long = "This involves red, through various means, in historical context and everything else"
        assert check_answer(long, ["Woman in Red"]) is False

    # K-041 Gibbons case (from Session 063).
    def test_gibbons_match(self):
        assert check_answer("Gibbons", ["Gibbons", "Stella Gibbons", "Edward Gibbon"]) is True

    def test_gibbons_specific(self):
        assert check_answer("Stella Gibbons", ["Gibbons", "Stella Gibbons"]) is True

    def test_flora_no_match_gibbons(self):
        assert check_answer("Flora", ["Gibbons", "Stella Gibbons", "Edward Gibbon"]) is False

    # Empty and edge cases.
    def test_empty_refs(self):
        assert check_answer("anything", []) is False

    def test_empty_answer(self):
        # Empty string is a substring of any string in Python,
        # so this technically "matches". In practice, models never
        # produce empty responses.
        assert check_answer("", ["Paris"]) is True


class TestDetectAbstention:
    """Test abstention detection."""

    def test_i_dont_know(self):
        assert detect_abstention("I don't know the answer") is True

    def test_not_sure(self):
        assert detect_abstention("I'm not sure about this") is True

    def test_cannot_determine(self):
        assert detect_abstention("I cannot determine the answer") is True

    def test_no_abstention(self):
        assert detect_abstention("Paris is the capital of France") is False

    def test_case_insensitive(self):
        assert detect_abstention("I DON'T KNOW") is True


class TestCheckCorrect:
    """Test category-aware correctness checking."""

    def test_k_question_correct(self):
        q = Question(id="K-001", text="What?", category=Category.K,
                     sub_category=SubCategory.K, reference_answers=["Paris"])
        assert check_correct("Paris", q) is True

    def test_k_question_wrong(self):
        q = Question(id="K-001", text="What?", category=Category.K,
                     sub_category=SubCategory.K, reference_answers=["Paris"])
        assert check_correct("London", q) is False

    def test_c1_abstention_correct(self):
        q = Question(id="C1-001", text="What?", category=Category.C,
                     sub_category=SubCategory.C1, reference_answers=[])
        assert check_correct("I don't know", q) is True

    def test_c1_answer_incorrect(self):
        q = Question(id="C1-001", text="What?", category=Category.C,
                     sub_category=SubCategory.C1, reference_answers=[])
        assert check_correct("Some made-up answer", q) is False

    def test_c3_abstention_correct(self):
        q = Question(id="C3-001", text="What?", category=Category.C,
                     sub_category=SubCategory.C3, reference_answers=[])
        assert check_correct("I'm not sure about this", q) is True

    def test_d2_answer_correct(self):
        q = Question(id="D2-001", text="What?", category=Category.D,
                     sub_category=SubCategory.D2, reference_answers=["42"])
        assert check_correct("42", q) is True


class TestParseConfidence:
    """Test confidence parsing (mirrors test_parse_confidence.py)."""

    def test_leading_bare(self):
        assert parse_confidence("85") == 0.85

    def test_confidence_label(self):
        assert parse_confidence("Confidence: 90") == 0.90

    def test_with_answer(self):
        assert parse_confidence("Answer: Paris\nConfidence (0-100): 95") == 0.95

    def test_correctness_confidence(self):
        assert parse_confidence("Correctness confidence (0-100): 80") == 0.80

    def test_percent(self):
        assert parse_confidence("I am 75% sure") == 0.75

    def test_fraction(self):
        assert parse_confidence("About 60/100") == 0.60

    def test_no_confidence(self):
        assert parse_confidence("Just some text with no numbers over hundred") is None

    def test_boundary_zero(self):
        assert parse_confidence("0") == 0.0

    def test_boundary_hundred(self):
        assert parse_confidence("100") == 1.0
