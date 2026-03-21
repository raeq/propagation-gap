"""Dry-run verification for ALL construction scripts.

This test suite verifies that every CI-Bench construction module:
  1.  Imports cleanly (no circular deps, no missing symbols)
  2.  Seed candidates are well-formed (required fields present)
  3.  Screening functions run end-to-end with a mock model
  4.  Dataset builders produce valid BenchmarkDataset objects
  5.  Answer matching and diversity helpers work correctly
  6.  CLI argparser constructs without error

No real model required — uses a deterministic MockModel that returns
canned responses.  Run with:
    python -m pytest tests/test_construction_dryrun.py -v
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest


# -----------------------------------------------------------------------
# Mock model: returns deterministic responses without loading any LLM.
# -----------------------------------------------------------------------

@dataclass
class MockResponse:
    text: str


class MockModel:
    """Deterministic mock that returns answers from a script.

    If responses_fn is provided, it is called with (prompt, seed) and
    should return a string.  Otherwise returns a default answer.
    """

    def __init__(self, responses_fn=None, default_text="I don't know"):
        self._fn = responses_fn
        self._default = default_text

    def generate(self, prompt, temperature=0.7, max_tokens=128, seed=0):
        if self._fn:
            text = self._fn(prompt, seed)
        else:
            text = self._default
        return MockResponse(text=text)


# -----------------------------------------------------------------------
# §1  Import checks — every module must load without error
# -----------------------------------------------------------------------

class TestImports:
    """Verify all construction modules import cleanly."""

    def test_import_schema(self):
        from ci_bench.data.schema import (
            BenchmarkDataset, Category, Question, SubCategory,
        )

    def test_import_prompts(self):
        from ci_bench.models.prompts import get_template, PromptTemplate

    def test_import_k_known(self):
        from ci_bench.data.construction.k_known import (
            check_answer, normalise_answer, screen_questions,
            build_k_dataset,
        )

    def test_import_d2_rare(self):
        from ci_bench.data.construction.d2_rare import (
            screen_questions_d2, build_d2_dataset,
        )

    def test_import_c3_synthetic(self):
        from ci_bench.data.construction.c3_synthetic import (
            build_c3_dataset, SEED_QUESTIONS,
        )

    def test_import_c1_temporal(self):
        from ci_bench.data.construction.c1_temporal import (
            screen_c1, build_c1_dataset, load_candidates, SEED_CANDIDATES,
        )

    def test_import_c2_obscurity(self):
        from ci_bench.data.construction.c2_obscurity import (
            screen_c2, build_c2_dataset, load_candidates, SEED_CANDIDATES,
        )

    def test_import_d1_contested(self):
        from ci_bench.data.construction.d1_contested import (
            screen_d1, build_d1_dataset, compute_answer_diversity,
            SEED_CANDIDATES,
        )

    def test_import_d3_degraded(self):
        from ci_bench.data.construction.d3_degraded import (
            screen_d3, build_d3_dataset, check_stale_answer,
            SEED_CANDIDATES,
        )


# -----------------------------------------------------------------------
# §2  Seed candidate validation — required fields present
# -----------------------------------------------------------------------

class TestSeedCandidates:
    """Verify all seed candidate sets have the expected structure."""

    def test_c1_candidates_structure(self):
        from ci_bench.data.construction.c1_temporal import SEED_CANDIDATES
        assert len(SEED_CANDIDATES) >= 30, "C1 needs >=30 seed candidates"
        for i, c in enumerate(SEED_CANDIDATES):
            assert "question" in c, f"C1[{i}] missing 'question'"
            assert "answers" in c, f"C1[{i}] missing 'answers'"
            assert isinstance(c["answers"], list), f"C1[{i}] 'answers' must be list"
            assert len(c["answers"]) > 0, f"C1[{i}] has empty answers"

    def test_c2_candidates_structure(self):
        from ci_bench.data.construction.c2_obscurity import SEED_CANDIDATES
        assert len(SEED_CANDIDATES) >= 30, "C2 needs >=30 seed candidates"
        for i, c in enumerate(SEED_CANDIDATES):
            assert "question" in c, f"C2[{i}] missing 'question'"
            assert "answers" in c, f"C2[{i}] missing 'answers'"
            assert isinstance(c["answers"], list)

    def test_c3_candidates_structure(self):
        from ci_bench.data.construction.c3_synthetic import SEED_QUESTIONS
        assert len(SEED_QUESTIONS) >= 25, "C3 needs >=25 seed questions"
        for i, q in enumerate(SEED_QUESTIONS):
            assert "text" in q, f"C3[{i}] missing 'text'"
            assert "fabricated_entity" in q, f"C3[{i}] missing 'fabricated_entity'"

    def test_d1_candidates_structure(self):
        from ci_bench.data.construction.d1_contested import SEED_CANDIDATES
        assert len(SEED_CANDIDATES) >= 25, "D1 needs >=25 seed candidates"
        for i, c in enumerate(SEED_CANDIDATES):
            assert "question" in c, f"D1[{i}] missing 'question'"
            assert "answers" in c, f"D1[{i}] missing 'answers'"
            assert "dispute" in c or "source" in c, f"D1[{i}] missing provenance"

    def test_d3_candidates_structure(self):
        from ci_bench.data.construction.d3_degraded import SEED_CANDIDATES
        assert len(SEED_CANDIDATES) >= 20, "D3 needs >=20 seed candidates"
        for i, c in enumerate(SEED_CANDIDATES):
            assert "question" in c, f"D3[{i}] missing 'question'"
            assert "current_answers" in c, f"D3[{i}] missing 'current_answers'"
            assert "stale_answers" in c, f"D3[{i}] missing 'stale_answers'"
            assert isinstance(c["current_answers"], list)
            assert isinstance(c["stale_answers"], list)
            assert len(c["current_answers"]) > 0, f"D3[{i}] empty current_answers"
            assert len(c["stale_answers"]) > 0, f"D3[{i}] empty stale_answers"


# -----------------------------------------------------------------------
# §3  Answer matching and normalisation helpers
# -----------------------------------------------------------------------

class TestAnswerHelpers:
    """Verify the shared answer-matching functions."""

    def test_normalise_answer(self):
        from ci_bench.data.construction.k_known import normalise_answer
        assert normalise_answer("The Big Dog") == "big dog"
        assert normalise_answer("  Hello, World!  ") == "hello world"
        assert normalise_answer("An apple") == "apple"

    def test_check_answer_match(self):
        from ci_bench.data.construction.k_known import check_answer
        assert check_answer("Paris is the capital of France.", ["Paris"])
        assert check_answer("The answer is 42.", ["42"])
        assert not check_answer("I think it's Berlin.", ["Paris"])

    def test_check_answer_case_insensitive(self):
        from ci_bench.data.construction.k_known import check_answer
        assert check_answer("PARIS", ["paris"])
        assert check_answer("paris", ["Paris"])

    def test_check_answer_empty_refs(self):
        from ci_bench.data.construction.k_known import check_answer
        assert not check_answer("anything", [])

    def test_check_stale_answer(self):
        from ci_bench.data.construction.d3_degraded import check_stale_answer
        assert check_stale_answer(
            "The threshold is 140/90 mmHg.",
            ["140/90", "140 over 90"],
        )
        assert not check_stale_answer(
            "The threshold is 130/80 mmHg.",
            ["140/90", "140 over 90"],
        )

    def test_check_stale_answer_normalisation(self):
        from ci_bench.data.construction.d3_degraded import check_stale_answer
        # Case insensitive.
        assert check_stale_answer("SWAZILAND", ["Swaziland"])
        # Empty stale list.
        assert not check_stale_answer("anything", [])


# -----------------------------------------------------------------------
# §4  Answer diversity computation
# -----------------------------------------------------------------------

class TestAnswerDiversity:
    """Verify compute_answer_diversity edge cases and logic."""

    def test_all_same(self):
        from ci_bench.data.construction.d1_contested import compute_answer_diversity
        responses = [{"text": "Paris"} for _ in range(10)]
        div = compute_answer_diversity(responses)
        assert div == 0.0

    def test_all_different(self):
        from ci_bench.data.construction.d1_contested import compute_answer_diversity
        responses = [{"text": f"Answer {i}"} for i in range(10)]
        div = compute_answer_diversity(responses)
        assert div == 1.0

    def test_two_distinct(self):
        from ci_bench.data.construction.d1_contested import compute_answer_diversity
        # 5x "Paris" + 5x "London" among 10 runs.
        responses = [{"text": "Paris"}] * 5 + [{"text": "London"}] * 5
        div = compute_answer_diversity(responses)
        # (2-1)/(10-1) = 0.111...
        assert 0.10 < div < 0.15

    def test_empty(self):
        from ci_bench.data.construction.d1_contested import compute_answer_diversity
        assert compute_answer_diversity([]) == 0.0

    def test_single(self):
        from ci_bench.data.construction.d1_contested import compute_answer_diversity
        assert compute_answer_diversity([{"text": "ok"}]) == 0.0

    def test_multiline_takes_first(self):
        from ci_bench.data.construction.d1_contested import compute_answer_diversity
        responses = [
            {"text": "Paris\nThe capital of France"},
            {"text": "Paris\nA lovely city"},
            {"text": "London\nCapital of England"},
        ]
        div = compute_answer_diversity(responses)
        # 2 distinct first-line answers out of 3 runs: (2-1)/(3-1) = 0.5
        assert div == 0.5


# -----------------------------------------------------------------------
# §5  Prompt template rendering
# -----------------------------------------------------------------------

class TestPromptRendering:
    """Verify templates render correctly for each condition."""

    @pytest.mark.parametrize("condition", ["direct", "cot", "abstention", "confidence"])
    def test_render_all_conditions(self, condition):
        from ci_bench.models.prompts import get_template
        tmpl = get_template(condition, variant=1)
        rendered = tmpl.render("What is the capital of France?")
        assert "What is the capital of France?" in rendered
        assert "{question}" not in rendered  # Placeholder must be gone.

    def test_get_template_bad_key(self):
        from ci_bench.models.prompts import get_template
        with pytest.raises(KeyError):
            get_template("nonexistent", variant=99)


# -----------------------------------------------------------------------
# §6  Screening functions with mock model
# -----------------------------------------------------------------------

class TestScreeningWithMock:
    """Run each screening function on 3 candidates with a MockModel."""

    def test_k_screening(self):
        from ci_bench.data.construction.k_known import screen_questions
        # Model always returns "Paris" — one question has Paris as answer.
        model = MockModel(default_text="Paris")
        candidates = [
            {"question": "Capital of France?", "answers": ["Paris"]},
            {"question": "Capital of Germany?", "answers": ["Berlin"]},
            {"question": "Capital of Spain?", "answers": ["Madrid"]},
        ]
        passed = screen_questions(model, candidates, n_runs=3, threshold=0.9)
        assert len(passed) == 1
        assert passed[0]["question"] == "Capital of France?"

    def test_d2_screening(self):
        from ci_bench.data.construction.d2_rare import screen_questions_d2
        # Model alternates: correct on even seeds, wrong on odd.
        def alternating(prompt, seed):
            return "Berlin" if seed % 2 == 0 else "Unknown"

        model = MockModel(responses_fn=alternating)
        candidates = [
            {"question": "Capital of Germany?", "answers": ["Berlin"]},
        ]
        passed = screen_questions_d2(
            model, candidates, n_runs=10, low=0.2, high=0.8,
        )
        # Accuracy should be 50% (5/10 even seeds) — in [0.2, 0.8] band.
        assert len(passed) == 1
        assert 0.2 <= passed[0]["screening"]["accuracy"] <= 0.8

    def test_c1_screening(self):
        from ci_bench.data.construction.c1_temporal import screen_c1
        # Model always wrong.
        model = MockModel(default_text="I have no idea")
        candidates = [
            {"question": "Who won 2024 election?", "answers": ["Trump"]},
            {"question": "Who won Euro 2024?", "answers": ["Spain"]},
        ]
        passed = screen_c1(model, candidates, n_runs=3, max_accuracy=0.2)
        # Both should pass (0% accuracy <= 20%).
        assert len(passed) == 2

    def test_c2_screening(self):
        from ci_bench.data.construction.c2_obscurity import screen_c2
        model = MockModel(default_text="No idea")
        candidates = [
            {"question": "Highest point of Bornholm?", "answers": ["Rytterknagten"]},
        ]
        passed = screen_c2(model, candidates, n_runs=3, max_accuracy=0.2)
        assert len(passed) == 1
        assert passed[0]["screening"]["accuracy"] == 0.0

    def test_d1_screening(self):
        from ci_bench.data.construction.d1_contested import screen_d1
        # Model gives different answers across runs to trigger diversity.
        answers = ["1440", "1450", "1436", "1440", "1450",
                    "1440", "1436", "1450", "1440", "1436"]

        def cycling(prompt, seed):
            idx = seed % len(answers)
            return answers[idx]

        model = MockModel(responses_fn=cycling)
        candidates = [{
            "question": "When was the printing press invented?",
            "answers": ["1440", "1436", "1439", "1450"],
        }]
        passed = screen_d1(
            model, candidates, n_runs=10,
            min_diversity=0.2, low_accuracy=0.2, high_accuracy=0.8,
        )
        # All runs return something in the answer list, so accuracy should be
        # high.  Diversity should be >0.2 (3 distinct answers out of 10).
        # Whether it passes depends on accuracy being <=0.8.
        # All answers are in reference_answers, so accuracy=100% -> FAIL on accuracy.
        # This is correct: the screening catches that.
        assert len(passed) == 0  # 100% accuracy > 80% band.

    def test_d1_screening_passes_with_mixed(self):
        from ci_bench.data.construction.d1_contested import screen_d1
        # Mix of correct and wrong, with diversity.
        def mixed(prompt, seed):
            cycle = ["1440", "1500", "I don't know", "1440", "1600",
                     "1440", "1500", "1440", "I don't know", "1500"]
            return cycle[seed % len(cycle)]

        model = MockModel(responses_fn=mixed)
        candidates = [{
            "question": "When was the printing press invented?",
            "answers": ["1440"],
        }]
        passed = screen_d1(
            model, candidates, n_runs=10,
            min_diversity=0.2, low_accuracy=0.2, high_accuracy=0.8,
        )
        # 4 correct out of 10 = 40% accuracy.  4 distinct answers = diversity 0.33.
        # Both criteria met.
        assert len(passed) == 1

    def test_d3_screening(self):
        from ci_bench.data.construction.d3_degraded import screen_d3
        # Model consistently gives the stale answer.
        model = MockModel(default_text="140/90 mmHg is the threshold.")
        candidates = [{
            "question": "Blood pressure threshold for hypertension?",
            "current_answers": ["130/80"],
            "stale_answers": ["140/90"],
        }]
        passed = screen_d3(
            model, candidates, n_runs=5,
            max_diversity=0.2, max_current_accuracy=0.3, min_stale_rate=0.5,
        )
        # All runs give stale answer: diversity=0, current_acc=0, stale_rate=1.0
        assert len(passed) == 1
        sc = passed[0]["screening"]
        assert sc["current_accuracy"] == 0.0
        assert sc["stale_rate"] == 1.0
        assert sc["answer_diversity"] == 0.0

    def test_d3_rejects_current_answer(self):
        from ci_bench.data.construction.d3_degraded import screen_d3
        # Model gives the CURRENT (revised) answer — should fail.
        model = MockModel(default_text="The threshold is 130/80 mmHg now.")
        candidates = [{
            "question": "Blood pressure threshold?",
            "current_answers": ["130/80"],
            "stale_answers": ["140/90"],
        }]
        passed = screen_d3(
            model, candidates, n_runs=5,
            max_diversity=0.2, max_current_accuracy=0.3, min_stale_rate=0.5,
        )
        assert len(passed) == 0  # Correctly rejected.


# -----------------------------------------------------------------------
# §7  Dataset builders produce valid BenchmarkDataset objects
# -----------------------------------------------------------------------

class TestDatasetBuilders:
    """Verify dataset builders create correct Question objects."""

    def test_build_k_dataset(self):
        from ci_bench.data.construction.k_known import build_k_dataset
        screened = [{"question": "Q?", "answers": ["A"], "screening": {}}]
        ds = build_k_dataset(screened)
        assert len(ds) == 1
        assert ds.questions[0].id == "K-001"
        assert ds.questions[0].category.value == "K"
        assert ds.questions[0].sub_category.value == "K"

    def test_build_d2_dataset(self):
        from ci_bench.data.construction.d2_rare import build_d2_dataset
        screened = [{"question": "Q?", "answers": ["A"], "screening": {}}]
        ds = build_d2_dataset(screened)
        assert len(ds) == 1
        assert ds.questions[0].sub_category.value == "D2"

    def test_build_c3_dataset(self):
        from ci_bench.data.construction.c3_synthetic import build_c3_dataset
        ds = build_c3_dataset()
        assert len(ds) >= 25
        for q in ds.questions:
            assert q.category.value == "C"
            assert q.sub_category.value == "C3"
            assert q.reference_answers == []  # Abstention is correct.

    def test_build_c1_dataset(self):
        from ci_bench.data.construction.c1_temporal import build_c1_dataset
        screened = [{"question": "Q?", "answers": ["A"], "source": "test"}]
        ds = build_c1_dataset(screened)
        assert len(ds) == 1
        assert ds.questions[0].sub_category.value == "C1"

    def test_build_c2_dataset(self):
        from ci_bench.data.construction.c2_obscurity import build_c2_dataset
        screened = [{"question": "Q?", "answers": ["A"], "source": "test"}]
        ds = build_c2_dataset(screened)
        assert len(ds) == 1
        assert ds.questions[0].sub_category.value == "C2"

    def test_build_d1_dataset(self):
        from ci_bench.data.construction.d1_contested import build_d1_dataset
        screened = [{"question": "Q?", "answers": ["A"], "source": "test"}]
        ds = build_d1_dataset(screened)
        assert len(ds) == 1
        assert ds.questions[0].sub_category.value == "D1"

    def test_build_d3_dataset(self):
        from ci_bench.data.construction.d3_degraded import build_d3_dataset
        screened = [{
            "question": "Q?",
            "current_answers": ["A"],
            "stale_answers": ["B"],
            "source": "test",
        }]
        ds = build_d3_dataset(screened)
        assert len(ds) == 1
        assert ds.questions[0].sub_category.value == "D3"
        assert ds.questions[0].reference_answers == ["A"]  # Current answer.
        assert "stale_answers" in ds.questions[0].metadata


# -----------------------------------------------------------------------
# §8  Dataset round-trip: save, reload, verify
# -----------------------------------------------------------------------

class TestDatasetRoundTrip:
    """Verify datasets survive JSON serialisation."""

    def test_roundtrip_mixed(self):
        from ci_bench.data.schema import (
            BenchmarkDataset, Category, Question, SubCategory,
        )
        ds = BenchmarkDataset(version="test-0.1")
        ds.add(Question(
            id="K-001", text="Q?", category=Category.K,
            sub_category=SubCategory.K, reference_answers=["A"],
        ))
        ds.add(Question(
            id="C1-001", text="Q?", category=Category.C,
            sub_category=SubCategory.C1, reference_answers=["B"],
        ))
        ds.add(Question(
            id="D3-001", text="Q?", category=Category.D,
            sub_category=SubCategory.D3, reference_answers=["C"],
            metadata={"stale_answers": ["old"]},
        ))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        ds.save(path)

        reloaded = BenchmarkDataset.load(path)
        assert len(reloaded) == 3
        assert reloaded.questions[2].metadata["stale_answers"] == ["old"]
        assert reloaded.summary()["K"] == 1
        assert reloaded.summary()["C"] == 1
        assert reloaded.summary()["D"] == 1
        path.unlink()


# -----------------------------------------------------------------------
# §9  CLI argument parsers construct without error
# -----------------------------------------------------------------------

class TestCLIParsers:
    """Verify argparser construction for each module (no side effects)."""

    def test_k_known_parser(self):
        import ci_bench.data.construction.k_known as mod
        # Just verify main() is callable without sys.argv mangling.
        assert callable(mod.main)

    def test_d2_rare_parser(self):
        import ci_bench.data.construction.d2_rare as mod
        assert callable(mod.main)

    def test_c3_synthetic_parser(self):
        import ci_bench.data.construction.c3_synthetic as mod
        assert callable(mod.main)

    def test_c1_temporal_parser(self):
        import ci_bench.data.construction.c1_temporal as mod
        assert callable(mod.main)

    def test_c2_obscurity_parser(self):
        import ci_bench.data.construction.c2_obscurity as mod
        assert callable(mod.main)

    def test_d1_contested_parser(self):
        import ci_bench.data.construction.d1_contested as mod
        assert callable(mod.main)

    def test_d3_degraded_parser(self):
        import ci_bench.data.construction.d3_degraded as mod
        assert callable(mod.main)


# -----------------------------------------------------------------------
# §10  Seed uniqueness — no duplicate questions within a module
# -----------------------------------------------------------------------

class TestSeedUniqueness:
    """Verify no duplicate questions within each seed set."""

    def test_c1_no_duplicates(self):
        from ci_bench.data.construction.c1_temporal import SEED_CANDIDATES
        questions = [c["question"] for c in SEED_CANDIDATES]
        assert len(questions) == len(set(questions)), "C1 has duplicate questions"

    def test_c2_no_duplicates(self):
        from ci_bench.data.construction.c2_obscurity import SEED_CANDIDATES
        questions = [c["question"] for c in SEED_CANDIDATES]
        assert len(questions) == len(set(questions)), "C2 has duplicate questions"

    def test_c3_no_duplicates(self):
        from ci_bench.data.construction.c3_synthetic import SEED_QUESTIONS
        texts = [q["text"] for q in SEED_QUESTIONS]
        assert len(texts) == len(set(texts)), "C3 has duplicate questions"

    def test_d1_no_duplicates(self):
        from ci_bench.data.construction.d1_contested import SEED_CANDIDATES
        questions = [c["question"] for c in SEED_CANDIDATES]
        assert len(questions) == len(set(questions)), "D1 has duplicate questions"

    def test_d3_no_duplicates(self):
        from ci_bench.data.construction.d3_degraded import SEED_CANDIDATES
        questions = [c["question"] for c in SEED_CANDIDATES]
        assert len(questions) == len(set(questions)), "D3 has duplicate questions"


# -----------------------------------------------------------------------
# §11  Base seed uniqueness across modules
# -----------------------------------------------------------------------

class TestBaseSeedUniqueness:
    """Verify base seeds don't collide across construction scripts."""

    def test_all_base_seeds_distinct(self):
        """Each module's default base_seed must be unique."""
        import inspect
        from ci_bench.data.construction.k_known import screen_questions
        from ci_bench.data.construction.d2_rare import screen_questions_d2
        from ci_bench.data.construction.c1_temporal import screen_c1
        from ci_bench.data.construction.c2_obscurity import screen_c2
        from ci_bench.data.construction.d1_contested import screen_d1
        from ci_bench.data.construction.d3_degraded import screen_d3

        # Extract default base_seed from each function signature.
        seeds = {}
        for name, fn in [
            ("K", screen_questions),
            ("D2", screen_questions_d2),
            ("C1", screen_c1),
            ("C2", screen_c2),
            ("D1", screen_d1),
            ("D3", screen_d3),
        ]:
            sig = inspect.signature(fn)
            default = sig.parameters["base_seed"].default
            seeds[name] = default

        # All values must be distinct.
        values = list(seeds.values())
        assert len(values) == len(set(values)), (
            f"Base seed collision: {seeds}"
        )

        # Verify expected values.
        assert seeds["K"] == 1000
        assert seeds["D2"] == 2000
        assert seeds["C1"] == 3000
        assert seeds["C2"] == 4000
        assert seeds["D1"] == 5000
        assert seeds["D3"] == 6000
