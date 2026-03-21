"""Tests for CI-Bench data schema: Question, BenchmarkDataset, round-trip."""

import json
import tempfile
from pathlib import Path

import pytest

from ci_bench.data.schema import (
    BenchmarkDataset,
    Category,
    Question,
    SubCategory,
    SUBCATEGORY_TO_CATEGORY,
)


class TestQuestion:
    def test_basic_construction(self):
        q = Question(
            id="K-001",
            text="What is the capital of France?",
            category=Category.K,
            sub_category=SubCategory.K,
            reference_answers=["Paris"],
            source="TriviaQA",
        )
        assert q.category == Category.K
        assert q.sub_category == SubCategory.K
        assert q.reference_answers == ["Paris"]

    def test_string_coercion(self):
        """Strings are coerced to enums."""
        q = Question(
            id="C3-001",
            text="Who is Dr. Elara Voss?",
            category="C",
            sub_category="C3",
        )
        assert q.category == Category.C
        assert q.sub_category == SubCategory.C3

    def test_mismatched_category_raises(self):
        """Sub-category must match parent category."""
        with pytest.raises(ValueError, match="belongs to category C"):
            Question(
                id="bad-001",
                text="Test",
                category=Category.D,
                sub_category=SubCategory.C1,
            )

    def test_round_trip(self):
        """to_dict -> from_dict preserves all fields."""
        q = Question(
            id="D2-017",
            text="What year was the Treaty of Tordesillas modified?",
            category=Category.D,
            sub_category=SubCategory.D2,
            reference_answers=["1506", "1529"],
            source="manual-curation",
            metadata={"screening_accuracy": 0.45, "domain": "history"},
        )
        d = q.to_dict()
        q2 = Question.from_dict(d)
        assert q2.id == q.id
        assert q2.text == q.text
        assert q2.category == q.category
        assert q2.sub_category == q.sub_category
        assert q2.reference_answers == q.reference_answers
        assert q2.source == q.source
        assert q2.metadata == q.metadata

    def test_json_serialisable(self):
        """to_dict output is JSON-serialisable."""
        q = Question(
            id="C1-005",
            text="Test question",
            category=Category.C,
            sub_category=SubCategory.C1,
        )
        s = json.dumps(q.to_dict())
        assert isinstance(s, str)


class TestSubcategoryMapping:
    def test_all_subcategories_mapped(self):
        """Every SubCategory has a parent in SUBCATEGORY_TO_CATEGORY."""
        for sub in SubCategory:
            assert sub in SUBCATEGORY_TO_CATEGORY

    def test_c_subcategories(self):
        for sub in [SubCategory.C1, SubCategory.C2, SubCategory.C3]:
            assert SUBCATEGORY_TO_CATEGORY[sub] == Category.C

    def test_d_subcategories(self):
        for sub in [SubCategory.D1, SubCategory.D2, SubCategory.D3]:
            assert SUBCATEGORY_TO_CATEGORY[sub] == Category.D


class TestBenchmarkDataset:
    def _make_dataset(self) -> BenchmarkDataset:
        """Create a small test dataset."""
        ds = BenchmarkDataset(version="test")
        ds.add(Question(id="K-001", text="Q1", category="K", sub_category="K",
                        reference_answers=["A1"]))
        ds.add(Question(id="C1-001", text="Q2", category="C", sub_category="C1"))
        ds.add(Question(id="C3-001", text="Q3", category="C", sub_category="C3"))
        ds.add(Question(id="D1-001", text="Q4", category="D", sub_category="D1"))
        ds.add(Question(id="D2-001", text="Q5", category="D", sub_category="D2"))
        return ds

    def test_length(self):
        ds = self._make_dataset()
        assert len(ds) == 5

    def test_summary(self):
        ds = self._make_dataset()
        s = ds.summary()
        assert s["total"] == 5
        assert s["K"] == 1
        assert s["C"] == 2
        assert s["D"] == 2
        assert s["C1"] == 1
        assert s["C3"] == 1

    def test_filter_by_category(self):
        ds = self._make_dataset()
        c_questions = ds.filter(category=Category.C)
        assert len(c_questions) == 2
        assert all(q.category == Category.C for q in c_questions)

    def test_filter_by_subcategory(self):
        ds = self._make_dataset()
        d2 = ds.filter(sub_category=SubCategory.D2)
        assert len(d2) == 1
        assert d2[0].id == "D2-001"

    def test_duplicate_id_raises(self):
        ds = self._make_dataset()
        with pytest.raises(ValueError, match="Duplicate"):
            ds.add(Question(id="K-001", text="Dup", category="K", sub_category="K"))

    def test_json_round_trip(self):
        """Save to JSON, load back, verify equality."""
        ds = self._make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_bench.json"
            ds.save(path)

            ds2 = BenchmarkDataset.load(path)
            assert len(ds2) == len(ds)
            assert ds2.version == ds.version
            for q1, q2 in zip(ds.questions, ds2.questions):
                assert q1.id == q2.id
                assert q1.text == q2.text
                assert q1.category == q2.category
                assert q1.sub_category == q2.sub_category
                assert q1.reference_answers == q2.reference_answers

    def test_repr(self):
        ds = self._make_dataset()
        r = repr(ds)
        assert "total=5" in r
        assert "K=1" in r
