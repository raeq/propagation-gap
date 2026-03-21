"""Data schema for CI-Bench questions and datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Category(str, Enum):
    """Top-level ignorance category."""

    K = "K"  # Known — model answers correctly and reliably
    C = "C"  # Coverage-ignorant — training data unlikely to support generalisation
    D = "D"  # Depth-ignorant — exposed to but not learned reliably


class SubCategory(str, Enum):
    """Fine-grained sub-category within K, C, or D."""

    K = "K"  # Known (no sub-categories)
    C1 = "C1"  # Temporal cutoff
    C2 = "C2"  # Extreme obscurity
    C3 = "C3"  # Synthetic (fabricated entities)
    D1 = "D1"  # Contested facts
    D2 = "D2"  # Rare-but-present
    D3 = "D3"  # Degraded knowledge


# Mapping from sub-category to parent category.
SUBCATEGORY_TO_CATEGORY: dict[SubCategory, Category] = {
    SubCategory.K: Category.K,
    SubCategory.C1: Category.C,
    SubCategory.C2: Category.C,
    SubCategory.C3: Category.C,
    SubCategory.D1: Category.D,
    SubCategory.D2: Category.D,
    SubCategory.D3: Category.D,
}


@dataclass
class Question:
    """A single CI-Bench question with ground-truth labels.

    Attributes:
        id: Unique identifier (e.g., "C3-042").
        text: The question text as presented to the model.
        category: Top-level category (K, C, or D).
        sub_category: Fine-grained sub-category.
        reference_answers: Acceptable answers. For C3 (synthetic), this is
            empty — the correct response is abstention.
        source: Provenance string (e.g., "TriviaQA", "manual-curation").
        metadata: Arbitrary additional fields (construction notes, screening
            results, ground-truth provenance).
    """

    id: str
    text: str
    category: Category
    sub_category: SubCategory
    reference_answers: list[str] = field(default_factory=list)
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce strings to enums if needed (e.g., when loading from JSON).
        if isinstance(self.category, str):
            self.category = Category(self.category)
        if isinstance(self.sub_category, str):
            self.sub_category = SubCategory(self.sub_category)

        # Validate sub-category matches category.
        expected_parent = SUBCATEGORY_TO_CATEGORY[self.sub_category]
        if self.category != expected_parent:
            raise ValueError(
                f"Sub-category {self.sub_category.value} belongs to category "
                f"{expected_parent.value}, not {self.category.value}"
            )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        d = asdict(self)
        d["category"] = self.category.value
        d["sub_category"] = self.sub_category.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Question:
        """Deserialise from a dict."""
        return cls(**d)


class BenchmarkDataset:
    """A collection of CI-Bench questions with filtering and I/O.

    Attributes:
        questions: The list of Question objects.
        version: Dataset version string.
    """

    def __init__(
        self,
        questions: Optional[list[Question]] = None,
        version: str = "0.1.0",
    ) -> None:
        self.questions: list[Question] = questions or []
        self.version = version
        self._id_set: set[str] = {q.id for q in self.questions}

    def add(self, question: Question) -> None:
        """Add a question. Raises ValueError on duplicate ID."""
        if question.id in self._id_set:
            raise ValueError(f"Duplicate question ID: {question.id}")
        self.questions.append(question)
        self._id_set.add(question.id)

    def filter(
        self,
        category: Optional[Category] = None,
        sub_category: Optional[SubCategory] = None,
    ) -> list[Question]:
        """Return questions matching the given category/sub-category filter."""
        result = self.questions
        if category is not None:
            result = [q for q in result if q.category == category]
        if sub_category is not None:
            result = [q for q in result if q.sub_category == sub_category]
        return result

    def summary(self) -> dict[str, int]:
        """Count questions per category and sub-category."""
        counts: dict[str, int] = {}
        for cat in Category:
            counts[cat.value] = sum(
                1 for q in self.questions if q.category == cat
            )
        for sub in SubCategory:
            counts[sub.value] = sum(
                1 for q in self.questions if q.sub_category == sub
            )
        counts["total"] = len(self.questions)
        return counts

    def save(self, path: str | Path) -> None:
        """Save to JSON."""
        path = Path(path)
        data = {
            "version": self.version,
            "questions": [q.to_dict() for q in self.questions],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkDataset":
        """Load from a single JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        questions = [Question.from_dict(q) for q in data["questions"]]
        return cls(questions=questions, version=data.get("version", "unknown"))

    @classmethod
    def from_directory(cls, directory: str | Path) -> "BenchmarkDataset":
        """Load and merge all JSON files in a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Not a directory: {directory}")
        combined = cls()
        for json_file in sorted(directory.glob("*.json")):
            partial = cls.from_file(json_file)
            for q in partial.questions:
                combined.add(q)
        if not combined.questions:
            raise ValueError(f"No questions found in {directory}")
        return combined

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkDataset:
        """Load from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        questions = [Question.from_dict(d) for d in data["questions"]]
        return cls(questions=questions, version=data.get("version", "0.0.0"))

    def __len__(self) -> int:
        return len(self.questions)

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"BenchmarkDataset(v{self.version}, "
            f"total={s['total']}, K={s['K']}, C={s['C']}, D={s['D']})"
        )
