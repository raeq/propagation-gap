"""Activation extraction for CI-Bench probing experiments.

Extracts per-layer last-token hidden states from open-weight models
using the MLXModel wrapper. Handles batching, progress reporting,
and saving/loading of extracted activations.

Usage:
    from ci_bench.models.mlx_model import MLXModel
    from ci_bench.probes.extract import extract_batch, save_activations

    model = MLXModel("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    questions = dataset.filter(category=Category.C)

    activations = extract_batch(model, questions)
    save_activations(activations, "activations/mistral7b_C.npz")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ci_bench.models.mlx_model import MLXModel

from ci_bench.data.schema import Question, Category, SubCategory


@dataclass
class ActivationData:
    """Container for extracted activations with metadata.

    Attributes:
        hidden_states: Array of shape (n_questions, n_layers + 1, hidden_dim).
            The +1 is for the post-norm hidden state.
        question_ids: List of question IDs in the same order.
        categories: List of category labels (K, C, D).
        sub_categories: List of sub-category labels (K, C1, ..., D3).
        model_id: Model that produced these activations.
        n_layers: Number of transformer layers (not counting norm).
        hidden_dim: Hidden dimension size.
    """

    hidden_states: np.ndarray  # (n_questions, n_layers + 1, hidden_dim)
    question_ids: list[str]
    categories: list[str]
    sub_categories: list[str]
    model_id: str
    n_layers: int
    hidden_dim: int

    def filter_by_category(self, category: Category) -> "ActivationData":
        """Return a new ActivationData with only the given category."""
        mask = [c == category.value for c in self.categories]
        indices = [i for i, m in enumerate(mask) if m]
        return ActivationData(
            hidden_states=self.hidden_states[indices],
            question_ids=[self.question_ids[i] for i in indices],
            categories=[self.categories[i] for i in indices],
            sub_categories=[self.sub_categories[i] for i in indices],
            model_id=self.model_id,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )

    def filter_by_sub_category(self, sub: SubCategory) -> "ActivationData":
        """Return a new ActivationData with only the given sub-category."""
        mask = [s == sub.value for s in self.sub_categories]
        indices = [i for i, m in enumerate(mask) if m]
        return ActivationData(
            hidden_states=self.hidden_states[indices],
            question_ids=[self.question_ids[i] for i in indices],
            categories=[self.categories[i] for i in indices],
            sub_categories=[self.sub_categories[i] for i in indices],
            model_id=self.model_id,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )

    def get_layer(self, layer_idx: int) -> np.ndarray:
        """Get activations for a single layer: shape (n_questions, hidden_dim)."""
        return self.hidden_states[:, layer_idx, :]

    def __len__(self) -> int:
        return len(self.question_ids)

    def __repr__(self) -> str:
        return (
            f"ActivationData(n={len(self)}, layers={self.n_layers}, "
            f"dim={self.hidden_dim}, model={self.model_id})"
        )


def extract_batch(
    model: "MLXModel",
    questions: list[Question],
    prompt_prefix: str = "",
    verbose: bool = True,
) -> ActivationData:
    """Extract last-token hidden states for a batch of questions.

    Processes questions one at a time (mlx doesn't batch well for
    variable-length inputs) and collects per-layer hidden states.

    Args:
        model: An MLXModel instance (already loaded).
        questions: List of Question objects to process.
        prompt_prefix: Optional prefix added before each question text
            (e.g., "Answer the following question: ").
        verbose: Print progress to stderr.

    Returns:
        ActivationData with shape (n_questions, n_layers + 1, hidden_dim).
    """
    import mlx.core as mx  # Deferred import.

    all_states: list[np.ndarray] = []
    question_ids: list[str] = []
    categories: list[str] = []
    sub_categories: list[str] = []

    n = len(questions)
    for i, q in enumerate(questions):
        if verbose and (i % 10 == 0 or i == n - 1):
            print(
                f"  Extracting [{i + 1}/{n}] {q.id}...",
                file=sys.stderr,
                flush=True,
            )

        text = prompt_prefix + q.text

        # Extract: returns mx.array of shape (n_layers + 1, hidden_dim).
        states_mx = model.extract_last_token_hidden_states(text)

        # Convert to numpy immediately to free mlx memory.
        states_np = np.array(states_mx)
        all_states.append(states_np)

        question_ids.append(q.id)
        categories.append(q.category.value)
        sub_categories.append(q.sub_category.value)

    # Stack: (n_questions, n_layers + 1, hidden_dim).
    hidden_states = np.stack(all_states, axis=0)

    return ActivationData(
        hidden_states=hidden_states,
        question_ids=question_ids,
        categories=categories,
        sub_categories=sub_categories,
        model_id=model.model_id,
        n_layers=model.n_layers,
        hidden_dim=model.hidden_dim,
    )


def save_activations(data: ActivationData, path: str | Path) -> None:
    """Save extracted activations to a compressed .npz file.

    Args:
        data: The ActivationData to save.
        path: Output file path (should end in .npz).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hidden_states=data.hidden_states,
        question_ids=np.array(data.question_ids),
        categories=np.array(data.categories),
        sub_categories=np.array(data.sub_categories),
        model_id=np.array(data.model_id),
        n_layers=np.array(data.n_layers),
        hidden_dim=np.array(data.hidden_dim),
    )


def load_activations(path: str | Path) -> ActivationData:
    """Load extracted activations from a .npz file.

    Args:
        path: Path to the .npz file saved by save_activations.

    Returns:
        ActivationData reconstructed from the file.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    return ActivationData(
        hidden_states=data["hidden_states"],
        question_ids=data["question_ids"].tolist(),
        categories=data["categories"].tolist(),
        sub_categories=data["sub_categories"].tolist(),
        model_id=str(data["model_id"]),
        n_layers=int(data["n_layers"]),
        hidden_dim=int(data["hidden_dim"]),
    )
