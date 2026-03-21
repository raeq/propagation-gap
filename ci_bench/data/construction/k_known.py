"""Construct the K (Known) set for CI-Bench pilot.

Pulls questions from TriviaQA (or Natural Questions) and screens them
against a target model: keep questions the model answers correctly on
>=9 out of 10 runs at temperature 0.7.

Usage:
    python -m ci_bench.data.construction.k_known \
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --n-source 50 \
        --n-runs 10 \
        --threshold 0.9 \
        --output data/pilot/k_known.json

Requires: pip install datasets
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ci_bench.data.schema import (
    BenchmarkDataset,
    Category,
    Question,
    SubCategory,
)


def load_triviaqa_questions(n: int, seed: int = 42) -> list[dict]:
    """Load n questions from TriviaQA via HuggingFace datasets.

    Returns list of dicts with 'question' and 'answers' keys.
    'answers' is a list of acceptable answer strings.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' package required. Install with:\n"
            "  pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    print("  Loading TriviaQA from HuggingFace...", file=sys.stderr, flush=True)
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    # Deterministic shuffle.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))[:n].tolist()

    questions = []
    for idx in indices:
        item = ds[int(idx)]
        q_text = item["question"]
        # TriviaQA stores answers in item["answer"]["aliases"] + item["answer"]["value"]
        answer_obj = item["answer"]
        aliases = list(answer_obj.get("aliases", []))
        value = answer_obj.get("value", "")
        all_answers = list(set([value] + aliases))
        all_answers = [a.strip() for a in all_answers if a.strip()]

        questions.append({
            "question": q_text,
            "answers": all_answers,
            "source_idx": int(idx),
        })

    print(f"  Loaded {len(questions)} source questions.", file=sys.stderr, flush=True)
    return questions


def normalise_answer(text: str) -> str:
    """Basic normalisation for answer comparison."""
    text = text.lower().strip()
    # Remove articles.
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation.
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_answer(model_output: str, reference_answers: list[str]) -> bool:
    """Check if the model output contains any reference answer."""
    model_norm = normalise_answer(model_output)
    for ref in reference_answers:
        ref_norm = normalise_answer(ref)
        if ref_norm and ref_norm in model_norm:
            return True
    return False


def screen_questions(
    model,
    source_questions: list[dict],
    n_runs: int = 10,
    threshold: float = 0.9,
    temperature: float = 0.7,
    base_seed: int = 1000,
) -> list[dict]:
    """Screen questions against the model.

    For each question, run the model n_runs times at the given temperature.
    Keep questions where the model answers correctly >= threshold fraction.

    Each run uses a deterministic seed derived from the question index
    and run number: seed = base_seed + (question_index * n_runs) + run.
    This makes screening fully reproducible while ensuring different
    random states across questions and runs.

    Returns the source dicts augmented with screening metadata.
    """
    from ci_bench.models.prompts import get_template

    template = get_template("direct", variant=1)
    passed = []

    n = len(source_questions)
    for i, sq in enumerate(source_questions):
        q_text = sq["question"]
        ref_answers = sq["answers"]
        prompt = template.render(q_text)

        correct_count = 0
        responses = []
        for run in range(n_runs):
            run_seed = base_seed + (i * n_runs) + run
            resp = model.generate(
                prompt, temperature=temperature, max_tokens=128, seed=run_seed,
            )
            is_correct = check_answer(resp.text, ref_answers)
            correct_count += int(is_correct)
            responses.append({
                "text": resp.text[:200],  # Truncate for storage.
                "correct": is_correct,
                "seed": run_seed,
            })

        accuracy = correct_count / n_runs
        sq["screening"] = {
            "accuracy": accuracy,
            "n_runs": n_runs,
            "temperature": temperature,
            "responses": responses,
        }

        status = "PASS" if accuracy >= threshold else "FAIL"
        print(
            f"  [{i + 1}/{n}] acc={accuracy:.1%} {status} — {q_text[:60]}",
            file=sys.stderr,
            flush=True,
        )

        if accuracy >= threshold:
            passed.append(sq)

    print(
        f"\n  Screening complete: {len(passed)}/{n} passed (threshold {threshold:.0%})",
        file=sys.stderr,
        flush=True,
    )
    return passed


def build_k_dataset(
    screened_questions: list[dict],
    start_id: int = 1,
) -> BenchmarkDataset:
    """Convert screened questions to a BenchmarkDataset."""
    dataset = BenchmarkDataset(version="pilot-0.1")
    for i, sq in enumerate(screened_questions):
        q = Question(
            id=f"K-{start_id + i:03d}",
            text=sq["question"],
            category=Category.K,
            sub_category=SubCategory.K,
            reference_answers=sq["answers"],
            source="TriviaQA",
            metadata={
                "screening": sq["screening"],
                "source_idx": sq.get("source_idx"),
            },
        )
        dataset.add(q)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct K (Known) set for CI-Bench pilot."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        help="Model path or HuggingFace ID.",
    )
    parser.add_argument(
        "--n-source",
        type=int,
        default=50,
        help="Number of source questions to pull from TriviaQA.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of model runs per question for screening.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum accuracy to qualify as K (default 0.9 = 9/10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question sampling.",
    )
    parser.add_argument(
        "--output",
        default="data/pilot/k_known.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    # Load source questions.
    source = load_triviaqa_questions(args.n_source, seed=args.seed)

    # Load model.
    print(
        f"\n{'=' * 60}\n  Loading model: {args.model}\n{'=' * 60}\n",
        file=sys.stderr,
    )
    from ci_bench.models.mlx_model import MLXModel

    model = MLXModel(args.model)

    # Screen.
    print(
        f"\n{'=' * 60}\n  Screening {len(source)} questions "
        f"(n_runs={args.n_runs}, threshold={args.threshold:.0%})\n{'=' * 60}\n",
        file=sys.stderr,
    )
    passed = screen_questions(
        model, source, n_runs=args.n_runs, threshold=args.threshold
    )

    # Build and save dataset.
    dataset = build_k_dataset(passed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    print(
        f"\n  Saved {len(dataset)} K questions to {output_path}",
        file=sys.stderr,
    )
    print(f"\n  Summary: {dataset.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
