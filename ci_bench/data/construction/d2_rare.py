"""Construct the D2 (Rare-but-present) set for CI-Bench pilot.

Pulls questions from TriviaQA and screens them against a target model:
keep questions the model answers correctly on 3-7 out of 10 runs at
temperature 0.7. These represent depth ignorance — the model was
exposed to the information but didn't learn it reliably.

Usage:
    python -m ci_bench.data.construction.d2_rare \
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --n-source 100 \
        --n-runs 10 \
        --output data/pilot/d2_rare.json

Requires: pip install datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ci_bench.data.construction.k_known import (
    check_answer,
    load_triviaqa_questions,
)
from ci_bench.data.schema import (
    BenchmarkDataset,
    Category,
    Question,
    SubCategory,
)


def screen_questions_d2(
    model,
    source_questions: list[dict],
    n_runs: int = 10,
    low: float = 0.3,
    high: float = 0.7,
    temperature: float = 0.7,
    base_seed: int = 2000,
) -> list[dict]:
    """Screen questions for the D2 accuracy band.

    Keep questions where the model answers correctly between low and high
    fraction of n_runs. This identifies questions the model has partial
    knowledge of — the hallmark of depth ignorance.

    Each run uses a deterministic seed derived from the question index
    and run number: seed = base_seed + (question_index * n_runs) + run.
    The base_seed differs from k_known.py (1000) to avoid correlated
    random states.

    Args:
        model: Model instance with generate().
        source_questions: Questions with 'question' and 'answers' keys.
        n_runs: Number of model runs per question.
        low: Lower bound of accuracy band (inclusive).
        high: Upper bound of accuracy band (inclusive).
        temperature: Sampling temperature for screening runs.
        base_seed: Base seed for deterministic per-run RNG pinning.

    Returns:
        Screened questions augmented with screening metadata.
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
                "text": resp.text[:200],
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

        in_band = low <= accuracy <= high
        status = "PASS" if in_band else "FAIL"
        print(
            f"  [{i + 1}/{n}] acc={accuracy:.1%} {status} — {q_text[:60]}",
            file=sys.stderr,
            flush=True,
        )

        if in_band:
            passed.append(sq)

    print(
        f"\n  Screening complete: {len(passed)}/{n} in D2 band "
        f"[{low:.0%}, {high:.0%}]",
        file=sys.stderr,
        flush=True,
    )
    return passed


def build_d2_dataset(
    screened_questions: list[dict],
    start_id: int = 1,
) -> BenchmarkDataset:
    """Convert screened D2 questions to a BenchmarkDataset."""
    dataset = BenchmarkDataset(version="pilot-0.1")
    for i, sq in enumerate(screened_questions):
        q = Question(
            id=f"D2-{start_id + i:03d}",
            text=sq["question"],
            category=Category.D,
            sub_category=SubCategory.D2,
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
        description="Construct D2 (Rare-but-present) set for CI-Bench pilot."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        help="Model path or HuggingFace ID.",
    )
    parser.add_argument(
        "--n-source",
        type=int,
        default=100,
        help="Number of source questions to pull from TriviaQA.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of model runs per question for screening.",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=0.3,
        help="Lower accuracy bound for D2 band.",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=0.7,
        help="Upper accuracy bound for D2 band.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,  # Different seed from K-set to avoid overlap.
        help="Random seed for question sampling.",
    )
    parser.add_argument(
        "--output",
        default="data/pilot/d2_rare.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    # Load source questions (different seed from K-set).
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
        f"(n_runs={args.n_runs}, band=[{args.low:.0%}, {args.high:.0%}])"
        f"\n{'=' * 60}\n",
        file=sys.stderr,
    )
    passed = screen_questions_d2(
        model, source,
        n_runs=args.n_runs,
        low=args.low,
        high=args.high,
    )

    # Build and save dataset.
    dataset = build_d2_dataset(passed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    print(
        f"\n  Saved {len(dataset)} D2 questions to {output_path}",
        file=sys.stderr,
    )
    print(f"\n  Summary: {dataset.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
