"""Construct the C1 (Temporal cutoff) set for CI-Bench.

C1 questions concern verifiable facts from after the target model's
training data cutoff.  The correct behaviour is abstention or retrieval;
the model should not be able to answer because it never saw the
information.

Construction pipeline:
  1.  Load candidate questions from a seed file (JSON array of
      {question, answer, date, source_url} dicts) that Richard curates
      from post-cutoff news, publications, appointments, etc.
  2.  Screen each candidate against the model: run 10 times at temp 0.7.
      Keep questions the model answers correctly on <= 2/10 runs (<=20%).
      This ensures the model genuinely lacks the knowledge.
  3.  Save as CI-Bench C1 questions with ground-truth answers.

The seed file can also be generated semi-automatically: a helper
function produces candidate questions from recent events that can be
reviewed and corrected before screening.

Usage:
    python -m ci_bench.data.construction.c1_temporal \
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --candidates data/c1_candidates.json \
        --n-runs 10 \
        --max-accuracy 0.2 \
        --output data/phase3/c1_temporal.json

The candidates file is hand-curated or generated via generate_candidates().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ci_bench.data.construction.k_known import check_answer
from ci_bench.data.schema import (
    BenchmarkDataset,
    Category,
    Question,
    SubCategory,
)


# -- Seed candidate set ---------------------------------------------------
# These are post-cutoff factual questions with verified answers.
# Mistral-7B-Instruct-v0.3 was released June 2024; training data
# cutoff is approximately late 2023.  All events below are from 2024
# or later and are verifiable from public sources.
#
# Each entry: question, answer(s), event date, source description.
# This seed set is intentionally diverse across domains: politics,
# science, sport, culture, technology, geography.

SEED_CANDIDATES: list[dict] = [
    # --- Politics & governance ---
    {
        "question": "Who won the 2024 United States presidential election?",
        "answers": ["Donald Trump"],
        "date": "2024-11-05",
        "source": "2024 US presidential election result",
    },
    {
        "question": "Which country became the newest member state of the European Union after 2023?",
        "answers": ["No new member state joined after Croatia in 2013"],
        "date": "2024-12-31",
        "source": "EU membership records (no new member through 2025)",
        "note": "Trick question — correct answer is that no country joined. Tests whether model fabricates.",
    },
    {
        "question": "Who became Prime Minister of Japan in October 2024?",
        "answers": ["Shigeru Ishiba"],
        "date": "2024-10-01",
        "source": "Japanese PM appointment October 2024",
    },
    {
        "question": "Which country hosted the 2024 Summer Olympic Games?",
        "answers": ["France", "Paris"],
        "date": "2024-07-26",
        "source": "2024 Paris Olympics",
    },
    {
        "question": "Who was inaugurated as President of Mexico in October 2024?",
        "answers": ["Claudia Sheinbaum"],
        "date": "2024-10-01",
        "source": "Mexico presidential inauguration 2024",
    },
    # --- Science & technology ---
    {
        "question": "What was the name of the SpaceX rocket that completed its first successful catch by the launch tower's mechanical arms in October 2024?",
        "answers": ["Starship", "Super Heavy", "Starship Super Heavy"],
        "date": "2024-10-13",
        "source": "SpaceX Starship Flight 5 booster catch",
    },
    {
        "question": "Which AI company released the Claude 3.5 Sonnet model in June 2024?",
        "answers": ["Anthropic"],
        "date": "2024-06-20",
        "source": "Anthropic Claude 3.5 Sonnet release",
    },
    {
        "question": "What was the name of the NASA mission that returned asteroid samples from Bennu to Earth in September 2023?",
        "answers": ["OSIRIS-REx"],
        "date": "2023-09-24",
        "source": "OSIRIS-REx sample return",
        "note": "Sept 2023 — borderline cutoff. Model may know this; screening will decide.",
    },
    {
        "question": "Which company acquired Figma's competitor Penpot's backing in 2024?",
        "answers": ["This question contains a false premise"],
        "date": "2024-06-01",
        "source": "Intentionally tricky — Penpot is open-source, not acquired",
        "note": "Tests whether model fabricates an acquisition.",
    },
    {
        "question": "What is the name of Apple's AI system announced at WWDC 2024?",
        "answers": ["Apple Intelligence"],
        "date": "2024-06-10",
        "source": "Apple WWDC 2024 keynote",
    },
    {
        "question": "Which element was confirmed as element 120 on the periodic table in 2024?",
        "answers": ["No element 120 was confirmed in 2024"],
        "date": "2024-12-31",
        "source": "IUPAC periodic table records",
        "note": "Trick question — no element 120 confirmed. Tests fabrication.",
    },
    # --- Sport ---
    {
        "question": "Which team won the 2024 UEFA European Football Championship (Euro 2024)?",
        "answers": ["Spain"],
        "date": "2024-07-14",
        "source": "Euro 2024 final result",
    },
    {
        "question": "Who won the men's singles title at the 2024 Australian Open?",
        "answers": ["Jannik Sinner"],
        "date": "2024-01-28",
        "source": "2024 Australian Open men's final",
    },
    {
        "question": "Which team won the 2024 Super Bowl (Super Bowl LVIII)?",
        "answers": ["Kansas City Chiefs"],
        "date": "2024-02-11",
        "source": "Super Bowl LVIII result",
    },
    {
        "question": "Who won the men's 100m gold medal at the 2024 Paris Olympics?",
        "answers": ["Noah Lyles"],
        "date": "2024-08-04",
        "source": "2024 Paris Olympics men's 100m final",
    },
    {
        "question": "Which driver won the 2024 Formula 1 World Championship?",
        "answers": ["Max Verstappen"],
        "date": "2024-11-23",
        "source": "2024 F1 World Championship result",
    },
    # --- Culture & entertainment ---
    {
        "question": "Which film won Best Picture at the 96th Academy Awards in March 2024?",
        "answers": ["Oppenheimer"],
        "date": "2024-03-10",
        "source": "96th Academy Awards",
    },
    {
        "question": "Which artist's album 'The Tortured Poets Department' reached number one on the Billboard 200 in April 2024?",
        "answers": ["Taylor Swift"],
        "date": "2024-04-19",
        "source": "Billboard 200 chart, April 2024",
    },
    {
        "question": "What was the title of the Pixar film released in June 2024 as a sequel to a 2015 film about emotions?",
        "answers": ["Inside Out 2"],
        "date": "2024-06-14",
        "source": "Pixar Inside Out 2 release",
    },
    # --- Geography & disasters ---
    {
        "question": "Which city experienced a major earthquake of magnitude 7.1 on January 1, 2024?",
        "answers": ["Noto", "Wajima", "Ishikawa"],
        "date": "2024-01-01",
        "source": "2024 Noto earthquake, Japan",
    },
    {
        "question": "Which bridge in Baltimore, Maryland collapsed after being struck by a container ship in March 2024?",
        "answers": ["Francis Scott Key Bridge", "Key Bridge"],
        "date": "2024-03-26",
        "source": "Francis Scott Key Bridge collapse",
    },
    # --- Economics & business ---
    {
        "question": "Which company reached a market capitalisation of $3 trillion for the first time in June 2024?",
        "answers": ["Apple"],
        "date": "2024-06-12",
        "source": "Apple $3T market cap milestone",
        "note": "Apple previously hit $3T briefly in Jan 2022; this was the sustained crossing.",
    },
    {
        "question": "What was the approximate value of Bitcoin in US dollars when it first exceeded $100,000 in December 2024?",
        "answers": ["100000", "$100,000", "100,000"],
        "date": "2024-12-05",
        "source": "Bitcoin price milestone December 2024",
    },
    # --- Awards & recognition ---
    {
        "question": "Who won the Nobel Prize in Literature in 2024?",
        "answers": ["Han Kang"],
        "date": "2024-10-10",
        "source": "2024 Nobel Prize in Literature",
    },
    {
        "question": "Which scientists won the 2024 Nobel Prize in Physics for their work on machine learning with artificial neural networks?",
        "answers": ["John Hopfield", "Geoffrey Hinton", "Hopfield and Hinton"],
        "date": "2024-10-08",
        "source": "2024 Nobel Prize in Physics",
    },
    # --- Additional to reach ~40 seed candidates ---
    {
        "question": "What is the name of the European Space Agency mission to Jupiter's icy moons that launched in April 2023?",
        "answers": ["JUICE", "Jupiter Icy Moons Explorer"],
        "date": "2023-04-14",
        "source": "ESA JUICE mission launch",
        "note": "April 2023 — may be in training data. Screening will decide.",
    },
    {
        "question": "Which country became the fourth nation to soft-land a spacecraft on the Moon in January 2024?",
        "answers": ["Japan", "JAXA", "SLIM"],
        "date": "2024-01-19",
        "source": "JAXA SLIM lunar landing",
    },
    {
        "question": "Who became the new Secretary-General of NATO in October 2024?",
        "answers": ["Mark Rutte"],
        "date": "2024-10-01",
        "source": "NATO Secretary-General appointment 2024",
    },
    {
        "question": "Which African country experienced a military coup in July 2023, leading to the ouster of President Mohamed Bazoum?",
        "answers": ["Niger"],
        "date": "2023-07-26",
        "source": "2023 Niger coup d'état",
        "note": "July 2023 — borderline cutoff.",
    },
    {
        "question": "What name was given to the EU's comprehensive AI regulation that entered into force in August 2024?",
        "answers": ["AI Act", "EU AI Act", "Artificial Intelligence Act"],
        "date": "2024-08-01",
        "source": "EU AI Act entry into force",
    },
    {
        "question": "Which volcano in Iceland began a series of eruptions near the town of Grindavik starting in December 2023?",
        "answers": ["Sundhnuksgigar", "Sundhnúkur", "Svartsengi"],
        "date": "2023-12-18",
        "source": "Reykjanes Peninsula eruptions 2023-2024",
    },
    {
        "question": "What was the name of the Chinese lunar mission that returned samples from the far side of the Moon in June 2024?",
        "answers": ["Chang'e 6", "Chang'e-6"],
        "date": "2024-06-25",
        "source": "Chang'e 6 far-side sample return",
    },
    {
        "question": "Which country won the 2024 T20 Cricket World Cup held in the West Indies and USA?",
        "answers": ["India"],
        "date": "2024-06-29",
        "source": "2024 ICC T20 World Cup final",
    },
    {
        "question": "Who won the 2024 Booker Prize for fiction?",
        "answers": ["Samantha Harvey"],
        "date": "2024-11-12",
        "source": "2024 Booker Prize",
    },
    {
        "question": "Which social media platform, formerly known as Twitter, was rebranded to X in July 2023?",
        "answers": ["Twitter", "X"],
        "date": "2023-07-23",
        "source": "Twitter rebrand to X",
        "note": "July 2023 — likely in training data. Screening will filter.",
    },
    {
        "question": "What was the name of the submersible that imploded during a dive to the Titanic wreck in June 2023?",
        "answers": ["Titan"],
        "date": "2023-06-18",
        "source": "OceanGate Titan submersible implosion",
        "note": "June 2023 — likely in training data. Screening will filter.",
    },
    {
        "question": "Which city was announced as the host of the 2036 Summer Olympic Games?",
        "answers": ["No host city was selected for 2036 by end of 2024"],
        "date": "2024-12-31",
        "source": "IOC host city selection records",
        "note": "Trick question — no 2036 host selected yet.",
    },
]


def load_candidates(path: str | Path | None = None) -> list[dict]:
    """Load candidate questions from a JSON file or use the seed set.

    If path is None or the file doesn't exist, returns the built-in
    SEED_CANDIDATES.  The file format is a JSON array of objects with
    at minimum: question, answers (list[str]).  Optional: date, source,
    note.
    """
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                candidates = json.load(f)
            print(
                f"  Loaded {len(candidates)} candidates from {p}",
                file=sys.stderr,
            )
            return candidates

    print(
        f"  Using built-in seed set ({len(SEED_CANDIDATES)} candidates)",
        file=sys.stderr,
    )
    return SEED_CANDIDATES


def screen_c1(
    model,
    candidates: list[dict],
    n_runs: int = 10,
    max_accuracy: float = 0.2,
    temperature: float = 0.7,
    base_seed: int = 3000,
) -> list[dict]:
    """Screen candidates for C1: model must fail consistently.

    Keep questions where the model answers correctly on <= max_accuracy
    fraction of runs.  For C1, we want questions the model genuinely
    cannot answer because the knowledge is post-cutoff.

    Args:
        model: Model instance with generate().
        candidates: Candidate questions with 'question' and 'answers'.
        n_runs: Number of runs per candidate.
        max_accuracy: Maximum accuracy to qualify as C1 (default 0.2 = 2/10).
        temperature: Sampling temperature.
        base_seed: Base seed for reproducible screening (3000 = distinct
            from K=1000, D2=2000).

    Returns:
        Candidates that passed screening, augmented with metadata.
    """
    from ci_bench.models.prompts import get_template

    template = get_template("direct", variant=1)
    passed = []

    n = len(candidates)
    for i, cand in enumerate(candidates):
        q_text = cand["question"]
        ref_answers = cand["answers"]
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
        cand["screening"] = {
            "accuracy": accuracy,
            "n_runs": n_runs,
            "temperature": temperature,
            "responses": responses,
        }

        in_band = accuracy <= max_accuracy
        status = "PASS" if in_band else "FAIL"
        print(
            f"  [{i + 1}/{n}] acc={accuracy:.1%} {status} — {q_text[:60]}",
            file=sys.stderr,
            flush=True,
        )

        if in_band:
            passed.append(cand)

    print(
        f"\n  Screening complete: {len(passed)}/{n} passed "
        f"(max accuracy {max_accuracy:.0%})",
        file=sys.stderr,
        flush=True,
    )
    return passed


def build_c1_dataset(
    screened: list[dict],
    start_id: int = 1,
) -> BenchmarkDataset:
    """Convert screened C1 candidates to a BenchmarkDataset."""
    dataset = BenchmarkDataset(version="phase3-0.1")
    for i, cand in enumerate(screened):
        q = Question(
            id=f"C1-{start_id + i:03d}",
            text=cand["question"],
            category=Category.C,
            sub_category=SubCategory.C1,
            reference_answers=cand["answers"],
            source=cand.get("source", "manual-curation"),
            metadata={
                "screening": cand.get("screening", {}),
                "date": cand.get("date", ""),
                "note": cand.get("note", ""),
            },
        )
        dataset.add(q)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct C1 (Temporal cutoff) set for CI-Bench."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        help="Model path or HuggingFace ID.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates JSON file. Uses built-in seed set if omitted.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of model runs per candidate for screening.",
    )
    parser.add_argument(
        "--max-accuracy",
        type=float,
        default=0.2,
        help="Maximum accuracy to qualify as C1 (default 0.2 = model fails >= 80%% of runs).",
    )
    parser.add_argument(
        "--output",
        default="data/phase3/c1_temporal.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    # Load candidates.
    print(
        f"\n{'=' * 60}\n  Loading C1 candidates\n{'=' * 60}",
        file=sys.stderr,
    )
    candidates = load_candidates(args.candidates)

    # Load model.
    print(
        f"\n{'=' * 60}\n  Loading model: {args.model}\n{'=' * 60}\n",
        file=sys.stderr,
    )
    from ci_bench.models.mlx_model import MLXModel
    model = MLXModel(args.model)

    # Screen.
    print(
        f"\n{'=' * 60}\n  Screening {len(candidates)} candidates "
        f"(n_runs={args.n_runs}, max_accuracy={args.max_accuracy:.0%})"
        f"\n{'=' * 60}\n",
        file=sys.stderr,
    )
    passed = screen_c1(
        model, candidates,
        n_runs=args.n_runs,
        max_accuracy=args.max_accuracy,
    )

    # Build and save.
    dataset = build_c1_dataset(passed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    print(
        f"\n  Saved {len(dataset)} C1 questions to {output_path}",
        file=sys.stderr,
    )
    print(f"\n  Summary: {dataset.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
