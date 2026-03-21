"""Construct the C2 (Extreme obscurity) set for CI-Bench.

C2 questions concern real entities that are verifiable but extremely
low-frequency: minor municipal officials, obscure species, niche
technical standards, small regional landmarks.  The claim is not that
these are literally absent from training data, but that they are
sparse enough that no reliable pattern was learned.

Construction pipeline:
  1.  Load candidate questions from a seed file or use the built-in set.
      Each candidate has a question, verified answer, and source.
  2.  Screen against the model: run 10 times at temp 0.7.
      Keep questions with <= 20% accuracy (model fails consistently).
  3.  Save as CI-Bench C2 questions.

Unlike C1, C2 questions are not about timing — the entities have always
existed.  The model's ignorance stems from extreme sparsity in the
training distribution, not from temporal cutoff.

Usage:
    python -m ci_bench.data.construction.c2_obscurity \
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --candidates data/c2_candidates.json \
        --output data/phase3/c2_obscurity.json
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
# Real entities that are verifiable but extremely obscure.
# All answers verified from authoritative sources.
# Deliberately diverse: biology, geography, governance, standards,
# history, infrastructure.

SEED_CANDIDATES: list[dict] = [
    # --- Minor municipal governance ---
    {
        "question": "What is the name of the mayor of Berwick-upon-Tweed as of 2023?",
        "answers": ["This is a civil parish with a town council, not a mayor in the traditional sense"],
        "source": "Berwick-upon-Tweed Town Council records",
        "domain": "governance",
        "note": "Trick: Berwick has a Town Council, not a mayor. Tests whether model fabricates a name.",
    },
    {
        "question": "What is the population of Ystad municipality in Skane County, Sweden, according to the 2023 census?",
        "answers": ["approximately 31000", "31000", "about 31,000"],
        "source": "Statistics Sweden (SCB) 2023 municipal population data",
        "domain": "geography",
    },
    {
        "question": "What is the name of the river that flows through the town of Brechin in Angus, Scotland?",
        "answers": ["South Esk", "River South Esk"],
        "source": "Ordnance Survey / Angus Council",
        "domain": "geography",
    },
    {
        "question": "What is the IATA airport code for Karup Airport in Denmark?",
        "answers": ["KRP"],
        "source": "IATA airport code registry",
        "domain": "infrastructure",
    },
    {
        "question": "Which Danish island is connected to the mainland by the Sallingsund Bridge?",
        "answers": ["Mors"],
        "source": "Danish Road Directorate / Vejdirektoratet",
        "domain": "geography",
    },
    # --- Obscure species taxonomy ---
    {
        "question": "What is the common name of the moth species Xestia c-nigrum?",
        "answers": ["Setaceous Hebrew Character"],
        "source": "UK Moths / Lepidoptera taxonomy",
        "domain": "biology",
    },
    {
        "question": "What family does the freshwater fish species Cobitis taenia belong to?",
        "answers": ["Cobitidae", "loach family"],
        "source": "FishBase / ITIS taxonomy",
        "domain": "biology",
    },
    {
        "question": "What is the conservation status of the Seychelles sheath-tailed bat (Coleura seychellensis) according to the IUCN Red List?",
        "answers": ["Critically Endangered", "CR"],
        "source": "IUCN Red List",
        "domain": "biology",
    },
    {
        "question": "What is the binomial name of the European mudminnow?",
        "answers": ["Umbra krameri"],
        "source": "FishBase taxonomy",
        "domain": "biology",
    },
    {
        "question": "Which genus does the rare British wildflower Lady's Slipper Orchid belong to?",
        "answers": ["Cypripedium"],
        "source": "Botanical Society of Britain and Ireland",
        "domain": "biology",
    },
    # --- Niche technical standards ---
    {
        "question": "What does the ISO standard ISO 8601-2:2019 specifically extend or refine compared to ISO 8601-1?",
        "answers": ["date and time extensions", "extensions to ISO 8601-1", "profile and extensions"],
        "source": "ISO 8601-2:2019 standard abstract",
        "domain": "standards",
    },
    {
        "question": "What is the purpose of the ASTM D3039 testing standard?",
        "answers": ["tensile properties of polymer matrix composite materials", "tensile testing of composites"],
        "source": "ASTM International standard D3039",
        "domain": "standards",
    },
    {
        "question": "What measurement does the Vickers hardness test (ISO 6507) use as its indenter?",
        "answers": ["diamond pyramid", "square-based diamond pyramid"],
        "source": "ISO 6507 / materials testing",
        "domain": "standards",
    },
    # --- Regional history ---
    {
        "question": "In which year was the Tay Road Bridge in Dundee, Scotland opened to traffic?",
        "answers": ["1966"],
        "source": "Tay Road Bridge Joint Board records",
        "domain": "history",
    },
    {
        "question": "What was the original name of the city now known as Kolkata, India, before the official name change in 2001?",
        "answers": ["Calcutta"],
        "source": "Government of West Bengal gazette notification",
        "domain": "history",
        "note": "Well-known — may be filtered out by screening.",
    },
    {
        "question": "What is the name of the medieval castle in Visby on the Swedish island of Gotland?",
        "answers": ["Visborg", "Visborgs slott"],
        "source": "Swedish National Heritage Board / Riksantikvarieambetet",
        "domain": "history",
    },
    {
        "question": "Which treaty ended the Dano-Swedish War of 1658?",
        "answers": ["Treaty of Roskilde"],
        "source": "Nordic history references",
        "domain": "history",
        "note": "May be known — screening decides.",
    },
    # --- Obscure infrastructure ---
    {
        "question": "What is the length in kilometres of the Oresund Bridge connecting Denmark and Sweden?",
        "answers": ["7.845", "approximately 8", "7.8"],
        "source": "Oresundsbron official specifications",
        "domain": "infrastructure",
        "note": "Bridge is well-known; exact length may not be.",
    },
    {
        "question": "What is the name of the railway station that serves Edinburgh Airport in Scotland?",
        "answers": ["Edinburgh Gateway"],
        "source": "ScotRail / Network Rail",
        "domain": "infrastructure",
    },
    {
        "question": "What is the gauge of the railway system in Sri Lanka in millimetres?",
        "answers": ["1676", "broad gauge", "Indian gauge"],
        "source": "Sri Lanka Railways / UIC railway gauge database",
        "domain": "infrastructure",
    },
    # --- Obscure geography ---
    {
        "question": "What is the highest point on the Danish island of Bornholm?",
        "answers": ["Rytterknagten"],
        "source": "Danish Geodata Agency / Geodatastyrelsen",
        "domain": "geography",
    },
    {
        "question": "What is the name of the deepest lake in Scotland?",
        "answers": ["Loch Morar"],
        "source": "British Geological Survey",
        "domain": "geography",
    },
    {
        "question": "What is the name of the strait between the islands of Skye and Raasay in Scotland?",
        "answers": ["Sound of Raasay"],
        "source": "Ordnance Survey",
        "domain": "geography",
    },
    {
        "question": "Which volcano is the highest point on the Azores archipelago?",
        "answers": ["Mount Pico", "Ponta do Pico", "Pico"],
        "source": "Instituto Geografico Portugues",
        "domain": "geography",
    },
    {
        "question": "What is the name of the largest lake entirely within the borders of Estonia?",
        "answers": ["Lake Vortsjarv", "Vortsjarv"],
        "source": "Estonian Land Board",
        "domain": "geography",
    },
    # --- Obscure cultural / academic ---
    {
        "question": "What is the name of the annual traditional horse race held in Siena, Italy?",
        "answers": ["Palio", "Palio di Siena", "Il Palio"],
        "source": "Comune di Siena",
        "domain": "culture",
        "note": "Moderately well-known — screening may filter.",
    },
    {
        "question": "What is the name of the professional body that regulates solicitors in Scotland?",
        "answers": ["Law Society of Scotland"],
        "source": "Law Society of Scotland official website",
        "domain": "governance",
    },
    {
        "question": "What is the Dewey Decimal Classification number range for the subject of chemistry?",
        "answers": ["540", "540-549"],
        "source": "OCLC Dewey Decimal Classification system",
        "domain": "standards",
    },
    {
        "question": "What is the name of the unit of measurement equal to one nautical mile per hour?",
        "answers": ["knot"],
        "source": "International Bureau of Weights and Measures",
        "domain": "standards",
        "note": "Well-known — will likely be filtered by screening.",
    },
    {
        "question": "What is the name of the traditional Scandinavian open-faced sandwich?",
        "answers": ["smorrebrod", "smørrebrød"],
        "source": "Danish culinary tradition",
        "domain": "culture",
        "note": "Moderately known — screening will filter.",
    },
    # --- More obscure to compensate for known items ---
    {
        "question": "What is the name of the highest mountain in the Faroe Islands?",
        "answers": ["Slaettaratindur"],
        "source": "Faroese Environment Agency",
        "domain": "geography",
    },
    {
        "question": "In which year was the Storstrøm Bridge in Denmark originally opened?",
        "answers": ["1937"],
        "source": "Danish Road Directorate historical records",
        "domain": "infrastructure",
    },
    {
        "question": "What is the name of the UNESCO World Heritage Site consisting of burial mounds near the town of Jelling in Denmark?",
        "answers": ["Jelling Mounds", "Jelling Stones", "Jelling"],
        "source": "UNESCO World Heritage List",
        "domain": "history",
        "note": "The Jelling stones are somewhat known. Screening decides.",
    },
    {
        "question": "What is the maximum depth in metres of Loch Ness?",
        "answers": ["230", "approximately 230", "227"],
        "source": "British Geological Survey bathymetric data",
        "domain": "geography",
        "note": "Loch Ness is famous; exact depth less known.",
    },
    {
        "question": "What ISO standard defines the format for International Standard Book Numbers (ISBN)?",
        "answers": ["ISO 2108"],
        "source": "ISO 2108:2017",
        "domain": "standards",
    },
    {
        "question": "What is the name of the type of traditional Norwegian wooden church from the medieval period?",
        "answers": ["stave church", "stavkirke"],
        "source": "Norwegian Directorate for Cultural Heritage",
        "domain": "culture",
        "note": "Moderately known — screening decides.",
    },
    {
        "question": "What was the former name of the country now known as Eswatini?",
        "answers": ["Swaziland"],
        "source": "UN member state records",
        "domain": "geography",
        "note": "Moderately known — screening decides.",
    },
    {
        "question": "What is the name of the smallest bone in the human body?",
        "answers": ["stapes", "stirrup"],
        "source": "Gray's Anatomy",
        "domain": "biology",
        "note": "Well-known fact — will be filtered by screening.",
    },
    {
        "question": "What is the name of the channel separating the Isle of Wight from mainland England?",
        "answers": ["Solent", "The Solent"],
        "source": "UK Hydrographic Office",
        "domain": "geography",
        "note": "Moderately known — screening decides.",
    },
    {
        "question": "What is the population of the Svalbard archipelago according to the most recent Norwegian census data?",
        "answers": ["approximately 2900", "about 2900", "2900", "roughly 3000"],
        "source": "Statistics Norway / Svalbard population register",
        "domain": "geography",
    },
]


def load_candidates(path: str | Path | None = None) -> list[dict]:
    """Load candidate questions from a JSON file or use the seed set."""
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


def screen_c2(
    model,
    candidates: list[dict],
    n_runs: int = 10,
    max_accuracy: float = 0.2,
    temperature: float = 0.7,
    base_seed: int = 4000,
) -> list[dict]:
    """Screen candidates for C2: model must fail consistently.

    Same logic as C1 screening, but with distinct base_seed (4000).
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


def build_c2_dataset(
    screened: list[dict],
    start_id: int = 1,
) -> BenchmarkDataset:
    """Convert screened C2 candidates to a BenchmarkDataset."""
    dataset = BenchmarkDataset(version="phase3-0.1")
    for i, cand in enumerate(screened):
        q = Question(
            id=f"C2-{start_id + i:03d}",
            text=cand["question"],
            category=Category.C,
            sub_category=SubCategory.C2,
            reference_answers=cand["answers"],
            source=cand.get("source", "manual-curation"),
            metadata={
                "screening": cand.get("screening", {}),
                "domain": cand.get("domain", ""),
                "note": cand.get("note", ""),
            },
        )
        dataset.add(q)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct C2 (Extreme obscurity) set for CI-Bench."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        help="Model path or HuggingFace ID.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates JSON. Uses built-in seed set if omitted.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of model runs per candidate.",
    )
    parser.add_argument(
        "--max-accuracy",
        type=float,
        default=0.2,
        help="Maximum accuracy to qualify as C2.",
    )
    parser.add_argument(
        "--output",
        default="data/phase3/c2_obscurity.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    print(
        f"\n{'=' * 60}\n  Loading C2 candidates\n{'=' * 60}",
        file=sys.stderr,
    )
    candidates = load_candidates(args.candidates)

    print(
        f"\n{'=' * 60}\n  Loading model: {args.model}\n{'=' * 60}\n",
        file=sys.stderr,
    )
    from ci_bench.models.mlx_model import MLXModel
    model = MLXModel(args.model)

    print(
        f"\n{'=' * 60}\n  Screening {len(candidates)} candidates "
        f"(n_runs={args.n_runs}, max_accuracy={args.max_accuracy:.0%})"
        f"\n{'=' * 60}\n",
        file=sys.stderr,
    )
    passed = screen_c2(
        model, candidates,
        n_runs=args.n_runs,
        max_accuracy=args.max_accuracy,
    )

    dataset = build_c2_dataset(passed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    print(
        f"\n  Saved {len(dataset)} C2 questions to {output_path}",
        file=sys.stderr,
    )
    print(f"\n  Summary: {dataset.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
