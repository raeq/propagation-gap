"""Construct the D1 (Contested facts) set for CI-Bench.

D1 questions concern topics where authoritative sources disagree:
disputed historical dates, conflicting scientific measurements,
evolving medical guidelines, ambiguous attribution.  The model
encountered multiple conflicting answers during training and learned
an unreliable representation.

Unlike D2 (rarity), D1 captures a *coherence* problem: the model saw
plenty of training examples but they disagree.  The signature is not
low accuracy but *inconsistent* answers across runs — the model
vacillates between competing learned patterns.

Construction pipeline:
  1.  Load candidate questions from a seed file or built-in set.
      Each candidate has a question, a set of plausible conflicting
      answers, and a source documenting the dispute.
  2.  Screen against the model: run 10 times at temp 0.7.
      D1 screening uses TWO criteria (both must be met):
        a. Answer entropy: the model gives >= 2 distinct answers
           across runs (normalised answer diversity >= 0.3).
        b. Moderate accuracy band: 20-80% (model is sometimes right,
           sometimes wrong, or gives different-but-plausible answers).
      This captures the specific D1 signature: the model has learned
      *something* but what it learned is incoherent.
  3.  Save as CI-Bench D1 questions.

Usage:
    python -m ci_bench.data.construction.d1_contested \
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --candidates data/d1_candidates.json \
        --output data/phase3/d1_contested.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from ci_bench.data.construction.k_known import check_answer, normalise_answer
from ci_bench.data.schema import (
    BenchmarkDataset,
    Category,
    Question,
    SubCategory,
)


# -- Seed candidate set ---------------------------------------------------
# Questions where authoritative sources genuinely disagree.
# Each entry documents the dispute and lists plausible conflicting answers.
# reference_answers contains the most widely accepted answer(s); the
# dispute_answers field lists plausible alternatives the model might give.
#
# Domains: history, science, medicine, geography, attribution.

SEED_CANDIDATES: list[dict] = [
    # --- Disputed historical dates & facts ---
    {
        "question": "In what year was the printing press invented by Johannes Gutenberg?",
        "answers": ["1440", "1436", "1439", "1450"],
        "dispute": "Sources cite dates ranging from 1436 to 1450. The commonly given date of '1440' or 'around 1440' is conventional but debated.",
        "source": "Multiple history of printing references",
        "domain": "history",
    },
    {
        "question": "How many people died in the Great Fire of London in 1666?",
        "answers": ["6", "8", "few"],
        "dispute": "Traditional accounts say 6-8 deaths, but historians argue the true toll was likely much higher as deaths among the poor and in outlying areas went unrecorded.",
        "source": "Museum of London / historical debate",
        "domain": "history",
    },
    {
        "question": "Who invented the telephone?",
        "answers": ["Alexander Graham Bell", "Bell"],
        "dispute": "Bell holds the patent, but Antonio Meucci, Elisha Gray, and others have competing claims. The US Congress passed a resolution in 2002 recognising Meucci's contributions.",
        "dispute_answers": ["Antonio Meucci", "Meucci", "Elisha Gray"],
        "source": "US House Resolution 269 (2002) / patent history",
        "domain": "history",
    },
    {
        "question": "Who discovered America?",
        "answers": ["Christopher Columbus", "Columbus"],
        "dispute": "Conventional Western answer is Columbus (1492), but Norse expeditions (Leif Erikson, c. 1000) preceded him, and indigenous peoples were already there.",
        "dispute_answers": ["Leif Erikson", "Leif Eriksson", "Vikings", "Indigenous peoples"],
        "source": "Historical consensus debate",
        "domain": "history",
    },
    {
        "question": "What was the population of the Roman Empire at its peak?",
        "answers": ["55 million", "60 million", "65 million", "70 million"],
        "dispute": "Estimates range from 55 million to 120 million depending on methodology. No consensus exists.",
        "source": "Scheidel (2007), Harper (2017), Frier (2000)",
        "domain": "history",
    },
    {
        "question": "In which year did the Western Roman Empire fall?",
        "answers": ["476", "480"],
        "dispute": "476 (deposition of Romulus Augustulus) is conventional but Julius Nepos ruled until 480. Some historians argue for a gradual process with no single date.",
        "dispute_answers": ["475", "480", "410"],
        "source": "Debate in late Roman historiography",
        "domain": "history",
    },
    # --- Conflicting scientific measurements ---
    {
        "question": "What is the age of the universe in billions of years?",
        "answers": ["13.8", "13.7", "13.787"],
        "dispute": "Planck satellite data gives 13.787 +/- 0.020 Gyr, but some measurements using the Hubble constant give younger estimates. The 'Hubble tension' is unresolved.",
        "dispute_answers": ["13.5", "12.5", "13.6"],
        "source": "Planck Collaboration (2020) vs. SH0ES Collaboration",
        "domain": "science",
    },
    {
        "question": "How many taste receptors does the human tongue have: four or five basic tastes?",
        "answers": ["five", "5"],
        "dispute": "Umami was added as the fifth basic taste, but older sources still list four. Some researchers propose additional tastes (fat, starch, calcium).",
        "dispute_answers": ["four", "4", "six"],
        "source": "Taste perception research / Chaudhari & Roper (2010)",
        "domain": "science",
    },
    {
        "question": "How many oceans are there on Earth?",
        "answers": ["5", "five"],
        "dispute": "Traditionally 4 (Atlantic, Pacific, Indian, Arctic). The Southern Ocean was recognised by NOAA in 2000 and IHO in 2021, but not all countries/organisations agree.",
        "dispute_answers": ["4", "four", "7", "seven"],
        "source": "IHO / NOAA recognition debate",
        "domain": "science",
    },
    {
        "question": "What is the boiling point of water in degrees Celsius?",
        "answers": ["100"],
        "dispute": "100°C at standard atmospheric pressure (101.325 kPa), but the exact value depends on pressure, dissolved substances, and the IAPWS-IF97 formulation gives 99.9743°C at 1 atm.",
        "dispute_answers": ["99.97", "99.974", "212"],
        "source": "IAPWS-IF97 / metrology references",
        "domain": "science",
        "note": "Model will likely say 100 consistently. May not pass D1 screen.",
    },
    {
        "question": "What is the speed of light in metres per second?",
        "answers": ["299792458"],
        "dispute": "The value is defined exactly since 1983, but models often give rounded or slightly wrong values (300000000, 299792, etc.).",
        "dispute_answers": ["300000000", "3e8", "186000 miles per second"],
        "source": "SI unit definitions / BIPM",
        "domain": "science",
        "note": "Tests whether model gives inconsistent precision across runs.",
    },
    {
        "question": "How many chromosomes do humans have?",
        "answers": ["46", "23 pairs"],
        "dispute": "46 is correct for typical humans, but models sometimes confuse with 48 (pre-1956 count) or cite rare conditions.",
        "dispute_answers": ["48", "23"],
        "source": "Tjio & Levan (1956) / cytogenetics",
        "domain": "science",
        "note": "Likely answered consistently. May not pass D1 screen.",
    },
    # --- Evolving medical/health guidelines ---
    {
        "question": "How many glasses of water should a person drink per day?",
        "answers": ["8", "eight", "varies"],
        "dispute": "The '8 glasses a day' rule has no scientific basis (Valtin, 2002). Modern guidelines say 'drink when thirsty' and vary by body weight, activity, and climate.",
        "dispute_answers": ["6", "10", "2 litres", "depends"],
        "source": "Valtin (2002) / Mayo Clinic / NHS guidelines",
        "domain": "medicine",
    },
    {
        "question": "Is it safe to eat eggs every day?",
        "answers": ["yes for most people", "generally safe"],
        "dispute": "US dietary guidelines removed the 300mg/day cholesterol limit in 2015, but some studies still associate high egg consumption with cardiovascular risk. Guidelines vary by country.",
        "dispute_answers": ["no", "in moderation", "depends on health conditions"],
        "source": "2015-2020 Dietary Guidelines / Zhong et al. JAMA 2019",
        "domain": "medicine",
    },
    {
        "question": "What is the recommended daily intake of sodium for adults in milligrams?",
        "answers": ["2300", "less than 2300"],
        "dispute": "WHO recommends <2000mg, US guidelines say <2300mg, AHA recommends <1500mg for most adults. Model may give any of these.",
        "dispute_answers": ["2000", "1500", "2400"],
        "source": "WHO / USDA / AHA conflicting guidelines",
        "domain": "medicine",
    },
    {
        "question": "At what age should women begin regular mammogram screening?",
        "answers": ["40", "50"],
        "dispute": "USPSTF changed recommendation from 50 to 40 in 2024. ACS says 40-44 optional, 45-54 annually. Different countries have different guidelines.",
        "dispute_answers": ["45", "50", "40"],
        "source": "USPSTF 2024 / ACS / NHS screening ages",
        "domain": "medicine",
    },
    {
        "question": "Is coffee good or bad for health?",
        "answers": ["generally beneficial in moderation", "moderate consumption is associated with health benefits"],
        "dispute": "Meta-analyses show reduced mortality at 3-5 cups/day, but studies also link excessive intake to anxiety, insomnia, and bone loss. Guidelines vary.",
        "dispute_answers": ["bad", "good", "depends", "mixed evidence"],
        "source": "Poole et al. BMJ 2017 / conflicting study results",
        "domain": "medicine",
    },
    # --- Disputed attribution ---
    {
        "question": "Who said 'The definition of insanity is doing the same thing over and over and expecting different results'?",
        "answers": ["Unknown", "often misattributed to Einstein"],
        "dispute": "Commonly attributed to Einstein, but there is no evidence he said it. Also attributed to Ben Franklin, Mark Twain, and various others. Earliest known version is from a 1981 Narcotics Anonymous text.",
        "dispute_answers": ["Albert Einstein", "Einstein", "Benjamin Franklin", "Mark Twain"],
        "source": "Quote Investigator / misattribution research",
        "domain": "attribution",
    },
    {
        "question": "Who said 'Let them eat cake'?",
        "answers": ["Marie Antoinette", "likely apocryphal"],
        "dispute": "Traditionally attributed to Marie Antoinette but the phrase appears in Rousseau's Confessions (written c. 1765) attributed to 'a great princess' when Marie Antoinette was 10 years old.",
        "dispute_answers": ["Marie Antoinette", "Rousseau", "unknown"],
        "source": "Rousseau Confessions / French Revolution historiography",
        "domain": "attribution",
    },
    {
        "question": "Who wrote the quote 'The only thing we have to fear is fear itself'?",
        "answers": ["Franklin D. Roosevelt", "FDR"],
        "dispute": "FDR's 1933 inaugural address, but the sentiment paraphrases earlier writers including Thoreau, Francis Bacon, and Michel de Montaigne.",
        "dispute_answers": ["Thoreau", "Francis Bacon"],
        "source": "FDR inaugural address / literary precedents",
        "domain": "attribution",
        "note": "FDR is the standard answer; model likely consistent. May not pass D1 screen.",
    },
    # --- Disputed geography/nomenclature ---
    {
        "question": "How many continents are there?",
        "answers": ["7", "seven"],
        "dispute": "Anglo-American convention says 7, but other models count 5 or 6 (combining Europe-Asia, or the Americas). The Olympic rings represent 5.",
        "dispute_answers": ["5", "6", "4"],
        "source": "Geographic convention variation by country",
        "domain": "geography",
    },
    {
        "question": "What is the longest river in the world?",
        "answers": ["Nile", "Amazon"],
        "dispute": "The Nile has traditionally been considered longest (~6,650 km), but recent measurements of the Amazon (~6,992 km including the Pará estuary) may make it longer. No consensus.",
        "dispute_answers": ["Nile", "Amazon", "Mississippi"],
        "source": "National Geographic / Brazilian geographic survey",
        "domain": "geography",
    },
    {
        "question": "What is the tallest mountain in the world?",
        "answers": ["Mount Everest", "Everest"],
        "dispute": "Everest is tallest above sea level, but Mauna Kea is tallest from base to peak, and Chimborazo's peak is farthest from Earth's centre.",
        "dispute_answers": ["Mauna Kea", "Chimborazo", "K2"],
        "source": "Depends on measurement criterion",
        "domain": "geography",
        "note": "Model will likely say Everest consistently. May not pass D1 screen.",
    },
    {
        "question": "Is Pluto a planet?",
        "answers": ["No", "dwarf planet"],
        "dispute": "IAU reclassified Pluto as a dwarf planet in 2006, but the decision remains contested. NASA's New Horizons team and some planetary scientists still argue for planet status.",
        "dispute_answers": ["yes", "no", "dwarf planet", "it depends"],
        "source": "IAU 2006 / ongoing debate in planetary science",
        "domain": "science",
    },
    # --- Conflicting nutritional science ---
    {
        "question": "Are saturated fats bad for heart health?",
        "answers": ["evidence is mixed", "current guidelines recommend limiting"],
        "dispute": "AHA says yes (reduce to <6% calories). But meta-analyses (Siri-Tarino 2010, de Souza 2015) found no clear association. Debate is ongoing.",
        "dispute_answers": ["yes", "no", "it depends", "not necessarily"],
        "source": "AHA guidelines vs. Siri-Tarino et al. meta-analysis",
        "domain": "medicine",
    },
    {
        "question": "How much protein should an average adult consume per kilogram of body weight per day?",
        "answers": ["0.8", "0.8g", "0.8 grams"],
        "dispute": "RDA is 0.8g/kg but many researchers argue this is a minimum to prevent deficiency, not optimal. Sports nutrition guidelines suggest 1.2-2.0g/kg.",
        "dispute_answers": ["1.0", "1.2", "1.6", "2.0"],
        "source": "RDA / ISSN position stand / Phillips & Van Loon (2011)",
        "domain": "medicine",
    },
    # --- Historical numbers & statistics ---
    {
        "question": "How many people died in the sinking of the Titanic?",
        "answers": ["1517", "1503", "1490"],
        "dispute": "Figures range from 1,490 to 1,635 depending on source. The British inquiry said 1,490; the US inquiry said 1,517. Modern estimates vary.",
        "source": "British and US inquiry reports / Titanic Historical Society",
        "domain": "history",
    },
    {
        "question": "What percentage of the Earth's surface is covered by water?",
        "answers": ["71", "about 71"],
        "dispute": "Commonly cited as 71%, but precise measurements give values from 70.8% to 71.2% depending on methodology and whether ice is included.",
        "dispute_answers": ["70", "72", "75"],
        "source": "NOAA / various earth science references",
        "domain": "science",
        "note": "Model likely consistent at 71%. May not pass D1 screen.",
    },
    {
        "question": "How long did the Hundred Years' War last?",
        "answers": ["116", "116 years"],
        "dispute": "Conventionally 1337-1453 = 116 years, but some historians date it differently (1328-1453 = 125 years) or argue it was multiple separate conflicts.",
        "dispute_answers": ["100", "hundred", "115", "119"],
        "source": "Medieval historiography",
        "domain": "history",
    },
    {
        "question": "When did the Middle Ages end?",
        "answers": ["1453", "1492", "1500"],
        "dispute": "No agreed date. Common candidates: fall of Constantinople (1453), Columbus (1492), or a round 1500. Some historians argue for the Reformation (1517).",
        "dispute_answers": ["1453", "1492", "1500", "1517"],
        "source": "Periodisation debate in European historiography",
        "domain": "history",
    },
]


def compute_answer_diversity(responses: list[dict]) -> float:
    """Compute normalised answer diversity across screening runs.

    Extracts the core answer from each response (first sentence,
    normalised), counts distinct answers, and returns a diversity
    score in [0, 1]:
      0.0 = all runs gave the same answer
      1.0 = every run gave a different answer

    The score is: (n_distinct - 1) / (n_runs - 1), clamped to [0, 1].
    """
    answers = []
    for r in responses:
        text = r.get("text", "")
        # Extract first substantive line as the "answer".
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            ans = normalise_answer(lines[0][:200])
            answers.append(ans)

    if len(answers) <= 1:
        return 0.0

    n_distinct = len(set(answers))
    diversity = (n_distinct - 1) / (len(answers) - 1)
    return min(1.0, max(0.0, diversity))


def screen_d1(
    model,
    candidates: list[dict],
    n_runs: int = 10,
    min_diversity: float = 0.2,
    low_accuracy: float = 0.2,
    high_accuracy: float = 0.8,
    temperature: float = 0.7,
    base_seed: int = 5000,
) -> list[dict]:
    """Screen candidates for D1: model must show answer inconsistency.

    D1 screening requires BOTH:
      - Answer diversity >= min_diversity (model gives different answers)
      - Accuracy in [low_accuracy, high_accuracy] (model is sometimes right)

    This captures the specific D1 signature: the model learned conflicting
    information and vacillates between competing patterns.

    Args:
        model: Model instance with generate().
        candidates: Candidate questions with 'question' and 'answers'.
        n_runs: Number of runs per candidate.
        min_diversity: Minimum normalised answer diversity (default 0.2).
        low_accuracy: Lower accuracy bound (default 0.2).
        high_accuracy: Upper accuracy bound (default 0.8).
        temperature: Sampling temperature.
        base_seed: Base seed (5000 = distinct from K=1000, D2=2000,
            C1=3000, C2=4000).

    Returns:
        Candidates that passed both criteria, with screening metadata.
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
        diversity = compute_answer_diversity(responses)

        cand["screening"] = {
            "accuracy": accuracy,
            "answer_diversity": round(diversity, 3),
            "n_runs": n_runs,
            "temperature": temperature,
            "responses": responses,
        }

        passes_diversity = diversity >= min_diversity
        passes_accuracy = low_accuracy <= accuracy <= high_accuracy
        in_band = passes_diversity and passes_accuracy

        # Also log questions that pass diversity but not accuracy (informative).
        if passes_diversity and not passes_accuracy:
            flag = "DIV_ONLY"
        elif passes_accuracy and not passes_diversity:
            flag = "ACC_ONLY"
        elif in_band:
            flag = "PASS"
        else:
            flag = "FAIL"

        print(
            f"  [{i + 1}/{n}] acc={accuracy:.1%} div={diversity:.2f} "
            f"{flag} — {q_text[:55]}",
            file=sys.stderr,
            flush=True,
        )

        if in_band:
            passed.append(cand)

    print(
        f"\n  Screening complete: {len(passed)}/{n} passed "
        f"(diversity >= {min_diversity}, accuracy [{low_accuracy:.0%}, "
        f"{high_accuracy:.0%}])",
        file=sys.stderr,
        flush=True,
    )
    return passed


def build_d1_dataset(
    screened: list[dict],
    start_id: int = 1,
) -> BenchmarkDataset:
    """Convert screened D1 candidates to a BenchmarkDataset."""
    dataset = BenchmarkDataset(version="phase3-0.1")
    for i, cand in enumerate(screened):
        q = Question(
            id=f"D1-{start_id + i:03d}",
            text=cand["question"],
            category=Category.D,
            sub_category=SubCategory.D1,
            reference_answers=cand["answers"],
            source=cand.get("source", "manual-curation"),
            metadata={
                "screening": cand.get("screening", {}),
                "dispute": cand.get("dispute", ""),
                "domain": cand.get("domain", ""),
                "dispute_answers": cand.get("dispute_answers", []),
                "note": cand.get("note", ""),
            },
        )
        dataset.add(q)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct D1 (Contested facts) set for CI-Bench."
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
        "--min-diversity",
        type=float,
        default=0.2,
        help="Minimum normalised answer diversity to qualify as D1.",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=0.2,
        help="Lower accuracy bound.",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=0.8,
        help="Upper accuracy bound.",
    )
    parser.add_argument(
        "--output",
        default="data/phase3/d1_contested.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    print(
        f"\n{'=' * 60}\n  Loading D1 candidates\n{'=' * 60}",
        file=sys.stderr,
    )
    if args.candidates and Path(args.candidates).exists():
        with open(args.candidates) as f:
            candidates = json.load(f)
        print(f"  Loaded {len(candidates)} from {args.candidates}", file=sys.stderr)
    else:
        candidates = SEED_CANDIDATES
        print(f"  Using built-in seed set ({len(candidates)} candidates)", file=sys.stderr)

    print(
        f"\n{'=' * 60}\n  Loading model: {args.model}\n{'=' * 60}\n",
        file=sys.stderr,
    )
    from ci_bench.models.mlx_model import MLXModel
    model = MLXModel(args.model)

    print(
        f"\n{'=' * 60}\n  Screening {len(candidates)} candidates "
        f"(n_runs={args.n_runs}, diversity >= {args.min_diversity}, "
        f"accuracy [{args.low:.0%}, {args.high:.0%}])"
        f"\n{'=' * 60}\n",
        file=sys.stderr,
    )
    passed = screen_d1(
        model, candidates,
        n_runs=args.n_runs,
        min_diversity=args.min_diversity,
        low_accuracy=args.low,
        high_accuracy=args.high,
    )

    dataset = build_d1_dataset(passed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    print(
        f"\n  Saved {len(dataset)} D1 questions to {output_path}",
        file=sys.stderr,
    )
    print(f"\n  Summary: {dataset.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
