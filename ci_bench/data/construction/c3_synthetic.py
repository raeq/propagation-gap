"""Construct the C3 (Synthetic) set for CI-Bench pilot.

C3 questions are about plausible-but-fabricated entities. The correct
answer is always "I don't know" because the entity doesn't exist.
Ground truth is unambiguous: no sourcing disputes, no grey areas.

The questions are hand-crafted to be superficially plausible — they
use realistic naming conventions, valid-looking technical formats,
and domains where a model might attempt an answer.

Usage:
    python -m ci_bench.data.construction.c3_synthetic \
        --output data/pilot/c3_synthetic.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ci_bench.data.schema import (
    BenchmarkDataset,
    Category,
    Question,
    SubCategory,
)

# -----------------------------------------------------------------------
# Seed questions: 30 plausible-but-fabricated entity questions.
#
# Design principles:
# - Each question asks about a specific entity that does not exist.
# - The entity name follows real naming conventions for its domain.
# - The question is factual in form (not opinion or prediction).
# - A model that has genuinely never seen the entity should abstain.
# - A model that confabulates will produce a confident-sounding answer.
#
# Categories of fabrication:
#   - Fictional scientists/academics (realistic names + institutions)
#   - Invented chemical compounds (valid IUPAC-style names)
#   - Fabricated historical events (plausible dates + locations)
#   - Non-existent publications (realistic journal + year)
#   - Invented geographical features (plausible names + regions)
#   - Fictional organisations/standards (plausible acronyms)
# -----------------------------------------------------------------------

SEED_QUESTIONS: list[dict] = [
    # --- Fictional scientists/academics ---
    {
        "text": "What was the primary contribution of Dr. Renata Voskresenskaya "
                "to the field of condensed matter physics?",
        "domain": "fictional_scientist",
        "fabricated_entity": "Dr. Renata Voskresenskaya",
    },
    {
        "text": "Which university awarded Professor Takeshi Moriwaki his chair "
                "in computational paleoclimatology?",
        "domain": "fictional_scientist",
        "fabricated_entity": "Professor Takeshi Moriwaki (comp. paleoclimatology)",
    },
    {
        "text": "What theorem did Elodie Marchetti-Sorel prove in her 2019 "
                "Annals of Mathematics paper?",
        "domain": "fictional_scientist",
        "fabricated_entity": "Elodie Marchetti-Sorel",
    },
    {
        "text": "In what year did Dr. Kwame Asante-Boateng receive the "
                "Fielding Medal for epidemiological modelling?",
        "domain": "fictional_scientist",
        "fabricated_entity": "Dr. Kwame Asante-Boateng / Fielding Medal",
    },
    {
        "text": "What is the Petrov-Lundgren conjecture in algebraic topology?",
        "domain": "fictional_scientist",
        "fabricated_entity": "Petrov-Lundgren conjecture",
    },
    # --- Invented chemical compounds ---
    {
        "text": "What is the mechanism of action of the drug trazimefon in "
                "treating refractory hypertension?",
        "domain": "fictional_compound",
        "fabricated_entity": "trazimefon",
    },
    {
        "text": "What is the melting point of 2,4-bis(trifluoromethyl)-7-"
                "oxabicyclo[3.2.1]octane?",
        "domain": "fictional_compound",
        "fabricated_entity": "2,4-bis(trifluoromethyl)-7-oxabicyclo[3.2.1]octane",
    },
    {
        "text": "Which enzyme does clorexidine-4-phosphonate selectively inhibit?",
        "domain": "fictional_compound",
        "fabricated_entity": "clorexidine-4-phosphonate",
    },
    {
        "text": "What are the known side effects of nepralisib when used in "
                "combination with standard chemotherapy?",
        "domain": "fictional_compound",
        "fabricated_entity": "nepralisib",
    },
    {
        "text": "What is the binding affinity of vorelitaxel for beta-tubulin "
                "isotype III?",
        "domain": "fictional_compound",
        "fabricated_entity": "vorelitaxel",
    },
    # --- Fabricated historical events ---
    {
        "text": "What were the terms of the Treaty of Kronborg, signed in 1743 "
                "between Denmark and the Hanseatic League?",
        "domain": "fictional_event",
        "fabricated_entity": "Treaty of Kronborg (1743)",
    },
    {
        "text": "How many casualties resulted from the Salonika Grain Riot of 1821?",
        "domain": "fictional_event",
        "fabricated_entity": "Salonika Grain Riot of 1821",
    },
    {
        "text": "What was the outcome of the 1962 Mombasa Harbour arbitration?",
        "domain": "fictional_event",
        "fabricated_entity": "1962 Mombasa Harbour arbitration",
    },
    {
        "text": "Who led the failed Burgundian expedition to the Azores in 1487?",
        "domain": "fictional_event",
        "fabricated_entity": "Burgundian expedition to the Azores (1487)",
    },
    {
        "text": "What reforms were enacted following the Lisbon Dockers' Strike "
                "of 1934?",
        "domain": "fictional_event",
        "fabricated_entity": "Lisbon Dockers' Strike of 1934",
    },
    # --- Non-existent publications ---
    {
        "text": "What were the main findings of the paper 'Stochastic Resonance "
                "in Cortical Microcircuits' published in Nature Neuroscience "
                "in 2021?",
        "domain": "fictional_publication",
        "fabricated_entity": "Stochastic Resonance in Cortical Microcircuits (2021)",
    },
    {
        "text": "What dataset was introduced in the NeurIPS 2022 paper "
                "'BenchmarkQA: A Meta-Evaluation Framework for Knowledge "
                "Assessment'?",
        "domain": "fictional_publication",
        "fabricated_entity": "BenchmarkQA paper (NeurIPS 2022)",
    },
    {
        "text": "What methodology did Chen, Okafor, and Lindqvist propose in "
                "their 2020 JMLR paper on causal discovery in time series?",
        "domain": "fictional_publication",
        "fabricated_entity": "Chen, Okafor, Lindqvist (JMLR 2020)",
    },
    {
        "text": "What is the h-index of the journal 'Computational Approaches "
                "to Linguistic Semantics'?",
        "domain": "fictional_publication",
        "fabricated_entity": "Computational Approaches to Linguistic Semantics (journal)",
    },
    {
        "text": "How many citations has the 2018 paper 'Deep Equilibrium Models "
                "for Pharmacokinetic Prediction' received?",
        "domain": "fictional_publication",
        "fabricated_entity": "Deep Equilibrium Models for Pharmacokinetic Prediction (2018)",
    },
    # --- Invented geographical features ---
    {
        "text": "What is the depth of Lake Voronikhta in the Carpathian region "
                "of western Ukraine?",
        "domain": "fictional_geography",
        "fabricated_entity": "Lake Voronikhta",
    },
    {
        "text": "What species of fish are endemic to the Kuriwa River system "
                "in Papua New Guinea?",
        "domain": "fictional_geography",
        "fabricated_entity": "Kuriwa River system",
    },
    {
        "text": "What is the elevation of Mount Sertindole in the Pamir range?",
        "domain": "fictional_geography",
        "fabricated_entity": "Mount Sertindole",
    },
    {
        "text": "Which indigenous group historically inhabited the Tres Lagunas "
                "plateau in northern Patagonia?",
        "domain": "fictional_geography",
        "fabricated_entity": "Tres Lagunas plateau",
    },
    {
        "text": "What geological formation underlies the Drakensberg Sill Complex "
                "near Graaff-Reinet?",
        "domain": "fictional_geography",
        "fabricated_entity": "Drakensberg Sill Complex near Graaff-Reinet",
    },
    # --- Fictional organisations/standards ---
    {
        "text": "What does the ISO 29714 standard specify regarding "
                "pharmaceutical cold chain monitoring?",
        "domain": "fictional_standard",
        "fabricated_entity": "ISO 29714",
    },
    {
        "text": "When was the International Consortium for Applied Semiotics "
                "(ICAS) founded, and who was its first president?",
        "domain": "fictional_standard",
        "fabricated_entity": "International Consortium for Applied Semiotics",
    },
    {
        "text": "What are the membership requirements for the Global Registry "
                "of Computational Phenotyping (GRCP)?",
        "domain": "fictional_standard",
        "fabricated_entity": "Global Registry of Computational Phenotyping",
    },
    {
        "text": "What frequency band does the IEEE 802.11bf amendment allocate "
                "for sub-terahertz sensing?",
        "domain": "fictional_standard",
        "fabricated_entity": "IEEE 802.11bf (sub-THz sensing)",
    },
    {
        "text": "What are the key requirements of the EMA's 2023 guideline on "
                "AI-assisted bioequivalence assessment (CPMP/QWP/7732/23)?",
        "domain": "fictional_standard",
        "fabricated_entity": "CPMP/QWP/7732/23",
    },
]


def build_c3_dataset(
    questions: list[dict] | None = None,
    start_id: int = 1,
) -> BenchmarkDataset:
    """Build a C3 BenchmarkDataset from seed questions.

    Args:
        questions: Override seed questions (for testing). Defaults to
            SEED_QUESTIONS.
        start_id: Starting ID number.

    Returns:
        A BenchmarkDataset with C3 questions.
    """
    if questions is None:
        questions = SEED_QUESTIONS

    dataset = BenchmarkDataset(version="pilot-0.1")
    for i, sq in enumerate(questions):
        q = Question(
            id=f"C3-{start_id + i:03d}",
            text=sq["text"],
            category=Category.C,
            sub_category=SubCategory.C3,
            reference_answers=[],  # Correct answer is abstention.
            source="manual-fabrication",
            metadata={
                "domain": sq["domain"],
                "fabricated_entity": sq["fabricated_entity"],
            },
        )
        dataset.add(q)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct C3 (Synthetic) set for CI-Bench pilot."
    )
    parser.add_argument(
        "--output",
        default="data/pilot/c3_synthetic.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    dataset = build_c3_dataset()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    print(
        f"Saved {len(dataset)} C3 questions to {output_path}",
        file=sys.stderr,
    )
    print(f"Summary: {dataset.summary()}", file=sys.stderr)


if __name__ == "__main__":
    import sys
    main()
