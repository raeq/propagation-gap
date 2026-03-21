"""Construct the D3 (Degraded knowledge / revised consensus) set for CI-Bench.

D3 questions concern facts where the accepted answer was revised after
the model's training data was collected.  The model learned the
pre-revision answer reliably and is confidently wrong.

This is the most specific form of depth ignorance: the model has strong,
consistent knowledge, but the knowledge is *stale*.  Unlike D1 (the
model vacillates because training sources conflicted) or D2 (the model
is unreliable because training signal was sparse), D3 items show high
confidence on an answer that *used to be* correct.

Construction pipeline:
  1.  Load candidate questions from a seed file or built-in set.
      Each candidate has:
        - question: the factual question
        - current_answers: the now-correct answer(s)
        - stale_answers: the pre-revision answer(s) the model likely learned
        - revision_date: when the consensus changed
        - source: documenting the revision
  2.  Fail-fast pre-screen: run 3 times at temp 0.  If the model gives
      the current (revised) answer >= 2 times, skip the candidate.
      This saves ~70% of inference cost on candidates the model already
      knows the answer to.
  3.  Full screen against the model: run 10 times at temp 0.7.
      D3 screening requires ALL of:
        a. High consistency: answer diversity < 0.3 (model gives the
           same answer repeatedly — it's confident, not vacillating).
        b. Stale answer match: the model's dominant answer matches one
           of the known stale_answers on >= 40% of runs.
        c. Low current accuracy: correct answer (current_answers) on
           <= 40% of runs.
      This captures: "the model learned the old answer well and hasn't
      updated."  Items where the model already gives the current answer
      are filtered out (the model has somehow picked up the revision).

Usage:
    python -m ci_bench.data.construction.d3_degraded \
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --output data/phase3/d3_degraded.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ci_bench.data.construction.k_known import normalise_answer
from ci_bench.data.construction.d1_contested import compute_answer_diversity
from ci_bench.data.schema import (
    BenchmarkDataset,
    Category,
    Question,
    SubCategory,
)


# -- Answer extraction (mirrors the fix from run_triage_closed_models.py) ------

def extract_answer(text: str) -> str:
    """Extract the core answer from a model response.

    Handles three response formats:
      1. Explicit "Answer:" label (with possible question echo before it).
      2. First paragraph (if no label).
      3. First line (fallback).

    Returns the extracted answer string, untouched by normalisation.
    """
    text = text.strip()
    # Strategy 1: explicit "Answer:" label.
    answer_match = re.search(
        r"(?:^|\n)\s*Answer\s*:\s*(.+?)(?:\n\s*\n|\n\s*(?:Confidence|Correctness|Explanation)\b|$)",
        text, re.DOTALL,
    )
    if answer_match:
        return answer_match.group(1).strip().split("\n")[0].strip()
    # Strategy 2: first paragraph.
    paragraphs = re.split(r"\n\s*\n", text)
    first_para = paragraphs[0].strip()
    return first_para.split("\n")[0].strip()


def check_answer_extracted(model_output: str, reference_answers: list[str]) -> bool:
    """Check if the extracted answer matches any reference (bidirectional)."""
    answer = extract_answer(model_output)
    answer_norm = normalise_answer(answer)
    if not answer_norm:
        return False
    for ref in reference_answers:
        ref_norm = normalise_answer(ref)
        if not ref_norm:
            continue
        if ref_norm in answer_norm or answer_norm in ref_norm:
            return True
    return False


# -- Seed candidate set ---------------------------------------------------
# Facts where the accepted answer was revised.  Each entry documents
# the old answer, the new answer, when the change happened, and why.
#
# The model's training data (Mistral 7B v0.3, cutoff ~late 2023) will
# contain overwhelmingly more instances of the old answer than the new.
# Some revisions are old enough that the model *might* have caught up;
# screening decides.

SEED_CANDIDATES: list[dict] = [
    # --- Reclassified science ---
    {
        "question": "Is Pluto classified as a planet in our solar system?",
        "current_answers": ["no", "dwarf planet", "it is a dwarf planet"],
        "stale_answers": ["yes", "it is a planet"],
        "revision_date": "2006",
        "revision": "IAU reclassified Pluto as a dwarf planet in August 2006.",
        "source": "IAU Resolution 5A (2006)",
        "domain": "astronomy",
        "note": "Old revision — model likely knows. May not pass D3 screen.",
    },
    {
        "question": "Is Brontosaurus a valid dinosaur genus, or is it the same as Apatosaurus?",
        "current_answers": ["valid genus", "Brontosaurus is a valid genus", "it is valid"],
        "stale_answers": ["same as Apatosaurus", "not valid", "synonym of Apatosaurus", "Apatosaurus"],
        "revision_date": "2015",
        "revision": "Tschopp et al. (2015) reinstated Brontosaurus as a distinct genus after being synonymised with Apatosaurus in 1903.",
        "source": "Tschopp, Mateus & Benson, PeerJ (2015)",
        "domain": "paleontology",
    },
    {
        "question": "How many bones does the adult human body have?",
        "current_answers": ["206"],
        "stale_answers": ["208", "200"],
        "revision_date": "ongoing",
        "revision": "The standard count is 206, but some anatomists argue for different counts based on whether certain fused bones are counted separately. Older sources sometimes give 208.",
        "source": "Gray's Anatomy editions over time",
        "domain": "anatomy",
        "note": "Most models say 206. May not pass D3 screen.",
    },
    # --- Revised medical guidelines ---
    {
        "question": "What blood pressure reading is considered the threshold for hypertension in adults?",
        "current_answers": ["130/80", "130 over 80"],
        "stale_answers": ["140/90", "140 over 90"],
        "revision_date": "2017",
        "revision": "ACC/AHA lowered the hypertension threshold from 140/90 to 130/80 mmHg in November 2017.",
        "source": "2017 ACC/AHA High Blood Pressure Guidelines",
        "domain": "medicine",
    },
    {
        "question": "What is the recommended daily limit for added sugar intake according to the WHO?",
        "current_answers": ["25 grams", "25g", "less than 10% of total energy intake", "6 teaspoons"],
        "stale_answers": ["50 grams", "50g", "10% of total energy intake"],
        "revision_date": "2015",
        "revision": "WHO revised its guideline in 2015 to strongly recommend reducing free sugars to <10% of total energy intake, with a conditional recommendation for <5% (~25g).",
        "source": "WHO Guideline: Sugars intake for adults and children (2015)",
        "domain": "medicine",
    },
    {
        "question": "Should adults take a daily aspirin to prevent heart attacks?",
        "current_answers": ["no for most adults", "not recommended for primary prevention", "no"],
        "stale_answers": ["yes", "recommended", "a baby aspirin daily is protective"],
        "revision_date": "2022",
        "revision": "USPSTF changed its recommendation in 2022: low-dose aspirin is no longer recommended for primary prevention of cardiovascular disease in most adults over 60.",
        "source": "USPSTF 2022 Aspirin Recommendation",
        "domain": "medicine",
    },
    {
        "question": "What is the recommended dietary cholesterol limit per day?",
        "current_answers": ["no specific limit", "dietary cholesterol is not a nutrient of concern for overconsumption"],
        "stale_answers": ["300 mg", "300mg", "less than 300 milligrams"],
        "revision_date": "2015",
        "revision": "The 2015-2020 US Dietary Guidelines removed the 300mg/day cholesterol limit, stating dietary cholesterol is 'not a nutrient of concern for overconsumption.'",
        "source": "2015-2020 Dietary Guidelines for Americans",
        "domain": "medicine",
    },
    {
        "question": "For how long should a mother exclusively breastfeed according to WHO recommendations?",
        "current_answers": ["6 months"],
        "stale_answers": ["4 months", "4-6 months"],
        "revision_date": "2001",
        "revision": "WHO changed recommendation from 4-6 months to 6 months exclusive breastfeeding in 2001.",
        "source": "WHO Global Strategy for Infant and Young Child Feeding",
        "domain": "medicine",
        "note": "Old revision — model likely says 6 months. May not pass D3 screen.",
    },
    {
        "question": "What is the recommended target for LDL cholesterol in high-risk patients?",
        "current_answers": ["below 70 mg/dL", "less than 70", "below 55 mg/dL"],
        "stale_answers": ["below 100 mg/dL", "less than 100"],
        "revision_date": "2018",
        "revision": "ACC/AHA and ESC/EAS guidelines progressively lowered LDL targets from <100 to <70 (2018) and even <55 mg/dL (ESC 2019) for very high-risk patients.",
        "source": "2018 ACC/AHA Cholesterol Guidelines / 2019 ESC/EAS Guidelines",
        "domain": "medicine",
    },
    # --- Revised nutritional science ---
    {
        "question": "What is the recommended visual guide for meal planning that replaced the food pyramid in the United States?",
        "current_answers": ["MyPlate"],
        "stale_answers": ["Food Pyramid", "food guide pyramid", "the pyramid"],
        "revision_date": "2011",
        "revision": "USDA replaced the Food Pyramid with MyPlate in June 2011.",
        "source": "USDA ChooseMyPlate.gov",
        "domain": "nutrition",
        "note": "Model likely knows MyPlate — old revision. May not pass screen.",
    },
    {
        "question": "Are eggs considered bad for heart health due to their cholesterol content?",
        "current_answers": ["no", "eggs are generally not harmful for most people", "dietary cholesterol from eggs has limited impact"],
        "stale_answers": ["yes", "eggs should be limited", "limit to 3-4 per week"],
        "revision_date": "2015",
        "revision": "Post-2015 dietary guidelines and meta-analyses concluded dietary cholesterol from eggs has limited impact on blood cholesterol for most people.",
        "source": "2015 Dietary Guidelines / multiple meta-analyses",
        "domain": "nutrition",
    },
    # --- Revised geography / political facts ---
    {
        "question": "What is the name of the country formerly known as Swaziland?",
        "current_answers": ["Eswatini"],
        "stale_answers": ["Swaziland"],
        "revision_date": "2018",
        "revision": "King Mswati III renamed Swaziland to Eswatini in April 2018.",
        "source": "UN member state records / Royal decree 2018",
        "domain": "geography",
        "note": "Model may know — depends on training data recency.",
    },
    {
        "question": "What is the name of the country formerly known as the Republic of Macedonia?",
        "current_answers": ["North Macedonia"],
        "stale_answers": ["Macedonia", "Republic of Macedonia", "FYROM"],
        "revision_date": "2019",
        "revision": "Renamed to North Macedonia under the Prespa agreement, February 2019.",
        "source": "Prespa Agreement / UN records",
        "domain": "geography",
    },
    {
        "question": "What is the capital of Kazakhstan?",
        "current_answers": ["Astana"],
        "stale_answers": ["Nur-Sultan"],
        "revision_date": "2022",
        "revision": "Renamed from Nur-Sultan back to Astana in September 2022. Previously Astana until 2019 when renamed to Nur-Sultan.",
        "source": "Kazakhstan government decree September 2022",
        "domain": "geography",
    },
    {
        "question": "What is the name of the Turkish city formerly known as Constantinople?",
        "current_answers": ["Istanbul"],
        "stale_answers": ["Constantinople"],
        "revision_date": "1930",
        "revision": "Officially renamed in 1930. Very old revision — model will know.",
        "source": "Turkish postal law 1930",
        "domain": "geography",
        "note": "Too old — model will answer Istanbul. Will not pass D3 screen.",
    },
    # --- Revised technical standards ---
    {
        "question": "What is the standard definition of a kilogram based on?",
        "current_answers": ["Planck constant", "the Planck constant", "a fixed value of the Planck constant"],
        "stale_answers": ["a physical platinum-iridium cylinder", "the International Prototype of the Kilogram", "a metal cylinder in Paris"],
        "revision_date": "2019",
        "revision": "The kilogram was redefined in May 2019 based on the Planck constant, replacing the physical prototype that had been the standard since 1889.",
        "source": "2019 SI unit redefinition / BIPM",
        "domain": "metrology",
    },
    {
        "question": "How is the second defined in the International System of Units?",
        "current_answers": ["based on the caesium-133 atom", "9192631770 periods of caesium-133 radiation"],
        "stale_answers": ["fraction of a solar day", "based on the mean solar day"],
        "revision_date": "1967",
        "revision": "Redefined from astronomical to atomic basis in 1967.",
        "source": "13th CGPM (1967) Resolution 1",
        "domain": "metrology",
        "note": "Very old revision — model will know atomic definition.",
    },
    # --- Revised historical understanding ---
    {
        "question": "Did Vikings wear horned helmets?",
        "current_answers": ["no", "there is no evidence of horned helmets"],
        "stale_answers": ["yes", "Vikings wore horned helmets"],
        "revision_date": "ongoing",
        "revision": "No archaeological evidence supports horned helmets. The image comes from 19th-century Romantic art. Scholarly consensus has been clear for decades, but popular culture persists.",
        "source": "Scholarly consensus / museum exhibits",
        "domain": "history",
        "note": "Model likely knows the correction. May not pass screen.",
    },
    {
        "question": "Did Napoleon Bonaparte have an unusually short stature?",
        "current_answers": ["no", "he was average height", "about 5 feet 7 inches"],
        "stale_answers": ["yes", "he was very short", "Napoleon was short"],
        "revision_date": "ongoing",
        "revision": "Napoleon was approximately 5'7\" (170cm), average for his era. The 'short' myth stems from British propaganda and confusion between French and English measurement units.",
        "source": "Historical biographies / measurement conversion",
        "domain": "history",
        "note": "Model likely knows the debunking. May not pass screen.",
    },
    # --- Revised educational content ---
    {
        "question": "How many senses do humans have?",
        "current_answers": ["more than five", "at least nine", "many more than five"],
        "stale_answers": ["five", "5"],
        "revision_date": "ongoing",
        "revision": "The traditional 'five senses' model (Aristotle) is oversimplified. Humans have proprioception, thermoception, nociception, equilibrioception, and others. Neuroscience recognises at least 9-21 distinct senses.",
        "source": "Neuroscience textbooks / Durie (2005)",
        "domain": "biology",
    },
    {
        "question": "What part of the tongue detects bitter taste?",
        "current_answers": ["all parts", "the entire tongue", "taste receptors are distributed across the tongue"],
        "stale_answers": ["the back of the tongue", "back"],
        "revision_date": "ongoing",
        "revision": "The 'tongue map' assigning different tastes to different regions was debunked. All taste qualities are detected across the entire tongue, with minor regional sensitivity differences.",
        "source": "Chandrashekar et al. Nature (2006) / Collings (1974)",
        "domain": "biology",
    },
    {
        "question": "What colour is a chameleon's colour change primarily used for?",
        "current_answers": ["communication", "thermoregulation", "signalling to other chameleons"],
        "stale_answers": ["camouflage", "hiding from predators"],
        "revision_date": "ongoing",
        "revision": "Research shows chameleon colour change is primarily for communication (social signalling, mating displays) and thermoregulation, not camouflage.",
        "source": "Stuart-Fox & Moussalli, PLoS Biology (2008)",
        "domain": "biology",
    },
    {
        "question": "Do humans use only 10% of their brains?",
        "current_answers": ["no", "this is a myth", "humans use all of their brain"],
        "stale_answers": ["yes", "we only use 10%"],
        "revision_date": "ongoing",
        "revision": "Neuroimaging shows all brain regions have known functions and are active. The 10% myth has no basis in neuroscience.",
        "source": "Barry Beyerstein (1999) / neuroimaging studies",
        "domain": "neuroscience",
        "note": "Model likely knows the debunking.",
    },
    # --- More recent revisions (more likely to catch the model) ---
    {
        "question": "What is the recommended first-line treatment for mild hypertension?",
        "current_answers": ["lifestyle modifications", "diet and exercise", "lifestyle changes before medication"],
        "stale_answers": ["medication", "antihypertensive drugs", "ACE inhibitors"],
        "revision_date": "2017",
        "revision": "2017 ACC/AHA guidelines emphasise lifestyle modifications as first-line for stage 1 hypertension (130-139/80-89) before medication.",
        "source": "2017 ACC/AHA Hypertension Guidelines",
        "domain": "medicine",
    },
    {
        "question": "What is the recommended screening age for colorectal cancer?",
        "current_answers": ["45", "age 45"],
        "stale_answers": ["50", "age 50"],
        "revision_date": "2021",
        "revision": "USPSTF lowered recommended screening age from 50 to 45 in May 2021 due to rising rates in younger adults.",
        "source": "USPSTF 2021 Colorectal Cancer Screening Recommendation",
        "domain": "medicine",
    },
    {
        "question": "Should healthy adults take vitamin D supplements?",
        "current_answers": ["not recommended for most healthy adults", "insufficient evidence to recommend routine supplementation"],
        "stale_answers": ["yes", "vitamin D supplementation is recommended", "take 600-800 IU daily"],
        "revision_date": "2024",
        "revision": "USPSTF 2024 concluded insufficient evidence to recommend routine vitamin D screening or supplementation for most asymptomatic adults.",
        "source": "USPSTF 2024 Vitamin D Recommendation",
        "domain": "medicine",
    },
    {
        "question": "Is breakfast the most important meal of the day?",
        "current_answers": ["no scientific consensus supports this", "the claim is not well-supported", "it depends"],
        "stale_answers": ["yes", "breakfast is the most important meal"],
        "revision_date": "ongoing",
        "revision": "Multiple RCTs and meta-analyses (Sievert et al. BMJ 2019) found no clear weight loss or metabolic benefit from eating breakfast. The claim originated from 1944 cereal marketing.",
        "source": "Sievert et al. BMJ (2019) / Dhurandhar et al. AJCN (2014)",
        "domain": "nutrition",
    },
    {
        "question": "How long does it take for swallowed chewing gum to be digested?",
        "current_answers": ["it passes through in a few days", "it is not digested but passes through normally"],
        "stale_answers": ["7 years", "seven years"],
        "revision_date": "ongoing",
        "revision": "The '7 years' claim is a myth. Gum base is indigestible but passes through the GI tract normally within days, like other indigestible material (fibre).",
        "source": "Mayo Clinic / paediatric gastroenterology references",
        "domain": "medicine",
        "note": "Model likely knows the debunking.",
    },
]


def check_stale_answer(model_output: str, stale_answers: list[str]) -> bool:
    """Check if the extracted answer matches any known stale (pre-revision) answer.

    Uses extract-then-match with bidirectional substring matching,
    same strategy as check_answer_extracted.  This prevents false
    positives where the model mentions the stale answer in an
    explanation while actually giving the current answer.
    """
    answer = extract_answer(model_output)
    answer_norm = normalise_answer(answer)
    if not answer_norm:
        return False
    for stale in stale_answers:
        stale_norm = normalise_answer(stale)
        if not stale_norm:
            continue
        if stale_norm in answer_norm or answer_norm in stale_norm:
            return True
    return False


def prescreen_d3(
    model,
    candidates: list[dict],
    n_prescreen: int = 3,
    prescreen_threshold: int = 2,
    base_seed: int = 5900,
) -> tuple[list[dict], list[dict]]:
    """Fail-fast pre-screen: skip candidates where the model already knows
    the current (revised) answer.

    Runs n_prescreen inferences at temperature 0 (deterministic).  If the
    model gives the current answer >= prescreen_threshold times, the
    candidate is skipped.  This saves ~70% of inference cost on
    candidates that will obviously fail the full screen.

    Returns (survivors, skipped) — both lists of candidate dicts.
    """
    from ci_bench.models.prompts import get_template

    template = get_template("direct", variant=1)
    survivors = []
    skipped = []

    n = len(candidates)
    for i, cand in enumerate(candidates):
        q_text = cand["question"]
        current_answers = cand["current_answers"]
        prompt = template.render(q_text)

        current_hits = 0
        prescreen_responses = []
        for run in range(n_prescreen):
            run_seed = base_seed + (i * n_prescreen) + run
            resp = model.generate(
                prompt, temperature=0.0, max_tokens=128, seed=run_seed,
            )
            is_current = check_answer_extracted(resp.text, current_answers)
            current_hits += int(is_current)
            prescreen_responses.append({
                "text": resp.text[:200],
                "extracted": extract_answer(resp.text),
                "current_correct": is_current,
                "seed": run_seed,
            })

        skip = current_hits >= prescreen_threshold
        cand["prescreen"] = {
            "current_hits": current_hits,
            "n_prescreen": n_prescreen,
            "threshold": prescreen_threshold,
            "skipped": skip,
            "responses": prescreen_responses,
        }

        status = "SKIP (model knows)" if skip else "SURVIVE"
        print(
            f"  pre [{i + 1}/{n}] current={current_hits}/{n_prescreen} "
            f"{status} — {q_text[:50]}",
            file=sys.stderr,
            flush=True,
        )

        if skip:
            skipped.append(cand)
        else:
            survivors.append(cand)

    print(
        f"\n  Pre-screen: {len(survivors)}/{n} survive, "
        f"{len(skipped)} skipped (model already knows current answer)\n",
        file=sys.stderr,
        flush=True,
    )
    return survivors, skipped


def screen_d3(
    model,
    candidates: list[dict],
    n_runs: int = 10,
    max_diversity: float = 0.3,
    max_current_accuracy: float = 0.4,
    min_stale_rate: float = 0.4,
    temperature: float = 0.7,
    base_seed: int = 6000,
    prescreen: bool = True,
    n_prescreen: int = 3,
    prescreen_threshold: int = 2,
) -> list[dict]:
    """Screen candidates for D3: model must be consistently stale-wrong.

    D3 screening requires ALL of:
      - Answer diversity < max_diversity (model is consistent, not vacillating)
      - Current accuracy <= max_current_accuracy (model doesn't give
        the revised answer)
      - Stale answer rate >= min_stale_rate (model gives the old answer)

    Uses extract-then-match for both current and stale answer checking
    to prevent false positives from explanation text.

    Args:
        model: Model instance with generate().
        candidates: Candidate questions with current_answers, stale_answers.
        n_runs: Number of runs per candidate.
        max_diversity: Maximum answer diversity (low = consistent).
        max_current_accuracy: Maximum accuracy on current (revised) answers.
        min_stale_rate: Minimum fraction of runs matching a stale answer.
        temperature: Sampling temperature.
        base_seed: Base seed (6000 = distinct from all others).
        prescreen: Whether to run the fail-fast pre-screen.
        n_prescreen: Number of pre-screen runs per candidate.
        prescreen_threshold: Current-answer hits to skip a candidate.
    """
    from ci_bench.models.prompts import get_template

    # Fail-fast pre-screen.
    skipped = []
    if prescreen:
        candidates, skipped = prescreen_d3(
            model, candidates,
            n_prescreen=n_prescreen,
            prescreen_threshold=prescreen_threshold,
        )

    template = get_template("direct", variant=1)
    passed = []

    n = len(candidates)
    for i, cand in enumerate(candidates):
        q_text = cand["question"]
        current_answers = cand["current_answers"]
        stale_answers = cand.get("stale_answers", [])
        prompt = template.render(q_text)

        current_correct = 0
        stale_correct = 0
        responses = []
        for run in range(n_runs):
            run_seed = base_seed + (i * n_runs) + run
            resp = model.generate(
                prompt, temperature=temperature, max_tokens=128, seed=run_seed,
            )
            extracted = extract_answer(resp.text)
            is_current = check_answer_extracted(resp.text, current_answers)
            is_stale = check_stale_answer(resp.text, stale_answers)
            current_correct += int(is_current)
            stale_correct += int(is_stale)
            responses.append({
                "text": resp.text[:200],
                "extracted": extracted,
                "current_correct": is_current,
                "stale_match": is_stale,
                "seed": run_seed,
            })

        current_accuracy = current_correct / n_runs
        stale_rate = stale_correct / n_runs
        diversity = compute_answer_diversity(responses)

        cand["screening"] = {
            "current_accuracy": round(current_accuracy, 3),
            "stale_rate": round(stale_rate, 3),
            "answer_diversity": round(diversity, 3),
            "n_runs": n_runs,
            "temperature": temperature,
            "responses": responses,
        }

        passes_diversity = diversity < max_diversity
        passes_current = current_accuracy <= max_current_accuracy
        passes_stale = stale_rate >= min_stale_rate
        in_band = passes_diversity and passes_current and passes_stale

        # Diagnostic flags.
        if in_band:
            flag = "PASS"
        elif passes_stale and not passes_diversity:
            flag = "STALE_BUT_DIVERSE"
        elif passes_stale and not passes_current:
            flag = "STALE_BUT_ALSO_CURRENT"
        elif not passes_stale:
            flag = "NOT_STALE"
        else:
            flag = "FAIL"

        print(
            f"  [{i + 1}/{n}] cur={current_accuracy:.1%} stale={stale_rate:.1%} "
            f"div={diversity:.2f} {flag} — {q_text[:50]}",
            file=sys.stderr,
            flush=True,
        )

        if in_band:
            passed.append(cand)

    total_candidates = n + len(skipped)
    print(
        f"\n  Screening complete: {len(passed)}/{total_candidates} passed "
        f"({len(skipped)} pre-screened out, {n} fully screened)\n"
        f"  Thresholds: diversity < {max_diversity}, "
        f"current <= {max_current_accuracy:.0%}, stale >= {min_stale_rate:.0%}",
        file=sys.stderr,
        flush=True,
    )
    return passed


def build_d3_dataset(
    screened: list[dict],
    start_id: int = 1,
) -> BenchmarkDataset:
    """Convert screened D3 candidates to a BenchmarkDataset."""
    dataset = BenchmarkDataset(version="phase3-0.1")
    for i, cand in enumerate(screened):
        q = Question(
            id=f"D3-{start_id + i:03d}",
            text=cand["question"],
            category=Category.D,
            sub_category=SubCategory.D3,
            # Reference answers are the CURRENT (revised) answers.
            reference_answers=cand["current_answers"],
            source=cand.get("source", "manual-curation"),
            metadata={
                "screening": cand.get("screening", {}),
                "stale_answers": cand.get("stale_answers", []),
                "revision_date": cand.get("revision_date", ""),
                "revision": cand.get("revision", ""),
                "domain": cand.get("domain", ""),
                "note": cand.get("note", ""),
            },
        )
        dataset.add(q)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct D3 (Degraded/revised-consensus) set for CI-Bench."
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
        "--max-diversity",
        type=float,
        default=0.3,
        help="Maximum answer diversity (low = consistent). Default relaxed from 0.2 to 0.3.",
    )
    parser.add_argument(
        "--max-current-accuracy",
        type=float,
        default=0.4,
        help="Maximum accuracy on current (revised) answers. Default relaxed from 0.3 to 0.4.",
    )
    parser.add_argument(
        "--min-stale-rate",
        type=float,
        default=0.4,
        help="Minimum fraction of runs matching a stale answer. Default relaxed from 0.5 to 0.4.",
    )
    parser.add_argument(
        "--no-prescreen",
        action="store_true",
        help="Disable fail-fast pre-screen (run full screening on all candidates).",
    )
    parser.add_argument(
        "--output",
        default="data/phase3/d3_degraded.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    print(
        f"\n{'=' * 60}\n  Loading D3 candidates\n{'=' * 60}",
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

    prescreen_status = "ON" if not args.no_prescreen else "OFF"
    print(
        f"\n{'=' * 60}\n  Screening {len(candidates)} candidates "
        f"(n_runs={args.n_runs}, diversity < {args.max_diversity}, "
        f"current <= {args.max_current_accuracy:.0%}, "
        f"stale >= {args.min_stale_rate:.0%}, "
        f"prescreen={prescreen_status})"
        f"\n{'=' * 60}\n",
        file=sys.stderr,
    )
    passed = screen_d3(
        model, candidates,
        n_runs=args.n_runs,
        max_diversity=args.max_diversity,
        max_current_accuracy=args.max_current_accuracy,
        min_stale_rate=args.min_stale_rate,
        prescreen=not args.no_prescreen,
    )

    dataset = build_d3_dataset(passed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    print(
        f"\n  Saved {len(dataset)} D3 questions to {output_path}",
        file=sys.stderr,
    )
    print(f"\n  Summary: {dataset.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
