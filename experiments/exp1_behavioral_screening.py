#!/usr/bin/env python3
"""Experiment 1: Behavioral Screening.

Paper claim: Uniform confidence — expressed confidence carries negligible
K-D information.

Loads V4 summaries (Tier 1) and phase4 summaries (Tier 2).
Computes all behavioral metrics from per-question data.
Outputs results/canonical/experiment_1.json.
"""

import sys
from pathlib import Path
from collections import Counter
from itertools import combinations

# Allow running from repo root or experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_all_summaries, extract_kd_labels, extract_kdc_labels,
    save_json, metadata_block, print_verification_header, print_check,
    ALL_MODELS, ALL_MODELS_ORDERED, TIER1_MODELS, CANONICAL,
)


def compute_model_behavioral(display_name, summary):
    """Compute all behavioral metrics for a single model."""
    cfg = ALL_MODELS[display_name]
    pq = summary["per_question"]
    cal = summary.get("calibration", {})

    # Label counts from model-relative labels
    label_counts = summary.get("model_label_counts", {})
    n_k = label_counts.get("K", 0)
    n_d = label_counts.get("D", 0)
    n_c = label_counts.get("C", 0)
    total = n_k + n_d + n_c

    d_rate = n_d / total if total > 0 else 0

    # Per-label calibration from summary (mean confidence, accuracy)
    calibration = {}
    for label in ("K", "D", "C"):
        if label in cal:
            c = cal[label]
            calibration[label] = {
                "mean_confidence": c.get("mean_confidence"),
                "mean_accuracy": c.get("accuracy_in_confidence_condition"),
                "n": c.get("n", 0),
                "n_with_confidence": c.get("n_with_confidence", 0),
            }
        else:
            calibration[label] = {
                "mean_confidence": None,
                "mean_accuracy": None,
                "n": 0,
                "n_with_confidence": 0,
            }

    # K-D confidence gap (percentage points)
    k_conf = calibration["K"]["mean_confidence"]
    d_conf = calibration["D"]["mean_confidence"]
    kd_confidence_gap_pp = None
    if k_conf is not None and d_conf is not None:
        kd_confidence_gap_pp = round((k_conf - d_conf) * 100, 1)

    # K-D accuracy gap
    k_acc = calibration["K"]["mean_accuracy"]
    d_acc = calibration["D"]["mean_accuracy"]
    kd_accuracy_gap_pp = None
    if k_acc is not None and d_acc is not None:
        kd_accuracy_gap_pp = round((k_acc - d_acc) * 100, 1)

    # D overconfidence (confidence - accuracy)
    d_overconfidence_pp = None
    if d_conf is not None and d_acc is not None:
        d_overconfidence_pp = round((d_conf - d_acc) * 100, 1)

    # Per-question confidence distributions for K and D
    k_confidences = []
    d_confidences = []
    for qid, q in pq.items():
        ml = q.get("model_label", q.get("predicted_label"))
        conf = q.get("confidence")
        if conf is not None:
            if ml == "K":
                k_confidences.append(conf)
            elif ml == "D":
                d_confidences.append(conf)

    return {
        "tier": cfg["tier"],
        "label_counts": {"K": n_k, "D": n_d, "C": n_c},
        "d_rate": round(d_rate, 3),
        "calibration": calibration,
        "kd_confidence_gap_pp": kd_confidence_gap_pp,
        "kd_accuracy_gap_pp": kd_accuracy_gap_pp,
        "d_overconfidence_pp": d_overconfidence_pp,
        "per_question_confidence": {
            "K": sorted(k_confidences),
            "D": sorted(d_confidences),
        },
    }


def compute_cross_model(summaries):
    """Compute cross-model D-set overlap and pairwise Jaccard indices."""
    # Collect D-labelled questions per model
    d_sets = {}
    for name in ALL_MODELS_ORDERED:
        pq = summaries[name]["per_question"]
        d_sets[name] = set()
        for qid, q in pq.items():
            ml = q.get("model_label", q.get("predicted_label"))
            if ml == "D":
                d_sets[name].add(qid)

    # D-overlap histogram: how many questions are D in exactly n models
    d_questions = {}  # qid -> count of models labelling it D
    for name, dset in d_sets.items():
        for qid in dset:
            d_questions[qid] = d_questions.get(qid, 0) + 1

    overlap_hist = Counter(d_questions.values())
    d_overlap_histogram = {str(n): overlap_hist.get(n, 0) for n in range(7)}

    # Also count questions that are D in 0 models (all 338 minus those in d_questions)
    total_questions = len(summaries[ALL_MODELS_ORDERED[0]]["per_question"])
    d_overlap_histogram["0"] = total_questions - len(d_questions)

    # Pairwise Jaccard for Tier 1 models
    tier1_names = [n for n in ALL_MODELS_ORDERED if ALL_MODELS[n]["tier"] == 1]
    pairwise_jaccard = {}
    for a, b in combinations(tier1_names, 2):
        overlap = len(d_sets[a] & d_sets[b])
        union = len(d_sets[a] | d_sets[b])
        jaccard = overlap / union if union > 0 else 0
        pair_key = f"{a.split()[0]}-{b.split()[0]}"
        pairwise_jaccard[pair_key] = {
            "overlap": overlap,
            "union": union,
            "jaccard": round(jaccard, 3),
        }

    # Count of models with K-D confidence gap < 3 pp
    n_gap_lt_3pp = 0
    for name in ALL_MODELS_ORDERED:
        cal = summaries[name].get("calibration", {})
        k_conf = cal.get("K", {}).get("mean_confidence")
        d_conf = cal.get("D", {}).get("mean_confidence")
        if k_conf is not None and d_conf is not None:
            gap_pp = abs(k_conf - d_conf) * 100
            if gap_pp < 3:
                n_gap_lt_3pp += 1

    return {
        "n_models_gap_lt_3pp": n_gap_lt_3pp,
        "d_overlap_histogram": d_overlap_histogram,
        "pairwise_jaccard": pairwise_jaccard,
    }


def verify(results):
    """Print verification table comparing computed values against manuscript claims."""
    print_verification_header("Experiment 1: Behavioral Screening")

    # Manuscript claims to verify
    print("\n  --- Per-model calibration ---")
    for name in ALL_MODELS_ORDERED:
        m = results["models"][name]
        gap = m["kd_confidence_gap_pp"]
        acc_gap = m["kd_accuracy_gap_pp"]
        overconf = m["d_overconfidence_pp"]
        n_d = m["label_counts"]["D"]
        gap_s = f"{gap:5.1f}" if gap is not None else "  N/A"
        acc_s = f"{acc_gap}" if acc_gap is not None else "N/A"
        oc_s = f"{overconf}" if overconf is not None else "N/A"
        print(f"  {name:20s}  N_D={n_d:3d}  "
              f"K-D conf gap={gap_s}pp  "
              f"K-D acc gap={acc_s}pp  "
              f"D overconf={oc_s}pp")

    print(f"\n  Models with K-D confidence gap < 3pp: "
          f"{results['cross_model']['n_models_gap_lt_3pp']}")

    print("\n  --- D-set overlap histogram ---")
    hist = results["cross_model"]["d_overlap_histogram"]
    for n in range(7):
        print(f"    D in {n} models: {hist[str(n)]}")

    print("\n  --- Pairwise Jaccard (Tier 1) ---")
    for pair, data in results["cross_model"]["pairwise_jaccard"].items():
        print(f"    {pair}: overlap={data['overlap']}, "
              f"union={data['union']}, Jaccard={data['jaccard']:.3f}")


def main():
    print("Experiment 1: Behavioral Screening")
    print("-" * 40)

    # Load data
    print("Loading summaries...")
    summaries = load_all_summaries()

    # Compute per-model metrics
    print("Computing per-model behavioral metrics...")
    models = {}
    for name in ALL_MODELS_ORDERED:
        models[name] = compute_model_behavioral(name, summaries[name])

    # Compute cross-model metrics
    print("Computing cross-model D-set overlap...")
    cross_model = compute_cross_model(summaries)

    # Assemble output
    results = {
        "metadata": metadata_block("exp1_behavioral_screening.py"),
        "models": models,
        "cross_model": cross_model,
    }

    # Save
    save_json(results, CANONICAL / "experiment_1.json")

    # Verify
    verify(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
