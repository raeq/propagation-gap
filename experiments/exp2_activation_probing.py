#!/usr/bin/env python3
"""Experiment 2: Activation Probing.

Paper claim: Hidden states contain a linearly decodable K-vs-D signal
(AUROC ~0.75).

Loads pre-computed probe results from mentor2_results.json (Analyses 2A, 2C, 2D),
surface_controls_results.json (Analysis 2B), and shuffled_mlp_baseline files (2E).

All sources recomputed from activations .npz files by their respective scripts.
This battery script assembles them into a single canonical output.

Outputs results/canonical/experiment_2.json.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_mentor2, load_surface_controls, load_shuffled_baseline,
    save_json, metadata_block, print_verification_header, print_check,
    TIER1_MODELS, CANONICAL,
)


def build_analysis_2a(mentor2):
    """Analysis 2A: Linear probing — per-layer K-vs-D AUROC."""
    mlp_linear = mentor2["experiment_2_mlp_vs_linear"]

    results = {}
    for display_name, cfg in TIER1_MODELS.items():
        key = cfg["key"]
        d = mlp_linear[key]

        aurocs_by_layer = d["linear_aurocs_by_layer"]
        best_layer = d["best_linear_layer"]
        best_auroc = d["best_linear_auroc"]
        l0_auroc = aurocs_by_layer[0]
        final_auroc = aurocs_by_layer[-1]

        results[display_name] = {
            "n_K": d["n_K"],
            "n_D": d["n_D"],
            "k_vs_d": {
                "by_layer": {str(i): round(a, 4)
                             for i, a in enumerate(aurocs_by_layer)},
                "best_layer": best_layer,
                "best_auroc": round(best_auroc, 4),
                "l0_auroc": round(l0_auroc, 4),
                "final_auroc": round(final_auroc, 4),
            },
        }

    return results


def build_analysis_2b(surface_controls):
    """Analysis 2B: Surface-form controls."""
    baselines = surface_controls.get("non_neural_baselines", {})
    year_pcts = surface_controls.get("year_string_percentages", {})
    grouped_cv = surface_controls.get("grouped_cv", {})

    # TF-IDF and surface feature baselines
    tfidf = baselines.get("tfidf_500", {})
    surface_7var = baselines.get("surface_features_7var", {})

    result = {
        "tfidf_500": {
            "c_vs_d": round(tfidf.get("c_vs_d", {}).get("mean_auroc", 0), 3),
            "k_vs_d": round(tfidf.get("k_vs_d", {}).get("mean_auroc", 0), 3),
        },
        "surface_features_7var": {
            "c_vs_d": round(surface_7var.get("c_vs_d", {}).get("mean_auroc", 0), 3),
            "k_vs_d": round(surface_7var.get("k_vs_d", {}).get("mean_auroc", 0), 3),
        },
        "year_string_pct": {
            k: round(v, 2) for k, v in year_pcts.items()
            if isinstance(v, (int, float))
        },
    }

    # Grouped CV (cross-corpus validation)
    if grouped_cv:
        gcv_results = {}
        for model_key, model_data in grouped_cv.items():
            if model_key.startswith("_"):
                continue
            kd_raw = model_data.get("k_vs_d_by_layer", [])
            cd_raw = model_data.get("c_vs_d_by_layer", [])
            # Elements are dicts: k_vs_d has {layer, auroc}, c_vs_d has {layer, c1_to_c2, c2_to_c1, mean}
            def _extract_auroc(e):
                if isinstance(e, dict):
                    return e.get("auroc", e.get("mean", 0))
                return e
            kd_aurocs = [_extract_auroc(e) for e in kd_raw]
            cd_aurocs = [_extract_auroc(e) for e in cd_raw]
            gcv_results[model_key] = {
                "k_vs_d_by_layer": [round(v, 4) for v in kd_aurocs],
                "c_vs_d_by_layer": [round(v, 4) for v in cd_aurocs],
                "k_vs_d_best": round(max(kd_aurocs), 4) if kd_aurocs else None,
                "c_vs_d_best": round(max(cd_aurocs), 4) if cd_aurocs else None,
            }
        result["grouped_cv"] = gcv_results

    # Matched-set controls
    matched = surface_controls.get("matched_set", {})
    if matched:
        result["matched_set"] = {
            "n_matched_pairs": matched.get("n_matched_pairs"),
            "n_total_available": matched.get("n_total"),
            "bow_auroc_cd": matched.get("bow_auroc_cd"),
            "bow_auroc_kd": matched.get("bow_auroc_kd"),
        }

    return result


def build_analysis_2c(mentor2):
    """Analysis 2C: Continuous reliability (ridge regression)."""
    continuous = mentor2["experiment_1_continuous_target"]

    results = {}
    for display_name, cfg in TIER1_MODELS.items():
        key = cfg["key"]
        d = continuous[key]
        results[display_name] = {
            "best_spearman_rho": round(d["best_spearman_rho"], 4),
            "best_r2": round(d["best_r2"], 4),
            "best_layer": d["best_layer"],
            "n_layers": d["n_layers"],
            "rhos_by_layer": [round(r, 4) for r in d["rhos_by_layer"]],
            "r2s_by_layer": [round(r, 4) for r in d["r2s_by_layer"]],
        }

    return results


def build_analysis_2d(mentor2):
    """Analysis 2D: Probe expressiveness (MLP vs linear)."""
    mlp_linear = mentor2["experiment_2_mlp_vs_linear"]

    results = {}
    deltas = []
    for display_name, cfg in TIER1_MODELS.items():
        key = cfg["key"]
        d = mlp_linear[key]
        delta = d["mlp_minus_linear"]
        deltas.append(delta)
        results[display_name] = {
            "linear_best": {
                "layer": d["best_linear_layer"],
                "auroc": round(d["best_linear_auroc"], 4),
            },
            "mlp_best": {
                "layer": d["best_mlp_layer"],
                "auroc": round(d["best_mlp_auroc"], 4),
            },
            "delta": round(delta, 4),
        }

    results["mean_delta"] = round(np.mean(deltas), 3)
    return results


def build_analysis_2e():
    """Analysis 2E: Shuffled-label baselines."""
    results = {}
    for display_name, cfg in TIER1_MODELS.items():
        d = load_shuffled_baseline(cfg["file_key"])
        layers = d["layers"]

        # Find best layer by real_auroc for linear probe
        best_layer_key = None
        best_auroc = 0
        for layer_key, layer_data in layers.items():
            if "linear" in layer_data:
                real = layer_data["linear"].get("real_auroc", 0)
                if real > best_auroc:
                    best_auroc = real
                    best_layer_key = layer_key

        lin = layers[best_layer_key]["linear"]
        z = (lin["real_auroc"] - lin["shuffled_mean"]) / lin["shuffled_std"]

        results[display_name] = {
            "test_layer": int(best_layer_key),
            "real_auroc": round(lin["real_auroc"], 4),
            "shuffled_mean": round(lin["shuffled_mean"], 4),
            "shuffled_std": round(lin["shuffled_std"], 4),
            "z_score": round(z, 2),
            "p_value": lin["p_value"],
            "n_permutations": len(lin.get("shuffled_aurocs", [])),
        }

    return results


def verify(results):
    """Print verification table."""
    print_verification_header("Experiment 2: Activation Probing")

    print("\n  --- Analysis 2A: Linear Probing (K-vs-D) ---")
    for name, d in results["analysis_2a_linear_probing"].items():
        kd = d["k_vs_d"]
        print(f"  {name:20s}  best L{kd['best_layer']:2d}={kd['best_auroc']:.4f}  "
              f"L0={kd['l0_auroc']:.4f}  final={kd['final_auroc']:.4f}  "
              f"(N_K={d['n_K']}, N_D={d['n_D']})")

    print("\n  --- Analysis 2B: Surface Controls ---")
    sc = results["analysis_2b_surface_controls"]
    print(f"  TF-IDF 500:      C-vs-D={sc['tfidf_500']['c_vs_d']:.3f}  "
          f"K-vs-D={sc['tfidf_500']['k_vs_d']:.3f}")
    print(f"  Surface 7var:    C-vs-D={sc['surface_features_7var']['c_vs_d']:.3f}  "
          f"K-vs-D={sc['surface_features_7var']['k_vs_d']:.3f}")
    print(f"  Year-string %:   K={sc['year_string_pct'].get('K', 'N/A')}  "
          f"D={sc['year_string_pct'].get('D', 'N/A')}  "
          f"C={sc['year_string_pct'].get('C', 'N/A')}")
    if "grouped_cv" in sc:
        print("  Grouped CV best K-vs-D:")
        for model, gcv in sc["grouped_cv"].items():
            print(f"    {model}: {gcv['k_vs_d_best']:.4f}")

    print("\n  --- Analysis 2C: Continuous Reliability ---")
    for name, d in results["analysis_2c_continuous"].items():
        print(f"  {name:20s}  best rho={d['best_spearman_rho']:.4f} "
              f"(L{d['best_layer']})  R²={d['best_r2']:.4f}")

    print("\n  --- Analysis 2D: MLP vs Linear ---")
    for name, d in results["analysis_2d_mlp_vs_linear"].items():
        if name == "mean_delta":
            continue
        print(f"  {name:20s}  linear={d['linear_best']['auroc']:.4f} (L{d['linear_best']['layer']})  "
              f"MLP={d['mlp_best']['auroc']:.4f} (L{d['mlp_best']['layer']})  "
              f"delta={d['delta']:+.4f}")
    print(f"  Mean delta: {results['analysis_2d_mlp_vs_linear']['mean_delta']:+.3f}")

    print("\n  --- Analysis 2E: Shuffled-Label Baselines ---")
    for name, d in results["analysis_2e_shuffled_baselines"].items():
        print(f"  {name:20s}  L{d['test_layer']:2d}  "
              f"real={d['real_auroc']:.4f}  "
              f"shuffled={d['shuffled_mean']:.4f}±{d['shuffled_std']:.4f}  "
              f"z={d['z_score']:.2f}  p={d['p_value']}")


def main():
    print("Experiment 2: Activation Probing")
    print("-" * 40)

    # Load source data
    print("Loading mentor2 results...")
    mentor2 = load_mentor2()

    print("Loading surface controls...")
    surface_controls = load_surface_controls()

    # Build analyses
    print("Building Analysis 2A (linear probing)...")
    a2a = build_analysis_2a(mentor2)

    print("Building Analysis 2B (surface controls)...")
    a2b = build_analysis_2b(surface_controls)

    print("Building Analysis 2C (continuous reliability)...")
    a2c = build_analysis_2c(mentor2)

    print("Building Analysis 2D (MLP vs linear)...")
    a2d = build_analysis_2d(mentor2)

    print("Building Analysis 2E (shuffled baselines)...")
    a2e = build_analysis_2e()

    # Assemble output
    results = {
        "metadata": metadata_block("exp2_activation_probing.py"),
        "analysis_2a_linear_probing": a2a,
        "analysis_2b_surface_controls": a2b,
        "analysis_2c_continuous": a2c,
        "analysis_2d_mlp_vs_linear": a2d,
        "analysis_2e_shuffled_baselines": a2e,
    }

    # Save
    save_json(results, CANONICAL / "experiment_2.json")

    # Verify
    verify(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
