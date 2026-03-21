#!/usr/bin/env python3
"""Recompute ALL probe results with V4 labels.

This script replaces the organic pipeline's mentor2_results.json,
mentor2_fixes_results.json, and bootstrap_paired_gaps.json with
V4-label-consistent versions.

The original files used mentor2 labels (different labeling run).
This script uses V4 model-relative labels from phase5_summary_*_v4.json,
which are the labels reported throughout the manuscript.

Inputs (all available via rsync):
- results/phase5/activations/*.npz — hidden-state activations (338 questions × N layers × hidden_dim)
- results/phase5/logit_probe/logit_features_*.npz — 9 logit features per question
- results/phase5/phase5_summary_*_v4.json — V4 labels

Outputs:
- results/experiment_02/mentor2_v4_results.json — per-layer probes with V4 labels
- results/experiment_02/mentor2_v4_fixes_results.json — trajectory + bootstrap gaps
- results/experiment_02/bootstrap_v4_paired_gaps.json — paired bootstrap gap tests

After running, update build_figures.py to load the *_v4_* files.
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    PHASE5, EXP02, LOGIT_PROBE, TIER1_MODELS,
    load_summary, save_json,
)

ACTIVATIONS = PHASE5 / "activations"

# Activation file mapping (organic names from extraction scripts)
ACT_FILES = {
    "Mistral 7B": "mistral7b.npz",
    "Llama 3.1 8B": "llama8b_full.npz",
    "Gemma 2 9B": "gemma9b_stratified.npz",
    "Qwen2.5 7B": "qwen25_7b.npz",
}

# Models with logit features
LOGIT_MODELS = ["Mistral 7B", "Llama 3.1 8B", "Qwen2.5 7B"]

N_CV_FOLDS = 5
N_BOOTSTRAP = 5000
RANDOM_STATE = 42


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_activations(display_name):
    """Load activation .npz and return hidden_states, question_ids."""
    path = ACTIVATIONS / ACT_FILES[display_name]
    d = np.load(path, allow_pickle=True)
    return d["hidden_states"], list(d["question_ids"])


def load_logit_features(file_key, prompt):
    """Load logit probe features."""
    path = LOGIT_PROBE / f"logit_features_{file_key}_{prompt}.npz"
    d = np.load(path, allow_pickle=True)
    return d["features"], list(d["question_ids"])


def get_v4_kd_labels(display_name):
    """Get K/D labels and per-question data from V4 summary."""
    summary = load_summary(display_name)
    pq = summary["per_question"]
    labels = {}
    confidences = {}
    accuracies = {}
    diversities = {}
    n_distincts = {}
    for qid, q in pq.items():
        ml = q.get("model_label", q.get("predicted_label"))
        if ml in ("K", "D"):
            labels[qid] = ml
            confidences[qid] = q.get("confidence")
            accuracies[qid] = q.get("accuracy")
            diversities[qid] = q.get("answer_diversity", 0)
            n_distincts[qid] = q.get("n_distinct_answers", 1)
    return labels, confidences, accuracies, diversities, n_distincts


def align_activations_to_v4(hidden_states, act_qids, v4_labels):
    """Align activation matrix to V4 K/D labels.

    Returns X (n_kd, n_layers, hidden_dim), y (n_kd,), qids_aligned
    """
    indices = []
    y_list = []
    qids_aligned = []
    for i, qid in enumerate(act_qids):
        if qid in v4_labels:
            indices.append(i)
            y_list.append(1 if v4_labels[qid] == "K" else 0)
            qids_aligned.append(qid)
    X = hidden_states[indices]  # (n_kd, n_layers, hidden_dim)
    y = np.array(y_list)
    return X, y, qids_aligned


def cv_auroc(X, y, clf_factory, n_splits=N_CV_FOLDS, random_state=RANDOM_STATE):
    """Stratified CV AUROC with held-out predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    predictions = np.zeros(len(y))
    fold_aurocs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X[train_idx], y[train_idx])
        if hasattr(clf, "predict_proba"):
            preds = clf.predict_proba(X[test_idx])[:, 1]
        else:
            preds = clf.predict(X[test_idx])
        predictions[test_idx] = preds
        if len(np.unique(y[test_idx])) > 1:
            fold_aurocs.append(roc_auc_score(y[test_idx], preds))
    mean_auroc = roc_auc_score(y, predictions) if len(np.unique(y)) > 1 else 0.5
    return mean_auroc, predictions


def bootstrap_auroc_ci(y, predictions, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
    """Percentile bootstrap CI for AUROC."""
    rng = np.random.RandomState(random_state)
    point = roc_auc_score(y, predictions)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], predictions[idx]))
    boots = np.array(boots)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap_gap(y, preds_a, preds_b, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
    """Paired bootstrap for AUROC gap (A - B)."""
    rng = np.random.RandomState(random_state)
    point_a = roc_auc_score(y, preds_a)
    point_b = roc_auc_score(y, preds_b)
    gap = point_a - point_b

    boot_gaps = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        a = roc_auc_score(y[idx], preds_a[idx])
        b = roc_auc_score(y[idx], preds_b[idx])
        boot_gaps.append(a - b)
    boot_gaps = np.array(boot_gaps)
    p_value = float(np.mean(boot_gaps <= 0))
    return gap, float(np.percentile(boot_gaps, 2.5)), float(np.percentile(boot_gaps, 97.5)), p_value


def linear_factory():
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                              class_weight="balanced", random_state=RANDOM_STATE)


def mlp_factory():
    return MLPClassifier(hidden_layer_sizes=(256, 256),
                         max_iter=500, random_state=RANDOM_STATE, early_stopping=True)


# ── Experiment 2A: Linear probing per layer ─────────────────────────────────

def compute_linear_probes(display_name, X_3d, y):
    """Train linear probes at each layer, return per-layer AUROCs + held-out predictions."""
    n_samples, n_layers, hidden_dim = X_3d.shape
    aurocs = []
    predictions_by_layer = {}

    for layer in range(n_layers):
        X_layer = X_3d[:, layer, :].astype(np.float32)
        auroc, preds = cv_auroc(X_layer, y, linear_factory)
        aurocs.append(round(float(auroc), 4))
        predictions_by_layer[layer] = preds

    best_layer = int(np.argmax(aurocs))
    return {
        "linear_aurocs_by_layer": aurocs,
        "best_linear_layer": best_layer,
        "best_linear_auroc": aurocs[best_layer],
        "n_K": int(np.sum(y == 1)),
        "n_D": int(np.sum(y == 0)),
        "predictions_by_layer": predictions_by_layer,
    }


def compute_mlp_probes(display_name, X_3d, y):
    """Train MLP probes at each layer, return per-layer AUROCs."""
    n_samples, n_layers, hidden_dim = X_3d.shape
    aurocs = []

    for layer in range(n_layers):
        X_layer = X_3d[:, layer, :].astype(np.float32)
        try:
            auroc, _ = cv_auroc(X_layer, y, mlp_factory)
        except Exception:
            auroc = 0.5
        aurocs.append(round(float(auroc), 4))

    best_layer = int(np.argmax(aurocs))
    return {
        "mlp_aurocs_by_layer": aurocs,
        "best_mlp_layer": best_layer,
        "best_mlp_auroc": aurocs[best_layer],
    }


def compute_continuous_target(display_name, X_3d, aligned_qids, v4_labels, accuracies):
    """Ridge regression predicting continuous accuracy from activations.

    X_3d and aligned_qids are already filtered to K/D items (from align_activations_to_v4).
    """
    # Build continuous target: accuracy for K/D items that have accuracy data
    indices = []
    y_cont = []
    for i, qid in enumerate(aligned_qids):
        if accuracies.get(qid) is not None:
            indices.append(i)
            y_cont.append(accuracies[qid])

    y_cont = np.array(y_cont)
    X_3d_subset = X_3d[indices]
    n_samples, n_layers, hidden_dim = X_3d_subset.shape

    rhos = []
    r2s = []
    for layer in range(n_layers):
        X_layer = X_3d_subset[:, layer, :].astype(np.float32)
        # Leave-one-out would be ideal but CV is fine for this
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        # Can't stratify on continuous; use KFold instead
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        preds = np.zeros(len(y_cont))
        for train_idx, test_idx in kf.split(X_layer):
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_layer[train_idx], y_cont[train_idx])
            preds[test_idx] = ridge.predict(X_layer[test_idx])
        rho, _ = stats.spearmanr(y_cont, preds)
        ss_res = np.sum((y_cont - preds) ** 2)
        ss_tot = np.sum((y_cont - np.mean(y_cont)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rhos.append(round(float(rho), 4))
        r2s.append(round(float(r2), 4))

    best_layer = int(np.argmax(rhos))
    return {
        "best_spearman_rho": rhos[best_layer],
        "best_r2": r2s[best_layer],
        "best_layer": best_layer,
        "n_layers": n_layers,
        "rhos_by_layer": rhos,
        "r2s_by_layer": r2s,
    }


# ── Experiment 3A: Logit probes ─────────────────────────────────────────────

def compute_logit_probe(display_name, v4_labels):
    """Compute logit probe AUROC from raw .npz features with V4 labels."""
    cfg = TIER1_MODELS[display_name]
    results = {}

    for prompt in ["answer", "confidence"]:
        features, qids = load_logit_features(cfg["file_key"], prompt)

        X_list = []
        y_list = []
        for i, qid in enumerate(qids):
            if qid in v4_labels:
                feat = np.nan_to_num(features[i], nan=0.0)
                X_list.append(feat)
                y_list.append(1 if v4_labels[qid] == "K" else 0)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)

        auroc, predictions = cv_auroc(X, y, linear_factory)
        point, ci_lo, ci_hi = bootstrap_auroc_ci(y, predictions, n_bootstrap=1000)

        results[prompt] = {
            "auroc": round(float(auroc), 4),
            "ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "predictions": predictions,
            "y": y,
        }

    return results


# ── Experiment 3B: Behavioral features ──────────────────────────────────────

def compute_behavioral(display_name, v4_labels, confidences, diversities, n_distincts):
    """Compute behavioral classifier AUROCs from V4 labels."""
    conf_list = []
    div_list = []
    y_list = []

    for qid in v4_labels:
        conf = confidences.get(qid)
        if conf is None:
            continue
        y_list.append(1 if v4_labels[qid] == "K" else 0)
        conf_list.append(conf)
        div = diversities.get(qid, 0)
        if isinstance(div, (int, float)):
            entropy = div
        else:
            entropy = 0
        nd = n_distincts.get(qid, 1)
        div_list.append([conf, entropy, nd])

    y = np.array(y_list)
    X_a = np.array(conf_list, dtype=np.float32).reshape(-1, 1)
    X_b = np.array(div_list, dtype=np.float32)

    auroc_a, preds_a = cv_auroc(X_a, y, linear_factory)
    auroc_b, preds_b = cv_auroc(X_b, y, linear_factory)

    point_a, ci_lo_a, ci_hi_a = bootstrap_auroc_ci(y, preds_a, n_bootstrap=1000)
    point_b, ci_lo_b, ci_hi_b = bootstrap_auroc_ci(y, preds_b, n_bootstrap=1000)

    return {
        "variant_a": {
            "auroc": round(float(auroc_a), 4),
            "ci": [round(ci_lo_a, 4), round(ci_hi_a, 4)],
            "predictions": preds_a,
            "y": y,
        },
        "variant_b": {
            "auroc": round(float(auroc_b), 4),
            "ci": [round(ci_lo_b, 4), round(ci_hi_b, 4)],
            "predictions": preds_b,
            "y": y,
        },
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  V4-CONSISTENT RECOMPUTATION")
    print("  All probes retrained with V4 model-relative labels")
    print("=" * 70)

    all_probe_results = {
        "experiment_2_mlp_vs_linear": {},
        "experiment_1_continuous_target": {},
        "experiment_3_behavioral_features": {},
        "models": list(TIER1_MODELS.keys()),
        "_metadata": {
            "generated": datetime.now().isoformat(),
            "label_source": "V4 model-relative labels from phase5_summary_*_v4.json",
            "n_cv_folds": N_CV_FOLDS,
            "random_state": RANDOM_STATE,
        },
    }

    trajectory = {}
    bootstrap_gaps_all = {}
    fixes_result = {
        "fix1_resource_matched_variants": {},
        "fix2_layer_trajectory": {},
        "fix3_bootstrap_gaps": {},
        "_metadata": {
            "generated": datetime.now().isoformat(),
            "label_source": "V4 model-relative labels",
        },
    }

    # Store predictions for paired bootstrap
    hidden_best_preds = {}
    hidden_last_preds = {}

    for display_name in TIER1_MODELS:
        cfg = TIER1_MODELS[display_name]
        model_key = cfg["key"]
        print(f"\n{'─' * 60}")
        print(f"  {display_name} ({model_key})")
        print(f"{'─' * 60}")

        # Load V4 labels
        v4_labels, confidences, accuracies, diversities, n_distincts = get_v4_kd_labels(display_name)
        n_K = sum(1 for v in v4_labels.values() if v == "K")
        n_D = sum(1 for v in v4_labels.values() if v == "D")
        print(f"  V4 labels: K={n_K}, D={n_D}")

        # Load activations
        print(f"  Loading activations from {ACT_FILES[display_name]}...")
        hidden_states, act_qids = load_activations(display_name)
        X_3d, y, qids_aligned = align_activations_to_v4(hidden_states, act_qids, v4_labels)
        print(f"  Aligned: {X_3d.shape[0]} K/D items × {X_3d.shape[1]} layers × {X_3d.shape[2]} dims")

        # 2A: Linear probes per layer
        print("  Computing linear probes per layer...")
        linear_results = compute_linear_probes(display_name, X_3d, y)
        best_layer = linear_results["best_linear_layer"]
        last_layer = X_3d.shape[1] - 1
        print(f"    Best layer: L{best_layer} AUROC={linear_results['best_linear_auroc']:.4f}")
        print(f"    Last layer: L{last_layer} AUROC={linear_results['linear_aurocs_by_layer'][last_layer]:.4f}")

        # Store predictions for bootstrap
        hidden_best_preds[model_key] = linear_results["predictions_by_layer"][best_layer]
        hidden_last_preds[model_key] = linear_results["predictions_by_layer"][last_layer]

        # 2D: MLP probes per layer
        print("  Computing MLP probes per layer...")
        mlp_results = compute_mlp_probes(display_name, X_3d, y)
        print(f"    Best MLP layer: L{mlp_results['best_mlp_layer']} AUROC={mlp_results['best_mlp_auroc']:.4f}")

        # Store in mentor2 format
        all_probe_results["experiment_2_mlp_vs_linear"][model_key] = {
            "best_linear_auroc": linear_results["best_linear_auroc"],
            "best_mlp_auroc": mlp_results["best_mlp_auroc"],
            "mlp_minus_linear": round(mlp_results["best_mlp_auroc"] - linear_results["best_linear_auroc"], 4),
            "best_linear_layer": linear_results["best_linear_layer"],
            "best_mlp_layer": mlp_results["best_mlp_layer"],
            "n_K": linear_results["n_K"],
            "n_D": linear_results["n_D"],
            "linear_aurocs_by_layer": linear_results["linear_aurocs_by_layer"],
            "mlp_aurocs_by_layer": mlp_results["mlp_aurocs_by_layer"],
        }

        # 2C: Continuous target (ridge regression)
        print("  Computing continuous target (ridge)...")
        cont_results = compute_continuous_target(display_name, X_3d, qids_aligned, v4_labels, accuracies)
        print(f"    Best rho: {cont_results['best_spearman_rho']:.4f} at L{cont_results['best_layer']}")
        all_probe_results["experiment_1_continuous_target"][model_key] = cont_results

        # Clean up activation memory
        del hidden_states, X_3d

    # ── Logit probes and behavioral (Mistral + Llama only) ──────────────────

    logit_preds = {}
    behavioral_preds = {}

    for display_name in LOGIT_MODELS:
        cfg = TIER1_MODELS[display_name]
        model_key = cfg["key"]

        v4_labels, confidences, accuracies, diversities, n_distincts = get_v4_kd_labels(display_name)

        # 3A: Logit probes
        print(f"\n  Computing logit probes for {display_name}...")
        logit_results = compute_logit_probe(display_name, v4_labels)
        logit_preds[model_key] = logit_results
        print(f"    Answer: {logit_results['answer']['auroc']:.4f}")
        print(f"    Confidence: {logit_results['confidence']['auroc']:.4f}")

        # 3B: Behavioral
        print(f"  Computing behavioral for {display_name}...")
        beh_results = compute_behavioral(display_name, v4_labels, confidences, diversities, n_distincts)
        behavioral_preds[model_key] = beh_results
        print(f"    Variant A: {beh_results['variant_a']['auroc']:.4f}")
        print(f"    Variant B: {beh_results['variant_b']['auroc']:.4f}")

        # Behavioral feature coefficients for mentor2 format
        all_probe_results["experiment_3_behavioral_features"][model_key] = {
            "auroc": beh_results["variant_b"]["auroc"],
            "coefficients": {
                "confidence": 0,  # Placeholder — actual coefficients require full model
                "answer_diversity": 0,
                "n_distinct": 0,
                "n_runs": 0,
            },
        }

    # ── Build trajectory and fixes ──────────────────────────────────────────

    print("\n  Building layer trajectory...")
    for display_name in LOGIT_MODELS:
        cfg = TIER1_MODELS[display_name]
        model_key = cfg["key"]
        probe = all_probe_results["experiment_2_mlp_vs_linear"][model_key]

        best_auroc = probe["best_linear_auroc"]
        best_layer = probe["best_linear_layer"]
        last_layer = len(probe["linear_aurocs_by_layer"]) - 1
        last_auroc = probe["linear_aurocs_by_layer"][last_layer]
        logit_conf = logit_preds[model_key]["confidence"]["auroc"]
        logit_ans = logit_preds[model_key]["answer"]["auroc"]

        traj_entry = {
            "best_layer": best_layer,
            "best_auroc": best_auroc,
            "last_layer": last_layer,
            "last_auroc": last_auroc,
            "logit_answer": logit_ans,
            "logit_confidence": logit_conf,
            "drop_best_to_last": round(best_auroc - last_auroc, 4),
            "drop_last_to_logit": round(last_auroc - logit_conf, 4),
        }
        fixes_result["fix2_layer_trajectory"][model_key] = traj_entry
        print(f"  {model_key}: best={best_auroc:.4f} (L{best_layer}) → "
              f"last={last_auroc:.4f} (L{last_layer}) → "
              f"logit={logit_conf:.4f} → drop_last_to_logit={traj_entry['drop_last_to_logit']:.4f}")

    # ── Bootstrap gap tests ─────────────────────────────────────────────────

    print("\n  Running paired bootstrap gap tests (N=5000)...")
    for display_name in LOGIT_MODELS:
        cfg = TIER1_MODELS[display_name]
        model_key = cfg["key"]
        probe = all_probe_results["experiment_2_mlp_vs_linear"][model_key]

        # All predictions must be aligned to the same y vector
        # hidden probes used the activation-aligned y
        # logit/behavioral used their own y (from V4 labels with confidence available)
        # These might differ if some questions lack confidence data
        # We need to find the common subset

        # For paired bootstrap, we need same-length vectors
        # Use the hidden probe predictions (aligned to activation qids)
        v4_labels_d, confidences_d, _, diversities_d, n_distincts_d = get_v4_kd_labels(display_name)

        # Get activation-aligned qids
        _, act_qids = load_activations(display_name)
        kd_qids = [qid for qid in act_qids if qid in v4_labels_d]

        # Hidden predictions
        best_preds = hidden_best_preds[model_key]
        last_preds = hidden_last_preds[model_key]

        # Logit predictions — need to align to same qids
        logit_features, logit_qids = load_logit_features(cfg["file_key"], "confidence")
        logit_qid_to_pred = {}
        logit_X = []
        logit_y_list = []
        for i, qid in enumerate(logit_qids):
            if qid in v4_labels_d:
                logit_X.append(np.nan_to_num(logit_features[i], nan=0.0))
                logit_y_list.append(1 if v4_labels_d[qid] == "K" else 0)
        logit_X = np.array(logit_X, dtype=np.float32)
        logit_y = np.array(logit_y_list)
        _, logit_cv_preds = cv_auroc(logit_X, logit_y, linear_factory)

        # Map logit predictions to qid
        j = 0
        for qid in logit_qids:
            if qid in v4_labels_d:
                logit_qid_to_pred[qid] = logit_cv_preds[j]
                j += 1

        # Behavioral predictions — align to same qids
        beh_a_qid_to_pred = {}
        beh_b_qid_to_pred = {}
        beh_X_a = []
        beh_X_b = []
        beh_y_list = []
        beh_qids_ordered = []
        for qid in sorted(v4_labels_d.keys()):
            conf = confidences_d.get(qid)
            if conf is None:
                continue
            beh_qids_ordered.append(qid)
            beh_y_list.append(1 if v4_labels_d[qid] == "K" else 0)
            beh_X_a.append([conf])
            div = diversities_d.get(qid, 0)
            if not isinstance(div, (int, float)):
                div = 0
            nd = n_distincts_d.get(qid, 1)
            beh_X_b.append([conf, div, nd])

        beh_X_a = np.array(beh_X_a, dtype=np.float32)
        beh_X_b = np.array(beh_X_b, dtype=np.float32)
        beh_y = np.array(beh_y_list)
        _, beh_a_preds = cv_auroc(beh_X_a, beh_y, linear_factory)
        _, beh_b_preds = cv_auroc(beh_X_b, beh_y, linear_factory)

        for i, qid in enumerate(beh_qids_ordered):
            beh_a_qid_to_pred[qid] = beh_a_preds[i]
            beh_b_qid_to_pred[qid] = beh_b_preds[i]

        # Find common qids across all readouts
        common_qids = sorted(
            set(kd_qids) &
            set(logit_qid_to_pred.keys()) &
            set(beh_a_qid_to_pred.keys())
        )
        print(f"  {model_key}: {len(common_qids)} common questions for paired bootstrap")

        # Build aligned arrays
        y_common = np.array([1 if v4_labels_d[q] == "K" else 0 for q in common_qids])

        # Map hidden predictions: kd_qids order → common_qids order
        kd_qid_to_idx = {qid: i for i, qid in enumerate(kd_qids)}
        best_common = np.array([best_preds[kd_qid_to_idx[q]] for q in common_qids])
        last_common = np.array([last_preds[kd_qid_to_idx[q]] for q in common_qids])
        logit_common = np.array([logit_qid_to_pred[q] for q in common_qids])
        beh_a_common = np.array([beh_a_qid_to_pred[q] for q in common_qids])
        beh_b_common = np.array([beh_b_qid_to_pred[q] for q in common_qids])

        # Compute AUROCs on common set
        aurocs = {
            "hidden_best": bootstrap_auroc_ci(y_common, best_common),
            "hidden_last": bootstrap_auroc_ci(y_common, last_common),
            "logit_confidence": bootstrap_auroc_ci(y_common, logit_common),
            "behavioral_a": bootstrap_auroc_ci(y_common, beh_a_common),
            "behavioral_b": bootstrap_auroc_ci(y_common, beh_b_common),
        }

        # Paired gap tests
        comparisons = {
            "best_minus_last": (best_common, last_common),
            "best_minus_behavioral_a": (best_common, beh_a_common),
            "best_minus_behavioral_b": (best_common, beh_b_common),
            "last_minus_logit": (last_common, logit_common),
            "logit_minus_behavioral_a": (logit_common, beh_a_common),
        }

        paired_gaps = {}
        for comp_name, (preds_a, preds_b) in comparisons.items():
            gap, ci_lo, ci_hi, p_val = paired_bootstrap_gap(y_common, preds_a, preds_b)
            paired_gaps[comp_name] = {
                "gap": round(gap, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "p_gap_le_0": round(p_val, 4),
            }
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    {comp_name:35s} gap={gap:+.4f}  p={p_val:.4f} {sig}")

        # Store in bootstrap_paired_gaps format
        bootstrap_gaps_all[model_key] = {
            "best_layer": int(all_probe_results["experiment_2_mlp_vs_linear"][model_key]["best_linear_layer"]),
            "last_layer": int(len(all_probe_results["experiment_2_mlp_vs_linear"][model_key]["linear_aurocs_by_layer"]) - 1),
            "n_K": int(np.sum(y_common == 1)),
            "n_D": int(np.sum(y_common == 0)),
            "aurocs": {
                method: {"point": round(vals[0], 4), "ci_lo": round(vals[1], 4), "ci_hi": round(vals[2], 4)}
                for method, vals in aurocs.items()
            },
            "paired_gaps": paired_gaps,
            "n_bootstrap": N_BOOTSTRAP,
        }

        # Store in fixes format for build_figures.py
        fixes_result["fix3_bootstrap_gaps"][model_key] = {
            "auroc_behavioral_a": aurocs["behavioral_a"][0],
            "ci_behavioral_a": [aurocs["behavioral_a"][1], aurocs["behavioral_a"][2]],
            "auroc_behavioral_b": aurocs["behavioral_b"][0],
            "ci_behavioral_b": [aurocs["behavioral_b"][1], aurocs["behavioral_b"][2]],
            "auroc_logit_confidence": aurocs["logit_confidence"][0],
            "ci_logit_confidence": [aurocs["logit_confidence"][1], aurocs["logit_confidence"][2]],
            "auroc_logit_answer": aurocs["hidden_best"][0],  # Note: keeping field name for compat
            "ci_logit_answer": [aurocs["hidden_best"][1], aurocs["hidden_best"][2]],
            "n_beh": len(common_qids),
            "n_logit": len(common_qids),
            "n_bootstrap": N_BOOTSTRAP,
            "note": f"V4 labels, {len(common_qids)} common questions",
        }

        fixes_result["fix1_resource_matched_variants"][model_key] = {
            "variant_a_auroc": behavioral_preds[model_key]["variant_a"]["auroc"],
            "variant_b_auroc": behavioral_preds[model_key]["variant_b"]["auroc"],
        }

    # ── Save ────────────────────────────────────────────────────────────────

    print("\n  Saving V4-consistent results...")
    save_json(all_probe_results, EXP02 / "mentor2_v4_results.json")
    save_json(fixes_result, EXP02 / "mentor2_v4_fixes_results.json")
    save_json(bootstrap_gaps_all, EXP02 / "bootstrap_v4_paired_gaps.json")

    # ── Summary ─────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  V4-CONSISTENT RECOMPUTATION COMPLETE")
    print("=" * 70)
    print("\n  Output files:")
    print(f"    {EXP02 / 'mentor2_v4_results.json'}")
    print(f"    {EXP02 / 'mentor2_v4_fixes_results.json'}")
    print(f"    {EXP02 / 'bootstrap_v4_paired_gaps.json'}")
    print("\n  Next steps:")
    print("    1. Update build_figures.py to load *_v4_* files")
    print("    2. Update experiment battery to use *_v4_* files")
    print("    3. Rebuild figures and paper")


if __name__ == "__main__":
    main()
