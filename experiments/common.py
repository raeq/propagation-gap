"""Shared utilities for the clean experiment battery.

Provides: paths, model configs, label loading, CV helpers, bootstrap,
and verification table printing.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats


# ── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PHASE4 = RESULTS / "phase4"
PHASE5 = RESULTS / "phase5"
EXP02 = RESULTS / "experiment_02"
ACTIVATIONS = PHASE5 / "activations"
LOGIT_PROBE = PHASE5 / "logit_probe"
ROBUSTNESS = PHASE5 / "robustness"
CANONICAL = RESULTS / "canonical"

CANONICAL.mkdir(parents=True, exist_ok=True)


# ── Model configs ───────────────────────────────────────────────────────────

TIER1_MODELS = {
    "Mistral 7B": {
        "key": "Mistral-7B",
        "file_key": "mistral7b",
        "summary_file": "phase5_summary_mistral7b_v4.json",
        "n_layers": 33,
        "tier": 1,
    },
    "Llama 3.1 8B": {
        "key": "Llama-3.1-8B",
        "file_key": "llama8b",
        # Use llama8b_v4 (D=95) not llama8b_full_v4 (D=88, different labeling run).
        # The manuscript reports D=95 throughout.
        "summary_file": "phase5_summary_llama8b_v4.json",
        "n_layers": 33,
        "tier": 1,
    },
    "Gemma 2 9B": {
        "key": "Gemma-2-9B",
        "file_key": "gemma9b",
        "summary_file": "phase5_summary_gemma9b_v4.json",
        "n_layers": 43,
        "tier": 1,
    },
    "Qwen2.5 7B": {
        "key": "Qwen2.5-7B",
        "file_key": "qwen25_7b",
        "summary_file": "phase5_summary_qwen25_7b_v4.json",
        "n_layers": 29,
        "tier": 1,
    },
}

TIER2_MODELS = {
    "GPT-4o": {
        "key": "GPT-4o",
        "file_key": "gpt-4o",
        "summary_file": "phase4_summary_gpt-4o.json",
        "tier": 2,
    },
    "Sonnet 3.5": {
        "key": "Sonnet",
        "file_key": "sonnet",
        "summary_file": "phase4_summary_sonnet.json",
        "tier": 2,
    },
    "Gemini 2.0 Flash": {
        "key": "Gemini",
        "file_key": "gemini",
        "summary_file": "phase4_summary_gemini.json",
        "tier": 2,
    },
}

ALL_MODELS = {**TIER1_MODELS, **TIER2_MODELS}
ALL_MODELS_ORDERED = [
    "Mistral 7B", "Llama 3.1 8B", "Gemma 2 9B", "Qwen2.5 7B",
    "GPT-4o", "Sonnet 3.5", "Gemini 2.0 Flash",
]


# ── Data loading ────────────────────────────────────────────────────────────

def load_json(path):
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    """Save data to JSON with formatting."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Saved: {path}")


def _json_default(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_summary(display_name):
    """Load the summary JSON for a model (V4 for Tier 1, phase4 for Tier 2)."""
    cfg = ALL_MODELS[display_name]
    if cfg["tier"] == 1:
        return load_json(PHASE5 / cfg["summary_file"])
    else:
        return load_json(PHASE4 / cfg["summary_file"])


def load_all_summaries():
    """Load summaries for all 6 models."""
    return {name: load_summary(name) for name in ALL_MODELS_ORDERED}


def load_mentor2():
    """Load mentor2_results.json."""
    return load_json(EXP02 / "mentor2_results.json")


def load_bootstrap_gaps():
    """Load bootstrap_paired_gaps.json."""
    return load_json(EXP02 / "bootstrap_paired_gaps.json")


def load_surface_controls():
    """Load surface_controls_results.json."""
    return load_json(EXP02 / "surface_controls_results.json")


def load_shuffled_baseline(file_key):
    """Load shuffled MLP baseline for a model."""
    path = ROBUSTNESS / file_key / f"shuffled_mlp_baseline_{file_key}.json"
    return load_json(path)


# ── Label extraction ────────────────────────────────────────────────────────

def extract_kd_labels(summary):
    """Extract per-question K/D labels and confidence from a summary.

    Returns:
        labels: dict mapping question_id -> 'K' or 'D' (C items excluded)
        confidences: dict mapping question_id -> mean confidence (float)
        accuracies: dict mapping question_id -> accuracy (float)
    """
    pq = summary["per_question"]
    labels = {}
    confidences = {}
    accuracies = {}
    for qid, q in pq.items():
        ml = q.get("model_label", q.get("predicted_label"))
        if ml in ("K", "D"):
            labels[qid] = ml
            confidences[qid] = q.get("confidence", None)
            accuracies[qid] = q.get("accuracy", None)
    return labels, confidences, accuracies


def extract_kdc_labels(summary):
    """Extract per-question K/D/C labels from a summary.

    Returns:
        labels: dict mapping question_id -> 'K', 'D', or 'C'
    """
    pq = summary["per_question"]
    labels = {}
    for qid, q in pq.items():
        ml = q.get("model_label", q.get("predicted_label"))
        if ml in ("K", "D", "C"):
            labels[qid] = ml
    return labels


# ── Cross-validation helpers ────────────────────────────────────────────────

def stratified_cv_auroc(X, y, n_splits=5, C=1.0, random_state=42):
    """5-fold stratified CV logistic regression AUROC.

    Returns mean AUROC across folds and per-fold AUROCs.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_aurocs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=random_state
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        fold_aurocs.append(roc_auc_score(y_test, y_prob))
    return np.mean(fold_aurocs), fold_aurocs


def stratified_cv_auroc_with_predictions(X, y, n_splits=5, C=1.0, random_state=42):
    """Like stratified_cv_auroc but also returns held-out predictions.

    Returns:
        mean_auroc: float
        fold_aurocs: list
        predictions: array of predicted probabilities (aligned with input order)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_aurocs = []
    predictions = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=random_state
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = y_prob
        fold_aurocs.append(roc_auc_score(y_test, y_prob))
    return np.mean(fold_aurocs), fold_aurocs, predictions


# ── Bootstrap ───────────────────────────────────────────────────────────────

def bootstrap_auroc(y_true, y_pred, n_bootstrap=1000, random_state=42):
    """Percentile bootstrap for AUROC confidence interval.

    Returns: point_estimate, ci_lo, ci_hi
    """
    rng = np.random.RandomState(random_state)
    point = roc_auc_score(y_true, y_pred)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_pred[idx]))
    boots = np.array(boots)
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def paired_bootstrap_gap(y_true, pred_a, pred_b, n_bootstrap=5000, random_state=42):
    """Paired bootstrap test for AUROC gap (A - B).

    Returns: gap, ci_lo, ci_hi, p_value (proportion of bootstraps where gap <= 0)
    """
    rng = np.random.RandomState(random_state)
    point_a = roc_auc_score(y_true, pred_a)
    point_b = roc_auc_score(y_true, pred_b)
    gap = point_a - point_b

    boot_gaps = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a = roc_auc_score(y_true[idx], pred_a[idx])
        b = roc_auc_score(y_true[idx], pred_b[idx])
        boot_gaps.append(a - b)
    boot_gaps = np.array(boot_gaps)
    p_value = np.mean(boot_gaps <= 0)
    return gap, np.percentile(boot_gaps, 2.5), np.percentile(boot_gaps, 97.5), p_value


# ── Verification printing ──────────────────────────────────────────────────

def print_verification_header(experiment_name):
    """Print a verification header."""
    print()
    print("=" * 70)
    print(f"  VERIFICATION: {experiment_name}")
    print("=" * 70)


def print_check(description, computed, manuscript=None, tolerance=0.005):
    """Print a verification check.

    If manuscript is provided, flags discrepancies beyond tolerance.
    """
    if manuscript is not None:
        match = abs(computed - manuscript) <= tolerance
        flag = "OK" if match else "*** MISMATCH ***"
        print(f"  {description}: {computed:.4f}  (manuscript: {manuscript})  [{flag}]")
    else:
        print(f"  {description}: {computed:.4f}")


def metadata_block(script_name, n_questions=338):
    """Generate a metadata dict for the output JSON."""
    return {
        "generated": datetime.now().isoformat(),
        "script": script_name,
        "n_questions": n_questions,
    }
