"""Cross-domain transfer analysis for CI-Bench probes (§5.4).

Tests whether probes trained on one sub-category generalise to others.
This distinguishes genuine knowledge-level signal from sub-category-
specific artefacts.

Key contrasts:
  - Train on C1, test on C2 (and vice versa)
  - Train on D2, test on D1 (if D1 is large enough)
  - C3 dropped from primary analysis (format confound)
  - Format-controlled: K-vs-D within TriviaQA-format questions only

Usage:
    from ci_bench.probes.transfer import TransferAnalyzer
    from ci_bench.probes.extract import load_activations

    acts = load_activations("activations/mistral7b.npz")
    analyzer = TransferAnalyzer(seed=42)

    # Full transfer matrix:
    matrix = analyzer.transfer_matrix(acts, layer=3)

    # Format-controlled K-vs-D:
    result = analyzer.format_controlled_kd(acts, layer=3)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ci_bench.probes.extract import ActivationData
from ci_bench.probes.train import ProbeResult


@dataclass
class TransferResult:
    """Result from a transfer experiment.

    Attributes:
        train_label: Sub-category used for training.
        test_label: Sub-category used for testing.
        layer: Layer index.
        accuracy: Classification accuracy on test set.
        auroc: AUROC on test set, or None if single-class test.
        n_train: Training set size.
        n_test: Test set size.
    """
    train_label: str
    test_label: str
    layer: int
    accuracy: float
    auroc: float | None = None
    n_train: int = 0
    n_test: int = 0

    def to_dict(self) -> dict:
        return {
            "train_label": self.train_label,
            "test_label": self.test_label,
            "layer": self.layer,
            "accuracy": round(self.accuracy, 4),
            "auroc": round(self.auroc, 4) if self.auroc is not None else None,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


class TransferAnalyzer:
    """Cross-domain transfer analysis."""

    def __init__(
        self,
        seed: int = 42,
        regularization: float = 1.0,
        max_iter: int = 1000,
        min_samples: int = 5,
    ) -> None:
        self.seed = seed
        self.regularization = regularization
        self.max_iter = max_iter
        self.min_samples = min_samples

    def _filter_by_sub(
        self,
        activations: ActivationData,
        sub_label: str,
    ) -> tuple[NDArray, list[str]]:
        """Filter activations to a single sub-category.

        Returns:
            (hidden_states, question_ids) for matching questions.
        """
        mask = np.array([s == sub_label for s in activations.sub_categories])
        return activations.hidden_states[mask], [
            qid for qid, m in zip(activations.question_ids, mask) if m
        ]

    def _filter_by_category(
        self,
        activations: ActivationData,
        cat_label: str,
    ) -> tuple[NDArray, list[str]]:
        """Filter activations to a top-level category."""
        mask = np.array([c == cat_label for c in activations.categories])
        return activations.hidden_states[mask], [
            qid for qid, m in zip(activations.question_ids, mask) if m
        ]

    def transfer_pair(
        self,
        activations: ActivationData,
        train_pos: str,
        train_neg: str,
        test_pos: str,
        test_neg: str,
        layer: int,
    ) -> TransferResult:
        """Train on one pair of sub-categories, test on another.

        Args:
            train_pos/train_neg: Sub-categories for training (pos=1, neg=0).
            test_pos/test_neg: Sub-categories for testing.
            layer: Layer index.

        Returns:
            TransferResult with accuracy and AUROC.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, accuracy_score

        # Build training set.
        X_pos_all, _ = self._filter_by_sub(activations, train_pos)
        X_neg_all, _ = self._filter_by_sub(activations, train_neg)

        if len(X_pos_all) < self.min_samples or len(X_neg_all) < self.min_samples:
            return TransferResult(
                train_label=f"{train_neg}-vs-{train_pos}",
                test_label=f"{test_neg}-vs-{test_pos}",
                layer=layer,
                accuracy=0.0,
                auroc=None,
                n_train=len(X_pos_all) + len(X_neg_all),
                n_test=0,
            )

        X_train = np.concatenate([X_pos_all[:, layer, :], X_neg_all[:, layer, :]])
        y_train = np.array([1] * len(X_pos_all) + [0] * len(X_neg_all))

        # Build test set.
        X_tpos_all, _ = self._filter_by_sub(activations, test_pos)
        X_tneg_all, _ = self._filter_by_sub(activations, test_neg)

        if len(X_tpos_all) < 1 or len(X_tneg_all) < 1:
            return TransferResult(
                train_label=f"{train_neg}-vs-{train_pos}",
                test_label=f"{test_neg}-vs-{test_pos}",
                layer=layer,
                accuracy=0.0,
                auroc=None,
                n_train=len(y_train),
                n_test=0,
            )

        X_test = np.concatenate([X_tpos_all[:, layer, :], X_tneg_all[:, layer, :]])
        y_test = np.array([1] * len(X_tpos_all) + [0] * len(X_tneg_all))

        # Train and evaluate.
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            C=self.regularization,
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.seed,
        )
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)

        auroc = None
        if len(np.unique(y_test)) == 2:
            proba = clf.predict_proba(X_test_s)[:, 1]
            auroc = roc_auc_score(y_test, proba)

        return TransferResult(
            train_label=f"{train_neg}-vs-{train_pos}",
            test_label=f"{test_neg}-vs-{test_pos}",
            layer=layer,
            accuracy=float(acc),
            auroc=float(auroc) if auroc is not None else None,
            n_train=len(y_train),
            n_test=len(y_test),
        )

    def transfer_matrix(
        self,
        activations: ActivationData,
        layer: int,
        verbose: bool = True,
    ) -> list[TransferResult]:
        """Run the full per-subcategory transfer matrix (§5.4.3).

        Tests all pairwise C-sub vs D-sub contrasts, plus K vs each D-sub.

        Returns:
            List of TransferResult for all valid pairwise tests.
        """
        # Available sub-categories with enough samples.
        sub_counts: dict[str, int] = {}
        for s in activations.sub_categories:
            sub_counts[s] = sub_counts.get(s, 0) + 1

        c_subs = [s for s in ["C1", "C2", "C3"] if sub_counts.get(s, 0) >= self.min_samples]
        d_subs = [s for s in ["D1", "D2", "D3"] if sub_counts.get(s, 0) >= self.min_samples]
        has_k = sub_counts.get("K", 0) >= self.min_samples

        results = []

        # K vs each D sub-category.
        if has_k:
            for d_sub in d_subs:
                r = self.transfer_pair(activations, d_sub, "K", d_sub, "K", layer)
                results.append(r)
                if verbose:
                    auroc_str = f"{r.auroc:.3f}" if r.auroc is not None else "N/A"
                    print(f"  K-vs-{d_sub}: auroc={auroc_str} (n_train={r.n_train}, n_test={r.n_test})",
                          file=sys.stderr, flush=True)

        # Each C sub vs each D sub (train on one pair, test on same pair).
        for c_sub in c_subs:
            for d_sub in d_subs:
                r = self.transfer_pair(activations, d_sub, c_sub, d_sub, c_sub, layer)
                results.append(r)
                if verbose:
                    auroc_str = f"{r.auroc:.3f}" if r.auroc is not None else "N/A"
                    print(f"  {c_sub}-vs-{d_sub}: auroc={auroc_str} (n={r.n_train}+{r.n_test})",
                          file=sys.stderr, flush=True)

        # Cross-C transfer: train on C1-vs-D2, test on C2-vs-D2 (and vice versa).
        if len(c_subs) >= 2 and "D2" in d_subs:
            for i, c1 in enumerate(c_subs):
                for c2 in c_subs[i+1:]:
                    # Train C1-vs-D2, test C2-vs-D2.
                    r = self.transfer_pair(activations, "D2", c1, "D2", c2, layer)
                    results.append(r)
                    if verbose:
                        auroc_str = f"{r.auroc:.3f}" if r.auroc is not None else "N/A"
                        print(f"  Train {c1}-vs-D2, Test {c2}-vs-D2: auroc={auroc_str}",
                              file=sys.stderr, flush=True)

                    # Reverse direction.
                    r = self.transfer_pair(activations, "D2", c2, "D2", c1, layer)
                    results.append(r)
                    if verbose:
                        auroc_str = f"{r.auroc:.3f}" if r.auroc is not None else "N/A"
                        print(f"  Train {c2}-vs-D2, Test {c1}-vs-D2: auroc={auroc_str}",
                              file=sys.stderr, flush=True)

        return results

    def format_controlled_kd(
        self,
        activations: ActivationData,
        layer: int,
        format_controlled_ids: set[str] | None = None,
    ) -> ProbeResult:
        """Format-controlled K-vs-D probe (§5.4.4).

        Uses only TriviaQA-format K and D questions (no hand-crafted C3).
        If format_controlled_ids is not provided, uses all K and D2
        questions (which are all TriviaQA-sourced).

        Returns:
            ProbeResult for the format-controlled K-vs-D contrast.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, accuracy_score

        # Filter to K and D2 (both TriviaQA-format).
        k_mask = np.array([s == "K" for s in activations.sub_categories])
        d2_mask = np.array([s == "D2" for s in activations.sub_categories])

        if format_controlled_ids is not None:
            id_mask = np.array([qid in format_controlled_ids for qid in activations.question_ids])
            k_mask = k_mask & id_mask
            d2_mask = d2_mask & id_mask

        mask = k_mask | d2_mask
        X = activations.hidden_states[mask][:, layer, :]
        y = np.array([1 if d2_mask[i] else 0 for i in range(len(mask)) if mask[i]])

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        if n_pos < self.min_samples or n_neg < self.min_samples:
            return ProbeResult(
                layer=layer,
                contrast="K-vs-D2 (format-controlled)",
                accuracy=0.0,
                auroc=None,
                n_pos=n_pos,
                n_neg=n_neg,
            )

        # Stratified split.
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(y))
        train_idx, test_idx = [], []
        for cls in [0, 1]:
            cls_idx = indices[y == cls]
            rng.shuffle(cls_idx)
            split = int(len(cls_idx) * 0.7)
            train_idx.extend(cls_idx[:split])
            test_idx.extend(cls_idx[split:])

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            C=self.regularization,
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.seed,
        )
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        auroc = None
        if len(np.unique(y_test)) == 2:
            proba = clf.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, proba)

        return ProbeResult(
            layer=layer,
            contrast="K-vs-D2 (format-controlled)",
            accuracy=float(acc),
            auroc=float(auroc) if auroc is not None else None,
            n_pos=n_pos,
            n_neg=n_neg,
            n_train=len(y_train),
            n_test=len(y_test),
        )
