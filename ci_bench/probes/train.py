"""Probe training for CI-Bench Experiment 2.

Trains linear probes (logistic regression) on extracted hidden-state
activations to test whether internal representations separate K, C,
and D knowledge categories.

Supports:
  - Binary probes: C-vs-D, K-vs-D, C-vs-K
  - Three-way probes: K/C/D
  - Per-layer analysis with AUROC and accuracy
  - Stratified train/val/test splits
  - Bootstrap confidence intervals on AUROC
  - Shuffled-label baselines (§5.4.1)
  - Learning curves (§5.4.2)

Usage:
    from ci_bench.probes.train import ProbeTrainer
    from ci_bench.probes.extract import load_activations

    acts = load_activations("activations/mistral7b.npz")
    trainer = ProbeTrainer(seed=42)

    # Binary C-vs-D probe at a specific layer:
    result = trainer.train_binary(acts, layer=3, pos_label="D", neg_label="C")

    # Full layer sweep:
    results = trainer.layer_sweep_binary(acts, pos_label="D", neg_label="C")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ci_bench.probes.extract import ActivationData


@dataclass
class ProbeResult:
    """Results from a single probe training run.

    Attributes:
        layer: Layer index (-1 for aggregated results).
        contrast: Description of the binary contrast (e.g., "C-vs-D").
        accuracy: Classification accuracy on test set.
        auroc: Area under the ROC curve on test set.
        auroc_ci: (lower, upper) 95% bootstrap CI on AUROC, or None.
        n_pos: Number of positive-class examples.
        n_neg: Number of negative-class examples.
        n_train: Training set size.
        n_test: Test set size.
        fold_aurocs: Per-fold AUROCs if cross-validation was used.
        metadata: Additional info (regularization, etc.).
    """
    layer: int
    contrast: str
    accuracy: float
    auroc: float | None = None
    auroc_ci: tuple[float, float] | None = None
    n_pos: int = 0
    n_neg: int = 0
    n_train: int = 0
    n_test: int = 0
    fold_aurocs: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "layer": self.layer,
            "contrast": self.contrast,
            "accuracy": round(self.accuracy, 4),
            "auroc": round(self.auroc, 4) if self.auroc is not None else None,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }
        if self.auroc_ci:
            d["auroc_ci_lower"] = round(self.auroc_ci[0], 4)
            d["auroc_ci_upper"] = round(self.auroc_ci[1], 4)
        if self.fold_aurocs:
            d["fold_aurocs"] = [round(a, 4) for a in self.fold_aurocs]
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class ProbeTrainer:
    """Trains linear probes on extracted activations.

    Handles data splitting, standardization, logistic regression
    training, and evaluation with bootstrap CIs.
    """

    def __init__(
        self,
        seed: int = 42,
        regularization: float = 1.0,
        max_iter: int = 1000,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
    ) -> None:
        self.seed = seed
        self.regularization = regularization
        self.max_iter = max_iter
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level

    def _make_binary_labels(
        self,
        activations: ActivationData,
        pos_label: str,
        neg_label: str,
    ) -> tuple[NDArray, NDArray]:
        """Filter activations to two categories and create binary labels.

        Args:
            activations: Full activation data.
            pos_label: Category or sub-category for positive class.
            neg_label: Category or sub-category for negative class.

        Returns:
            (X, y) where X is (n_samples, n_layers+1, hidden_dim) and
            y is binary {0, 1}.
        """
        pos_mask = np.array([
            c == pos_label or s == pos_label
            for c, s in zip(activations.categories, activations.sub_categories)
        ])
        neg_mask = np.array([
            c == neg_label or s == neg_label
            for c, s in zip(activations.categories, activations.sub_categories)
        ])

        mask = pos_mask | neg_mask
        X = activations.hidden_states[mask]
        y = np.array([1 if pos_mask[i] else 0 for i in range(len(mask)) if mask[i]])

        return X, y

    def _split_data(
        self,
        X: NDArray,
        y: NDArray,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
    ) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
        """Stratified train/val/test split.

        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        rng = np.random.default_rng(self.seed)
        n = len(y)

        # Stratified split: maintain class balance in each partition.
        indices = np.arange(n)
        train_idx, val_idx, test_idx = [], [], []

        for cls in np.unique(y):
            cls_indices = indices[y == cls]
            rng.shuffle(cls_indices)
            n_cls = len(cls_indices)
            n_train = int(n_cls * train_frac)
            n_val = int(n_cls * val_frac)

            train_idx.extend(cls_indices[:n_train])
            val_idx.extend(cls_indices[n_train:n_train + n_val])
            test_idx.extend(cls_indices[n_train + n_val:])

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)

        return (
            (X[train_idx], y[train_idx]),
            (X[val_idx], y[val_idx]),
            (X[test_idx], y[test_idx]),
        )

    def _train_probe_at_layer(
        self,
        X_all: NDArray,
        y: NDArray,
        layer: int,
        use_cv: bool = False,
        n_folds: int = 5,
    ) -> ProbeResult:
        """Train a logistic regression probe at a single layer.

        Args:
            X_all: Full activations (n_samples, n_layers+1, hidden_dim).
            y: Binary labels.
            layer: Layer index to probe.
            use_cv: If True, use cross-validation instead of fixed split.
            n_folds: Number of CV folds (only if use_cv=True).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, accuracy_score

        X = X_all[:, layer, :]
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        if use_cv:
            return self._train_probe_cv(X, y, layer, n_folds)

        # Fixed train/val/test split.
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._split_data(X, y)

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
        auroc_ci = None
        if len(np.unique(y_test)) == 2:
            proba = clf.predict_proba(X_test_s)[:, 1]
            auroc = roc_auc_score(y_test, proba)

            # Bootstrap CI on test set AUROC.
            if self.n_bootstrap > 0:
                auroc_ci = self._bootstrap_auroc(y_test, proba)

        return ProbeResult(
            layer=layer,
            contrast="",  # Caller fills this in.
            accuracy=float(acc),
            auroc=float(auroc) if auroc is not None else None,
            auroc_ci=auroc_ci,
            n_pos=n_pos,
            n_neg=n_neg,
            n_train=len(y_train),
            n_test=len(y_test),
        )

    def _train_probe_cv(
        self,
        X: NDArray,
        y: NDArray,
        layer: int,
        n_folds: int = 5,
    ) -> ProbeResult:
        """Train a probe with stratified cross-validation."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, accuracy_score

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        fold_accs = []
        fold_aurocs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

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

            acc = accuracy_score(y_test, clf.predict(X_test_s))
            fold_accs.append(acc)

            if len(np.unique(y_test)) == 2:
                proba = clf.predict_proba(X_test_s)[:, 1]
                fold_aurocs.append(roc_auc_score(y_test, proba))

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        return ProbeResult(
            layer=layer,
            contrast="",
            accuracy=float(np.mean(fold_accs)),
            auroc=float(np.mean(fold_aurocs)) if fold_aurocs else None,
            n_pos=n_pos,
            n_neg=n_neg,
            n_train=int(len(y) * (n_folds - 1) / n_folds),
            n_test=int(len(y) / n_folds),
            fold_aurocs=[float(a) for a in fold_aurocs],
        )

    def _bootstrap_auroc(
        self,
        y_true: NDArray,
        y_score: NDArray,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval on AUROC."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(self.seed + 1000)
        estimates = []

        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            y_b = y_true[idx]
            s_b = y_score[idx]
            if len(np.unique(y_b)) < 2:
                continue
            estimates.append(roc_auc_score(y_b, s_b))

        if not estimates:
            return (float("nan"), float("nan"))

        alpha = (1.0 - self.ci_level) / 2.0
        lo = float(np.percentile(estimates, 100 * alpha))
        hi = float(np.percentile(estimates, 100 * (1 - alpha)))
        return (lo, hi)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def train_binary(
        self,
        activations: ActivationData,
        layer: int,
        pos_label: str = "D",
        neg_label: str = "C",
        use_cv: bool = False,
    ) -> ProbeResult:
        """Train a binary probe at a specific layer.

        Args:
            activations: Extracted activations.
            layer: Layer index.
            pos_label: Positive class category/sub-category.
            neg_label: Negative class category/sub-category.
            use_cv: Use cross-validation instead of fixed split.

        Returns:
            ProbeResult with accuracy, AUROC, and CIs.
        """
        X_all, y = self._make_binary_labels(activations, pos_label, neg_label)
        result = self._train_probe_at_layer(X_all, y, layer, use_cv=use_cv)
        result.contrast = f"{neg_label}-vs-{pos_label}"
        return result

    def layer_sweep_binary(
        self,
        activations: ActivationData,
        pos_label: str = "D",
        neg_label: str = "C",
        use_cv: bool = False,
        verbose: bool = True,
    ) -> list[ProbeResult]:
        """Run binary probe at every layer.

        Returns list of ProbeResult, one per layer (including post-norm).
        """
        X_all, y = self._make_binary_labels(activations, pos_label, neg_label)
        n_layers = X_all.shape[1]
        contrast = f"{neg_label}-vs-{pos_label}"

        results = []
        for layer_idx in range(n_layers):
            result = self._train_probe_at_layer(X_all, y, layer_idx, use_cv=use_cv)
            result.contrast = contrast
            results.append(result)

            if verbose:
                auroc_str = f"{result.auroc:.3f}" if result.auroc is not None else "N/A"
                ci_str = ""
                if result.auroc_ci:
                    ci_str = f" [{result.auroc_ci[0]:.3f}, {result.auroc_ci[1]:.3f}]"
                print(
                    f"  Layer {layer_idx:2d}: acc={result.accuracy:.3f}  "
                    f"auroc={auroc_str}{ci_str}",
                    file=sys.stderr, flush=True,
                )

        return results

    def shuffled_label_baseline(
        self,
        activations: ActivationData,
        layer: int,
        pos_label: str = "D",
        neg_label: str = "C",
        n_shuffles: int = 10,
    ) -> dict:
        """Run shuffled-label baseline (§5.4.1).

        Trains probes with randomly permuted labels to establish
        the chance-level AUROC. Should be ~0.5.

        Returns dict with mean and std of shuffled AUROCs.
        """
        X_all, y = self._make_binary_labels(activations, pos_label, neg_label)
        X = X_all[:, layer, :]

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(self.seed + 2000)
        shuffled_aurocs = []

        for i in range(n_shuffles):
            y_shuffled = rng.permutation(y)

            (X_train, y_train), _, (X_test, y_test) = self._split_data(X_all[:, layer:layer+1, :], y_shuffled)
            # Reshape for single layer.
            X_train_flat = X_train[:, 0, :]
            X_test_flat = X_test[:, 0, :]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_flat)
            X_test_s = scaler.transform(X_test_flat)

            clf = LogisticRegression(
                C=self.regularization,
                max_iter=self.max_iter,
                class_weight="balanced",
                random_state=self.seed + i,
            )
            clf.fit(X_train_s, y_train)

            if len(np.unique(y_test)) == 2:
                proba = clf.predict_proba(X_test_s)[:, 1]
                shuffled_aurocs.append(roc_auc_score(y_test, proba))

        return {
            "contrast": f"{neg_label}-vs-{pos_label}",
            "layer": layer,
            "n_shuffles": n_shuffles,
            "mean_auroc": float(np.mean(shuffled_aurocs)) if shuffled_aurocs else None,
            "std_auroc": float(np.std(shuffled_aurocs)) if shuffled_aurocs else None,
            "all_aurocs": [round(a, 4) for a in shuffled_aurocs],
        }

    def learning_curve(
        self,
        activations: ActivationData,
        layer: int,
        pos_label: str = "D",
        neg_label: str = "C",
        fractions: list[float] | None = None,
        n_repeats: int = 5,
    ) -> list[dict]:
        """Compute learning curve: AUROC vs training set size (§5.4.2).

        Args:
            fractions: Training set fractions to evaluate. Default:
                [0.1, 0.2, 0.3, 0.5, 0.7, 1.0].
            n_repeats: Number of random subsamples per fraction.

        Returns:
            List of dicts with fraction, n_train, mean_auroc, std_auroc.
        """
        if fractions is None:
            fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

        X_all, y = self._make_binary_labels(activations, pos_label, neg_label)
        X = X_all[:, layer, :]

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        # Fixed test set (last 20%).
        n = len(y)
        rng = np.random.default_rng(self.seed)
        indices = np.arange(n)

        # Stratified test split.
        test_idx = []
        train_pool_idx = []
        for cls in np.unique(y):
            cls_idx = indices[y == cls]
            rng.shuffle(cls_idx)
            n_test = max(1, int(len(cls_idx) * 0.2))
            test_idx.extend(cls_idx[:n_test])
            train_pool_idx.extend(cls_idx[n_test:])

        test_idx = np.array(test_idx)
        train_pool_idx = np.array(train_pool_idx)
        X_test, y_test = X[test_idx], y[test_idx]

        if len(np.unique(y_test)) < 2:
            return []

        results = []
        for frac in fractions:
            frac_aurocs = []
            n_train = max(2, int(len(train_pool_idx) * frac))

            for rep in range(n_repeats):
                rep_rng = np.random.default_rng(self.seed + rep * 100)
                sample_idx = rep_rng.choice(train_pool_idx, size=n_train, replace=False)
                X_train, y_train = X[sample_idx], y[sample_idx]

                if len(np.unique(y_train)) < 2:
                    continue

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

                proba = clf.predict_proba(X_test_s)[:, 1]
                frac_aurocs.append(roc_auc_score(y_test, proba))

            results.append({
                "fraction": frac,
                "n_train": n_train,
                "mean_auroc": float(np.mean(frac_aurocs)) if frac_aurocs else None,
                "std_auroc": float(np.std(frac_aurocs)) if frac_aurocs else None,
                "n_repeats": len(frac_aurocs),
            })

        return results
