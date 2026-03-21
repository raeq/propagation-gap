"""Evaluation metrics for CI-Bench, all category-aware by design.

Every public function accepts an optional category_mask parameter: a boolean
array that selects which items to include. This enforces the paper's core
design principle — no metric is computed without knowing which ignorance
type it applies to.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def expected_calibration_error(
    confidences: NDArray[np.floating],
    correctness: NDArray[np.bool_],
    n_bins: int = 10,
    category_mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Standard binned Expected Calibration Error (ECE).

    Args:
        confidences: Model-reported confidence scores in [0, 1].
        correctness: Binary correctness labels.
        n_bins: Number of equal-width bins.
        category_mask: Boolean mask to select a subset of items.

    Returns:
        Weighted average of |accuracy - confidence| per bin.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.bool_)

    if category_mask is not None:
        category_mask = np.asarray(category_mask, dtype=np.bool_)
        confidences = confidences[category_mask]
        correctness = correctness[category_mask]

    if len(confidences) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(confidences)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            in_bin = (confidences >= low) & (confidences < high)
        else:
            # Last bin includes the right edge.
            in_bin = (confidences >= low) & (confidences <= high)

        n_bin = in_bin.sum()
        if n_bin == 0:
            continue

        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = correctness[in_bin].mean()
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def abstention_precision(
    abstained: NDArray[np.bool_],
    should_abstain: NDArray[np.bool_],
    category_mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Precision of abstention decisions.

    Of the items where the model abstained, what fraction should have
    been abstained on (according to ground-truth labels)?

    Args:
        abstained: Whether the model abstained on each item.
        should_abstain: Ground-truth label for whether abstention is correct.
        category_mask: Boolean mask to select a subset.

    Returns:
        Precision in [0, 1], or 0.0 if the model never abstained.
    """
    abstained = np.asarray(abstained, dtype=np.bool_)
    should_abstain = np.asarray(should_abstain, dtype=np.bool_)

    if category_mask is not None:
        category_mask = np.asarray(category_mask, dtype=np.bool_)
        abstained = abstained[category_mask]
        should_abstain = should_abstain[category_mask]

    n_abstained = abstained.sum()
    if n_abstained == 0:
        return 0.0

    true_positives = (abstained & should_abstain).sum()
    return float(true_positives / n_abstained)


def abstention_recall(
    abstained: NDArray[np.bool_],
    should_abstain: NDArray[np.bool_],
    category_mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Recall of abstention decisions.

    Of the items where the model should have abstained, what fraction
    did it actually abstain on?

    Args:
        abstained: Whether the model abstained on each item.
        should_abstain: Ground-truth label for whether abstention is correct.
        category_mask: Boolean mask to select a subset.

    Returns:
        Recall in [0, 1], or 0.0 if no items should have been abstained on.
    """
    abstained = np.asarray(abstained, dtype=np.bool_)
    should_abstain = np.asarray(should_abstain, dtype=np.bool_)

    if category_mask is not None:
        category_mask = np.asarray(category_mask, dtype=np.bool_)
        abstained = abstained[category_mask]
        should_abstain = should_abstain[category_mask]

    n_should = should_abstain.sum()
    if n_should == 0:
        return 0.0

    true_positives = (abstained & should_abstain).sum()
    return float(true_positives / n_should)


def auroc(
    scores: NDArray[np.floating],
    labels: NDArray[np.bool_],
    category_mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Area Under the ROC Curve (numpy-only implementation).

    Uses the trapezoidal rule on sorted thresholds. Equivalent to
    sklearn.metrics.roc_auc_score but with no sklearn dependency for
    the core metrics module.

    Args:
        scores: Continuous scores (higher = more likely positive).
        labels: Binary ground-truth labels.
        category_mask: Boolean mask to select a subset.

    Returns:
        AUROC in [0, 1].

    Raises:
        ValueError: If labels contain only one class.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)

    if category_mask is not None:
        category_mask = np.asarray(category_mask, dtype=np.bool_)
        scores = scores[category_mask]
        labels = labels[category_mask]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "AUROC is undefined when labels contain only one class "
            f"(n_pos={n_pos}, n_neg={n_neg})."
        )

    # Sort by descending score.
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    # Compute TPR and FPR at each threshold.
    tps = np.cumsum(labels_sorted).astype(np.float64)
    fps = np.cumsum(~labels_sorted).astype(np.float64)
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend origin.
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Trapezoidal integration.
    return float(np.trapezoid(tpr, fpr))


def reasoning_penalty(
    abstention_recall_baseline: float,
    abstention_recall_cot: float,
) -> float:
    """Change in abstention recall after CoT prompting.

    Negative values mean CoT degraded abstention (the reasoning penalty).

    Args:
        abstention_recall_baseline: Recall under direct prompting.
        abstention_recall_cot: Recall under CoT prompting.

    Returns:
        Delta (CoT - baseline). Negative = penalty.
    """
    return abstention_recall_cot - abstention_recall_baseline
