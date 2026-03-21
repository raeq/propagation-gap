"""Bootstrap confidence interval estimation."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: NDArray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    **metric_kwargs,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for any metric function.

    Args:
        metric_fn: A function that takes one or more arrays and returns
            a scalar metric. Must accept the same positional array
            arguments as passed here.
        *arrays: Arrays to resample (all resampled with the same indices).
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        **metric_kwargs: Additional keyword arguments passed to metric_fn.

    Returns:
        (point_estimate, ci_lower, ci_upper)

    Example:
        >>> from ci_bench.eval.metrics import expected_calibration_error
        >>> point, lo, hi = bootstrap_ci(
        ...     expected_calibration_error,
        ...     confidences, correctness,
        ...     n_bins=10,
        ... )
    """
    rng = np.random.default_rng(seed)
    n = len(arrays[0])

    # Validate all arrays have the same length.
    for i, arr in enumerate(arrays):
        if len(arr) != n:
            raise ValueError(
                f"Array {i} has length {len(arr)}, expected {n}."
            )

    # Point estimate on the full data.
    point = metric_fn(*arrays, **metric_kwargs)

    # Bootstrap resamples.
    estimates = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled = tuple(np.asarray(arr)[idx] for arr in arrays)
        try:
            estimates[b] = metric_fn(*resampled, **metric_kwargs)
        except (ValueError, ZeroDivisionError):
            # Some resamples may have degenerate data (e.g., single class
            # for AUROC). Use NaN and exclude from percentile calculation.
            estimates[b] = np.nan

    # Exclude failed resamples.
    valid = estimates[~np.isnan(estimates)]
    if len(valid) == 0:
        return point, float("nan"), float("nan")

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(valid, 100 * alpha))
    hi = float(np.percentile(valid, 100 * (1 - alpha)))

    return point, lo, hi
