"""Robust batch statistics.

Optional. Nothing imports this - it is here because the detection plan is
batch-median scoring across all five failure modes and these are the three
functions that keeps needing. If you would rather write your own, delete the
file; nothing breaks.

Median and MAD rather than mean and standard deviation, for a specific
reason: with 18 samples, a single genuinely failed vial moves the mean and
inflates the standard deviation enough to hide itself. That is the failure
mode being guarded against, so the estimator has to be one the outlier cannot
influence.
"""

from __future__ import annotations

import math

import numpy as np

# Scale factor making MAD a consistent estimator of sigma for normal data,
# so a robust z reads on roughly the same scale as an ordinary one.
MAD_TO_SIGMA = 1.4826


def median(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.median(arr)) if arr.size else math.nan


def mad(values, scale: bool = True) -> float:
    """Median absolute deviation from the median.

    Returns 0.0 when more than half the sample is identical - which happens
    for real, e.g. a feature that saturates. robust_z handles that case; do
    not divide by this without checking.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return math.nan
    med = np.median(arr)
    m = float(np.median(np.abs(arr - med)))
    return m * MAD_TO_SIGMA if scale else m


def robust_z(values, min_n: int = 5, floor: float = 1e-9) -> list[float]:
    """Per-sample (value - median) / MAD.

    Returns zeros when the sample is smaller than min_n or when the MAD
    collapses to zero. Both are real situations - early in a run there are
    few peers, and a saturated feature has no spread - and in both the honest
    answer is "no evidence of divergence", not a divide-by-zero or an
    infinite score.

    min_n exists because a median over three vials is not a batch consensus.
    DETECTION.min_vials_for_batch_stats is the project's value for it.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return []
    if arr.size < min_n:
        return [0.0] * arr.size

    spread = mad(arr)
    if not np.isfinite(spread) or spread <= floor:
        return [0.0] * arr.size
    return [float(z) for z in (arr - np.median(arr)) / spread]


def iqr(values) -> float:
    """Interquartile range. An alternative spread estimator to MAD.

    Worth having because MAD is zero whenever a majority of the sample is
    identical, whereas the IQR still measures something as long as the
    quartiles differ.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 4:
        return math.nan
    q1, q3 = np.percentile(arr, [25, 75])
    return float(q3 - q1)
