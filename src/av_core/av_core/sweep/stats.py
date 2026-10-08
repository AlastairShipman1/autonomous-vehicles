"""Interval estimates: Wilson 95% for rates, percentile bootstrap (2,000 resamples) for means."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Z95 = 1.959963984540054
N_BOOTSTRAP = 2000


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """(rate, low, high). With n = 0 all three are nan."""
    if n == 0:
        return math.nan, math.nan, math.nan
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean(values: Sequence[float | None] | np.ndarray, n_resamples: int = N_BOOTSTRAP, seed: int = 0, level: float = 0.95
                   ) -> tuple[float, float, float]:
    """(mean, low, high) of the mean of the finite values; nan when there are none."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_resamples, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(v.mean()), float(lo), float(hi)
