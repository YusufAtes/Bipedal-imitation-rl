"""Statistical helpers for the rebuttal tables (C6).

All C-track tables report mean +- 95% CI. Pairwise comparisons between
configurations are run with both a paired bootstrap (on matched seeds, if
available) and Welch's unequal-variance t-test.

Functions
---------
* :func:`mean_ci`                mean and bootstrap 95% CI of a single group
* :func:`paired_bootstrap`       bootstrap p-value for the mean paired diff
* :func:`welch_ttest`            classical Welch two-sample t-test
* :func:`pairwise_table`         DataFrame of pairwise comparisons across a
                                 dict[name -> seed_values]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class MeanCI:
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def as_tuple(self) -> tuple[float, float, float, int]:
        return self.mean, self.ci_lo, self.ci_hi, self.n


def _clean(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=np.float64)
    return arr[np.isfinite(arr)]


def mean_ci(
    x: Sequence[float], ci: float = 0.95, n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> MeanCI:
    """Mean and percentile bootstrap CI for a single sample."""
    arr = _clean(x)
    n = arr.size
    if n == 0:
        return MeanCI(float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        return MeanCI(float(arr[0]), float(arr[0]), float(arr[0]), 1)
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return MeanCI(float(arr.mean()), float(lo), float(hi), n)


def paired_bootstrap(
    x: Sequence[float], y: Sequence[float], ci: float = 0.95,
    n_boot: int = 10_000, rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Bootstrap the distribution of the mean paired difference ``x - y``.

    Requires that ``x`` and ``y`` come from matched sources (e.g. seeds). If
    the two vectors have different lengths, the shorter prefix is used.

    Returns mean diff, CI bounds, and a two-sided bootstrap p-value (the
    probability that a resampled mean diff has the opposite sign of the
    observed one, times two, clipped to [0, 1]).
    """
    a = _clean(x)
    b = _clean(y)
    n = min(a.size, b.size)
    if n < 2:
        return dict(mean_diff=float("nan"), ci_lo=float("nan"),
                    ci_hi=float("nan"), p_value=float("nan"), n=n)
    d = a[:n] - b[:n]
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    obs = float(d.mean())
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    if obs >= 0:
        tail = float(np.mean(means <= 0.0))
    else:
        tail = float(np.mean(means >= 0.0))
    p_two = float(min(1.0, 2.0 * tail))
    return dict(mean_diff=obs, ci_lo=float(lo), ci_hi=float(hi),
                p_value=p_two, n=n)


def welch_ttest(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Welch's two-sample t-test (unequal variance, two-sided)."""
    a = _clean(x)
    b = _clean(y)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return dict(t=float("nan"), df=float("nan"), p_value=float("nan"),
                    n1=n1, n2=n2)
    m1, m2 = float(a.mean()), float(b.mean())
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return dict(t=float("inf") if m1 != m2 else 0.0, df=float("nan"),
                    p_value=0.0 if m1 != m2 else 1.0, n1=n1, n2=n2)
    t = (m1 - m2) / se
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    # Two-sided p from Student-t survival function via the regularized
    # incomplete beta function (SciPy-free implementation).
    p = _student_t_sf(abs(t), df) * 2.0
    return dict(t=float(t), df=float(df), p_value=float(min(1.0, p)),
                n1=n1, n2=n2)


def _student_t_sf(t: float, df: float) -> float:
    """Student-t survival function without SciPy (regularized incomplete beta)."""
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float, max_iter: int = 200,
             tol: float = 1e-12) -> float:
    """Regularized incomplete beta I_x(a, b) via continued fraction.

    Numerical Recipes 6.4: the continued fraction is evaluated for whichever
    of ``x`` or ``1-x`` gives faster convergence, then the symmetry
    ``I_x(a, b) = 1 - I_{1-x}(b, a)`` is used.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def _front(a: float, b: float, x: float) -> float:
        ln_beta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
        return float(np.exp(a * np.log(x) + b * np.log(1.0 - x) - ln_beta) / a)

    if x < (a + 1.0) / (a + b + 2.0):
        return _front(a, b, x) * _betacf(a, b, x, max_iter, tol)
    return 1.0 - _front(b, a, 1.0 - x) * _betacf(b, a, 1.0 - x, max_iter, tol)


def _betacf(a: float, b: float, x: float, max_iter: int, tol: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            return h
    return h


def _lgamma(x: float) -> float:
    from math import lgamma
    return float(lgamma(x))


def pairwise_table(
    data: Mapping[str, Sequence[float]], baseline: str | None = None,
    metric_name: str = "metric", n_boot: int = 10_000,
) -> pd.DataFrame:
    """Build a pairwise comparison table.

    If ``baseline`` is provided, compares every other group against it. Else,
    emits all ordered pairs (A, B) with A != B.

    Returned columns: ``metric, group_a, group_b, mean_a, mean_b, diff,
    boot_ci_lo, boot_ci_hi, boot_p, welch_t, welch_p, n_paired``.
    """
    rows: list[dict[str, object]] = []
    names = list(data.keys())
    if baseline is not None:
        pairs = [(baseline, n) for n in names if n != baseline]
    else:
        pairs = [(a, b) for a in names for b in names if a != b]
    for a, b in pairs:
        xa = _clean(data[a])
        xb = _clean(data[b])
        m_a = float(xa.mean()) if xa.size else float("nan")
        m_b = float(xb.mean()) if xb.size else float("nan")
        pb = paired_bootstrap(xa, xb, n_boot=n_boot)
        we = welch_ttest(xa, xb)
        rows.append(dict(
            metric=metric_name, group_a=a, group_b=b,
            mean_a=m_a, mean_b=m_b, diff=pb["mean_diff"],
            boot_ci_lo=pb["ci_lo"], boot_ci_hi=pb["ci_hi"],
            boot_p=pb["p_value"],
            welch_t=we["t"], welch_p=we["p_value"],
            n_paired=pb["n"],
        ))
    return pd.DataFrame(rows)


__all__ = [
    "MeanCI", "mean_ci", "paired_bootstrap", "welch_ttest", "pairwise_table",
]
