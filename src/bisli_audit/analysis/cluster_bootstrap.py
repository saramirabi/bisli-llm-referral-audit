"""Child-clustered bootstrap for paired binary outcomes.

Design (frozen for strengthening revision):
- Estimand: transcript-weighted mean of paired deltas (or of a value column).
- Resample whole children with replacement; keep all rows for drawn children.
- For bilingual-minus-monolingual contrasts, resample the two child groups
  independently then form the contrast of group means.
- Null for two-sided p: recentre each child's contribution by subtracting the
  observed mean (clustered recentred bootstrap).
- Monte Carlo floor: report p >= 1/(B+1); never print p=0.
- Degenerate case: if all deltas are exactly 0, p=1 and CI is [0, 0].
- Pointwise percentile CIs are separate from Holm-adjusted p-values.

Default B=10_000, seed=42 (override per call; record seeds in outputs).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable

import numpy as np
import pandas as pd

DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 42


def _groups(df: pd.DataFrame, cluster_col: str) -> dict[Hashable, list[int]]:
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, c in enumerate(df[cluster_col].to_numpy()):
        groups[c].append(i)
    return groups


def cluster_boot_ci(
    df: pd.DataFrame,
    values: pd.Series | np.ndarray,
    *,
    cluster_col: str = "person_id",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for the transcript-weighted mean."""
    vals = np.asarray(values, dtype=float)
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan")
    if not np.any(np.isfinite(vals)):
        return float("nan"), float("nan"), float("nan")
    obs = float(np.nanmean(vals))
    if np.nanstd(vals) == 0.0 or np.allclose(vals, vals[0]):
        return obs, obs, obs
    groups = _groups(df, cluster_col)
    persons = list(groups)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.choice(persons, size=len(persons), replace=True)
        idx = [j for p in draw for j in groups[p]]
        stats[b] = float(np.mean(vals[idx]))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return obs, lo, hi


def recentered_cluster_boot_p(
    df: pd.DataFrame,
    values: pd.Series | np.ndarray,
    *,
    cluster_col: str = "person_id",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided recentred clustered bootstrap p-value for H0: mean=0.

    Recentres the observed values by subtracting the observed mean, then
    resamples children and compares |boot mean| to |observed mean|.
    Returns max(raw_p, 1/(B+1)).
    """
    vals = np.asarray(values, dtype=float)
    if len(df) == 0:
        return float("nan")
    obs = float(np.mean(vals))
    floor = 1.0 / (n_boot + 1)
    if np.allclose(vals, 0.0) or abs(obs) == 0.0:
        return 1.0
    centered = vals - obs
    groups = _groups(df, cluster_col)
    persons = list(groups)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.choice(persons, size=len(persons), replace=True)
        idx = [j for p in draw for j in groups[p]]
        stats[b] = float(np.mean(centered[idx]))
    raw = float(np.mean(np.abs(stats) >= abs(obs)))
    return max(raw, floor)


def two_group_contrast_boot(
    df_a: pd.DataFrame,
    values_a: pd.Series | np.ndarray,
    df_b: pd.DataFrame,
    values_b: pd.Series | np.ndarray,
    *,
    cluster_col: str = "person_id",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Contrast mean(A)-mean(B) with independent child resampling per group."""
    va = np.asarray(values_a, dtype=float)
    vb = np.asarray(values_b, dtype=float)
    obs_a = float(np.mean(va)) if len(va) else float("nan")
    obs_b = float(np.mean(vb)) if len(vb) else float("nan")
    obs = obs_a - obs_b
    floor = 1.0 / (n_boot + 1)
    if len(va) == 0 or len(vb) == 0:
        return {
            "estimate": obs,
            "lo": float("nan"),
            "hi": float("nan"),
            "p_recentered": float("nan"),
        }
    if abs(obs) == 0.0 and np.allclose(va, va[0]) and np.allclose(vb, vb[0]):
        return {"estimate": obs, "lo": obs, "hi": obs, "p_recentered": 1.0}

    ga = _groups(df_a, cluster_col)
    gb = _groups(df_b, cluster_col)
    pa, pb = list(ga), list(gb)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    # Recentred null: subtract group means from each group's values
    ca = va - obs_a
    cb = vb - obs_b
    null_stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        da = rng.choice(pa, size=len(pa), replace=True)
        db = rng.choice(pb, size=len(pb), replace=True)
        ia = [j for p in da for j in ga[p]]
        ib = [j for p in db for j in gb[p]]
        stats[b] = float(np.mean(va[ia]) - np.mean(vb[ib]))
        null_stats[b] = float(np.mean(ca[ia]) - np.mean(cb[ib]))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    raw = float(np.mean(np.abs(null_stats) >= abs(obs)))
    return {
        "estimate": obs,
        "lo": lo,
        "hi": hi,
        "p_recentered": max(raw, floor),
    }


def holm_adjust(pvalues: list[float] | np.ndarray) -> list[float]:
    """Holm step-down adjustment; monotone nondecreasing in rank order output
    aligned to the *sorted* input order then mapped back.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order, start=1):
        val = min(1.0, (m - rank + 1) * float(p[idx]))
        val = max(val, prev)
        prev = val
        adj[idx] = val
    return adj.tolist()
