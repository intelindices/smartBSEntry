"""Triple-barrier labels: the training target, independent of the Risk Manager.

From each bar, place an upper barrier at ``+k*ATR`` and a lower barrier at
``-k*ATR`` and look forward at most ``horizon`` bars. Whichever barrier is
touched first names the class; if neither is touched the bar is FLAT.

Nothing here consults swing-R, entry gates, or model probabilities. Two
consequences follow:

1. The Risk Manager can be retuned without invalidating any checkpoint.
2. Labels no longer depend on a bootstrap model, so **two-pass labeling is not
   needed** — there is a single training pass.

Labels are also engine-independent, so they are computed once per asset and
reused across every ``STEngine``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartbs_engines.config import SmartBSConfig
from smartbs_engines.structure import atr as _atr

DEFAULT_BARRIER_K = 1.0
DEFAULT_BARRIER_HORIZON = 24


@dataclass(frozen=True)
class BarrierLabels:
    labels: np.ndarray  # CLASS_FLAT / CLASS_LONG / CLASS_SHORT
    resolved: np.ndarray  # bool: a barrier was actually touched within the horizon
    ambiguous: np.ndarray  # bool: both barriers touched on the same bar

    def stats(self) -> dict[str, float]:
        n = max(len(self.labels), 1)
        return {
            "n": float(len(self.labels)),
            "flat": float((self.labels == SmartBSConfig.CLASS_FLAT).sum()) / n,
            "long": float((self.labels == SmartBSConfig.CLASS_LONG).sum()) / n,
            "short": float((self.labels == SmartBSConfig.CLASS_SHORT).sum()) / n,
            "resolved": float(self.resolved.sum()) / n,
            "ambiguous": float(self.ambiguous.sum()) / n,
        }


def label_triple_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    atr: np.ndarray | None = None,
    k_up: float = DEFAULT_BARRIER_K,
    k_dn: float = DEFAULT_BARRIER_K,
    horizon: int = DEFAULT_BARRIER_HORIZON,
) -> BarrierLabels:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if atr is None:
        atr = _atr(high, low, close, 14)
    atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)

    upper = close + float(k_up) * atr
    lower = close - float(k_dn) * atr

    sentinel = n + 1
    first_up = np.full(n, sentinel, dtype=np.int64)
    first_dn = np.full(n, sentinel, dtype=np.int64)

    # One vectorized pass per forward offset: O(horizon * n), not O(n * horizon)
    # in Python-level loops.
    for j in range(1, int(horizon) + 1):
        if j >= n:
            break
        fut_high = np.full(n, -np.inf)
        fut_low = np.full(n, np.inf)
        fut_high[: n - j] = high[j:]
        fut_low[: n - j] = low[j:]
        hit_up = (fut_high >= upper) & (first_up == sentinel)
        hit_dn = (fut_low <= lower) & (first_dn == sentinel)
        first_up = np.where(hit_up, j, first_up)
        first_dn = np.where(hit_dn, j, first_dn)

    labels = np.full(n, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    labels = np.where(first_up < first_dn, SmartBSConfig.CLASS_LONG, labels)
    labels = np.where(first_dn < first_up, SmartBSConfig.CLASS_SHORT, labels)

    touched = np.minimum(first_up, first_dn)
    ambiguous = (first_up == first_dn) & (first_up != sentinel)
    resolved = (touched != sentinel) & ~ambiguous
    # Barriers hit on the same bar give no ordering, so the bar teaches nothing.
    labels = np.where(ambiguous, SmartBSConfig.CLASS_FLAT, labels)
    # The tail cannot resolve; leave it FLAT and exclude it from training.
    labels[max(n - int(horizon), 0) :] = SmartBSConfig.CLASS_FLAT

    return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)


def select_barrier_train_indices(
    candidates: list[int],
    labels: np.ndarray,
    *,
    seed: int = 42,
    max_flat_ratio: float = 2.0,
) -> list[int]:
    """Cap FLAT dominance and balance LONG vs SHORT.

    Triple-barrier labels are roughly symmetric by construction, so this is a
    much lighter touch than the old policy-simulated labels needed.
    """
    rng = np.random.default_rng(seed)
    longs = [i for i in candidates if labels[i] == SmartBSConfig.CLASS_LONG]
    shorts = [i for i in candidates if labels[i] == SmartBSConfig.CLASS_SHORT]
    flats = [i for i in candidates if labels[i] == SmartBSConfig.CLASS_FLAT]

    keep_dir = min(len(longs), len(shorts))
    if keep_dir == 0:
        return sorted(candidates)
    if len(longs) > keep_dir:
        longs = list(rng.choice(longs, size=keep_dir, replace=False))
    if len(shorts) > keep_dir:
        shorts = list(rng.choice(shorts, size=keep_dir, replace=False))

    max_flat = int(max(keep_dir * 2 * float(max_flat_ratio), 1))
    if len(flats) > max_flat:
        flats = list(rng.choice(flats, size=max_flat, replace=False))
    return sorted(int(i) for i in (*longs, *shorts, *flats))
