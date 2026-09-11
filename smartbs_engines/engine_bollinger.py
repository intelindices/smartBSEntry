"""BollingerBandSTEngine — double Bollinger Band (DBB) features.

20 channels: BB(20, EMA, 1σ+2σ) primary block (13) + BB(50, EMA, 1σ+2σ) slow
context (7). Replaces the prior dual-length single-2σ layout.

v2 geometry:
  - Inner 1σ + outer 2σ on the same period (classic DBB zones).
  - Continuous ``zone_score`` = σ-position vs EMA basis (replaces hard outs).
  - ``squeeze_pct`` = 1 − bandwidth percentile over 100 bars.
  - ``keltner_diff`` = BB outer width − 4×ATR for squeeze-vs-trend context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.registry import BaseSTEngine, EngineResult, safe_div, slope_norm
from smartbs_engines.structure import atr as _atr, ema as _ema

BB_FAST = 20
BB_SLOW = 50
INNER_MULT = 1.0
OUTER_MULT = 2.0
SQUEEZE_WINDOW = 100
KELTNER_ATR_MULT = 2.0
PCTB_SLOPE_LOOKBACK = 4
SQUEEZE_THRESH = 0.8
ZONE_CLIP = 2.5
BARS_SINCE_NORM = 100.0

FAST_CHANNELS: tuple[str, ...] = (
    "zone_score",
    "pctb_inner",
    "pctb_outer",
    "dist_u1",
    "dist_l1",
    "dist_u2",
    "dist_l2",
    "bandwidth",
    "squeeze_pct",
    "basis_slope",
    "keltner_diff",
    "pctb_slope",
    "bars_since_squeeze",
)

SLOW_CHANNELS: tuple[str, ...] = (
    "s_zone_score",
    "s_pctb_outer",
    "s_bandwidth",
    "s_squeeze_pct",
    "s_basis_slope",
    "s_dist_u2",
    "s_dist_l2",
)


def _squeeze_pct(bandwidth: np.ndarray, window: int) -> np.ndarray:
    """1 when bandwidth is at its rolling minimum (max compression), 0 when at max."""
    bw = pd.Series(bandwidth)
    roll_min = bw.rolling(window, min_periods=window).min().to_numpy(dtype=np.float64)
    roll_max = bw.rolling(window, min_periods=window).max().to_numpy(dtype=np.float64)
    expansion = np.clip(safe_div(bandwidth - roll_min, roll_max - roll_min), 0.0, 1.0)
    return 1.0 - expansion


def _bars_since_squeeze(squeeze_pct: np.ndarray) -> np.ndarray:
    n = len(squeeze_pct)
    out = np.zeros(n, dtype=np.float64)
    last = -1
    for i in range(n):
        if squeeze_pct[i] >= SQUEEZE_THRESH:
            last = i
        out[i] = min((i - last) / BARS_SINCE_NORM, 1.0) if last >= 0 else 1.0
    return out


def _dbb_bands(close: np.ndarray, length: int) -> tuple[np.ndarray, ...]:
    basis = _ema(close, length)
    dev = (
        pd.Series(close)
        .rolling(length, min_periods=length)
        .std(ddof=0)
        .to_numpy(dtype=np.float64)
    )
    upper1 = basis + INNER_MULT * dev
    lower1 = basis - INNER_MULT * dev
    upper2 = basis + OUTER_MULT * dev
    lower2 = basis - OUTER_MULT * dev
    return basis, dev, upper1, lower1, upper2, lower2


def _dbb_core(
    *,
    close: np.ndarray,
    atr14: np.ndarray,
    length: int,
) -> dict[str, np.ndarray]:
    basis, dev, upper1, lower1, upper2, lower2 = _dbb_bands(close, length)
    inner_w = upper1 - lower1
    outer_w = upper2 - lower2

    zone_score = np.clip(safe_div(close - basis, dev), -ZONE_CLIP, ZONE_CLIP)
    pctb_inner = np.clip(safe_div(close - lower1, inner_w), -0.5, 1.5)
    pctb_outer = np.clip(safe_div(close - lower2, outer_w), -0.5, 1.5)
    bandwidth = safe_div(outer_w, basis)
    sq = _squeeze_pct(bandwidth, SQUEEZE_WINDOW)
    keltner_w = 2.0 * KELTNER_ATR_MULT * atr14

    pctb_slope = np.zeros_like(pctb_outer)
    if len(pctb_outer) > PCTB_SLOPE_LOOKBACK:
        delta = (
            pctb_outer[PCTB_SLOPE_LOOKBACK:] - pctb_outer[:-PCTB_SLOPE_LOOKBACK]
        ) / float(PCTB_SLOPE_LOOKBACK)
        pctb_slope[PCTB_SLOPE_LOOKBACK:] = np.clip(delta, -1.0, 1.0)

    return {
        "zone_score": zone_score,
        "pctb_inner": pctb_inner,
        "pctb_outer": pctb_outer,
        "dist_u1": safe_div(close - upper1, atr14),
        "dist_l1": safe_div(close - lower1, atr14),
        "dist_u2": safe_div(close - upper2, atr14),
        "dist_l2": safe_div(close - lower2, atr14),
        "bandwidth": bandwidth,
        "squeeze_pct": sq,
        "basis_slope": slope_norm(np.nan_to_num(basis, nan=close), atr14),
        "keltner_diff": safe_div(outer_w - keltner_w, atr14),
        "pctb_slope": pctb_slope,
        "bars_since_squeeze": _bars_since_squeeze(sq),
    }


class BollingerBandSTEngine(BaseSTEngine):
    name = "bollinger"
    feature_names = list(FAST_CHANNELS) + list(SLOW_CHANNELS)
    warmup_bars = 3 * (max(BB_FAST, BB_SLOW) + SQUEEZE_WINDOW)

    def compute(self, df: pd.DataFrame) -> EngineResult:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        atr14 = _atr(high, low, close, 14)

        fast = _dbb_core(close=close, atr14=atr14, length=BB_FAST)
        slow = _dbb_core(close=close, atr14=atr14, length=BB_SLOW)

        cols: list[np.ndarray] = [fast[c] for c in FAST_CHANNELS] + [
            slow["zone_score"],
            slow["pctb_outer"],
            slow["bandwidth"],
            slow["squeeze_pct"],
            slow["basis_slope"],
            slow["dist_u2"],
            slow["dist_l2"],
        ]

        return self._finalize(cols, n)
