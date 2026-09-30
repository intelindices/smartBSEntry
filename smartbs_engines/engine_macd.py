"""MACDSTEngine — EMA stack trend + MACD histogram timing.

13 channels: classic 12/26 EMA geometry (8) plus a slim MACD block (5).
Dropped ``hist_atr``, ``bull_soft``, ``macd_zero_dist`` (overlap with hist_z / persist).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.registry import BaseSTEngine, EngineResult, atr_unit, safe_div, slope_unit
from smartbs_engines.structure import atr as _atr, ema as _ema

EMA_FAST = 12
EMA_SLOW = 26
EMA_SIGNAL = 9
EMA_CTX_FAST = 5
EMA_CTX_SLOW = 35
STACK_CLIP = 2.0
SLOPE_CLIP = 2.0
HIST_Z_WINDOW = 50
HIST_PERSIST_WINDOW = 10
CROSS_DECAY_CAP = 20
HIST_Z_CLIP = 3.0

EMA_CHANNELS: tuple[str, ...] = (
    "stack_12_26",
    "dist_fast_12",
    "dist_slow_26",
    "slope_fast_12",
    "slope_slow_26",
    "stack_slope",
    "close_vs_stack",
    "stack_5_35",
)

MACD_CHANNELS: tuple[str, ...] = (
    "hist_z",
    "hist_slope",
    "hist_persist",
    "cross_up_decay",
    "cross_dn_decay",
)


def _prev(a: np.ndarray) -> np.ndarray:
    p = np.roll(a, 1)
    p[0] = a[0]
    return p


def _bars_since_flag(flag: np.ndarray, cap: int = CROSS_DECAY_CAP) -> np.ndarray:
    n = len(flag)
    out = np.zeros(n, dtype=np.float64)
    last = -1
    for i in range(n):
        if flag[i]:
            last = i
        if last >= 0:
            out[i] = max(0.0, 1.0 - (i - last) / float(cap))
    return out


def _hist_zscore(hist: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(hist)
    mu = s.rolling(window, min_periods=window).mean().to_numpy(dtype=np.float64)
    sd = s.rolling(window, min_periods=window).std(ddof=0).to_numpy(dtype=np.float64)
    z = safe_div(hist - mu, sd)
    return np.clip(z, -HIST_Z_CLIP, HIST_Z_CLIP) / HIST_Z_CLIP


class MACDSTEngine(BaseSTEngine):
    name = "macd"
    feature_names = list(EMA_CHANNELS) + list(MACD_CHANNELS)
    warmup_bars = 3 * max(EMA_SLOW * 3, EMA_CTX_SLOW, HIST_Z_WINDOW)

    def compute(self, df: pd.DataFrame) -> EngineResult:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        atr14 = _atr(high, low, close, 14)

        ema12 = _ema(close, EMA_FAST)
        ema26 = _ema(close, EMA_SLOW)
        ema5 = _ema(close, EMA_CTX_FAST)
        ema35 = _ema(close, EMA_CTX_SLOW)
        stack = ema12 - ema26

        macd_line = stack.copy()
        signal_line = _ema(macd_line, EMA_SIGNAL)
        hist = macd_line - signal_line
        bull = macd_line > signal_line
        cross_up = bull & ~_prev(bull)
        cross_dn = ~bull & _prev(bull)

        dist_fast = atr_unit(close - ema12, atr14, k=STACK_CLIP)
        dist_slow = atr_unit(close - ema26, atr14, k=STACK_CLIP)
        close_vs_stack = 0.5 * (dist_fast + dist_slow)

        ema_cols = [
            atr_unit(stack, atr14, k=STACK_CLIP),
            dist_fast,
            dist_slow,
            slope_unit(ema12, atr14, k=SLOPE_CLIP),
            slope_unit(ema26, atr14, k=SLOPE_CLIP),
            slope_unit(stack, atr14, k=SLOPE_CLIP),
            close_vs_stack,
            atr_unit(ema5 - ema35, atr14, k=STACK_CLIP),
        ]

        macd_cols = [
            _hist_zscore(hist, HIST_Z_WINDOW),
            slope_unit(hist, atr14, k=SLOPE_CLIP),
            pd.Series(np.sign(hist))
            .rolling(HIST_PERSIST_WINDOW, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float64),
            _bars_since_flag(cross_up.astype(np.float64)),
            _bars_since_flag(cross_dn.astype(np.float64)),
        ]

        return self._finalize(ema_cols + macd_cols, n)
