"""MACDSTEngine v2 — EMA stack trend + MACD histogram timing.

16 channels: classic 12/26 EMA geometry (8) plus a single 12/26/9 MACD block (8).
Slow 5/35 stack is kept as context only; the duplicate fast MACD config is dropped.

v2 geometry:
  - ``stack_12_26`` / ``stack_5_35`` = clipped (fast − slow) / ATR.
  - ``close_vs_stack`` = soft price position vs both EMAs.
  - ``bull_soft`` = tanh((macd − signal) / ATR); cross decays replace 0/1 flags.
  - ``hist_z`` / ``hist_persist`` = regime, not one-bar crosses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.registry import BaseSTEngine, EngineResult, safe_div, slope_norm
from smartbs_engines.structure import atr as _atr, ema as _ema

EMA_FAST = 12
EMA_SLOW = 26
EMA_SIGNAL = 9
EMA_CTX_FAST = 5
EMA_CTX_SLOW = 35
STACK_CLIP = 2.0
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
    "hist_atr",
    "hist_z",
    "hist_slope",
    "bull_soft",
    "hist_persist",
    "macd_zero_dist",
    "cross_up_decay",
    "cross_dn_decay",
)


def _clip_stack(x: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(x, nan=0.0), -STACK_CLIP, STACK_CLIP)


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

        dist_fast = safe_div(close - ema12, atr14)
        dist_slow = safe_div(close - ema26, atr14)
        close_vs_stack = np.clip(0.5 * (dist_fast + dist_slow), -STACK_CLIP, STACK_CLIP)

        ema_cols = [
            _clip_stack(safe_div(stack, atr14)),
            dist_fast,
            dist_slow,
            slope_norm(ema12, atr14),
            slope_norm(ema26, atr14),
            slope_norm(stack, atr14),
            close_vs_stack,
            _clip_stack(safe_div(ema5 - ema35, atr14)),
        ]

        macd_cols = [
            safe_div(hist, atr14),
            _hist_zscore(hist, HIST_Z_WINDOW),
            slope_norm(hist, atr14),
            np.tanh(safe_div(macd_line - signal_line, atr14)),
            pd.Series(np.sign(hist))
            .rolling(HIST_PERSIST_WINDOW, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float64),
            safe_div(macd_line, atr14),
            _bars_since_flag(cross_up.astype(np.float64)),
            _bars_since_flag(cross_dn.astype(np.float64)),
        ]

        return self._finalize(ema_cols + macd_cols, n)
