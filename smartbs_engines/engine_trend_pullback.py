"""TrendPullbackSTEngine — trend state plus depth of the current pullback.

18 channels encoding EMA stack strength, vol regime, pullback depth on two
horizons (20 and 48 bars), and continuous setup scores.

v2 geometry:
  - Soft stack scores instead of binary trend_up/trend_dn.
  - ``atr_ratio`` = ATR14 / long ATR for spike/chop context (NATGAS).
  - Parallel 48-bar retrace/depth for slower commodities.
  - ``setup_long`` / ``setup_short`` = retrace × trend stack (replaces arm_*).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.registry import BaseSTEngine, EngineResult, safe_div, slope_norm
from smartbs_engines.structure import atr as _atr, ema as _ema

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
IMPULSE_WINDOW = 20
IMPULSE_WINDOW_LONG = 48
ATR_LONG = 100
TREND_CLIP = 2.0


def _clip_trend(x: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(x, nan=0.0), -TREND_CLIP, TREND_CLIP)


def _pullback_block(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    window: int,
    trend_up: np.ndarray,
    trend_dn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Depth, retrace fraction, bars-since-extreme, and range/ATR for one window."""
    s_high = pd.Series(high)
    s_low = pd.Series(low)
    run_max = s_high.rolling(window, min_periods=1).max().to_numpy(dtype=np.float64)
    run_min = s_low.rolling(window, min_periods=1).min().to_numpy(dtype=np.float64)
    rng = run_max - run_min

    depth = np.where(
        trend_up,
        run_max - close,
        np.where(trend_dn, close - run_min, 0.0),
    )
    retrace = np.clip(safe_div(depth, rng), 0.0, 1.5)

    idx_max = s_high.rolling(window, min_periods=1).apply(np.argmax, raw=True).to_numpy()
    idx_min = s_low.rolling(window, min_periods=1).apply(np.argmin, raw=True).to_numpy()
    n = len(close)
    span = np.minimum(np.arange(n) + 1, window)
    bars_since = np.where(
        trend_up,
        span - 1 - idx_max,
        np.where(trend_dn, span - 1 - idx_min, 0.0),
    )
    impulse = safe_div(rng, atr14)
    return depth, retrace, bars_since / float(window), impulse, rng


class TrendPullbackSTEngine(BaseSTEngine):
    name = "trend_pullback"
    feature_names = [
        "stack_fast_mid",
        "stack_mid_slow",
        "atr_ratio",
        "dist_fast",
        "dist_mid",
        "dist_slow",
        "slope_fast",
        "slope_mid",
        "slope_slow",
        "depth_20",
        "retrace_20",
        "bars_since_20",
        "retrace_48",
        "depth_48",
        "impulse_20",
        "impulse_vs_long",
        "setup_long",
        "setup_short",
    ]
    warmup_bars = 3 * max(EMA_SLOW, IMPULSE_WINDOW_LONG, ATR_LONG)

    def compute(self, df: pd.DataFrame) -> EngineResult:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        atr14 = _atr(high, low, close, 14)
        atr_long = (
            pd.Series(atr14)
            .rolling(ATR_LONG, min_periods=ATR_LONG)
            .mean()
            .to_numpy(dtype=np.float64)
        )
        atr_ratio = np.clip(safe_div(atr14, atr_long), 0.0, 3.0)

        fast = _ema(close, EMA_FAST)
        mid = _ema(close, EMA_MID)
        slow = _ema(close, EMA_SLOW)

        stack_fast_mid = _clip_trend(safe_div(fast - mid, atr14))
        stack_mid_slow = _clip_trend(safe_div(mid - slow, atr14))
        stack_sum = stack_fast_mid + stack_mid_slow

        # Internal trend side for pullback geometry (stack still soft in outputs).
        trend_up = (fast > mid) & (mid > slow)
        trend_dn = (fast < mid) & (mid < slow)

        d20, r20, bs20, imp20, rng20 = _pullback_block(
            high=high,
            low=low,
            close=close,
            atr14=atr14,
            window=IMPULSE_WINDOW,
            trend_up=trend_up,
            trend_dn=trend_dn,
        )
        d48, r48, _bs48, _imp48, rng48 = _pullback_block(
            high=high,
            low=low,
            close=close,
            atr14=atr14,
            window=IMPULSE_WINDOW_LONG,
            trend_up=trend_up,
            trend_dn=trend_dn,
        )

        bull_setup = np.clip(stack_sum, 0.0, TREND_CLIP) / TREND_CLIP
        bear_setup = np.clip(-stack_sum, 0.0, TREND_CLIP) / TREND_CLIP
        setup_long = np.clip(r20 * bull_setup, 0.0, 1.5)
        setup_short = np.clip(r20 * bear_setup, 0.0, 1.5)

        cols = [
            stack_fast_mid,
            stack_mid_slow,
            atr_ratio,
            safe_div(close - fast, atr14),
            safe_div(close - mid, atr14),
            safe_div(close - slow, atr14),
            slope_norm(fast, atr14),
            slope_norm(mid, atr14),
            slope_norm(slow, atr14),
            safe_div(d20, atr14),
            r20,
            bs20,
            r48,
            safe_div(d48, atr14),
            imp20,
            np.clip(safe_div(rng20, rng48), 0.0, 3.0),
            setup_long,
            setup_short,
        ]
        return self._finalize(cols, n)
