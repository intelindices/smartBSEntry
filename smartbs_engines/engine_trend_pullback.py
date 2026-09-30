"""TrendPullbackSTEngine — trend state plus depth of the current pullback.

15 channels: EMA stack, distances/slopes, retrace on two horizons, setup scores.
Dropped ``atr_ratio`` (ablation-noisy), plus earlier ``depth_20`` / ``depth_48``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.registry import BaseSTEngine, EngineResult, atr_unit, safe_div, slope_unit
from smartbs_engines.structure import atr as _atr, ema as _ema

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
IMPULSE_WINDOW = 20
IMPULSE_WINDOW_LONG = 48
TREND_CLIP = 2.0
DIST_CLIP = 2.0
SLOPE_CLIP = 2.0
SIZE_CLIP = 3.0


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
    """Retrace fraction, bars-since-extreme, impulse, and window range."""
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
    impulse = atr_unit(rng, atr14, k=SIZE_CLIP)
    return retrace, bars_since / float(window), impulse, rng


class TrendPullbackSTEngine(BaseSTEngine):
    name = "trend_pullback"
    feature_names = [
        "stack_fast_mid",
        "stack_mid_slow",
        "dist_fast",
        "dist_mid",
        "dist_slow",
        "slope_fast",
        "slope_mid",
        "slope_slow",
        "retrace_20",
        "bars_since_20",
        "retrace_48",
        "impulse_20",
        "impulse_vs_long",
        "setup_long",
        "setup_short",
    ]
    warmup_bars = 3 * max(EMA_SLOW, IMPULSE_WINDOW_LONG)

    def compute(self, df: pd.DataFrame) -> EngineResult:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        atr14 = _atr(high, low, close, 14)

        fast = _ema(close, EMA_FAST)
        mid = _ema(close, EMA_MID)
        slow = _ema(close, EMA_SLOW)

        stack_fast_mid = atr_unit(fast - mid, atr14, k=TREND_CLIP)
        stack_mid_slow = atr_unit(mid - slow, atr14, k=TREND_CLIP)
        stack_sum = stack_fast_mid + stack_mid_slow

        trend_up = (fast > mid) & (mid > slow)
        trend_dn = (fast < mid) & (mid < slow)

        r20, bs20, imp20, rng20 = _pullback_block(
            high=high,
            low=low,
            close=close,
            atr14=atr14,
            window=IMPULSE_WINDOW,
            trend_up=trend_up,
            trend_dn=trend_dn,
        )
        r48, _bs48, _imp48, rng48 = _pullback_block(
            high=high,
            low=low,
            close=close,
            atr14=atr14,
            window=IMPULSE_WINDOW_LONG,
            trend_up=trend_up,
            trend_dn=trend_dn,
        )

        bull_setup = np.clip(stack_sum, 0.0, 1.0)
        bear_setup = np.clip(-stack_sum, 0.0, 1.0)
        setup_long = np.clip(r20 * bull_setup, 0.0, 1.5)
        setup_short = np.clip(r20 * bear_setup, 0.0, 1.5)

        cols = [
            stack_fast_mid,
            stack_mid_slow,
            atr_unit(close - fast, atr14, k=DIST_CLIP),
            atr_unit(close - mid, atr14, k=DIST_CLIP),
            atr_unit(close - slow, atr14, k=DIST_CLIP),
            slope_unit(fast, atr14, k=SLOPE_CLIP),
            slope_unit(mid, atr14, k=SLOPE_CLIP),
            slope_unit(slow, atr14, k=SLOPE_CLIP),
            r20,
            bs20,
            r48,
            imp20,
            np.clip(safe_div(rng20, rng48), 0.0, 3.0),
            setup_long,
            setup_short,
        ]
        return self._finalize(cols, n)
