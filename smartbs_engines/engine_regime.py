"""RegimeEngine — Pine trend-regime (fast/mid + close vs mid).

Slim channels (12): regime flags, stack flags, ATR dists, slopes.
``regime_long`` / ``regime_short`` for the signals bus are derived from
``regime_up`` / ``regime_dn`` at extract time (not duplicated here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.registry import BaseSTEngine, EngineResult, atr_unit, slope_unit
from smartbs_engines.structure import atr as _atr, ema as _ema, sma as _sma

EMA_FAST = 14
EMA_MID = 48
EMA_SLOW = 120
SMA_LONG = 180
DIST_CLIP = 3.0


class RegimeEngineSTEngine(BaseSTEngine):
    name = "regime_engine"
    feature_names = [
        "regime_up",
        "regime_dn",
        "regime_side",
        "fast_above_mid",
        "close_above_mid",
        "dist_fast_mid",
        "dist_close_mid",
        "dist_fast",
        "dist_slow",
        "dist_long",
        "slope_fast",
        "slope_mid",
    ]
    warmup_bars = 3 * SMA_LONG

    def compute(self, df: pd.DataFrame) -> EngineResult:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        atr14 = _atr(high, low, close, 14)

        ma_fast = _ema(close, EMA_FAST)
        ma_mid = _ema(close, EMA_MID)
        ma_slow = _ema(close, EMA_SLOW)
        ma_long = _sma(close, SMA_LONG)

        fast_above_mid = (ma_fast > ma_mid).astype(np.float64)
        close_above_mid = (close > ma_mid).astype(np.float64)
        regime_up = ((ma_fast > ma_mid) & (close > ma_mid)).astype(np.float64)
        regime_dn = ((ma_fast < ma_mid) & (close < ma_mid)).astype(np.float64)
        regime_side = (1.0 - np.maximum(regime_up, regime_dn)).astype(np.float64)

        cols = [
            regime_up,
            regime_dn,
            regime_side,
            fast_above_mid,
            close_above_mid,
            atr_unit(ma_fast - ma_mid, atr14, k=DIST_CLIP),
            atr_unit(close - ma_mid, atr14, k=DIST_CLIP),
            atr_unit(close - ma_fast, atr14, k=DIST_CLIP),
            atr_unit(close - ma_slow, atr14, k=DIST_CLIP),
            atr_unit(close - ma_long, atr14, k=DIST_CLIP),
            slope_unit(ma_fast, atr14, k=2.0),
            slope_unit(ma_mid, atr14, k=2.0),
        ]
        return self._finalize(cols, n)
