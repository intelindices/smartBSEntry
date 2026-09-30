"""PivotEngine — potential + selected common channels (core-only).

Channels (12), no auto-attached common pack (``_attach_common = False``):

  0  potential = ohlcv_volume * ohlcv_close_displacement
  1..5  ohlcv_open/high/low/volume/close_displacement
  6  trend_regime
  7..8  tod_sin, tod_cos
  9..10 cd_vs_fast, cd_body_cross_fast
  11 last_candle_strength

Formulas match ``common_channels``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.common_channels import (
    COMMON_WARMUP,
    EMA_FAST,
    EMA_SLOW,
    VOL_SMA,
    _candle_direction_cols,
    _last_candle_strength_col,
    _ohlcv_1h_cols,
    _tod_cols,
    _trend_regime_col,
)
from smartbs_engines.registry import BaseSTEngine, EngineResult
from smartbs_engines.structure import ema as _ema

FEATURE_NAMES: tuple[str, ...] = (
    "potential",
    "ohlcv_open",
    "ohlcv_high",
    "ohlcv_low",
    "ohlcv_volume",
    "ohlcv_close_displacement",
    "trend_regime",
    "tod_sin",
    "tod_cos",
    "cd_vs_fast",
    "cd_body_cross_fast",
    "last_candle_strength",
)


class PivotSTEngine(BaseSTEngine):
    """Core-only: do not auto-attach the common pack."""

    name = "pivot_engine"
    feature_names = list(FEATURE_NAMES)
    warmup_bars = max(3 * VOL_SMA, COMMON_WARMUP, 3 * EMA_SLOW)
    _attach_common = False

    def compute(self, df: pd.DataFrame, **kwargs) -> EngineResult:
        n = len(df)
        times = df["open_time"].to_numpy(dtype=np.int64)
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        ma_fast = _ema(c, EMA_FAST)

        o_n, h_n, l_n, v_n, close_disp = _ohlcv_1h_cols(df)
        potential = v_n * close_disp
        trend = _trend_regime_col(c)
        tod_sin, tod_cos = _tod_cols(times)
        cd_vs, cd_cross = _candle_direction_cols(o, c, ma_fast)
        last_str = _last_candle_strength_col(o, h, l, c)

        return self._finalize(
            [
                potential,
                o_n,
                h_n,
                l_n,
                v_n,
                close_disp,
                trend,
                tod_sin,
                tod_cos,
                cd_vs,
                cd_cross,
                last_str,
            ],
            n,
        )
