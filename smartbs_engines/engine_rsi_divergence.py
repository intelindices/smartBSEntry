"""RSIDivergenceSTEngine — dual-timeframe RSI setup detector.

Major timeframe = bar series (typically 1h). Minor = 15m RSI snapshotted to
each 1h bar (last 15m close inside the hour), same pattern as SmartMoney.

Setups encoded as decaying event channels (not one-hot flags):

  **Continuation (trend + minor cross)**
  - Uptrend: EMA20 > EMA50, major RSI in band (30–70), minor RSI crosses up through 30.
  - Downtrend: EMA20 < EMA50, major RSI in band, minor RSI crosses down through 70.

  **Reversal (simultaneous major + minor cross)**
  - Bull: both major and minor RSI cross up through 30 on the same 1h bar.
  - Bear: both cross down through 70 on the same 1h bar.

26 scale-free channels — see ``FEATURE_NAMES`` for audit.

v2 channel tweaks (still 26 inputs):
  - ``has_15m`` replaces redundant ``ema_spread_pct`` (minor RSI data quality).
  - ``bull_div_decay`` / ``bear_div_decay`` replace slope "pressure" with classic
    pivot RSI divergence on the major (1h) series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.smart_money_structure import (
    load_aligned_15m,
    map_15m_end_of_hour_to_1h,
    prev_array,
)
from smartbs_engines.registry import BaseSTEngine, EngineResult, safe_div, slope_norm
from smartbs_engines.structure import atr as _atr, ema as _ema, pivothigh, pivotlow, rsi as _rsi

RSI_PERIOD = 14
RSI_LOW = 30.0
RSI_HIGH = 70.0
EMA_TREND_FAST = 20
EMA_TREND_SLOW = 50
SWING_LEN = 5
DECAY_CAP = 24
MINOR_FALLBACK_PERIOD = 7  # when 15m cache missing: faster RSI on 1h

TREND_CHANNELS: tuple[str, ...] = (
    "trend_stack_atr",
    "trend_slope_atr",
    "close_vs_trend_atr",
    "bull_trend_soft",
    "bear_trend_soft",
    "has_15m",
)

MAJOR_RSI_CHANNELS: tuple[str, ...] = (
    "rsi_maj_norm",
    "rsi_maj_in_band",
    "dist_maj_from_30",
    "dist_maj_from_70",
    "maj_xup_30_decay",
    "maj_xdn_70_decay",
)

MINOR_RSI_CHANNELS: tuple[str, ...] = (
    "rsi_min_norm",
    "rsi_min_xup_30_decay",
    "rsi_min_xdn_70_decay",
    "rsi_spread_min_maj",
    "min_oversold_soft",
    "min_overbought_soft",
)

SETUP_CHANNELS: tuple[str, ...] = (
    "sim_xup_30_decay",
    "sim_xdn_70_decay",
    "cont_long_decay",
    "cont_short_decay",
    "rev_long_decay",
    "rev_short_decay",
    "bull_div_decay",
    "bear_div_decay",
)

FEATURE_NAMES: tuple[str, ...] = (
    TREND_CHANNELS + MAJOR_RSI_CHANNELS + MINOR_RSI_CHANNELS + SETUP_CHANNELS
)


def _bars_since_decay(flag: np.ndarray, cap: int = DECAY_CAP) -> np.ndarray:
    n = len(flag)
    out = np.zeros(n, dtype=np.float64)
    last = -1
    for i in range(n):
        if flag[i]:
            last = i
        if last >= 0:
            out[i] = max(0.0, 1.0 - (i - last) / float(cap))
    return out


def _cross_up(level: float, series: np.ndarray) -> np.ndarray:
    prev = prev_array(series)
    return (prev <= level) & (series > level)


def _cross_dn(level: float, series: np.ndarray) -> np.ndarray:
    prev = prev_array(series)
    return (prev >= level) & (series < level)


def _rsi_divergence_events(
    high: np.ndarray,
    low: np.ndarray,
    rsi: np.ndarray,
    swing_len: int = SWING_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Classic RSI divergence at confirmed swing pivots on the major series."""
    ph = pivothigh(high, swing_len, swing_len)
    pl = pivotlow(low, swing_len, swing_len)
    n = len(rsi)
    bull = np.zeros(n, dtype=np.float64)
    bear = np.zeros(n, dtype=np.float64)
    prev_low_px = prev_low_rsi = np.nan
    prev_high_px = prev_high_rsi = np.nan
    for i in range(n):
        if np.isfinite(pl[i]):
            pi = i - swing_len
            px = low[pi]
            rv = rsi[pi]
            if np.isfinite(prev_low_px) and px < prev_low_px and rv > prev_low_rsi + 1e-6:
                bull[i] = 1.0
            prev_low_px, prev_low_rsi = px, rv
        if np.isfinite(ph[i]):
            pi = i - swing_len
            px = high[pi]
            rv = rsi[pi]
            if np.isfinite(prev_high_px) and px > prev_high_px and rv < prev_high_rsi - 1e-6:
                bear[i] = 1.0
            prev_high_px, prev_high_rsi = px, rv
    return bull, bear


def _soft_band(rsi: np.ndarray, lo: float = RSI_LOW, hi: float = RSI_HIGH) -> np.ndarray:
    """1 inside (lo, hi), smooth roll-off outside."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return np.clip(1.0 - np.abs(rsi - mid) / half, 0.0, 1.0)


def _minor_rsi_on_1h(
    df: pd.DataFrame,
    close_1h: np.ndarray,
    *,
    symbol: str | None,
    data_source: str | None,
) -> tuple[np.ndarray, bool]:
    """Return minor RSI aligned to 1h index; bool = used real 15m."""
    times_1h = df["open_time"].to_numpy(dtype=np.int64)
    df15 = load_aligned_15m(df, symbol=symbol, data_source=data_source)
    if df15 is not None and len(df15) >= RSI_PERIOD + 5:
        c15 = df15["close"].to_numpy(dtype=np.float64)
        rsi15 = _rsi(c15, RSI_PERIOD)
        mapped = map_15m_end_of_hour_to_1h(
            times_1h,
            df15["open_time"].to_numpy(dtype=np.int64),
            rsi15,
        )
        return mapped, True
    return _rsi(close_1h, MINOR_FALLBACK_PERIOD), False


def compute_rsi_divergence_overlay(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    data_source: str | None = None,
) -> dict[str, np.ndarray]:
    """Raw RSI levels + setup flags for replay chart overlay (audit)."""
    close = df["close"].to_numpy(dtype=np.float64)
    rsi_maj = _rsi(close, RSI_PERIOD)
    rsi_min, used_15m = _minor_rsi_on_1h(df, close, symbol=symbol, data_source=data_source)
    ema_f = _ema(close, EMA_TREND_FAST)
    ema_s = _ema(close, EMA_TREND_SLOW)
    uptrend = ema_f > ema_s
    downtrend = ema_f < ema_s
    maj_band = (rsi_maj > RSI_LOW) & (rsi_maj < RSI_HIGH)
    min_up = _cross_up(RSI_LOW, rsi_min)
    min_dn = _cross_dn(RSI_HIGH, rsi_min)
    maj_up = _cross_up(RSI_LOW, rsi_maj)
    maj_dn = _cross_dn(RSI_HIGH, rsi_maj)
    sim_up = maj_up & min_up
    sim_dn = maj_dn & min_dn
    return {
        "rsi_maj": rsi_maj,
        "rsi_min": rsi_min,
        "used_15m": np.full(len(close), 1.0 if used_15m else 0.0),
        "cont_long": uptrend & maj_band & min_up,
        "cont_short": downtrend & maj_band & min_dn,
        "rev_long": sim_up,
        "rev_short": sim_dn,
    }


class RSIDivergenceSTEngine(BaseSTEngine):
    name = "rsi_divergence"
    feature_names = list(FEATURE_NAMES)
    warmup_bars = 3 * max(EMA_TREND_SLOW, RSI_PERIOD * 3, DECAY_CAP, SWING_LEN * 6)

    def compute(
        self,
        df: pd.DataFrame,
        *,
        symbol: str | None = None,
        data_source: str | None = None,
    ) -> EngineResult:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        atr14 = _atr(high, low, close, 14)
        ema_f = _ema(close, EMA_TREND_FAST)
        ema_s = _ema(close, EMA_TREND_SLOW)
        stack = ema_f - ema_s

        rsi_maj = _rsi(close, RSI_PERIOD)
        rsi_min, used_15m = _minor_rsi_on_1h(
            df, close, symbol=symbol, data_source=data_source
        )
        bull_div_evt, bear_div_evt = _rsi_divergence_events(high, low, rsi_maj)

        uptrend = ema_f > ema_s
        downtrend = ema_f < ema_s
        bull_soft = np.tanh(safe_div(stack, atr14))
        bear_soft = np.tanh(safe_div(-stack, atr14))

        maj_up = _cross_up(RSI_LOW, rsi_maj)
        maj_dn = _cross_dn(RSI_HIGH, rsi_maj)
        min_up = _cross_up(RSI_LOW, rsi_min)
        min_dn = _cross_dn(RSI_HIGH, rsi_min)
        sim_up = maj_up & min_up
        sim_dn = maj_dn & min_dn
        maj_band = (rsi_maj > RSI_LOW) & (rsi_maj < RSI_HIGH)

        cont_long = uptrend & maj_band & min_up
        cont_short = downtrend & maj_band & min_dn
        rev_long = sim_up & downtrend
        rev_short = sim_dn & uptrend

        has_15m = np.full(n, 1.0 if used_15m else 0.0, dtype=np.float64)

        trend_cols = [
            np.clip(safe_div(stack, atr14), -3.0, 3.0),
            slope_norm(stack, atr14),
            np.tanh(safe_div(close - ema_f, atr14)),
            bull_soft,
            bear_soft,
            has_15m,
        ]

        maj_cols = [
            (rsi_maj - 50.0) / 50.0,
            _soft_band(rsi_maj),
            np.clip((rsi_maj - RSI_LOW) / 40.0, -1.0, 1.0),
            np.clip((RSI_HIGH - rsi_maj) / 40.0, -1.0, 1.0),
            _bars_since_decay(maj_up.astype(np.float64)),
            _bars_since_decay(maj_dn.astype(np.float64)),
        ]

        min_cols = [
            (rsi_min - 50.0) / 50.0,
            _bars_since_decay(min_up.astype(np.float64)),
            _bars_since_decay(min_dn.astype(np.float64)),
            safe_div(rsi_min - rsi_maj, 100.0),
            np.clip((35.0 - rsi_min) / 35.0, 0.0, 1.0),
            np.clip((rsi_min - 65.0) / 35.0, 0.0, 1.0),
        ]

        setup_cols = [
            _bars_since_decay(sim_up.astype(np.float64)),
            _bars_since_decay(sim_dn.astype(np.float64)),
            _bars_since_decay(cont_long.astype(np.float64)),
            _bars_since_decay(cont_short.astype(np.float64)),
            _bars_since_decay(rev_long.astype(np.float64)),
            _bars_since_decay(rev_short.astype(np.float64)),
            _bars_since_decay(bull_div_evt),
            _bars_since_decay(bear_div_evt),
        ]

        return self._finalize(trend_cols + maj_cols + min_cols + setup_cols, n)
