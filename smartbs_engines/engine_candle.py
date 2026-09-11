"""CandlePatternSTEngine v2 — 1h price-rejection features.

20 channels focused on long-wick / short-body rejection (pins, failed breaks,
range-context setups). All geometry is range-fraction or ATR-normalized.

v2 geometry:
  - Continuous ``reject_bull`` / ``reject_bear`` scores (replaces pin_* flags).
  - ``wick_body_*`` = wick length vs body (classic pin ratio).
  - ``failed_break_*`` = new high/low with close rejection + large upper/lower wick.
  - ``bars_since_reject_*`` cooldown to reduce overtrade pressure on the TCN.
  - ``setup_reject_*`` = rejection × 50-bar range position (discount/premium).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.registry import BaseSTEngine, EngineResult, safe_div
from smartbs_engines.structure import atr as _atr

RANGE_WINDOW = 50
RANGE_ATR_Z_WINDOW = 50
REJECT_EVENT_THRESH = 0.35
REJECT_DECAY_CAP = 20
WICK_BODY_CLIP = 5.0
RANGE_ATR_Z_CLIP = 3.0

FEATURE_NAMES: tuple[str, ...] = (
    "body_frac",
    "upper_wick_frac",
    "lower_wick_frac",
    "wick_body_up",
    "wick_body_dn",
    "reject_bull",
    "reject_bear",
    "wick_up_atr",
    "wick_dn_atr",
    "close_pos",
    "failed_break_up",
    "failed_break_dn",
    "bars_since_reject_bull",
    "bars_since_reject_bear",
    "setup_reject_long",
    "setup_reject_short",
    "range_atr_z",
    "body_atr",
    "engulf_bull_score",
    "engulf_bear_score",
)


def _prev(a: np.ndarray) -> np.ndarray:
    p = np.roll(a, 1)
    p[0] = a[0]
    return p


def _bars_since_event(score: np.ndarray, thresh: float, cap: int) -> np.ndarray:
    """Linear decay 1→0 over ``cap`` bars after ``score >= thresh``."""
    n = len(score)
    out = np.zeros(n, dtype=np.float64)
    last = -1
    for i in range(n):
        if score[i] >= thresh:
            last = i
        if last >= 0:
            out[i] = max(0.0, 1.0 - (i - last) / float(cap))
    return out


def _range_position(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    window: int,
) -> np.ndarray:
    hi = pd.Series(high).rolling(window, min_periods=1).max().to_numpy(dtype=np.float64)
    lo = pd.Series(low).rolling(window, min_periods=1).min().to_numpy(dtype=np.float64)
    return np.clip(safe_div(close - lo, hi - lo), 0.0, 1.0)


def _range_atr_z(range_atr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(range_atr)
    mu = s.rolling(window, min_periods=window).mean().to_numpy(dtype=np.float64)
    sd = s.rolling(window, min_periods=window).std(ddof=0).to_numpy(dtype=np.float64)
    z = safe_div(range_atr - mu, sd)
    return np.clip(z, -RANGE_ATR_Z_CLIP, RANGE_ATR_Z_CLIP) / RANGE_ATR_Z_CLIP


class CandlePatternSTEngine(BaseSTEngine):
    name = "candle"
    feature_names = list(FEATURE_NAMES)
    warmup_bars = 3 * max(14 * 2, RANGE_WINDOW, RANGE_ATR_Z_WINDOW)

    def compute(self, df: pd.DataFrame) -> EngineResult:
        open_ = df["open"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        atr14 = _atr(high, low, close, 14)

        rng = np.maximum(high - low, 1e-12)
        body = np.abs(close - open_)
        body_frac = np.clip(safe_div(body, rng), 0.0, 1.0)
        top = np.maximum(open_, close)
        bot = np.minimum(open_, close)
        up_wick_frac = np.clip(safe_div(high - top, rng), 0.0, 1.0)
        dn_wick_frac = np.clip(safe_div(bot - low, rng), 0.0, 1.0)
        up_wick = high - top
        dn_wick = bot - low
        close_pos = np.clip(safe_div(close - low, rng), 0.0, 1.0)
        mid = 0.5 * (high + low)

        body_eps = np.maximum(body, atr14 * 0.01)
        wick_body_up = np.clip(safe_div(up_wick, body_eps), 0.0, WICK_BODY_CLIP) / WICK_BODY_CLIP
        wick_body_dn = np.clip(safe_div(dn_wick, body_eps), 0.0, WICK_BODY_CLIP) / WICK_BODY_CLIP

        reject_bull = np.clip(dn_wick_frac * (1.0 - body_frac) * close_pos, 0.0, 1.5)
        reject_bear = np.clip(up_wick_frac * (1.0 - body_frac) * (1.0 - close_pos), 0.0, 1.5)

        wick_up_atr = safe_div(up_wick, atr14)
        wick_dn_atr = safe_div(dn_wick, atr14)

        p_high, p_low = _prev(high), _prev(low)
        new_high = high > p_high
        new_low = low < p_low
        failed_break_up = np.clip(
            wick_up_atr * new_high.astype(np.float64) * (close < mid).astype(np.float64),
            0.0,
            3.0,
        )
        failed_break_dn = np.clip(
            wick_dn_atr * new_low.astype(np.float64) * (close > mid).astype(np.float64),
            0.0,
            3.0,
        )

        range_pos = _range_position(close, high, low, RANGE_WINDOW)
        setup_reject_long = np.clip(reject_bull * (1.0 - range_pos), 0.0, 1.5)
        setup_reject_short = np.clip(reject_bear * range_pos, 0.0, 1.5)

        range_atr = safe_div(rng, atr14)
        range_atr_z = _range_atr_z(range_atr, RANGE_ATR_Z_WINDOW)

        p_open, p_close = _prev(open_), _prev(close)
        prev_bear_body = np.clip(p_open - p_close, 0.0, None)
        prev_bull_body = np.clip(p_close - p_open, 0.0, None)
        curr_bull_body = np.clip(close - open_, 0.0, None)
        curr_bear_body = np.clip(open_ - close, 0.0, None)
        engulf_bull_score = np.clip(
            safe_div(curr_bull_body, prev_bear_body + atr14 * 0.01), 0.0, 2.0
        ) / 2.0 * (p_close < p_open).astype(np.float64) * (close > open_).astype(np.float64)
        engulf_bear_score = np.clip(
            safe_div(curr_bear_body, prev_bull_body + atr14 * 0.01), 0.0, 2.0
        ) / 2.0 * (p_close > p_open).astype(np.float64) * (close < open_).astype(np.float64)

        cols = [
            body_frac,
            up_wick_frac,
            dn_wick_frac,
            wick_body_up,
            wick_body_dn,
            reject_bull,
            reject_bear,
            wick_up_atr,
            wick_dn_atr,
            close_pos,
            failed_break_up,
            failed_break_dn,
            _bars_since_event(reject_bull, REJECT_EVENT_THRESH, REJECT_DECAY_CAP),
            _bars_since_event(reject_bear, REJECT_EVENT_THRESH, REJECT_DECAY_CAP),
            setup_reject_long,
            setup_reject_short,
            range_atr_z,
            safe_div(body, atr14),
            engulf_bull_score,
            engulf_bear_score,
        ]
        return self._finalize(cols, n)
