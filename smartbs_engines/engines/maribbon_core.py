"""MA-ribbon feature engine — 39 channels (13 base × 1h + 15m + 5m).

No common pack (``MARibbonSTEngine._attach_common = False``).

Ribbon EMAs: 9, 14, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240 (12).

Base 13 (per TF):
  trend_strength, direction_strength, flat_strength, r_point, supertrend,
  open_t, close_t, high_t, low_t, open_d, close_d, high_d, low_d

Twins: ``*_15m`` / ``*_5m`` = last completed LTF bar inside each primary bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smartbs_engines.engines.dbb import _map_last_completed
from smartbs_engines.registry import atr_unit
from smartbs_engines.smart_money_structure import (
    M15_MS,
    load_aligned_5m,
    load_aligned_15m,
)
from smartbs_engines.structure import atr as _atr
from smartbs_engines.structure import ema as _ema

MARIBBON_EMA_LENS: tuple[int, ...] = (
    9,
    14,
    24,
    36,
    48,
    60,
    72,
    84,
    96,
    120,
    180,
    240,
)
MARIBBON_N_MA = len(MARIBBON_EMA_LENS)  # 12
MARIBBON_N_PAIRS = MARIBBON_N_MA - 1  # 11
MARIBBON_MAX_MA_LEN = max(MARIBBON_EMA_LENS)
MARIBBON_WARMUP_BARS = 3 * MARIBBON_MAX_MA_LEN

ST_ATR_LEN = 9
ST_MULT = 3.0
OHLC_CLIP = 2.0

M5_MS = 300_000  # keep local alias for twin map (matches smart_money_structure)

BASE_FEATURE_NAMES: tuple[str, ...] = (
    "trend_strength",
    "direction_strength",
    "flat_strength",
    "r_point",
    "supertrend",
    "open_t",
    "close_t",
    "high_t",
    "low_t",
    "open_d",
    "close_d",
    "high_d",
    "low_d",
)
BASE_N = len(BASE_FEATURE_NAMES)  # 13

MARIBBON_FEATURE_NAMES: list[str] = (
    list(BASE_FEATURE_NAMES)
    + [f"{n}_15m" for n in BASE_FEATURE_NAMES]
    + [f"{n}_5m" for n in BASE_FEATURE_NAMES]
)
MARIBBON_NUM_INPUTS = len(MARIBBON_FEATURE_NAMES)  # 39


def _ribbon_emas(close: np.ndarray) -> np.ndarray:
    """Return ``(n, 12)`` EMA matrix, fast→slow."""
    cols = [_ema(close, int(L)) for L in MARIBBON_EMA_LENS]
    return np.column_stack(cols)


def _consecutive_stack(mas: np.ndarray, *, from_fast: bool) -> np.ndarray:
    """Signed consecutive pair alignment from one end of the ribbon (±11).

    Bull pair: ``ema[i] > ema[i+1]`` (faster above slower).
    ``from_fast=True`` → direction_strength (short→long / fast→slow).
    ``from_fast=False`` → trend_strength (long→short / slow→fast).
    """
    n = mas.shape[0]
    bull = mas[:, :-1] > mas[:, 1:]
    out = np.zeros(n, dtype=np.float64)
    pair_order = (
        range(MARIBBON_N_PAIRS)
        if from_fast
        else range(MARIBBON_N_PAIRS - 1, -1, -1)
    )
    for t in range(n):
        first: bool | None = None
        count = 0
        for p in pair_order:
            b = bool(bull[t, p])
            if first is None:
                first = b
                count = 1
            elif b == first:
                count += 1
            else:
                break
        if first is None:
            out[t] = 0.0
        else:
            out[t] = float(count if first else -count)
    return out


def _flat_strength(mas: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Count of EMAs inside the bar range ``[low, high]`` (0..12)."""
    inside = (mas >= low[:, None]) & (mas <= high[:, None])
    return inside.sum(axis=1).astype(np.float64)


def _full_stack_state(mas: np.ndarray) -> np.ndarray:
    """+1 full bull, −1 full bear, else 0."""
    bull = mas[:, :-1] > mas[:, 1:]
    bear = mas[:, :-1] < mas[:, 1:]
    all_bull = bull.all(axis=1)
    all_bear = bear.all(axis=1)
    out = np.zeros(len(mas), dtype=np.float64)
    out[all_bull] = 1.0
    out[all_bear] = -1.0
    return out


def _r_point(full: np.ndarray) -> np.ndarray:
    """Pulse on full-stack form/break: +1/−1/0."""
    n = len(full)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    prev = 0.0
    for t in range(n):
        cur = float(full[t])
        if cur == 1.0 and prev != 1.0:
            out[t] = 1.0
        elif cur == -1.0 and prev != -1.0:
            out[t] = -1.0
        elif prev == 1.0 and cur != 1.0:
            out[t] = -1.0
        elif prev == -1.0 and cur != -1.0:
            out[t] = 1.0
        prev = cur
    return out


def _supertrend_dir(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    atr_len: int = ST_ATR_LEN,
    mult: float = ST_MULT,
) -> np.ndarray:
    """Classic SuperTrend direction: +1 uptrend, −1 downtrend."""
    n = len(close)
    atr = _atr(high, low, close, int(atr_len))
    hl2 = 0.5 * (high + low)
    basic_ub = hl2 + float(mult) * atr
    basic_lb = hl2 - float(mult) * atr
    final_ub = np.zeros(n, dtype=np.float64)
    final_lb = np.zeros(n, dtype=np.float64)
    direction = np.ones(n, dtype=np.float64)  # +1 up
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    for i in range(1, n):
        final_ub[i] = (
            basic_ub[i]
            if (basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1])
            else final_ub[i - 1]
        )
        final_lb[i] = (
            basic_lb[i]
            if (basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1])
            else final_lb[i - 1]
        )
        if close[i] > final_ub[i - 1]:
            direction[i] = 1.0
        elif close[i] < final_lb[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
    return direction


def _section_means(
    mas: np.ndarray,
    strength: np.ndarray,
    *,
    from_fast: bool,
) -> np.ndarray:
    """Mean of consecutive EMAs counted from the fast or slow end."""
    n = mas.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        k = int(abs(strength[t]))
        n_take = max(1, min(k + 1, MARIBBON_N_MA))
        if from_fast:
            out[t] = float(np.mean(mas[t, :n_take]))
        else:
            out[t] = float(np.mean(mas[t, -n_take:]))
    return out


def pack_maribbon_base(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> list[np.ndarray]:
    """Compute the 13 base channels for one OHLCV frame."""
    n = len(close)
    if n == 0:
        return [np.zeros(0, dtype=np.float64) for _ in BASE_FEATURE_NAMES]

    mas = _ribbon_emas(close)
    atr14 = _atr(high, low, close, 14)

    trend_s = _consecutive_stack(mas, from_fast=False)  # long→short
    dir_s = _consecutive_stack(mas, from_fast=True)  # short→long
    flat_s = _flat_strength(mas, low, high)
    full = _full_stack_state(mas)
    r_pt = _r_point(full)
    st = _supertrend_dir(high, low, close)

    sec_t = _section_means(mas, trend_s, from_fast=False)
    sec_d = _section_means(mas, dir_s, from_fast=True)

    def _vs(px: np.ndarray, sec: np.ndarray) -> np.ndarray:
        return atr_unit(px - sec, atr14, k=OHLC_CLIP)

    return [
        trend_s,
        dir_s,
        flat_s,
        r_pt,
        st,
        _vs(open_, sec_t),
        _vs(close, sec_t),
        _vs(high, sec_t),
        _vs(low, sec_t),
        _vs(open_, sec_d),
        _vs(close, sec_d),
        _vs(high, sec_d),
        _vs(low, sec_d),
    ]


@dataclass
class MARibbonEngine:
    features: np.ndarray
    valid_from: int


def compute_maribbon_engine(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    data_source: str | None = None,
    frames: dict[str, pd.DataFrame] | None = None,
) -> MARibbonEngine:
    """Build ``(n, 39)`` features aligned to ``df`` (primary TF + 15m + 5m twins)."""
    n = len(df)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    times = df["open_time"].to_numpy(dtype=np.int64)

    cols_1h = pack_maribbon_base(o, h, l, c)

    bag = dict(frames) if frames else {}
    if bag.get("15m") is None or not len(bag.get("15m", [])):
        bag["15m"] = load_aligned_15m(df, symbol=symbol, data_source=data_source)
    if bag.get("5m") is None or not len(bag.get("5m", [])):
        bag["5m"] = load_aligned_5m(df, symbol=symbol, data_source=data_source)

    def _twin(df_ltf: pd.DataFrame | None, dur_ms: int) -> list[np.ndarray]:
        if df_ltf is None or len(df_ltf) == 0:
            return [np.zeros(n, dtype=np.float64) for _ in BASE_FEATURE_NAMES]
        pack = pack_maribbon_base(
            df_ltf["open"].to_numpy(dtype=np.float64),
            df_ltf["high"].to_numpy(dtype=np.float64),
            df_ltf["low"].to_numpy(dtype=np.float64),
            df_ltf["close"].to_numpy(dtype=np.float64),
        )
        t_ltf = df_ltf["open_time"].to_numpy(dtype=np.int64)
        return [_map_last_completed(times, t_ltf, ch, dur_ms=dur_ms) for ch in pack]

    cols_15 = _twin(bag.get("15m"), M15_MS)
    cols_5 = _twin(bag.get("5m"), M5_MS)

    feats = np.column_stack([*cols_1h, *cols_15, *cols_5])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if feats.shape != (n, MARIBBON_NUM_INPUTS):
        raise ValueError(f"maribbon features {feats.shape} != ({n}, {MARIBBON_NUM_INPUTS})")
    return MARibbonEngine(features=feats, valid_from=min(n, MARIBBON_WARMUP_BARS))


def build_maribbon_features(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    data_source: str | None = None,
) -> np.ndarray:
    """``(n, 39)`` maribbon matrix."""
    return compute_maribbon_engine(df, symbol=symbol, data_source=data_source).features


# ---------------------------------------------------------------------------
# Classic MA-ribbon channels (dist + slope on current 12 EMAs) — used by
# all_blend in place of the 39ch stack/OHLC twin pack.
# ---------------------------------------------------------------------------
DIST_CLIP = 2.0
SLOPE_CLIP = 2.0
SLOPE_LOOKBACK = 1

MA_RIBBON_MA_NAMES: tuple[str, ...] = tuple(f"ema{L}" for L in MARIBBON_EMA_LENS)
MA_RIBBON_CHANNEL_NAMES: list[str] = (
    [f"dist_{n}" for n in MA_RIBBON_MA_NAMES]
    + [f"slope_{n}" for n in MA_RIBBON_MA_NAMES]
)
MA_RIBBON_NUM_CHANNELS = len(MA_RIBBON_CHANNEL_NAMES)  # 24 = 12 dist + 12 slope


def compute_ma_ribbon_channels(df: pd.DataFrame) -> np.ndarray:
    """``(n, 24)`` classic ribbon: ATR-norm dist + slope on EMAs 9…240.

    Dist layout matches legacy maribbon: ``dist_ema9 = (close−ema9)/ATR``,
    then ``dist_ema{L} = (ema_prev − ema_L)/ATR`` for slower MAs.
    """
    from smartbs_engines.registry import atr_unit, slope_unit

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    n = len(close)
    if n == 0:
        return np.zeros((0, MA_RIBBON_NUM_CHANNELS), dtype=np.float32)

    atr14 = _atr(high, low, close, 14)
    mas = _ribbon_emas(close)  # (n, 12)

    dists = np.zeros((n, MARIBBON_N_MA), dtype=np.float64)
    dists[:, 0] = atr_unit(close - mas[:, 0], atr14, k=DIST_CLIP)
    for i in range(1, MARIBBON_N_MA):
        dists[:, i] = atr_unit(mas[:, i - 1] - mas[:, i], atr14, k=DIST_CLIP)

    slopes = np.stack(
        [
            slope_unit(mas[:, i], atr14, lookback=SLOPE_LOOKBACK, k=SLOPE_CLIP)
            for i in range(MARIBBON_N_MA)
        ],
        axis=1,
    )
    feats = np.column_stack([dists, slopes])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
