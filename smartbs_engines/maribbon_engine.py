"""MA-ribbon feature engine for SmartBS.

OHLCV in, 22 MA-ribbon channels out. Entry, exit, and stops live in the Risk
Manager / PositionManager and are independent of this engine.

Nine EMAs → dist (9) + slope (9) + 4 RSI = 22.
Permanently dropped: EMA420, EMA500 (and former dist_ema420 skip).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# EMA9, EMA14, EMA24, EMA40, EMA60, EMA100, EMA160, EMA240, EMA320
MARIBBON_SPECS: tuple[tuple[str, str, int], ...] = (
    ("ema9", "ema", 9),
    ("ema14", "ema", 14),
    ("ema24", "ema", 24),
    ("ema40", "ema", 40),
    ("ema60", "ema", 60),
    ("ema100", "ema", 100),
    ("ema160", "ema", 160),
    ("ema240", "ema", 240),
    ("ema320", "ema", 320),
)

MARIBBON_MAX_MA_LEN = max(spec[2] for spec in MARIBBON_SPECS)

# An EMA(span=L, adjust=False) seeded at close[0] still carries ~5% of that seed
# after 3L bars. Feeding fewer bars than this silently produces a different
# ribbon than the one the model trained on.
MARIBBON_WARMUP_BARS = 3 * MARIBBON_MAX_MA_LEN

MARIBBON_MA_NAMES: tuple[str, ...] = tuple(spec[0] for spec in MARIBBON_SPECS)

RSI_PERIOD = 14
RSI_LOWER = 30.0
RSI_UPPER = 70.0
RSI_MID = 50.0

from smartbs_engines.registry import atr_unit, slope_unit

DIST_CLIP = 2.0
SLOPE_CLIP = 2.0

MARIBBON_RSI_NAMES: tuple[str, ...] = (
    "rsi_norm",
    "rsi_vs_mid",
    "rsi_slope",
    "rsi_to_band",
)

MARIBBON_FEATURE_NAMES: list[str] = (
    [f"dist_{n}" for n in MARIBBON_MA_NAMES]
    + [f"slope_{n}" for n in MARIBBON_MA_NAMES]
    + list(MARIBBON_RSI_NAMES)
)

MARIBBON_NUM_INPUTS = len(MARIBBON_FEATURE_NAMES)  # 22 = 9 dist + 9 slope + 4 RSI
SLOPE_LOOKBACK = 1


def _ema(series: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(series).ewm(span=length, adjust=False).mean().to_numpy(dtype=np.float64)


def _sma(series: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(series).rolling(length, min_periods=length).mean().to_numpy(dtype=np.float64)


def _rsi(close: np.ndarray, period: int = RSI_PERIOD) -> np.ndarray:
    from smartbs_engines.structure import rsi

    return rsi(close, period)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).ewm(alpha=1.0 / length, adjust=False).mean().to_numpy(dtype=np.float64)


def _norm_slope(
    ma: np.ndarray,
    atr: np.ndarray,
    lookback: int = SLOPE_LOOKBACK,
    k: float = SLOPE_CLIP,
) -> np.ndarray:
    return slope_unit(ma, atr, lookback=lookback, k=k)


def _rsi_to_band(rsi: np.ndarray) -> np.ndarray:
    """Signed distance to the nearer of 30 / 70, in units of 20 RSI points."""
    above = rsi - RSI_UPPER
    below = RSI_LOWER - rsi
    outside = np.maximum(above, below)
    inside_room = np.minimum(rsi - RSI_LOWER, RSI_UPPER - rsi)
    raw = np.where(outside > 0, outside, -inside_room)
    return np.clip(raw / 20.0, -3.0, 3.0)


@dataclass
class MARibbonEngine:
    """Ribbon state + feature matrix aligned to bar index."""

    mas: dict[str, np.ndarray]
    atr14: np.ndarray
    rsi14: np.ndarray
    features: np.ndarray
    valid_from: int


def compute_maribbon_engine(df: pd.DataFrame) -> MARibbonEngine:
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    n = len(close)
    atr14 = _atr(high, low, close, 14)

    mas: dict[str, np.ndarray] = {}
    for name, kind, length in MARIBBON_SPECS:
        raw = _ema(close, length) if kind == "ema" else _sma(close, length)
        mas[name] = np.nan_to_num(raw, nan=close, posinf=close, neginf=close)

    names = MARIBBON_MA_NAMES
    dists = np.zeros((n, len(names)), dtype=np.float64)
    dists[:, 0] = atr_unit(close - mas["ema9"], atr14, k=DIST_CLIP)
    for i in range(1, len(names)):
        dists[:, i] = atr_unit(mas[names[i - 1]] - mas[names[i]], atr14, k=DIST_CLIP)

    slopes = np.stack(
        [_norm_slope(mas[name], atr14, SLOPE_LOOKBACK, k=SLOPE_CLIP) for name in names],
        axis=1,
    )

    rsi14 = _rsi(close, RSI_PERIOD)
    rsi_prev = np.roll(rsi14, 1)
    rsi_prev[0] = rsi14[0]
    rsi_slope = np.clip((rsi14 - rsi_prev) / 10.0, -3.0, 3.0)
    rsi_block = np.stack(
        [
            rsi14 / 100.0,
            (rsi14 - RSI_MID) / 50.0,
            rsi_slope,
            _rsi_to_band(rsi14),
        ],
        axis=1,
    )

    features = np.concatenate([dists, slopes, rsi_block], axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if features.shape != (n, MARIBBON_NUM_INPUTS):
        raise ValueError(f"maribbon features {features.shape} != ({n}, {MARIBBON_NUM_INPUTS})")
    return MARibbonEngine(
        mas=mas,
        atr14=atr14,
        rsi14=rsi14,
        features=features,
        valid_from=min(n, MARIBBON_WARMUP_BARS),
    )


def build_maribbon_features(df: pd.DataFrame) -> np.ndarray:
    """(n, 22) matrix: dist×9 + slope×9 EMAs, then 4 RSI channels."""
    return compute_maribbon_engine(df).features
