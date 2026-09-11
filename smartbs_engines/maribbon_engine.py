"""MA-ribbon feature engine for SmartBS.

OHLCV in, 48 MA-ribbon channels out. Entry, exit, and stops live in the Risk
Manager / PositionManager and are independent of this engine.

Eleven EMAs → 44 channels (soft dir, ATR dist, 1-bar slope, 4-bar slope),
plus 4 continuous RSI channels = 48.

v2 geometry (A–E):
  A. All distances / ATR14 (not / price) so gold and NATGAS share a scale.
  B. ``dist_ema9`` = (close − EMA9) / ATR — same family as the fan, not a wick.
  C. ``angle_*`` replaced by ``slope4_*`` (4-bar ΔMA / ATR).
  D. Soft direction: clip((close − MA) / ATR) — no hard 0 on cross bars.
  E. RSI one-hots replaced by continuous mid / slope / band-edge channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# EMA9, EMA14, EMA24, EMA40, EMA60, EMA100, EMA160, EMA240, EMA320, EMA420, EMA500
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
    ("ema420", "ema", 420),
    ("ema500", "ema", 500),
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

# Soft direction clip in ATR units (D).
DIR_CLIP = 2.0

# RSI occupies the last 4 channels. Continuous (E) so the TCN gets gradient
# through mid-band and turns, not just ternary flags.
MARIBBON_RSI_NAMES: tuple[str, ...] = (
    "rsi_norm",  # RSI / 100 ∈ [0, 1]
    "rsi_vs_mid",  # (RSI − 50) / 50
    "rsi_slope",  # 1-bar ΔRSI / 10, clipped
    "rsi_to_band",  # signed distance to nearest 30/70 edge / 20
)

MARIBBON_FEATURE_NAMES: list[str] = (
    [f"dir_{n}" for n in MARIBBON_MA_NAMES]
    + [f"dist_{n}" for n in MARIBBON_MA_NAMES]
    + [f"slope_{n}" for n in MARIBBON_MA_NAMES]
    + [f"slope4_{n}" for n in MARIBBON_MA_NAMES]
    + list(MARIBBON_RSI_NAMES)
)

MARIBBON_NUM_INPUTS = len(MARIBBON_FEATURE_NAMES)  # 48 = 11*4 + 4
SLOPE_LOOKBACK = 1  # 1-bar MA change, ATR-normalized
SLOPE4_LOOKBACK = 4  # replaces angle_* (C)


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


def _atr_safe(atr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(atr) & (atr > 1e-12), atr, np.nan)


def _ma_soft_dir(close: np.ndarray, ma: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """(D) Soft direction: clip((close − MA) / ATR). Cross bars keep signed penetration."""
    denom = _atr_safe(atr)
    raw = (close - ma) / denom
    out = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), -DIR_CLIP, DIR_CLIP)
    out = np.where(~np.isfinite(ma), 0.0, out)
    return out.astype(np.float64)


def _norm_slope(ma: np.ndarray, atr: np.ndarray, lookback: int = SLOPE_LOOKBACK) -> np.ndarray:
    """NormalizedSlope = (MA − MA[lookback]) / ATR."""
    prev = np.roll(ma, lookback)
    prev[:lookback] = ma[:lookback]
    denom = _atr_safe(atr)
    slope = (ma - prev) / denom
    return np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)


def _rsi_to_band(rsi: np.ndarray) -> np.ndarray:
    """Signed distance to the nearer of 30 / 70, in units of 20 RSI points.

    Inside the band → negative (room to edge). Outside → positive (how far past).
    """
    # Distance above upper or below lower when extreme; else −min(rsi−30, 70−rsi).
    above = rsi - RSI_UPPER
    below = RSI_LOWER - rsi
    outside = np.maximum(above, below)  # >0 when outside either edge
    inside_room = np.minimum(rsi - RSI_LOWER, RSI_UPPER - rsi)  # >0 inside
    raw = np.where(outside > 0, outside, -inside_room)
    return np.clip(raw / 20.0, -3.0, 3.0)


@dataclass
class MARibbonEngine:
    """Ribbon state + 48-channel feature matrix aligned to bar index."""

    mas: dict[str, np.ndarray]
    atr14: np.ndarray
    rsi14: np.ndarray
    features: np.ndarray
    # First bar whose ribbon is fully warmed up. Rows before this are filled in
    # (see below) and are not comparable to training-time features.
    valid_from: int


def compute_maribbon_engine(df: pd.DataFrame) -> MARibbonEngine:
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    n = len(close)
    atr14 = _atr(high, low, close, 14)
    denom = _atr_safe(atr14)

    mas: dict[str, np.ndarray] = {}
    for name, kind, length in MARIBBON_SPECS:
        raw = _ema(close, length) if kind == "ema" else _sma(close, length)
        mas[name] = np.nan_to_num(raw, nan=close, posinf=close, neginf=close)

    # (D) Soft ATR-normalized direction — no hard zero on straddles.
    dirs = np.stack(
        [_ma_soft_dir(close, mas[name], atr14) for name in MARIBBON_MA_NAMES], axis=1
    )

    # (A)(B) Fan distances in ATR units. dist_ema9 = close−EMA9; rest = faster−slower.
    dists = np.zeros((n, len(MARIBBON_MA_NAMES)), dtype=np.float64)
    dists[:, 0] = np.nan_to_num((close - mas["ema9"]) / denom, nan=0.0, posinf=0.0, neginf=0.0)
    names = MARIBBON_MA_NAMES
    for i in range(1, len(names)):
        dists[:, i] = np.nan_to_num(
            (mas[names[i - 1]] - mas[names[i]]) / denom,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    slopes = np.stack(
        [_norm_slope(mas[name], atr14, SLOPE_LOOKBACK) for name in MARIBBON_MA_NAMES],
        axis=1,
    )
    # (C) 4-bar slope block replaces arctan(slope) angles.
    slope4 = np.stack(
        [_norm_slope(mas[name], atr14, SLOPE4_LOOKBACK) for name in MARIBBON_MA_NAMES],
        axis=1,
    )

    # (E) Continuous RSI channels.
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

    features = np.concatenate([dirs, dists, slopes, slope4, rsi_block], axis=1)
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
    """(n, 48) matrix: soft-dir/dist/slope/slope4 x 11 EMAs, then 4 RSI channels."""
    return compute_maribbon_engine(df).features
