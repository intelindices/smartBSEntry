"""SmartBSEntryEngine — compact multi-TF entry features.

Six major/minor pairs (week/day … 5m/1m). Per pair, only these signed channels:

  candle:  strength, dist, movement, dir
  regime:  fast_dir, slow_dir, long_dir, regime_status
  smart$:  choch, bos, fgv   (bull +, bear −)
  RSI:     bull/bear reverse + bull/bear entry (bull +, bear −)

Primary bar series for train/infer remains **1h**; coarser/finer TFs are
resampled or loaded and snapshotted onto each 1h bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_entry.structure_sm import (
    bars_since_decay,
    compute_structure_block,
    prev_array,
)
from smartbs_entry.registry import BaseSTEngine, EngineResult, safe_div
from smartbs_entry.indicators import atr as _atr, ema as _ema, rsi as _rsi

# MA ribbon used for regime / candle_dist / candle_dir (fast = shortest).
MA_FAST = 8
MA_MID = 21
MA_SLOW = 50
MA_LONG = 100
RSI_PERIOD = 14
RSI_LOW = 30.0
RSI_HIGH = 70.0
SWING_LEN = 5
DECAY_CAP = 24

# (major_tf, minor_tf, name_prefix)
TF_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("1w", "1d", "wd"),
    ("1d", "4h", "d4h"),
    ("4h", "1h", "4h1h"),
    ("1h", "15m", "1h15m"),
    ("15m", "5m", "15m5m"),
    ("5m", "1m", "5m1m"),
)

PAIR_CHANNELS: tuple[str, ...] = (
    "candle_strength",
    "candle_dist",
    "candle_movement",
    "candle_dir",
    "fast_dir",
    "slow_dir",
    "long_dir",
    "regime_status",
    "choch",
    "bos",
    "fgv",
    "rsi_bull_reverse",
    "rsi_bear_reverse",
    "rsi_bull_entry",
    "rsi_bear_entry",
)

FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{prefix}_{ch}" for _, _, prefix in TF_PAIRS for ch in PAIR_CHANNELS
)

_RESAMPLE_RULE: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W-MON",
}


def _cross_up(level: float, series: np.ndarray) -> np.ndarray:
    prev = prev_array(series)
    return (prev <= level) & (series > level)


def _cross_dn(level: float, series: np.ndarray) -> np.ndarray:
    prev = prev_array(series)
    return (prev >= level) & (series < level)


def _signed_decay(flag: np.ndarray, sign: float, cap: int = DECAY_CAP) -> np.ndarray:
    return sign * bars_since_decay(flag.astype(np.float64), cap=cap)


def _map_last_le(times_dst: np.ndarray, times_src: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map src series onto dst bars: last src bar with open_time <= dst open_time."""
    n = len(times_dst)
    out = np.zeros(n, dtype=np.float64)
    if len(times_src) == 0:
        return out
    idx = np.searchsorted(times_src, times_dst, side="right") - 1
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    if not valid[0]:
        # Leading dst bars before first src bar stay 0; ffill once we have data.
        first = int(np.argmax(valid)) if valid.any() else -1
        if first > 0:
            out[:first] = out[first]
    return out


def _resample_from_base(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _RESAMPLE_RULE[tf]
    if df is None or df.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    g = df.set_index(ts).resample(rule, label="left", closed="left")
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum() if "volume" in df.columns else 0.0,
        }
    ).dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index().rename(columns={"index": "open_time"})
    out["open_time"] = (pd.to_datetime(out["open_time"], utc=True).astype("int64") // 10**6).astype(
        np.int64
    )
    return out


def _load_tf_frame(
    tf: str,
    *,
    df_1h: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if tf in frames and frames[tf] is not None and len(frames[tf]):
        return frames[tf]
    if tf == "1h":
        return df_1h
    # Prefer finer base when available.
    if tf in ("4h", "1d", "1w"):
        base = frames.get("1h", df_1h)
        out = _resample_from_base(base, tf)
        frames[tf] = out
        return out
    if tf == "15m":
        base = frames.get("15m")
        if base is None or base.empty:
            base = frames.get("5m")
            if base is not None and len(base):
                out = _resample_from_base(base, "15m")
            elif frames.get("1m") is not None and len(frames["1m"]):
                out = _resample_from_base(frames["1m"], "15m")
            else:
                out = _resample_from_base(df_1h, "15m")  # coarse fallback
            frames["15m"] = out
            return out
        return base
    if tf == "5m":
        base = frames.get("5m")
        if base is not None and len(base):
            return base
        if frames.get("1m") is not None and len(frames["1m"]):
            out = _resample_from_base(frames["1m"], "5m")
            frames["5m"] = out
            return out
        # Fallback: synthesize 5m from 15m (3 identical sub-slices is wrong);
        # use 15m as stand-in so channels stay defined.
        fb = frames.get("15m")
        if fb is not None and len(fb):
            frames["5m"] = fb.copy()
            return frames["5m"]
        return _resample_from_base(df_1h, "5m")
    if tf == "1m":
        base = frames.get("1m")
        if base is not None and len(base):
            return base
        fb = frames.get("5m")
        if fb is not None and len(fb):
            frames["1m"] = fb.copy()
            return frames["1m"]
        fb = frames.get("15m")
        if fb is not None and len(fb):
            frames["1m"] = fb.copy()
            return frames["1m"]
        return df_1h.copy()
    raise KeyError(tf)


def _gather_frames(
    df_1h: pd.DataFrame,
    *,
    symbol: str | None,
    data_source: str | None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {"1h": df_1h}
    src = (data_source or "").lower()
    sym = (symbol or "").replace("/", "").upper() or None
    if src in ("dukascopy", "duka") and sym:
        from smartbs_entry.dukascopy import load_dukascopy_klines

        t0 = int(df_1h["open_time"].iloc[0])
        t1 = int(df_1h["open_time"].iloc[-1]) + 3_600_000
        for tf in ("15m", "5m", "1m"):
            raw = load_dukascopy_klines(sym, interval=tf, max_candles=None)
            if raw is None or raw.empty:
                continue
            mask = (raw["open_time"] >= t0 - 7 * 86_400_000) & (raw["open_time"] < t1)
            clipped = raw.loc[mask].reset_index(drop=True)
            if len(clipped):
                frames[tf] = clipped
    elif src and sym:
        from smartbs_entry.structure_sm import load_aligned_15m

        df15 = load_aligned_15m(df_1h, symbol=sym, data_source=src)
        if df15 is not None and len(df15):
            frames["15m"] = df15
    return frames


def _ohlc_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        df["open_time"].to_numpy(dtype=np.int64),
        df["open"].to_numpy(dtype=np.float64),
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
    )


def _candle_block(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
    rng = high - low
    strength = safe_div(open_ - close, rng)
    movement = safe_div(open_ - close, close)
    fast = _ema(close, MA_FAST)
    atr14 = _atr(high, low, close, 14)
    dist = safe_div(close - fast, atr14)
    lo = np.minimum(open_, close)
    hi = np.maximum(open_, close)
    between = (fast >= lo) & (fast <= hi)
    dir_ = np.where(between, 0.0, np.where(close > fast, 1.0, -1.0))
    return {
        "candle_strength": np.clip(strength, -1.0, 1.0),
        "candle_dist": np.clip(dist, -5.0, 5.0),
        "candle_movement": np.clip(movement * 100.0, -5.0, 5.0),
        "candle_dir": dir_.astype(np.float64),
    }


def _regime_block(close: np.ndarray) -> dict[str, np.ndarray]:
    f = _ema(close, MA_FAST)
    m = _ema(close, MA_MID)
    s = _ema(close, MA_SLOW)
    lng = _ema(close, MA_LONG)

    fast_dir = np.where(f > s, 1.0, -1.0)
    slow_dir = np.where(
        (f > s) & (s > m),
        1.0,
        np.where((f < s) & (s < m), -1.0, 0.0),
    )
    long_dir = np.where(
        (f > s) & (s > m) & (m > lng),
        1.0,
        np.where((f < s) & (s < m) & (m < lng), -1.0, 0.0),
    )

    spread = np.maximum.reduce([f, m, s, lng]) - np.minimum.reduce([f, m, s, lng])
    prev_spread = prev_array(spread)
    # Soft signed status: + expanding, − squeezing (ATR-free, price-norm).
    regime_status = np.tanh(safe_div(spread - prev_spread, np.maximum(np.abs(close), 1e-9) * 0.001))
    return {
        "fast_dir": fast_dir.astype(np.float64),
        "slow_dir": slow_dir.astype(np.float64),
        "long_dir": long_dir.astype(np.float64),
        "regime_status": regime_status.astype(np.float64),
    }


def _smart_money_signed(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> dict[str, np.ndarray]:
    atr14 = _atr(high, low, close, 14)
    block = compute_structure_block(
        high=high, low=low, close=close, atr14=atr14, swing_len=SWING_LEN
    )
    choch = block["choch_up_decay"] - block["choch_dn_decay"]
    bos = block["bos_up_decay"] - block["bos_dn_decay"]
    fgv = block["fvg_prox_up"] - block["fvg_prox_dn"]
    return {
        "choch": np.clip(choch, -1.0, 1.0),
        "bos": np.clip(bos, -1.0, 1.0),
        "fgv": np.clip(fgv, -1.0, 1.0),
    }


def _rsi_pair_block(
    times_1h: np.ndarray,
    t_maj: np.ndarray,
    c_maj: np.ndarray,
    t_min: np.ndarray,
    c_min: np.ndarray,
) -> dict[str, np.ndarray]:
    """Joint RSI setups via native-TF decays mapped onto 1h."""
    rsi_maj = _rsi(c_maj, RSI_PERIOD)
    rsi_min = _rsi(c_min, RSI_PERIOD)

    maj_dn70 = _map_last_le(times_1h, t_maj, bars_since_decay(_cross_dn(RSI_HIGH, rsi_maj).astype(np.float64)))
    min_dn70 = _map_last_le(times_1h, t_min, bars_since_decay(_cross_dn(RSI_HIGH, rsi_min).astype(np.float64)))
    maj_up30 = _map_last_le(times_1h, t_maj, bars_since_decay(_cross_up(RSI_LOW, rsi_maj).astype(np.float64)))
    min_up30 = _map_last_le(times_1h, t_min, bars_since_decay(_cross_up(RSI_LOW, rsi_min).astype(np.float64)))

    rsi_maj_1h = _map_last_le(times_1h, t_maj, rsi_maj)
    maj_band = ((rsi_maj_1h > RSI_LOW) & (rsi_maj_1h < RSI_HIGH)).astype(np.float64)

    # Joint: both sides recently active → product keeps sign via explicit ±.
    bull_rev = maj_dn70 * min_dn70
    bear_rev = maj_up30 * min_up30
    bull_entry = maj_band * min_up30
    bear_entry = maj_band * min_dn70

    return {
        "rsi_bull_reverse": np.clip(bull_rev, 0.0, 1.0),
        "rsi_bear_reverse": -np.clip(bear_rev, 0.0, 1.0),
        "rsi_bull_entry": np.clip(bull_entry, 0.0, 1.0),
        "rsi_bear_entry": -np.clip(bear_entry, 0.0, 1.0),
    }


def _pair_features_on_1h(
    times_1h: np.ndarray,
    df_maj: pd.DataFrame,
    df_min: pd.DataFrame,
) -> dict[str, np.ndarray]:
    t_maj, o_maj, h_maj, l_maj, c_maj = _ohlc_arrays(df_maj)
    t_min, o_min, h_min, l_min, c_min = _ohlc_arrays(df_min)

    candle = _candle_block(o_min, h_min, l_min, c_min)
    regime = _regime_block(c_maj)
    sm = _smart_money_signed(h_min, l_min, c_min)
    rsi = _rsi_pair_block(times_1h, t_maj, c_maj, t_min, c_min)

    out: dict[str, np.ndarray] = {}
    for name, arr in {**candle, **sm}.items():
        out[name] = _map_last_le(times_1h, t_min, arr)
    for name, arr in regime.items():
        out[name] = _map_last_le(times_1h, t_maj, arr)
    out.update(rsi)
    return out


class SmartBSEntryEngine(BaseSTEngine):
    """Compact signed multi-TF entry engine (``entry``)."""

    name = "entry"
    feature_names = list(FEATURE_NAMES)
    # ~100 weekly bars × ~5d × ~24h, with EMA slack.
    warmup_bars = 3 * MA_LONG * 24

    def compute(
        self,
        df: pd.DataFrame,
        *,
        symbol: str | None = None,
        data_source: str | None = None,
        frames: dict[str, pd.DataFrame] | None = None,
    ) -> EngineResult:
        n = len(df)
        times_1h = df["open_time"].to_numpy(dtype=np.int64)
        bag = dict(frames) if frames else _gather_frames(df, symbol=symbol, data_source=data_source)
        bag["1h"] = df

        cols: list[np.ndarray] = []
        for major_tf, minor_tf, prefix in TF_PAIRS:
            df_maj = _load_tf_frame(major_tf, df_1h=df, frames=bag)
            df_min = _load_tf_frame(minor_tf, df_1h=df, frames=bag)
            if df_maj is None or len(df_maj) < MA_LONG + 5:
                df_maj = df
            if df_min is None or len(df_min) < MA_LONG + 5:
                df_min = df
            block = _pair_features_on_1h(times_1h, df_maj, df_min)
            for ch in PAIR_CHANNELS:
                cols.append(block[ch])

        return self._finalize(cols, n)


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)
