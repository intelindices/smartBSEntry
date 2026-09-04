"""Shared market-structure primitives: swings, ATR, EMA, RSI, SMA.

Owned by the Risk Manager and reused by feature engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(series).ewm(span=length, adjust=False).mean().to_numpy(dtype=np.float64)


def sma(series: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(series).rolling(length, min_periods=length).mean().to_numpy(dtype=np.float64)


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).ewm(alpha=1.0 / length, adjust=False).mean().to_numpy(dtype=np.float64)


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_gain = pd.Series(gain).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    return 100.0 - (100.0 / (1.0 + rs))


def pivothigh(high: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(left + right, n):
        pivot_i = i - right
        window = high[pivot_i - left : pivot_i + right + 1]
        if high[pivot_i] >= np.max(window) - 1e-15 and np.argmax(window) == left:
            out[i] = high[pivot_i]
    return out


def pivotlow(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(left + right, n):
        pivot_i = i - right
        window = low[pivot_i - left : pivot_i + right + 1]
        if low[pivot_i] <= np.min(window) + 1e-15 and np.argmin(window) == left:
            out[i] = low[pivot_i]
    return out


def valuewhen(condition: np.ndarray, source: np.ndarray) -> np.ndarray:
    n = len(source)
    out = np.full(n, np.nan, dtype=np.float64)
    last = np.nan
    for i in range(n):
        if condition[i]:
            last = source[i]
        out[i] = last
    return out


def swing_levels(high: np.ndarray, low: np.ndarray, swing_len: int) -> tuple[np.ndarray, np.ndarray]:
    ph = pivothigh(high, swing_len, swing_len)
    pl = pivotlow(low, swing_len, swing_len)
    return valuewhen(~np.isnan(ph), ph), valuewhen(~np.isnan(pl), pl)


FOUR_H_MS = 4 * 3_600_000


def compress_ohlc_4h(
    times_ms: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate 1h (or finer) bars into completed UTC 4h OHLC buckets.

    Returns ``(bucket_start_ms, open, high, low, close)`` sorted by time.
    Incomplete trailing buckets are dropped so values are causal once a
    4h window has closed.
    """
    ms = np.asarray(times_ms, dtype=np.int64).reshape(-1)
    o = np.asarray(open_, dtype=np.float64).reshape(-1)
    h = np.asarray(high, dtype=np.float64).reshape(-1)
    l = np.asarray(low, dtype=np.float64).reshape(-1)
    c = np.asarray(close, dtype=np.float64).reshape(-1)
    if not (len(ms) == len(o) == len(h) == len(l) == len(c)):
        raise ValueError("OHLC arrays must share length")
    if len(ms) == 0:
        empty = np.array([], dtype=np.float64)
        return np.array([], dtype=np.int64), empty, empty, empty, empty

    bucket = (ms // FOUR_H_MS) * FOUR_H_MS
    # Drop the forming (latest) bucket so OHLC is only from completed 4h windows.
    keep = bucket < int(bucket[-1])
    if not np.any(keep):
        empty = np.array([], dtype=np.float64)
        return np.array([], dtype=np.int64), empty, empty, empty, empty

    b = bucket[keep]
    o_k, h_k, l_k, c_k = o[keep], h[keep], l[keep], c[keep]
    starts: list[int] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    i = 0
    n = len(b)
    while i < n:
        start = int(b[i])
        j = i + 1
        while j < n and int(b[j]) == start:
            j += 1
        starts.append(start)
        opens.append(float(o_k[i]))
        highs.append(float(np.max(h_k[i:j])))
        lows.append(float(np.min(l_k[i:j])))
        closes.append(float(c_k[j - 1]))
        i = j
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(opens, dtype=np.float64),
        np.asarray(highs, dtype=np.float64),
        np.asarray(lows, dtype=np.float64),
        np.asarray(closes, dtype=np.float64),
    )


def regime_emas_4h_ffill(
    times_ms: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lengths: dict[str, int],
    kinds: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Regime MA series from 4h-compressed closes, forward-filled onto 1h bars.

    ``kinds`` maps name → ``\"ema\"`` or ``\"sma\"`` (default ema).
    """
    ctx = regime_4h_ffill(times_ms, open_, high, low, close, lengths, kinds=kinds)
    return {name: ctx[name] for name in lengths}


def regime_4h_ffill(
    times_ms: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lengths: dict[str, int],
    *,
    atr_length: int = 14,
    kinds: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """4h regime MAs + range/ATR context, forward-filled onto 1h bars.

    Keys: MA names from ``lengths``, plus ``range`` (4h high−low) and ``atr``.
    Until the first 4h bar completes (and series warm up), values are NaN.
    """
    n = len(times_ms)
    out = {name: np.full(n, np.nan, dtype=np.float64) for name in lengths}
    out["range"] = np.full(n, np.nan, dtype=np.float64)
    out["atr"] = np.full(n, np.nan, dtype=np.float64)
    b_ms, _o, b_high, b_low, b_close = compress_ohlc_4h(times_ms, open_, high, low, close)
    if len(b_close) == 0:
        return out
    kind_map = {str(k): str(v).lower() for k, v in (kinds or {}).items()}
    mas_4h: dict[str, np.ndarray] = {}
    for name, ln in lengths.items():
        kind = kind_map.get(name, "ema")
        if kind == "sma":
            mas_4h[name] = sma(b_close, int(ln))
        else:
            mas_4h[name] = ema(b_close, int(ln))
    range_4h = b_high - b_low
    atr_4h = atr(b_high, b_low, b_close, length=int(atr_length))
    ms = np.asarray(times_ms, dtype=np.int64).reshape(-1)
    cur_bucket = (ms // FOUR_H_MS) * FOUR_H_MS
    target = cur_bucket - FOUR_H_MS
    idx = np.searchsorted(b_ms, target, side="right") - 1
    valid = idx >= 0
    for name, series in mas_4h.items():
        out[name][valid] = series[idx[valid]]
    out["range"][valid] = range_4h[idx[valid]]
    out["atr"][valid] = atr_4h[idx[valid]]
    return out
