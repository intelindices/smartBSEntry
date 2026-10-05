"""Shared BOS / ChoCH / FVG / sweep structure primitives for SmartMoney."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.registry import atr_unit, safe_div

HOUR_MS = 3_600_000
M15_MS = 900_000
H4_MS = 14_400_000
DAY_MS = 86_400_000


def infer_bar_ms(df: pd.DataFrame) -> int:
    """Median bar duration in ms from open_time (defaults to 1h)."""
    if df is None or len(df) < 2:
        return HOUR_MS
    diffs = np.diff(df["open_time"].to_numpy(dtype=np.int64))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return HOUR_MS
    return int(np.median(diffs))


def is_15m_bars(df: pd.DataFrame) -> bool:
    ms = infer_bar_ms(df)
    return 600_000 <= ms <= 1_200_000
DECAY_CAP = 50


def prev_array(a: np.ndarray) -> np.ndarray:
    p = np.roll(a, 1)
    p[0] = a[0]
    return p


def bars_since_decay(flag: np.ndarray, cap: int = DECAY_CAP) -> np.ndarray:
    """1 at event bar, linear decay to 0 over ``cap`` bars."""
    n = len(flag)
    out = np.zeros(n, dtype=np.float64)
    last = -1
    for i in range(n):
        if flag[i]:
            last = i
        if last >= 0:
            out[i] = max(0.0, 1.0 - (i - last) / float(cap))
    return out


def compute_structure_block(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    swing_len: int,
) -> dict[str, np.ndarray]:
    """Structure channels on one OHLCV series (1h or 15m)."""
    from smartbs_engines.structure import swing_levels

    sh, sl = swing_levels(high, low, swing_len)
    sh = np.nan_to_num(sh, nan=high[0])
    sl = np.nan_to_num(sl, nan=low[0])
    p_sh, p_sl = prev_array(sh), prev_array(sl)

    broke_up = close > p_sh
    broke_dn = close < p_sl
    new_bos_up = broke_up & ~prev_array(broke_up)
    new_bos_dn = broke_dn & ~prev_array(broke_dn)
    choch_up = new_bos_up & prev_array(broke_dn)
    choch_dn = new_bos_dn & prev_array(broke_up)

    swept_up = np.clip(safe_div(high - p_sh, atr14), 0.0, 3.0) / 3.0 * (close <= p_sh).astype(
        np.float64
    )
    swept_dn = np.clip(safe_div(p_sl - low, atr14), 0.0, 3.0) / 3.0 * (close >= p_sl).astype(
        np.float64
    )

    h2, l2 = prev_array(prev_array(high)), prev_array(prev_array(low))
    fvg_up = np.clip(low - h2, 0.0, None)
    fvg_dn = np.clip(l2 - high, 0.0, None)
    fvg_up_sz = atr_unit(fvg_up, atr14, k=3.0)
    fvg_dn_sz = atr_unit(fvg_dn, atr14, k=3.0)

    # Distance to nearest active gap edge (0 when inside the gap), ATR-unit [-1, 1].
    dist_fvg_up = np.where(
        fvg_up > 0, atr_unit(close - h2, atr14, k=3.0), atr_unit(close - low, atr14, k=3.0)
    )
    dist_fvg_dn = np.where(
        fvg_dn > 0, atr_unit(l2 - close, atr14, k=3.0), atr_unit(high - close, atr14, k=3.0)
    )
    fvg_prox_up = np.clip(1.0 - np.abs(dist_fvg_up), 0.0, 1.0) * (fvg_up_sz > 0).astype(np.float64)
    fvg_prox_dn = np.clip(1.0 - np.abs(dist_fvg_dn), 0.0, 1.0) * (fvg_dn_sz > 0).astype(np.float64)

    return {
        "bos_up_decay": bars_since_decay(new_bos_up.astype(np.float64)),
        "bos_dn_decay": bars_since_decay(new_bos_dn.astype(np.float64)),
        "choch_up_decay": bars_since_decay(choch_up.astype(np.float64)),
        "choch_dn_decay": bars_since_decay(choch_dn.astype(np.float64)),
        "fvg_up_sz": fvg_up_sz,
        "fvg_dn_sz": fvg_dn_sz,
        "dist_fvg_up": dist_fvg_up,
        "dist_fvg_dn": dist_fvg_dn,
        "fvg_prox_up": fvg_prox_up,
        "fvg_prox_dn": fvg_prox_dn,
        "sweep_up": swept_up,
        "sweep_dn": swept_dn,
        "dist_sh": atr_unit(close - sh, atr14, k=3.0),
        "dist_sl": atr_unit(close - sl, atr14, k=3.0),
    }


def map_15m_end_of_hour_to_1h(
    times_1h: np.ndarray,
    times_15m: np.ndarray,
    values_15m: np.ndarray,
) -> np.ndarray:
    """Snapshot each 15m series at the last 15m bar inside each 1h bucket."""
    n = len(times_1h)
    out = np.zeros(n, dtype=np.float64)
    if len(times_15m) == 0:
        return out
    frame = pd.DataFrame({"t": times_15m, "v": values_15m})
    frame["hour"] = (frame["t"] // HOUR_MS) * HOUR_MS
    last_by_hour = frame.groupby("hour", sort=False)["v"].last()
    mapped = pd.Series(times_1h).map(last_by_hour)
    return mapped.ffill().fillna(0.0).to_numpy(dtype=np.float64)


def load_aligned_15m(
    df_1h: pd.DataFrame,
    *,
    symbol: str | None,
    data_source: str | None,
) -> pd.DataFrame | None:
    """Load 15m OHLCV covering the primary window when cache/API allows."""
    return _load_aligned_interval(df_1h, interval="15m", symbol=symbol, data_source=data_source)


def load_aligned_5m(
    df_1h: pd.DataFrame,
    *,
    symbol: str | None,
    data_source: str | None,
) -> pd.DataFrame | None:
    """Load 5m OHLCV covering the primary window when cache/API allows."""
    return _load_aligned_interval(df_1h, interval="5m", symbol=symbol, data_source=data_source)


def load_aligned_1h(
    df_primary: pd.DataFrame,
    *,
    symbol: str | None,
    data_source: str | None,
) -> pd.DataFrame | None:
    """Load 1h OHLCV covering the primary window when cache/API allows."""
    return _load_aligned_interval(df_primary, interval="1h", symbol=symbol, data_source=data_source)


def load_aligned_4h(
    df_1h: pd.DataFrame,
    *,
    symbol: str | None,
    data_source: str | None,
) -> pd.DataFrame | None:
    """Load 4h OHLCV covering the primary window when cache/API allows."""
    return _load_aligned_interval(df_1h, interval="4h", symbol=symbol, data_source=data_source)


def load_aligned_1d(
    df_1h: pd.DataFrame,
    *,
    symbol: str | None,
    data_source: str | None,
) -> pd.DataFrame | None:
    """Load daily OHLCV covering the primary window when cache/API allows.

    Falls back to aggregating the primary bars into UTC days when 1d history
    is unavailable (common when only 1h parquet cache exists).
    """
    out = _load_aligned_interval(df_1h, interval="1d", symbol=symbol, data_source=data_source)
    if out is not None and len(out):
        return out
    return aggregate_daily_ohlcv(df_1h)


def aggregate_daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
    """Build D1 OHLCV from lower-TF bars (UTC day buckets)."""
    if df is None or df.empty:
        return None
    day = (df["open_time"].to_numpy(dtype=np.int64) // DAY_MS) * DAY_MS
    frame = df.copy()
    frame["_day"] = day
    rows: list[dict] = []
    has_vol = "volume" in frame.columns
    for d, g in frame.groupby("_day", sort=True):
        rows.append(
            {
                "open_time": int(d),
                "open": float(g["open"].iloc[0]),
                "high": float(g["high"].max()),
                "low": float(g["low"].min()),
                "close": float(g["close"].iloc[-1]),
                "volume": float(g["volume"].sum()) if has_vol else 0.0,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def _load_aligned_interval(
    df_1h: pd.DataFrame,
    *,
    interval: str,
    symbol: str | None,
    data_source: str | None,
) -> pd.DataFrame | None:
    if df_1h is None or df_1h.empty or not symbol:
        return None
    sym = str(symbol).replace("/", "").upper()
    src = (data_source or "").lower()
    t0 = int(df_1h["open_time"].iloc[0])
    pad = max(infer_bar_ms(df_1h), HOUR_MS)
    t1 = int(df_1h["open_time"].iloc[-1]) + pad
    iv = interval.strip().lower()

    if src in ("dukascopy", "duka"):
        from smartbs_engines.dukascopy import load_dukascopy_klines

        df_htf = load_dukascopy_klines(sym, interval=iv, max_candles=None)
        if df_htf is None or df_htf.empty:
            return None
        mask = (df_htf["open_time"] >= t0) & (df_htf["open_time"] < t1)
        out = df_htf.loc[mask].reset_index(drop=True)
        return out if len(out) else None

    if src in ("mt5", "metatrader", "broker"):
        from smartbs_engines.mt5_data import load_mt5_klines

        df_htf = load_mt5_klines(sym, iv, max_candles=None, refresh=False)
        if df_htf is None or df_htf.empty:
            return None
        mask = (df_htf["open_time"] >= t0) & (df_htf["open_time"] < t1)
        out = df_htf.loc[mask].reset_index(drop=True)
        return out if len(out) else None

    if src in ("yahoo", "tradingview", "binance"):
        from smartbs_engines.data import fetch_klines

        mult = 4 if iv in ("15m", "15min") else 1
        need = min(max(len(df_1h) * mult + 200, 400), 5000)
        try:
            df_htf = fetch_klines(
                symbol=sym,
                interval=iv,
                max_candles=need,
                source=src if src != "tradingview" else "tradingview",
            )
        except Exception:
            return None
        if df_htf is None or df_htf.empty:
            return None
        mask = (df_htf["open_time"] >= t0) & (df_htf["open_time"] < t1)
        out = df_htf.loc[mask].reset_index(drop=True)
        return out if len(out) else None

    return None
