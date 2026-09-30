"""SmartBSDbbEngine — double Bollinger close-zone location (1h + 15m).

Primary series: **1h**. Slim BB pack on 1h and on last completed 15m.

Channels (12):
  1h pack (6) + 15m pack (6)

Per-TF pack: body peak zone code + 5 close-in-zone one-hots.
Body%/range% occupancy dropped (close zone is the actionable signal).

Five zones (inner mid bands merged into one mid zone):
  upper_outer (+2), upper_outer_to_inner (+1), mid_zone (0),
  lower_inner_to_outer (-1), lower_outer (-2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smartbs_engines.registry import BaseSTEngine, EngineResult
from smartbs_engines.smart_money_structure import (
    HOUR_MS,
    is_15m_bars,
    load_aligned_15m,
    load_aligned_1h,
)
from smartbs_engines.structure import sma

BB_LENGTH = 20
BB_MULT_INNER = 1.0
BB_MULT_OUTER = 2.0
M15_MS = 900_000

# Per-symbol overrides from BB sweep v2 (independent 1h/15m lengths).
BB_PARAMS_BY_SYMBOL: dict[str, dict[str, float | int]] = {
    "XAUUSD": {"length_1h": 40, "length_15m": 20, "mult_inner": 2.0, "mult_outer": 3.0},
    "XAGUSD": {"length_1h": 35, "length_15m": 20, "mult_inner": 1.5, "mult_outer": 3.0},
    "XTIUSD": {"length_1h": 35, "length_15m": 40, "mult_inner": 2.0, "mult_outer": 3.0},
    "NATGAS": {"length_1h": 25, "length_15m": 25, "mult_inner": 2.0, "mult_outer": 3.0},
    "PLATINUM": {"length_1h": 35, "length_15m": 35, "mult_inner": 1.0, "mult_outer": 1.5},
}
WARMUP = max(256, 40 * 8)


def rolling_stdev(series: np.ndarray, length: int) -> np.ndarray:
    """Population stdev (ddof=0), matching TradingView ``ta.stdev`` default."""
    return (
        pd.Series(series)
        .rolling(length, min_periods=length)
        .std(ddof=0)
        .to_numpy(dtype=np.float64)
    )


def resolve_bb_params(
    symbol: str | None = None,
    *,
    length_1h: int | None = None,
    length_15m: int | None = None,
    mult1: float | None = None,
    mult2: float | None = None,
) -> tuple[int, int, float, float]:
    """Resolve (length_1h, length_15m, mult_inner, mult_outer)."""
    ov: dict[str, float | int] = {}
    if symbol:
        key = str(symbol).replace("/", "").upper()
        ov = dict(BB_PARAMS_BY_SYMBOL.get(key, {}))
    legacy = ov.get("length")
    L1 = int(
        length_1h
        if length_1h is not None
        else ov.get("length_1h", legacy if legacy is not None else BB_LENGTH)
    )
    L15 = int(
        length_15m
        if length_15m is not None
        else ov.get("length_15m", legacy if legacy is not None else BB_LENGTH)
    )
    m1 = float(mult1 if mult1 is not None else ov.get("mult_inner", BB_MULT_INNER))
    m2 = float(mult2 if mult2 is not None else ov.get("mult_outer", BB_MULT_OUTER))
    if m2 <= m1:
        raise ValueError(f"BB outer mult must exceed inner (got mult1={m1}, mult2={m2})")
    return L1, L15, m1, m2


_ZONE_SUFFIXES: tuple[str, ...] = (
    "upper_outer",
    "upper_outer_to_inner",
    "mid_zone",
    "lower_inner_to_outer",
    "lower_outer",
)

# mid_zone code is 0 (valid peak); doji / empty body still uses 0 only when best<=0
# so peak channel uses codes and falls back to 0 for empty — same numeric mid.
_ZONE_CODES: tuple[float, ...] = (2.0, 1.0, 0.0, -1.0, -2.0)

_BASE_CHANNEL_NAMES: tuple[str, ...] = (
    "body_peak_zone",
    *(f"close_in_{suf}" for suf in _ZONE_SUFFIXES),
)

FEATURE_NAMES: tuple[str, ...] = (
    *_BASE_CHANNEL_NAMES,
    *(f"{n}_15m" for n in _BASE_CHANNEL_NAMES),
)


def _map_last_completed(
    times_dst: np.ndarray,
    times_src: np.ndarray,
    values: np.ndarray,
    *,
    dur_ms: int,
) -> np.ndarray:
    n = len(times_dst)
    out = np.zeros(n, dtype=np.float64)
    if len(times_src) == 0:
        return out
    complete_by = times_src + int(dur_ms)
    idx = np.searchsorted(complete_by, times_dst, side="right") - 1
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


@dataclass(frozen=True)
class DoubleBollinger:
    """Double-BB geometry on one SMA basis."""

    basis: np.ndarray
    upper1: np.ndarray
    lower1: np.ndarray
    upper2: np.ndarray
    lower2: np.ndarray
    stdev: np.ndarray


def double_bollinger(
    close: np.ndarray,
    *,
    length: int = BB_LENGTH,
    mult1: float = BB_MULT_INNER,
    mult2: float = BB_MULT_OUTER,
) -> DoubleBollinger:
    """Two Bollinger envelopes on one SMA basis."""
    basis = sma(close, length)
    sd = rolling_stdev(close, length)
    m1 = float(mult1)
    m2 = float(mult2)
    return DoubleBollinger(
        basis=basis,
        upper1=basis + m1 * sd,
        lower1=basis - m1 * sd,
        upper2=basis + m2 * sd,
        lower2=basis - m2 * sd,
        stdev=sd,
    )


def _frac_overlap(
    seg_lo: np.ndarray,
    seg_hi: np.ndarray,
    zone_lo: np.ndarray | float,
    zone_hi: np.ndarray | float,
) -> np.ndarray:
    """Fraction of [seg_lo, seg_hi] that overlaps [zone_lo, zone_hi] (closed)."""
    length = seg_hi - seg_lo
    ov_lo = np.maximum(seg_lo, zone_lo)
    ov_hi = np.minimum(seg_hi, zone_hi)
    overlap = np.maximum(ov_hi - ov_lo, 0.0)
    out = np.zeros_like(length, dtype=np.float64)
    ok = np.isfinite(length) & (length > 0.0)
    if isinstance(zone_lo, np.ndarray):
        ok &= np.isfinite(zone_lo)
    if isinstance(zone_hi, np.ndarray):
        ok &= np.isfinite(zone_hi)
    out[ok] = overlap[ok] / length[ok]
    return out


def region_occupancy_pct(
    seg_lo: np.ndarray,
    seg_hi: np.ndarray,
    bb: DoubleBollinger,
) -> list[np.ndarray]:
    """Five region occupancy fractions (top -> bottom), each in [0, 1].

    Mid zone merges former upper_inner_to_mid + mid_to_lower_inner
    into [lower1, upper1].
    """
    pos_inf = np.full_like(seg_lo, np.inf)
    neg_inf = np.full_like(seg_lo, -np.inf)
    return [
        _frac_overlap(seg_lo, seg_hi, bb.upper2, pos_inf),
        _frac_overlap(seg_lo, seg_hi, bb.upper1, bb.upper2),
        _frac_overlap(seg_lo, seg_hi, bb.lower1, bb.upper1),
        _frac_overlap(seg_lo, seg_hi, bb.lower2, bb.lower1),
        _frac_overlap(seg_lo, seg_hi, neg_inf, bb.lower2),
    ]


def body_peak_zone_code(body_pcts: list[np.ndarray]) -> np.ndarray:
    """Signed zone where the largest body share sits; 0 if none / doji.

    Mid zone peak is also coded 0, so empty-body and mid-peak share the value.
    """
    mat = np.stack(body_pcts, axis=1)
    codes = np.asarray(_ZONE_CODES, dtype=np.float64)
    peak = np.argmax(mat, axis=1)
    best = mat[np.arange(len(mat)), peak]
    out = np.zeros(len(mat), dtype=np.float64)
    ok = best > 0.0
    out[ok] = codes[peak[ok]]
    return out


def close_zone_onehots(close: np.ndarray, bb: DoubleBollinger) -> list[np.ndarray]:
    """Mutually exclusive one-hots: which zone the candle close sits in."""
    n = len(close)
    outs = [np.zeros(n, dtype=np.float64) for _ in range(5)]
    ok = (
        np.isfinite(close)
        & np.isfinite(bb.upper1)
        & np.isfinite(bb.lower1)
        & np.isfinite(bb.upper2)
        & np.isfinite(bb.lower2)
    )
    c = close
    # Top → bottom, closed on lower edge of each band gap except outer top.
    z0 = ok & (c >= bb.upper2)
    z1 = ok & ~z0 & (c >= bb.upper1)
    z2 = ok & ~z0 & ~z1 & (c >= bb.lower1)
    z3 = ok & ~z0 & ~z1 & ~z2 & (c >= bb.lower2)
    z4 = ok & ~z0 & ~z1 & ~z2 & ~z3
    outs[0][z0] = 1.0
    outs[1][z1] = 1.0
    outs[2][z2] = 1.0
    outs[3][z3] = 1.0
    outs[4][z4] = 1.0
    return outs


def pack_dbb_channels(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    length: int = BB_LENGTH,
    mult1: float = BB_MULT_INNER,
    mult2: float = BB_MULT_OUTER,
    bb: DoubleBollinger | None = None,
) -> list[np.ndarray]:
    """6 channels for one TF: peak zone + 5 close-in one-hots."""
    bands = bb if bb is not None else double_bollinger(
        close, length=length, mult1=mult1, mult2=mult2
    )
    body_lo = np.minimum(open_, close)
    body_hi = np.maximum(open_, close)
    body_pcts = region_occupancy_pct(body_lo, body_hi, bands)
    peak = body_peak_zone_code(body_pcts)
    close_oh = close_zone_onehots(close, bands)
    return [peak, *close_oh]


class SmartBSDbbEngine(BaseSTEngine):
    """dBB: 12 channels — 1h + last-completed 15m close-zone packs."""

    name = "dbb"
    feature_names = list(FEATURE_NAMES)
    warmup_bars = WARMUP

    def __init__(
        self,
        *,
        length_1h: int | None = None,
        length_15m: int | None = None,
        mult_inner: float | None = None,
        mult_outer: float | None = None,
    ) -> None:
        self._length_1h = length_1h
        self._length_15m = length_15m
        self._mult_inner = mult_inner
        self._mult_outer = mult_outer

    def compute(
        self,
        df: pd.DataFrame,
        *,
        symbol: str | None = None,
        data_source: str | None = None,
        frames: dict[str, pd.DataFrame] | None = None,
    ) -> EngineResult:
        n = len(df)
        times = df["open_time"].to_numpy(dtype=np.int64)
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        L1, L15, m1, m2 = resolve_bb_params(
            symbol,
            length_1h=self._length_1h,
            length_15m=self._length_15m,
            mult1=self._mult_inner,
            mult2=self._mult_outer,
        )
        cols_primary = pack_dbb_channels(o, h, l, c, length=L1, mult1=m1, mult2=m2)

        bag = dict(frames) if frames else {}
        if is_15m_bars(df):
            # Primary=15m → second pack is last-completed 1h (reuse _15m channel slots)
            if bag.get("1h") is None or not len(bag.get("1h", [])):
                bag["1h"] = load_aligned_1h(df, symbol=symbol, data_source=data_source)
            df_htf = bag["1h"]
            if df_htf is None or len(df_htf) == 0:
                cols_htf = [np.zeros(n, dtype=np.float64) for _ in _BASE_CHANNEL_NAMES]
            else:
                t_htf = df_htf["open_time"].to_numpy(dtype=np.int64)
                pack_htf = pack_dbb_channels(
                    df_htf["open"].to_numpy(dtype=np.float64),
                    df_htf["high"].to_numpy(dtype=np.float64),
                    df_htf["low"].to_numpy(dtype=np.float64),
                    df_htf["close"].to_numpy(dtype=np.float64),
                    length=L15,
                    mult1=m1,
                    mult2=m2,
                )
                cols_htf = [
                    _map_last_completed(times, t_htf, ch, dur_ms=HOUR_MS) for ch in pack_htf
                ]
            return self._finalize([*cols_primary, *cols_htf], n)

        if bag.get("15m") is None or not len(bag.get("15m", [])):
            bag["15m"] = load_aligned_15m(df, symbol=symbol, data_source=data_source)
        df15 = bag["15m"]
        if df15 is None or len(df15) == 0:
            cols_15m = [np.zeros(n, dtype=np.float64) for _ in _BASE_CHANNEL_NAMES]
        else:
            t15 = df15["open_time"].to_numpy(dtype=np.int64)
            pack15 = pack_dbb_channels(
                df15["open"].to_numpy(dtype=np.float64),
                df15["high"].to_numpy(dtype=np.float64),
                df15["low"].to_numpy(dtype=np.float64),
                df15["close"].to_numpy(dtype=np.float64),
                length=L15,
                mult1=m1,
                mult2=m2,
            )
            cols_15m = [
                _map_last_completed(times, t15, ch, dur_ms=M15_MS) for ch in pack15
            ]

        return self._finalize([*cols_primary, *cols_15m], n)


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)
