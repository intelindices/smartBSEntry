"""Common channels appended to every ST engine feature matrix.

Attached via ``attach_common_channels`` (see ``registry.get_engine``).

Pack (16 channels), always last:

  OHLCV(1h) close-normalized: ohlcv_open, ohlcv_high, ohlcv_low,
                              ohlcv_volume (vol/SMA48, clip 0..3),
                              ohlcv_close_displacement (Δclose/prev_close, clip ±5%)
  timestamp (no date):        tod_sin, tod_cos
  session (UTC):              session_tokyo, session_london, session_newyork
  candle vs EMA14:            cd_vs_fast (−1/0/+1), cd_body_cross_fast
  session strength:           session_strength (final of last completed session),
                              current_session_strength (running sum in forming session)
  3-MA stack:                 trend_regime (+1 bull EMA14>48>120 / −1 bear / 0)
  candle body/range:          last_candle_strength (open−close)/(high−low)

Session hours (UTC) — default **normal** (overlapping FX):
  tokyo  00–09, london 07–16, newyork 12–21
Adjusted (non-overlapping) via ``set_session_hours_mode("adjusted")`` /
``--session-hours adjusted``: tokyo 00–07, london 07–12, newyork 12–20.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from smartbs_engines.registry import BaseSTEngine, EngineResult, safe_div
from smartbs_engines.structure import ema as _ema

DAY_MS = 86_400_000
HOUR_MS = 3_600_000

EMA_FAST = 14
EMA_MID = 48
EMA_SLOW = 120
COMMON_WARMUP = 3 * EMA_SLOW

# Default overlapping FX sessions (UTC).
SESSION_HOURS_NORMAL: tuple[tuple[str, int, int], ...] = (
    ("tokyo", 0, 9),
    ("london", 7, 16),
    ("newyork", 12, 21),
)
# Non-overlapping adjusted windows (opt-in via --session-hours adjusted).
SESSION_HOURS_ADJUSTED: tuple[tuple[str, int, int], ...] = (
    ("tokyo", 0, 7),
    ("london", 7, 12),
    ("newyork", 12, 20),
)

SESSION_HOURS_MODES: tuple[str, ...] = ("normal", "adjusted")

# Active pack (mutated by set_session_hours_mode). Default = normal.
SESSION_HOURS: tuple[tuple[str, int, int], ...] = SESSION_HOURS_NORMAL
_SESSION_HOURS: dict[str, tuple[int, int]] = {n: (lo, hi) for n, lo, hi in SESSION_HOURS}
_SESSION_HOURS_MODE: str = "normal"


def normalize_session_hours_mode(mode: str | None) -> str:
    m = str(mode or "normal").strip().lower()
    if m in ("", "normal", "legacy", "classic", "fx", "default"):
        return "normal"
    if m in ("adj", "adjusted"):
        return "adjusted"
    raise ValueError(f"session_hours={mode!r} unknown; use normal | adjusted")


def session_hours_for_mode(mode: str | None = None) -> tuple[tuple[str, int, int], ...]:
    m = normalize_session_hours_mode(mode if mode is not None else _SESSION_HOURS_MODE)
    return SESSION_HOURS_ADJUSTED if m == "adjusted" else SESSION_HOURS_NORMAL


def set_session_hours_mode(mode: str | None = "normal") -> str:
    """Set process-wide session windows used by common channels + labels."""
    global SESSION_HOURS, _SESSION_HOURS, _SESSION_HOURS_MODE
    m = normalize_session_hours_mode(mode)
    SESSION_HOURS = session_hours_for_mode(m)
    _SESSION_HOURS = {n: (lo, hi) for n, lo, hi in SESSION_HOURS}
    _SESSION_HOURS_MODE = m
    return m


def get_session_hours_mode() -> str:
    return _SESSION_HOURS_MODE


def session_bounds(mode: str | None = None) -> tuple[tuple[int, int], ...]:
    """``(lo, hi)`` hour bounds in UTC for the given (or active) mode."""
    return tuple((lo, hi) for _, lo, hi in session_hours_for_mode(mode))


def max_session_hours(mode: str | None = None) -> int:
    return max(hi - lo for lo, hi in session_bounds(mode))


COMMON_FEATURE_NAMES: tuple[str, ...] = (
    "ohlcv_open",
    "ohlcv_high",
    "ohlcv_low",
    "ohlcv_volume",
    "ohlcv_close_displacement",
    "tod_sin",
    "tod_cos",
    "session_tokyo",
    "session_london",
    "session_newyork",
    "cd_vs_fast",
    "cd_body_cross_fast",
    "session_strength",
    "current_session_strength",
    "trend_regime",
    "last_candle_strength",
)
COMMON_NUM_INPUTS = len(COMMON_FEATURE_NAMES)

# Signed close return clip (fraction of price).
CLOSE_DISP_CLIP = 0.05
VOL_SMA = 48
VOL_CLIP = 3.0


def _ohlcv_1h_cols(df: pd.DataFrame) -> list[np.ndarray]:
    n = len(df)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    if "volume" in df.columns:
        v = df["volume"].to_numpy(dtype=np.float64)
    else:
        v = np.zeros(n, dtype=np.float64)

    o_n = safe_div(o, c)
    h_n = safe_div(h, c)
    l_n = safe_div(l, c)
    vol_ma = pd.Series(v).rolling(VOL_SMA, min_periods=1).mean().to_numpy(dtype=np.float64)
    v_n = np.clip(safe_div(v, vol_ma), 0.0, VOL_CLIP)

    # Signed bar return: (close - prev_close) / prev_close, clipped ±5%.
    close_disp = np.zeros(n, dtype=np.float64)
    if n >= 2:
        prev = c[:-1]
        ret = safe_div(c[1:] - prev, prev)
        close_disp[1:] = np.clip(ret, -CLOSE_DISP_CLIP, CLOSE_DISP_CLIP)

    bad = ~(np.isfinite(c) & (np.abs(c) > 1e-12))
    if np.any(bad):
        o_n[bad] = 1.0
        h_n[bad] = 1.0
        l_n[bad] = 1.0
        v_n[bad] = 0.0
        close_disp[bad] = 0.0
    return [o_n, h_n, l_n, v_n, close_disp]


def _tod_cols(times_ms: np.ndarray) -> list[np.ndarray]:
    frac = (times_ms.astype(np.int64) % DAY_MS).astype(np.float64) / float(DAY_MS)
    ang = 2.0 * np.pi * frac
    return [np.sin(ang), np.cos(ang)]


def _session_cols(times_ms: np.ndarray) -> list[np.ndarray]:
    hour = ((times_ms.astype(np.int64) % DAY_MS) // HOUR_MS).astype(np.int64)
    cols: list[np.ndarray] = []
    for key in ("tokyo", "london", "newyork"):
        lo, hi = _SESSION_HOURS[key]
        cols.append(((hour >= lo) & (hour < hi)).astype(np.float64))
    return cols


def _candle_direction_cols(
    open_: np.ndarray,
    close: np.ndarray,
    ma_fast: np.ndarray,
) -> list[np.ndarray]:
    # Signed close vs fast EMA: +1 above, −1 below, 0 equal.
    vs = np.sign(close - ma_fast)
    lo = np.minimum(open_, close)
    hi = np.maximum(open_, close)
    body_cross = ((lo < ma_fast) & (hi > ma_fast)).astype(np.float64)
    return [vs.astype(np.float64), body_cross]


def _bar_session_score(
    open_: np.ndarray,
    close: np.ndarray,
    ma_fast: np.ndarray,
) -> np.ndarray:
    """Per-bar score: body-cross → 0; else close>MA → +1; close<MA → −1."""
    lo = np.minimum(open_, close)
    hi = np.maximum(open_, close)
    body_cross = (lo < ma_fast) & (hi > ma_fast)
    score = np.sign(close - ma_fast).astype(np.float64)
    score[body_cross] = 0.0
    score[~np.isfinite(ma_fast)] = 0.0
    return score


def _session_strength_pair(
    times_ms: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    ma_fast: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(session_strength, current_session_strength)``.

    ``current_session_strength`` — running sum of bar scores in the forming UTC
    session (0 off-hours / 1h before session open).

    ``session_strength`` — final strength of the **last completed** session
    (carried forward); 0 before any session ends.
    """
    from smartbs_engines.labels import session_start_end_ms

    n = len(close)
    last_str = np.zeros(n, dtype=np.float64)
    cur_str = np.zeros(n, dtype=np.float64)
    if n == 0:
        return last_str, cur_str

    score = _bar_session_score(open_, close, ma_fast)
    start, end = session_start_end_ms(times_ms)
    times = np.asarray(times_ms, dtype=np.int64)

    run = 0.0
    prev_start = -2
    completed = 0.0
    have_completed = False

    for i in range(n):
        s0 = int(start[i])
        if s0 < 0:
            run = 0.0
            prev_start = -1
            cur_str[i] = 0.0
            last_str[i] = completed if have_completed else 0.0
            continue

        if s0 != prev_start:
            run = 0.0
            prev_start = s0
        run += float(score[i])
        cur_str[i] = run
        last_str[i] = completed if have_completed else 0.0

        e0 = int(end[i])
        if i + 1 >= n or int(times[i + 1]) >= e0 or int(start[i + 1]) != s0:
            completed = run
            have_completed = True

    return last_str, cur_str


def _trend_regime_col(close: np.ndarray) -> np.ndarray:
    """+1 when EMA14>EMA48>EMA120, −1 when fully inverted, else 0."""
    ma_fast = _ema(close, EMA_FAST)
    ma_mid = _ema(close, EMA_MID)
    ma_slow = _ema(close, EMA_SLOW)
    bull = (ma_fast > ma_mid) & (ma_mid > ma_slow)
    bear = (ma_fast < ma_mid) & (ma_mid < ma_slow)
    out = np.zeros(len(close), dtype=np.float64)
    out[bull] = 1.0
    out[bear] = -1.0
    return out


def _last_candle_strength_col(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """(open − close) / (high − low); 0 when range is flat."""
    rng = high - low
    return safe_div(open_ - close, rng)


def compute_common_channels(df: pd.DataFrame) -> np.ndarray:
    """Return ``(n, COMMON_NUM_INPUTS)`` float32 common pack."""
    n = len(df)
    if n == 0:
        return np.zeros((0, COMMON_NUM_INPUTS), dtype=np.float32)

    times = df["open_time"].to_numpy(dtype=np.int64)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    ma_fast = _ema(c, EMA_FAST)

    cols: list[np.ndarray] = []
    cols.extend(_ohlcv_1h_cols(df))
    cols.extend(_tod_cols(times))
    cols.extend(_session_cols(times))
    cols.extend(_candle_direction_cols(o, c, ma_fast))
    last_s, cur_s = _session_strength_pair(times, o, c, ma_fast)
    cols.append(last_s)
    cols.append(cur_s)
    cols.append(_trend_regime_col(c))
    cols.append(_last_candle_strength_col(o, h, l, c))

    if len(cols) != COMMON_NUM_INPUTS:
        raise ValueError(f"common: built {len(cols)} channels, expected {COMMON_NUM_INPUTS}")
    out = np.stack(cols, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _call_compute(eng: BaseSTEngine, df: pd.DataFrame, **kwargs) -> EngineResult:
    params = inspect.signature(eng.compute).parameters
    call_kw = {k: v for k, v in kwargs.items() if k in params}
    return eng.compute(df, **call_kw)


class CommonChannelEngine(BaseSTEngine):
    """Wrap any ST engine and append the common channel pack."""

    _has_common_channels = True

    def __init__(self, inner: BaseSTEngine) -> None:
        if getattr(inner, "_has_common_channels", False):
            raise ValueError(f"engine {inner.name!r} already has common channels")
        self._inner = inner
        self.name = inner.name
        self.feature_names = list(inner.feature_names) + list(COMMON_FEATURE_NAMES)
        self.warmup_bars = max(int(inner.warmup_bars), COMMON_WARMUP)

    def compute(self, df: pd.DataFrame, **kwargs) -> EngineResult:
        n = len(df)
        core = _call_compute(self._inner, df, **kwargs)
        feats = np.asarray(core.features, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] != n:
            raise ValueError(
                f"{self.name}: core features {feats.shape} incompatible with n={n}"
            )
        common = compute_common_channels(df)
        stacked = np.concatenate([feats, common], axis=1)
        if stacked.shape != (n, self.num_inputs):
            raise ValueError(
                f"{self.name}+common: stacked {stacked.shape} != ({n}, {self.num_inputs})"
            )
        valid = max(int(core.valid_from), min(n, self.warmup_bars))
        return EngineResult(features=stacked, valid_from=valid)


def attach_common_channels(eng: BaseSTEngine) -> BaseSTEngine:
    if getattr(eng, "_has_common_channels", False):
        return eng
    return CommonChannelEngine(eng)
