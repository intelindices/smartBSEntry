"""Triple-barrier labels: the training target, independent of the Risk Manager.

From each bar, place an upper barrier at ``+k*ATR`` and a lower barrier at
``-k*ATR`` and look forward at most ``horizon`` bars. Whichever barrier is
touched first names the class; if neither is touched the bar is FLAT.

Nothing here consults swing-R, entry gates, or model probabilities. Two
consequences follow:

1. The Risk Manager can be retuned without invalidating any checkpoint.
2. Labels no longer depend on a bootstrap model, so **two-pass labeling is not
   needed** — there is a single training pass.

Labels are also engine-independent, so they are computed once per asset and
reused across every ``STEngine``.

Also: ``label_next_direction`` — next-bar range break of the current candle:
LONG if next high > high; SHORT if next low < low; inside → FLAT; both-side
break → side with larger excursion wins.

Also: ``label_session_direction`` — session open→close return vs ±pct
(default ±0.5%): LONG / SHORT / FLAT for the whole session.

Also: ``label_session_trend`` — at each 1h bar in the pred window
(``[session_start−1h, session_end−1h)``), first touch of
``±k*ATR(14)`` before session end (default k=2): LONG / SHORT / FLAT.

Also: ``label_pivot_breakout`` — rolling extremes over ``pivot_length``:
update window high → LONG, update window low → SHORT; else first strict
break of the frozen range within the next ``L`` bars, else FLAT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartbs_engines.config import SmartBSConfig
from smartbs_engines.structure import atr as _atr

DEFAULT_BARRIER_K = 1.0
DEFAULT_BARRIER_HORIZON = 24
DEFAULT_SESSION_DIRECTION_PCT = 0.005  # ±0.5% open→close
DEFAULT_SESSION_TREND_K = 2.0  # ±2*ATR(14) vs session-end close
DEFAULT_SESSION_TREND_HORIZON = 0  # unused (resolved mask handles tails)
DEFAULT_SESSION_TREND_PCT = 0.0  # unused
DEFAULT_DAY_TREND_PCT = 0.01  # ±1% vs today's NY session end close
DEFAULT_PIVOT_BREAKOUT_LEN = 15  # rolling window / forward scan length
H4_MS = 14_400_000  # 4h bucket

DAY_MS = 86_400_000
HOUR_MS = 3_600_000

BARRIER_LABEL_MODES: tuple[str, ...] = (
    "triple_barrier",
    "pct_barrier",
    "session_direction",
    "session_trend",
    "day_trend",
    "pivot_breakout",
)


@dataclass(frozen=True)
class BarrierLabels:
    labels: np.ndarray  # CLASS_FLAT / CLASS_LONG / CLASS_SHORT
    resolved: np.ndarray  # bool: a barrier was actually touched within the horizon
    ambiguous: np.ndarray  # bool: both barriers touched on the same bar

    def stats(self) -> dict[str, float]:
        n = max(len(self.labels), 1)
        return {
            "n": float(len(self.labels)),
            "flat": float((self.labels == SmartBSConfig.CLASS_FLAT).sum()) / n,
            "long": float((self.labels == SmartBSConfig.CLASS_LONG).sum()) / n,
            "short": float((self.labels == SmartBSConfig.CLASS_SHORT).sum()) / n,
            "resolved": float(self.resolved.sum()) / n,
            "ambiguous": float(self.ambiguous.sum()) / n,
        }


def label_next_direction(
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    """Next-candle range-break labels (LONG / SHORT / FLAT).

    At bar ``t`` (closed candle), look at candle ``t+1``:
      - LONG  if ``high[t+1] > high[t]`` and not ``low[t+1] < low[t]``
      - SHORT if ``low[t+1] < low[t]`` and not ``high[t+1] > high[t]``
      - both breaks: compare ``high[t+1]-high[t]`` vs ``low[t]-low[t+1]``;
        larger distance wins; equal distances → FLAT
      - FLAT  if inside ``[low[t], high[t]]`` (neither side breaks), or last bar

    CLASS ids 0/1/2 keep the 3-logit head compatible. FLAT is a real train target.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if high.shape != low.shape:
        raise ValueError("high/low length mismatch")
    n = len(high)
    labels = np.full(n, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    if n < 2:
        return labels

    dist_up = high[1:] - high[:-1]
    dist_dn = low[:-1] - low[1:]
    broke_hi = dist_up > 0.0
    broke_lo = dist_dn > 0.0
    both = broke_hi & broke_lo
    long_only = broke_hi & ~broke_lo
    short_only = broke_lo & ~broke_hi

    out = np.full(n - 1, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    out[long_only] = SmartBSConfig.CLASS_LONG
    out[short_only] = SmartBSConfig.CLASS_SHORT
    # Outside bar: larger excursion wins; tie → FLAT
    out[both & (dist_up > dist_dn)] = SmartBSConfig.CLASS_LONG
    out[both & (dist_dn > dist_up)] = SmartBSConfig.CLASS_SHORT
    labels[:-1] = out
    return labels


def label_pivot_breakout(
    high: np.ndarray,
    low: np.ndarray,
    *,
    pivot_len: int = DEFAULT_PIVOT_BREAKOUT_LEN,
    horizon: int | None = None,
) -> BarrierLabels:
    """Rolling-extreme pivot breakout labels (window = forward scan = ``L``).

    At closed bar ``t`` with ``L = pivot_len``:

      win_hi = max(high[t-L+1 .. t])
      win_lo = min(low[t-L+1 .. t])
      Updates PH if ``high[t] == win_hi``; Updates PL if ``low[t] == win_lo``.

    Rules:
      1. Update PH only → LONG (immediate, resolved)
      2. Update PL only → SHORT (immediate, resolved)
      3. Update both → FLAT ambiguous (unresolved / dropped)
      4. Else freeze ``(win_hi, win_lo)`` and scan forward up to ``L`` bars:
         first ``high > win_hi`` → LONG; first ``low < win_lo`` → SHORT;
         same-bar both → FLAT ambiguous; no break → FLAT range-hold (resolved)
      5. Warmup ``t < L-1`` and tail ``t > n-1-L`` → unresolved

    ``horizon`` is ignored; the forward scan length is always ``pivot_len``.
    CLASS ids 0/1/2 keep the 3-logit head compatible.
    """
    del horizon  # forced to pivot_len
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if high.shape != low.shape:
        raise ValueError("high/low length mismatch")
    n = len(high)
    labels = np.full(n, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    resolved = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)
    if n == 0:
        return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)

    L = max(int(pivot_len), 1)
    # Valid decision bars: full L-window behind and L bars of lookahead ahead.
    for t in range(L - 1, n - L):
        sl = slice(t - L + 1, t + 1)
        win_hi = float(np.max(high[sl]))
        win_lo = float(np.min(low[sl]))
        upd_ph = high[t] == win_hi
        upd_pl = low[t] == win_lo

        if upd_ph and upd_pl:
            labels[t] = SmartBSConfig.CLASS_FLAT
            ambiguous[t] = True
            continue
        if upd_ph:
            labels[t] = SmartBSConfig.CLASS_LONG
            resolved[t] = True
            continue
        if upd_pl:
            labels[t] = SmartBSConfig.CLASS_SHORT
            resolved[t] = True
            continue

        # Between extremes: first strict break of the frozen window within L bars.
        decided = False
        for j in range(1, L + 1):
            i = t + j
            broke_hi = high[i] > win_hi
            broke_lo = low[i] < win_lo
            if broke_hi and broke_lo:
                labels[t] = SmartBSConfig.CLASS_FLAT
                ambiguous[t] = True
                decided = True
                break
            if broke_hi:
                labels[t] = SmartBSConfig.CLASS_LONG
                resolved[t] = True
                decided = True
                break
            if broke_lo:
                labels[t] = SmartBSConfig.CLASS_SHORT
                resolved[t] = True
                decided = True
                break
        if not decided:
            labels[t] = SmartBSConfig.CLASS_FLAT
            resolved[t] = True

    return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)


def label_triple_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    atr: np.ndarray | None = None,
    k_up: float = DEFAULT_BARRIER_K,
    k_dn: float = DEFAULT_BARRIER_K,
    horizon: int = DEFAULT_BARRIER_HORIZON,
) -> BarrierLabels:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if atr is None:
        atr = _atr(high, low, close, 14)
    atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)

    upper = close + float(k_up) * atr
    lower = close - float(k_dn) * atr

    sentinel = n + 1
    first_up = np.full(n, sentinel, dtype=np.int64)
    first_dn = np.full(n, sentinel, dtype=np.int64)

    # One vectorized pass per forward offset: O(horizon * n), not O(n * horizon)
    # in Python-level loops.
    for j in range(1, int(horizon) + 1):
        if j >= n:
            break
        fut_high = np.full(n, -np.inf)
        fut_low = np.full(n, np.inf)
        fut_high[: n - j] = high[j:]
        fut_low[: n - j] = low[j:]
        hit_up = (fut_high >= upper) & (first_up == sentinel)
        hit_dn = (fut_low <= lower) & (first_dn == sentinel)
        first_up = np.where(hit_up, j, first_up)
        first_dn = np.where(hit_dn, j, first_dn)

    labels = np.full(n, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    labels = np.where(first_up < first_dn, SmartBSConfig.CLASS_LONG, labels)
    labels = np.where(first_dn < first_up, SmartBSConfig.CLASS_SHORT, labels)

    touched = np.minimum(first_up, first_dn)
    ambiguous = (first_up == first_dn) & (first_up != sentinel)
    resolved = (touched != sentinel) & ~ambiguous
    # Barriers hit on the same bar give no ordering, so the bar teaches nothing.
    labels = np.where(ambiguous, SmartBSConfig.CLASS_FLAT, labels)
    # The tail cannot resolve; leave it FLAT and exclude it from training.
    labels[max(n - int(horizon), 0) :] = SmartBSConfig.CLASS_FLAT

    return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)


def label_pct_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    pct_up: float = 0.02,
    pct_dn: float = 0.02,
    horizon: int = 24,
) -> BarrierLabels:
    """Same first-touch logic as triple_barrier, but barriers are ±% of price.

    Upper = close * (1 + pct_up), lower = close * (1 - pct_dn).
    Default: ±2% within ``horizon`` bars (24 on 1h ≈ 24 calendar hours).
    Does not use ATR; ``label_triple_barrier`` is unchanged.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    pu = float(pct_up)
    pdn = float(pct_dn)
    upper = close * (1.0 + pu)
    lower = close * (1.0 - pdn)

    sentinel = n + 1
    first_up = np.full(n, sentinel, dtype=np.int64)
    first_dn = np.full(n, sentinel, dtype=np.int64)

    for j in range(1, int(horizon) + 1):
        if j >= n:
            break
        fut_high = np.full(n, -np.inf)
        fut_low = np.full(n, np.inf)
        fut_high[: n - j] = high[j:]
        fut_low[: n - j] = low[j:]
        hit_up = (fut_high >= upper) & (first_up == sentinel)
        hit_dn = (fut_low <= lower) & (first_dn == sentinel)
        first_up = np.where(hit_up, j, first_up)
        first_dn = np.where(hit_dn, j, first_dn)

    labels = np.full(n, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    labels = np.where(first_up < first_dn, SmartBSConfig.CLASS_LONG, labels)
    labels = np.where(first_dn < first_up, SmartBSConfig.CLASS_SHORT, labels)

    touched = np.minimum(first_up, first_dn)
    ambiguous = (first_up == first_dn) & (first_up != sentinel)
    resolved = (touched != sentinel) & ~ambiguous
    labels = np.where(ambiguous, SmartBSConfig.CLASS_FLAT, labels)
    labels[max(n - int(horizon), 0) :] = SmartBSConfig.CLASS_FLAT

    return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)


def session_end_ms(times_ms: np.ndarray) -> np.ndarray:
    """UTC ms when the current session(s) end for each bar.

    If a bar sits in overlapping sessions, the latest end among them is used.
    Bars outside all sessions get ``end == open_time`` (no forward window).
    """
    _start, end = session_start_end_ms(times_ms)
    t = np.asarray(times_ms, dtype=np.int64)
    # Outside session: end == open_time (legacy contract).
    return np.where(_start >= 0, end, t)


def session_start_end_ms(
    times_ms: np.ndarray,
    *,
    session_hours: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar UTC session ``[start, end)`` (latest-ending session if overlapping).

    Outside all sessions → ``start == -1``, ``end == -1``.

    ``session_hours``: ``normal`` (default) | ``adjusted`` | None → active mode.
    """
    from smartbs_engines.common_channels import session_bounds

    t = np.asarray(times_ms, dtype=np.int64)
    n = len(t)
    start = np.full(n, -1, dtype=np.int64)
    end = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return start, end
    hour = ((t % DAY_MS) // HOUR_MS).astype(np.int64)
    day0 = (t // DAY_MS) * DAY_MS
    for lo, hi in session_bounds(session_hours):
        in_sess = (hour >= lo) & (hour < hi)
        cand_start = day0 + int(lo) * HOUR_MS
        cand_end = day0 + int(hi) * HOUR_MS
        better = in_sess & ((end < 0) | (cand_end > end))
        start = np.where(better, cand_start, start)
        end = np.where(better, cand_end, end)
    return start, end


def _infer_bar_ms(times_ms: np.ndarray) -> int:
    t = np.asarray(times_ms, dtype=np.int64)
    if len(t) < 2:
        return HOUR_MS
    d = np.diff(t)
    d = d[d > 0]
    if len(d) == 0:
        return HOUR_MS
    return max(int(np.median(d)), 60_000)


def label_session_direction(
    open_: np.ndarray,
    close: np.ndarray,
    times_ms: np.ndarray,
    *,
    pct: float = DEFAULT_SESSION_DIRECTION_PCT,
) -> BarrierLabels:
    """Session open→close return vs ±``pct`` (default ±0.5%).

    For each Tokyo / London / NY window (active session_hours mode):
      ret = (session_close - session_open) / session_open
      LONG  if ret > +pct
      SHORT if ret < -pct
      FLAT  otherwise

    Default windows are **normal** overlapping FX (UTC 00–09 / 07–16 / 12–21).
    Use ``set_session_hours_mode("adjusted")`` for non-overlapping 00–07 / 07–12 / 12–20.

    All bars inside a completed session share that label. Incomplete trailing
    sessions and bars outside sessions stay FLAT.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    times = np.asarray(times_ms, dtype=np.int64)
    n = len(close)
    labels = np.full(n, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    resolved = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)
    if n == 0:
        return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)

    start, end = session_start_end_ms(times)
    bar_ms = _infer_bar_ms(times)
    thr = float(pct)
    last_t = int(times[-1])

    # Unique session starts among in-session bars.
    in_sess = start >= 0
    if not np.any(in_sess):
        return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)

    for s0 in np.unique(start[in_sess]):
        mask = start == s0
        e0 = int(end[mask][0])
        # Session finished in the series (covered through session end).
        if last_t + bar_ms < e0:
            continue
        win = (times >= s0) & (times < e0)
        if not np.any(win):
            continue
        idxs = np.flatnonzero(win)
        i0 = int(idxs[0])
        i1 = int(idxs[-1])
        o = float(open_[i0])
        c = float(close[i1])
        if not np.isfinite(o) or o <= 0.0 or not np.isfinite(c):
            continue
        ret = (c - o) / o
        if ret > thr:
            lab = SmartBSConfig.CLASS_LONG
        elif ret < -thr:
            lab = SmartBSConfig.CLASS_SHORT
        else:
            lab = SmartBSConfig.CLASS_FLAT
        labels[win] = lab
        resolved[win] = True

    return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)


def label_session_trend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    times_ms: np.ndarray | None = None,
    *,
    atr: np.ndarray | None = None,
    k: float = DEFAULT_SESSION_TREND_K,
    horizon: int = DEFAULT_SESSION_TREND_HORIZON,
    pct: float | None = None,
    session_hours: str | None = None,
) -> BarrierLabels:
    """Label bars in ``[session_start−1h, session_end−1h)`` by ±k*ATR first-touch.

    For each completed Tokyo / London / NY window, every pred-window bar ``t``:

      upper = close[t] + k * ATR(14)[t]   (default k=2)
      lower = close[t] - k * ATR(14)[t]
      scan forward while open_time < session_end
      LONG  if high touches upper first
      SHORT if low touches lower first
      FLAT  if neither barrier is touched before session end
      ambiguous same-bar both-touch → FLAT

    Pred window starts 1h before session open and stops 1h before session end.
    Incomplete trailing sessions stay unresolved. Overlapping windows: latest-
    ending wins. ``horizon`` / ``pct`` ignored.
    """
    _ = horizon
    _ = pct
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    labels = np.full(n, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    resolved = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)
    if n == 0 or times_ms is None:
        return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)

    times = np.asarray(times_ms, dtype=np.int64)
    if len(times) != n:
        raise ValueError("times_ms length must match close")

    kk = float(k)
    if kk <= 0.0:
        raise ValueError(f"session_trend k must be > 0, got {k!r}")
    if atr is None:
        atr = _atr(high, low, close, 14)
    atr_arr = np.asarray(atr, dtype=np.float64)
    atr_arr = np.where(np.isfinite(atr_arr) & (atr_arr > 0), atr_arr, np.nan)

    bar_ms = _infer_bar_ms(times)
    last_t = int(times[-1])
    best_end = np.full(n, -1, dtype=np.int64)

    from smartbs_engines.common_channels import max_session_hours, session_bounds

    day0 = (times // DAY_MS) * DAY_MS
    days = np.unique(day0)
    if len(days):
        days = np.unique(np.concatenate([days, days - DAY_MS]))

    for d0 in days:
        d0 = int(d0)
        for lo, hi in session_bounds(session_hours):
            s0 = d0 + int(lo) * HOUR_MS
            e0 = d0 + int(hi) * HOUR_MS
            if last_t + bar_ms < e0:
                continue
            win_sess = (times >= s0) & (times < e0)
            if not np.any(win_sess):
                continue
            pred = (times >= s0 - bar_ms) & (times < e0 - bar_ms)
            better = pred & ((best_end < 0) | (e0 > best_end))
            best_end = np.where(better, e0, best_end)

    # Barriers only on pred-window bars with valid ATR; others get end<=t → no scan.
    band = kk * atr_arr
    upper = close + band
    lower = close - band
    end_ms = np.where(best_end >= 0, best_end, times)
    bad = ~(np.isfinite(band) & (band > 0) & np.isfinite(close))
    end_ms = np.where(bad, times, end_ms)

    max_h = max(int(max_session_hours(session_hours)) + 2, 8)
    ft = _first_touch_until(
        high,
        low,
        upper,
        lower,
        times_ms=times,
        end_ms=end_ms,
        max_horizon=max_h,
    )

    # Completed pred-window bars: resolve even when neither barrier is touched (FLAT).
    mask = best_end >= 0
    labels = np.where(mask, ft.labels, labels)
    ambiguous = np.where(mask, ft.ambiguous, ambiguous)
    resolved = np.where(mask, ~ft.ambiguous, resolved)
    labels = np.where(ambiguous, SmartBSConfig.CLASS_FLAT, labels)

    return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)


def _ny_hour_bounds(session_hours: str | None = None) -> tuple[int, int]:
    """UTC ``[lo, hi)`` for New York under the active / requested session pack."""
    from smartbs_engines.common_channels import session_hours_for_mode

    for name, lo, hi in session_hours_for_mode(session_hours):
        if name == "newyork":
            return int(lo), int(hi)
    return 12, 21


def label_day_trend(
    close: np.ndarray,
    times_ms: np.ndarray,
    *,
    pct: float = DEFAULT_DAY_TREND_PCT,
    session_hours: str | None = None,
) -> BarrierLabels:
    """Label each 1h bar by **today's NY session end close** vs ±``pct``.

    NY end close = close of the last H1 bar whose UTC hour is in the NY window
    (normal 12–21 → hour 20 bar; adjusted 12–20 → hour 19 bar). A day is
    labeled only once that final NY hour bar exists in the series.

    For each earlier bar on that UTC day:

      ret = (ny_end_close - close[t]) / close[t]
      LONG  if ret >  +pct   (default 1%)
      SHORT if ret <  -pct
      FLAT  if |ret| <= pct

    Bars on/after the NY end bar, incomplete trailing days, and days without
    a full NY window stay unresolved FLAT.
    """
    close = np.asarray(close, dtype=np.float64)
    times = np.asarray(times_ms, dtype=np.int64)
    n = len(close)
    labels = np.full(n, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    resolved = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)
    if n == 0:
        return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)

    pp = float(pct)
    if pp <= 0.0:
        raise ValueError(f"day_trend pct must be > 0, got {pct!r}")

    ny_lo, ny_hi = _ny_hour_bounds(session_hours)
    last_ny_hour = ny_hi - 1
    hour = ((times % DAY_MS) // HOUR_MS).astype(np.int64)
    day = times // DAY_MS

    # Per UTC day: index of the final NY-hour bar (hour == last_ny_hour), if present.
    end_idx_by_day: dict[int, int] = {}
    for i in range(n):
        if int(hour[i]) == last_ny_hour:
            end_idx_by_day[int(day[i])] = i

    for i in range(n):
        d = int(day[i])
        end_i = end_idx_by_day.get(d)
        if end_i is None or i >= end_i:
            continue
        c0 = float(close[i])
        c1 = float(close[end_i])
        if not np.isfinite(c0) or not np.isfinite(c1) or abs(c0) < 1e-12:
            continue
        ret = (c1 - c0) / c0
        if ret > pp:
            labels[i] = SmartBSConfig.CLASS_LONG
        elif ret < -pp:
            labels[i] = SmartBSConfig.CLASS_SHORT
        else:
            labels[i] = SmartBSConfig.CLASS_FLAT
        resolved[i] = True

    return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)


def _first_touch_until(
    high: np.ndarray,
    low: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    *,
    times_ms: np.ndarray,
    end_ms: np.ndarray,
    max_horizon: int,
) -> BarrierLabels:
    """First-touch LONG/SHORT while ``times[i+j] < end_ms[i]``."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    times = np.asarray(times_ms, dtype=np.int64)
    end = np.asarray(end_ms, dtype=np.int64)
    n = len(upper)

    sentinel = n + 1
    first_up = np.full(n, sentinel, dtype=np.int64)
    first_dn = np.full(n, sentinel, dtype=np.int64)
    h_max = max(1, int(max_horizon))

    for j in range(1, h_max + 1):
        if j >= n:
            break
        fut_high = np.full(n, -np.inf)
        fut_low = np.full(n, np.inf)
        fut_t = np.full(n, np.iinfo(np.int64).max)
        fut_high[: n - j] = high[j:]
        fut_low[: n - j] = low[j:]
        fut_t[: n - j] = times[j:]
        still = fut_t < end
        hit_up = still & (fut_high >= upper) & (first_up == sentinel)
        hit_dn = still & (fut_low <= lower) & (first_dn == sentinel)
        first_up = np.where(hit_up, j, first_up)
        first_dn = np.where(hit_dn, j, first_dn)

    labels = np.full(n, SmartBSConfig.CLASS_FLAT, dtype=np.int64)
    labels = np.where(first_up < first_dn, SmartBSConfig.CLASS_LONG, labels)
    labels = np.where(first_dn < first_up, SmartBSConfig.CLASS_SHORT, labels)

    touched = np.minimum(first_up, first_dn)
    ambiguous = (first_up == first_dn) & (first_up != sentinel)
    resolved = (touched != sentinel) & ~ambiguous
    labels = np.where(ambiguous, SmartBSConfig.CLASS_FLAT, labels)
    # No forward session window → FLAT
    labels = np.where(end <= times, SmartBSConfig.CLASS_FLAT, labels)
    return BarrierLabels(labels=labels, resolved=resolved, ambiguous=ambiguous)


def select_barrier_train_indices(
    candidates: list[int],
    labels: np.ndarray,
    *,
    seed: int = 42,
    max_flat_ratio: float = 2.0,
) -> list[int]:
    """Cap FLAT dominance and balance LONG vs SHORT.

    Triple-barrier labels are roughly symmetric by construction, so this is a
    much lighter touch than the old policy-simulated labels needed.
    """
    rng = np.random.default_rng(seed)
    longs = [i for i in candidates if labels[i] == SmartBSConfig.CLASS_LONG]
    shorts = [i for i in candidates if labels[i] == SmartBSConfig.CLASS_SHORT]
    flats = [i for i in candidates if labels[i] == SmartBSConfig.CLASS_FLAT]

    keep_dir = min(len(longs), len(shorts))
    if keep_dir == 0:
        return sorted(candidates)
    if len(longs) > keep_dir:
        longs = list(rng.choice(longs, size=keep_dir, replace=False))
    if len(shorts) > keep_dir:
        shorts = list(rng.choice(shorts, size=keep_dir, replace=False))

    max_flat = int(max(keep_dir * 2 * float(max_flat_ratio), 1))
    if len(flats) > max_flat:
        flats = list(rng.choice(flats, size=max_flat, replace=False))
    return sorted(int(i) for i in (*longs, *shorts, *flats))
