"""Train all ST engines × all dump assets (MT5) + comparison replay.

Window:
  train_years (CLI, default 10) → (no fixed from-date) → align to max(1h_start, 15m_start)
  → holdout 360d for replay

Usage:
  python -m smartbs_engines._train_replay_all_st
  python -m smartbs_engines._train_replay_all_st --train-years 5
  python -m smartbs_engines._train_replay_all_st --label-mode forward_return --interval 15m --ckpt-root .../checkpoints_fwd_15m --out-stem asset_engine_fwd_15m
  python -m smartbs_engines._train_replay_all_st --skip-train
  python -m smartbs_engines._train_replay_all_st --symbols XAUUSD,BTCUSD
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from smartbs_engines.config import SmartBSConfig
from smartbs_engines.data import fetch_klines
from smartbs_engines.features import num_inputs_for
from smartbs_engines.mt5_data import DEFAULT_DUMP_ASSETS
from smartbs_engines.predict import predict_probs
from smartbs_engines.registry import list_engines
from smartbs_engines.structure import atr as _atr
from smartbs_engines.structure import sma as _sma
from smartbs_engines.structure import swing_levels
from smartbs_engines.train import train_model

ALL_ENGINES = tuple(list_engines())
ALL_ASSETS = tuple(DEFAULT_DUMP_ASSETS)
COMMODITIES = ("XAUUSD", "XAGUSD", "XTIUSD", "NATGAS", "PLATINUM")
WINDOWS = (90, 180, 360)
LOTS = {
    "XAUUSD": 0.1,
    "XAGUSD": 0.5,
    "XTIUSD": 0.1,
    "NATGAS": 0.1,
    "PLATINUM": 0.1,
    "BTCUSD": 0.01,
    "ETHUSD": 0.1,
    "LTCUSD": 0.1,
    "ADAUSD": 1.0,
    "SOLUSD": 0.1,
    "EURUSD": 0.1,
    "GBPUSD": 0.1,
    "AUDUSD": 0.1,
    "USDCHF": 0.1,
    "USDJPY": 0.1,
}
PVS = {
    "XAUUSD": 1.0,
    "XAGUSD": 5.0,
    "XTIUSD": 1.0,
    "NATGAS": 1.0,
    "PLATINUM": 1.0,
    "BTCUSD": 1.0,
    "ETHUSD": 1.0,
    "LTCUSD": 1.0,
    "ADAUSD": 1.0,
    "SOLUSD": 1.0,
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "AUDUSD": 10.0,
    "USDCHF": 10.0,
    "USDJPY": 10.0,
}


def _stats(ret: np.ndarray, pos: np.ndarray) -> dict:
    eq = np.cumsum(ret)
    peak = np.maximum.accumulate(eq) if len(eq) else eq
    dd = float((eq - peak).min()) if len(eq) else 0.0
    pnl = float(eq[-1]) if len(eq) else 0.0
    segs: list[float] = []
    cur = 0.0
    side = 0
    for s, rr in zip(pos, ret):
        if int(s) != side:
            if side != 0:
                segs.append(cur)
            cur = 0.0
            side = int(s)
        if side != 0:
            cur += float(rr)
    if side != 0:
        segs.append(cur)
    wins = [x for x in segs if x > 0]
    losses = [x for x in segs if x <= 0]
    gp = sum(wins)
    gl = -sum(losses)
    pf = (gp / gl) if gl > 1e-12 else float("inf")
    wr = (100.0 * len(wins) / len(segs)) if segs else 0.0
    return {"pnl": pnl, "dd": dd, "trades": len(segs), "wr": wr, "pf": pf}


def _classes_from_probs(
    probs: np.ndarray,
    ai_threshold: float,
    *,
    force_side: bool = False,
) -> np.ndarray:
    """Map (n,3) probs → class ids. Gate: max(P_long,P_short) < thr → FLAT.

    ``force_side=True``: always pick LONG vs SHORT, never FLAT (legacy close-only
    next_direction). Range-break next_direction trains FLAT and uses argmax.
    """
    p_long = probs[:, 1]
    p_short = probs[:, 2]
    side = np.where(p_long >= p_short, 1, 2).astype(np.int64)
    if force_side:
        return side
    conf = np.maximum(p_long, p_short)
    thr = float(ai_threshold)
    if thr <= 0.0:
        return probs.argmax(axis=1).astype(np.int64)
    return np.where(conf < thr, 0, side).astype(np.int64)


def _make_swing_stop(
    side: int,
    entry: float,
    swing_high: float,
    swing_low: float,
    atr: float,
    *,
    r_min_atr: float = 0.25,
    r_max_atr: float = 3.0,
    r_max_pct: float = 0.012,
    chase_frac: float = 0.45,
) -> tuple[bool, float, float]:
    """EA ``SB_MakeSwingStop`` parity. Returns ``(ok, sl, r)``."""
    if side == 0 or not np.isfinite(entry):
        return False, 0.0, 0.0
    if not (np.isfinite(swing_high) and np.isfinite(swing_low)):
        return False, 0.0, 0.0
    rng = float(swing_high - swing_low)
    if rng <= 0.0:
        return False, 0.0, 0.0
    if side > 0:
        if swing_low >= entry:
            return False, 0.0, 0.0
        sl = float(swing_low)
        r = float(entry - swing_low)
    else:
        if swing_high <= entry:
            return False, 0.0, 0.0
        sl = float(swing_high)
        r = float(swing_high - entry)
    if r <= 0.0:
        return False, 0.0, 0.0
    if chase_frac > 0.0 and (r / rng) > chase_frac:
        return False, 0.0, 0.0
    if atr > 0.0 and r < r_min_atr * atr:
        return False, 0.0, 0.0
    if atr > 0.0 and r > r_max_atr * atr:
        return False, 0.0, 0.0
    if r_max_pct > 0.0 and r > entry * r_max_pct:
        return False, 0.0, 0.0
    return True, sl, r


def _positions_ma_ai_gate(
    cls: np.ndarray,
    close: np.ndarray,
    sma: np.ndarray,
    *,
    start: int,
    n: int,
) -> np.ndarray:
    """EA-parity MA+AI gate → next-bar position.

    Entry: LONG if close>SMA and AI LONG; SHORT if close<SMA and AI SHORT.
    Exit: LONG when close<SMA; SHORT when close>SMA (AI ignored on exit).
    While in a trade, hold even if AI flips/FLAT.
    """
    pos = np.zeros(n, dtype=np.int8)
    have = 0
    for i in range(int(start), n - 1):
        c = float(close[i])
        m = float(sma[i])
        if not np.isfinite(m):
            pos[i + 1] = have
            continue
        ai = 0
        ci = int(cls[i])
        if ci == 1:
            ai = 1
        elif ci == 2:
            ai = -1

        if have > 0 and c < m:
            have = 0
        elif have < 0 and c > m:
            have = 0

        if have == 0:
            if ai > 0 and c > m:
                have = 1
            elif ai < 0 and c < m:
                have = -1

        pos[i + 1] = have
    return pos


def _exit_ma_confirm(position: int, close: float, fma: float) -> bool:
    """True when close confirms exit arm (below MA for LONG, above for SHORT)."""
    if position > 0:
        return close < fma
    if position < 0:
        return close > fma
    return False


def _positions_raw_arm(
    cls: np.ndarray,
    close: np.ndarray,
    sma: np.ndarray,
    *,
    start: int,
    n: int,
    entry_arm: bool = True,
    exit_arm: bool = True,
) -> np.ndarray:
    """Bot-parity raw_ai + optional entry/exit arms on SMA → next-bar position.

    * ``entry_arm``: AI LONG/SHORT arms; fill when close > SMA (L) / < SMA (S).
      Arm clears on AI FLAT; re-arms on flip.
    * ``exit_arm``: on AI FLAT/flip, wait until MA confirms before flattening.
    * Same-bar reverse allowed after an exit confirms (matches Bot ``rm=off``).
    """
    pos = np.zeros(n, dtype=np.int8)
    have = 0
    armed = 0
    for i in range(int(start), n - 1):
        c = float(close[i])
        m = float(sma[i])
        ai = 0
        ci = int(cls[i])
        if ci == 1:
            ai = 1
        elif ci == 2:
            ai = -1

        if not np.isfinite(m):
            pos[i + 1] = have
            continue

        # Exit when AI disagrees with open side.
        if have != 0 and ai != have:
            if exit_arm and not _exit_ma_confirm(have, c, m):
                pos[i + 1] = have
                continue
            have = 0
            armed = 0

        if have != 0:
            pos[i + 1] = have
            continue

        # Flat book — entry path.
        if entry_arm:
            if armed != 0:
                if (armed > 0 and c > m) or (armed < 0 and c < m):
                    have = armed
                    armed = 0
                elif ai == 0:
                    armed = 0
                elif ai != armed:
                    armed = ai
                    if (armed > 0 and c > m) or (armed < 0 and c < m):
                        have = armed
                        armed = 0

            if have == 0:
                if ai == 0:
                    armed = 0
                else:
                    if armed != ai:
                        armed = ai
                    if (armed > 0 and c > m) or (armed < 0 and c < m):
                        have = armed
                        armed = 0
        elif ai != 0:
            have = ai

        pos[i + 1] = have
    return pos


def _simulate_session_hold(
    cls: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    times: np.ndarray,
    *,
    start: int,
    n: int,
    lot: float,
    point_value: float,
    session_hours: str = "normal",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold AI side from pre-session bar: open at session start, close at end.

    Decision = last bar before session ``start``. LONG/SHORT → enter at the
    first in-session open, exit at the last in-session close. FLAT → skip.
    Returns ``(pos, ret, session_start_ms_per_bar)``; PnL is realized on the
    exit bar. ``session_start_ms`` is ``s0`` on bars belonging to a traded
    session (else -1) for window attribution.
    """
    from smartbs_engines.labels import _infer_bar_ms, session_start_end_ms

    pos = np.zeros(n, dtype=np.int8)
    ret = np.zeros(n, dtype=np.float64)
    sess_tag = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return pos, ret, sess_tag

    start_ms, end_ms = session_start_end_ms(times, session_hours=session_hours)
    in_sess = start_ms >= 0
    if not np.any(in_sess):
        return pos, ret, sess_tag

    bar_ms = _infer_bar_ms(times)
    last_t = int(times[-1])
    for s0 in np.unique(start_ms[in_sess]):
        e0 = int(end_ms[start_ms == s0][0])
        if last_t + bar_ms < e0:
            continue
        pre = np.flatnonzero(times < s0)
        if len(pre) == 0:
            continue
        i_pre = int(pre[-1])
        if i_pre < int(start):
            continue
        c = int(cls[i_pre])
        if c == 0:
            continue
        side = 1 if c == 1 else -1
        win = np.flatnonzero((times >= s0) & (times < e0))
        if len(win) == 0:
            continue
        i0 = int(win[0])
        i1 = int(win[-1])
        if i0 >= n or i1 >= n:
            continue
        entry = float(opens[i0])
        exit_px = float(closes[i1])
        if not (np.isfinite(entry) and np.isfinite(exit_px)):
            continue
        pnl = float(side) * (exit_px - entry) * float(lot) * float(point_value)
        pos[i0 : i1 + 1] = side
        ret[i1] = pnl
        sess_tag[i0 : i1 + 1] = int(s0)
    return pos, ret, sess_tag


def _simulate_raw_ai_swing_sl(
    cls: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    atr: np.ndarray,
    *,
    start: int,
    n: int,
    lot: float,
    point_value: float,
    r_min_atr: float,
    r_max_atr: float,
    r_max_pct: float,
    chase_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw AI entries + initial swing-R stop only (no MA gate, no trail)."""
    pos = np.zeros(n, dtype=np.int8)
    ret = np.zeros(n, dtype=np.float64)
    have = 0
    stop = float("nan")
    scale = float(lot) * float(point_value)

    for i in range(int(start), n - 1):
        if have != 0 and np.isfinite(stop):
            hit = False
            exit_px = float(opens[i + 1])
            if have > 0 and float(lows[i]) <= stop:
                exit_px = float(stop)
                hit = True
            elif have < 0 and float(highs[i]) >= stop:
                exit_px = float(stop)
                hit = True
            ret[i] = have * (exit_px - float(opens[i])) * scale
            if hit:
                have = 0
                stop = float("nan")
                pos[i + 1] = 0
                continue

        if have == 0:
            c = int(cls[i])
            if c == 0:
                pos[i + 1] = 0
                continue
            side = 1 if c == 1 else -1
            ok, sl, _r = _make_swing_stop(
                side,
                float(opens[i + 1]),
                float(swing_high[i]),
                float(swing_low[i]),
                float(atr[i]),
                r_min_atr=r_min_atr,
                r_max_atr=r_max_atr,
                r_max_pct=r_max_pct,
                chase_frac=chase_frac,
            )
            if not ok:
                pos[i + 1] = 0
                continue
            have = side
            stop = sl
            pos[i + 1] = have
        else:
            pos[i + 1] = have
    return pos, ret


def _day_trend_entry_ok(times_ms: np.ndarray, *, session_hours: str = "normal") -> np.ndarray:
    """True where UTC hour is in ``[0, ny_end_hour - 1)`` (entry window for day_trend).

    Normal NY ends 21:00 → entries only hours 0–19. Adjusted ends 20:00 → 0–18.
    """
    from smartbs_engines.labels import DAY_MS, HOUR_MS, _ny_hour_bounds

    t = np.asarray(times_ms, dtype=np.int64)
    _lo, ny_hi = _ny_hour_bounds(session_hours)
    cutoff = int(ny_hi) - 1
    hour = ((t % DAY_MS) // HOUR_MS).astype(np.int64)
    return hour < cutoff


def _session_trend_entry_ok(
    times_ms: np.ndarray, *, session_hours: str = "normal"
) -> np.ndarray:
    """True in pred window ``[session_start−1h, session_end−1h)`` (any session)."""
    from smartbs_engines.common_channels import session_bounds
    from smartbs_engines.labels import DAY_MS, HOUR_MS, _infer_bar_ms

    t = np.asarray(times_ms, dtype=np.int64)
    n = len(t)
    ok = np.zeros(n, dtype=bool)
    if n == 0:
        return ok
    bar_ms = _infer_bar_ms(t)
    day0 = (t // DAY_MS) * DAY_MS
    days = np.unique(day0)
    if len(days):
        days = np.unique(np.concatenate([days, days - DAY_MS, days + DAY_MS]))
    for d0 in days:
        d0 = int(d0)
        for lo, hi in session_bounds(session_hours):
            s0 = d0 + int(lo) * HOUR_MS
            e0 = d0 + int(hi) * HOUR_MS
            ok |= (t >= s0 - bar_ms) & (t < e0 - bar_ms)
    return ok


def replay_raw_ai(
    symbol: str,
    checkpoint: str,
    *,
    windows_days: list[int],
    lot: float,
    point_value: float,
    interval: str = "1h",
    ai_threshold: float = 0.0,
    ma_ai_gate: bool = False,
    ma_len: int = 14,
    entry_arm: bool = False,
    exit_arm: bool = False,
    swing_sl: bool = False,
    swing_len: int = 9,
    r_min_atr: float = 0.25,
    r_max_atr: float = 3.0,
    r_max_pct: float = 0.012,
    chase_frac: float = 0.45,
    raw_ai_override: str | None = None,
    signal_policy_override: str | None = None,
    session_hold: bool = False,
    session_hours: str = "normal",
    day_trend_window: bool | None = None,
) -> list[dict]:
    if ma_ai_gate and swing_sl:
        raise ValueError("Use either --ma-ai-gate or --swing-sl, not both")
    if (entry_arm or exit_arm) and (ma_ai_gate or swing_sl):
        raise ValueError("--entry-arm/--exit-arm cannot combine with --ma-ai-gate / --swing-sl")
    if session_hold and (ma_ai_gate or swing_sl or entry_arm or exit_arm):
        raise ValueError("--session-hold cannot combine with MA/arm/swing gates")
    iv = (interval or "1h").strip().lower()
    df = fetch_klines(symbol, iv, max_candles=0, source="mt5")
    end_ms = int(df["open_time"].iloc[-1])
    max_days = max(windows_days)
    cutoff_max = end_ms - max_days * 86_400_000
    hold = df[df["open_time"] >= cutoff_max].reset_index(drop=True)
    warm_n = 12_000 if iv in ("15m", "15min", "m15") else 3_000
    warm = df[df["open_time"] < cutoff_max].tail(warm_n)
    full = pd.concat([warm, hold], ignore_index=True)
    hold_start = len(warm)

    probs, cfg = predict_probs(full, checkpoint, symbol=symbol, data_source="mt5")
    lookback = int(cfg.get("lookback", 64))
    force_side = False
    cls = _classes_from_probs(probs, ai_threshold, force_side=force_side)
    opens = full["open"].to_numpy(dtype=np.float64)
    highs = full["high"].to_numpy(dtype=np.float64)
    lows = full["low"].to_numpy(dtype=np.float64)
    closes = full["close"].to_numpy(dtype=np.float64)
    times = full["open_time"].to_numpy(dtype=np.int64)
    n = len(full)
    start = max(lookback - 1, hold_start)

    from smartbs_engines.common_channels import set_session_hours_mode
    from smartbs_engines.pipeline import PipelineSpec
    from smartbs_engines.signal_policy import RAW_AI_AI_ONLY, apply_raw_ai_strategy

    sh_mode = (
        str(session_hours or cfg.get("session_hours") or "normal").strip().lower()
        or "normal"
    )
    set_session_hours_mode(sh_mode)

    use_day_win = day_trend_window
    if use_day_win is None:
        use_day_win = str(cfg.get("label_mode", "")).strip().lower() == "day_trend"
    use_sess_win = str(cfg.get("label_mode", "")).strip().lower() == "session_trend"
    if use_day_win:
        entry_ok = _day_trend_entry_ok(times, session_hours=sh_mode)
        win_tag = "+day_win"
    elif use_sess_win:
        entry_ok = _session_trend_entry_ok(times, session_hours=sh_mode)
        win_tag = "+sess_win"
    else:
        entry_ok = np.ones(n, dtype=bool)
        win_tag = ""
    use_entry_win = bool(use_day_win or use_sess_win)

    cfg_eff = dict(cfg)
    if raw_ai_override:
        cfg_eff["raw_ai_strategy"] = str(raw_ai_override)
        cfg_eff["signal_point_gate"] = str(raw_ai_override).strip().lower() == "signal_gate"
    if signal_policy_override:
        cfg_eff["signal_policy"] = str(signal_policy_override)
    pipe = PipelineSpec.from_config(cfg_eff)
    gate_sp = ""
    if (
        pipe.raw_ai != RAW_AI_AI_ONLY
        and not ma_ai_gate
        and not swing_sl
        and not entry_arm
        and not exit_arm
        and not session_hold
    ):
        from smartbs_engines.registry import get_engine

        policy = pipe.make_signal_policy()
        sig_eng = get_engine("signals", signal_sources=pipe.engines)
        sig_res = sig_eng.compute(full)
        sig_feats = np.asarray(sig_res.features, dtype=np.float64)
        decision = policy.decide(
            full,
            features=sig_feats,
            feature_names=list(sig_eng.feature_names),
            symbol=symbol,
            data_source="mt5",
        )
        cls, _reason_idx = apply_raw_ai_strategy(
            cls,
            decision,
            pipe.raw_ai,
            require_agree=pipe.signal_require_agree,
            features=sig_feats,
            feature_names=list(sig_eng.feature_names),
            thr=pipe.signal_thr,
        )
        gate_sp = f"{pipe.signal_policy}:{pipe.raw_ai}"
    else:
        gate_sp = "raw_ai" if pipe.raw_ai == RAW_AI_AI_ONLY else ""

    sess_tag: np.ndarray | None = None
    if session_hold:
        pos, ret, sess_tag = _simulate_session_hold(
            cls,
            opens,
            closes,
            times,
            start=start,
            n=n,
            lot=lot,
            point_value=point_value,
            session_hours=sh_mode,
        )
        ret[:hold_start] = 0.0
        pos[:hold_start] = 0
        sess_tag[:hold_start] = -1
        gate = "session_hold"
    elif ma_ai_gate:
        ml = max(1, int(ma_len))
        ma = _sma(closes, ml)
        pos = _positions_ma_ai_gate(cls, closes, ma, start=start, n=n)
        if use_entry_win:
            # Drop entries decided outside the label pred window.
            for i in range(start, n - 1):
                if not entry_ok[i]:
                    pos[i + 1] = 0
        ret = np.zeros(n, dtype=np.float64)
        for i in range(hold_start, n - 1):
            if pos[i] == 0:
                continue
            ret[i] = pos[i] * (opens[i + 1] - opens[i]) * lot * point_value
        gate = "ma_ai" + win_tag
    elif entry_arm or exit_arm:
        ml = max(1, int(ma_len))
        ma = _sma(closes, ml)
        pos = _positions_raw_arm(
            cls,
            closes,
            ma,
            start=start,
            n=n,
            entry_arm=bool(entry_arm),
            exit_arm=bool(exit_arm),
        )
        if use_entry_win:
            for i in range(start, n - 1):
                if not entry_ok[i]:
                    pos[i + 1] = 0
        ret = np.zeros(n, dtype=np.float64)
        for i in range(hold_start, n - 1):
            if pos[i] == 0:
                continue
            ret[i] = pos[i] * (opens[i + 1] - opens[i]) * lot * point_value
        arms = []
        if entry_arm:
            arms.append("entry_arm")
        if exit_arm:
            arms.append("exit_arm")
        gate = "+".join(arms) + f"(SMA{ml})" + win_tag
    elif swing_sl:
        sh, slv = swing_levels(highs, lows, int(swing_len))
        atr14 = _atr(highs, lows, closes, 14)
        pos, ret = _simulate_raw_ai_swing_sl(
            cls,
            opens,
            highs,
            lows,
            sh,
            slv,
            atr14,
            start=start,
            n=n,
            lot=lot,
            point_value=point_value,
            r_min_atr=float(r_min_atr),
            r_max_atr=float(r_max_atr),
            r_max_pct=float(r_max_pct),
            chase_frac=float(chase_frac),
        )
        if use_entry_win:
            for i in range(start, n - 1):
                if not entry_ok[i]:
                    if i + 1 < n and pos[i] == 0:
                        pos[i + 1] = 0
        ret[:hold_start] = 0.0
        gate = "swing_sl" + win_tag
    else:
        pos = np.zeros(n, dtype=np.int8)
        for i in range(start, n - 1):
            if use_entry_win and not entry_ok[i]:
                continue
            c = int(cls[i])
            if c == 0:
                continue
            pos[i + 1] = 1 if c == 1 else -1
        ret = np.zeros(n, dtype=np.float64)
        for i in range(hold_start, n - 1):
            if pos[i] == 0:
                continue
            ret[i] = pos[i] * (opens[i + 1] - opens[i]) * lot * point_value
        gate = (gate_sp or "raw_ai") + win_tag

    rows = []
    for days in windows_days:
        cut = end_ms - days * 86_400_000
        if session_hold and sess_tag is not None:
            # Whole-session trades attributed by session start time.
            mask = (sess_tag >= 0) & (sess_tag >= cut) & (np.arange(n) >= hold_start)
        else:
            mask = (np.arange(n) >= hold_start) & (np.arange(n) < n - 1) & (times >= cut)
        st = _stats(ret[mask], pos[mask])
        idx = np.where(mask)[0]
        if len(idx):
            t0 = pd.to_datetime(times[idx[0]], unit="ms", utc=True)
            t1 = pd.to_datetime(times[idx[-1]], unit="ms", utc=True)
            fr, to = str(t0.date()), str(t1.date())
        else:
            fr = to = "?"
        st.update(
            {
                "engine": str(cfg.get("feature_engine", "?")),
                "symbol": symbol,
                "days": days,
                "from": fr,
                "to": to,
                "gate": gate,
            }
        )
        rows.append(st)
    return rows


def train_one(
    sym: str,
    eng: str,
    ckpt_root: Path,
    *,
    train_years: float,
    label_mode: str,
    horizon: int,
    return_threshold: float,
    barrier_k: float,
    barrier_horizon: int,
    ma_len: int = 14,
    barrier_pct: float = 0.02,
    session_hours: str = "normal",
    pivot_len: int = 5,
    interval: str = "1h",
    backbone: str = "tcn",
    kernel_size: int | None = None,
    signal_engines: str = "",
    signal_policy: str = "none",
    raw_ai_strategy: str = "ai_only",
) -> str:
    from smartbs_engines.engines import resolve_feature_engine
    from smartbs_engines.model import default_kernel_size, normalize_backbone
    from smartbs_engines.pipeline import PipelineSpec
    from smartbs_engines.signals import normalize_signal_sources

    bb = normalize_backbone(backbone)
    if kernel_size is None or int(kernel_size) <= 0:
        ks = default_kernel_size(bb)
    else:
        ks = int(kernel_size)

    eng_name, sources = resolve_feature_engine(eng, signal_engines.strip() or None)
    sig_tuple = normalize_signal_sources(sources) if eng_name == "signals" else ()
    pipe = PipelineSpec(
        engines=sig_tuple if eng_name == "signals" else (eng_name,),
        signal_policy=str(signal_policy),
        backbone=bb,
        raw_ai=str(raw_ai_strategy),
    )
    cfg = SmartBSConfig(
        trade_pair=sym,
        data_source="mt5",
        interval=interval,
        train_candles=0,
        train_years=float(train_years),
        train_from_date="",
        train_align_15m=True,
        holdout_days=360,
        epochs=15,
        batch_size=128,
        feature_engine=eng_name,
        signal_engines=sig_tuple,
        signal_policy=pipe.signal_policy,
        raw_ai_strategy=pipe.raw_ai,
        signal_point_gate=pipe.raw_ai == "signal_gate",
        backbone=bb,
        kernel_size=ks,
        num_inputs=num_inputs_for(eng_name, signal_engines=sig_tuple or None),
        train_assets=[sym],
        checkpoint_path="",
        label_mode=label_mode,
        horizon=1 if label_mode == "next_direction" else int(horizon),
        return_threshold=float(return_threshold),
        barrier_k=float(barrier_k),
        barrier_horizon=int(barrier_horizon),
        ma_len=int(ma_len),
        barrier_pct=float(barrier_pct),
        pivot_len=int(pivot_len),
        session_hours=str(session_hours),
    )
    os.environ["SMARTBS_CHECKPOINT_DIR"] = str(ckpt_root)
    return train_model(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(ALL_ASSETS))
    ap.add_argument("--engines", default=",".join(ALL_ENGINES))
    ap.add_argument(
        "--train-years",
        type=float,
        default=10.0,
        help="Calendar years back from series end before 15m align + holdout (default 10)",
    )
    ap.add_argument(
        "--label-mode",
        choices=[
            "triple_barrier",
            "pct_barrier",
            "session_direction",
            "session_trend",
            "day_trend",
            "pivot_breakout",
            "forward_return",
            "next_direction",
        ],
        default="triple_barrier",
    )
    ap.add_argument("--horizon", type=int, default=4, help="forward_return horizon; ma_stay = which future bar (1=next)")
    ap.add_argument(
        "--return-threshold",
        type=float,
        default=0.0015,
        help="forward_return |fwd| threshold",
    )
    ap.add_argument("--ma-len", type=int, default=14, help="ma_stay SMA length; also MA+AI gate SMA")
    ap.add_argument(
        "--ma-ai-gate",
        action="store_true",
        help="Replay with EA MA+AI gate: entry close vs SMA + AI side; exit on MA flip",
    )
    ap.add_argument(
        "--entry-arm",
        action="store_true",
        help="Bot arm_entry: AI arms LONG/SHORT; fill when close vs SMA(ma-len)",
    )
    ap.add_argument(
        "--exit-arm",
        action="store_true",
        help="Bot arm_exit: on AI FLAT/flip wait for MA confirm before flatten",
    )
    ap.add_argument(
        "--swing-sl",
        action="store_true",
        help="Replay raw AI + initial swing-R stop only (EA SB_MakeSwingStop; no MA gate)",
    )
    ap.add_argument("--swing-len", type=int, default=9, help="Swing pivot L/R for --swing-sl")
    ap.add_argument("--r-min-atr", type=float, default=0.25)
    ap.add_argument("--r-max-atr", type=float, default=3.0)
    ap.add_argument("--r-max-pct", type=float, default=0.012)
    ap.add_argument("--chase-frac", type=float, default=0.45)
    ap.add_argument(
        "--barrier-k",
        type=float,
        default=1.0,
        help="triple_barrier / session_trend ATR multiple (session_trend default k=1)",
    )
    ap.add_argument(
        "--barrier-horizon",
        type=int,
        default=4,
        help="triple_barrier / pct_barrier / pivot_breakout horizon bars "
        "(pivot_breakout default 24 when left at 4)",
    )
    ap.add_argument(
        "--pivot-len",
        type=int,
        default=5,
        help="pivot_breakout lookback L/R (default 5)",
    )
    ap.add_argument(
        "--barrier-pct",
        type=float,
        default=0.02,
        help="pct_barrier ±fraction; session_direction / day_trend override",
    )
    ap.add_argument(
        "--session-hours",
        choices=["normal", "adjusted"],
        default="normal",
        help="UTC sessions: normal 0-9/7-16/12-21 (default) | adjusted 0-7/7-12/12-20",
    )
    ap.add_argument(
        "--ckpt-root",
        default="",
        help="Checkpoint root (default: smartbs_engines/checkpoints)",
    )
    ap.add_argument(
        "--interval",
        default="1h",
        choices=["1h", "15m"],
        help="Primary bar interval for train + replay (default 1h)",
    )
    ap.add_argument(
        "--ai-threshold",
        type=float,
        default=0.0,
        help="Min max(P_long,P_short); below → FLAT (0=raw argmax). EA default 0.35",
    )
    ap.add_argument(
        "--backbone",
        default="tcn",
        help="Model backbone: tcn | smartBSEntryV2 | smartBSTF | smartBSDualTF",
    )
    ap.add_argument(
        "--signal-engines",
        default="",
        help="When --engines includes signals: comma/+ sources "
        "(default=ACTIVE_SIGNAL_SOURCES)",
    )
    ap.add_argument(
        "--signal-policy",
        default="none",
        help="SignalDecision policy: none | onset_side | ma_decay",
    )
    ap.add_argument(
        "--raw-ai-strategy",
        default="ai_only",
        help="raw_ai combine: ai_only | signal_gate | signal_prior",
    )
    ap.add_argument(
        "--kernel-size",
        type=int,
        default=0,
        help="Conv kernel (default: 3 tcn / 7 V2 / 1 TF unused)",
    )
    ap.add_argument(
        "--session-hold",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replay: pre-session AI → open at session start, close at session end "
        "(off by default; use --session-hold to enable)",
    )
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="Skip train if .pt exists")
    ap.add_argument(
        "--out-stem",
        default="asset_engine",
        help="Output stem under sweep_st/ (writes {stem}_replay.csv + {stem}_summary.json)",
    )
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    engines = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    train_years = float(args.train_years)
    label_mode = str(args.label_mode)
    interval = str(args.interval).strip().lower()
    ai_threshold = float(args.ai_threshold)
    base = Path(__file__).resolve().parent
    ckpt_root = Path(args.ckpt_root) if str(args.ckpt_root).strip() else base / "checkpoints"
    out_dir = base / "sweep_st"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = str(args.out_stem).strip() or "asset_engine"

    print(f"Symbols: {symbols}")
    print(f"Engines: {engines}")
    print(f"Interval: {interval}")
    print(f"Backbone: {args.backbone}")
    print(f"Signal policy: {args.signal_policy} | raw_ai: {args.raw_ai_strategy}")
    if str(args.signal_engines).strip():
        print(f"Signal sources: {args.signal_engines}")
    print(f"AI threshold: {ai_threshold:g} ({'raw argmax' if ai_threshold <= 0 else 'gated'})")
    if args.session_hold is None:
        session_hold = False
    else:
        session_hold = bool(args.session_hold)
    if session_hold:
        print(
            f"Replay gate: session_hold "
            f"(pre-session predict → open@start close@end, hours={args.session_hours})"
        )
    else:
        print("Replay gate: every-bar (no session_hold)")
    print(
        "Entry windows: day_trend → UTC [00:00, NY_end-1h); "
        "session_trend → [session_start-1h, session_end-1h)"
    )
    if args.ma_ai_gate and args.swing_sl:
        raise SystemExit("Choose only one of --ma-ai-gate / --swing-sl")
    if (args.entry_arm or args.exit_arm) and (args.ma_ai_gate or args.swing_sl):
        raise SystemExit("--entry-arm/--exit-arm cannot combine with --ma-ai-gate / --swing-sl")
    if args.ma_ai_gate:
        gate_desc = f"MA+AI (SMA{args.ma_len})"
    elif args.entry_arm or args.exit_arm:
        arms = []
        if args.entry_arm:
            arms.append("entry_arm")
        if args.exit_arm:
            arms.append("exit_arm")
        gate_desc = "+".join(arms) + f" (SMA{args.ma_len}, raw_ai)"
    elif args.swing_sl:
        gate_desc = (
            f"swing_sl (len={args.swing_len} R∈[{args.r_min_atr:g},{args.r_max_atr:g}]ATR "
            f"pct≤{args.r_max_pct:g} chase≤{args.chase_frac:g})"
        )
    else:
        gate_desc = f"{args.signal_policy}:{args.raw_ai_strategy}"
    print(f"Replay gate: {gate_desc}")
    print(f"Checkpoint root: {ckpt_root}")
    if label_mode == "forward_return":
        label_desc = (
            f"forward_return horizon={args.horizon} thr={args.return_threshold:g}"
        )
    elif label_mode == "next_direction":
        label_desc = (
            "next_direction (range break: hi>prev→L, lo<prev→S; "
            "both→larger excursion; inside→FLAT)"
        )
    elif label_mode == "pct_barrier":
        label_desc = (
            f"pct_barrier ±{args.barrier_pct:g} horizon={args.barrier_horizon}"
        )
    elif label_mode == "session_direction":
        pct = 0.005 if abs(float(args.barrier_pct) - 0.02) < 1e-15 else float(args.barrier_pct)
        label_desc = (
            f"session_direction open→close ±{pct:g} "
            f"(session_hours={args.session_hours})"
        )
    elif label_mode == "session_trend":
        raw_k = float(args.barrier_k)
        k = 2.0 if abs(raw_k - 1.0) < 1e-15 else raw_k
        label_desc = (
            f"session_trend ±{k:g}*ATR first-touch before session end "
            f"pred=[start-1h,end-1h) (session_hours={args.session_hours})"
        )
    elif label_mode == "day_trend":
        raw_pct = float(args.barrier_pct)
        pct = 0.01 if abs(raw_pct - 0.02) < 1e-15 else raw_pct
        label_desc = (
            f"day_trend today's NY end close vs ±{pct:g} "
            f"(session_hours={args.session_hours})"
        )
    elif label_mode == "pivot_breakout":
        raw_h = int(args.barrier_horizon)
        hor = 24 if raw_h == 4 else raw_h
        label_desc = (
            f"pivot_breakout last unbroken pivot H/L break "
            f"within {hor} bars (pivot_len={args.pivot_len})"
        )
    else:
        label_desc = f"triple_barrier k={args.barrier_k:g} horizon={args.barrier_horizon}"
    print(
        f"Labels: {label_desc} "
        f"| Window: train_years={train_years:g} → 1h∩15m align → holdout 360d"
    )

    if not args.skip_train:
        from smartbs_engines.engines import normalize_feature_engine

        for sym in symbols:
            for eng in engines:
                ckpt_eng = normalize_feature_engine(eng)
                ckpt = ckpt_root / ckpt_eng / f"{sym}.pt"
                if args.skip_existing and ckpt.is_file():
                    print(f"\n========== SKIP {sym}/{eng} (exists) ==========")
                    continue
                print(f"\n========== TRAIN {sym} / {eng} ==========")
                try:
                    path = train_one(
                        sym,
                        eng,
                        ckpt_root,
                        train_years=train_years,
                        label_mode=label_mode,
                        horizon=int(args.horizon),
                        return_threshold=float(args.return_threshold),
                        barrier_k=float(args.barrier_k),
                        barrier_horizon=int(args.barrier_horizon),
                        ma_len=int(args.ma_len),
                        barrier_pct=float(args.barrier_pct),
                        session_hours=str(args.session_hours),
                        pivot_len=int(args.pivot_len),
                        interval=interval,
                        backbone=str(args.backbone),
                        kernel_size=int(args.kernel_size) or None,
                        signal_engines=str(args.signal_engines or ""),
                        signal_policy=str(args.signal_policy),
                        raw_ai_strategy=str(args.raw_ai_strategy),
                    )
                    print(f"  -> {path}")
                except Exception as exc:
                    print(f"  FAIL {sym}/{eng}: {exc}")

    rows: list[dict] = []
    print(
        f"\n{'sym':8} {'eng':16} {'days':>5} {'pnl':>10} {'dd':>10} "
        f"{'tr':>5} {'wr%':>6} {'pf':>6}"
    )
    from smartbs_engines.engines import normalize_feature_engine

    for sym in symbols:
        for eng in engines:
            ckpt_eng = normalize_feature_engine(eng)
            ckpt = ckpt_root / ckpt_eng / f"{sym}.pt"
            if not ckpt.is_file():
                print(f"{sym:8} {eng:16} missing {ckpt.name}")
                continue
            try:
                st_rows = replay_raw_ai(
                    sym,
                    str(ckpt),
                    windows_days=list(WINDOWS),
                    lot=LOTS.get(sym, 0.1),
                    point_value=PVS.get(sym, 1.0),
                    interval=interval,
                    ai_threshold=ai_threshold,
                    ma_ai_gate=bool(args.ma_ai_gate),
                    ma_len=int(args.ma_len),
                    entry_arm=bool(args.entry_arm),
                    exit_arm=bool(args.exit_arm),
                    swing_sl=bool(args.swing_sl),
                    swing_len=int(args.swing_len),
                    r_min_atr=float(args.r_min_atr),
                    r_max_atr=float(args.r_max_atr),
                    r_max_pct=float(args.r_max_pct),
                    chase_frac=float(args.chase_frac),
                    raw_ai_override=str(args.raw_ai_strategy),
                    signal_policy_override=str(args.signal_policy),
                    session_hold=session_hold,
                    session_hours=str(args.session_hours),
                )
            except Exception as exc:
                print(f"{sym:8} {eng:16} FAIL replay: {exc}")
                continue
            for st in st_rows:
                rows.append(st)
                print(
                    f"{st['symbol']:8} {st['engine']:16} {st['days']:5d} "
                    f"{st['pnl']:10.1f} {st['dd']:10.1f} {st['trades']:5d} "
                    f"{st['wr']:6.1f} {st['pf']:6.2f}"
                )

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{out_stem}_replay.csv"
    df.to_csv(csv_path, index=False)

    summary: dict[str, dict] = {}
    for days in WINDOWS:
        sub = df[df["days"] == days]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="symbol", columns="engine", values="pnl", aggfunc="first")
        piv_wr = sub.pivot_table(index="symbol", columns="engine", values="wr", aggfunc="first")
        best = {}
        for sym in piv.index:
            row = piv.loc[sym].dropna()
            if len(row):
                eng = str(row.idxmax())
                wr_v = (
                    float(piv_wr.loc[sym, eng])
                    if eng in piv_wr.columns and pd.notna(piv_wr.loc[sym, eng])
                    else None
                )
                best[sym] = {"engine": eng, "pnl": float(row[eng]), "wr": wr_v}
        summary[str(days)] = {
            "matrix": {
                s: {e: float(v) if pd.notna(v) else None for e, v in piv.loc[s].items()}
                for s in piv.index
            },
            "wr_matrix": {
                s: {
                    e: float(v) if pd.notna(v) else None
                    for e, v in piv_wr.loc[s].items()
                }
                for s in piv_wr.index
            },
            "best": best,
        }
        print(f"\n=== PnL matrix @ {days}d ===")
        print(piv.round(1).to_string())
        print(f"\n=== Win-rate % @ {days}d ===")
        print(piv_wr.round(1).to_string())
        print("Best per asset:")
        for s, b in best.items():
            wr_s = f", wr={b['wr']:.1f}%" if b.get("wr") is not None else ""
            print(f"  {s}: {b['engine']} ({b['pnl']:+.1f}{wr_s})")

    summary_path = out_dir / f"{out_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")
    print("========== ALL DONE ==========")


if __name__ == "__main__":
    main()
