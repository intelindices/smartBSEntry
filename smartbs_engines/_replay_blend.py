"""Replay equal-weight mean blend (probs → argmax) for dump assets.

Defaults run two pools:
  blend5 = ACTIVE_BLEND_ENGINES (live pool)
  blend7 = BLEND_ENGINES (research pool)
  blend9 = candle,dbb,macd,maribbon,regime,rsi_div,signals,smart_money,trend_pb

Usage:
  python -m smartbs_engines._replay_blend
  python -m smartbs_engines._replay_blend --pools blend5
  python -m smartbs_engines._replay_blend --pools blend9 --symbols XAUUSD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartbs_engines.data import fetch_klines
from smartbs_engines.mt5_data import DEFAULT_DUMP_ASSETS
from smartbs_engines.predict import predict_blend_probs
from smartbs_engines.registry import ACTIVE_BLEND_ENGINES, BLEND_ENGINES

WINDOWS = (90, 180, 360)
# Forex lot=1; commodities + crypto lot=0.1
LOTS = {
    "XAUUSD": 0.1,
    "XAGUSD": 0.5,
    "XTIUSD": 0.1,
    "NATGAS": 0.1,
    "PLATINUM": 0.1,
    "BTCUSD": 0.1,
    "ETHUSD": 0.1,
    "LTCUSD": 0.1,
    "ADAUSD": 0.1,
    "SOLUSD": 0.1,
    "EURUSD": 1.0,
    "GBPUSD": 1.0,
    "AUDUSD": 1.0,
    "USDCHF": 1.0,
    "USDJPY": 1.0,
}
# Non-FX: price dollars × lot × PV. XAG uses 5 as a simple silver scale.
PVS = {s: 1.0 for s in DEFAULT_DUMP_ASSETS}
PVS["XAGUSD"] = 5.0

# Mini-scale FX: 1_000 units (= 1/100 of a standard 100k lot) so PnL stays readable vs crypto.
FX_USD_QUOTE = frozenset({"EURUSD", "GBPUSD", "AUDUSD"})
FX_USD_BASE = frozenset({"USDJPY", "USDCHF"})
FX_CONTRACT = 1_000.0  # was 100_000; user scale = 1/100 of standard lot

POOLS = {
    "blend_live": tuple(ACTIVE_BLEND_ENGINES),
    "blend5": tuple(ACTIVE_BLEND_ENGINES),  # legacy alias
    "blend_research": tuple(BLEND_ENGINES),
    "blend7": tuple(BLEND_ENGINES),  # legacy alias
    "blend9": (
        "dbb",
        "macd",
        "maribbon",
        "rsi_divergence",
        "smart_money",
        "trend_pullback",
        "common",
    ),
}


def _bar_pnl_usd(
    symbol: str,
    side: int,
    entry: float,
    exit_: float,
    lot: float,
    point_value: float,
) -> float:
    """Signed USD PnL for one bar hold (next-bar open model)."""
    dpx = (exit_ - entry) * float(side)
    if symbol in FX_USD_QUOTE:
        return dpx * FX_CONTRACT * lot
    if symbol in FX_USD_BASE:
        # Quote-currency PnL -> USD at exit
        if exit_ <= 0:
            return 0.0
        return (dpx * FX_CONTRACT * lot) / exit_
    return dpx * lot * point_value


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


def _classes_from_probs(probs: np.ndarray, ai_threshold: float) -> np.ndarray:
    """Map (n,3) probs → class ids. Gate: max(P_long,P_short) < thr → FLAT (EA InpAiThreshold)."""
    p_long = probs[:, 1]
    p_short = probs[:, 2]
    side = np.where(p_long >= p_short, 1, 2).astype(np.int64)
    conf = np.maximum(p_long, p_short)
    thr = float(ai_threshold)
    if thr <= 0.0:
        return probs.argmax(axis=1).astype(np.int64)
    return np.where(conf < thr, 0, side).astype(np.int64)


def replay_blend(
    symbol: str,
    blend_root: str,
    *,
    pool_name: str,
    engines: tuple[str, ...],
    windows_days: list[int],
    lot: float,
    point_value: float,
    interval: str = "1h",
    ai_threshold: float = 0.0,
    session_hold: bool = False,
    session_hours: str = "normal",
) -> list[dict]:
    from smartbs_engines.common_channels import set_session_hours_mode
    from smartbs_engines._train_replay_all_st import _simulate_session_hold

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

    probs, meta = predict_blend_probs(
        full,
        blend_root,
        trade_pair=symbol,
        engines=engines,
        symbol=symbol,
        data_source="mt5",
    )
    lookback = int(meta.get("lookback", 64))
    cls = _classes_from_probs(probs, ai_threshold)
    opens = full["open"].to_numpy(dtype=np.float64)
    closes = full["close"].to_numpy(dtype=np.float64)
    times = full["open_time"].to_numpy(dtype=np.int64)
    n = len(full)
    start = max(lookback - 1, hold_start)

    sh_mode = str(session_hours or "normal").strip().lower() or "normal"
    set_session_hours_mode(sh_mode)

    from smartbs_engines._train_replay_all_st import (
        _day_trend_entry_ok,
        _session_trend_entry_ok,
    )

    label_mode = str(meta.get("label_mode", "")).strip().lower()
    if label_mode == "day_trend":
        entry_ok = _day_trend_entry_ok(times, session_hours=sh_mode)
        win_tag = "+day_win"
    elif label_mode == "session_trend":
        entry_ok = _session_trend_entry_ok(times, session_hours=sh_mode)
        win_tag = "+sess_win"
    else:
        entry_ok = np.ones(n, dtype=bool)
        win_tag = ""

    sess_tag: np.ndarray | None = None
    if session_hold:
        pos, _unit_ret, sess_tag = _simulate_session_hold(
            cls,
            opens,
            closes,
            times,
            start=start,
            n=n,
            lot=1.0,
            point_value=1.0,
            session_hours=sh_mode,
        )
        ret = np.zeros(n, dtype=np.float64)
        for s0 in np.unique(sess_tag[sess_tag >= 0]):
            win = np.flatnonzero(sess_tag == s0)
            if len(win) == 0:
                continue
            i0, i1 = int(win[0]), int(win[-1])
            side = int(pos[i0])
            if side == 0:
                continue
            ret[i1] = _bar_pnl_usd(
                symbol,
                side,
                float(opens[i0]),
                float(closes[i1]),
                lot,
                point_value,
            )
        ret[:hold_start] = 0.0
        pos[:hold_start] = 0
        sess_tag[:hold_start] = -1
        gate = "session_hold"
    else:
        pos = np.zeros(n, dtype=np.int8)
        for i in range(start, n - 1):
            if not entry_ok[i]:
                continue
            c = int(cls[i])
            if c == 0:
                continue
            pos[i + 1] = 1 if c == 1 else -1

        ret = np.zeros(n, dtype=np.float64)
        for i in range(hold_start, n - 1):
            if pos[i] == 0:
                continue
            ret[i] = _bar_pnl_usd(
                symbol,
                int(pos[i]),
                float(opens[i]),
                float(opens[i + 1]),
                lot,
                point_value,
            )
        gate = "raw_ai" + win_tag

    rows = []
    for days in windows_days:
        cut = end_ms - days * 86_400_000
        if session_hold and sess_tag is not None:
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
                "engine": pool_name,
                "symbol": symbol,
                "days": days,
                "from": fr,
                "to": to,
                "engines": ",".join(engines),
                "gate": gate,
            }
        )
        rows.append(st)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(DEFAULT_DUMP_ASSETS))
    ap.add_argument(
        "--pools",
        default="blend5,blend7",
        help="Comma list: blend5 | blend7 | blend9 | blend_live | blend_research",
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
        help="Bar interval for replay series (default 1h)",
    )
    ap.add_argument(
        "--ai-threshold",
        type=float,
        default=0.0,
        help="Min max(P_long,P_short); below → FLAT (0=raw argmax). EA default 0.35",
    )
    ap.add_argument(
        "--out-stem",
        default="blend",
        help="Output stem under sweep_st/ (writes {stem}_replay.csv + {stem}_summary.json)",
    )
    ap.add_argument(
        "--session-hold",
        action="store_true",
        help="Pre-session AI → open at session start, close at session end",
    )
    ap.add_argument(
        "--session-hours",
        choices=["normal", "adjusted"],
        default="normal",
        help="UTC sessions for --session-hold (default normal; use adjusted for 0-7/7-12/12-20)",
    )
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    pool_names = [p.strip().lower() for p in args.pools.split(",") if p.strip()]
    for p in pool_names:
        if p not in POOLS:
            raise SystemExit(f"Unknown pool {p!r}; choose from {list(POOLS)}")

    base = Path(__file__).resolve().parent
    ckpt_root = Path(args.ckpt_root) if str(args.ckpt_root).strip() else base / "checkpoints"
    interval = str(args.interval).strip().lower()
    ai_threshold = float(args.ai_threshold)
    out_stem = str(args.out_stem).strip() or "blend"
    out_dir = base / "sweep_st"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Symbols: {symbols}")
    print(f"Interval: {interval}")
    print(f"AI threshold: {ai_threshold:g} ({'raw argmax' if ai_threshold <= 0 else 'gated'})")
    print(f"Checkpoint root: {ckpt_root}")
    if args.session_hold:
        print(
            f"Method: session_hold (pre-session predict → open@start close@end, "
            f"hours={args.session_hours})"
        )
    else:
        print("Method: equal-weight mean probs -> gated side (raw_ai, next-bar open)")
    print(
        "Lots: forex=1.0, commodities+crypto=0.1 | "
        f"FX contract={FX_CONTRACT:g} (USD-quote / USD-base via exit)"
    )

    rows: list[dict] = []
    print(
        f"\n{'sym':8} {'pool':8} {'days':>5} {'pnl':>10} {'dd':>10} "
        f"{'tr':>5} {'wr%':>6} {'pf':>6}"
    )
    for pool_name in pool_names:
        engines = POOLS[pool_name]
        print(f"\n========== {pool_name} ({len(engines)}): {engines} ==========")
        for sym in symbols:
            try:
                st_rows = replay_blend(
                    sym,
                    str(ckpt_root),
                    pool_name=pool_name,
                    engines=engines,
                    windows_days=list(WINDOWS),
                    lot=LOTS.get(sym, 0.1),
                    point_value=PVS.get(sym, 1.0),
                    interval=interval,
                    ai_threshold=ai_threshold,
                    session_hold=bool(args.session_hold),
                    session_hours=str(args.session_hours),
                )
            except Exception as exc:
                print(f"{sym:8} {pool_name:8} FAIL: {exc}")
                continue
            for st in st_rows:
                rows.append(st)
                print(
                    f"{st['symbol']:8} {st['engine']:8} {st['days']:5d} "
                    f"{st['pnl']:10.1f} {st['dd']:10.1f} {st['trades']:5d} "
                    f"{st['wr']:6.1f} {st['pf']:6.2f}"
                )

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{out_stem}_replay.csv"
    df.to_csv(csv_path, index=False)

    summary: dict = {
        "method": (
            "session_hold"
            if args.session_hold
            else "simple_mean→gated_side"
        ),
        "ai_threshold": ai_threshold,
        "interval": interval,
        "ckpt_root": str(ckpt_root),
        "session_hold": bool(args.session_hold),
        "session_hours": str(args.session_hours),
        "pools": {p: list(POOLS[p]) for p in pool_names},
        "windows": {},
    }
    for days in WINDOWS:
        day_key = str(days)
        summary["windows"][day_key] = {}
        print(f"\n=== Blend comparison @ {days}d ===")
        hdr = f"{'sym':8}" + "".join(f" {p:>10}" for p in pool_names)
        print(hdr)
        for sym in symbols:
            cell: dict = {}
            for pool_name in pool_names:
                sub = df[(df["days"] == days) & (df["symbol"] == sym) & (df["engine"] == pool_name)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                cell[pool_name] = {
                    "pnl": float(r["pnl"]),
                    "dd": float(r["dd"]),
                    "trades": int(r["trades"]),
                    "wr": float(r["wr"]),
                    "pf": float(r["pf"]),
                }
            if cell:
                summary["windows"][day_key][sym] = cell
                parts = [f"{sym:8}"]
                for p in pool_names:
                    v = cell.get(p, {}).get("pnl")
                    parts.append(f" {v:+.1f}" if v is not None else f" {'—':>10}")
                    if v is not None:
                        parts[-1] = f" {v:+10.1f}"
                print("".join(parts))

    summary_path = out_dir / f"{out_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Keep legacy filenames only when writing the default stem.
    if out_stem == "blend":
        b5 = df[df["engine"] == "blend5"]
        if not b5.empty:
            b5.to_csv(out_dir / "blend5_replay.csv", index=False)
            legacy = {
                "engines": list(POOLS["blend5"]),
                "method": "simple_mean→argmax",
                "windows": {
                    d: {
                        s: v["blend5"]
                        for s, v in summary["windows"][d].items()
                        if "blend5" in v
                    }
                    for d in summary["windows"]
                },
            }
            (out_dir / "blend5_summary.json").write_text(
                json.dumps(legacy, indent=2), encoding="utf-8"
            )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")
    print("========== BLEND REPLAY DONE ==========")


if __name__ == "__main__":
    main()
