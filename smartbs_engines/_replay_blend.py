"""Replay equal-weight ACTIVE_BLEND_ENGINES (mean probs → argmax) for all assets.

Engines default: ACTIVE_BLEND_ENGINES (maribbon, dbb, trend_pullback, smart_money, macd).

Usage:
  python -m smartbs_engines._replay_blend
  python -m smartbs_engines._replay_blend --engines maribbon,macd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartbs_engines.data import fetch_klines
from smartbs_engines.predict import predict_blend_probs
from smartbs_engines.registry import ACTIVE_BLEND_ENGINES

COMMODITIES = ("XAUUSD", "XAGUSD", "XTIUSD", "NATGAS", "PLATINUM")
WINDOWS = (90, 180, 360)
LOTS = {
    "XAUUSD": 0.1,
    "XAGUSD": 0.5,
    "XTIUSD": 0.1,
    "NATGAS": 0.1,
    "PLATINUM": 0.1,
}
PVS = {
    "XAUUSD": 1.0,
    "XAGUSD": 5.0,
    "XTIUSD": 1.0,
    "NATGAS": 1.0,
    "PLATINUM": 1.0,
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


def replay_blend(
    symbol: str,
    blend_root: str,
    *,
    engines: tuple[str, ...],
    windows_days: list[int],
    lot: float,
    point_value: float,
) -> list[dict]:
    df = fetch_klines(symbol, "1h", max_candles=0, source="mt5")
    end_ms = int(df["open_time"].iloc[-1])
    max_days = max(windows_days)
    cutoff_max = end_ms - max_days * 86_400_000
    hold = df[df["open_time"] >= cutoff_max].reset_index(drop=True)
    warm = df[df["open_time"] < cutoff_max].tail(3000)
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
    cls = probs.argmax(axis=1)
    opens = full["open"].to_numpy(dtype=np.float64)
    times = full["open_time"].to_numpy(dtype=np.int64)
    n = len(full)

    pos = np.zeros(n, dtype=np.int8)
    for i in range(max(lookback - 1, hold_start), n - 1):
        c = int(cls[i])
        if c == 0:
            continue
        pos[i + 1] = 1 if c == 1 else -1

    ret = np.zeros(n, dtype=np.float64)
    for i in range(hold_start, n - 1):
        if pos[i] == 0:
            continue
        ret[i] = pos[i] * (opens[i + 1] - opens[i]) * lot * point_value

    rows = []
    for days in windows_days:
        cut = end_ms - days * 86_400_000
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
                "engine": "blend5",
                "symbol": symbol,
                "days": days,
                "from": fr,
                "to": to,
                "engines": ",".join(engines),
            }
        )
        rows.append(st)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(COMMODITIES))
    ap.add_argument("--engines", default=",".join(ACTIVE_BLEND_ENGINES))
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    engines = tuple(e.strip().lower() for e in args.engines.split(",") if e.strip())
    ckpt_root = Path(__file__).resolve().parent / "checkpoints"
    out_dir = Path(__file__).resolve().parent / "sweep_st"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Symbols: {symbols}")
    print(f"Blend engines ({len(engines)}): {engines}")
    print(f"Checkpoint root: {ckpt_root}")
    print(f"Method: equal-weight mean probs → argmax (raw_ai, next-bar open)")

    rows: list[dict] = []
    print(
        f"\n{'sym':8} {'eng':8} {'days':>5} {'pnl':>10} {'dd':>10} "
        f"{'tr':>5} {'wr%':>6} {'pf':>6}"
    )
    for sym in symbols:
        try:
            st_rows = replay_blend(
                sym,
                str(ckpt_root),
                engines=engines,
                windows_days=list(WINDOWS),
                lot=LOTS.get(sym, 0.1),
                point_value=PVS.get(sym, 1.0),
            )
        except Exception as exc:
            print(f"{sym:8} blend5   FAIL: {exc}")
            continue
        for st in st_rows:
            rows.append(st)
            print(
                f"{st['symbol']:8} {st['engine']:8} {st['days']:5d} "
                f"{st['pnl']:10.1f} {st['dd']:10.1f} {st['trades']:5d} "
                f"{st['wr']:6.1f} {st['pf']:6.2f}"
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "blend5_replay.csv"
    df.to_csv(csv_path, index=False)

    summary: dict[str, dict] = {}
    for days in WINDOWS:
        sub = df[df["days"] == days]
        if sub.empty:
            continue
        by_sym = {
            str(r["symbol"]): {
                "pnl": float(r["pnl"]),
                "dd": float(r["dd"]),
                "trades": int(r["trades"]),
                "wr": float(r["wr"]),
                "pf": float(r["pf"]),
            }
            for _, r in sub.iterrows()
        }
        summary[str(days)] = by_sym
        print(f"\n=== Blend5 raw_ai @ {days}d ===")
        for s in symbols:
            if s not in by_sym:
                continue
            b = by_sym[s]
            print(
                f"  {s:8} pnl={b['pnl']:+8.1f}  dd={b['dd']:8.1f}  "
                f"tr={b['trades']:4d}  wr={b['wr']:5.1f}%  pf={b['pf']:5.2f}"
            )

    (out_dir / "blend5_summary.json").write_text(
        json.dumps(
            {"engines": list(engines), "method": "simple_mean→argmax", "windows": summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {csv_path}")
    print("========== BLEND5 DONE ==========")


if __name__ == "__main__":
    main()
