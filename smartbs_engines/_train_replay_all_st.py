"""Train all ST engines × commodities (MT5, 15m-aligned) + comparison replay.

Window (same as dBB):
  train_years=10 → train_from_date=2022-06-14 → align 15m → holdout 360d

Usage:
  python -m smartbs_engines._train_replay_all_st
  python -m smartbs_engines._train_replay_all_st --skip-train   # replay only
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
from smartbs_engines.predict import predict_probs
from smartbs_engines.registry import BLEND_ENGINES
from smartbs_engines.train import train_model

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


def replay_raw_ai(
    symbol: str,
    checkpoint: str,
    *,
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

    probs, cfg = predict_probs(full, checkpoint, symbol=symbol, data_source="mt5")
    lookback = int(cfg.get("lookback", 64))
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
                "engine": str(cfg.get("feature_engine", "?")),
                "symbol": symbol,
                "days": days,
                "from": fr,
                "to": to,
            }
        )
        rows.append(st)
    return rows


def train_one(sym: str, eng: str, ckpt_root: Path) -> str:
    cfg = SmartBSConfig(
        trade_pair=sym,
        data_source="mt5",
        interval="1h",
        train_candles=0,
        train_years=10.0,
        train_from_date="2022-06-14",
        train_align_15m=True,
        holdout_days=360,
        epochs=15,
        batch_size=128,
        feature_engine=eng,
        num_inputs=num_inputs_for(eng),
        train_assets=[sym],
        checkpoint_path="",
    )
    os.environ["SMARTBS_CHECKPOINT_DIR"] = str(ckpt_root)
    return train_model(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(COMMODITIES))
    ap.add_argument("--engines", default=",".join(BLEND_ENGINES))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="Skip train if .pt exists")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    engines = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    ckpt_root = Path(__file__).resolve().parent / "checkpoints"
    out_dir = Path(__file__).resolve().parent / "sweep_st"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Symbols: {symbols}")
    print(f"Engines: {engines}")
    print(f"Checkpoint root: {ckpt_root}")

    if not args.skip_train:
        for sym in symbols:
            for eng in engines:
                ckpt = ckpt_root / eng / f"{sym}.pt"
                if args.skip_existing and ckpt.is_file():
                    print(f"\n========== SKIP {sym}/{eng} (exists) ==========")
                    continue
                print(f"\n========== TRAIN {sym} / {eng} ==========")
                try:
                    path = train_one(sym, eng, ckpt_root)
                    print(f"  -> {path}")
                except Exception as exc:
                    print(f"  FAIL {sym}/{eng}: {exc}")

    rows: list[dict] = []
    print(
        f"\n{'sym':8} {'eng':16} {'days':>5} {'pnl':>10} {'dd':>10} "
        f"{'tr':>5} {'wr%':>6} {'pf':>6}"
    )
    for sym in symbols:
        for eng in engines:
            ckpt = ckpt_root / eng / f"{sym}.pt"
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
    csv_path = out_dir / "asset_engine_replay.csv"
    df.to_csv(csv_path, index=False)

    # Pivot summaries per horizon
    summary: dict[str, dict] = {}
    for days in WINDOWS:
        sub = df[df["days"] == days]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="symbol", columns="engine", values="pnl", aggfunc="first")
        best = {}
        for sym in piv.index:
            row = piv.loc[sym].dropna()
            if len(row):
                eng = str(row.idxmax())
                best[sym] = {"engine": eng, "pnl": float(row[eng])}
        summary[str(days)] = {
            "matrix": {s: {e: float(v) if pd.notna(v) else None for e, v in piv.loc[s].items()} for s in piv.index},
            "best": best,
        }
        print(f"\n=== PnL matrix @ {days}d (lot 0.1 scale) ===")
        print(piv.round(1).to_string())
        print("Best per asset:")
        for s, b in best.items():
            print(f"  {s}: {b['engine']} ({b['pnl']:+.1f})")

    (out_dir / "asset_engine_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {csv_path}")
    print("========== ALL DONE ==========")


if __name__ == "__main__":
    main()
