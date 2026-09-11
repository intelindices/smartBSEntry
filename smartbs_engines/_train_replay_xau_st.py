"""Train each ST engine on XAUUSD (MT5, 15m-aligned window) + raw_ai replay.

Same train window as dBB:
  train_years=10 → train_from_date=2022-06-14 → align 15m → holdout 360d

Usage:
  python -m smartbs_engines._train_replay_xau_st
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smartbs_engines.predict import predict_probs
from smartbs_engines.registry import BLEND_ENGINES
from smartbs_engines.train import train_model
from smartbs_engines.config import SmartBSConfig
from smartbs_engines.features import num_inputs_for

from smartbs_engines.data import fetch_klines


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
    lot: float = 0.1,
    point_value: float = 1.0,
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
                "engine": cfg.get("feature_engine", "?"),
                "symbol": symbol,
                "days": days,
                "from": fr,
                "to": to,
            }
        )
        rows.append(st)
    return rows


def main() -> None:
    ckpt_root = Path(__file__).resolve().parent / "checkpoints"
    engines = list(BLEND_ENGINES)
    print(f"Engines: {engines}")
    print(f"Checkpoint root: {ckpt_root}")

    for eng in engines:
        print(f"\n========== TRAIN XAUUSD / {eng} ==========")
        cfg = SmartBSConfig(
            trade_pair="XAUUSD",
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
            train_assets=["XAUUSD"],
            checkpoint_path="",
        )
        # Force writes under smartbs_engines/checkpoints/<eng>/XAUUSD.pt
        import os

        os.environ["SMARTBS_CHECKPOINT_DIR"] = str(ckpt_root)
        path = train_model(cfg)
        print(f"  -> {path}")

    print(f"\n{'eng':16} {'days':>5} {'pnl':>10} {'dd':>10} {'tr':>5} {'wr%':>6} {'pf':>6}  window")
    for eng in engines:
        ckpt = ckpt_root / eng / "XAUUSD.pt"
        if not ckpt.is_file():
            print(f"{eng}: missing {ckpt}")
            continue
        rows = replay_raw_ai("XAUUSD", str(ckpt), windows_days=[90, 180, 360])
        for st in rows:
            print(
                f"{st['engine']:16} {st['days']:5d} "
                f"{st['pnl']:10.1f} {st['dd']:10.1f} {st['trades']:5d} "
                f"{st['wr']:6.1f} {st['pf']:6.2f}  {st['from']}->{st['to']}"
            )
    print("\n========== ALL DONE ==========")


if __name__ == "__main__":
    main()
