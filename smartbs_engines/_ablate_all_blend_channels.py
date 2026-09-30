"""Per-channel leave-one-out ablation on ``all_blend`` (core + common).

Default mode ``occlusion``: freeze the full all_blend checkpoint, zero one
channel at a time at inference, replay 90/180/360d, rank by ΔPnL / ΔWR.

``retrain`` mode: LOGO retrain with that channel zeroed (slow; 155× train).

Importance = baseline − ablated (higher ⇒ channel helped more).

Example::

  python -m smartbs_engines._ablate_all_blend_channels \\
    --symbols XAUUSD --backbone smartBSEntryV2 \\
    --ckpt E:/.../checkpoints_all_blend_pivot_breakout_p5_xau_entryv2/all_blend/XAUUSD.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from smartbs_engines._train_replay_all_st import (
    LOTS,
    PVS,
    WINDOWS,
    _classes_from_probs,
    _stats,
)
from smartbs_engines.calibration import softmax_np
from smartbs_engines.checkpoint import load_classifier
from smartbs_engines.data import fetch_klines
from smartbs_engines.engine_all_blend import all_blend_channel_catalog, all_blend_feature_names
from smartbs_engines.features import build_feature_matrix


def _probs_from_feature_matrix(
    model: torch.nn.Module,
    feats: np.ndarray,
    *,
    lookback: int,
    temperature: float,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Windowed forward over ``feats`` → ``(n, 3)`` probs (same layout as predict)."""
    n = len(feats)
    out = np.zeros((n, 3), dtype=np.float64)
    if n < lookback:
        return out
    idxs = list(range(lookback - 1, n))
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s : s + batch_size]
            windows = np.stack(
                [feats[i - lookback + 1 : i + 1].T for i in chunk],
                axis=0,
            )
            x = torch.from_numpy(windows).float().to(device)
            logits = model(x).detach().cpu().numpy()
            probs = softmax_np(logits, temperature)
            for j, i in enumerate(chunk):
                out[i] = probs[j]
    return out


def _replay_from_probs(
    *,
    probs: np.ndarray,
    full: pd.DataFrame,
    hold_start: int,
    end_ms: int,
    windows_days: list[int],
    lot: float,
    point_value: float,
    ai_threshold: float,
    lookback: int,
    symbol: str,
    engine: str,
) -> list[dict]:
    opens = full["open"].to_numpy(dtype=np.float64)
    times = full["open_time"].to_numpy(dtype=np.int64)
    n = len(full)
    start = max(lookback - 1, hold_start)
    cls = _classes_from_probs(probs, ai_threshold, force_side=False)
    pos = np.zeros(n, dtype=np.int8)
    for i in range(start, n - 1):
        c = int(cls[i])
        if c == 0:
            continue
        pos[i + 1] = 1 if c == 1 else -1
    ret = np.zeros(n, dtype=np.float64)
    for i in range(hold_start, n - 1):
        if pos[i] == 0:
            continue
        ret[i] = pos[i] * (opens[i + 1] - opens[i]) * lot * point_value

    rows: list[dict] = []
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
                "engine": engine,
                "symbol": symbol,
                "days": days,
                "from": fr,
                "to": to,
                "gate": "raw_ai",
            }
        )
        rows.append(st)
    return rows


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default="XAUUSD")
    ap.add_argument("--backbone", default="smartBSEntryV2")
    ap.add_argument(
        "--ckpt",
        default="",
        help="Full all_blend checkpoint (default: baseline root all_blend/SYM.pt)",
    )
    ap.add_argument(
        "--baseline-ckpt-root",
        default="",
        help="Root with all_blend/{SYM}.pt",
    )
    ap.add_argument("--ai-threshold", type=float, default=0.0)
    ap.add_argument("--session-hours", default="normal")
    ap.add_argument(
        "--mode",
        choices=["occlusion", "retrain"],
        default="occlusion",
        help="occlusion=zero at infer on frozen ckpt (default); retrain=LOGO train",
    )
    ap.add_argument(
        "--channels",
        default="",
        help="Comma channel names or indices (default=all)",
    )
    ap.add_argument("--out-stem", default="all_blend_channel_ablation")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    catalog = all_blend_channel_catalog(with_common=True)
    names = all_blend_feature_names(with_common=True)
    n_ch = len(catalog)
    if n_ch == 0 or len(names) != n_ch:
        raise SystemExit(f"all_blend catalog/names mismatch: {n_ch} vs {len(names)}")

    if str(args.channels).strip():
        selected: list[dict] = []
        for tok in args.channels.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.isdigit():
                idx = int(tok)
                selected.append(catalog[idx])
            else:
                idx = names.index(tok)
                selected.append(catalog[idx])
        channels = selected
    else:
        channels = catalog

    base = Path(__file__).resolve().parent
    bb_tag = {
        "smartbsentryv2": "entryv2",
        "tcn": "tcn",
        "smartbstf": "tf",
    }.get(str(args.backbone).strip().lower().replace("_", ""), str(args.backbone))
    baseline_root = (
        Path(args.baseline_ckpt_root)
        if str(args.baseline_ckpt_root).strip()
        else base / f"checkpoints_all_blend_pivot_breakout_p5_xau_{bb_tag}"
    )
    out_dir = base / "sweep_st"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = str(args.out_stem).strip() or "all_blend_channel_ablation"

    if args.mode != "occlusion":
        raise SystemExit(
            "retrain mode for 155 channels is disabled here; "
            "use --mode occlusion (frozen-ckpt channel zeroing). "
            "Group-level retrain LOGO: _ablate_all_blend_groups."
        )

    print(f"Symbols: {symbols}")
    print(f"Mode: occlusion | channels: {len(channels)} / {len(catalog)}")
    print(f"Baseline root: {baseline_root}")

    from smartbs_engines.common_channels import set_session_hours_mode

    set_session_hours_mode(str(args.session_hours))

    all_rows: list[dict] = []

    for sym in symbols:
        ckpt = (
            Path(args.ckpt)
            if str(args.ckpt).strip()
            else baseline_root / "all_blend" / f"{sym}.pt"
        )
        if not ckpt.is_file():
            raise SystemExit(f"Missing checkpoint: {ckpt}")
        print(f"\n========== LOAD {sym} {ckpt} ==========")

        model, cfg = load_classifier(str(ckpt))
        lookback = int(cfg.get("lookback", 64))
        temperature = float(cfg.get("temperature", 1.0) or 1.0)
        device = next(model.parameters()).device

        df = fetch_klines(sym, "1h", max_candles=0, source="mt5")
        end_ms = int(df["open_time"].iloc[-1])
        max_days = max(WINDOWS)
        cutoff_max = end_ms - max_days * 86_400_000
        hold = df[df["open_time"] >= cutoff_max].reset_index(drop=True)
        warm = df[df["open_time"] < cutoff_max].tail(3_000)
        full = pd.concat([warm, hold], ignore_index=True)
        hold_start = len(warm)

        feats = build_feature_matrix(
            full,
            feature_engine="all_blend",
            symbol=sym,
            data_source="mt5",
            ablation_zero_group="",
        )
        if feats.shape[1] != len(names):
            raise RuntimeError(
                f"feature width {feats.shape[1]} != catalog {len(names)}"
            )
        print(f"  bars={len(full)} hold_start={hold_start} feats={feats.shape}")

        # Baseline
        base_probs = _probs_from_feature_matrix(
            model,
            feats,
            lookback=lookback,
            temperature=temperature,
            device=device,
            batch_size=int(args.batch_size),
        )
        base_rows = _replay_from_probs(
            probs=base_probs,
            full=full,
            hold_start=hold_start,
            end_ms=end_ms,
            windows_days=list(WINDOWS),
            lot=LOTS.get(sym, 0.1),
            point_value=PVS.get(sym, 1.0),
            ai_threshold=float(args.ai_threshold),
            lookback=lookback,
            symbol=sym,
            engine="all_blend",
        )
        base_map = {int(r["days"]): r for r in base_rows}
        for r in base_rows:
            print(
                f"  baseline {r['days']:3d}d  pnl={r['pnl']:8.1f}  "
                f"wr={r['wr']:5.1f}%  tr={r['trades']}"
            )

        lot = LOTS.get(sym, 0.1)
        pv = PVS.get(sym, 1.0)
        for k, meta in enumerate(channels):
            idx = int(meta["idx"])
            name = str(meta["name"])
            group = str(meta["group"])
            if (k + 1) % 10 == 1 or k == 0:
                print(
                    f"  [{k+1}/{len(channels)}] zero {idx:3d} {group}/{name}",
                    flush=True,
                )
            feats_z = np.array(feats, dtype=np.float32, copy=True)
            feats_z[:, idx] = 0.0
            probs = _probs_from_feature_matrix(
                model,
                feats_z,
                lookback=lookback,
                temperature=temperature,
                device=device,
                batch_size=int(args.batch_size),
            )
            st_rows = _replay_from_probs(
                probs=probs,
                full=full,
                hold_start=hold_start,
                end_ms=end_ms,
                windows_days=list(WINDOWS),
                lot=lot,
                point_value=pv,
                ai_threshold=float(args.ai_threshold),
                lookback=lookback,
                symbol=sym,
                engine="all_blend",
            )
            row: dict = {
                "symbol": sym,
                "idx": idx,
                "channel": name,
                "group": group,
                "safe": _safe_name(name),
                "backbone": str(args.backbone),
                "mode": "occlusion",
            }
            for st in st_rows:
                d = int(st["days"])
                b = base_map[d]
                pnl = float(st["pnl"])
                wr = float(st["wr"])
                bp = float(b["pnl"])
                bw = float(b["wr"])
                row[f"pnl_{d}"] = pnl
                row[f"wr_{d}"] = wr
                row[f"trades_{d}"] = int(st["trades"])
                row[f"dd_{d}"] = float(st["dd"])
                row[f"base_pnl_{d}"] = bp
                row[f"base_wr_{d}"] = bw
                row[f"dPnL_{d}"] = bp - pnl
                row[f"dWR_{d}"] = bw - wr
            all_rows.append(row)

    if not all_rows:
        raise SystemExit("No channel ablation rows")

    df_out = pd.DataFrame(all_rows)
    csv_path = out_dir / f"{out_stem}_replay.csv"
    df_out.to_csv(csv_path, index=False)

    rankings: dict = {}
    for days in WINDOWS:
        by_pnl = sorted(all_rows, key=lambda r: float(r[f"dPnL_{days}"]), reverse=True)
        by_wr = sorted(all_rows, key=lambda r: float(r[f"dWR_{days}"]), reverse=True)
        rankings[f"rank_by_dPnL_{days}d"] = [
            {
                "rank": i + 1,
                "idx": r["idx"],
                "channel": r["channel"],
                "group": r["group"],
                "dPnL": r[f"dPnL_{days}"],
                "pnl": r[f"pnl_{days}"],
                "base_pnl": r[f"base_pnl_{days}"],
            }
            for i, r in enumerate(by_pnl)
        ]
        rankings[f"rank_by_dWR_{days}d"] = [
            {
                "rank": i + 1,
                "idx": r["idx"],
                "channel": r["channel"],
                "group": r["group"],
                "dWR": r[f"dWR_{days}"],
                "wr": r[f"wr_{days}"],
                "base_wr": r[f"base_wr_{days}"],
            }
            for i, r in enumerate(by_wr)
        ]

    summary = {
        "symbols": symbols,
        "backbone": str(args.backbone),
        "mode": "occlusion",
        "n_channels": len(channels),
        "importance": (
            "dPnL/dWR = baseline − ablated; "
            "higher means zeroing the channel hurt more (channel more valuable)"
        ),
        "baseline_ckpt_root": str(baseline_root),
        **rankings,
    }
    json_path = out_dir / f"{out_stem}_summary.json"
    # Keep summary lean (ranks only); full rows in CSV.
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print("\n=== Top 20 by ΔPnL @ 360d ===")
    for item in rankings["rank_by_dPnL_360d"][:20]:
        print(
            f"  #{item['rank']:3d}  [{item['group']:16}] {item['channel']:40}  "
            f"dPnL={item['dPnL']:+8.1f}"
        )
    print("\n=== Bottom 10 by ΔPnL @ 360d (hurting / noisy) ===")
    for item in rankings["rank_by_dPnL_360d"][-10:]:
        print(
            f"  #{item['rank']:3d}  [{item['group']:16}] {item['channel']:40}  "
            f"dPnL={item['dPnL']:+8.1f}"
        )
    print("\n=== Top 20 by ΔWR @ 360d ===")
    for item in rankings["rank_by_dWR_360d"][:20]:
        print(
            f"  #{item['rank']:3d}  [{item['group']:16}] {item['channel']:40}  "
            f"dWR={item['dWR']:+6.1f}pp"
        )


if __name__ == "__main__":
    main()
