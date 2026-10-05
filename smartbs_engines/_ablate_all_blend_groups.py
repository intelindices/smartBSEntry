"""Leave-one-group-out ablation on ``all_blend`` channel groups.

For each source group (regime_engine … rsi_divergence) + ``common``:
  1. Retrain with that group's columns zeroed (155-dim kept)
  2. Replay 90 / 180 / 360d raw-AI argmax
  3. Rank by ΔPnL and ΔWR vs full all_blend baseline
     (importance = baseline − ablated; higher = group helped more)

Example::

  python -m smartbs_engines._ablate_all_blend_groups \\
    --symbols XAUUSD --backbone smartBSEntryV2 \\
    --label-mode pivot_breakout --pivot-len 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from smartbs_engines._train_replay_all_st import LOTS, PVS, WINDOWS, replay_raw_ai
from smartbs_engines.engine_all_blend import (
    ALL_BLEND_ALIAS_GROUP_NAMES,
    ALL_BLEND_GROUPS,
    all_blend_group_channel_names,
    all_blend_group_indices,
)


def _rank_key_pnl(row: dict, days: int = 360) -> float:
    return float(row.get(f"dPnL_{days}", 0.0))


def _rank_key_wr(row: dict, days: int = 360) -> float:
    return float(row.get(f"dWR_{days}", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default="XAUUSD")
    ap.add_argument("--backbone", default="smartBSEntryV2")
    ap.add_argument("--label-mode", default="pivot_breakout")
    ap.add_argument("--barrier-horizon", type=int, default=24)
    ap.add_argument("--pivot-len", type=int, default=15)
    ap.add_argument("--train-years", type=float, default=10.0)
    ap.add_argument("--session-hours", default="normal")
    ap.add_argument("--ai-threshold", type=float, default=0.0)
    ap.add_argument(
        "--baseline-ckpt",
        default="",
        help="Full all_blend .pt (default: ckpt-root/all_blend/{SYM}.pt)",
    )
    ap.add_argument(
        "--ckpt-root",
        default="",
        help="Root for ablated ckpts (default: checkpoints_all_blend_ablate_…)",
    )
    ap.add_argument(
        "--baseline-ckpt-root",
        default="",
        help="Root containing full all_blend baseline",
    )
    ap.add_argument("--out-stem", default="all_blend_group_ablation")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--groups",
        default="",
        help="Comma subset of groups (default=all)",
    )
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    known = set(ALL_BLEND_GROUPS) | set(ALL_BLEND_ALIAS_GROUP_NAMES)
    groups = (
        [g.strip().lower() for g in args.groups.split(",") if g.strip()]
        if str(args.groups).strip()
        else list(ALL_BLEND_GROUPS)
    )
    for g in groups:
        if g not in known:
            raise SystemExit(f"Unknown group {g!r}; known={sorted(known)}")

    base = Path(__file__).resolve().parent
    bb_tag = {
        "smartbsentryv2": "entryv2",
        "tcn": "tcn",
        "smartbstf": "tf",
    }.get(str(args.backbone).strip().lower().replace("_", ""), str(args.backbone))
    label_tag = str(args.label_mode)
    if label_tag == "pivot_breakout":
        label_tag = f"pivot_breakout_p{int(args.pivot_len)}"

    ckpt_root = (
        Path(args.ckpt_root)
        if str(args.ckpt_root).strip()
        else base / f"checkpoints_all_blend_ablate_{label_tag}_xau_{bb_tag}"
    )
    baseline_root = (
        Path(args.baseline_ckpt_root)
        if str(args.baseline_ckpt_root).strip()
        else base / f"checkpoints_all_blend_{label_tag}_xau_{bb_tag}"
    )
    out_dir = base / "sweep_st"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = str(args.out_stem).strip() or "all_blend_group_ablation"

    print(f"Symbols: {symbols}")
    print(f"Backbone: {args.backbone}")
    print(f"Groups ({len(groups)}): {groups}")
    print(f"Baseline root: {baseline_root}")
    print(f"Ablate ckpt root: {ckpt_root}")
    print(
        f"Labels: {args.label_mode} horizon={args.barrier_horizon} "
        f"pivot_len={args.pivot_len} | LOGO zero-group retrain"
    )

    # --- baseline replay (reuse existing full all_blend ckpt) ---
    baseline_by_sym: dict[str, dict[int, dict]] = {}
    for sym in symbols:
        bckpt = (
            Path(args.baseline_ckpt)
            if str(args.baseline_ckpt).strip()
            else baseline_root / "all_blend" / f"{sym}.pt"
        )
        if not bckpt.is_file():
            raise SystemExit(f"Missing baseline checkpoint: {bckpt}")
        print(f"\n========== BASELINE replay {sym} ==========")
        print(f"  ckpt={bckpt}")
        rows = replay_raw_ai(
            sym,
            str(bckpt),
            windows_days=list(WINDOWS),
            lot=LOTS.get(sym, 0.1),
            point_value=PVS.get(sym, 1.0),
            interval="1h",
            ai_threshold=float(args.ai_threshold),
            session_hours=str(args.session_hours),
        )
        baseline_by_sym[sym] = {int(r["days"]): r for r in rows}
        for r in rows:
            print(
                f"  baseline {r['days']:3d}d  pnl={r['pnl']:8.1f}  "
                f"wr={r['wr']:5.1f}%  tr={r['trades']}"
            )

    # --- LOGO trains + replays ---
    from smartbs_engines.config import SmartBSConfig
    from smartbs_engines.features import num_inputs_for
    from smartbs_engines.model import default_kernel_size, normalize_backbone
    from smartbs_engines.pipeline import PipelineSpec
    from smartbs_engines.train import train_model

    results: list[dict] = []
    for sym in symbols:
        for group in groups:
            n_ch = len(all_blend_group_indices(group, with_common=True))
            ch_names = all_blend_group_channel_names(group, with_common=True)
            print(
                f"\n========== ABLATE {sym} / wo_{group} "
                f"({n_ch} ch zeroed) =========="
            )
            print(f"  channels: {', '.join(ch_names[:8])}"
                  + ("…" if len(ch_names) > 8 else ""))

            eng_dir = ckpt_root / f"wo_{group}"
            eng_dir.mkdir(parents=True, exist_ok=True)
            # Store under all_blend/ so resolved_checkpoint path works,
            # but isolate by group via SMARTBS_CHECKPOINT_DIR + subfolder.
            group_root = ckpt_root / f"wo_{group}"
            ckpt = group_root / "all_blend" / f"{sym}.pt"

            if not args.skip_train:
                if args.skip_existing and ckpt.is_file():
                    print(f"  SKIP train (exists): {ckpt}")
                else:
                    bb = normalize_backbone(str(args.backbone))
                    ks = default_kernel_size(bb)
                    pipe = PipelineSpec(
                        engines=("all_blend",),
                        signal_policy="none",
                        backbone=bb,
                        raw_ai="ai_only",
                    )
                    cfg = SmartBSConfig(
                        trade_pair=sym,
                        data_source="mt5",
                        interval="1h",
                        train_candles=0,
                        train_years=float(args.train_years),
                        train_from_date="",
                        train_align_15m=True,
                        holdout_days=360,
                        epochs=15,
                        batch_size=128,
                        feature_engine="all_blend",
                        ablation_zero_group=group,
                        signal_engines=(),
                        signal_policy=pipe.signal_policy,
                        raw_ai_strategy=pipe.raw_ai,
                        backbone=bb,
                        kernel_size=ks,
                        num_inputs=num_inputs_for("all_blend"),
                        train_assets=[sym],
                        checkpoint_path="",
                        label_mode=str(args.label_mode),
                        horizon=1,
                        barrier_horizon=int(args.barrier_horizon),
                        pivot_len=int(args.pivot_len),
                        session_hours=str(args.session_hours),
                    )
                    os.environ["SMARTBS_CHECKPOINT_DIR"] = str(group_root)
                    path = train_model(cfg)
                    print(f"  -> {path}")

            if not ckpt.is_file():
                # train_model may write via resolved path
                alt = Path(os.environ.get("SMARTBS_CHECKPOINT_DIR", group_root)) / "all_blend" / f"{sym}.pt"
                if alt.is_file():
                    ckpt = alt
                else:
                    print(f"  MISSING ckpt {ckpt}")
                    continue

            st_rows = replay_raw_ai(
                sym,
                str(ckpt),
                windows_days=list(WINDOWS),
                lot=LOTS.get(sym, 0.1),
                point_value=PVS.get(sym, 1.0),
                interval="1h",
                ai_threshold=float(args.ai_threshold),
                session_hours=str(args.session_hours),
            )
            base_map = baseline_by_sym[sym]
            row: dict = {
                "symbol": sym,
                "group": group,
                "n_channels": n_ch,
                "channels": ",".join(ch_names),
                "backbone": str(args.backbone),
                "label_mode": str(args.label_mode),
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
                # Positive = removing group hurt performance → group was helpful
                row[f"dPnL_{d}"] = bp - pnl
                row[f"dWR_{d}"] = bw - wr
                print(
                    f"  wo_{group:16} {d:3d}d  pnl={pnl:8.1f} (Δ{bp-pnl:+7.1f})  "
                    f"wr={wr:5.1f}% (Δ{bw-wr:+5.1f})  tr={st['trades']}"
                )
            results.append(row)

    if not results:
        raise SystemExit("No ablation results")

    df = pd.DataFrame(results)
    csv_path = out_dir / f"{out_stem}_replay.csv"
    df.to_csv(csv_path, index=False)

    # Rankings @ 360d (also emit 90/180)
    rankings: dict = {"baseline_root": str(baseline_root), "ckpt_root": str(ckpt_root)}
    for days in WINDOWS:
        by_pnl = sorted(results, key=lambda r: _rank_key_pnl(r, days), reverse=True)
        by_wr = sorted(results, key=lambda r: _rank_key_wr(r, days), reverse=True)
        rankings[f"rank_by_dPnL_{days}d"] = [
            {
                "rank": i + 1,
                "group": r["group"],
                "n_channels": r["n_channels"],
                "dPnL": r[f"dPnL_{days}"],
                "pnl": r[f"pnl_{days}"],
                "base_pnl": r[f"base_pnl_{days}"],
            }
            for i, r in enumerate(by_pnl)
        ]
        rankings[f"rank_by_dWR_{days}d"] = [
            {
                "rank": i + 1,
                "group": r["group"],
                "n_channels": r["n_channels"],
                "dWR": r[f"dWR_{days}"],
                "wr": r[f"wr_{days}"],
                "base_wr": r[f"base_wr_{days}"],
            }
            for i, r in enumerate(by_wr)
        ]

    summary = {
        "symbols": symbols,
        "backbone": str(args.backbone),
        "label_mode": str(args.label_mode),
        "pivot_len": int(args.pivot_len),
        "barrier_horizon": int(args.barrier_horizon),
        "groups": groups,
        "group_sizes": {
            g: len(all_blend_group_indices(g)) for g in groups
        },
        "importance": (
            "dPnL/dWR = baseline − ablated; "
            "higher means removing the group hurt more (group more valuable)"
        ),
        **rankings,
        "rows": results,
    }
    json_path = out_dir / f"{out_stem}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print("\n=== Rank by ΔPnL @ 360d (group importance) ===")
    for item in rankings["rank_by_dPnL_360d"]:
        print(
            f"  #{item['rank']:2d}  {item['group']:16}  "
            f"dPnL={item['dPnL']:+8.1f}  "
            f"(ablated={item['pnl']:+8.1f}  base={item['base_pnl']:+8.1f})  "
            f"n={item['n_channels']}"
        )
    print("\n=== Rank by ΔWR @ 360d (group importance) ===")
    for item in rankings["rank_by_dWR_360d"]:
        print(
            f"  #{item['rank']:2d}  {item['group']:16}  "
            f"dWR={item['dWR']:+6.1f}pp  "
            f"(ablated={item['wr']:5.1f}%  base={item['base_wr']:5.1f}%)  "
            f"n={item['n_channels']}"
        )


if __name__ == "__main__":
    main()
