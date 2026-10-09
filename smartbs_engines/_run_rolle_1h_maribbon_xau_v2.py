"""Train XAU maribbon+V2 rolle_breakout 1h L=5, then export 90/180/360d replay."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENG = Path(__file__).resolve().parent
OUT = ENG / "replay_chart"
SYMBOL = "XAUUSD"
PLEN = 5
WINDOWS = "90,180,360"


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)}")


def main() -> None:
    py = sys.executable
    ckpt = ROOT / f"checkpoints_rolle_breakout_p{PLEN}_1h_entryv2"
    stem = f"asset_engine_maribbon_rolle_breakout_p{PLEN}_1h_xau_entryv2"
    print(f"\n========== TRAIN L={PLEN} 1h maribbon V2 {SYMBOL} ==========", flush=True)
    run(
        [
            py,
            "-u",
            "-m",
            "smartbs_engines._train_replay_all_st",
            "--label-mode",
            "rolle_breakout",
            "--rolle-len",
            str(PLEN),
            "--interval",
            "1h",
            "--backbone",
            "smartBSEntryV2",
            "--engines",
            "maribbon",
            "--symbols",
            SYMBOL,
            "--ckpt-root",
            str(ckpt),
            "--out-stem",
            stem,
            "--holdout-days",
            "360",
        ]
    )

    from smartbs_engines._export_replay_chart import (
        _rebuild_catalog_from_data,
        export_checkpoint_windows,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    windows = [int(x) for x in WINDOWS.split(",")]
    ckpt_pt = ckpt / "maribbon" / f"{SYMBOL}.pt"
    print("\n=== export replay charts ===", flush=True)
    if not ckpt_pt.is_file():
        raise SystemExit(f"missing {ckpt_pt}")
    print(f"  EXPORT L={PLEN} {SYMBOL} maribbon", flush=True)
    rows = export_checkpoint_windows(
        SYMBOL,
        ckpt_pt,
        windows_days=windows,
        out_dir=OUT,
        ai_threshold=0.0,
    )
    for row in rows:
        print(
            f"    {row['days']:3d}d pnl={row['pnl']:+.1f} trades={row['trades']}",
            flush=True,
        )

    items = _rebuild_catalog_from_data(OUT)
    cat_path = OUT / "catalog.json"
    cat_path.write_text(
        json.dumps(
            {
                "title": "SmartBS replay chart",
                "ckpt_root": f"rolle_breakout_p{PLEN}_1h_maribbon_xau_v2",
                "items": items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    focus = [
        i
        for i in items
        if i.get("engine") == "maribbon"
        and i.get("interval") == "1h"
        and i.get("symbol") == SYMBOL
        and i.get("label_mode") == "rolle_breakout"
        and i.get("rolle_len") == PLEN
    ]
    print(f"\nCatalog {len(items)} items; focus: {len(focus)}", flush=True)
    for i in sorted(focus, key=lambda x: x.get("days") or 0):
        print(
            f"  L={i.get('rolle_len')} {i.get('symbol')} {i.get('days')}d "
            f"pnl={i.get('pnl')} trades={i.get('trades')}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
