"""Train common+V2 rolle_breakout 15m for L=5,9,12 × 5 assets, then export replay."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENG = Path(__file__).resolve().parent
OUT = ENG / "replay_chart"
SYMBOLS = "XAUUSD,XAGUSD,XTIUSD,BTCUSD,ETHUSD"
LENS = (5, 9, 12)
WINDOWS = "90,180,360"


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)}")


def main() -> None:
    py = sys.executable
    for plen in LENS:
        ckpt = ROOT / f"checkpoints_rolle_breakout_p{plen}_15m_entryv2"
        stem = f"asset_engine_common_rolle_breakout_p{plen}_15m_5assets_entryv2"
        run(
            [
                py,
                "-m",
                "smartbs_engines._train_replay_all_st",
                "--label-mode",
                "rolle_breakout",
                "--rolle-len",
                str(plen),
                "--interval",
                "15m",
                "--backbone",
                "smartBSEntryV2",
                "--engines",
                "common",
                "--symbols",
                SYMBOLS,
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
    symbols = [s.strip() for s in SYMBOLS.split(",") if s.strip()]

    print("\n=== export replay charts ===", flush=True)
    for plen in LENS:
        for sym in symbols:
            ckpt = ROOT / f"checkpoints_rolle_breakout_p{plen}_15m_entryv2" / "common" / f"{sym}.pt"
            if not ckpt.is_file():
                print(f"  SKIP missing {ckpt}", flush=True)
                continue
            print(f"  EXPORT L={plen} {sym}", flush=True)
            try:
                rows = export_checkpoint_windows(
                    sym,
                    ckpt,
                    windows_days=windows,
                    out_dir=OUT,
                    ai_threshold=0.0,
                )
                for row in rows:
                    print(
                        f"    {row['days']:3d}d pnl={row['pnl']:+.1f} trades={row['trades']}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"    FAIL: {exc}", flush=True)

    items = _rebuild_catalog_from_data(OUT)
    cat_path = OUT / "catalog.json"
    cat_path.write_text(
        json.dumps(
            {
                "title": "SmartBS replay chart",
                "ckpt_root": "multi_rolle_len_15m_common_v2",
                "items": items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    focus = [
        i
        for i in items
        if i.get("engine") == "common"
        and i.get("interval") == "15m"
        and i.get("label_mode") == "rolle_breakout"
        and i.get("rolle_len") in LENS
    ]
    print(f"\nCatalog {len(items)} items; focus 15m common L={LENS}: {len(focus)}", flush=True)
    for i in sorted(
        focus, key=lambda x: (x.get("rolle_len") or 0, x.get("symbol") or "", x.get("days") or 0)
    ):
        print(
            f"  L={i.get('rolle_len')} {i.get('symbol')} {i.get('days')}d "
            f"pnl={i.get('pnl')} trades={i.get('trades')}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
