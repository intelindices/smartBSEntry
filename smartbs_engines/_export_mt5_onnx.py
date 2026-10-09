"""Export 5m / 15m / 1h / 4h checkpoints into MQL5/Files/SmartBSEntry/{SYM}_{eng}_{tf}.onnx."""
from __future__ import annotations

import argparse
from pathlib import Path

from smartbs_engines.export_onnx import export_onnx

ENGINES = [
    "maribbon",
    "common",
]
SYMBOLS = ["XAUUSD", "XAGUSD", "XTIUSD", "BTCUSD", "ETHUSD"]
TF_ROOTS = {
    "5m": "checkpoints_rolle_breakout_roll_p15_5m_entryv2",
    "15m": "checkpoints_rolle_breakout_roll_p15_15m_entryv2",
    "4h": "checkpoints_rolle_breakout_roll_p15_4h_entryv2",
    "1h": "checkpoints_rolle_breakout_roll_p15_xau_xag_xti_entryv2",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--tf", nargs="*", default=["5m", "15m", "1h", "4h"])
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    out_dir = args.repo / "mql5" / "Files" / "SmartBSEntry"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for tf in args.tf:
        root = args.repo / TF_ROOTS[tf]
        if not root.is_dir():
            root = args.repo / "smartbs_engines" / TF_ROOTS[tf]
        if not root.is_dir():
            print(f"SKIP missing ckpt root for {tf}: {TF_ROOTS[tf]}")
            continue
        for eng in ENGINES:
            for sym in SYMBOLS:
                pt = root / eng / f"{sym}.pt"
                if not pt.is_file():
                    n_skip += 1
                    continue
                dest = out_dir / f"{sym}_{eng}_{tf}.onnx"
                try:
                    export_onnx(str(pt), str(dest), verify=not args.no_verify)
                    n_ok += 1
                except Exception as exc:
                    n_fail += 1
                    print(f"FAIL {pt}: {exc}")
    print(f"export done ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
