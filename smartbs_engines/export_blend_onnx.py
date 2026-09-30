"""Export ONNX (+ JSON sidecar) for ST engines for one or more symbols.

Default engines = blend5 (ACTIVE_BLEND_ENGINES).
Default checkpoints = checkpoints_fwd (1h forward_return).

Usage:
  python -m smartbs_engines.export_blend_onnx --symbols BTCUSD
  python -m smartbs_engines.export_blend_onnx --symbols XAUUSD --engines blend7
  python -m smartbs_engines.export_blend_onnx --ckpt-root smartbs_engines/checkpoints --symbols BTCUSD
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smartbs_engines.export_onnx import export_onnx
from smartbs_engines.registry import ACTIVE_BLEND_ENGINES, BLEND_ENGINES
from smartbs_engines.stdio_compat import configure_stdio

configure_stdio()

POOLS = {
    "blend5": tuple(ACTIVE_BLEND_ENGINES),
    "blend7": tuple(BLEND_ENGINES),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default="XAUUSD")
    ap.add_argument(
        "--engines",
        default="blend5",
        help="blend5 (default) | blend7 | comma list of engine names",
    )
    ap.add_argument(
        "--ckpt-root",
        default="",
        help="Default: smartbs_engines/checkpoints_fwd (1h forward_return)",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Default: mql5/Models",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else root / "checkpoints_fwd"
    out_dir = Path(args.out_dir) if args.out_dir else root.parent / "mql5" / "Models"
    out_dir.mkdir(parents=True, exist_ok=True)

    eng_arg = args.engines.strip().lower()
    if eng_arg in POOLS:
        engines = list(POOLS[eng_arg])
    else:
        engines = [e.strip().lower() for e in args.engines.split(",") if e.strip()]

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"Symbols: {symbols}")
    print(f"Engines: {engines}")
    print(f"Checkpoints: {ckpt_root}")
    print(f"Out: {out_dir}")

    for sym in symbols:
        for eng in engines:
            ckpt = ckpt_root / eng / f"{sym}.pt"
            if not ckpt.is_file():
                print(f"  SKIP missing {ckpt}")
                continue
            out = out_dir / f"{sym}_{eng}.onnx"
            try:
                meta = export_onnx(str(ckpt), str(out))
                print(
                    f"  OK {sym}/{eng} → {out.name} "
                    f"in={meta.get('num_inputs')} lb={meta.get('lookback')}"
                )
            except Exception as exc:
                print(f"  FAIL {sym}/{eng}: {exc}")

    print("Copy *.onnx + *.json to terminal MQL5/Files/SmartBSEntry/")
    print("========== EXPORT DONE ==========")


if __name__ == "__main__":
    main()
