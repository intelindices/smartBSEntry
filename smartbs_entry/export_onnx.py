"""Export a SmartBS Entry ``.pt`` checkpoint to ONNX for MetaTrader 5.

Input:  ``features`` float32 [1, num_inputs, lookback]  (e.g. 1×90×64)
Output: ``logits``   float32 [1, 3]

Also writes a JSON sidecar with temperature / lookback / spec hash for the EA.

    python -m smartbs_entry.export_onnx \\
      --checkpoint path/to/XAUUSD.pt \\
      --out mql5/Models/XAUUSD.onnx
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from smartbs_entry.checkpoint import load_classifier
from smartbs_entry.stdio_compat import configure_stdio

configure_stdio()


def _remove_weight_norm(module: nn.Module) -> None:
    """Bake weight_norm parametrizations so ONNX export stays simple for MT5."""
    for child in list(module.children()):
        _remove_weight_norm(child)
    # torch.nn.utils.parametrizations.weight_norm
    try:
        from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

        for name, _ in list(module.named_parameters(recurse=False)):
            pass
        # Conv1d layers may carry parametrizations on "weight"
        if is_parametrized(module, "weight"):
            remove_parametrizations(module, "weight", leave_parametrized=False)
    except Exception:
        pass
    # Legacy torch.nn.utils.weight_norm
    try:
        from torch.nn.utils import remove_weight_norm

        if hasattr(module, "weight_g") or hasattr(module, "weight_v"):
            remove_weight_norm(module)
    except Exception:
        pass


def export_onnx(
    checkpoint_path: str,
    out_path: str,
    *,
    opset: int = 18,
    verify: bool = True,
) -> dict[str, Any]:
    model, cfg = load_classifier(checkpoint_path, device=torch.device("cpu"))
    model.eval()
    _remove_weight_norm(model)

    num_inputs = int(cfg["num_inputs"])
    lookback = int(cfg.get("lookback", 64))
    temperature = float(cfg.get("temperature", 1.0) or 1.0)

    dummy = torch.randn(1, num_inputs, lookback, dtype=torch.float32)
    with torch.no_grad():
        ref_logits = model(dummy).detach().cpu().numpy()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    meta = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "onnx": str(out.resolve()),
        "trade_pair": cfg.get("trade_pair"),
        "feature_engine": cfg.get("feature_engine", "entry"),
        "feature_spec_hash": cfg.get("feature_spec_hash"),
        "num_inputs": num_inputs,
        "lookback": lookback,
        "num_classes": int(cfg.get("num_classes", 3)),
        "temperature": temperature,
        "num_channels": list(cfg.get("num_channels") or []),
        "opset": int(opset),
    }
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed — skip runtime verify (file still written)")
        else:
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            got = sess.run(None, {"features": dummy.numpy()})[0]
            import numpy as np

            err = float(np.max(np.abs(got - ref_logits)))
            print(f"ONNX vs PyTorch max|Δlogits| = {err:.6e}")
            if err > 1e-4:
                raise RuntimeError(f"ONNX parity failed: max abs err {err}")
            meta["verify_max_abs_err"] = err

    print(f"Wrote {out}")
    print(f"Wrote {sidecar}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SmartBS Entry checkpoint to ONNX for MT5")
    ap.add_argument("--checkpoint", required=True, help="Path to entry/*.pt")
    ap.add_argument("--out", required=True, help="Output .onnx path")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    export_onnx(args.checkpoint, args.out, opset=args.opset, verify=not args.no_verify)


if __name__ == "__main__":
    main()
