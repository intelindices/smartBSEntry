"""Dump a Python feature window + logits for MQL5 parity checks.

Prints JSON to stdout and optionally writes a file the EA Journal can be
compared against (feature L2 / signal match).

    python -m smartbs_entry.mql5_parity \\
      --checkpoint path/to/XAUUSD.pt \\
      --symbol XAUUSD --data-source dukascopy \\
      --out mql5/Models/XAUUSD_parity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from smartbs_entry.calibration import softmax_np
from smartbs_entry.checkpoint import load_classifier
from smartbs_entry.data import fetch_mtf_klines
from smartbs_entry.engine import FEATURE_NAMES, SmartBSEntryEngine
from smartbs_entry.features import build_feature_matrix
from smartbs_entry.stdio_compat import configure_stdio

configure_stdio()


def dump_parity(
    checkpoint: str,
    *,
    symbol: str = "XAUUSD",
    data_source: str = "dukascopy",
    train_candles: int = 8000,
) -> dict[str, Any]:
    model, cfg = load_classifier(checkpoint, device=torch.device("cpu"))
    lookback = int(cfg.get("lookback", 64))
    temperature = float(cfg.get("temperature", 1.0) or 1.0)
    df = fetch_mtf_klines(symbol, source=data_source, max_1h_candles=train_candles)
    feats = build_feature_matrix(df, feature_engine="entry", symbol=symbol, data_source=data_source)
    n = len(feats)
    end = n - 1
    window = feats[end - lookback + 1 : end + 1].T.astype(np.float32)  # (90, lookback)
    x = torch.from_numpy(window[None, ...])
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy()[0]
    probs = softmax_np(logits[None, :], temperature)[0]
    cls = int(np.argmax(probs))
    names = ["FLAT", "LONG", "SHORT"]
    eng = SmartBSEntryEngine()
    return {
        "symbol": symbol,
        "data_source": data_source,
        "checkpoint": str(Path(checkpoint).resolve()),
        "n_bars": n,
        "end_bar": end,
        "lookback": lookback,
        "temperature": temperature,
        "feature_spec_hash": eng.spec_hash(),
        "ckpt_spec_hash": cfg.get("feature_spec_hash"),
        "num_inputs": int(cfg["num_inputs"]),
        "feature_names": list(FEATURE_NAMES),
        "window_chw": window.reshape(-1).tolist(),  # channel-major flat
        "window_last_bar_features": feats[end].tolist(),
        "logits": logits.tolist(),
        "probs": {"FLAT": float(probs[0]), "LONG": float(probs[1]), "SHORT": float(probs[2])},
        "signal": names[cls],
        "class": cls,
        "confidence": float(probs[cls]),
        "last_open_time_ms": int(df["open_time"].iloc[end]),
        "last_close": float(df["close"].iloc[end]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump SmartBS Entry parity vectors for MT5")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--data-source", default="dukascopy")
    ap.add_argument("--train-candles", type=int, default=8000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    payload = dump_parity(
        args.checkpoint,
        symbol=args.symbol,
        data_source=args.data_source,
        train_candles=args.train_candles,
    )
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        # Drop huge window from default file? keep it for full parity.
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    print(
        f"signal={payload['signal']} conf={payload['confidence']:.4f} "
        f"T={payload['temperature']:.4f} bars={payload['n_bars']} "
        f"hash={payload['feature_spec_hash']}"
    )


if __name__ == "__main__":
    main()
