"""Raw AI inference — probabilities + argmax (no arm / R / day-DD / REST)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from smartbs_entry.calibration import softmax_np
from smartbs_entry.checkpoint import load_classifier
from smartbs_entry.config import SmartBSConfig
from smartbs_entry.features import build_feature_matrix

CLASS_TO_ORDER = {
    SmartBSConfig.CLASS_FLAT: "FLAT",
    SmartBSConfig.CLASS_LONG: "LONG",
    SmartBSConfig.CLASS_SHORT: "SHORT",
}


def predict_probs(
    df_1h: pd.DataFrame,
    checkpoint_path: str,
    *,
    symbol: str | None = None,
    data_source: str | None = None,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(n, 3)`` softmax probs aligned to ``df_1h`` bars + ckpt config.

    Rows before lookback-1 are left as zeros. Temperature comes from the checkpoint.
    """
    model, cfg = load_classifier(checkpoint_path, device=device)
    lookback = int(cfg.get("lookback", 64))
    temperature = float(cfg.get("temperature", 1.0) or 1.0)
    feats = build_feature_matrix(
        df_1h,
        feature_engine=cfg.get("feature_engine", "entry"),
        symbol=symbol or cfg.get("trade_pair"),
        data_source=data_source or cfg.get("data_source"),
    )
    n = len(feats)
    out = np.zeros((n, 3), dtype=np.float64)
    if n < lookback:
        return out, cfg

    device = next(model.parameters()).device
    windows = []
    idxs = []
    for i in range(lookback - 1, n):
        windows.append(feats[i - lookback + 1 : i + 1].T)
        idxs.append(i)
    x = torch.from_numpy(np.stack(windows, axis=0)).float().to(device)
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy()
    probs = softmax_np(logits, temperature)
    for j, i in enumerate(idxs):
        out[i] = probs[j]
    return out, cfg


def predict_raw_ai(
    df_1h: pd.DataFrame,
    checkpoint_path: str,
    *,
    symbol: str | None = None,
    data_source: str | None = None,
    bar_index: int = -1,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Argmax FLAT/LONG/SHORT at one bar — raw AI (no arm / R / max-DD)."""
    probs, cfg = predict_probs(
        df_1h,
        checkpoint_path,
        symbol=symbol,
        data_source=data_source,
        device=device,
    )
    i = bar_index if bar_index >= 0 else len(probs) + bar_index
    if i < 0 or i >= len(probs):
        raise IndexError(f"bar_index {bar_index} out of range for n={len(probs)}")
    p = probs[i]
    cls = int(np.argmax(p))
    return {
        "signal": CLASS_TO_ORDER[cls],
        "class": cls,
        "probs": {
            "FLAT": float(p[0]),
            "LONG": float(p[1]),
            "SHORT": float(p[2]),
        },
        "confidence": float(p[cls]),
        "bar_index": i,
        "checkpoint": checkpoint_path,
        "feature_engine": cfg.get("feature_engine", "entry"),
        "temperature": float(cfg.get("temperature", 1.0) or 1.0),
    }
