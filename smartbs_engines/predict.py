"""Single-engine and equal-weight blend inference (no RM / arm / REST)."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from smartbs_engines.blend import simple_mean_blend
from smartbs_engines.calibration import softmax_np
from smartbs_engines.checkpoint import load_classifier
from smartbs_engines.config import SmartBSConfig, PAIR_ALIASES
from smartbs_engines.features import build_feature_matrix
from smartbs_engines.registry import ACTIVE_BLEND_ENGINES

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
    """Return ``(n, 3)`` softmax probs aligned to ``df_1h`` + ckpt config."""
    model, cfg = load_classifier(checkpoint_path, device=device)
    lookback = int(cfg.get("lookback", 64))
    temperature = float(cfg.get("temperature", 1.0) or 1.0)
    feats = build_feature_matrix(
        df_1h,
        feature_engine=cfg.get("feature_engine", "maribbon"),
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
    """Argmax FLAT/LONG/SHORT at one bar for a single engine checkpoint."""
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
        "feature_engine": cfg.get("feature_engine", "maribbon"),
        "temperature": float(cfg.get("temperature", 1.0) or 1.0),
    }


def resolve_engine_checkpoint(
    blend_root: str,
    engine: str,
    trade_pair: str,
) -> str:
    pair = PAIR_ALIASES.get(trade_pair.replace("/", "").upper(), trade_pair.replace("/", "").upper())
    path = os.path.join(blend_root, engine, f"{pair}.pt")
    if not os.path.isfile(path):
        alt = os.path.join(blend_root, f"{engine}_{pair}.pt")
        if os.path.isfile(alt):
            return alt
        raise FileNotFoundError(f"Missing checkpoint for {engine}/{pair}: {path}")
    return path


def predict_blend_probs(
    df_1h: pd.DataFrame,
    blend_root: str,
    *,
    trade_pair: str,
    engines: tuple[str, ...] | list[str] | None = None,
    symbol: str | None = None,
    data_source: str | None = None,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Equal-weight mean blend over ``ACTIVE_BLEND_ENGINES`` (or ``engines``).

    Checkpoints expected at ``{blend_root}/{engine}/{PAIR}.pt``.
    """
    names = list(engines) if engines is not None else list(ACTIVE_BLEND_ENGINES)
    probs_by: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"engines": [], "checkpoints": {}}
    for eng in names:
        path = resolve_engine_checkpoint(blend_root, eng, trade_pair)
        probs, cfg = predict_probs(
            df_1h,
            path,
            symbol=symbol or trade_pair,
            data_source=data_source,
            device=device,
        )
        probs_by[eng] = probs
        meta["engines"].append(eng)
        meta["checkpoints"][eng] = path
        meta.setdefault("lookback", int(cfg.get("lookback", 64)))
    blended = simple_mean_blend(probs_by, engines=names)
    meta["blend"] = "simple_mean"
    return blended, meta


def predict_blend_raw_ai(
    df_1h: pd.DataFrame,
    blend_root: str,
    *,
    trade_pair: str,
    engines: tuple[str, ...] | list[str] | None = None,
    symbol: str | None = None,
    data_source: str | None = None,
    bar_index: int = -1,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Argmax of the equal-weight blend at one bar."""
    probs, meta = predict_blend_probs(
        df_1h,
        blend_root,
        trade_pair=trade_pair,
        engines=engines,
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
        "blend_root": blend_root,
        "trade_pair": trade_pair,
        **meta,
    }
