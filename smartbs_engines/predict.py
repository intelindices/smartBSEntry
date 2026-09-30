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
    df_15m: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(n, 3)`` softmax probs aligned to ``df_1h`` + ckpt config."""
    from smartbs_engines.model import BACKBONE_SMARTBS_DUAL_TF, normalize_backbone

    model, cfg = load_classifier(checkpoint_path, device=device)
    from smartbs_engines.common_channels import set_session_hours_mode

    # Match common-channel session windows to the checkpoint.
    # Missing key → normal (legacy overlapping hours used before this field existed).
    _sh = cfg.get("session_hours", None)
    set_session_hours_mode("normal" if _sh in (None, "") else _sh)
    lookback = int(cfg.get("lookback", 64))
    temperature = float(cfg.get("temperature", 1.0) or 1.0)
    backbone = normalize_backbone(cfg.get("backbone"))
    sym = symbol or cfg.get("trade_pair")
    src = data_source or cfg.get("data_source")
    feats = build_feature_matrix(
        df_1h,
        feature_engine=cfg.get("feature_engine", "maribbon"),
        symbol=sym,
        data_source=src,
        signal_engines=cfg.get("signal_engines") or None,
        ablation_zero_group=str(cfg.get("ablation_zero_group", "") or ""),
    )
    n = len(feats)
    out = np.zeros((n, 3), dtype=np.float64)
    if n < lookback:
        return out, cfg

    device = next(model.parameters()).device

    if backbone == BACKBONE_SMARTBS_DUAL_TF:
        from smartbs_engines.features import build_1h_to_15m_end_index
        from smartbs_engines.smart_money_structure import load_aligned_15m

        if df_15m is None:
            df_15m = load_aligned_15m(df_1h, symbol=sym, data_source=src)
        if df_15m is None or len(df_15m) < 16:
            raise RuntimeError("predict DualTF: aligned 15m data unavailable")
        feats_15 = build_feature_matrix(
            df_15m,
            feature_engine=cfg.get("feature_engine", "maribbon"),
            symbol=sym,
            data_source=src,
            signal_engines=cfg.get("signal_engines") or None,
            ablation_zero_group=str(cfg.get("ablation_zero_group", "") or ""),
        )
        ratio = int(cfg.get("tf_ratio", 4) or 4)
        lb15 = int(cfg.get("lookback_15m", 0) or 0) or lookback * ratio
        end15 = build_1h_to_15m_end_index(
            df_1h["open_time"].to_numpy(dtype=np.int64),
            df_15m["open_time"].to_numpy(dtype=np.int64),
        )
        w1_list, w15_list, idxs = [], [], []
        for i in range(lookback - 1, n):
            e15 = int(end15[i])
            if e15 < lb15 - 1:
                continue
            w1_list.append(feats[i - lookback + 1 : i + 1].T)
            w15_list.append(feats_15[e15 - lb15 + 1 : e15 + 1].T)
            idxs.append(i)
        if not idxs:
            return out, cfg
        x1 = torch.from_numpy(np.stack(w1_list, axis=0)).float().to(device)
        x15 = torch.from_numpy(np.stack(w15_list, axis=0)).float().to(device)
        with torch.no_grad():
            logits = model(x1, x15).detach().cpu().numpy()
        probs = softmax_np(logits, temperature)
        for j, i in enumerate(idxs):
            out[i] = probs[j]
        return out, cfg

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
    signal_point_gate: bool | None = None,
    raw_ai_strategy: str | None = None,
) -> dict[str, Any]:
    """Argmax FLAT/LONG/SHORT at one bar for a single engine checkpoint.

    Combines AI probs with the checkpoint ``signal_policy`` via ``raw_ai_strategy``
    (``ai_only`` | ``signal_gate`` | ``signal_prior``). Legacy
    ``signal_point_gate=True`` maps to ``signal_gate``.
    """
    from smartbs_engines.pipeline import PipelineSpec
    from smartbs_engines.signal_policy import (
        RAW_AI_AI_ONLY,
        apply_raw_ai_strategy,
        resolve_raw_ai_strategy,
    )

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
    reason = ""

    cfg_eff = dict(cfg)
    if signal_point_gate is True:
        cfg_eff["signal_point_gate"] = True
    elif signal_point_gate is False:
        cfg_eff["signal_point_gate"] = False
        cfg_eff["raw_ai_strategy"] = cfg_eff.get("raw_ai_strategy") or RAW_AI_AI_ONLY
    if raw_ai_strategy is not None:
        cfg_eff["raw_ai_strategy"] = raw_ai_strategy

    strategy = resolve_raw_ai_strategy(cfg_eff)
    pipe = PipelineSpec.from_config(cfg_eff)
    if strategy != RAW_AI_AI_ONLY:
        policy = pipe.make_signal_policy()
        # Include previous bar so onset (cross thr from below) is defined.
        i0 = max(0, i - 1)
        sl = slice(i0, i + 1)
        sub = df_1h.iloc[sl].reset_index(drop=True)
        feats = build_feature_matrix(
            sub,
            feature_engine=cfg.get("feature_engine", "maribbon"),
            symbol=symbol or cfg.get("trade_pair"),
            data_source=data_source or cfg.get("data_source"),
            signal_engines=cfg.get("signal_engines") or None,
        )
        # Prefer full signals bus for onset policy when model features are raw engine.
        decision = policy.decide(
            sub,
            features=feats,
            feature_names=None,
            symbol=symbol or cfg.get("trade_pair"),
            data_source=data_source or cfg.get("data_source"),
        )
        cls_win = np.zeros(i - i0 + 1, dtype=np.int64)
        cls_win[-1] = cls
        gated, reason_idx = apply_raw_ai_strategy(
            cls_win,
            decision,
            strategy,
            require_agree=pipe.signal_require_agree,
            features=feats if feats.shape[1] == len(decision.reason_ids) else None,
            feature_names=decision.reason_ids or None,
            thr=pipe.signal_thr,
        )
        cls = int(gated[-1])
        ri = int(reason_idx[-1])
        ids = list(decision.reason_ids)
        if 0 <= ri < len(ids):
            reason = ids[ri]
        p = probs[i]
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
        "signal_policy": pipe.signal_policy,
        "raw_ai_strategy": strategy,
        "signal_point_gate": strategy == "signal_gate",
        "signal_reason": reason,
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
        meta.setdefault("label_mode", str(cfg.get("label_mode", "") or ""))
        meta.setdefault("session_hours", str(cfg.get("session_hours", "") or ""))
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
