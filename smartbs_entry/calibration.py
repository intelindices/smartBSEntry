"""AI-layer calibration: temperature scaling and label-space meta thresholds.

Entry, exit, size, and stops are Risk Manager settings — they are not fitted or
written here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from smartbs_entry.config import SmartBSConfig


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 0.05)
    z = np.asarray(logits, dtype=np.float64) / t
    if z.ndim == 1:
        z = z - np.max(z)
        e = np.exp(z)
        return (e / np.clip(e.sum(), 1e-12, None)).astype(np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float64)


def entropy_np(probs: np.ndarray) -> np.ndarray | float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    if p.ndim == 1:
        return float(-(p * np.log(p)).sum())
    return -(p * np.log(p)).sum(axis=1)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, max_iter: int = 80) -> float:
    """Minimize NLL of labels under softmax(logits / T). Regularized toward T=1."""
    import torch
    import torch.nn.functional as F

    if logits is None or len(logits) < 8:
        return 1.0
    x = torch.tensor(np.asarray(logits, dtype=np.float32))
    y = torch.tensor(np.asarray(labels, dtype=np.int64))
    log_t = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([log_t], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.05, 5.0)
        nll = F.cross_entropy(x / t, y)
        loss = nll + 0.02 * (t - 1.0).pow(2)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        return 1.0
    t = float(torch.exp(log_t).clamp(0.05, 5.0).item())
    return 1.0 if not np.isfinite(t) else t


def fit_meta_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    min_take_frac: float = 0.25,
    max_take_frac: float = 0.90,
) -> tuple[float, dict[str, float]]:
    """Pick P(LONG) cutoff on val windows (diagnostics only; not an RM gate)."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    cand = (y == SmartBSConfig.CLASS_LONG) | (y == SmartBSConfig.CLASS_FLAT)
    if int(cand.sum()) < 8:
        return 0.42, {"n": float(cand.sum()), "fallback": 1.0}

    p_side = p[cand, SmartBSConfig.CLASS_LONG]
    win = y[cand] == SmartBSConfig.CLASS_LONG
    grid = np.unique(
        np.concatenate(
            [
                np.quantile(p_side, np.linspace(0.05, 0.95, 19)),
                np.array([0.33, 0.36, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60]),
            ]
        )
    )
    n = int(cand.sum())
    best_th = 0.42
    best_ev = -1e18
    best_stats: dict[str, float] = {}
    for th in grid:
        take = p_side >= float(th)
        n_take = int(take.sum())
        frac = n_take / max(n, 1)
        if frac < min_take_frac or frac > max_take_frac:
            continue
        tp = int((take & win).sum())
        fp = int((take & ~win).sum())
        fn = int((~take & win).sum())
        ev = float(tp - fp) - 0.15 * float(fn)
        if ev > best_ev:
            best_ev = ev
            best_th = float(th)
            best_stats = {
                "n": float(n),
                "take_frac": float(frac),
                "precision": float(tp / max(n_take, 1)),
                "ev": ev,
            }
    if not best_stats:
        best_th = float(np.quantile(p_side, 0.20))
        best_stats = {"n": float(n), "fallback_quantile": 0.20}
    return best_th, best_stats


def collect_logits(
    model,
    loader,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(y.numpy() if hasattr(y, "numpy") else np.asarray(y))
    if not all_logits:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)
