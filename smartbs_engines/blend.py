"""Blend per-engine probabilities into one (FLAT, LONG, SHORT).

**Live / replay default (simple):**
  ``p_blend[side] = sum_e p_e[side] / N`` over ``ACTIVE_BLEND_ENGINES``, then
  side = argmax(p_blend). No vote, no disagreement veto, no fitted temperature.

Research helpers below (vote blend, regime-side blend, disagreement veto,
blend temperature) stay for offline eval only — they are not the live path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartbs_engines.config import SmartBSConfig

DEFAULT_OPPOSITE_VETO = 0.60
DEFAULT_MIN_WEIGHT_FOR_VETO = 0.05
# Starting point only; per-asset thresholds are fitted from trade rate.
DEFAULT_AI_THRESHOLD = 0.58
BARS_PER_MONTH_1H = 24 * 30


@dataclass(frozen=True)
class BlendDecision:
    side: str | None  # "LONG" | "SHORT" | None
    confidence: float
    probs: np.ndarray  # blended, calibrated (FLAT, LONG, SHORT)
    vetoed: bool = False
    veto_reason: str = ""


def simple_mean_blend(
    probs_by_engine: dict[str, np.ndarray],
    *,
    engines: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Equal-weight mean: ``p[side] = sum_e p_e[side] / N``, renormalized.

    ``engines`` defaults to every key present in ``probs_by_engine``. Works for
    a single bar ``(3,)`` or a series ``(n, 3)``.
    """
    names = list(engines) if engines is not None else list(probs_by_engine.keys())
    names = [n for n in names if n in probs_by_engine]
    if not names:
        raise ValueError("simple_mean_blend: no engines with probabilities")
    mats = [np.asarray(probs_by_engine[n], dtype=np.float64) for n in names]
    stacked = np.stack(mats, axis=0)  # (N, 3) or (N, n, 3)
    out = stacked.mean(axis=0)
    return out / np.clip(out.sum(axis=-1, keepdims=True), 1e-12, None)


def _as_matrix(
    probs_by_engine: dict[str, np.ndarray], weights: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = [n for n in probs_by_engine if weights.get(n, 0.0) > 0.0]
    if not names:
        raise ValueError("no engine has positive weight")
    mat = np.stack([np.asarray(probs_by_engine[n], dtype=np.float64) for n in names], axis=0)
    w = np.array([float(weights[n]) for n in names], dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    return mat, w, names


def blend_probs(
    probs_by_engine: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Weighted average of per-engine probabilities (live uses equal weights).

    Prefer ``simple_mean_blend`` for the explicit sum/N path. Works for a single
    bar (each value shape (3,)) or a series (shape (n, 3)).
    """
    # Equal weights → identical to simple_mean_blend over those engines.
    names = [n for n in probs_by_engine if weights.get(n, 0.0) > 0.0]
    if names and abs(max(weights[n] for n in names) - min(weights[n] for n in names)) < 1e-12:
        return simple_mean_blend(probs_by_engine, engines=names)
    mat, w, _ = _as_matrix(probs_by_engine, weights)
    if mat.ndim == 2:  # (engines, 3)
        out = (mat * w[:, None]).sum(axis=0)
    else:  # (engines, n, 3)
        out = (mat * w[:, None, None]).sum(axis=0)
    return out / np.clip(out.sum(axis=-1, keepdims=True), 1e-12, None)


def vote_blend_probs(
    probs_by_engine: dict[str, np.ndarray],
    *,
    engines: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Majority vote on per-engine argmax; confidence = max p(side) among engines.

    Each engine votes +1 (LONG argmax), -1 (SHORT), or 0 (FLAT). Sum of votes:
      >0 → LONG,  <0 → SHORT,  ==0 → FLAT.

    Encoding (unused side is exactly 0 so debug/RM-off can read the vote):
      LONG  → ``[1 - c, c, 0]`` where ``c = max_e p_e(LONG)``
      SHORT → ``[1 - c, 0, c]`` where ``c = max_e p_e(SHORT)``
      FLAT  → ``[1, 0, 0]``

    Works for a single bar (each value shape (3,)) or a series (shape (n, 3)).
    """
    flat_c = SmartBSConfig.CLASS_FLAT
    long_c = SmartBSConfig.CLASS_LONG
    short_c = SmartBSConfig.CLASS_SHORT

    names = [n for n in (engines or list(probs_by_engine)) if n in probs_by_engine]
    if not names:
        raise ValueError("no engines to vote-blend")
    mats = [np.asarray(probs_by_engine[n], dtype=np.float64) for n in names]
    stack = np.stack(mats, axis=0)  # (E, 3) or (E, n, 3)
    single = stack.ndim == 2
    if single:
        stack = stack[:, None, :]  # (E, 1, 3)

    arg = np.argmax(stack, axis=-1)  # (E, n)
    votes = np.zeros(arg.shape, dtype=np.float64)
    votes[arg == long_c] = 1.0
    votes[arg == short_c] = -1.0
    vote_sum = votes.sum(axis=0)  # (n,)

    conf_long = stack[:, :, long_c].max(axis=0)
    conf_short = stack[:, :, short_c].max(axis=0)
    n = vote_sum.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)

    long_m = vote_sum > 0
    short_m = vote_sum < 0
    flat_m = ~long_m & ~short_m

    out[flat_m, flat_c] = 1.0
    c_l = np.clip(conf_long[long_m], 0.0, 1.0)
    out[long_m, long_c] = c_l
    out[long_m, flat_c] = 1.0 - c_l
    c_s = np.clip(conf_short[short_m], 0.0, 1.0)
    out[short_m, short_c] = c_s
    out[short_m, flat_c] = 1.0 - c_s

    out = out / np.clip(out.sum(axis=-1, keepdims=True), 1e-12, None)
    return out[0] if single else out


def regime_side_blend_probs(
    probs_by_engine: dict[str, np.ndarray],
    regime_side: np.ndarray,
    *,
    engines: list[str] | tuple[str, ...] | None = None,
    min_agree: int = 2,
) -> np.ndarray:
    """Regime-gated entry signal from per-engine argmax.

    Opposite-regime votes are ignored. Need at least ``min_agree`` engines
    whose argmax matches the regime side; confidence = max p(side) among
    those agreeing engines. Regime sideways/off, or too few agrees → FLAT.

    ``regime_side`` is +1 bull / -1 bear / 0 sideways, length ``n`` matching
    the per-engine series. Encoding matches ``vote_blend_probs`` (unused
    side = 0).
    """
    flat_c = SmartBSConfig.CLASS_FLAT
    long_c = SmartBSConfig.CLASS_LONG
    short_c = SmartBSConfig.CLASS_SHORT
    need = max(1, int(min_agree))

    names = [n for n in (engines or list(probs_by_engine)) if n in probs_by_engine]
    if not names:
        raise ValueError("no engines to regime-blend")
    mats = [np.asarray(probs_by_engine[n], dtype=np.float64) for n in names]
    stack = np.stack(mats, axis=0)  # (E, n, 3) or (E, 3)
    single = stack.ndim == 2
    if single:
        stack = stack[:, None, :]
    n = stack.shape[1]
    side = np.asarray(regime_side, dtype=np.int64).reshape(-1)
    if side.shape[0] != n:
        raise ValueError(f"regime_side length {side.shape[0]} != probs length {n}")

    arg = np.argmax(stack, axis=-1)  # (E, n)
    out = np.zeros((n, 3), dtype=np.float64)
    out[:, flat_c] = 1.0

    for i in range(n):
        rs = int(side[i])
        if rs > 0:
            agree = arg[:, i] == long_c
            if int(agree.sum()) < need:
                continue
            c = float(np.clip(stack[agree, i, long_c].max(), 0.0, 1.0))
            out[i, long_c] = c
            out[i, flat_c] = 1.0 - c
            out[i, short_c] = 0.0
        elif rs < 0:
            agree = arg[:, i] == short_c
            if int(agree.sum()) < need:
                continue
            c = float(np.clip(stack[agree, i, short_c].max(), 0.0, 1.0))
            out[i, short_c] = c
            out[i, flat_c] = 1.0 - c
            out[i, long_c] = 0.0

    out = out / np.clip(out.sum(axis=-1, keepdims=True), 1e-12, None)
    return out[0] if single else out


def regime_filter_engine(probs: np.ndarray, regime_side: np.ndarray) -> np.ndarray:
    """Single-engine: keep argmax only when it matches regime; else FLAT."""
    flat_c = SmartBSConfig.CLASS_FLAT
    long_c = SmartBSConfig.CLASS_LONG
    short_c = SmartBSConfig.CLASS_SHORT
    p = np.asarray(probs, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p[None, :]
    side = np.asarray(regime_side, dtype=np.int64).reshape(-1)
    n = p.shape[0]
    if side.shape[0] != n:
        raise ValueError(f"regime_side length {side.shape[0]} != probs length {n}")
    out = np.zeros((n, 3), dtype=np.float64)
    arg = np.argmax(p, axis=-1)
    for i in range(n):
        rs = int(side[i])
        k = int(arg[i])
        want = long_c if rs > 0 else short_c if rs < 0 else flat_c
        if rs == 0 or k != want:
            out[i, flat_c] = 1.0
            continue
        c = float(np.clip(p[i, k], 0.0, 1.0))
        out[i, k] = c
        out[i, flat_c] = 1.0 - c
    out = out / np.clip(out.sum(axis=-1, keepdims=True), 1e-12, None)
    return out[0] if single else out


def directional_encode(probs: np.ndarray) -> np.ndarray:
    """Encode one engine's probs as vote-style directional mass for debug PM.

    Argmax side keeps its raw probability ``c``; unused directional class is 0;
    FLAT gets ``1 - c``. Shape ``(3,)`` or ``(n, 3)``.
    """
    flat_c = SmartBSConfig.CLASS_FLAT
    long_c = SmartBSConfig.CLASS_LONG
    short_c = SmartBSConfig.CLASS_SHORT
    p = np.asarray(probs, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p[None, :]
    n = p.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    arg = np.argmax(p, axis=-1)
    for i in range(n):
        k = int(arg[i])
        if k == flat_c:
            out[i, flat_c] = 1.0
            continue
        c = float(np.clip(p[i, k], 0.0, 1.0))
        out[i, k] = c
        out[i, flat_c] = 1.0 - c
    out = out / np.clip(out.sum(axis=-1, keepdims=True), 1e-12, None)
    return out[0] if single else out


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-sharpen or flatten a probability vector: p^(1/T), renormalized."""
    t = max(float(temperature), 0.05)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    z = np.log(p) / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def fit_blend_temperature(
    blended: np.ndarray, labels: np.ndarray, *, max_iter: int = 80
) -> float:
    """Fit T on the weight-fit split so the blend's confidence means something.

    Minimizes negative log-likelihood of the observed labels under p^(1/T).
    """
    from smartbs_engines.calibration import fit_temperature

    p = np.clip(np.asarray(blended, dtype=np.float64), 1e-12, 1.0)
    # fit_temperature works on logits; log-probs are logits up to a constant,
    # which softmax is invariant to.
    return float(fit_temperature(np.log(p), np.asarray(labels), max_iter=max_iter))


def threshold_for_trade_rate(
    p_side: np.ndarray,
    *,
    trades_per_month: float,
    bars_per_month: int = BARS_PER_MONTH_1H,
    floor: float = 0.34,
    ceil: float = 0.95,
) -> float:
    """Pick the gate that yields the requested trade frequency.

    A fixed constant does not transfer from one engine to a six-way blend, since
    averaging pulls every probability toward 1/3. Choosing by quantile keeps the
    trade rate stable no matter how many engines are in the blend.
    """
    p = np.asarray(p_side, dtype=np.float64).reshape(-1)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float(DEFAULT_AI_THRESHOLD)
    months = max(p.size / float(bars_per_month), 1e-9)
    target_hits = max(float(trades_per_month) * months, 1.0)
    frac = min(max(target_hits / p.size, 1e-6), 1.0)
    thr = float(np.quantile(p, 1.0 - frac))
    return float(min(max(thr, floor), ceil))


def disagreement_veto(
    probs_by_engine: dict[str, np.ndarray],
    weights: dict[str, float],
    side: str,
    *,
    opposite_threshold: float = DEFAULT_OPPOSITE_VETO,
    min_weight: float = DEFAULT_MIN_WEIGHT_FOR_VETO,
    require_top2_agree: bool = True,
) -> tuple[bool, str]:
    """Block a trade the engines do not actually agree on.

    Returns ``(vetoed, reason)``.
    """
    opp_cls = SmartBSConfig.CLASS_SHORT if side == "LONG" else SmartBSConfig.CLASS_LONG
    side_cls = SmartBSConfig.CLASS_LONG if side == "LONG" else SmartBSConfig.CLASS_SHORT

    for name, probs in probs_by_engine.items():
        w = float(weights.get(name, 0.0))
        if w < min_weight:
            continue
        if float(np.asarray(probs).reshape(-1)[opp_cls]) > opposite_threshold:
            return True, f"{name} opposes at {float(np.asarray(probs).reshape(-1)[opp_cls]):.2f}"

    if require_top2_agree:
        ranked = sorted(
            (n for n in probs_by_engine if weights.get(n, 0.0) > 0.0),
            key=lambda n: -float(weights.get(n, 0.0)),
        )[:2]
        for name in ranked:
            p = np.asarray(probs_by_engine[name]).reshape(-1)
            if int(np.argmax(p)) != side_cls:
                return True, f"top-weighted {name} does not agree on {side}"
    return False, ""


def decide_blend(
    probs_by_engine: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    temperature: float = 1.0,
    threshold: float = DEFAULT_AI_THRESHOLD,
    opposite_threshold: float = DEFAULT_OPPOSITE_VETO,
    require_top2_agree: bool = True,
) -> BlendDecision:
    """Blend, calibrate, gate, and veto — for one bar."""
    blended = apply_temperature(blend_probs(probs_by_engine, weights), temperature)
    p_long = float(blended[SmartBSConfig.CLASS_LONG])
    p_short = float(blended[SmartBSConfig.CLASS_SHORT])

    if max(p_long, p_short) < float(threshold):
        return BlendDecision(side=None, confidence=max(p_long, p_short), probs=blended)

    side = "LONG" if p_long >= p_short else "SHORT"
    vetoed, reason = disagreement_veto(
        probs_by_engine,
        weights,
        side,
        opposite_threshold=opposite_threshold,
        require_top2_agree=require_top2_agree,
    )
    if vetoed:
        return BlendDecision(
            side=None,
            confidence=max(p_long, p_short),
            probs=blended,
            vetoed=True,
            veto_reason=reason,
        )
    return BlendDecision(side=side, confidence=max(p_long, p_short), probs=blended)
