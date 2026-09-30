"""Pluggable signal decisions + raw_ai strategies.

Pipeline shape::

    engines → features → SignalPolicy.decide() → SignalDecision
                       → Backbone → probs → raw_ai strategy → FLAT/LONG/SHORT

``SignalPolicy`` implementations are registered by name. Ship defaults:

* ``none`` — no discrete signal (all FLAT); AI sees engine features alone
* ``onset_side`` — engine-side ``*_long/*_short`` onset → LONG/SHORT
* ``ma_decay`` (default) — regime FLAT/LONG/SHORT + other-engine MA decay

Custom policies: implement ``SignalPolicy`` and ``register_signal_policy(name, factory)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from smartbs_engines.config import SmartBSConfig

CLASS_FLAT = SmartBSConfig.CLASS_FLAT
CLASS_LONG = SmartBSConfig.CLASS_LONG
CLASS_SHORT = SmartBSConfig.CLASS_SHORT

SIGNAL_POLICY_NONE = "none"
SIGNAL_POLICY_ONSET_SIDE = "onset_side"
SIGNAL_POLICY_MA_DECAY = "ma_decay"

RAW_AI_AI_ONLY = "ai_only"
RAW_AI_SIGNAL_GATE = "signal_gate"
RAW_AI_SIGNAL_PRIOR = "signal_prior"

DEFAULT_SIGNAL_POLICY = SIGNAL_POLICY_NONE
DEFAULT_RAW_AI_STRATEGY = RAW_AI_AI_ONLY

# ma_decay: pure time fade 1.0 → 0 in this many bars (0.1/step default).
MA_DECAY_BARS = 10
MA_DECAY_STEP = 0.1
MA_DECAY_FAST_LEN = 14  # same as regime_engine EMA fast


@dataclass
class SignalDecision:
    """Per-bar discrete signal-layer output (aligned to the feature bar index)."""

    cls: np.ndarray  # (n,) int64 — FLAT/LONG/SHORT
    strength: np.ndarray  # (n,) float64 ∈ [0, 1]
    reason_idx: np.ndarray  # (n,) int64 — index into reason_ids (-1 none)
    onset: np.ndarray  # (n,) bool — True on the decision bar (e.g. onset)
    reason_ids: tuple[str, ...] = ()

    @staticmethod
    def all_flat(n: int, *, reason_ids: tuple[str, ...] = ()) -> "SignalDecision":
        n = int(n)
        return SignalDecision(
            cls=np.zeros(n, dtype=np.int64),
            strength=np.zeros(n, dtype=np.float64),
            reason_idx=np.full(n, -1, dtype=np.int64),
            onset=np.zeros(n, dtype=bool),
            reason_ids=tuple(reason_ids),
        )


@runtime_checkable
class SignalPolicy(Protocol):
    name: str

    def decide(
        self,
        df: pd.DataFrame,
        *,
        features: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        symbol: str | None = None,
        data_source: str | None = None,
    ) -> SignalDecision: ...


class NoneSignalPolicy:
    """No discrete signal — placeholder until a custom policy is plugged in."""

    name = SIGNAL_POLICY_NONE

    def decide(
        self,
        df: pd.DataFrame,
        *,
        features: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        symbol: str | None = None,
        data_source: str | None = None,
    ) -> SignalDecision:
        del features, feature_names, symbol, data_source
        return SignalDecision.all_flat(len(df))


class OnsetSideSignalPolicy:
    """LONG/SHORT on engine-side aggregate onsets (legacy signal-point gate source)."""

    name = SIGNAL_POLICY_ONSET_SIDE

    def __init__(
        self,
        *,
        sources: Sequence[str] | None = None,
        thr: float = 0.35,
        reason_ids: Sequence[str] | None = None,
    ):
        self.sources = tuple(sources) if sources else ()
        self.thr = float(thr)
        self.reason_ids = tuple(reason_ids) if reason_ids else None

    def decide(
        self,
        df: pd.DataFrame,
        *,
        features: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        symbol: str | None = None,
        data_source: str | None = None,
    ) -> SignalDecision:
        from smartbs_engines.registry import get_engine
        from smartbs_engines.signals import (
            ENGINE_SIDE_REASON_IDS,
            signal_ids_for_sources,
            signal_point_onset,
        )

        names = [str(x) for x in (feature_names or ())]
        mat = None if features is None else np.asarray(features, dtype=np.float64)
        use_bus = (
            mat is not None
            and len(names) == (mat.shape[1] if mat.ndim == 2 else -1)
            and any(n.endswith("_long") or n.endswith("_short") for n in names)
        )
        if not use_bus:
            eng = get_engine(
                "signals",
                signal_sources=self.sources or None,
            )
            # signals engine may need symbol on df context via compute only
            del symbol, data_source
            res = eng.compute(df)
            mat = np.asarray(res.features, dtype=np.float64)
            names = list(eng.feature_names)

        ids = tuple(names)
        want = self.reason_ids or signal_ids_for_sources(self.sources or None)
        # Prefer engine-side aggregates when available.
        side_set = set(ENGINE_SIDE_REASON_IDS)
        if any(i in side_set for i in ids):
            want = tuple(i for i in ids if i in side_set) or want

        fire_long, fire_short, primary = signal_point_onset(
            mat, ids, thr=self.thr, reason_ids=want
        )
        n = len(df)
        cls = np.zeros(n, dtype=np.int64)
        strength = np.zeros(n, dtype=np.float64)
        only_long = fire_long & ~fire_short
        only_short = fire_short & ~fire_long
        both = fire_long & fire_short
        cls[only_long] = CLASS_LONG
        cls[only_short] = CLASS_SHORT
        # Both sides onset → leave FLAT; raw_ai gate/prior can still use AI.
        onset = fire_long | fire_short
        for i in np.flatnonzero(onset):
            ri = int(primary[i])
            if 0 <= ri < mat.shape[1]:
                strength[i] = float(np.clip(mat[i, ri], 0.0, 1.0))
        # Ambiguous both-fire: strength = max of sides, cls stays FLAT
        for i in np.flatnonzero(both):
            ri = int(primary[i])
            if 0 <= ri < mat.shape[1]:
                strength[i] = float(np.clip(mat[i, ri], 0.0, 1.0))
        return SignalDecision(
            cls=cls,
            strength=strength,
            reason_idx=primary.astype(np.int64, copy=False),
            onset=onset,
            reason_ids=ids,
        )


class MaDecaySignalPolicy:
    """Regime class + MA-decay strengths for other engines.

    * **regime_engine** — every bar: bull→LONG, bear→SHORT, side→FLAT
      (strength 1 when trending, 0 when sideways).
    * **other engines** — on ``*_long`` / ``*_short`` onset, strength starts at 1.0
      and decays by ``decay_step`` each bar (default 0.1 → dead in 10 bars), with:

      - **re-strengthen to 1** while close agrees with fastMA (LONG: close>EMA14;
        SHORT: close<EMA14)
      - **fade** (one decay step) when the candle **body** crosses fastMA
      - **kill to 0** when close disagrees with fastMA
      - **kill to 0** when price breaks the signal candle extreme
        (LONG: low < origin_low; SHORT: high > origin_high)

    Final ``cls`` prefers regime when not sideways; otherwise the active decay
    side (or FLAT).
    """

    name = SIGNAL_POLICY_MA_DECAY

    def __init__(
        self,
        *,
        sources: Sequence[str] | None = None,
        thr: float = 0.35,
        decay_bars: int = MA_DECAY_BARS,
        decay_step: float | None = None,
        fast_ma_len: int = MA_DECAY_FAST_LEN,
    ):
        self.sources = tuple(sources) if sources else ()
        self.thr = float(thr)
        self.decay_bars = max(1, int(decay_bars))
        self.decay_step = (
            float(decay_step)
            if decay_step is not None
            else 1.0 / float(self.decay_bars)
        )
        self.fast_ma_len = max(1, int(fast_ma_len))

    def decide(
        self,
        df: pd.DataFrame,
        *,
        features: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        symbol: str | None = None,
        data_source: str | None = None,
    ) -> SignalDecision:
        from smartbs_engines.registry import get_engine
        from smartbs_engines.structure import ema as _ema
        from smartbs_engines.signals import (
            ENGINE_SIDE_REASON_IDS,
            signal_ids_for_sources,
            signal_point_onset,
        )

        del symbol, data_source
        n = len(df)
        if n == 0:
            return SignalDecision.all_flat(0)

        open_ = df["open"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        fast = _ema(close, self.fast_ma_len)

        # --- regime path (Pine-identical) ---
        reg_eng = get_engine("regime_engine")
        reg = reg_eng.compute(df)
        reg_names = list(reg_eng.feature_names)
        reg_feats = np.asarray(reg.features, dtype=np.float64)
        iu = reg_names.index("regime_up")
        idn = reg_names.index("regime_dn")
        regime_up = reg_feats[:, iu] >= 0.5
        regime_dn = reg_feats[:, idn] >= 0.5

        # --- other engines: onset on side aggregates (exclude regime_*) ---
        names = [str(x) for x in (feature_names or ())]
        mat = None if features is None else np.asarray(features, dtype=np.float64)
        use_bus = (
            mat is not None
            and mat.ndim == 2
            and len(names) == mat.shape[1]
            and any(n.endswith("_long") or n.endswith("_short") for n in names)
        )
        if not use_bus:
            if self.sources:
                src = tuple(
                    s
                    for s in self.sources
                    if s not in ("regime_engine", "regime", "regimeengine")
                )
            else:
                from smartbs_engines.signals import ACTIVE_SIGNAL_SOURCES

                src = tuple(
                    s for s in ACTIVE_SIGNAL_SOURCES if s != "regime_engine"
                )
            eng = get_engine("signals", signal_sources=src or None)
            res = eng.compute(df)
            mat = np.asarray(res.features, dtype=np.float64)
            names = list(eng.feature_names)

        ids = tuple(names)
        side_set = {
            x
            for x in ENGINE_SIDE_REASON_IDS
            if not x.startswith("regime_")
        }
        want = tuple(i for i in ids if i in side_set)
        fire_long, fire_short, primary = signal_point_onset(
            mat, ids, thr=self.thr, reason_ids=want or None
        )

        cls = np.zeros(n, dtype=np.int64)
        strength = np.zeros(n, dtype=np.float64)
        reason_idx = np.full(n, -1, dtype=np.int64)
        onset_out = np.zeros(n, dtype=bool)

        # Active decay state (at most one side at a time).
        active_side = 0  # +1 long, -1 short, 0 none
        active_str = 0.0
        origin_hi = np.nan
        origin_lo = np.nan
        active_reason = -1

        step = float(self.decay_step)
        for i in range(n):
            # New onsets (restart / flip).
            fl, fs = bool(fire_long[i]), bool(fire_short[i])
            if fl and not fs:
                active_side = 1
                active_str = 1.0
                origin_hi = float(high[i])
                origin_lo = float(low[i])
                active_reason = int(primary[i])
                onset_out[i] = True
            elif fs and not fl:
                active_side = -1
                active_str = 1.0
                origin_hi = float(high[i])
                origin_lo = float(low[i])
                active_reason = int(primary[i])
                onset_out[i] = True
            elif fl and fs:
                # Ambiguous dual onset — cancel active decay.
                active_side = 0
                active_str = 0.0
                active_reason = -1

            if active_side != 0:
                o = float(open_[i])
                c = float(close[i])
                h = float(high[i])
                l = float(low[i])
                ma = float(fast[i])
                # 1) Origin candle break → invalid
                if active_side > 0 and l < origin_lo:
                    active_side, active_str, active_reason = 0, 0.0, -1
                elif active_side < 0 and h > origin_hi:
                    active_side, active_str, active_reason = 0, 0.0, -1
                else:
                    agree = (c > ma) if active_side > 0 else (c < ma)
                    disagree = (c < ma) if active_side > 0 else (c > ma)
                    body_cross = (o - ma) * (c - ma) < 0.0
                    # 2) Close disagrees with fastMA → kill
                    if disagree:
                        active_side, active_str, active_reason = 0, 0.0, -1
                    # 3) Close agrees → re-strengthen / hold at 1
                    elif agree:
                        active_str = 1.0
                    # 4) Body crosses MA → fade one step
                    elif body_cross:
                        active_str = max(0.0, active_str - step)
                    # 5) Default time decay
                    else:
                        active_str = max(0.0, active_str - step)
                    if active_str <= 1e-12:
                        active_side, active_str, active_reason = 0, 0.0, -1

            decay_cls = 0
            decay_str = 0.0
            decay_ri = -1
            if active_side > 0 and active_str > 0:
                decay_cls = CLASS_LONG
                decay_str = float(active_str)
                decay_ri = int(active_reason)
            elif active_side < 0 and active_str > 0:
                decay_cls = CLASS_SHORT
                decay_str = float(active_str)
                decay_ri = int(active_reason)

            # Regime overrides class when trending; sideways defers to decay.
            if bool(regime_up[i]):
                cls[i] = CLASS_LONG
                strength[i] = 1.0
                reason_idx[i] = -1  # regime (not a bus column)
                if onset_out[i] and decay_cls == CLASS_LONG:
                    pass
            elif bool(regime_dn[i]):
                cls[i] = CLASS_SHORT
                strength[i] = 1.0
                reason_idx[i] = -1
            else:
                cls[i] = decay_cls
                strength[i] = decay_str
                reason_idx[i] = decay_ri

        return SignalDecision(
            cls=cls,
            strength=strength,
            reason_idx=reason_idx,
            onset=onset_out,
            reason_ids=ids,
        )


_POLICY_FACTORIES: dict[str, Callable[..., SignalPolicy]] = {
    SIGNAL_POLICY_NONE: lambda **_kw: NoneSignalPolicy(),
    SIGNAL_POLICY_ONSET_SIDE: lambda **kw: OnsetSideSignalPolicy(**kw),
    SIGNAL_POLICY_MA_DECAY: lambda **kw: MaDecaySignalPolicy(**kw),
}


def register_signal_policy(name: str, factory: Callable[..., SignalPolicy]) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("signal policy name must be non-empty")
    _POLICY_FACTORIES[key] = factory


def list_signal_policies() -> list[str]:
    return sorted(_POLICY_FACTORIES)


def normalize_signal_policy(name: str | None) -> str:
    key = str(name or DEFAULT_SIGNAL_POLICY).strip().lower() or DEFAULT_SIGNAL_POLICY
    if key not in _POLICY_FACTORIES:
        raise ValueError(
            f"Unknown signal_policy={name!r}; expected one of {list_signal_policies()}"
        )
    return key


def normalize_raw_ai_strategy(name: str | None) -> str:
    key = str(name or DEFAULT_RAW_AI_STRATEGY).strip().lower() or DEFAULT_RAW_AI_STRATEGY
    allowed = {RAW_AI_AI_ONLY, RAW_AI_SIGNAL_GATE, RAW_AI_SIGNAL_PRIOR}
    if key not in allowed:
        raise ValueError(
            f"Unknown raw_ai_strategy={name!r}; expected one of {sorted(allowed)}"
        )
    return key


def get_signal_policy(
    name: str | None = None,
    *,
    sources: Sequence[str] | None = None,
    thr: float = 0.35,
    reason_ids: Sequence[str] | None = None,
) -> SignalPolicy:
    key = normalize_signal_policy(name)
    factory = _POLICY_FACTORIES[key]
    if key == SIGNAL_POLICY_ONSET_SIDE:
        return factory(sources=sources, thr=thr, reason_ids=reason_ids)
    if key == SIGNAL_POLICY_MA_DECAY:
        return factory(sources=sources, thr=thr)
    return factory()


def resolve_raw_ai_strategy(cfg: dict) -> str:
    """Checkpoint-aware: honor ``raw_ai_strategy``, else legacy ``signal_point_gate``."""
    raw = cfg.get("raw_ai_strategy")
    if raw is not None and str(raw).strip():
        return normalize_raw_ai_strategy(str(raw))
    if bool(cfg.get("signal_point_gate")):
        return RAW_AI_SIGNAL_GATE
    return RAW_AI_AI_ONLY


def resolve_signal_policy_name(cfg: dict) -> str:
    raw = cfg.get("signal_policy")
    if raw is not None and str(raw).strip():
        return normalize_signal_policy(str(raw))
    if resolve_raw_ai_strategy(cfg) in (RAW_AI_SIGNAL_GATE, RAW_AI_SIGNAL_PRIOR):
        return SIGNAL_POLICY_ONSET_SIDE
    return SIGNAL_POLICY_NONE


def apply_raw_ai_strategy(
    ai_cls: np.ndarray,
    decision: SignalDecision,
    strategy: str | None,
    *,
    require_agree: bool = True,
    features: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    thr: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine AI argmax classes with ``SignalDecision``.

    Returns ``(gated_cls, reason_idx)``.
    """
    from smartbs_engines.signals import apply_signal_point_gate

    strategy = normalize_raw_ai_strategy(strategy)
    raw = np.asarray(ai_cls, dtype=np.int64).reshape(-1).copy()
    n = len(raw)
    if len(decision.cls) != n:
        raise ValueError(f"ai_cls len {n} != decision len {len(decision.cls)}")

    if strategy == RAW_AI_AI_ONLY:
        return raw, np.full(n, -1, dtype=np.int64)

    if strategy == RAW_AI_SIGNAL_PRIOR:
        out = raw.copy()
        use = decision.cls != CLASS_FLAT
        out[use] = decision.cls[use]
        reason = decision.reason_idx.copy()
        reason[~use] = -1
        return out, reason

    # signal_gate — prefer legacy onset gate when signal matrix available
    ids = list(feature_names or decision.reason_ids)
    if features is not None and ids and np.asarray(features).shape[0] == n:
        return apply_signal_point_gate(
            raw,
            features,
            ids,
            thr=thr,
            require_agree=require_agree,
        )

    # Fallback: gate with decision onset/side
    gated = np.zeros(n, dtype=np.int64)
    reason = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if not bool(decision.onset[i]) and int(decision.cls[i]) == CLASS_FLAT:
            continue
        ai = int(raw[i])
        sig = int(decision.cls[i])
        if require_agree:
            if sig == CLASS_FLAT:
                # onset both-sides / ambiguous — allow AI L/S
                if bool(decision.onset[i]) and ai in (CLASS_LONG, CLASS_SHORT):
                    gated[i] = ai
                    reason[i] = int(decision.reason_idx[i])
            elif ai == sig:
                gated[i] = ai
                reason[i] = int(decision.reason_idx[i])
        else:
            if ai in (CLASS_LONG, CLASS_SHORT):
                gated[i] = ai
                reason[i] = int(decision.reason_idx[i])
            elif sig in (CLASS_LONG, CLASS_SHORT):
                gated[i] = sig
                reason[i] = int(decision.reason_idx[i])
    return gated, reason
