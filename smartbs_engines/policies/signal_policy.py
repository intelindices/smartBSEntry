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
        # signals bus removed — policy kept for API compat; emits FLAT.
        del features, feature_names, symbol, data_source
        return SignalDecision.all_flat(len(df))


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
        # signals / regime_engine bus removed — API compat stub.
        del features, feature_names, symbol, data_source
        return SignalDecision.all_flat(len(df))


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
    del features, feature_names, thr
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

    # signal_gate — decision onset/side (signals bus removed)
    gated = np.zeros(n, dtype=np.int64)
    reason = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if not bool(decision.onset[i]) and int(decision.cls[i]) == CLASS_FLAT:
            continue
        ai = int(raw[i])
        sig = int(decision.cls[i])
        if require_agree:
            if sig == CLASS_FLAT:
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
