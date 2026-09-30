"""Composable train/infer pipeline: engines × signal policy × backbone × raw_ai.

Free combination::

    PipelineSpec(engines=(\"macd\",), signal_policy=\"none\", backbone=\"tcn\", raw_ai=\"ai_only\")
    PipelineSpec(engines=(\"dbb\", \"macd\"), signal_policy=\"onset_side\",
                 backbone=\"smartBSDualTF\", raw_ai=\"signal_gate\")

Defaults: signal_policy=none, raw_ai=ai_only (raw AI side; no signal gate).

Feature resolution:

* one engine + ``signal_policy=none`` → that engine's channels (legacy path)
* multiple engines → ``signals:`` bus
* ``signal_policy=onset_side`` may still train on a single engine's channels;
  the policy builds the signals bus internally when deciding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from smartbs_engines.engines import resolve_feature_engine
from smartbs_engines.model import normalize_backbone
from smartbs_engines.signal_policy import (
    DEFAULT_RAW_AI_STRATEGY,
    DEFAULT_SIGNAL_POLICY,
    SIGNAL_POLICY_NONE,
    get_signal_policy,
    normalize_raw_ai_strategy,
    normalize_signal_policy,
    resolve_raw_ai_strategy,
    resolve_signal_policy_name,
)
from smartbs_engines.signals import ACTIVE_SIGNAL_SOURCES, normalize_signal_sources


@dataclass(frozen=True)
class PipelineSpec:
    """Declarative plug-board for engines, signal layer, backbone, raw_ai."""

    engines: tuple[str, ...]
    signal_policy: str = DEFAULT_SIGNAL_POLICY
    backbone: str = "tcn"
    raw_ai: str = DEFAULT_RAW_AI_STRATEGY
    signal_thr: float = 0.35
    signal_require_agree: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "signal_policy", normalize_signal_policy(self.signal_policy)
        )
        object.__setattr__(self, "raw_ai", normalize_raw_ai_strategy(self.raw_ai))
        object.__setattr__(self, "backbone", normalize_backbone(self.backbone))
        engs = tuple(
            str(e).strip().lower() for e in self.engines if str(e).strip()
        )
        if not engs:
            raise ValueError("PipelineSpec.engines must be non-empty")
        object.__setattr__(self, "engines", engs)

    @classmethod
    def from_config(cls, cfg: Any) -> "PipelineSpec":
        """Build from ``SmartBSConfig`` or checkpoint dict."""
        if hasattr(cfg, "feature_engine"):
            fe = getattr(cfg, "feature_engine", "maribbon")
            sig_eng = getattr(cfg, "signal_engines", ()) or None
            backbone = getattr(cfg, "backbone", "tcn")
            sp = getattr(cfg, "signal_policy", None)
            raw = getattr(cfg, "raw_ai_strategy", None)
            thr = float(getattr(cfg, "signal_point_thr", 0.35) or 0.35)
            agree = bool(getattr(cfg, "signal_point_require_agree", True))
            gate = bool(getattr(cfg, "signal_point_gate", False))
            d = {
                "feature_engine": fe,
                "signal_engines": list(sig_eng) if sig_eng else [],
                "backbone": backbone,
                "signal_policy": sp,
                "raw_ai_strategy": raw,
                "signal_point_gate": gate,
                "signal_point_thr": thr,
                "signal_point_require_agree": agree,
            }
        else:
            d = dict(cfg)

        name, sources = resolve_feature_engine(
            d.get("feature_engine"),
            d.get("signal_engines") or None,
        )
        if name == "signals":
            engines = tuple(sources or ACTIVE_SIGNAL_SOURCES)
        else:
            engines = (name,)

        raw_ai = resolve_raw_ai_strategy(d)
        signal_policy = resolve_signal_policy_name(d)
        if sp := d.get("signal_policy"):
            signal_policy = normalize_signal_policy(str(sp))

        return cls(
            engines=engines,
            signal_policy=signal_policy,
            backbone=str(d.get("backbone") or "tcn"),
            raw_ai=raw_ai,
            signal_thr=float(d.get("signal_point_thr", 0.35) or 0.35),
            signal_require_agree=bool(d.get("signal_point_require_agree", True)),
        )

    def feature_engine_name(self) -> str:
        """Canonical ``feature_engine`` string for train/checkpoint."""
        if len(self.engines) == 1 and self.signal_policy == SIGNAL_POLICY_NONE:
            return self.engines[0]
        if len(self.engines) == 1:
            # Single engine as model features; signal policy may use its own bus.
            return self.engines[0]
        return "signals"

    def signal_engines(self) -> tuple[str, ...]:
        if self.feature_engine_name() == "signals":
            return self.engines
        if self.signal_policy != SIGNAL_POLICY_NONE:
            return self.engines
        return ()

    def resolve_feature_engine(self) -> tuple[str, tuple[str, ...] | None]:
        name = self.feature_engine_name()
        if name == "signals":
            return "signals", normalize_signal_sources(self.engines)
        return name, None

    def make_signal_policy(self):
        return get_signal_policy(
            self.signal_policy,
            sources=self.engines if self.signal_policy != SIGNAL_POLICY_NONE else None,
            thr=self.signal_thr,
        )

    def to_checkpoint_fields(self) -> dict[str, Any]:
        name, sources = self.resolve_feature_engine()
        return {
            "feature_engine": name,
            "signal_engines": list(sources or self.signal_engines()),
            "pipeline_engines": list(self.engines),
            "signal_policy": self.signal_policy,
            "raw_ai_strategy": self.raw_ai,
            "backbone": self.backbone,
            "signal_point_gate": self.raw_ai == "signal_gate",
            "signal_point_thr": float(self.signal_thr),
            "signal_point_require_agree": bool(self.signal_require_agree),
        }


def engines_from_arg(
    engines: str | Sequence[str] | None,
    *,
    feature_engine: str | None = None,
    signal_engines: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Parse CLI/config engine lists into a normalized tuple."""
    if engines is not None and str(engines).strip():
        if isinstance(engines, str):
            parts = [
                p.strip().lower()
                for p in engines.replace("+", ",").split(",")
                if p.strip()
            ]
            return tuple(parts)
        return tuple(str(e).strip().lower() for e in engines if str(e).strip())
    name, sources = resolve_feature_engine(feature_engine, signal_engines)
    if name == "signals":
        return tuple(sources or ACTIVE_SIGNAL_SOURCES)
    return (name,)
