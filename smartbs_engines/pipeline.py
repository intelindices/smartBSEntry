"""Composable train/infer pipeline: engines × signal policy × backbone × raw_ai."""

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
        if len(engs) != 1:
            raise ValueError(
                f"PipelineSpec supports one feature engine (got {engs}); "
                "signals bus was removed"
            )
        object.__setattr__(self, "engines", engs)

    @classmethod
    def from_config(cls, cfg: Any) -> "PipelineSpec":
        if hasattr(cfg, "feature_engine"):
            fe = getattr(cfg, "feature_engine", "maribbon")
            backbone = getattr(cfg, "backbone", "tcn")
            sp = getattr(cfg, "signal_policy", None)
            raw = getattr(cfg, "raw_ai_strategy", None)
            thr = float(getattr(cfg, "signal_point_thr", 0.35) or 0.35)
            agree = bool(getattr(cfg, "signal_point_require_agree", True))
            gate = bool(getattr(cfg, "signal_point_gate", False))
            d = {
                "feature_engine": fe,
                "backbone": backbone,
                "signal_policy": sp,
                "raw_ai_strategy": raw,
                "signal_point_gate": gate,
                "signal_point_thr": thr,
                "signal_point_require_agree": agree,
            }
        else:
            d = dict(cfg)

        name, _sources = resolve_feature_engine(d.get("feature_engine"), None)
        raw_ai = resolve_raw_ai_strategy(d)
        signal_policy = resolve_signal_policy_name(d)
        if sp := d.get("signal_policy"):
            signal_policy = normalize_signal_policy(str(sp))

        return cls(
            engines=(name,),
            signal_policy=signal_policy,
            backbone=str(d.get("backbone") or "tcn"),
            raw_ai=raw_ai,
            signal_thr=float(d.get("signal_point_thr", 0.35) or 0.35),
            signal_require_agree=bool(d.get("signal_point_require_agree", True)),
        )

    def feature_engine_name(self) -> str:
        return self.engines[0]

    def signal_engines(self) -> tuple[str, ...]:
        return ()

    def resolve_feature_engine(self) -> tuple[str, tuple[str, ...] | None]:
        return self.feature_engine_name(), None

    def make_signal_policy(self):
        return get_signal_policy(
            self.signal_policy,
            sources=None,
            thr=self.signal_thr,
        )

    def to_checkpoint_fields(self) -> dict[str, Any]:
        name, _ = self.resolve_feature_engine()
        return {
            "feature_engine": name,
            "signal_engines": [],
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
    del signal_engines
    if engines is not None and str(engines).strip():
        if isinstance(engines, str):
            parts = [
                p.strip().lower()
                for p in engines.replace("+", ",").split(",")
                if p.strip()
            ]
            return tuple(parts[:1]) if parts else ()
        return tuple(str(e).strip().lower() for e in list(engines)[:1] if str(e).strip())
    name, _ = resolve_feature_engine(feature_engine, None)
    return (name,)
