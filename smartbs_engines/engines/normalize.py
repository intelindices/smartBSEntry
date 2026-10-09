"""Normalize feature-engine names against registered plugins."""

from __future__ import annotations

FEATURE_ENGINE_MARIBBON = "maribbon"


def normalize_feature_engine(engine: str | None) -> str:
    eng = (engine or FEATURE_ENGINE_MARIBBON).strip().lower()
    if eng.startswith("signals"):
        eng = "signals"
    from smartbs_engines.registry import list_engines

    known = list_engines()
    if eng in known:
        return eng
    raise ValueError(f"Unknown feature engine {engine!r}; expected one of {known}")


def resolve_feature_engine(
    engine: str | None,
    signal_engines: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[str, tuple[str, ...] | None]:
    del signal_engines
    return normalize_feature_engine(engine), None


def engine_warmup_bars(engine: str | None) -> int:
    from smartbs_engines.registry import get_engine

    name, _ = resolve_feature_engine(engine)
    return int(get_engine(name).warmup_bars)
