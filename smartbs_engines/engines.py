"""Feature-engine helpers used by train / infer.

Engines are registered in ``st_engines``. This module is the small facade those
callers import: normalize the name, report warm-up, refuse retired engines.
"""

from __future__ import annotations

FEATURE_ENGINE_MARIBBON = "maribbon"
REMOVED_ENGINES = frozenset({"sbs", "pine", "pbs", "entry", "bollinger"})


def normalize_feature_engine(engine: str | None) -> str:
    eng = (engine or FEATURE_ENGINE_MARIBBON).strip().lower()
    if eng in REMOVED_ENGINES:
        hint = "dbb" if eng == "bollinger" else "maribbon"
        raise ValueError(
            f"feature_engine={eng!r} was removed. "
            f"Retrain with --feature-engine {hint} (or another ST engine)."
        )
    from smartbs_engines.registry import list_engines

    if eng == FEATURE_ENGINE_MARIBBON or eng in list_engines():
        return eng
    raise ValueError(f"Unknown feature engine {engine!r}; expected one of {list_engines()}")


def engine_warmup_bars(engine: str | None) -> int:
    """Minimum bars before an engine's features are comparable to training."""
    from smartbs_engines.registry import get_engine

    return int(get_engine(normalize_feature_engine(engine)).warmup_bars)
