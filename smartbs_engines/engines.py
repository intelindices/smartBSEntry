"""Feature-engine helpers used by train / infer.

Engines are registered in ``st_engines``. This module is the small facade those
callers import: normalize the name, report warm-up, refuse retired engines.
"""

from __future__ import annotations

FEATURE_ENGINE_MARIBBON = "maribbon"
REMOVED_ENGINES = frozenset(
    {
        "sbs",
        "pine",
        "pbs",
        "entry",
        "bollinger",
        "ohlcv",
        "temp_engine",
        "tempengine",
        "tempe",
        "temp",
    }
)


def normalize_feature_engine(engine: str | None) -> str:
    eng = (engine or FEATURE_ENGINE_MARIBBON).strip().lower()
    if eng in REMOVED_ENGINES:
        if eng == "bollinger":
            hint = "dbb"
        elif eng in ("ohlcv", "temp_engine", "tempengine", "tempe", "temp"):
            hint = "macd (common OHLCV/tod/session channels are attached to every engine)"
        else:
            hint = "maribbon"
        raise ValueError(
            f"feature_engine={eng!r} was removed. "
            f"Retrain with --feature-engine {hint}."
        )
    # regimeEngine / regime → regime_engine
    if eng in ("regimeengine", "regime_engine", "regime"):
        eng = "regime_engine"
    # signals / signals:macd,rsi_divergence → canonical name "signals"
    if eng.startswith("signals"):
        from smartbs_engines.signals import parse_signals_engine_name

        parse_signals_engine_name(eng)  # validate
        return "signals"
    from smartbs_engines.registry import list_engines

    if eng == FEATURE_ENGINE_MARIBBON or eng in list_engines():
        return eng
    raise ValueError(f"Unknown feature engine {engine!r}; expected one of {list_engines()}")


def resolve_feature_engine(
    engine: str | None,
    signal_engines: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[str, tuple[str, ...] | None]:
    """Return ``(canonical_engine, signal_sources|None)``.

    Examples::

        resolve_feature_engine("macd")
        resolve_feature_engine("signals")
        resolve_feature_engine("signals:rsi_divergence,macd")
        resolve_feature_engine("signals", signal_engines="macd+candle")
    """
    raw = (engine or FEATURE_ENGINE_MARIBBON).strip().lower()
    if raw.startswith("signals"):
        from smartbs_engines.signals import normalize_signal_sources, parse_signals_engine_name

        _, parsed = parse_signals_engine_name(raw)
        if signal_engines:
            sources = normalize_signal_sources(signal_engines)
        else:
            sources = normalize_signal_sources(parsed)
        return "signals", sources
    return normalize_feature_engine(engine), None


def engine_warmup_bars(engine: str | None) -> int:
    """Minimum bars before an engine's features are comparable to training."""
    from smartbs_engines.registry import get_engine

    name, sources = resolve_feature_engine(engine)
    if name == "signals":
        return int(get_engine("signals", signal_sources=sources).warmup_bars)
    return int(get_engine(name).warmup_bars)
