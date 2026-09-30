"""ST-engine registry for all SmartBS feature engines.

Live blend pool: ``ACTIVE_BLEND_ENGINES`` (regime_engine first, then maribbon,
dbb, trend_pullback, smart_money, macd). ``candle`` and ``rsi_divergence``
stay registered/trainable.

Every engine from ``get_engine`` gets the shared common channel pack appended
(OHLCV 1h, tod, session, candle_direction, near_term_trend) — see
``common_channels.py``.

``signals`` is the modular signal bus; compose with
``get_engine(\"signals\", signal_sources=(...))`` or ``signals:macd,rsi_divergence``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineResult:
    """Feature matrix aligned to the bar index, plus its warm-up boundary."""

    features: np.ndarray  # (n, num_inputs) float32
    valid_from: int  # first bar whose channels are fully warmed up


@runtime_checkable
class STEngine(Protocol):
    name: str
    feature_names: list[str]
    warmup_bars: int

    def compute(self, df: pd.DataFrame) -> EngineResult: ...


class BaseSTEngine:
    """Shared plumbing: shape checks, NaN scrubbing, warm-up bookkeeping."""

    name: str = "base"
    feature_names: list[str] = []
    warmup_bars: int = 0

    @property
    def num_inputs(self) -> int:
        return len(self.feature_names)

    def compute(self, df: pd.DataFrame) -> EngineResult:  # pragma: no cover
        raise NotImplementedError

    def _finalize(self, columns: list[np.ndarray], n: int) -> EngineResult:
        if len(columns) != self.num_inputs:
            raise ValueError(
                f"{self.name}: built {len(columns)} channels but declared {self.num_inputs}"
            )
        feats = np.stack(columns, axis=1)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if feats.shape != (n, self.num_inputs):
            raise ValueError(f"{self.name}: features {feats.shape} != ({n}, {self.num_inputs})")
        return EngineResult(features=feats, valid_from=min(n, self.warmup_bars))

    def spec_hash(self) -> str:
        payload = f"{self.name}|{self.warmup_bars}|" + ",".join(self.feature_names)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    d = np.where(np.isfinite(den) & (np.abs(den) > 1e-12), den, np.nan)
    return np.nan_to_num(np.asarray(num, dtype=np.float64) / d, nan=0.0, posinf=0.0, neginf=0.0)


def atr_unit(value: np.ndarray, atr: np.ndarray, k: float = 2.0) -> np.ndarray:
    """ATR-normalized unit: ``clip(value/ATR, -k, k) / k`` → ``[-1, 1]``."""
    kk = float(k)
    if kk <= 0.0:
        raise ValueError(f"atr_unit k must be > 0, got {k!r}")
    raw = safe_div(value, atr)
    return np.clip(raw, -kk, kk) / kk


def slope_norm(series: np.ndarray, scale: np.ndarray, lookback: int = 1) -> np.ndarray:
    prev = np.roll(series, lookback)
    prev[:lookback] = series[:lookback]
    return safe_div(series - prev, scale)


def slope_unit(
    series: np.ndarray,
    atr: np.ndarray,
    lookback: int = 1,
    k: float = 2.0,
) -> np.ndarray:
    """1-bar (or N-bar) slope in ATR units, mapped to ``[-1, 1]``."""
    prev = np.roll(series, lookback)
    prev[:lookback] = series[:lookback]
    return atr_unit(series - prev, atr, k=k)


_REGISTRY: dict[str, Callable[[], BaseSTEngine]] = {}


def register_engine(name: str, factory: Callable[[], BaseSTEngine]) -> None:
    _REGISTRY[name.strip().lower()] = factory


def get_engine(
    name: str,
    *,
    signal_sources: list[str] | tuple[str, ...] | str | None = None,
    attach_common: bool = True,
) -> BaseSTEngine:
    """Instantiate an engine. By default appends the common channel pack."""
    _ensure_registered()
    from smartbs_engines.signals import SignalsSTEngine, parse_signals_engine_name

    key, parsed_sources = parse_signals_engine_name(name)
    if key == "signals":
        sources = signal_sources if signal_sources is not None else parsed_sources
        eng: BaseSTEngine = SignalsSTEngine(sources=sources)
    else:
        key = (name or "").strip().lower()
        if key not in _REGISTRY:
            raise ValueError(f"Unknown engine {name!r}; known: {sorted(_REGISTRY)}")
        eng = _REGISTRY[key]()

    if attach_common:
        from smartbs_engines.common_channels import attach_common_channels

        # Engines may opt out via ``_attach_common = False`` (core-only).
        if getattr(eng, "_attach_common", True):
            eng = attach_common_channels(eng)
    return eng


def list_engines() -> list[str]:
    _ensure_registered()
    return sorted(_REGISTRY)


def engine_warmup_bars(name: str) -> int:
    return int(get_engine(name).warmup_bars)


def engine_num_inputs(name: str) -> int:
    return int(get_engine(name).num_inputs)


class MARibbonSTEngine(BaseSTEngine):
    """11 EMAs -> 26 channels (distance, slope, RSI)."""

    name = "maribbon"

    def __init__(self) -> None:
        from smartbs_engines.maribbon_engine import (
            MARIBBON_FEATURE_NAMES,
            MARIBBON_WARMUP_BARS,
        )

        self.feature_names = list(MARIBBON_FEATURE_NAMES)
        self.warmup_bars = int(MARIBBON_WARMUP_BARS)

    def compute(self, df: pd.DataFrame) -> EngineResult:
        from smartbs_engines.maribbon_engine import compute_maribbon_engine

        eng = compute_maribbon_engine(df)
        return EngineResult(features=eng.features, valid_from=eng.valid_from)


_REGISTERED = False


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    # Mark early so nested get_engine() during signals init cannot recurse.
    _REGISTERED = True
    from smartbs_engines.engine_candle import CandlePatternSTEngine
    from smartbs_engines.engine_common import CommonOnlySTEngine
    from smartbs_engines.engine_dbb import SmartBSDbbEngine
    from smartbs_engines.engine_macd import MACDSTEngine
    from smartbs_engines.engine_pivot import PivotSTEngine
    from smartbs_engines.engine_regime import RegimeEngineSTEngine
    from smartbs_engines.engine_rsi_divergence import RSIDivergenceSTEngine
    from smartbs_engines.engine_signals import SignalsSTEngine
    from smartbs_engines.engine_smart_money import SmartMoneySTEngine
    from smartbs_engines.engine_trend_pullback import TrendPullbackSTEngine
    from smartbs_engines.engine_all_blend import AllBlendSTEngine

    # regime_engine first — Pine trend-regime (EMA14/48 + close vs mid).
    register_engine("regime_engine", RegimeEngineSTEngine)
    register_engine("maribbon", MARibbonSTEngine)
    register_engine("dbb", SmartBSDbbEngine)
    register_engine("trend_pullback", TrendPullbackSTEngine)
    register_engine("smart_money", SmartMoneySTEngine)
    register_engine("candle", CandlePatternSTEngine)
    register_engine("macd", MACDSTEngine)
    register_engine("rsi_divergence", RSIDivergenceSTEngine)
    register_engine("pivot_engine", PivotSTEngine)
    register_engine("signals", SignalsSTEngine)
    register_engine("common", CommonOnlySTEngine)
    register_engine("all_blend", AllBlendSTEngine)


from smartbs_engines.engine_catalog import (
    BLEND_LIVE as ACTIVE_BLEND_ENGINES,
    BLEND_RESEARCH as BLEND_ENGINES,
)

# Naming aliases (prefer BLEND_LIVE / BLEND_RESEARCH in new code).
BLEND_LIVE = ACTIVE_BLEND_ENGINES
BLEND_RESEARCH = BLEND_ENGINES
