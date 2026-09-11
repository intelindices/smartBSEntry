"""ST-engine registry for all SmartBS feature engines.

Live blend pool: ``ACTIVE_BLEND_ENGINES`` (maribbon, dbb, trend_pullback,
smart_money, macd). ``candle`` and ``rsi_divergence`` stay registered/trainable.
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


def slope_norm(series: np.ndarray, scale: np.ndarray, lookback: int = 1) -> np.ndarray:
    prev = np.roll(series, lookback)
    prev[:lookback] = series[:lookback]
    return safe_div(series - prev, scale)


_REGISTRY: dict[str, Callable[[], BaseSTEngine]] = {}


def register_engine(name: str, factory: Callable[[], BaseSTEngine]) -> None:
    _REGISTRY[name.strip().lower()] = factory


def get_engine(name: str) -> BaseSTEngine:
    _ensure_registered()
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown engine {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def list_engines() -> list[str]:
    _ensure_registered()
    return sorted(_REGISTRY)


def engine_warmup_bars(name: str) -> int:
    return int(get_engine(name).warmup_bars)


def engine_num_inputs(name: str) -> int:
    return int(get_engine(name).num_inputs)


class MARibbonSTEngine(BaseSTEngine):
    """11 EMAs -> 48 channels (direction, distance, slope, angle, RSI)."""

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
    from smartbs_engines.engine_candle import CandlePatternSTEngine
    from smartbs_engines.engine_dbb import SmartBSDbbEngine
    from smartbs_engines.engine_macd import MACDSTEngine
    from smartbs_engines.engine_rsi_divergence import RSIDivergenceSTEngine
    from smartbs_engines.engine_smart_money import SmartMoneySTEngine
    from smartbs_engines.engine_trend_pullback import TrendPullbackSTEngine

    register_engine("maribbon", MARibbonSTEngine)
    register_engine("dbb", SmartBSDbbEngine)
    register_engine("trend_pullback", TrendPullbackSTEngine)
    register_engine("smart_money", SmartMoneySTEngine)
    register_engine("candle", CandlePatternSTEngine)
    register_engine("macd", MACDSTEngine)
    register_engine("rsi_divergence", RSIDivergenceSTEngine)
    _REGISTERED = True


BLEND_ENGINES: tuple[str, ...] = (
    "maribbon",
    "dbb",
    "trend_pullback",
    "smart_money",
    "candle",
    "macd",
    "rsi_divergence",
)

# Production mean blend.
ACTIVE_BLEND_ENGINES: tuple[str, ...] = (
    "maribbon",
    "dbb",
    "trend_pullback",
    "smart_money",
    "macd",
)
