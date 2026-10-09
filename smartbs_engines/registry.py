"""ST-engine runtime registry — backed by ``engines.ENGINE_PLUGINS``."""

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

    def compute(self, df: pd.DataFrame, **kwargs) -> EngineResult: ...


class BaseSTEngine:
    """Shared plumbing: shape checks, NaN scrubbing, warm-up bookkeeping."""

    name: str = "base"
    feature_names: list[str] = []
    warmup_bars: int = 0

    @property
    def num_inputs(self) -> int:
        return len(self.feature_names)

    def compute(self, df: pd.DataFrame, **kwargs) -> EngineResult:  # pragma: no cover
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


def _plugins():
    from smartbs_engines.engines import ENGINE_PLUGINS
    from smartbs_engines.plugins.base import ensure_plugins_loaded

    ensure_plugins_loaded()
    return ENGINE_PLUGINS


def register_engine(name: str, factory: Callable[[], BaseSTEngine]) -> None:
    """Legacy helper — prefer ``engines.register_engine_plugin``."""
    from smartbs_engines.engines import EnginePlugin, register_engine_plugin

    register_engine_plugin(
        EnginePlugin(
            name=name,
            core_channels=0,
            factory=factory,
            mql_define=f"SB_{name.upper()}_CH",
        )
    )


def get_engine(
    name: str,
    *,
    signal_sources: list[str] | tuple[str, ...] | str | None = None,
    attach_common: bool = True,
) -> BaseSTEngine:
    """Instantiate an engine. By default appends the common channel pack."""
    del signal_sources
    plugs = _plugins()
    key = (name or "").strip().lower()
    if key not in plugs:
        raise ValueError(f"Unknown engine {name!r}; known: {plugs.names()}")
    plugin = plugs.get(key)
    eng: BaseSTEngine = plugin.factory()

    if attach_common and plugin.attach_common:
        from smartbs_engines.common_channels import attach_common_channels

        if getattr(eng, "_attach_common", True):
            eng = attach_common_channels(eng)
    return eng


def list_engines() -> list[str]:
    return _plugins().names()


def engine_warmup_bars(name: str) -> int:
    return int(get_engine(name).warmup_bars)


def engine_num_inputs(name: str) -> int:
    return int(get_engine(name).num_inputs)


def _blend_names(*, live: bool) -> tuple[str, ...]:
    plugs = _plugins()
    out = []
    for name, p in plugs.items():
        if live and p.in_blend_live:
            out.append(name)
        elif (not live) and p.in_blend_research:
            out.append(name)
    return tuple(out)


def _lazy_blend(live: bool) -> tuple[str, ...]:
    return _blend_names(live=live)


# Eager aliases for ``from registry import ACTIVE_BLEND_ENGINES``.
def __getattr__(name: str):
    if name in ("ACTIVE_BLEND_ENGINES", "BLEND_LIVE"):
        return _lazy_blend(True)
    if name in ("BLEND_ENGINES", "BLEND_RESEARCH"):
        return _lazy_blend(False)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


