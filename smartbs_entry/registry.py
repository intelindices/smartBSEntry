"""Slim ST-engine registry — entry engine only (no maribbon/blend/other engines)."""

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

    def compute(self, df: pd.DataFrame) -> EngineResult:  # pragma: no cover - abstract
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
        """Identity of this engine's feature contract (stored on checkpoints)."""
        payload = f"{self.name}|{self.warmup_bars}|" + ",".join(self.feature_names)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    d = np.where(np.isfinite(den) & (np.abs(den) > 1e-12), den, np.nan)
    return np.nan_to_num(np.asarray(num, dtype=np.float64) / d, nan=0.0, posinf=0.0, neginf=0.0)


_REGISTRY: dict[str, Callable[[], BaseSTEngine]] = {}


def register_engine(name: str, factory: Callable[[], BaseSTEngine]) -> None:
    _REGISTRY[name.strip().lower()] = factory


def get_engine(name: str = "entry") -> BaseSTEngine:
    key = (name or "entry").strip().lower()
    if key not in _REGISTRY:
        if key == "entry":
            from smartbs_entry.engine import SmartBSEntryEngine

            register_engine("entry", SmartBSEntryEngine)
        else:
            raise ValueError(f"Unknown engine {name!r}; known: {sorted(_REGISTRY)}")
    if key not in _REGISTRY:
        raise ValueError(f"Unknown engine {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def list_engines() -> list[str]:
    if "entry" not in _REGISTRY:
        get_engine("entry")
    return sorted(_REGISTRY)


def engine_warmup_bars(name: str = "entry") -> int:
    return int(get_engine(name).warmup_bars)


def engine_num_inputs(name: str = "entry") -> int:
    return int(get_engine(name).num_inputs)


def normalize_feature_engine(engine: str | None) -> str:
    eng = (engine or "entry").strip().lower()
    if eng not in _REGISTRY:
        # Lazy import avoids circular import with engine.py
        if eng == "entry" and "entry" not in _REGISTRY:
            from smartbs_entry.engine import SmartBSEntryEngine

            register_engine("entry", SmartBSEntryEngine)
        if eng not in _REGISTRY:
            raise ValueError(f"Unknown feature engine {engine!r}; expected one of {list_engines()}")
    return eng
