"""Maribbon ST engine plugin — 39ch multi-TF EMA ribbon (no common attach)."""

from __future__ import annotations

import pandas as pd

from smartbs_engines.engines.maribbon_core import (
    MARIBBON_FEATURE_NAMES,
    MARIBBON_WARMUP_BARS,
    compute_maribbon_engine,
)
from smartbs_engines.registry import BaseSTEngine, EngineResult


class MARibbonSTEngine(BaseSTEngine):
    """12 EMAs → 13 base × (1h + 15m + 5m) = 39 channels; no common pack."""

    name = "maribbon"
    _attach_common = False

    def __init__(self) -> None:
        self.feature_names = list(MARIBBON_FEATURE_NAMES)
        self.warmup_bars = int(MARIBBON_WARMUP_BARS)

    def compute(
        self,
        df: pd.DataFrame,
        *,
        symbol: str | None = None,
        data_source: str | None = None,
        frames: dict | None = None,
        **_kwargs,
    ) -> EngineResult:
        eng = compute_maribbon_engine(
            df, symbol=symbol, data_source=data_source, frames=frames
        )
        return EngineResult(features=eng.features, valid_from=eng.valid_from)
