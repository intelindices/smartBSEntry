"""Common-channels-only ST engine (no core feature pack).

Use for ablations / labels that should train on the shared 16 common channels
alone. ``get_engine(\"common\")`` skips a second attach because this engine
already is the common pack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.common_channels import (
    COMMON_FEATURE_NAMES,
    COMMON_WARMUP,
    compute_common_channels,
)
from smartbs_engines.registry import BaseSTEngine, EngineResult


class CommonOnlySTEngine(BaseSTEngine):
    """Features = common channel pack only."""

    _has_common_channels = True
    name = "common"

    def __init__(self) -> None:
        self.feature_names = list(COMMON_FEATURE_NAMES)
        self.warmup_bars = int(COMMON_WARMUP)

    def compute(self, df: pd.DataFrame, **kwargs) -> EngineResult:
        n = len(df)
        feats = compute_common_channels(df)
        if feats.shape != (n, self.num_inputs):
            raise ValueError(
                f"common: features {feats.shape} != ({n}, {self.num_inputs})"
            )
        return EngineResult(
            features=np.asarray(feats, dtype=np.float32),
            valid_from=min(n, self.warmup_bars),
        )
