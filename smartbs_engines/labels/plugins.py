"""Label plugin registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from smartbs_engines.plugins.base import PluginRegistry

LABEL_PLUGINS: PluginRegistry["LabelPlugin"] = PluginRegistry("label")

LABEL_MODE_ALIASES = {
    "pivot_breakout": "rolle_breakout",
}


@dataclass(frozen=True)
class LabelPlugin:
    name: str
    compute: Callable[[pd.DataFrame, Any], np.ndarray]
    is_barrier: bool = False
    notes: str = ""


def register_label_plugin(plugin: LabelPlugin) -> LabelPlugin:
    return LABEL_PLUGINS.register(plugin.name, plugin)


def list_label_modes() -> list[str]:
    from smartbs_engines.plugins.base import ensure_plugins_loaded

    ensure_plugins_loaded()
    return LABEL_PLUGINS.names()


def normalize_label_mode(mode: str | None) -> str:
    m = str(mode or "").strip().lower()
    m = LABEL_MODE_ALIASES.get(m, m)
    from smartbs_engines.plugins.base import ensure_plugins_loaded

    ensure_plugins_loaded()
    if m not in LABEL_PLUGINS:
        raise ValueError(
            f"label_mode={m!r} unknown; expected one of {LABEL_PLUGINS.names()}"
        )
    return m
