"""Label-mode plugins + re-exports of label helpers."""

from __future__ import annotations

from smartbs_engines.labels.core import *  # noqa: F403
from smartbs_engines.labels.plugins import (  # noqa: F401
    LABEL_PLUGINS,
    LabelPlugin,
    list_label_modes,
    register_label_plugin,
)
from smartbs_engines.labels.plugins import normalize_label_mode
from smartbs_engines.labels import modes as _modes  # noqa: F401

__all__ = sorted(
    {
        *[n for n in globals() if not n.startswith("_")],
        "LABEL_PLUGINS",
        "LabelPlugin",
        "list_label_modes",
        "normalize_label_mode",
        "register_label_plugin",
    }
)
