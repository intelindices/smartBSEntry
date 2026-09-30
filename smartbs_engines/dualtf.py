"""DualTF (1h+15m) dataset + predict facade.

Keeps DualTF behind one import surface so callers do not reach into
``features.make_dual_datasets`` / DualTF predict branches directly.
"""

from __future__ import annotations

from smartbs_engines.features import (
    DualWindowDataset,
    build_1h_to_15m_end_index,
    make_dual_datasets,
)
from smartbs_engines.model import BACKBONE_SMARTBS_DUAL_TF, normalize_backbone

__all__ = [
    "BACKBONE_SMARTBS_DUAL_TF",
    "DualWindowDataset",
    "build_1h_to_15m_end_index",
    "is_dualtf",
    "make_dual_datasets",
]


def is_dualtf(backbone: str | None) -> bool:
    return normalize_backbone(backbone) == BACKBONE_SMARTBS_DUAL_TF
