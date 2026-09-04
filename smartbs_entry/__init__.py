"""SmartBS Entry — modular multi-TF entry features + TCN AI.

Standalone package (no other ST engines, no Vanta miner, no blend / RM).
Compatible with existing ``entry/*.pt`` checkpoints.
"""

from __future__ import annotations

from smartbs_entry.checkpoint import load_classifier, promote_checkpoint, checkpoint_spec_fields
from smartbs_entry.config import SmartBSConfig, package_version, smartbs_version
from smartbs_entry.engine import (
    FEATURE_NAMES,
    PAIR_CHANNELS,
    SmartBSEntryEngine,
    TF_PAIRS,
    feature_names,
)
from smartbs_entry.features import build_feature_matrix, num_inputs_for
from smartbs_entry.model import SmartBSClassifier, TemporalConvNet
from smartbs_entry.predict import predict_probs, predict_raw_ai
from smartbs_entry.registry import (
    BaseSTEngine,
    EngineResult,
    get_engine,
    list_engines,
    safe_div,
)

__version__ = package_version()

__all__ = [
    "SmartBSEntryEngine",
    "SmartBSClassifier",
    "TemporalConvNet",
    "SmartBSConfig",
    "EngineResult",
    "BaseSTEngine",
    "FEATURE_NAMES",
    "PAIR_CHANNELS",
    "TF_PAIRS",
    "feature_names",
    "build_feature_matrix",
    "num_inputs_for",
    "get_engine",
    "list_engines",
    "safe_div",
    "load_classifier",
    "promote_checkpoint",
    "checkpoint_spec_fields",
    "predict_probs",
    "predict_raw_ai",
    "package_version",
    "smartbs_version",
    "__version__",
]
