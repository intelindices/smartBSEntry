"""SmartBS engines + equal-weight blend — modular ST feature engines + TCN AI.

Standalone package (no Entry engine — see sibling ``smartbs_entry``; no Vanta
miner / Risk Manager / hydrate). Compatible with existing per-engine ``.pt``
checkpoints under ``{root}/{engine}/{PAIR}.pt``.
"""

from __future__ import annotations

from smartbs_engines.blend import BlendDecision, blend_probs, simple_mean_blend
from smartbs_engines.checkpoint import checkpoint_spec_fields, load_classifier, promote_checkpoint
from smartbs_engines.config import SmartBSConfig, package_version, smartbs_version
from smartbs_engines.features import build_feature_matrix, num_inputs_for
from smartbs_engines.model import SmartBSClassifier, TemporalConvNet
from smartbs_engines.predict import (
    predict_blend_probs,
    predict_blend_raw_ai,
    predict_probs,
    predict_raw_ai,
)
from smartbs_engines.registry import (
    ACTIVE_BLEND_ENGINES,
    BLEND_ENGINES,
    BaseSTEngine,
    EngineResult,
    engine_num_inputs,
    engine_warmup_bars,
    get_engine,
    list_engines,
    safe_div,
)
from smartbs_engines.weights import EnginePerf, compute_weights, equal_weights

__version__ = package_version()

__all__ = [
    "ACTIVE_BLEND_ENGINES",
    "BLEND_ENGINES",
    "BaseSTEngine",
    "BlendDecision",
    "EnginePerf",
    "EngineResult",
    "SmartBSClassifier",
    "SmartBSConfig",
    "TemporalConvNet",
    "blend_probs",
    "build_feature_matrix",
    "checkpoint_spec_fields",
    "compute_weights",
    "engine_num_inputs",
    "engine_warmup_bars",
    "equal_weights",
    "get_engine",
    "list_engines",
    "load_classifier",
    "num_inputs_for",
    "package_version",
    "predict_blend_probs",
    "predict_blend_raw_ai",
    "predict_probs",
    "predict_raw_ai",
    "promote_checkpoint",
    "safe_div",
    "simple_mean_blend",
    "smartbs_version",
    "__version__",
]
