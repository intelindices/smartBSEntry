"""SmartBS engines + equal-weight blend — modular ST feature engines + TCN AI.

Standalone package (no Entry engine; no Vanta miner / Risk Manager / hydrate).
Train each engine independently; blend any subset; export ONNX for the EA.
"""

from __future__ import annotations

from smartbs_engines.blend import BlendDecision, blend_probs, simple_mean_blend
from smartbs_engines.checkpoint import checkpoint_spec_fields, load_classifier, promote_checkpoint
from smartbs_engines.config import SmartBSConfig, package_version, smartbs_version
from smartbs_engines.features import build_feature_matrix, num_inputs_for
from smartbs_engines.model import (
    SmartBSClassifier,
    SmartBSDualTF,
    SmartBSEntryV2,
    SmartBSTF,
    TemporalConvNet,
    build_classifier,
)
from smartbs_engines.model import (
    BACKBONE_SMARTBS_DUAL_TF,
    BACKBONE_SMARTBS_ENTRY_V2,
    BACKBONE_SMARTBS_TF,
    BACKBONE_TCN,
)
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
from smartbs_engines.pipeline import PipelineSpec, engines_from_arg
from smartbs_engines.signal_policy import (
    RAW_AI_AI_ONLY,
    RAW_AI_SIGNAL_GATE,
    RAW_AI_SIGNAL_PRIOR,
    SIGNAL_POLICY_MA_DECAY,
    SIGNAL_POLICY_NONE,
    SIGNAL_POLICY_ONSET_SIDE,
    SignalDecision,
    apply_raw_ai_strategy,
    get_signal_policy,
    list_signal_policies,
    register_signal_policy,
)
from smartbs_engines.signals import (
    ACTIVE_SIGNAL_SOURCES,
    ENGINE_SIDE_REASON_IDS,
    SIGNAL_POOL_ENGINES,
    SignalsSTEngine,
    apply_signal_point_gate,
    list_signal_catalog,
    normalize_signal_sources,
    signal_point_onset,
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
    "BACKBONE_SMARTBS_DUAL_TF",
    "BACKBONE_SMARTBS_ENTRY_V2",
    "BACKBONE_SMARTBS_TF",
    "BACKBONE_TCN",
    "SmartBSClassifier",
    "SmartBSConfig",
    "SmartBSDualTF",
    "SmartBSEntryV2",
    "SmartBSTF",
    "TemporalConvNet",
    "blend_probs",
    "build_classifier",
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
    "PipelineSpec",
    "RAW_AI_AI_ONLY",
    "RAW_AI_SIGNAL_GATE",
    "RAW_AI_SIGNAL_PRIOR",
    "SIGNAL_POLICY_MA_DECAY",
    "SIGNAL_POLICY_NONE",
    "SIGNAL_POLICY_ONSET_SIDE",
    "SignalDecision",
    "ACTIVE_SIGNAL_SOURCES",
    "ENGINE_SIDE_REASON_IDS",
    "SIGNAL_POOL_ENGINES",
    "SignalsSTEngine",
    "apply_raw_ai_strategy",
    "apply_signal_point_gate",
    "engines_from_arg",
    "get_signal_policy",
    "list_signal_catalog",
    "list_signal_policies",
    "normalize_signal_sources",
    "register_signal_policy",
    "safe_div",
    "signal_point_onset",
    "simple_mean_blend",
    "smartbs_version",
    "__version__",
]
