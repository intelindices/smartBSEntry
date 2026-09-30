"""all_blend: union of every ST engine's channels, deduped by name.

Sources (catalog order): regime_engine, maribbon, dbb, trend_pullback,
smart_money, macd, candle, rsi_divergence.

- Skips ``common`` / ``signals`` / ``all_blend`` as sources.
- First occurrence of a channel name wins (6 cross-engine collisions today).
- Common pack is embedded here (filtered) so noisy common cols can be dropped;
  ``get_engine`` does not double-attach (``_has_common_channels = True``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.common_channels import (
    COMMON_FEATURE_NAMES,
    COMMON_WARMUP,
    _call_compute,
    compute_common_channels,
)
from smartbs_engines.registry import BaseSTEngine, EngineResult, get_engine

# Feature engines to union (excludes common, signals, all_blend).
ALL_BLEND_SOURCES: tuple[str, ...] = (
    "regime_engine",
    "maribbon",
    "dbb",
    "trend_pullback",
    "smart_money",
    "macd",
    "candle",
    "rsi_divergence",
)

# Ablation-driven drops — empty (core 134 + common 16 = 150).
ALL_BLEND_DROP_CHANNELS: frozenset[str] = frozenset()

# Near-duplicate cliques (|r|≥0.95 on XAU 5y). Used for LOGO clique ablation
# via ``ablation_zero_group=alias_N`` (dim kept; columns zeroed at train+infer).
ALL_BLEND_ALIAS_CLIQUES: dict[str, tuple[str, ...]] = {
    "alias_1": (
        "dist_close_mid",
        "dist_fast",
        "slope_fast",
        "slope_mid",
        "dist_ema9",
        "dist_ema24",
        "slope_ema9",
        "slope_ema14",
        "slope_ema24",
        "slope_ema40",
        "slope_ema60",
        "slope_ema100",
        "slope_ema160",
        "slope_ema240",
        "slope_ema320",
        "rsi_norm",
        "rsi_vs_mid",
        "dist_mid",
        "slope_slow",
        "stack_12_26",
        "dist_fast_12",
        "dist_slow_26",
        "slope_fast_12",
        "slope_slow_26",
        "close_vs_stack",
        "stack_5_35",
        "close_vs_trend_atr",
        "rsi_maj_norm",
        "dist_maj_from_30",
        "dist_maj_from_70",
        "rsi_min_norm",
    ),
    "alias_2": (
        "dist_fast_mid",
        "dist_ema40",
        "stack_fast_mid",
        "trend_stack_atr",
        "bull_trend_soft",
        "bear_trend_soft",
    ),
    "alias_3": ("rsi_to_band", "rsi_maj_in_band"),
    "alias_4": ("range_pos", "mr_stretch"),
    "alias_5": ("sim_xdn_70_decay", "rev_short_decay"),
    "alias_6": ("ohlcv_open", "ohlcv_close_displacement"),
}


def _plan_channels() -> tuple[list[str], list[tuple[str, int]], int]:
    """Return ``(feature_names, [(source, local_col_idx), ...], warmup)``."""
    names: list[str] = []
    plan: list[tuple[str, int]] = []
    seen: set[str] = set()
    warmup = 0
    drop = ALL_BLEND_DROP_CHANNELS
    for src in ALL_BLEND_SOURCES:
        eng = get_engine(src, attach_common=False)
        warmup = max(warmup, int(eng.warmup_bars))
        for j, name in enumerate(eng.feature_names):
            key = str(name)
            if key in seen or key in drop:
                continue
            seen.add(key)
            names.append(key)
            plan.append((src, int(j)))
    return names, plan, warmup


def _common_keep() -> tuple[list[str], list[int]]:
    """Common feature names + column indices kept after drops."""
    names: list[str] = []
    idxs: list[int] = []
    drop = ALL_BLEND_DROP_CHANNELS
    for i, name in enumerate(COMMON_FEATURE_NAMES):
        if name in drop:
            continue
        names.append(str(name))
        idxs.append(int(i))
    return names, idxs


_CACHED_NAMES: list[str] | None = None
_CACHED_PLAN: list[tuple[str, int]] | None = None
_CACHED_WARMUP: int | None = None
_CACHED_COMMON_NAMES: list[str] | None = None
_CACHED_COMMON_IDX: list[int] | None = None


def _ensure_cache() -> None:
    global _CACHED_NAMES, _CACHED_PLAN, _CACHED_WARMUP
    global _CACHED_COMMON_NAMES, _CACHED_COMMON_IDX
    if _CACHED_NAMES is None or _CACHED_PLAN is None or _CACHED_WARMUP is None:
        _CACHED_NAMES, _CACHED_PLAN, _CACHED_WARMUP = _plan_channels()
    if _CACHED_COMMON_NAMES is None or _CACHED_COMMON_IDX is None:
        _CACHED_COMMON_NAMES, _CACHED_COMMON_IDX = _common_keep()


def all_blend_core_channel_count() -> int:
    _ensure_cache()
    assert _CACHED_NAMES is not None
    return len(_CACHED_NAMES)


# Ablation groups: each source engine + the shared common pack + alias cliques.
ALL_BLEND_GROUPS: tuple[str, ...] = ALL_BLEND_SOURCES + ("common",)
ALL_BLEND_ALIAS_GROUP_NAMES: tuple[str, ...] = tuple(ALL_BLEND_ALIAS_CLIQUES.keys())


def all_blend_group_indices(
    group: str,
    *,
    with_common: bool = True,
) -> list[int]:
    """Column indices of ``group`` in an all_blend matrix (core [+ common])."""
    _ensure_cache()
    assert _CACHED_NAMES is not None and _CACHED_PLAN is not None
    assert _CACHED_COMMON_NAMES is not None
    g = (group or "").strip().lower()
    if g in ALL_BLEND_ALIAS_CLIQUES:
        names = all_blend_feature_names(with_common=with_common)
        want = set(ALL_BLEND_ALIAS_CLIQUES[g])
        return [i for i, n in enumerate(names) if n in want]
    n_core = len(_CACHED_NAMES)
    if g == "common":
        if not with_common:
            return []
        return list(range(n_core, n_core + len(_CACHED_COMMON_NAMES)))
    if g not in ALL_BLEND_SOURCES:
        raise ValueError(
            f"Unknown all_blend group {group!r}; expected one of "
            f"{ALL_BLEND_GROUPS + ALL_BLEND_ALIAS_GROUP_NAMES}"
        )
    return [i for i, (src, _) in enumerate(_CACHED_PLAN) if src == g]


def all_blend_group_channel_names(
    group: str,
    *,
    with_common: bool = True,
) -> list[str]:
    """Feature names belonging to ``group``."""
    _ensure_cache()
    assert _CACHED_NAMES is not None and _CACHED_PLAN is not None
    assert _CACHED_COMMON_NAMES is not None
    g = (group or "").strip().lower()
    if g in ALL_BLEND_ALIAS_CLIQUES:
        names = all_blend_feature_names(with_common=with_common)
        want = set(ALL_BLEND_ALIAS_CLIQUES[g])
        return [n for n in names if n in want]
    if g == "common":
        if not with_common:
            return []
        return list(_CACHED_COMMON_NAMES)
    return [
        _CACHED_NAMES[i] for i, (src, _) in enumerate(_CACHED_PLAN) if src == g
    ]


def zero_all_blend_group(
    features: np.ndarray,
    group: str,
    *,
    with_common: bool = True,
) -> np.ndarray:
    """Return a copy of ``features`` with ``group`` columns set to 0."""
    idxs = all_blend_group_indices(group, with_common=with_common)
    if not idxs:
        return features
    out = np.array(features, dtype=np.float32, copy=True)
    out[:, idxs] = 0.0
    return out


def all_blend_feature_names(*, with_common: bool = True) -> list[str]:
    """Full all_blend feature name list (core [+ common])."""
    _ensure_cache()
    assert _CACHED_NAMES is not None and _CACHED_COMMON_NAMES is not None
    out = list(_CACHED_NAMES)
    if with_common:
        out.extend(_CACHED_COMMON_NAMES)
    return out


def all_blend_channel_catalog(*, with_common: bool = True) -> list[dict]:
    """Per-channel metadata: ``idx``, ``name``, ``group``."""
    _ensure_cache()
    assert _CACHED_NAMES is not None and _CACHED_PLAN is not None
    assert _CACHED_COMMON_NAMES is not None
    rows: list[dict] = []
    for i, ((src, _), name) in enumerate(zip(_CACHED_PLAN, _CACHED_NAMES)):
        rows.append({"idx": i, "name": str(name), "group": str(src)})
    if with_common:
        base = len(_CACHED_NAMES)
        for j, name in enumerate(_CACHED_COMMON_NAMES):
            rows.append({"idx": base + j, "name": str(name), "group": "common"})
    return rows


def zero_all_blend_channel(
    features: np.ndarray,
    channel: int | str,
    *,
    with_common: bool = True,
) -> np.ndarray:
    """Zero one column by index or exact feature name."""
    if isinstance(channel, int):
        idx = int(channel)
    else:
        names = all_blend_feature_names(with_common=with_common)
        key = str(channel)
        try:
            idx = names.index(key)
        except ValueError as exc:
            raise ValueError(f"Unknown all_blend channel {channel!r}") from exc
    if idx < 0 or idx >= features.shape[1]:
        raise IndexError(f"channel idx {idx} out of range for C={features.shape[1]}")
    out = np.array(features, dtype=np.float32, copy=True)
    out[:, idx] = 0.0
    return out


class AllBlendSTEngine(BaseSTEngine):
    """Concatenate unique core channels from every feature engine + filtered common."""

    name = "all_blend"
    # Embed common ourselves (with drops) so get_engine does not double-attach.
    _has_common_channels = True

    def __init__(self) -> None:
        _ensure_cache()
        assert _CACHED_NAMES is not None and _CACHED_PLAN is not None
        assert _CACHED_WARMUP is not None
        assert _CACHED_COMMON_NAMES is not None and _CACHED_COMMON_IDX is not None
        self.feature_names = list(_CACHED_NAMES) + list(_CACHED_COMMON_NAMES)
        self.warmup_bars = max(int(_CACHED_WARMUP), int(COMMON_WARMUP))
        self._plan = list(_CACHED_PLAN)
        self._common_idx = list(_CACHED_COMMON_IDX)
        self._n_core = len(_CACHED_NAMES)

    def compute(self, df: pd.DataFrame, **kwargs) -> EngineResult:
        n = len(df)
        if n == 0:
            return EngineResult(
                features=np.zeros((0, self.num_inputs), dtype=np.float32),
                valid_from=0,
            )

        by_src: dict[str, np.ndarray] = {}
        valid = 0
        for src in ALL_BLEND_SOURCES:
            eng = get_engine(src, attach_common=False)
            res = _call_compute(eng, df, **kwargs)
            feats = np.asarray(res.features, dtype=np.float32)
            if feats.ndim != 2 or feats.shape[0] != n:
                raise ValueError(
                    f"all_blend: {src} features {feats.shape} incompatible with n={n}"
                )
            by_src[src] = feats
            valid = max(valid, int(res.valid_from))

        cols: list[np.ndarray] = []
        for src, j in self._plan:
            cols.append(by_src[src][:, j])
        core = np.stack(cols, axis=1)
        core = np.nan_to_num(core, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        common_full = compute_common_channels(df)
        common = common_full[:, self._common_idx]
        stacked = np.concatenate([core, common], axis=1)
        if stacked.shape != (n, self.num_inputs):
            raise ValueError(
                f"all_blend: stacked {stacked.shape} != ({n}, {self.num_inputs})"
            )
        return EngineResult(
            features=stacked,
            valid_from=min(n, max(valid, self.warmup_bars)),
        )
