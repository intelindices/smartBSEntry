"""Feature matrices from ST engines + training datasets.

Engines only supply numbers. The Risk Manager owns entry, exit, and stops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from smartbs_engines.config import SmartBSConfig
from smartbs_engines.labels import select_barrier_train_indices
from smartbs_engines.registry import get_engine


def feature_names_for(
    engine: str | None,
    *,
    signal_engines: str | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    from smartbs_engines.engines import resolve_feature_engine

    name, _sources = resolve_feature_engine(engine, signal_engines)
    return list(get_engine(name).feature_names)


def num_inputs_for(
    engine: str | None,
    *,
    signal_engines: str | list[str] | tuple[str, ...] | None = None,
) -> int:
    return len(feature_names_for(engine, signal_engines=signal_engines))


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    feature_engine: str | None = None,
    symbol: str | None = None,
    data_source: str | None = None,
    signal_engines: str | list[str] | tuple[str, ...] | None = None,
    ablation_zero_group: str | None = None,
) -> np.ndarray:
    """Build ``(n, num_features)`` from 1H OHLCV via the selected engine."""
    from smartbs_engines.engines import resolve_feature_engine

    name, _sources = resolve_feature_engine(feature_engine, signal_engines)
    eng = get_engine(name)
    feats = eng.compute(df, symbol=symbol, data_source=data_source).features
    _ = ablation_zero_group
    return feats


def label_forward_returns(
    close: np.ndarray,
    horizon: int,
    threshold: float,
) -> np.ndarray:
    n = len(close)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n - horizon):
        fwd = close[i + horizon] / close[i] - 1.0
        if fwd > threshold:
            labels[i] = SmartBSConfig.CLASS_LONG
        elif fwd < -threshold:
            labels[i] = SmartBSConfig.CLASS_SHORT
        else:
            labels[i] = SmartBSConfig.CLASS_FLAT
    return labels


class CandleWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        lookback: int,
        end_idx: int | None = None,
        indices: list[int] | None = None,
    ):
        self.features = features
        self.labels = labels
        self.lookback = lookback
        self.end_idx = end_idx if end_idx is not None else len(features)
        if indices is not None:
            self.indices = list(indices)
        else:
            self.indices = list(range(lookback - 1, self.end_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        end = self.indices[idx]
        start = end - self.lookback + 1
        window = self.features[start : end + 1].T
        label = self.labels[end]
        return torch.from_numpy(window), torch.tensor(label, dtype=torch.long)


def _build_labels(df: pd.DataFrame, cfg: SmartBSConfig) -> np.ndarray:
    from smartbs_engines.labels import LABEL_PLUGINS, normalize_label_mode
    from smartbs_engines.plugins.base import ensure_plugins_loaded

    ensure_plugins_loaded()
    mode = normalize_label_mode(cfg.label_mode)
    return LABEL_PLUGINS.get(mode).compute(df, cfg)


def _session_trend_barrier_params(cfg: SmartBSConfig) -> tuple[float, int]:
    """Resolve session_trend: ±k*ATR first-touch before session end (default k=2)."""
    from smartbs_engines.labels import (
        DEFAULT_SESSION_TREND_HORIZON,
        DEFAULT_SESSION_TREND_K,
    )

    raw_k = float(getattr(cfg, "barrier_k", DEFAULT_SESSION_TREND_K) or DEFAULT_SESSION_TREND_K)
    # Config default barrier_k=1.0 → mode contract ±2*ATR.
    k = DEFAULT_SESSION_TREND_K if abs(raw_k - 1.0) < 1e-15 else raw_k
    return k, DEFAULT_SESSION_TREND_HORIZON


def _day_trend_pct(cfg: SmartBSConfig) -> float:
    """day_trend ±pct vs NY end (default 1%). barrier_pct=0.02 → use mode default."""
    from smartbs_engines.labels import DEFAULT_DAY_TREND_PCT

    raw = float(getattr(cfg, "barrier_pct", DEFAULT_DAY_TREND_PCT) or DEFAULT_DAY_TREND_PCT)
    return DEFAULT_DAY_TREND_PCT if abs(raw - 0.02) < 1e-15 else raw


def _rolle_breakout_params(cfg: SmartBSConfig) -> tuple[int, int]:
    """rolle_breakout: rolling window L = rolle_len; horizon forced to L.

    ``barrier_horizon`` is ignored for this mode.
    """
    from smartbs_engines.labels import rolle_len_of

    plen = rolle_len_of(cfg)
    return plen, plen


def _session_trend_resolved(df: pd.DataFrame, cfg: SmartBSConfig) -> np.ndarray:
    """Boolean mask: bars with a finished session-end outcome."""
    from smartbs_engines.labels import label_session_trend

    k, horizon = _session_trend_barrier_params(cfg)
    return label_session_trend(
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        df["open_time"].to_numpy(dtype=np.int64),
        k=k,
        horizon=horizon,
        session_hours=getattr(cfg, "session_hours", None),
    ).resolved


def _day_trend_resolved(df: pd.DataFrame, cfg: SmartBSConfig) -> np.ndarray:
    from smartbs_engines.labels import label_day_trend

    return label_day_trend(
        df["close"].to_numpy(dtype=np.float64),
        df["open_time"].to_numpy(dtype=np.int64),
        pct=_day_trend_pct(cfg),
        session_hours=getattr(cfg, "session_hours", None),
    ).resolved


def _rolle_breakout_resolved(df: pd.DataFrame, cfg: SmartBSConfig) -> np.ndarray:
    from smartbs_engines.labels import label_rolle_breakout

    plen, hor = _rolle_breakout_params(cfg)
    return label_rolle_breakout(
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        rolle_len=plen,
        horizon=hor,
    ).resolved


def make_datasets(df: pd.DataFrame, cfg: SmartBSConfig, *, symbol: str | None = None):
    from smartbs_engines.engines import resolve_feature_engine
    from smartbs_engines.labels import normalize_label_mode

    engine, _sources = resolve_feature_engine(
        getattr(cfg, "feature_engine", "maribbon"),
        getattr(cfg, "signal_engines", ()) or None,
    )
    sym = symbol or cfg.trade_pair
    label_mode = normalize_label_mode(cfg.label_mode)
    features = build_feature_matrix(
        df,
        feature_engine=engine,
        symbol=sym,
        data_source=cfg.data_source,
    )
    labels = _build_labels(df, cfg)

    horizon_tail = 0
    if label_mode == "forward_return":
        horizon_tail = int(cfg.horizon)
    elif label_mode == "session_trend":
        _, horizon_tail = _session_trend_barrier_params(cfg)
    elif label_mode == "day_trend":
        # NY end is same UTC day; resolved mask drops unfinished days.
        horizon_tail = 0
    elif label_mode == "rolle_breakout":
        _, horizon_tail = _rolle_breakout_params(cfg)
    usable = len(df) - horizon_tail
    usable = max(usable, cfg.lookback)
    split = int(usable * (1.0 - cfg.val_ratio))
    split = max(split, cfg.lookback)

    train_idxs = list(range(cfg.lookback - 1, split))
    val_idxs = list(range(split, usable))

    if label_mode in (
        "triple_barrier",
        "session_trend",
        "day_trend",
        "rolle_breakout",
    ):
        warm = max(get_engine(engine).warmup_bars, cfg.lookback - 1)
        if label_mode in ("session_trend", "day_trend", "rolle_breakout"):
            # usable already excludes horizon_tail; resolved mask drops unfinished bars.
            tail = usable
        else:
            tail = usable - cfg.barrier_horizon
        cand = [i for i in train_idxs if warm <= i < tail]
        val_idxs = [i for i in val_idxs if warm <= i < tail]
        if label_mode == "session_trend":
            resolved = _session_trend_resolved(df, cfg)
            cand = [i for i in cand if resolved[i]]
            val_idxs = [i for i in val_idxs if resolved[i]]
        elif label_mode == "day_trend":
            resolved = _day_trend_resolved(df, cfg)
            cand = [i for i in cand if resolved[i]]
            val_idxs = [i for i in val_idxs if resolved[i]]
        elif label_mode == "rolle_breakout":
            resolved = _rolle_breakout_resolved(df, cfg)
            cand = [i for i in cand if resolved[i]]
            val_idxs = [i for i in val_idxs if resolved[i]]
        train_idxs = select_barrier_train_indices(cand, labels, seed=cfg.seed)
    elif label_mode != "forward_return":
        raise ValueError(
            f"label_mode={label_mode!r} unknown; use triple_barrier / "
            "session_trend / day_trend / rolle_breakout / forward_return"
        )

    train_ds = CandleWindowDataset(
        features, labels, cfg.lookback, end_idx=split, indices=train_idxs
    )
    val_ds = CandleWindowDataset(
        features, labels, cfg.lookback, end_idx=usable, indices=val_idxs
    )
    return train_ds, val_ds, features, labels


def build_1h_to_15m_end_index(
    times_1h: np.ndarray,
    times_15m: np.ndarray,
    *,
    hour_ms: int = 3_600_000,
) -> np.ndarray:
    """For each 1h bar, index of the last 15m bar with ``open_time < hour_end``."""
    t1 = np.asarray(times_1h, dtype=np.int64)
    t15 = np.asarray(times_15m, dtype=np.int64)
    n = len(t1)
    out = np.full(n, -1, dtype=np.int64)
    if len(t15) == 0:
        return out
    j = 0
    n15 = len(t15)
    for i in range(n):
        end = int(t1[i]) + int(hour_ms)
        while j + 1 < n15 and int(t15[j + 1]) < end:
            j += 1
        if int(t15[j]) < end:
            out[i] = j
        elif j > 0 and int(t15[j - 1]) < end:
            out[i] = j - 1
    return out


class DualWindowDataset(Dataset):
    """Paired 1h + 15m windows labeled on the 1h bar."""

    def __init__(
        self,
        features_1h: np.ndarray,
        features_15m: np.ndarray,
        labels: np.ndarray,
        lookback_1h: int,
        lookback_15m: int,
        end15_of_1h: np.ndarray,
        indices: list[int] | None = None,
    ):
        self.features_1h = features_1h
        self.features_15m = features_15m
        self.labels = labels
        self.lookback_1h = int(lookback_1h)
        self.lookback_15m = int(lookback_15m)
        self.end15_of_1h = np.asarray(end15_of_1h, dtype=np.int64)
        if indices is not None:
            self.indices = [
                i
                for i in indices
                if i >= self.lookback_1h - 1
                and 0 <= int(self.end15_of_1h[i])
                and int(self.end15_of_1h[i]) >= self.lookback_15m - 1
            ]
        else:
            self.indices = [
                i
                for i in range(self.lookback_1h - 1, len(features_1h))
                if 0 <= int(self.end15_of_1h[i])
                and int(self.end15_of_1h[i]) >= self.lookback_15m - 1
            ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        end1 = self.indices[idx]
        start1 = end1 - self.lookback_1h + 1
        end15 = int(self.end15_of_1h[end1])
        start15 = end15 - self.lookback_15m + 1
        w1 = self.features_1h[start1 : end1 + 1].T
        w15 = self.features_15m[start15 : end15 + 1].T
        label = self.labels[end1]
        return (torch.from_numpy(w1), torch.from_numpy(w15)), torch.tensor(
            label, dtype=torch.long
        )


def make_dual_datasets(
    df_1h: pd.DataFrame,
    cfg: SmartBSConfig,
    *,
    symbol: str | None = None,
    df_15m: pd.DataFrame | None = None,
):
    """Build DualWindowDataset train/val for ``smartBSDualTF``."""
    from smartbs_engines.engines import resolve_feature_engine
    from smartbs_engines.labels import normalize_label_mode
    from smartbs_engines.smart_money_structure import load_aligned_15m

    engine, _sources = resolve_feature_engine(
        getattr(cfg, "feature_engine", "maribbon"),
        getattr(cfg, "signal_engines", ()) or None,
    )
    sym = symbol or cfg.trade_pair
    label_mode = normalize_label_mode(cfg.label_mode)
    if df_15m is None:
        df_15m = load_aligned_15m(df_1h, symbol=sym, data_source=cfg.data_source)
    if df_15m is None or len(df_15m) < 64:
        raise RuntimeError(f"dual_tf: no aligned 15m data for {sym}")

    feats_1h = build_feature_matrix(
        df_1h,
        feature_engine=engine,
        symbol=sym,
        data_source=cfg.data_source,
    )
    feats_15m = build_feature_matrix(
        df_15m,
        feature_engine=engine,
        symbol=sym,
        data_source=cfg.data_source,
    )
    labels = _build_labels(df_1h, cfg)
    end15 = build_1h_to_15m_end_index(
        df_1h["open_time"].to_numpy(dtype=np.int64),
        df_15m["open_time"].to_numpy(dtype=np.int64),
    )

    lb1 = int(cfg.lookback)
    ratio = int(getattr(cfg, "tf_ratio", 4) or 4)
    lb15 = int(getattr(cfg, "lookback_15m", 0) or 0) or lb1 * ratio

    horizon_tail = 0
    if label_mode == "forward_return":
        horizon_tail = int(cfg.horizon)
    elif label_mode == "session_trend":
        _, horizon_tail = _session_trend_barrier_params(cfg)
    elif label_mode == "day_trend":
        horizon_tail = 0
    elif label_mode == "rolle_breakout":
        _, horizon_tail = _rolle_breakout_params(cfg)
    usable = len(df_1h) - horizon_tail
    usable = max(usable, lb1)
    split = int(usable * (1.0 - cfg.val_ratio))
    split = max(split, lb1)

    train_idxs = list(range(lb1 - 1, split))
    val_idxs = list(range(split, usable))

    if label_mode in (
        "triple_barrier",
        "session_trend",
        "day_trend",
        "rolle_breakout",
    ):
        warm = max(get_engine(engine).warmup_bars, lb1 - 1)
        if label_mode in ("session_trend", "day_trend", "rolle_breakout"):
            tail = usable
        else:
            tail = usable - cfg.barrier_horizon
        cand = [i for i in train_idxs if warm <= i < tail]
        val_idxs = [i for i in val_idxs if warm <= i < tail]
        if label_mode == "session_trend":
            resolved = _session_trend_resolved(df_1h, cfg)
            cand = [i for i in cand if resolved[i]]
            val_idxs = [i for i in val_idxs if resolved[i]]
        elif label_mode == "day_trend":
            resolved = _day_trend_resolved(df_1h, cfg)
            cand = [i for i in cand if resolved[i]]
            val_idxs = [i for i in val_idxs if resolved[i]]
        elif label_mode == "rolle_breakout":
            resolved = _rolle_breakout_resolved(df_1h, cfg)
            cand = [i for i in cand if resolved[i]]
            val_idxs = [i for i in val_idxs if resolved[i]]
        train_idxs = select_barrier_train_indices(cand, labels, seed=cfg.seed)
    elif label_mode != "forward_return":
        raise ValueError(
            f"label_mode={label_mode!r} unknown; use triple_barrier / "
            "session_trend / day_trend / rolle_breakout / forward_return"
        )

    train_ds = DualWindowDataset(
        feats_1h, feats_15m, labels, lb1, lb15, end15, indices=train_idxs
    )
    val_ds = DualWindowDataset(
        feats_1h, feats_15m, labels, lb1, lb15, end15, indices=val_idxs
    )
    return train_ds, val_ds, feats_1h, feats_15m, labels, end15


def make_multi_asset_datasets(
    asset_frames: list[tuple[pd.DataFrame, str]],
    cfg: SmartBSConfig,
):
    train_parts = []
    val_parts = []
    all_labels = []
    for df, _pair in asset_frames:
        if df is None or len(df) < cfg.lookback + 50:
            continue
        train_ds, val_ds, _, labels = make_datasets(df, cfg, symbol=_pair)
        if len(train_ds):
            train_parts.append(train_ds)
        if len(val_ds):
            val_parts.append(val_ds)
        all_labels.append(labels)
    if not train_parts:
        raise RuntimeError("No usable asset frames for multi-asset training")
    train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    val_ds = ConcatDataset(val_parts) if len(val_parts) > 1 else val_parts[0]
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    return train_ds, val_ds, labels
