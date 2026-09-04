"""Feature matrices from SmartBSEntryEngine + training datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from smartbs_entry.config import SmartBSConfig
from smartbs_entry.labels import select_barrier_train_indices
from smartbs_entry.registry import get_engine, normalize_feature_engine


def feature_names_for(engine: str | None = "entry") -> list[str]:
    return list(get_engine(normalize_feature_engine(engine)).feature_names)


def num_inputs_for(engine: str | None = "entry") -> int:
    return len(feature_names_for(engine))


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    feature_engine: str | None = "entry",
    symbol: str | None = None,
    data_source: str | None = None,
) -> np.ndarray:
    """Build ``(n, num_features)`` from 1H OHLCV via the entry engine."""
    name = normalize_feature_engine(feature_engine)
    eng = get_engine(name)
    return eng.compute(df, symbol=symbol, data_source=data_source).features


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
    mode = cfg.label_mode
    if mode == "triple_barrier":
        from smartbs_entry.labels import label_triple_barrier

        return label_triple_barrier(
            df["high"].to_numpy(dtype=np.float64),
            df["low"].to_numpy(dtype=np.float64),
            df["close"].to_numpy(dtype=np.float64),
            k_up=cfg.barrier_k,
            k_dn=cfg.barrier_k,
            horizon=cfg.barrier_horizon,
        ).labels
    if mode == "forward_return":
        return label_forward_returns(df["close"].to_numpy(), cfg.horizon, cfg.return_threshold)
    raise ValueError(f"label_mode={mode!r}; use triple_barrier or forward_return")


def make_datasets(df: pd.DataFrame, cfg: SmartBSConfig, *, symbol: str | None = None):
    engine = normalize_feature_engine(getattr(cfg, "feature_engine", "entry"))
    sym = symbol or cfg.trade_pair
    features = build_feature_matrix(
        df,
        feature_engine=engine,
        symbol=sym,
        data_source=cfg.data_source,
    )
    labels = _build_labels(df, cfg)

    usable = len(df) - (0 if cfg.label_mode != "forward_return" else cfg.horizon)
    usable = max(usable, cfg.lookback)
    split = int(usable * (1.0 - cfg.val_ratio))
    split = max(split, cfg.lookback)

    train_idxs = list(range(cfg.lookback - 1, split))
    val_idxs = list(range(split, usable))

    if cfg.label_mode == "triple_barrier":
        warm = max(get_engine(engine).warmup_bars, cfg.lookback - 1)
        tail = usable - cfg.barrier_horizon
        train_idxs = select_barrier_train_indices(
            [i for i in train_idxs if warm <= i < tail], labels, seed=cfg.seed
        )
        val_idxs = [i for i in val_idxs if warm <= i < tail]

    train_ds = CandleWindowDataset(
        features, labels, cfg.lookback, end_idx=split, indices=train_idxs
    )
    val_ds = CandleWindowDataset(
        features, labels, cfg.lookback, end_idx=usable, indices=val_idxs
    )
    return train_ds, val_ds, features, labels


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
