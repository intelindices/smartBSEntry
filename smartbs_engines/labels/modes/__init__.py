"""Register live label modes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.labels import core as L
from smartbs_engines.labels.params import (
    day_trend_pct,
    rolle_breakout_params,
    session_trend_barrier_params,
)
from smartbs_engines.labels.plugins import LabelPlugin, register_label_plugin


def _triple(df: pd.DataFrame, cfg) -> np.ndarray:
    return L.label_triple_barrier(
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        k_up=cfg.barrier_k,
        k_dn=cfg.barrier_k,
        horizon=cfg.barrier_horizon,
    ).labels


def _session_trend(df: pd.DataFrame, cfg) -> np.ndarray:
    k, horizon = session_trend_barrier_params(cfg)
    return L.label_session_trend(
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        df["open_time"].to_numpy(dtype=np.int64),
        k=k,
        horizon=horizon,
        session_hours=getattr(cfg, "session_hours", None),
    ).labels


def _day_trend(df: pd.DataFrame, cfg) -> np.ndarray:
    return L.label_day_trend(
        df["close"].to_numpy(dtype=np.float64),
        df["open_time"].to_numpy(dtype=np.int64),
        pct=day_trend_pct(cfg),
        session_hours=getattr(cfg, "session_hours", None),
    ).labels


def _rolle(df: pd.DataFrame, cfg) -> np.ndarray:
    plen, hor = rolle_breakout_params(cfg)
    return L.label_rolle_breakout(
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        rolle_len=plen,
        horizon=hor,
    ).labels


def _forward(df: pd.DataFrame, cfg) -> np.ndarray:
    from smartbs_engines.config import SmartBSConfig

    close = df["close"].to_numpy(dtype=np.float64)
    horizon = int(cfg.horizon)
    threshold = float(cfg.return_threshold)
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


register_label_plugin(LabelPlugin("triple_barrier", _triple, is_barrier=True))
register_label_plugin(LabelPlugin("session_trend", _session_trend, is_barrier=True))
register_label_plugin(LabelPlugin("day_trend", _day_trend, is_barrier=True))
register_label_plugin(LabelPlugin("rolle_breakout", _rolle, is_barrier=True))
register_label_plugin(LabelPlugin("forward_return", _forward, is_barrier=False))
