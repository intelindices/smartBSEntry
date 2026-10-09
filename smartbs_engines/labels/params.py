"""Shared label-mode parameter resolution (avoids features↔labels cycles)."""

from __future__ import annotations

from smartbs_engines.labels.core import (
    DEFAULT_DAY_TREND_PCT,
    DEFAULT_SESSION_TREND_HORIZON,
    DEFAULT_SESSION_TREND_K,
    rolle_len_of,
)


def session_trend_barrier_params(cfg) -> tuple[float, int]:
    raw_k = float(getattr(cfg, "barrier_k", DEFAULT_SESSION_TREND_K) or DEFAULT_SESSION_TREND_K)
    k = DEFAULT_SESSION_TREND_K if abs(raw_k - 1.0) < 1e-15 else raw_k
    return k, DEFAULT_SESSION_TREND_HORIZON


def day_trend_pct(cfg) -> float:
    raw = float(getattr(cfg, "barrier_pct", DEFAULT_DAY_TREND_PCT) or DEFAULT_DAY_TREND_PCT)
    return DEFAULT_DAY_TREND_PCT if abs(raw - 0.02) < 1e-15 else raw


def rolle_breakout_params(cfg) -> tuple[int, int]:
    plen = rolle_len_of(cfg)
    return plen, plen
