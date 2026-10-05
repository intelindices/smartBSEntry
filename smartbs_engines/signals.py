"""Signal layer — named, modular emphasis signals between engines and AI.

Each source ST engine (any subset of the pool) is mapped
to exact reason ids such as ``rsi_divergence_long`` / ``macd_cross_up``. Soft
strengths in ``[0, 1]`` feed any backbone (tcn / V2 / smartBSTF / DualTF).

Default sources: ``dbb, rsi_divergence, trend_pullback, smart_money, macd``.
Optional ``signal_point_gate`` restricts entries to onsets of the engine-side
aggregates (``ENGINE_SIDE_REASON_IDS``) so LONG/SHORT only happen at a signal
point.

Compose freely::

    SignalsSTEngine()                           # ACTIVE_SIGNAL_SOURCES
    SignalsSTEngine(sources=(\"macd\",))        # single engine
    SignalsSTEngine(sources=(\"rsi_divergence\", \"smart_money\", \"macd\"))

Feature-engine strings::

    signals
    signals:rsi_divergence,macd
    signals:dbb+rsi_divergence+trend_pullback+smart_money+macd
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from smartbs_engines.engine_catalog import ACTIVE_SIGNAL_SOURCES
from smartbs_engines.registry import BaseSTEngine, EngineResult, get_engine

# Production signal pool (ST engines; common channels are not signal sources).
SIGNAL_POOL_ENGINES: tuple[str, ...] = (
    "regime_engine",
    "maribbon",
    "dbb",
    "trend_pullback",
    "smart_money",
    "candle",
    "macd",
    "rsi_divergence",
)

# Engine-level long/short reason ids — the "signal points" used for onset gating.
ENGINE_SIDE_REASON_IDS: tuple[str, ...] = (
    "regime_long",
    "regime_short",
    "dbb_long",
    "dbb_short",
    "rsi_divergence_long",
    "rsi_divergence_short",
    "trend_pullback_long",
    "trend_pullback_short",
    "smart_money_long",
    "smart_money_short",
    "macd_long",
    "macd_short",
)

FORBIDDEN_SIGNAL_ENGINES: frozenset[str] = frozenset(
    {"temp_engine", "tempengine", "tempe", "temp", "ohlcv"}
)
DEFAULT_SIGNAL_POINT_THR: float = 0.35


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def _idx(names: list[str], key: str) -> int:
    try:
        return names.index(key)
    except ValueError as exc:
        raise KeyError(f"channel {key!r} missing; have {names}") from exc


def _col(feats: np.ndarray, names: list[str], key: str) -> np.ndarray:
    return np.asarray(feats[:, _idx(names, key)], dtype=np.float64)


def _decay_as_strength(x: np.ndarray) -> np.ndarray:
    """Event decays are already in ~[0,1]; clip for safety."""
    return _clip01(x)


def _threshold_strength(x: np.ndarray, thr: float) -> np.ndarray:
    """Map continuous score to soft strength above threshold."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    above = x >= thr
    out[above] = _clip01((x[above] - thr) / max(1e-6, 1.0 - thr) * 0.5 + 0.5)
    # still pass partial credit below thr (scaled)
    below = ~above
    out[below] = _clip01(x[below] / max(thr, 1e-6) * 0.49)
    return out


@dataclass(frozen=True)
class SignalDef:
    """One named signal (exact reason id) produced by the signal layer."""

    signal_id: str
    engine: str
    side: int  # +1 long, -1 short, 0 context/neutral
    description: str = ""


Extractor = Callable[[np.ndarray, list[str]], dict[str, np.ndarray]]


def _extract_rsi_divergence(feats: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    cont_l = _decay_as_strength(_col(feats, names, "cont_long_decay"))
    cont_s = _decay_as_strength(_col(feats, names, "cont_short_decay"))
    rev_l = _decay_as_strength(_col(feats, names, "rev_long_decay"))
    rev_s = _decay_as_strength(_col(feats, names, "rev_short_decay"))
    bull_d = _decay_as_strength(_col(feats, names, "bull_div_decay"))
    bear_d = _decay_as_strength(_col(feats, names, "bear_div_decay"))
    sim_up = _decay_as_strength(_col(feats, names, "sim_xup_30_decay"))
    sim_dn = _decay_as_strength(_col(feats, names, "sim_xdn_70_decay"))
    return {
        "rsi_divergence_long": np.maximum.reduce([cont_l, rev_l, bull_d, sim_up]),
        "rsi_divergence_short": np.maximum.reduce([cont_s, rev_s, bear_d, sim_dn]),
    }


def _extract_smart_money(feats: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    setup_l = _clip01(_col(feats, names, "setup_long") / 1.5)
    setup_s = _clip01(_col(feats, names, "setup_short") / 1.5)
    bos_up = _decay_as_strength(_col(feats, names, "m15_bos_up_decay"))
    bos_dn = _decay_as_strength(_col(feats, names, "m15_bos_dn_decay"))
    choch_up = _decay_as_strength(_col(feats, names, "m15_choch_up_decay"))
    choch_dn = _decay_as_strength(_col(feats, names, "m15_choch_dn_decay"))
    return {
        "smart_money_long": np.maximum.reduce([setup_l, choch_up, bos_up]),
        "smart_money_short": np.maximum.reduce([setup_s, choch_dn, bos_dn]),
    }


def _extract_trend_pullback(feats: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    setup_l = _clip01(_col(feats, names, "setup_long") / 1.5)
    setup_s = _clip01(_col(feats, names, "setup_short") / 1.5)
    return {
        "trend_pullback_long": setup_l,
        "trend_pullback_short": setup_s,
    }


def _extract_macd(feats: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    cross_up = _decay_as_strength(_col(feats, names, "cross_up_decay"))
    cross_dn = _decay_as_strength(_col(feats, names, "cross_dn_decay"))
    persist = _clip01((_col(feats, names, "hist_persist") + 1.0) * 0.5)
    return {
        "macd_long": np.maximum(cross_up, persist),
        "macd_short": np.maximum(cross_dn, 1.0 - persist),
    }


def _extract_candle(feats: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    rej_l = _threshold_strength(_col(feats, names, "reject_bull"), 0.35)
    rej_s = _threshold_strength(_col(feats, names, "reject_bear"), 0.35)
    setup_l = _clip01(_col(feats, names, "setup_reject_long") / 1.5)
    setup_s = _clip01(_col(feats, names, "setup_reject_short") / 1.5)
    engulf_l = _clip01(_col(feats, names, "engulf_bull_score"))
    engulf_s = _clip01(_col(feats, names, "engulf_bear_score"))
    fail_up = _clip01(_col(feats, names, "failed_break_up") / 3.0)
    fail_dn = _clip01(_col(feats, names, "failed_break_dn") / 3.0)
    return {
        "candle_long": np.maximum.reduce([rej_l, setup_l, engulf_l, fail_dn]),
        "candle_short": np.maximum.reduce([rej_s, setup_s, engulf_s, fail_up]),
    }


def _extract_dbb(feats: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    lo_outer = _clip01(_col(feats, names, "close_in_lower_outer"))
    up_outer = _clip01(_col(feats, names, "close_in_upper_outer"))
    lo_outer_15 = _clip01(_col(feats, names, "close_in_lower_outer_15m"))
    up_outer_15 = _clip01(_col(feats, names, "close_in_upper_outer_15m"))
    return {
        "dbb_long": np.maximum(lo_outer, lo_outer_15),
        "dbb_short": np.maximum(up_outer, up_outer_15),
    }


def _extract_regime_engine(feats: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    up = _clip01(_col(feats, names, "regime_up"))
    dn = _clip01(_col(feats, names, "regime_dn"))
    return {
        "regime_long": up,
        "regime_short": dn,
    }


def _extract_maribbon(feats: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    # Stack bias from 1h direction/trend strength (±11) + r_point pulse.
    ds = _col(feats, names, "direction_strength")
    ts = _col(feats, names, "trend_strength")
    rp = _col(feats, names, "r_point")
    bull = np.maximum(ds, 0.0) + np.maximum(ts, 0.0) + np.maximum(rp, 0.0) * 11.0
    bear = np.maximum(-ds, 0.0) + np.maximum(-ts, 0.0) + np.maximum(-rp, 0.0) * 11.0
    return {
        "maribbon_long": _clip01(bull / 22.0),
        "maribbon_short": _clip01(bear / 22.0),
    }


_EXTRACTORS: dict[str, Extractor] = {
    "regime_engine": _extract_regime_engine,
    "rsi_divergence": _extract_rsi_divergence,
    "smart_money": _extract_smart_money,
    "trend_pullback": _extract_trend_pullback,
    "macd": _extract_macd,
    "candle": _extract_candle,
    "dbb": _extract_dbb,
    "maribbon": _extract_maribbon,
}


def normalize_signal_sources(
    sources: Sequence[str] | str | None = None,
) -> tuple[str, ...]:
    """Resolve engine list; empty/None/all → full pool. Rejects removed engines."""
    if sources is None or (isinstance(sources, str) and not sources.strip()):
        return ACTIVE_SIGNAL_SOURCES
    if isinstance(sources, str):
        raw = sources.strip().lower()
        if raw in ("all", "*", "pool"):
            return SIGNAL_POOL_ENGINES
        parts = [p.strip().lower() for p in raw.replace("+", ",").split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in sources if str(p).strip()]
    if not parts:
        return ACTIVE_SIGNAL_SOURCES
    out: list[str] = []
    for p in parts:
        if p in ("regimeengine", "regime"):
            p = "regime_engine"
        if p in FORBIDDEN_SIGNAL_ENGINES:
            raise ValueError(
                f"signal engine {p!r} was removed (use common channels on each ST engine). "
                f"Pool: {SIGNAL_POOL_ENGINES}"
            )
        if p not in _EXTRACTORS:
            raise ValueError(
                f"Unknown signal source {p!r}; known: {sorted(_EXTRACTORS)} "
                f"(pool default: {SIGNAL_POOL_ENGINES})"
            )
        if p not in out:
            out.append(p)
    return tuple(out)


def parse_signals_engine_name(name: str | None) -> tuple[str, tuple[str, ...] | None]:
    """Parse ``signals`` / ``signals:macd,rsi_divergence`` → (\"signals\", sources|None)."""
    raw = (name or "").strip().lower()
    if not raw.startswith("signals"):
        return raw, None
    if raw == "signals":
        return "signals", None
    if raw.startswith("signals:") or raw.startswith("signals="):
        body = raw.split(":", 1)[-1] if ":" in raw else raw.split("=", 1)[-1]
        return "signals", normalize_signal_sources(body)
    raise ValueError(f"Bad signals engine spec {name!r}; use signals or signals:eng1,eng2")


def signal_ids_for_sources(sources: Sequence[str] | None = None) -> list[str]:
    """Stable ordered signal ids for the given source engines."""
    srcs = normalize_signal_sources(sources)
    # Deterministic order: pool order, then signal keys sorted within each extractor
    # but keep extractor's insertion order (Python 3.7+ dict).
    ids: list[str] = []
    for eng in srcs:
        # Probe channel names via a tiny fake — better: call extractor on zeros with real names
        child = get_engine(eng, attach_common=False)
        n = 1
        fake = np.zeros((n, child.num_inputs), dtype=np.float32)
        got = _EXTRACTORS[eng](fake, list(child.feature_names))
        ids.extend(got.keys())
    return ids


def list_signal_catalog(sources: Sequence[str] | None = None) -> list[SignalDef]:
    """Human-readable catalog of signal reason ids."""
    srcs = normalize_signal_sources(sources)
    catalog: list[SignalDef] = []
    for eng in srcs:
        for sid in signal_ids_for_sources((eng,)):
            if sid.endswith("_long") or sid.endswith("_up") or "bull" in sid or sid.endswith("_discount"):
                side = 1
            elif sid.endswith("_short") or sid.endswith("_dn") or sid.endswith("_down") or "bear" in sid or sid.endswith("_premium"):
                side = -1
            else:
                side = 0
            catalog.append(SignalDef(signal_id=sid, engine=eng, side=side, description=sid))
    return catalog


def build_signal_matrix(
    df: pd.DataFrame,
    *,
    sources: Sequence[str] | None = None,
    symbol: str | None = None,
    data_source: str | None = None,
) -> tuple[np.ndarray, list[str], int]:
    """Compute ``(n, n_signals)`` float32 matrix + ids + warmup."""
    import inspect

    srcs = normalize_signal_sources(sources)
    n = len(df)
    parts: list[np.ndarray] = []
    ids: list[str] = []
    warmup = 0
    for eng_name in srcs:
        child = get_engine(eng_name, attach_common=False)
        warmup = max(warmup, int(child.warmup_bars))
        params = inspect.signature(child.compute).parameters
        kwargs: dict = {}
        if "symbol" in params:
            kwargs["symbol"] = symbol
        if "data_source" in params:
            kwargs["data_source"] = data_source
        result = child.compute(df, **kwargs)
        feats = np.asarray(result.features, dtype=np.float32)
        if feats.shape != (n, child.num_inputs):
            raise ValueError(
                f"signals/{eng_name}: got {feats.shape}, expected ({n}, {child.num_inputs})"
            )
        got = _EXTRACTORS[eng_name](feats, list(child.feature_names))
        for sid, col in got.items():
            ids.append(sid)
            parts.append(np.asarray(col, dtype=np.float32).reshape(n))
    if not parts:
        raise ValueError("signals: no source engines selected")
    stacked = np.nan_to_num(np.column_stack(parts), nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )
    return stacked, ids, warmup


class SignalsSTEngine(BaseSTEngine):
    """Modular signal bus: plug any subset of ``SIGNAL_POOL_ENGINES``."""

    name = "signals"

    def __init__(self, sources: Sequence[str] | str | None = None) -> None:
        self.sources = normalize_signal_sources(sources)
        self.feature_names = signal_ids_for_sources(self.sources)
        self.warmup_bars = max(
            int(get_engine(s, attach_common=False).warmup_bars) for s in self.sources
        )

    def compute(
        self,
        df: pd.DataFrame,
        *,
        symbol: str | None = None,
        data_source: str | None = None,
    ) -> EngineResult:
        stacked, ids, warmup = build_signal_matrix(
            df, sources=self.sources, symbol=symbol, data_source=data_source
        )
        if ids != self.feature_names:
            raise ValueError(
                f"signals: id drift {ids[:3]}… vs declared {self.feature_names[:3]}…"
            )
        n = len(df)
        if stacked.shape != (n, self.num_inputs):
            raise ValueError(f"signals: {stacked.shape} != ({n}, {self.num_inputs})")
        return EngineResult(features=stacked, valid_from=min(n, warmup))

    def reason_vector(self, row: np.ndarray, *, min_strength: float = 0.35) -> list[str]:
        """Active signal reason ids for one bar (for logs / EA attribution)."""
        return [
            sid
            for sid, v in zip(self.feature_names, np.asarray(row).ravel())
            if float(v) >= min_strength
        ]


def _side_of_reason(signal_id: str) -> int:
    """+1 long, -1 short, 0 neutral/context."""
    sid = str(signal_id).lower()
    if sid.endswith("_long") or sid.endswith("_up") or "bull" in sid or sid.endswith("_discount"):
        return 1
    if (
        sid.endswith("_short")
        or sid.endswith("_dn")
        or sid.endswith("_down")
        or "bear" in sid
        or sid.endswith("_premium")
    ):
        return -1
    return 0


def signal_point_onset(
    matrix: np.ndarray,
    signal_ids: Sequence[str],
    *,
    thr: float = DEFAULT_SIGNAL_POINT_THR,
    reason_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect directional signal-point onsets on the reason bus.

    A fire is ``strength[t] >= thr`` and ``strength[t-1] < thr`` on an
    engine-side aggregate (``dbb_long``, ``macd_short``, …).

    Returns
    -------
    fire_long, fire_short : bool (n,)
    primary_reason_idx : int (n,) index into ``signal_ids`` (-1 if none)
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"signal matrix must be (n, C), got {mat.shape}")
    n, c = mat.shape
    ids = [str(x) for x in signal_ids]
    if len(ids) != c:
        raise ValueError(f"signal_ids len {len(ids)} != channels {c}")

    want = set(reason_ids) if reason_ids is not None else set(ENGINE_SIDE_REASON_IDS)
    cols: list[tuple[int, int]] = []  # (col, side)
    for i, sid in enumerate(ids):
        if sid not in want:
            continue
        side = _side_of_reason(sid)
        if side != 0:
            cols.append((i, side))

    fire_long = np.zeros(n, dtype=bool)
    fire_short = np.zeros(n, dtype=bool)
    primary = np.full(n, -1, dtype=np.int64)
    if not cols or n == 0:
        return fire_long, fire_short, primary

    prev = np.vstack([np.zeros((1, c), dtype=np.float64), mat[:-1]])
    best_str = np.full(n, -1.0, dtype=np.float64)
    thr_f = float(thr)
    for col, side in cols:
        onset = (mat[:, col] >= thr_f) & (prev[:, col] < thr_f)
        if side > 0:
            fire_long |= onset
        else:
            fire_short |= onset
        stronger = onset & (mat[:, col] > best_str)
        primary = np.where(stronger, col, primary)
        best_str = np.where(stronger, mat[:, col], best_str)
    return fire_long, fire_short, primary


def apply_signal_point_gate(
    cls: np.ndarray,
    matrix: np.ndarray,
    signal_ids: Sequence[str],
    *,
    thr: float = DEFAULT_SIGNAL_POINT_THR,
    require_agree: bool = True,
    reason_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Force FLAT unless a directional signal-point fires this bar.

    When ``require_agree`` is True, AI LONG/SHORT must match the fire side
    (both fires → AI may pick either side). Returns ``(gated_cls, reason_idx)``.
    """
    raw = np.asarray(cls, dtype=np.int64).reshape(-1).copy()
    fire_long, fire_short, primary = signal_point_onset(
        matrix, signal_ids, thr=thr, reason_ids=reason_ids
    )
    n = len(raw)
    if len(fire_long) != n:
        raise ValueError(f"cls len {n} != matrix rows {len(fire_long)}")

    gated = np.zeros(n, dtype=np.int64)
    reason_idx = np.full(n, -1, dtype=np.int64)
    any_fire = fire_long | fire_short
    for i in range(n):
        if not any_fire[i]:
            continue
        ai = int(raw[i])
        fl, fs = bool(fire_long[i]), bool(fire_short[i])
        if require_agree:
            if fl and not fs:
                if ai == 1:
                    gated[i] = 1
                    reason_idx[i] = int(primary[i])
            elif fs and not fl:
                if ai == 2:
                    gated[i] = 2
                    reason_idx[i] = int(primary[i])
            else:
                # both sides onset — allow AI L/S only
                if ai in (1, 2):
                    gated[i] = ai
                    reason_idx[i] = int(primary[i])
        else:
            if ai in (1, 2):
                gated[i] = ai
                reason_idx[i] = int(primary[i])
            elif fl and not fs:
                gated[i] = 1
                reason_idx[i] = int(primary[i])
            elif fs and not fl:
                gated[i] = 2
                reason_idx[i] = int(primary[i])
    return gated, reason_idx
