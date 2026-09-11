"""SmartMoneySTEngine v2 — 1h mean-reversion context + 15m structure.

22 channels:
  - 1h: dealing-range MR, swing bias, fused setup scores
  - 15m: BOS/ChoCH decay, FVG proximity, sweeps (end-of-hour snapshot onto 1h)

When 15m data is unavailable (tests, missing cache), 15m channels fall back to
structure computed on 1h with a shorter swing length so train/live paths stay
compatible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbs_engines.smart_money_structure import (
    compute_structure_block,
    load_aligned_15m,
    map_15m_end_of_hour_to_1h,
    prev_array,
)
from smartbs_engines.registry import BaseSTEngine, EngineResult, safe_div
from smartbs_engines.structure import atr as _atr, swing_levels

SWING_LEN_1H = 9
SWING_LEN_15M = 5
RANGE_WINDOW = 50
DECAY_CAP = 50

MR_CHANNELS: tuple[str, ...] = (
    "struct_bias",
    "range_pos",
    "mr_stretch",
    "dist_sh",
    "dist_sl",
    "discount",
    "premium",
    "setup_long",
    "setup_short",
    "bars_since_sh",
)

M15_CHANNELS: tuple[str, ...] = (
    "m15_bos_up_decay",
    "m15_bos_dn_decay",
    "m15_choch_up_decay",
    "m15_choch_dn_decay",
    "m15_fvg_up_sz",
    "m15_fvg_dn_sz",
    "m15_dist_fvg_up",
    "m15_dist_fvg_dn",
    "m15_sweep_up",
    "m15_sweep_dn",
    "m15_dist_sh",
    "m15_dist_sl",
)


def _bars_since_level_change(level: np.ndarray, cap: int = DECAY_CAP) -> np.ndarray:
    n = len(level)
    out = np.full(n, float(cap), dtype=np.float64)
    last = -1
    for i in range(n):
        if i > 0 and level[i] != level[i - 1]:
            last = i
        out[i] = float(cap) if last < 0 else min(float(i - last), float(cap))
    return out / float(cap)


def _mr_block(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
) -> dict[str, np.ndarray]:
    sh, sl = swing_levels(high, low, SWING_LEN_1H)
    sh = np.nan_to_num(sh, nan=high[0])
    sl = np.nan_to_num(sl, nan=low[0])
    p_sh, p_sl = prev_array(sh), prev_array(sl)

    rng_hi = pd.Series(high).rolling(RANGE_WINDOW, min_periods=1).max().to_numpy(dtype=np.float64)
    rng_lo = pd.Series(low).rolling(RANGE_WINDOW, min_periods=1).min().to_numpy(dtype=np.float64)
    span = rng_hi - rng_lo
    range_pos = np.clip(safe_div(close - rng_lo, span), 0.0, 1.0)
    mr_stretch = np.clip(2.0 * (range_pos - 0.5), -1.0, 1.0)
    discount = np.clip(0.5 - range_pos, 0.0, 0.5) * 2.0
    premium = np.clip(range_pos - 0.5, 0.0, 0.5) * 2.0

    hh = (sh > p_sh).astype(np.float64)
    hl = (sl > p_sl).astype(np.float64)
    struct_bias = pd.Series(hh + hl - 1.0).rolling(20, min_periods=1).mean().to_numpy(dtype=np.float64)

    return {
        "struct_bias": struct_bias,
        "range_pos": range_pos,
        "mr_stretch": mr_stretch,
        "dist_sh": safe_div(close - sh, atr14),
        "dist_sl": safe_div(close - sl, atr14),
        "discount": discount,
        "premium": premium,
        "bars_since_sh": _bars_since_level_change(sh),
    }


def _map_m15_to_1h(
    times_1h: np.ndarray,
    df_15m: pd.DataFrame,
    block: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    t15 = df_15m["open_time"].to_numpy(dtype=np.int64)
    return {
        key: map_15m_end_of_hour_to_1h(times_1h, t15, block[key])
        for key in block
    }


class SmartMoneySTEngine(BaseSTEngine):
    name = "smart_money"
    feature_names = list(MR_CHANNELS) + list(M15_CHANNELS)
    warmup_bars = 3 * max(RANGE_WINDOW, SWING_LEN_1H * 4, SWING_LEN_15M * 16)

    def compute(
        self,
        df: pd.DataFrame,
        *,
        df_15m: pd.DataFrame | None = None,
        symbol: str | None = None,
        data_source: str | None = None,
    ) -> EngineResult:
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        n = len(close)
        times_1h = df["open_time"].to_numpy(dtype=np.int64)
        atr14 = _atr(high, low, close, 14)

        mr = _mr_block(high=high, low=low, close=close, atr14=atr14)

        if df_15m is None and symbol:
            df_15m = load_aligned_15m(df, symbol=symbol, data_source=data_source)

        if df_15m is not None and len(df_15m) >= SWING_LEN_15M * 3:
            h15 = df_15m["high"].to_numpy(dtype=np.float64)
            l15 = df_15m["low"].to_numpy(dtype=np.float64)
            c15 = df_15m["close"].to_numpy(dtype=np.float64)
            atr15 = _atr(h15, l15, c15, 14)
            block15 = compute_structure_block(
                high=h15, low=l15, close=c15, atr14=atr15, swing_len=SWING_LEN_15M
            )
            mapped = _map_m15_to_1h(times_1h, df_15m, block15)
            m15 = {f"m15_{k}": v for k, v in mapped.items()}
        else:
            block_fb = compute_structure_block(
                high=high, low=low, close=close, atr14=atr14, swing_len=SWING_LEN_1H
            )
            m15 = {f"m15_{k}": v for k, v in block_fb.items()}

        setup_long = np.clip(
            m15["m15_choch_up_decay"] * mr["discount"] * m15["m15_fvg_prox_up"],
            0.0,
            1.5,
        )
        setup_short = np.clip(
            m15["m15_choch_dn_decay"] * mr["premium"] * m15["m15_fvg_prox_dn"],
            0.0,
            1.5,
        )
        mr["setup_long"] = setup_long
        mr["setup_short"] = setup_short

        cols = [mr[c] for c in MR_CHANNELS] + [m15[c] for c in M15_CHANNELS]
        return self._finalize(cols, n)
