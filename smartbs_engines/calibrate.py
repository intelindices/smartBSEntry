"""Temperature / meta calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from smartbs_engines.calibration import (
    collect_logits,
    fit_meta_threshold,
    fit_temperature,
    softmax_np,
)
from smartbs_engines.config import SmartBSConfig
from smartbs_engines.data import TRAIN_ASSET_SPECS, fetch_mtf_klines

_MS_PER_DAY = 86_400_000


def _is_1h_interval(interval: str | None) -> bool:
    tag = str(interval or "1h").strip().lower()
    return tag in ("1h", "60m", "60", "h1")


def trim_train_years(df_1h: pd.DataFrame, years: float) -> pd.DataFrame:
    if df_1h is None or len(df_1h) == 0 or years is None or float(years) <= 0.0:
        return df_1h
    end_ms = int(df_1h["open_time"].iloc[-1])
    start_ms = end_ms - int(float(years) * 365.25 * _MS_PER_DAY)
    cache_from = int(df_1h["open_time"].iloc[0])
    kept = df_1h[df_1h["open_time"] >= start_ms].reset_index(drop=True)
    if len(kept) == 0:
        raise ValueError(f"train_years={years} left zero bars")
    t0 = pd.to_datetime(int(kept["open_time"].iloc[0]), unit="ms", utc=True).date()
    t1 = pd.to_datetime(end_ms, unit="ms", utc=True).date()
    clipped = " (cache shorter than window)" if cache_from > start_ms else ""
    print(f"Unified train window {float(years):g}y: {t0} -> {t1} ({len(kept)} bars){clipped}")
    return kept


def trim_from_date(df_1h: pd.DataFrame, date_str: str) -> pd.DataFrame:
    if df_1h is None or len(df_1h) == 0:
        return df_1h
    tag = str(date_str or "").strip()
    if not tag:
        return df_1h
    start_ms = int(pd.Timestamp(tag, tz="UTC").timestamp() * 1000)
    before = len(df_1h)
    kept = df_1h[df_1h["open_time"] >= start_ms].reset_index(drop=True)
    if len(kept) == 0:
        raise ValueError(f"train_from_date={tag} left zero bars")
    t0 = pd.to_datetime(int(kept["open_time"].iloc[0]), unit="ms", utc=True).date()
    t1 = pd.to_datetime(int(kept["open_time"].iloc[-1]), unit="ms", utc=True).date()
    print(f"Train from date {tag}: {t0} -> {t1} ({len(kept)}/{before} bars)")
    return kept


def trim_to_15m_start(
    df_1h: pd.DataFrame,
    symbol: str,
    *,
    data_source: str,
) -> pd.DataFrame:
    """Clip 1h train series to first available 15m bar (parity with dBB / SM)."""
    if df_1h is None or len(df_1h) == 0:
        return df_1h
    src = (data_source or "").lower()
    sym = symbol.replace("/", "").upper()
    m15 = None
    try:
        if src in ("mt5", "metatrader", "broker"):
            from smartbs_engines.mt5_data import load_mt5_klines

            m15 = load_mt5_klines(sym, "15m", max_candles=None, refresh=False)
        elif src in ("dukascopy", "duka"):
            from smartbs_engines.dukascopy import load_dukascopy_klines

            m15 = load_dukascopy_klines(sym, interval="15m", max_candles=None)
    except Exception as exc:
        print(f"15m align skip ({sym}): {exc}")
        return df_1h
    if m15 is None or len(m15) == 0:
        print(f"15m align skip ({sym}): no 15m cache")
        return df_1h
    start_ms = int(m15["open_time"].iloc[0])
    before = len(df_1h)
    kept = df_1h[df_1h["open_time"] >= start_ms].reset_index(drop=True)
    if len(kept) == 0:
        raise ValueError(f"15m align left zero 1h bars for {sym}")
    t0 = pd.to_datetime(int(kept["open_time"].iloc[0]), unit="ms", utc=True).date()
    t1 = pd.to_datetime(int(kept["open_time"].iloc[-1]), unit="ms", utc=True).date()
    m15_0 = pd.to_datetime(start_ms, unit="ms", utc=True).date()
    print(
        f"Aligned to 15m start {m15_0}: 1h train {t0} -> {t1} ({len(kept)}/{before} bars)"
    )
    return kept


def trim_holdout_tail(df_1h: pd.DataFrame, holdout_days: int) -> pd.DataFrame:
    if df_1h is None or len(df_1h) == 0 or holdout_days <= 0:
        return df_1h
    cutoff = int(df_1h["open_time"].iloc[-1]) - int(holdout_days) * _MS_PER_DAY
    kept = df_1h[df_1h["open_time"] < cutoff].reset_index(drop=True)
    if len(kept) == 0:
        raise ValueError(
            f"holdout_days={holdout_days} consumed the entire {len(df_1h)}-bar training window"
        )
    t0 = pd.to_datetime(int(kept["open_time"].iloc[0]), unit="ms", utc=True).date()
    t1 = pd.to_datetime(int(kept["open_time"].iloc[-1]), unit="ms", utc=True).date()
    print(
        f"Reserved last {holdout_days}d for replay: training on {len(kept)}/{len(df_1h)} bars "
        f"({t0} -> {t1})"
    )
    return kept


def fetch_training_frame(symbol: str, cfg: SmartBSConfig) -> pd.DataFrame:
    spec = TRAIN_ASSET_SPECS.get(
        symbol.upper(),
        {
            "symbol": symbol,
            "source": cfg.data_source
            if cfg.data_source != "tradingview" or cfg.train_candles <= 5000
            else "yahoo",
            "exchange": cfg.tv_exchange,
        },
    )
    source = cfg.data_source if getattr(cfg, "data_source", None) else spec.get("source", "yahoo")
    if source == "tradingview" and cfg.train_candles > 5000:
        source = "yahoo"
    interval = getattr(cfg, "interval", None) or "1h"
    years = float(getattr(cfg, "train_years", 0.0) or 0.0)
    max_bars = cfg.train_candles
    if _is_1h_interval(interval) and years > 0.0:
        max_bars = 0
    df = fetch_mtf_klines(
        symbol=spec.get("symbol", symbol),
        source=source,
        exchange=spec.get("exchange", "OANDA"),
        max_1h_candles=max_bars,
        binance_symbol=spec.get("binance_symbol"),
    )
    if _is_1h_interval(interval) and years > 0.0:
        df = trim_train_years(df, years)
    from_date = str(getattr(cfg, "train_from_date", "") or "").strip()
    if _is_1h_interval(interval) and from_date:
        df = trim_from_date(df, from_date)
    if _is_1h_interval(interval) and bool(getattr(cfg, "train_align_15m", True)):
        df = trim_to_15m_start(df, str(spec.get("symbol", symbol)), data_source=str(source))
    return trim_holdout_tail(df, int(getattr(cfg, "holdout_days", 0) or 0))


@dataclass
class CalibrationResult:
    temperature: float = 1.0
    meta_threshold: float = 0.33
    meta_stats: dict[str, float] = field(default_factory=dict)

    def to_config_patch(self) -> dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "meta_threshold": float(self.meta_threshold),
            "meta_stats": {k: float(v) for k, v in self.meta_stats.items()},
        }


def fit_model_calibration(
    model: torch.nn.Module,
    cfg: SmartBSConfig,
    val_ds,
    primary_frame: tuple | None,
    device: torch.device,
) -> CalibrationResult:
    _ = primary_frame
    out = CalibrationResult(meta_threshold=float(getattr(cfg, "confidence_threshold", 0.0) or 0.33))
    model.eval()

    if len(val_ds) < 8:
        print("Calibration skipped (val set too small); T=1.0")
        return out

    cal_loader = DataLoader(val_ds, batch_size=min(256, max(len(val_ds), 1)), shuffle=False)
    val_logits, val_y = collect_logits(model, cal_loader, device)
    out.temperature = fit_temperature(val_logits, val_y)
    cal_probs = softmax_np(val_logits, out.temperature)
    out.meta_threshold, out.meta_stats = fit_meta_threshold(cal_probs, val_y)
    print(
        f"Calibration: T={out.temperature:.3f} meta_th={out.meta_threshold:.3f} "
        f"val_n={len(val_y)} stats={out.meta_stats}"
    )
    return out
