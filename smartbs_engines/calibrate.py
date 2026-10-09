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


def _is_5m_interval(interval: str | None) -> bool:
    tag = str(interval or "").strip().lower()
    return tag in ("5m", "5min", "m5")


def _is_15m_interval(interval: str | None) -> bool:
    tag = str(interval or "").strip().lower()
    return tag in ("15m", "15min", "m15")


def _is_4h_interval(interval: str | None) -> bool:
    tag = str(interval or "").strip().lower()
    return tag in ("4h", "240m", "h4")


def _is_train_window_interval(interval: str | None) -> bool:
    """Intervals that use train_years / from_date / 1h∩15m align."""
    return (
        _is_1h_interval(interval)
        or _is_15m_interval(interval)
        or _is_4h_interval(interval)
        or _is_5m_interval(interval)
    )


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


def _load_tf_cache(symbol: str, interval: str, data_source: str):
    src = (data_source or "").lower()
    sym = symbol.replace("/", "").upper()
    if src in ("mt5", "metatrader", "broker"):
        from smartbs_engines.mt5_data import load_mt5_klines

        return load_mt5_klines(sym, interval, max_candles=None, refresh=False)
    if src in ("dukascopy", "duka"):
        from smartbs_engines.dukascopy import load_dukascopy_klines

        return load_dukascopy_klines(sym, interval=interval, max_candles=None)
    return None


def trim_to_15m_start(
    df_1h: pd.DataFrame,
    symbol: str,
    *,
    data_source: str,
) -> pd.DataFrame:
    """Clip primary series to the first bar where **both** 1h and 15m exist.

    Start = max(first_1h, first_15m). Works for 1h or 15m primary so both
    TFs share the same calendar window ("same dataset").
    """
    if df_1h is None or len(df_1h) == 0:
        return df_1h
    sym = symbol.replace("/", "").upper()
    try:
        h1 = _load_tf_cache(sym, "1h", data_source)
        m15 = _load_tf_cache(sym, "15m", data_source)
    except Exception as exc:
        print(f"1h∩15m align skip ({sym}): {exc}")
        return df_1h
    if h1 is None or len(h1) == 0 or m15 is None or len(m15) == 0:
        print(f"1h∩15m align skip ({sym}): missing 1h or 15m cache")
        return df_1h
    h1_0 = int(h1["open_time"].iloc[0])
    m15_0 = int(m15["open_time"].iloc[0])
    start_ms = max(h1_0, m15_0, int(df_1h["open_time"].iloc[0]))
    before = len(df_1h)
    kept = df_1h[df_1h["open_time"] >= start_ms].reset_index(drop=True)
    if len(kept) == 0:
        raise ValueError(f"1h∩15m align left zero bars for {sym}")
    t0 = pd.to_datetime(int(kept["open_time"].iloc[0]), unit="ms", utc=True).date()
    t1 = pd.to_datetime(int(kept["open_time"].iloc[-1]), unit="ms", utc=True).date()
    h1_d = pd.to_datetime(h1_0, unit="ms", utc=True).date()
    m15_d = pd.to_datetime(m15_0, unit="ms", utc=True).date()
    print(
        f"Aligned to both 1h+15m (1h from {h1_d}, 15m from {m15_d}): "
        f"train {t0} -> {t1} ({len(kept)}/{before} bars)"
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
    use_years_window = _is_train_window_interval(interval) and years > 0.0
    if use_years_window:
        max_bars = 0
    sym = spec.get("symbol", symbol)
    if _is_15m_interval(interval) or _is_4h_interval(interval) or _is_5m_interval(interval):
        from smartbs_engines.data import fetch_klines

        if _is_5m_interval(interval):
            iv = "5m"
        elif _is_15m_interval(interval):
            iv = "15m"
        else:
            iv = "4h"
        print(f"Fetching {iv} series ({max_bars} bars) from {source}...")
        df = fetch_klines(
            symbol=sym,
            interval=iv,
            max_candles=max_bars,
            source=source,
            exchange=spec.get("exchange", "OANDA"),
            binance_symbol=spec.get("binance_symbol"),
        )
    else:
        df = fetch_mtf_klines(
            symbol=sym,
            source=source,
            exchange=spec.get("exchange", "OANDA"),
            max_1h_candles=max_bars,
            binance_symbol=spec.get("binance_symbol"),
        )
    if use_years_window:
        df = trim_train_years(df, years)
    from_date = str(getattr(cfg, "train_from_date", "") or "").strip()
    if _is_train_window_interval(interval) and from_date:
        df = trim_from_date(df, from_date)
    if _is_train_window_interval(interval) and bool(
        getattr(cfg, "train_align_15m", True)
    ):
        df = trim_to_15m_start(df, str(sym), data_source=str(source))
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
