"""Temperature / meta calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from smartbs_entry.calibration import (
    collect_logits,
    fit_meta_threshold,
    fit_temperature,
    softmax_np,
)
from smartbs_entry.config import SmartBSConfig
from smartbs_entry.data import TRAIN_ASSET_SPECS, fetch_mtf_klines


def trim_holdout_tail(df_1h: pd.DataFrame, holdout_days: int) -> pd.DataFrame:
    if df_1h is None or len(df_1h) == 0 or holdout_days <= 0:
        return df_1h
    cutoff = int(df_1h["open_time"].iloc[-1]) - int(holdout_days) * 86_400_000
    kept = df_1h[df_1h["open_time"] < cutoff].reset_index(drop=True)
    if len(kept) == 0:
        raise ValueError(
            f"holdout_days={holdout_days} consumed the entire {len(df_1h)}-bar training window"
        )
    print(
        f"Reserved last {holdout_days}d for replay: training on {len(kept)}/{len(df_1h)} 1H bars"
    )
    return kept


def fetch_training_frame(symbol: str, cfg: SmartBSConfig) -> pd.DataFrame:
    spec = TRAIN_ASSET_SPECS.get(
        symbol.upper(),
        {
            "symbol": symbol,
            "source": cfg.data_source if cfg.data_source != "tradingview" or cfg.train_candles <= 5000 else "yahoo",
            "exchange": cfg.tv_exchange,
        },
    )
    source = cfg.data_source if getattr(cfg, "data_source", None) else spec.get("source", "yahoo")
    if source == "tradingview" and cfg.train_candles > 5000:
        source = "yahoo"
    df_1h = fetch_mtf_klines(
        symbol=spec.get("symbol", symbol),
        source=source,
        exchange=spec.get("exchange", "OANDA"),
        max_1h_candles=cfg.train_candles,
        binance_symbol=spec.get("binance_symbol"),
    )
    return trim_holdout_tail(df_1h, int(getattr(cfg, "holdout_days", 0) or 0))


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
