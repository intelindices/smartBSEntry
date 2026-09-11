"""Train SmartBS ST-engine AI (TCN) and save a checkpoint."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from smartbs_engines.calibrate import fetch_training_frame, fit_model_calibration
from smartbs_engines.checkpoint import checkpoint_spec_fields, promote_checkpoint
from smartbs_engines.config import PAIR_ALIASES, SmartBSConfig, resolve_checkpoint_dir
from smartbs_engines.features import make_datasets, make_multi_asset_datasets, num_inputs_for
from smartbs_engines.model import SmartBSClassifier
from smartbs_engines.stdio_compat import configure_stdio

configure_stdio()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(cfg: SmartBSConfig) -> str:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assets = [a.replace("/", "").upper() for a in (cfg.train_assets or [cfg.trade_pair])]
    if cfg.trade_pair.replace("/", "").upper() not in assets:
        assets = [cfg.trade_pair.replace("/", "").upper()] + assets

    print(f"Multi-asset train set: {assets}")
    print(f"Feature engine: {cfg.feature_engine} ({cfg.num_inputs} inputs)")
    frames: list[tuple[pd.DataFrame, str]] = []
    primary_df = None
    for sym in assets:
        try:
            df = fetch_training_frame(sym, cfg)
        except Exception as e:
            print(f"  skip {sym}: {e}")
            continue
        if len(df) < cfg.lookback + 50:
            print(f"  skip {sym}: only {len(df)} bars")
            continue
        t0 = pd.to_datetime(df["open_time"].iloc[0], unit="ms", utc=True)
        t1 = pd.to_datetime(df["open_time"].iloc[-1], unit="ms", utc=True)
        print(f"  {sym}: {len(df)}x1H | {t0.date()} -> {t1.date()}")
        frames.append((df, sym))
        if sym == cfg.trade_pair.replace("/", "").upper():
            primary_df = df

    if not frames:
        raise RuntimeError("No asset data available for training")

    if len(frames) == 1:
        train_ds, val_ds, _, labels = make_datasets(frames[0][0], cfg)
    else:
        train_ds, val_ds, labels = make_multi_asset_datasets(frames, cfg)

    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")
    train_ys = [int(train_ds[i][1].item()) for i in range(len(train_ds))]
    counts = np.bincount(np.asarray(train_ys, dtype=np.int64), minlength=3)
    print(f"Train label counts (FLAT/LONG/SHORT): {counts.tolist()}")

    class_counts = np.maximum(counts.astype(np.float64), 1.0)
    weights = (class_counts.sum() / (len(class_counts) * class_counts)).astype(np.float32)
    weights[counts == 0] = 1.0
    weights = np.minimum(weights, float(cfg.max_class_weight)).astype(np.float32)
    print(f"Class weights (capped<={cfg.max_class_weight}): {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, max(len(train_ds), 1)),
        shuffle=True,
        drop_last=len(train_ds) >= cfg.batch_size * 2,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SmartBSClassifier(
        num_inputs=cfg.num_inputs,
        num_channels=cfg.num_channels,
        num_classes=cfg.num_classes,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = int(cfg.early_stop_patience)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = train_correct = train_n = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_n += y.size(0)

        model.eval()
        val_loss = val_correct = val_n = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_n += y.size(0)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs}  "
            f"train_loss={train_loss/max(train_n,1):.4f} acc={train_correct/max(train_n,1):.3f}  "
            f"val_loss={val_loss/max(val_n,1):.4f} acc={val_correct/max(val_n,1):.3f}"
        )

        if val_loss / max(val_n, 1) < best_val - 1e-6:
            best_val = val_loss / max(val_n, 1)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(cfg.early_stop_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is None:
        best_state = model.state_dict()
    else:
        model.load_state_dict(best_state)

    primary_frame = frames[0] if frames else None
    cal = fit_model_calibration(model, cfg, val_ds, primary_frame, device)

    checkpoint_path = cfg.resolved_checkpoint()
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    ref_df = primary_df if primary_df is not None else frames[0][0]
    base_config = {
        "trade_pair": cfg.trade_pair,
        "data_source": cfg.data_source,
        "tv_symbol": cfg.tv_symbol,
        "tv_exchange": cfg.tv_exchange,
        "binance_symbol": cfg.binance_symbol,
        "interval": cfg.interval,
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "return_threshold": cfg.return_threshold,
        "num_inputs": cfg.num_inputs,
        "num_channels": cfg.num_channels,
        "kernel_size": cfg.kernel_size,
        "dropout": cfg.dropout,
        "num_classes": cfg.num_classes,
        "leverage": float(cfg.leverage),
        "label_mode": cfg.label_mode,
        "feature_engine": cfg.feature_engine,
        **cal.to_config_patch(),
        "train_candles": cfg.train_candles,
        "train_assets": assets,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val) if best_val < float("inf") else None,
        "data_bars": len(ref_df),
        "data_from_ms": int(ref_df["open_time"].iloc[0]),
        "data_to_ms": int(ref_df["open_time"].iloc[-1]),
    }
    saved = []
    for sym in assets:
        defaults = SmartBSConfig.asset_data_defaults(sym)
        pair_cfg = dict(base_config)
        pair_cfg.update(defaults)
        pair_cfg["leverage"] = float(cfg.leverage)
        pair_cfg.update(checkpoint_spec_fields(cfg.feature_engine))
        pair_cfg_path = SmartBSConfig(
            trade_pair=sym,
            feature_engine=cfg.feature_engine,
            checkpoint_path="",
        ).resolved_checkpoint()
        path = pair_cfg_path
        _install({"model_state": best_state, "config": pair_cfg}, path)
        saved.append(path)
    if checkpoint_path not in saved:
        base_config.update(checkpoint_spec_fields(cfg.feature_engine))
        _install({"model_state": best_state, "config": base_config}, checkpoint_path)
    return saved[0] if saved else checkpoint_path


def _install(payload: dict, path: str) -> None:
    res = promote_checkpoint(payload, path)
    if not res.promoted:
        raise RuntimeError(f"Refused to install checkpoint at {path}: {res.reason}")
    note = f" (previous kept at {os.path.basename(res.backup)})" if res.backup else ""
    print(f"Saved checkpoint -> {path}{note}")


def parse_args() -> SmartBSConfig:
    parser = argparse.ArgumentParser(description="Train SmartBS ST-engine AI (TCN)")
    parser.add_argument("--trade-pair", default="XAUUSD")
    parser.add_argument("--interval", default="1h")
    parser.add_argument(
        "--data-source",
        default="dukascopy",
        choices=["tradingview", "binance", "yahoo", "dukascopy"],
    )
    parser.add_argument("--tv-symbol", default="XAUUSD")
    parser.add_argument("--tv-exchange", default="OANDA")
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-candles", type=int, default=12000)
    parser.add_argument("--holdout-days", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--label-mode", choices=["triple_barrier", "forward_return"], default="triple_barrier")
    parser.add_argument("--train-assets", default="")
    parser.add_argument("--leverage", type=float, default=0.1, help="Lot size stamped into checkpoint metadata")
    from smartbs_engines.registry import BLEND_ENGINES, list_engines
    parser.add_argument(
        "--feature-engine",
        default="maribbon",
        choices=list(BLEND_ENGINES),
        help="ST engine to train (not entry — use smartbs_entry for that)",
    )
    args = parser.parse_args()

    trade_pair = PAIR_ALIASES.get(args.trade_pair.replace("/", "").upper(), args.trade_pair.replace("/", "").upper())
    if not str(args.train_assets).strip():
        assets = [trade_pair]
    else:
        assets = SmartBSConfig.normalize_trade_pairs(args.train_assets)
        if trade_pair not in assets:
            assets = [trade_pair] + assets
    eng = args.feature_engine
    cfg = SmartBSConfig(
        trade_pair=trade_pair if trade_pair in assets else assets[0],
        interval=args.interval,
        data_source=args.data_source,
        tv_symbol=args.tv_symbol.upper(),
        tv_exchange=args.tv_exchange.upper(),
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_candles=args.train_candles,
        holdout_days=args.holdout_days,
        checkpoint_path=args.checkpoint,
        label_mode=args.label_mode,
        train_assets=assets,
        feature_engine=eng,
        num_inputs=num_inputs_for(eng),
        leverage=float(args.leverage),
    )
    defaults = SmartBSConfig.asset_data_defaults(cfg.trade_pair)
    cfg.tv_symbol = defaults["tv_symbol"]
    cfg.tv_exchange = defaults["tv_exchange"]
    cfg.binance_symbol = defaults["binance_symbol"]
    return cfg


if __name__ == "__main__":
    train_model(parse_args())


def main() -> None:
    train_model(parse_args())
