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
from smartbs_engines.features import make_datasets, make_dual_datasets, make_multi_asset_datasets, num_inputs_for
from smartbs_engines.model import (
    BACKBONE_SMARTBS_DUAL_TF,
    build_classifier,
    default_kernel_size,
    model_forward,
    normalize_backbone,
)
from smartbs_engines.pipeline import PipelineSpec
from smartbs_engines.stdio_compat import configure_stdio

configure_stdio()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(cfg: SmartBSConfig) -> str:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu (install CUDA torch to use GPU: pip install torch --index-url https://download.pytorch.org/whl/cu128)")

    assets = [a.replace("/", "").upper() for a in (cfg.train_assets or [cfg.trade_pair])]
    if cfg.trade_pair.replace("/", "").upper() not in assets:
        assets = [cfg.trade_pair.replace("/", "").upper()] + assets

    from smartbs_engines.common_channels import (
        get_session_hours_mode,
        set_session_hours_mode,
    )

    sess_mode = set_session_hours_mode(getattr(cfg, "session_hours", "normal"))
    cfg.session_hours = sess_mode

    backbone = normalize_backbone(getattr(cfg, "backbone", None))
    kernel_size = int(cfg.kernel_size)
    # Remap classic default k=3 to backbone-specific defaults when unset by caller.
    if kernel_size == 3 and backbone not in ("tcn", BACKBONE_SMARTBS_DUAL_TF):
        kernel_size = default_kernel_size(backbone)
    cfg.backbone = backbone
    cfg.kernel_size = kernel_size
    if backbone == BACKBONE_SMARTBS_DUAL_TF:
        ratio = int(getattr(cfg, "tf_ratio", 4) or 4)
        if int(getattr(cfg, "lookback_15m", 0) or 0) <= 0:
            cfg.lookback_15m = int(cfg.lookback) * ratio
        if int(getattr(cfg, "num_inputs_15m", 0) or 0) <= 0:
            cfg.num_inputs_15m = int(cfg.num_inputs)

    print(f"Multi-asset train set: {assets}")
    print(f"Feature engine: {cfg.feature_engine} ({cfg.num_inputs} inputs)")
    print(f"Session hours: {get_session_hours_mode()} (UTC)")
    if str(cfg.feature_engine).lower() == "signals":
        src = tuple(getattr(cfg, "signal_engines", ()) or ())
        print(f"Signal sources: {','.join(src) if src else 'all(pool)'}")
    print(f"Backbone: {backbone} (kernel={kernel_size})")
    if backbone == BACKBONE_SMARTBS_DUAL_TF:
        print(
            f"DualTF: lookback_1h={cfg.lookback} lookback_15m={cfg.lookback_15m} "
            f"C15={cfg.num_inputs_15m} ratio={cfg.tf_ratio}"
        )
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
        print(f"  {sym}: {len(df)}x{cfg.interval} | {t0.date()} -> {t1.date()}")
        frames.append((df, sym))
        if sym == cfg.trade_pair.replace("/", "").upper():
            primary_df = df

    if not frames:
        raise RuntimeError("No asset data available for training")

    if len(frames) == 1:
        if backbone == BACKBONE_SMARTBS_DUAL_TF:
            train_ds, val_ds, _, _, labels, _ = make_dual_datasets(frames[0][0], cfg)
        else:
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

    model = build_classifier(
        num_inputs=cfg.num_inputs,
        num_channels=cfg.num_channels,
        num_classes=cfg.num_classes,
        kernel_size=kernel_size,
        dropout=cfg.dropout,
        backbone=backbone,
        num_inputs_15m=int(getattr(cfg, "num_inputs_15m", 0) or 0) or None,
        tf_ratio=int(getattr(cfg, "tf_ratio", 4) or 4),
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
            y = y.to(device)
            optimizer.zero_grad()
            logits = model_forward(model, x, device)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_n += y.size(0)

        model.eval()
        val_loss = val_correct = val_n = 0.0
        # Directional precision: among pred LONG/SHORT, share with label==pred.
        dir_hit = dir_n = 0.0
        both_hit = both_n = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                y = y.to(device)
                logits = model_forward(model, x, device)
                loss = criterion(logits, y)
                pred = logits.argmax(dim=1)
                val_loss += loss.item() * y.size(0)
                val_correct += (pred == y).sum().item()
                val_n += y.size(0)
                dir_mask = pred != 0
                if dir_mask.any():
                    dir_hit += (pred[dir_mask] == y[dir_mask]).sum().item()
                    dir_n += int(dir_mask.sum().item())
                both = dir_mask & (y != 0)
                if both.any():
                    both_hit += (pred[both] == y[both]).sum().item()
                    both_n += int(both.sum().item())

        dir_prec = dir_hit / dir_n if dir_n > 0 else float("nan")
        dir_both = both_hit / both_n if both_n > 0 else float("nan")
        print(
            f"Epoch {epoch:02d}/{cfg.epochs}  "
            f"train_loss={train_loss/max(train_n,1):.4f} acc={train_correct/max(train_n,1):.3f}  "
            f"val_loss={val_loss/max(val_n,1):.4f} acc={val_correct/max(val_n,1):.3f}  "
            f"dir_prec={dir_prec:.3f} dir_both={dir_both:.3f}"
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
        "lookback_15m": int(getattr(cfg, "lookback_15m", 0) or 0),
        "tf_ratio": int(getattr(cfg, "tf_ratio", 4) or 4),
        "horizon": cfg.horizon,
        "return_threshold": cfg.return_threshold,
        "ma_len": int(getattr(cfg, "ma_len", 14) or 14),
        "barrier_k": float(getattr(cfg, "barrier_k", 1.0) or 1.0),
        "barrier_horizon": int(getattr(cfg, "barrier_horizon", 4) or 4),
        "barrier_pct": float(getattr(cfg, "barrier_pct", 0.02) or 0.02),
        "pivot_len": int(getattr(cfg, "pivot_len", 5) or 5),
        "ablation_zero_group": str(getattr(cfg, "ablation_zero_group", "") or ""),
        "num_inputs": cfg.num_inputs,
        "num_inputs_15m": int(getattr(cfg, "num_inputs_15m", 0) or 0),
        "num_channels": cfg.num_channels,
        "kernel_size": kernel_size,
        "dropout": cfg.dropout,
        "num_classes": cfg.num_classes,
        "backbone": backbone,
        "leverage": float(cfg.leverage),
        "label_mode": cfg.label_mode,
        "session_hours": str(getattr(cfg, "session_hours", "normal") or "normal"),
        **PipelineSpec.from_config(cfg).to_checkpoint_fields(),
        **cal.to_config_patch(),
        "train_candles": cfg.train_candles,
        "train_years": float(getattr(cfg, "train_years", 0.0) or 0.0),
        "train_from_date": str(getattr(cfg, "train_from_date", "") or ""),
        "train_align_15m": bool(getattr(cfg, "train_align_15m", True)),
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
        pair_cfg.update(
            checkpoint_spec_fields(
                cfg.feature_engine,
                signal_engines=getattr(cfg, "signal_engines", ()) or None,
            )
        )
        pair_cfg_path = SmartBSConfig(
            trade_pair=sym,
            feature_engine=cfg.feature_engine,
            checkpoint_path="",
        ).resolved_checkpoint()
        path = pair_cfg_path
        _install({"model_state": best_state, "config": pair_cfg}, path)
        saved.append(path)
    if checkpoint_path not in saved:
        base_config.update(
            checkpoint_spec_fields(
                cfg.feature_engine,
                signal_engines=getattr(cfg, "signal_engines", ()) or None,
            )
        )
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
        default="mt5",
        choices=["tradingview", "binance", "yahoo", "dukascopy", "mt5"],
    )
    parser.add_argument("--tv-symbol", default="XAUUSD")
    parser.add_argument("--tv-exchange", default="OANDA")
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-candles", type=int, default=0)
    parser.add_argument(
        "--train-years",
        type=float,
        default=10.0,
        help="1h only: unified calendar years back from series end (0=off)",
    )
    parser.add_argument(
        "--train-from-date",
        default="",
        help="1h only: UTC inclusive train start YYYY-MM-DD (empty=off; then 10y + 1h∩15m)",
    )
    parser.add_argument(
        "--train-align-15m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="1h only: also clip to first available 15m bar (default on)",
    )
    parser.add_argument("--holdout-days", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--label-mode",
        choices=[
            "triple_barrier",
            "pct_barrier",
            "session_direction",
            "session_trend",
            "day_trend",
            "pivot_breakout",
            "forward_return",
            "next_direction",
        ],
        default="triple_barrier",
    )
    parser.add_argument("--ma-len", type=int, default=14, help="MA+AI replay gate SMA length")
    parser.add_argument(
        "--barrier-pct",
        type=float,
        default=0.02,
        help="pct_barrier ±fraction; session_direction / day_trend override",
    )
    parser.add_argument(
        "--barrier-k",
        type=float,
        default=1.0,
        help="triple_barrier ATR multiple; session_trend ±k*ATR (default k=1)",
    )
    parser.add_argument(
        "--barrier-horizon",
        type=int,
        default=4,
        help="triple_barrier / pct_barrier / pivot_breakout horizon "
        "(pivot_breakout default 24 when left at 4)",
    )
    parser.add_argument(
        "--pivot-len",
        type=int,
        default=5,
        help="pivot_breakout lookback L/R (default 5)",
    )
    parser.add_argument(
        "--session-hours",
        choices=["normal", "adjusted"],
        default="normal",
        help="UTC session windows: normal 0-9/7-16/12-21 (default) | adjusted 0-7/7-12/12-20",
    )
    parser.add_argument("--train-assets", default="")
    parser.add_argument("--leverage", type=float, default=0.1, help="Lot size stamped into checkpoint metadata")
    parser.add_argument(
        "--feature-engine",
        default="maribbon",
        help="ST engine: maribbon|dbb|…|signals|signals:macd,rsi_divergence",
    )
    parser.add_argument(
        "--signal-engines",
        default="",
        help="When feature-engine=signals: comma/+ list of sources "
        "(default=ACTIVE_SIGNAL_SOURCES). Ex: rsi_divergence,macd,smart_money",
    )
    parser.add_argument(
        "--backbone",
        default="tcn",
        help="Model backbone: tcn | smartBSEntryV2 | smartBSTF | smartBSDualTF",
    )
    parser.add_argument(
        "--signal-policy",
        default="none",
        help="SignalDecision policy: none | onset_side | ma_decay | (custom registered)",
    )
    parser.add_argument(
        "--raw-ai-strategy",
        default="ai_only",
        help="raw_ai combine: ai_only | signal_gate | signal_prior",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        help="Conv kernel (default: 3 tcn / 7 V2 / 1 TF unused)",
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
    from smartbs_engines.engines import resolve_feature_engine
    from smartbs_engines.pipeline import PipelineSpec
    from smartbs_engines.signals import normalize_signal_sources

    eng_name, sig_sources = resolve_feature_engine(
        eng, str(args.signal_engines).strip() or None
    )
    if eng_name == "signals":
        sig_tuple = normalize_signal_sources(sig_sources)
    else:
        sig_tuple = ()
    backbone = normalize_backbone(args.backbone)
    if args.kernel_size is not None:
        kernel_size = int(args.kernel_size)
    else:
        kernel_size = default_kernel_size(backbone)
    pipe = PipelineSpec(
        engines=sig_tuple if eng_name == "signals" else (eng_name,),
        signal_policy=str(args.signal_policy),
        backbone=backbone,
        raw_ai=str(args.raw_ai_strategy),
    )
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
        train_years=float(args.train_years),
        train_from_date=str(args.train_from_date or ""),
        train_align_15m=bool(args.train_align_15m),
        holdout_days=args.holdout_days,
        checkpoint_path=args.checkpoint,
        label_mode=args.label_mode,
        ma_len=int(args.ma_len),
        barrier_pct=float(args.barrier_pct),
        barrier_k=float(args.barrier_k),
        barrier_horizon=int(args.barrier_horizon),
        pivot_len=int(args.pivot_len),
        session_hours=str(args.session_hours),
        train_assets=assets,
        feature_engine=eng_name,
        signal_engines=sig_tuple,
        signal_policy=pipe.signal_policy,
        raw_ai_strategy=pipe.raw_ai,
        signal_point_gate=pipe.raw_ai == "signal_gate",
        backbone=backbone,
        kernel_size=kernel_size,
        num_inputs=num_inputs_for(eng_name, signal_engines=sig_tuple or None),
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
