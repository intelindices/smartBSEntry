"""AI / train / blend config for SmartBS engines (no Entry, no Vanta miner/RM)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DEFAULT_BLEND_CHECKPOINT_ROOT = os.path.join(DEFAULT_CHECKPOINT_DIR, "blend")

PAIR_ALIASES = {
    "BTCUSDC": "BTCUSD",
    "ETHUSDC": "ETHUSD",
    "ADAUSDC": "ADAUSD",
    "LTCUSDC": "LTCUSD",
    "BCHUSDC": "BCHUSD",
    "GOLDUSDC": "XAUUSD",
    "SILVERUSDC": "XAGUSD",
    "WTIOILUSDC": "XTIUSD",
}

TRAINABLE_TRADE_PAIRS = (
    "XAUUSD",
    "XAGUSD",
    "XTIUSD",
    "COPPERUSDC",
    "NATGASUSDC",
    "PLATINUMUSDC",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCHF",
    "USDCAD",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "EURGBP",
    "BTCUSD",
    "ETHUSD",
    "ADAUSD",
    "LTCUSD",
    "BCHUSD",
)


def package_version() -> str:
    path = os.path.join(BASE_DIR, "VERSION")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip() or "unknown"
    except OSError:
        return "unknown"


smartbs_version = package_version


def resolve_checkpoint_dir() -> str:
    return os.environ.get("SMARTBS_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)


def resolve_blend_checkpoint_root() -> str:
    return os.environ.get(
        "SMARTBS_BLEND_CHECKPOINT_ROOT",
        os.environ.get("SMARTBS_CHECKPOINT_DIR", DEFAULT_BLEND_CHECKPOINT_ROOT),
    )


@dataclass
class SmartBSConfig:
    """Train / infer hyperparameters for ST-engine TCNs and blend."""

    trade_pair: str = "XAUUSD"
    data_source: str = "dukascopy"
    tv_symbol: str = "XAUUSD"
    tv_exchange: str = "OANDA"
    binance_symbol: str = ""
    interval: str = "1h"
    lookback: int = 64
    horizon: int = 4
    return_threshold: float = 0.0015
    label_mode: str = "triple_barrier"
    barrier_k: float = 2.0
    barrier_horizon: int = 24
    train_assets: List[str] = field(default_factory=lambda: ["XAUUSD"])

    feature_engine: str = "maribbon"
    num_inputs: int = 48
    num_channels: List[int] = field(default_factory=lambda: [32, 32, 48, 48])
    kernel_size: int = 3
    dropout: float = 0.25
    num_classes: int = 3

    batch_size: int = 64
    epochs: int = 15
    early_stop_patience: int = 3
    learning_rate: float = 5e-4
    train_candles: int = 12000
    holdout_days: int = 0
    val_ratio: float = 0.15
    seed: int = 42
    max_class_weight: float = 8.0

    leverage: float = 0.1
    confidence_threshold: float = 0.0
    checkpoint_path: str = ""
    blend_checkpoint_root: str = ""

    CLASS_FLAT: int = 0
    CLASS_LONG: int = 1
    CLASS_SHORT: int = 2

    def resolved_checkpoint(self) -> str:
        if self.checkpoint_path and not os.path.isdir(self.checkpoint_path):
            return self.checkpoint_path
        ckpt_dir = resolve_checkpoint_dir()
        os.makedirs(ckpt_dir, exist_ok=True)
        pair = PAIR_ALIASES.get(
            self.trade_pair.replace("/", "").upper(),
            self.trade_pair.replace("/", "").upper(),
        )
        eng = (self.feature_engine or "maribbon").strip().lower()
        # Prefer engine/PAIR.pt layout used by live blend roots.
        eng_dir = os.path.join(ckpt_dir, eng)
        if os.path.isdir(eng_dir) or self.feature_engine:
            os.makedirs(eng_dir, exist_ok=True)
            return os.path.join(eng_dir, f"{pair}.pt")
        return os.path.join(ckpt_dir, f"{pair}.pt")

    def resolved_blend_root(self) -> str:
        if self.blend_checkpoint_root:
            return self.blend_checkpoint_root
        return resolve_blend_checkpoint_root()

    @staticmethod
    def normalize_trade_pairs(raw: str | List[str] | None) -> List[str]:
        if raw is None or raw == "":
            return ["XAUUSD"]
        if isinstance(raw, str):
            parts = [p.strip().replace("/", "").upper() for p in raw.split(",") if p.strip()]
        else:
            parts = [str(p).replace("/", "").upper() for p in raw if str(p).strip()]
        out: List[str] = []
        for p in parts:
            p = PAIR_ALIASES.get(p, p)
            if p not in out:
                out.append(p)
        return out or ["XAUUSD"]

    @staticmethod
    def asset_data_defaults(trade_pair: str) -> dict:
        from smartbs_engines.data import TRAIN_ASSET_SPECS

        key = PAIR_ALIASES.get(
            trade_pair.replace("/", "").upper(), trade_pair.replace("/", "").upper()
        )
        spec = TRAIN_ASSET_SPECS.get(
            key, {"symbol": key, "source": "yahoo", "exchange": "BINANCE"}
        )
        return {
            "trade_pair": key,
            "data_source": spec.get("source", "yahoo"),
            "tv_symbol": spec.get("symbol", key),
            "tv_exchange": spec.get("exchange", "BINANCE"),
            "binance_symbol": spec.get("binance_symbol")
            or SmartBSConfig.trade_pair_to_binance(key),
        }

    @staticmethod
    def trade_pair_to_binance(trade_pair: str) -> str:
        mapping = {
            "BTCUSD": "BTCUSDT",
            "ETHUSD": "ETHUSDT",
            "ADAUSD": "ADAUSDT",
            "LTCUSD": "LTCUSDT",
            "BCHUSD": "BCHUSDT",
            "XAUUSD": "PAXGUSDT",
            "GOLDUSDC": "PAXGUSDT",
        }
        key = trade_pair.replace("/", "").upper()
        if key in mapping:
            return mapping[key]
        if key.endswith("USDC"):
            return key[:-4] + "USDT"
        return key
