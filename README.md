# smartbs-entry

Standalone **SmartBS Entry engine + TCN AI** — extractable from Vanta Network for reuse in other projects.

## What this is

| Included | Excluded |
|----------|----------|
| `SmartBSEntryEngine` (90-channel multi-TF features) | Other ST engines (maribbon, bollinger, …) |
| TCN classifier (`SmartBSClassifier`) | Blend / mean voting |
| Train + temperature calibration | Risk Manager / position manager |
| Raw AI infer (`predict_raw_ai` = argmax) | Vanta miner REST / hydrate |
| Dukascopy / Yahoo / Binance / TV data loaders | Polygon / vali_objects |

Compatible with existing `entry/*.pt` checkpoints (`feature_engine=entry`, `num_inputs=90`).

## Install

```bash
cd packages/smartbs_entry   # or clone this repo root
pip install -e .
# optional feeds:
pip install -e ".[yahoo,tradingview]"
```

## Quick start — raw AI

```python
from smartbs_entry import predict_raw_ai, build_feature_matrix, load_classifier
from smartbs_entry.data import fetch_mtf_klines

df = fetch_mtf_klines("XAUUSD", source="dukascopy", max_1h_candles=8000)
out = predict_raw_ai(df, "path/to/XAUUSD.pt", symbol="XAUUSD", data_source="dukascopy")
print(out["signal"], out["probs"], out["confidence"])
```

## Train

```bash
export SMARTBS_CHECKPOINT_DIR=./checkpoints/entry
export SMARTBS_DUKASCOPY_DIR=/path/to/dukascopy/cache   # optional
python -m smartbs_entry.train --trade-pair XAUUSD --data-source dukascopy --feature-engine entry
```

## MetaTrader 5 Expert Advisor

Full fidelity port: **MQL5 features + ONNX TCN** (raw_ai). See [`mql5/README.md`](mql5/README.md).

```bash
pip install onnx onnxruntime onnxscript
python -m smartbs_entry.export_onnx \
  --checkpoint path/to/XAUUSD.pt \
  --out mql5/Models/XAUUSD.onnx
```

Copy `mql5/Experts`, `mql5/Include/SmartBSEntry`, and the `.onnx` into your MT5 data folder; attach `SmartBSEntry` to an **H1** chart.

## Package layout

```
smartbs_entry/
  engine.py       # SmartBSEntryEngine
  registry.py     # BaseSTEngine + entry-only registry
  indicators.py   # ema / atr / rsi / swings
  structure_sm.py # BOS / ChoCH / FVG primitives
  model.py        # TCN + SmartBSClassifier
  features.py     # build_feature_matrix + datasets
  labels.py       # triple-barrier labels
  config.py       # SmartBSConfig (AI knobs)
  train.py        # training CLI
  calibrate.py    # temperature fit
  checkpoint.py   # load / promote .pt
  predict.py      # predict_probs / predict_raw_ai
  export_onnx.py  # .pt → ONNX for MT5
  mql5_parity.py  # parity dump for EA Journal
  data.py         # OHLCV fetch
  dukascopy.py    # Dukascopy cache I/O
mql5/             # Expert Advisor + includes
```

## Version

See `smartbs_entry/VERSION` (aligned with SmartBS v0.9.7 at extraction).
