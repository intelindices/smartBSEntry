# SmartBS Entry modularization report

**Date:** 2026-09-04  
**Source:** `vanta-network` / SmartBS v0.9.7  
**Extract location:** `packages/smartbs_entry/`  
**Intended new repo contents:** everything under `packages/smartbs_entry/` (this folder is self-contained)

---

## Goal

Make **SmartBSEntryEngine** + **SmartBS AI (TCN)** a reusable module with **no dependency** on other ST engines, blend, Risk Manager, or Vanta miner REST — so it can be committed to a separate GitHub repository.

---

## What was extracted

### Public API (`import smartbs_entry`)

| Symbol | Role |
|--------|------|
| `SmartBSEntryEngine` | 90-channel multi-TF feature engine |
| `FEATURE_NAMES` / `TF_PAIRS` / `PAIR_CHANNELS` | Feature contract |
| `SmartBSClassifier` / `TemporalConvNet` | AI backbone (`self.tcn` key preserved for .pt compat) |
| `SmartBSConfig` | Train/infer knobs (default `feature_engine=entry`, `num_inputs=90`) |
| `build_feature_matrix` | OHLCV → `(n, 90)` |
| `load_classifier` | Load `.pt` + verify `feature_spec_hash` |
| `predict_probs` / `predict_raw_ai` | Softmax probs / argmax FLAT·LONG·SHORT (no arm / R / day-DD) |
| `train_model` (`python -m smartbs_entry.train`) | Train + calibrate + promote checkpoint |

### Geometry (unchanged)

- **6 TF pairs** × **15 channels** = **90 inputs**
- Warmup **7200** 1h bars; lookback **64**; TCN channels `[32,32,48,48]`
- Spec hash must match live engine or load fails (same as miner)

### Explicitly excluded

- maribbon / bollinger / candle / macd / smart_money / trend_pullback / rsi_divergence
- Blend mean voting, `ACTIVE_BLEND_ENGINES`
- Risk Manager, position manager, raw_arm MA gates, day-DD flatten
- Vanta `submit-order`, hydrate, miner profile / PM2 scripts
- Polygon / `vali_objects` data path

---

## How to publish as a new GitHub repo

```bash
# From vanta-network root
cd packages/smartbs_entry
git init
git add .
git commit -m "Initial smartbs-entry package (Entry engine + TCN AI v0.9.7)"
# create empty repo on GitHub, then:
git remote add origin git@github.com:<org>/<repo>.git
git push -u origin main
```

Optional: copy existing checkpoints separately (often gitignored / large):

```bash
# example — do not commit secrets; checkpoints are large binaries
mkdir -p checkpoints/entry
cp ../../mining/smartbs_strategy/checkpoints/dukascopy_10y/entry/*.pt checkpoints/entry/
```

Dukascopy cache: set `SMARTBS_DUKASCOPY_DIR` to your parquet cache, or run `python -m smartbs_entry.dukascopy`.

---

## Install / smoke

```bash
cd packages/smartbs_entry
pip install -e .
python -c "from smartbs_entry import SmartBSEntryEngine, predict_raw_ai, __version__; print(__version__, SmartBSEntryEngine().num_inputs)"
```

Expected: `0.9.7 90`

---

## Relation to live miner (`mining/smartbs_strategy`)

- Live miner **still uses** its in-tree modules (`engine_entry.py`, `model.py`, …) so production is unchanged.
- This package is the **portable copy** for other projects / a new repo.
- Future optional step: make mining re-export from `smartbs_entry` after `pip install -e packages/smartbs_entry` (single source of truth).

---

## File inventory (`packages/smartbs_entry/`)

```
pyproject.toml
README.md
REPORT.md                 ← this file
smartbs_entry/
  VERSION                 # 0.9.7
  __init__.py
  engine.py
  registry.py
  indicators.py
  structure_sm.py
  model.py
  features.py
  labels.py
  config.py
  train.py
  calibrate.py
  calibration.py
  checkpoint.py
  predict.py
  data.py
  dukascopy.py
  stdio_compat.py
```

---

## Dependencies

**Required:** `numpy`, `pandas`, `torch`, `requests`  
**Optional:** `yfinance`, `tradingview-datafeed`  
**Not required:** Bittensor, Flask, Polygon, Vanta subnet packages
