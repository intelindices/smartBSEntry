# SmartBS modularization report

**Date:** 2026-09-10 (engines + blend added)  
**Source:** `vanta-network` / SmartBS v0.9.7  
**Repo:** https://github.com/intelindices/smartBSEntry.git  

Two independent Python packages in one repo:

| Package | Role |
|---------|------|
| `smartbs_entry` | Entry engine (90-ch) + TCN + MQL5/ONNX |
| `smartbs_engines` | All other ST engines + equal-weight blend + TCN |

Neither depends on Vanta miner / Risk Manager / hydrate / REST.

---

## `smartbs_entry` (existing)

- `SmartBSEntryEngine`, `predict_raw_ai`, train CLI, MQL5 EA
- Excludes other ST engines and blend

## `smartbs_engines` (new)

### Engines

`maribbon`, `bollinger`, `trend_pullback`, `smart_money`, `macd`, `candle`, `rsi_divergence`

### Blend

- `ACTIVE_BLEND_ENGINES` = five live engines (no candle / rsi_divergence / entry)
- `simple_mean_blend` / `predict_blend_raw_ai`
- Checkpoint layout: `{root}/{engine}/{PAIR}.pt`

### Shared

- TCN (`SmartBSClassifier`), data loaders (Dukascopy/Yahoo/Binance/TV), train CLI
- `weights.equal_weights` / research helpers (live = 1/k)

### Explicitly excluded

- Entry engine (sibling package)
- Risk Manager, position manager, raw_arm, hydrate, miner REST, `VANTA_SUBMIT_MAP`
- Polygon / `vali_objects`

---

## Install

```bash
pip install -e .
pip install -e ".[yahoo,tradingview]"
```

## Smoke

```python
import smartbs_entry, smartbs_engines
assert smartbs_entry.list_engines() == ["entry"]
assert "entry" not in smartbs_engines.list_engines()
assert "maribbon" in smartbs_engines.ACTIVE_BLEND_ENGINES
```
