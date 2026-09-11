# SmartBS modularization report

**Date:** 2026-09-11 (Entry removed; dBB replaces bollinger)  
**Repo:** https://github.com/intelindices/smartBSEntry.git  

Single Python package:

| Package | Role |
|---------|------|
| `smartbs_engines` | ST feature engines + equal-weight blend + TCN + ONNX export |

No Entry engine. No Vanta miner / Risk Manager / hydrate / REST.

## Engines

`maribbon`, `dbb` (26ch double-BB), `trend_pullback`, `smart_money`, `macd`, `candle`, `rsi_divergence`

Retired: `entry`, `bollinger` (use `dbb`).

### Blend

- `ACTIVE_BLEND_ENGINES` = maribbon, dbb, trend_pullback, smart_money, macd
- `simple_mean_blend` / `predict_blend_raw_ai` (any subset via `engines=`)
- Checkpoint layout: `{root}/{engine}/{PAIR}.pt`

### MQL5

EA is an ONNX **raw_ai** shell only. Feature builder ported today: `dbb`. Bake models with `python -m smartbs_engines.export_onnx`.
