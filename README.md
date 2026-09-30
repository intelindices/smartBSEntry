# smartbs

Standalone **SmartBS ST engines** — train independently, blend any subset, export ONNX for MT5.

| Package | Import | What |
|---------|--------|------|
| **smartbs_engines** | `smartbs_engines` | ST feature engines + equal-weight **blend** + TCN + ONNX export |

No Entry engine. No Vanta miner / Risk Manager / hydrate.

## Install

```bash
git clone https://github.com/intelindices/smartBSEntry.git
cd smartBSEntry
pip install -e .
pip install -e ".[yahoo,tradingview,onnx,mt5]"   # optional feeds / export / MT5 dump
```

## Engines

| Engine | Core ch | +common | Live blend? |
|--------|---------|---------|-------------|
| `regime_engine` | 15 | 29 | yes |
| `maribbon` | 48 | 62 | yes |
| `dbb` | 32 | 46 | yes (double BB 1h+15m) |
| `trend_pullback` | 18 | 32 | yes |
| `smart_money` | 22 | 36 | yes |
| `macd` | 16 | 30 | yes |
| `candle` | 20 | 34 | registered / train only |
| `rsi_divergence` | 26 | 40 | registered / train only |
| `signals` | variable | +14 | signal bus |

Every engine’s feature matrix ends with the shared **common** pack (14): OHLCV(1h) close-norm, time-of-day (no date), Tokyo/London/NY sessions, candle direction vs EMA14, near-term trend EMA14>EMA48.

Live blend = equal-weight mean softmax over `ACTIVE_BLEND_ENGINES`, then argmax (`raw_ai`).

```bash
# Train one engine
python -m smartbs_engines.train --trade-pair XAUUSD --data-source mt5 --feature-engine dbb

# Dump MT5 bars (cache under data_cache/mt5/)
python -m smartbs_engines.mt5_data --symbol XAUUSD

# Export ONNX for the EA (after train)
python -m smartbs_engines.export_onnx --checkpoint smartbs_engines/checkpoints/dbb/XAUUSD.pt --out mql5/Models/XAUUSD_dbb.onnx

# Blend any subset (Python replay)
python -m smartbs_engines._replay_blend --engines maribbon,dbb,macd
```

```python
from smartbs_engines import predict_raw_ai, predict_blend_raw_ai

out = predict_raw_ai(df, "checkpoints/dbb/XAUUSD.pt", symbol="XAUUSD", data_source="mt5")
blend = predict_blend_raw_ai(
    df, "smartbs_engines/checkpoints", trade_pair="XAUUSD",
    engines=("maribbon", "dbb", "macd"), data_source="mt5",
)
```

## MQL5 EA

See [mql5/README.md](mql5/README.md). The Expert Advisor loads ONNX and emits **raw_ai** only (FLAT / LONG / SHORT). Risk management and order execution are left for a later EA layer.

1h training defaults to `train_from_date=2022-06-14` (15m history alignment). See `.cursor/rules/1h-train-window.mdc`.
