# smartbs

Standalone **SmartBS AI packages** — extractable from Vanta Network for reuse in other projects.

| Package | Import | What |
|---------|--------|------|
| **smartbs_entry** | `smartbs_entry` | Entry engine (90-ch multi-TF) + TCN + ONNX/MQL5 |
| **smartbs_engines** | `smartbs_engines` | All other ST engines + equal-weight **blend** + TCN |

Neither package includes the Vanta miner, Risk Manager, hydrate, or REST submit path.

## Install

```bash
git clone https://github.com/intelindices/smartBSEntry.git
cd smartBSEntry
pip install -e .
# optional feeds:
pip install -e ".[yahoo,tradingview]"
# optional MT5 export for Entry:
pip install -e ".[onnx]"
```

---

## `smartbs_entry` — Entry engine

See also [mql5/README.md](mql5/README.md) for the MetaTrader 5 EA.

```python
from smartbs_entry import predict_raw_ai
from smartbs_entry.data import fetch_mtf_klines

df = fetch_mtf_klines("XAUUSD", source="dukascopy", max_1h_candles=8000)
out = predict_raw_ai(df, "path/to/XAUUSD.pt", symbol="XAUUSD", data_source="dukascopy")
print(out["signal"], out["probs"])
```

```bash
python -m smartbs_entry.train --trade-pair XAUUSD --data-source dukascopy --feature-engine entry
```

---

## `smartbs_engines` — ST engines + blend

### Engines (no Entry)

| Engine | Channels | Live blend? |
|--------|----------|-------------|
| `maribbon` | 48 | yes |
| `bollinger` | 20 | yes |
| `trend_pullback` | 18 | yes |
| `smart_money` | 22 | yes |
| `macd` | 16 | yes |
| `candle` | 20 | registered / train only |
| `rsi_divergence` | 26 | registered / train only |

Live blend = equal-weight mean over `ACTIVE_BLEND_ENGINES` (the five “yes” rows).

### Single-engine raw AI

```python
from smartbs_engines import predict_raw_ai, get_engine, list_engines
from smartbs_engines.data import fetch_mtf_klines

print(list_engines())
df = fetch_mtf_klines("XAUUSD", source="dukascopy", max_1h_candles=8000)
out = predict_raw_ai(df, "checkpoints/maribbon/XAUUSD.pt", symbol="XAUUSD")
print(out["signal"], out["feature_engine"])
```

### Equal-weight blend

Checkpoints layout (same as Vanta live):

```
{blend_root}/
  maribbon/XAUUSD.pt
  bollinger/XAUUSD.pt
  trend_pullback/XAUUSD.pt
  smart_money/XAUUSD.pt
  macd/XAUUSD.pt
```

```python
from smartbs_engines import predict_blend_raw_ai, simple_mean_blend, ACTIVE_BLEND_ENGINES
from smartbs_engines.data import fetch_mtf_klines

df = fetch_mtf_klines("XAUUSD", source="dukascopy", max_1h_candles=8000)
out = predict_blend_raw_ai(
    df,
    blend_root="./checkpoints/dukascopy_10y",
    trade_pair="XAUUSD",
    data_source="dukascopy",
)
print(out["signal"], out["probs"], out["engines"])
```

### Train one engine

```bash
export SMARTBS_CHECKPOINT_DIR=./checkpoints
python -m smartbs_engines.train \
  --trade-pair XAUUSD \
  --feature-engine maribbon \
  --data-source dukascopy
# writes checkpoints/maribbon/XAUUSD.pt
```

---

## Package layout

```
smartbs_entry/     # Entry-only AI + MQL5 helpers
smartbs_engines/   # ST engines + blend + shared TCN/data
mql5/              # MetaTrader 5 EA for Entry ONNX
```

## Version

Both packages ship `VERSION` **0.9.7** (aligned with SmartBS at extraction).
