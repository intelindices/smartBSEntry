# MQL5 Expert Advisor — pivot_engine (pivot_breakout)

`Experts/SmartBSEntry.mq5` loads `{asset}_pivot_engine.onnx` (12 channels).

| | Rule (defaults) |
|--|------|
| **Engine** | `pivot_engine` (potential + OHLCV + trend + tod + cd + candle) |
| **Labels** | trained with `pivot_breakout` pivot_len=5 |
| **Signal** | argmax FLAT / LONG / SHORT (`raw_ai`) |
| **Trade** | every H1 bar sync (`InpTradePolicy=bar`) |
| **Gates / SL** | off |

Export:

```bash
python -m smartbs_engines.export_onnx \
  --checkpoint smartbs_engines/checkpoints_pivot_engine_core_pivot_breakout_p5_commodity_entryv2/pivot_engine/XAUUSD.pt \
  --out mql5/Models/SmartBSEntry/XAUUSD_pivot_engine.onnx
```

Copy `mql5/Models/SmartBSEntry/*_pivot_engine.{onnx,json}` → terminal `MQL5/Files/SmartBSEntry/`.

### Strategy Tester

1. MetaEditor: compile `SmartBSEntry.mq5` (F7).
2. Tester: Expert = SmartBSEntry, Symbol = XAUUSD (or XAG/XTI/NATGAS/PLATINUM), Period = **H1**.
3. Model: 1 minute OHLC or Open prices.
4. Journal should show `pivot_engine` ready and class probs each bar.
