# MQL5 Expert Advisor — maribbon (rolle_breakout)

`Experts/SmartBSEntry.mq5` loads `{asset}_maribbon_1h.onnx` by default (39 channels).

| | Rule (defaults) |
|--|------|
| **Engine** | `maribbon` |
| **TF** | **H1** |
| **Labels** | `rolle_breakout` rolle_len=5 (forward REH/REL updates) |
| **Backbone** | smartBSEntryV2 |
| **Signal** | argmax FLAT / LONG / SHORT (`raw_ai`) |
| **Trade** | every bar (`InpTradePolicy=bar`) + SL ladder + ST filter |

Export:

```bash
python -m smartbs_engines.export_onnx \
  --checkpoint checkpoints_rolle_breakout_p5_1h_entryv2/maribbon/XAUUSD.pt \
  --out mql5/Models/SmartBSEntry/XAUUSD_maribbon_1h.onnx
```

Copy `mql5/Models/SmartBSEntry/XAUUSD_maribbon_1h.{onnx,json}` → terminal `MQL5/Files/SmartBSEntry/`.

### Strategy Tester

1. MetaEditor: compile `SmartBSEntry.mq5` (F7).
2. Tester: Expert = SmartBSEntry, Symbol = XAUUSD, Period = **H1**.
3. Optional: load `Presets/SmartBSEntry.set`.
4. Model: 1 minute OHLC or Open prices.
5. Journal should show `maribbon` ready and class probs each bar.
