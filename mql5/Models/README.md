# SmartBSEntry ONNX models

Not stored in git (except when checked in for a release pack). Current EA default:
**bar raw_ai + `maribbon`** on **XAUUSD 1h** (39ch, rolle_breakout L=5, smartBSEntryV2).

## Export (default pack)

```bash
python -m smartbs_engines.export_onnx `
  --checkpoint checkpoints_rolle_breakout_p5_1h_entryv2/maribbon/XAUUSD.pt `
  --out mql5/Models/SmartBSEntry/XAUUSD_maribbon_1h.onnx
```

## Install for MT5

Copy `mql5/Models/SmartBSEntry/XAUUSD_maribbon_1h.{onnx,json}` → terminal `MQL5/Files/SmartBSEntry/`.

(Or recompile EA — `#property tester_file` / `TesterFiles.mqh` ships models into the Strategy Tester.)

## EA inputs (Strategy Tester)

| Input | Value |
|---|---|
| `InpEngines` | `maribbon` |
| `InpTimeframe` | `H1` |
| `InpTradePolicy` | `bar` |
| `InpSignalMode` | `raw_ai` |
| `InpAiThreshold` | `0` |
| `InpStAtrLen` | `5` (match rolle_len) |
| `InpExitMode` | `SL ladder` |
| Chart | XAUUSD **H1** |

Model sidecar: 39 inputs × lookback 64, `label_mode=rolle_breakout`, `rolle_len=5`.
