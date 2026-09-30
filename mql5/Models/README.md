# SmartBSEntry ONNX models

Not stored in git. Current EA default: **bar raw_ai + `pivot_engine`** (12ch, pivot_breakout p5, smartBSEntryV2).

## Export (commodities, pivot_engine)

```bash
$ckpt = "smartbs_engines/checkpoints_pivot_engine_core_pivot_breakout_p5_commodity_entryv2/pivot_engine"
$out  = "mql5/Models/SmartBSEntry"
foreach ($sym in @("XAUUSD","XAGUSD","XTIUSD","NATGAS","PLATINUM")) {
  python -m smartbs_engines.export_onnx `
    --checkpoint "$ckpt/$sym.pt" `
    --out "$out/${sym}_pivot_engine.onnx"
}
```

## Install for MT5

Copy `mql5/Models/SmartBSEntry/*_pivot_engine.{onnx,json}` → terminal `MQL5/Files/SmartBSEntry/`.

(Or recompile EA — `#property tester_file` ships the models into the Strategy Tester.)

## EA inputs (Strategy Tester)

Defaults match Python every-bar pivot replay:

| Input | Value |
|---|---|
| `InpEngines` | `pivot_engine` |
| `InpTradePolicy` | `bar` (every H1) |
| `InpSignalMode` | `raw_ai` |
| `InpAiThreshold` | `0` |
| `InpSignalPointGate` / `InpMaAiGate` / `InpUseStopLoss` | `false` |
| `InpLot` | `0.1` (XAU) |
| Chart | XAUUSD / XAG / XTI / NATGAS / PLATINUM **H1** |

Model sidecar: 12 inputs × lookback 64, `label_mode=pivot_breakout`, `pivot_len=5`.
