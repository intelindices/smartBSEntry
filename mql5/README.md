# MQL5 Expert Advisor — raw_ai shell

`Experts/SmartBSEntry.mq5` loads an ONNX TCN and prints **raw_ai** (`FLAT` / `LONG` / `SHORT`) on each closed H1 bar. It does **not** place orders or manage risk.

## Features (today)

| Engine | Channels | MQL builder |
|--------|----------|-------------|
| `dbb` | 26 | `Include/SmartBSEntry/DbbFeatures.mqh` |

Other ST engines (`maribbon`, `macd`, …) are trained in Python; MQL feature ports come later. Set `InpEngines=dbb` until then.

## Bake a model

```bash
python -m smartbs_engines.train --trade-pair XAUUSD --data-source mt5 --feature-engine dbb
python -m smartbs_engines.export_onnx \
  --checkpoint smartbs_engines/checkpoints/dbb/XAUUSD.pt \
  --out mql5/Models/XAUUSD_dbb.onnx
```

Copy `XAUUSD_dbb.onnx` (+ `.json` sidecar) to the terminal’s `MQL5/Files/SmartBSEntry/` (or set `InpOnnxFile`).

## Inputs

- `InpOnnxFile` — path under `Files/` (blank → `SmartBSEntry\{ASSET}_dbb.onnx`)
- `InpEngines` — comma list for future multi-ONNX mean blend (single `dbb` for now)
- `InpSignalMode` — raw argmax or confidence threshold
- `InpLookback` / `InpNumInputs` / `InpTemperature` — override sidecar JSON

## Include layout

```
Include/SmartBSEntry/
  Common.mqh       # OHLC copy
  DbbFeatures.mqh  # 26ch dBB
  OnnxModel.mqh    # ONNX runner (max 48×64)
  Softmax.mqh
  Indicators.mqh
```
