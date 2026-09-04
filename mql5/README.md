# SmartBS Entry — MetaTrader 5 Expert Advisor

H1 **raw_ai** EA: 90-channel Entry features in MQL5 + trained TCN via ONNX.

## Layout

```
mql5/
  Experts/SmartBSEntry.mq5
  Include/SmartBSEntry/
    Indicators.mqh
    Structure.mqh
    EntryFeatures.mqh
    Softmax.mqh
    OnnxModel.mqh
  Models/          # export ONNX here, then copy into MT5 Files
    XAUUSD.onnx
    XAUUSD.json
```

## Install in MetaTrader 5

1. Copy `Include/SmartBSEntry/` → `<DataFolder>/MQL5/Include/SmartBSEntry/`
2. Copy `Experts/SmartBSEntry.mq5` → `<DataFolder>/MQL5/Experts/`
3. Export ONNX (from Python package root):

```bash
pip install -e ".[dev]"  # or: pip install onnx onnxruntime onnxscript
python -m smartbs_entry.export_onnx \
  --checkpoint /path/to/entry/XAUUSD.pt \
  --out mql5/Models/XAUUSD.onnx
```

4. Copy `XAUUSD.onnx` → `<DataFolder>/MQL5/Files/SmartBSEntry/XAUUSD.onnx`  
   (create the `SmartBSEntry` subfolder under `Files`)
5. In MetaEditor: compile `SmartBSEntry.mq5`
6. Attach to an **H1** chart (XAUUSD / gold symbol matching the checkpoint)
7. Inputs:
   - `InpOnnxFile` = `SmartBSEntry\XAUUSD.onnx`
   - `InpTemperature` = value from `XAUUSD.json` (`temperature`)
   - `InpLot` = `0.1` (fixed)
   - `InpOnlyClosedBar` = true

## Behaviour

- On each **new closed H1 bar**: build 90×64 feature window → ONNX logits → temperature softmax → argmax
- `FLAT` → close positions (this EA’s magic)
- `LONG` / `SHORT` → flatten opposite, open fixed lot
- No arm / R / day-DD (matches Python `predict_raw_ai`)

## History requirements

Engine warmup is **7200 H1 bars** (~10 months) plus lookback 64. Fine TFs (M1/M5/M15) should cover the same window. If the broker truncates M1, the EA falls back (same spirit as Python) and logs warmup failures until enough H1 is present.

In MT5: right-click chart → **Tools → History Center** / scroll chart left to force download, or use a broker with deep history.

## Parity check

```bash
python -m smartbs_entry.mql5_parity \
  --checkpoint /path/to/XAUUSD.pt \
  --symbol XAUUSD --data-source dukascopy \
  --out mql5/Models/XAUUSD_parity.json
```

Compare Journal `SmartBS raw_ai …` line (signal + probs) to the JSON `signal` / `probs`. Feature L2 against `window_chw` requires dumping the same bar from MQL5 (advanced).

## Notes

- One ONNX file per symbol (retrain/export per pair).
- `#property strict` EA; requires MT5 build with ONNX support.
- Structure assignment of multi-TF OHLC uses MQL5 dynamic arrays — use a recent MetaTrader 5 build.
