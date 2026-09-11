# smartbs_engines

ST feature engines + equal-weight blend for SmartBS.

## Layout

```
smartbs_engines/
  engine_*.py / maribbon_engine.py   # feature engines
  train.py / predict.py / blend.py   # train one, infer one, mean-blend subset
  export_onnx.py                     # .pt → ONNX + JSON sidecar for MT5
  mt5_data.py                        # broker bar dump / cache
  checkpoints/{engine}/{PAIR}.pt     # local (gitignored)
```

## Train

```bash
python -m smartbs_engines.train --trade-pair XAUUSD --data-source mt5 --feature-engine macd
```

Checkpoints land at `{SMARTBS_CHECKPOINT_DIR|smartbs_engines/checkpoints}/{engine}/{PAIR}.pt`.

## Blend

```python
from smartbs_engines import predict_blend_probs, ACTIVE_BLEND_ENGINES

probs, meta = predict_blend_probs(
    df_1h, "smartbs_engines/checkpoints",
    trade_pair="XAUUSD",
    engines=ACTIVE_BLEND_ENGINES,  # or any subset
    data_source="mt5",
)
# side = argmax(probs[i])
```

`ACTIVE_BLEND_ENGINES` = maribbon, **dbb**, trend_pullback, smart_money, macd.
