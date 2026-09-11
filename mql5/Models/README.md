ONNX models are not stored in git. Train + export:

  python -m smartbs_engines.export_onnx --checkpoint ... --out mql5/Models/XAUUSD_dbb.onnx

Then copy into the MT5 terminal `MQL5/Files/SmartBSEntry/`.
