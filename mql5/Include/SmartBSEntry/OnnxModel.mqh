//+------------------------------------------------------------------+
//| SmartBS Entry — ONNX TCN loader / runner                         |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ENTRY_ONNX_MQH
#define SMARTBS_ENTRY_ONNX_MQH

#include "Softmax.mqh"
#include "EntryFeatures.mqh"

class CSBOnnxModel
  {
private:
   long     m_handle;
   int      m_lookback;
   int      m_num_inputs;
   double   m_temperature;
   bool     m_ready;

public:
   CSBOnnxModel(void) : m_handle(INVALID_HANDLE), m_lookback(64), m_num_inputs(90),
                        m_temperature(1.0), m_ready(false) {}

   ~CSBOnnxModel(void) { Shutdown(); }

   void SetTemperature(const double t) { m_temperature = (t > 1e-8 ? t : 1.0); }
   void SetLookback(const int lb) { m_lookback = lb; }
   int  Lookback(void) const { return m_lookback; }
   bool Ready(void) const { return m_ready; }

   bool Load(const string onnx_filename)
     {
      Shutdown();
      // Place file under MQL5/Files/ or Terminal Common\Files
      m_handle = OnnxCreate(onnx_filename, ONNX_DEFAULT);
      if(m_handle == INVALID_HANDLE)
        {
         Print("OnnxCreate failed for ", onnx_filename, " err=", GetLastError());
         return false;
        }

      // Shape: features [1, 90, lookback]
      ulong input_shape[];
      ArrayResize(input_shape, 3);
      input_shape[0] = 1;
      input_shape[1] = (ulong)m_num_inputs;
      input_shape[2] = (ulong)m_lookback;
      if(!OnnxSetInputShape(m_handle, 0, input_shape))
        {
         Print("OnnxSetInputShape failed err=", GetLastError());
         Shutdown();
         return false;
        }

      ulong output_shape[];
      ArrayResize(output_shape, 2);
      output_shape[0] = 1;
      output_shape[1] = 3;
      if(!OnnxSetOutputShape(m_handle, 0, output_shape))
        {
         Print("OnnxSetOutputShape failed err=", GetLastError());
         Shutdown();
         return false;
        }

      m_ready = true;
      Print("ONNX loaded: ", onnx_filename, " in=", m_num_inputs, "x", m_lookback);
      return true;
     }

   void Shutdown(void)
     {
      if(m_handle != INVALID_HANDLE)
        {
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
        }
      m_ready = false;
     }

   // window: flat float[90 * lookback] channel-major
   // returns class 0=FLAT 1=LONG 2=SHORT; fills probs[3]
   int Predict(const float &window[], double &probs[], double &logits[])
     {
      ArrayResize(probs, 3);
      ArrayResize(logits, 3);
      if(!m_ready)
         return 0;

      // MT5 OnnxRun expects typed arrays matching shapes
      float in_data[];
      ArrayResize(in_data, m_num_inputs * m_lookback);
      int n = ArraySize(window);
      int copy = MathMin(n, m_num_inputs * m_lookback);
      for(int i = 0; i < copy; i++)
         in_data[i] = window[i];

      float out_data[];
      ArrayResize(out_data, 3);

      if(!OnnxRun(m_handle, ONNX_NO_CONVERSION, in_data, out_data))
        {
         Print("OnnxRun failed err=", GetLastError());
         return 0;
        }

      for(int i = 0; i < 3; i++)
         logits[i] = (double)out_data[i];
      SB_SoftmaxTemp(logits, m_temperature, probs);
      return SB_Argmax(probs);
     }
  };

#endif
