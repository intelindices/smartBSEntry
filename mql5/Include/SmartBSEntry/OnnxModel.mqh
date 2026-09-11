//+------------------------------------------------------------------+
//| SmartBS — ONNX TCN loader / runner (max 48 channels × lookback)  |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ONNX_MQH
#define SMARTBS_ONNX_MQH

#include "Softmax.mqh"

#define SB_ONNX_MAX_INPUTS 48
#define SB_ONNX_MAX_LOOKBACK 64

class CSBOnnxModel
  {
private:
   long     m_handle;
   int      m_lookback;
   int      m_num_inputs;
   double   m_temperature;
   bool     m_ready;
   // Fixed 3-D buffers — MT5 OnnxRun does not reliably bind dynamic 1-D float[]
   float    m_in[1][SB_ONNX_MAX_INPUTS][SB_ONNX_MAX_LOOKBACK];
   float    m_out[1][3];

public:
   CSBOnnxModel(void) : m_handle(INVALID_HANDLE), m_lookback(64), m_num_inputs(26),
                        m_temperature(1.0), m_ready(false)
     {
      ArrayInitialize(m_in, 0.0f);
      ArrayInitialize(m_out, 0.0f);
     }

   ~CSBOnnxModel(void) { Shutdown(); }

   void SetTemperature(const double t) { m_temperature = (t > 1e-8 ? t : 1.0); }
   void SetLookback(const int lb)
     {
      if(lb > 0 && lb <= SB_ONNX_MAX_LOOKBACK)
         m_lookback = lb;
     }
   void SetNumInputs(const int n)
     {
      if(n > 0 && n <= SB_ONNX_MAX_INPUTS)
         m_num_inputs = n;
     }
   int  Lookback(void) const { return m_lookback; }
   int  NumInputs(void) const { return m_num_inputs; }
   bool Ready(void) const { return m_ready; }

   bool Load(const string onnx_filename)
     {
      Shutdown();
      m_handle = OnnxCreate(onnx_filename, ONNX_USE_CPU_ONLY);
      if(m_handle == INVALID_HANDLE)
         m_handle = OnnxCreate(onnx_filename, ONNX_USE_CPU_ONLY | ONNX_COMMON_FOLDER);
      if(m_handle == INVALID_HANDLE)
        {
         Print("OnnxCreate failed for ", onnx_filename, " err=", GetLastError(),
               " exist=", FileIsExist(onnx_filename),
               " common=", FileIsExist(onnx_filename, FILE_COMMON));
         return false;
        }

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

   // window: flat float[num_inputs * lookback] channel-major (c * lookback + t)
   // returns class 0=FLAT 1=LONG 2=SHORT; fills probs[3]
   int Predict(const float &window[], double &probs[], double &logits[])
     {
      ArrayResize(probs, 3);
      ArrayResize(logits, 3);
      if(!m_ready)
         return 0;

      int n = ArraySize(window);
      int need = m_num_inputs * m_lookback;
      if(n < need)
        {
         Print("ONNX window too short: ", n, " need ", need);
         return 0;
        }

      ArrayInitialize(m_in, 0.0f);
      for(int c = 0; c < m_num_inputs; c++)
         for(int t = 0; t < m_lookback; t++)
            m_in[0][c][t] = window[c * m_lookback + t];

      if(!OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in, m_out))
        {
         Print("OnnxRun failed err=", GetLastError());
         return 0;
        }

      for(int i = 0; i < 3; i++)
         logits[i] = (double)m_out[0][i];
      SB_SoftmaxTemp(logits, m_temperature, probs);
      return SB_Argmax(probs);
     }
  };

#endif
