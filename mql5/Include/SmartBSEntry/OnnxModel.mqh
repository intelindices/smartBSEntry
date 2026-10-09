//+------------------------------------------------------------------+
//| SmartBS — ONNX TCN loader / runner (max 128 channels × lookback)  |
//| OnnxRun requires the buffer byte-size to match the input shape.  |
//| We dispatch exact [1][C][64] tensors per channel count.           |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ONNX_MQH
#define SMARTBS_ONNX_MQH

#include "Softmax.mqh"

#define SB_ONNX_MAX_INPUTS 128
#define SB_ONNX_MAX_LOOKBACK 64

class CSBOnnxModel
  {
private:
   long     m_handle;
   int      m_lookback;
   int      m_num_inputs;
   double   m_temperature;
   bool     m_ready;
   float    m_out[1][3];

   // Exact tensors — MT5 validates sizeof(buffer) against the ONNX shape.
   // Slim totals (+common16): common16; pivot12; signals26; regime/dbb28; macd29; tp32; sm/candle36; maribbon/rsi42.
   float    m_in12[1][12][SB_ONNX_MAX_LOOKBACK];
   float    m_in13[1][13][SB_ONNX_MAX_LOOKBACK];
   float    m_in15[1][15][SB_ONNX_MAX_LOOKBACK];
   float    m_in16[1][16][SB_ONNX_MAX_LOOKBACK];
   float    m_in18[1][18][SB_ONNX_MAX_LOOKBACK];
   float    m_in20[1][20][SB_ONNX_MAX_LOOKBACK];
   float    m_in22[1][22][SB_ONNX_MAX_LOOKBACK];
   float    m_in23[1][23][SB_ONNX_MAX_LOOKBACK];
   float    m_in24[1][24][SB_ONNX_MAX_LOOKBACK];
   float    m_in25[1][25][SB_ONNX_MAX_LOOKBACK];
   float    m_in26[1][26][SB_ONNX_MAX_LOOKBACK];
   float    m_in27[1][27][SB_ONNX_MAX_LOOKBACK];
   float    m_in28[1][28][SB_ONNX_MAX_LOOKBACK];
   float    m_in29[1][29][SB_ONNX_MAX_LOOKBACK];
   float    m_in31[1][31][SB_ONNX_MAX_LOOKBACK];
   float    m_in32[1][32][SB_ONNX_MAX_LOOKBACK];
   float    m_in33[1][33][SB_ONNX_MAX_LOOKBACK];
   float    m_in35[1][35][SB_ONNX_MAX_LOOKBACK];
   float    m_in36[1][36][SB_ONNX_MAX_LOOKBACK];
   float    m_in38[1][38][SB_ONNX_MAX_LOOKBACK];
   float    m_in39[1][39][SB_ONNX_MAX_LOOKBACK];
   float    m_in41[1][41][SB_ONNX_MAX_LOOKBACK];
   float    m_in42[1][42][SB_ONNX_MAX_LOOKBACK];
   float    m_in44[1][44][SB_ONNX_MAX_LOOKBACK];
   float    m_in45[1][45][SB_ONNX_MAX_LOOKBACK];
   float    m_in48[1][48][SB_ONNX_MAX_LOOKBACK];
   float    m_in115[1][115][SB_ONNX_MAX_LOOKBACK];
   float    m_in128[1][128][SB_ONNX_MAX_LOOKBACK];

   bool FillAndRun(const float &window[])
     {
      const int C = m_num_inputs;
      const int T = m_lookback;
      if(T != SB_ONNX_MAX_LOOKBACK)
        {
         // Non-64 lookback: use a sized flat buffer (product must match shape).
         float flat[];
         ArrayResize(flat, C * T);
         for(int i = 0; i < C * T; i++)
            flat[i] = window[i];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, flat, m_out);
        }

      if(C == 12)
        {
         ArrayInitialize(m_in12, 0.0f);
         for(int c = 0; c < 12; c++)
            for(int t = 0; t < T; t++)
               m_in12[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in12, m_out);
        }
      if(C == 13)
        {
         ArrayInitialize(m_in13, 0.0f);
         for(int c = 0; c < 13; c++)
            for(int t = 0; t < T; t++)
               m_in13[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in13, m_out);
        }
      if(C == 15)
        {
         ArrayInitialize(m_in15, 0.0f);
         for(int c = 0; c < 15; c++)
            for(int t = 0; t < T; t++)
               m_in15[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in15, m_out);
        }
      if(C == 16)
        {
         ArrayInitialize(m_in16, 0.0f);
         for(int c = 0; c < 16; c++)
            for(int t = 0; t < T; t++)
               m_in16[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in16, m_out);
        }
      if(C == 18)
        {
         ArrayInitialize(m_in18, 0.0f);
         for(int c = 0; c < 18; c++)
            for(int t = 0; t < T; t++)
               m_in18[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in18, m_out);
        }
      if(C == 20)
        {
         ArrayInitialize(m_in20, 0.0f);
         for(int c = 0; c < 20; c++)
            for(int t = 0; t < T; t++)
               m_in20[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in20, m_out);
        }
      if(C == 22)
        {
         ArrayInitialize(m_in22, 0.0f);
         for(int c = 0; c < 22; c++)
            for(int t = 0; t < T; t++)
               m_in22[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in22, m_out);
        }
      if(C == 23)
        {
         ArrayInitialize(m_in23, 0.0f);
         for(int c = 0; c < 23; c++)
            for(int t = 0; t < T; t++)
               m_in23[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in23, m_out);
        }
      if(C == 24)
        {
         ArrayInitialize(m_in24, 0.0f);
         for(int c = 0; c < 24; c++)
            for(int t = 0; t < T; t++)
               m_in24[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in24, m_out);
        }
      if(C == 25)
        {
         ArrayInitialize(m_in25, 0.0f);
         for(int c = 0; c < 25; c++)
            for(int t = 0; t < T; t++)
               m_in25[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in25, m_out);
        }
      if(C == 26)
        {
         ArrayInitialize(m_in26, 0.0f);
         for(int c = 0; c < 26; c++)
            for(int t = 0; t < T; t++)
               m_in26[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in26, m_out);
        }
      if(C == 27)
        {
         ArrayInitialize(m_in27, 0.0f);
         for(int c = 0; c < 27; c++)
            for(int t = 0; t < T; t++)
               m_in27[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in27, m_out);
        }
      if(C == 28)
        {
         ArrayInitialize(m_in28, 0.0f);
         for(int c = 0; c < 28; c++)
            for(int t = 0; t < T; t++)
               m_in28[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in28, m_out);
        }
      if(C == 29)
        {
         ArrayInitialize(m_in29, 0.0f);
         for(int c = 0; c < 29; c++)
            for(int t = 0; t < T; t++)
               m_in29[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in29, m_out);
        }
      if(C == 31)
        {
         ArrayInitialize(m_in31, 0.0f);
         for(int c = 0; c < 31; c++)
            for(int t = 0; t < T; t++)
               m_in31[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in31, m_out);
        }
      if(C == 32)
        {
         ArrayInitialize(m_in32, 0.0f);
         for(int c = 0; c < 32; c++)
            for(int t = 0; t < T; t++)
               m_in32[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in32, m_out);
        }
      if(C == 33)
        {
         ArrayInitialize(m_in33, 0.0f);
         for(int c = 0; c < 33; c++)
            for(int t = 0; t < T; t++)
               m_in33[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in33, m_out);
        }
      if(C == 35)
        {
         ArrayInitialize(m_in35, 0.0f);
         for(int c = 0; c < 35; c++)
            for(int t = 0; t < T; t++)
               m_in35[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in35, m_out);
        }
      if(C == 36)
        {
         ArrayInitialize(m_in36, 0.0f);
         for(int c = 0; c < 36; c++)
            for(int t = 0; t < T; t++)
               m_in36[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in36, m_out);
        }
      if(C == 38)
        {
         ArrayInitialize(m_in38, 0.0f);
         for(int c = 0; c < 38; c++)
            for(int t = 0; t < T; t++)
               m_in38[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in38, m_out);
        }
      if(C == 39)
        {
         ArrayInitialize(m_in39, 0.0f);
         for(int c = 0; c < 39; c++)
            for(int t = 0; t < T; t++)
               m_in39[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in39, m_out);
        }
      if(C == 41)
        {
         ArrayInitialize(m_in41, 0.0f);
         for(int c = 0; c < 41; c++)
            for(int t = 0; t < T; t++)
               m_in41[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in41, m_out);
        }
      if(C == 42)
        {
         ArrayInitialize(m_in42, 0.0f);
         for(int c = 0; c < 42; c++)
            for(int t = 0; t < T; t++)
               m_in42[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in42, m_out);
        }
      if(C == 44)
        {
         ArrayInitialize(m_in44, 0.0f);
         for(int c = 0; c < 44; c++)
            for(int t = 0; t < T; t++)
               m_in44[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in44, m_out);
        }
      if(C == 45)
        {
         ArrayInitialize(m_in45, 0.0f);
         for(int c = 0; c < 45; c++)
            for(int t = 0; t < T; t++)
               m_in45[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in45, m_out);
        }
      if(C == 48)
        {
         ArrayInitialize(m_in48, 0.0f);
         for(int c = 0; c < 48; c++)
            for(int t = 0; t < T; t++)
               m_in48[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in48, m_out);
        }
      if(C == 115)
        {
         ArrayInitialize(m_in115, 0.0f);
         for(int c = 0; c < 115; c++)
            for(int t = 0; t < T; t++)
               m_in115[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in115, m_out);
        }
      if(C == 128)
        {
         ArrayInitialize(m_in128, 0.0f);
         for(int c = 0; c < 128; c++)
            for(int t = 0; t < T; t++)
               m_in128[0][c][t] = window[c * T + t];
         return OnnxRun(m_handle, ONNX_NO_CONVERSION, m_in128, m_out);
        }

      Print("Unsupported num_inputs for ONNX buffer: ", C);
      return false;
     }

public:
   CSBOnnxModel(void) : m_handle(INVALID_HANDLE), m_lookback(64), m_num_inputs(26),
                        m_temperature(1.0), m_ready(false)
     {
      ArrayInitialize(m_out, 0.0f);
      ArrayInitialize(m_in12, 0.0f);
      ArrayInitialize(m_in13, 0.0f);
      ArrayInitialize(m_in15, 0.0f);
      ArrayInitialize(m_in16, 0.0f);
      ArrayInitialize(m_in18, 0.0f);
      ArrayInitialize(m_in20, 0.0f);
      ArrayInitialize(m_in22, 0.0f);
      ArrayInitialize(m_in23, 0.0f);
      ArrayInitialize(m_in26, 0.0f);
      ArrayInitialize(m_in27, 0.0f);
      ArrayInitialize(m_in29, 0.0f);
      ArrayInitialize(m_in31, 0.0f);
      ArrayInitialize(m_in32, 0.0f);
      ArrayInitialize(m_in33, 0.0f);
      ArrayInitialize(m_in35, 0.0f);
      ArrayInitialize(m_in36, 0.0f);
      ArrayInitialize(m_in39, 0.0f);
      ArrayInitialize(m_in41, 0.0f);
      ArrayInitialize(m_in42, 0.0f);
      ArrayInitialize(m_in44, 0.0f);
      ArrayInitialize(m_in45, 0.0f);
      ArrayInitialize(m_in48, 0.0f);
      ArrayInitialize(m_in115, 0.0f);
      ArrayInitialize(m_in128, 0.0f);
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
      string base = onnx_filename;
      int slash = StringLen(base) - 1;
      for(; slash >= 0; slash--)
        {
         ushort ch = StringGetCharacter(base, slash);
         if(ch == '\\' || ch == '/')
           {
            base = StringSubstr(base, slash + 1);
            break;
           }
        }
      string tries[4];
      tries[0] = onnx_filename;
      tries[1] = base;
      tries[2] = onnx_filename;
      tries[3] = base;
      int flags[4];
      flags[0] = ONNX_USE_CPU_ONLY;
      flags[1] = ONNX_USE_CPU_ONLY;
      flags[2] = ONNX_USE_CPU_ONLY | ONNX_COMMON_FOLDER;
      flags[3] = ONNX_USE_CPU_ONLY | ONNX_COMMON_FOLDER;

      for(int t = 0; t < 4; t++)
        {
         ResetLastError();
         m_handle = OnnxCreate(tries[t], flags[t]);
         if(m_handle != INVALID_HANDLE)
           {
            if(tries[t] != onnx_filename || (flags[t] & ONNX_COMMON_FOLDER) != 0)
               Print("ONNX opened via fallback: ", tries[t],
                     ((flags[t] & ONNX_COMMON_FOLDER) != 0 ? " (COMMON)" : ""));
            break;
           }
        }
      if(m_handle == INVALID_HANDLE)
        {
         Print("OnnxCreate failed for ", onnx_filename, " err=", GetLastError(),
               " exist=", FileIsExist(onnx_filename),
               " common=", FileIsExist(onnx_filename, FILE_COMMON),
               " base_exist=", FileIsExist(base),
               " base_common=", FileIsExist(base, FILE_COMMON));
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
      Print("ONNX loaded: ", onnx_filename, " in=", m_num_inputs, "x", m_lookback,
            " bytes=", m_num_inputs * m_lookback * 4);
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
   // returns 0=FLAT 1=LONG 2=SHORT; on hard failure returns -1
   int Predict(const float &window[], double &probs[], double &logits[])
     {
      ArrayResize(probs, 3);
      ArrayResize(logits, 3);
      probs[0] = probs[1] = probs[2] = 0.0;
      logits[0] = logits[1] = logits[2] = 0.0;
      if(!m_ready)
         return -1;

      int n = ArraySize(window);
      int need = m_num_inputs * m_lookback;
      if(n < need)
        {
         Print("ONNX window too short: ", n, " need ", need);
         return -1;
        }

      ResetLastError();
      if(!FillAndRun(window))
        {
         Print("OnnxRun failed err=", GetLastError(),
               " in_bytes=", need * 4,
               " shape=", m_num_inputs, "x", m_lookback);
         return -1;
        }

      for(int i = 0; i < 3; i++)
         logits[i] = (double)m_out[0][i];
      SB_SoftmaxTemp(logits, m_temperature, probs);
      return SB_Argmax(probs);
     }
  };

#endif
