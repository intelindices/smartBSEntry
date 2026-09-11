//+------------------------------------------------------------------+
//| SmartBS — shared indicator helpers (SMA / stdev / ATR / swings)  |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ENTRY_INDICATORS_MQH
#define SMARTBS_ENTRY_INDICATORS_MQH

double SB_SafeDiv(const double num, const double den)
  {
   if(!MathIsValidNumber(den) || MathAbs(den) <= 1e-12)
      return 0.0;
   double v = num / den;
   if(!MathIsValidNumber(v))
      return 0.0;
   return v;
  }

void SB_PrevArray(const double &src[], double &dst[])
  {
   int n = ArraySize(src);
   ArrayResize(dst, n);
   if(n <= 0)
      return;
   dst[0] = src[0];
   for(int i = 1; i < n; i++)
      dst[i] = src[i - 1];
  }

void SB_EMA(const double &src[], const int length, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   if(n <= 0 || length <= 0)
      return;
   double alpha = 2.0 / (length + 1.0);
   out[0] = src[0];
   for(int i = 1; i < n; i++)
      out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1];
  }

void SB_SMA(const double &src[], const int length, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || length <= 0)
      return;
   double sum = 0.0;
   for(int i = 0; i < n; i++)
     {
      sum += src[i];
      if(i >= length)
         sum -= src[i - length];
      if(i >= length - 1)
         out[i] = sum / (double)length;
      else
         out[i] = sum / (double)(i + 1);
     }
  }

// pandas rolling(length, min_periods=length).mean — 0 until window is full
void SB_SMA_Strict(const double &src[], const int length, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || length <= 0)
      return;
   double sum = 0.0;
   for(int i = 0; i < n; i++)
     {
      sum += src[i];
      if(i >= length)
         sum -= src[i - length];
      if(i >= length - 1)
         out[i] = sum / (double)length;
     }
  }

// Population stdev (ddof=0), TradingView ta.stdev default; 0 until window full
void SB_StdevPop(const double &src[], const int length, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || length <= 0)
      return;
   for(int i = length - 1; i < n; i++)
     {
      double mean = 0.0;
      for(int j = i - length + 1; j <= i; j++)
         mean += src[j];
      mean /= (double)length;
      double acc = 0.0;
      for(int j = i - length + 1; j <= i; j++)
        {
         double d = src[j] - mean;
         acc += d * d;
        }
      out[i] = MathSqrt(acc / (double)length);
     }
  }

void SB_Slope(const double &src[], double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   for(int i = 1; i < n; i++)
      out[i] = MathArctan(src[i] - src[i - 1]);
  }

void SB_RsiSign(const double &close[], double &out[])
  {
   int n = ArraySize(close);
   double rsi[];
   SB_RSI(close, 14, rsi);
   ArrayResize(out, n);
   for(int i = 0; i < n; i++)
     {
      if(rsi[i] > 70.0)
         out[i] = 1.0;
      else if(rsi[i] < 30.0)
         out[i] = -1.0;
      else
         out[i] = 0.0;
     }
  }

void SB_ATR(const double &high[], const double &low[], const double &close[],
            const int length, double &out[])
  {
   int n = ArraySize(close);
   ArrayResize(out, n);
   if(n <= 0)
      return;
   double tr[];
   ArrayResize(tr, n);
   tr[0] = high[0] - low[0];
   for(int i = 1; i < n; i++)
     {
      double a = high[i] - low[i];
      double b = MathAbs(high[i] - close[i - 1]);
      double c = MathAbs(low[i] - close[i - 1]);
      tr[i] = MathMax(a, MathMax(b, c));
     }
   // Wilder EMA: alpha = 1/length
   double alpha = 1.0 / length;
   out[0] = tr[0];
   for(int i = 1; i < n; i++)
      out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1];
  }

// Wilder RMA helper (same recursion as ATR)
void SB_RMA(const double &src[], const int length, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || length <= 0)
      return;
   double alpha = 1.0 / length;
   out[0] = src[0];
   for(int i = 1; i < n; i++)
      out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1];
  }

// TradingView-style ADX (0..100). Caller may scale /100.
void SB_ADX(const double &high[], const double &low[], const double &close[],
            const int length, double &out[])
  {
   int n = ArraySize(close);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n < 2 || length <= 0)
      return;

   double tr[], plus_dm[], minus_dm[];
   ArrayResize(tr, n);
   ArrayResize(plus_dm, n);
   ArrayResize(minus_dm, n);
   tr[0] = 0.0;
   plus_dm[0] = 0.0;
   minus_dm[0] = 0.0;
   for(int i = 1; i < n; i++)
     {
      double up = high[i] - high[i - 1];
      double down = low[i - 1] - low[i];
      plus_dm[i] = (up > down && up > 0.0) ? up : 0.0;
      minus_dm[i] = (down > up && down > 0.0) ? down : 0.0;
      double a = high[i] - low[i];
      double b = MathAbs(high[i] - close[i - 1]);
      double c = MathAbs(low[i] - close[i - 1]);
      tr[i] = MathMax(a, MathMax(b, c));
     }

   double tr_s[], plus_s[], minus_s[];
   SB_RMA(tr, length, tr_s);
   SB_RMA(plus_dm, length, plus_s);
   SB_RMA(minus_dm, length, minus_s);

   double dx[];
   ArrayResize(dx, n);
   ArrayInitialize(dx, 0.0);
   for(int i = 0; i < n; i++)
     {
      if(tr_s[i] <= 1e-12)
         continue;
      double pdi = 100.0 * plus_s[i] / tr_s[i];
      double mdi = 100.0 * minus_s[i] / tr_s[i];
      double den = pdi + mdi;
      if(den > 1e-12)
         dx[i] = 100.0 * MathAbs(pdi - mdi) / den;
     }
   SB_RMA(dx, length, out);
   for(int i = 0; i < length && i < n; i++)
      out[i] = 0.0;
  }

void SB_RSI(const double &close[], const int period, double &out[])
  {
   int n = ArraySize(close);
   ArrayResize(out, n);
   if(n <= 0)
      return;
   double gain[], loss[];
   ArrayResize(gain, n);
   ArrayResize(loss, n);
   gain[0] = 0.0;
   loss[0] = 0.0;
   for(int i = 1; i < n; i++)
     {
      double d = close[i] - close[i - 1];
      gain[i] = (d > 0.0 ? d : 0.0);
      loss[i] = (d < 0.0 ? -d : 0.0);
     }
   double alpha = 1.0 / period;
   double avg_g = gain[0], avg_l = loss[0];
   out[0] = 50.0;
   for(int i = 1; i < n; i++)
     {
      avg_g = alpha * gain[i] + (1.0 - alpha) * avg_g;
      avg_l = alpha * loss[i] + (1.0 - alpha) * avg_l;
      if(avg_l == 0.0)
         out[i] = 100.0;
      else
         out[i] = 100.0 - (100.0 / (1.0 + avg_g / avg_l));
     }
  }

void SB_PivotHigh(const double &high[], const int left, const int right, double &out[])
  {
   int n = ArraySize(high);
   ArrayResize(out, n);
   ArrayInitialize(out, EMPTY_VALUE);
   for(int i = left + right; i < n; i++)
     {
      int pivot = i - right;
      double mx = high[pivot - left];
      int arg = 0;
      for(int k = 1; k <= left + right; k++)
        {
         double v = high[pivot - left + k];
         if(v > mx)
           {
            mx = v;
            arg = k;
           }
        }
      if(high[pivot] >= mx - 1e-15 && arg == left)
         out[i] = high[pivot];
     }
  }

void SB_PivotLow(const double &low[], const int left, const int right, double &out[])
  {
   int n = ArraySize(low);
   ArrayResize(out, n);
   ArrayInitialize(out, EMPTY_VALUE);
   for(int i = left + right; i < n; i++)
     {
      int pivot = i - right;
      double mn = low[pivot - left];
      int arg = 0;
      for(int k = 1; k <= left + right; k++)
        {
         double v = low[pivot - left + k];
         if(v < mn)
           {
            mn = v;
            arg = k;
           }
        }
      if(low[pivot] <= mn + 1e-15 && arg == left)
         out[i] = low[pivot];
     }
  }

void SB_ValueWhen(const bool &cond[], const double &src[], double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   double last = EMPTY_VALUE;
   for(int i = 0; i < n; i++)
     {
      if(cond[i])
         last = src[i];
      out[i] = last;
     }
  }

void SB_SwingLevels(const double &high[], const double &low[], const int swing_len,
                    double &sh[], double &sl[])
  {
   double ph[], pl[];
   SB_PivotHigh(high, swing_len, swing_len, ph);
   SB_PivotLow(low, swing_len, swing_len, pl);
   int n = ArraySize(high);
   bool ch[], cl[];
   ArrayResize(ch, n);
   ArrayResize(cl, n);
   for(int i = 0; i < n; i++)
     {
      ch[i] = (ph[i] != EMPTY_VALUE);
      cl[i] = (pl[i] != EMPTY_VALUE);
     }
   SB_ValueWhen(ch, ph, sh);
   SB_ValueWhen(cl, pl, sl);
  }

#endif
