//+------------------------------------------------------------------+
//| SmartBS Entry — indicators (parity with smartbs_entry.indicators)|
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
