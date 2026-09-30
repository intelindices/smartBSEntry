//+------------------------------------------------------------------+
//| SmartBS — shared feature helpers (clip / slope / rolling / map)  |
//+------------------------------------------------------------------+
#ifndef SMARTBS_FEATURE_UTILS_MQH
#define SMARTBS_FEATURE_UTILS_MQH

#include "Indicators.mqh"

#define SB_HOUR_SEC 3600

double SB_Clip(const double v, const double lo, const double hi)
  {
   if(!MathIsValidNumber(v))
      return 0.0;
   if(v < lo)
      return lo;
   if(v > hi)
      return hi;
   return v;
  }

double SB_Nan0(const double v)
  {
   return MathIsValidNumber(v) ? v : 0.0;
  }

// (src[i] - src[i-lookback]) / atr[i]  — first lookback bars use src[i] as prev
void SB_SlopeNorm(const double &src[], const double &atr[], const int lookback, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || lookback <= 0)
      return;
   for(int i = 0; i < n; i++)
     {
      int j = i - lookback;
      double prev = (j >= 0) ? src[j] : src[i];
      out[i] = SB_Nan0(SB_SafeDiv(src[i] - prev, atr[i]));
     }
  }

void SB_RollingMax(const double &src[], const int window, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || window <= 0)
      return;
   for(int i = 0; i < n; i++)
     {
      int start = i - window + 1;
      if(start < 0)
         start = 0;
      double mx = src[start];
      for(int j = start + 1; j <= i; j++)
         if(src[j] > mx)
            mx = src[j];
      out[i] = mx;
     }
  }

void SB_RollingMin(const double &src[], const int window, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || window <= 0)
      return;
   for(int i = 0; i < n; i++)
     {
      int start = i - window + 1;
      if(start < 0)
         start = 0;
      double mn = src[start];
      for(int j = start + 1; j <= i; j++)
         if(src[j] < mn)
            mn = src[j];
      out[i] = mn;
     }
  }

// Rolling mean with min_periods=1 (partial windows allowed)
void SB_RollingMeanMin1(const double &src[], const int window, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || window <= 0)
      return;
   double sum = 0.0;
   for(int i = 0; i < n; i++)
     {
      sum += src[i];
      if(i >= window)
         sum -= src[i - window];
      int cnt = (i + 1 < window) ? (i + 1) : window;
      out[i] = sum / (double)cnt;
     }
  }

// Index of max within rolling window (relative to window start), min_periods=1
void SB_RollingArgMax(const double &src[], const int window, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || window <= 0)
      return;
   for(int i = 0; i < n; i++)
     {
      int start = i - window + 1;
      if(start < 0)
         start = 0;
      int arg = 0;
      double mx = src[start];
      for(int j = start + 1; j <= i; j++)
        {
         if(src[j] > mx)
           {
            mx = src[j];
            arg = j - start;
           }
        }
      out[i] = (double)arg;
     }
  }

void SB_RollingArgMin(const double &src[], const int window, double &out[])
  {
   int n = ArraySize(src);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || window <= 0)
      return;
   for(int i = 0; i < n; i++)
     {
      int start = i - window + 1;
      if(start < 0)
         start = 0;
      int arg = 0;
      double mn = src[start];
      for(int j = start + 1; j <= i; j++)
        {
         if(src[j] < mn)
           {
            mn = src[j];
            arg = j - start;
           }
        }
      out[i] = (double)arg;
     }
  }

// 1 at event bar, linear decay to 0 over ``cap`` bars (bool flags)
void SB_BarsSinceFlagDecay(const bool &flag[], const int cap, double &out[])
  {
   int n = ArraySize(flag);
   ArrayResize(out, n);
   int last = -1;
   for(int i = 0; i < n; i++)
     {
      if(flag[i])
         last = i;
      if(last >= 0)
         out[i] = MathMax(0.0, 1.0 - (i - last) / (double)cap);
      else
         out[i] = 0.0;
     }
  }

void SB_BarsSinceScoreDecay(const double &score[], const double thresh,
                            const int cap, double &out[])
  {
   int n = ArraySize(score);
   ArrayResize(out, n);
   int last = -1;
   for(int i = 0; i < n; i++)
     {
      if(score[i] >= thresh)
         last = i;
      if(last >= 0)
         out[i] = MathMax(0.0, 1.0 - (i - last) / (double)cap);
      else
         out[i] = 0.0;
     }
  }

// Snapshot last 15m value inside each 1h open-time hour bucket; ffill, else 0
void SB_Map15mEndOfHour(const datetime &h1_times[], const datetime &m15_times[],
                        const double &m15_values[], double &mapped[])
  {
   int n1 = ArraySize(h1_times);
   int n15 = ArraySize(m15_times);
   ArrayResize(mapped, n1);
   ArrayInitialize(mapped, 0.0);
   if(n1 <= 0 || n15 <= 0)
      return;

   // Build last-value-per-hour from 15m (chrono ascending)
   datetime hours[];
   double   lasts[];
   ArrayResize(hours, 0);
   ArrayResize(lasts, 0);
   datetime cur_h = -1;
   double   cur_v = 0.0;
   for(int k = 0; k < n15; k++)
     {
      datetime h = (datetime)(((long)m15_times[k] / (long)SB_HOUR_SEC) * (long)SB_HOUR_SEC);
      if(cur_h < 0 || h != cur_h)
        {
         if(cur_h >= 0)
           {
            int m = ArraySize(hours);
            ArrayResize(hours, m + 1);
            ArrayResize(lasts, m + 1);
            hours[m] = cur_h;
            lasts[m] = cur_v;
           }
         cur_h = h;
        }
      cur_v = m15_values[k];
     }
   if(cur_h >= 0)
     {
      int m = ArraySize(hours);
      ArrayResize(hours, m + 1);
      ArrayResize(lasts, m + 1);
      hours[m] = cur_h;
      lasts[m] = cur_v;
     }

   int nh = ArraySize(hours);
   int hi = 0;
   double last_mapped = 0.0;
   bool have = false;
   for(int i = 0; i < n1; i++)
     {
      datetime h = (datetime)(((long)h1_times[i] / (long)SB_HOUR_SEC) * (long)SB_HOUR_SEC);
      while(hi + 1 < nh && hours[hi + 1] <= h)
         hi++;
      if(hi < nh && hours[hi] == h)
        {
         last_mapped = lasts[hi];
         have = true;
         mapped[i] = last_mapped;
        }
      else if(have)
         mapped[i] = last_mapped;
      else
         mapped[i] = 0.0;
     }
  }

// Pack channel-major float window from flat feat[bar * ch + c]
bool SB_ExtractLookbackWindowFlat(const double &feat[], const int n_bars, const int ch,
                                  const int lookback, const int end_bar,
                                  float &window[], string &err)
  {
   err = "";
   if(ch <= 0 || lookback <= 0)
     {
      err = "invalid ch/lookback";
      return false;
     }
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   if(ArraySize(feat) < n_bars * ch)
     {
      err = "flat feature buffer too small";
      return false;
     }
   ArrayResize(window, ch * lookback);
   for(int c = 0; c < ch; c++)
      for(int t = 0; t < lookback; t++)
        {
         int bar = end_bar - lookback + 1 + t;
         double v = feat[bar * ch + c];
         window[c * lookback + t] = (float)(MathIsValidNumber(v) ? v : 0.0);
        }
   return true;
  }

#endif
