//+------------------------------------------------------------------+
//| SmartBS dBB — double Bollinger occupancy (1h + M15), 26 channels |
//| Parity with smartbs_engines/engine_dbb.py                        |
//+------------------------------------------------------------------+
#ifndef SMARTBS_DBB_FEATURES_MQH
#define SMARTBS_DBB_FEATURES_MQH

#include "Indicators.mqh"
#include "Common.mqh"

#define SB_NUM_INPUTS      26
#define SB_DBB_BB_LEN      20
#define SB_DBB_MULT1       1.0
#define SB_DBB_MULT2       2.0
#define SB_DBB_WARMUP      256
#define SB_DBB_PACK        13
#define SB_M15_SEC         900

double SB_FracOverlap(const double seg_lo, const double seg_hi,
                      const double zone_lo, const double zone_hi)
  {
   double length = seg_hi - seg_lo;
   if(!MathIsValidNumber(length) || length <= 0.0)
      return 0.0;
   if(!MathIsValidNumber(zone_lo) || !MathIsValidNumber(zone_hi))
      return 0.0;
   double ov_lo = MathMax(seg_lo, zone_lo);
   double ov_hi = MathMin(seg_hi, zone_hi);
   double overlap = MathMax(ov_hi - ov_lo, 0.0);
   return overlap / length;
  }

void SB_DbbPack13(const double &open[], const double &high[],
                  const double &low[], const double &close[],
                  const int bb_len, const double mult1, const double mult2,
                  double &out[][SB_DBB_PACK])
  {
   int n = ArraySize(close);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0 || bb_len < 2)
      return;

   double basis[], sd[];
   SB_SMA_Strict(close, bb_len, basis);
   SB_StdevPop(close, bb_len, sd);

   for(int i = 0; i < n; i++)
     {
      if(i < bb_len - 1)
         continue;
      double b = basis[i];
      double s = sd[i];
      if(!MathIsValidNumber(b) || !MathIsValidNumber(s))
         continue;

      double upper1 = b + mult1 * s;
      double lower1 = b - mult1 * s;
      double upper2 = b + mult2 * s;
      double lower2 = b - mult2 * s;

      double blo = MathMin(open[i], close[i]);
      double bhi = MathMax(open[i], close[i]);
      double body[6];
      body[0] = SB_FracOverlap(blo, bhi, upper2, 1.0e100);
      body[1] = SB_FracOverlap(blo, bhi, upper1, upper2);
      body[2] = SB_FracOverlap(blo, bhi, b, upper1);
      body[3] = SB_FracOverlap(blo, bhi, lower1, b);
      body[4] = SB_FracOverlap(blo, bhi, lower2, lower1);
      body[5] = SB_FracOverlap(blo, bhi, -1.0e100, lower2);

      double rng[6];
      rng[0] = SB_FracOverlap(low[i], high[i], upper2, 1.0e100);
      rng[1] = SB_FracOverlap(low[i], high[i], upper1, upper2);
      rng[2] = SB_FracOverlap(low[i], high[i], b, upper1);
      rng[3] = SB_FracOverlap(low[i], high[i], lower1, b);
      rng[4] = SB_FracOverlap(low[i], high[i], lower2, lower1);
      rng[5] = SB_FracOverlap(low[i], high[i], -1.0e100, lower2);

      double codes[6] = {3.0, 2.0, 1.0, -1.0, -2.0, -3.0};
      int peak = 0;
      double best = body[0];
      for(int z = 1; z < 6; z++)
        {
         if(body[z] > best)
           {
            best = body[z];
            peak = z;
           }
        }
      double peak_code = (best > 0.0) ? codes[peak] : 0.0;

      for(int z = 0; z < 6; z++)
         out[i][z] = body[z];
      for(int z = 0; z < 6; z++)
         out[i][6 + z] = rng[z];
      out[i][12] = peak_code;
     }
  }

void SB_MapLastCompletedM15(const SBOhlc &h1, const SBOhlc &m15,
                            const double &pack15[][SB_DBB_PACK],
                            double &mapped[][SB_DBB_PACK])
  {
   ArrayResize(mapped, h1.n);
   ArrayInitialize(mapped, 0.0);
   int k = -1;
   for(int i = 0; i < h1.n; i++)
     {
      while(k + 1 < m15.n && (long)m15.time[k + 1] + SB_M15_SEC <= (long)h1.time[i])
         k++;
      if(k < 0)
         continue;
      for(int c = 0; c < SB_DBB_PACK; c++)
         mapped[i][c] = pack15[k][c];
     }
  }

bool SB_BuildDbbFeatureMatrix(const string symbol, double &features[][SB_NUM_INPUTS],
                              int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1, m15;
   int want = MathMax(SB_DBB_WARMUP + 512, SB_DBB_BB_LEN * 40);
   if(!SB_CopyRatesChrono(symbol, PERIOD_H1, want, h1))
     {
      err = "not enough H1 bars";
      return false;
     }
   if(!SB_CopyRatesChrono(symbol, PERIOD_M15, MathMax(2000, want * 4 + 200), m15))
     {
      err = "not enough M15 bars";
      return false;
     }

   double pack1h[][SB_DBB_PACK];
   double pack15[][SB_DBB_PACK];
   double map15[][SB_DBB_PACK];
   SB_DbbPack13(h1.open, h1.high, h1.low, h1.close,
                SB_DBB_BB_LEN, SB_DBB_MULT1, SB_DBB_MULT2, pack1h);
   SB_DbbPack13(m15.open, m15.high, m15.low, m15.close,
                SB_DBB_BB_LEN, SB_DBB_MULT1, SB_DBB_MULT2, pack15);
   SB_MapLastCompletedM15(h1, m15, pack15, map15);

   n_bars = h1.n;
   ArrayResize(features, n_bars);
   ArrayInitialize(features, 0.0);
   for(int i = 0; i < n_bars; i++)
     {
      for(int c = 0; c < SB_DBB_PACK; c++)
         features[i][c] = pack1h[i][c];
      for(int c = 0; c < SB_DBB_PACK; c++)
         features[i][SB_DBB_PACK + c] = map15[i][c];
     }
   return true;
  }

bool SB_BuildDbbLookbackWindow(const double &features[][SB_NUM_INPUTS], const int n_bars,
                               const int lookback, const int end_bar,
                               float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_NUM_INPUTS * lookback);
   for(int c = 0; c < SB_NUM_INPUTS; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
