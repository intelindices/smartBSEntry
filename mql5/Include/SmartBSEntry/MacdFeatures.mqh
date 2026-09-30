//+------------------------------------------------------------------+
//| SmartBS MACD — EMA stack + MACD histogram, 13 channels           |
//| Parity with smartbs_engines/engine_macd.py                       |
//+------------------------------------------------------------------+
#ifndef SMARTBS_MACD_FEATURES_MQH
#define SMARTBS_MACD_FEATURES_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"

#define SB_MACD_CH            13
#define SB_MACD_EMA_FAST      12
#define SB_MACD_EMA_SLOW      26
#define SB_MACD_EMA_SIGNAL    9
#define SB_MACD_CTX_FAST      5
#define SB_MACD_CTX_SLOW      35
#define SB_MACD_STACK_CLIP    2.0
#define SB_MACD_HIST_Z_WIN    50
#define SB_MACD_HIST_PERSIST  10
#define SB_MACD_CROSS_CAP     20
#define SB_MACD_HIST_Z_CLIP   3.0
#define SB_MACD_WARMUP        (3 * 78)

double SB_MacdAtrUnit(const double x, const double atr, const double k)
  {
   return SB_Clip(SB_SafeDiv(x, atr), -k, k) / k;
  }

bool SB_BuildMacdFeatureMatrix(const string symbol, double &features[][SB_MACD_CH],
                               int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1;
   int want = MathMax(SB_MACD_WARMUP + 512, 400);
   if(!SB_CopyRatesChrono(symbol, PERIOD_H1, want, h1))
     {
      err = "not enough H1 bars";
      return false;
     }
   n_bars = h1.n;
   ArrayResize(features, n_bars);
   ArrayInitialize(features, 0.0);

   double atr14[];
   SB_ATR(h1.high, h1.low, h1.close, 14, atr14);

   double ema12[], ema26[], ema5[], ema35[];
   SB_EMA(h1.close, SB_MACD_EMA_FAST, ema12);
   SB_EMA(h1.close, SB_MACD_EMA_SLOW, ema26);
   SB_EMA(h1.close, SB_MACD_CTX_FAST, ema5);
   SB_EMA(h1.close, SB_MACD_CTX_SLOW, ema35);

   double stack[], signal[], hist[];
   ArrayResize(stack, n_bars);
   ArrayResize(hist, n_bars);
   for(int i = 0; i < n_bars; i++)
      stack[i] = ema12[i] - ema26[i];
   SB_EMA(stack, SB_MACD_EMA_SIGNAL, signal);
   for(int i = 0; i < n_bars; i++)
      hist[i] = stack[i] - signal[i];

   bool bull[], cross_up[], cross_dn[];
   ArrayResize(bull, n_bars);
   ArrayResize(cross_up, n_bars);
   ArrayResize(cross_dn, n_bars);
   bull[0] = (stack[0] > signal[0]);
   cross_up[0] = false;
   cross_dn[0] = false;
   for(int i = 1; i < n_bars; i++)
     {
      bull[i] = (stack[i] > signal[i]);
      cross_up[i] = bull[i] && !bull[i - 1];
      cross_dn[i] = !bull[i] && bull[i - 1];
     }

   double slope_f[], slope_s[], slope_st[], slope_h[];
   SB_SlopeNorm(ema12, atr14, 1, slope_f);
   SB_SlopeNorm(ema26, atr14, 1, slope_s);
   SB_SlopeNorm(stack, atr14, 1, slope_st);
   SB_SlopeNorm(hist, atr14, 1, slope_h);

   double hist_mu[], hist_sd[];
   SB_SMA_Strict(hist, SB_MACD_HIST_Z_WIN, hist_mu);
   SB_StdevPop(hist, SB_MACD_HIST_Z_WIN, hist_sd);

   double hist_sign[];
   ArrayResize(hist_sign, n_bars);
   for(int i = 0; i < n_bars; i++)
     {
      if(hist[i] > 0.0)
         hist_sign[i] = 1.0;
      else if(hist[i] < 0.0)
         hist_sign[i] = -1.0;
      else
         hist_sign[i] = 0.0;
     }
   double hist_persist[];
   SB_RollingMeanMin1(hist_sign, SB_MACD_HIST_PERSIST, hist_persist);

   double xu_dec[], xd_dec[];
   SB_BarsSinceFlagDecay(cross_up, SB_MACD_CROSS_CAP, xu_dec);
   SB_BarsSinceFlagDecay(cross_dn, SB_MACD_CROSS_CAP, xd_dec);

   for(int i = 0; i < n_bars; i++)
     {
      double dist_fast = SB_MacdAtrUnit(h1.close[i] - ema12[i], atr14[i], SB_MACD_STACK_CLIP);
      double dist_slow = SB_MacdAtrUnit(h1.close[i] - ema26[i], atr14[i], SB_MACD_STACK_CLIP);
      double close_vs = 0.5 * (dist_fast + dist_slow);
      double z = SB_SafeDiv(hist[i] - hist_mu[i], hist_sd[i]);
      z = SB_Clip(z, -SB_MACD_HIST_Z_CLIP, SB_MACD_HIST_Z_CLIP) / SB_MACD_HIST_Z_CLIP;

      features[i][0]  = SB_Nan0(SB_MacdAtrUnit(stack[i], atr14[i], SB_MACD_STACK_CLIP));
      features[i][1]  = SB_Nan0(dist_fast);
      features[i][2]  = SB_Nan0(dist_slow);
      features[i][3]  = SB_Nan0(SB_Clip(slope_f[i], -SB_MACD_STACK_CLIP, SB_MACD_STACK_CLIP) / SB_MACD_STACK_CLIP);
      features[i][4]  = SB_Nan0(SB_Clip(slope_s[i], -SB_MACD_STACK_CLIP, SB_MACD_STACK_CLIP) / SB_MACD_STACK_CLIP);
      features[i][5]  = SB_Nan0(SB_Clip(slope_st[i], -SB_MACD_STACK_CLIP, SB_MACD_STACK_CLIP) / SB_MACD_STACK_CLIP);
      features[i][6]  = SB_Nan0(close_vs);
      features[i][7]  = SB_Nan0(SB_MacdAtrUnit(ema5[i] - ema35[i], atr14[i], SB_MACD_STACK_CLIP));
      features[i][8]  = SB_Nan0(z);
      features[i][9]  = SB_Nan0(SB_Clip(slope_h[i], -SB_MACD_STACK_CLIP, SB_MACD_STACK_CLIP) / SB_MACD_STACK_CLIP);
      features[i][10] = SB_Nan0(hist_persist[i]);
      features[i][11] = SB_Nan0(xu_dec[i]);
      features[i][12] = SB_Nan0(xd_dec[i]);
     }
   return true;
  }

bool SB_BuildMacdLookbackWindow(const double &features[][SB_MACD_CH], const int n_bars,
                                const int lookback, const int end_bar,
                                float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_MACD_CH * lookback);
   for(int c = 0; c < SB_MACD_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
