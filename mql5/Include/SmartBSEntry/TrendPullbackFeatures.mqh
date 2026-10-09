//+------------------------------------------------------------------+
//| SmartBS TrendPullback — EMA stack + pullback, 15 channels        |
//| Parity with smartbs_engines/engine_trend_pullback.py             |
//+------------------------------------------------------------------+
#ifndef SMARTBS_TREND_PULLBACK_FEATURES_MQH
#define SMARTBS_TREND_PULLBACK_FEATURES_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"

#define SB_TP_CH              15
#define SB_TP_EMA_FAST        20
#define SB_TP_EMA_MID         50
#define SB_TP_EMA_SLOW        200
#define SB_TP_IMPULSE_20      20
#define SB_TP_IMPULSE_48      48
#define SB_TP_TREND_CLIP      2.0
#define SB_TP_WARMUP          (3 * 200)

double SB_TpAtrUnit(const double x, const double atr, const double k)
  {
   return SB_Clip(SB_SafeDiv(x, atr), -k, k) / k;
  }

bool SB_BuildTrendPullbackFeatureMatrix(const string symbol, double &features[][SB_TP_CH],
                                        int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1;
   int want = MathMax(SB_TP_WARMUP + 512, 800);
   if(!SB_CopyRatesChrono(symbol, SB_PrimaryTf(), want, h1))
     {
      err = "not enough H1 bars";
      return false;
     }
   n_bars = h1.n;
   ArrayResize(features, n_bars);
   ArrayInitialize(features, 0.0);

   double atr14[];
   SB_ATR(h1.high, h1.low, h1.close, 14, atr14);

   double fast[], mid[], slow[];
   SB_EMA(h1.close, SB_TP_EMA_FAST, fast);
   SB_EMA(h1.close, SB_TP_EMA_MID, mid);
   SB_EMA(h1.close, SB_TP_EMA_SLOW, slow);

   double slope_f[], slope_m[], slope_s[];
   SB_SlopeNorm(fast, atr14, 1, slope_f);
   SB_SlopeNorm(mid, atr14, 1, slope_m);
   SB_SlopeNorm(slow, atr14, 1, slope_s);

   double run_max20[], run_min20[], run_max48[], run_min48[];
   double argmax20[], argmin20[];
   SB_RollingMax(h1.high, SB_TP_IMPULSE_20, run_max20);
   SB_RollingMin(h1.low, SB_TP_IMPULSE_20, run_min20);
   SB_RollingMax(h1.high, SB_TP_IMPULSE_48, run_max48);
   SB_RollingMin(h1.low, SB_TP_IMPULSE_48, run_min48);
   SB_RollingArgMax(h1.high, SB_TP_IMPULSE_20, argmax20);
   SB_RollingArgMin(h1.low, SB_TP_IMPULSE_20, argmin20);

   for(int i = 0; i < n_bars; i++)
     {
      double stack_fm = SB_TpAtrUnit(fast[i] - mid[i], atr14[i], SB_TP_TREND_CLIP);
      double stack_ms = SB_TpAtrUnit(mid[i] - slow[i], atr14[i], SB_TP_TREND_CLIP);
      double stack_sum = stack_fm + stack_ms;

      bool trend_up = (fast[i] > mid[i]) && (mid[i] > slow[i]);
      bool trend_dn = (fast[i] < mid[i]) && (mid[i] < slow[i]);

      double rng20 = run_max20[i] - run_min20[i];
      double rng48 = run_max48[i] - run_min48[i];
      double depth20 = 0.0, depth48 = 0.0;
      if(trend_up)
        {
         depth20 = run_max20[i] - h1.close[i];
         depth48 = run_max48[i] - h1.close[i];
        }
      else if(trend_dn)
        {
         depth20 = h1.close[i] - run_min20[i];
         depth48 = h1.close[i] - run_min48[i];
        }

      double retrace20 = SB_Clip(SB_SafeDiv(depth20, rng20), 0.0, 1.5);
      double retrace48 = SB_Clip(SB_SafeDiv(depth48, rng48), 0.0, 1.5);

      int span20 = (i + 1 < SB_TP_IMPULSE_20) ? (i + 1) : SB_TP_IMPULSE_20;
      double bs20 = 0.0;
      if(trend_up)
         bs20 = (span20 - 1 - argmax20[i]) / (double)SB_TP_IMPULSE_20;
      else if(trend_dn)
         bs20 = (span20 - 1 - argmin20[i]) / (double)SB_TP_IMPULSE_20;

      double impulse20 = SB_TpAtrUnit(rng20, atr14[i], 3.0);
      double impulse_vs = SB_Clip(SB_SafeDiv(rng20, rng48), 0.0, 3.0);

      double bull_setup = SB_Clip(stack_sum, 0.0, 1.0);
      double bear_setup = SB_Clip(-stack_sum, 0.0, 1.0);
      double setup_long  = SB_Clip(retrace20 * bull_setup, 0.0, 1.5);
      double setup_short = SB_Clip(retrace20 * bear_setup, 0.0, 1.5);

      features[i][0]  = SB_Nan0(stack_fm);
      features[i][1]  = SB_Nan0(stack_ms);
      features[i][2]  = SB_Nan0(SB_TpAtrUnit(h1.close[i] - fast[i], atr14[i], 2.0));
      features[i][3]  = SB_Nan0(SB_TpAtrUnit(h1.close[i] - mid[i], atr14[i], 2.0));
      features[i][4]  = SB_Nan0(SB_TpAtrUnit(h1.close[i] - slow[i], atr14[i], 2.0));
      features[i][5]  = SB_Nan0(SB_Clip(slope_f[i], -2.0, 2.0) / 2.0);
      features[i][6]  = SB_Nan0(SB_Clip(slope_m[i], -2.0, 2.0) / 2.0);
      features[i][7]  = SB_Nan0(SB_Clip(slope_s[i], -2.0, 2.0) / 2.0);
      features[i][8]  = SB_Nan0(retrace20);
      features[i][9]  = SB_Nan0(bs20);
      features[i][10] = SB_Nan0(retrace48);
      features[i][11] = SB_Nan0(impulse20);
      features[i][12] = SB_Nan0(impulse_vs);
      features[i][13] = SB_Nan0(setup_long);
      features[i][14] = SB_Nan0(setup_short);
     }
   return true;
  }

bool SB_BuildTrendPullbackLookbackWindow(const double &features[][SB_TP_CH], const int n_bars,
                                         const int lookback, const int end_bar,
                                         float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_TP_CH * lookback);
   for(int c = 0; c < SB_TP_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
