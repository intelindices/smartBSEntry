//+------------------------------------------------------------------+
//| SmartBS pivot_engine — 12 channels (core-only, no common pack)   |
//| Parity with smartbs_engines/engine_pivot.py                      |
//| 0 potential=vol_n*disp | 1..5 ohlcv | 6 trend | 7..8 tod         |
//| 9..10 cd_vs/cross | 11 last_candle_strength                      |
//+------------------------------------------------------------------+
#ifndef SMARTBS_PIVOT_FEATURES_MQH
#define SMARTBS_PIVOT_FEATURES_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"
#include "EngineCatalog.mqh"

#ifndef SB_PIVOT_CH
#define SB_PIVOT_CH           12
#endif
#define SB_PIVOT_VOL_SMA      48
#define SB_PIVOT_VOL_CLIP     3.0
#define SB_PIVOT_DISP_CLIP    0.05
#define SB_PIVOT_EMA_FAST     14
#define SB_PIVOT_EMA_MID      48
#define SB_PIVOT_EMA_SLOW     120
#define SB_PIVOT_DAY_SEC      86400
#define SB_PIVOT_WARMUP       (3 * SB_PIVOT_EMA_SLOW)

datetime SB_PivotBarTimeGMT(const datetime bar_server_time)
  {
   return bar_server_time - TimeCurrent() + TimeGMT();
  }

bool SB_BuildPivotFeatureMatrix(const string symbol, double &features[][SB_PIVOT_CH],
                                int &n_bars, string &err)
  {
   err = "";
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int want = MathMax(SB_PIVOT_WARMUP + 512, 400);
   int got = CopyRates(symbol, PERIOD_H1, 0, want, rates);
   if(got < 10)
     {
      err = "pivot: not enough H1 bars";
      return false;
     }
   n_bars = got;
   ArrayResize(features, n_bars);
   ArrayInitialize(features, 0.0);

   double open[], high[], low[], close[], volume[];
   ArrayResize(open, n_bars);
   ArrayResize(high, n_bars);
   ArrayResize(low, n_bars);
   ArrayResize(close, n_bars);
   ArrayResize(volume, n_bars);
   datetime times[];
   ArrayResize(times, n_bars);
   for(int i = 0; i < n_bars; i++)
     {
      times[i] = rates[i].time;
      open[i] = rates[i].open;
      high[i] = rates[i].high;
      low[i] = rates[i].low;
      close[i] = rates[i].close;
      volume[i] = (double)rates[i].tick_volume;
     }

   double vol_sma[], ema_f[], ema_mid[], ema_slow[];
   SB_SMA(volume, SB_PIVOT_VOL_SMA, vol_sma);
   SB_EMA(close, SB_PIVOT_EMA_FAST, ema_f);
   SB_EMA(close, SB_PIVOT_EMA_MID, ema_mid);
   SB_EMA(close, SB_PIVOT_EMA_SLOW, ema_slow);

   for(int i = 0; i < n_bars; i++)
     {
      double c = close[i];
      double o_n = SB_Nan0(SB_SafeDiv(open[i], c));
      double h_n = SB_Nan0(SB_SafeDiv(high[i], c));
      double l_n = SB_Nan0(SB_SafeDiv(low[i], c));
      double vol_n = SB_Clip(SB_Nan0(SB_SafeDiv(volume[i], vol_sma[i])), 0.0, SB_PIVOT_VOL_CLIP);
      double disp = 0.0;
      if(i > 0)
         disp = SB_Clip(SB_Nan0(SB_SafeDiv(c - close[i - 1], close[i - 1])),
                        -SB_PIVOT_DISP_CLIP, SB_PIVOT_DISP_CLIP);
      features[i][0] = SB_Nan0(vol_n * disp);
      features[i][1] = o_n;
      features[i][2] = h_n;
      features[i][3] = l_n;
      features[i][4] = vol_n;
      features[i][5] = disp;

      if(ema_f[i] > ema_mid[i] && ema_mid[i] > ema_slow[i])
         features[i][6] = 1.0;
      else if(ema_f[i] < ema_mid[i] && ema_mid[i] < ema_slow[i])
         features[i][6] = -1.0;
      else
         features[i][6] = 0.0;

      datetime bar_gmt = SB_PivotBarTimeGMT(times[i]);
      int sec = (int)(bar_gmt % SB_PIVOT_DAY_SEC);
      if(sec < 0)
         sec += SB_PIVOT_DAY_SEC;
      double frac = (double)sec / (double)SB_PIVOT_DAY_SEC;
      double ang = 2.0 * M_PI * frac;
      features[i][7] = MathSin(ang);
      features[i][8] = MathCos(ang);

      if(close[i] > ema_f[i])
         features[i][9] = 1.0;
      else if(close[i] < ema_f[i])
         features[i][9] = -1.0;
      else
         features[i][9] = 0.0;
      double body_lo = MathMin(open[i], close[i]);
      double body_hi = MathMax(open[i], close[i]);
      features[i][10] = (body_lo < ema_f[i] && body_hi > ema_f[i]) ? 1.0 : 0.0;

      features[i][11] = SB_Nan0(SB_SafeDiv(open[i] - close[i], high[i] - low[i]));
     }
   return true;
  }

bool SB_BuildPivotLookbackWindow(const double &features[][SB_PIVOT_CH], const int n_bars,
                                 const int lookback, const int end_bar,
                                 float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_PIVOT_CH * lookback);
   for(int c = 0; c < SB_PIVOT_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
