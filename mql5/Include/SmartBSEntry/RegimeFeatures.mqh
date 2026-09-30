//+------------------------------------------------------------------+
//| SmartBS regime_engine — 12 channels                              |
//| Parity with smartbs_engines/engine_regime.py                     |
//+------------------------------------------------------------------+
#ifndef SMARTBS_REGIME_FEATURES_MQH
#define SMARTBS_REGIME_FEATURES_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"
#include "EngineCatalog.mqh"

#ifndef SB_REGIME_CH
#define SB_REGIME_CH          12
#endif
#define SB_REGIME_EMA_FAST    14
#define SB_REGIME_EMA_MID     48
#define SB_REGIME_EMA_SLOW    120
#define SB_REGIME_SMA_LONG    180
#define SB_REGIME_DIST_CLIP   3.0
#define SB_REGIME_SLOPE_CLIP  2.0
#define SB_REGIME_WARMUP      (3 * SB_REGIME_SMA_LONG)

double SB_RegimeAtrUnit(const double x, const double atr, const double k)
  {
   return SB_Clip(SB_SafeDiv(x, atr), -k, k) / k;
  }

bool SB_BuildRegimeFeatureMatrix(const string symbol, double &features[][SB_REGIME_CH],
                                 int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1;
   int want = MathMax(SB_REGIME_WARMUP + 512, 400);
   if(!SB_CopyRatesChrono(symbol, PERIOD_H1, want, h1))
     {
      err = "regime: not enough H1 bars";
      return false;
     }
   n_bars = h1.n;
   ArrayResize(features, n_bars);
   ArrayInitialize(features, 0.0);

   double atr14[];
   SB_ATR(h1.high, h1.low, h1.close, 14, atr14);

   double ma_fast[], ma_mid[], ma_slow[], ma_long[];
   SB_EMA(h1.close, SB_REGIME_EMA_FAST, ma_fast);
   SB_EMA(h1.close, SB_REGIME_EMA_MID, ma_mid);
   SB_EMA(h1.close, SB_REGIME_EMA_SLOW, ma_slow);
   SB_SMA_Strict(h1.close, SB_REGIME_SMA_LONG, ma_long);

   double slope_f[], slope_m[];
   SB_SlopeNorm(ma_fast, atr14, 1, slope_f);
   SB_SlopeNorm(ma_mid, atr14, 1, slope_m);

   for(int i = 0; i < n_bars; i++)
     {
      const bool fam = (ma_fast[i] > ma_mid[i]);
      const bool fbm = (ma_fast[i] < ma_mid[i]);
      const bool cam = (h1.close[i] > ma_mid[i]);
      const bool cbm = (h1.close[i] < ma_mid[i]);
      const double up = (fam && cam) ? 1.0 : 0.0;
      const double dn = (fbm && cbm) ? 1.0 : 0.0;
      features[i][0]  = up;
      features[i][1]  = dn;
      features[i][2]  = 1.0 - MathMax(up, dn);
      features[i][3]  = fam ? 1.0 : 0.0;
      features[i][4]  = cam ? 1.0 : 0.0;
      features[i][5]  = SB_Nan0(SB_RegimeAtrUnit(ma_fast[i] - ma_mid[i], atr14[i], SB_REGIME_DIST_CLIP));
      features[i][6]  = SB_Nan0(SB_RegimeAtrUnit(h1.close[i] - ma_mid[i], atr14[i], SB_REGIME_DIST_CLIP));
      features[i][7]  = SB_Nan0(SB_RegimeAtrUnit(h1.close[i] - ma_fast[i], atr14[i], SB_REGIME_DIST_CLIP));
      features[i][8]  = SB_Nan0(SB_RegimeAtrUnit(h1.close[i] - ma_slow[i], atr14[i], SB_REGIME_DIST_CLIP));
      features[i][9]  = SB_Nan0(SB_RegimeAtrUnit(h1.close[i] - ma_long[i], atr14[i], SB_REGIME_DIST_CLIP));
      features[i][10] = SB_Nan0(SB_Clip(slope_f[i], -SB_REGIME_SLOPE_CLIP, SB_REGIME_SLOPE_CLIP) / SB_REGIME_SLOPE_CLIP);
      features[i][11] = SB_Nan0(SB_Clip(slope_m[i], -SB_REGIME_SLOPE_CLIP, SB_REGIME_SLOPE_CLIP) / SB_REGIME_SLOPE_CLIP);
     }
   return true;
  }

bool SB_BuildRegimeLookbackWindow(const double &features[][SB_REGIME_CH], const int n_bars,
                                  const int lookback, const int end_bar,
                                  float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "regime: end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_REGIME_CH * lookback);
   for(int c = 0; c < SB_REGIME_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
