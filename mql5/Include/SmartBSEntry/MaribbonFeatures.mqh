//+------------------------------------------------------------------+
//| SmartBS MA-Ribbon — 9 EMAs × dist/slope + RSI4                   |
//| Parity with smartbs_engines/maribbon_engine.py (22 channels)     |
//| EMA420 / EMA500 permanently dropped                              |
//+------------------------------------------------------------------+
#ifndef SMARTBS_MARIBBON_FEATURES_MQH
#define SMARTBS_MARIBBON_FEATURES_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"

#define SB_MARIBBON_CH         22
#define SB_MARIBBON_N_MA       9
#define SB_MARIBBON_DIST_CLIP  2.0
#define SB_MARIBBON_SLOPE_CLIP 2.0
#define SB_MARIBBON_RSI_PERIOD 14
#define SB_MARIBBON_RSI_LO     30.0
#define SB_MARIBBON_RSI_HI     70.0
#define SB_MARIBBON_RSI_MID    50.0
#define SB_MARIBBON_WARMUP     (3 * 320)

int SB_MaribbonLens(const int idx)
  {
   int lens[SB_MARIBBON_N_MA] = {9, 14, 24, 40, 60, 100, 160, 240, 320};
   if(idx < 0 || idx >= SB_MARIBBON_N_MA)
      return 9;
   return lens[idx];
  }

double SB_MaribbonRsiToBand(const double rsi)
  {
   double above = rsi - SB_MARIBBON_RSI_HI;
   double below = SB_MARIBBON_RSI_LO - rsi;
   double outside = MathMax(above, below);
   double inside_room = MathMin(rsi - SB_MARIBBON_RSI_LO, SB_MARIBBON_RSI_HI - rsi);
   double raw = (outside > 0.0) ? outside : -inside_room;
   return SB_Clip(raw / 20.0, -3.0, 3.0);
  }

double SB_MaribbonAtrUnit(const double x, const double atr, const double k)
  {
   return SB_Clip(SB_SafeDiv(x, atr), -k, k) / k;
  }

bool SB_BuildMaribbonFeatureMatrix(const string symbol, double &features[][SB_MARIBBON_CH],
                                   int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1;
   int want = MathMax(SB_MARIBBON_WARMUP + 512, 2000);
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

   double mas[][SB_MARIBBON_N_MA];
   ArrayResize(mas, n_bars);
   for(int m = 0; m < SB_MARIBBON_N_MA; m++)
     {
      double ema[];
      SB_EMA(h1.close, SB_MaribbonLens(m), ema);
      for(int i = 0; i < n_bars; i++)
         mas[i][m] = ema[i];
     }

   double rsi14[];
   SB_RSI(h1.close, SB_MARIBBON_RSI_PERIOD, rsi14);

   for(int i = 0; i < n_bars; i++)
     {
      // Dist fan (9): close-ema9, then consecutive faster-slower
      features[i][0] = SB_Nan0(SB_MaribbonAtrUnit(h1.close[i] - mas[i][0], atr14[i], SB_MARIBBON_DIST_CLIP));
      for(int m = 1; m < SB_MARIBBON_N_MA; m++)
         features[i][m] = SB_Nan0(SB_MaribbonAtrUnit(mas[i][m - 1] - mas[i][m], atr14[i], SB_MARIBBON_DIST_CLIP));

      // Slope 1-bar (9)
      for(int m = 0; m < SB_MARIBBON_N_MA; m++)
        {
         double prev = (i >= 1) ? mas[i - 1][m] : mas[i][m];
         features[i][9 + m] = SB_Nan0(SB_MaribbonAtrUnit(mas[i][m] - prev, atr14[i], SB_MARIBBON_SLOPE_CLIP));
        }
      // RSI4
      double r = rsi14[i];
      double r_prev = (i >= 1) ? rsi14[i - 1] : r;
      features[i][18] = SB_Nan0(r / 100.0);
      features[i][19] = SB_Nan0((r - SB_MARIBBON_RSI_MID) / 50.0);
      features[i][20] = SB_Clip((r - r_prev) / 10.0, -3.0, 3.0);
      features[i][21] = SB_Nan0(SB_MaribbonRsiToBand(r));
     }
   return true;
  }

bool SB_BuildMaribbonLookbackWindow(const double &features[][SB_MARIBBON_CH], const int n_bars,
                                    const int lookback, const int end_bar,
                                    float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_MARIBBON_CH * lookback);
   for(int c = 0; c < SB_MARIBBON_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
