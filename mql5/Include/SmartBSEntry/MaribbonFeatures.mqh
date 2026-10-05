//+------------------------------------------------------------------+
//| SmartBS MA-Ribbon — 39 channels (13 × 1h + 15m + 5m)             |
//| Parity with smartbs_engines/maribbon_engine.py                   |
//| EMAs: 9,14,24,36,48,60,72,84,96,120,180,240 — no common pack     |
//+------------------------------------------------------------------+
#ifndef SMARTBS_MARIBBON_FEATURES_MQH
#define SMARTBS_MARIBBON_FEATURES_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"

#define SB_MARIBBON_CH         39
#define SB_MARIBBON_BASE       13
#define SB_MARIBBON_N_MA       12
#define SB_MARIBBON_N_PAIRS    11
#define SB_MARIBBON_OHLC_CLIP  2.0
#define SB_MARIBBON_ST_ATR     9
#define SB_MARIBBON_ST_MULT    3.0
#define SB_MARIBBON_WARMUP     (3 * 240)
#define SB_MARIBBON_M15_SEC    900
#define SB_MARIBBON_M5_SEC     300

int SB_MaribbonLens(const int idx)
  {
   int lens[SB_MARIBBON_N_MA] = {9, 14, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240};
   if(idx < 0 || idx >= SB_MARIBBON_N_MA)
      return 9;
   return lens[idx];
  }

double SB_MaribbonAtrUnit(const double x, const double atr, const double k)
  {
   return SB_Clip(SB_SafeDiv(x, atr), -k, k) / k;
  }

void SB_MaribbonPackBase(const double &open[], const double &high[],
                         const double &low[], const double &close[],
                         double &out[][SB_MARIBBON_BASE])
  {
   int n = ArraySize(close);
   ArrayResize(out, n);
   ArrayInitialize(out, 0.0);
   if(n <= 0)
      return;

   double atr14[];
   SB_ATR(high, low, close, 14, atr14);

   double mas[][SB_MARIBBON_N_MA];
   ArrayResize(mas, n);
   for(int m = 0; m < SB_MARIBBON_N_MA; m++)
     {
      double ema[];
      SB_EMA(close, SB_MaribbonLens(m), ema);
      for(int i = 0; i < n; i++)
         mas[i][m] = ema[i];
     }

   // SuperTrend ATR9×3
   double atr_st[];
   SB_ATR(high, low, close, SB_MARIBBON_ST_ATR, atr_st);
   double final_ub[], final_lb[], st_dir[];
   ArrayResize(final_ub, n);
   ArrayResize(final_lb, n);
   ArrayResize(st_dir, n);
   ArrayInitialize(final_ub, 0.0);
   ArrayInitialize(final_lb, 0.0);
   ArrayInitialize(st_dir, 1.0);
   if(n > 0)
     {
      double hl2_0 = 0.5 * (high[0] + low[0]);
      final_ub[0] = hl2_0 + SB_MARIBBON_ST_MULT * atr_st[0];
      final_lb[0] = hl2_0 - SB_MARIBBON_ST_MULT * atr_st[0];
      st_dir[0] = 1.0;
     }
   for(int i = 1; i < n; i++)
     {
      double hl2 = 0.5 * (high[i] + low[i]);
      double bub = hl2 + SB_MARIBBON_ST_MULT * atr_st[i];
      double blb = hl2 - SB_MARIBBON_ST_MULT * atr_st[i];
      final_ub[i] = (bub < final_ub[i - 1] || close[i - 1] > final_ub[i - 1]) ? bub : final_ub[i - 1];
      final_lb[i] = (blb > final_lb[i - 1] || close[i - 1] < final_lb[i - 1]) ? blb : final_lb[i - 1];
      if(close[i] > final_ub[i - 1])
         st_dir[i] = 1.0;
      else if(close[i] < final_lb[i - 1])
         st_dir[i] = -1.0;
      else
         st_dir[i] = st_dir[i - 1];
     }

   double full[];
   ArrayResize(full, n);
   ArrayInitialize(full, 0.0);
   double prev_full = 0.0;

   for(int i = 0; i < n; i++)
     {
      // Consecutive stack: trend (slow→fast), direction (fast→slow)
      int trend_count = 0;
      int trend_sign = 0;
      for(int p = SB_MARIBBON_N_PAIRS - 1; p >= 0; p--)
        {
         bool bull = (mas[i][p] > mas[i][p + 1]);
         int s = bull ? 1 : -1;
         if(trend_sign == 0)
           {
            trend_sign = s;
            trend_count = 1;
           }
         else if(s == trend_sign)
            trend_count++;
         else
            break;
        }
      double trend_s = (double)(trend_sign * trend_count);

      int dir_count = 0;
      int dir_sign = 0;
      for(int p = 0; p < SB_MARIBBON_N_PAIRS; p++)
        {
         bool bull = (mas[i][p] > mas[i][p + 1]);
         int s = bull ? 1 : -1;
         if(dir_sign == 0)
           {
            dir_sign = s;
            dir_count = 1;
           }
         else if(s == dir_sign)
            dir_count++;
         else
            break;
        }
      double dir_s = (double)(dir_sign * dir_count);

      int flat = 0;
      for(int m = 0; m < SB_MARIBBON_N_MA; m++)
         if(mas[i][m] >= low[i] && mas[i][m] <= high[i])
            flat++;

      bool all_bull = true;
      bool all_bear = true;
      for(int p = 0; p < SB_MARIBBON_N_PAIRS; p++)
        {
         if(!(mas[i][p] > mas[i][p + 1]))
            all_bull = false;
         if(!(mas[i][p] < mas[i][p + 1]))
            all_bear = false;
        }
      double cur_full = 0.0;
      if(all_bull)
         cur_full = 1.0;
      else if(all_bear)
         cur_full = -1.0;
      full[i] = cur_full;

      double r_pt = 0.0;
      if(cur_full == 1.0 && prev_full != 1.0)
         r_pt = 1.0;
      else if(cur_full == -1.0 && prev_full != -1.0)
         r_pt = -1.0;
      else if(prev_full == 1.0 && cur_full != 1.0)
         r_pt = -1.0;
      else if(prev_full == -1.0 && cur_full != -1.0)
         r_pt = 1.0;
      prev_full = cur_full;

      // Section means
      int kt = (int)MathAbs(trend_s);
      int nt = MathMax(1, MathMin(kt + 1, SB_MARIBBON_N_MA));
      double sec_t = 0.0;
      for(int m = SB_MARIBBON_N_MA - nt; m < SB_MARIBBON_N_MA; m++)
         sec_t += mas[i][m];
      sec_t /= (double)nt;

      int kd = (int)MathAbs(dir_s);
      int nd = MathMax(1, MathMin(kd + 1, SB_MARIBBON_N_MA));
      double sec_d = 0.0;
      for(int m = 0; m < nd; m++)
         sec_d += mas[i][m];
      sec_d /= (double)nd;

      out[i][0]  = trend_s;
      out[i][1]  = dir_s;
      out[i][2]  = (double)flat;
      out[i][3]  = r_pt;
      out[i][4]  = st_dir[i];
      out[i][5]  = SB_Nan0(SB_MaribbonAtrUnit(open[i]  - sec_t, atr14[i], SB_MARIBBON_OHLC_CLIP));
      out[i][6]  = SB_Nan0(SB_MaribbonAtrUnit(close[i] - sec_t, atr14[i], SB_MARIBBON_OHLC_CLIP));
      out[i][7]  = SB_Nan0(SB_MaribbonAtrUnit(high[i]  - sec_t, atr14[i], SB_MARIBBON_OHLC_CLIP));
      out[i][8]  = SB_Nan0(SB_MaribbonAtrUnit(low[i]   - sec_t, atr14[i], SB_MARIBBON_OHLC_CLIP));
      out[i][9]  = SB_Nan0(SB_MaribbonAtrUnit(open[i]  - sec_d, atr14[i], SB_MARIBBON_OHLC_CLIP));
      out[i][10] = SB_Nan0(SB_MaribbonAtrUnit(close[i] - sec_d, atr14[i], SB_MARIBBON_OHLC_CLIP));
      out[i][11] = SB_Nan0(SB_MaribbonAtrUnit(high[i]  - sec_d, atr14[i], SB_MARIBBON_OHLC_CLIP));
      out[i][12] = SB_Nan0(SB_MaribbonAtrUnit(low[i]   - sec_d, atr14[i], SB_MARIBBON_OHLC_CLIP));
     }
  }

void SB_MaribbonMapLastCompleted(const SBOhlc &h1, const SBOhlc &ltf,
                                 const long dur_sec,
                                 const double &pack[][SB_MARIBBON_BASE],
                                 double &mapped[][SB_MARIBBON_BASE])
  {
   ArrayResize(mapped, h1.n);
   ArrayInitialize(mapped, 0.0);
   int k = -1;
   for(int i = 0; i < h1.n; i++)
     {
      while(k + 1 < ltf.n && (long)ltf.time[k + 1] + dur_sec <= (long)h1.time[i])
         k++;
      if(k < 0)
         continue;
      for(int c = 0; c < SB_MARIBBON_BASE; c++)
         mapped[i][c] = pack[k][c];
     }
  }

bool SB_BuildMaribbonFeatureMatrix(const string symbol, double &features[][SB_MARIBBON_CH],
                                   int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1, m15, m5;
   int want = MathMax(SB_MARIBBON_WARMUP + 512, 2000);
   if(!SB_CopyRatesChrono(symbol, PERIOD_H1, want, h1))
     {
      err = "not enough H1 bars";
      return false;
     }
   n_bars = h1.n;
   ArrayResize(features, n_bars);
   ArrayInitialize(features, 0.0);

   double pack1h[][SB_MARIBBON_BASE];
   SB_MaribbonPackBase(h1.open, h1.high, h1.low, h1.close, pack1h);

   double map15[][SB_MARIBBON_BASE];
   ArrayResize(map15, n_bars);
   ArrayInitialize(map15, 0.0);
   if(SB_CopyRatesChrono(symbol, PERIOD_M15, MathMax(2000, want * 4 + 200), m15))
     {
      double pack15[][SB_MARIBBON_BASE];
      SB_MaribbonPackBase(m15.open, m15.high, m15.low, m15.close, pack15);
      SB_MaribbonMapLastCompleted(h1, m15, SB_MARIBBON_M15_SEC, pack15, map15);
     }

   double map5[][SB_MARIBBON_BASE];
   ArrayResize(map5, n_bars);
   ArrayInitialize(map5, 0.0);
   if(SB_CopyRatesChrono(symbol, PERIOD_M5, MathMax(4000, want * 12 + 400), m5))
     {
      double pack5[][SB_MARIBBON_BASE];
      SB_MaribbonPackBase(m5.open, m5.high, m5.low, m5.close, pack5);
      SB_MaribbonMapLastCompleted(h1, m5, SB_MARIBBON_M5_SEC, pack5, map5);
     }

   for(int i = 0; i < n_bars; i++)
     {
      for(int c = 0; c < SB_MARIBBON_BASE; c++)
         features[i][c] = pack1h[i][c];
      for(int c = 0; c < SB_MARIBBON_BASE; c++)
         features[i][SB_MARIBBON_BASE + c] = map15[i][c];
      for(int c = 0; c < SB_MARIBBON_BASE; c++)
         features[i][2 * SB_MARIBBON_BASE + c] = map5[i][c];
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

// all_blend maribbon slot: 1h base 13 only (no 15m/5m twins).
bool SB_BuildMaribbonBase1hMatrix(const string symbol,
                                  double &features[][SB_MARIBBON_BASE],
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
   SB_MaribbonPackBase(h1.open, h1.high, h1.low, h1.close, features);
   return true;
  }

bool SB_BuildMaribbonBase1hLookbackWindow(const double &features[][SB_MARIBBON_BASE],
                                          const int n_bars, const int lookback,
                                          const int end_bar, float &window[],
                                          string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_MARIBBON_BASE * lookback);
   for(int c = 0; c < SB_MARIBBON_BASE; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
