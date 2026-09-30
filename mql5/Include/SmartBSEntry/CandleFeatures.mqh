//+------------------------------------------------------------------+
//| SmartBS Candle — price rejection / pin / engulf, 20 channels     |
//| Parity with smartbs_engines/engine_candle.py                     |
//+------------------------------------------------------------------+
#ifndef SMARTBS_CANDLE_FEATURES_MQH
#define SMARTBS_CANDLE_FEATURES_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"

#define SB_CANDLE_CH              20
#define SB_CANDLE_RANGE_WIN       50
#define SB_CANDLE_RANGE_ATR_Z     50
#define SB_CANDLE_REJECT_THRESH   0.35
#define SB_CANDLE_REJECT_CAP      20
#define SB_CANDLE_WICK_BODY_CLIP  5.0
#define SB_CANDLE_RANGE_Z_CLIP    3.0
#define SB_CANDLE_WARMUP          (3 * 50)

bool SB_BuildCandleFeatureMatrix(const string symbol, double &features[][SB_CANDLE_CH],
                                 int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1;
   int want = MathMax(SB_CANDLE_WARMUP + 512, 400);
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

   double run_hi[], run_lo[];
   SB_RollingMax(h1.high, SB_CANDLE_RANGE_WIN, run_hi);
   SB_RollingMin(h1.low, SB_CANDLE_RANGE_WIN, run_lo);

   double range_atr[], range_mu[], range_sd[];
   ArrayResize(range_atr, n_bars);
   for(int i = 0; i < n_bars; i++)
     {
      double rng = MathMax(h1.high[i] - h1.low[i], 1e-12);
      range_atr[i] = SB_SafeDiv(rng, atr14[i]);
     }
   SB_SMA_Strict(range_atr, SB_CANDLE_RANGE_ATR_Z, range_mu);
   SB_StdevPop(range_atr, SB_CANDLE_RANGE_ATR_Z, range_sd);

   double reject_bull[], reject_bear[];
   ArrayResize(reject_bull, n_bars);
   ArrayResize(reject_bear, n_bars);

   for(int i = 0; i < n_bars; i++)
     {
      double o = h1.open[i], c = h1.close[i], h = h1.high[i], l = h1.low[i];
      double rng = MathMax(h - l, 1e-12);
      double body = MathAbs(c - o);
      double body_frac = SB_Clip(SB_SafeDiv(body, rng), 0.0, 1.0);
      double top = MathMax(o, c);
      double bot = MathMin(o, c);
      double up_wick_frac = SB_Clip(SB_SafeDiv(h - top, rng), 0.0, 1.0);
      double dn_wick_frac = SB_Clip(SB_SafeDiv(bot - l, rng), 0.0, 1.0);
      double up_wick = h - top;
      double dn_wick = bot - l;
      double close_pos = SB_Clip(SB_SafeDiv(c - l, rng), 0.0, 1.0);
      double mid = 0.5 * (h + l);

      double body_eps = MathMax(body, atr14[i] * 0.01);
      double wick_body_up = SB_Clip(SB_SafeDiv(up_wick, body_eps), 0.0, SB_CANDLE_WICK_BODY_CLIP) / SB_CANDLE_WICK_BODY_CLIP;
      double wick_body_dn = SB_Clip(SB_SafeDiv(dn_wick, body_eps), 0.0, SB_CANDLE_WICK_BODY_CLIP) / SB_CANDLE_WICK_BODY_CLIP;

      reject_bull[i] = SB_Clip(dn_wick_frac * (1.0 - body_frac) * close_pos, 0.0, 1.5);
      reject_bear[i] = SB_Clip(up_wick_frac * (1.0 - body_frac) * (1.0 - close_pos), 0.0, 1.5);

      double wick_up_atr = SB_SafeDiv(up_wick, atr14[i]);
      double wick_dn_atr = SB_SafeDiv(dn_wick, atr14[i]);

      double p_high = (i > 0) ? h1.high[i - 1] : h;
      double p_low  = (i > 0) ? h1.low[i - 1]  : l;
      bool new_high = (h > p_high);
      bool new_low  = (l < p_low);
      double failed_up = SB_Clip(wick_up_atr * (new_high ? 1.0 : 0.0) * ((c < mid) ? 1.0 : 0.0), 0.0, 3.0);
      double failed_dn = SB_Clip(wick_dn_atr * (new_low ? 1.0 : 0.0) * ((c > mid) ? 1.0 : 0.0), 0.0, 3.0);

      double range_pos = SB_Clip(SB_SafeDiv(c - run_lo[i], run_hi[i] - run_lo[i]), 0.0, 1.0);
      double setup_long  = SB_Clip(reject_bull[i] * (1.0 - range_pos), 0.0, 1.5);
      double setup_short = SB_Clip(reject_bear[i] * range_pos, 0.0, 1.5);

      double z = SB_SafeDiv(range_atr[i] - range_mu[i], range_sd[i]);
      z = SB_Clip(z, -SB_CANDLE_RANGE_Z_CLIP, SB_CANDLE_RANGE_Z_CLIP) / SB_CANDLE_RANGE_Z_CLIP;

      double p_o = (i > 0) ? h1.open[i - 1] : o;
      double p_c = (i > 0) ? h1.close[i - 1] : c;
      double prev_bear = MathMax(p_o - p_c, 0.0);
      double prev_bull = MathMax(p_c - p_o, 0.0);
      double curr_bull = MathMax(c - o, 0.0);
      double curr_bear = MathMax(o - c, 0.0);
      double engulf_bull = SB_Clip(SB_SafeDiv(curr_bull, prev_bear + atr14[i] * 0.01), 0.0, 2.0) / 2.0
                           * ((p_c < p_o) ? 1.0 : 0.0) * ((c > o) ? 1.0 : 0.0);
      double engulf_bear = SB_Clip(SB_SafeDiv(curr_bear, prev_bull + atr14[i] * 0.01), 0.0, 2.0) / 2.0
                           * ((p_c > p_o) ? 1.0 : 0.0) * ((c < o) ? 1.0 : 0.0);

      features[i][0]  = SB_Nan0(body_frac);
      features[i][1]  = SB_Nan0(up_wick_frac);
      features[i][2]  = SB_Nan0(dn_wick_frac);
      features[i][3]  = SB_Nan0(wick_body_up);
      features[i][4]  = SB_Nan0(wick_body_dn);
      features[i][5]  = SB_Nan0(reject_bull[i]);
      features[i][6]  = SB_Nan0(reject_bear[i]);
      features[i][7]  = SB_Nan0(wick_up_atr);
      features[i][8]  = SB_Nan0(wick_dn_atr);
      features[i][9]  = SB_Nan0(close_pos);
      features[i][10] = SB_Nan0(failed_up);
      features[i][11] = SB_Nan0(failed_dn);
      // 12-13 filled after decay pass
      features[i][14] = SB_Nan0(setup_long);
      features[i][15] = SB_Nan0(setup_short);
      features[i][16] = SB_Nan0(z);
      features[i][17] = SB_Nan0(SB_SafeDiv(body, atr14[i]));
      features[i][18] = SB_Nan0(engulf_bull);
      features[i][19] = SB_Nan0(engulf_bear);
     }

   double bs_bull[], bs_bear[];
   SB_BarsSinceScoreDecay(reject_bull, SB_CANDLE_REJECT_THRESH, SB_CANDLE_REJECT_CAP, bs_bull);
   SB_BarsSinceScoreDecay(reject_bear, SB_CANDLE_REJECT_THRESH, SB_CANDLE_REJECT_CAP, bs_bear);
   for(int i = 0; i < n_bars; i++)
     {
      features[i][12] = SB_Nan0(bs_bull[i]);
      features[i][13] = SB_Nan0(bs_bear[i]);
     }
   return true;
  }

bool SB_BuildCandleLookbackWindow(const double &features[][SB_CANDLE_CH], const int n_bars,
                                  const int lookback, const int end_bar,
                                  float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_CANDLE_CH * lookback);
   for(int c = 0; c < SB_CANDLE_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
