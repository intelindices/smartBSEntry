//+------------------------------------------------------------------+
//| SmartBS RSI Divergence — dual-TF RSI setups, 25 channels         |
//| Parity with smartbs_engines/engine_rsi_divergence.py             |
//+------------------------------------------------------------------+
#ifndef SMARTBS_RSI_DIVERGENCE_FEATURES_MQH
#define SMARTBS_RSI_DIVERGENCE_FEATURES_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"

#define SB_RSI_DIV_CH              26
#define SB_RSI_DIV_PERIOD          14
#define SB_RSI_DIV_LO              30.0
#define SB_RSI_DIV_HI              70.0
#define SB_RSI_DIV_EMA_FAST        20
#define SB_RSI_DIV_EMA_SLOW        50
#define SB_RSI_DIV_SWING           5
#define SB_RSI_DIV_DECAY           24
#define SB_RSI_DIV_FALLBACK_PERIOD 7
#define SB_RSI_DIV_WARMUP          (3 * 50)

double SB_RsiSoftBand(const double rsi, const double lo, const double hi)
  {
   double mid = 0.5 * (lo + hi);
   double half = 0.5 * (hi - lo);
   return SB_Clip(1.0 - MathAbs(rsi - mid) / half, 0.0, 1.0);
  }

void SB_RsiDivEvents(const double &high[], const double &low[], const double &rsi[],
                     const int swing_len, bool &bull[], bool &bear[])
  {
   int n = ArraySize(rsi);
   ArrayResize(bull, n);
   ArrayResize(bear, n);
   ArrayInitialize(bull, false);
   ArrayInitialize(bear, false);

   double ph[], pl[];
   SB_PivotHigh(high, swing_len, swing_len, ph);
   SB_PivotLow(low, swing_len, swing_len, pl);

   double prev_low_px = EMPTY_VALUE, prev_low_rsi = EMPTY_VALUE;
   double prev_high_px = EMPTY_VALUE, prev_high_rsi = EMPTY_VALUE;
   for(int i = 0; i < n; i++)
     {
      if(pl[i] != EMPTY_VALUE && MathIsValidNumber(pl[i]))
        {
         int pi = i - swing_len;
         if(pi >= 0)
           {
            double px = low[pi];
            double rv = rsi[pi];
            if(prev_low_px != EMPTY_VALUE && MathIsValidNumber(prev_low_px) &&
               MathIsValidNumber(prev_low_rsi) &&
               px < prev_low_px && rv > prev_low_rsi + 1e-6)
               bull[i] = true;
            prev_low_px = px;
            prev_low_rsi = rv;
           }
        }
      if(ph[i] != EMPTY_VALUE && MathIsValidNumber(ph[i]))
        {
         int pi = i - swing_len;
         if(pi >= 0)
           {
            double px = high[pi];
            double rv = rsi[pi];
            if(prev_high_px != EMPTY_VALUE && MathIsValidNumber(prev_high_px) &&
               MathIsValidNumber(prev_high_rsi) &&
               px > prev_high_px && rv < prev_high_rsi - 1e-6)
               bear[i] = true;
            prev_high_px = px;
            prev_high_rsi = rv;
           }
        }
     }
  }

bool SB_BuildRsiDivergenceFeatureMatrix(const string symbol, double &features[][SB_RSI_DIV_CH],
                                        int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1, m15;
   int want = MathMax(SB_RSI_DIV_WARMUP + 512, 400);
   if(!SB_CopyRatesChrono(symbol, SB_PrimaryTf(), want, h1))
     {
      err = "not enough H1 bars";
      return false;
     }
   bool have15 = SB_CopyRatesChrono(symbol, PERIOD_M15, MathMax(2000, want * 4 + 200), m15);

   n_bars = h1.n;
   ArrayResize(features, n_bars);
   ArrayInitialize(features, 0.0);

   double atr14[];
   SB_ATR(h1.high, h1.low, h1.close, 14, atr14);

   double ema_f[], ema_s[];
   SB_EMA(h1.close, SB_RSI_DIV_EMA_FAST, ema_f);
   SB_EMA(h1.close, SB_RSI_DIV_EMA_SLOW, ema_s);

   double rsi_maj[];
   SB_RSI(h1.close, SB_RSI_DIV_PERIOD, rsi_maj);

   double rsi_min[];
   bool used15 = false;
   if(have15 && m15.n >= SB_RSI_DIV_PERIOD + 5)
     {
      double rsi15[];
      SB_RSI(m15.close, SB_RSI_DIV_PERIOD, rsi15);
      SB_Map15mEndOfHour(h1.time, m15.time, rsi15, rsi_min);
      used15 = true;
     }
   else
     {
      SB_RSI(h1.close, SB_RSI_DIV_FALLBACK_PERIOD, rsi_min);
     }

   double stack[], stack_slope[];
   ArrayResize(stack, n_bars);
   for(int i = 0; i < n_bars; i++)
      stack[i] = ema_f[i] - ema_s[i];
   SB_SlopeNorm(stack, atr14, 1, stack_slope);

   bool bull_div[], bear_div[];
   SB_RsiDivEvents(h1.high, h1.low, rsi_maj, SB_RSI_DIV_SWING, bull_div, bear_div);

   bool maj_up[], maj_dn[], min_up[], min_dn[];
   bool sim_up[], sim_dn[], cont_long[], cont_short[], rev_long[], rev_short[];
   ArrayResize(maj_up, n_bars);
   ArrayResize(maj_dn, n_bars);
   ArrayResize(min_up, n_bars);
   ArrayResize(min_dn, n_bars);
   ArrayResize(sim_up, n_bars);
   ArrayResize(sim_dn, n_bars);
   ArrayResize(cont_long, n_bars);
   ArrayResize(cont_short, n_bars);
   ArrayResize(rev_long, n_bars);
   ArrayResize(rev_short, n_bars);

   for(int i = 0; i < n_bars; i++)
     {
      double p_maj = (i > 0) ? rsi_maj[i - 1] : rsi_maj[i];
      double p_min = (i > 0) ? rsi_min[i - 1] : rsi_min[i];
      maj_up[i] = (p_maj <= SB_RSI_DIV_LO) && (rsi_maj[i] > SB_RSI_DIV_LO);
      maj_dn[i] = (p_maj >= SB_RSI_DIV_HI) && (rsi_maj[i] < SB_RSI_DIV_HI);
      min_up[i] = (p_min <= SB_RSI_DIV_LO) && (rsi_min[i] > SB_RSI_DIV_LO);
      min_dn[i] = (p_min >= SB_RSI_DIV_HI) && (rsi_min[i] < SB_RSI_DIV_HI);
      sim_up[i] = maj_up[i] && min_up[i];
      sim_dn[i] = maj_dn[i] && min_dn[i];

      bool uptrend = ema_f[i] > ema_s[i];
      bool downtrend = ema_f[i] < ema_s[i];
      bool maj_band = (rsi_maj[i] > SB_RSI_DIV_LO) && (rsi_maj[i] < SB_RSI_DIV_HI);
      cont_long[i]  = uptrend && maj_band && min_up[i];
      cont_short[i] = downtrend && maj_band && min_dn[i];
      rev_long[i]   = sim_up[i] && downtrend;
      rev_short[i]  = sim_dn[i] && uptrend;
     }

   double d_maj_up[], d_maj_dn[], d_min_up[], d_min_dn[];
   double d_sim_up[], d_sim_dn[], d_cont_l[], d_cont_s[], d_rev_l[], d_rev_s[];
   double d_bull[], d_bear[];
   SB_BarsSinceFlagDecay(maj_up, SB_RSI_DIV_DECAY, d_maj_up);
   SB_BarsSinceFlagDecay(maj_dn, SB_RSI_DIV_DECAY, d_maj_dn);
   SB_BarsSinceFlagDecay(min_up, SB_RSI_DIV_DECAY, d_min_up);
   SB_BarsSinceFlagDecay(min_dn, SB_RSI_DIV_DECAY, d_min_dn);
   SB_BarsSinceFlagDecay(sim_up, SB_RSI_DIV_DECAY, d_sim_up);
   SB_BarsSinceFlagDecay(sim_dn, SB_RSI_DIV_DECAY, d_sim_dn);
   SB_BarsSinceFlagDecay(cont_long, SB_RSI_DIV_DECAY, d_cont_l);
   SB_BarsSinceFlagDecay(cont_short, SB_RSI_DIV_DECAY, d_cont_s);
   SB_BarsSinceFlagDecay(rev_long, SB_RSI_DIV_DECAY, d_rev_l);
   SB_BarsSinceFlagDecay(rev_short, SB_RSI_DIV_DECAY, d_rev_s);
   SB_BarsSinceFlagDecay(bull_div, SB_RSI_DIV_DECAY, d_bull);
   SB_BarsSinceFlagDecay(bear_div, SB_RSI_DIV_DECAY, d_bear);

   double has15 = used15 ? 1.0 : 0.0;
   for(int i = 0; i < n_bars; i++)
     {
      // Trend (6)
      features[i][0]  = SB_Clip(SB_SafeDiv(stack[i], atr14[i]), -3.0, 3.0);
      features[i][1]  = SB_Nan0(stack_slope[i]);
      features[i][2]  = SB_Nan0(MathTanh(SB_SafeDiv(h1.close[i] - ema_f[i], atr14[i])));
      features[i][3]  = SB_Nan0(MathTanh(SB_SafeDiv(stack[i], atr14[i])));
      features[i][4]  = SB_Nan0(MathTanh(SB_SafeDiv(-stack[i], atr14[i])));
      features[i][5]  = has15;
      // Major RSI (6)
      features[i][6]  = SB_Nan0((rsi_maj[i] - 50.0) / 50.0);
      features[i][7]  = SB_Nan0(SB_RsiSoftBand(rsi_maj[i], SB_RSI_DIV_LO, SB_RSI_DIV_HI));
      features[i][8]  = SB_Clip((rsi_maj[i] - SB_RSI_DIV_LO) / 40.0, -1.0, 1.0);
      features[i][9]  = SB_Clip((SB_RSI_DIV_HI - rsi_maj[i]) / 40.0, -1.0, 1.0);
      features[i][10] = SB_Nan0(d_maj_up[i]);
      features[i][11] = SB_Nan0(d_maj_dn[i]);
      // Minor RSI (6)
      features[i][12] = SB_Nan0((rsi_min[i] - 50.0) / 50.0);
      features[i][13] = SB_Nan0(d_min_up[i]);
      features[i][14] = SB_Nan0(d_min_dn[i]);
      features[i][15] = SB_Nan0(SB_SafeDiv(rsi_min[i] - rsi_maj[i], 100.0));
      features[i][16] = SB_Clip((35.0 - rsi_min[i]) / 35.0, 0.0, 1.0);
      features[i][17] = SB_Clip((rsi_min[i] - 65.0) / 35.0, 0.0, 1.0);
      // Setups (8)
      features[i][18] = SB_Nan0(d_sim_up[i]);
      features[i][19] = SB_Nan0(d_sim_dn[i]);
      features[i][20] = SB_Nan0(d_cont_l[i]);
      features[i][21] = SB_Nan0(d_cont_s[i]);
      features[i][22] = SB_Nan0(d_rev_l[i]);
      features[i][23] = SB_Nan0(d_rev_s[i]);
      features[i][24] = SB_Nan0(d_bull[i]);
      features[i][25] = SB_Nan0(d_bear[i]);
     }
   return true;
  }

bool SB_BuildRsiDivergenceLookbackWindow(const double &features[][SB_RSI_DIV_CH], const int n_bars,
                                         const int lookback, const int end_bar,
                                         float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_RSI_DIV_CH * lookback);
   for(int c = 0; c < SB_RSI_DIV_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
