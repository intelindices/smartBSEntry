//+------------------------------------------------------------------+
//| SmartBS signals — side aggregates only (10 channels)             |
//| Parity with smartbs_engines/signals.py ACTIVE_SIGNAL_SOURCES:    |
//|   dbb, rsi_divergence, trend_pullback, smart_money, macd         |
//+------------------------------------------------------------------+
#ifndef SMARTBS_SIGNALS_FEATURES_MQH
#define SMARTBS_SIGNALS_FEATURES_MQH

#include "DbbFeatures.mqh"
#include "RsiDivergenceFeatures.mqh"
#include "TrendPullbackFeatures.mqh"
#include "SmartMoneyFeatures.mqh"
#include "MacdFeatures.mqh"
#include "FeatureUtils.mqh"

#define SB_SIGNALS_CH      10
#define SB_SIGNALS_WARMUP  SB_TP_WARMUP

// Engine-side aggregate channel indices (long/short pairs).
#define SB_SIG_DBB_LONG              0
#define SB_SIG_DBB_SHORT             1
#define SB_SIG_RSI_LONG              2
#define SB_SIG_RSI_SHORT             3
#define SB_SIG_TP_LONG               4
#define SB_SIG_TP_SHORT              5
#define SB_SIG_SM_LONG               6
#define SB_SIG_SM_SHORT              7
#define SB_SIG_MACD_LONG             8
#define SB_SIG_MACD_SHORT            9

double SB_SigClip01(const double v)
  {
   return SB_Clip(v, 0.0, 1.0);
  }

double SB_SigMax2(const double a, const double b)
  {
   return (a > b) ? a : b;
  }

double SB_SigMax3(const double a, const double b, const double c)
  {
   return SB_SigMax2(a, SB_SigMax2(b, c));
  }

double SB_SigMax4(const double a, const double b, const double c, const double d)
  {
   return SB_SigMax2(SB_SigMax2(a, b), SB_SigMax2(c, d));
  }

bool SB_BuildSignalsFeatureMatrix(const string symbol, double &features[][SB_SIGNALS_CH],
                                  int &n_bars, string &err)
  {
   err = "";
   double dbb[][SB_DBB_CH];
   double rsi[][SB_RSI_DIV_CH];
   double tp[][SB_TP_CH];
   double sm[][SB_SM_CH];
   double macd[][SB_MACD_CH];
   int n_dbb = 0, n_rsi = 0, n_tp = 0, n_sm = 0, n_macd = 0;
   string e2 = "";

   if(!SB_BuildDbbFeatureMatrix(symbol, dbb, n_dbb, err))
      return false;
   if(!SB_BuildRsiDivergenceFeatureMatrix(symbol, rsi, n_rsi, e2))
     {
      err = e2;
      return false;
     }
   if(!SB_BuildTrendPullbackFeatureMatrix(symbol, tp, n_tp, e2))
     {
      err = e2;
      return false;
     }
   if(!SB_BuildSmartMoneyFeatureMatrix(symbol, sm, n_sm, e2))
     {
      err = e2;
      return false;
     }
   if(!SB_BuildMacdFeatureMatrix(symbol, macd, n_macd, e2))
     {
      err = e2;
      return false;
     }

   n_bars = n_dbb;
   if(n_rsi < n_bars) n_bars = n_rsi;
   if(n_tp < n_bars) n_bars = n_tp;
   if(n_sm < n_bars) n_bars = n_sm;
   if(n_macd < n_bars) n_bars = n_macd;
   if(n_bars <= 0)
     {
      err = "signals: empty feature matrices";
      return false;
     }

   const int off_dbb = n_dbb - n_bars;
   const int off_rsi = n_rsi - n_bars;
   const int off_tp = n_tp - n_bars;
   const int off_sm = n_sm - n_bars;
   const int off_macd = n_macd - n_bars;

   ArrayResize(features, n_bars);
   ArrayInitialize(features, 0.0);

   for(int i = 0; i < n_bars; i++)
     {
      // dbb pack6: peak=0, close_in upper_outer=1 ... lower_outer=5; 15m at +6
      double up_outer = SB_SigClip01(dbb[off_dbb + i][1]);
      double lo_outer = SB_SigClip01(dbb[off_dbb + i][5]);
      double up_outer_15 = SB_SigClip01(dbb[off_dbb + i][7]);
      double lo_outer_15 = SB_SigClip01(dbb[off_dbb + i][11]);
      features[i][0] = SB_SigMax2(lo_outer, lo_outer_15);
      features[i][1] = SB_SigMax2(up_outer, up_outer_15);

      double cont_l = SB_SigClip01(rsi[off_rsi + i][20]);
      double cont_s = SB_SigClip01(rsi[off_rsi + i][21]);
      double rev_l  = SB_SigClip01(rsi[off_rsi + i][22]);
      double rev_s  = SB_SigClip01(rsi[off_rsi + i][23]);
      double bull_d = SB_SigClip01(rsi[off_rsi + i][24]);
      double bear_d = SB_SigClip01(rsi[off_rsi + i][25]);
      double sim_up = SB_SigClip01(rsi[off_rsi + i][18]);
      double sim_dn = SB_SigClip01(rsi[off_rsi + i][19]);
      features[i][2] = SB_SigMax4(cont_l, rev_l, bull_d, sim_up);
      features[i][3] = SB_SigMax4(cont_s, rev_s, bear_d, sim_dn);

      features[i][4] = SB_SigClip01(tp[off_tp + i][14] / 1.5);
      features[i][5] = SB_SigClip01(tp[off_tp + i][15] / 1.5);

      double sm_setup_l = SB_SigClip01(sm[off_sm + i][5] / 1.5);
      double sm_setup_s = SB_SigClip01(sm[off_sm + i][6] / 1.5);
      double bos_up = SB_SigClip01(sm[off_sm + i][8]);
      double bos_dn = SB_SigClip01(sm[off_sm + i][9]);
      double choch_up = SB_SigClip01(sm[off_sm + i][10]);
      double choch_dn = SB_SigClip01(sm[off_sm + i][11]);
      features[i][6] = SB_SigMax3(sm_setup_l, choch_up, bos_up);
      features[i][7] = SB_SigMax3(sm_setup_s, choch_dn, bos_dn);

      double cross_up = SB_SigClip01(macd[off_macd + i][11]);
      double cross_dn = SB_SigClip01(macd[off_macd + i][12]);
      double persist = SB_SigClip01((macd[off_macd + i][10] + 1.0) * 0.5);
      features[i][8] = SB_SigMax2(cross_up, persist);
      features[i][9] = SB_SigMax2(cross_dn, 1.0 - persist);
     }
   return true;
  }

bool SB_BuildSignalsLookbackWindow(const double &features[][SB_SIGNALS_CH], const int n_bars,
                                   const int lookback, const int end_bar,
                                   float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_SIGNALS_CH * lookback);
   for(int c = 0; c < SB_SIGNALS_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

int SB_SignalsApplyOnsetGate(const double &features[][SB_SIGNALS_CH], const int n_bars,
                             const int end_bar, const int ai_cls,
                             const double thr, const bool require_agree,
                             string &reason)
  {
   reason = "";
   if(end_bar < 1 || end_bar >= n_bars)
      return 0;

   const int cols_long[5]  = {SB_SIG_DBB_LONG, SB_SIG_RSI_LONG, SB_SIG_TP_LONG,
                              SB_SIG_SM_LONG, SB_SIG_MACD_LONG};
   const int cols_short[5] = {SB_SIG_DBB_SHORT, SB_SIG_RSI_SHORT, SB_SIG_TP_SHORT,
                              SB_SIG_SM_SHORT, SB_SIG_MACD_SHORT};
   const string ids_long[5]  = {"dbb_long", "rsi_divergence_long", "trend_pullback_long",
                                "smart_money_long", "macd_long"};
   const string ids_short[5] = {"dbb_short", "rsi_divergence_short", "trend_pullback_short",
                                "smart_money_short", "macd_short"};

   bool fire_long = false;
   bool fire_short = false;
   double best = -1.0;
   string best_id = "";

   for(int k = 0; k < 5; k++)
     {
      double cur = features[end_bar][cols_long[k]];
      double prev = features[end_bar - 1][cols_long[k]];
      if(cur >= thr && prev < thr)
        {
         fire_long = true;
         if(cur > best)
           {
            best = cur;
            best_id = ids_long[k];
           }
        }
     }
   for(int k = 0; k < 5; k++)
     {
      double cur = features[end_bar][cols_short[k]];
      double prev = features[end_bar - 1][cols_short[k]];
      if(cur >= thr && prev < thr)
        {
         fire_short = true;
         if(cur > best)
           {
            best = cur;
            best_id = ids_short[k];
           }
        }
     }

   if(!fire_long && !fire_short)
      return 0;

   int gated = 0;
   if(require_agree)
     {
      if(fire_long && !fire_short)
        {
         if(ai_cls == 1)
            gated = 1;
        }
      else if(fire_short && !fire_long)
        {
         if(ai_cls == 2)
            gated = 2;
        }
      else if(ai_cls == 1 || ai_cls == 2)
         gated = ai_cls;
     }
   else
     {
      if(ai_cls == 1 || ai_cls == 2)
         gated = ai_cls;
      else if(fire_long && !fire_short)
         gated = 1;
      else if(fire_short && !fire_long)
         gated = 2;
     }

   if(gated != 0)
      reason = best_id;
   return gated;
  }

#endif
