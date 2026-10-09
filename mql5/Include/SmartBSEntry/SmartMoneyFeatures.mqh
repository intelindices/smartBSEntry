//+------------------------------------------------------------------+
//| SmartBS SmartMoney — 1h MR + mapped 15m structure, 19 channels   |
//| Parity with smartbs_engines/engine_smart_money.py                |
//+------------------------------------------------------------------+
#ifndef SMARTBS_SMART_MONEY_FEATURES_MQH
#define SMARTBS_SMART_MONEY_FEATURES_MQH

#include "Common.mqh"
#include "Indicators.mqh"
#include "FeatureUtils.mqh"
#include "Structure.mqh"

#define SB_SM_CH              20
#define SB_SM_SWING_1H        9
#define SB_SM_SWING_15M       9
#define SB_SM_RANGE_WIN       50
#define SB_SM_DECAY_CAP       50
#define SB_SM_WARMUP          (3 * 80)

void SB_SmBarsSinceLevelChange(const double &level[], const int cap, double &out[])
  {
   int n = ArraySize(level);
   ArrayResize(out, n);
   int last = -1;
   for(int i = 0; i < n; i++)
     {
      if(i > 0 && level[i] != level[i - 1])
         last = i;
      double v = (last < 0) ? (double)cap : MathMin((double)(i - last), (double)cap);
      out[i] = v / (double)cap;
     }
  }

void SB_SmMapStructureTo1h(const datetime &h1_t[], const datetime &m15_t[],
                           const SBStructureBlock &blk15, SBStructureBlock &mapped)
  {
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.bos_up_decay, mapped.bos_up_decay);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.bos_dn_decay, mapped.bos_dn_decay);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.choch_up_decay, mapped.choch_up_decay);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.choch_dn_decay, mapped.choch_dn_decay);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.fvg_up_sz, mapped.fvg_up_sz);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.fvg_dn_sz, mapped.fvg_dn_sz);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.dist_fvg_up, mapped.dist_fvg_up);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.dist_fvg_dn, mapped.dist_fvg_dn);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.fvg_prox_up, mapped.fvg_prox_up);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.fvg_prox_dn, mapped.fvg_prox_dn);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.sweep_up, mapped.sweep_up);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.sweep_dn, mapped.sweep_dn);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.dist_sh, mapped.dist_sh);
   SB_Map15mEndOfHour(h1_t, m15_t, blk15.dist_sl, mapped.dist_sl);
  }

bool SB_BuildSmartMoneyFeatureMatrix(const string symbol, double &features[][SB_SM_CH],
                                     int &n_bars, string &err)
  {
   err = "";
   SBOhlc h1, m15;
   int want = MathMax(SB_SM_WARMUP + 512, 400);
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

   double sh[], sl[];
   SB_SwingLevels(h1.high, h1.low, SB_SM_SWING_1H, sh, sl);
   for(int i = 0; i < n_bars; i++)
     {
      if(sh[i] == EMPTY_VALUE || !MathIsValidNumber(sh[i]))
         sh[i] = h1.high[0];
      if(sl[i] == EMPTY_VALUE || !MathIsValidNumber(sl[i]))
         sl[i] = h1.low[0];
     }
   double p_sh[], p_sl[];
   SB_PrevArray(sh, p_sh);
   SB_PrevArray(sl, p_sl);

   double run_hi[], run_lo[];
   SB_RollingMax(h1.high, SB_SM_RANGE_WIN, run_hi);
   SB_RollingMin(h1.low, SB_SM_RANGE_WIN, run_lo);

   double hhhl[];
   ArrayResize(hhhl, n_bars);
   for(int i = 0; i < n_bars; i++)
     {
      double hh = (sh[i] > p_sh[i]) ? 1.0 : 0.0;
      double hl = (sl[i] > p_sl[i]) ? 1.0 : 0.0;
      hhhl[i] = hh + hl - 1.0;
     }
   double struct_bias[];
   SB_RollingMeanMin1(hhhl, 20, struct_bias);

   double bars_sh[];
   SB_SmBarsSinceLevelChange(sh, SB_SM_DECAY_CAP, bars_sh);

   // 15m structure (or 1h fallback)
   SBStructureBlock m15b;
   if(have15 && m15.n >= SB_SM_SWING_15M * 3)
     {
      double atr15[];
      SB_ATR(m15.high, m15.low, m15.close, 14, atr15);
      SBStructureBlock blk15;
      SB_ComputeStructureBlock(m15.high, m15.low, m15.close, atr15, SB_SM_SWING_15M, blk15);
      SB_SmMapStructureTo1h(h1.time, m15.time, blk15, m15b);
     }
   else
     {
      SB_ComputeStructureBlock(h1.high, h1.low, h1.close, atr14, SB_SM_SWING_1H, m15b);
     }

   for(int i = 0; i < n_bars; i++)
     {
      double span = run_hi[i] - run_lo[i];
      double range_pos = SB_Clip(SB_SafeDiv(h1.close[i] - run_lo[i], span), 0.0, 1.0);
      double mr_stretch = SB_Clip(2.0 * (range_pos - 0.5), -1.0, 1.0);
      double discount = SB_Clip(0.5 - range_pos, 0.0, 0.5) * 2.0;
      double premium  = SB_Clip(range_pos - 0.5, 0.0, 0.5) * 2.0;

      double setup_long = SB_Clip(
         m15b.choch_up_decay[i] * discount * m15b.fvg_prox_up[i], 0.0, 1.5);
      double setup_short = SB_Clip(
         m15b.choch_dn_decay[i] * premium * m15b.fvg_prox_dn[i], 0.0, 1.5);

      // MR (8) — discount/premium used in setup only
      features[i][0]  = SB_Nan0(struct_bias[i]);
      features[i][1]  = SB_Nan0(range_pos);
      features[i][2]  = SB_Nan0(mr_stretch);
      features[i][3]  = SB_Nan0(SB_Clip(SB_SafeDiv(h1.close[i] - sh[i], atr14[i]), -3.0, 3.0) / 3.0);
      features[i][4]  = SB_Nan0(SB_Clip(SB_SafeDiv(h1.close[i] - sl[i], atr14[i]), -3.0, 3.0) / 3.0);
      features[i][5]  = SB_Nan0(setup_long);
      features[i][6]  = SB_Nan0(setup_short);
      features[i][7]  = SB_Nan0(bars_sh[i]);
      // M15 (12) — fvg_prox used in setup only, not exported
      features[i][8]  = SB_Nan0(m15b.bos_up_decay[i]);
      features[i][9]  = SB_Nan0(m15b.bos_dn_decay[i]);
      features[i][10] = SB_Nan0(m15b.choch_up_decay[i]);
      features[i][11] = SB_Nan0(m15b.choch_dn_decay[i]);
      features[i][12] = SB_Nan0(m15b.fvg_up_sz[i]);
      features[i][13] = SB_Nan0(m15b.fvg_dn_sz[i]);
      features[i][14] = SB_Nan0(m15b.dist_fvg_up[i]);
      features[i][15] = SB_Nan0(m15b.dist_fvg_dn[i]);
      features[i][16] = SB_Nan0(m15b.sweep_up[i]);
      features[i][17] = SB_Nan0(m15b.sweep_dn[i]);
      features[i][18] = SB_Nan0(m15b.dist_sh[i]);
      features[i][19] = SB_Nan0(m15b.dist_sl[i]);
     }
   return true;
  }

bool SB_BuildSmartMoneyLookbackWindow(const double &features[][SB_SM_CH], const int n_bars,
                                      const int lookback, const int end_bar,
                                      float &window[], string &err)
  {
   err = "";
   if(end_bar < lookback - 1 || end_bar >= n_bars)
     {
      err = "end_bar out of range for lookback window";
      return false;
     }
   ArrayResize(window, SB_SM_CH * lookback);
   for(int c = 0; c < SB_SM_CH; c++)
      for(int t = 0; t < lookback; t++)
         window[c * lookback + t] = (float)features[end_bar - lookback + 1 + t][c];
   return true;
  }

#endif
