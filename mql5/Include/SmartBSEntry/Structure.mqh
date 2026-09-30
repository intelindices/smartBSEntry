//+------------------------------------------------------------------+
//| SmartBS Entry — BOS/ChoCH/FVG/sweep structure                    |
//| Parity with smartbs_engines/smart_money_structure.py             |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ENTRY_STRUCTURE_MQH
#define SMARTBS_ENTRY_STRUCTURE_MQH

#include "Indicators.mqh"
#include "FeatureUtils.mqh"

void SB_BarsSinceDecay(const bool &flag[], const int cap, double &out[])
  {
   SB_BarsSinceFlagDecay(flag, cap, out);
  }

void SB_BarsSinceDecayFromDouble(const double &flag[], const int cap, double &out[])
  {
   int n = ArraySize(flag);
   bool b[];
   ArrayResize(b, n);
   for(int i = 0; i < n; i++)
      b[i] = (flag[i] > 0.5);
   SB_BarsSinceDecay(b, cap, out);
  }

struct SBStructureBlock
  {
   double bos_up_decay[];
   double bos_dn_decay[];
   double choch_up_decay[];
   double choch_dn_decay[];
   double fvg_up_sz[];
   double fvg_dn_sz[];
   double dist_fvg_up[];
   double dist_fvg_dn[];
   double fvg_prox_up[];
   double fvg_prox_dn[];
   double sweep_up[];
   double sweep_dn[];
   double dist_sh[];
   double dist_sl[];
  };

void SB_ComputeStructureBlock(const double &high[], const double &low[], const double &close[],
                              const double &atr14[], const int swing_len, SBStructureBlock &blk)
  {
   int n = ArraySize(close);
   double sh[], sl[];
   SB_SwingLevels(high, low, swing_len, sh, sl);
   for(int i = 0; i < n; i++)
     {
      if(sh[i] == EMPTY_VALUE || !MathIsValidNumber(sh[i]))
         sh[i] = high[0];
      if(sl[i] == EMPTY_VALUE || !MathIsValidNumber(sl[i]))
         sl[i] = low[0];
     }
   double p_sh[], p_sl[];
   SB_PrevArray(sh, p_sh);
   SB_PrevArray(sl, p_sl);

   bool broke_up[], broke_dn[];
   ArrayResize(broke_up, n);
   ArrayResize(broke_dn, n);
   for(int i = 0; i < n; i++)
     {
      broke_up[i] = (close[i] > p_sh[i]);
      broke_dn[i] = (close[i] < p_sl[i]);
     }
   bool prev_up[], prev_dn[];
   ArrayResize(prev_up, n);
   ArrayResize(prev_dn, n);
   prev_up[0] = broke_up[0];
   prev_dn[0] = broke_dn[0];
   for(int i = 1; i < n; i++)
     {
      prev_up[i] = broke_up[i - 1];
      prev_dn[i] = broke_dn[i - 1];
     }

   bool new_bos_up[], new_bos_dn[], choch_up[], choch_dn[];
   ArrayResize(new_bos_up, n);
   ArrayResize(new_bos_dn, n);
   ArrayResize(choch_up, n);
   ArrayResize(choch_dn, n);
   for(int i = 0; i < n; i++)
     {
      new_bos_up[i] = broke_up[i] && !prev_up[i];
      new_bos_dn[i] = broke_dn[i] && !prev_dn[i];
      choch_up[i] = new_bos_up[i] && prev_dn[i];
      choch_dn[i] = new_bos_dn[i] && prev_up[i];
     }

   SB_BarsSinceDecay(new_bos_up, 50, blk.bos_up_decay);
   SB_BarsSinceDecay(new_bos_dn, 50, blk.bos_dn_decay);
   SB_BarsSinceDecay(choch_up, 50, blk.choch_up_decay);
   SB_BarsSinceDecay(choch_dn, 50, blk.choch_dn_decay);

   ArrayResize(blk.sweep_up, n);
   ArrayResize(blk.sweep_dn, n);
   ArrayResize(blk.dist_sh, n);
   ArrayResize(blk.dist_sl, n);
   for(int i = 0; i < n; i++)
     {
      double su = SB_Clip(SB_SafeDiv(high[i] - p_sh[i], atr14[i]), 0.0, 3.0);
      double sd = SB_Clip(SB_SafeDiv(p_sl[i] - low[i], atr14[i]), 0.0, 3.0);
      blk.sweep_up[i] = SB_Nan0(su * ((close[i] <= p_sh[i]) ? 1.0 : 0.0));
      blk.sweep_dn[i] = SB_Nan0(sd * ((close[i] >= p_sl[i]) ? 1.0 : 0.0));
      blk.dist_sh[i] = SB_Nan0(SB_SafeDiv(close[i] - sh[i], atr14[i]));
      blk.dist_sl[i] = SB_Nan0(SB_SafeDiv(close[i] - sl[i], atr14[i]));
     }

   double h2[], l2[], ph[], pl[];
   SB_PrevArray(high, ph);
   SB_PrevArray(low, pl);
   SB_PrevArray(ph, h2);
   SB_PrevArray(pl, l2);

   ArrayResize(blk.fvg_up_sz, n);
   ArrayResize(blk.fvg_dn_sz, n);
   ArrayResize(blk.dist_fvg_up, n);
   ArrayResize(blk.dist_fvg_dn, n);
   ArrayResize(blk.fvg_prox_up, n);
   ArrayResize(blk.fvg_prox_dn, n);
   for(int i = 0; i < n; i++)
     {
      double fvg_up = MathMax(low[i] - h2[i], 0.0);
      double fvg_dn = MathMax(l2[i] - high[i], 0.0);
      double fvg_up_sz = SB_SafeDiv(fvg_up, atr14[i]);
      double fvg_dn_sz = SB_SafeDiv(fvg_dn, atr14[i]);
      double dist_up = (fvg_up > 0.0)
                       ? SB_SafeDiv(close[i] - h2[i], atr14[i])
                       : SB_SafeDiv(close[i] - low[i], atr14[i]);
      double dist_dn = (fvg_dn > 0.0)
                       ? SB_SafeDiv(l2[i] - close[i], atr14[i])
                       : SB_SafeDiv(high[i] - close[i], atr14[i]);
      dist_up = SB_Clip(dist_up, -3.0, 3.0);
      dist_dn = SB_Clip(dist_dn, -3.0, 3.0);
      blk.fvg_up_sz[i] = SB_Nan0(fvg_up_sz);
      blk.fvg_dn_sz[i] = SB_Nan0(fvg_dn_sz);
      blk.dist_fvg_up[i] = SB_Nan0(dist_up);
      blk.dist_fvg_dn[i] = SB_Nan0(dist_dn);
      blk.fvg_prox_up[i] = SB_Clip(1.0 - MathAbs(dist_up) / 2.0, 0.0, 1.0) * (fvg_up_sz > 0.0 ? 1.0 : 0.0);
      blk.fvg_prox_dn[i] = SB_Clip(1.0 - MathAbs(dist_dn) / 2.0, 0.0, 1.0) * (fvg_dn_sz > 0.0 ? 1.0 : 0.0);
     }
  }

#endif
