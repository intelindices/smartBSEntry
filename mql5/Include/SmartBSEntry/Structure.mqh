//+------------------------------------------------------------------+
//| SmartBS Entry — BOS/ChoCH/FVG structure (structure_sm.py parity)  |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ENTRY_STRUCTURE_MQH
#define SMARTBS_ENTRY_STRUCTURE_MQH

#include "Indicators.mqh"

void SB_BarsSinceDecay(const bool &flag[], const int cap, double &out[])
  {
   int n = ArraySize(flag);
   ArrayResize(out, n);
   int last = -1;
   for(int i = 0; i < n; i++)
     {
      if(flag[i])
         last = i;
      if(last >= 0)
         out[i] = MathMax(0.0, 1.0 - (i - last) / (double)cap);
      else
         out[i] = 0.0;
     }
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
   double fvg_prox_up[];
   double fvg_prox_dn[];
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

   double h2[], l2[], ph[], pl[];
   SB_PrevArray(high, ph);
   SB_PrevArray(low, pl);
   SB_PrevArray(ph, h2);
   SB_PrevArray(pl, l2);

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
      dist_up = MathMax(-3.0, MathMin(3.0, dist_up));
      dist_dn = MathMax(-3.0, MathMin(3.0, dist_dn));
      blk.fvg_prox_up[i] = MathMax(0.0, MathMin(1.0, 1.0 - MathAbs(dist_up) / 2.0)) * (fvg_up_sz > 0.0 ? 1.0 : 0.0);
      blk.fvg_prox_dn[i] = MathMax(0.0, MathMin(1.0, 1.0 - MathAbs(dist_dn) / 2.0)) * (fvg_dn_sz > 0.0 ? 1.0 : 0.0);
     }
  }

#endif
