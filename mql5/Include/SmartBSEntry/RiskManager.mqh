//+------------------------------------------------------------------+
//| SmartBSEntry — Risk Manager (swing SL, BE/ladder, arm MA)        |
//| Parity targets: smartBS-Bot mechanical SL + arm_entry/arm_exit.  |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ENTRY_RISK_MANAGER_MQH
#define SMARTBS_ENTRY_RISK_MANAGER_MQH

#include "Common.mqh"
#include "Indicators.mqh"

// Last confirmed H1 swing high/low at the closed bar (shift 1).
bool SB_LastSwingR(const string symbol, const int swing_len,
                   double &swing_high, double &swing_low, string &err)
  {
   err = "";
   swing_high = 0.0;
   swing_low = 0.0;
   SBOhlc h1;
   int need = MathMax(200, swing_len * 40 + 64);
   if(!SB_CopyRatesChrono(symbol, PERIOD_H1, need, h1))
     {
      err = "swing: not enough H1";
      return false;
     }
   double sh[], sl[];
   SB_SwingLevels(h1.high, h1.low, swing_len, sh, sl);
   int i = h1.n - 2; // last closed bar
   if(i < 0)
      i = h1.n - 1;
   if(i < 0)
     {
      err = "swing: empty";
      return false;
     }
   swing_high = sh[i];
   swing_low = sl[i];
   if(!MathIsValidNumber(swing_high) || swing_high == EMPTY_VALUE ||
      !MathIsValidNumber(swing_low) || swing_low == EMPTY_VALUE)
     {
      err = "swing: empty level";
      return false;
     }
   return true;
  }

// Closed-bar H1 close + SMA(fast_len). Returns false if not ready.
bool SB_ClosedBarFastMa(const string symbol, const int fast_len,
                        double &close_out, double &ma_out, string &err)
  {
   err = "";
   close_out = 0.0;
   ma_out = 0.0;
   SBOhlc h1;
   int need = MathMax(fast_len + 8, 64);
   if(!SB_CopyRatesChrono(symbol, PERIOD_H1, need, h1))
     {
      err = "ma: not enough H1";
      return false;
     }
   if(h1.n < fast_len + 2)
     {
      err = "ma: warm-up";
      return false;
     }
   double sma[];
   SB_SMA(h1.close, fast_len, sma);
   int i = h1.n - 2; // closed bar
   close_out = h1.close[i];
   ma_out = sma[i];
   if(!MathIsValidNumber(ma_out) || ma_out == EMPTY_VALUE)
     {
      err = "ma: nan";
      return false;
     }
   return true;
  }

bool SB_EntryMaOk(const int side, const double close_px, const double fma)
  {
   if(side > 0)
      return (close_px > fma);
   if(side < 0)
      return (close_px < fma);
   return false;
  }

bool SB_ExitMaConfirm(const int position, const double close_px, const double fma)
  {
   if(position > 0)
      return (close_px < fma);
   if(position < 0)
      return (close_px > fma);
   return false;
  }

// SuperTrend direction at last closed H1 (+1 bull / -1 bear).
// Matches replay_chart/index.html (Wilder ATR length × factor; default 15×3).
bool SB_ClosedBarSuperTrendDir(const string symbol, const int atr_len,
                               const double factor, int &dir_out, string &err)
  {
   err = "";
   dir_out = 1;
   const int L = MathMax(1, atr_len);
   const double f = (factor > 0.0) ? factor : 3.0;
   SBOhlc h1;
   int need = MathMax(L * 20 + 64, 400);
   if(!SB_CopyRatesChrono(symbol, PERIOD_H1, need, h1))
     {
      err = "st: not enough H1";
      return false;
     }
   if(h1.n < L + 3)
     {
      err = "st: warm-up";
      return false;
     }
   int end = h1.n - 2; // last closed bar
   if(end < 1)
      end = h1.n - 1;
   double atr[];
   SB_ATR(h1.high, h1.low, h1.close, L, atr);
   double final_ub = 0.5 * (h1.high[0] + h1.low[0]) + f * atr[0];
   double final_lb = 0.5 * (h1.high[0] + h1.low[0]) - f * atr[0];
   int dir = 1;
   for(int i = 1; i <= end; i++)
     {
      double hl2 = 0.5 * (h1.high[i] + h1.low[i]);
      double bub = hl2 + f * atr[i];
      double blb = hl2 - f * atr[i];
      double prev_c = h1.close[i - 1];
      double prev_ub = final_ub;
      double prev_lb = final_lb;
      final_ub = (bub < prev_ub || prev_c > prev_ub) ? bub : prev_ub;
      final_lb = (blb > prev_lb || prev_c < prev_lb) ? blb : prev_lb;
      if(h1.close[i] > prev_ub)
         dir = 1;
      else if(h1.close[i] < prev_lb)
         dir = -1;
     }
   dir_out = dir;
   return true;
  }

// ATR(14) at last closed H1 bar.
bool SB_ClosedBarAtr(const string symbol, const int atr_len, double &atr_out, string &err)
  {
   err = "";
   atr_out = 0.0;
   SBOhlc h1;
   int need = MathMax(atr_len * 5 + 64, 200);
   if(!SB_CopyRatesChrono(symbol, PERIOD_H1, need, h1))
     {
      err = "atr: not enough H1";
      return false;
     }
   double atr[];
   SB_ATR(h1.high, h1.low, h1.close, atr_len, atr);
   int i = h1.n - 2;
   atr_out = atr[i];
   if(!MathIsValidNumber(atr_out) || atr_out <= 0.0)
     {
      err = "atr: invalid";
      return false;
     }
   return true;
  }

// Build swing stop. When validate_r=false: geometry only (no chase/ATR/pct caps).
bool SB_MakeSwingStop(const int side, const double entry,
                      const int swing_len, const double r_min_atr, const double r_max_atr,
                      const double r_max_pct, const double chase_frac,
                      double &sl_out, double &r_out, string &err,
                      const bool validate_r=true)
  {
   sl_out = 0.0;
   r_out = 0.0;
   double sh = 0.0, slv = 0.0;
   if(!SB_LastSwingR(_Symbol, swing_len, sh, slv, err))
      return false;
   double atr = 0.0;
   if(validate_r)
     {
      string aerr;
      if(!SB_ClosedBarAtr(_Symbol, 14, atr, aerr))
        {
         err = aerr;
         return false;
        }
     }
   double rng = sh - slv;
   if(rng <= 0.0)
     {
      err = "swing: bad range";
      return false;
     }
   double sl = 0.0;
   double r = 0.0;
   if(side > 0)
     {
      if(slv >= entry)
        {
         err = "swing: long SL above entry";
         return false;
        }
      sl = slv;
      r = entry - slv;
      if(validate_r && chase_frac > 0.0 && (r / rng) > chase_frac)
        {
         err = "swing: chase";
         return false;
        }
     }
   else if(side < 0)
     {
      if(sh <= entry)
        {
         err = "swing: short SL below entry";
         return false;
        }
      sl = sh;
      r = sh - entry;
      if(validate_r && chase_frac > 0.0 && (r / rng) > chase_frac)
        {
         err = "swing: chase";
         return false;
        }
     }
   else
     {
      err = "swing: flat";
      return false;
     }
   if(r <= 0.0)
     {
      err = "swing: R<=0";
      return false;
     }
   if(validate_r)
     {
      if(atr > 0.0 && r < r_min_atr * atr)
        {
         err = "swing: R too tight";
         return false;
        }
      if(atr > 0.0 && r > r_max_atr * atr)
        {
         err = "swing: R too wide (ATR)";
         return false;
        }
      if(r_max_pct > 0.0 && r > entry * r_max_pct)
        {
         err = "swing: R too wide (pct)";
         return false;
        }
     }
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   sl_out = NormalizeDouble(sl, digits);
   r_out = r;
   return true;
  }

#endif
