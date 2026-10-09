//+------------------------------------------------------------------+
//| SmartBS — shared OHLC copy helpers                               |
//+------------------------------------------------------------------+
#ifndef SMARTBS_COMMON_MQH
#define SMARTBS_COMMON_MQH

struct SBOhlc
  {
   datetime time[];
   double   open[];
   double   high[];
   double   low[];
   double   close[];
   int      n;
  };

// Chart/EA primary TF: 5m, 15m, 1h, or 4h (parity with replay_chart/index.html).
ENUM_TIMEFRAMES g_sb_primary_tf = PERIOD_M5;

void SB_SetPrimaryTf(const ENUM_TIMEFRAMES tf)
  {
   if(tf == PERIOD_M5 || tf == PERIOD_M15 || tf == PERIOD_H4 || tf == PERIOD_H1)
      g_sb_primary_tf = tf;
   else
      g_sb_primary_tf = PERIOD_M5;
  }

ENUM_TIMEFRAMES SB_PrimaryTf(void)
  {
   return g_sb_primary_tf;
  }

string SB_TfTag(void)
  {
   if(g_sb_primary_tf == PERIOD_M5)
      return "5m";
   if(g_sb_primary_tf == PERIOD_M15)
      return "15m";
   if(g_sb_primary_tf == PERIOD_H4)
      return "4h";
   return "1h";
  }

int SB_PrimaryBarSeconds(void)
  {
   return (int)PeriodSeconds(g_sb_primary_tf);
  }

int SB_CopyRatesFloor(const ENUM_TIMEFRAMES tf)
  {
   // Replay export warmup (_export_replay_chart._warm_bars_for_interval).
   // Short windows (400–2000) on M5 leave EMA/SMA half-warm vs Python.
   if(tf == PERIOD_M5)
      return 12000;
   if(tf == PERIOD_M15)
      return 8000;
   if(tf == PERIOD_H1)
      return 3000;
   if(tf == PERIOD_H4)
      return 1500;
   return 0;
  }

bool SB_CopyRatesChrono(const string symbol, const ENUM_TIMEFRAMES tf, const int count, SBOhlc &out)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   const int need = MathMax(count, SB_CopyRatesFloor(tf));
   int got = CopyRates(symbol, tf, 0, need, rates);
   if(got < 10)
      return false;
   out.n = got;
   ArrayResize(out.time, got);
   ArrayResize(out.open, got);
   ArrayResize(out.high, got);
   ArrayResize(out.low, got);
   ArrayResize(out.close, got);
   for(int i = 0; i < got; i++)
     {
      out.time[i]  = rates[i].time;
      out.open[i]  = rates[i].open;
      out.high[i]  = rates[i].high;
      out.low[i]   = rates[i].low;
      out.close[i] = rates[i].close;
     }
   return true;
  }

#endif
