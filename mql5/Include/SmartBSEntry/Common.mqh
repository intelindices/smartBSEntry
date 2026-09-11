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

bool SB_CopyRatesChrono(const string symbol, const ENUM_TIMEFRAMES tf, const int count, SBOhlc &out)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int got = CopyRates(symbol, tf, 0, count, rates);
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
