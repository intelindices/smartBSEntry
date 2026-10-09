//+------------------------------------------------------------------+
//| Auto-generated FeatureDispatch from smartbs_engines.catalog     |
//| Do not edit by hand — run: python -m smartbs_engines.catalog --write-mql
//+------------------------------------------------------------------+
#ifndef SMARTBS_FEATURE_DISPATCH_MQH
#define SMARTBS_FEATURE_DISPATCH_MQH

#include "CommonChannels.mqh"
#include "DbbFeatures.mqh"
#include "MacdFeatures.mqh"
#include "MaribbonFeatures.mqh"
#include "RsiDivergenceFeatures.mqh"
#include "SmartMoneyFeatures.mqh"
#include "TrendPullbackFeatures.mqh"
#include "EngineCatalog.mqh"
#include "ChartAllowlist.mqh"

#define SB_MAX_ENGINE_CH SB_MAX_TOTAL_CH

string SB_EngineNormalize(const string eng)
  {
   string e = eng;
   StringTrimLeft(e);
   StringTrimRight(e);
   StringToLower(e);
   return e;
  }

int SB_EngineCoreChannelCount(const string eng)
  {
   string e = SB_EngineNormalize(eng);
   if(e == "common")
      return SB_COMMON_ONLY; // 0 = common-only (no core pack)
   if(e == "dbb")
      return SB_DBB_CH;
   if(e == "macd")
      return SB_MACD_CH;
   if(e == "maribbon")
      return SB_MARIBBON_CH;
   if(e == "rsi_divergence")
      return SB_RSI_DIV_CH;
   if(e == "smart_money")
      return SB_SM_CH;
   if(e == "trend_pullback")
      return SB_TP_CH;
   return -1;
  }

int SB_EngineChannelCount(const string eng)
  {
   string e = SB_EngineNormalize(eng);
   if(e == "common")
      return SB_COMMON_CH;
   if(e == "maribbon")
      return SB_EngineCoreChannelCount(e);
   int core = SB_EngineCoreChannelCount(eng);
   if(core < 0)
      return 0;
   return core + SB_COMMON_CH;
  }

int SB_EngineWarmup(const string eng)
  {
   string e = SB_EngineNormalize(eng);
   int w = 0;
   if(e == "common")
      return SB_COMMON_WARMUP;
   if(e == "dbb")
      w = SB_DBB_WARMUP;
   else if(e == "macd")
      w = SB_MACD_WARMUP;
   else if(e == "maribbon")
      w = SB_MARIBBON_WARMUP;
   else if(e == "rsi_divergence")
      w = SB_RSI_DIV_WARMUP;
   else if(e == "smart_money")
      w = SB_SM_WARMUP;
   else if(e == "trend_pullback")
      w = SB_TP_WARMUP;
   else
      return 0;
   if(e == "maribbon")
      return w;
   return MathMax(w, SB_COMMON_WARMUP);
  }

void SB_ParseEngineList(const string inp, string &out[], int &n)
  {
   n = 0;
   ArrayResize(out, 0);
   string parts[];
   int cnt = StringSplit(inp, ',', parts);
   for(int i = 0; i < cnt; i++)
     {
      string e = SB_EngineNormalize(parts[i]);
      if(StringLen(e) == 0)
         continue;
      ArrayResize(out, n + 1);
      out[n] = e;
      n++;
     }
  }

bool SB_BuildEngineLookbackWindow(const string eng, const string symbol,
                                  const int lookback,
                                  const int end_bar_closed_h1_index_or_use_shift,
                                  float &window[], int &num_inputs, string &err)
  {
   err = "";
   num_inputs = 0;
   string e = SB_EngineNormalize(eng);
   int core_ch = SB_EngineCoreChannelCount(e);
   num_inputs = SB_EngineChannelCount(e);
   if(core_ch < 0 || num_inputs <= 0)
     {
      err = "unknown engine: " + eng;
      return false;
     }
   if(lookback <= 0)
     {
      err = "invalid lookback";
      return false;
     }

   if(e == "common")
      return SB_BuildCommonLookbackWindow(symbol, lookback, window, err);

   int n_bars = 0;
   int end_bar = end_bar_closed_h1_index_or_use_shift;
   bool ok = false;

   if(e == "dbb")
     {
      double features[][SB_NUM_INPUTS];
      if(!SB_BuildDbbFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildDbbLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(e == "macd")
     {
      double features[][SB_MACD_CH];
      if(!SB_BuildMacdFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildMacdLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(e == "maribbon")
     {
      double features[][SB_MARIBBON_CH];
      if(!SB_BuildMaribbonFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildMaribbonLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(e == "rsi_divergence")
     {
      double features[][SB_RSI_DIV_CH];
      if(!SB_BuildRsiDivergenceFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildRsiDivergenceLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(e == "smart_money")
     {
      double features[][SB_SM_CH];
      if(!SB_BuildSmartMoneyFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildSmartMoneyLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(e == "trend_pullback")
     {
      double features[][SB_TP_CH];
      if(!SB_BuildTrendPullbackFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildTrendPullbackLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else
     {
      err = "unknown engine: " + eng;
      return false;
     }

   if(!ok)
      return false;
   if(e == "maribbon")
      return true;
   return SB_ExtendWindowWithCommon(symbol, lookback, end_bar, core_ch, window, err);
  }

#endif
