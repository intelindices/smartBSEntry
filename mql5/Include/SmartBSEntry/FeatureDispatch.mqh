//+------------------------------------------------------------------+
//| SmartBS — dispatch lookback windows for all ST engines           |
//| Most engines append the shared common pack; all_blend embeds it, |
//| and pivot_engine / maribbon are core-only.                       |
//+------------------------------------------------------------------+
#ifndef SMARTBS_FEATURE_DISPATCH_MQH
#define SMARTBS_FEATURE_DISPATCH_MQH

#include "DbbFeatures.mqh"
#include "MacdFeatures.mqh"
#include "CandleFeatures.mqh"
#include "TrendPullbackFeatures.mqh"
#include "MaribbonFeatures.mqh"
#include "SmartMoneyFeatures.mqh"
#include "RsiDivergenceFeatures.mqh"
#include "RegimeFeatures.mqh"
#include "PivotFeatures.mqh"
#include "SignalsFeatures.mqh"
#include "CommonChannels.mqh"
#include "AllBlendFeatures.mqh"
#include "EngineCatalog.mqh"

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
   if(e == "candle")
      return SB_CANDLE_CH;
   if(e == "trend_pullback")
      return SB_TP_CH;
   if(e == "maribbon")
      return SB_MARIBBON_CH;
   if(e == "smart_money")
      return SB_SM_CH;
   if(e == "rsi_divergence")
      return SB_RSI_DIV_CH;
   if(e == "regime_engine")
      return SB_REGIME_CH;
   if(e == "pivot_engine")
      return SB_PIVOT_CH;
   if(e == "signals")
      return SB_SIGNALS_CH;
   if(e == "all_blend")
      return SB_ALL_BLEND_CH; // 151 core (common embedded separately)
   return -1;
  }

int SB_EngineChannelCount(const string eng)
  {
   string e = SB_EngineNormalize(eng);
   if(e == "common")
      return SB_COMMON_CH; // common-only: 16 channels (not core+common)
   if(e == "pivot_engine")
      return SB_PIVOT_CH; // core-only: no common pack
   if(e == "maribbon")
      return SB_MARIBBON_CH; // core-only: no common pack
   if(e == "all_blend")
      return SB_ALL_BLEND_TOTAL_CH; // 167 = 151 core + 16 common (embedded)
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
   if(e == "all_blend")
      return SB_ALL_BLEND_WARMUP; // max(sources, common); currently 720
   if(e == "dbb")
      w = SB_DBB_WARMUP;
   else if(e == "macd")
      w = SB_MACD_WARMUP;
   else if(e == "candle")
      w = SB_CANDLE_WARMUP;
   else if(e == "trend_pullback")
      w = SB_TP_WARMUP;
   else if(e == "maribbon")
      w = SB_MARIBBON_WARMUP;
   else if(e == "smart_money")
      w = SB_SM_WARMUP;
   else if(e == "rsi_divergence")
      w = SB_RSI_DIV_WARMUP;
   else if(e == "regime_engine")
      w = SB_REGIME_WARMUP;
   else if(e == "pivot_engine")
      w = SB_PIVOT_WARMUP;
   else if(e == "signals")
      w = SB_SIGNALS_WARMUP;
   else
      return 0;
   if(e == "pivot_engine" || e == "maribbon")
      return w; // core-only: no common warmup floor
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

   // common-only: 16 channels, no extra append.
   if(e == "common")
      return SB_BuildCommonLookbackWindow(symbol, lookback, window, err);

   // all_blend embeds common itself — do not SB_ExtendWindowWithCommon after.
   if(e == "all_blend")
     {
      if(!SB_BuildAllBlendLookbackWindow(symbol, lookback,
                                         end_bar_closed_h1_index_or_use_shift,
                                         window, err))
         return false;
      return true;
     }

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
   else if(e == "candle")
     {
      double features[][SB_CANDLE_CH];
      if(!SB_BuildCandleFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildCandleLookbackWindow(features, n_bars, lookback, end_bar, window, err);
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
   else if(e == "maribbon")
     {
      double features[][SB_MARIBBON_CH];
      if(!SB_BuildMaribbonFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildMaribbonLookbackWindow(features, n_bars, lookback, end_bar, window, err);
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
   else if(e == "rsi_divergence")
     {
      double features[][SB_RSI_DIV_CH];
      if(!SB_BuildRsiDivergenceFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildRsiDivergenceLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(e == "regime_engine")
     {
      double features[][SB_REGIME_CH];
      if(!SB_BuildRegimeFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildRegimeLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(e == "pivot_engine")
     {
      double features[][SB_PIVOT_CH];
      if(!SB_BuildPivotFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildPivotLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(e == "signals")
     {
      double features[][SB_SIGNALS_CH];
      if(!SB_BuildSignalsFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildSignalsLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else
     {
      err = "unknown engine: " + eng;
      return false;
     }

   if(!ok)
      return false;
   // pivot_engine / maribbon are core-only (no common pack).
   if(e == "pivot_engine" || e == "maribbon")
      return true;
   return SB_ExtendWindowWithCommon(symbol, lookback, end_bar, core_ch, window, err);
  }

#endif
