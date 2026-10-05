//+------------------------------------------------------------------+
//| SmartBS all_blend — 99 core + 16 common = 115 channels       |
//| Auto-generated from smartbs_engines/engine_all_blend.py          |
//| Do not edit by hand — run:                                      |
//|   python smartbs_engines/_gen_all_blend_features_mqh.py         |
//+------------------------------------------------------------------+
#ifndef SMARTBS_ALL_BLEND_FEATURES_MQH
#define SMARTBS_ALL_BLEND_FEATURES_MQH

#include "RegimeFeatures.mqh"
#include "MaribbonFeatures.mqh"
#include "DbbFeatures.mqh"
#include "TrendPullbackFeatures.mqh"
#include "SmartMoneyFeatures.mqh"
#include "MacdFeatures.mqh"
#include "CandleFeatures.mqh"
#include "RsiDivergenceFeatures.mqh"
#include "CommonChannels.mqh"
#include "EngineCatalog.mqh"

#define SB_ALL_BLEND_CORE_CH   99
#define SB_ALL_BLEND_TOTAL_CH  115
#define SB_ALL_BLEND_WARMUP    720

// Source engine ids for the static plan (matches ALL_BLEND_SOURCES order).
#define SB_AB_SRC_REGIME    0
#define SB_AB_SRC_MARIBBON  1
#define SB_AB_SRC_DBB       2
#define SB_AB_SRC_TP        3
#define SB_AB_SRC_SM        4
#define SB_AB_SRC_MACD      5
#define SB_AB_SRC_CANDLE    6
#define SB_AB_SRC_RSI_DIV   7
#define SB_AB_SRC_COUNT     8

// Per core output column: (source_engine_id, local_col). First-name-wins plan.
static const int SB_ALL_BLEND_PLAN_SRC[SB_ALL_BLEND_CORE_CH] =
  {
   SB_AB_SRC_REGIME, SB_AB_SRC_REGIME, SB_AB_SRC_REGIME, SB_AB_SRC_REGIME, SB_AB_SRC_REGIME, SB_AB_SRC_REGIME, SB_AB_SRC_REGIME, SB_AB_SRC_REGIME,
   SB_AB_SRC_REGIME, SB_AB_SRC_REGIME, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON,
   SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_MARIBBON, SB_AB_SRC_DBB,
   SB_AB_SRC_DBB, SB_AB_SRC_DBB, SB_AB_SRC_DBB, SB_AB_SRC_DBB, SB_AB_SRC_DBB, SB_AB_SRC_TP, SB_AB_SRC_TP, SB_AB_SRC_TP,
   SB_AB_SRC_TP, SB_AB_SRC_TP, SB_AB_SRC_TP, SB_AB_SRC_TP, SB_AB_SRC_TP, SB_AB_SRC_TP, SB_AB_SRC_SM, SB_AB_SRC_SM,
   SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM,
   SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_SM, SB_AB_SRC_MACD,
   SB_AB_SRC_MACD, SB_AB_SRC_MACD, SB_AB_SRC_MACD, SB_AB_SRC_MACD, SB_AB_SRC_MACD, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE,
   SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE,
   SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE, SB_AB_SRC_CANDLE,
   SB_AB_SRC_CANDLE, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV,
   SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV,
   SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV, SB_AB_SRC_RSI_DIV
  };

static const int SB_ALL_BLEND_PLAN_COL[SB_ALL_BLEND_CORE_CH] =
  {
   0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
   6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 1, 7, 8,
   9, 10, 11, 12, 13, 14, 0, 2, 3, 4, 7, 8, 9, 10, 11, 12,
   13, 14, 15, 16, 17, 18, 19, 5, 8, 9, 10, 11, 12, 0, 1, 2,
   3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
   19, 1, 5, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
   23, 24, 25
  };

// Plan is first-name-wins over ALL_BLEND_SOURCES; cross-engine name
// collisions drop later locals; ALL_BLEND_DROP_CHANNELS removed too.
// Core 99 + common 16 = 115.
// Python warmup=720; common keep=16.

void SB_AllBlendCopyColumn(const float &src[], const int src_col,
                           const int lookback, float &dst[], const int dst_col)
  {
   for(int t = 0; t < lookback; t++)
      dst[dst_col * lookback + t] = src[src_col * lookback + t];
  }

bool SB_AllBlendBuildSourceWindow(const int src_id, const string symbol,
                                  const int lookback,
                                  const int end_bar_or_neg1,
                                  float &window[], string &err)
  {
   err = "";
   int n_bars = 0;
   int end_bar = end_bar_or_neg1;
   bool ok = false;

   if(src_id == SB_AB_SRC_REGIME)
     {
      double features[][SB_REGIME_CH];
      if(!SB_BuildRegimeFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildRegimeLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(src_id == SB_AB_SRC_MARIBBON)
     {
      // all_blend uses 1h base 13 only (no 15m/5m twins)
      double features[][SB_MARIBBON_BASE];
      if(!SB_BuildMaribbonBase1hMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildMaribbonBase1hLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(src_id == SB_AB_SRC_DBB)
     {
      double features[][SB_NUM_INPUTS];
      if(!SB_BuildDbbFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildDbbLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(src_id == SB_AB_SRC_TP)
     {
      double features[][SB_TP_CH];
      if(!SB_BuildTrendPullbackFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildTrendPullbackLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(src_id == SB_AB_SRC_SM)
     {
      double features[][SB_SM_CH];
      if(!SB_BuildSmartMoneyFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildSmartMoneyLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(src_id == SB_AB_SRC_MACD)
     {
      double features[][SB_MACD_CH];
      if(!SB_BuildMacdFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildMacdLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(src_id == SB_AB_SRC_CANDLE)
     {
      double features[][SB_CANDLE_CH];
      if(!SB_BuildCandleFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildCandleLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else if(src_id == SB_AB_SRC_RSI_DIV)
     {
      double features[][SB_RSI_DIV_CH];
      if(!SB_BuildRsiDivergenceFeatureMatrix(symbol, features, n_bars, err))
         return false;
      if(end_bar < 0)
         end_bar = n_bars - 2;
      ok = SB_BuildRsiDivergenceLookbackWindow(features, n_bars, lookback, end_bar, window, err);
     }
   else
     {
      err = StringFormat("all_blend: unknown source id %d", src_id);
      return false;
     }
   return ok;
  }

// Build all_blend lookback window: 99 core (union plan) + 16 common.
// Layout: channel-major float[ch * lookback], c*lookback+t.
// Common is embedded here — FeatureDispatch must NOT append again.
bool SB_BuildAllBlendLookbackWindow(const string symbol,
                                    const int lookback,
                                    const int end_bar_or_neg1,
                                    float &window[],
                                    string &err)
  {
   err = "";
   if(lookback <= 0)
     {
      err = "all_blend: invalid lookback";
      return false;
     }

   // Build each source core lookback window once.
   // Separate locals — MQL5 has no float[][].
   float win_regime[], win_maribbon[], win_dbb[], win_tp[];
   float win_sm[], win_macd[], win_candle[], win_rsi[];

   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_REGIME, symbol, lookback,
                                   end_bar_or_neg1, win_regime, err))
      return false;
   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_MARIBBON, symbol, lookback,
                                   end_bar_or_neg1, win_maribbon, err))
      return false;
   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_DBB, symbol, lookback,
                                   end_bar_or_neg1, win_dbb, err))
      return false;
   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_TP, symbol, lookback,
                                   end_bar_or_neg1, win_tp, err))
      return false;
   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_SM, symbol, lookback,
                                   end_bar_or_neg1, win_sm, err))
      return false;
   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_MACD, symbol, lookback,
                                   end_bar_or_neg1, win_macd, err))
      return false;
   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_CANDLE, symbol, lookback,
                                   end_bar_or_neg1, win_candle, err))
      return false;
   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_RSI_DIV, symbol, lookback,
                                   end_bar_or_neg1, win_rsi, err))
      return false;

   ArrayResize(window, SB_ALL_BLEND_CORE_CH * lookback);
   for(int o = 0; o < SB_ALL_BLEND_CORE_CH; o++)
     {
      const int s = SB_ALL_BLEND_PLAN_SRC[o];
      const int c = SB_ALL_BLEND_PLAN_COL[o];
      if(s == SB_AB_SRC_REGIME)
         SB_AllBlendCopyColumn(win_regime, c, lookback, window, o);
      else if(s == SB_AB_SRC_MARIBBON)
         SB_AllBlendCopyColumn(win_maribbon, c, lookback, window, o);
      else if(s == SB_AB_SRC_DBB)
         SB_AllBlendCopyColumn(win_dbb, c, lookback, window, o);
      else if(s == SB_AB_SRC_TP)
         SB_AllBlendCopyColumn(win_tp, c, lookback, window, o);
      else if(s == SB_AB_SRC_SM)
         SB_AllBlendCopyColumn(win_sm, c, lookback, window, o);
      else if(s == SB_AB_SRC_MACD)
         SB_AllBlendCopyColumn(win_macd, c, lookback, window, o);
      else if(s == SB_AB_SRC_CANDLE)
         SB_AllBlendCopyColumn(win_candle, c, lookback, window, o);
      else if(s == SB_AB_SRC_RSI_DIV)
         SB_AllBlendCopyColumn(win_rsi, c, lookback, window, o);
      else
        {
         err = StringFormat("all_blend: bad plan src at out=%d", o);
         return false;
        }
     }

   // Append full common pack (16).
   if(!SB_ExtendWindowWithCommon(symbol, lookback, end_bar_or_neg1,
                                 SB_ALL_BLEND_CORE_CH, window, err))
      return false;

   if(ArraySize(window) != SB_ALL_BLEND_TOTAL_CH * lookback)
     {
      err = StringFormat("all_blend: window size %d != %d*%d",
                         ArraySize(window), SB_ALL_BLEND_TOTAL_CH, lookback);
      return false;
     }
   return true;
  }

#endif // SMARTBS_ALL_BLEND_FEATURES_MQH
