//+------------------------------------------------------------------+
//| Auto-generated from smartbs_engines/engine_catalog.py           |
//| Do not edit by hand — run: python -m smartbs_engines.engine_catalog --write-mql
//+------------------------------------------------------------------+
#ifndef SMARTBS_ENGINE_CATALOG_MQH
#define SMARTBS_ENGINE_CATALOG_MQH

#define SB_COMMON_CH_CATALOG 16
#define SB_MAX_TOTAL_CH      150

#ifndef SB_REGIME_CH
#define SB_REGIME_CH 12  // regime_engine
#endif
#ifndef SB_MARIBBON_CH
#define SB_MARIBBON_CH 22  // maribbon
#endif
#ifndef SB_DBB_CH
#define SB_DBB_CH 12  // dbb
#endif
#ifndef SB_TP_CH
#define SB_TP_CH 15  // trend_pullback
#endif
#ifndef SB_SM_CH
#define SB_SM_CH 20  // smart_money
#endif
#ifndef SB_MACD_CH
#define SB_MACD_CH 13  // macd
#endif
#ifndef SB_CANDLE_CH
#define SB_CANDLE_CH 20  // candle
#endif
#ifndef SB_RSI_DIV_CH
#define SB_RSI_DIV_CH 26  // rsi_divergence
#endif
#ifndef SB_PIVOT_CH
#define SB_PIVOT_CH 12  // pivot_engine
#endif
#ifndef SB_COMMON_ONLY
#define SB_COMMON_ONLY 0  // common
#endif
#ifndef SB_ALL_BLEND_CH
#define SB_ALL_BLEND_CH 134  // all_blend
#endif

// Live blend pool (Python BLEND_LIVE / ACTIVE_BLEND_ENGINES)
string SB_BlendLiveEngines() { return "regime_engine,maribbon,dbb,trend_pullback,smart_money,macd"; }

#endif // SMARTBS_ENGINE_CATALOG_MQH
