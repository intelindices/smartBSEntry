//+------------------------------------------------------------------+
//| Auto-generated from smartbs_engines.catalog                     |
//| Do not edit by hand — run: python -m smartbs_engines.catalog --write-mql
//+------------------------------------------------------------------+
#ifndef SMARTBS_ENGINE_CATALOG_MQH
#define SMARTBS_ENGINE_CATALOG_MQH

#define SB_COMMON_CH_CATALOG 16
#define SB_MAX_TOTAL_CH      42

#ifndef SB_COMMON_ONLY
#define SB_COMMON_ONLY 0  // common
#endif
#ifndef SB_DBB_CH
#define SB_DBB_CH 12  // dbb
#endif
#ifndef SB_MACD_CH
#define SB_MACD_CH 13  // macd
#endif
#ifndef SB_MARIBBON_CH
#define SB_MARIBBON_CH 39  // maribbon
#endif
#ifndef SB_RSI_DIV_CH
#define SB_RSI_DIV_CH 26  // rsi_divergence
#endif
#ifndef SB_SM_CH
#define SB_SM_CH 20  // smart_money
#endif
#ifndef SB_TP_CH
#define SB_TP_CH 15  // trend_pullback
#endif

string SB_BlendLiveEngines() { return "dbb,macd,maribbon,smart_money,trend_pullback"; }

#endif // SMARTBS_ENGINE_CATALOG_MQH
