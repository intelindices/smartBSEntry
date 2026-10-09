//+------------------------------------------------------------------+
//| Auto-generated chart-pack allowlist from smartbs_engines.catalog |
//| Do not edit by hand — run: python -m smartbs_engines.catalog --write-mql
//+------------------------------------------------------------------+
#ifndef SMARTBS_CHART_ALLOWLIST_MQH
#define SMARTBS_CHART_ALLOWLIST_MQH

#define SB_CHART_PACK_ENGINES "common,maribbon"

bool SB_IsChartPackEngine(const string e)
  {
   return (e == "common" || e == "maribbon");
  }

#endif // SMARTBS_CHART_ALLOWLIST_MQH
