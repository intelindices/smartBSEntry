"""Generate mql5/Include/SmartBSEntry/AllBlendFeatures.mqh from engine_all_blend plan.

Run:
    python smartbs_engines/_gen_all_blend_features_mqh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartbs_engines.engine_all_blend import (  # noqa: E402
    all_blend_feature_names,
    _common_keep,
    _plan_channels,
)
OUT = ROOT / "mql5" / "Include" / "SmartBSEntry" / "AllBlendFeatures.mqh"

SRC_CONST = {
    "regime_engine": "SB_AB_SRC_REGIME",
    "maribbon": "SB_AB_SRC_MARIBBON",
    "dbb": "SB_AB_SRC_DBB",
    "trend_pullback": "SB_AB_SRC_TP",
    "smart_money": "SB_AB_SRC_SM",
    "macd": "SB_AB_SRC_MACD",
    "candle": "SB_AB_SRC_CANDLE",
    "rsi_divergence": "SB_AB_SRC_RSI_DIV",
}


def main() -> None:
    names, plan, warm = _plan_channels()
    ck_names, ck_idxs = _common_keep()
    full = all_blend_feature_names(with_common=True)

    n_core = len(plan)
    n_common = len(ck_names)
    n_total = len(full)
    assert n_core == len(names) > 0, (n_core, len(names))
    assert 0 < n_common <= 16, n_common
    assert len(ck_idxs) == n_common
    assert n_total == n_core + n_common, (n_total, n_core, n_common)
    print(
        f"OK: len(plan)={n_core} common_keep={n_common}/{16} "
        f"total_with_common={n_total} warmup={warm}"
    )

    srcs = [SRC_CONST[s] for s, _ in plan]
    cols = [int(j) for _, j in plan]

    lines: list[str] = []
    a = lines.append
    a("//+------------------------------------------------------------------+")
    a(f"//| SmartBS all_blend — {n_core} core + {n_common} common = {n_total} channels"
      f"{' ' * max(0, 10 - len(str(n_total)))}|")
    a("//| Auto-generated from smartbs_engines/engine_all_blend.py          |")
    a("//| Do not edit by hand — run:                                      |")
    a("//|   python smartbs_engines/_gen_all_blend_features_mqh.py         |")
    a("//+------------------------------------------------------------------+")
    a("#ifndef SMARTBS_ALL_BLEND_FEATURES_MQH")
    a("#define SMARTBS_ALL_BLEND_FEATURES_MQH")
    a("")
    a('#include "RegimeFeatures.mqh"')
    a('#include "MaribbonFeatures.mqh"')
    a('#include "DbbFeatures.mqh"')
    a('#include "TrendPullbackFeatures.mqh"')
    a('#include "SmartMoneyFeatures.mqh"')
    a('#include "MacdFeatures.mqh"')
    a('#include "CandleFeatures.mqh"')
    a('#include "RsiDivergenceFeatures.mqh"')
    a('#include "CommonChannels.mqh"')
    a('#include "EngineCatalog.mqh"')
    a("")
    a(f"#define SB_ALL_BLEND_CORE_CH   {n_core}")
    a(f"#define SB_ALL_BLEND_TOTAL_CH  {n_total}")
    a(f"#define SB_ALL_BLEND_WARMUP    {int(warm)}")
    a("")
    a("// Source engine ids for the static plan (matches ALL_BLEND_SOURCES order).")
    a("#define SB_AB_SRC_REGIME    0")
    a("#define SB_AB_SRC_MARIBBON  1")
    a("#define SB_AB_SRC_DBB       2")
    a("#define SB_AB_SRC_TP        3")
    a("#define SB_AB_SRC_SM        4")
    a("#define SB_AB_SRC_MACD      5")
    a("#define SB_AB_SRC_CANDLE    6")
    a("#define SB_AB_SRC_RSI_DIV   7")
    a("#define SB_AB_SRC_COUNT     8")
    a("")
    a("// Per core output column: (source_engine_id, local_col). First-name-wins plan.")
    a("static const int SB_ALL_BLEND_PLAN_SRC[SB_ALL_BLEND_CORE_CH] =")
    a("  {")
    for i in range(0, len(srcs), 8):
        chunk = srcs[i : i + 8]
        comma = "," if i + 8 < len(srcs) else ""
        a("   " + ", ".join(chunk) + comma)
    a("  };")
    a("")
    a("static const int SB_ALL_BLEND_PLAN_COL[SB_ALL_BLEND_CORE_CH] =")
    a("  {")
    for i in range(0, len(cols), 16):
        chunk = [str(c) for c in cols[i : i + 16]]
        comma = "," if i + 16 < len(cols) else ""
        a("   " + ", ".join(chunk) + comma)
    a("  };")
    a("")
    a("// Plan is first-name-wins over ALL_BLEND_SOURCES; cross-engine name")
    a("// collisions drop later locals; ALL_BLEND_DROP_CHANNELS removed too.")
    a(f"// Core {n_core} + common {n_common} = {n_total}.")
    a(f"// Python warmup={int(warm)}; common keep={len(ck_names)}.")
    a("")
    a("void SB_AllBlendCopyColumn(const float &src[], const int src_col,")
    a("                           const int lookback, float &dst[], const int dst_col)")
    a("  {")
    a("   for(int t = 0; t < lookback; t++)")
    a("      dst[dst_col * lookback + t] = src[src_col * lookback + t];")
    a("  }")
    a("")
    a("bool SB_AllBlendBuildSourceWindow(const int src_id, const string symbol,")
    a("                                  const int lookback,")
    a("                                  const int end_bar_or_neg1,")
    a("                                  float &window[], string &err)")
    a("  {")
    a('   err = "";')
    a("   int n_bars = 0;")
    a("   int end_bar = end_bar_or_neg1;")
    a("   bool ok = false;")
    a("")
    a("   if(src_id == SB_AB_SRC_REGIME)")
    a("     {")
    a("      double features[][SB_REGIME_CH];")
    a("      if(!SB_BuildRegimeFeatureMatrix(symbol, features, n_bars, err))")
    a("         return false;")
    a("      if(end_bar < 0)")
    a("         end_bar = n_bars - 2;")
    a("      ok = SB_BuildRegimeLookbackWindow(features, n_bars, lookback, end_bar, window, err);")
    a("     }")
    a("   else if(src_id == SB_AB_SRC_MARIBBON)")
    a("     {")
    a("      // all_blend uses 1h base 13 only (no 15m/5m twins)")
    a("      double features[][SB_MARIBBON_BASE];")
    a("      if(!SB_BuildMaribbonBase1hMatrix(symbol, features, n_bars, err))")
    a("         return false;")
    a("      if(end_bar < 0)")
    a("         end_bar = n_bars - 2;")
    a("      ok = SB_BuildMaribbonBase1hLookbackWindow(features, n_bars, lookback, end_bar, window, err);")
    a("     }")
    a("   else if(src_id == SB_AB_SRC_DBB)")
    a("     {")
    a("      double features[][SB_NUM_INPUTS];")
    a("      if(!SB_BuildDbbFeatureMatrix(symbol, features, n_bars, err))")
    a("         return false;")
    a("      if(end_bar < 0)")
    a("         end_bar = n_bars - 2;")
    a("      ok = SB_BuildDbbLookbackWindow(features, n_bars, lookback, end_bar, window, err);")
    a("     }")
    a("   else if(src_id == SB_AB_SRC_TP)")
    a("     {")
    a("      double features[][SB_TP_CH];")
    a("      if(!SB_BuildTrendPullbackFeatureMatrix(symbol, features, n_bars, err))")
    a("         return false;")
    a("      if(end_bar < 0)")
    a("         end_bar = n_bars - 2;")
    a("      ok = SB_BuildTrendPullbackLookbackWindow(features, n_bars, lookback, end_bar, window, err);")
    a("     }")
    a("   else if(src_id == SB_AB_SRC_SM)")
    a("     {")
    a("      double features[][SB_SM_CH];")
    a("      if(!SB_BuildSmartMoneyFeatureMatrix(symbol, features, n_bars, err))")
    a("         return false;")
    a("      if(end_bar < 0)")
    a("         end_bar = n_bars - 2;")
    a("      ok = SB_BuildSmartMoneyLookbackWindow(features, n_bars, lookback, end_bar, window, err);")
    a("     }")
    a("   else if(src_id == SB_AB_SRC_MACD)")
    a("     {")
    a("      double features[][SB_MACD_CH];")
    a("      if(!SB_BuildMacdFeatureMatrix(symbol, features, n_bars, err))")
    a("         return false;")
    a("      if(end_bar < 0)")
    a("         end_bar = n_bars - 2;")
    a("      ok = SB_BuildMacdLookbackWindow(features, n_bars, lookback, end_bar, window, err);")
    a("     }")
    a("   else if(src_id == SB_AB_SRC_CANDLE)")
    a("     {")
    a("      double features[][SB_CANDLE_CH];")
    a("      if(!SB_BuildCandleFeatureMatrix(symbol, features, n_bars, err))")
    a("         return false;")
    a("      if(end_bar < 0)")
    a("         end_bar = n_bars - 2;")
    a("      ok = SB_BuildCandleLookbackWindow(features, n_bars, lookback, end_bar, window, err);")
    a("     }")
    a("   else if(src_id == SB_AB_SRC_RSI_DIV)")
    a("     {")
    a("      double features[][SB_RSI_DIV_CH];")
    a("      if(!SB_BuildRsiDivergenceFeatureMatrix(symbol, features, n_bars, err))")
    a("         return false;")
    a("      if(end_bar < 0)")
    a("         end_bar = n_bars - 2;")
    a("      ok = SB_BuildRsiDivergenceLookbackWindow(features, n_bars, lookback, end_bar, window, err);")
    a("     }")
    a("   else")
    a("     {")
    a('      err = StringFormat("all_blend: unknown source id %d", src_id);')
    a("      return false;")
    a("     }")
    a("   return ok;")
    a("  }")
    a("")
    a(f"// Build all_blend lookback window: {n_core} core (union plan) + {n_common} common.")
    a("// Layout: channel-major float[ch * lookback], c*lookback+t.")
    a("// Common is embedded here — FeatureDispatch must NOT append again.")
    a("bool SB_BuildAllBlendLookbackWindow(const string symbol,")
    a("                                    const int lookback,")
    a("                                    const int end_bar_or_neg1,")
    a("                                    float &window[],")
    a("                                    string &err)")
    a("  {")
    a('   err = "";')
    a("   if(lookback <= 0)")
    a("     {")
    a('      err = "all_blend: invalid lookback";')
    a("      return false;")
    a("     }")
    a("")
    a("   // Build each source core lookback window once.")
    a("   float src_win[];")
    a("   // Jagged per-source windows held as separate locals (MQL5 has no float[][]).")
    a("   float win_regime[], win_maribbon[], win_dbb[], win_tp[];")
    a("   float win_sm[], win_macd[], win_candle[], win_rsi[];")
    a("")
    a("   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_REGIME, symbol, lookback,")
    a("                                   end_bar_or_neg1, win_regime, err))")
    a("      return false;")
    a("   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_MARIBBON, symbol, lookback,")
    a("                                   end_bar_or_neg1, win_maribbon, err))")
    a("      return false;")
    a("   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_DBB, symbol, lookback,")
    a("                                   end_bar_or_neg1, win_dbb, err))")
    a("      return false;")
    a("   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_TP, symbol, lookback,")
    a("                                   end_bar_or_neg1, win_tp, err))")
    a("      return false;")
    a("   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_SM, symbol, lookback,")
    a("                                   end_bar_or_neg1, win_sm, err))")
    a("      return false;")
    a("   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_MACD, symbol, lookback,")
    a("                                   end_bar_or_neg1, win_macd, err))")
    a("      return false;")
    a("   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_CANDLE, symbol, lookback,")
    a("                                   end_bar_or_neg1, win_candle, err))")
    a("      return false;")
    a("   if(!SB_AllBlendBuildSourceWindow(SB_AB_SRC_RSI_DIV, symbol, lookback,")
    a("                                   end_bar_or_neg1, win_rsi, err))")
    a("      return false;")
    a("")
    a("   ArrayResize(window, SB_ALL_BLEND_CORE_CH * lookback);")
    a("   for(int o = 0; o < SB_ALL_BLEND_CORE_CH; o++)")
    a("     {")
    a("      const int s = SB_ALL_BLEND_PLAN_SRC[o];")
    a("      const int c = SB_ALL_BLEND_PLAN_COL[o];")
    a("      if(s == SB_AB_SRC_REGIME)")
    a("         SB_AllBlendCopyColumn(win_regime, c, lookback, window, o);")
    a("      else if(s == SB_AB_SRC_MARIBBON)")
    a("         SB_AllBlendCopyColumn(win_maribbon, c, lookback, window, o);")
    a("      else if(s == SB_AB_SRC_DBB)")
    a("         SB_AllBlendCopyColumn(win_dbb, c, lookback, window, o);")
    a("      else if(s == SB_AB_SRC_TP)")
    a("         SB_AllBlendCopyColumn(win_tp, c, lookback, window, o);")
    a("      else if(s == SB_AB_SRC_SM)")
    a("         SB_AllBlendCopyColumn(win_sm, c, lookback, window, o);")
    a("      else if(s == SB_AB_SRC_MACD)")
    a("         SB_AllBlendCopyColumn(win_macd, c, lookback, window, o);")
    a("      else if(s == SB_AB_SRC_CANDLE)")
    a("         SB_AllBlendCopyColumn(win_candle, c, lookback, window, o);")
    a("      else if(s == SB_AB_SRC_RSI_DIV)")
    a("         SB_AllBlendCopyColumn(win_rsi, c, lookback, window, o);")
    a("      else")
    a("        {")
    a('         err = StringFormat("all_blend: bad plan src at out=%d", o);')
    a("         return false;")
    a("        }")
    a("     }")
    a("")
    if n_common == 16 and list(ck_idxs) == list(range(16)):
        a("   // Append full common pack (16).")
        a("   if(!SB_ExtendWindowWithCommon(symbol, lookback, end_bar_or_neg1,")
        a("                                 SB_ALL_BLEND_CORE_CH, window, err))")
        a("      return false;")
    else:
        a(f"   // Append filtered common pack ({n_common} of 16; drops via ALL_BLEND_DROP).")
        a("   {")
        a("    double common[][SB_COMMON_CH];")
        a("    int n_bars_c = 0;")
        a("    if(!SB_BuildCommonFeatureMatrix(symbol, common, n_bars_c, err))")
        a("       return false;")
        a("    const int end_c = n_bars_c - 2;")
        a("    if(end_c < lookback - 1 || end_c < 0)")
        a("      {")
        a('       err = StringFormat("all_blend common: need lookback=%d, have n=%d",')
        a("                          lookback, n_bars_c);")
        a("       return false;")
        a("      }")
        a(f"    static const int SB_AB_COMMON_SRC_COL[{n_common}] =")
        a("      {")
        for i in range(0, n_common, 16):
            chunk = [str(int(x)) for x in ck_idxs[i : i + 16]]
            comma = "," if i + 16 < n_common else ""
            a("       " + ", ".join(chunk) + comma)
        a("      };")
        a("    ArrayResize(window, SB_ALL_BLEND_TOTAL_CH * lookback);")
        a(f"    for(int k = 0; k < {n_common}; k++)")
        a("      {")
        a("       const int src_c = SB_AB_COMMON_SRC_COL[k];")
        a("       const int dst_c = SB_ALL_BLEND_CORE_CH + k;")
        a("       for(int t = 0; t < lookback; t++)")
        a("          window[dst_c * lookback + t] =")
        a("             (float)common[end_c - lookback + 1 + t][src_c];")
        a("      }")
        a("   }")
    a("")
    a("   if(ArraySize(window) != SB_ALL_BLEND_TOTAL_CH * lookback)")
    a("     {")
    a('      err = StringFormat("all_blend: window size %d != %d*%d",')
    a("                         ArraySize(window), SB_ALL_BLEND_TOTAL_CH, lookback);")
    a("      return false;")
    a("     }")
    a("   return true;")
    a("  }")
    a("")
    a("#endif // SMARTBS_ALL_BLEND_FEATURES_MQH")
    a("")

    # Drop unused local to keep compiler quiet in generated intent comments
    text = "\n".join(lines)
    # Remove the unused `float src_win[];` line we left as a comment placeholder
    text = text.replace(
        "   // Build each source core lookback window once.\n"
        "   float src_win[];\n"
        "   // Jagged per-source windows held as separate locals (MQL5 has no float[][]).\n",
        "   // Build each source core lookback window once.\n"
        "   // Separate locals — MQL5 has no float[][].\n",
    )

    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
