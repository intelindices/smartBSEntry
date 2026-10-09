"""Catalog SSOT derived from registered plugins + MQL codegen CLI.

    python -m smartbs_engines.catalog --write-mql
    python -m smartbs_engines.catalog --print
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smartbs_engines.plugins.base import ensure_plugins_loaded

COMMON_CHANNELS = 16
COMMON_FEATURE_NAMES: tuple[str, ...] = (
    "ohlcv_open",
    "ohlcv_high",
    "ohlcv_low",
    "ohlcv_volume",
    "ohlcv_close_displacement",
    "tod_sin",
    "tod_cos",
    "session_tokyo",
    "session_london",
    "session_newyork",
    "cd_vs_fast",
    "cd_body_cross_fast",
    "session_strength",
    "current_session_strength",
    "trend_regime",
    "last_candle_strength",
)


@dataclass(frozen=True)
class EngineSpec:
    name: str
    core_channels: int
    module: str
    mql_define: str
    in_blend_live: bool = False
    in_blend_research: bool = False
    in_signal_sources: bool = False
    in_chart_pack: bool = False
    notes: str = ""


def _engine_plugins():
    ensure_plugins_loaded()
    from smartbs_engines.engines import ENGINE_PLUGINS

    return ENGINE_PLUGINS


def engine_specs() -> tuple[EngineSpec, ...]:
    plugs = _engine_plugins()
    out: list[EngineSpec] = []
    for name, p in plugs.items():
        out.append(
            EngineSpec(
                name=p.name,
                core_channels=p.core_channels,
                module=f"engines/{p.name}.py",
                mql_define=p.mql_define,
                in_blend_live=p.in_blend_live,
                in_blend_research=p.in_blend_research,
                in_signal_sources=p.in_signal_sources,
                in_chart_pack=p.in_chart_pack,
                notes=p.notes,
            )
        )
    return tuple(out)


def __getattr__(name: str):
    if name == "ENGINE_SPECS":
        return engine_specs()
    if name == "BLEND_LIVE":
        return tuple(p.name for _, p in _engine_plugins().items() if p.in_blend_live)
    if name == "BLEND_RESEARCH":
        return tuple(p.name for _, p in _engine_plugins().items() if p.in_blend_research)
    if name == "ACTIVE_SIGNAL_SOURCES":
        return tuple(p.name for _, p in _engine_plugins().items() if p.in_signal_sources)
    if name == "CHART_PACK_ENGINES":
        return tuple(p.name for _, p in _engine_plugins().items() if p.in_chart_pack)
    if name == "LABEL_MODES":
        ensure_plugins_loaded()
        from smartbs_engines.labels import list_label_modes

        return tuple(list_label_modes())
    if name == "BACKBONES":
        ensure_plugins_loaded()
        from smartbs_engines.backbones import list_backbones

        return tuple(list_backbones())
    raise AttributeError(name)


def core_channels(name: str) -> int:
    return int(_engine_plugins().get(name).core_channels)


def total_channels(name: str, *, with_common: bool = True) -> int:
    key = (name or "").strip().lower()
    p = _engine_plugins().get(key)
    n = int(p.core_channels)
    if key == "common":
        return COMMON_CHANNELS if with_common else 0
    if not p.attach_common:
        return n
    return n + (COMMON_CHANNELS if with_common else 0)


def max_total_channels(*, with_common: bool = True) -> int:
    return max(total_channels(n, with_common=with_common) for n in _engine_plugins().names())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mql_include_dir() -> Path:
    return _repo_root() / "mql5" / "Include" / "SmartBSEntry"


def render_mql_catalog() -> str:
    specs = engine_specs()
    blend = ",".join(p.name for _, p in _engine_plugins().items() if p.in_blend_live)
    lines = [
        "//+------------------------------------------------------------------+",
        "//| Auto-generated from smartbs_engines.catalog                     |",
        "//| Do not edit by hand — run: python -m smartbs_engines.catalog --write-mql",
        "//+------------------------------------------------------------------+",
        "#ifndef SMARTBS_ENGINE_CATALOG_MQH",
        "#define SMARTBS_ENGINE_CATALOG_MQH",
        "",
        f"#define SB_COMMON_CH_CATALOG {COMMON_CHANNELS}",
        f"#define SB_MAX_TOTAL_CH      {max_total_channels()}",
        "",
    ]
    for e in specs:
        lines.append(f"#ifndef {e.mql_define}")
        lines.append(f"#define {e.mql_define} {e.core_channels}  // {e.name}")
        lines.append("#endif")
    lines.extend(
        [
            "",
            "string SB_BlendLiveEngines() { return \"" + blend + "\"; }",
            "",
            "#endif // SMARTBS_ENGINE_CATALOG_MQH",
            "",
        ]
    )
    return "\n".join(lines)


def render_mql_chart_allowlist() -> str:
    names = [p.name for _, p in _engine_plugins().items() if p.in_chart_pack]
    joined = ",".join(names)
    checks = " || ".join(f'e == "{n}"' for n in names) or "false"
    return "\n".join(
        [
            "//+------------------------------------------------------------------+",
            "//| Auto-generated chart-pack allowlist from smartbs_engines.catalog |",
            "//| Do not edit by hand — run: python -m smartbs_engines.catalog --write-mql",
            "//+------------------------------------------------------------------+",
            "#ifndef SMARTBS_CHART_ALLOWLIST_MQH",
            "#define SMARTBS_CHART_ALLOWLIST_MQH",
            "",
            f'#define SB_CHART_PACK_ENGINES "{joined}"',
            "",
            "bool SB_IsChartPackEngine(const string e)",
            "  {",
            f"   return ({checks});",
            "  }",
            "",
            "#endif // SMARTBS_CHART_ALLOWLIST_MQH",
            "",
        ]
    )


def render_mql_feature_dispatch() -> str:
    plugs = [p for _, p in _engine_plugins().items()]
    includes = []
    seen = set()
    for p in plugs:
        if p.mql_include and p.mql_include not in seen:
            includes.append(f'#include "{p.mql_include}"')
            seen.add(p.mql_include)
    if "CommonChannels.mqh" not in seen:
        includes.append('#include "CommonChannels.mqh"')
    includes.append('#include "EngineCatalog.mqh"')
    includes.append('#include "ChartAllowlist.mqh"')

    core_branches = []
    for p in plugs:
        if p.name == "common":
            core_branches.append(
                '   if(e == "common")\n'
                "      return SB_COMMON_ONLY; // 0 = common-only (no core pack)"
            )
        else:
            core_branches.append(
                f'   if(e == "{p.name}")\n'
                f"      return {p.mql_define};"
            )

    warm_lines = [
        '   if(e == "common")',
        "      return SB_COMMON_WARMUP;",
    ]
    others = [p for p in plugs if p.name != "common"]
    for i, p in enumerate(others):
        kw = "if" if i == 0 else "else if"
        warm_lines.append(f'   {kw}(e == "{p.name}")')
        warm_lines.append(f"      w = {p.mql_warmup_macro};")
    warm_lines.extend(["   else", "      return 0;"])

    build_branches = []
    for p in plugs:
        if p.name == "common":
            continue
        build_branches.append(
            f'   else if(e == "{p.name}")\n'
            "     {\n"
            f"      double features[][{p.mql_matrix_ch}];\n"
            f"      if(!{p.mql_build_matrix}(symbol, features, n_bars, err))\n"
            "         return false;\n"
            "      if(end_bar < 0)\n"
            "         end_bar = n_bars - 2;\n"
            f"      ok = {p.mql_build_window}(features, n_bars, lookback, end_bar, window, err);\n"
            "     }"
        )
    # first branch should be if not else if
    if build_branches:
        build_branches[0] = build_branches[0].replace("else if", "if", 1)

    core_only = [p.name for p in plugs if not p.attach_common and p.name != "common"]
    core_only_check = " || ".join(f'e == "{n}"' for n in core_only) or "false"

    lines = [
        "//+------------------------------------------------------------------+",
        "//| Auto-generated FeatureDispatch from smartbs_engines.catalog     |",
        "//| Do not edit by hand — run: python -m smartbs_engines.catalog --write-mql",
        "//+------------------------------------------------------------------+",
        "#ifndef SMARTBS_FEATURE_DISPATCH_MQH",
        "#define SMARTBS_FEATURE_DISPATCH_MQH",
        "",
        *includes,
        "",
        "#define SB_MAX_ENGINE_CH SB_MAX_TOTAL_CH",
        "",
        "string SB_EngineNormalize(const string eng)",
        "  {",
        "   string e = eng;",
        "   StringTrimLeft(e);",
        "   StringTrimRight(e);",
        "   StringToLower(e);",
        "   return e;",
        "  }",
        "",
        "int SB_EngineCoreChannelCount(const string eng)",
        "  {",
        "   string e = SB_EngineNormalize(eng);",
        *core_branches,
        "   return -1;",
        "  }",
        "",
        "int SB_EngineChannelCount(const string eng)",
        "  {",
        "   string e = SB_EngineNormalize(eng);",
        '   if(e == "common")',
        "      return SB_COMMON_CH;",
        f"   if({core_only_check})",
        "      return SB_EngineCoreChannelCount(e);",
        "   int core = SB_EngineCoreChannelCount(eng);",
        "   if(core < 0)",
        "      return 0;",
        "   return core + SB_COMMON_CH;",
        "  }",
        "",
        "int SB_EngineWarmup(const string eng)",
        "  {",
        "   string e = SB_EngineNormalize(eng);",
        "   int w = 0;",
        *warm_lines,
        f"   if({core_only_check})",
        "      return w;",
        "   return MathMax(w, SB_COMMON_WARMUP);",
        "  }",
        "",
        "void SB_ParseEngineList(const string inp, string &out[], int &n)",
        "  {",
        "   n = 0;",
        "   ArrayResize(out, 0);",
        "   string parts[];",
        "   int cnt = StringSplit(inp, ',', parts);",
        "   for(int i = 0; i < cnt; i++)",
        "     {",
        "      string e = SB_EngineNormalize(parts[i]);",
        "      if(StringLen(e) == 0)",
        "         continue;",
        "      ArrayResize(out, n + 1);",
        "      out[n] = e;",
        "      n++;",
        "     }",
        "  }",
        "",
        "bool SB_BuildEngineLookbackWindow(const string eng, const string symbol,",
        "                                  const int lookback,",
        "                                  const int end_bar_closed_h1_index_or_use_shift,",
        "                                  float &window[], int &num_inputs, string &err)",
        "  {",
        '   err = "";',
        "   num_inputs = 0;",
        "   string e = SB_EngineNormalize(eng);",
        "   int core_ch = SB_EngineCoreChannelCount(e);",
        "   num_inputs = SB_EngineChannelCount(e);",
        "   if(core_ch < 0 || num_inputs <= 0)",
        "     {",
        '      err = "unknown engine: " + eng;',
        "      return false;",
        "     }",
        "   if(lookback <= 0)",
        "     {",
        '      err = "invalid lookback";',
        "      return false;",
        "     }",
        "",
        '   if(e == "common")',
        "      return SB_BuildCommonLookbackWindow(symbol, lookback, window, err);",
        "",
        "   int n_bars = 0;",
        "   int end_bar = end_bar_closed_h1_index_or_use_shift;",
        "   bool ok = false;",
        "",
        *build_branches,
        "   else",
        "     {",
        '      err = "unknown engine: " + eng;',
        "      return false;",
        "     }",
        "",
        "   if(!ok)",
        "      return false;",
        f"   if({core_only_check})",
        "      return true;",
        "   return SB_ExtendWindowWithCommon(symbol, lookback, end_bar, core_ch, window, err);",
        "  }",
        "",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def write_tester_files() -> Path:
    SYMS = ["XAUUSD", "XAGUSD", "XTIUSD", "BTCUSD", "ETHUSD"]
    TFS = ["1h", "5m", "15m", "4h"]

    def props(stem: str) -> list[str]:
        return [
            f'#property tester_file "SmartBSEntry\\\\{stem}.onnx"',
            f'#property tester_file "SmartBSEntry\\\\{stem}.json"',
        ]

    engines = [p.name for _, p in _engine_plugins().items() if p.in_chart_pack]
    lines = [
        f"// Auto-generated tester_file copies: {', '.join(engines)}",
        "// Default: XAUUSD_maribbon_1h (rolle_breakout L=5 V2).",
        "#ifndef SMARTBS_TESTER_FILES_MQH",
        "#define SMARTBS_TESTER_FILES_MQH",
        "",
        "// --- default EA pack ---",
    ]
    lines += props("XAUUSD_maribbon_1h")
    lines += ["", "// --- TF-suffixed ---"]
    for tf in TFS:
        for s in SYMS:
            for e in engines:
                stem = f"{s}_{e}_{tf}"
                if stem == "XAUUSD_maribbon_1h":
                    continue
                lines += props(stem)
    lines += ["", "// --- legacy 1h unsuffixed ---"]
    for s in SYMS:
        for e in engines:
            lines += props(f"{s}_{e}")
    lines += ["", "#endif", ""]
    out = _mql_include_dir() / "TesterFiles.mqh"
    out.write_text("\n".join(lines), encoding="ascii")
    return out


def write_mql_all() -> list[Path]:
    out_dir = _mql_include_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    mapping = {
        "EngineCatalog.mqh": render_mql_catalog(),
        "ChartAllowlist.mqh": render_mql_chart_allowlist(),
        "FeatureDispatch.mqh": render_mql_feature_dispatch(),
    }
    for name, text in mapping.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    written.append(write_tester_files())
    return written


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-mql", action="store_true")
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()
    ensure_plugins_loaded()
    if args.write_mql:
        for p in write_mql_all():
            print(f"Wrote {p}")
    if args.do_print or not args.write_mql:
        for e in engine_specs():
            print(
                f"{e.name:16} core={e.core_channels:3d} total={total_channels(e.name):3d}  {e.module}"
            )
        print(f"BLEND_LIVE={__getattr__('BLEND_LIVE')}")
        print(f"BLEND_RESEARCH={__getattr__('BLEND_RESEARCH')}")
        print(f"CHART_PACK={__getattr__('CHART_PACK_ENGINES')}")
        print(f"BACKBONES={__getattr__('BACKBONES')}")
        print(f"LABEL_MODES={__getattr__('LABEL_MODES')}")


if __name__ == "__main__":
    main()
