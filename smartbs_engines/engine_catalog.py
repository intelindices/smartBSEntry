"""Single source of truth for ST engine names, widths, and pools.

Python train/infer and MQL5 dispatch should agree with this table.
Regenerate the MQL header with::

    python -m smartbs_engines.engine_catalog --write-mql
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

# Canonical backbones (normalize_backbone aliases live in model.py).
BACKBONES: tuple[str, ...] = (
    "tcn",  # alias: smartBSEntry
    "smartBSEntryV2",
    "smartBSTF",
    "smartBSDualTF",
)

# Label modes still supported after cleanup.
LABEL_MODES: tuple[str, ...] = (
    "triple_barrier",
    "pct_barrier",
    "session_direction",
    "session_trend",
    "day_trend",
    "pivot_breakout",
    "forward_return",
    "next_direction",
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
    notes: str = ""


# Core widths must match engine feature_names (before common pack).
ENGINE_SPECS: tuple[EngineSpec, ...] = (
    EngineSpec("regime_engine", 12, "engine_regime.py", "SB_REGIME_CH", in_blend_live=True, in_blend_research=True),
    EngineSpec("maribbon", 22, "maribbon_engine.py", "SB_MARIBBON_CH", in_blend_live=True, in_blend_research=True),
    EngineSpec("dbb", 12, "engine_dbb.py", "SB_DBB_CH", in_blend_live=True, in_blend_research=True, in_signal_sources=True),
    EngineSpec("trend_pullback", 15, "engine_trend_pullback.py", "SB_TP_CH", in_blend_live=True, in_blend_research=True, in_signal_sources=True),
    EngineSpec("smart_money", 20, "engine_smart_money.py", "SB_SM_CH", in_blend_live=True, in_blend_research=True, in_signal_sources=True),
    EngineSpec("macd", 13, "engine_macd.py", "SB_MACD_CH", in_blend_live=True, in_blend_research=True, in_signal_sources=True),
    EngineSpec("candle", 20, "engine_candle.py", "SB_CANDLE_CH", in_blend_research=True),
    EngineSpec("rsi_divergence", 26, "engine_rsi_divergence.py", "SB_RSI_DIV_CH", in_blend_research=True, in_signal_sources=True),
    EngineSpec("pivot_engine", 12, "engine_pivot.py", "SB_PIVOT_CH", notes="potential=vol_n*disp +OHLCV+tod+cd+trend+candle; no common attach"),
    EngineSpec("signals", 10, "signals.py", "SB_SIGNALS_CH", notes="side aggregates over ACTIVE_SIGNAL_SOURCES"),
    EngineSpec("common", 0, "engine_common.py", "SB_COMMON_ONLY", notes="common pack only (16 ch); no core"),
    EngineSpec(
        "all_blend",
        134,
        "engine_all_blend.py",
        "SB_ALL_BLEND_CH",
        notes="union of unique cores + common 16 = 150 (maribbon EMA420/500 dropped)",
    ),
)

_SPEC_BY_NAME = {e.name: e for e in ENGINE_SPECS}

BLEND_LIVE: tuple[str, ...] = tuple(e.name for e in ENGINE_SPECS if e.in_blend_live)
BLEND_RESEARCH: tuple[str, ...] = tuple(e.name for e in ENGINE_SPECS if e.in_blend_research)
# Stable order for the signals bus (must match SignalsFeatures.mqh channel layout).
ACTIVE_SIGNAL_SOURCES: tuple[str, ...] = (
    "dbb",
    "rsi_divergence",
    "trend_pullback",
    "smart_money",
    "macd",
)


def core_channels(name: str) -> int:
    key = (name or "").strip().lower()
    if key not in _SPEC_BY_NAME:
        raise KeyError(f"Unknown engine {name!r}; known: {sorted(_SPEC_BY_NAME)}")
    return int(_SPEC_BY_NAME[key].core_channels)


def total_channels(name: str, *, with_common: bool = True) -> int:
    key = (name or "").strip().lower()
    # all_blend embeds a filtered common pack — width ≠ core + full COMMON_CHANNELS.
    if key == "all_blend":
        from smartbs_engines.engine_all_blend import (
            all_blend_core_channel_count,
            all_blend_feature_names,
        )

        if not with_common:
            return all_blend_core_channel_count()
        return len(all_blend_feature_names(with_common=True))
    n = core_channels(name)
    # ``common`` engine *is* the common pack (no second attach).
    if key == "common":
        return COMMON_CHANNELS if with_common else 0
    return n + (COMMON_CHANNELS if with_common else 0)


def max_total_channels(*, with_common: bool = True) -> int:
    return max(total_channels(e.name, with_common=with_common) for e in ENGINE_SPECS)


def render_mql_catalog() -> str:
    """Emit EngineCatalog.mqh content from this table."""
    lines = [
        "//+------------------------------------------------------------------+",
        "//| Auto-generated from smartbs_engines/engine_catalog.py           |",
        "//| Do not edit by hand — run: python -m smartbs_engines.engine_catalog --write-mql",
        "//+------------------------------------------------------------------+",
        "#ifndef SMARTBS_ENGINE_CATALOG_MQH",
        "#define SMARTBS_ENGINE_CATALOG_MQH",
        "",
        f"#define SB_COMMON_CH_CATALOG {COMMON_CHANNELS}",
        f"#define SB_MAX_TOTAL_CH      {max_total_channels()}",
        "",
    ]
    for e in ENGINE_SPECS:
        if e.name == "signals":
            continue  # signals width comes from SignalsFeatures.mqh composition
        lines.append(f"#ifndef {e.mql_define}")
        lines.append(f"#define {e.mql_define} {e.core_channels}  // {e.name}")
        lines.append("#endif")
    lines.extend(
        [
            "",
            "// Live blend pool (Python BLEND_LIVE / ACTIVE_BLEND_ENGINES)",
            "string SB_BlendLiveEngines() { return \""
            + ",".join(BLEND_LIVE)
            + "\"; }",
            "",
            "#endif // SMARTBS_ENGINE_CATALOG_MQH",
            "",
        ]
    )
    return "\n".join(lines)


def write_mql_catalog(
    path: Path | None = None,
) -> Path:
    root = Path(__file__).resolve().parents[1]
    out = path or (root / "mql5" / "Include" / "SmartBSEntry" / "EngineCatalog.mqh")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_mql_catalog(), encoding="utf-8")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-mql", action="store_true")
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()
    if args.write_mql:
        p = write_mql_catalog()
        print(f"Wrote {p}")
    if args.do_print or not args.write_mql:
        for e in ENGINE_SPECS:
            print(
                f"{e.name:16} core={e.core_channels:3d} total={total_channels(e.name):3d}  {e.module}"
            )
        print(f"BLEND_LIVE={BLEND_LIVE}")
        print(f"BLEND_RESEARCH={BLEND_RESEARCH}")
        print(f"ACTIVE_SIGNAL_SOURCES={ACTIVE_SIGNAL_SOURCES}")
        print(f"BACKBONES={BACKBONES}")
        print(f"LABEL_MODES={LABEL_MODES}")


if __name__ == "__main__":
    main()
