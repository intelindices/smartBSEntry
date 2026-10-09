"""Feature-engine plugins (self-registering)."""

from __future__ import annotations

from smartbs_engines.engines.normalize import (
    FEATURE_ENGINE_MARIBBON,
    engine_warmup_bars,
    normalize_feature_engine,
    resolve_feature_engine,
)
from smartbs_engines.engines.spec import EnginePlugin
from smartbs_engines.plugins.base import PluginRegistry

ENGINE_PLUGINS: PluginRegistry[EnginePlugin] = PluginRegistry("engine")


def register_engine_plugin(plugin: EnginePlugin) -> EnginePlugin:
    return ENGINE_PLUGINS.register(plugin.name, plugin)


def _register_all() -> None:
    from smartbs_engines.engines.common import CommonOnlySTEngine
    from smartbs_engines.engines.dbb import SmartBSDbbEngine
    from smartbs_engines.engines.macd import MACDSTEngine
    from smartbs_engines.engines.maribbon import MARibbonSTEngine
    from smartbs_engines.engines.rsi_divergence import RSIDivergenceSTEngine
    from smartbs_engines.engines.smart_money import SmartMoneySTEngine
    from smartbs_engines.engines.trend_pullback import TrendPullbackSTEngine

    register_engine_plugin(
        EnginePlugin(
            name="maribbon",
            core_channels=39,
            factory=MARibbonSTEngine,
            mql_define="SB_MARIBBON_CH",
            mql_include="MaribbonFeatures.mqh",
            mql_warmup_macro="SB_MARIBBON_WARMUP",
            mql_matrix_ch="SB_MARIBBON_CH",
            mql_build_matrix="SB_BuildMaribbonFeatureMatrix",
            mql_build_window="SB_BuildMaribbonLookbackWindow",
            attach_common=False,
            in_blend_live=True,
            in_blend_research=True,
            in_chart_pack=True,
            notes="13 base ×1h/15m/5m; EMAs 9..240; no common attach",
        )
    )
    register_engine_plugin(
        EnginePlugin(
            name="dbb",
            core_channels=12,
            factory=SmartBSDbbEngine,
            mql_define="SB_DBB_CH",
            mql_include="DbbFeatures.mqh",
            mql_warmup_macro="SB_DBB_WARMUP",
            mql_matrix_ch="SB_NUM_INPUTS",
            mql_build_matrix="SB_BuildDbbFeatureMatrix",
            mql_build_window="SB_BuildDbbLookbackWindow",
            in_blend_live=True,
            in_blend_research=True,
            in_signal_sources=True,
        )
    )
    register_engine_plugin(
        EnginePlugin(
            name="trend_pullback",
            core_channels=15,
            factory=TrendPullbackSTEngine,
            mql_define="SB_TP_CH",
            mql_include="TrendPullbackFeatures.mqh",
            mql_warmup_macro="SB_TP_WARMUP",
            mql_matrix_ch="SB_TP_CH",
            mql_build_matrix="SB_BuildTrendPullbackFeatureMatrix",
            mql_build_window="SB_BuildTrendPullbackLookbackWindow",
            in_blend_live=True,
            in_blend_research=True,
            in_signal_sources=True,
        )
    )
    register_engine_plugin(
        EnginePlugin(
            name="smart_money",
            core_channels=20,
            factory=SmartMoneySTEngine,
            mql_define="SB_SM_CH",
            mql_include="SmartMoneyFeatures.mqh",
            mql_warmup_macro="SB_SM_WARMUP",
            mql_matrix_ch="SB_SM_CH",
            mql_build_matrix="SB_BuildSmartMoneyFeatureMatrix",
            mql_build_window="SB_BuildSmartMoneyLookbackWindow",
            in_blend_live=True,
            in_blend_research=True,
            in_signal_sources=True,
        )
    )
    register_engine_plugin(
        EnginePlugin(
            name="macd",
            core_channels=13,
            factory=MACDSTEngine,
            mql_define="SB_MACD_CH",
            mql_include="MacdFeatures.mqh",
            mql_warmup_macro="SB_MACD_WARMUP",
            mql_matrix_ch="SB_MACD_CH",
            mql_build_matrix="SB_BuildMacdFeatureMatrix",
            mql_build_window="SB_BuildMacdLookbackWindow",
            in_blend_live=True,
            in_blend_research=True,
            in_signal_sources=True,
        )
    )
    register_engine_plugin(
        EnginePlugin(
            name="rsi_divergence",
            core_channels=26,
            factory=RSIDivergenceSTEngine,
            mql_define="SB_RSI_DIV_CH",
            mql_include="RsiDivergenceFeatures.mqh",
            mql_warmup_macro="SB_RSI_DIV_WARMUP",
            mql_matrix_ch="SB_RSI_DIV_CH",
            mql_build_matrix="SB_BuildRsiDivergenceFeatureMatrix",
            mql_build_window="SB_BuildRsiDivergenceLookbackWindow",
            in_blend_research=True,
            in_signal_sources=True,
        )
    )
    register_engine_plugin(
        EnginePlugin(
            name="common",
            core_channels=0,
            factory=CommonOnlySTEngine,
            mql_define="SB_COMMON_ONLY",
            mql_include="CommonChannels.mqh",
            mql_warmup_macro="SB_COMMON_WARMUP",
            attach_common=False,
            in_chart_pack=True,
            notes="common pack only (16 ch); no core",
        )
    )


_register_all()

__all__ = [
    "ENGINE_PLUGINS",
    "EnginePlugin",
    "FEATURE_ENGINE_MARIBBON",
    "engine_warmup_bars",
    "normalize_feature_engine",
    "register_engine_plugin",
    "resolve_feature_engine",
]
