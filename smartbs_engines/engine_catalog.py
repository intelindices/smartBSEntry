"""Shim — catalog SSOT lives in ``smartbs_engines.catalog``."""

from __future__ import annotations

from smartbs_engines.catalog import (  # noqa: F401
    ACTIVE_SIGNAL_SOURCES,
    BACKBONES,
    BLEND_LIVE,
    BLEND_RESEARCH,
    CHART_PACK_ENGINES,
    COMMON_CHANNELS,
    COMMON_FEATURE_NAMES,
    ENGINE_SPECS,
    EngineSpec,
    LABEL_MODES,
    core_channels,
    max_total_channels,
    render_mql_catalog,
    total_channels,
    write_mql_all,
)


def write_mql_catalog(path=None):
    from pathlib import Path

    from smartbs_engines.catalog import render_mql_catalog, _mql_include_dir

    out = Path(path) if path else (_mql_include_dir() / "EngineCatalog.mqh")
    out.write_text(render_mql_catalog(), encoding="utf-8")
    return out


def main() -> None:
    from smartbs_engines.catalog import main as _main

    _main()


if __name__ == "__main__":
    main()
