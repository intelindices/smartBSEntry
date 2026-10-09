"""Engine plugin specification (Python + MQL codegen metadata)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class EnginePlugin:
    name: str
    core_channels: int
    factory: Callable[[], Any]
    mql_define: str
    mql_include: str = ""
    mql_warmup_macro: str = ""
    mql_matrix_ch: str = ""
    mql_build_matrix: str = ""
    mql_build_window: str = ""
    attach_common: bool = True
    in_blend_live: bool = False
    in_blend_research: bool = False
    in_chart_pack: bool = False
    in_signal_sources: bool = False
    notes: str = ""

    @property
    def is_common_only(self) -> bool:
        return self.name == "common"
