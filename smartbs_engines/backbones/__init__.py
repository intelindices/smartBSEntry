"""Backbone plugins (tcn / V2 / smartBSTF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from smartbs_engines.plugins.base import PluginRegistry

BACKBONE_PLUGINS: PluginRegistry["BackbonePlugin"] = PluginRegistry("backbone")

BACKBONE_TCN = "tcn"
BACKBONE_SMARTBS_ENTRY = "smartBSEntry"
BACKBONE_SMARTBS_ENTRY_V2 = "smartBSEntryV2"
BACKBONE_SMARTBS_TF = "smartBSTF"


@dataclass(frozen=True)
class BackbonePlugin:
    name: str
    aliases: tuple[str, ...]
    default_kernel: int
    stem_tag: str
    factory: Callable[..., Any]


def register_backbone_plugin(plugin: BackbonePlugin) -> BackbonePlugin:
    BACKBONE_PLUGINS.register(plugin.name, plugin)
    for a in plugin.aliases:
        BACKBONE_PLUGINS.register(a, plugin)
    return plugin


def _register_all() -> None:
    from smartbs_engines.model import SmartBSClassifier, SmartBSEntryV2, SmartBSTF

    def _tcn(**kw):
        return SmartBSClassifier(**kw)

    def _v2(**kw):
        return SmartBSEntryV2(**kw)

    def _tf(**kw):
        return SmartBSTF(**kw)

    register_backbone_plugin(
        BackbonePlugin(
            name=BACKBONE_TCN,
            aliases=("smartbsentry", "classic", "v1"),
            default_kernel=3,
            stem_tag="tcn",
            factory=_tcn,
        )
    )
    register_backbone_plugin(
        BackbonePlugin(
            name=BACKBONE_SMARTBS_ENTRY_V2,
            aliases=("moderntcn", "v2"),
            default_kernel=7,
            stem_tag="v2",
            factory=_v2,
        )
    )
    register_backbone_plugin(
        BackbonePlugin(
            name=BACKBONE_SMARTBS_TF,
            aliases=("transformer", "tf", "attn", "attention"),
            default_kernel=1,
            stem_tag="tf",
            factory=_tf,
        )
    )


def normalize_backbone(name: str | None) -> str:
    from smartbs_engines.plugins.base import ensure_plugins_loaded

    ensure_plugins_loaded()
    raw = str(name or BACKBONE_TCN).strip()
    key = raw.lower().replace("-", "").replace("_", "")
    if not key:
        return BACKBONE_TCN
    if key in BACKBONE_PLUGINS:
        return BACKBONE_PLUGINS.get(key).name
    if raw in BACKBONE_PLUGINS:
        return BACKBONE_PLUGINS.get(raw).name
    for p in BACKBONE_PLUGINS.values():
        if raw == p.name:
            return p.name
    raise ValueError(
        f"Unknown backbone {name!r}; expected one of "
        f"{sorted({p.name for p in BACKBONE_PLUGINS.values()})}"
    )


def default_kernel_size(backbone: str | None) -> int:
    return int(BACKBONE_PLUGINS.get(normalize_backbone(backbone)).default_kernel)


def backbone_stem_tag(backbone: str | None) -> str:
    return BACKBONE_PLUGINS.get(normalize_backbone(backbone)).stem_tag


def list_backbones() -> list[str]:
    from smartbs_engines.plugins.base import ensure_plugins_loaded

    ensure_plugins_loaded()
    return sorted({p.name for p in BACKBONE_PLUGINS.values()})


def build_classifier(
    *,
    num_inputs: int,
    num_channels: list[int],
    num_classes: int = 3,
    kernel_size: int | None = None,
    dropout: float = 0.2,
    backbone: str | None = None,
    num_inputs_15m: int | None = None,
    tf_ratio: int = 4,
):
    del num_inputs_15m, tf_ratio
    bb = normalize_backbone(backbone)
    plugin = BACKBONE_PLUGINS.get(bb)
    ks = plugin.default_kernel if kernel_size is None else int(kernel_size)
    return plugin.factory(
        num_inputs=num_inputs,
        num_channels=list(num_channels),
        num_classes=num_classes,
        kernel_size=ks,
        dropout=dropout,
    )


_register_all()

__all__ = [
    "BACKBONE_PLUGINS",
    "BACKBONE_SMARTBS_ENTRY",
    "BACKBONE_SMARTBS_ENTRY_V2",
    "BACKBONE_SMARTBS_TF",
    "BACKBONE_TCN",
    "BackbonePlugin",
    "backbone_stem_tag",
    "build_classifier",
    "default_kernel_size",
    "list_backbones",
    "normalize_backbone",
    "register_backbone_plugin",
]
