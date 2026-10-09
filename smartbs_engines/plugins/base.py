"""Generic plugin registry + one-shot discovery loader."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")

_LOADED = False


class PluginRegistry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> T:
        key = str(name or "").strip().lower()
        if not key:
            raise ValueError(f"{self.kind}: empty plugin name")
        self._items[key] = item
        return item

    def get(self, name: str) -> T:
        key = str(name or "").strip().lower()
        if key not in self._items:
            raise KeyError(
                f"Unknown {self.kind} {name!r}; known: {sorted(self._items)}"
            )
        return self._items[key]

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[tuple[str, T]]:
        return sorted(self._items.items(), key=lambda kv: kv[0])

    def values(self) -> list[T]:
        return [v for _, v in self.items()]

    def __contains__(self, name: object) -> bool:
        return str(name or "").strip().lower() in self._items


def ensure_plugins_loaded() -> None:
    """Import all plugin packages once so self-registration runs."""
    global _LOADED
    if _LOADED:
        return
    import smartbs_engines.engines  # noqa: F401
    import smartbs_engines.backbones  # noqa: F401
    import smartbs_engines.labels  # noqa: F401
    import smartbs_engines.policies  # noqa: F401
    _LOADED = True
