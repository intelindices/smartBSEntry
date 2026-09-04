"""Windows consoles default to cp1252 and crash on arrows like ``→``.

Ubuntu is already UTF-8; reconfiguring to UTF-8 there is a no-op. On Windows it
stops ``print`` from raising ``UnicodeEncodeError`` mid-training and skipping
the asset as if the data were missing.
"""

from __future__ import annotations

import sys


def configure_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass
