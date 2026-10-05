"""Smoke-test rolling-extreme pivot_breakout labels."""

from __future__ import annotations

import numpy as np

from smartbs_engines.config import SmartBSConfig
from smartbs_engines.features import _pivot_breakout_params
from smartbs_engines.labels import label_pivot_breakout


def _case(name: str, high, low, L: int, expect_at: dict) -> None:
    bl = label_pivot_breakout(high, low, pivot_len=L)
    print(f"\n== {name} L={L} n={len(high)} ==")
    for t, exp in expect_at.items():
        lab, res, amb = int(bl.labels[t]), bool(bl.resolved[t]), bool(bl.ambiguous[t])
        ok = lab == exp["lab"] and res == exp["res"] and amb == exp.get("amb", False)
        print(f"  t={t}: lab={lab} res={res} amb={amb} expect={exp} {'OK' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(f"FAIL at t={t} ({name})")


def main() -> None:
    # A: immediate LONG
    _case(
        "immediate LONG",
        np.array([1.0, 1.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]),
        np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_LONG, res=True)},
    )
    # B: immediate SHORT (prior bar owns win_hi so PH is not updated)
    _case(
        "immediate SHORT",
        np.array([3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_SHORT, res=True)},
    )
    # C: both update → ambiguous
    _case(
        "both update FLAT amb",
        np.array([1.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_FLAT, res=False, amb=True)},
    )
    # D: range hold
    _case(
        "range hold FLAT",
        np.array([3.0, 3.0, 2.5, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6]),
        np.array([1.0, 1.0, 1.5, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_FLAT, res=True)},
    )
    # E: forward LONG
    _case(
        "forward LONG",
        np.array([3.0, 3.0, 2.5, 2.6, 3.1, 2.6, 2.6, 2.6, 2.6]),
        np.array([1.0, 1.0, 1.5, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_LONG, res=True)},
    )
    # F: forward SHORT
    _case(
        "forward SHORT",
        np.array([3.0, 3.0, 2.5, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6]),
        np.array([1.0, 1.0, 1.5, 1.4, 0.9, 1.4, 1.4, 1.4, 1.4]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_SHORT, res=True)},
    )
    # G: same-bar both break
    _case(
        "same-bar both break amb",
        np.array([3.0, 3.0, 2.5, 3.1, 2.6, 2.6, 2.6, 2.6, 2.6]),
        np.array([1.0, 1.0, 1.5, 0.9, 1.4, 1.4, 1.4, 1.4, 1.4]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_FLAT, res=False, amb=True)},
    )

    h = np.array([3.0, 3.0, 2.5, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6])
    l = np.array([1.0, 1.0, 1.5, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])
    bl = label_pivot_breakout(h, l, pivot_len=3)
    assert not bl.resolved[0] and not bl.resolved[1], "warmup unresolved"
    assert not any(bl.resolved[-3:]), "tail unresolved"
    print("\nwarmup/tail unresolved OK")

    class _Cfg:
        pivot_len = 15
        barrier_horizon = 99

    plen, hor = _pivot_breakout_params(_Cfg())
    assert plen == 15 and hor == 15, (plen, hor)
    print("params horizon=pivot_len OK", plen, hor)

    try:
        from smartbs_engines.data import fetch_klines

        df = fetch_klines("XAUUSD", "1h", max_candles=5000, source="mt5")
        bl = label_pivot_breakout(
            df["high"].to_numpy(), df["low"].to_numpy(), pivot_len=5
        )
        labs = bl.labels[bl.resolved]
        n = len(labs)
        print("\nXAU sample (resolved only):")
        print(f"  n_resolved={n}/{len(bl.labels)} amb={int(bl.ambiguous.sum())}")
        print(f"  FLAT={(labs == 0).sum()} ({100 * (labs == 0).mean():.1f}%)")
        print(f"  LONG={(labs == 1).sum()} ({100 * (labs == 1).mean():.1f}%)")
        print(f"  SHORT={(labs == 2).sum()} ({100 * (labs == 2).mean():.1f}%)")
    except Exception as exc:  # noqa: BLE001
        print("XAU sample skipped:", exc)

    print("\nALL SMOKE OK")


if __name__ == "__main__":
    main()
