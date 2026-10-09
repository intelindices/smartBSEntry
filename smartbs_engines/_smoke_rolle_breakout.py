"""Smoke-test rolling-extreme rolle_breakout labels (forward-update rule)."""

from __future__ import annotations

import numpy as np

from smartbs_engines.config import SmartBSConfig
from smartbs_engines.features import _rolle_breakout_params
from smartbs_engines.labels import label_rolle_breakout


def _case(name: str, high, low, L: int, expect_at: dict) -> None:
    bl = label_rolle_breakout(high, low, rolle_len=L)
    print(f"\n== {name} L={L} n={len(high)} ==")
    for t, exp in expect_at.items():
        lab, res, amb = int(bl.labels[t]), bool(bl.resolved[t]), bool(bl.ambiguous[t])
        ok = lab == exp["lab"] and res == exp["res"] and amb == exp.get("amb", False)
        print(f"  t={t}: lab={lab} res={res} amb={amb} expect={exp} {'OK' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(f"FAIL at t={t} ({name})")


def main() -> None:
    # L=3: at t, look at bars t+1..t+3 for RollE updates.
    # A: only REH in forward window -> LONG (rising lows so no REL)
    _case(
        "forward REH only -> LONG",
        np.array([2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0]),
        np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_LONG, res=True)},
    )
    # B: only REL in forward window -> SHORT (falling highs so no REH)
    _case(
        "forward REL only -> SHORT",
        np.array([3.0, 3.0, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4]),
        np.array([2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_SHORT, res=True)},
    )
    # C: both REH and REL in forward window -> FLAT amb
    _case(
        "forward both -> FLAT amb",
        np.array([2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0]),
        np.array([1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 1.5, 1.5, 1.5]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_FLAT, res=False, amb=True)},
    )
    # D: nothing in forward window -> FLAT resolved
    _case(
        "forward none -> FLAT",
        np.array([3.0, 3.0, 2.5, 2.6, 2.55, 2.55, 2.55, 2.55, 2.55]),
        np.array([1.0, 1.0, 1.5, 1.4, 1.45, 1.45, 1.45, 1.45, 1.45]),
        3,
        {2: dict(lab=SmartBSConfig.CLASS_FLAT, res=True)},
    )

    h = np.array([3.0, 3.0, 2.5, 2.6, 2.55, 2.55, 2.55, 2.55, 2.55])
    l = np.array([1.0, 1.0, 1.5, 1.4, 1.45, 1.45, 1.45, 1.45, 1.45])
    bl = label_rolle_breakout(h, l, rolle_len=3)
    assert not any(bl.resolved[-3:]), "tail unresolved"
    print("\ntail unresolved OK")

    class _Cfg:
        rolle_len = 5
        barrier_horizon = 99

    plen, hor = _rolle_breakout_params(_Cfg())
    assert plen == 5 and hor == 5, (plen, hor)
    print("params horizon=rolle_len OK", plen, hor)

    try:
        from smartbs_engines.data import fetch_klines

        df = fetch_klines("XAUUSD", "1h", max_candles=5000, source="mt5")
        bl = label_rolle_breakout(
            df["high"].to_numpy(), df["low"].to_numpy(), rolle_len=5
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
