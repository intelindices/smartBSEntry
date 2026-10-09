"""Load OHLCV from a running MetaTrader 5 terminal (same bars the tester uses).

Requires the ``MetaTrader5`` package and a logged-in terminal. Bars are cached
under ``data_cache/mt5/`` so training can reuse a dump.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CACHE_DIR = Path(os.environ.get("SMARTBS_MT5_DIR", REPO_ROOT / "data_cache" / "mt5"))

_TF_TO_MT5 = {
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
    "1w": "TIMEFRAME_W1",
}

# Ask for the longest practical history; MT5 servers often cap ~80–100k bars.
_DEFAULT_COUNTS = {
    "1m": 80_000,
    "5m": 200_000,
    "15m": 80_000,
    "1h": 80_000,
    "4h": 40_000,
    "1d": 10_000,
    "1w": 2_000,
}

_DEFAULT_TERMINAL = r"C:\Program Files\MetaTrader 5\terminal64.exe"

SYMBOL_CANDIDATES: dict[str, list[str]] = {
    "XAUUSD": ["XAUUSD"],
    "XAGUSD": ["XAGUSD"],
    "XTIUSD": ["XTIUSD", "USOUSD", "WTIUSD"],
    "NATGAS": ["XNGUSD", "NATGAS", "NGAS"],
    "NATGASUSDC": ["XNGUSD", "NATGAS", "NGAS"],
    "PLATINUM": ["XPTUSD", "PLATINUM"],
    "PLATINUMUSDC": ["XPTUSD", "PLATINUM"],
    "EURUSD": ["EURUSD"],
    "GBPUSD": ["GBPUSD"],
    "AUDUSD": ["AUDUSD"],
    "USDJPY": ["USDJPY"],
    "USDCHF": ["USDCHF"],
    "USDNZD": ["NZDUSD", "USDNZD"],
    "NZDUSD": ["NZDUSD"],
    "USDCAD": ["USDCAD"],
    "BTCUSD": ["BTCUSD"],
    "ETHUSD": ["ETHUSD"],
    "LTCUSD": ["LTCUSD"],
    "BCHUSD": ["BCHUSD"],
    "ADAUSD": ["ADAUSD"],
    "SOLUSD": ["SOLUSD", "SOLUSDT"],
}

# Permanent dump set (15m / 1h / 4h) used by train + EA.
DEFAULT_DUMP_ASSETS = (
    "XAUUSD",
    "XAGUSD",
    "XTIUSD",
    "NATGAS",
    "PLATINUM",
    "BTCUSD",
    "ETHUSD",
    "BCHUSD",
    "LTCUSD",
    "ADAUSD",
    "SOLUSD",
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "USDCHF",
    "USDJPY",
)
DEFAULT_DUMP_INTERVALS = ("5m", "15m", "1h", "4h")

DEFAULT_TRAIN_ASSETS = DEFAULT_DUMP_ASSETS


def _cache_path(symbol: str, interval: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_{interval}.parquet"


def _init_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise ImportError("pip install MetaTrader5") from exc

    if mt5.initialize():
        return mt5
    path = os.environ.get("SMARTBS_MT5_TERMINAL", _DEFAULT_TERMINAL)
    if path and os.path.isfile(path) and mt5.initialize(path):
        return mt5
    raise RuntimeError(f"MetaTrader5.initialize failed: {mt5.last_error()}")


def resolve_mt5_symbol(symbol: str, mt5=None) -> str:
    """Map package names (NATGAS, USDNZD, …) to a ticket the terminal knows."""
    want = symbol.replace("/", "").upper()
    own = mt5 is None
    if own:
        mt5 = _init_mt5()
    try:
        for cand in SYMBOL_CANDIDATES.get(want, [want]):
            if mt5.symbol_info(cand) is not None:
                mt5.symbol_select(cand, True)
                return cand
        if mt5.symbol_info(want) is not None:
            mt5.symbol_select(want, True)
            return want
        raise RuntimeError(
            f"MT5 symbol not found: {symbol} (tried {SYMBOL_CANDIDATES.get(want, [want])})"
        )
    finally:
        if own:
            pass


def _rates_to_df(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    out = pd.DataFrame(
        {
            "open_time": (df["time"].astype("int64") * 1000),
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "volume": df["tick_volume"].astype("float64") if "tick_volume" in df.columns else 0.0,
        }
    )
    return out.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


def copy_mt5_klines(symbol: str, interval: str, max_candles: int | None = None) -> pd.DataFrame:
    """Pull the longest available bars from the live terminal (broker clock).

    MT5 ``copy_rates_from_pos(…, 0, N)`` only returns the *most recent* N bars
    (IC Markets often caps a single call at ~80k). We page backward with
    increasing ``start_pos`` and stitch until the terminal has no more history —
    that is the same depth the EA can see once charts are fully loaded.
    """
    mt5 = _init_mt5()
    tf_name = _TF_TO_MT5.get(interval)
    if tf_name is None:
        raise ValueError(f"unsupported MT5 interval: {interval}")
    timeframe = getattr(mt5, tf_name)
    ticket = resolve_mt5_symbol(symbol, mt5)
    if ticket != symbol.replace("/", "").upper():
        print(f"  MT5 {symbol} -> {ticket}", flush=True)

    page = 20_000
    # Soft upper bound so a wedged terminal cannot loop forever.
    hard_cap = int(max_candles) if max_candles else 2_000_000
    frames: list[pd.DataFrame] = []
    pos = 0
    last_err = None
    while pos < hard_cap:
        rates = mt5.copy_rates_from_pos(ticket, timeframe, pos, page)
        if rates is None or len(rates) == 0:
            last_err = mt5.last_error()
            break
        frames.append(_rates_to_df(rates))
        n = len(rates)
        pos += n
        if n < page:
            break

    if not frames:
        # Fallback: single-shot from an early date.
        want = int(max_candles or _DEFAULT_COUNTS.get(interval, 10_000))
        rates = mt5.copy_rates_from(ticket, timeframe, datetime(2016, 9, 1), want)
        if rates is not None and len(rates) > 0:
            df = _rates_to_df(rates)
        else:
            raise RuntimeError(
                f"copy_rates {ticket} {interval} failed: {last_err or mt5.last_error()}"
            )
    else:
        df = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True)
        )

    if max_candles and len(df) > int(max_candles):
        df = df.iloc[-int(max_candles) :].reset_index(drop=True)
    return df


def save_mt5_klines(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, interval)
    df.to_parquet(path, index=False)
    return path


def merge_ohlcv(existing: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """Union bars by ``open_time`` (keep longest history; prefer fresh on ties)."""
    if existing is None or existing.empty:
        return fresh.reset_index(drop=True)
    if fresh is None or fresh.empty:
        return existing.reset_index(drop=True)
    out = (
        pd.concat([existing, fresh], ignore_index=True)
        .drop_duplicates("open_time", keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return out


def load_mt5_klines(
    symbol: str,
    interval: str,
    max_candles: int | None = None,
    *,
    refresh: bool = False,
    merge_cache: bool = True,
) -> pd.DataFrame | None:
    """Cache-first load. ``refresh=True`` hits the terminal and rewrites parquet.

    When ``merge_cache=True`` (default), newly fetched bars are unioned with any
    existing parquet so earlier history is never discarded.
    """
    canon = symbol.replace("/", "").upper()
    aliases = [canon] + [c for c in SYMBOL_CANDIDATES.get(canon, []) if c != canon]
    path = None
    for name in aliases:
        cand = _cache_path(name, interval)
        if cand.is_file():
            path = cand
            break
    if path is None:
        path = _cache_path(canon, interval)

    cached: pd.DataFrame | None = None
    if path.is_file():
        try:
            cached = pd.read_parquet(path)
        except Exception:
            cached = None

    if refresh or cached is None or cached.empty:
        try:
            df = copy_mt5_klines(symbol, interval, max_candles)
        except Exception as exc:
            if cached is not None and not cached.empty:
                print(f"MT5 live fetch failed ({exc}); using cache {path.name}")
                df = cached
            else:
                print(f"MT5 {symbol} {interval}: {exc}")
                return None
        else:
            if merge_cache and cached is not None and not cached.empty:
                before = len(cached)
                df = merge_ohlcv(cached, df)
                added = len(df) - before
                if added > 0:
                    print(
                        f"  merged +{added} bars into cache "
                        f"({before} -> {len(df)})",
                        flush=True,
                    )
            save_mt5_klines(df, canon, interval)
            try:
                mt5 = _init_mt5()
                ticket = resolve_mt5_symbol(symbol, mt5)
                if ticket != canon:
                    # Also merge under broker ticket name if different.
                    alt = _cache_path(ticket, interval)
                    alt_cached = pd.read_parquet(alt) if alt.is_file() else None
                    save_mt5_klines(
                        merge_ohlcv(alt_cached, df) if merge_cache else df,
                        ticket,
                        interval,
                    )
            except Exception:
                pass
    else:
        df = cached
    if df is None or df.empty:
        return None
    if max_candles and len(df) > max_candles:
        df = df.iloc[-int(max_candles) :].reset_index(drop=True)
    return df


def _find_cache_path(symbol: str, interval: str) -> Path | None:
    """First existing parquet for symbol or broker alias."""
    canon = symbol.replace("/", "").upper()
    aliases = [canon] + [c for c in SYMBOL_CANDIDATES.get(canon, []) if c != canon]
    for name in aliases:
        cand = _cache_path(name, interval)
        if cand.is_file():
            return cand
    return None


def _cache_span_years(symbol: str, interval: str) -> float | None:
    """Return cached series span in years, or None if missing/unreadable."""
    path = _find_cache_path(symbol, interval)
    if path is None:
        return None
    try:
        df = pd.read_parquet(path, columns=["open_time"])
    except Exception:
        return None
    if df is None or df.empty:
        return None
    t0 = pd.to_datetime(df["open_time"].iloc[0], unit="ms", utc=True)
    t1 = pd.to_datetime(df["open_time"].iloc[-1], unit="ms", utc=True)
    return float((t1 - t0).total_seconds()) / (365.25 * 24 * 3600)


def _max_candles_for_years(interval: str, years: float) -> int:
    """Upper bound on bars to request for ``years`` of history (24/7 markets)."""
    y = max(float(years), 0.25)
    per_year = {
        "1m": 366 * 24 * 60,
        "5m": 366 * 24 * 12,
        "15m": 366 * 24 * 4,
        "1h": 366 * 24,
        "4h": 366 * 6,
        "1d": 366,
        "1w": 54,
    }.get(interval, 80_000)
    # Headroom for DST / broker quirks; floor at existing default counts.
    return max(int(per_year * y * 1.05) + 5_000, int(_DEFAULT_COUNTS.get(interval, 10_000)))


def dump_symbol(
    symbol: str = "XAUUSD",
    intervals: tuple[str, ...] = DEFAULT_DUMP_INTERVALS,
    *,
    min_years: float = 0.0,
    skip_existing: bool = False,
) -> dict[str, int]:
    """Download TFs from the logged-in terminal into ``data_cache/mt5/``.

    When ``skip_existing`` and ``min_years`` > 0, skip any TF whose cache already
    spans at least ``min_years`` (no re-download).
    """
    counts: dict[str, int] = {}
    want_years = float(min_years) if min_years and min_years > 0 else 0.0
    for tf in intervals:
        if skip_existing and want_years > 0:
            span = _cache_span_years(symbol, tf)
            path = _find_cache_path(symbol, tf)
            if span is not None and path is not None and span + 1e-9 >= want_years:
                print(
                    f"  skip {symbol} {tf}: cache already {span:.2f}y "
                    f"(need {want_years:g}y) -> {path.name}",
                    flush=True,
                )
                try:
                    n = len(pd.read_parquet(path, columns=["open_time"]))
                except Exception:
                    n = 0
                counts[tf] = n
                continue

        max_n = _max_candles_for_years(tf, want_years or 10.0)
        print(f"  copying {symbol} {tf} (max~{max_n})...", flush=True)
        df = load_mt5_klines(symbol, tf, max_candles=max_n, refresh=True)
        n = 0 if df is None else len(df)
        counts[tf] = n
        if df is not None and n:
            t0 = pd.to_datetime(df["open_time"].iloc[0], unit="ms", utc=True)
            t1 = pd.to_datetime(df["open_time"].iloc[-1], unit="ms", utc=True)
            span = (t1 - t0).total_seconds() / (365.25 * 24 * 3600)
            print(f"  {symbol} {tf}: {n} bars  {t0} -> {t1}  ({span:.2f}y)", flush=True)
        else:
            print(f"  {symbol} {tf}: FAILED / empty", flush=True)
    return counts


def dump_assets(
    symbols: list[str] | tuple[str, ...] = DEFAULT_DUMP_ASSETS,
    intervals: tuple[str, ...] = DEFAULT_DUMP_INTERVALS,
    *,
    min_years: float = 0.0,
    skip_existing: bool = False,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Intervals: {intervals}")
    if skip_existing and min_years > 0:
        print(f"Skip if cache span >= {min_years:g}y", flush=True)
    for sym in symbols:
        print(f"Dumping {sym} from MetaTrader 5...", flush=True)
        try:
            out[sym] = dump_symbol(
                sym,
                intervals=intervals,
                min_years=min_years,
                skip_existing=skip_existing,
            )
        except Exception as exc:
            print(f"  skip {sym}: {exc}", flush=True)
            out[sym] = {}
    return out


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Dump MT5 broker bars for SmartBS training")
    ap.add_argument("--symbol", default="", help="Single symbol (default: dump the full set)")
    ap.add_argument(
        "--assets",
        default="",
        help="Comma-separated symbols (default: dump set)",
    )
    ap.add_argument(
        "--intervals",
        default=",".join(DEFAULT_DUMP_INTERVALS),
        help="Comma-separated TFs (default: 15m,1h,4h)",
    )
    ap.add_argument(
        "--min-years",
        type=float,
        default=0.0,
        help="Target history depth in years (sets fetch size; used with --skip-existing)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip TF if cache already spans --min-years",
    )
    args = ap.parse_args()
    intervals = tuple(x.strip() for x in args.intervals.split(",") if x.strip())
    min_years = float(args.min_years)
    skip_existing = bool(args.skip_existing)
    if args.symbol.strip():
        print(f"Dumping {args.symbol} from MetaTrader 5...", flush=True)
        print(f"Cache dir: {CACHE_DIR}")
        dump_symbol(
            args.symbol.strip(),
            intervals=intervals,
            min_years=min_years,
            skip_existing=skip_existing,
        )
        return
    assets = [a.strip() for a in args.assets.split(",") if a.strip()] or list(DEFAULT_DUMP_ASSETS)
    summary = dump_assets(
        assets,
        intervals=intervals,
        min_years=min_years,
        skip_existing=skip_existing,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nWrote parquets under {CACHE_DIR}")


if __name__ == "__main__":
    main()
