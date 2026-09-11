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

_DEFAULT_COUNTS = {
    "1m": 80_000,
    "5m": 80_000,
    "15m": 80_000,
    "1h": 70_000,
    "4h": 20_000,
    "1d": 4_000,
    "1w": 600,
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
}

DEFAULT_TRAIN_ASSETS = (
    "XAUUSD",
    "XAGUSD",
    "XTIUSD",
    "NATGAS",
    "PLATINUM",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDNZD",
    "USDCAD",
    "BTCUSD",
    "ETHUSD",
    "LTCUSD",
    "BCHUSD",
    "ADAUSD",
)


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
    """Pull the longest available bars from the live terminal (broker clock)."""
    mt5 = _init_mt5()
    tf_name = _TF_TO_MT5.get(interval)
    if tf_name is None:
        raise ValueError(f"unsupported MT5 interval: {interval}")
    timeframe = getattr(mt5, tf_name)
    ticket = resolve_mt5_symbol(symbol, mt5)
    want = int(max_candles or _DEFAULT_COUNTS.get(interval, 10_000))

    counts = [want]
    for step in (80_000, 60_000, 40_000, 20_000, 10_000):
        if step < want and step not in counts:
            counts.append(step)

    last_err = None
    for count in counts:
        rates = mt5.copy_rates_from_pos(ticket, timeframe, 0, count)
        if rates is not None and len(rates) > 0:
            df = _rates_to_df(rates)
            if ticket != symbol.replace("/", "").upper():
                print(f"  MT5 {symbol} -> {ticket}")
            return df
        last_err = mt5.last_error()

    start = datetime(2016, 9, 1)
    rates = mt5.copy_rates_from(ticket, timeframe, start, want)
    if rates is not None and len(rates) > 0:
        return _rates_to_df(rates)
    raise RuntimeError(f"copy_rates {ticket} {interval} failed: {last_err or mt5.last_error()}")


def save_mt5_klines(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, interval)
    df.to_parquet(path, index=False)
    return path


def load_mt5_klines(
    symbol: str,
    interval: str,
    max_candles: int | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame | None:
    """Cache-first load. ``refresh=True`` hits the terminal and rewrites parquet."""
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

    if refresh or not path.is_file():
        try:
            df = copy_mt5_klines(symbol, interval, max_candles)
        except Exception as exc:
            if path.is_file():
                print(f"MT5 live fetch failed ({exc}); using cache {path.name}")
                df = pd.read_parquet(path)
            else:
                print(f"MT5 {symbol} {interval}: {exc}")
                return None
        else:
            save_mt5_klines(df, canon, interval)
            try:
                mt5 = _init_mt5()
                ticket = resolve_mt5_symbol(symbol, mt5)
                if ticket != canon:
                    save_mt5_klines(df, ticket, interval)
            except Exception:
                pass
    else:
        df = pd.read_parquet(path)
    if df is None or df.empty:
        return None
    if max_candles and len(df) > max_candles:
        df = df.iloc[-int(max_candles) :].reset_index(drop=True)
    return df


def dump_symbol(
    symbol: str = "XAUUSD",
    intervals: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d", "1w"),
) -> dict[str, int]:
    """Download TFs used by ST engines from the logged-in terminal."""
    counts: dict[str, int] = {}
    for tf in intervals:
        print(f"  copying {symbol} {tf}…", flush=True)
        df = load_mt5_klines(symbol, tf, refresh=True)
        n = 0 if df is None else len(df)
        counts[tf] = n
        if df is not None and n:
            t0 = pd.to_datetime(df["open_time"].iloc[0], unit="ms", utc=True)
            t1 = pd.to_datetime(df["open_time"].iloc[-1], unit="ms", utc=True)
            print(f"  {symbol} {tf}: {n} bars  {t0} -> {t1}", flush=True)
    return counts


def dump_assets(symbols: list[str] | tuple[str, ...] = DEFAULT_TRAIN_ASSETS) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for sym in symbols:
        print(f"Dumping {sym} from MetaTrader 5…", flush=True)
        try:
            out[sym] = dump_symbol(sym)
        except Exception as exc:
            print(f"  skip {sym}: {exc}", flush=True)
            out[sym] = {}
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Dump MT5 broker bars for SmartBS training")
    ap.add_argument("--symbol", default="", help="Single symbol (default: dump the full train set)")
    ap.add_argument(
        "--assets",
        default="",
        help="Comma-separated symbols (default: commodities + majors + crypto)",
    )
    args = ap.parse_args()
    if args.symbol.strip():
        print(f"Dumping {args.symbol} from MetaTrader 5…", flush=True)
        dump_symbol(args.symbol.strip())
        return
    assets = [a.strip() for a in args.assets.split(",") if a.strip()] or list(DEFAULT_TRAIN_ASSETS)
    dump_assets(assets)


if __name__ == "__main__":
    main()
