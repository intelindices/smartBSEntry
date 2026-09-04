"""OHLCV fetchers for the SmartBS strategy (TradingView, Yahoo, Binance)."""

from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd
import requests


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Map our interval strings → tvDatafeed Interval enum names
_TV_INTERVAL_MAP = {
    "1m": "in_1_minute",
    "3m": "in_3_minute",
    "5m": "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "45m": "in_45_minute",
    "1h": "in_1_hour",
    "60": "in_1_hour",
    "1H": "in_1_hour",
    "2h": "in_2_hour",
    "3h": "in_3_hour",
    "4h": "in_4_hour",
    "1d": "in_daily",
    "1D": "in_daily",
    "1w": "in_weekly",
    "1W": "in_weekly",
}

# Yahoo intervals / gold proxies for XAUUSD when TV history is capped at 5000 bars
_YAHOO_INTERVAL_MAP = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60": "1h",
    "1h": "1h",
    "1H": "1h",
    "1d": "1d",
    "1D": "1d",
    "1w": "1wk",
    "1W": "1wk",
}

_YAHOO_SYMBOL_MAP = {
    "XAUUSD": "GC=F",
    "GOLDUSDC": "GC=F",
    "XAGUSD": "SI=F",
    "SILVERUSDC": "SI=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "BTCUSD": "BTC-USD",
    "BTCUSDC": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "ETHUSDC": "ETH-USD",
    "SOLUSDC": "SOL-USD",
    "XTIUSD": "CL=F",
    "WTIOILUSDC": "CL=F",
    "COPPERUSDC": "HG=F",
    "NATGASUSDC": "NG=F",
    "PLATINUMUSDC": "PL=F",
}

# Specs used by multi-asset AI training / per-pair inference (symbol → fetch knobs)
TRAIN_ASSET_SPECS: dict[str, dict] = {
    "XAUUSD": {"symbol": "XAUUSD", "source": "yahoo", "exchange": "OANDA"},
    "XAGUSD": {"symbol": "XAGUSD", "source": "yahoo", "exchange": "OANDA"},
    "EURUSD": {"symbol": "EURUSD", "source": "yahoo", "exchange": "OANDA"},
    "XTIUSD": {"symbol": "XTIUSD", "source": "yahoo", "exchange": "NYMEX"},
    "COPPERUSDC": {"symbol": "COPPERUSDC", "source": "yahoo", "exchange": "COMEX"},
    "NATGASUSDC": {"symbol": "NATGASUSDC", "source": "yahoo", "exchange": "NYMEX"},
    "PLATINUMUSDC": {"symbol": "PLATINUMUSDC", "source": "yahoo", "exchange": "NYMEX"},
    "ETHUSD": {
        "symbol": "ETHUSD",
        "source": "yahoo",
        "exchange": "BINANCE",
        "binance_symbol": "ETHUSDT",
    },
    "BTCUSD": {
        "symbol": "BTCUSD",
        "source": "yahoo",
        "exchange": "BINANCE",
        "binance_symbol": "BTCUSDT",
    },
    "BTCUSDC": {
        "symbol": "BTCUSDC",
        "source": "yahoo",
        "exchange": "BINANCE",
        "binance_symbol": "BTCUSDT",
    },
    "ADAUSD": {
        "symbol": "ADAUSD",
        "source": "yahoo",
        "exchange": "BINANCE",
        "binance_symbol": "ADAUSDT",
    },
    "LTCUSD": {
        "symbol": "LTCUSD",
        "source": "yahoo",
        "exchange": "BINANCE",
        "binance_symbol": "LTCUSDT",
    },
    "BCHUSD": {
        "symbol": "BCHUSD",
        "source": "yahoo",
        "exchange": "BINANCE",
        "binance_symbol": "BCHUSDT",
    },
}


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns: open_time, open, high, low, close, volume."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    out = df.copy()
    if "open_time" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
        rename = {}
        for cand in ("datetime", "date", "time", "Datetime", "Date", "index"):
            if cand in out.columns:
                rename[cand] = "open_time"
                break
        out = out.rename(columns=rename)

    if "open_time" not in out.columns:
        raise ValueError(f"Could not find timestamp column in frame: {list(out.columns)}")

    if pd.api.types.is_numeric_dtype(out["open_time"]):
        sample = float(out["open_time"].iloc[-1])
        out["open_time"] = out["open_time"].astype("int64")
        if sample < 1e11:
            out["open_time"] = out["open_time"] * 1000
    else:
        ts = pd.to_datetime(out["open_time"], utc=True)
        out["open_time"] = (ts.astype("int64") // 10**6).astype("int64")

    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            alt = col.capitalize()
            if alt in out.columns:
                out[col] = out[alt]
            elif col == "volume":
                out[col] = 0.0
            else:
                raise ValueError(f"Missing column {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return out[["open_time", "open", "high", "low", "close", "volume"]]


def fetch_tradingview_klines(
    symbol: str = "XAUUSD",
    exchange: str = "OANDA",
    interval: str = "1h",
    max_candles: int = 5000,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV from TradingView via ``tvDatafeed`` (hard cap ~5000 bars)."""
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError as e:
        raise ImportError(
            "TradingView data requires tvDatafeed. Install with:\n"
            "  pip install tradingview-datafeed"
        ) from e

    interval_key = _TV_INTERVAL_MAP.get(interval, interval)
    if not hasattr(Interval, interval_key):
        raise ValueError(f"Unsupported TradingView interval '{interval}'. Known: {sorted(_TV_INTERVAL_MAP)}")
    tv_interval = getattr(Interval, interval_key)

    user = username if username is not None else os.environ.get("TRADINGVIEW_USERNAME")
    pwd = password if password is not None else os.environ.get("TRADINGVIEW_PASSWORD")
    tv = TvDatafeed(username=user, password=pwd) if user and pwd else TvDatafeed()

    n_bars = min(int(max_candles), 5000)
    raw = tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=n_bars)
    if raw is None or len(raw) == 0:
        raise ConnectionError(
            f"TradingView returned no data for {exchange}:{symbol} ({interval}). "
            "Try FX_IDC or use --data-source yahoo for longer history."
        )
    return _normalize_ohlcv(raw)


def fetch_yahoo_klines(
    symbol: str = "XAUUSD",
    interval: str = "1h",
    max_candles: int = 43800,
) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance.

    Notes:
    - GC=F / forex 1H is typically capped near ~730 calendar days by Yahoo.
    - Requests for longer windows still ask for max history and truncate to ``max_candles``.
    """
    # Keep yfinance sqlite cache inside the repo so sandboxed runs can write.
    cache_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Yahoo data requires yfinance: pip install yfinance") from e

    y_symbol = _YAHOO_SYMBOL_MAP.get(symbol.replace("/", "").upper(), symbol)
    y_interval = _YAHOO_INTERVAL_MAP.get(interval, interval)

    # period must cover requested history; 1h is hard-capped by Yahoo (~2y)
    if y_interval in ("1h", "60m", "90m"):
        period = "max" if max_candles > 8760 else "2y"
    elif y_interval in ("1m", "2m", "5m", "15m", "30m"):
        period = "60d"
    else:
        period = "max"

    raw = yf.download(
        y_symbol,
        interval=y_interval,
        period=period,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if raw is None or len(raw) == 0:
        # Fallback: explicit 2y window if max failed
        raw = yf.download(
            y_symbol,
            interval=y_interval,
            period="2y" if y_interval in ("1h", "60m", "90m") else "max",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    if raw is None or len(raw) == 0:
        raise ConnectionError(f"Yahoo returned no data for {y_symbol} ({y_interval})")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    df = _normalize_ohlcv(raw)
    if len(df) > max_candles:
        df = df.iloc[-int(max_candles) :].reset_index(drop=True)
    return df


def fetch_binance_klines(
    symbol: str,
    interval: str = "15m",
    limit: int = 1000,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    max_candles: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles from Binance spot REST API (crypto fallback)."""
    rows: list[list] = []
    remaining = max_candles or limit
    cursor_end = end_ms

    while remaining > 0:
        batch = min(1000, remaining)
        params = {"symbol": symbol, "interval": interval, "limit": batch}
        if start_ms is not None:
            params["startTime"] = start_ms
        if cursor_end is not None:
            params["endTime"] = cursor_end

        for attempt in range(5):
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            if resp.status_code == 200:
                break
            time.sleep(0.5 * (attempt + 1))
        else:
            raise ConnectionError(f"Failed to fetch Binance klines for {symbol}: {resp.status_code} {resp.text}")

        data = resp.json()
        if not data:
            break
        rows = data + rows
        remaining -= len(data)
        cursor_end = int(data[0][0]) - 1
        if len(data) < batch:
            break
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    return _normalize_ohlcv(df)


def fetch_klines(
    symbol: str | None = None,
    interval: str = "1h",
    max_candles: int = 8760,
    *,
    source: str = "tradingview",
    exchange: str = "OANDA",
    binance_symbol: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Unified candle fetch.

    ``tradingview`` — OANDA/FX XAUUSD etc. (max ~5000 bars).
    ``yahoo`` — GC=F proxy for gold; supports ~1–2y of 1H (needed for 1y trains).
    ``binance`` — crypto spot.
    """
    source = (source or "tradingview").lower()
    sym = symbol or "XAUUSD"

    if source in ("tradingview", "tv"):
        if max_candles > 5000:
            print(
                f"TradingView hard-caps ~5000 bars; for {max_candles} × {interval} "
                f"falling back to Yahoo {_YAHOO_SYMBOL_MAP.get(sym.replace('/', '').upper(), sym)}."
            )
            return fetch_yahoo_klines(symbol=sym, interval=interval, max_candles=max_candles)
        return fetch_tradingview_klines(
            symbol=sym,
            exchange=exchange,
            interval=interval,
            max_candles=max_candles,
            username=kwargs.get("username"),
            password=kwargs.get("password"),
        )

    if source in ("yahoo", "yfinance"):
        return fetch_yahoo_klines(symbol=sym, interval=interval, max_candles=max_candles)

    if source in ("dukascopy", "duka"):
        from smartbs_entry.dukascopy import load_dukascopy_klines

        cached = load_dukascopy_klines(sym, interval=interval, max_candles=max_candles)
        if cached is None or cached.empty:
            raise FileNotFoundError(
                f"dukascopy cache missing for {sym} {interval}. "
                f"Run: python -m smartbs_entry.dukascopy"
            )
        print(f"Using permanent Dukascopy cache for {sym} {interval} ({len(cached)} bars)")
        return cached

    if source == "binance":
        return fetch_binance_klines(
            symbol=binance_symbol or symbol or "BTCUSDT",
            interval=interval,
            max_candles=max_candles,
        )
    if source in ("polygon", "polygon_cache", "polygon-debug", "debug_polygon"):
        raise ImportError(
            "polygon data_source is not bundled in smartbs_entry; "
            "use dukascopy / yahoo / binance / tradingview"
        )

    raise ValueError(f"Unknown data source: {source}")


def fetch_mtf_klines(
    symbol: str = "XAUUSD",
    *,
    source: str = "yahoo",
    exchange: str = "OANDA",
    max_1h_candles: int = 8760,
    binance_symbol: str | None = None,
) -> pd.DataFrame:
    """Fetch the 1H series used by the Entry engine / AI."""
    sym = symbol.replace("/", "").upper()
    print(f"Fetching 1H series ({max_1h_candles} bars) from {source}...")
    return fetch_klines(
        symbol=sym,
        interval="1h",
        max_candles=max_1h_candles,
        source=source,
        exchange=exchange,
        binance_symbol=binance_symbol,
    )
