"""Download and persist Dukascopy 1H + 15m OHLCV for SmartBS commodities.

Writes under ``mining/smartbs_strategy/data_cache/dukascopy/``:

    {PAIR}_1h.parquet / .csv.gz / .meta.json
    {PAIR}_15m.parquet / .csv.gz / .meta.json

Uses Dukascopy's jetta candle API (hourly monthly files, minute daily files).
15m bars are resampled from 1m. Existing cache files are reused.

Re-run:

    python -m smartbs_engines.dukascopy
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from smartbs_engines.data import _normalize_ohlcv

PACKAGE_DIR = Path(__file__).resolve().parent
# Prefer env, then monorepo mining cache (vanta-network), then package-local.
_MONOREPO_CACHE = PACKAGE_DIR.parents[2] / "mining" / "smartbs_strategy" / "data_cache" / "dukascopy"
_ENV_CACHE = os.environ.get("SMARTBS_DUKASCOPY_DIR", "").strip()
DEFAULT_CACHE_DIR = (
    Path(_ENV_CACHE)
    if _ENV_CACHE
    else (
        _MONOREPO_CACHE
        if _MONOREPO_CACHE.is_dir()
        else PACKAGE_DIR / "data_cache" / "dukascopy"
    )
)
REPO_ROOT = PACKAGE_DIR.parents[2] if (PACKAGE_DIR.parents[2] / "mining").is_dir() else PACKAGE_DIR
JETTA_ROOT = "https://jetta.dukascopy.com/v1"

# Trade-pair keys → Dukascopy instrument codes.
COMMODITY_INSTRUMENTS: dict[str, str] = {
    "XAUUSD": "XAU-USD",
    "XAGUSD": "XAG-USD",
    "XTIUSD": "LIGHT.CMD-USD",
    "COPPERUSDC": "COPPER.CMD-USD",
    "NATGASUSDC": "GAS.CMD-USD",
    "PLATINUMUSDC": "XPT.CMD-USD",
}

# Five popular Vanta crypto spot pairs with Dukascopy jetta coverage (SOL/XRP/DOGE unavailable).
VANTA_CRYPTO_INSTRUMENTS: dict[str, str] = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "ADAUSD": "ADA-USD",
    "LTCUSD": "LTC-USD",
    "BCHUSD": "BCH-USD",
}

# Ten liquid Vanta forex majors (G1–G3).
VANTA_FOREX_INSTRUMENTS: dict[str, str] = {
    "EURUSD": "EUR-USD",
    "GBPUSD": "GBP-USD",
    "USDJPY": "USD-JPY",
    "USDCHF": "USD-CHF",
    "AUDUSD": "AUD-USD",
    "USDCAD": "USD-CAD",
    "NZDUSD": "NZD-USD",
    "EURJPY": "EUR-JPY",
    "GBPJPY": "GBP-JPY",
    "EURGBP": "EUR-GBP",
}

ALL_INSTRUMENTS: dict[str, str] = {
    **COMMODITY_INSTRUMENTS,
    **VANTA_CRYPTO_INSTRUMENTS,
    **VANTA_FOREX_INSTRUMENTS,
}

# Minute/hour history does not always reach 10y (Dukascopy catalog).
INSTRUMENT_START: dict[str, str] = {
    "XPT.CMD-USD": "2021-11-01",
    "ADA-USD": "2018-04-01",
    "BCH-USD": "2017-08-01",
}


def _normalize_interval_tag(interval: str) -> str:
    key = (interval or "").strip().lower()
    aliases = {
        "h1": "1h",
        "60": "1h",
        "60m": "1h",
        "m1": "1m",
        "m5": "5m",
        "m15": "15m",
        "h4": "4h",
        "d1": "1d",
        "day": "1d",
        "daily": "1d",
        "w1": "1w",
        "week": "1w",
        "weekly": "1w",
    }
    return aliases.get(key, key)


def cache_paths(symbol: str, interval: str, data_dir: Path | None = None) -> dict[str, Path]:
    root = Path(data_dir) if data_dir is not None else DEFAULT_CACHE_DIR
    sym = symbol.replace("/", "").upper()
    tf = _normalize_interval_tag(interval)
    return {
        "dir": root,
        "parquet": root / f"{sym}_{tf}.parquet",
        "csv": root / f"{sym}_{tf}.csv.gz",
        "meta": root / f"{sym}_{tf}.meta.json",
    }


def load_dukascopy_klines(
    symbol: str,
    interval: str = "1h",
    max_candles: int | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame | None:
    paths = cache_paths(symbol, interval, data_dir)
    src = paths["parquet"] if paths["parquet"].is_file() else paths["csv"]
    if not src.is_file():
        return None
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    out = _normalize_ohlcv(df)
    if max_candles is not None and max_candles > 0 and len(out) > max_candles:
        out = out.iloc[-int(max_candles) :].reset_index(drop=True)
    return out


def _already_saved(symbol: str, interval: str, data_dir: Path) -> dict[str, Any] | None:
    paths = cache_paths(symbol, interval, data_dir)
    if not paths["meta"].is_file() or not paths["parquet"].is_file():
        return None
    try:
        meta = json.loads(paths["meta"].read_text())
    except json.JSONDecodeError:
        return None
    if int(meta.get("n_bars") or 0) < 100:
        return None
    return meta


def _save(df: pd.DataFrame, *, symbol: str, interval: str, instrument: str, data_dir: Path) -> dict[str, Any]:
    paths = cache_paths(symbol, interval, data_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    cols = ["open_time", "open", "high", "low", "close", "volume"]
    out = df[cols].copy()
    out.to_parquet(paths["parquet"], index=False)
    out.to_csv(paths["csv"], index=False, compression="gzip")
    t0 = datetime.fromtimestamp(int(out["open_time"].iloc[0]) / 1000, tz=timezone.utc)
    t1 = datetime.fromtimestamp(int(out["open_time"].iloc[-1]) / 1000, tz=timezone.utc)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    meta = {
        "symbol": symbol.replace("/", "").upper(),
        "dukascopy_instrument": instrument,
        "interval": _normalize_interval_tag(interval),
        "source": "dukascopy",
        "price_type": "bid",
        "permanent": True,
        "n_bars": int(len(out)),
        "from_utc": t0.isoformat(),
        "to_utc": t1.isoformat(),
        "close_min": float(out["close"].min()),
        "close_max": float(out["close"].max()),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "parquet": _rel(paths["parquet"]),
        "csv_gz": _rel(paths["csv"]),
        "usage": "python -m smartbs_engines.train --data-source dukascopy",
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"  saved {meta['symbol']} {meta['interval']}: {meta['n_bars']} bars "
        f"{t0.date()} → {t1.date()} → {paths['parquet']}"
    )
    return meta


def _decode_jetta_candles(payload: dict[str, Any], *, include_flats: bool = False) -> list[tuple]:
    """Decode Dukascopy jetta delta-encoded candle JSON to (ts_ms, o, h, l, c, v)."""
    times = payload.get("times") or []
    if not times:
        return []
    mult = float(payload["multiplier"])
    shift = int(payload["shift"])
    ts = int(payload["timestamp"])
    open_u = round(float(payload["open"]) / mult)
    high_u = round(float(payload["high"]) / mult)
    low_u = round(float(payload["low"]) / mult)
    close_u = round(float(payload["close"]) / mult)
    prev_close = close_u
    opens = payload["opens"]
    highs = payload["highs"]
    lows = payload["lows"]
    closes = payload["closes"]
    volumes = payload.get("volumes") or [0.0] * len(times)
    rows: list[tuple] = []
    for i, delta in enumerate(times):
        delta = int(delta)
        gap_n = delta - (0 if i == 0 else 1)
        if include_flats:
            for gap in range(gap_n):
                flat_ts = ts + (gap if i == 0 else gap + 1) * shift
                px = prev_close * mult
                rows.append((flat_ts, px, px, px, px, 0.0))
        ts += delta * shift
        open_u += int(opens[i])
        high_u += int(highs[i])
        low_u += int(lows[i])
        close_u += int(closes[i])
        prev_close = close_u
        rows.append(
            (
                ts,
                open_u * mult,
                high_u * mult,
                low_u * mult,
                close_u * mult,
                float(volumes[i] or 0.0),
            )
        )
    return rows


def _cache_file(raw_cache: Path, rel: str) -> Path:
    return raw_cache / urllib.parse.quote(rel, safe="")


def _fetch_json(
    session: requests.Session,
    url: str,
    cache_path: Path,
    *,
    min_interval_s: float,
    last_request: list[float],
) -> dict[str, Any] | None:
    if cache_path.is_file() and cache_path.stat().st_size > 2:
        try:
            data = json.loads(cache_path.read_text())
            if not data or not data.get("times"):
                return None
            return data
        except json.JSONDecodeError:
            cache_path.unlink(missing_ok=True)
    wait = min_interval_s - (time.monotonic() - last_request[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(1, 9):
        last_request[0] = time.monotonic()
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException as e:
            sleep = min(120, 10 * attempt)
            print(f"  net {e}; sleep {sleep}s")
            time.sleep(sleep)
            continue
        if resp.status_code in (400, 404):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text('{"times":[]}')
            return None
        if resp.status_code == 429:
            sleep = min(180, 20 * attempt)
            print(f"  429 {url} try {attempt}; sleep {sleep}s")
            time.sleep(sleep)
            continue
        if resp.status_code >= 500:
            sleep = min(120, 8 * attempt)
            print(f"  {resp.status_code} {url}; sleep {sleep}s")
            time.sleep(sleep)
            continue
        resp.raise_for_status()
        if not resp.content or resp.content.strip() in (b"{}", b"[]", b"null"):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text('{"times":[]}')
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        return resp.json()
    raise RuntimeError(f"failed after retries: {url}")


def _month_starts(d0: date, d1: date) -> list[date]:
    cur = date(d0.year, d0.month, 1)
    end = date(d1.year, d1.month, 1)
    out: list[date] = []
    while cur <= end:
        out.append(cur)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def _days(d0: date, d1: date) -> list[date]:
    n = (d1 - d0).days + 1
    return [d0 + timedelta(days=i) for i in range(n)]


def _rows_to_frame(rows: list[tuple]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    return _normalize_ohlcv(df)


def _resample_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return _resample_ohlcv(df_1m, "15min")


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    g = df.set_index(ts).resample(rule, label="left", closed="left")
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
        }
    ).dropna(subset=["open", "high", "low", "close"])
    if "volume" in out.columns:
        out = out[out["volume"] > 0]
    out = out.reset_index().rename(columns={"index": "open_time"})
    return _normalize_ohlcv(out)


def build_fine_tf_from_minute_cache(
    symbol: str,
    *,
    data_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Assemble 1m (+5m) parquet from already-downloaded jetta minute JSON cache."""
    root = Path(data_dir) if data_dir is not None else DEFAULT_CACHE_DIR
    sym = symbol.replace("/", "").upper()
    if sym not in ALL_INSTRUMENTS:
        raise KeyError(f"unknown pair {sym}")
    instrument = ALL_INSTRUMENTS[sym]
    if not force:
        m1 = _already_saved(sym, "1m", root)
        m5 = _already_saved(sym, "5m", root)
        if m1 is not None and m5 is not None:
            print(f"=== skip {sym} 1m/5m (cache hit)")
            return {"1m": m1, "5m": m5}

    raw_cache = root / ".dukascopy-cache"
    # File names are URL-encoded: candles%2Fminute%2F{instrument}%2FBID%2F{y}%2F{m}%2F{d}.json
    pat = f"candles%2Fminute%2F{instrument}%2FBID%2F"
    files = sorted(raw_cache.glob(f"{pat}*.json")) if raw_cache.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"no minute cache under {raw_cache} for {instrument}; "
            f"run download_dukascopy --symbols {sym} --intervals 15m first"
        )

    rows: list[tuple] = []
    for i, path in enumerate(files, 1):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload and payload.get("times"):
            rows.extend(_decode_jetta_candles(payload))
        if i % 500 == 0 or i == len(files):
            print(f"  {sym} minute cache {i}/{len(files)} files, {len(rows)} bars")

    df_1m = _rows_to_frame(rows)
    if df_1m.empty:
        raise ValueError(f"empty 1m frame for {sym}")
    meta_1m = _save(df_1m, symbol=sym, interval="1m", instrument=instrument, data_dir=root)
    df_5m = _resample_ohlcv(df_1m, "5min")
    meta_5m = _save(df_5m, symbol=sym, interval="5m", instrument=instrument, data_dir=root)
    return {"1m": meta_1m, "5m": meta_5m}


def download_one(
    symbol: str,
    interval: str,
    *,
    date_from: str,
    date_to: str,
    data_dir: Path,
    force: bool = False,
    min_interval_s: float = 0.35,
) -> dict[str, Any]:
    if not force:
        existing = _already_saved(symbol, interval, data_dir)
        if existing is not None:
            print(
                f"=== skip {symbol} {interval} "
                f"({existing['n_bars']} bars {existing['from_utc'][:10]} → {existing['to_utc'][:10]})"
            )
            return existing

    instrument = ALL_INSTRUMENTS[symbol]
    start_s = max(date_from, INSTRUMENT_START.get(instrument, date_from))
    d0 = date.fromisoformat(start_s)
    d1 = date.fromisoformat(date_to)
    raw_cache = data_dir / ".dukascopy-cache"
    raw_cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "smartbs-ai-bot-dukascopy/1.0", "Accept": "application/json"})
    last_request = [0.0]
    rows: list[tuple] = []

    if interval.lower() in ("1h", "h1", "60"):
        buckets = _month_starts(d0, d1)
        kind = "hour"
        print(f"=== {symbol} 1h ({instrument}) {d0} → {d1}  months={len(buckets)}")
        for i, m in enumerate(buckets, 1):
            rel = f"candles/hour/{instrument}/BID/{m.year}/{m.month}"
            url = f"{JETTA_ROOT}/{rel}"
            payload = _fetch_json(
                session,
                url,
                _cache_file(raw_cache, rel + ".json"),
                min_interval_s=min_interval_s,
                last_request=last_request,
            )
            if payload and payload.get("times"):
                rows.extend(_decode_jetta_candles(payload))
            if i % 24 == 0 or i == len(buckets):
                print(f"  {symbol} 1h {i}/{len(buckets)} months, {len(rows)} bars")
        df = _rows_to_frame(rows)
    else:
        buckets = _days(d0, d1)
        print(f"=== {symbol} 15m ({instrument}) {d0} → {d1}  days={len(buckets)} (from 1m)")
        for i, d in enumerate(buckets, 1):
            rel = f"candles/minute/{instrument}/BID/{d.year}/{d.month}/{d.day}"
            url = f"{JETTA_ROOT}/{rel}"
            payload = _fetch_json(
                session,
                url,
                _cache_file(raw_cache, rel + ".json"),
                min_interval_s=min_interval_s,
                last_request=last_request,
            )
            if payload and payload.get("times"):
                rows.extend(_decode_jetta_candles(payload))
            if i % 200 == 0 or i == len(buckets):
                print(f"  {symbol} 1m {i}/{len(buckets)} days, {len(rows)} minute bars")
        df = _resample_15m(_rows_to_frame(rows))

    if df.empty:
        raise ValueError(f"empty OHLCV for {symbol} {interval}")
    # Clip to requested window (monthly files can start before d0).
    start_ms = int(datetime(d0.year, d0.month, d0.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(
        (datetime(d1.year, d1.month, d1.day, tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1000
    )
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] < end_ms)].reset_index(drop=True)
    return _save(df, symbol=symbol, interval=interval, instrument=instrument, data_dir=data_dir)


def download_all(
    *,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    date_from: str = "2016-08-21",
    date_to: str = "2026-08-21",
    data_dir: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    root = Path(data_dir) if data_dir is not None else DEFAULT_CACHE_DIR
    pairs = symbols or list(COMMODITY_INSTRUMENTS)
    tfs = intervals or ["1h", "15m"]
    metas: list[dict[str, Any]] = []
    for tf in tfs:
        for sym in pairs:
            if sym not in ALL_INSTRUMENTS:
                raise KeyError(f"unknown pair {sym}; known: {sorted(ALL_INSTRUMENTS)}")
            metas.append(
                download_one(
                    sym,
                    tf,
                    date_from=date_from,
                    date_to=date_to,
                    data_dir=root,
                    force=force,
                    min_interval_s=0.45 if tf == "15m" else 0.35,
                )
            )
    index = {
        "permanent": True,
        "source": "dukascopy",
        "date_from": date_from,
        "date_to": date_to,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "assets": metas,
    }
    (root / "index.meta.json").write_text(json.dumps(index, indent=2) + "\n")
    return metas


def download_vanta_crypto_forex(
    *,
    intervals: list[str] | None = None,
    date_from: str = "2016-08-21",
    date_to: str = "2026-08-21",
    data_dir: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Download 5 crypto + 10 forex Vanta pairs to permanent dukascopy cache."""
    symbols = list(VANTA_CRYPTO_INSTRUMENTS) + list(VANTA_FOREX_INSTRUMENTS)
    return download_all(
        symbols=symbols,
        intervals=intervals,
        date_from=date_from,
        date_to=date_to,
        data_dir=data_dir,
        force=force,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Download 10y Dukascopy 1H+15m for SmartBS commodities")
    p.add_argument("--symbols", default="", help="Comma-separated pairs (default: all 7 commodities)")
    p.add_argument("--intervals", default="1h,15m")
    p.add_argument("--from", dest="date_from", default="2016-08-21")
    p.add_argument("--to", dest="date_to", default="2026-08-21")
    p.add_argument("--dir", dest="data_dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--force", action="store_true", help="Re-download even if parquet already exists")
    p.add_argument(
        "--build-fine",
        action="store_true",
        help="Build 1m/5m parquet from existing minute JSON cache (no network)",
    )
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    root = Path(args.data_dir)
    if args.build_fine:
        for sym in symbols or list(COMMODITY_INSTRUMENTS):
            build_fine_tf_from_minute_cache(sym, data_dir=root, force=args.force)
        return
    metas = download_all(
        symbols=symbols,
        intervals=intervals,
        date_from=args.date_from,
        date_to=args.date_to,
        data_dir=root,
        force=args.force,
    )
    print("=== done ===")
    for m in metas:
        print(f"{m['symbol']:14} {m['interval']:4} {m['n_bars']:7}  {m['from_utc'][:10]} → {m['to_utc'][:10]}")


if __name__ == "__main__":
    main()
