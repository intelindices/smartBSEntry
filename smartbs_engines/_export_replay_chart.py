"""Export replay candles + MA + trade markers for the HTML chart viewer.

Usage:
  python -m smartbs_engines._export_replay_chart \\
    --ckpt-root checkpoints_rolle_breakout_roll_p5_xau_xag_entryv2 \\
    --symbols XAUUSD,XAGUSD --windows 90,180,360

  python -m smartbs_engines._export_replay_chart --serve
"""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from smartbs_engines._train_replay_all_st import (
    LOTS,
    PVS,
    _classes_from_probs,
    _positions_raw_arm,
)
from smartbs_engines.data import fetch_klines
from smartbs_engines.predict import predict_probs
from smartbs_engines.structure import ema as _ema
from smartbs_engines.structure import sma as _sma

# Match pine/SmartBSRMIndicator.pine MA stack.
MA_FAST = 14   # EMA
MA_MID = 48    # EMA
MA_SLOW = 120  # EMA
MA_LONG = 180  # SMA
ARM_SMA_LEN = 14  # unused alias; exit_arm chart/EA uses EMA14

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(__file__).resolve().parent / "replay_chart"
VIEWER_NAME = "index.html"


def _trades_from_pos(
    pos: np.ndarray,
    opens: np.ndarray,
    times: np.ndarray,
    ret: np.ndarray,
    *,
    hold_start: int,
    cut_ms: int,
) -> list[dict]:
    """Build open/close trade list from bar positions (held on bar open).

    Only trades whose entry bar is inside the chart window are emitted.
    """
    n = len(pos)
    trades: list[dict] = []
    side = 0
    entry_i = -1
    entry_px = 0.0
    entry_t = 0
    pnl_acc = 0.0

    def _close(i: int) -> None:
        nonlocal pnl_acc, entry_i
        if entry_i < 0 or side == 0:
            return
        # Skip trades that opened before the selected window.
        if int(entry_t) < cut_ms:
            entry_i = -1
            return
        exit_px = float(opens[i])
        trade_pnl = float(ret[entry_i:i].sum()) if i > entry_i else 0.0
        pnl_acc += trade_pnl
        trades.append(
            {
                "side": "long" if side > 0 else "short",
                "entryTime": int(entry_t) // 1000,
                "exitTime": int(times[i]) // 1000,
                "entryPrice": float(entry_px),
                "exitPrice": exit_px,
                "pnl": trade_pnl,
                "pnlCum": pnl_acc,
            }
        )
        entry_i = -1

    for i in range(hold_start, n):
        p = int(pos[i])
        if p == side:
            continue
        if side != 0 and entry_i >= 0:
            _close(i)
        side = p
        if p != 0:
            entry_i = i
            entry_px = float(opens[i])
            entry_t = int(times[i])
        else:
            entry_i = -1

    if side != 0 and entry_i >= 0:
        _close(n - 1)
    return trades


def _raw_ai_positions(
    cls: np.ndarray,
    *,
    start: int,
    n: int,
) -> np.ndarray:
    pos = np.zeros(n, dtype=np.int8)
    for i in range(start, n - 1):
        c = int(cls[i])
        if c == 0:
            continue
        pos[i + 1] = 1 if c == 1 else -1
    return pos


def _peek_ckpt_interval(checkpoint: Path) -> str:
    """Read interval from checkpoint config (default 1h)."""
    import torch

    ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    if isinstance(cfg, dict):
        from smartbs_engines.labels import normalize_ckpt_config

        normalize_ckpt_config(cfg)
    iv = str(cfg.get("interval") or "1h").strip().lower()
    if iv in ("60m", "60", "h1"):
        return "1h"
    if iv in ("5min", "m5"):
        return "5m"
    if iv in ("15min", "m15"):
        return "15m"
    if iv in ("240m", "h4"):
        return "4h"
    return iv or "1h"


def _warm_bars_for_interval(interval: str) -> int:
    """Warmup bars ahead of the holdout window (features need history)."""
    iv = (interval or "1h").strip().lower()
    if iv == "5m":
        return 12_000
    if iv == "15m":
        return 8_000
    if iv == "4h":
        return 1_500
    return 3_000


def export_checkpoint_windows(
    symbol: str,
    checkpoint: Path,
    *,
    windows_days: list[int],
    out_dir: Path,
    ai_threshold: float = 0.0,
) -> list[dict]:
    """Predict once, then write one JSON series per window."""
    interval = _peek_ckpt_interval(checkpoint)
    df = fetch_klines(symbol, interval, max_candles=0, source="mt5")
    end_ms = int(df["open_time"].iloc[-1])
    max_days = max(windows_days)
    cutoff_max = end_ms - int(max_days) * 86_400_000
    hold = df[df["open_time"] >= cutoff_max].reset_index(drop=True)
    warm = df[df["open_time"] < cutoff_max].tail(_warm_bars_for_interval(interval))
    full = pd.concat([warm, hold], ignore_index=True)
    hold_start = len(warm)

    probs, cfg = predict_probs(full, str(checkpoint), symbol=symbol, data_source="mt5")
    # Prefer config stamped at train time; fall back to peeked interval.
    interval = str(cfg.get("interval") or interval or "1h").strip().lower()
    lookback = int(cfg.get("lookback", 64))
    cls = _classes_from_probs(probs, ai_threshold, force_side=False)
    opens = full["open"].to_numpy(dtype=np.float64)
    highs = full["high"].to_numpy(dtype=np.float64)
    lows = full["low"].to_numpy(dtype=np.float64)
    closes = full["close"].to_numpy(dtype=np.float64)
    times = full["open_time"].to_numpy(dtype=np.int64)
    n = len(full)
    start = max(lookback - 1, hold_start)

    lot = float(LOTS.get(symbol, 0.1))
    pv = float(PVS.get(symbol, 1.0))
    ema_arm = _ema(closes, ARM_SMA_LEN)
    pos_raw = _raw_ai_positions(cls, start=start, n=n)
    pos_exit = _positions_raw_arm(
        cls,
        closes,
        ema_arm,
        start=start,
        n=n,
        entry_arm=False,
        exit_arm=True,
    )

    def _ret_for(pos: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        for i in range(hold_start, n - 1):
            if pos[i] == 0:
                continue
            out[i] = pos[i] * (opens[i + 1] - opens[i]) * lot * pv
        return out

    ret_raw = _ret_for(pos_raw)
    ret_exit = _ret_for(pos_exit)

    ma_full = {
        "fast": _ema(closes, MA_FAST),
        "mid": _ema(closes, MA_MID),
        "slow": _ema(closes, MA_SLOW),
        "long": _sma(closes, MA_LONG),
        "arm": ema_arm,  # EMA14 used by exit_arm
    }
    rows: list[dict] = []
    out_dir.joinpath("data").mkdir(parents=True, exist_ok=True)

    def _pack_trades(pos: np.ndarray, ret: np.ndarray, cut_ms: int) -> list[dict]:
        trades = _trades_from_pos(
            pos, opens, times, ret, hold_start=hold_start, cut_ms=cut_ms
        )
        pnl_acc = 0.0
        for t in trades:
            pnl_acc += float(t["pnl"])
            t["pnlCum"] = pnl_acc
        return trades

    def _pack_equity(ret: np.ndarray, cut_ms: int) -> list[dict]:
        mask = (np.arange(n) >= hold_start) & (np.arange(n) < n - 1) & (times >= cut_ms)
        eq = np.cumsum(np.where(mask, ret, 0.0))
        first_i = next((i for i in range(n) if int(times[i]) >= cut_ms), 0)
        base = float(eq[first_i]) if n else 0.0
        return [
            {"time": int(t) // 1000, "value": float(eq[i] - base)}
            for i, t in enumerate(times)
            if int(t) >= cut_ms
        ]

    _CLS_LABEL = {0: "FLAT", 1: "LONG", 2: "SHORT"}

    for days in windows_days:
        cutoff = end_ms - int(days) * 86_400_000
        candles = []
        signals = []
        for i, (t, o, h, lo, c) in enumerate(zip(times, opens, highs, lows, closes)):
            if int(t) < cutoff:
                continue
            t_sec = int(t) // 1000
            label = _CLS_LABEL.get(int(cls[i]), "FLAT")
            p = probs[i]
            p_flat = round(float(p[0]), 4)
            p_long = round(float(p[1]), 4)
            p_short = round(float(p[2]), 4)
            candles.append(
                {
                    "time": t_sec,
                    "open": float(o),
                    "high": float(h),
                    "low": float(lo),
                    "close": float(c),
                    "signal": label,
                    "pFlat": p_flat,
                    "pLong": p_long,
                    "pShort": p_short,
                }
            )
            signals.append(
                {
                    "time": t_sec,
                    "signal": label,
                    "pFlat": p_flat,
                    "pLong": p_long,
                    "pShort": p_short,
                }
            )
        ma_series = {
            name: [
                {"time": int(t) // 1000, "value": float(v)}
                for t, v in zip(times, arr)
                if int(t) >= cutoff and np.isfinite(v)
            ]
            for name, arr in ma_full.items()
        }
        trades = _pack_trades(pos_raw, ret_raw, cutoff)
        trades_exit = _pack_trades(pos_exit, ret_exit, cutoff)
        equity = _pack_equity(ret_raw, cutoff)
        equity_exit = _pack_equity(ret_exit, cutoff)

        wins = [t for t in trades_exit if t["pnl"] > 0]
        pnl = float(sum(t["pnl"] for t in trades_exit))
        from smartbs_engines.labels import normalize_label_mode, rolle_len_of

        eng = str(cfg.get("feature_engine", "?"))
        label_mode = normalize_label_mode(cfg.get("label_mode", "?"))
        meta = {
            "symbol": symbol,
            "engine": eng,
            "backbone": str(cfg.get("backbone", "?")),
            "label_mode": label_mode,
            "interval": interval,
            "rolle_len": rolle_len_of(cfg) if label_mode == "rolle_breakout" else cfg.get("rolle_len"),
            "days": int(days),
            "from": candles[0]["time"] if candles else None,
            "to": candles[-1]["time"] if candles else None,
            "checkpoint": str(checkpoint).replace("\\", "/"),
            "ai_threshold": float(ai_threshold),
            "lot": lot,
            "point_value": pv,
            "balance": 10000.0,
            "arm_sma": ARM_SMA_LEN,
            "trades": len(trades_exit),
            "pnl": pnl,
            "wr": (100.0 * len(wins) / len(trades_exit)) if trades_exit else 0.0,
            "wins": len(wins),
            "losses": len(trades_exit) - len(wins),
            "ma": {
                "fast": {"len": MA_FAST, "type": "EMA"},
                "mid": {"len": MA_MID, "type": "EMA"},
                "slow": {"len": MA_SLOW, "type": "EMA"},
                "long": {"len": MA_LONG, "type": "SMA"},
                "arm": {"len": ARM_SMA_LEN, "type": "SMA"},
            },
        }
        label_tag = str(label_mode or "label").strip().lower().replace("/", "_")
        plen = meta.get("rolle_len")
        if label_tag == "rolle_breakout" and plen is not None:
            try:
                label_tag = f"rolle_breakout_p{int(plen)}"
            except (TypeError, ValueError):
                pass
        from smartbs_engines.model import backbone_stem_tag

        bb_tag = backbone_stem_tag(meta.get("backbone"))
        stem = f"{symbol}_{eng}_{label_tag}_{bb_tag}_{interval}_{days}d"
        out_path = out_dir / "data" / f"{stem}.json"
        out_path.write_text(
            json.dumps(
                {
                    "meta": meta,
                    "candles": candles,
                    "signals": signals,  # AI class per bar: FLAT|LONG|SHORT
                    "ma": ma_series,
                    "trades": trades,
                    "equity": equity,
                    "tradesExitArm": trades_exit,
                    "equityExitArm": equity_exit,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "id": stem,
                "file": f"data/{stem}.json",
                "symbol": symbol,
                "engine": eng,
                "days": int(days),
                "label_mode": meta["label_mode"],
                "rolle_len": meta.get("rolle_len"),
                "interval": interval,
                "backbone": meta["backbone"],
                "pnl": pnl,
                "trades": len(trades_exit),
                "wr": meta["wr"],
            }
        )
    return rows


def _discover_ckpts(ckpt_root: Path, symbols: list[str]) -> list[tuple[str, str, Path]]:
    """Return (symbol, engine, path) for each available checkpoint."""
    found: list[tuple[str, str, Path]] = []
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"ckpt-root not found: {ckpt_root}")
    for eng_dir in sorted(ckpt_root.iterdir()):
        if not eng_dir.is_dir():
            continue
        eng = eng_dir.name
        for sym in symbols:
            pt = eng_dir / f"{sym}.pt"
            if pt.is_file():
                found.append((sym, eng, pt))
    return found


def _rolle_len_from_stem(stem: str):
    """Parse ``_p12_`` / ``_p15_`` style tag from export stem, else None."""
    import re

    m = re.search(r"_p(\d+)_(?:5m|15m|1h|4h)_\d+d$", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"_rolle_breakout_p(\d+)_", stem)
    if m:
        return int(m.group(1))
    return None


def _load_meta_fast(path: Path) -> dict:
    """Read only the leading ``meta`` object from a replay JSON (avoid full parse)."""
    with path.open("r", encoding="utf-8") as f:
        head = f.read(200_000)
    for marker in (',"candles"', ',"signals"', ',"ma"', ',"trades"'):
        i = head.find(marker)
        if i > 0:
            frag = head[:i] + "}"
            try:
                return dict(json.loads(frag).get("meta") or {})
            except json.JSONDecodeError:
                break
    # Fallback: full load (small files / unusual layout).
    return dict((json.loads(path.read_text(encoding="utf-8")).get("meta")) or {})


def _rebuild_catalog_from_data(out_dir: Path) -> list[dict]:
    """Scan data/*.json and build catalog items (keeps all label modes)."""
    items: list[dict] = []
    data_dir = out_dir / "data"
    if not data_dir.is_dir():
        return items
    for p in sorted(data_dir.glob("*.json")):
        try:
            m = _load_meta_fast(p)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        iv = str(m.get("interval") or "").strip().lower() or _interval_from_stem(p.stem)
        from smartbs_engines.labels import normalize_label_mode

        plen = m.get("rolle_len", m.get("pivot_len"))
        if plen is None:
            plen = _rolle_len_from_stem(p.stem)
        try:
            plen = int(plen) if plen is not None else None
        except (TypeError, ValueError):
            plen = None
        items.append(
            {
                "id": p.stem,
                "file": f"data/{p.name}",
                "symbol": m.get("symbol") or p.stem.split("_")[0],
                "engine": m.get("engine") or "?",
                "days": int(m.get("days") or 0),
                "label_mode": normalize_label_mode(m.get("label_mode") or "unknown"),
                "rolle_len": plen,
                "interval": iv,
                "backbone": m.get("backbone"),
                "pnl": m.get("pnl"),
                "trades": m.get("trades"),
                "wr": m.get("wr"),
            }
        )
    return items


def _interval_from_stem(stem: str) -> str:
    """Parse interval tag from stem (..._{5m|15m|1h|4h}_{days}d); default 1h."""
    parts = stem.rsplit("_", 2)
    if len(parts) >= 2 and parts[-2] in ("5m", "15m", "1h", "4h"):
        return parts[-2]
    return "1h"


def write_viewer(out_dir: Path) -> Path:
    src = Path(__file__).resolve().parent / "replay_chart" / VIEWER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / VIEWER_NAME
    if src.resolve() != dst.resolve() and src.is_file():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt-root",
        default=str(ROOT / "checkpoints_rolle_breakout_roll_p15_xau_xag_xti_entryv2"),
    )
    ap.add_argument("--symbols", default="XAUUSD,XAGUSD")
    ap.add_argument("--windows", default="90,180,360")
    ap.add_argument("--engines", default="", help="Comma filter (default: all under ckpt-root)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--ai-threshold", type=float, default=0.0)
    ap.add_argument("--serve", action="store_true", help="Serve viewer over HTTP and open browser")
    ap.add_argument(
        "--serve-only",
        action="store_true",
        help="Only serve existing catalog/data (skip export)",
    )
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ensure viewer HTML exists next to data/.
    viewer_src = Path(__file__).resolve().parent / "replay_chart" / VIEWER_NAME
    if not viewer_src.is_file():
        raise SystemExit(f"Missing viewer at {viewer_src}")
    if out_dir.resolve() != viewer_src.parent.resolve():
        (out_dir / VIEWER_NAME).write_text(viewer_src.read_text(encoding="utf-8"), encoding="utf-8")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    eng_filter = {e.strip().lower() for e in args.engines.split(",") if e.strip()}

    catalog: list[dict] = []
    ckpt_root = Path(args.ckpt_root) if str(args.ckpt_root).strip() else None
    do_serve = bool(args.serve or args.serve_only)
    do_export = (not args.serve_only) and ckpt_root is not None and ckpt_root.is_dir()
    if do_export:
        ckpts = _discover_ckpts(ckpt_root, symbols)
        if eng_filter:
            ckpts = [c for c in ckpts if c[1].lower() in eng_filter]
        if not ckpts:
            raise SystemExit(f"No checkpoints found under {ckpt_root}")
        print(f"Exporting {len(ckpts)} checkpoints × windows={windows} → {out_dir}")
        for sym, eng, pt in ckpts:
            print(f"  {sym} / {eng} ...", flush=True)
            try:
                rows = export_checkpoint_windows(
                    sym,
                    pt,
                    windows_days=windows,
                    out_dir=out_dir,
                    ai_threshold=float(args.ai_threshold),
                )
                catalog.extend(rows)
                for row in rows:
                    print(
                        f"    {row['days']:3d}d  pnl={row['pnl']:+.1f}  "
                        f"trades={row['trades']}  wr={row['wr']:.1f}%",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"    FAIL: {exc}", flush=True)

        # Always rebuild from all data/*.json so a partial export never wipes
        # other label modes / symbols already on disk.
        n_written = len(catalog)
        catalog = _rebuild_catalog_from_data(out_dir)
        cat_path = out_dir / "catalog.json"
        cat_path.write_text(
            json.dumps(
                {
                    "title": "SmartBS replay chart",
                    "ckpt_root": str(ckpt_root).replace("\\", "/"),
                    "items": catalog,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        modes = sorted({str(i.get("label_mode") or "?") for i in catalog})
        print(
            f"Wrote {cat_path} ({len(catalog)} series, "
            f"+{n_written} this run; label_modes={modes})"
        )
    elif not do_serve:
        raise SystemExit(f"ckpt-root not found: {args.ckpt_root}")

    if do_serve:
        os.chdir(out_dir)
        server = ThreadingHTTPServer(("127.0.0.1", int(args.port)), SimpleHTTPRequestHandler)
        url = f"http://127.0.0.1:{int(args.port)}/{VIEWER_NAME}"
        print(f"Serving {out_dir} at {url}")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
