"""One-shot rename: rolling-extreme pivot_breakout → rolle_breakout / rolle_len.

Does NOT touch fractal/swing pivot APIs (SB_PivotHigh, ta.pivothigh, pine PivotChannel).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENG = ROOT / "smartbs_engines"
MQL5 = ROOT / "mql5"

# Source / docs to rewrite (not pine fractal indicators).
TEXT_GLOBS = [
    ENG / "*.py",
    ENG / "replay_chart" / "index.html",
    MQL5 / "Experts" / "SmartBSEntry.mq5",
    MQL5 / "README.md",
    MQL5 / "Models" / "README.md",
    MQL5 / "Presets" / "*.set",
]

# Ordered replacements (longer / more specific first).
REPLACEMENTS: list[tuple[str, str]] = [
    ("label_pivot_breakout", "label_rolle_breakout"),
    ("DEFAULT_PIVOT_BREAKOUT_LEN", "DEFAULT_ROLLE_LEN"),
    ("_pivot_breakout_params", "_rolle_breakout_params"),
    ("_pivot_breakout_resolved", "_rolle_breakout_resolved"),
    ("_pivot_len_from_stem", "_rolle_len_from_stem"),
    ("_smoke_pivot_breakout", "_smoke_rolle_breakout"),
    ("_ablate_pivot_len_5m_xau", "_ablate_rolle_len_5m_xau"),
    ("_finish_pivot_len_chart", "_finish_rolle_len_chart"),
    ("pivot_len_ablation", "rolle_len_ablation"),
    ("multi_pivot_len", "multi_rolle_len"),
    ("pivot_breakout_p", "rolle_breakout_p"),
    ("checkpoints_pivot_breakout", "checkpoints_rolle_breakout"),
    ('"pivot_breakout"', '"rolle_breakout"'),
    ("'pivot_breakout'", "'rolle_breakout'"),
    ("pivot_breakout", "rolle_breakout"),
    ("--pivot-len", "--rolle-len"),
    ("args.pivot_len", "args.rolle_len"),
    ("itemPivotLen", "itemRolleLen"),
    ("selectedSwingLen", "selectedSwingLen"),  # no-op keep
    ('id="pivotLen"', 'id="rolleLen"'),
    ("$(\"pivotLen\")", "$(\"rolleLen\")"),
    ("$('pivotLen')", "$('rolleLen')"),
    ("Pivot L", "RollE L"),
    ("pivot_len=", "rolle_len="),
    ("pivot_len:", "rolle_len:"),
    ('"pivot_len"', '"rolle_len"'),
    ("'pivot_len'", "'rolle_len'"),
    ("pivot_len", "rolle_len"),
    ("pivotLen", "rolleLen"),
    ("pivot-len", "rolle-len"),
]

FILE_RENAMES = [
    (ENG / "_smoke_pivot_breakout.py", ENG / "_smoke_rolle_breakout.py"),
    (ENG / "_ablate_pivot_len_5m_xau.py", ENG / "_ablate_rolle_len_5m_xau.py"),
    (ENG / "_finish_pivot_len_chart.py", ENG / "_finish_rolle_len_chart.py"),
]


def iter_text_files() -> list[Path]:
    out: list[Path] = []
    for g in TEXT_GLOBS:
        if g.is_file():
            out.append(g)
        else:
            out.extend(sorted(Path(g.parent).glob(g.name)))
    # Dedup
    seen = set()
    uniq = []
    for p in out:
        rp = p.resolve()
        if rp in seen or not p.is_file():
            continue
        # Skip this migrator and tmp scripts
        if p.name.startswith("_tmp_") or p.name == "_migrate_rolle_rename.py":
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def rewrite_text(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    new = raw
    for a, b in REPLACEMENTS:
        if a == b:
            continue
        new = new.replace(a, b)
    if new == raw:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def rename_paths() -> list[str]:
    log = []
    # Source file renames
    for src, dst in FILE_RENAMES:
        if src.is_file() and not dst.exists():
            src.rename(dst)
            log.append(f"file {src.name} → {dst.name}")

    # Checkpoint roots
    for p in sorted(ROOT.glob("checkpoints_pivot_breakout*")):
        dst = ROOT / p.name.replace("checkpoints_pivot_breakout", "checkpoints_rolle_breakout")
        if p.resolve() != dst.resolve():
            if dst.exists():
                log.append(f"SKIP dir exists {dst.name}")
            else:
                p.rename(dst)
                log.append(f"dir {p.name} → {dst.name}")

    for p in sorted((ENG).glob("checkpoints_*pivot_breakout*")):
        dst = ENG / p.name.replace("pivot_breakout", "rolle_breakout")
        if p.is_dir() and not dst.exists():
            p.rename(dst)
            log.append(f"dir {p.name} → {dst.name}")

    # Chart JSON stems
    data = ENG / "replay_chart" / "data"
    if data.is_dir():
        for p in sorted(data.glob("*pivot_breakout*")):
            dst = data / p.name.replace("pivot_breakout", "rolle_breakout")
            if dst.exists():
                # Prefer keeping newer rolle name; drop legacy duplicate
                if p.resolve() != dst.resolve():
                    p.unlink()
                    log.append(f"removed legacy {p.name}")
            else:
                p.rename(dst)
                log.append(f"data {p.name} → {dst.name}")

    # Parity ablation outputs
    po = ENG / "parity_out"
    if po.is_dir():
        for p in sorted(po.glob("*pivot_len*")):
            dst = po / p.name.replace("pivot_len", "rolle_len")
            if not dst.exists():
                p.rename(dst)
                log.append(f"parity {p.name} → {dst.name}")

    return log


def patch_catalog() -> None:
    cat = ENG / "replay_chart" / "catalog.json"
    if not cat.is_file():
        return
    obj = json.loads(cat.read_text(encoding="utf-8"))
    if "ckpt_root" in obj and isinstance(obj["ckpt_root"], str):
        obj["ckpt_root"] = obj["ckpt_root"].replace("pivot_breakout", "rolle_breakout")
    for it in obj.get("items") or []:
        for k in ("id", "file", "label_mode"):
            if k in it and isinstance(it[k], str):
                it[k] = it[k].replace("pivot_breakout", "rolle_breakout")
        if "pivot_len" in it:
            it["rolle_len"] = it.pop("pivot_len")
        # stem-derived
        if "label_mode" in it and it["label_mode"] == "pivot_breakout":
            it["label_mode"] = "rolle_breakout"
    cat.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def inject_compat_helpers() -> None:
    """Ensure labels.py exports normalize helpers after mechanical replace."""
    path = ENG / "labels.py"
    text = path.read_text(encoding="utf-8")
    if "def normalize_label_mode" in text:
        return
    if "DEFAULT_ROLLE_LEN" not in text:
        return
    helper = '''
# Legacy alias (pre-RollE rename). Fractal/swing pivots are unrelated.
LABEL_MODE_ALIASES = {
    "pivot_breakout": "rolle_breakout",
}


def normalize_label_mode(mode: str | None) -> str:
    m = str(mode or "").strip().lower()
    return LABEL_MODE_ALIASES.get(m, m)


def rolle_len_of(cfg) -> int:
    """Read ``rolle_len`` from config/dict; accept legacy ``pivot_len``."""
    if isinstance(cfg, dict):
        v = cfg.get("rolle_len", cfg.get("pivot_len", DEFAULT_ROLLE_LEN))
    else:
        v = getattr(cfg, "rolle_len", None)
        if v is None:
            v = getattr(cfg, "pivot_len", DEFAULT_ROLLE_LEN)
    try:
        return max(int(v or DEFAULT_ROLLE_LEN), 1)
    except (TypeError, ValueError):
        return DEFAULT_ROLLE_LEN


'''
    anchor = "DEFAULT_ROLLE_LEN = 15  # rolling window / forward scan length\n"
    if anchor in text and "def normalize_label_mode" not in text:
        path.write_text(text.replace(anchor, anchor + "\n" + helper, 1), encoding="utf-8")


def main() -> None:
    changed = []
    for p in iter_text_files():
        if rewrite_text(p):
            changed.append(str(p.relative_to(ROOT)))
    inject_compat_helpers()
    # Fix double-replace accidents
    for p in iter_text_files():
        t = p.read_text(encoding="utf-8")
        t2 = t.replace("rolle_rolle_", "rolle_").replace("RollE RollE ", "RollE ")
        # selectedSwingLen no-op already fine
        if t2 != t:
            p.write_text(t2, encoding="utf-8")
    logs = rename_paths()
    patch_catalog()
    print(f"Rewrote {len(changed)} files")
    for c in changed[:80]:
        print(" ", c)
    if len(changed) > 80:
        print(f"  ... +{len(changed)-80} more")
    print(f"Path renames: {len(logs)}")
    for L in logs[:60]:
        print(" ", L)


if __name__ == "__main__":
    main()
