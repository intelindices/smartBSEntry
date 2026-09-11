"""Checkpoint load / promote for Entry AI models."""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from smartbs_engines.engines import normalize_feature_engine
from smartbs_engines.model import SmartBSClassifier
from smartbs_engines.registry import get_engine

CHECKPOINT_KEEP = 3
SPEC_HASH_KEY = "feature_spec_hash"
ENGINE_KEY = "feature_engine"
WARMUP_KEY = "warmup_bars"


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    path: str
    reason: str = ""
    backup: str = ""


def checkpoint_spec_fields(engine_name: str = "maribbon") -> dict[str, Any]:
    eng = get_engine(engine_name)
    return {
        ENGINE_KEY: eng.name,
        SPEC_HASH_KEY: eng.spec_hash(),
        WARMUP_KEY: int(eng.warmup_bars),
        "num_inputs": int(eng.num_inputs),
    }


def validate_checkpoint_config(
    ckpt_cfg: dict[str, Any],
    *,
    strict: bool = True,
) -> tuple[bool, str]:
    try:
        engine_name = normalize_feature_engine(ckpt_cfg.get(ENGINE_KEY) or "maribbon")
        eng = get_engine(engine_name)
    except ValueError as e:
        return False, str(e)

    n_in = ckpt_cfg.get("num_inputs")
    if n_in is not None and int(n_in) != eng.num_inputs:
        return False, f"num_inputs {n_in} != {engine_name} width {eng.num_inputs}"

    stored = ckpt_cfg.get(SPEC_HASH_KEY)
    if stored is None:
        msg = f"{engine_name} checkpoint predates feature-spec hashing"
        return (not strict), msg
    if str(stored) != eng.spec_hash():
        return False, (
            f"{engine_name} feature spec changed (checkpoint {stored}, "
            f"current {eng.spec_hash()}); retrain required"
        )
    return True, ""


def _assert_feature_spec_current(cfg_dict: dict, checkpoint_path: str) -> None:
    claimed = str(cfg_dict.get("feature_spec_hash") or "").strip()
    engine = normalize_feature_engine(cfg_dict.get("feature_engine") or "maribbon")
    eng = get_engine(engine)
    n_in = cfg_dict.get("num_inputs")
    if n_in is not None and int(n_in) != eng.num_inputs:
        raise RuntimeError(
            f"{checkpoint_path}: num_inputs {n_in} != {engine} width {eng.num_inputs}."
        )
    if not claimed:
        return
    current = str(eng.spec_hash())
    if current != claimed:
        raise RuntimeError(
            f"{checkpoint_path}: stale feature spec for engine '{engine}' "
            f"(checkpoint {claimed[:12]}, current {current[:12]})."
        )


def load_classifier(
    checkpoint_path: str, device: Optional[torch.device] = None
) -> tuple[SmartBSClassifier, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    cfg_dict = payload["config"]
    _assert_feature_spec_current(cfg_dict, checkpoint_path)
    model = SmartBSClassifier(
        num_inputs=cfg_dict["num_inputs"],
        num_channels=list(cfg_dict["num_channels"]),
        num_classes=cfg_dict.get("num_classes", 3),
        kernel_size=cfg_dict.get("kernel_size", 3),
        dropout=cfg_dict.get("dropout", 0.15),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, cfg_dict


def _rotate_backups(target: str, keep: int) -> str:
    if not os.path.isfile(target):
        return ""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"{target}.bak.{stamp}"
    shutil.copy2(target, backup)
    old = sorted(glob.glob(f"{target}.bak.*"))
    for path in old[: max(0, len(old) - keep)]:
        try:
            os.remove(path)
        except OSError:
            pass
    return backup


def promote_checkpoint(
    payload: dict[str, Any],
    target_path: str,
    *,
    keep: int = CHECKPOINT_KEEP,
    strict_spec: bool = True,
) -> PromotionResult:
    cfg = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(cfg, dict):
        return PromotionResult(False, target_path, "payload has no config dict")

    ok, why = validate_checkpoint_config(cfg, strict=strict_spec)
    if not ok:
        return PromotionResult(False, target_path, why)

    target_dir = os.path.dirname(os.path.abspath(target_path)) or "."
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=target_dir, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        reloaded = torch.load(tmp, map_location="cpu", weights_only=False)
        if not isinstance(reloaded, dict) or "config" not in reloaded:
            raise ValueError("temp checkpoint did not round-trip")
        backup = _rotate_backups(target_path, keep)
        os.replace(tmp, target_path)
        return PromotionResult(True, target_path, backup=backup)
    except Exception as e:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return PromotionResult(False, target_path, f"{type(e).__name__}: {e}")
