#!/usr/bin/env python3
"""Wait for and fail-closed audit the first Local-5 fullres resume anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "neuron_experiments/H9_bipolar_self_attention"
RUN = EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml"
MODEL = RUN / "checkpoint_epoch9.pth"
STATE = RUN / "checkpoint_epoch9_state_dict.pth"
REPORT = RUN / "checkpoint_epoch9_early_audit.json"
LOG = RUN / "checkpoint_epoch9_early_audit.log"
EXPECTED_LRS = [1e-4, 1e-4, 5e-5, 5e-5, 5e-6]


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_milestones(value: Any) -> dict[int, int]:
    if value is None:
        return {}
    return {int(key): int(count) for key, count in dict(value).items()}


def close_list(actual: list[float], expected: list[float]) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(got, want, rel_tol=0.0, abs_tol=1e-12)
        for got, want in zip(actual, expected)
    )


def validate_state(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    optimizer = payload.get("optimizer") or {}
    scheduler = payload.get("scheduler") or {}
    scaler = payload.get("scaler")
    optimizer_lrs = [float(group["lr"]) for group in optimizer.get("param_groups", [])]
    scheduler_lrs = [float(value) for value in scheduler.get("_last_lr", [])]
    facts = {
        "state_epoch": payload.get("epoch"),
        "scheduler_last_epoch": scheduler.get("last_epoch"),
        "scheduler_milestones": normalize_milestones(scheduler.get("milestones")),
        "optimizer_lrs": optimizer_lrs,
        "scheduler_last_lrs": scheduler_lrs,
        "scaler_present": isinstance(scaler, dict) and bool(scaler),
    }
    checks = {
        "state_epoch9": facts["state_epoch"] == 9,
        "scheduler_epoch9": facts["scheduler_last_epoch"] == 9,
        "milestones_13_20": facts["scheduler_milestones"] == {13: 1, 20: 1},
        "optimizer_five_group_lrs": close_list(optimizer_lrs, EXPECTED_LRS),
        "scheduler_five_group_lrs": close_list(scheduler_lrs, EXPECTED_LRS),
        "amp_scaler_present": bool(facts["scaler_present"]),
    }
    return facts, checks


def wait_stable(paths: tuple[Path, ...], poll_seconds: int, timeout_hours: float) -> None:
    deadline = time.monotonic() + timeout_hours * 3600.0
    previous: tuple[tuple[int, int], ...] | None = None
    while time.monotonic() < deadline:
        if all(path.is_file() for path in paths):
            current = tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
            if all(size > 100 * 1024 * 1024 for size, _ in current) and current == previous:
                return
            previous = current
            record("WAIT ep9 model/state files to become stable")
        else:
            record("WAIT ep9 model/state pair")
        time.sleep(poll_seconds)
    raise TimeoutError(f"ep9 model/state did not stabilize within {timeout_hours} hours")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    args = parser.parse_args()

    if REPORT.is_file():
        existing = json.loads(REPORT.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            record("REUSE existing PASS ep9 early audit")
            return 0

    wait_stable((MODEL, STATE), args.poll_seconds, args.timeout_hours)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload = torch.load(STATE, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"training state must be a mapping, got {type(payload).__name__}")
    facts, state_checks = validate_state(payload)

    runtime = config.get("runtime") or {}
    loader = config.get("loader") or {}
    optimizer_cfg = config.get("optimizer") or {}
    train_log = (RUN / "train.log").read_text(encoding="utf-8", errors="replace")
    load_audits = re.findall(
        r"\[H9\] load audit: checkpoint_overlay_keys=(\d+), missing=(\d+), unexpected=(\d+)",
        train_log,
    )
    latest_load = tuple(int(item) for item in load_audits[-1]) if load_audits else None
    checks = {
        **state_checks,
        "model_nonempty": MODEL.stat().st_size > 100 * 1024 * 1024,
        "state_nonempty": STATE.stat().st_size > 100 * 1024 * 1024,
        "config_fullres_480x640": loader.get("resolution") == [480, 640]
        and loader.get("crop") is None,
        "config_window_t2x15x15": config.get("swin_transformer", {}).get("window_size")
        == [2, 15, 15],
        "config_force_save_epochs": runtime.get("force_save_epochs") == [9, 14, 19, 24, 29],
        "config_state_save_epochs": runtime.get("state_save_epochs") == [9, 19, 29],
        "config_milestones": optimizer_cfg.get("milestones") == [13, 20],
        "load_overlay210_missing0_unexpected0": latest_load == (210, 0, 0),
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "schema": "local5_ep9_early_checkpoint_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "scope": "first_resume_anchor_model_state_and_training_load_not_accuracy",
        "model_path": str(MODEL.resolve()),
        "state_path": str(STATE.resolve()),
        "model_size_bytes": MODEL.stat().st_size,
        "state_size_bytes": STATE.stat().st_size,
        "model_sha256": sha256(MODEL),
        "state_sha256": sha256(STATE),
        "state_facts": facts,
        "latest_training_load_audit": latest_load,
        "checks": checks,
        "failed_checks": failed,
    }
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT)
    if failed:
        raise RuntimeError(f"Local-5 ep9 early checkpoint audit failed: {failed}")
    record(
        f"PASS ep9 model/state audit model_sha={report['model_sha256']} "
        f"state_sha={report['state_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
