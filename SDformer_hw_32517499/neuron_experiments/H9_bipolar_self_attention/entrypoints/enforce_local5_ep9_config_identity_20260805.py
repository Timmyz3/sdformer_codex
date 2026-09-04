#!/usr/bin/env python3
"""Bind Local-5 to its actual ep9 state and repair only a stale LR milestone."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RUN = EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml"
GENERATOR = EXP / "entrypoints/make_dsec_fullres_w15_h66d_local5_bb1e4_config.py"
SOURCE_CONFIG = EXP / "configs/generated/dsec_fullres_paper_w15_h66d_local5_ep29_ft30.yml"
MODEL = RUN / "checkpoint_epoch9.pth"
STATE = RUN / "checkpoint_epoch9_state_dict.pth"
ARCHIVE = RUN / "checkpoint_epoch9_state_dict_pre_config_identity_repair.pth"
REPORT = RUN / "training_config_identity.json"
LOG = RUN / "training_config_identity.log"
LOCK = RUN / "training_config_identity.lock"
EARLY_AUDITOR = EXP / "entrypoints/audit_local5_ep9_checkpoint_20260805.py"
EARLY_AUDIT_REPORT = RUN / "checkpoint_epoch9_early_audit.json"
ACTIVE_LAUNCH_REPORT = RUN / "active_launch_provenance.json"
EXPECTED_LRS = (1e-4, 1e-4, 5e-5, 5e-5, 5e-6)
EXPECTED_MILESTONES = {13: 1, 20: 1}


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generated_config_bytes() -> bytes:
    spec = importlib.util.spec_from_file_location("local5_config_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import config generator: {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return yaml.safe_dump(module.build_config(), sort_keys=False).encode("utf-8")


def active_launch_binding() -> dict[str, Any]:
    launch = json.loads(ACTIVE_LAUNCH_REPORT.read_text(encoding="utf-8"))
    artifact = launch.get("artifact_identity") or {}
    train = launch.get("active_train") or {}
    checks = launch.get("checks") or {}
    validations = {
        "schema": launch.get("schema") == "local5_active_launch_provenance_v1",
        "status": launch.get("status") == "PASS_ACTIVE_CAPTURE",
        "capture_checks": bool(checks) and all(checks.values()),
        "config_sha_at_capture": artifact.get("config_sha256_at_capture")
        == sha256(CONFIG),
        "source_checkpoint_sha": artifact.get("source_checkpoint_sha256")
        == sha256(
            EXP
            / "results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
        ),
        "active_started_before_final_config_mtime": float(train.get("start_epoch", 0.0))
        < CONFIG.stat().st_mtime,
    }
    failed = [name for name, passed in validations.items() if not passed]
    if failed:
        raise RuntimeError(f"Local-5 active launch provenance failed: {failed}")
    return {
        "path": str(ACTIVE_LAUNCH_REPORT.resolve()),
        "sha256": sha256(ACTIVE_LAUNCH_REPORT),
        "train_pid_at_capture": int(train["pid"]),
        "train_start_utc": str(train["start_utc"]),
        "scope": str(launch.get("scope", "")),
    }


def state_facts(payload: dict[str, Any]) -> dict[str, Any]:
    optimizer = payload.get("optimizer") or {}
    scheduler = payload.get("scheduler") or {}
    milestones = {
        int(key): int(value) for key, value in dict(scheduler.get("milestones") or {}).items()
    }
    return {
        "state_epoch": int(payload.get("epoch", -1)),
        "scheduler_last_epoch": int(scheduler.get("last_epoch", -1)),
        "scheduler_milestones": milestones,
        "optimizer_lrs": [float(group["lr"]) for group in optimizer.get("param_groups", [])],
        "scheduler_last_lrs": [float(value) for value in scheduler.get("_last_lr", [])],
        "scaler_present": bool(payload.get("scaler")),
    }


def close_lrs(actual: list[float]) -> bool:
    return len(actual) == len(EXPECTED_LRS) and all(
        math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12)
        for got, expected in zip(actual, EXPECTED_LRS)
    )


def state_checks(facts: dict[str, Any]) -> dict[str, bool]:
    return {
        "state_epoch9": facts["state_epoch"] == 9,
        "scheduler_epoch9": facts["scheduler_last_epoch"] == 9,
        "milestones_13_20": facts["scheduler_milestones"] == EXPECTED_MILESTONES,
        "optimizer_five_group_lrs": close_lrs(facts["optimizer_lrs"]),
        "scheduler_five_group_lrs": close_lrs(facts["scheduler_last_lrs"]),
        "amp_scaler_present": facts["scaler_present"],
    }


def repairable_milestone_only(checks: dict[str, bool]) -> bool:
    return not checks["milestones_13_20"] and all(
        passed for name, passed in checks.items() if name != "milestones_13_20"
    )


def train_pids() -> list[int]:
    result = []
    expected_config = str(CONFIG.resolve())
    expected_save = str((RUN / "checkpoint_epoch{}.pth").resolve())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [item.decode(errors="replace") for item in (entry / "cmdline").read_bytes().split(b"\0")]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        joined = "\0".join(argv)
        if "entrypoints/train.py" not in joined or expected_config not in argv or expected_save not in argv:
            continue
        try:
            parent_pid = int((entry / "stat").read_text(encoding="utf-8").split()[3])
            parent_cmdline = Path(f"/proc/{parent_pid}/cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        if b"run_dsec_fullres_w15_h66d_local5_bb1e4_full_pipeline.py" in parent_cmdline:
            result.append(int(entry.name))
    return result


def stop_training() -> list[int]:
    pids = train_pids()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and any(Path(f"/proc/{pid}").exists() for pid in pids):
        time.sleep(1)
    survivors = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
    if survivors:
        raise RuntimeError(f"Local-5 train processes did not stop after SIGTERM: {survivors}")
    return pids


def repair_scheduler_state(payload: dict[str, Any]) -> None:
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict):
        raise TypeError("training state has no scheduler mapping")
    scheduler["milestones"] = Counter(EXPECTED_MILESTONES)


def write_report(payload: dict[str, Any]) -> None:
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT)


def base_identity() -> dict[str, Any]:
    current = CONFIG.read_bytes()
    regenerated = generated_config_bytes()
    return {
        "schema": "local5_training_config_identity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(CONFIG.resolve()),
        "config_sha256": sha256_bytes(current),
        "config_size_bytes": len(current),
        "deterministic_regeneration_sha256": sha256_bytes(regenerated),
        "deterministic_regeneration_equal": current == regenerated,
        "generator_path": str(GENERATOR.resolve()),
        "generator_sha256": sha256(GENERATOR),
        "generator_mtime_ns": GENERATOR.stat().st_mtime_ns,
        "source_config_path": str(SOURCE_CONFIG.resolve()),
        "source_config_sha256": sha256(SOURCE_CONFIG),
        "config_mtime_ns": CONFIG.stat().st_mtime_ns,
        "launch_timeline_disclosure": (
            "The active train process started before the generator/config final mtimes; "
            "the ep9 optimizer/scheduler state is therefore the authoritative runtime contract."
        ),
        "active_launch_provenance": active_launch_binding(),
    }


def wait_stable(poll_seconds: int, timeout_hours: float) -> None:
    deadline = time.monotonic() + timeout_hours * 3600
    previous = None
    while time.monotonic() < deadline:
        if MODEL.is_file() and STATE.is_file():
            current = tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in (MODEL, STATE))
            if min(size for size, _ in current) > 100 * 1024 * 1024 and current == previous:
                return
            previous = current
        record("WAIT ep9 model/state for runtime config identity")
        time.sleep(poll_seconds)
    raise TimeoutError("Local-5 ep9 pair did not stabilize")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)

    lock_handle = LOCK.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        record("SKIP config identity enforcer lock held")
        return 0

    identity = base_identity()
    identity.update({"status": "PENDING_EP9_RUNTIME_STATE", "checks": {}})
    write_report(identity)
    wait_stable(args.poll_seconds, args.timeout_hours)

    payload = torch.load(STATE, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Local-5 training state is not a mapping")
    before_sha = sha256(STATE)
    before = state_facts(payload)
    checks = state_checks(before)
    repaired = False
    stopped_pids: list[int] = []

    if not all(checks.values()):
        if not repairable_milestone_only(checks):
            stopped_pids = stop_training()
            identity.update(
                {
                    "status": "FAIL_UNREPAIRABLE_RUNTIME_STATE",
                    "state_facts_before": before,
                    "checks": checks,
                    "stopped_train_pids": stopped_pids,
                }
            )
            write_report(identity)
            raise RuntimeError(f"unrepairable Local-5 ep9 runtime state: {checks}")

        stopped_pids = stop_training()
        if not ARCHIVE.exists():
            os.link(STATE, ARCHIVE)
        repair_scheduler_state(payload)
        temporary = STATE.with_suffix(".pth.config_identity_repair.tmp")
        torch.save(payload, temporary)
        temporary.replace(STATE)
        repaired = True
        record(
            f"REPAIRED ep9 scheduler milestones {before['scheduler_milestones']} -> "
            f"{EXPECTED_MILESTONES}; preserved {ARCHIVE.name}"
        )

    final_payload = torch.load(STATE, map_location="cpu", weights_only=False)
    final_facts = state_facts(final_payload)
    final_checks = state_checks(final_facts)
    if not all(final_checks.values()):
        raise RuntimeError(f"Local-5 final ep9 state contract failed: {final_checks}")

    subprocess.run(
        [os.sys.executable, str(EARLY_AUDITOR), "--poll-seconds", "1", "--timeout-hours", "0.1"],
        cwd=REPO,
        check=True,
    )
    early_audit = json.loads(EARLY_AUDIT_REPORT.read_text(encoding="utf-8"))
    if (
        early_audit.get("status") != "PASS"
        or early_audit.get("state_sha256") != sha256(STATE)
        or early_audit.get("model_sha256") != sha256(MODEL)
    ):
        raise RuntimeError("ep9 early audit did not bind the final model/state pair")
    identity.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "authority": "ep9_optimizer_scheduler_state",
            "model_path": str(MODEL.resolve()),
            "model_sha256": sha256(MODEL),
            "state_path": str(STATE.resolve()),
            "state_sha256_before": before_sha,
            "state_sha256": sha256(STATE),
            "state_facts_before": before,
            "state_facts": final_facts,
            "checks": final_checks,
            "scheduler_repaired_at_ep9_boundary": repaired,
            "preserved_pre_repair_state": str(ARCHIVE.resolve()) if repaired else None,
            "preserved_pre_repair_state_sha256": sha256(ARCHIVE) if repaired else None,
            "stopped_train_pids": stopped_pids,
            "early_audit_path": str(EARLY_AUDIT_REPORT.resolve()),
            "early_audit_sha256": sha256(EARLY_AUDIT_REPORT),
        }
    )
    write_report(identity)
    record(f"PASS Local-5 runtime config identity repaired={repaired} state_sha={identity['state_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
