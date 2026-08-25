#!/usr/bin/env python3
"""Write fail-closed launch/postflight receipts for the M29 valid40 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


FROZEN_BASE_SHA256 = (
    "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49"
)
FROZEN_SOURCE_CHECKPOINT_SHA256 = (
    "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
)
PREFLIGHT_STATUS = (
    "PASS_FLOATING_FACTOR_VALID40_INTERNAL_SCREEN_PREFLIGHT_NOT_INT8_"
    "NOT_ACCURACY_RESULT"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, str | int]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("expected JSON object: {}".format(path))
    return value


def write_exclusive(path: Path, value: dict) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_common(
    config: Path, source_checkpoint: Path, preflight_path: Path
) -> tuple[dict, dict[str, str | int], dict[str, str | int]]:
    config_identity = file_identity(config)
    checkpoint_identity = file_identity(source_checkpoint)
    preflight = read_json(preflight_path.resolve())
    if checkpoint_identity["sha256"] != FROZEN_SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("M29 source checkpoint does not match frozen H67 manifest")
    if (
        preflight.get("schema") != "m29_h67_rank3_launch_preflight_v1"
        or preflight.get("status") != PREFLIGHT_STATUS
        or preflight.get("config_sha256") != config_identity["sha256"]
        or preflight.get("checkpoint_sha256") != checkpoint_identity["sha256"]
        or int(preflight.get("trainable_tensors", -1)) != 180
        or int(preflight.get("expected_trainable_tensors", -1)) != 180
        or preflight.get("validation_scope")
        != "valid40_internal_screen_not_valid825_admission"
        or preflight.get("headline_admitted") is not False
    ):
        raise RuntimeError("M29 persisted preflight identity or scope drift")
    return preflight, config_identity, checkpoint_identity


def launch_receipt(args: argparse.Namespace) -> dict:
    preflight, config_identity, checkpoint_identity = verify_common(
        args.config, args.source_checkpoint, args.preflight
    )
    result_dir = args.result_dir.resolve()
    existing = sorted(result_dir.glob("checkpoint_epoch*.pth")) if result_dir.exists() else []
    if existing:
        raise RuntimeError("M29 result directory already contains checkpoints")
    return {
        "schema": "m29_h67_rank3_run_receipt_v1",
        "phase": "launch",
        "status": "READY_VALID40_INTERNAL_SCREEN_NOT_ACCURACY_NOT_SPEEDUP",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "watcher_pid": int(args.watcher_pid),
        "config": config_identity,
        "source_checkpoint": checkpoint_identity,
        "preflight": file_identity(args.preflight),
        "preflight_schema": preflight["schema"],
        "result_dir": str(result_dir),
        "train_log": str(args.train_log.resolve()),
        "frozen_base_sha256": FROZEN_BASE_SHA256,
        "frozen_source_checkpoint_sha256": FROZEN_SOURCE_CHECKPOINT_SHA256,
        "expected_checkpoint_epochs": list(range(36, 41)),
        "headline_admitted": False,
    }


def postflight_receipt(args: argparse.Namespace) -> dict:
    _, config_identity, checkpoint_identity = verify_common(
        args.config, args.source_checkpoint, args.preflight
    )
    launch_path = args.launch_receipt.resolve()
    launch = read_json(launch_path)
    if (
        launch.get("schema") != "m29_h67_rank3_run_receipt_v1"
        or launch.get("phase") != "launch"
        or launch.get("config", {}).get("sha256") != config_identity["sha256"]
        or launch.get("source_checkpoint", {}).get("sha256")
        != checkpoint_identity["sha256"]
        or launch.get("preflight", {}).get("sha256") != sha256(args.preflight)
        or launch.get("headline_admitted") is not False
    ):
        raise RuntimeError("M29 launch receipt identity drift")
    result_dir = args.result_dir.resolve()
    expected_epochs = list(range(36, 41))
    checkpoints = []
    states = []
    missing = []
    for epoch in expected_epochs:
        checkpoint = result_dir / "checkpoint_epoch{}.pth".format(epoch)
        state = result_dir / "checkpoint_epoch{}_state_dict.pth".format(epoch)
        if checkpoint.is_file():
            checkpoints.append(file_identity(checkpoint))
        else:
            missing.append(str(checkpoint))
        if state.is_file():
            states.append(file_identity(state))
        else:
            missing.append(str(state))
    train_log = file_identity(args.train_log) if args.train_log.is_file() else None
    exit_code = int(args.exit_code)
    complete = exit_code == 0 and not missing
    receipt = {
        "schema": "m29_h67_rank3_run_receipt_v1",
        "phase": "postflight",
        "status": (
            "PASS_TRAIN_EXIT_AND_EPOCH36_TO40_PRESENT_NOT_ACCURACY_RESULT"
            if complete
            else "FAIL_TRAIN_EXIT_OR_EXPECTED_CHECKPOINT_MISSING"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "watcher_pid": int(args.watcher_pid),
        "train_exit_code": exit_code,
        "complete": complete,
        "config": config_identity,
        "source_checkpoint": checkpoint_identity,
        "preflight": file_identity(args.preflight),
        "launch_receipt": file_identity(launch_path),
        "train_log": train_log,
        "result_dir": str(result_dir),
        "checkpoints": checkpoints,
        "training_states": states,
        "missing_expected_files": missing,
        "validation_scope": "valid40_internal_screen_not_valid825_admission",
        "headline_admitted": False,
    }
    if exit_code == 0 and missing:
        raise RuntimeError("M29 train returned zero but expected checkpoint/state files are missing")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("launch", "postflight"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watcher-pid", type=int, default=os.getppid())
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    if args.phase == "launch":
        value = launch_receipt(args)
    else:
        if args.launch_receipt is None or args.exit_code is None:
            parser.error("postflight requires --launch-receipt and --exit-code")
        value = postflight_receipt(args)
    write_exclusive(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
