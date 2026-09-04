#!/usr/bin/env python3
"""Capture the one active Local-5 root train process before its first checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RUN = EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml"
SOURCE_CHECKPOINT = (
    EXP
    / "results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
)
TRAIN_ENTRYPOINT = EXP / "entrypoints/train.py"
PIPELINE_ENTRYPOINT = (
    EXP / "entrypoints/run_dsec_fullres_w15_h66d_local5_bb1e4_full_pipeline.py"
)
STATUS_LOG = RUN / "status.log"
REPORT = RUN / "active_launch_provenance.json"
EXPECTED_SAVE = RUN / "checkpoint_epoch{}.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_argv(pid: int) -> list[str]:
    payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode(errors="replace") for part in payload.split(b"\0") if part]


def parse_proc_stat(raw: str) -> dict[str, int | str]:
    prefix, suffix = raw.rstrip().rsplit(") ", 1)
    pid_text, comm = prefix.split(" (", 1)
    fields = suffix.split()
    return {
        "pid": int(pid_text),
        "comm": comm,
        "state": fields[0],
        "ppid": int(fields[1]),
        "start_ticks": int(fields[19]),
    }


def boot_time_epoch() -> int:
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        if line.startswith("btime "):
            return int(line.split()[1])
    raise RuntimeError("/proc/stat has no btime")


def process_row(pid: int) -> dict[str, Any]:
    proc = Path(f"/proc/{pid}")
    facts = parse_proc_stat((proc / "stat").read_text(encoding="utf-8"))
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    start_epoch = boot_time_epoch() + int(facts["start_ticks"]) / clock_ticks
    return {
        **facts,
        "argv": read_argv(pid),
        "exe": str((proc / "exe").resolve()),
        "start_utc": datetime.fromtimestamp(start_epoch, timezone.utc).isoformat(),
        "start_epoch": start_epoch,
    }


def active_root_train_rows() -> list[dict[str, Any]]:
    expected_config = str(CONFIG.resolve())
    expected_save = str(EXPECTED_SAVE.resolve())
    expected_source = str(SOURCE_CHECKPOINT.resolve())
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            row = process_row(pid)
            argv = row["argv"]
            assert isinstance(argv, list)
            if (
                not any(value.endswith("/entrypoints/train.py") for value in argv)
                or expected_config not in argv
                or expected_save not in argv
                or expected_source not in argv
            ):
                continue
            parent = process_row(int(row["ppid"]))
        except (AssertionError, FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        parent_argv = parent["argv"]
        if not isinstance(parent_argv, list) or not any(
            value.endswith("run_dsec_fullres_w15_h66d_local5_bb1e4_full_pipeline.py")
            for value in parent_argv
        ):
            continue
        row["parent"] = parent
        rows.append(row)
    return rows


def option_value(argv: list[str], option: str) -> str | None:
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return None


def main() -> int:
    rows = active_root_train_rows()
    if len(rows) != 1:
        raise RuntimeError(f"expected one active Local-5 root train, found {len(rows)}")
    train = rows[0]
    argv = train["argv"]
    assert isinstance(argv, list)
    checks = {
        "one_active_root_train": len(rows) == 1,
        "config_arg": option_value(argv, "--config") == str(CONFIG.resolve()),
        "source_checkpoint_arg": option_value(argv, "--prev_runid")
        == str(SOURCE_CHECKPOINT.resolve()),
        "save_path_arg": option_value(argv, "--save_path") == str(EXPECTED_SAVE.resolve()),
        "finetune_arg": option_value(argv, "--finetune") == "1",
        "source_checkpoint_exists": SOURCE_CHECKPOINT.is_file(),
        "config_exists": CONFIG.is_file(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"active Local-5 launch checks failed: {failed}")

    status_lines = []
    if STATUS_LOG.is_file():
        status_lines = [
            line
            for line in STATUS_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
            if "START Local-5 bb1e4 fullres train30" in line
            or "END Local-5 bb1e4 fullres train30" in line
        ]
    report = {
        "schema": "local5_active_launch_provenance_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_ACTIVE_CAPTURE",
        "scope": "active_process_launch_identity_not_proof_of_bytes_read_before_later_config_rewrites",
        "active_train": train,
        "checks": checks,
        "artifact_identity": {
            "config_path": str(CONFIG.resolve()),
            "config_sha256_at_capture": sha256(CONFIG),
            "config_mtime_ns_at_capture": CONFIG.stat().st_mtime_ns,
            "source_checkpoint_path": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
            "train_entrypoint_path": str(TRAIN_ENTRYPOINT.resolve()),
            "train_entrypoint_sha256": sha256(TRAIN_ENTRYPOINT),
            "pipeline_entrypoint_path": str(PIPELINE_ENTRYPOINT.resolve()),
            "pipeline_entrypoint_sha256": sha256(PIPELINE_ENTRYPOINT),
        },
        "historical_pipeline_start_end_lines": status_lines,
        "disclosure": (
            "The active root process started before the final config mtime. This receipt proves "
            "the process/argv/source-checkpoint identity; ep9 optimizer/scheduler state remains "
            "the authority for the runtime training contract."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT)
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
