#!/usr/bin/env python3
"""Prune superseded MDR/FAPS intermediate checkpoints with an audit trail."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = REPO / "neuron_autoresearch/cleanup_audits/legacy_mdr_faps_intermediates_20260805.json"

POLICIES = {
    "mdr_fast_local_ckpts_20260624": {
        "preserve": set(),
        "reason": "invalid resume: model weights were not restored; explicitly excluded from paper",
        "minimum_files": 17,
    },
    "mdr_valid_resume_local_ckpts_20260625_164239": {
        "preserve": {"checkpoint_epoch41.pth", "checkpoint_epoch47.pth"},
        "reason": "retain only the two MVSEC-evaluated baseline anchors",
        "minimum_files": 19,
    },
    "ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep0_20260630_201141/local_ckpts": {
        "preserve": {"checkpoint_epoch10.pth", "checkpoint_epoch10_state_dict.pth"},
        "reason": "retain the ep10 model/training-state handoff into the completed continuation",
        "minimum_files": 20,
    },
    "ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep10_20260702_161956/local_ckpts": {
        "preserve": {
            "checkpoint_epoch20.pth",
            "checkpoint_epoch20_state_dict.pth",
            "checkpoint_epoch40.pth",
            "checkpoint_epoch43.pth",
            "checkpoint_epoch43_state_dict.pth",
        },
        "reason": "retain paper-evaluated ep20/40/43 plus ep20 and final resume states",
        "minimum_files": 44,
    },
    "date11full_all_binary_atlif_faps_all12_nokmag_slowlr_ft_txep19_ft10_bs8_20260627_030855_setsid": {
        "preserve": {"checkpoint_epoch8.pth", "checkpoint_epoch9.pth"},
        "reason": "retain standard-valid825 rank-1 ep8 and final ep9 from the retired FAPS FT10 run",
        "minimum_files": 20,
    },
}


def active_commands() -> list[str]:
    commands = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if text:
            commands.append(text)
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    running = active_commands()
    rows = []
    preserved = []
    for relative_dir, policy in POLICIES.items():
        directory = RESULTS / relative_dir
        if not directory.is_dir():
            raise RuntimeError(f"missing cleanup directory: {directory}")
        files = sorted(directory.glob("checkpoint_epoch*.pth"))
        if len(files) < int(policy["minimum_files"]):
            raise RuntimeError(f"unexpectedly sparse cleanup directory: {directory}: {len(files)}")
        preserve = set(policy["preserve"])
        missing_preserve = preserve.difference(path.name for path in files)
        if missing_preserve:
            raise RuntimeError(f"missing preserved checkpoints: {directory}: {sorted(missing_preserve)}")
        if any(str(directory) in command for command in running):
            raise RuntimeError(f"cleanup target is referenced by a running process: {directory}")
        for path in files:
            stat = path.stat()
            record = {
                "path": str(path.relative_to(REPO)),
                "bytes": stat.st_size,
                "inode": stat.st_ino,
                "nlink": stat.st_nlink,
                "reason": policy["reason"],
            }
            if path.name in preserve:
                preserved.append(record)
            else:
                rows.append(record)

    free_before = os.statvfs(REPO).f_bavail * os.statvfs(REPO).f_frsize
    if args.execute:
        for row in rows:
            path = REPO / row["path"]
            path.unlink()
        for row in rows:
            if (REPO / row["path"]).exists():
                raise RuntimeError(f"checkpoint survived deletion: {row['path']}")
        for row in preserved:
            if not (REPO / row["path"]).is_file():
                raise RuntimeError(f"preserved checkpoint missing after deletion: {row['path']}")
    free_after = os.statvfs(REPO).f_bavail * os.statvfs(REPO).f_frsize
    audit = {
        "schema": "legacy_mdr_faps_checkpoint_cleanup_v1",
        "executed": args.execute,
        "deleted_files": len(rows) if args.execute else 0,
        "candidate_files": len(rows),
        "candidate_bytes": sum(int(row["bytes"]) for row in rows),
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "free_bytes_delta": free_after - free_before,
        "candidates": rows,
        "preserved": preserved,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "executed", "deleted_files", "candidate_files", "candidate_bytes",
        "free_bytes_before", "free_bytes_after", "free_bytes_delta"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
