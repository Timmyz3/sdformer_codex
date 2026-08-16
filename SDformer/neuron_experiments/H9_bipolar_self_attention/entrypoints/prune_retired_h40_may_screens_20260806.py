#!/usr/bin/env python3
"""Remove retired May H40 screen checkpoints while retaining all measurements."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = ROOT / (
    "neuron_autoresearch/cleanup_audits/"
    "retired_h40_may_screen_checkpoints_20260806.json"
)
H40_PREFIX = "h40_p4_ang05_screen160_bs4x2_"
PROBE_NAMES = {
    "probe_parallel_bs4_a_20260522_015732",
    "probe_parallel_bs4_b_20260522_015732",
}
EXPECTED_H40_RUNS = 40
EXPECTED_TOTAL_RUNS = EXPECTED_H40_RUNS + len(PROBE_NAMES)


def free_bytes() -> int:
    stat = os.statvfs(ROOT)
    return int(stat.f_bavail * stat.f_frsize)


def active_cmdlines() -> str:
    rows: list[str] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            rows.append(
                (process / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return "\n".join(rows)


def selected_run_dirs() -> list[Path]:
    h40 = sorted(
        path
        for path in RESULTS.iterdir()
        if path.is_dir()
        and path.name.startswith(H40_PREFIX)
        and "_20260522_" in path.name
    )
    if len(h40) != EXPECTED_H40_RUNS:
        raise RuntimeError(
            f"expected {EXPECTED_H40_RUNS} H40 runs, found {len(h40)}"
        )
    probes = [RESULTS / name for name in sorted(PROBE_NAMES)]
    missing = [path for path in probes if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing probe directories: {missing}")
    return h40 + probes


def describe(path: Path, reason: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": int(stat.st_size),
        "blocks_bytes": int(stat.st_blocks * 512),
        "inode": int(stat.st_ino),
        "links": int(stat.st_nlink),
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    active = active_cmdlines()
    candidates: list[dict[str, object]] = []
    retained_measurements: list[dict[str, object]] = []
    run_dirs = selected_run_dirs()
    if len(run_dirs) != EXPECTED_TOTAL_RUNS:
        raise RuntimeError("retired run count changed unexpectedly")

    for run_dir in run_dirs:
        checkpoints = sorted(run_dir.rglob("*.pth"))
        if len(checkpoints) != 1 or checkpoints[0].name != "checkpoint_epoch0.pth":
            raise RuntimeError(
                f"expected exactly one epoch0 checkpoint in {run_dir}, got {checkpoints}"
            )
        checkpoint = checkpoints[0]
        stat = checkpoint.stat()
        if stat.st_nlink != 1:
            raise RuntimeError(f"refuse to unlink shared checkpoint inode: {checkpoint}")
        if str(checkpoint.resolve()) in active:
            raise RuntimeError(f"refuse active checkpoint: {checkpoint}")
        candidates.append(
            describe(
                checkpoint,
                "retired 160-step H40/probe screen; measurements are retained and "
                "the run is unrelated to the NB0/TTX/H67/Local5 final lineage",
            )
        )
        retained = sorted(
            path
            for path in run_dir.rglob("*")
            if path.is_file() and path.suffix != ".pth"
        )
        if not retained:
            raise RuntimeError(f"no retained measurements found in {run_dir}")
        retained_measurements.append(
            {
                "run_dir": str(run_dir.relative_to(ROOT)),
                "file_count": len(retained),
                "files": [str(path.relative_to(ROOT)) for path in retained],
            }
        )

    before = free_bytes()
    if args.execute:
        for row in candidates:
            path = ROOT / str(row["path"])
            path.unlink()
            row["exists_after"] = path.exists()
            if row["exists_after"]:
                raise RuntimeError(f"checkpoint survived deletion: {path}")
    after = free_bytes()

    report = {
        "schema": "retired_h40_may_screen_checkpoint_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed": bool(args.execute),
        "scope": (
            "40 May-22 H40 160-step screens and two probe runs; delete only each "
            "run's unique checkpoint_epoch0.pth"
        ),
        "protected_scope": (
            "all NB0, TTX/BTTX, H67, Local5, full-resolution, resume, optimizer-state, "
            "valid825, rank-1, deployment-profile, and RTL-exact artifacts"
        ),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(row["size_bytes"]) for row in candidates),
        "free_bytes_before": before,
        "free_bytes_after": after,
        "observed_free_bytes_delta": after - before,
        "deleted": candidates if args.execute else [],
        "candidates": [] if args.execute else candidates,
        "retained_measurements": retained_measurements,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "executed",
                    "candidate_count",
                    "candidate_bytes",
                    "free_bytes_before",
                    "free_bytes_after",
                    "observed_free_bytes_delta",
                )
            },
            indent=2,
        )
    )
    print(f"audit={AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
