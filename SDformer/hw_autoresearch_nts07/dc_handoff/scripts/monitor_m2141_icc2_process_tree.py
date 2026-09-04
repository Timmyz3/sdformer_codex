#!/usr/bin/python3.12
"""Observe every descendant of one M2141 top-level timeout/ICC2 process."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def proc_record(path: Path) -> dict[str, object] | None:
    try:
        fields = (path / "stat").read_text().split()
        cmdline = [
            x.decode(errors="replace")
            for x in (path / "cmdline").read_bytes().split(b"\0")
            if x
        ]
        return {
            "pid": int(path.name),
            "ppid": int(fields[3]),
            "starttime_ticks": int(fields[21]),
            "comm": (path / "comm").read_text().strip(),
            "cmdline": cmdline,
        }
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed: dict[tuple[int, int], dict[str, object]] = {}
    samples = 0
    root_seen = False
    stop_seen_samples = 0
    while True:
        records = []
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                try:
                    same_uid = path.stat().st_uid == os.getuid()
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                record = proc_record(path)
                if record is not None and same_uid:
                    records.append(record)
        by_pid = {int(record["pid"]): record for record in records}
        descendants = {args.root_pid}
        changed = True
        while changed:
            changed = False
            for record in records:
                pid = int(record["pid"])
                if int(record["ppid"]) in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        for pid in descendants:
            if pid in by_pid:
                record = by_pid[pid]
                observed[(pid, int(record["starttime_ticks"]))] = record
                if pid == args.root_pid:
                    root_seen = True
        samples += 1
        if args.stop_file.exists():
            stop_seen_samples += 1
            if stop_seen_samples >= 5:
                break
        if samples > 720000:
            raise SystemExit("M2141 process monitor exceeded sample budget")
        time.sleep(0.02)

    records = sorted(observed.values(), key=lambda x: (int(x["starttime_ticks"]), int(x["pid"])))
    tool_child_tokens = {
        "icc2_lm_shell",
        "lm_shell",
        "milkyway_exec",
        "icc_shell_exec",
        "icc2_shell_exec",
    }
    conversion_children = [
        record
        for record in records
        if int(record["pid"]) != args.root_pid
        and (
            str(record["comm"]) in tool_child_tokens
            or any(Path(arg).name in tool_child_tokens for arg in record["cmdline"])
        )
    ]
    payload = {
        "schema": "m2141_icc2_process_tree_r1_v1",
        "root_pid": args.root_pid,
        "root_seen": root_seen,
        "sample_count": samples,
        "unique_process_identity_count": len(records),
        "tool_spawned_conversion_child_count": len(conversion_children),
        "tool_spawned_conversion_children": conversion_children,
        "all_observed_processes": records,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not root_seen:
        raise SystemExit("M2141 process monitor never observed root process")
    print("PASS_M2141_PROCESS_TREE_CENSUS")
    print(f"unique_process_identity_count={len(records)}")
    print(f"tool_spawned_conversion_child_count={len(conversion_children)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
