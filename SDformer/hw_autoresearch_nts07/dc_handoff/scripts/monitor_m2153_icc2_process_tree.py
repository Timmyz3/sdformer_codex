#!/usr/bin/python3.12
"""Census PID/start-time identities and exec transitions for one M2155 run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


SELECTED_ENV = {"HOME", "TMPDIR", "XDG_CACHE_HOME", "M2153_ISOLATED_CWD"}
ICC2_EXEC_NAMES = {"icc2_exec", "icc2_exec-sle", "dgcom_exec"}
TOOL_CHILD_NAMES = {
    "icc2_lm_shell",
    "icc2_lm_shell_exec",
    "lm_shell",
    "lm_shell_exec",
    "milkyway_exec",
    "icc_shell_exec",
    "icc2_shell_exec",
    "common_shell_exec",
    "common_shell_exe",
}


def proc_record(path: Path) -> dict[str, object] | None:
    try:
        fields = (path / "stat").read_text().split()
        cmdline = [
            item.decode(errors="replace")
            for item in (path / "cmdline").read_bytes().split(b"\0")
            if item
        ]
        environment: dict[str, str] = {}
        for item in (path / "environ").read_bytes().split(b"\0"):
            if b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            decoded_key = key.decode(errors="replace")
            if decoded_key in SELECTED_ENV:
                environment[decoded_key] = value.decode(errors="replace")
        try:
            executable = os.readlink(path / "exe")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            executable = ""
        return {
            "pid": int(path.name),
            "ppid": int(fields[3]),
            "starttime_ticks": int(fields[21]),
            "comm": (path / "comm").read_text().strip(),
            "exe_path": executable.removesuffix(" (deleted)"),
            "cmdline": cmdline,
            "selected_environment": environment,
        }
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None


def observation_key(record: dict[str, object]) -> tuple[object, ...]:
    return (
        record["comm"],
        record["exe_path"],
        tuple(record["cmdline"]),
        tuple(sorted(record["selected_environment"].items())),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--wrapper-path", type=Path, required=True)
    parser.add_argument("--actual-exec-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wrapper_path = str(args.wrapper_path.resolve(strict=True))
    actual_exec_path = str(args.actual_exec_path.resolve(strict=True))

    identities: dict[tuple[int, int], dict[str, object]] = {}
    samples = 0
    root_seen = False
    actual_exec_seen = False
    stop_seen_samples = 0
    while True:
        records: list[dict[str, object]] = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                if path.stat().st_uid != os.getuid():
                    continue
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            record = proc_record(path)
            if record is not None:
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
        for pid in sorted(descendants):
            if pid not in by_pid:
                continue
            record = by_pid[pid]
            start = int(record["starttime_ticks"])
            identity_key = (pid, start)
            parent = by_pid.get(int(record["ppid"]))
            parent_link = {
                "ppid": int(record["ppid"]),
                "parent_starttime_ticks": (
                    int(parent["starttime_ticks"]) if parent is not None else None
                ),
            }
            identity = identities.setdefault(
                identity_key,
                {
                    "pid": pid,
                    "starttime_ticks": start,
                    "first_ppid": int(record["ppid"]),
                    "parent_links": [],
                    "exec_observations": [],
                },
            )
            if parent_link not in identity["parent_links"]:
                identity["parent_links"].append(parent_link)
            key = observation_key(record)
            existing = {observation_key(item) for item in identity["exec_observations"]}
            if key not in existing:
                identity["exec_observations"].append(
                    {key: record[key] for key in ("comm", "exe_path", "cmdline", "selected_environment")}
                )
            if pid == args.root_pid:
                root_seen = True
            if record["exe_path"] == actual_exec_path:
                actual_exec_seen = True
        samples += 1
        if root_seen and not args.ready_file.exists():
            args.ready_file.write_text("M2153_MONITOR_READY\n")
        if args.stop_file.exists():
            stop_seen_samples += 1
            if stop_seen_samples >= 5:
                break
        if samples > 2_000_000:
            raise SystemExit("M2153 process monitor exceeded sample budget")
        time.sleep(0.05 if actual_exec_seen else 0.005)

    ordered = sorted(identities.values(), key=lambda item: (int(item["starttime_ticks"]), int(item["pid"])))
    flat: list[dict[str, object]] = []
    for identity in ordered:
        for observation in identity["exec_observations"]:
            flat.append(
                {
                    "pid": identity["pid"],
                    "starttime_ticks": identity["starttime_ticks"],
                    **observation,
                }
            )
    wrappers = [item for item in flat if wrapper_path in item["cmdline"]]
    actuals = [
        item
        for item in flat
        if item["exe_path"] == actual_exec_path
        or Path(str(item["exe_path"])).name in ICC2_EXEC_NAMES
    ]
    children = [
        item
        for item in flat
        if Path(str(item["exe_path"])).name in TOOL_CHILD_NAMES
        or str(item["comm"]) in TOOL_CHILD_NAMES
        or any(Path(arg).name in TOOL_CHILD_NAMES for arg in item["cmdline"])
    ]
    payload = {
        "schema": "m2153_icc2_process_tree_r1_v1",
        "root_pid": args.root_pid,
        "root_seen": root_seen,
        "sample_count": samples,
        "unique_process_identity_count": len(ordered),
        "exec_observation_count": len(flat),
        "icc2_wrapper_observation_count": len(wrappers),
        "icc2_wrapper_observations": wrappers,
        "icc2_actual_exec_observation_count": len(actuals),
        "icc2_actual_exec_observations": actuals,
        "tool_spawned_conversion_exec_observation_count": len(children),
        "tool_spawned_conversion_exec_observations": children,
        "all_observed_processes": ordered,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not root_seen:
        raise SystemExit("M2153 process monitor never observed root process")
    if not actuals:
        raise SystemExit("M2153 process monitor never observed actual ICC2 executable")
    print("PASS_M2153_PROCESS_TREE_CENSUS")
    print(f"unique_process_identity_count={len(ordered)}")
    print(f"exec_observation_count={len(flat)}")
    print(f"icc2_actual_exec_observation_count={len(actuals)}")
    print(f"tool_spawned_conversion_exec_observation_count={len(children)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
