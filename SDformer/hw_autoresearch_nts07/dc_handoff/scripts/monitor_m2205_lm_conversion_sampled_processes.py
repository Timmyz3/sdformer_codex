#!/usr/bin/python3.12
"""Gate LM conversion and audit sampled live processes without spawning helpers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


SELECTED_ENV = {"HOME", "TMPDIR", "XDG_CACHE_HOME", "M2205_ISOLATED_CWD"}
GATE_TOKEN = "M2205_MONITOR_RELEASE_ACTUAL_STABLE\n"


class MonitorFailure(RuntimeError):
    pass


def record(path: Path) -> dict[str, object] | None:
    try:
        fields = (path / "stat").read_text().split()
        cmdline = [item.decode(errors="replace") for item in
                   (path / "cmdline").read_bytes().split(b"\0") if item]
        environment: dict[str, str] = {}
        for item in (path / "environ").read_bytes().split(b"\0"):
            if b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            name = key.decode(errors="replace")
            if name in SELECTED_ENV:
                environment[name] = value.decode(errors="replace")
        try:
            exe = os.readlink(path / "exe").removesuffix(" (deleted)")
        except OSError:
            exe = ""
        return {"pid": int(path.name), "ppid": int(fields[3]),
                "starttime_ticks": int(fields[21]),
                "comm": (path / "comm").read_text().strip(),
                "exe_path": exe, "cmdline": cmdline,
                "selected_environment": environment}
    except (OSError, ValueError, IndexError):
        return None


def snapshot() -> list[dict[str, object]]:
    return [item for path in Path("/proc").iterdir()
            if path.name.isdigit() and (item := record(path)) is not None]


def descendant_pids(records: list[dict[str, object]], root_pid: int) -> set[int]:
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for item in records:
            pid, ppid = int(item["pid"]), int(item["ppid"])
            if ppid in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def obs_key(item: dict[str, object]) -> tuple[object, ...]:
    return (item["comm"], item["exe_path"], tuple(item["cmdline"]),
            tuple(sorted(item["selected_environment"].items())))


def add_observation(store: dict[tuple[int, int], dict[str, object]],
                    item: dict[str, object], phase: str,
                    parent_starttime_ticks: int | None) -> None:
    key = (int(item["pid"]), int(item["starttime_ticks"]))
    identity = store.setdefault(key, {
        "pid": key[0], "starttime_ticks": key[1], "first_ppid": int(item["ppid"]),
        "parent_links": [], "exec_observations": []})
    link = {"ppid": int(item["ppid"]),
            "parent_starttime_ticks": parent_starttime_ticks}
    if link not in identity["parent_links"]:
        identity["parent_links"].append(link)
    observation = {name: item[name] for name in
                   ("comm", "exe_path", "cmdline", "selected_environment")}
    observation["phase"] = phase
    seen = {(obs_key(obs), obs.get("phase")) for obs in identity["exec_observations"]}
    if (obs_key(observation), phase) not in seen:
        identity["exec_observations"].append(observation)


def terminate_sampled_tree(root_pid: int, root_starttime_ticks: int | None) -> None:
    records = snapshot()
    root = next((item for item in records if int(item["pid"]) == root_pid), None)
    if (root is None or root_starttime_ticks is None or
            int(root["starttime_ticks"]) != root_starttime_ticks):
        return
    pids = descendant_pids(records, root_pid)
    identities = {int(item["pid"]): int(item["starttime_ticks"]) for item in records
                  if int(item["pid"]) in pids}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        current = {int(item["pid"]): int(item["starttime_ticks"]) for item in snapshot()}
        for pid in sorted(identities, reverse=True):
            if current.get(pid) != identities[pid]:
                continue
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--gate-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--actual-exec-path", type=Path, required=True)
    parser.add_argument("--milkyway-path", type=Path, required=True)
    parser.add_argument("--tcl-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-samples", type=int, default=5)
    parser.add_argument("--pre-gate-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.stable_samples < 3:
        raise SystemExit("stable-samples must be at least three")
    actual_path = str(args.actual_exec_path.resolve(strict=True))
    milkyway_path = str(args.milkyway_path.resolve(strict=True))
    tcl_path = str(args.tcl_path.resolve(strict=True))
    if args.gate_file.exists() or any(args.frame_dir.iterdir()):
        raise SystemExit("M2205 gate/output not fresh")

    all_sampled: dict[tuple[int, int], dict[str, object]] = {}
    post_sampled: dict[tuple[int, int], dict[str, object]] = {}
    actual_keys: set[tuple[int, int]] = set()
    milkyway_keys: set[tuple[int, int]] = set()
    unexpected_post: list[dict[str, object]] = []
    pre_gate_milkyway: list[dict[str, object]] = []
    root_seen = gate_released = wait_marker_seen = frame_absent = False
    root_starttime_ticks: int | None = None
    stable_key: tuple[int, int] | None = None
    stable_count = max_stable_count = 0
    release_time_ns: int | None = None
    sample_count = post_gate_samples = 0
    stop_samples = 0
    violation = ""
    start = time.monotonic()

    try:
        while True:
            records = snapshot()
            by_pid = {int(item["pid"]): item for item in records}
            root_desc = descendant_pids(records, args.root_pid)
            current = [by_pid[pid] for pid in sorted(root_desc) if pid in by_pid]
            if args.root_pid in by_pid:
                observed_root_start = int(by_pid[args.root_pid]["starttime_ticks"])
                if root_starttime_ticks is None:
                    root_starttime_ticks = observed_root_start
                elif observed_root_start != root_starttime_ticks:
                    raise MonitorFailure("root PID reuse")
                root_seen = True
            phase = "post_gate" if gate_released else "bootstrap_pre_gate"
            for item in current:
                parent = by_pid.get(int(item["ppid"]))
                add_observation(all_sampled, item, phase,
                                int(parent["starttime_ticks"]) if parent else None)
            current_actual = [item for item in current if item["exe_path"] == actual_path]
            for item in current_actual:
                actual_keys.add((int(item["pid"]), int(item["starttime_ticks"])))
            if len(actual_keys) > 1 or len(current_actual) > 1:
                raise MonitorFailure("multiple sampled lm_shell_exec identities")

            if not gate_released:
                if args.stop_file.exists():
                    raise MonitorFailure("LM exited before conversion gate release")
                early_mw = [item for item in current if item["exe_path"] == milkyway_path]
                if early_mw:
                    pre_gate_milkyway.extend(early_mw)
                    raise MonitorFailure("Milkyway sampled before conversion gate release")
                if any(args.frame_dir.iterdir()):
                    raise MonitorFailure("frame output exists before conversion gate release")
                if len(current_actual) == 1:
                    item = current_actual[0]
                    key = (int(item["pid"]), int(item["starttime_ticks"]))
                    expected_args = {"-no_init", "-f", tcl_path}
                    if not expected_args.issubset(set(item["cmdline"])):
                        raise MonitorFailure("lm_shell_exec command identity mismatch")
                    if key == stable_key:
                        stable_count += 1
                    else:
                        stable_key, stable_count = key, 1
                    max_stable_count = max(max_stable_count, stable_count)
                    marker = (f"M2205_GATE0_TCL_WAITING actual_pid={key[0]} "
                              f"gate={args.gate_file}")
                    try:
                        text = args.log_file.read_text(errors="replace")
                    except OSError:
                        text = ""
                    wait_marker_seen = text.splitlines().count(marker) == 1
                    if stable_count >= args.stable_samples and wait_marker_seen:
                        frame_absent = not any(args.frame_dir.iterdir())
                        if not frame_absent:
                            raise MonitorFailure("frame appeared at gate release")
                        with args.gate_file.open("x") as stream:
                            stream.write(GATE_TOKEN)
                            stream.flush()
                            os.fsync(stream.fileno())
                        gate_released = True
                        release_time_ns = time.monotonic_ns()
                else:
                    stable_key, stable_count = None, 0
                if time.monotonic() - start > args.pre_gate_timeout_seconds:
                    raise MonitorFailure("stable lm_shell_exec/Tcl wait marker timeout")
            else:
                post_gate_samples += 1
                if stable_key is None:
                    raise MonitorFailure("missing stable actual identity")
                actual_pid, actual_start = stable_key
                actual_now = by_pid.get(actual_pid)
                if actual_now is not None and int(actual_now["starttime_ticks"]) != actual_start:
                    raise MonitorFailure("actual PID reuse")
                actual_desc = descendant_pids(records, actual_pid) if actual_now else set()
                post_current = [by_pid[pid] for pid in sorted(actual_desc) if pid in by_pid]
                for item in post_current:
                    parent = by_pid.get(int(item["ppid"]))
                    add_observation(post_sampled, item, "post_gate",
                                    int(parent["starttime_ticks"]) if parent else None)
                    key = (int(item["pid"]), int(item["starttime_ticks"]))
                    if key == stable_key:
                        if item["exe_path"] != actual_path:
                            unexpected_post.append(item)
                            raise MonitorFailure("actual identity executable drift")
                    elif item["exe_path"] == milkyway_path:
                        milkyway_keys.add(key)
                        if len(milkyway_keys) > 1:
                            raise MonitorFailure("multiple sampled Milkyway identities")
                    else:
                        unexpected_post.append(item)
                        raise MonitorFailure("unexpected sampled post-gate LM descendant")
                if args.stop_file.exists():
                    stop_samples += 1
                    if stop_samples >= 5:
                        break
                else:
                    stop_samples = 0
            sample_count += 1
            if sample_count > 2_000_000:
                raise MonitorFailure("sample budget exceeded")
            time.sleep(0.005)
        if not root_seen or not gate_released or stable_key is None:
            raise MonitorFailure("gate/root/actual evidence incomplete")
        if len(milkyway_keys) != 1:
            raise MonitorFailure("exactly one sampled Milkyway identity required")
    except Exception as exc:
        violation = f"{type(exc).__name__}: {exc}"
        terminate_sampled_tree(args.root_pid, root_starttime_ticks)

    ordered_all = sorted(all_sampled.values(), key=lambda row: (row["starttime_ticks"], row["pid"]))
    ordered_post = sorted(post_sampled.values(), key=lambda row: (row["starttime_ticks"], row["pid"]))
    payload = {
        "schema": "m2205_lm_conversion_sampled_process_contract_r1_v1",
        "status": ("PASS_M2205_SAMPLED_POST_GATE_PROCESS_CONTRACT" if not violation else
                   "FAIL_M2205_SAMPLED_POST_GATE_PROCESS_CONTRACT"),
        "claim_scope": {
            "sampled_live_processes_only": True,
            "exhaustive_short_lived_processes": False,
            "sampling_interval_seconds": 0.005,
            "bootstrap_helpers_permitted_before_gate": True,
            "post_gate_actual_subtree_allowlist": [actual_path, milkyway_path],
        },
        "root_pid": args.root_pid,
        "root_starttime_ticks": root_starttime_ticks,
        "root_seen": root_seen,
        "sample_count": sample_count,
        "gate": {
            "released": gate_released,
            "created_by_monitor": gate_released,
            "token": GATE_TOKEN.rstrip("\n") if gate_released else "",
            "tcl_wait_marker_seen": wait_marker_seen,
            "actual_stable_samples_required": args.stable_samples,
            "actual_stable_samples_observed": max_stable_count,
            "frame_absent_before_release": frame_absent,
            "release_monotonic_ns": release_time_ns,
        },
        "actual_identity": ({"pid": stable_key[0], "starttime_ticks": stable_key[1],
                             "exe_path": actual_path} if stable_key else None),
        "sampled_actual_identity_count": len(actual_keys),
        "sampled_milkyway_identity_count": len(milkyway_keys),
        "pre_gate_milkyway_observations": pre_gate_milkyway,
        "post_gate_sample_count": post_gate_samples,
        "unexpected_sampled_post_gate_descendants": unexpected_post,
        "all_sampled_processes": ordered_all,
        "post_gate_actual_subtree_processes": ordered_post,
        "violation": violation,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if violation:
        print(violation, file=__import__("sys").stderr)
        return 1
    print(payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
