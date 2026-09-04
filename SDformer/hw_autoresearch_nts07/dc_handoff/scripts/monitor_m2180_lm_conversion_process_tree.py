#!/usr/bin/python3.12
"""Monitor exactly one LM root, its lm_shell_exec, and Milkyway child."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


SELECTED_ENV = {"HOME", "TMPDIR", "XDG_CACHE_HOME", "M2180_ISOLATED_CWD"}
KNOWN_TOOL_NAMES = {
    "lm_shell", "lm_shell_exec", "icc2_lm_shell", "icc2_lm_shell_exec",
    "icc2_shell", "icc2_exec", "dgcom_exec", "dc_shell", "pt_shell",
    "fm_shell", "vcs", "simv", "Milkyway", "milkyway_exec",
}


def record(path: Path) -> dict[str, object] | None:
    try:
        fields = (path / "stat").read_text().split()
        cmdline = [item.decode(errors="replace") for item in
                   (path / "cmdline").read_bytes().split(b"\0") if item]
        env: dict[str, str] = {}
        for item in (path / "environ").read_bytes().split(b"\0"):
            if b"=" in item:
                key, value = item.split(b"=", 1)
                name = key.decode(errors="replace")
                if name in SELECTED_ENV:
                    env[name] = value.decode(errors="replace")
        try:
            exe = os.readlink(path / "exe").removesuffix(" (deleted)")
        except OSError:
            exe = ""
        return {"pid": int(path.name), "ppid": int(fields[3]),
                "starttime_ticks": int(fields[21]),
                "comm": (path / "comm").read_text().strip(),
                "exe_path": exe, "cmdline": cmdline,
                "selected_environment": env}
    except (OSError, ValueError, IndexError):
        return None


def obs_key(item: dict[str, object]) -> tuple[object, ...]:
    return (item["comm"], item["exe_path"], tuple(item["cmdline"]),
            tuple(sorted(item["selected_environment"].items())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--ready-file", type=Path, required=True)
    ap.add_argument("--wrapper-path", type=Path, required=True)
    ap.add_argument("--actual-exec-path", type=Path, required=True)
    ap.add_argument("--milkyway-path", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    wrapper = str(args.wrapper_path.resolve(strict=True))
    actual = str(args.actual_exec_path.resolve(strict=True))
    milkyway = str(args.milkyway_path.resolve(strict=True))
    identities: dict[tuple[int, int], dict[str, object]] = {}
    root_seen = False
    samples = 0
    stop_samples = 0
    while True:
        records = [item for path in Path("/proc").iterdir()
                   if path.name.isdigit() and (item := record(path)) is not None]
        by_pid = {int(item["pid"]): item for item in records}
        descendants = {args.root_pid}
        changed = True
        while changed:
            changed = False
            for item in records:
                if int(item["ppid"]) in descendants and int(item["pid"]) not in descendants:
                    descendants.add(int(item["pid"]))
                    changed = True
        for pid in sorted(descendants):
            if pid not in by_pid:
                continue
            item = by_pid[pid]
            key = (pid, int(item["starttime_ticks"]))
            parent = by_pid.get(int(item["ppid"]))
            link = {"ppid": int(item["ppid"]),
                    "parent_starttime_ticks": int(parent["starttime_ticks"]) if parent else None}
            identity = identities.setdefault(key, {
                "pid": pid, "starttime_ticks": key[1], "first_ppid": int(item["ppid"]),
                "parent_links": [], "exec_observations": []})
            if link not in identity["parent_links"]:
                identity["parent_links"].append(link)
            existing = {obs_key(obs) for obs in identity["exec_observations"]}
            if obs_key(item) not in existing:
                identity["exec_observations"].append(
                    {name: item[name] for name in ("comm", "exe_path", "cmdline", "selected_environment")})
            root_seen |= pid == args.root_pid
        samples += 1
        if root_seen and not args.ready_file.exists():
            args.ready_file.write_text("M2180_MONITOR_READY\n")
        if args.stop_file.exists():
            stop_samples += 1
            if stop_samples >= 5:
                break
        if samples > 2_000_000:
            raise SystemExit("M2180 monitor sample budget")
        time.sleep(0.005)

    ordered = sorted(identities.values(), key=lambda x: (x["starttime_ticks"], x["pid"]))
    flat = [{"pid": ident["pid"], "starttime_ticks": ident["starttime_ticks"], **obs}
            for ident in ordered for obs in ident["exec_observations"]]
    wrappers = [obs for obs in flat if wrapper in obs["cmdline"]]
    actuals = [obs for obs in flat if obs["exe_path"] == actual]
    milkyways = [obs for obs in flat if obs["exe_path"] == milkyway]
    unexpected_tools = []
    for obs in flat:
        names = {Path(str(obs["exe_path"])).name, str(obs["comm"]),
                 *(Path(arg).name for arg in obs["cmdline"])}
        if names & KNOWN_TOOL_NAMES and not (
                wrapper in obs["cmdline"] or obs["exe_path"] in {actual, milkyway}):
            unexpected_tools.append(obs)
    payload = {
        "schema": "m2180_lm_conversion_process_tree_r1_v1",
        "root_pid": args.root_pid, "root_seen": root_seen,
        "sample_count": samples, "unique_process_identity_count": len(ordered),
        "exec_observation_count": len(flat),
        "lm_wrapper_observations": wrappers,
        "lm_actual_exec_observations": actuals,
        "milkyway_child_observations": milkyways,
        "unexpected_tool_observations": unexpected_tools,
        "all_observed_processes": ordered,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not root_seen or not wrappers or not actuals or not milkyways or unexpected_tools:
        raise SystemExit("M2180 exact LM/Milkyway process census failed")
    if len({(x["pid"], x["starttime_ticks"]) for x in actuals}) != 1:
        raise SystemExit("M2180 lm_shell_exec identity count != 1")
    if len({(x["pid"], x["starttime_ticks"]) for x in milkyways}) != 1:
        raise SystemExit("M2180 Milkyway identity count != 1")
    print("PASS_M2180_LM_PROCESS_TREE_CENSUS")
    print(f"lm_actual_identities=1 milkyway_identities=1 observations={len(flat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
