#!/usr/bin/python3.12
"""Write a same-UID live-tool census without invoking ps or any EDA tool."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BLOCKED = {"vcs", "simv", "dc_shell", "pt_shell", "fm_shell", "icc2_shell",
           "icc2_exec", "dgcom_exec", "lm_shell", "lm_shell_exec", "Milkyway",
           "lmutil", "lmstat"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("M2205 census output not fresh")
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / "comm").read_text().strip()
            exe = Path(os.readlink(proc / "exe")).name
            argv = [item.decode(errors="replace") for item in
                    (proc / "cmdline").read_bytes().split(b"\0") if item]
            argv_names = {Path(item).name for item in argv}
        except (OSError, ValueError):
            continue
        if comm in BLOCKED or exe in BLOCKED or BLOCKED & argv_names:
            hits.append({"pid": int(proc.name), "comm": comm, "exe": exe,
                         "argv_names": sorted(argv_names)})
    payload = {"schema": "m2205_same_uid_tool_census_r1_v1", "phase": args.phase,
               "uid": os.getuid(), "blocked_names": sorted(BLOCKED),
               "matching_processes": hits, "matching_process_count": len(hits),
               "status": "PASS_EMPTY" if not hits else "FAIL_NONEMPTY"}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if hits:
        raise SystemExit("M2205 same-UID tool census is not empty")
    print(f"PASS_M2205_SAME_UID_CENSUS phase={args.phase} count=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
