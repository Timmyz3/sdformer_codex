"""Protect the active DSEC GPU queue from known CPU-only architecture probes."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
STATUS = (
    REPO
    / "neuron_experiments/H9_bipolar_self_attention/results/"
    "dsec_fullres_paper_w15_queue_status.log"
)
DONE_MARKER = "ALL COMPLETE DSEC PAPER-W15 QUEUE"
MATCHES = (
    "scripts/run_prosperity_official_probe.py",
    "scripts/phi_prosperity_dual_line_simulator.py",
    "tests.test_phi_prosperity_dual_line_simulator",
)


def record(message: str) -> None:
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] {message}",
        flush=True,
    )


def matching_processes() -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own_pid:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(pattern in cmdline for pattern in MATCHES):
            matches.append((pid, cmdline.strip()))
    return matches


def main() -> int:
    seen: set[int] = set()
    record("START DSEC CPU priority guard")
    while True:
        status = (
            STATUS.read_text(encoding="utf-8", errors="ignore")
            if STATUS.is_file()
            else ""
        )
        if DONE_MARKER in status:
            record("STOP main DSEC queue complete")
            return 0
        for pid, cmdline in matching_processes():
            try:
                os.setpriority(os.PRIO_PROCESS, pid, 19)
            except (PermissionError, ProcessLookupError):
                continue
            if pid not in seen:
                record(f"RENICE pid={pid} nice=19 cmd={cmdline[:180]}")
                seen.add(pid)
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
