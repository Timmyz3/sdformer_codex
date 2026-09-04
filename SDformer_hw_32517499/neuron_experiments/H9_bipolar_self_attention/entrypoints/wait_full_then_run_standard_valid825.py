"""Wait for a full-training run to produce target checkpoints, then run standard valid825 eval.

This is a small orchestration helper for long H9 full runs. It does not change
training behavior or existing experiment code paths; it only waits for a run to
finish the requested epochs and then invokes `run_h9_standard_valid825_eval.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]


def stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def append(log_path: Path, message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epoch", action="append", type=int, default=[19, 24, 29])
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    args = parser.parse_args()

    config = args.config.resolve()
    run_dir = args.run_dir.resolve()
    epochs = sorted(set(args.epoch))
    control_dir = run_dir / "standard_valid825_waiter"
    control_dir.mkdir(parents=True, exist_ok=True)
    control_log = control_dir / "wait.log"

    append(control_log, f"watching run_dir={run_dir}")
    append(control_log, f"target epochs={epochs}")

    deadline = time.time() + args.timeout_hours * 3600.0
    while True:
        missing = [epoch for epoch in epochs if not (run_dir / f"checkpoint_epoch{epoch}.pth").exists()]
        if not missing:
            append(control_log, "all target checkpoints found; launching standard valid825 eval")
            break
        if time.time() > deadline:
            append(control_log, f"timeout waiting for checkpoints; still missing={missing}")
            return 2
        append(control_log, f"still waiting; missing={missing}")
        time.sleep(max(30, args.poll_seconds))

    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/run_h9_standard_valid825_eval.py"),
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
    ]
    for epoch in epochs:
        command.extend(["--epoch", str(epoch)])
    rc = run(command, control_dir / "standard_valid825_eval.log")
    append(control_log, f"standard valid825 eval return code={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
