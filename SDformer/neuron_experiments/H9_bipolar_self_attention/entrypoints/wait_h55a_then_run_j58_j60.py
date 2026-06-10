"""Wait for H55a, then profile it and run J58-J60 ATLIF-control short screens."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = EXP_ROOT / "results"


def stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def append(log: Path, message: str) -> None:
    text = f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {message}"
    print(text, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def process_alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("SDFORMER_USE_MLFLOW", "0")
    env.setdefault("SDFORMER_MLFLOW_MODEL_LOGGING", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[j58-j60-watcher] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h55a-pid", type=int, required=True)
    parser.add_argument("--h55a-run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=180)
    args = parser.parse_args()

    control_dir = RESULTS_DIR / f"j58_j60_after_h55a_{stamp()}"
    control_dir.mkdir(parents=True, exist_ok=True)
    control_log = control_dir / "watcher.log"
    append(control_log, f"watching H55a pid={args.h55a_pid}, run_dir={args.h55a_run_dir}")
    while process_alive(args.h55a_pid):
        append(control_log, "H55a still running; sleeping")
        time.sleep(max(30, args.poll_seconds))

    append(control_log, "H55a process ended; profiling selected checkpoints")
    profile_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_checkpoints.py"),
        "--config",
        str(args.h55a_run_dir / "config.yml"),
        "--run-dir",
        str(args.h55a_run_dir),
        "--output-root",
        str(control_dir / "h55a_profiles"),
        "--samples",
        "816",
        "--epoch",
        "4",
        "--epoch",
        "9",
        "--epoch",
        "14",
        "--epoch",
        "19",
        "--epoch",
        "24",
        "--epoch",
        "29",
    ]
    profile_rc = run_command(profile_cmd, control_dir / "h55a_selected_profile.log")
    append(control_log, f"H55a profile return code={profile_rc}")

    append(control_log, "running J58/J59/J60 360-step valid40 rapid screen")
    screen_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--config",
        "generated/j58a_importance_h54b_steps360.yml",
        "--config",
        "generated/j59a_quantile_h54b_steps360.yml",
        "--config",
        "generated/j60a_quantile_importance_h54b_steps360.yml",
        "--steps",
        "360",
        "--batch-size",
        "8",
        "--workers",
        "8",
        "--prefetch-factor",
        "4",
        "--pin-memory",
        "--amp",
        "--valid-samples",
        "40",
        "--no-promote-valid40",
        "--promote-aee",
        "2.05",
        "--promote-aae",
        "11.0",
        "--promote-sops-g",
        "3.6",
        "--tag",
        "j58_j60_atlif_controls",
    ]
    screen_rc = run_command(screen_cmd, control_dir / "j58_j60_rapid_screen.log")
    append(control_log, f"J58/J59/J60 rapid screen return code={screen_rc}")
    return 0 if screen_rc == 0 else screen_rc


if __name__ == "__main__":
    raise SystemExit(main())
