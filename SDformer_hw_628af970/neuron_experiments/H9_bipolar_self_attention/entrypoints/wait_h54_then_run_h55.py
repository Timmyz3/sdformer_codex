"""Wait for the active H54b full run, then profile and launch H55a."""

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
BASELINE_CHECKPOINT = "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def append(log: Path, message: str) -> None:
    text = f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {message}"
    print(text, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    return Path(f"/proc/{pid}").exists()


def run_command(command: list[str], log_path: Path, cwd: Path = REPO_ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def launch_training(config: Path, run_dir: Path, log_path: Path) -> subprocess.Popen:
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(config),
        "--prev_runid",
        BASELINE_CHECKPOINT,
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    env = os.environ.copy()
    env.setdefault("SDFORMER_USE_MLFLOW", "0")
    env.setdefault("SDFORMER_MLFLOW_MODEL_LOGGING", "0")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    return subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h54-pid", type=int, required=True)
    parser.add_argument("--h54-run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--profile-samples", type=int, default=816)
    args = parser.parse_args()

    control_dir = RESULTS_DIR / f"h55_autopilot_{stamp()}"
    control_dir.mkdir(parents=True, exist_ok=True)
    control_log = control_dir / "autopilot.log"
    append(control_log, f"watching H54 pid={args.h54_pid}, run_dir={args.h54_run_dir}")

    while process_alive(args.h54_pid):
        append(control_log, "H54 still running; sleeping")
        time.sleep(max(10, args.poll_seconds))

    append(control_log, "H54 process ended")
    h54_config = args.h54_run_dir / "config.yml"
    h54_epoch29 = args.h54_run_dir / "checkpoint_epoch29.pth"
    if h54_config.exists() and h54_epoch29.exists():
        profile_cmd = [
            sys.executable,
            "-u",
            str(EXP_ROOT / "entrypoints/profile_checkpoints.py"),
            "--config",
            str(h54_config),
            "--run-dir",
            str(args.h54_run_dir),
            "--output-root",
            str(control_dir / "profiles"),
            "--samples",
            str(args.profile_samples),
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
        append(control_log, "profiling selected H54 checkpoints before H55")
        profile_rc = run_command(profile_cmd, control_dir / "h54_pareto_profile.log")
        append(control_log, f"H54 profile return code={profile_rc}")
    else:
        append(control_log, "H54 epoch29/config missing; skipping profile and launching H55 anyway")

    config = EXP_ROOT / "configs/generated/h55a_h54b_teacher_epe_full30.yml"
    run_dir = RESULTS_DIR / f"h55a_h54b_teacher_epe_full30_{stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    target_config = run_dir / "config.yml"
    target_config.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    proc = launch_training(target_config, run_dir, run_dir / "train.log")
    append(control_log, f"launched H55a pid={proc.pid}, run_dir={run_dir}")
    (control_dir / "launched_h55a.txt").write_text(f"pid={proc.pid}\nrun_dir={run_dir}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
