"""Wait for H55a, profile selected checkpoints, then launch H55b or H55c."""

from __future__ import annotations

import argparse
import os
import re
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
    return pid > 0 and Path(f"/proc/{pid}").exists()


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def parse_best_profile(report: Path) -> dict[str, float | str]:
    if not report.exists():
        return {}
    for line in report.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| 1 |"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        return {
            "checkpoint": cells[1],
            "AEE": float(cells[2]),
            "AAE": float(cells[3]),
            "SOPs_G": float(cells[4]),
            "firing": float(cells[5]),
        }
    return {}


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


def choose_next(best: dict[str, float | str]) -> tuple[str, Path, str]:
    if not best:
        return (
            "h55b",
            EXP_ROOT / "configs/generated/h55b_h54b_teacher_epe_dir_full30.yml",
            "H55a profile missing; run H55b direction-distill as the safer diagnostic.",
        )

    aae = float(best["AAE"])
    firing = float(best["firing"])
    checkpoint = str(best["checkpoint"])
    match = re.search(r"epoch(\d+)", checkpoint)
    best_epoch = int(match.group(1)) if match else -1

    if aae > 9.3:
        return (
            "h55b",
            EXP_ROOT / "configs/generated/h55b_h54b_teacher_epe_dir_full30.yml",
            f"Best AAE={aae:.4f} is still high; try direction-aware teacher distillation.",
        )
    if firing < 0.0725 or (0 <= best_epoch < 20):
        return (
            "h55c",
            EXP_ROOT / "configs/generated/h55c_h54b_teacher_epe_slowffn_full30.yml",
            f"Best firing={firing:.5f}, best epoch={best_epoch}; try slower FFN ATLIF sparsification.",
        )
    return (
        "h55b",
        EXP_ROOT / "configs/generated/h55b_h54b_teacher_epe_dir_full30.yml",
        "H55a looks usable; run H55b to test whether direction distillation further improves AAE.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h55a-pid", type=int, required=True)
    parser.add_argument("--h55a-run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--profile-samples", type=int, default=816)
    args = parser.parse_args()

    control_dir = RESULTS_DIR / f"h55_followup_{stamp()}"
    control_dir.mkdir(parents=True, exist_ok=True)
    control_log = control_dir / "followup.log"
    append(control_log, f"watching H55a pid={args.h55a_pid}, run_dir={args.h55a_run_dir}")

    while process_alive(args.h55a_pid):
        append(control_log, "H55a still running; sleeping")
        time.sleep(max(30, args.poll_seconds))

    append(control_log, "H55a process ended")
    profile_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_checkpoints.py"),
        "--config",
        str(args.h55a_run_dir / "config.yml"),
        "--run-dir",
        str(args.h55a_run_dir),
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
    append(control_log, "profiling selected H55a checkpoints")
    profile_rc = run_command(profile_cmd, control_dir / "h55a_selected_profile.log")
    append(control_log, f"H55a profile return code={profile_rc}")

    best = parse_best_profile(args.h55a_run_dir / f"profile_ranking_valid{args.profile_samples}.md")
    next_name, next_config, reason = choose_next(best)
    append(control_log, f"next={next_name}: {reason}")

    run_dir = RESULTS_DIR / f"{next_name}_{next_config.stem}_{stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    target_config = run_dir / "config.yml"
    target_config.write_text(next_config.read_text(encoding="utf-8"), encoding="utf-8")
    proc = launch_training(target_config, run_dir, run_dir / "train.log")
    append(control_log, f"launched {next_name} pid={proc.pid}, run_dir={run_dir}")
    (control_dir / f"launched_{next_name}.txt").write_text(f"pid={proc.pid}\nrun_dir={run_dir}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
