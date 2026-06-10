"""Continue ATLIF-budget experiments after the J59 720-step confirmation."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = EXP_ROOT / "configs"
RESULTS_DIR = EXP_ROOT / "results"
DEFAULT_BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid: int, poll_seconds: int) -> None:
    while pid_alive(pid):
        time.sleep(poll_seconds)


def read_rows(summary_csv: Path) -> list[dict[str, Any]]:
    if not summary_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for key in ("AEE", "AAE", "SOPs_G", "firing", "score"):
                parsed[key] = float(parsed[key])
            parsed["samples"] = int(parsed["samples"])
            rows.append(parsed)
    return rows


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("samples") == 40]
    if not valid:
        return None
    return min(valid, key=lambda row: row["score"])


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[wait-j59] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def launch_full(config_name: str, tag: str, prev_runid: Path) -> int:
    run_stamp = stamp()
    config_path = CONFIG_DIR / "generated" / config_name
    run_dir = RESULTS_DIR / f"{config_path.stem}_{tag}_{run_stamp}"
    cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(config_path),
        "--prev_runid",
        str(prev_runid.resolve()),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    return run_command(cmd, run_dir / "train.log")


def run_budget_screen(prev_runid: Path) -> tuple[int, Path]:
    tag = f"j61_j62_budget_followup_{stamp()}"
    out_dir_hint = RESULTS_DIR / tag
    cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--config",
        "generated/j61a_quantile_budget_q98_fullguard_steps360.yml",
        "--config",
        "generated/j62a_quantile_budget_weak_importance_steps360.yml",
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
        "2.00",
        "--promote-aae",
        "10.5",
        "--promote-sops-g",
        "3.6",
        "--tag",
        tag,
        "--prev-runid",
        str(prev_runid.resolve()),
    ]
    exit_code = run_command(cmd, RESULTS_DIR / f"{tag}_stdout.log")
    candidates = sorted(RESULTS_DIR.glob(f"{tag}_*"))
    return exit_code, candidates[-1] if candidates else out_dir_hint


def config_for_row(row_name: str) -> str | None:
    if row_name.startswith("j59a_quantile_h54b"):
        return "j59a_quantile_h54b_full30.yml"
    if row_name.startswith("j61a_quantile_budget_q98_fullguard"):
        return "j61a_quantile_budget_q98_fullguard_full30.yml"
    if row_name.startswith("j62a_quantile_budget_weak_importance"):
        return "j62a_quantile_budget_weak_importance_full30.yml"
    return None


def good_enough_for_full(row: dict[str, Any]) -> bool:
    return row["AEE"] <= 1.80 and row["AAE"] <= 8.80 and row["SOPs_G"] <= 3.45


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--j59-summary", type=Path, required=True)
    parser.add_argument("--prev-runid", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args(argv)

    print(f"waiting_for_pid={args.wait_pid}", flush=True)
    wait_for_pid(args.wait_pid, args.poll_seconds)
    time.sleep(5)

    j59 = best_row(read_rows(args.j59_summary))
    print(f"j59_row={j59}", flush=True)
    if j59 is not None and good_enough_for_full(j59):
        config = config_for_row(str(j59["name"]))
        if config is None:
            return 2
        print(f"launch_full={config}", flush=True)
        return launch_full(config, "auto_after_j59_720", args.prev_runid)

    print("j59_not_good_enough_running_j61_j62", flush=True)
    screen_exit, screen_dir = run_budget_screen(args.prev_runid)
    if screen_exit != 0:
        return screen_exit

    row = best_row(read_rows(screen_dir / "summary.csv"))
    print(f"budget_row={row}", flush=True)
    if row is None:
        return 3
    config = config_for_row(str(row["name"]))
    if config is None:
        return 4
    print(f"launch_full={config}", flush=True)
    return launch_full(config, "auto_after_budget_screen", args.prev_runid)


if __name__ == "__main__":
    raise SystemExit(main())
