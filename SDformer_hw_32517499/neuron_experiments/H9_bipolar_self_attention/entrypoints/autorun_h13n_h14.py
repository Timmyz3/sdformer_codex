"""Continue H13/H14 experiments without leaving the GPU idle.

This controller is intentionally conservative: it waits for an already running
H13n full run, profiles useful checkpoints, then screens H13 review guards and
strict-BSA H14 guards. Only guards that are not clearly worse are promoted to
full training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = EXP_ROOT / "configs"
RESULTS_DIR = EXP_ROOT / "results"
BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_command(command: list[str], log_path: Path, cwd: Path = REPO_ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[autorun] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def profile_checkpoint(config_name: str, checkpoint: Path, out_dir: Path) -> dict | None:
    if not checkpoint.exists():
        return None
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_sops.py"),
        "--config",
        str(CONFIG_DIR / config_name),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(out_dir),
        "--split",
        "valid",
        "--num-samples",
        "40",
        "--batch-size",
        "1",
        "--num-workers",
        "4",
        "--metric",
        "AEE",
        "--metric",
        "AAE",
    ]
    exit_code = run_command(command, out_dir / "profile.log")
    summary_path = out_dir / "sops_summary.json"
    if exit_code != 0 or not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_metrics(summary: dict) -> dict:
    metrics = summary.get("metrics", {})
    return {
        "AEE": float(metrics.get("AEE", 999.0)),
        "AAE": float(metrics.get("AAE", 999.0)),
        "SOPs_G": float(summary.get("estimated_total_sops", 0.0)) / 1.0e9,
        "firing": float(summary.get("global_firing_rate", 0.0)),
    }


def guard_score(metrics: dict) -> float:
    return metrics["AEE"] + 0.02 * metrics["AAE"] + 0.04 * max(0.0, metrics["SOPs_G"] - 3.1)


def train_config(config_name: str, stem: str) -> Path:
    stamp = now_stamp()
    run_dir = RESULTS_DIR / f"{stem}_bs8_{stamp}_setsid"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_pattern = run_dir / "checkpoint_epoch{}.pth"
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(CONFIG_DIR / config_name),
        "--prev_runid",
        str(BASELINE_CKPT),
        "--save_path",
        str(save_pattern),
    ]
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    exit_code = run_command(command, run_dir / "train.log")
    (run_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--h13-run", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as log:
        def record(message: str) -> None:
            line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        record(f"waiting for H13n pid {args.wait_pid}")
        while pid_alive(args.wait_pid):
            time.sleep(60)
        record("H13n process finished; profiling checkpoints")

        h13_config = "h13n_biascenter_shiftmax_target05_halfffn_down02_full.yml"
        h13_epochs = [0, 4, 7, 9, 14, 19, 24, 29]
        h13_results: list[tuple[int, dict]] = []
        for epoch in h13_epochs:
            summary = profile_checkpoint(
                h13_config,
                args.h13_run / f"checkpoint_epoch{epoch}.pth",
                RESULTS_DIR / f"profile_h13n_full_epoch{epoch}_valid40_{now_stamp()}",
            )
            if summary:
                metrics = compact_metrics(summary)
                h13_results.append((epoch, metrics))
                record(f"H13n epoch{epoch}: {metrics}")
        if h13_results:
            best_epoch, best_metrics = min(h13_results, key=lambda item: guard_score(item[1]))
            record(f"best H13n profile epoch{best_epoch}: {best_metrics}")
        else:
            record("no H13n checkpoints were profiled successfully")

        variants = [
            ("h14a_strict_bsa_thetav_sqrt_guard120.yml", "h14a_strict_bsa_thetav_sqrt_guard120"),
            ("h14b_strict_bsa_signv_sqrt_guard120.yml", "h14b_strict_bsa_signv_sqrt_guard120"),
            ("h14c_strict_bsa_thetav_mild_guard120.yml", "h14c_strict_bsa_thetav_mild_guard120"),
            ("h13t_popcount_l1_h13n_guard120.yml", "h13t_popcount_l1_h13n_guard120"),
            ("h13s_shiftnorm_h13n_guard120.yml", "h13s_shiftnorm_h13n_guard120"),
            ("h13u_negtarget_h13n_guard120.yml", "h13u_negtarget_h13n_guard120"),
        ]
        guard_results: list[tuple[str, Path, dict]] = []
        for config_name, stem in variants:
            record(f"starting {stem}")
            run_dir = train_config(config_name, stem)
            summary = profile_checkpoint(
                config_name,
                run_dir / "checkpoint_epoch0.pth",
                RESULTS_DIR / f"profile_{stem}_valid40_{now_stamp()}",
            )
            if summary:
                metrics = compact_metrics(summary)
                guard_results.append((stem, run_dir, metrics))
                record(f"{stem}: {metrics}")
            else:
                record(f"{stem}: profile failed")

        if not guard_results:
            record("no review/H14 guard succeeded; autorun stops")
            return 1

        best_stem, _, best_guard = min(guard_results, key=lambda item: guard_score(item[2]))
        record(f"best review/H14 guard: {best_stem} {best_guard}")
        passes = best_guard["AEE"] <= 1.56 and best_guard["AAE"] <= 8.0 and best_guard["SOPs_G"] <= 3.85
        if not passes:
            record("best guard did not pass promotion thresholds; autorun stops")
            return 0

        full_config = best_stem.replace("_guard120", "_full") + ".yml"
        full_stem = best_stem.replace("_guard120", "_full")
        record(f"promoting {full_stem}")
        full_run = train_config(full_config, full_stem)
        summary = profile_checkpoint(
            full_config,
            full_run / "checkpoint_epoch29.pth",
            RESULTS_DIR / f"profile_{full_stem}_epoch29_valid40_{now_stamp()}",
        )
        if summary:
            record(f"{full_stem} epoch29: {compact_metrics(summary)}")
        else:
            record(f"{full_stem} epoch29 profile failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
