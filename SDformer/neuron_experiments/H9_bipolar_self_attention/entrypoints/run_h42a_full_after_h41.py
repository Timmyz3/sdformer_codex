"""Wait for the H41 continuation run, then launch H42a full training.

H42a is the mild-threshold SN S02 C variant from the H42 sweep.  The watcher is
deliberately conservative: it only starts H42a after the H41 launch log reports
that checkpoint profiling finished.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = EXP_ROOT / "configs" / "generated"
RESULTS_DIR = EXP_ROOT / "results"
BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_status(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {text.rstrip()}\n")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_h41(run_dir: Path, status: Path, poll_seconds: int) -> bool:
    pid_path = run_dir / "run.pid"
    train_log = run_dir / "train.log"
    if not pid_path.exists():
        append_status(status, f"H41 pid file missing: {pid_path}")
        return False
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    append_status(status, f"waiting for H41 pid={pid} run_dir={run_dir}")
    while pid_alive(pid):
        time.sleep(poll_seconds)
    text = train_log.read_text(encoding="utf-8", errors="ignore") if train_log.exists() else ""
    if "[profile_done]" not in text:
        append_status(status, "H41 process ended, but profile_done marker is missing; H42a full not launched.")
        return False
    append_status(status, "H41 profile_done detected; launching H42a full.")
    return True


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_h42a_full_config(out_dir: Path) -> Path:
    config = read_yaml(GENERATED_DIR / "h40_p3_SNS02_C.yml")
    run_stamp = out_dir.name.rsplit("_", 1)[-1]
    config["experiment"] = f"h42a_mild_theta_full30_{run_stamp}"

    loader = config.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True

    runtime = config.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = list(range(30))
    runtime["use_mlflow_model_logging"] = False

    optimizer = config.setdefault("optimizer", {})
    optimizer["use_amp"] = True
    optimizer.setdefault("param_groups", {})
    optimizer["param_groups"]["threshold_lr"] = 1.0e-6
    optimizer["param_groups"]["neuron_lr"] = 1.0e-5

    atlif = config.setdefault("atlif_ternary_psn", {})
    atlif["threshold_base_lr"] = 1.0e-6
    atlif["threshold_eta"] = 0.00025
    atlif["threshold_lr_scale"] = 20000.0
    atlif["target_rate_eta"] = 0.03
    atlif["max_threshold"] = 1.45

    loss = config.setdefault("loss", {})
    loss["use_angular_loss"] = False
    loss["lambda_ang"] = 0

    config["note"] = (
        "H42a full30: SN S02 C topology from H41, but with milder ATLIF threshold "
        "growth to reduce late-epoch firing collapse. Q/K remain ternary PSN+ATLIF; "
        "S0 FFN and stage2 half FFN remain binary PSN+ATLIF. No angular loss."
    )
    path = GENERATED_DIR / f"{config['experiment']}.yml"
    write_yaml(path, config)
    return path


def run_command(command: list[str], log_path: Path, status: Path, label: str) -> int:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        code = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"\n[h42a_full] {label} exit_code={code}\n")
    append_status(status, f"{label} exit={code}")
    return int(code)


def launch_h42a_full(status: Path) -> int:
    run_stamp = stamp()
    out_dir = RESULTS_DIR / f"h42a_mild_theta_full30_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = make_h42a_full_config(out_dir)
    append_status(status, f"H42a full config={config}")
    append_status(status, f"H42a full output={out_dir}")

    train_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(config),
        "--prev_runid",
        str(BASELINE_CKPT),
        "--save_path",
        str(out_dir / "checkpoint_epoch{}.pth"),
    ]
    code = run_command(train_cmd, out_dir / "train.log", status, "h42a_full_train")
    if code != 0:
        return code

    profile_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_checkpoints.py"),
        "--config",
        str(config),
        "--run-dir",
        str(out_dir),
        "--output-root",
        str(out_dir / "profiles"),
        "--samples",
        "40",
    ]
    for epoch in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29):
        profile_cmd.extend(["--epoch", str(epoch)])
    return run_command(profile_cmd, out_dir / "profile.log", status, "h42a_full_profile")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h41-run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    watcher_dir = RESULTS_DIR / f"h42a_after_h41_watcher_{stamp()}"
    status = watcher_dir / "status.log"
    watcher_dir.mkdir(parents=True, exist_ok=True)
    append_status(status, "watcher started")
    if not wait_for_h41(args.h41_run_dir.resolve(), status, args.poll_seconds):
        return 1
    return launch_h42a_full(status)


if __name__ == "__main__":
    raise SystemExit(main())
