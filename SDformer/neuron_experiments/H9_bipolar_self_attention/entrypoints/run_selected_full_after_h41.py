"""Wait for H41 continuation, then launch one selected full experiment.

This is used to keep the GPU busy after the current H41 continuation finishes.
It intentionally waits for the H41 profile marker so the current run records its
metrics before the next full run starts.
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


def wait_for_run(run_dir: Path, status: Path, poll_seconds: int) -> bool:
    pid_path = run_dir / "run.pid"
    train_log = run_dir / "train.log"
    if not pid_path.exists():
        append_status(status, f"pid file missing: {pid_path}")
        return False
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    append_status(status, f"waiting for H41 pid={pid} run_dir={run_dir}")
    while pid_alive(pid):
        time.sleep(poll_seconds)
    text = train_log.read_text(encoding="utf-8", errors="ignore") if train_log.exists() else ""
    if "[profile_done]" not in text:
        append_status(status, "H41 ended, but profile_done marker is missing; selected full not launched.")
        return False
    append_status(status, "H41 profile_done detected; selected full can launch.")
    return True


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_full_config(source_config: Path, run_name: str, output_dir: Path) -> Path:
    config = read_yaml(source_config)
    config["experiment"] = run_name

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

    config.setdefault("optimizer", {})["use_amp"] = True
    config["note"] = (
        f"{run_name}: full30 promoted after H41 continuation. Source config={source_config}. "
        "The run starts from the preserved PSN baseline checkpoint and saves every epoch "
        "for checkpoint-curve profiling."
    )
    out = GENERATED_DIR / f"{run_name}.yml"
    write_yaml(out, config)
    (output_dir / "README.md").write_text(
        "# Selected Full Run After H41\n\n"
        f"- source config: `{source_config}`\n"
        f"- generated full config: `{out}`\n"
        f"- baseline checkpoint: `{BASELINE_CKPT}`\n"
        "- n_epochs: 30\n"
        "- force_save_epochs: 0..29\n"
        "- profiling after train: valid40 on epochs 0,3,6,9,12,15,18,21,24,27,29\n",
        encoding="utf-8",
    )
    return out


def run_command(command: list[str], log_path: Path, status: Path, label: str) -> int:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        code = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"\n[selected_full] {label} exit_code={code}\n")
    append_status(status, f"{label} exit={code}")
    return int(code)


def launch_full(source_config: Path, short_name: str, status: Path) -> int:
    run_stamp = stamp()
    run_name = f"{short_name}_full30_{run_stamp}"
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_full_config(source_config, run_name, output_dir)
    append_status(status, f"launching {run_name}")
    append_status(status, f"config={config}")
    append_status(status, f"output={output_dir}")

    train_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(config),
        "--prev_runid",
        str(BASELINE_CKPT),
        "--save_path",
        str(output_dir / "checkpoint_epoch{}.pth"),
    ]
    code = run_command(train_cmd, output_dir / "train.log", status, f"{short_name}_train")
    if code != 0:
        return code

    profile_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_checkpoints.py"),
        "--config",
        str(config),
        "--run-dir",
        str(output_dir),
        "--output-root",
        str(output_dir / "profiles"),
        "--samples",
        "40",
    ]
    for epoch in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29):
        profile_cmd.extend(["--epoch", str(epoch)])
    return run_command(profile_cmd, output_dir / "profile.log", status, f"{short_name}_profile")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h41-run-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--short-name", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    watcher_dir = RESULTS_DIR / f"{args.short_name}_after_h41_watcher_{stamp()}"
    watcher_dir.mkdir(parents=True, exist_ok=True)
    status = watcher_dir / "status.log"
    append_status(status, "watcher started")
    append_status(status, f"source_config={args.source_config.resolve()}")
    if not wait_for_run(args.h41_run_dir.resolve(), status, args.poll_seconds):
        return 1
    return launch_full(args.source_config.resolve(), args.short_name, status)


if __name__ == "__main__":
    raise SystemExit(main())
