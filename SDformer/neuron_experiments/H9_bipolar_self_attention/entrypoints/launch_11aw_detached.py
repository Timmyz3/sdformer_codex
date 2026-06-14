#!/usr/bin/env python3
"""Double-fork launcher so harness SIGKILL does not take down training."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO = EXP_ROOT.parents[1]
RUN_DIR = EXP_ROOT / "results/nts11aw_hw_h60_s23_sn2qbin_w720_stdlr_bs8_20260613_133609_setsid"
CFG = EXP_ROOT / "configs/generated/nts11aw_hw_h60_s23_sn2qbin_w720_stdlr_scope_full30.yml"
CKPT = RUN_DIR / "checkpoint_epoch22.pth"
PY = "/opt/conda/bin/python3"
TRAIN = EXP_ROOT / "entrypoints/train.py"
GUARDIAN = EXP_ROOT / "entrypoints/run_nts11_hw_friendly_guardian.py"
MODE = sys.argv[1] if len(sys.argv) > 1 else "train"


def daemonize() -> None:
    if os.fork() > 0:
        raise SystemExit(0)
    os.setsid()
    if os.fork() > 0:
        raise SystemExit(0)
    os.chdir(REPO)
    os.umask(0o22)


def env() -> dict[str, str]:
    out = os.environ.copy()
    out["SDFORMER_USE_MLFLOW"] = "0"
    out["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    out["SDFORMER_SNN_BACKEND"] = "cupy"
    out["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return out


def launch_train() -> None:
    state = CKPT.with_name(CKPT.name.replace(".pth", "_state_dict.pth"))
    cmd = [PY, "-u", str(TRAIN), "--config", str(CFG), "--prev_runid", str(CKPT), "--save_path", str(RUN_DIR / "checkpoint_epoch{}.pth")]
    if state.is_file():
        cmd.extend(["--resume", "1"])
    log_path = RUN_DIR / "train.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[detached_launch {datetime.now().isoformat(timespec='seconds')}] $ {' '.join(cmd)}\n")
        handle.flush()
        subprocess.Popen(cmd, cwd=REPO, env=env(), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)


def launch_guardian() -> None:
    log_path = EXP_ROOT / "results/nts11_hw_friendly_autopilot_overnight.log"
    cmd = [PY, "-u", str(GUARDIAN), "--run-dir", str(RUN_DIR), "--poll-sec", "180"]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[detached_launch {datetime.now().isoformat(timespec='seconds')}] guardian\n")
        handle.flush()
        subprocess.Popen(cmd, cwd=REPO, env=env(), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)


def main() -> None:
    daemonize()
    if MODE == "guardian":
        launch_guardian()
    elif MODE == "both":
        launch_train()
        launch_guardian()
    else:
        launch_train()


if __name__ == "__main__":
    main()