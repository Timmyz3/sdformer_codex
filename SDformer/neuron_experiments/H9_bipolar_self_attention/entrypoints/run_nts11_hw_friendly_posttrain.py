"""Wait for an in-flight full30 run, then run standard valid825 eval."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
STD_EVAL = EXP_ROOT / "entrypoints/run_h9_standard_valid825_eval.py"
PY = os.environ.get("HW_FRIENDLY_PYTHON", "/opt/conda/bin/python3")
DEFAULT_CFG = EXP_ROOT / "configs/generated/nts11aw_hw_h60_s23_sn2qbin_w720_stdlr_scope_full30.yml"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def train_pids_for_run(run_dir: Path) -> list[int]:
    markers = (run_dir.name, str(run_dir))
    pids: list[int] = []
    try:
        out = subprocess.check_output(["pgrep", "-af", "entrypoints/train.py"], text=True)
    except subprocess.CalledProcessError:
        return pids
    for line in out.splitlines():
        if "train.py" not in line:
            continue
        if any(marker in line for marker in markers):
            pid = int(line.split(None, 1)[0])
            pids.append(pid)
    return pids


def training_complete(run_dir: Path, target_epoch: int = 29) -> bool:
    final_ckpt = run_dir / f"checkpoint_epoch{target_epoch}.pth"
    if final_ckpt.is_file():
        return True
    train_log = run_dir / "train.log"
    if not train_log.is_file():
        return False
    tail = train_log.read_text(encoding="utf-8", errors="ignore")[-8000:]
    return f"Epoch {target_epoch}" in tail and "Epoch loss (Validation):" in tail.split(f"Epoch {target_epoch}")[-1]


def wait_for_training(run_dir: Path, status: Path, poll_sec: int = 120, target_epoch: int = 29) -> None:
    while True:
        pids = train_pids_for_run(run_dir)
        ckpts = sorted(
            p for p in run_dir.glob("checkpoint_epoch*.pth") if "state_dict" not in p.name
        )
        latest = ckpts[-1].name if ckpts else "none"
        done = training_complete(run_dir, target_epoch=target_epoch)
        log(status, f"poll: train_pids={pids or 'none'} latest_ckpt={latest} complete={done}")
        if done and not pids:
            return
        if not pids and latest != "none":
            latest_ep = int(ckpts[-1].stem.replace("checkpoint_epoch", ""))
            if latest_ep >= target_epoch:
                return
        time.sleep(poll_sec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", default=str(DEFAULT_CFG), type=Path)
    parser.add_argument("--poll-sec", default=120, type=int)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = args.config.resolve()
    driver_dir = EXP_ROOT / "results" / f"nts11_hw_friendly_posttrain_{stamp()}"
    status = driver_dir / "status.log"
    log(status, f"watching run_dir={run_dir}")

    wait_for_training(run_dir, status, poll_sec=args.poll_sec, target_epoch=29)

    eval_epochs = [ep for ep in (9, 14, 19, 24, 28, 29) if (run_dir / f"checkpoint_epoch{ep}.pth").is_file()]
    if not eval_epochs:
        ckpts = sorted(
            p for p in run_dir.glob("checkpoint_epoch*.pth") if "state_dict" not in p.name
        )
        if not ckpts:
            log(status, "no checkpoints found after training stopped")
            return 1
        eval_epochs = [int(ckpts[-1].stem.replace("checkpoint_epoch", ""))]

    log(status, f"valid825 epochs={eval_epochs}")
    cmd = [PY, "-u", str(STD_EVAL), "--config", str(config), "--run-dir", str(run_dir)]
    for ep in eval_epochs:
        cmd.extend(["--epoch", str(ep)])
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path = driver_dir / "valid825.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[posttrain] exit_code={proc.returncode}\n")
    log(status, f"valid825 done exit_code={proc.returncode}")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())