"""Keep 11bd rank2 full30 alive; run valid825 when ep29 is done."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN = EXP_ROOT / "entrypoints/train.py"
STD_EVAL = EXP_ROOT / "entrypoints/run_h9_standard_valid825_eval.py"
PY = os.environ.get("RANK2_PYTHON", "/opt/conda/bin/python3")
TARGET_EPOCH = 29
AW_LOCK = EXP_ROOT / "results/GPU_LOCK_11aw_full30"
RANK2_LOCK = EXP_ROOT / "results/GPU_LOCK_11bd_rank2_full30"


def aw_lock_blocks_rank2() -> bool:
    """11aw lock blocks rank2 only when ACTIVE and rank2 lock is not ACTIVE."""
    if RANK2_LOCK.is_file() and "ACTIVE" in RANK2_LOCK.read_text(encoding="utf-8", errors="ignore").upper():
        return False
    if not AW_LOCK.is_file():
        return False
    text = AW_LOCK.read_text(encoding="utf-8", errors="ignore").upper()
    return "PAUSED" not in text


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def train_pids(run_dir: Path) -> list[int]:
    markers = (run_dir.name, str(run_dir))
    pids: list[int] = []
    for pattern in ("entrypoints/train.py", "H9_bipolar_self_attention/entrypoints/train.py"):
        try:
            out = subprocess.check_output(["pgrep", "-af", pattern], text=True)
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            if "train.py" not in line:
                continue
            if any(m in line for m in markers):
                pid = int(line.split(None, 1)[0])
                if pid not in pids:
                    pids.append(pid)
    return pids


def train_log_active(run_dir: Path, within_sec: int = 30) -> bool:
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        pipeline = run_dir / "pipeline.log"
        if not pipeline.is_file():
            return False
        log_path = pipeline
    return time.time() - log_path.stat().st_mtime < within_sec


def latest_checkpoint(run_dir: Path) -> tuple[int, Path] | tuple[None, None]:
    ckpts = sorted(
        (int(p.stem.replace("checkpoint_epoch", "")), p)
        for p in run_dir.glob("checkpoint_epoch*.pth")
        if "state_dict" not in p.name
    )
    if not ckpts:
        return None, None
    return ckpts[-1]


def training_complete(run_dir: Path, target_epoch: int = TARGET_EPOCH) -> bool:
    if (run_dir / f"checkpoint_epoch{target_epoch}.pth").is_file():
        return True
    for name in ("train.log", "pipeline.log"):
        log_path = run_dir / name
        if not log_path.is_file():
            continue
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-12000:]
        if f"Epoch {target_epoch}" not in tail:
            continue
        if "Epoch loss (Validation):" in tail.rsplit(f"Epoch {target_epoch}", 1)[-1]:
            return True
    return False


def launch_training(run_dir: Path, config: Path, ckpt: Path, status: Path) -> None:
    state = ckpt.with_name(ckpt.name.replace(".pth", "_state_dict.pth"))
    cmd = [
        PY,
        "-u",
        str(TRAIN),
        "--config",
        str(config),
        "--prev_runid",
        str(ckpt),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    if state.is_file():
        cmd.extend(["--resume", "1"])
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path = run_dir / "train.log"
    log(status, f"launch train from {ckpt.name} resume={int(state.is_file())}")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[rank2-guardian {datetime.now().isoformat(timespec='seconds')}] $ {' '.join(cmd)}\n")
        handle.flush()
        subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def run_valid825(run_dir: Path, config: Path, status: Path) -> int:
    eval_epochs = [ep for ep in (9, 14, 19, 24, 28, 29) if (run_dir / f"checkpoint_epoch{ep}.pth").is_file()]
    if not eval_epochs:
        ep, _ = latest_checkpoint(run_dir)
        if ep is None:
            log(status, "valid825 skipped: no checkpoints")
            return 1
        eval_epochs = [ep]
    cmd = [PY, "-u", str(STD_EVAL), "--config", str(config), "--run-dir", str(run_dir)]
    for ep in eval_epochs:
        cmd.extend(["--epoch", str(ep)])
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(status, f"valid825 epochs={eval_epochs}")
    log_path = run_dir / "valid825.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[rank2-guardian] valid825 exit_code={proc.returncode}\n")
    log(status, f"valid825 done exit_code={proc.returncode}")
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--poll-sec", default=180, type=int)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = args.config.resolve()
    baseline = args.baseline.resolve()
    status = run_dir / "rank2_guardian_status.log"
    log(status, f"rank2 guardian start run_dir={run_dir}")

    while True:
        if aw_lock_blocks_rank2():
            log(status, "paused: GPU_LOCK_11aw_full30 ACTIVE (not PAUSED)")
            time.sleep(args.poll_sec)
            continue

        if training_complete(run_dir):
            log(status, "training complete")
            return run_valid825(run_dir, config, status)

        pids = train_pids(run_dir)
        active = train_log_active(run_dir)
        ep, ckpt = latest_checkpoint(run_dir)
        log(status, f"poll: train_pids={pids or 'none'} log_active={active} latest={ckpt.name if ckpt else 'none'}")

        stale = not train_log_active(run_dir, within_sec=30)
        if not pids and (not active or stale):
            start_ckpt = ckpt if ckpt is not None else baseline
            launch_training(run_dir, config, start_ckpt, status)

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())