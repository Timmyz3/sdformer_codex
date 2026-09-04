"""Keep 11aw full30 alive: auto-resume train, then valid825 when ep29 is done."""

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
PY = os.environ.get("HW_FRIENDLY_PYTHON", "/opt/conda/bin/python3")
DEFAULT_CFG = EXP_ROOT / "configs/generated/nts11aw_hw_h60_s23_sn2qbin_w720_stdlr_scope_full30.yml"
TARGET_EPOCH = 29


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def train_pids(run_dir: Path) -> list[int]:
    """Return all train.py PIDs for this run (main + DataLoader workers)."""
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


def train_alive(run_dir: Path) -> bool:
    return bool(train_pids(run_dir))


def launch_cooldown_active(run_dir: Path, within_sec: int = 600) -> bool:
    marker = run_dir / ".guardian_last_launch"
    if not marker.is_file():
        return False
    return (time.time() - marker.stat().st_mtime) < within_sec


def train_log_active(run_dir: Path, within_sec: int = 300) -> bool:
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        return False
    age = time.time() - log_path.stat().st_mtime
    return age < within_sec


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
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="ignore")[-12000:]
    if f"Epoch {target_epoch}" not in tail:
        return False
    return "Epoch loss (Validation):" in tail.rsplit(f"Epoch {target_epoch}", 1)[-1]


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
    (run_dir / ".guardian_last_launch").write_text(datetime.now().isoformat(), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[guardian {datetime.now().isoformat(timespec='seconds')}] $ {' '.join(cmd)}\n")
        handle.flush()
        subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def run_valid825(run_dir: Path, config: Path, status: Path, driver_dir: Path) -> int:
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
    log_path = driver_dir / "valid825.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[guardian] valid825 exit_code={proc.returncode}\n")
    log(status, f"valid825 done exit_code={proc.returncode}")
    return int(proc.returncode)


def gpu_lock_paused() -> bool:
    rank2 = EXP_ROOT / "results" / "GPU_LOCK_11bd_rank2_full30"
    if rank2.is_file() and "ACTIVE" in rank2.read_text(encoding="utf-8", errors="ignore").upper():
        return True
    lock = EXP_ROOT / "results" / "GPU_LOCK_11aw_full30"
    if not lock.is_file():
        return False
    first = lock.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[0].upper()
    return first.startswith("PAUSED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", default=str(DEFAULT_CFG), type=Path)
    parser.add_argument("--poll-sec", default=180, type=int)
    parser.add_argument("--baseline", default="", help="NB0 ckpt if run-dir has no checkpoints")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = args.config.resolve()
    driver_dir = EXP_ROOT / "results" / f"nts11_hw_friendly_guardian_{stamp()}"
    status = driver_dir / "status.log"
    log(status, f"guardian start run_dir={run_dir}")

    while True:
        if gpu_lock_paused():
            log(status, "GPU_LOCK_11aw_full30=PAUSED; guardian exiting")
            return 0
        if training_complete(run_dir):
            log(status, "training complete")
            return run_valid825(run_dir, config, status, driver_dir)

        pids = train_pids(run_dir)
        alive = bool(pids)
        active = train_log_active(run_dir)
        ep, ckpt = latest_checkpoint(run_dir)
        log(
            status,
            f"poll: train_alive={alive} n_pids={len(pids)} log_active={active} latest={ckpt.name if ckpt else 'none'}",
        )

        # Resume only when no train.py (main or workers) and log stale (>120s during CUDA load).
        stale = not train_log_active(run_dir, within_sec=120)
        cooldown = launch_cooldown_active(run_dir)
        if not alive and not cooldown and (not active or stale):
            if ckpt is None:
                if not args.baseline:
                    log(status, "no checkpoint and no --baseline; abort")
                    return 1
                ckpt = Path(args.baseline).resolve()
                log(status, f"cold start from baseline {ckpt}")
            launch_training(run_dir, config, ckpt, status)

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())