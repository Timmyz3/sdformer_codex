"""Resume interrupted NTS10d full30 from latest checkpoint, then run standard valid825."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
REDESIGN_MD = REPO_ROOT / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"

CONFIG = EXP_ROOT / "configs/nts10d_hw_h60_s23_freeze1224_s1224_steps1224_auto_full_20260610_151207.yml"
RUN_DIR = EXP_ROOT / "results/nts10d_hw_h60_s23_freeze1224_s1224_steps1224_auto_full_bs6_20260610_151207_setsid"
SHORT_DIR = EXP_ROOT / "results/nts10_blocks_20260610_141114"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_status(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    checkpoints = [item for item in checkpoints if "_state_dict" not in item.name]
    return checkpoints[-1] if checkpoints else None


def run(command: list[str], log_path: Path, *, check: bool = True) -> int:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[nts10d-crash-resume] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}; log={log_path}")
    return int(proc.returncode)


def metric_float(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, "nan"))


def append_md(rows: list[dict[str, Any]]) -> None:
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### NTS-10d S23 full30 宕机续训 + valid825（自动追加）\n\n")
        handle.write(f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write(f"- 短测目录：`{SHORT_DIR}`\n")
        handle.write(f"- 全量配置：`{CONFIG}`\n")
        handle.write(f"- 全量目录：`{RUN_DIR}`\n")
        handle.write("- 方法：NTS09e freeze1224 基座，S2+S3（8 block）扩大替换。\n\n")
        handle.write("| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                f"{row['PE1']:.4f} | {row['PE2']:.4f} | {row['outlier']:.4f} | "
                f"{row['spikes_g']:.4f}G | {row['firing'] * 100:.4f}% | {row['energy_uj']:.2f} |\n"
            )


def main() -> int:
    driver_dir = EXP_ROOT / "results" / f"nts10d_crash_resume_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, f"run_dir={RUN_DIR}")

    resume_ckpt = latest_checkpoint(RUN_DIR)
    if resume_ckpt is None:
        append_status(status, "no checkpoint found")
        return 1
    append_status(status, f"resume_from={resume_ckpt.name}")

    target_epochs = [19, 24, 29]
    missing = [epoch for epoch in target_epochs if not (RUN_DIR / f"checkpoint_epoch{epoch}.pth").exists()]
    if not missing and all((RUN_DIR / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json").exists() for epoch in target_epochs):
        append_status(status, "full30 + valid825 already complete")
        return 0

    if missing:
        train_cmd = [
            sys.executable,
            "-u",
            str(EXP_ROOT / "entrypoints/train.py"),
            "--config",
            str(CONFIG),
            "--prev_runid",
            str(resume_ckpt),
            "--resume",
            "True",
            "--save_path",
            str(RUN_DIR / "checkpoint_epoch{}.pth"),
        ]
        append_status(status, "resume full30 training start")
        run(train_cmd, RUN_DIR / "train.log")
        append_status(status, "resume full30 training done")

    waiter_log = RUN_DIR / "standard_valid825_waiter" / "wait.log"
    waiter_log.parent.mkdir(parents=True, exist_ok=True)
    waiter_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/wait_full_then_run_standard_valid825.py"),
        "--config",
        str(CONFIG),
        "--run-dir",
        str(RUN_DIR),
        "--epoch",
        "19",
        "--epoch",
        "24",
        "--epoch",
        "29",
        "--timeout-hours",
        "12",
    ]
    append_status(status, "standard valid825 waiter start")
    run(waiter_cmd, waiter_log)
    append_status(status, "standard valid825 waiter done")

    rows: list[dict[str, Any]] = []
    for epoch in target_epochs:
        summary = RUN_DIR / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"
        if not summary.exists():
            append_status(status, f"missing {summary}")
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        rows.append(
            {
                "epoch": epoch,
                "AEE": metric_float(metrics, "AEE"),
                "AAE": metric_float(metrics, "AAE"),
                "PE1": metric_float(metrics, "AEE_PE1"),
                "PE2": metric_float(metrics, "AEE_PE2"),
                "outlier": metric_float(metrics, "AEE_outliers"),
                "spikes_g": float(data.get("total_spikes", 0.0) or 0.0) / 1e9,
                "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
                "energy_uj": float(data.get("energy_uj", 0.0) or 0.0),
            }
        )
        append_status(
            status,
            f"valid825 epoch{epoch} AEE={rows[-1]['AEE']:.4f} AAE={rows[-1]['AAE']:.4f} spikes={rows[-1]['spikes_g']:.4f}G",
        )

    if rows:
        append_md(rows)
    append_status(status, "NTS10d crash resume complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())