"""Follow up NSC-04 resume training with standard valid825 evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
REDESIGN_MD = REPO_ROOT / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
RUN_DIR = (
    REPO_ROOT
    / "neuron_experiments/H9_bipolar_self_attention/results/"
    "nsc04d_blend_mu05_l06_tr03_ang02_auto_full_auto_full_bs8_20260602_142622_setsid"
)
CONFIG = (
    REPO_ROOT
    / "neuron_experiments/H9_bipolar_self_attention/configs/"
    "nsc04d_blend_mu05_l06_tr03_ang02_resume11_29_20260602.yml"
)
PID_FILE = RUN_DIR / "resume11_29.pid"
STATUS = RUN_DIR / "resume_followup_status.log"
EPOCHS = [19, 23, 28, 29]


def append_status(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_resume() -> None:
    if not PID_FILE.exists():
        append_status(f"pid file missing: {PID_FILE}")
        return
    pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    append_status(f"waiting for resume pid={pid}")
    while pid_alive(pid):
        time.sleep(120)
    append_status(f"resume pid finished: {pid}")


def run(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[nsc04-followup] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def metric_float(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, "nan"))


def standard_eval() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        checkpoint = RUN_DIR / f"checkpoint_epoch{epoch}.pth"
        if not checkpoint.exists():
            append_status(f"standard eval skip missing {checkpoint.name}")
            continue
        out_dir = RUN_DIR / "standard_valid825" / f"epoch{epoch}"
        profile = out_dir / "spike_profile.json"
        if profile.exists():
            profile.unlink()
        append_status(f"standard eval epoch{epoch} start")
        code = run(
            [
                sys.executable,
                "-u",
                "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
                "--config",
                str(CONFIG),
                "--checkpoint",
                str(checkpoint),
                "--path_results",
                str(out_dir),
                "--mode",
                "valid",
            ],
            out_dir / "eval.log",
        )
        if code != 0 or not profile.exists():
            append_status(f"standard eval epoch{epoch} failed code={code}")
            continue
        data = json.loads(profile.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        dense = float(data.get("dense_flops", 0.0) or 0.0)
        effective = float(data.get("effective_flops", 0.0) or 0.0)
        row = {
            "epoch": epoch,
            "AEE": metric_float(metrics, "AEE"),
            "AAE": metric_float(metrics, "AAE"),
            "PE1": metric_float(metrics, "AEE_PE1"),
            "PE2": metric_float(metrics, "AEE_PE2"),
            "outlier": metric_float(metrics, "AEE_outliers"),
            "spikes_g": float(data.get("total_spikes", 0.0) or 0.0) / 1e9,
            "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
            "effective_g": effective / 1e9,
            "sparsity": 1.0 - effective / dense if dense else 0.0,
            "energy_uj": float(data.get("energy_uj", 0.0) or 0.0),
        }
        rows.append(row)
        append_status(f"standard eval epoch{epoch} done AEE={row['AEE']:.4f} AAE={row['AAE']:.4f}")
    return rows


def append_md(rows: list[dict[str, Any]]) -> None:
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### 31.13.1 NSC-04d resume 后标准 valid825 结果（自动追加）\n\n")
        handle.write(f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write(f"- 配置：`{CONFIG}`\n")
        handle.write(f"- 目录：`{RUN_DIR}`\n")
        handle.write("- 推理：`eval_DSEC_flow_SNN.py` full valid825，CuPy backend。\n\n")
        handle.write("| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | effective_flops | sparsity | energy_uj |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                f"{row['PE1']:.4f} | {row['PE2']:.4f} | {row['outlier']:.4f} | "
                f"{row['spikes_g']:.4f}G | {row['firing'] * 100:.4f}% | "
                f"{row['effective_g']:.4f}G | {row['sparsity'] * 100:.2f}% | {row['energy_uj']:.2f} |\n"
            )
        if rows:
            best = min(rows, key=lambda row: row["AEE"])
            handle.write(f"\n当前最优：epoch{best['epoch']}，AEE `{best['AEE']:.4f}`，AAE `{best['AAE']:.4f}`。\n")
        else:
            handle.write("\n标准推理未生成有效结果，需要检查 `resume_followup_status.log` 和各 epoch `eval.log`。\n")


def main() -> int:
    append_status("followup start")
    wait_for_resume()
    missing = [epoch for epoch in EPOCHS if not (RUN_DIR / f"checkpoint_epoch{epoch}.pth").exists()]
    if missing:
        append_status(f"missing checkpoints after resume: {missing}")
    rows = standard_eval()
    append_md(rows)
    append_status("followup complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
