"""Run standard valid825 eval for completed H57 full30 runs."""

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

RUNS = [
    {
        "label": "NSC-06b",
        "config": EXP_ROOT / "configs/generated/nsc06b_h57_all_mu010_l03_full30.yml",
        "run_dir": EXP_ROOT / "results/nsc06b_h57_all_mu010_l03_auto_full_bs8_20260603_021724_setsid",
        "epochs": [19, 23, 27, 29],
    },
    {
        "label": "NSC-07f",
        "config": EXP_ROOT / "configs/generated/nsc07f_h57_all_mu010_l04.yml",
        "run_dir": EXP_ROOT / "results/nsc07f_h57_all_mu010_l04_full30_bs8_20260603_025119_setsid",
        "epochs": [19, 24, 27, 29],
    },
]


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
        handle.write(f"\n[h57-standard-eval] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def metric_float(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, "nan"))


def parse_profile(profile: Path) -> dict[str, float]:
    data = json.loads(profile.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    dense = float(data.get("dense_flops", 0.0) or 0.0)
    effective = float(data.get("effective_flops", 0.0) or 0.0)
    return {
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


def main() -> int:
    rows: list[dict[str, Any]] = []
    started = datetime.now().isoformat(timespec="seconds")
    for item in RUNS:
        config = Path(item["config"])
        run_dir = Path(item["run_dir"])
        for epoch in item["epochs"]:
            checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
            if not checkpoint.exists():
                continue
            out_dir = run_dir / "standard_valid825" / f"epoch{epoch}"
            profile = out_dir / "spike_profile.json"
            if profile.exists():
                result = parse_profile(profile)
            else:
                exit_code = run(
                    [
                        sys.executable,
                        "-u",
                        "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
                        "--config",
                        str(config),
                        "--checkpoint",
                        str(checkpoint),
                        "--path_results",
                        str(out_dir),
                        "--mode",
                        "valid",
                    ],
                    out_dir / "eval.log",
                )
                if exit_code != 0:
                    raise RuntimeError(f"eval failed: {item['label']} epoch{epoch}; log={out_dir / 'eval.log'}")
                result = parse_profile(profile)
            rows.append({"label": item["label"], "epoch": epoch, "out_dir": out_dir, **result})

    best = min(rows, key=lambda row: row["AEE"]) if rows else None
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### 31.17 H57 NSC-06/NSC-07 full30 标准 valid825 结果（自动追加）\n\n")
        handle.write(f"- 时间：`{started}`\n")
        handle.write("- 推理：`eval_DSEC_flow_SNN.py` full valid825，CuPy backend。\n")
        handle.write("- 对比锚点：NTX-01 epoch28 AEE `1.5340`，AAE `10.2880`，total_spikes `34.6119G`。\n\n")
        handle.write("| line | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | effective_flops | sparsity | energy_uj |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['label']} | {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                f"{row['PE1']:.4f} | {row['PE2']:.4f} | {row['outlier']:.4f} | "
                f"{row['spikes_g']:.4f}G | {row['firing'] * 100:.4f}% | "
                f"{row['effective_g']:.4f}G | {row['sparsity'] * 100:.2f}% | {row['energy_uj']:.2f} |\n"
            )
        if best is not None:
            handle.write(
                f"\n当前 H57 标准 valid825 最优：{best['label']} epoch{best['epoch']}，"
                f"AEE `{best['AEE']:.4f}`，AAE `{best['AAE']:.4f}`，"
                f"total_spikes `{best['spikes_g']:.4f}G`。\n"
            )
            handle.write(
                "判断：若 AEE 接近 NTX-01 但 AAE/PE/outlier 仍高，说明 H57 residual 有保 AEE 潜力，"
                "但方向角还需要 teacher/direction loss 或更弱 FFN 稀疏来修正。\n"
            )
    print(f"evaluated {len(rows)} checkpoints")
    if best is not None:
        print(f"best {best['label']} epoch{best['epoch']} AEE={best['AEE']:.4f} AAE={best['AAE']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
