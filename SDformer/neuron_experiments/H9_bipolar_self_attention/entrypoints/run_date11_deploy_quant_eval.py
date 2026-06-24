"""Evaluate DATE11 all-binary NTS/H60 deployment quantization configs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
REDESIGN_MD = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)
CHECKPOINT = (
    EXP
    / "results/date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid/checkpoint_epoch2.pth"
)
CONFIGS = [
    GEN / "date11_binary_nts_ft_ep29_deploy_float_ref.yml",
    GEN / "date11_binary_nts_ft_ep29_deploy_score_int8.yml",
    GEN / "date11_binary_nts_ft_ep29_deploy_score_int8_mu_pow2.yml",
    GEN / "date11_binary_nts_ft_ep29_deploy_score_int8_mu_pow2_gate_int8.yml",
]


def run_eval(config: Path, out_dir: Path) -> None:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PY),
        "-u",
        "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
        "--config",
        str(config),
        "--checkpoint",
        str(CHECKPOINT),
        "--path_results",
        str(out_dir),
        "--mode",
        "valid",
    ]
    with (out_dir / "eval.log").open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[date11-deploy-quant] exit_code={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed for {config}; log={out_dir / 'eval.log'}")


def parse_profile(profile: Path) -> dict[str, str]:
    data = json.loads(profile.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "AEE": f"{float(metrics.get('AEE', 0.0)):.4f}",
        "AAE": f"{float(metrics.get('AAE', 0.0)):.4f}",
        "PE1": f"{float(metrics.get('AEE_PE1', 0.0)):.4f}",
        "PE2": f"{float(metrics.get('AEE_PE2', 0.0)):.4f}",
        "outlier": f"{float(metrics.get('AEE_outliers', 0.0)):.4f}",
        "spikes": f"{float(data.get('total_spikes', 0.0)) / 1e9:.4f}G",
        "firing": f"{float(data.get('global_firing_rate', 0.0)) * 100.0:.4f}%",
        "energy": f"{float(data.get('energy_uj', 0.0)):.2f}",
    }


def append_md(run_dir: Path, rows: list[dict[str, str]]) -> None:
    marker = f"DATE11_DEPLOY_QUANT::{run_dir.name}"
    text = REDESIGN_MD.read_text(encoding="utf-8")
    if marker in text:
        return
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n\n### DATE11 自动结果追加：all-binary NTS/H60 deploy quant（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）\n\n"
        )
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- 主 checkpoint：`{CHECKPOINT.relative_to(REPO)}`\n")
        handle.write(f"- 运行目录：`{run_dir.relative_to(REPO)}`\n")
        handle.write("- 目的：验证 all-binary NTS/H60 FT ep2 主线在 int8 score / pow2 μ / int8 gate 下是否保持等价。\n\n")
        handle.write("| config | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| `{row['config']}` | {row['AEE']} | {row['AAE']} | {row['PE1']} | {row['PE2']} | "
                f"{row['outlier']} | {row['spikes']} | {row['firing']} | {row['energy']} |\n"
            )


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXP / "results" / f"date11_binary_nts_deploy_quant_full825_{stamp}"
    rows: list[dict[str, str]] = []
    for config in CONFIGS:
        out_dir = run_dir / config.stem
        run_eval(config, out_dir)
        row = parse_profile(out_dir / "spike_profile.json")
        row["config"] = config.stem
        rows.append(row)
    append_md(run_dir, rows)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
