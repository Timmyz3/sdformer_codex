"""Run standard MVSEC dt1 eval for one or more DSEC-trained checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEQUENCES = ["indoor_flying1", "indoor_flying2", "indoor_flying3"]


def run(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = os.environ.get("SDFORMER_SNN_BACKEND", "auto")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[h9-standard-mvsec] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def metric_float(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, "nan"))


def parse_profile(profile: Path) -> dict[str, float]:
    data = json.loads(profile.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    dense = float(data.get("dense_flops", 0.0) or 0.0)
    effective = float(data.get("effective_flops", 0.0) or 0.0)
    return {
        "sequence": str(data.get("sequence", "")),
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


def patch_config_sequence(config_path: Path, sequence: str, out_path: Path) -> None:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    patched: list[str] = []
    for line in lines:
        if line.strip().startswith("test_sequence:"):
            patched.append(f'  test_sequence: {sequence}')
        else:
            patched.append(line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(patched) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sequence", action="append", default=[])
    args = parser.parse_args()

    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    sequences = args.sequence or DEFAULT_SEQUENCES

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        seq_dir = out_dir / sequence
        profile = seq_dir / "spike_profile.json"
        if not profile.exists():
            seq_config = seq_dir / "eval_config.yml"
            patch_config_sequence(config, sequence, seq_config)
            exit_code = run(
                [
                    sys.executable,
                    "-u",
                    "third_party/SDformerFlow/eval_MV_flow_SNN.py",
                    "--config",
                    str(seq_config),
                    "--checkpoint",
                    str(checkpoint),
                    "--path_results",
                    str(seq_dir),
                    "--mode",
                    "valid",
                ],
                seq_dir / "eval.log",
            )
            if exit_code != 0:
                raise RuntimeError(f"MVSEC eval failed for {sequence}; log={seq_dir / 'eval.log'}")
        rows.append({"sequence": sequence, **parse_profile(profile)})

    ranking = out_dir / "mvsec_ranking.md"
    with ranking.open("w", encoding="utf-8") as handle:
        handle.write("# MVSEC dt1 Ranking\n\n")
        handle.write("| sequence | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['sequence']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                f"{row['PE1']:.4f} | {row['PE2']:.4f} | {row['outlier']:.4f} | "
                f"{row['spikes_g']:.4f}G | {row['firing'] * 100:.4f}% | {row['energy_uj']:.2f} |\n"
            )

    mean_aee = sum(r["AEE"] for r in rows) / len(rows)
    mean_aae = sum(r["AAE"] for r in rows) / len(rows)
    print(f"MVSEC mean AEE={mean_aee:.4f} AAE={mean_aae:.4f} over {len(rows)} sequences")
    print(f"ranking={ranking}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())