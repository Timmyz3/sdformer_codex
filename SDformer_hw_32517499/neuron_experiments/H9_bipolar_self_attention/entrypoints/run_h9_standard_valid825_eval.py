"""Run standard valid825 eval for one completed H9 full-training run.

This is the post-full counterpart to rapid-screen promotion. It evaluates a set
of saved checkpoints with `eval_DSEC_flow_SNN.py`, writes `standard_valid825`
artifacts under the run directory, and produces a compact ranking markdown for
later inclusion in the redesign notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        handle.write(f"\n[h9-standard-valid825] exit_code={proc.returncode}\n")
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
        "AAE_Benchmark": metric_float(metrics, "AAE_Benchmark"),
        "PE1": metric_float(metrics, "AEE_PE1"),
        "PE2": metric_float(metrics, "AEE_PE2"),
        "outlier": metric_float(metrics, "AEE_outliers"),
        "spikes_g": float(data.get("total_spikes", 0.0) or 0.0) / 1e9,
        "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
        "effective_g": effective / 1e9,
        "sparsity": 1.0 - effective / dense if dense else 0.0,
        "energy_uj": float(data.get("energy_uj", 0.0) or 0.0),
    }


def candidate_score(row: dict[str, float]) -> float:
    return (
        float(row["AEE"])
        + 0.025 * float(row["AAE"])
        + 0.20 * max(0.0, float(row["spikes_g"]) - 34.5)
    )


def profile_artifact_status(
    profile: Path,
    config: Path,
    checkpoint: Path,
) -> str:
    raw = json.loads(profile.read_text(encoding="utf-8"))
    identity = raw.get("artifact_identity")
    if not isinstance(identity, dict):
        return "legacy"
    stat = checkpoint.stat()
    expected = {
        "config_path": str(config.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    return (
        "match"
        if all(identity.get(key) == value for key, value in expected.items())
        else "mismatch"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epoch", action="append", type=int, default=[])
    parser.add_argument(
        "--ranking-mode",
        choices=("candidate", "aee"),
        default="candidate",
        help=(
            "Sort by the historical crop precision/spike candidate score, or by "
            "AEE. Full-resolution evaluations must use 'aee' because the crop "
            "34.5G spike target is not resolution invariant."
        ),
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = args.config.resolve()
    config_data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    h9_enabled = bool(
        config_data.get("atlif_ternary_psn", {}).get("enabled")
        or config_data.get("bsa_attention", {}).get("enabled")
        or config_data.get("simple_ternary_psn", {}).get("enabled")
    )
    epochs = args.epoch or [19, 24, 29]

    rows: list[dict[str, Any]] = []
    skipped_epochs: list[int] = []
    for epoch in epochs:
        checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
        if not checkpoint.exists():
            skipped_epochs.append(epoch)
            continue
        out_dir = run_dir / "standard_valid825" / f"epoch{epoch}"
        profile = out_dir / "spike_profile.json"
        if profile.exists():
            status = profile_artifact_status(profile, config, checkpoint)
            if status == "mismatch":
                raise RuntimeError(
                    f"epoch{epoch} profile artifact mismatch: {profile}; "
                    "refusing silent reuse"
                )
            if status == "legacy":
                print(
                    f"epoch{epoch}: reusing legacy profile without artifact identity: "
                    f"{profile}"
                )
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
                raise RuntimeError(f"eval failed for epoch{epoch}; log={out_dir / 'eval.log'}")
            if profile_artifact_status(profile, config, checkpoint) != "match":
                raise RuntimeError(
                    f"epoch{epoch} newly generated profile lacks matching provenance: "
                    f"{profile}"
                )
        raw_profile = json.loads(profile.read_text(encoding="utf-8"))
        if int(raw_profile.get("samples", 0)) != 825:
            raise RuntimeError(
                f"epoch{epoch} standard valid825 sample count is not 825: "
                f"{raw_profile.get('samples')}"
            )
        # Current H9 configs require a complete audited overlay load. Baseline
        # configs legitimately have no H9 load-audit payload.
        if h9_enabled:
            audit = raw_profile.get("checkpoint_load_audit")
            counts = raw_profile.get("module_counts")
            if not isinstance(audit, dict) or not isinstance(counts, dict):
                raise RuntimeError(f"epoch{epoch} H9 profile lacks load/module audit")
            if int(audit.get("missing_count", -1)) or int(
                audit.get("unexpected_count", -1)
            ):
                raise RuntimeError(f"epoch{epoch} H9 checkpoint load is incomplete: {audit}")
            if int(counts.get("ATLIFTernaryPSN", 0)) <= 0 or int(
                counts.get("ShiftmaxAttention", 0)
            ) <= 0:
                raise RuntimeError(f"epoch{epoch} H9 module coverage is empty: {counts}")
        rows.append({"epoch": epoch, **parse_profile(profile)})

    if not rows:
        raise RuntimeError(f"no standard valid825 rows produced for {run_dir}")
    if skipped_epochs:
        print(f"skipped missing checkpoints: {', '.join(str(epoch) for epoch in skipped_epochs)}")

    ranking_key = candidate_score if args.ranking_mode == "candidate" else lambda row: float(row["AEE"])
    rows = sorted(rows, key=ranking_key)
    ranking = run_dir / "profile_ranking_valid825.md"
    with ranking.open("w", encoding="utf-8") as handle:
        handle.write("# Standard Valid825 Ranking\n\n")
        handle.write(f"Ranking mode: `{args.ranking_mode}`.\n\n")
        handle.write("The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.\n\n")
        handle.write("| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(rows, 1):
            handle.write(
                f"| {rank} | {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                f"{row['AAE_Benchmark']:.4f} | {row['PE1']:.4f} | {row['PE2']:.4f} | {row['outlier']:.4f} | "
                f"{row['spikes_g']:.4f}G | {row['firing'] * 100:.4f}% | {row['energy_uj']:.2f} |\n"
            )
    best = rows[0]
    print(f"best epoch{best['epoch']} AEE={best['AEE']:.4f} AAE={best['AAE']:.4f} spikes={best['spikes_g']:.4f}G")
    print(f"ranking={ranking}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
