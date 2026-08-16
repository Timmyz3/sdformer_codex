"""Run standard MVSEC dt1 eval for one or more DSEC-trained checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEQUENCES = ["outdoor_day1", "indoor_flying1", "indoor_flying2", "indoor_flying3"]


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
        "PE1": metric_float(metrics, "AEE_PE1"),
        "PE2": metric_float(metrics, "AEE_PE2"),
        "outlier": metric_float(metrics, "AEE_outliers"),
        "gt_fl_percent": metric_float(metrics, "DSEC_Fl"),
        "spikes_g": float(data.get("total_spikes", 0.0) or 0.0) / 1e9,
        "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
        "effective_g": effective / 1e9,
        "sparsity": 1.0 - effective / dense if dense else 0.0,
        "energy_uj": float(data.get("energy_uj", 0.0) or 0.0),
        "samples": int(data.get("samples", 0) or 0),
        "valid_pixels": int(data.get("valid_pixels", 0) or 0),
        "weighted_aee": float(data.get("valid_pixel_weighted_aee", "nan")),
    }


def patch_config_sequence(
    config_path: Path,
    sequence: str,
    out_path: Path,
    fixed800_manifest: Path | None,
) -> None:
    patched = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    patched["data"]["test_sequence"] = sequence
    metric_names = list(patched.get("metrics", {}).get("name") or [])
    if "DSEC_Fl" not in metric_names:
        metric_names.append("DSEC_Fl")
    patched["metrics"]["name"] = metric_names
    if fixed800_manifest is not None:
        patched["data"]["mvsec_split_manifest"] = str(fixed800_manifest.resolve())
        patched["data"]["mvsec_eval_split"] = f"test_fixed800_{sequence}"
    else:
        patched["data"].pop("mvsec_eval_split", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")


def audit_eval_load(log_path: Path, config_data: dict[str, Any]) -> None:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    h9_enabled = bool(config_data.get("atlif_ternary_psn", {}).get("enabled")) or bool(
        config_data.get("bsa_attention", {}).get("enabled")
    )
    if h9_enabled:
        required = (
            "[H9] eval installed ATLIFTernaryPSN: 105 modules",
            "[H9] eval installed Shiftmax attention: 12 modules",
            "checkpoint_overlay_keys=210, model_overlay_keys=210, missing=0, unexpected=0",
        )
    else:
        required = (
            "checkpoint_overlay_keys=0, model_overlay_keys=0, missing=0, unexpected=0",
        )
    missing = [marker for marker in required if marker not in log_text]
    if missing:
        raise RuntimeError(f"MVSEC eval load audit failed for {log_path}: missing {missing}")


def mvsec_sequence_ready(sequence: str) -> tuple[bool, str]:
    seq_root = REPO_ROOT / "third_party" / "SDformerFlow" / "data" / "Datasets" / "MVSEC" / "MVSEC_test" / sequence
    event_dir = seq_root / "event"
    flow_dir = seq_root / "flowgt_dt1"
    if not event_dir.is_dir():
        return False, f"missing {event_dir}"
    if not flow_dir.is_dir():
        return False, f"missing {flow_dir}"
    if not any(event_dir.glob("*.h5")):
        return False, f"empty {event_dir}"
    if not any(flow_dir.glob("*.npy")):
        return False, f"empty {flow_dir}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--fixed800-manifest", type=Path)
    args = parser.parse_args()

    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    sequences = args.sequence or DEFAULT_SEQUENCES

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    source_config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    source_config_sha256 = file_sha256(config)
    checkpoint_sha256 = file_sha256(checkpoint)
    fixed800_manifest_sha256 = (
        file_sha256(args.fixed800_manifest.resolve()) if args.fixed800_manifest else None
    )

    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for sequence in sequences:
        ready, reason = mvsec_sequence_ready(sequence)
        if not ready:
            print(f"[skip] {sequence}: {reason}")
            skipped.append((sequence, reason))
            continue
        seq_dir = out_dir / sequence
        profile = seq_dir / "spike_profile.json"
        seq_config = seq_dir / "eval_config.yml"
        patch_config_sequence(config, sequence, seq_config, args.fixed800_manifest)
        identity = {
            "schema": "mvsec_sequence_evaluation_identity_v1",
            "sequence": sequence,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "source_config": str(config),
            "source_config_sha256": source_config_sha256,
            "eval_config_sha256": file_sha256(seq_config),
            "fixed800_manifest_sha256": fixed800_manifest_sha256,
        }
        identity_path = seq_dir / "evaluation_identity.json"
        reusable = False
        if profile.is_file() and identity_path.is_file():
            reusable = json.loads(identity_path.read_text(encoding="utf-8")) == identity
        if not reusable:
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
            if not profile.is_file():
                raise RuntimeError(f"MVSEC eval produced no profile for {sequence}: {profile}")
            audit_eval_load(seq_dir / "eval.log", source_config_data)
            identity_path.write_text(
                json.dumps(identity, indent=2) + "\n", encoding="utf-8"
            )
        else:
            audit_eval_load(seq_dir / "eval.log", source_config_data)
        rows.append({"sequence": sequence, **parse_profile(profile)})

    ranking = out_dir / "mvsec_ranking.md"
    with ranking.open("w", encoding="utf-8") as handle:
        handle.write("# MVSEC dt1 Ranking\n\n")
        handle.write("| sequence | samples | AEE | weighted AEE | valid pixels | PE1 | PE2 | legacy outlier | GT Fl(%) | total_spikes | firing | energy_uj |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['sequence']} | {row['samples']} | {row['AEE']:.4f} | "
                f"{row['weighted_aee']:.4f} | {row['valid_pixels']} | "
                f"{row['PE1']:.4f} | {row['PE2']:.4f} | {row['outlier']:.4f} | "
                f"{row['gt_fl_percent']:.4f} | "
                f"{row['spikes_g']:.4f}G | {row['firing'] * 100:.4f}% | {row['energy_uj']:.2f} |\n"
            )
        if skipped:
            handle.write("\nSkipped sequences:\n\n")
            for sequence, reason in skipped:
                handle.write(f"- `{sequence}`: {reason}\n")

    if not rows:
        raise RuntimeError("No MVSEC sequence was evaluated; check MVSEC_test preprocessing.")
    mean_aee = sum(r["AEE"] for r in rows) / len(rows)
    total_valid_pixels = sum(r["valid_pixels"] for r in rows)
    weighted_aee = (
        sum(r["weighted_aee"] * r["valid_pixels"] for r in rows) / total_valid_pixels
        if total_valid_pixels
        else float("nan")
    )
    summary = {
        "schema": "mvsec_evaluation_summary_v1",
        "protocol": "fixed800" if args.fixed800_manifest else "full_sequence",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "source_config_sha256": source_config_sha256,
        "fixed800_manifest": (
            str(args.fixed800_manifest.resolve()) if args.fixed800_manifest else None
        ),
        "fixed800_manifest_sha256": fixed800_manifest_sha256,
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "upstream_eval_sha256": file_sha256(
            REPO_ROOT / "third_party/SDformerFlow/eval_MV_flow_SNN.py"
        ),
        "sequences": rows,
        "skipped": [
            {"sequence": sequence, "reason": reason}
            for sequence, reason in skipped
        ],
        "mean_aee": mean_aee,
        "valid_pixel_weighted_aee": weighted_aee,
        "total_valid_pixels": total_valid_pixels,
    }
    (out_dir / "mvsec_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"MVSEC mean AEE={mean_aee:.4f}, valid-pixel-weighted AEE={weighted_aee:.4f} "
        f"over {len(rows)} evaluated sequences"
    )
    print(f"ranking={ranking}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
