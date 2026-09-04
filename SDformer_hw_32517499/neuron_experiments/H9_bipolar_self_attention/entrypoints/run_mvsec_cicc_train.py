#!/usr/bin/env python3
"""Run one direct-MVSEC training config with local-only checkpoints and logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = REPO_ROOT / "third_party/SDformerFlow"
TRAINER = UPSTREAM / "train_mdr_supervised_SNN.py"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prev-runid", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(TRAINER), "--config", str(config)]
    if args.prev_runid:
        command.extend(["--prev_runid", str(args.prev_runid.resolve())])
    if args.resume:
        command.extend(["--resume", "1"])

    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_MDR_USE_MLFLOW": "0",
            "SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG": "1",
            "SDFORMER_MDR_SKIP_MLFLOW_STATE_LOG": "1",
            "SDFORMER_MDR_LOCAL_CHECKPOINT_DIR": str(output_dir),
            "SDFORMER_MDR_VOXEL_GPU": env.get("SDFORMER_MDR_VOXEL_GPU", "0"),
            "SDFORMER_MDR_DETECT_ANOMALY": env.get("SDFORMER_MDR_DETECT_ANOMALY", "0"),
            "SDFORMER_SNN_BACKEND": env.get("SDFORMER_SNN_BACKEND", "cupy"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    provenance = {
        "schema": "mvsec_cicc_direct_train_launch_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config),
        "output_dir": str(output_dir),
        "prev_runid": str(args.prev_runid.resolve()) if args.prev_runid else None,
        "resume": args.resume,
        "command": command,
        "source_sha256": {
            "config": file_sha256(config),
            "trainer": file_sha256(TRAINER),
            "mvsec_loader": file_sha256(UPSTREAM / "MDR_dataloader/MVSEC.py"),
            "mvsec_protocol": file_sha256(UPSTREAM / "MDR_dataloader/mvsec_protocol.py"),
        },
        "environment": {
            key: env[key]
            for key in (
                "SDFORMER_MDR_USE_MLFLOW",
                "SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG",
                "SDFORMER_MDR_SKIP_MLFLOW_STATE_LOG",
                "SDFORMER_MDR_LOCAL_CHECKPOINT_DIR",
                "SDFORMER_MDR_VOXEL_GPU",
                "SDFORMER_MDR_DETECT_ANOMALY",
                "SDFORMER_SNN_BACKEND",
            )
        },
    }
    import yaml

    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    manifest_value = config_data.get("data", {}).get("mvsec_split_manifest")
    if manifest_value:
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        manifest_path = manifest_path.resolve()
        provenance["manifest"] = {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        }
    (output_dir / "launch_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "train.log").open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=UPSTREAM,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(f"\n[mvsec-cicc-train] exit_code={result.returncode}\n")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
