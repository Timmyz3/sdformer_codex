#!/usr/bin/env python3
"""Run the standard valid825 evaluation for the fixed D1/H87 ep5 checkpoint."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
RUN = EXP / "results/dsec_fullres_w15_H87_motion_t5_quotient_ft5_short_20260818"
OUT = RUN / "standard_valid825" / "epoch5"
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H87_motion_t5_quotient_ft5_short_20260818.yml"
CHECKPOINT = RUN / "checkpoint_epoch5.pth"
OPERATOR = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"
CONTRACT = REPO / "neuron_autoresearch/D1_MOTION_T5_IMPLEMENTATION_20260818.json"
PYTHON = Path("/opt/conda/envs/sdformerflow/bin/python")
EVALUATOR = REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
LOCK = Path("/tmp/sdformer_h87_ep5_valid825.lock")

EXPECTED_SHA256 = {
    OPERATOR: "0f77f66dbd331daa77a284199cda33125a1959a005b6f4d592e2e6cda5317187",
    CONFIG: "cee9684bba4b5aa6dc2b3bb8b8e6ac3989c5a9c52f50c8d7fdb21316edf69ec8",
    CHECKPOINT: "1a1c2e31154ad0a6fabeca78f7967fa05df15f33bda3365e047e6d7bd36dd5f0",
    CONTRACT: "5e20ed3d18e179df07cede358195e6101ee2c31116f463672311aacca56c5eb7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "valid825_ep5_watcher.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def audit_profile(profile_path: Path) -> dict[str, object]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    load = profile.get("checkpoint_load_audit", {})
    modules = profile.get("module_counts", {})
    checks = {
        "samples_825": profile.get("samples") == 825,
        "atlif_105": modules.get("ATLIFTernaryPSN") == 105,
        "shiftmax_12": modules.get("ShiftmaxAttention") == 12,
        "overlay_keys_210": load.get("checkpoint_overlay_keys") == 210
        and load.get("model_overlay_keys") == 210,
        "missing_zero": load.get("missing_count") == 0
        and load.get("overlay_missing_count") == 0,
        "unexpected_zero": load.get("unexpected_count") == 0
        and load.get("overlay_unexpected_count") == 0,
        "checkpoint_identity": load.get("checkpoint") == str(CHECKPOINT),
    }
    metrics = profile.get("metrics", {})
    return {
        "schema": "h87_ep5_valid825_acceptance_v1",
        "timestamp_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
        "total_spikes": profile.get("total_spikes"),
        "energy_uj": profile.get("energy_uj"),
        "profile_sha256": sha256(profile_path),
    }


def main() -> int:
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("REFUSED: another H87 ep5 valid825 launcher owns the lock")
            return 2

        for path, expected in EXPECTED_SHA256.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"identity drift: {path}: {actual} != {expected}")

        OUT.mkdir(parents=True, exist_ok=True)
        profile_path = OUT / "spike_profile.json"
        acceptance_path = OUT / "acceptance.json"
        if profile_path.is_file():
            acceptance = audit_profile(profile_path)
            acceptance_path.write_text(
                json.dumps(acceptance, indent=2) + "\n", encoding="utf-8"
            )
            record(f"existing profile audited: {acceptance['status']}")
            return 0 if acceptance["status"] == "PASS" else 3

        command = [
            str(PYTHON),
            "-u",
            str(EVALUATOR.relative_to(REPO)),
            "--config",
            str(CONFIG),
            "--checkpoint",
            str(CHECKPOINT),
            "--path_results",
            str(OUT),
            "--mode",
            "valid",
        ]
        provenance = {
            "schema": "h87_ep5_valid825_provenance_v1",
            "timestamp_utc": utc_now(),
            "claim_boundary": "standard valid825 algorithm evidence; no RTL or PPA claim",
            "expected_sha256": {str(path): value for path, value in EXPECTED_SHA256.items()},
            "command": command,
        }
        (OUT / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )

        env = os.environ.copy()
        env["SDFORMER_USE_MLFLOW"] = "0"
        env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
        env["SDFORMER_SNN_BACKEND"] = "cupy"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        record("launching standard valid825")
        with (OUT / "eval.log").open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            process = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.write(f"\n[h87-valid825-ep5] exit_code={process.returncode}\n")

        if process.returncode != 0:
            record(f"evaluation failed: exit_code={process.returncode}")
            return process.returncode
        if not profile_path.is_file():
            record("evaluation failed: spike_profile.json missing")
            return 4

        acceptance = audit_profile(profile_path)
        acceptance_path.write_text(
            json.dumps(acceptance, indent=2) + "\n", encoding="utf-8"
        )
        record(f"evaluation complete: {acceptance['status']}")
        return 0 if acceptance["status"] == "PASS" else 5


if __name__ == "__main__":
    sys.exit(main())
