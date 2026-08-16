#!/usr/bin/env python3
"""Bind the production Local-5 post-G0 profiler to the bb1e4 fullres winner."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run_local5_qfsa_profile_after_fullres as watcher
from evidence_provenance import validate_local5_atlif_provenance


HW_ROOT = Path(__file__).resolve().parents[1]
REPO = HW_ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RUN = EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
TRAINING_CONFIG = (
    EXP / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml"
)
TRAINING_IDENTITY = RUN / "training_config_identity.json"
TAG = "local5_fullres_bb1e4_postg0_profile100_20260805"


watcher.DEPLOY_STATUS = RUN / "status.log"
watcher.RUN_DIR = RUN
watcher.RANKING = RUN / "profile_ranking_valid825.md"
watcher.CONFIG = (
    EXP
    / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30_hardware_order_q7q17_deploy.yml"
)
watcher.OUTPUT = HW_ROOT / f"results/{TAG}"
watcher.REPLAY = HW_ROOT / "results/local5_fullres_bb1e4_postg0_replay_20260805"
watcher.DESCRIPTOR_ANALYSIS = HW_ROOT / "results/local5_fullres_bb1e4_descriptor_analysis_20260805"
watcher.ACCEPTANCE = HW_ROOT / "results/local5_fullres_bb1e4_postg0_acceptance_20260805"
watcher.STATUS = HW_ROOT / "results/local5_fullres_bb1e4_postg0_watcher_20260805.log"
watcher.LOCK = HW_ROOT / "results/local5_fullres_bb1e4_postg0_watcher_20260805.lock"
watcher.RUN_IDENTITY = watcher.OUTPUT / "post_g0_run_identity.json"
watcher.RELEASE_RECEIPT = watcher.OUTPUT / "post_g0_release_receipt.json"

ATLIF_VECTORS = HW_ROOT / "tb_hitflow/vectors/local5_bb1e4_checkpoint_atlif_postg0_20260805"
ATLIF_RESULTS = HW_ROOT / "results/local5_bb1e4_checkpoint_atlif_dptme_rtl_20260805"
ATLIF_LOCK = HW_ROOT / "results/local5_bb1e4_checkpoint_atlif_dptme_rtl_20260805.lock"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_identity() -> dict[str, object] | None:
    if not TRAINING_IDENTITY.is_file():
        return None
    value = json.loads(TRAINING_IDENTITY.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        return None
    state_path = Path(str(value.get("state_path", "")))
    checks = value.get("checks") or {}
    validations = {
        "schema": value.get("schema") == "local5_training_config_identity_v1",
        "authority": value.get("authority") == "ep9_optimizer_scheduler_state",
        "deterministic_config": value.get("deterministic_regeneration_equal") is True,
        "training_config_path": Path(str(value.get("config_path", ""))).resolve()
        == TRAINING_CONFIG.resolve(),
        "training_config_sha": value.get("config_sha256") == file_sha256(TRAINING_CONFIG),
        "state_exists": state_path.is_file(),
        "state_sha": state_path.is_file()
        and value.get("state_sha256") == file_sha256(state_path),
        "runtime_checks": bool(checks) and all(checks.values()),
    }
    failed = [name for name, passed in validations.items() if not passed]
    if failed:
        raise RuntimeError(f"Local-5 training identity failed: {failed}")
    return value


def wait_for_training_identity(poll_seconds: int = 60, timeout_hours: float = 48.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout_hours * 3600
    while time.monotonic() < deadline:
        value = validate_training_identity()
        if value is not None:
            watcher.record(
                "RELEASE Local-5 ep9 runtime config identity sha="
                + file_sha256(TRAINING_IDENTITY)
            )
            return value
        if TRAINING_IDENTITY.is_file():
            pending = json.loads(TRAINING_IDENTITY.read_text(encoding="utf-8"))
            if str(pending.get("status", "")).startswith("FAIL"):
                raise RuntimeError(
                    f"Local-5 training identity is fail-closed: {pending.get('status')}"
                )
        watcher.record("WAIT Local-5 ep9 runtime config identity PASS")
        time.sleep(poll_seconds)
    raise TimeoutError("Local-5 training identity did not pass")


_source_binding_paths = watcher.source_binding_paths


def source_binding_paths_with_training_identity() -> dict[str, Path]:
    paths = _source_binding_paths()
    paths["training_config_identity"] = TRAINING_IDENTITY
    return paths


watcher.source_binding_paths = source_binding_paths_with_training_identity


def run_atlif_checkpoint_replay() -> None:
    identity = json.loads(watcher.RUN_IDENTITY.read_text(encoding="utf-8"))
    ATLIF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with ATLIF_LOCK.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        report_path = ATLIF_RESULTS / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_identity = report.get("checkpoint_identity", {})
            try:
                validate_local5_atlif_provenance(report)
            except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError):
                pass
            else:
                if (
                    report_identity.get("checkpoint_sha256")
                    == identity.get("checkpoint_sha256")
                    and report_identity.get("config_sha256")
                    == identity.get("config_sha256")
                ):
                    return
        command = [
            PYTHON,
            str(HW_ROOT / "scripts/generate_checkpoint_atlif_dptme_vectors.py"),
            "--config",
            str(identity["config"]),
            "--checkpoint",
            str(identity["checkpoint"]),
            "--output-dir",
            str(ATLIF_VECTORS),
            "--sample-index",
            "0",
        ]
        env = os.environ.copy()
        env.update({
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        subprocess.run(command, cwd=REPO, env=env, check=True)
        env.update({"VECTOR_DIR": str(ATLIF_VECTORS), "RESULT_DIR": str(ATLIF_RESULTS)})
        subprocess.run(
            ["bash", str(HW_ROOT / "sim_hitflow/run_checkpoint_atlif_dptme_checks.sh")],
            cwd=REPO,
            env=env,
            check=True,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_identity = report.get("checkpoint_identity", {})
        if (
            report.get("status") != "PASS"
            or report_identity.get("checkpoint_sha256")
            != identity.get("checkpoint_sha256")
            or report_identity.get("config_sha256") != identity.get("config_sha256")
        ):
            raise RuntimeError("Local-5 ATLIF RTL report is not bound to post-G0 rank-1")


if __name__ == "__main__":
    wait_for_training_identity()
    sys.argv = [sys.argv[0], "--samples", "100", "--ordered-groups", "4", "--poll-seconds", "60"]
    result = watcher.main()
    if result:
        raise SystemExit(result)
    # watcher.main() returns 0 both on full success and on "lock already held".
    # Only produce ATLIF vectors after this process owns a finished acceptance.
    acceptance_json = watcher.ACCEPTANCE / "acceptance.json"
    if not acceptance_json.is_file():
        watcher.record(
            "SKIP ATLIF replay: acceptance not produced by this process "
            "(likely lock held by another post-G0 producer)"
        )
        raise SystemExit(0)
    run_atlif_checkpoint_replay()
    raise SystemExit(0)
