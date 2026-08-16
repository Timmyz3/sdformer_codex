#!/usr/bin/env python3
"""Profile the final H67 full-resolution checkpoint after Local-5 releases the GPU."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from evidence_provenance import validate_projection_provenance


HW_ROOT = Path(__file__).resolve().parents[1]
REPO = HW_ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
LOCAL5_RUN = (
    EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
)
LOCAL5_RANKING = LOCAL5_RUN / "profile_ranking_valid825.md"
LOCAL5_HARDWARE_CONFIG = EXP / (
    "configs/generated/"
    "dsec_fullres_w15_H66d_local5_bb1e4_ft30_hardware_order_q7q17_deploy.yml"
)
LOCAL5_RTL = HW_ROOT / (
    "results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/"
    "checkpoint_bound_scope.json"
)
LOCAL5_RTL_STATUS = (
    HW_ROOT / "results/local5_bb1e4_checkpoint_bound_rtl_watcher_20260805.log"
)
LOCAL5_RTL_COMPLETE = "ALL COMPLETE Local-5 bb1e4 checkpoint-bound component RTL exact"
LOCAL5_RTL_COMPLETE_MARKERS = (
    LOCAL5_RTL_COMPLETE,
    "ALL COMPLETE checkpoint-bound Local-5 score/Shiftmax, projection partial "
    "accumulator, and ATLIF temporal-matrix component RTL exact",
)
LOCAL5_PROJECTION_WEIGHT_MODE = "checkpoint_theta_folded_dyadic_int8_head_slice"
CONFIG = (
    EXP
    / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep30_hardware_order_q7q17_deploy.yml"
)
CHECKPOINT = (
    EXP
    / "results/dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch30.pth"
)
PROFILE = HW_ROOT / "results/h67_fullres_ep30_t450_profile100_20260805"
TRACE = HW_ROOT / "results/h67_fullres_ep30_t450_all12_bit_trace_20260805"
AUDIT = HW_ROOT / "results/h67_fullres_ep30_t450_all12_bit_trace_audit_20260805"
RTL_VECTORS = HW_ROOT / "tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805"
RTL_RESULT = HW_ROOT / "results/h67_fullres_ep30_t450_score_shiftmax_rtl_20260805"
ATLIF_VECTORS = HW_ROOT / "tb_hitflow/vectors/h67_ep30_checkpoint_atlif_20260805"
ATLIF_RESULT = HW_ROOT / "results/h67_ep30_checkpoint_atlif_dptme_rtl_20260805"
PROJECTION_RESULT = HW_ROOT / "results/h67_ep30_checkpoint_projection_rtl_20260805"
STATUS = HW_ROOT / "results/h67_fullres_ep30_t450_profile_watcher_20260805.log"
LOCK = HW_ROOT / "results/h67_fullres_ep30_t450_profile_watcher_20260805.lock"
GPU_LEASE = HW_ROOT / "results/gpu_profile_lease.lock"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"
MAX_GPU_USED_MIB = 8192
PROJECTION_RUNNER = HW_ROOT / "sim_hitflow/run_gatestack_dctf96_real_trace_checks.sh"
COMPLETE_MARKER = (
    "ALL COMPLETE H67 ep30 fullres T450 profile100/all12 trace audit/"
    "score-Shiftmax, ATLIF DP-TME, and real-weight projection component RTL"
)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(
    command: list[str],
    label: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> None:
    record(f"START {label}: {' '.join(command)}")
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(command, cwd=REPO, env=env)
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed")


def gpu_used_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_evidence_matches_checkpoint() -> bool:
    profile_path = PROFILE / "nts11_hardware_p0_profile.json"
    manifest_path = TRACE / "manifest.json"
    audit_path = AUDIT / "audit.json"
    score_path = RTL_RESULT / "report.json"
    atlif_path = ATLIF_RESULT / "report.json"
    projection_path = PROJECTION_RESULT / "report.json"
    if not all(
        path.is_file()
        for path in (
            profile_path,
            manifest_path,
            audit_path,
            score_path,
            atlif_path,
            projection_path,
            CONFIG,
            CHECKPOINT,
        )
    ):
        return False
    try:
        score = json.loads(score_path.read_text(encoding="utf-8"))
        atlif = json.loads(atlif_path.read_text(encoding="utf-8"))
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    try:
        validate_projection_provenance(projection)
    except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    checkpoint_sha = sha256(CHECKPOINT)
    profile_identity = profile.get("artifact_identity", {})
    profile_protocol = profile.get("eval_protocol", {})
    profile_counts = profile.get("module_counts", {})
    return (
        trace_matches_checkpoint()
        and profile_identity.get("checkpoint_sha256") == checkpoint_sha
        and profile_identity.get("config_sha256") == sha256(CONFIG)
        and int(profile.get("samples", 0)) == 100
        and profile_protocol.get("resolution") == [480, 640]
        and profile_protocol.get("crop") is None
        and profile_protocol.get("window_size") == [2, 15, 15]
        and int(profile_protocol.get("tokens_per_window", 0)) == 450
        and profile_counts.get("ATLIFTernaryPSN") == 105
        and profile_counts.get("ShiftmaxAttention") == 12
        and int(profile.get("bit_trace_records", 0)) == 12
        and audit.get("status") == "PASS"
        and Path(str(audit.get("source_manifest", ""))).resolve()
        == manifest_path.resolve()
        and audit.get("coverage", {}).get("four_stage_complete") is True
        and audit.get("coverage", {}).get("stages") == [0, 1, 2, 3]
        and len(audit.get("records") or []) == 12
        and all(record.get("sha256_ok") is True for record in audit.get("records") or [])
        and score.get("status") == "PASS"
        and "component_rtl_exact" in str(score.get("scope", ""))
        and score.get("run_context", {}).get("artifact_identity", {}).get(
            "checkpoint_sha256"
        )
        == checkpoint_sha
        and score.get("run_context", {}).get("artifact_identity", {}).get(
            "config_sha256"
        )
        == sha256(CONFIG)
        and atlif.get("status") == "PASS"
        and atlif.get("checkpoint_identity", {}).get("checkpoint_sha256")
        == checkpoint_sha
        and atlif.get("checkpoint_identity", {}).get("config_sha256")
        == sha256(CONFIG)
        and projection.get("status") == "PASS"
        and "projection_component_rtl_exact" in str(projection.get("scope", ""))
        and projection.get("checkpoint_identity", {}).get("checkpoint_sha256")
        == checkpoint_sha
        and projection.get("checkpoint_identity", {}).get("config_sha256")
        == sha256(CONFIG)
        and int(projection.get("record_count", 0)) == 12
        and projection.get("required_stage_coverage") == [0, 1, 2, 3]
        and int(projection.get("temporal_tokens", 0)) == 450
        and int(projection.get("token_id_width", 0)) == 9
    )


def trace_matches_checkpoint() -> bool:
    manifest_path = TRACE / "manifest.json"
    if not manifest_path.is_file() or not CONFIG.is_file() or not CHECKPOINT.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("records") or []
        identity = manifest.get("run_context", {}).get("artifact_identity", {})
        return (
            identity.get("checkpoint_sha256") == sha256(CHECKPOINT)
            and identity.get("config_sha256") == sha256(CONFIG)
            and len(records) == 12
            and {int(record.get("temporal_tokens", 0)) for record in records} == {450}
            and all(
                Path(str(record.get("file", ""))).is_file()
                and record.get("sha256") == sha256(Path(str(record["file"])))
                for record in records
            )
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return False


def profile_matches_checkpoint() -> bool:
    profile_path = PROFILE / "nts11_hardware_p0_profile.json"
    if not profile_path.is_file() or not CONFIG.is_file() or not CHECKPOINT.is_file():
        return False
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        identity = profile.get("artifact_identity", {})
        protocol = profile.get("eval_protocol", {})
        counts = profile.get("module_counts", {})
        return (
            identity.get("checkpoint_sha256") == sha256(CHECKPOINT)
            and identity.get("config_sha256") == sha256(CONFIG)
            and int(profile.get("samples", 0)) == 100
            and int(profile.get("bit_trace_records", 0)) == 12
            and protocol.get("resolution") == [480, 640]
            and protocol.get("crop") is None
            and protocol.get("window_size") == [2, 15, 15]
            and int(protocol.get("tokens_per_window", 0)) == 450
            and counts.get("ATLIFTernaryPSN") == 105
            and counts.get("ShiftmaxAttention") == 12
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def local5_rtl_evidence_complete() -> bool:
    required = (
        LOCAL5_RANKING,
        LOCAL5_HARDWARE_CONFIG,
        LOCAL5_RTL,
        LOCAL5_RTL_STATUS,
    )
    if not all(path.is_file() for path in required):
        return False
    status_text = LOCAL5_RTL_STATUS.read_text(encoding="utf-8", errors="replace")
    if not any(marker in status_text for marker in LOCAL5_RTL_COMPLETE_MARKERS):
        return False
    rank1_epoch = None
    for line in LOCAL5_RANKING.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            rank1_epoch = int(match.group(1))
            break
    if rank1_epoch is None:
        return False
    checkpoint = LOCAL5_RUN / f"checkpoint_epoch{rank1_epoch}.pth"
    if not checkpoint.is_file():
        return False
    try:
        result = json.loads(LOCAL5_RTL.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    identity = result.get("checkpoint_identity") or {}
    score = result.get("score_shiftmax") or {}
    projection = result.get("projection") or {}
    projection_verification = projection.get("verification") or {}
    atlif = result.get("atlif_temporal_matrix") or {}
    return (
        result.get("status") == "PASS"
        and "component_rtl_exact" in str(result.get("evidence_scope", ""))
        and "not_full_network" in str(result.get("evidence_scope", ""))
        and int(identity.get("best_epoch", -1)) == rank1_epoch
        and Path(str(identity.get("checkpoint", ""))).resolve()
        == checkpoint.resolve()
        and identity.get("checkpoint_sha256") == sha256(checkpoint)
        and Path(str(identity.get("config", ""))).resolve()
        == LOCAL5_HARDWARE_CONFIG.resolve()
        and identity.get("config_sha256") == sha256(LOCAL5_HARDWARE_CONFIG)
        and score.get("status") == "PASS"
        and all(
            projection_verification.get(name) == "PASS"
            for name in (
                "checkpoint_weight_binding",
                "random_sva",
                "verilator_lint",
                "yosys_check",
            )
        )
        and projection.get("weight_mode") == LOCAL5_PROJECTION_WEIGHT_MODE
        and atlif.get("status") == "PASS"
    )


def wait_for_release() -> None:
    while not local5_rtl_evidence_complete():
        record("WAIT Local-5 checkpoint-bound RTL release")
        time.sleep(300)
    while True:
        used = gpu_used_mib()
        if used <= MAX_GPU_USED_MIB:
            record(f"RELEASE GPU memory_used={used}MiB")
            return
        record(f"WAIT GPU memory_used={used}MiB > {MAX_GPU_USED_MIB}MiB")
        time.sleep(300)


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("EXIT another H67 ep30 profile watcher owns the lock")
            return 0
        if completed_evidence_matches_checkpoint():
            record("REUSE completed H67 ep30 T450 profile/trace/RTL evidence")
            record(COMPLETE_MARKER)
            return 0
        if not CONFIG.is_file() or not CHECKPOINT.is_file():
            raise FileNotFoundError(f"missing config/checkpoint: {CONFIG} {CHECKPOINT}")
        wait_for_release()
        if not (trace_matches_checkpoint() and profile_matches_checkpoint()):
            GPU_LEASE.parent.mkdir(parents=True, exist_ok=True)
            with GPU_LEASE.open("a", encoding="utf-8") as gpu_handle:
                fcntl.flock(gpu_handle, fcntl.LOCK_EX)
                run(
                    [
                        PYTHON,
                        "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py",
                        "--config",
                        str(CONFIG),
                        "--checkpoint",
                        str(CHECKPOINT),
                        "--output-dir",
                        str(PROFILE),
                        "--samples",
                        "100",
                        "--num-workers",
                        "0",
                        "--ordered-trace",
                        "--bit-trace-dir",
                        str(TRACE),
                        "--bit-trace-samples",
                        "1",
                        "--bit-trace-windows",
                        "1",
                        "--bit-trace-all-blocks",
                    ],
                    "H67 ep30 fullres T450 profile100 and all12 real bit trace",
                )
        run(
            [
                PYTHON,
                "hw_autoresearch_nts07/scripts/audit_h67_bit_trace.py",
                "--manifest",
                str(TRACE / "manifest.json"),
                "--output-dir",
                str(AUDIT),
                "--require-four-stages",
                "--require-records",
                "12",
            ],
            "H67 ep30 all12 real bit trace audit",
        )
        run(
            ["bash", "hw_autoresearch_nts07/sim_h67/run_h67_checkpoint_row_trace_checks.sh"],
            "H67 ep30 checkpoint-bound T450 score/Shiftmax RTL",
            extra_env={
                "TRACE_MANIFEST": str(TRACE / "manifest.json"),
                "VECTOR_DIR": str(RTL_VECTORS),
                "RESULT_DIR": str(RTL_RESULT),
                "PYTHON": PYTHON,
            },
        )
        run(
            [
                PYTHON,
                "hw_autoresearch_nts07/scripts/generate_checkpoint_atlif_dptme_vectors.py",
                "--config",
                str(CONFIG),
                "--checkpoint",
                str(CHECKPOINT),
                "--output-dir",
                str(ATLIF_VECTORS),
                "--sample-index",
                "0",
            ],
            "H67 ep30 checkpoint-bound ATLIF fixed-point vectors",
        )
        run(
            ["bash", "hw_autoresearch_nts07/sim_hitflow/run_checkpoint_atlif_dptme_checks.sh"],
            "H67 ep30 checkpoint-bound ATLIF DP-TME RTL",
            extra_env={"VECTOR_DIR": str(ATLIF_VECTORS), "RESULT_DIR": str(ATLIF_RESULT)},
        )
        run(
            ["bash", str(PROJECTION_RUNNER)],
            "H67 ep30 checkpoint-bound all12 real-weight projection RTL",
            extra_env={
                "SOURCE_MANIFEST": str(TRACE / "manifest.json"),
                "RESULT_DIR": str(PROJECTION_RESULT),
                "PYTHON": PYTHON,
            },
        )
        if not completed_evidence_matches_checkpoint():
            raise RuntimeError("H67 ep30 component RTL evidence is incomplete or stale")
        record(COMPLETE_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
