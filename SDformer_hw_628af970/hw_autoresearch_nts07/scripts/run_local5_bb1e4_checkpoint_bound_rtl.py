#!/usr/bin/env python3
"""Run checkpoint-bound Local-5 projection RTL after the new profile passes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from .evidence_provenance import (
        validate_local5_atlif_provenance,
        validate_local5_projection_provenance,
    )
except ImportError:  # Direct script execution.
    from evidence_provenance import (
        validate_local5_atlif_provenance,
        validate_local5_projection_provenance,
    )


HW_ROOT = Path(__file__).resolve().parents[1]
REPO = HW_ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RUN = EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
RANKING = RUN / "profile_ranking_valid825.md"
TRAINING_CONFIG = EXP / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml"
HARDWARE_CONFIG = EXP / (
    "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30_"
    "hardware_order_q7q17_deploy.yml"
)
TRAINING_IDENTITY = RUN / "training_config_identity.json"
PROFILE = HW_ROOT / "results/local5_fullres_bb1e4_postg0_profile100_20260805"
ACCEPTANCE = (
    HW_ROOT / "results/local5_fullres_bb1e4_postg0_acceptance_20260805/acceptance.json"
)
VECTORS = HW_ROOT / "tb_qfit/vectors/local5_bb1e4_active_projection_postg0_100"
RESULTS = HW_ROOT / "results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805"
SCORE_VECTORS = HW_ROOT / "tb_local5/vectors/local5_bb1e4_checkpoint_score_postg0_100"
SCORE_RESULTS = HW_ROOT / "results/local5_bb1e4_checkpoint_score_shiftmax_rtl_20260805"
RUN_IDENTITY = PROFILE / "post_g0_run_identity.json"
ATLIF_VECTORS = HW_ROOT / "tb_hitflow/vectors/local5_bb1e4_checkpoint_atlif_postg0_20260805"
ATLIF_RESULTS = HW_ROOT / "results/local5_bb1e4_checkpoint_atlif_dptme_rtl_20260805"
ATLIF_LOCK = HW_ROOT / "results/local5_bb1e4_checkpoint_atlif_dptme_rtl_20260805.lock"
STATUS = HW_ROOT / "results/local5_bb1e4_checkpoint_bound_rtl_watcher_20260805.log"
PROFILE_WRAPPER = HW_ROOT / "scripts/run_local5_bb1e4_postg0_profile.py"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"


def file_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def projection_source_bindings(
    source_manifest: Path, vector_manifest: Path
) -> list[dict[str, object]]:
    bindings = []
    for line in source_manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        path = (HW_ROOT / relative.lstrip("* ")).resolve()
        binding = file_binding(path)
        if binding["sha256"] != digest:
            raise RuntimeError(f"projection source manifest SHA mismatch: {path}")
        if path == vector_manifest.resolve():
            continue
        bindings.append(binding)
    if len(bindings) != 20:
        raise RuntimeError(f"expected 20 projection RTL/SVA sources, got {len(bindings)}")
    return bindings


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank1_checkpoint() -> tuple[int, Path]:
    for line in RANKING.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            epoch = int(match.group(1))
            checkpoint = RUN / f"checkpoint_epoch{epoch}.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            return epoch, checkpoint
    raise RuntimeError(f"cannot parse Local-5 rank-1 checkpoint: {RANKING}")


def validate_training_identity_binding() -> dict:
    value = json.loads(TRAINING_IDENTITY.read_text(encoding="utf-8"))
    state_path = Path(str(value.get("state_path", "")))
    checks = value.get("checks") or {}
    validations = {
        "status": value.get("status") == "PASS",
        "schema": value.get("schema") == "local5_training_config_identity_v1",
        "authority": value.get("authority") == "ep9_optimizer_scheduler_state",
        "training_config_path": Path(str(value.get("config_path", ""))).resolve()
        == TRAINING_CONFIG.resolve(),
        "training_config_sha": value.get("config_sha256") == file_sha256(TRAINING_CONFIG),
        "state_sha": state_path.is_file()
        and value.get("state_sha256") == file_sha256(state_path),
        "runtime_checks": bool(checks) and all(checks.values()),
    }
    failed = [name for name, passed in validations.items() if not passed]
    if failed:
        raise RuntimeError(f"Local-5 training identity binding failed: {failed}")
    run_identity = json.loads(RUN_IDENTITY.read_text(encoding="utf-8"))
    binding = (run_identity.get("source_bindings") or {}).get(
        "training_config_identity"
    ) or {}
    if (
        Path(str(binding.get("path", ""))).resolve() != TRAINING_IDENTITY.resolve()
        or binding.get("sha256") != file_sha256(TRAINING_IDENTITY)
    ):
        raise RuntimeError("post-G0 run identity is not bound to training config identity")
    return value


def validate_profile_acceptance_binding() -> tuple[dict, Path]:
    epoch, checkpoint = rank1_checkpoint()
    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    identity_path = Path(str(acceptance.get("run_identity", "")))
    manifest_path = Path(str(acceptance.get("manifest", "")))
    if not identity_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("post-G0 acceptance identity/manifest is missing")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_checks = (
        "loader_provenance",
        "formal_qualification",
        "relation_rtl_binding",
        "descriptor_geometry",
        "replay_binding",
        "descriptor_report_binding",
        "reports_recomputed_equal",
        "source_software_binding",
        "release_receipt_binding",
        "checkpoint_projection_weight_binding",
        "checkpoint_projection_payload_recomputed",
        "checkpoint_projection_topology_abi",
        "threshold_training_deployment_semantics",
    )
    checkpoint_sha = file_sha256(checkpoint)
    checks = acceptance.get("checks") or {}
    validations = {
        "schema": acceptance.get("schema") == "local5_post_g0_acceptance_v1",
        "accepted": acceptance.get("accepted") is True,
        "samples": int(acceptance.get("samples", 0)) == 100,
        "blocks": int(acceptance.get("blocks", 0)) == 12,
        "checks": all(checks.get(name) is True for name in required_checks),
        "acceptance manifest path": manifest_path.resolve()
        == Path(str(acceptance.get("manifest", ""))).resolve(),
        "acceptance manifest SHA": acceptance.get("manifest_sha256")
        == file_sha256(manifest_path),
        "acceptance identity path": identity_path.resolve() == RUN_IDENTITY.resolve(),
        "acceptance identity SHA": acceptance.get("run_identity_sha256")
        == file_sha256(identity_path),
        "identity epoch": int(identity.get("best_epoch", -1)) == epoch,
        "identity checkpoint path": Path(str(identity.get("checkpoint", ""))).resolve()
        == checkpoint.resolve(),
        "identity checkpoint SHA": identity.get("checkpoint_sha256") == checkpoint_sha,
        "identity config path": Path(str(identity.get("config", ""))).resolve()
        == HARDWARE_CONFIG.resolve(),
        "identity config exists": HARDWARE_CONFIG.is_file(),
        "identity config SHA": HARDWARE_CONFIG.is_file()
        and identity.get("config_sha256") == file_sha256(HARDWARE_CONFIG),
        "manifest checkpoint path": Path(str(manifest.get("checkpoint", ""))).resolve()
        == checkpoint.resolve(),
        "manifest checkpoint SHA": manifest.get("checkpoint_sha256") == checkpoint_sha,
        "manifest identity SHA": manifest.get("run_identity_file_sha256")
        == file_sha256(identity_path),
    }
    failed = [name for name, passed in validations.items() if not passed]
    if failed:
        raise RuntimeError(f"post-G0 acceptance/rank-1 binding failed: {failed}")
    validate_training_identity_binding()
    return identity, checkpoint


def ordered_manifest_identity(vector_manifest: dict) -> dict[str, str]:
    source = Path(str(vector_manifest.get("source_manifest", "")))
    if not source.is_file() or vector_manifest.get("source_manifest_sha256") != file_sha256(source):
        raise RuntimeError("ordered source manifest binding failed")
    source_raw = json.loads(source.read_text(encoding="utf-8"))
    checkpoint_sha = str(source_raw.get("checkpoint_sha256", ""))
    config_sha = str(source_raw.get("config_sha256", ""))
    if len(checkpoint_sha) != 64 or len(config_sha) != 64:
        raise RuntimeError("ordered source manifest lacks checkpoint/config SHA")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
    }


def ordered_manifest_checkpoint_sha(vector_manifest: dict) -> str:
    return ordered_manifest_identity(vector_manifest)["checkpoint_sha256"]


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], label: str, env: dict[str, str] | None = None) -> None:
    record(f"START {label}: {' '.join(command)}")
    result = subprocess.run(command, cwd=HW_ROOT, env=env)
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed")


def run_atlif_checkpoint_replay() -> dict:
    identity = json.loads(RUN_IDENTITY.read_text(encoding="utf-8"))
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
                    record(
                        "REUSE provenance-valid checkpoint-bound ATLIF DP-TME report "
                        "for identical rank-1 SHA"
                    )
                    return report
        env = os.environ.copy()
        env.update(
            {
                "SDFORMER_USE_MLFLOW": "0",
                "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
                "SDFORMER_SNN_BACKEND": "cupy",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        run(
            [
                PYTHON,
                "scripts/generate_checkpoint_atlif_dptme_vectors.py",
                "--config",
                str(identity["config"]),
                "--checkpoint",
                str(identity["checkpoint"]),
                "--output-dir",
                str(ATLIF_VECTORS),
                "--sample-index",
                "0",
            ],
            "generate checkpoint-bound ATLIF fixed-point vectors",
            env=env,
        )
        env.update({"VECTOR_DIR": str(ATLIF_VECTORS), "RESULT_DIR": str(ATLIF_RESULTS)})
        run(
            ["bash", "sim_hitflow/run_checkpoint_atlif_dptme_checks.sh"],
            "checkpoint-bound ATLIF DP-TME RTL/SVA/lint/synthesis",
            env=env,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_identity = report.get("checkpoint_identity", {})
        if (
            report.get("status") != "PASS"
            or report_identity.get("checkpoint_sha256")
            != identity.get("checkpoint_sha256")
            or report_identity.get("config_sha256") != identity.get("config_sha256")
        ):
            raise RuntimeError("ATLIF RTL report is not bound to Local-5 rank-1 checkpoint")
        return report


def ensure_profile_acceptance(
    *,
    poll_seconds: int = 120,
    timeout_hours: float = 48.0,
) -> None:
    """Wait for post-G0 acceptance; start producer only if none is active."""
    import time

    deadline = time.monotonic() + timeout_hours * 3600
    launched = False
    while time.monotonic() < deadline:
        try:
            validate_profile_acceptance_binding()
            return
        except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            # Prefer joining an already-running producer over racing a second one.
            lock_path = (
                HW_ROOT
                / "results/local5_fullres_bb1e4_postg0_watcher_20260805.lock"
            )
            lock_held = False
            if lock_path.is_file():
                try:
                    with lock_path.open("a", encoding="utf-8") as handle:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(handle, fcntl.LOCK_UN)
                except BlockingIOError:
                    lock_held = True
            if lock_held:
                record(
                    "WAIT Local-5 bb1e4 post-G0 acceptance "
                    f"(producer lock held; last={type(exc).__name__})"
                )
                time.sleep(max(30, poll_seconds))
                continue
            if not launched:
                env = os.environ.copy()
                env.update(
                    {
                        "SDFORMER_USE_MLFLOW": "0",
                        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
                        "SDFORMER_SNN_BACKEND": "cupy",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    }
                )
                # One non-blocking start attempt; subsequent loops only wait.
                try:
                    run(
                        [PYTHON, "-u", str(PROFILE_WRAPPER)],
                        "supervised Local-5 post-G0 profile/replay/acceptance producer",
                        env=env,
                    )
                except RuntimeError as run_exc:
                    record(f"WAIT post-G0 producer start deferred: {run_exc}")
                launched = True
                try:
                    validate_profile_acceptance_binding()
                    return
                except (
                    FileNotFoundError,
                    json.JSONDecodeError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    pass
            record(
                "WAIT Local-5 bb1e4 post-G0 acceptance "
                f"(last={type(exc).__name__})"
            )
            time.sleep(max(30, poll_seconds))
    raise TimeoutError("Local-5 post-G0 acceptance did not become available")


def main() -> int:
    ensure_profile_acceptance()
    identity, checkpoint = validate_profile_acceptance_binding()
    training_identity = validate_training_identity_binding()
    checkpoint_sha = file_sha256(checkpoint)
    record("RELEASE Local-5 bb1e4 accepted profile")

    # Serialize the only post-acceptance producer shared with the profile wrapper.
    # The ATLIF file lock makes either process produce once while the other reuses it.
    atlif_result = run_atlif_checkpoint_replay()
    atlif_identity = atlif_result.get("checkpoint_identity", {})
    if (
        atlif_identity.get("checkpoint_sha256") != checkpoint_sha
        or atlif_identity.get("config_sha256") != identity.get("config_sha256")
    ):
        raise RuntimeError("ATLIF RTL report is not bound to Local-5 rank-1")
    record("RELEASE serialized ATLIF replay before score/projection vector generation")

    score_env = os.environ.copy()
    score_env["TRACE_DIR"] = str(PROFILE)
    score_env["VECTOR_DIR"] = str(SCORE_VECTORS)
    score_env["RESULT_DIR"] = str(SCORE_RESULTS)
    run(
        ["bash", "sim_local5/run_local5_checkpoint_score_trace_checks.sh"],
        "checkpoint-bound T450 Q/K score/Shiftmax RTL exact",
        env=score_env,
    )

    run(
        [
            PYTHON,
            "scripts/generate_local5_active_projection_postg0_vectors.py",
            "--input-dir",
            str(PROFILE),
            "--output-dir",
            str(VECTORS),
            "--per-stage",
            "25",
            "--out-dim",
            "2",
            "--weight-mode",
            "checkpoint_theta_folded_dyadic_int8_head_slice",
        ],
        "generate 100 checkpoint-bound T450 real-weight projection vectors",
    )
    env = os.environ.copy()
    env["RESULT_DIR"] = str(RESULTS.relative_to(HW_ROOT))
    env["VECTOR_DIR"] = str(VECTORS.relative_to(HW_ROOT))
    env["CHECKPOINT_WEIGHTS"] = "1"
    run(
        ["bash", "sim_new_arch/run_local5_qgasr2c_fivebank_checks.sh"],
        "direct/QGASR checkpoint-bound projection RTL/SVA/lint/synthesis",
        env=env,
    )
    report = RESULTS / "report.json"
    if not report.is_file():
        raise FileNotFoundError(report)
    projection_result = json.loads(report.read_text(encoding="utf-8"))
    vector_manifest = VECTORS / "manifest.json"
    projection_manifest = json.loads(vector_manifest.read_text(encoding="utf-8"))
    source_manifest = RESULTS / "source_sha256.txt"
    required_verification = {
        "checkpoint_weight_binding",
        "random_sva",
        "verilator_lint",
        "yosys_check",
    }
    verification = projection_result.get("verification", {})
    if (
        projection_result.get("weight_mode")
        != "checkpoint_theta_folded_dyadic_int8_head_slice"
        or any(verification.get(key) != "PASS" for key in required_verification)
        or projection_result.get("vector_manifest_sha256")
        != hashlib.sha256(vector_manifest.read_bytes()).hexdigest()
        or not source_manifest.is_file()
        or projection_result.get("source_manifest_sha256")
        != hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    ):
        raise RuntimeError("projection RTL report未完整绑定checkpoint vectors/RTL验证")
    projection_result["evidence_scope"] = (
        "checkpoint_bound_post_g0_real_weight_head_slice_projection_accumulator_"
        "rtl_exact_not_cross_head_bn_requant_full_attention_or_full_network"
    )
    score_report = SCORE_RESULTS / "report.json"
    if not score_report.is_file():
        raise FileNotFoundError(score_report)
    score_result = json.loads(score_report.read_text(encoding="utf-8"))
    score_manifest = Path(str(score_result.get("vector_manifest", "")))
    score_vector_manifest = (
        json.loads(score_manifest.read_text(encoding="utf-8"))
        if score_manifest.is_file()
        else {}
    )
    score_source_identity = ordered_manifest_identity(score_vector_manifest)
    if (
        score_result.get("status") != "PASS"
        or not all(score_result.get("checks", {}).values())
        or not score_manifest.is_file()
        or score_result.get("vector_manifest_sha256") != file_sha256(score_manifest)
        or score_source_identity.get("checkpoint_sha256") != checkpoint_sha
        or score_source_identity.get("config_sha256")
        != identity.get("config_sha256")
    ):
        raise RuntimeError("checkpoint score/Shiftmax RTL report未通过fail-closed检查")
    projection_source_identity = ordered_manifest_identity(
        projection_manifest
    )
    if (
        projection_source_identity.get("checkpoint_sha256") != checkpoint_sha
        or projection_source_identity.get("config_sha256")
        != identity.get("config_sha256")
    ):
        raise RuntimeError("projection RTL vectors are not bound to Local-5 rank-1")
    result = {
        "schema": "local5_checkpoint_bound_component_rtl_exact_v2",
        "status": "PASS",
        "evidence_scope": (
            "checkpoint_bound_score_shiftmax_projection_partial_accumulator_and_"
            "atlif_temporal_matrix_component_rtl_exact_not_full_network"
        ),
        "training_config_identity": {
            "path": str(TRAINING_IDENTITY.resolve()),
            "sha256": file_sha256(TRAINING_IDENTITY),
            "config_sha256": training_identity["config_sha256"],
            "ep9_state_sha256": training_identity["state_sha256"],
        },
        "checkpoint_identity": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "config": str(HARDWARE_CONFIG.resolve()),
            "config_sha256": identity["config_sha256"],
            "best_epoch": int(identity["best_epoch"]),
            "run_identity": str(RUN_IDENTITY.resolve()),
            "run_identity_sha256": file_sha256(RUN_IDENTITY),
            "acceptance": str(ACCEPTANCE.resolve()),
            "acceptance_sha256": file_sha256(ACCEPTANCE),
        },
        "score_shiftmax": score_result,
        "projection": projection_result,
        "atlif_temporal_matrix": atlif_result,
        "source_artifacts": {
            "projection_report": file_binding(report),
            "projection_vector_manifest": file_binding(vector_manifest),
            "projection_source_manifest": file_binding(source_manifest),
            "projection_source_trace_manifest": file_binding(
                Path(str(projection_manifest["source_manifest"]))
            ),
            "projection_source_trace_payload": file_binding(
                Path(str(projection_manifest["source_payload"]))
            ),
            "projection_payloads": [
                file_binding(VECTORS / str(artifact["file"]))
                for artifact in projection_manifest["artifacts"].values()
            ],
            "projection_sources": projection_source_bindings(
                source_manifest, vector_manifest
            ),
            "score_report": file_binding(score_report),
            "atlif_report": file_binding(ATLIF_RESULTS / "report.json"),
            "runner": file_binding(Path(__file__)),
            "projection_shell": file_binding(
                HW_ROOT / "sim_new_arch/run_local5_qgasr2c_fivebank_checks.sh"
            ),
            "projection_generator": file_binding(
                HW_ROOT / "scripts/generate_local5_active_projection_postg0_vectors.py"
            ),
            "projection_summarizer": file_binding(
                HW_ROOT / "scripts/summarize_local5_gasr2c_fivebank_rtl.py"
            ),
            "score_shell": file_binding(
                HW_ROOT / "sim_local5/run_local5_checkpoint_score_trace_checks.sh"
            ),
            "score_generator": file_binding(
                HW_ROOT / "scripts/generate_local5_checkpoint_score_vectors.py"
            ),
            "score_reporter": file_binding(
                HW_ROOT / "scripts/report_local5_checkpoint_score_rtl.py"
            ),
            "provenance_validator": file_binding(
                HW_ROOT / "scripts/evidence_provenance.py"
            ),
        },
    }
    validate_local5_projection_provenance(result)
    (RESULTS / "checkpoint_bound_scope.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    record(
        "ALL COMPLETE checkpoint-bound Local-5 score/Shiftmax, projection partial "
        "accumulator, and ATLIF temporal-matrix component RTL exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
