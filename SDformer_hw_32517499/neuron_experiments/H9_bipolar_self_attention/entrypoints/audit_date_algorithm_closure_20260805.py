#!/usr/bin/env python3
"""Fail-closed final audit for the DSEC DATE algorithm and RTL evidence chain."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RESULTS = EXP / "results"
HW = REPO / "hw_autoresearch_nts07"
sys.path.insert(0, str(HW / "scripts"))

from evidence_provenance import (  # noqa: E402
    validate_local5_projection_provenance,
    validate_projection_provenance,
)

LOCAL = RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
LOCAL_RANKING = LOCAL / "profile_ranking_valid825.md"
LOCAL_DEPLOY = LOCAL / "deploy_summary.json"
LOCAL_CONFIG_IDENTITY = LOCAL / "training_config_identity.json"
LOCAL_TRAINING_CONFIG = EXP / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml"
LOCAL_DYADIC_CONFIG = EXP / (
    "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30_dyadic_q7q17_deploy.yml"
)
LOCAL_HARDWARE_CONFIG = EXP / (
    "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30_hardware_order_q7q17_deploy.yml"
)
LOCAL_ACTIVE_LAUNCH = LOCAL / "active_launch_provenance.json"
LOCAL_RTL = HW / "results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/checkpoint_bound_scope.json"
LOCAL_ACCEPTANCE = (
    HW / "results/local5_fullres_bb1e4_postg0_acceptance_20260805/acceptance.json"
)
CONVERGENCE = RESULTS / "dsec_fullres_w15_equal_plus10_convergence_summary_20260805.json"
H67_FINAL = HW / "results/h67_postconvergence_rank1_hardware_evidence_20260805.json"
H67_LINEAGE = REPO / "neuron_autoresearch/H67_FULLRES_LINEAGE_RECEIPT_20260805.json"
H67_LINEAGE_GENERATOR = EXP / "entrypoints/generate_h67_fullres_lineage_receipt_20260805.py"
H67_CROP_SOURCE = RESULTS / (
    "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_"
    "full30_20260711_setsid/checkpoint_epoch19.pth"
)
H67_EP30 = RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch30.pth"

REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
AAE_DIAGNOSTIC = REPO / "neuron_autoresearch/AAE_BASELINE_DIAGNOSTIC_20260717.md"
AAE_RECEIPT = REPO / "neuron_autoresearch/AAE_METRIC_TEST_RECEIPT_20260805.json"
AAE_GAP_RECEIPT = REPO / "neuron_autoresearch/NB0_AAE_GAP_DIAGNOSTIC_20260806.json"
HW_DOC = HW / "docs/247_H67与Local5最终Checkpoint硬件证据闭环_20260805.md"
OUTPUT_JSON = REPO / "neuron_autoresearch/DATE_ALGORITHM_CLOSURE_AUDIT_20260805.json"
OUTPUT_MD = REPO / "neuron_autoresearch/DATE_ALGORITHM_CLOSURE_AUDIT_20260805.md"
STATUS = RESULTS / "date_algorithm_closure_audit_20260805.log"
LOCK = RESULTS / "date_algorithm_closure_audit_20260805.lock"

REQUIRED = (
    LOCAL_RANKING,
    LOCAL_DEPLOY,
    LOCAL_CONFIG_IDENTITY,
    LOCAL_ACCEPTANCE,
    LOCAL_RTL,
    CONVERGENCE,
    H67_FINAL,
    H67_LINEAGE,
    AAE_RECEIPT,
    AAE_GAP_RECEIPT,
)
LOCAL_EPOCHS = (9, 14, 19, 24, 29)
LOCAL_PROJECTION_WEIGHT_MODE = "checkpoint_theta_folded_dyadic_int8_head_slice"
EXPECTED_VALIDATION_LIST_SHA256 = (
    "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0"
)
EXPECTED_SEQUENCE_FRAME_COUNTS = {
    "thun_00_a": 5,
    "zurich_city_01_a": 10,
    "zurich_city_02_a": 7,
    "zurich_city_02_c": 80,
    "zurich_city_02_d": 37,
    "zurich_city_02_e": 47,
    "zurich_city_03_a": 44,
    "zurich_city_05_a": 63,
    "zurich_city_05_b": 40,
    "zurich_city_06_a": 65,
    "zurich_city_07_a": 10,
    "zurich_city_08_a": 35,
    "zurich_city_09_a": 64,
    "zurich_city_10_a": 76,
    "zurich_city_10_b": 39,
    "zurich_city_11_a": 24,
    "zurich_city_11_b": 97,
    "zurich_city_11_c": 82,
}


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_aae_gap_receipt() -> dict[str, Any]:
    raw = load_json(AAE_GAP_RECEIPT)
    metric_checks = raw.get("metric_receipt_checks") or {}
    head_checks = raw.get("head_to_head_checks") or {}
    diagnosis = raw.get("diagnosis") or {}
    checks = {
        "status": raw.get("status")
        == "PASS_LOCAL_DIAGNOSIS_OFFICIAL_TEST_REPRODUCTION_UNAVAILABLE",
        "scope_not_official_test": "not an official hidden-test reproduction"
        in str(raw.get("scope", "")),
        "all_metric_checks": bool(metric_checks) and all(metric_checks.values()),
        "all_head_to_head_checks": bool(head_checks) and all(head_checks.values()),
        "no_formula_bug": diagnosis.get("formula_bug") is False,
        "nb0_aee_undertraining_plausible": diagnosis.get(
            "NB0_AEE_undertraining_plausible"
        )
        is True,
        "angle_not_explained_by_training_alone": diagnosis.get(
            "NB0_angle_gap_explained_by_undertraining_alone"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"AAE gap receipt failed: {checks}")
    return {
        "path": str(AAE_GAP_RECEIPT),
        "sha256": sha256(AAE_GAP_RECEIPT),
        "checks": checks,
        "late_trends": raw.get("late_trends"),
    }


def parse_ranking(path: Path) -> list[dict[str, int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
        if match:
            rows.append({"rank": int(match.group(1)), "epoch": int(match.group(2))})
    if not rows or [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"invalid ranking table: {path}")
    return rows


def validate_profile(
    profile_path: Path,
    checkpoint: Path,
    config: Path,
    *,
    overlay: int,
    atlif: int,
    shiftmax: int,
) -> dict[str, Any]:
    raw = load_json(profile_path)
    protocol = raw.get("eval_protocol") or {}
    contract = raw.get("metric_contract") or {}
    aggregation_audit = raw.get("metric_aggregation_audit") or {}
    frame_equal = aggregation_audit.get("frame_equal_mean") or {}
    pixel_global = aggregation_audit.get("pixel_global_mean") or {}
    sequence_balanced = aggregation_audit.get("sequence_balanced_mean") or {}
    validation_file_list = raw.get("validation_file_list") or {}
    validation_file = Path(str(validation_file_list.get("path", "")))
    identity = raw.get("artifact_identity") or {}
    audit = raw.get("checkpoint_load_audit") or {}
    counts = raw.get("module_counts") or {}
    metrics = raw.get("metrics") or {}
    checks = {
        "resolution": protocol.get("resolution") == [480, 640],
        "crop": protocol.get("crop") is None,
        "window": protocol.get("window_size") == [2, 15, 15],
        "bn": protocol.get("bn_policy") == "no_running",
        "batch": int(protocol.get("eval_batch_size", 0)) == 1,
        "legacy_aae": contract.get("AAE") == "legacy_2d_direction_angle_degrees_between_uv",
        "benchmark_aae": contract.get("AAE_Benchmark")
        == "middlebury_barron_3d_angle_degrees_between_normalized_uv1",
        "dsec_fl": contract.get("DSEC_Fl")
        == "percentage_epe_gt_3px_and_gt_5pct_of_ground_truth_flow_magnitude",
        "aggregation": contract.get("aggregation")
        == "masked_mean_per_frame_then_equal_mean_over_validation_frames",
        "population": contract.get("population")
        == "local_DSEC_valid_file_list_not_official_hidden_test",
        "samples": int(raw.get("samples", 0)) == 825,
        "aggregation_schema": aggregation_audit.get("schema")
        == "flow_metric_aggregation_audit_v1",
        "aggregation_frames": int(aggregation_audit.get("frame_count", 0)) == 825,
        "aggregation_sequences": int(aggregation_audit.get("sequence_count", 0)) == 18,
        "aggregation_valid_pixels": float(aggregation_audit.get("valid_pixels", 0)) > 0,
        "aggregation_per_sequence": len(aggregation_audit.get("per_sequence") or {}) == 18,
        "aggregation_matches_metrics": all(
            key in metrics
            and key in frame_equal
            and abs(float(metrics[key]) - float(frame_equal[key])) <= 1.0e-5
            for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
        ),
        "aggregation_modes_complete": all(
            key in pixel_global and key in sequence_balanced
            for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
        ),
        "validation_list_exists": validation_file.is_file(),
        "validation_list_sha": validation_file.is_file()
        and validation_file_list.get("sha256") == sha256(validation_file),
        "checkpoint_path": identity.get("checkpoint_path") == str(checkpoint.resolve()),
        "checkpoint_sha": identity.get("checkpoint_sha256") == sha256(checkpoint),
        "config_path": identity.get("config_path") == str(config.resolve()),
        "config_sha": identity.get("config_sha256") == sha256(config),
        "overlay": audit.get("checkpoint_overlay_keys") == overlay
        and audit.get("model_overlay_keys") == overlay,
        "missing": audit.get("missing_count") == 0,
        "unexpected": audit.get("unexpected_count") == 0,
        "atlif": counts.get("ATLIFTernaryPSN") == atlif,
        "shiftmax": counts.get("ShiftmaxAttention") == shiftmax,
        "metrics": all(
            key in metrics for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"profile contract failed {profile_path}: {failed}")
    return {
        "AEE": float(metrics["AEE"]),
        "AAE": float(metrics["AAE"]),
        "AAE_Benchmark": float(metrics["AAE_Benchmark"]),
        "DSEC_Fl": float(metrics["DSEC_Fl"]),
        "total_spikes_g": float(raw["total_spikes"]) / 1e9,
        "population_identity": {
            "validation_file_path": str(validation_file.resolve()),
            "validation_file_sha256": validation_file_list["sha256"],
            "frame_count": int(aggregation_audit["frame_count"]),
            "sequence_frame_counts": {
                str(sequence): int(value["frame_count"])
                for sequence, value in sorted(
                    (aggregation_audit.get("per_sequence") or {}).items()
                )
            },
        },
        "aggregation": {
            "valid_pixels": float(aggregation_audit["valid_pixels"]),
            "sequence_count": int(aggregation_audit["sequence_count"]),
            "frame_equal_mean": {
                key: float(frame_equal[key])
                for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
            },
            "pixel_global_mean": {
                key: float(pixel_global[key])
                for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
            },
            "sequence_balanced_mean": {
                key: float(sequence_balanced[key])
                for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
            },
        },
    }


def validate_local_paired_states(root: Path = LOCAL) -> dict[str, Any]:
    import torch

    base_lrs = (1.0e-4, 1.0e-4, 5.0e-5, 5.0e-5, 5.0e-6)
    expected_factors = {9: 1.0, 19: 0.5, 29: 0.25}
    output: dict[str, Any] = {}
    for epoch, factor in expected_factors.items():
        checkpoint = root / f"checkpoint_epoch{epoch}.pth"
        state_path = root / f"checkpoint_epoch{epoch}_state_dict.pth"
        if not checkpoint.is_file() or not state_path.is_file():
            raise RuntimeError(f"Local-5 paired checkpoint missing at epoch{epoch}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        scheduler = state.get("scheduler") or {}
        optimizer = state.get("optimizer") or {}
        optimizer_lrs = tuple(
            float(group.get("lr", float("nan")))
            for group in optimizer.get("param_groups", [])
        )
        expected_lrs = tuple(value * factor for value in base_lrs)
        scheduler_lrs = tuple(float(value) for value in scheduler.get("_last_lr", []))
        checks = {
            "state epoch": int(state.get("epoch", -1)) == epoch,
            "scheduler epoch": int(scheduler.get("last_epoch", -1)) == epoch,
            "scheduler milestones": dict(scheduler.get("milestones", {}))
            == {13: 1, 20: 1},
            "optimizer groups": len(optimizer_lrs) == len(expected_lrs),
            "optimizer LR": len(optimizer_lrs) == len(expected_lrs)
            and all(abs(a - b) <= 1.0e-12 for a, b in zip(optimizer_lrs, expected_lrs)),
            "scheduler LR": len(scheduler_lrs) == len(expected_lrs)
            and all(abs(a - b) <= 1.0e-12 for a, b in zip(scheduler_lrs, expected_lrs)),
            "AMP scaler": bool(state.get("scaler")),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"Local-5 ep{epoch} paired state contract failed: {failed}")
        output[str(epoch)] = {
            "checkpoint_sha256": sha256(checkpoint),
            "state_sha256": sha256(state_path),
            "state_epoch": epoch,
            "scheduler_last_epoch": epoch,
            "optimizer_lrs": optimizer_lrs,
            "scaler_present": True,
        }
        del state
    return output


def validate_local_config_identity() -> dict[str, Any]:
    identity = load_json(LOCAL_CONFIG_IDENTITY)
    launch_binding = identity.get("active_launch_provenance") or {}
    launch = load_json(LOCAL_ACTIVE_LAUNCH)
    launch_artifact = launch.get("artifact_identity") or {}
    launch_checks = launch.get("checks") or {}
    facts = identity.get("state_facts") or {}
    checks = identity.get("checks") or {}
    validations = {
        "schema": identity.get("schema") == "local5_training_config_identity_v1",
        "status": identity.get("status") == "PASS",
        "authority": identity.get("authority") == "ep9_optimizer_scheduler_state",
        "deterministic config": identity.get("deterministic_regeneration_equal") is True,
        "state epoch": facts.get("state_epoch") == 9,
        "scheduler epoch": facts.get("scheduler_last_epoch") == 9,
        "milestones": facts.get("scheduler_milestones") == {"13": 1, "20": 1}
        or facts.get("scheduler_milestones") == {13: 1, 20: 1},
        "runtime checks": bool(checks) and all(checks.values()),
        "state binding": identity.get("state_sha256")
        == sha256(LOCAL / "checkpoint_epoch9_state_dict.pth"),
        "launch path": Path(str(launch_binding.get("path", ""))).resolve()
        == LOCAL_ACTIVE_LAUNCH.resolve(),
        "launch sha": launch_binding.get("sha256") == sha256(LOCAL_ACTIVE_LAUNCH),
        "launch schema": launch.get("schema")
        == "local5_active_launch_provenance_v1",
        "launch status": launch.get("status") == "PASS_ACTIVE_CAPTURE",
        "launch checks": bool(launch_checks) and all(launch_checks.values()),
        "launch source checkpoint": launch_artifact.get("source_checkpoint_sha256")
        == sha256(
            LOCAL.parent
            / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
        ),
    }
    failed = [name for name, passed in validations.items() if not passed]
    if failed:
        raise RuntimeError(f"Local-5 training config identity failed: {failed}")
    return {
        "config_sha256": identity["config_sha256"],
        "ep9_state_sha256": identity["state_sha256"],
        "active_launch_provenance_sha256": launch_binding["sha256"],
        "scheduler_repaired_at_ep9_boundary": bool(
            identity.get("scheduler_repaired_at_ep9_boundary")
        ),
    }


def ordered_manifest_identity(vector_manifest: dict[str, Any]) -> dict[str, str]:
    source = Path(str(vector_manifest.get("source_manifest", "")))
    if not source.is_file() or sha256(source) != vector_manifest.get("source_manifest_sha256"):
        raise RuntimeError("ordered source manifest binding failed")
    source_raw = load_json(source)
    checkpoint_sha = str(source_raw.get("checkpoint_sha256", ""))
    config_sha = str(source_raw.get("config_sha256", ""))
    if len(checkpoint_sha) != 64 or len(config_sha) != 64:
        raise RuntimeError("ordered source manifest lacks checkpoint/config SHA")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
    }


def ordered_manifest_checkpoint_sha(vector_manifest: dict[str, Any]) -> str:
    return ordered_manifest_identity(vector_manifest)["checkpoint_sha256"]


def validate_local_acceptance(
    checkpoint: Path,
    acceptance_path: Path = LOCAL_ACCEPTANCE,
    training_identity_path: Path = LOCAL_CONFIG_IDENTITY,
) -> dict[str, Any]:
    acceptance = load_json(acceptance_path)
    checks = acceptance.get("checks") or {}
    manifest_path = Path(str(acceptance.get("manifest", "")))
    identity_path = Path(str(acceptance.get("run_identity", "")))
    if not manifest_path.is_file() or not identity_path.is_file():
        raise RuntimeError("Local-5 acceptance manifest/run identity missing")
    manifest = load_json(manifest_path)
    identity = load_json(identity_path)
    training_identity = load_json(training_identity_path)
    training_binding = (identity.get("source_bindings") or {}).get(
        "training_config_identity"
    ) or {}
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
        "threshold_training_deployment_semantics",
    )
    checkpoint_sha = sha256(checkpoint)
    validations = {
        "schema": acceptance.get("schema") == "local5_post_g0_acceptance_v1",
        "accepted": acceptance.get("accepted") is True,
        "samples100": int(acceptance.get("samples", 0)) == 100,
        "blocks12": int(acceptance.get("blocks", 0)) == 12,
        "required checks": all(checks.get(name) is True for name in required_checks),
        "manifest SHA": acceptance.get("manifest_sha256") == sha256(manifest_path),
        "identity SHA": acceptance.get("run_identity_sha256") == sha256(identity_path),
        "manifest checkpoint": manifest.get("checkpoint_sha256") == checkpoint_sha,
        "identity checkpoint": identity.get("checkpoint_sha256") == checkpoint_sha,
        "manifest identity": manifest.get("run_identity_file_sha256") == sha256(identity_path),
        "rank1 epoch": int(identity.get("best_epoch", -1))
        == int(checkpoint.stem.removeprefix("checkpoint_epoch")),
        "training identity path": Path(str(training_binding.get("path", ""))).resolve()
        == training_identity_path.resolve(),
        "training identity SHA": training_binding.get("sha256")
        == sha256(training_identity_path),
        "training identity status": training_identity.get("status") == "PASS",
    }
    failed = [name for name, passed in validations.items() if not passed]
    if failed:
        raise RuntimeError(f"Local-5 post-G0 acceptance failed: {failed}")
    return {
        "acceptance": str(acceptance_path),
        "acceptance_sha256": sha256(acceptance_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "run_identity": str(identity_path),
        "run_identity_sha256": sha256(identity_path),
        "samples": 100,
        "blocks": 12,
        "required_checks": list(required_checks),
    }


def validate_local_rtl(checkpoint: Path) -> dict[str, Any]:
    result = load_json(LOCAL_RTL)
    validate_local5_projection_provenance(result)
    if result.get("status") != "PASS" or "not_full_network" not in str(
        result.get("evidence_scope", "")
    ):
        raise RuntimeError("Local-5 aggregate RTL scope/status failed")
    checkpoint_sha = sha256(checkpoint)
    acceptance_identity_path = Path(
        str(load_json(LOCAL_ACCEPTANCE).get("run_identity", ""))
    )
    acceptance_identity = load_json(acceptance_identity_path)
    checkpoint_binding = result.get("checkpoint_identity") or {}
    if (
        Path(str(checkpoint_binding.get("checkpoint", ""))).resolve()
        != checkpoint.resolve()
        or checkpoint_binding.get("checkpoint_sha256") != checkpoint_sha
        or Path(str(checkpoint_binding.get("config", ""))).resolve()
        != Path(str(acceptance_identity.get("config", ""))).resolve()
        or checkpoint_binding.get("config_sha256")
        != acceptance_identity.get("config_sha256")
        or int(checkpoint_binding.get("best_epoch", -1))
        != int(checkpoint.stem.removeprefix("checkpoint_epoch"))
        or Path(str(checkpoint_binding.get("run_identity", ""))).resolve()
        != acceptance_identity_path.resolve()
        or checkpoint_binding.get("acceptance_sha256") != sha256(LOCAL_ACCEPTANCE)
    ):
        raise RuntimeError("Local-5 aggregate RTL/checkpoint identity failed")
    training_binding = result.get("training_config_identity") or {}
    if (
        Path(str(training_binding.get("path", ""))).resolve()
        != LOCAL_CONFIG_IDENTITY.resolve()
        or training_binding.get("sha256") != sha256(LOCAL_CONFIG_IDENTITY)
    ):
        raise RuntimeError("Local-5 RTL report/training config identity mismatch")

    score = result.get("score_shiftmax") or {}
    if score.get("status") != "PASS" or not all((score.get("checks") or {}).values()):
        raise RuntimeError("Local-5 score/Shiftmax report failed")
    score_manifest_path = Path(str(score.get("vector_manifest", "")))
    if not score_manifest_path.is_file() or sha256(score_manifest_path) != score.get(
        "vector_manifest_sha256"
    ):
        raise RuntimeError("Local-5 score vector manifest binding failed")
    score_source_identity = ordered_manifest_identity(load_json(score_manifest_path))
    if (
        score_source_identity.get("checkpoint_sha256") != checkpoint_sha
        or score_source_identity.get("config_sha256")
        != acceptance_identity.get("config_sha256")
    ):
        raise RuntimeError("Local-5 score vectors/checkpoint SHA mismatch")

    projection = result.get("projection") or {}
    validate_local_projection_weight_mode(projection)
    verification = projection.get("verification") or {}
    if any(
        verification.get(key) != "PASS"
        for key in ("checkpoint_weight_binding", "random_sva", "verilator_lint", "yosys_check")
    ):
        raise RuntimeError("Local-5 projection verification failed")
    projection_manifest_path = Path(str(projection.get("vector_manifest", "")))
    if not projection_manifest_path.is_file() or sha256(
        projection_manifest_path
    ) != projection.get("vector_manifest_sha256"):
        raise RuntimeError("Local-5 projection vector manifest binding failed")
    projection_manifest = load_json(projection_manifest_path)
    projection_source_identity = ordered_manifest_identity(projection_manifest)
    if (
        projection_source_identity.get("checkpoint_sha256") != checkpoint_sha
        or projection_source_identity.get("config_sha256")
        != acceptance_identity.get("config_sha256")
    ):
        raise RuntimeError("Local-5 projection vectors/checkpoint SHA mismatch")
    binding = projection_manifest.get("projection_contract_binding") or {}
    contract_path = Path(str(binding.get("manifest", "")))
    if (
        not contract_path.is_file()
        or sha256(contract_path) != binding.get("manifest_sha256")
        or load_json(contract_path).get("checkpoint_sha256") != checkpoint_sha
    ):
        raise RuntimeError("Local-5 projection contract/checkpoint SHA mismatch")

    atlif = result.get("atlif_temporal_matrix") or {}
    if (
        atlif.get("status") != "PASS"
        or atlif.get("checkpoint_identity", {}).get("checkpoint_sha256") != checkpoint_sha
        or atlif.get("checkpoint_identity", {}).get("config_sha256")
        != acceptance_identity.get("config_sha256")
        or atlif.get("numeric_bridge", {}).get("deployment_accuracy_signoff") is not False
    ):
        raise RuntimeError("Local-5 ATLIF report/checkpoint boundary failed")
    return {
        "aggregate_scope": result["evidence_scope"],
        "score_vectors": int(score.get("vectors", 0)),
        "projection_weight_mode": projection["weight_mode"],
        "atlif_hidden_events": int(atlif.get("rtl", {}).get("events", 0)),
        "training_config_identity_sha256": training_binding["sha256"],
    }


def validate_local_projection_weight_mode(projection: dict[str, Any]) -> None:
    if projection.get("weight_mode") != LOCAL_PROJECTION_WEIGHT_MODE:
        raise RuntimeError(
            "Local-5 projection weight mode is not the theta-folded production mode"
        )


def convergence_profiles(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("schema") != "dsec_fullres_equal_plus10_convergence_v1":
        raise RuntimeError("unexpected convergence summary schema")
    specs = {
        "H67": (
            RESULTS / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805",
            EXP / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml",
            (30, 35, 40), 210, 105, 12,
        ),
        "NB0": (
            RESULTS / "dsec_fullres_w15_NB0_equal_plus10_ep40_20260805",
            EXP / "configs/generated/dsec_fullres_w15_NB0_equal_plus10_ep40.yml",
            (29, 34, 39), 0, 0, 0,
        ),
    }
    output = {}
    for name, (root, config, labels, overlay, atlif, shiftmax) in specs.items():
        candidate = (summary.get("candidates") or {}).get(name) or {}
        points = candidate.get("points") or []
        if (
            len(points) != 3
            or tuple(int(point["checkpoint_label"]) for point in points) != labels
            or tuple(int(point["budget"]) for point in points) != (30, 35, 40)
        ):
            raise RuntimeError(f"{name} convergence point labels failed")
        validated_profiles = {}
        for point, label in zip(points, labels, strict=True):
            checkpoint = root / f"checkpoint_epoch{label}.pth"
            metrics = validate_profile(
                root / f"standard_valid825/epoch{label}/spike_profile.json",
                checkpoint,
                config,
                overlay=overlay,
                atlif=atlif,
                shiftmax=shiftmax,
            )
            validated_profiles[label] = metrics
            for key in (
                "AEE",
                "AAE",
                "AAE_Benchmark",
                "DSEC_Fl",
                "total_spikes_g",
            ):
                if abs(float(point[key]) - metrics[key]) > 1e-9:
                    raise RuntimeError(f"{name} convergence summary metric drift: {label} {key}")
        ordered = [validated_profiles[label] for label in labels]
        rank1_index = min(range(3), key=lambda index: ordered[index]["AEE"])

        def improvement(metric: str, previous: int, current: int) -> float:
            return 100.0 * (
                ordered[previous][metric] - ordered[current][metric]
            ) / ordered[previous][metric]

        def change(metric: str, previous: int, current: int) -> float:
            return 100.0 * (
                ordered[current][metric] - ordered[previous][metric]
            ) / ordered[previous][metric]

        recomputed = {
            "rank1_budget": (30, 35, 40)[rank1_index],
            "rank1_checkpoint_label": labels[rank1_index],
            "aee_last5_improvement_pct": improvement("AEE", 1, 2),
            "aee_last10_improvement_pct": improvement("AEE", 0, 2),
            "aae2d_last5_improvement_pct": improvement("AAE", 1, 2),
            "aae2d_last10_improvement_pct": improvement("AAE", 0, 2),
            "ae3d_last5_improvement_pct": improvement("AAE_Benchmark", 1, 2),
            "ae3d_last10_improvement_pct": improvement("AAE_Benchmark", 0, 2),
            "spikes_last5_change_pct": change("total_spikes_g", 1, 2),
            "spikes_last10_change_pct": change("total_spikes_g", 0, 2),
        }
        recomputed["decision"] = (
            "not_plateaued"
            if recomputed["rank1_budget"] == 40
            else "operationally_plateaued_or_overfit"
        )
        recomputed["angle_decision"] = (
            "angle_plateaued"
            if abs(recomputed["aae2d_last5_improvement_pct"]) <= 1.0
            and abs(recomputed["ae3d_last5_improvement_pct"]) <= 1.0
            else "angle_not_plateaued_or_noisy"
        )
        for key, expected in recomputed.items():
            observed = candidate.get(key)
            if isinstance(expected, float):
                if observed is None or abs(float(observed) - expected) > 1e-9:
                    raise RuntimeError(
                        f"{name} convergence summary derived-field drift: {key}"
                    )
            elif observed != expected:
                raise RuntimeError(
                    f"{name} convergence summary derived-field drift: {key}"
                )
        output[name] = {
            **recomputed,
            "profiles": {str(label): validated_profiles[label] for label in labels},
            "rank1_metrics": validated_profiles[labels[rank1_index]],
        }
    return output


def validate_common_population(profile_groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    identities = {
        name: value.get("population_identity") or {}
        for name, value in profile_groups.items()
    }
    if not identities:
        raise RuntimeError("no profile population identities to compare")
    reference_name = next(iter(identities))
    reference = identities[reference_name]
    required = (
        "validation_file_path",
        "validation_file_sha256",
        "frame_count",
        "sequence_frame_counts",
    )
    if any(key not in reference for key in required):
        raise RuntimeError(f"population identity incomplete: {reference_name}")
    mismatched = [
        name for name, identity in identities.items() if identity != reference
    ]
    if mismatched:
        raise RuntimeError(
            "candidate validation population mismatch: " + ", ".join(mismatched)
        )
    if (
        int(reference["frame_count"]) != 825
        or len(reference["sequence_frame_counts"]) != 18
        or sum(int(value) for value in reference["sequence_frame_counts"].values())
        != 825
    ):
        raise RuntimeError("common validation population frame distribution is invalid")
    if reference["validation_file_sha256"] != EXPECTED_VALIDATION_LIST_SHA256:
        raise RuntimeError("common validation population list SHA is not the frozen valid825")
    if reference["sequence_frame_counts"] != EXPECTED_SEQUENCE_FRAME_COUNTS:
        raise RuntimeError("common validation sequence distribution is not the frozen valid825")
    return {"reference_profile": reference_name, **reference}


def select_qualifying_mainline(
    *,
    local5: dict[str, Any],
    h67: dict[str, Any],
    nb0: dict[str, Any],
    convergence_eligible: dict[str, bool] | None = None,
) -> dict[str, Any]:
    baseline_aee = float(nb0["AEE"])
    baseline_spikes = float(nb0["total_spikes_g"])
    comparisons: dict[str, Any] = {}
    for name, metrics in (("H67", h67), ("Local5", local5)):
        aee = float(metrics["AEE"])
        spikes = float(metrics["total_spikes_g"])
        aee_change_pct = 100.0 * (aee - baseline_aee) / baseline_aee
        spikes_change_pct = 100.0 * (spikes - baseline_spikes) / baseline_spikes
        comparisons[name] = {
            "AEE": aee,
            "total_spikes_g": spikes,
            "AEE_change_pct_vs_NB0": aee_change_pct,
            "total_spikes_change_pct_vs_NB0": spikes_change_pct,
            "AEE_within_5pct": aee <= 1.05 * baseline_aee,
            "spikes_reduction_at_least_20pct": spikes <= 0.80 * baseline_spikes,
            "convergence_eligible": (
                True
                if convergence_eligible is None
                else bool(convergence_eligible.get(name, False))
            ),
        }
        comparisons[name]["qualifies"] = (
            comparisons[name]["AEE_within_5pct"]
            and comparisons[name]["spikes_reduction_at_least_20pct"]
            and comparisons[name]["convergence_eligible"]
        )
    qualified = [name for name, row in comparisons.items() if row["qualifies"]]
    if not qualified:
        raise RuntimeError(
            "no convergence-eligible H67/Local5 candidate meets AEE within 5% "
            "and spikes reduction >=20%"
        )
    selected = min(qualified, key=lambda name: comparisons[name]["AEE"])
    return {
        "baseline": {
            "name": "NB0",
            "AEE": baseline_aee,
            "total_spikes_g": baseline_spikes,
        },
        "criteria": {
            "AEE_max_relative_degradation_pct": 5.0,
            "minimum_total_spikes_reduction_pct": 20.0,
        },
        "candidates": comparisons,
        "qualified_candidates": qualified,
        "selected_mainline": selected,
    }


def classify_local5_convergence(
    profiles: dict[str, dict[str, Any]], rank1_epoch: int
) -> dict[str, Any]:
    previous = float(profiles["24"]["AEE"])
    boundary = float(profiles["29"]["AEE"])
    improvement = 100.0 * (previous - boundary) / previous
    # A boundary optimum is right-censored regardless of how small the last
    # measured slope is. The slope is descriptive, not a convergence proof.
    not_plateaued = rank1_epoch == 29
    return {
        "rank1_epoch": rank1_epoch,
        "late_interval": [24, 29],
        "aee_last5_improvement_pct": improvement,
        "decision": (
            "not_plateaued"
            if not_plateaued
            else "operationally_plateaued_or_overfit"
        ),
        "final_mainline_eligible": not not_plateaued,
    }


def validate_aae_receipt() -> dict[str, Any]:
    receipt = load_json(AAE_RECEIPT)
    contracts = receipt.get("contracts") or {}
    sources = receipt.get("sources") or {}
    expected = {
        "metric": REPO / "third_party/SDformerFlow/loss/flow_supervised.py",
        "evaluator": REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
        "tests": REPO / "third_party/SDformerFlow/tests/test_aae_metrics.py",
        "aggregation": REPO / "third_party/SDformerFlow/utils/metric_aggregation.py",
        "aggregation_tests": REPO / "third_party/SDformerFlow/tests/test_metric_aggregation.py",
    }
    checks = {
        "schema": receipt.get("schema") == "aae_metric_test_receipt_v2",
        "status": receipt.get("status") == "PASS",
        "tests": int(receipt.get("test_count", 0)) == 8,
        "legacy": contracts.get("legacy_aae") == "2d_direction_angle_degrees_between_uv",
        "benchmark": contracts.get("benchmark_ae")
        == "middlebury_barron_3d_angle_degrees_between_normalized_uv1",
        "aggregation": contracts.get("aggregation")
        == "masked_mean_per_frame_then_equal_mean_over_validation_frames",
        "dsec_fl": contracts.get("dsec_fl")
        == "percentage_epe_gt_3px_and_gt_5pct_of_ground_truth_flow_magnitude",
        "batch": int(contracts.get("eval_batch_size", 0)) == 1,
        "audited aggregations": contracts.get("audited_aggregations")
        == ["frame_equal_mean", "pixel_global_mean", "sequence_balanced_mean"],
    }
    for name, path in expected.items():
        binding = sources.get(name) or {}
        checks[f"{name} path"] = Path(str(binding.get("path", ""))).resolve() == path.resolve()
        checks[f"{name} sha"] = binding.get("sha256") == sha256(path)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"AAE metric test receipt failed: {failed}")
    return receipt


def validate_h67_lineage(path: Path = H67_LINEAGE) -> dict[str, Any]:
    receipt = load_json(path)

    def validate_file_binding(value: dict[str, Any], label: str) -> Path:
        artifact = Path(str(value.get("path", "")))
        checks = {
            "exists": artifact.is_file(),
            "sha": artifact.is_file() and value.get("sha256") == sha256(artifact),
            "size": artifact.is_file()
            and int(value.get("size_bytes", -1)) == artifact.stat().st_size,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"H67 lineage {label} binding failed: {failed}")
        return artifact

    generator = validate_file_binding(receipt.get("generator") or {}, "generator")
    initial = validate_file_binding(receipt.get("initial_checkpoint") or {}, "initial")
    final = validate_file_binding(receipt.get("final_checkpoint") or {}, "final")
    deletion_path = validate_file_binding(receipt.get("deletion_audit") or {}, "deletion audit")
    deletion = load_json(deletion_path)
    deleted = {item["path"]: item for item in deletion.get("deleted", [])}
    scheduler = receipt.get("scheduler_alignment") or {}
    validate_file_binding(scheduler.get("artifact") or {}, "scheduler alignment")

    stages = receipt.get("stages") or []
    for index, stage in enumerate(stages):
        if stage.get("status") != "PASS" or not all((stage.get("checks") or {}).values()):
            raise RuntimeError(f"H67 lineage stage{index} status/checks failed")
        validate_file_binding(stage.get("config") or {}, f"stage{index} config")
        validate_file_binding(stage.get("log") or {}, f"stage{index} log")
        for field in ("source", "output", "resume_state"):
            value = stage.get(field)
            if value is None:
                continue
            artifact = Path(str(value.get("path", "")))
            if artifact.is_file():
                validate_file_binding(value, f"stage{index} {field}")
                continue
            deletion_record = value.get("deletion_audit") or {}
            try:
                relative_path = str(artifact.resolve().relative_to(REPO.resolve()))
            except ValueError as exc:
                raise RuntimeError(f"H67 lineage deleted artifact outside repo: {artifact}") from exc
            if deletion_record != deleted.get(relative_path):
                raise RuntimeError(f"H67 lineage stage{index} {field} deletion binding failed")

    checks = {
        "schema": receipt.get("schema") == "h67_fullres_lineage_receipt_v1",
        "status": receipt.get("status") == "PASS",
        "generator": generator.resolve() == H67_LINEAGE_GENERATOR.resolve(),
        "initial": initial.resolve() == H67_CROP_SOURCE.resolve(),
        "final": final.resolve() == H67_EP30.resolve(),
        "five stages": len(stages) == 5,
        "lineage checks": bool(receipt.get("checks"))
        and all((receipt.get("checks") or {}).values()),
        "scheduler checks": bool(scheduler.get("checks"))
        and all((scheduler.get("checks") or {}).values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"H67 lineage receipt failed: {failed}")
    return {
        "receipt": str(path.resolve()),
        "receipt_sha256": sha256(path),
        "initial_checkpoint": str(initial),
        "initial_checkpoint_sha256": sha256(initial),
        "final_checkpoint": str(final),
        "final_checkpoint_sha256": sha256(final),
        "stages": len(stages),
        "no_nb0_or_local5_initialization": True,
    }


def validate_h67_final(convergence: dict[str, Any]) -> dict[str, Any]:
    final = load_json(H67_FINAL)
    epoch = int(final.get("rank1_epoch", -1))
    if final.get("status") != "PASS" or epoch != convergence["H67"]["rank1_checkpoint_label"]:
        raise RuntimeError("H67 final hardware evidence/rank-1 mismatch")
    if "component_rtl_exact" not in str(final.get("scope", "")):
        raise RuntimeError("H67 final hardware evidence lacks component RTL-exact scope")
    checkpoint = Path(str(final.get("checkpoint", "")))
    if not checkpoint.is_file():
        raise RuntimeError("H67 final checkpoint missing")
    checkpoint_sha = sha256(checkpoint)
    score_path = Path(
        str(final.get("rtl_report") or final.get("reused_ep30_checkpoint_bound_report") or "")
    )
    atlif_path = Path(
        str(final.get("atlif_rtl_report") or final.get("reused_ep30_atlif_checkpoint_bound_report") or "")
    )
    projection_path = Path(
        str(
            final.get("projection_rtl_report")
            or final.get("reused_ep30_projection_checkpoint_bound_report")
            or ""
        )
    )
    config_path = Path(str(final.get("hardware_order_config", "")))
    profile_path = Path(str(final.get("profile", "")))
    manifest_path = Path(str(final.get("trace_manifest", "")))
    audit_path = Path(str(final.get("trace_audit", "")))
    if not all(
        path.is_file()
        for path in (config_path, profile_path, manifest_path, audit_path)
    ):
        raise RuntimeError("H67 final profile/trace/audit evidence missing")
    profile = load_json(profile_path)
    manifest = load_json(manifest_path)
    trace_audit = load_json(audit_path)
    profile_identity = profile.get("artifact_identity", {})
    profile_protocol = profile.get("eval_protocol", {})
    profile_counts = profile.get("module_counts", {})
    manifest_identity = manifest.get("run_context", {}).get("artifact_identity", {})
    manifest_records = manifest.get("records") or []
    profile_trace_checks = {
        "profile checkpoint": profile_identity.get("checkpoint_sha256") == checkpoint_sha,
        "profile config": profile_identity.get("config_sha256") == sha256(config_path),
        "profile samples": int(profile.get("samples", 0)) == 100,
        "profile resolution": profile_protocol.get("resolution") == [480, 640],
        "profile crop": profile_protocol.get("crop") is None,
        "profile window": profile_protocol.get("window_size") == [2, 15, 15],
        "profile tokens": int(profile_protocol.get("tokens_per_window", 0)) == 450,
        "profile ATLIF": profile_counts.get("ATLIFTernaryPSN") == 105,
        "profile Shiftmax": profile_counts.get("ShiftmaxAttention") == 12,
        "profile trace records": int(profile.get("bit_trace_records", 0)) == 12,
        "manifest checkpoint": manifest_identity.get("checkpoint_sha256") == checkpoint_sha,
        "manifest config": manifest_identity.get("config_sha256") == sha256(config_path),
        "manifest records": len(manifest_records) == 12,
        "manifest tokens": {int(row.get("temporal_tokens", 0)) for row in manifest_records}
        == {450},
        "manifest payload SHA": all(
            Path(str(row.get("file", ""))).is_file()
            and row.get("sha256") == sha256(Path(str(row["file"])))
            for row in manifest_records
        ),
        "audit status": trace_audit.get("status") == "PASS",
        "audit manifest": Path(str(trace_audit.get("source_manifest", ""))).resolve()
        == manifest_path.resolve(),
        "audit stages": trace_audit.get("coverage", {}).get("stages") == [0, 1, 2, 3],
        "audit four stages": trace_audit.get("coverage", {}).get("four_stage_complete")
        is True,
        "audit records": len(trace_audit.get("records") or []) == 12,
        "audit payload SHA": all(
            row.get("sha256_ok") is True for row in (trace_audit.get("records") or [])
        ),
    }
    failed = [name for name, passed in profile_trace_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"H67 profile/trace/audit mismatch: {failed}")
    score = load_json(score_path)
    atlif = load_json(atlif_path)
    projection = load_json(projection_path)
    validate_projection_provenance(projection)
    if score.get("status") != "PASS" or score.get("run_context", {}).get(
        "artifact_identity", {}
    ).get("checkpoint_sha256") != checkpoint_sha or score.get("run_context", {}).get(
        "artifact_identity", {}
    ).get("config_sha256") != sha256(config_path):
        raise RuntimeError("H67 score/Shiftmax report/checkpoint mismatch")
    if atlif.get("status") != "PASS" or atlif.get("checkpoint_identity", {}).get(
        "checkpoint_sha256"
    ) != checkpoint_sha or atlif.get("checkpoint_identity", {}).get(
        "config_sha256"
    ) != sha256(config_path):
        raise RuntimeError("H67 ATLIF report/checkpoint mismatch")
    projection_checks = {
        "status": projection.get("status") == "PASS",
        "scope": "projection_component_rtl_exact"
        in str(projection.get("scope", "")),
        "checkpoint": projection.get("checkpoint_identity", {}).get(
            "checkpoint_sha256"
        )
        == checkpoint_sha,
        "config": projection.get("checkpoint_identity", {}).get("config_sha256")
        == sha256(config_path),
        "records": int(projection.get("record_count", 0)) == 12,
        "stages": projection.get("required_stage_coverage") == [0, 1, 2, 3],
        "tokens": int(projection.get("temporal_tokens", 0)) == 450,
        "token_id_width": int(projection.get("token_id_width", 0)) == 9,
        "weight_mode": projection.get("weight_mode")
        == "checkpoint_dyadic_int8_projection_weight",
    }
    failed = [name for name, passed in projection_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"H67 projection report/checkpoint mismatch: {failed}")
    return {
        "rank1_epoch": epoch,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "scope": str(final.get("scope", "")),
        "hardware_order_config": str(config_path),
        "profile": str(profile_path),
        "trace_manifest": str(manifest_path),
        "trace_audit": str(audit_path),
        "score_report": str(score_path),
        "atlif_report": str(atlif_path),
        "projection_report": str(projection_path),
    }


def append_docs(
    local_epoch: int,
    local_convergence: dict[str, Any],
    convergence: dict[str, Any],
) -> None:
    marker = "DATE_ALGORITHM_CLOSURE_AUDIT_PASS_20260805"
    for path, heading in (
        (REDESIGN, "DATE 算法/RTL 最终证据闭环"),
        (HW_DOC, "DATE 算法/RTL 最终证据闭环"),
    ):
        text = path.read_text(encoding="utf-8")
        if marker in text:
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n### {heading}\n\n<!-- {marker} -->\n\n")
            handle.write(
                f"- fail-closed closure audit PASS；Local-5 rank-1 ep{local_epoch}，"
                f"H67 rank-1 ep{convergence['H67']['rank1_checkpoint_label']}。\n"
            )
            handle.write(
                f"- H67 收敛判定 `{convergence['H67']['decision']}`，NB0 收敛判定 "
                f"`{convergence['NB0']['decision']}`；AAE-2D 与 AE-3D 仍分口径报告。\n"
            )
            handle.write(
                f"- Local-5 收敛判定 `{local_convergence['decision']}`，ep24到ep29 "
                f"AEE改善 `{local_convergence['aee_last5_improvement_pct']:.3f}%`；边界仍改善时"
                "不得选为最终主线。\n"
            )
            handle.write(
                "- H67 训练血缘由机器收据绑定为自身 Motion-XOR crop ep19 经五段续训到 "
                "fullres ep30；没有从 NB0 或 Local-5 初始化。\n"
            )
            handle.write(
                "- Local-5 仅声明 score/Shiftmax、真实权重 per-head projection partial "
                "accumulator、ATLIF temporal matrix 三项 component RTL-exact；H67 同样不外推为 full network。\n"
            )
            handle.write(f"- 机器审计：`{OUTPUT_JSON.relative_to(REPO)}`。\n")


def run_audit() -> dict[str, Any]:
    ranking = parse_ranking(LOCAL_RANKING)
    if len(ranking) != 5 or {row["epoch"] for row in ranking} != set(LOCAL_EPOCHS):
        raise RuntimeError("Local-5 ranking does not cover all five frozen checkpoints")
    local_epoch = ranking[0]["epoch"]
    local_checkpoint = LOCAL / f"checkpoint_epoch{local_epoch}.pth"
    local_profiles = {
        str(epoch): validate_profile(
            LOCAL / f"standard_valid825/epoch{epoch}/spike_profile.json",
            LOCAL / f"checkpoint_epoch{epoch}.pth",
            LOCAL_TRAINING_CONFIG,
            overlay=210,
            atlif=105,
            shiftmax=12,
        )
        for epoch in LOCAL_EPOCHS
    }
    deploy = load_json(LOCAL_DEPLOY)
    if int(deploy.get("best_epoch", -1)) != local_epoch or Path(
        str(deploy.get("checkpoint", ""))
    ).resolve() != local_checkpoint.resolve():
        raise RuntimeError("Local-5 deploy summary/rank-1 mismatch")
    for key in ("float", "dyadic", "hardware_order"):
        if not isinstance(deploy.get(key), dict) or "AEE" not in deploy[key]:
            raise RuntimeError(f"Local-5 deploy path missing: {key}")
    deploy_profiles = {
        "dyadic": validate_profile(
            Path(str(deploy["dyadic_profile"])),
            local_checkpoint,
            LOCAL_DYADIC_CONFIG,
            overlay=210,
            atlif=105,
            shiftmax=12,
        ),
        "hardware_order": validate_profile(
            Path(str(deploy["hardware_profile"])),
            local_checkpoint,
            LOCAL_HARDWARE_CONFIG,
            overlay=210,
            atlif=105,
            shiftmax=12,
        ),
    }
    for key, metrics in deploy_profiles.items():
        for metric in (
            "AEE",
            "AAE",
            "AAE_Benchmark",
            "DSEC_Fl",
            "total_spikes_g",
        ):
            if abs(float(deploy[key][metric]) - float(metrics[metric])) > 1e-9:
                raise RuntimeError(f"Local-5 deploy summary metric drift: {key} {metric}")
    local_rtl = validate_local_rtl(local_checkpoint)
    local_acceptance = validate_local_acceptance(local_checkpoint)
    local_paired_states = validate_local_paired_states()
    local_config_identity = validate_local_config_identity()
    local_convergence = classify_local5_convergence(local_profiles, local_epoch)

    convergence_summary = load_json(CONVERGENCE)
    convergence = convergence_profiles(convergence_summary)
    population_profiles = {
        **{f"Local5-standard-ep{epoch}": local_profiles[str(epoch)] for epoch in LOCAL_EPOCHS},
        **{f"Local5-{name}": metrics for name, metrics in deploy_profiles.items()},
        **{
            f"{candidate}-ep{epoch}": metrics
            for candidate, result in convergence.items()
            for epoch, metrics in result["profiles"].items()
        },
    }
    common_population = validate_common_population(population_profiles)
    if convergence["NB0"]["decision"] == "not_plateaued":
        raise RuntimeError(
            "NB0 equal-budget reference remains best at a non-plateaued boundary; "
            "continue the baseline before final DATE selection"
        )
    algorithm_targets = select_qualifying_mainline(
        local5=local_profiles[str(local_epoch)],
        h67=convergence["H67"]["rank1_metrics"],
        nb0=convergence["NB0"]["rank1_metrics"],
        convergence_eligible={
            "H67": convergence["H67"]["decision"] != "not_plateaued",
            "Local5": bool(local_convergence["final_mainline_eligible"]),
        },
    )
    h67_lineage = validate_h67_lineage()
    h67 = validate_h67_final(convergence)
    aae_receipt = validate_aae_receipt()
    aae_gap_receipt = validate_aae_gap_receipt()
    aae_text = AAE_DIAGNOSTIC.read_text(encoding="utf-8")
    if not all(token in aae_text for token in ("4.871", "6.1803", "angle nearly plateaued")):
        raise RuntimeError("AAE diagnostic lacks final population/convergence evidence")
    redesign_text = REDESIGN.read_text(encoding="utf-8")
    required_markers = (
        "DSEC_FULLRES_W15_H66D_LOCAL5_BB1E4_RESULT_20260805",
        "DSEC_FULLRES_W15_H67_NB0_EQUAL_PLUS10_RESULT_20260805",
    )
    missing_markers = [marker for marker in required_markers if marker not in redesign_text]
    if missing_markers:
        raise RuntimeError(f"redesign lacks final result markers: {missing_markers}")

    output = {
        "schema": "date_algorithm_closure_audit_v1",
        "status": "PASS",
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": "checkpoint_bound_component_rtl_exact_not_full_network",
        "local5": {
            "rank1_epoch": local_epoch,
            "checkpoint": str(local_checkpoint),
            "checkpoint_sha256": sha256(local_checkpoint),
            "standard_valid825": local_profiles,
            "paired_training_states": local_paired_states,
            "training_config_identity": local_config_identity,
            "convergence": local_convergence,
            "deploy": deploy,
            "post_g0_acceptance": local_acceptance,
            "rtl": local_rtl,
        },
        "convergence": convergence,
        "algorithm_targets": algorithm_targets,
        "h67_training_lineage": h67_lineage,
        "h67_hardware": h67,
        "aae_diagnostic": {
            "path": str(AAE_DIAGNOSTIC),
            "sha256": sha256(AAE_DIAGNOSTIC),
            "official_4p871_not_local_valid825_target": True,
            "metric_test_receipt": str(AAE_RECEIPT),
            "metric_test_receipt_sha256": sha256(AAE_RECEIPT),
            "metric_tests": int(aae_receipt["test_count"]),
            "gap_diagnostic_receipt": aae_gap_receipt,
            "common_population": common_population,
            "same_population_rank1_aggregation": {
                "Local5": local_profiles[str(local_epoch)]["aggregation"],
                "H67": convergence["H67"]["rank1_metrics"]["aggregation"],
                "NB0": convergence["NB0"]["rank1_metrics"]["aggregation"],
            },
        },
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                LOCAL_RANKING,
                LOCAL_DEPLOY,
                LOCAL_CONFIG_IDENTITY,
                LOCAL_ACCEPTANCE,
                LOCAL_RTL,
                CONVERGENCE,
                H67_FINAL,
                H67_LINEAGE,
                AAE_RECEIPT,
                AAE_GAP_RECEIPT,
                Path(__file__).resolve(),
            )
        },
    }
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    aggregation_rows = []
    for name, values in output["aae_diagnostic"][
        "same_population_rank1_aggregation"
    ].items():
        aggregation_rows.append(
            f"| {name} | {values['frame_equal_mean']['AAE_Benchmark']:.4f} | "
            f"{values['pixel_global_mean']['AAE_Benchmark']:.4f} | "
            f"{values['sequence_balanced_mean']['AAE_Benchmark']:.4f} |"
        )
    OUTPUT_MD.write_text(
        "\n".join(
            [
                "# DATE Algorithm Closure Audit",
                "",
                "Status: **PASS**",
                "",
                f"- Local-5 rank-1: epoch {local_epoch}",
                f"- Local-5 convergence: `{local_convergence['decision']}`",
                f"- H67 rank-1: epoch {h67['rank1_epoch']}",
                "- H67 training lineage: own Motion-XOR crop ep19 -> fullres ep30; "
                "no NB0/Local-5 initialization.",
                f"- H67 convergence: `{convergence['H67']['decision']}`",
                f"- NB0 convergence: `{convergence['NB0']['decision']}`",
                f"- Selected mainline under AEE<=NB0+5% and spikes<=NB0-20%: "
                f"`{algorithm_targets['selected_mainline']}`",
                "- RTL claim: checkpoint-bound component exact only; not full-network RTL-exact.",
                "",
                "| model | AE-3D frame-equal | AE-3D pixel-global | AE-3D sequence-balanced |",
                "|---|---:|---:|---:|",
                *aggregation_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )
    append_docs(local_epoch, local_convergence, convergence)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("EXIT another closure auditor owns the lock")
            return 0
        while True:
            missing = [path for path in REQUIRED if not path.is_file()]
            if not missing:
                break
            if not args.wait:
                record("PENDING " + ", ".join(str(path.relative_to(REPO)) for path in missing))
                return 3
            record("WAIT " + ", ".join(path.name for path in missing))
            time.sleep(args.poll_seconds)
        output = run_audit()
        record(
            f"ALL COMPLETE DATE algorithm closure audit PASS Local5=ep{output['local5']['rank1_epoch']} "
            f"H67=ep{output['h67_hardware']['rank1_epoch']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
