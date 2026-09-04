#!/usr/bin/env python3
"""Validate legacy marker or ranked-checkpoint Local5 release receipts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


LEGACY_SCHEMA = "local5_release_receipt_v2"
RANKED_SCHEMA = "local5_ranked_checkpoint_release_receipt_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _bound_file(value: dict[str, Any], stem: str) -> Path:
    path = Path(str(value.get(f"{stem}_path", ""))).resolve()
    if not path.is_file():
        raise ValueError(f"release receipt missing bound file: {stem}")
    if value.get(f"{stem}_sha256") != file_sha256(path):
        raise ValueError(f"release receipt SHA256 mismatch: {stem}")
    return path


def _ranking_rank1(path: Path) -> tuple[int, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(
            r"\|\s*1\s*\|\s*(\d+)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|",
            line,
        )
        if match:
            return int(match.group(1)), float(match.group(2))
    raise ValueError(f"cannot parse rank-1 row: {path}")


def _validate_profile_identity(
    profile_path: Path,
    *,
    config_path: Path,
    checkpoint_path: Path,
    expected_deployment_scope: str | None,
) -> dict[str, Any]:
    profile = _load_json(profile_path)
    identity = profile.get("artifact_identity")
    load_audit = profile.get("checkpoint_load_audit")
    protocol = profile.get("eval_protocol")
    if not all(isinstance(item, dict) for item in (identity, load_audit, protocol)):
        raise ValueError(f"profile provenance is incomplete: {profile_path}")
    stat = checkpoint_path.stat()
    expected_identity = {
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    if any(identity.get(key) != expected for key, expected in expected_identity.items()):
        raise ValueError(f"profile artifact identity mismatch: {profile_path}")
    if (
        load_audit.get("missing_count") != 0
        or load_audit.get("unexpected_count") != 0
        or load_audit.get("checkpoint_overlay_keys") != 210
        or load_audit.get("model_overlay_keys") != 210
    ):
        raise ValueError(f"profile checkpoint load audit failed: {profile_path}")
    expected_protocol = {
        "resolution": [480, 640],
        "crop": None,
        "window_size": [2, 15, 15],
        "remap": "v1",
        "bn_policy": "no_running",
        "eval_batch_size": 1,
    }
    if any(protocol.get(key) != expected for key, expected in expected_protocol.items()):
        raise ValueError(f"profile protocol mismatch: {profile_path}")
    if int(profile.get("samples", 0)) != 825:
        raise ValueError(f"profile is not valid825: {profile_path}")
    if expected_deployment_scope is not None:
        contract = profile.get("deployment_contract")
        expected_shiftmax = {
            "attention_core_numeric": "float_exp2",
            "attention_core_hardware_order_numeric": (
                "Q8_LUT_integer_rowsum_ceil_pow2"
            ),
        }.get(expected_deployment_scope)
        invalid_mask_ok = (
            expected_deployment_scope == "attention_core_numeric"
            or contract.get("invalid_candidate_mask") is True
        ) if isinstance(contract, dict) else False
        if (
            not isinstance(contract, dict)
            or expected_shiftmax is None
            or contract.get("scope") != expected_deployment_scope
            or contract.get("score_quantization") != "Q7_step_2^-7"
            or contract.get("shiftmax") != expected_shiftmax
            or contract.get("gate_quantization") != "Q1.7_RNE"
            or not invalid_mask_ok
        ):
            raise ValueError(f"hardware-order deployment contract mismatch: {profile_path}")
    return profile


def _validate_legacy(value: dict[str, Any]) -> dict[str, Any]:
    status_path = Path(str(value.get("status_path", ""))).resolve()
    status = status_path.read_bytes()
    prefix_bytes = int(value.get("status_prefix_bytes", -1))
    marker_start = int(value.get("marker_start_offset", -1))
    marker_end = int(value.get("marker_end_offset", -1))
    marker_line = str(value.get("marker_line", ""))
    if (
        value.get("release_marker") != "ALL COMPLETE fullres deploy followup"
        or not 0 <= prefix_bytes <= marker_start < marker_end <= len(status)
        or hashlib.sha256(status[:prefix_bytes]).hexdigest()
        != value.get("status_prefix_sha256")
        or status[marker_start:marker_end]
        .decode("utf-8", errors="strict")
        .rstrip("\n")
        != marker_line
        or "ALL COMPLETE fullres deploy followup" not in marker_line
        or "H67" not in marker_line
        or "H66d" not in marker_line
    ):
        raise ValueError("legacy Local5 release receipt is invalid")
    return value


def _validate_ranked(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("status") != "PASS":
        raise ValueError("ranked Local5 release receipt is not PASS")
    ranking_path = _bound_file(value, "ranking")
    convergence_path = _bound_file(value, "convergence_summary")
    checkpoint_path = _bound_file(value, "checkpoint")
    training_config_path = _bound_file(value, "training_config")
    origin_training_identity_path = _bound_file(
        value, "origin_training_identity"
    )
    resume_30_to_40_path = _bound_file(value, "resume_30_to_40")
    resume_40_to_50_path = _bound_file(value, "resume_40_to_50")
    dyadic_config_path = _bound_file(value, "dyadic_config")
    dyadic_profile_path = _bound_file(value, "dyadic_profile")
    hardware_config_path = _bound_file(value, "config")
    float_profile_path = _bound_file(value, "float_profile")
    hardware_profile_path = _bound_file(value, "hardware_profile")

    rank_epoch, rank_aee_rounded = _ranking_rank1(ranking_path)
    convergence = _load_json(convergence_path)
    best_epoch = int(value.get("best_epoch", -1))
    if (
        best_epoch != rank_epoch
        or convergence.get("status") != "PASS"
        or int(convergence.get("rank1_checkpoint_label", -1)) != best_epoch
        or convergence.get("decision") != "operationally_plateaued_or_overfit"
        or value.get("selection_metric") != "AEE"
        or value.get("selection_decision") != convergence.get("decision")
        or not str(value.get("watcher_session_uuid", ""))
    ):
        raise ValueError("ranked Local5 selection contract mismatch")

    points = convergence.get("points")
    if not isinstance(points, list):
        raise ValueError("ranked Local5 convergence points missing")
    selected = next(
        (
            point
            for point in points
            if int(point.get("checkpoint_label", -1)) == best_epoch
        ),
        None,
    )
    if not isinstance(selected, dict):
        raise ValueError("ranked Local5 selected convergence point missing")
    if (
        selected.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or Path(str(selected.get("profile", ""))).resolve()
        != float_profile_path.resolve()
        or selected.get("profile_sha256") != file_sha256(float_profile_path)
    ):
        raise ValueError("ranked Local5 convergence artifact binding mismatch")

    float_profile = _validate_profile_identity(
        float_profile_path,
        config_path=training_config_path,
        checkpoint_path=checkpoint_path,
        expected_deployment_scope=None,
    )
    _validate_profile_identity(
        dyadic_profile_path,
        config_path=dyadic_config_path,
        checkpoint_path=checkpoint_path,
        expected_deployment_scope="attention_core_numeric",
    )
    hardware_profile = _validate_profile_identity(
        hardware_profile_path,
        config_path=hardware_config_path,
        checkpoint_path=checkpoint_path,
        expected_deployment_scope="attention_core_hardware_order_numeric",
    )
    float_aee = float((float_profile.get("metrics") or {}).get("AEE", "nan"))
    selected_aee = float(selected.get("AEE", "nan"))
    if (
        abs(float_aee - selected_aee) > 1e-12
        or abs(float_aee - rank_aee_rounded) > 5e-5
        or not isinstance(hardware_profile.get("metrics"), dict)
    ):
        raise ValueError("ranked Local5 metric binding mismatch")
    origin_identity = _load_json(origin_training_identity_path)
    resume_30_to_40 = _load_json(resume_30_to_40_path)
    resume_40_to_50 = _load_json(resume_40_to_50_path)
    origin_state = Path(str(origin_identity.get("state_path", ""))).resolve()
    audit_sources_ok = True
    for audit in (resume_30_to_40, resume_40_to_50):
        for stem in ("source_model", "source_state"):
            path = Path(str(audit.get(stem, ""))).resolve()
            expected_sha = audit.get(f"{stem}_sha256")
            audit_sources_ok = (
                audit_sources_ok
                and path.is_file()
                and expected_sha == file_sha256(path)
            )
    if (
        origin_identity.get("status") != "PASS"
        or not origin_state.is_file()
        or origin_identity.get("state_sha256") != file_sha256(origin_state)
        or not all((origin_identity.get("checks") or {}).values())
        or not audit_sources_ok
        or resume_30_to_40.get("scope")
        != "model_optimizer_scheduler_scaler_resume_not_rng_bit_exact"
        or resume_40_to_50.get("status") != "PASS"
        or resume_40_to_50.get("scope")
        != "model_optimizer_scheduler_scaler_resume_not_rng_bit_exact"
        or resume_40_to_50.get("config_sha256")
        != file_sha256(training_config_path)
        or resume_40_to_50.get("does_not_inherit_ep29_hardware_provenance")
        is not True
        or Path(str(resume_40_to_50.get("config", ""))).resolve()
        != training_config_path.resolve()
    ):
        raise ValueError("ranked Local5 continuation lineage mismatch")
    return value


def validate_release_receipt(
    receipt_path: Path,
    expected_hash: str | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.resolve()
    if not receipt_path.is_file():
        raise ValueError("Local5 release receipt is missing")
    if expected_hash is not None and file_sha256(receipt_path) != expected_hash:
        raise ValueError("Local5 release receipt SHA256 binding failed")
    value = _load_json(receipt_path)
    schema = value.get("schema")
    if schema == LEGACY_SCHEMA:
        return _validate_legacy(value)
    if schema == RANKED_SCHEMA:
        return _validate_ranked(value)
    raise ValueError(f"unsupported Local5 release receipt schema: {schema}")
