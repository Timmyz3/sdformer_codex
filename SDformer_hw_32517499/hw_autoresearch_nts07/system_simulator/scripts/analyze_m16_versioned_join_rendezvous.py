#!/usr/bin/env python3
"""Classify exact-version residual joins for a future tile rendezvous engine."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_m4_keys(path: Path) -> set[tuple[int, str, str, int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (
                int(row["sample_id"]), row["sequence_key"], row["name"],
                int(row["operator_call_index"]),
            )
            for row in csv.DictReader(handle)
        }


def load_tile_evidence(path: Path) -> tuple[dict[str, Any], Path]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("status") != "PASS_IMMUTABLE_PROFILE_EVIDENCE_NOT_FULL_SPATIAL_OR_CYCLE_PROOF":
        raise ValueError("real-tile evidence is not admitted")
    entry = evidence.get("files", {}).get("real_tiles/tile_records.csv")
    if not entry or not entry.get("sha256"):
        raise ValueError("real-tile evidence does not bind tile_records.csv")
    tile_records = path.parent / entry["path"]
    if not tile_records.is_file() or sha256(tile_records) != entry["sha256"]:
        raise ValueError("real-tile record hash mismatch")
    manifest_entry = evidence.get("files", {}).get("real_tiles/manifest.json")
    if not manifest_entry or not manifest_entry.get("sha256"):
        raise ValueError("real-tile evidence does not bind its manifest")
    tile_manifest_path = path.parent / manifest_entry["path"]
    if not tile_manifest_path.is_file() or sha256(tile_manifest_path) != manifest_entry["sha256"]:
        raise ValueError("real-tile manifest hash mismatch")
    tile_manifest = json.loads(tile_manifest_path.read_text(encoding="utf-8"))
    if (
        tile_manifest.get("schema") != "dual_line_real_tile_trace_v2"
        or int(tile_manifest.get("cluster_contexts", 0)) != 4
    ):
        raise ValueError("M16 requires the adjacent-C4 v2 tile schema")
    return evidence, tile_records


def require_same_model(dependency: dict[str, Any], evidence: dict[str, Any]) -> None:
    dependency_identity = dependency.get("identities", {}).get("dependency_manifest", {})
    dependency_artifact = dependency_identity.get("artifact_identity", {})
    tile_artifact = evidence.get("artifact_identity", {})
    for field in ("checkpoint_sha256", "config_sha256"):
        if not dependency_artifact.get(field) or dependency_artifact[field] != tile_artifact.get(field):
            raise ValueError(f"dependency/tile {field} mismatch")
    dependency_samples = int(dependency_identity.get("samples", -1))
    command = evidence.get("command_tokens", [])
    try:
        tile_samples = int(command[command.index("--samples") + 1])
    except (ValueError, IndexError):
        raise ValueError("real-tile evidence does not bind --samples")
    if dependency_samples != tile_samples:
        raise ValueError("dependency/tile sample-count mismatch")
    if dependency_samples != 1:
        raise ValueError("M16 r2 freezes single-sample call-index semantics")


def elements(shape: list[int]) -> int:
    return math.prod(int(value) for value in shape)


def tensor_identity_sha256(tensor: dict[str, Any]) -> str:
    payload = {
        field: tensor.get(field)
        for field in (
            "dtype", "shape", "stride", "storage_cdata", "storage_offset", "version",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def classify_join_edge(
    row: dict[str, Any], m4_keys: set[tuple[int, str, str, int]],
) -> dict[str, Any]:
    result = {
        "atlif": row["name"],
        "sample_id": row["sample_id"],
        "sequence_key": row["sequence_key"],
        "atlif_call_index": row["module_call_index"],
        "service_cycles_l16": int(row["service_cycles_l16"]),
        "candidate": False,
        "reasons": [],
    }
    joins = row.get("join_operands", [])
    if row.get("live") is not True:
        result["reasons"].append("NOT_LIVE_JOIN")
        return result
    if not joins:
        result["reasons"].append("MISSING_JOIN_OPERANDS")
        return result
    top = max(joins, key=lambda item: int(item["join_event_index"]))
    result["join_event_index"] = int(top["join_event_index"])
    result["join_name"] = top["join_name"]
    if top["join_name"] != "aten.add.Tensor":
        result["reasons"].append("NOT_POINTWISE_ADD")
    output = top["output_tensor"]
    output_shape = list(output["shape"])
    result["output_shape"] = output_shape
    result["output_elements"] = elements(output_shape)
    result["output_tensor_identity_sha256"] = tensor_identity_sha256(output)
    if len(top.get("operands", [])) != 2:
        result["reasons"].append("NOT_BINARY_ADD")
    if row.get("transformed_seen") is not False or row.get("functional_ops") != ["aten.add.Tensor"]:
        result["reasons"].append("DOWNSTREAM_OR_JOIN_TRANSFORM_NOT_EXCLUDED")
    endpoints = []
    for operand in top["operands"]:
        input_tensor = operand["input_tensor"]
        if list(input_tensor["shape"]) != output_shape:
            result["reasons"].append("BROADCAST_OR_SHAPE_CHANGE")
        if (
            input_tensor.get("stride") != output.get("stride")
            or int(input_tensor.get("storage_offset", -1)) != int(output.get("storage_offset", -2))
        ):
            result["reasons"].append("NON_POINTWISE_VIEW_LAYOUT")
        nearest = operand.get("nearest_boundaries", [])
        if len(nearest) != 1:
            result["reasons"].append("NON_UNIQUE_OPERAND_BOUNDARY")
            continue
        endpoint = nearest[0]
        if endpoint.get("match_quality") not in {"exact_tensor_version", "exact_view_version"}:
            result["reasons"].append("NON_EXACT_OPERAND_VERSION")
        if endpoint.get("kind") not in {"module", "join", "persistent"}:
            result["reasons"].append("UNSUPPORTED_OPERAND_BOUNDARY")
        endpoints.append({
            "operand_index": int(operand["operand_index"]),
            "kind": endpoint.get("kind"),
            "name": endpoint.get("name"),
            "event_index": int(endpoint.get("event_index", -1)),
            "module_call_index": endpoint.get("module_call_index"),
            "tensor_version": input_tensor.get("version"),
            "tensor_dtype": input_tensor.get("dtype"),
            "tensor_identity_sha256": tensor_identity_sha256(input_tensor),
            "match_quality": endpoint.get("match_quality"),
            "m4_stream": (
                endpoint.get("kind") == "module"
                and (
                    int(row["sample_id"]), row["sequence_key"], endpoint.get("name"),
                    int(endpoint.get("module_call_index", -1)),
                ) in m4_keys
            ),
            "resident_before_join": int(endpoint.get("event_index", -1)) < int(top["join_event_index"]),
        })
        if int(endpoint.get("event_index", -1)) >= int(top["join_event_index"]):
            result["reasons"].append("OPERAND_NOT_AVAILABLE_BEFORE_JOIN")
    result["operands"] = endpoints
    stream_operands = [item for item in endpoints if item["m4_stream"]]
    if not stream_operands:
        result["reasons"].append("NO_M4_STREAM_OPERAND")
    if endpoints:
        latest_event = max(item["event_index"] for item in endpoints)
        result["latest_operand_boundary_event_index"] = latest_event
        latest_stream = [item for item in stream_operands if item["event_index"] == latest_event]
        if not latest_stream:
            result["reasons"].append("M4_NOT_LAST_BOUNDARY_IN_SOFTWARE_TRACE")
        result["stream_operands"] = latest_stream
        resident = [item for item in endpoints if item not in latest_stream]
        result["resident_operands"] = resident
        # INT8 is a lower-bound traffic contract; no claim that this tensor is
        # retained in the new rendezvous buffer rather than existing SRAM/DRAM.
        result["logical_resident_operand_payload_bytes_at_1B_per_element"] = (
            len(resident) * result["output_elements"]
        )
    result["candidate"] = not result["reasons"]
    result["candidate_scope"] = "structural_software_boundary_only"
    result["rtl_contract_missing"] = [
        "add_alpha", "operand_quant_scale_zero_point_signedness",
        "saturating_add_rounding_requant", "tile_ready_and_p_done_timestamps",
        "resident_lifetime_capacity_and_sram_ports",
    ]
    result["future_tag_fields_not_encoded_by_m16_r2"] = (
        "sample,sequence,producer_call,join_event,atlif_call,spatial,lane16,t,version"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-audit", type=Path, required=True)
    parser.add_argument("--real-tile-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dependency = json.loads(args.dependency_audit.read_text(encoding="utf-8"))
    if dependency.get("status") != "PASS_CAUSAL_DEPENDENCY_CLASSIFICATION":
        raise ValueError("dependency audit is not admitted")
    tile_evidence, tile_records = load_tile_evidence(args.real_tile_evidence)
    require_same_model(dependency, tile_evidence)
    m4_keys = load_m4_keys(tile_records)
    all_join = [
        row for row in dependency["rows"]
        if row["live"] is True and row.get("join_operands")
    ]
    if not all_join:
        raise ValueError("dependency audit contains no live join operands")
    call_keys = [
        (row["sample_id"], row["sequence_key"], row["name"], row["module_call_index"])
        for row in all_join
    ]
    if len(call_keys) != len(set(call_keys)):
        raise ValueError("duplicate live ATLIF join call identity")
    rows = [classify_join_edge(row, m4_keys) for row in all_join]
    candidates = [row for row in rows if row["candidate"]]
    candidate_cycles = sum(row["service_cycles_l16"] for row in candidates)
    total_cycles = sum(int(row["service_cycles_l16"]) for row in all_join)
    m4_join_rows = [row for row in rows if any(item.get("m4_stream") for item in row.get("operands", []))]
    m4_join_cycles = sum(row["service_cycles_l16"] for row in m4_join_rows)
    live_atlif_cycles = int(dependency["summary"]["live_service_cycles_l16"])
    payload = {
        "schema": "m16_versioned_pointwise_join_rendezvous_census_v2",
        "status": "PASS_STRUCTURAL_JOIN_CANDIDATE_CENSUS_NOT_TIMING_ADMISSION",
        "summary": {
            "all_live_join_calls": len(all_join),
            "all_live_join_service_cycles_l16": total_cycles,
            "current_tile_evidence_join_with_m4_calls": len(m4_join_rows),
            "current_tile_evidence_join_with_m4_service_cycles_l16": m4_join_cycles,
            "candidate_calls": len(candidates),
            "candidate_service_cycles_l16": candidate_cycles,
            "candidate_fraction_of_current_m4_join_service": (
                candidate_cycles / m4_join_cycles if m4_join_cycles else 0.0
            ),
            "candidate_fraction_of_all_live_atlif_service": candidate_cycles / live_atlif_cycles,
            "rejection_reason_histogram": dict(sorted(Counter(
                reason for row in rows for reason in row["reasons"]
            ).items())),
        },
        "rows": rows,
        "architecture_contract": {
            "rendezvous_future": "all operand-ready bits for the same immutable version tag",
            "commit": "pointwise INT8 add then ATLIF enqueue; no credit release before t=T-1",
            "default": "any missing/version-mismatched/transform operand remains fully serial",
            "m16_r2_admission": "none; a future quantized tile trace and finite-credit model are required",
        },
        "identities": {
            "dependency_audit_sha256": sha256(args.dependency_audit),
            "real_tile_evidence_sha256": sha256(args.real_tile_evidence),
            "real_tiles_sha256": sha256(tile_records),
            "checkpoint_sha256": tile_evidence["artifact_identity"]["checkpoint_sha256"],
            "config_sha256": tile_evidence["artifact_identity"]["config_sha256"],
            "source_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "Call-level exact-version pointwise-add structural candidate census only. Software "
            "event order is not tile arrival time. No alpha/quantization proof, per-tile arrival "
            "timestamps, resident-memory bandwidth proof, hidden cycles, system speedup, RTL, or PPA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "PASS_M16_JOIN_CENSUS candidates={}/{} cycles={}/{}".format(
            len(candidates), len(all_join), candidate_cycles, total_cycles
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
