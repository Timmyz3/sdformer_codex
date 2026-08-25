#!/usr/bin/env python3
"""Build an exact dynamic-BN barrier census and a bounded fused-replay model.

This milestone deliberately starts at the real no-running BatchNorm boundary.
It does not treat a Conv/Linear completion as ATLIF-ready.  The cycle model is
limited to the 13 direct-M4 source kernels, their dynamic BN materialization,
and the following L16 ATLIF service; it is not a full-network simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from analyze_h67_atlif_dependency_dag import (
    build_output_index,
    find_latest_producer,
    load_events,
    sha256,
    tensor_identity,
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def movement_cycle_variants(
    *, source_cycles: int, moment_serialization_cycles: int,
    moment_update_cycles: int, consumer_cycles: int,
    materialized_bytes: int, memory_bytes_per_cycle: float,
) -> dict[str, int]:
    """Return the honest 5/4/3/2-movement ablation matrix for one edge."""
    one_way = int(math.ceil(materialized_bytes / memory_bytes_per_cycle))
    source_write = max(int(source_cycles), one_way)
    source_write_online = max(
        int(source_cycles) + int(moment_serialization_cycles), one_way
    )
    statistics_read = max(int(moment_update_cycles), one_way)
    normalize_read_write = max(int(consumer_cycles), 2 * one_way)
    fused_bn_atlif_read = max(int(consumer_cycles), one_way)
    atlif_read = max(int(consumer_cycles), one_way)
    online_plus_fusion = source_write_online + fused_bn_atlif_read
    return {
        "unfused_five_movement": (
            source_write + statistics_read + normalize_read_write + atlif_read
        ),
        "online_only_four_movement": (
            source_write_online + normalize_read_write + atlif_read
        ),
        "fusion_only_three_movement": (
            source_write + statistics_read + fused_bn_atlif_read
        ),
        "online_plus_fusion_two_movement": online_plus_fusion,
        "proposed_two_movement": online_plus_fusion,
    }


def index_unique_atlif_enters(
    events: list[dict[str, Any]],
) -> dict[tuple[int, str, str, int], dict[str, Any]]:
    result: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    for event in events:
        if not (
            event.get("kind") == "leaf_module_enter"
            and event.get("module_type") == "ATLIFTernaryPSN"
        ):
            continue
        key = (
            int(event.get("sample_id", -1)),
            str(event.get("sequence_key", "")),
            str(event.get("name", "")),
            int(event.get("module_call_index", 0)),
        )
        if key in result:
            raise ValueError("duplicate ATLIF enter identity")
        result[key] = event
    return result


def trace_unique_direct_path(
    events: list[dict[str, Any]],
    output_index: dict[Any, Any],
    enter: dict[str, Any],
    boundary: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(enter.get("inputs", [])) != 1:
        raise ValueError("direct ATLIF boundary must have exactly one input")
    ref = enter["inputs"][0]
    upper = int(enter["event_index"])
    sample_id = int(enter["sample_id"])
    sequence_key = str(enter["sequence_key"])
    path: list[dict[str, Any]] = []
    visited: set[tuple[int, int, int, int]] = set()
    while True:
        key = (
            int(ref.get("storage_cdata", 0)),
            int(ref.get("storage_offset", 0)),
            int(ref.get("version", -1)),
            upper,
        )
        if key in visited:
            raise ValueError("cycle in direct-edge tensor ancestry")
        visited.add(key)
        match = find_latest_producer(
            ref, upper, output_index, sample_id, sequence_key
        )
        if match is None:
            raise ValueError("direct-edge path has an unresolved tensor")
        event, output_ref, quality = match
        if quality not in {"exact_tensor_version", "exact_view_version"}:
            raise ValueError("direct-edge path contains a non-exact tensor match")
        item = {
            "event_index": int(event["event_index"]),
            "kind": str(event.get("kind", "")),
            "name": str(event.get("name", "")),
            "module_type": event.get("module_type"),
            "module_call_index": event.get("module_call_index"),
            "match_quality": quality,
            "output_tensor": tensor_identity(output_ref),
        }
        path.append(item)
        if int(event["event_index"]) == int(boundary["producer_event_index"]):
            if str(event.get("name", "")) != str(boundary["producer"]):
                raise ValueError("producer event/name mismatch")
            break
        inputs = event.get("inputs", [])
        if len(inputs) != 1:
            raise ValueError("direct-edge pass path is not uniquely single-input")
        ref = inputs[0]
        upper = int(event["event_index"])
    return path


def batchnorm_contract(
    path: list[dict[str, Any]], events_by_index: dict[int, dict[str, Any]], input_bits: int
) -> dict[str, Any]:
    bn_items = [
        item for item in path
        if item.get("module_type") in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}
    ]
    if len(bn_items) != 1:
        raise ValueError("each admitted direct edge must cross exactly one BatchNorm")
    item = bn_items[0]
    event = events_by_index[item["event_index"]]
    if len(event.get("inputs", [])) != 1 or len(event.get("outputs", [])) != 1:
        raise ValueError("BatchNorm boundary is not one-input/one-output")
    logical_input = tensor_identity(event["inputs"][0])
    logical_output = tensor_identity(event["outputs"][0])
    shape = [int(value) for value in logical_input["shape"]]
    module_type = str(item["module_type"])
    expected_rank = {"BatchNorm1d": 3, "BatchNorm2d": 5, "BatchNorm3d": 6}[module_type]
    if len(shape) != expected_rank:
        raise ValueError("spiking BatchNorm logical tensor rank mismatch")
    # SpikingJelly multi-step BN flattens [T, B] into the native BN batch.
    # The logical channel is therefore axis 2 for all supported ranks.
    channel_axis = 2
    channels = shape[channel_axis]
    reduction_population = product(shape[:channel_axis] + shape[channel_axis + 1 :])
    elements = product(shape)
    if channels <= 0 or reduction_population <= 1:
        raise ValueError("invalid dynamic-BN reduction population")
    reduction_growth_bits = int(math.ceil(math.log(reduction_population, 2)))
    sum_bits = int(input_bits) + reduction_growth_bits
    maximum_signed_magnitude = 1 << (int(input_bits) - 1)
    maximum_sumsq = reduction_population * maximum_signed_magnitude**2
    sumsq_bits = int(maximum_sumsq.bit_length())
    return {
        "barrier_class": "BN_BLOCKED_GLOBAL_INPUT_STATISTICS",
        "module_event_index": int(item["event_index"]),
        "name": item["name"],
        "module_type": module_type,
        "module_call_index": item["module_call_index"],
        "logical_input_tensor": logical_input,
        "logical_output_tensor": logical_output,
        "channel_axis": channel_axis,
        "channels": channels,
        "reduction_axes": [index for index in range(len(shape)) if index != channel_axis],
        "reduction_population_per_channel": reduction_population,
        "elements": elements,
        "moment_state_bits_per_channel": {
            "signed_sum": sum_bits,
            "unsigned_sumsq": sumsq_bits,
            "total": sum_bits + sumsq_bits,
        },
        "moment_state_bits": channels * (sum_bits + sumsq_bits),
        "statistics_ready_only_after_elements": elements,
        "consumer_ready_definition": (
            "BN statistics complete, dynamic affine coefficients committed, and the "
            "corresponding materialized value is read into the fused BN+ATLIF stage"
        ),
    }


def build_model(
    events: list[dict[str, Any]],
    manifest: dict[str, Any],
    boundaries: dict[str, Any],
    m17_manifest: dict[str, Any],
    m17: dict[str, Any],
    *,
    consumer_lanes: int,
    moment_lanes: int,
    input_bits: int,
    frequency_mhz: float,
    dram_bandwidth_gbps: float,
) -> dict[str, Any]:
    if manifest["run_context"]["eval_protocol"].get("bn_policy") != "no_running":
        raise ValueError("M19 requires the real no_running evaluation contract")
    if boundaries.get("schema") != "m18_direct_m4_bn_blocked_path_certificates_v2":
        raise ValueError("M19 requires the fail-closed M18 BN-blocked certificate schema")
    if boundaries.get("status") != "PASS_EXACT_PATH_CERTIFICATES_ALL_BN_BLOCKED_M15_PROHIBITED":
        raise ValueError("M18 BN-blocked certificates are not admitted")
    m18_summary = boundaries.get("summary", {})
    expected_edges = len(boundaries.get("rows", []))
    if not (
        int(m18_summary.get("path_certified_edges", -1)) == expected_edges
        and int(m18_summary.get("bn_blocked_edges", -1)) == expected_edges
        and int(m18_summary.get("global_reduction_barrier_edges", -1)) == expected_edges
        and int(m18_summary.get("m15_admitted_edges", -1)) == 0
    ):
        raise ValueError("M18 summary is not fail-closed at the BN barrier")
    for field in ("checkpoint_sha256", "config_sha256"):
        dep_value = manifest["run_context"]["artifact_identity"][field]
        m17_value = m17_manifest["run_context"]["artifact_identity"][field]
        if dep_value != m17_value or dep_value != boundaries["identities"][field]:
            raise ValueError("M17/M18 artifact identity mismatch: " + field)
    if consumer_lanes <= 0 or moment_lanes <= 0 or input_bits <= 0:
        raise ValueError("lane and value widths must be positive")
    if frequency_mhz <= 0.0 or dram_bandwidth_gbps <= 0.0:
        raise ValueError("frequency and DRAM bandwidth must be positive")

    events_by_index = {int(event["event_index"]): event for event in events}
    if len(events_by_index) != len(events):
        raise ValueError("duplicate event_index")
    enters = index_unique_atlif_enters(events)
    output_index = build_output_index(events)
    m17_rows = {
        (str(row["name"]), int(row["operator_call_index"])): row
        for row in m17["rows"]
    }
    if len(m17_rows) != len(m17["rows"]):
        raise ValueError("duplicate M17 producer call")
    source_output_lanes = int(m17_manifest.get("architecture", {}).get("output_lanes", 0))
    if source_output_lanes != 96 or moment_lanes > source_output_lanes:
        raise ValueError("M19 requires the frozen 96-lane source and <=96 moment lanes")
    bytes_per_value = ceil_div(input_bits, 8)
    bytes_per_cycle = dram_bandwidth_gbps * 1000.0 / frequency_mhz
    rows = []
    used_m17: set[tuple[str, int]] = set()
    for boundary in boundaries["rows"]:
        enter_key = (
            int(boundary["sample_id"]), str(boundary["sequence_key"]),
            str(boundary["edge"]), int(boundary["edge_call_index"]),
        )
        if enter_key not in enters:
            raise ValueError("M18 ATLIF enter is absent from dependency events")
        enter = enters[enter_key]
        if int(enter["event_index"]) != int(boundary["edge_enter_event_index"]):
            raise ValueError("M18 ATLIF enter_event_index mismatch")
        path = trace_unique_direct_path(events, output_index, enter, boundary)
        bn = batchnorm_contract(path, events_by_index, input_bits)
        if not (
            boundary.get("causal_classification") == "BN_BLOCKED"
            and boundary.get("readiness_boundary") == "GLOBAL_REDUCTION_STATISTICS_BARRIER"
            and boundary.get("m15_admitted") is False
            and boundary.get("m15_rejection_reason")
            == "DYNAMIC_BN_OUTPUT_NOT_READY_AT_PRODUCER_P_DONE"
        ):
            raise ValueError("M18 row is not fail-closed at dynamic BN")
        m18_path = boundary.get("path_certificate", {}).get("producer_to_atlif_path", [])
        if [int(item["event_index"]) for item in reversed(path)] != [
            int(item["event_index"]) for item in m18_path
        ]:
            raise ValueError("independent M19 path does not match the M18 certificate")
        bn_barriers = boundary.get("bn_barriers", [])
        if not (
            len(bn_barriers) == 1
            and int(bn_barriers[0]["event_index"]) == int(bn["module_event_index"])
            and bn_barriers[0].get("bn_policy") == "no_running"
            and bn_barriers[0].get("barrier_kind")
            == "GLOBAL_REDUCTION_STATISTICS_BARRIER"
        ):
            raise ValueError("M18/M19 dynamic BN barrier mismatch")
        producer_key = (str(boundary["producer"]), int(boundary["producer_call_index"]))
        if producer_key not in m17_rows:
            raise ValueError("M18 producer is absent from M17 exact source census")
        if producer_key in used_m17:
            raise ValueError("M17 producer was consumed more than once")
        used_m17.add(producer_key)
        source = m17_rows[producer_key]
        elements = int(bn["elements"])
        if int(boundary["service_cycles_l16"]) != ceil_div(elements, 16):
            raise ValueError("r8 ATLIF L16 service is inconsistent with BN population")
        consumer_cycles = ceil_div(elements, consumer_lanes)
        source_output_cycles = ceil_div(elements, source_output_lanes)
        moment_update_cycles = ceil_div(elements, moment_lanes)
        moment_serialization_cycles = moment_update_cycles - source_output_cycles
        if moment_serialization_cycles < 0:
            raise ValueError("negative moment serialization")
        materialized_bytes = elements * bytes_per_value
        one_way_memory_cycles = int(math.ceil(materialized_bytes / bytes_per_cycle))

        cycle_matrix = {
            variant: movement_cycle_variants(
                source_cycles=int(source_cycles),
                moment_serialization_cycles=moment_serialization_cycles,
                moment_update_cycles=moment_update_cycles,
                consumer_cycles=consumer_cycles,
                materialized_bytes=materialized_bytes,
                memory_bytes_per_cycle=bytes_per_cycle,
            )
            for variant, source_cycles in {
                "local": source["local_m4_wall_cycles"],
                "hybrid": source["hybrid_m4_wall_cycles"],
                "p1_sparse_local": source["local_p1_sparse_wall_cycles"],
                "p1_sparse_hybrid": source["hybrid_p1_sparse_wall_cycles"],
                "same_width_dense": source["same_width_dense_wall_cycles"],
            }.items()
        }

        rows.append({
            "producer": boundary["producer"],
            "producer_call_index": int(boundary["producer_call_index"]),
            "edge": boundary["edge"],
            "edge_call_index": int(boundary["edge_call_index"]),
            "path_certificate": path,
            "path_certificate_sha256": canonical_sha256(path),
            "batchnorm": bn,
            "consumer_cycles": consumer_cycles,
            "source_output_cycles_at_96_lanes": source_output_cycles,
            "moment_update_cycles": moment_update_cycles,
            "moment_serialization_cycles": moment_serialization_cycles,
            "materialized_bytes": materialized_bytes,
            "one_way_memory_cycles": one_way_memory_cycles,
            "source_cycles": {
                "local": int(source["local_m4_wall_cycles"]),
                "hybrid": int(source["hybrid_m4_wall_cycles"]),
                "p1_sparse_local": int(source["local_p1_sparse_wall_cycles"]),
                "p1_sparse_hybrid": int(source["hybrid_p1_sparse_wall_cycles"]),
                "same_width_dense": int(source["same_width_dense_wall_cycles"]),
            },
            "bounded_proposed_two_pass_cycles": {
                variant: values["proposed_two_movement"]
                for variant, values in cycle_matrix.items()
            },
            "bounded_online_moments_four_pass_cycles": {
                variant: values["online_only_four_movement"]
                for variant, values in cycle_matrix.items()
            },
            "bounded_fusion_only_three_pass_cycles": {
                variant: values["fusion_only_three_movement"]
                for variant, values in cycle_matrix.items()
            },
            "bounded_online_plus_fusion_two_pass_cycles": {
                variant: values["online_plus_fusion_two_movement"]
                for variant, values in cycle_matrix.items()
            },
            "bounded_unfused_five_pass_cycles": {
                variant: values["unfused_five_movement"]
                for variant, values in cycle_matrix.items()
            },
        })
    if used_m17 != set(m17_rows):
        raise ValueError("M17 and M18 producer-call populations are not bijective")

    def total(section: str, variant: str) -> int:
        return sum(int(row[section][variant]) for row in rows)

    proposed = {
        variant: total("bounded_proposed_two_pass_cycles", variant)
        for variant in (
            "local", "hybrid", "p1_sparse_local", "p1_sparse_hybrid",
            "same_width_dense",
        )
    }
    total_elements = sum(int(row["batchnorm"]["elements"]) for row in rows)
    total_materialized_bytes = sum(int(row["materialized_bytes"]) for row in rows)
    online_four_pass = {
        variant: total("bounded_online_moments_four_pass_cycles", variant)
        for variant in proposed
    }
    fusion_three_pass = {
        variant: total("bounded_fusion_only_three_pass_cycles", variant)
        for variant in proposed
    }
    online_plus_fusion_two_pass = {
        variant: total("bounded_online_plus_fusion_two_pass_cycles", variant)
        for variant in proposed
    }
    unfused_five_pass = {
        variant: total("bounded_unfused_five_pass_cycles", variant)
        for variant in proposed
    }
    if proposed != online_plus_fusion_two_pass:
        raise ValueError("proposal does not close to the strongest composable two-pass baseline")
    sensitivity = []
    for bandwidth in (8.0, 16.0, 32.0, 48.0, 64.0):
        sensitivity_bytes_per_cycle = bandwidth * 1000.0 / frequency_mhz
        variant_totals = {}
        for variant in proposed:
            value = 0
            for row in rows:
                materialized_bytes = int(row["materialized_bytes"])
                one_way = int(math.ceil(materialized_bytes / sensitivity_bytes_per_cycle))
                phase_1 = max(
                    int(row["source_cycles"][variant])
                    + int(row["moment_serialization_cycles"]),
                    one_way,
                )
                phase_2 = max(int(row["consumer_cycles"]), one_way)
                value += phase_1 + phase_2
            variant_totals[variant] = value
        sensitivity.append({
            "dram_bandwidth_gbps": bandwidth,
            "dram_bytes_per_cycle": sensitivity_bytes_per_cycle,
            "bounded_proposed_two_pass_cycles": variant_totals,
            "bounded_local_speedup_vs_same_width_dense": (
                variant_totals["same_width_dense"] / variant_totals["local"]
            ),
            "bounded_hybrid_speedup_vs_same_width_dense": (
                variant_totals["same_width_dense"] / variant_totals["hybrid"]
            ),
        })
    summary = {
        "edges": len(rows),
        "all_edges_bn_blocked": all(
            row["batchnorm"]["barrier_class"] == "BN_BLOCKED_GLOBAL_INPUT_STATISTICS"
            for row in rows
        ),
        "elements": total_elements,
        "consumer_cycles": sum(int(row["consumer_cycles"]) for row in rows),
        "materialized_payload_bytes": total_materialized_bytes,
        "fused_write_plus_read_bytes": 2 * total_materialized_bytes,
        "online_plus_fusion_two_movement_bytes": 2 * total_materialized_bytes,
        "fusion_only_three_movement_bytes": 3 * total_materialized_bytes,
        "online_only_four_movement_bytes": 4 * total_materialized_bytes,
        "unfused_five_movement_bytes": 5 * total_materialized_bytes,
        "fused_bytes_saved_vs_unfused": 3 * total_materialized_bytes,
        "fused_total_movement_reduction_vs_unfused": 3.0 / 5.0,
        "fused_post_producer_movement_reduction_vs_unfused": 3.0 / 4.0,
        "proposed_movement_reduction_vs_online_only": 2.0 / 4.0,
        "proposed_movement_reduction_vs_fusion_only": 1.0 / 3.0,
        "proposed_movement_reduction_vs_online_plus_fusion": 0.0,
        "peak_single_tensor_bytes": max(int(row["materialized_bytes"]) for row in rows),
        "peak_moment_state_bits": max(int(row["batchnorm"]["moment_state_bits"]) for row in rows),
        "moment_serialization_cycles": sum(int(row["moment_serialization_cycles"]) for row in rows),
        "bounded_proposed_two_pass_cycles": proposed,
        "bounded_unfused_five_pass_cycles": unfused_five_pass,
        "bounded_online_only_four_pass_cycles": online_four_pass,
        "bounded_fusion_only_three_pass_cycles": fusion_three_pass,
        "bounded_online_plus_fusion_two_pass_cycles": online_plus_fusion_two_pass,
        "bounded_proposed_speedup_vs_online_only_local": online_four_pass["local"] / proposed["local"],
        "bounded_proposed_speedup_vs_online_only_hybrid": online_four_pass["hybrid"] / proposed["hybrid"],
        "bounded_proposed_speedup_vs_fusion_only_local": fusion_three_pass["local"] / proposed["local"],
        "bounded_proposed_speedup_vs_fusion_only_hybrid": fusion_three_pass["hybrid"] / proposed["hybrid"],
        "bounded_proposed_speedup_vs_online_plus_fusion_local": online_plus_fusion_two_pass["local"] / proposed["local"],
        "bounded_proposed_speedup_vs_online_plus_fusion_hybrid": online_plus_fusion_two_pass["hybrid"] / proposed["hybrid"],
        "bounded_13edge_hybrid_reduction_vs_local": 1.0 - proposed["hybrid"] / proposed["local"],
        "bounded_13edge_local_speedup_vs_same_width_dense": proposed["same_width_dense"] / proposed["local"],
        "bounded_13edge_hybrid_speedup_vs_same_width_dense": proposed["same_width_dense"] / proposed["hybrid"],
        "bounded_13edge_local_speedup_vs_p1_sparse": proposed["p1_sparse_local"] / proposed["local"],
        "bounded_13edge_hybrid_speedup_vs_p1_sparse": proposed["p1_sparse_hybrid"] / proposed["hybrid"],
        "bandwidth_sensitivity": sensitivity,
    }
    if total_elements != 552_960_000:
        raise ValueError("unexpected exact direct-edge BN population")
    if summary["consumer_cycles"] != ceil_div(total_elements, consumer_lanes):
        raise ValueError("consumer cycle total does not close")
    return {
        "summary": summary,
        "rows": rows,
        "architecture_contract": {
            "name": "producer_fused_moments_materialize_then_bn_atlif_replay",
            "phase_1": (
                "M4 produces each value, accumulates per-channel sum/sumsq in the producer "
                "stream, and materializes the unnormalized value"
            ),
            "barrier": "all reduction elements complete before dynamic scale/bias commit",
            "phase_2": (
                "read each materialized value once; fuse dynamic BN affine and ATLIF so no "
                "normalized tensor is written"
            ),
            "eliminated_tensor_movements_vs_unfused": [
                "standalone statistics-pass read",
                "normalized-output write",
                "ATLIF normalized-input read",
            ],
            "retained_tensor_movements": [
                "unnormalized producer-output write",
                "unnormalized fused-replay read",
            ],
            "consumer_ready": "defined only in phase 2 after the global BN barrier",
            "no_producer_p_done_admission": True,
            "moment_datapath_requirement": (
                "16 signed-sum and 16 full-precision square/sumsq updates per cycle; "
                "each 96-lane source output tile is serialized into six moment subtiles"
            ),
            "competitive_baseline_matrix": {
                "unfused": "five movements",
                "online_only": "four movements",
                "fusion_only": "three movements",
                "strongest_composable_online_plus_fusion": (
                    "two movements; cycle-identical to this proposal"
                ),
            },
            "novel_speedup_over_strongest_composable_baseline": 1.0,
        },
        "resource_model": {
            "consumer_lanes": consumer_lanes,
            "moment_lanes": moment_lanes,
            "source_output_lanes": source_output_lanes,
            "input_bits": input_bits,
            "bytes_per_value": bytes_per_value,
            "frequency_mhz": frequency_mhz,
            "dram_bandwidth_gbps": dram_bandwidth_gbps,
            "dram_bytes_per_cycle": bytes_per_cycle,
            "phase_cycles": (
                "max(source+16lane_moment_serialization,write) + "
                "max(fused_BN_ATLIF,read), summed per edge"
            ),
            "unproven_cycle_assumption": (
                "16-lane moment arithmetic, coefficient generation, BN affine, and ATLIF "
                "must pass VCS and Synopsys admission before bounded cycles are promoted"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--dependency-manifest", type=Path, required=True)
    parser.add_argument("--m18-boundaries", type=Path, required=True)
    parser.add_argument("--m17-manifest", type=Path, required=True)
    parser.add_argument("--m17-reconciliation", type=Path, required=True)
    parser.add_argument("--consumer-lanes", type=int, default=16)
    parser.add_argument("--moment-lanes", type=int, default=16)
    parser.add_argument("--input-bits", type=int, default=32)
    parser.add_argument("--frequency-mhz", type=float, default=333.333333333)
    parser.add_argument("--dram-bandwidth-gbps", type=float, default=64.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.dependency_manifest.read_text(encoding="utf-8"))
    if manifest.get("dependency_events_sha256") != sha256(args.events):
        raise ValueError("dependency event/manifest hash mismatch")
    boundaries = json.loads(args.m18_boundaries.read_text(encoding="utf-8"))
    m17_manifest = json.loads(args.m17_manifest.read_text(encoding="utf-8"))
    m17 = json.loads(args.m17_reconciliation.read_text(encoding="utf-8"))
    if m17["identities"]["oracle_manifest_sha256"] != sha256(args.m17_manifest):
        raise ValueError("M17 reconciliation/oracle manifest hash mismatch")
    model = build_model(
        load_events(args.events), manifest, boundaries, m17_manifest, m17,
        consumer_lanes=args.consumer_lanes,
        moment_lanes=args.moment_lanes,
        input_bits=args.input_bits,
        frequency_mhz=args.frequency_mhz,
        dram_bandwidth_gbps=args.dram_bandwidth_gbps,
    )
    payload = {
        "schema": "m19_dynamic_bn_barrier_fused_replay_v2",
        "revision": 3,
        "status": "PASS_EXACT_DYNAMIC_BN_BARRIER_CENSUS_BOUNDED_SOURCE_BN_ATLIF_MODEL",
        **model,
        "identities": {
            "dependency_events_sha256": sha256(args.events),
            "dependency_manifest_sha256": sha256(args.dependency_manifest),
            "m18_boundaries_sha256": sha256(args.m18_boundaries),
            "m17_manifest_sha256": sha256(args.m17_manifest),
            "m17_reconciliation_sha256": sha256(args.m17_reconciliation),
            "checkpoint_sha256": manifest["run_context"]["artifact_identity"]["checkpoint_sha256"],
            "config_sha256": manifest["run_context"]["artifact_identity"]["config_sha256"],
            "source_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "Exact one-sample census of the 13 no-running BN barriers plus a same-resource, "
            "sequential source+BN+ATLIF phase model at the stated lane/DRAM assumptions. Not "
            "full-network timing, measured DRAM timing, VCS/RTL, physical PPA, energy, FPS, "
            "cross-sequence evidence, a claim that Motion alone provides material speedup, "
            "or a speedup over the equally composable online-moments plus BN-ATLIF-fusion "
            "two-movement baseline (which is explicitly 1.000x)."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = payload["summary"]
    print(
        "PASS_M19_DYNAMIC_BN edges={} local_dense_speedup={:.6f} "
        "hybrid_dense_speedup={:.6f}".format(
            summary["edges"],
            summary["bounded_13edge_local_speedup_vs_same_width_dense"],
            summary["bounded_13edge_hybrid_speedup_vs_same_width_dense"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
