#!/usr/bin/env python3
"""Frozen-H67 premodel for the M218 tagged slice-coalesced FC2 service.

This is deliberately a pre-RTL service model.  It materializes exact K1/K8
group counts per token from the frozen payload, accounts for six 16-lane
requests per output block, and schedules fixed-latency memory requests under
outstanding, initiation-interval and accumulator-context hazard constraints.
It reports only an envelope for frontend/service temporal composition.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST = "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
EXPECTED_M172 = "ae633daee1e07f16396570c1ef757c11bac7f1f72be108f4411d75f6dcb5f6d9"
EXPECTED_M216_REPLAY = "06a7b682b55fec8100bfa134017e186411eedfba0b170b024eb29606fd437422"
EXPECTED_M216_ADMISSION = "8059a908cb47534995d928bfc95893da93e979d82a333cbc5199ab8f53a34894"
EXPECTED_CONTRACT = "b8d76adb1fcc182e9e21721d97d4e769e1059e36acbaf9d54541cdd9dbc177d7"
EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_RECORDS = 120
EXPECTED_TOKENS = 5580000
EXPECTED_EVENTS = 143894510
EXPECTED_RAW_BEATS = 36480000
EXPECTED_DESCRIPTORS = 18869376
EXPECTED_K1_GROUP_COMMANDS = 412900394
EXPECTED_K8_GROUP_COMMANDS = 73380812
EXPECTED_K8_BANK_QUEUE_ORACLE = 70657362
WIDTH = 96
SLICES = 6
SLICE_LANES = 16
WEIGHT_BYTES_PER_ACTIVE_BANK_SLICE = 16
LATENCIES = (1, 2, 4, 8, 16)
OUTSTANDING_POINTS = (1, 2, 4, 8)
II_POINTS = (1, 2)
PRIMARY = (4, 8, 1)
ORACLE = (1, 8, 1)
STAGE_GEOMETRY = {
    0: (384, 96, 2),
    1: (768, 192, 4),
    2: (1536, 384, 8),
    3: (3072, 768, 8),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path, expected, name):
    require(sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def point_name(latency, outstanding, initiation_interval):
    return "L{}_O{}_II{}".format(latency, outstanding,
                                 initiation_interval)


def service_span_table(max_requests, output_blocks, latency, outstanding,
                       initiation_interval, enforce_context_hazard=True):
    """Return cycles through final response for 0..max_requests.

    Request zero is accepted at edge zero.  A response is accepted exactly L
    edges later.  A response may free an outstanding or context slot for a new
    request on that same edge.  The returned span counts both endpoint edges,
    so one L=1 request occupies two service-model cycles.
    """
    require(latency >= 1 and outstanding >= 1
            and initiation_interval >= 1, "illegal service point")
    issue = np.zeros(max_requests, dtype=np.int64)
    span = np.zeros(max_requests + 1, dtype=np.int64)
    context_period = output_blocks * SLICES
    for index in range(max_requests):
        if index:
            value = int(issue[index - 1]) + initiation_interval
        else:
            value = 0
        if index >= outstanding:
            value = max(value, int(issue[index - outstanding]) + latency)
        if enforce_context_hazard and index >= context_period:
            value = max(value, int(issue[index - context_period]) + latency)
        issue[index] = value
        span[index + 1] = value + latency + 1
    return span


def empty_ledger():
    return {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "raw96_beats": 0,
        "nonzero96_descriptors": 0,
        "zero_tokens": 0,
        "k1_group_commands": 0,
        "k8_group_commands": 0,
        "source_terms": 0,
        "active_bank_reads": 0,
        "weight_bytes": 0,
        "result_beats": 0,
        "service": {},
        "service_without_context_hazard": {},
    }


def initialize_points(ledger):
    for latency in LATENCIES:
        for outstanding in OUTSTANDING_POINTS:
            for initiation_interval in II_POINTS:
                name = point_name(latency, outstanding,
                                  initiation_interval)
                ledger["service"][name] = {"k1_cycles": 0, "k8_cycles": 0}
                ledger["service_without_context_hazard"][name] = {
                    "k1_cycles": 0, "k8_cycles": 0}


def merge(target, source):
    for key in ("records", "tokens", "events", "raw96_beats",
                "nonzero96_descriptors", "zero_tokens",
                "k1_group_commands", "k8_group_commands", "source_terms",
                "active_bank_reads", "weight_bytes", "result_beats"):
        target[key] += int(source[key])
    for family in ("service", "service_without_context_hazard"):
        for name in target[family]:
            target[family][name]["k1_cycles"] += int(
                source[family][name]["k1_cycles"])
            target[family][name]["k8_cycles"] += int(
                source[family][name]["k8_cycles"])


def k8_base_groups(pooled, output_blocks):
    if output_blocks == 1:
        return pooled.max(axis=2).sum(axis=1, dtype=np.int64)
    groups = np.zeros(pooled.shape[0], dtype=np.int64)
    for first in range(0, pooled.shape[1], 2):
        combined = pooled[:, first].astype(np.int64)
        if first + 1 < pooled.shape[1]:
            combined = combined + pooled[:, first + 1]
        groups += combined.max(axis=1)
    return groups


def build_tables(input_width, output_blocks):
    max_requests = input_width * output_blocks * SLICES
    tables = {}
    for latency in LATENCIES:
        for outstanding in OUTSTANDING_POINTS:
            for initiation_interval in II_POINTS:
                key = (latency, outstanding, initiation_interval)
                tables[key] = (
                    service_span_table(
                        max_requests, output_blocks, latency, outstanding,
                        initiation_interval, True),
                    service_span_table(
                        max_requests, output_blocks, latency, outstanding,
                        initiation_interval, False),
                )
    return tables


def audit_record(record, payload_root, m172, chunk_tokens, table_cache):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    input_width, output_width, depth = STAGE_GEOMETRY[stage]
    require((shape[-1], output_shape[-1]) == (input_width, output_width),
            "FC2 geometry drift")
    output_blocks = output_width // WIDTH
    beats_per_token = input_width // WIDTH
    bytes_per_token = input_width // 8
    tokens = int(np.prod(shape[:-1], dtype=np.int64))
    payload = payload_root / record["relative_path"]
    require(payload.is_file() and payload.stat().st_size == record["packed_bytes"],
            "payload extent drift")
    require(sha256(payload) == record["file_sha256"], "payload SHA drift")
    raw = np.memmap(payload, dtype=np.uint8, mode="r").reshape(
        tokens, beats_per_token, bytes_per_token // beats_per_token)

    ledger = empty_ledger()
    initialize_points(ledger)
    ledger["records"] = 1
    ledger["tokens"] = tokens
    ledger["raw96_beats"] = tokens * beats_per_token
    if stage not in table_cache:
        table_cache[stage] = build_tables(input_width, output_blocks)
    tables = table_cache[stage]
    result_beats_each = output_blocks * SLICES

    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        byte_bits = m172.BYTE_BITS[np.asarray(raw[start:stop])]
        bank_counts = byte_bits.sum(axis=2, dtype=np.int16)
        beat_events = bank_counts.sum(axis=2, dtype=np.int16)
        nonzero = beat_events != 0
        descriptor_count = nonzero.sum(axis=1, dtype=np.int16)
        positions = np.cumsum(nonzero, axis=1, dtype=np.int16) - 1
        row, beat = np.nonzero(nonzero)
        maximum_windows = (beats_per_token + depth - 1) // depth
        pooled = np.zeros((stop - start, maximum_windows, 8),
                          dtype=np.int16)
        if row.size:
            window = (positions[row, beat] // depth).astype(np.intp)
            np.add.at(pooled, (row, window), bank_counts[row, beat])

        event_per_token = beat_events.sum(axis=1, dtype=np.int64)
        k1_groups = event_per_token * output_blocks
        k8_groups = k8_base_groups(pooled, output_blocks) * output_blocks
        require(np.all(k8_groups <= k1_groups), "K8 group regression")
        k1_requests = k1_groups * SLICES
        k8_requests = k8_groups * SLICES
        common_tail = result_beats_each + 1

        ledger["events"] += int(event_per_token.sum(dtype=np.int64))
        ledger["nonzero96_descriptors"] += int(
            descriptor_count.sum(dtype=np.int64))
        ledger["zero_tokens"] += int(np.count_nonzero(event_per_token == 0))
        ledger["k1_group_commands"] += int(k1_groups.sum(dtype=np.int64))
        ledger["k8_group_commands"] += int(k8_groups.sum(dtype=np.int64))
        source_terms = int(k1_groups.sum(dtype=np.int64))
        ledger["source_terms"] += source_terms
        ledger["active_bank_reads"] += source_terms * SLICES
        ledger["weight_bytes"] += (
            source_terms * SLICES * WEIGHT_BYTES_PER_ACTIVE_BANK_SLICE)
        ledger["result_beats"] += (stop - start) * result_beats_each

        for key, (hazard_table, no_hazard_table) in tables.items():
            name = point_name(*key)
            ledger["service"][name]["k1_cycles"] += int(
                hazard_table[k1_requests].sum(dtype=np.int64)
                + (stop - start) * common_tail)
            ledger["service"][name]["k8_cycles"] += int(
                hazard_table[k8_requests].sum(dtype=np.int64)
                + (stop - start) * common_tail)
            ledger["service_without_context_hazard"][name][
                "k1_cycles"] += int(
                    no_hazard_table[k1_requests].sum(dtype=np.int64)
                    + (stop - start) * common_tail)
            ledger["service_without_context_hazard"][name][
                "k8_cycles"] += int(
                    no_hazard_table[k8_requests].sum(dtype=np.int64)
                    + (stop - start) * common_tail)

    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return stage, ledger


def decorate_points(aggregate, frontend):
    points = {}
    frontend_k1 = int(frontend["aggregate"]["k1_cycles"])
    frontend_k8 = int(frontend["aggregate"]["k8_cycles"])
    for name, cycles in aggregate["service"].items():
        k1 = int(cycles["k1_cycles"])
        k8 = int(cycles["k8_cycles"])
        no_hazard = aggregate["service_without_context_hazard"][name]
        k1_lower = max(frontend_k1, k1)
        k1_upper = frontend_k1 + k1
        k8_lower = max(frontend_k8, k8)
        k8_upper = frontend_k8 + k8
        points[name] = {
            "service_k1_cycles": k1,
            "service_k8_cycles": k8,
            "service_k8_speedup_vs_k1": fraction(k1, k8),
            "k1_context_hazard_cycles": k1 - int(no_hazard["k1_cycles"]),
            "k8_context_hazard_cycles": k8 - int(no_hazard["k8_cycles"]),
            "composed_k1_cycle_interval": [k1_lower, k1_upper],
            "composed_k8_cycle_interval": [k8_lower, k8_upper],
            "conservative_composed_speedup_lower_bound": fraction(
                k1_lower, k8_upper),
            "composed_speedup_upper_bound": fraction(k1_upper, k8_lower),
        }
    oracle_k8 = points[point_name(*ORACLE)]["service_k8_cycles"]
    for point in points.values():
        point["k8_throughput_retention_vs_l1_o8_ii1_oracle"] = fraction(
            oracle_k8, point["service_k8_cycles"])
    return points


def slim_stage(stage, ledger):
    primary_name = point_name(*PRIMARY)
    oracle_name = point_name(*ORACLE)
    result = {key: ledger[key] for key in (
        "records", "tokens", "events", "raw96_beats",
        "nonzero96_descriptors", "zero_tokens", "k1_group_commands",
        "k8_group_commands", "source_terms", "active_bank_reads",
        "weight_bytes", "result_beats")}
    result["output_blocks"] = STAGE_GEOMETRY[stage][1] // WIDTH
    result["primary_service"] = ledger["service"][primary_name]
    result["oracle_service"] = ledger["service"][oracle_name]
    result["primary_service_speedup"] = fraction(
        result["primary_service"]["k1_cycles"],
        result["primary_service"]["k8_cycles"])
    result["primary_context_hazard_cycles"] = {
        cap: (ledger["service"][primary_name][cap + "_cycles"]
              - ledger["service_without_context_hazard"][primary_name][
                  cap + "_cycles"])
        for cap in ("k1", "k8")
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m216-replay", required=True, type=Path)
    parser.add_argument("--m216-admission", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST, "manifest drift")
    require(sha256(args.m216_replay) == EXPECTED_M216_REPLAY,
            "M216 replay drift")
    require(sha256(args.m216_admission) == EXPECTED_M216_ADMISSION,
            "M216 admission drift")
    require(sha256(args.contract) == EXPECTED_CONTRACT, "contract drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359, "docs359 drift")
    m172 = load_pinned(args.m172_analyzer, EXPECTED_M172,
                       "m172_pinned_m218")
    frontend = json.loads(args.m216_replay.read_text())
    admission = json.loads(args.m216_admission.read_text())
    require(frontend["status"]
            == "PASS_EXACT_FROZEN_H67_SCOPE_MATCHED_K1_K8_FRONTEND_REPLAY",
            "M216 replay not admitted")
    require(admission["status"]
            == "ADMITTED_STANDALONE_SPARSE_FC2_FRONTEND",
            "M216 admission not admitted")
    require(frontend["aggregate"]["k8_service_cycle_floor"]
            == EXPECTED_K8_BANK_QUEUE_ORACLE,
            "K8 independent bank-queue oracle drift")
    manifest = json.loads(args.manifest.read_text())
    records = [record for record in manifest["records"]
               if record["operator"] == "Linear"
               and ".mlp.fc2" in record["name"]]
    require(len(records) == EXPECTED_RECORDS, "FC2 record count drift")

    aggregate = empty_ledger()
    initialize_points(aggregate)
    per_stage = defaultdict(empty_ledger)
    for stage in STAGE_GEOMETRY:
        initialize_points(per_stage[stage])
    table_cache = {}
    for ordinal, record in enumerate(records, start=1):
        stage, ledger = audit_record(record, args.payload_root, m172,
                                     args.chunk_tokens, table_cache)
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print("[M218 premodel] {}/120".format(ordinal), flush=True)

    require(aggregate["tokens"] == EXPECTED_TOKENS, "token identity drift")
    require(aggregate["events"] == EXPECTED_EVENTS, "event identity drift")
    require(aggregate["raw96_beats"] == EXPECTED_RAW_BEATS,
            "raw beat identity drift")
    require(aggregate["nonzero96_descriptors"] == EXPECTED_DESCRIPTORS,
            "descriptor identity drift")
    require(aggregate["k1_group_commands"] == EXPECTED_K1_GROUP_COMMANDS,
            "K1 group identity drift")
    require(aggregate["k8_group_commands"] == EXPECTED_K8_GROUP_COMMANDS,
            "K8 group identity drift: observed={} expected={}".format(
                aggregate["k8_group_commands"],
                EXPECTED_K8_GROUP_COMMANDS))
    require(aggregate["source_terms"] == EXPECTED_K1_GROUP_COMMANDS,
            "source-term identity drift")
    require(aggregate["active_bank_reads"]
            == EXPECTED_K1_GROUP_COMMANDS * SLICES,
            "active-bank-read conservation failure")
    points = decorate_points(aggregate, frontend)
    primary = points[point_name(*PRIMARY)]
    gate = {
        "primary_k8_service_speedup_vs_k1_ge_2p5":
            primary["service_k8_speedup_vs_k1"]["float"] >= 2.5,
        "primary_k8_throughput_retention_vs_oracle_ge_0p85":
            primary[
                "k8_throughput_retention_vs_l1_o8_ii1_oracle"]["float"]
            >= 0.85,
        "conservative_composed_speedup_lower_bound_ge_2p5":
            primary[
                "conservative_composed_speedup_lower_bound"]["float"]
            >= 2.5,
        "k1_k8_active_bank_reads_exactly_equal": True,
        "k1_k8_weight_bytes_exactly_equal": True,
        "context_hazard_enforced": True,
        "all_payload_sha_size_popcount_checked": True,
    }
    require(all(gate.values()), "M218 premodel GO gate failed")

    identity_keys = ("records", "tokens", "events", "raw96_beats",
                     "nonzero96_descriptors", "zero_tokens",
                     "k1_group_commands", "k8_group_commands",
                     "source_terms", "active_bank_reads", "weight_bytes",
                     "result_beats")
    result = {
        "schema": "m218_h67_fc2_tagged_slice_service_premodel_v1",
        "status": "PASS_FROZEN_H67_TAGGED_SLICE_SERVICE_PREMODEL_GO",
        "identity": {
            "analyzer_start_sha256": script_start,
            "manifest_sha256": EXPECTED_MANIFEST,
            "m172_analyzer_sha256": EXPECTED_M172,
            "m216_replay_sha256": EXPECTED_M216_REPLAY,
            "m216_admission_sha256": EXPECTED_M216_ADMISSION,
            "contract_sha256": EXPECTED_CONTRACT,
            "docs359_sha256": EXPECTED_DOCS359,
            "all_120_payload_sha_size_popcount_checked": True,
        },
        "architecture": {
            "physical_weight_banks": 8,
            "slice_lanes": SLICE_LANES,
            "slices_per_output_block": SLICES,
            "maximum_k8_response_bits": 1024,
            "accumulator_context_bits_per_slice": 384,
            "request_order": "source-group/output-block/slice",
            "context_hazard_period_requests": "6 x output_blocks",
            "same_cycle_response_slot_reuse": True,
        },
        "aggregate_work": {key: aggregate[key] for key in identity_keys},
        "ordered_grouping_overhead": {
            "m216_ordered_k8_group_commands":
                aggregate["k8_group_commands"],
            "independent_bank_queue_oracle_group_commands":
                EXPECTED_K8_BANK_QUEUE_ORACLE,
            "ordering_overhead_group_commands":
                aggregate["k8_group_commands"]
                - EXPECTED_K8_BANK_QUEUE_ORACLE,
            "ordering_overhead_percent_vs_oracle":
                (float(aggregate["k8_group_commands"])
                 / float(EXPECTED_K8_BANK_QUEUE_ORACLE) - 1.0) * 100.0,
        },
        "memory_conservation": {
            "k1_active_bank_reads": aggregate["active_bank_reads"],
            "k8_active_bank_reads": aggregate["active_bank_reads"],
            "k1_weight_bytes": aggregate["weight_bytes"],
            "k8_weight_bytes": aggregate["weight_bytes"],
            "exactly_equal": True,
        },
        "points": points,
        "primary_point": point_name(*PRIMARY),
        "oracle_point": point_name(*ORACLE),
        "go_gate": gate,
        "per_stage": {str(stage): slim_stage(stage, per_stage[stage])
                      for stage in STAGE_GEOMETRY},
        "claim_boundary": {
            "exact_frozen_group_and_memory_work": True,
            "fixed_latency_in_order_service_premodel": True,
            "context_hazard_aware": True,
            "service_cycles": True,
            "frontend_service_temporal_composition": False,
            "composed_cycle_envelope_only": True,
            "rtl_exists": False,
            "synopsys_vcs_calibrated": False,
            "out_of_order_response": False,
            "complete_fc2": False,
            "complete_ffn": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "analyzer mutated during run")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "primary": primary,
        "go_gate": gate,
        "memory_conservation": result["memory_conservation"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
