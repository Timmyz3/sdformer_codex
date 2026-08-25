#!/usr/bin/env python3
"""Exact M229 fixed-latency recurrence on the frozen H67 FC1 population.

This closes the gap between M229's directed service-island observations and
the 100-record M51 trace.  It does not model a physical SRAM macro: accepted
weight requests are one per cycle, responses are in order after a frozen one-
or two-cycle latency, and neither the response nor Acc19 port is stalled.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

import analyze_m224_h67_fc1_parent_delta_bank_service_screen as m224
import analyze_m225_h67_fc1_held_weight_context_multicast_screen as m225


EXPECTED = {
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "m225_seal": "628150798123f70345e92b9dd01b376a8da9e9cbd3878fadd0250b760b72a83a",
    "m226_seal": "09f43c9ec47f9ae8276aeb814f881521aebcaa6af778c5844e66fa2b205c3568",
    "m226_review_seal": "6c77665dfbae822186fe656b9d746e4e94b9aaaf5f1c3311389389275fe3c898",
    "m229_vcs_seal": "7591869a0e519f32e309794a5f66d43bfd1b57d059f4cc2261d9be4ae5f9186e",
    "m229_dc_seal": "6a62aaed1096946aaf337334566494787e975fd092478991895ed1e8be7d5e75",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
FANOUTS = (1, 2, 4)
LATENCIES = (1, 2)
CONTEXT_GROUP = 8
SOURCE_CHUNK = 32
DEPTH = 4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ratio(numerator, denominator):
    require(int(denominator) > 0, "zero ratio denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def recurrence_period(service_lengths, response_latency):
    """M229 edge recurrence including the no-hot-restart fence.

    Edge zero accepts the header.  A descriptor may enter at edge one, its
    request is visible no earlier than the following edge, and a response is
    usable by replay one edge after it is registered.  The next header cannot
    be accepted on the done-accept edge, so the returned period ends at the
    earliest next-header edge.
    """
    if not service_lengths:
        return 1
    enqueue = []
    request = []
    pop = []
    for index, duration in enumerate(service_lengths):
        require(int(duration) >= 1, "zero descriptor service")
        input_ready = 1 if index == 0 else enqueue[-1] + 1
        credit_ready = pop[index - DEPTH] if index >= DEPTH else 1
        enqueue_edge = max(input_ready, credit_ready)
        request_port = 2 if index == 0 else request[-1] + 1
        request_edge = max(request_port, enqueue_edge + 1)
        response_usable = request_edge + int(response_latency) + 1
        replay_order = 0 if index == 0 else pop[-1] + 1
        start_edge = max(response_usable, replay_order)
        enqueue.append(enqueue_edge)
        request.append(request_edge)
        pop.append(start_edge + int(duration) - 1)
    return pop[-1] + 2


def prove_short_latency_closed_form():
    rng = random.Random(2300825)
    checked = 0
    extrema = {str(latency): {"min_length": None, "max_length": 0}
               for latency in LATENCIES}
    directed = [
        [1], [8], [1, 1, 1, 1], [2, 1, 2, 1, 2],
        [1] * 32, [2] * 32, [8, 1, 7, 2, 6, 3, 5, 4],
    ]
    cases = list(directed)
    for _ in range(20000):
        length = rng.randint(1, 384)
        cases.append([rng.randint(1, 8) for _ in range(length)])
    for service in cases:
        for latency in LATENCIES:
            observed = recurrence_period(service, latency)
            expected = sum(service) + latency + 4
            require(observed == expected,
                    "closed-form recurrence mismatch L{} n{}: {} != {}".
                    format(latency, len(service), observed, expected))
            row = extrema[str(latency)]
            row["min_length"] = (len(service) if row["min_length"] is None
                                 else min(row["min_length"], len(service)))
            row["max_length"] = max(row["max_length"], len(service))
            checked += 1
    return {"cases_times_latencies": checked, "seed": 2300825,
            "extrema": extrema, "zero_mismatch": True}


def grouped_occurrence(residual):
    time, batch, height, width, channel_bytes = residual.shape
    rows = residual.reshape(time * batch * height, width, channel_bytes)
    groups_per_row = m224.ceil_div(width, CONTEXT_GROUP)
    padded_width = groups_per_row * CONTEXT_GROUP
    if padded_width != width:
        padded = np.zeros((rows.shape[0], padded_width, channel_bytes),
                          dtype=np.uint8)
        padded[:, :width, :] = rows
        rows = padded
    grouped = rows.reshape(rows.shape[0], groups_per_row, CONTEXT_GROUP,
                           channel_bytes)
    # Little-endian unpack preserves source index byte*8+bit.
    bitplanes = np.unpackbits(grouped, axis=-1, bitorder="little")
    occurrence = bitplanes.sum(axis=2, dtype=np.uint8)
    return occurrence.reshape(-1, occurrence.shape[-1])


def exact_record_stats(record, residual, output_blocks):
    occurrence = grouped_occurrence(residual)
    union = occurrence != 0
    group_count = int(occurrence.shape[0])
    nonempty_groups = int(np.count_nonzero(np.any(union, axis=1)))
    unique_sources = int(np.count_nonzero(union))
    source_occurrences = int(occurrence.sum(dtype=np.uint64))
    chunks = m224.ceil_div(occurrence.shape[1], SOURCE_CHUNK)
    padded_sources = chunks * SOURCE_CHUNK
    if padded_sources != occurrence.shape[1]:
        padded = np.zeros((group_count, padded_sources), dtype=bool)
        padded[:, :occurrence.shape[1]] = union
        union = padded
    active_chunks = int(np.count_nonzero(
        np.any(union.reshape(group_count, chunks, SOURCE_CHUNK), axis=2)))
    service = {}
    occurrence16 = occurrence.astype(np.uint16)
    for fanout in FANOUTS:
        service[fanout] = int(
            ((occurrence16 + fanout - 1) // fanout).sum(dtype=np.uint64))
    streams = group_count * int(output_blocks)
    nonempty_streams = nonempty_groups * int(output_blocks)
    return {
        "sample_id": int(record["sample_id"]),
        "module_index": int(record["module_index"]),
        "name": record["name"],
        "input_channels": int(record["input_shape"][-1]),
        "output_blocks": int(output_blocks),
        "group_streams": streams,
        "nonempty_group_streams": nonempty_streams,
        "empty_group_streams": streams - nonempty_streams,
        "unique_source_weight_reads": unique_sources * int(output_blocks),
        "source_occurrences": source_occurrences,
        "active_32source_chunk_streams": active_chunks * int(output_blocks),
        "dense_32source_chunk_streams": group_count * chunks * int(output_blocks),
        "service_cycles": dict((str(fanout), service[fanout] * int(output_blocks))
                               for fanout in FANOUTS),
    }


def add_stats(total, row):
    for key in ("group_streams", "nonempty_group_streams",
                "empty_group_streams", "unique_source_weight_reads",
                "source_occurrences", "active_32source_chunk_streams",
                "dense_32source_chunk_streams"):
        total[key] = total.get(key, 0) + int(row[key])
    service = total.setdefault("service_cycles", dict((str(f), 0)
                                                       for f in FANOUTS))
    for fanout in FANOUTS:
        service[str(fanout)] += int(row["service_cycles"][str(fanout)])


def parse_dc_receipts(run_dir):
    variants = {}
    for fanout in FANOUTS:
        values = {}
        path = run_dir / "f{}".format(fanout) / "RUN_COMPLETE.txt"
        require(path.is_file(), "missing M229 DC receipt {}".format(path))
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        require(values.get("exact_sha") == "true", "non-exact M229 DC")
        require(values.get("setup_met") == "true" and
                values.get("hold_met") == "true", "M229 DC timing failure")
        variants[str(fanout)] = {
            "logic_area_um2": float(values["cell_area_um2"]),
            "cell_count": int(values["cell_count"]),
            "sequential_cells": int(values["sequential_cells"]),
            "setup_worst_slack_ns": float(values["setup_worst_slack_ns"]),
            "hold_worst_slack_ns": float(values["hold_worst_slack_ns"]),
            "acc_capacity_port_cut_bits": int(values["acc_capacity_port_cut_bits"]),
            "macro_count": int(values["macro_count"]),
        }
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m225-result", required=True, type=Path)
    parser.add_argument("--m225-seal", required=True, type=Path)
    parser.add_argument("--m226-seal", required=True, type=Path)
    parser.add_argument("--m226-review-seal", required=True, type=Path)
    parser.add_argument("--m229-vcs-seal", required=True, type=Path)
    parser.add_argument("--m229-dc-seal", required=True, type=Path)
    parser.add_argument("--m229-dc-run", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    identities = {
        "manifest": sha256(args.manifest),
        "m225_seal": sha256(args.m225_seal),
        "m226_seal": sha256(args.m226_seal),
        "m226_review_seal": sha256(args.m226_review_seal),
        "m229_vcs_seal": sha256(args.m229_vcs_seal),
        "m229_dc_seal": sha256(args.m229_dc_seal),
        "docs359": sha256(args.docs359),
    }
    require(identities == EXPECTED, "frozen input identity drift")
    manifest = json.loads(args.manifest.read_text())
    m225_result = json.loads(args.m225_result.read_text())
    records = [row for row in manifest["records"]
               if int(row["module_index"]) in m224.SELECTED_MODULE_INDICES]
    require(len(records) == 100, "expected 100 binary FC1 records")
    recurrence_proof = prove_short_latency_closed_form()
    aggregates = dict((mode, {}) for mode in m224.MODE_ORDER)
    per_record = []
    payload_hashes = []
    for ordinal, record in enumerate(sorted(
            records, key=lambda row: (int(row["sample_id"]),
                                      int(row["module_index"])))):
        current = m224.decode_record(record, args.payload_root)
        output_blocks = m224.ceil_div(int(record["output_shape"][-1]), 96)
        record_result = {"ordinal": ordinal, "sample_id": int(record["sample_id"]),
                         "module_index": int(record["module_index"]),
                         "name": record["name"], "modes": {}}
        payload_hashes.append({"relative_path": record["relative_path"],
                               "sha256": record["file_sha256"]})
        for mode in m224.MODE_ORDER:
            residual, choice = m224.select_parent(current, mode)
            row = exact_record_stats(record, residual, output_blocks)
            require(int(row["source_occurrences"]) == int(choice["source_events"]),
                    "signed source conservation drift")
            record_result["modes"][mode] = row
            add_stats(aggregates[mode], row)
            del residual
        per_record.append(record_result)
        del current

    ledger_mismatches = []
    for mode in m224.MODE_ORDER:
        total = aggregates[mode]
        for fanout in FANOUTS:
            prior = m225_result["aggregate_points"][
                "{}_K8_F{}".format(mode, fanout)]
            checks = {
                "group_streams": int(prior["group_descriptor_cycles"]),
                "unique_source_weight_reads": int(prior["weight_vector_reads"]),
                "service_cycles.{}".format(fanout): int(prior["service_cycles"]),
            }
            observed = {
                "group_streams": int(total["group_streams"]),
                "unique_source_weight_reads": int(total["unique_source_weight_reads"]),
                "service_cycles.{}".format(fanout):
                    int(total["service_cycles"][str(fanout)]),
            }
            for key, expected in checks.items():
                if observed[key] != expected:
                    ledger_mismatches.append({"mode": mode, "fanout": fanout,
                                              "field": key, "observed": observed[key],
                                              "expected": expected})
        total["active_chunk_fraction"] = (
            float(total["active_32source_chunk_streams"]) /
            float(total["dense_32source_chunk_streams"]))
        total["active_chunk_reduction"] = ratio(
            total["dense_32source_chunk_streams"],
            total["active_32source_chunk_streams"])
    require(not ledger_mismatches, "M225 ledger mismatch")

    dc = parse_dc_receipts(args.m229_dc_run)
    points = {}
    raw_k1_reference = int(m225_result["aggregate_points"]["raw_K1_F1"]
                           ["serial_cycles"])
    for mode in m224.MODE_ORDER:
        total = aggregates[mode]
        points[mode] = {}
        for latency in LATENCIES:
            latency_points = {}
            for fanout in FANOUTS:
                prior = m225_result["aggregate_points"][
                    "{}_K8_F{}".format(mode, fanout)]
                fixed = (int(prior["serial_cycles"]) -
                         int(prior["group_descriptor_cycles"]) -
                         int(prior["service_cycles"]))
                engine = (int(total["service_cycles"][str(fanout)]) +
                          int(total["nonempty_group_streams"]) * (latency + 4) +
                          int(total["empty_group_streams"]))
                cycles = fixed + engine
                latency_points[str(fanout)] = {
                    "fixed_non_engine_cycles": fixed,
                    "m229_engine_cycles": engine,
                    "trace_cycles": cycles,
                    "speedup_vs_raw_k1_f1": ratio(raw_k1_reference, cycles),
                    "logic_area_um2": dc[str(fanout)]["logic_area_um2"],
                }
            f1_cycles = latency_points["1"]["trace_cycles"]
            f1_area = dc["1"]["logic_area_um2"]
            for fanout in FANOUTS:
                point = latency_points[str(fanout)]
                speed = float(f1_cycles) / float(point["trace_cycles"])
                area_ratio = point["logic_area_um2"] / f1_area
                point["speedup_vs_same_mode_k8_f1"] = speed
                point["logic_area_ratio_vs_f1"] = area_ratio
                point["trace_throughput_per_logic_area_vs_f1"] = speed / area_ratio
            points[mode][str(latency)] = latency_points

    latency2_raw = points["raw"]["2"]
    latency2_spatial = points["spatial"]["2"]
    output = {
        "schema": "m230_h67_fc1_m229_fixed_latency_trace_recurrence_v1",
        "status": "PASS_EXACT_TRACE_RECURRENCE_LOGIC_ONLY_PARETO",
        "identity": identities,
        "population": {"records": len(records), "samples": 10, "modules": 10,
                       "stage3_nonbinary_fc1_modules": 2,
                       "stage3_policy": "conventional fallback",
                       "payloads": payload_hashes},
        "recurrence": {
            "credits": DEPTH,
            "request_accept_interval_cycles": 1,
            "response_latencies_cycles": list(LATENCIES),
            "nonempty_period": "service + latency + 4",
            "empty_bypass_period": 1,
            "hot_restart": False,
            "no_backpressure": True,
            "random_and_directed_closed_form_check": recurrence_proof,
        },
        "aggregate_trace": aggregates,
        "points": points,
        "dc_logic_only": dc,
        "ledger_mismatches": ledger_mismatches,
        "decision": {
            "latency2_raw_f2_speedup_vs_k8_f1":
                latency2_raw["2"]["speedup_vs_same_mode_k8_f1"],
            "latency2_raw_f4_speedup_vs_k8_f1":
                latency2_raw["4"]["speedup_vs_same_mode_k8_f1"],
            "latency2_raw_f2_throughput_per_logic_area":
                latency2_raw["2"]["trace_throughput_per_logic_area_vs_f1"],
            "latency2_raw_f4_throughput_per_logic_area":
                latency2_raw["4"]["trace_throughput_per_logic_area_vs_f1"],
            "latency2_spatial_f4_composed_speedup_vs_raw_k1_f1":
                latency2_spatial["4"]["speedup_vs_raw_k1_f1"],
            "f2_role": "logic-area-efficiency point",
            "f4_role": "absolute-throughput point",
            "active_chunk_skip_is_next_structural_target": True,
        },
        "admission": {
            "exact_100_record_trace_recurrence": True,
            "all_payload_sha_verified": True,
            "m225_ledger_zero_mismatch": True,
            "m229_vcs_bound": True,
            "m229_matched_dc_bound": True,
            "macro_complete": False,
            "saif_ptpx": False,
            "complete_fc1": False,
            "complete_ffn": False,
            "system_speedup": False,
            "headline": False,
        },
        "claim_boundary": {
            "ratios_are": "fixed-latency no-stall binary-FC1 trace recurrence",
            "ratios_are_not": ["physical SRAM timing", "complete FC1 speedup",
                               "complete FFN speedup", "system speedup",
                               "macro-aware energy or PPA"],
            "directed_m229_service_ratios_not_reused_as_trace_ratios": True,
        },
        "per_record": per_record,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / \
        "m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    readme = args.output_dir / "README.md"
    readme.write_text(
        "# M230 H67 FC1 M229 fixed-latency trace recurrence\n\n"
        "All 100 frozen binary-FC1 payloads were re-decoded and matched the "
        "M225 group, read and service ledger with zero mismatch.  At a two-cycle "
        "accepted-request-to-response latency, raw K8 F2/F4 are {:.6f}x/{:.6f}x "
        "versus raw K8 F1; logic-only throughput/area is {:.6f}x/{:.6f}x.  "
        "Spatial-parent F4 composes to {:.6f}x versus raw K1/F1.\n\n"
        "These are no-stall trace-recurrence results, not physical SRAM, complete "
        "FC1/FFN, or system speedups.  Stage-3 nonbinary FC1 stays conventional.\n".
        format(latency2_raw["2"]["speedup_vs_same_mode_k8_f1"],
               latency2_raw["4"]["speedup_vs_same_mode_k8_f1"],
               latency2_raw["2"]["trace_throughput_per_logic_area_vs_f1"],
               latency2_raw["4"]["trace_throughput_per_logic_area_vs_f1"],
               latency2_spatial["4"]["speedup_vs_raw_k1_f1"]["float"]))
    print("PASS M230 raw-L2 F2={:.6f} F4={:.6f} TPA={:.6f}/{:.6f}".
          format(latency2_raw["2"]["speedup_vs_same_mode_k8_f1"],
                 latency2_raw["4"]["speedup_vs_same_mode_k8_f1"],
                 latency2_raw["2"]["trace_throughput_per_logic_area_vs_f1"],
                 latency2_raw["4"]["trace_throughput_per_logic_area_vs_f1"]))


if __name__ == "__main__":
    main()
