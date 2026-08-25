#!/usr/bin/env python3
"""Exact heldout cycle DSE for sparse-mask-to-K4 descriptor production.

M132 assumes the producer fills a ping-pong descriptor bank one raw source per
cycle.  M140 evaluates a standalone descriptorizer that accepts already-sparse
nonzero 16-bit source masks and extracts four ordered source IDs per lane per
cycle.  The correction consumer, PWP512 service, weights, window recurrence and
all heldout records remain unchanged.  A dual-lane extractor is included only
as an upper-bound DSE; neither variant is RTL or physical evidence here.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M132_SCRIPT = HW / "system_simulator/scripts/analyze_m132_dualrow512_pwp_compact_k4_schedule.py"
M132_RESULT = HW / "results/m132_dualrow512_pwp_compact_k4_schedule_r1_20260824/m132_dualrow512_pwp_compact_k4_schedule.json"
M132_OVERLAY = HW / "contracts/m132_r1_independent_review_correction_overlay_r1_20260824.json"
M109_RESULT = HW / "results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/m109_r2_window_storage_dual_timeline_frontier.json"

EXPECTED_SHA256 = {
    "m132_script": "f140b6b72559f04cdac374eaf696c3f6650b20d3b00bd580419b88494d89c952",
    "m132_result": "f74444576ec487b9b1034aced7add0da868a9dea5d4185e0a62c1e33fe1ad755",
    "m132_overlay": "82ca925af73a7fecb55c4a47d6d95fbba5eb5c22698a2c27695b6a68fbda36a9",
    "m109_result": "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, path)
    require(spec is not None and spec.loader is not None,
            "cannot load module " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def producer_result(schedule, fixed_baseline_service_tokens,
                    source_events, packed_descriptors, lanes):
    result = schedule.result(fixed_baseline_service_tokens)
    payload_tokens = int(result.pop("raw_event_tokens"))
    result["source_events"] = int(source_events)
    result["packed_k4_descriptors"] = int(packed_descriptors)
    result["descriptorizer_lanes"] = int(lanes)
    result["producer_fill_payload_tokens"] = payload_tokens
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M140 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())

    frozen_paths = {
        "m132_script": M132_SCRIPT,
        "m132_result": M132_RESULT,
        "m132_overlay": M132_OVERLAY,
        "m109_result": M109_RESULT,
    }
    for label, path in frozen_paths.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)

    m132 = load_module("m140_frozen_m132", M132_SCRIPT)
    for label, path in {
        **{
            key: getattr(m132, {
                "m122_script": "M122_SCRIPT",
                "m122_result": "M122_RESULT",
                "m129_result": "M129_RESULT",
                "m129_overlay": "M129_OVERLAY",
                "m129_review": "M129_REVIEW",
                "m131_receipt": "M131_RECEIPT",
                "m109_script": "M109_SCRIPT",
                "m108_script": "M108_SCRIPT",
                "m105_script": "M105_SCRIPT",
                "m40_manifest": "M40_MANIFEST",
                "m72_result": "M72_RESULT",
                "m41_result": "M41_RESULT",
            }[key]) for key in m132.EXPECTED_SHA256
        }
    }.items():
        require(sha256(path) == m132.EXPECTED_SHA256[label],
                "M132 transitive input identity drift: " + label)

    frozen_m132 = strict_json(M132_RESULT)
    m122 = load_module("m140_frozen_m122", m132.M122_SCRIPT)
    m109 = load_module("m140_frozen_m109", m132.M109_SCRIPT)
    m108 = load_module("m140_frozen_m108", m132.M108_SCRIPT)
    m105 = load_module("m140_frozen_m105", m132.M105_SCRIPT)
    manifest = strict_json(m132.M40_MANIFEST)
    m72 = strict_json(m132.M72_RESULT)
    m41 = strict_json(m132.M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "heldout record extent drift")

    frozen_m109 = strict_json(M109_RESULT)
    w384 = next(row for row in frozen_m109["frontier"]
                if int(row["window_rows"]) == m132.WINDOW_ROWS)
    fixed_baseline_service_tokens = (
        int(w384["dual_timeline_recurrence"]
            ["fair_fixed8_baseline_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_commit_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_pipeline_flush_cycles"]))

    popcount = np.fromiter(
        (int(value).bit_count() for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    schedules = {
        "raw_source_one_per_cycle": m122.FoldSchedule(),
        "sparse_mask_k4x1": m122.FoldSchedule(),
        "sparse_mask_k4x2_upper_bound": m122.FoldSchedule(),
    }
    totals = Counter()
    starts = np.arange(0, m132.ROWS, m132.WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + m132.WINDOW_ROWS, m132.ROWS)

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, _, pwp512_rows, _ = m132.build_record_rows(
            m105, masks, record["operator_index"], centers, widths, popcount)
        source_counts = popcount[event_masks]
        event_prefix = np.concatenate((
            np.zeros((m132.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(source_counts.sum(axis=2, dtype=np.uint16),
                      axis=1, dtype=np.uint32)), axis=1)
        folded_per_row = ((source_counts.astype(np.uint16) + m132.FOLD - 1)
                          // m132.FOLD).sum(axis=2, dtype=np.uint16)
        folded_prefix = np.concatenate((
            np.zeros((m132.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(folded_per_row, axis=1, dtype=np.uint32)), axis=1)
        pwp512_prefix = np.concatenate((
            np.zeros((m132.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(pwp512_rows, axis=1, dtype=np.uint32)), axis=1)
        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)

        for window_index, (start, end) in enumerate(zip(starts, ends)):
            events = event_prefix[:, end] - event_prefix[:, start]
            folded = folded_prefix[:, end] - folded_prefix[:, start]
            pwp512 = pwp512_prefix[:, end] - pwp512_prefix[:, start]
            for partition in range(m132.PARTITIONS):
                event_count = int(events[partition])
                folded_count = int(folded[partition])
                group_count = int(groups[partition, window_index])
                pwp_tokens = int(pwp512[partition])
                startup = int(folded_count != 0)
                correction_cycles = folded_count + startup
                schedules["raw_source_one_per_cycle"].descriptor(
                    window_index, partition, event_count, group_count,
                    pwp_tokens, correction_cycles)
                schedules["sparse_mask_k4x1"].descriptor(
                    window_index, partition, folded_count, group_count,
                    pwp_tokens, correction_cycles)
                schedules["sparse_mask_k4x2_upper_bound"].descriptor(
                    window_index, partition, (folded_count + 1) // 2,
                    group_count, pwp_tokens, correction_cycles)
                totals["source_events"] += event_count
                totals["packed_k4_descriptors"] += folded_count
                totals["pwp512_tokens"] += pwp_tokens
        print("[M140 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    recurrence = {
        "raw_source_one_per_cycle": producer_result(
            schedules["raw_source_one_per_cycle"],
            fixed_baseline_service_tokens, totals["source_events"],
            totals["packed_k4_descriptors"], 0),
        "sparse_mask_k4x1": producer_result(
            schedules["sparse_mask_k4x1"],
            fixed_baseline_service_tokens, totals["source_events"],
            totals["packed_k4_descriptors"], 1),
        "sparse_mask_k4x2_upper_bound": producer_result(
            schedules["sparse_mask_k4x2_upper_bound"],
            fixed_baseline_service_tokens, totals["source_events"],
            totals["packed_k4_descriptors"], 2),
    }
    baseline = recurrence["raw_source_one_per_cycle"]
    k4x1 = recurrence["sparse_mask_k4x1"]
    k4x2 = recurrence["sparse_mask_k4x2_upper_bound"]
    require(baseline["candidate_cycles"] == frozen_m132["cycle_models"]
            ["compact_k4_dualrow_pwp512"]["candidate_cycles"],
            "M132 dualrow baseline cycle replay drift")
    require(totals["source_events"]
            == frozen_m132["exact_work"]["events"],
            "source-event conservation drift")
    require(totals["packed_k4_descriptors"]
            == frozen_m132["exact_work"]["k4_descriptors"],
            "K4 descriptor conservation drift")
    require(totals["pwp512_tokens"]
            == frozen_m132["exact_work"]["pwp512_tokens"],
            "PWP512 token conservation drift")
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M140 analyzer changed during execution")

    payload = {
        "schema": "m140_sparse_mask_k4_descriptorizer_dse_v1",
        "status": "PASS_EXACT_HELDOUT_DESCRIPTOR_PRODUCER_CYCLE_DSE",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED_SHA256,
            "m132_transitive_inputs_sha256": m132.EXPECTED_SHA256,
            "heldout_samples": list(range(5, 10)),
            "heldout_records": len(heldout),
        },
        "exact_work": dict(totals),
        "cycle_models": recurrence,
        "comparisons": {
            "k4x1_speedup_vs_raw_source_fill":
                baseline["candidate_cycles"] / k4x1["candidate_cycles"],
            "k4x2_upper_bound_speedup_vs_raw_source_fill":
                baseline["candidate_cycles"] / k4x2["candidate_cycles"],
            "k4x1_speedup_vs_m132_compact256":
                frozen_m132["cycle_models"]["compact_k4_pwp256"]
                ["candidate_cycles"] / k4x1["candidate_cycles"],
            "k4x2_upper_bound_speedup_vs_m132_compact256":
                frozen_m132["cycle_models"]["compact_k4_pwp256"]
                ["candidate_cycles"] / k4x2["candidate_cycles"],
            "k4x1_fill_payload_reduction_fraction":
                1.0 - (k4x1["producer_fill_payload_tokens"]
                       / baseline["producer_fill_payload_tokens"]),
            "k4x2_incremental_speedup_vs_k4x1":
                k4x1["candidate_cycles"] / k4x2["candidate_cycles"],
        },
        "architecture_assumption": {
            "input": "already-sparse nonzero 16-bit source-mask record with row/block/sign metadata",
            "k4x1": "one ordered up-to-four-source descriptor emitted per cycle",
            "k4x2_upper_bound": "two independent ordered K4 descriptors emitted per cycle",
            "consumer_and_pwp_unchanged": True,
            "sparse_mask_producer_upstream_implemented": False,
            "complete_row_losslessness_proved": False,
        },
        "model_boundary": {
            "exact_heldout_width_placement_and_recurrence": True,
            "m132_baseline_exactly_replayed": True,
            "descriptorizer_rtl": False,
            "producer_area_timing": False,
            "foundry_macro": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m140_sparse_mask_k4_descriptorizer_dse.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "PASS M140 baseline_cycles={} k4x1_cycles={} k4x2_cycles={} "
        "k4x1_vs_raw={:.9f}x k4x2_vs_raw={:.9f}x "
        "k4x1_vs_compact256={:.9f}x k4x2_vs_compact256={:.9f}x "
        "k4x1_rtl=false k4x2_rtl=false physical_speedup=false "
        "system_speedup=false headline=false".format(
            baseline["candidate_cycles"], k4x1["candidate_cycles"],
            k4x2["candidate_cycles"],
            baseline["candidate_cycles"] / k4x1["candidate_cycles"],
            baseline["candidate_cycles"] / k4x2["candidate_cycles"],
            frozen_m132["cycle_models"]["compact_k4_pwp256"]
            ["candidate_cycles"] / k4x1["candidate_cycles"],
            frozen_m132["cycle_models"]["compact_k4_pwp256"]
            ["candidate_cycles"] / k4x2["candidate_cycles"]),
        flush=True)


if __name__ == "__main__":
    main()
