#!/usr/bin/env python3
"""Exact heldout accounting for row admission bubbles and K4 descriptors.

M122 assumes one folded accumulator update per cycle.  M125/M127 accept a
row mask separately and therefore spend one non-update admission cycle for
every active (row, output-block).  M128 consumes canonical K1-K4 descriptors
and may replace a retiring descriptor across row boundaries.  This analyzer
replays the frozen M122 heldout reconstruction and charges those interfaces
explicitly.  It is a cycle-model audit, not physical or system evidence.
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
M122_SCRIPT = HW / (
    "system_simulator/scripts/"
    "analyze_m122_w384_row_synchronous_source_fold.py")
M122_RESULT = HW / (
    "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/"
    "m122_w384_row_synchronous_source_fold_dse.json")
M128_VCS_RECEIPT = HW / (
    "dc_handoff/runs/"
    "m128_descriptor_streamed_k4_row_fold_vcs_r1_sealed_20260824/"
    "RUN_COMPLETE.txt")

EXPECTED_SHA256 = {
    "m122_script":
        "ecf2ae43e1282ac483b6832f5a21af6d1b6259c3595eb6150e840f0dc7a55cd3",
    "m122_result":
        "be11341211b92d85dc42cb7b79b98a826a782765a4780e1207e7bad5368d27b2",
    "m128_vcs_receipt":
        "d9e320092d381999ec158fa31d8aaf32be47c02283d50e3e7ba463cfd7751f28",
}

FOLD = 4
ROW_MASK_DESCRIPTOR_BITS = 44
M128_DESCRIPTOR_BITS = 53
COMPACT_DESCRIPTOR_BITS = 35


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, path)
    require(spec is not None and spec.loader is not None,
            "cannot load module " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M129 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    for label, path in {
        "m122_script": M122_SCRIPT,
        "m122_result": M122_RESULT,
        "m128_vcs_receipt": M128_VCS_RECEIPT,
    }.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)

    m122 = load_module("m129_frozen_m122", M122_SCRIPT)
    m109 = m122.load_module("m129_frozen_m109", m122.M109_SCRIPT)
    m108 = m109.load_module("m129_frozen_m108", m109.M108_R1_SCRIPT)
    m105 = m108.load_m105_module()
    manifest = m108.strict_json(m105.M40_MANIFEST)
    m72 = m108.strict_json(m105.M72_RESULT)
    m41 = m108.strict_json(m105.M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "heldout record extent drift")

    frozen_m122 = m122.strict_json(M122_RESULT)
    frozen_k4 = next(
        row for row in frozen_m122["fold_dse"]
        if int(row["fold_sources_per_update"]) == FOLD)
    frozen_m109 = m122.strict_json(m122.M109_RESULT)
    w384 = next(row for row in frozen_m109["frontier"]
                if int(row["window_rows"]) == m122.WINDOW_ROWS)
    fixed_baseline_service_tokens = (
        int(w384["dual_timeline_recurrence"]
            ["fair_fixed8_baseline_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_commit_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_pipeline_flush_cycles"]))

    popcount = np.fromiter(
        (int(value).bit_count() for value in range(1 << m122.SOURCES)),
        dtype=np.uint8, count=1 << m122.SOURCES)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    schedules = {
        "m122_ideal_no_interface_bubble": m122.FoldSchedule(),
        "m125_m127_row_mask_admission": m122.FoldSchedule(),
        "m128_descriptor_conservative_startup": m122.FoldSchedule(),
    }
    totals = Counter()
    starts = np.arange(0, m122.ROWS, m122.WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + m122.WINDOW_ROWS, m122.ROWS)

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, pwp_rows = m109.build_record_rows(
            m105, m108, masks, record["operator_index"],
            centers, widths, popcount)
        source_counts = popcount[event_masks]
        event_prefix = np.concatenate((
            np.zeros((m122.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(source_counts.sum(axis=2, dtype=np.uint16),
                      axis=1, dtype=np.uint32)), axis=1)
        pwp_prefix = np.concatenate((
            np.zeros((m122.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(pwp_rows, axis=1, dtype=np.uint32)), axis=1)
        folded_per_row = ((source_counts.astype(np.uint16) + FOLD - 1)
                          // FOLD).sum(axis=2, dtype=np.uint16)
        folded_prefix = np.concatenate((
            np.zeros((m122.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(folded_per_row, axis=1, dtype=np.uint32)), axis=1)
        active_per_row = (source_counts != 0).sum(axis=2, dtype=np.uint16)
        active_prefix = np.concatenate((
            np.zeros((m122.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(active_per_row, axis=1, dtype=np.uint32)), axis=1)
        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)

        for window_index, (start, end) in enumerate(zip(starts, ends)):
            events = event_prefix[:, end] - event_prefix[:, start]
            pwp = pwp_prefix[:, end] - pwp_prefix[:, start]
            folded = folded_prefix[:, end] - folded_prefix[:, start]
            active = active_prefix[:, end] - active_prefix[:, start]
            for partition in range(m122.PARTITIONS):
                event_count = int(events[partition])
                group_count = int(groups[partition, window_index])
                pwp_count = int(pwp[partition])
                folded_count = int(folded[partition])
                active_count = int(active[partition])
                totals["events"] += event_count
                totals["active_row_blocks"] += active_count
                totals["k4_descriptors"] += folded_count
                totals["active_correction_descriptors"] += int(
                    folded_count != 0)

                schedules["m122_ideal_no_interface_bubble"].descriptor(
                    window_index, partition, event_count, group_count,
                    pwp_count, folded_count)
                schedules["m125_m127_row_mask_admission"].descriptor(
                    window_index, partition, event_count, group_count,
                    pwp_count, folded_count + active_count)
                schedules["m128_descriptor_conservative_startup"].descriptor(
                    window_index, partition, event_count, group_count,
                    pwp_count, folded_count + int(folded_count != 0))
        print("[M129 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    require(totals["events"] == int(frozen_m122["exact_work"]["events"]),
            "event conservation drift")
    require(totals["k4_descriptors"]
            == int(frozen_k4["exact_fold_event_cycles"]),
            "K4 descriptor conservation drift")
    expected_active = sum(
        int(value) for key, value in
        frozen_m122["same_row_source_count_histogram"].items()
        if int(key) != 0)
    require(totals["active_row_blocks"] == expected_active,
            "active row/block conservation drift")

    recurrence = {
        name: schedule.result(fixed_baseline_service_tokens)
        for name, schedule in schedules.items()
    }
    ideal = recurrence["m122_ideal_no_interface_bubble"]
    require(ideal == frozen_k4["dual_timeline_recurrence"],
            "M129 ideal replay does not exactly reproduce frozen M122 K4")
    row_mask = recurrence["m125_m127_row_mask_admission"]
    descriptor = recurrence["m128_descriptor_conservative_startup"]

    row_mask_bits = totals["active_row_blocks"] * ROW_MASK_DESCRIPTOR_BITS
    m128_bits = totals["k4_descriptors"] * M128_DESCRIPTOR_BITS
    compact_bits = totals["k4_descriptors"] * COMPACT_DESCRIPTOR_BITS
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M129 analyzer changed during execution")
    payload = {
        "schema": "m129_row_admission_bubble_and_descriptor_cost_v1",
        "status": "PASS_EXACT_HELDOUT_INTERFACE_CYCLE_ACCOUNTING",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED_SHA256,
            "heldout_samples": list(range(5, 10)),
            "heldout_records": len(heldout),
        },
        "exact_work": dict(totals),
        "cycle_models": {
            "m122_ideal_no_interface_bubble": ideal,
            "m125_m127_row_mask_admission": row_mask,
            "m128_descriptor_conservative_startup": descriptor,
        },
        "comparisons": {
            "m128_candidate_cycle_speedup_vs_m125_m127_row_mask":
                row_mask["candidate_cycles"] / descriptor["candidate_cycles"],
            "m128_candidate_cycles_removed_vs_m125_m127_row_mask":
                row_mask["candidate_cycles"] - descriptor["candidate_cycles"],
            "m128_fixed8_same_clock_service_island_ratio":
                descriptor["same_clock_service_island_ratio"],
            "m125_m127_row_mask_fixed8_same_clock_service_island_ratio":
                row_mask["same_clock_service_island_ratio"],
            "m128_conservative_startup_cycles_above_m122_ideal":
                descriptor["candidate_cycles"] - ideal["candidate_cycles"],
        },
        "descriptor_traffic": {
            "row_mask_descriptor_bits_each": ROW_MASK_DESCRIPTOR_BITS,
            "row_mask_total_bits": row_mask_bits,
            "m128_descriptor_bits_each": M128_DESCRIPTOR_BITS,
            "m128_total_bits": m128_bits,
            "m128_fraction_vs_row_mask": m128_bits / row_mask_bits,
            "compact_successor_descriptor_bits_each":
                COMPACT_DESCRIPTOR_BITS,
            "compact_successor_total_bits": compact_bits,
            "compact_successor_fraction_vs_row_mask":
                compact_bits / row_mask_bits,
        },
        "model_boundary": {
            "m125_m127_charge_one_row_admission_cycle_per_active_row_block":
                True,
            "m128_charge_one_pipeline_startup_cycle_per_nonempty_partition_window":
                True,
            "m128_cross_row_descriptor_ii1_vcs": True,
            "external_descriptor_generation_cycles": False,
            "descriptor_storage_memory_energy": False,
            "foundry_weight_macro": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m129_row_admission_bubble_and_descriptor_cost.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "PASS M129 rowmask_cycles={} descriptor_cycles={} speedup={:.9f}x "
        "rowmask_ratio={:.9f}x descriptor_ratio={:.9f}x "
        "m128_descriptor_fraction={:.9f} compact_fraction={:.9f} "
        "physical_speedup=false system_speedup=false headline=false".format(
            row_mask["candidate_cycles"], descriptor["candidate_cycles"],
            row_mask["candidate_cycles"] / descriptor["candidate_cycles"],
            row_mask["same_clock_service_island_ratio"],
            descriptor["same_clock_service_island_ratio"],
            m128_bits / row_mask_bits, compact_bits / row_mask_bits),
        flush=True)


if __name__ == "__main__":
    main()
