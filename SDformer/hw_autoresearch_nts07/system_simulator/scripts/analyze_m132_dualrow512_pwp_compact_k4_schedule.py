#!/usr/bin/env python3
"""Exact heldout cycle DSE for a dual-row 512-bit PWP service port.

M131 preserves the M129 compact K4 correction schedule.  This analyzer changes
only the PWP fetch geometry from one 256-bit row/cycle to two 256-bit rows/cycle.
Signed 8/9/10/11-bit 96-lane vectors therefore require 2/2/2/3 cycles instead
of 3/4/4/5.  The exact W384 recurrence is replayed per descriptor; banking,
macro area, energy and frequency remain outside this cycle-model admission.
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
M122_SCRIPT = HW / "system_simulator/scripts/analyze_m122_w384_row_synchronous_source_fold.py"
M122_RESULT = HW / "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/m122_w384_row_synchronous_source_fold_dse.json"
M129_RESULT = HW / "results/m129_row_admission_bubble_and_descriptor_cost_r1_20260824/m129_row_admission_bubble_and_descriptor_cost.json"
M129_OVERLAY = HW / "contracts/m129_r1_independent_review_identity_correction_r1_20260824.json"
M129_REVIEW = HW / "reviews/m129_row_admission_bubble_and_descriptor_cost_independent_hammer_r1_20260824/manifest.sha256"
M131_RECEIPT = HW / "dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
M109_SCRIPT = HW / "system_simulator/scripts/analyze_m109_r2_window_storage_dual_timeline_frontier.py"
M108_SCRIPT = HW / "system_simulator/scripts/analyze_m108_w64_fused_pwp_accumulator_schedule.py"
M105_SCRIPT = HW / "reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/audit_m105_bounded_row_transpose.py"
M40_MANIFEST = HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json"
M72_RESULT = HW / "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/m72_phi_kmeans_k16q16_valid825_internal_screen.json"
M41_RESULT = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/m41_h67_ep35_bottleneck_int8_bridge.json"

EXPECTED_SHA256 = {
    "m122_script": "ecf2ae43e1282ac483b6832f5a21af6d1b6259c3595eb6150e840f0dc7a55cd3",
    "m122_result": "be11341211b92d85dc42cb7b79b98a826a782765a4780e1207e7bad5368d27b2",
    "m129_result": "2443a651675763c9e867a2186e83440c323cf20e381e7a49724d6cb0d9ab411e",
    "m129_overlay": "9b4073183c8ecd541758a693472b1b2c92f829de915d836428a2b9e5e7a9968d",
    "m129_review": "eeada044c1199099de574dc8ed131bc81c33e0063d1581cd312c9f4649bd284d",
    "m131_receipt": "e30e273ff791475d7f015ae4fb580a8c5fa0b018a432adf666519ffd44184316",
    "m109_script": "4eed1e1ef25cdbea0fdd40d1602d6b1eb7661b15b5ae47541c80e149fd060ada",
    "m108_script": "4404e5825ece95fbf0a28dd580c03c7e9f34bcfa9ec12fa3b66d226a9042cbe2",
    "m105_script": "5e5c07631dd8c4bb328cd234da5c04fde8eb9800d1516b3fe462124b2b661ed5",
    "m40_manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m72_result": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
}

ROWS = 3000
PARTITIONS = 432
OUTPUT_BLOCKS = 8
WINDOW_ROWS = 384
FOLD = 4
PWP_BEATS_256 = np.zeros(33, dtype=np.uint8)
PWP_BEATS_256[8:12] = (3, 4, 4, 5)
PWP_BEATS_512 = np.zeros(33, dtype=np.uint8)
PWP_BEATS_512[8:12] = (2, 2, 2, 3)
EXPECTED_WIDTH_USES = {8: 11164284, 9: 32360036,
                       10: 13936011, 11: 1509043}


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


def build_record_rows(m105, masks, operator, centers, widths, popcount):
    event_masks = np.zeros((PARTITIONS, ROWS, OUTPUT_BLOCKS),
                           dtype=np.uint16)
    pwp256_rows = np.zeros((PARTITIONS, ROWS), dtype=np.uint16)
    pwp512_rows = np.zeros((PARTITIONS, ROWS), dtype=np.uint16)
    width_uses = Counter()
    for partition in range(PARTITIONS):
        values = masks[partition]
        center_values = centers[operator, partition]
        order = np.argsort(center_values, kind="stable")
        ordered_centers = center_values[order]
        distances = popcount[np.bitwise_xor(
            values[:, None], ordered_centers[None, :])]
        ordered_choice = distances.argmin(axis=1)
        best_index = order[ordered_choice]
        best_distance = distances[np.arange(ROWS), ordered_choice]
        best_center = center_values[best_index]
        population = popcount[values]
        beneficial = (1 + best_distance) < population
        delta = np.bitwise_xor(values, best_center)
        selected_widths = widths[operator, partition, best_index]
        eligible = beneficial[:, None] & (selected_widths <= m105.CAP)
        event_masks[partition] = np.where(
            eligible, delta[:, None], values[:, None]).astype(np.uint16)
        pwp256_rows[partition] = np.where(
            eligible, PWP_BEATS_256[selected_widths], 0).sum(
                axis=1, dtype=np.uint16)
        pwp512_rows[partition] = np.where(
            eligible, PWP_BEATS_512[selected_widths], 0).sum(
                axis=1, dtype=np.uint16)
        for width in range(8, 12):
            width_uses[width] += int(
                (eligible & (selected_widths == width)).sum())
    return event_masks, pwp256_rows, pwp512_rows, width_uses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M132 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    frozen_paths = {
        "m122_script": M122_SCRIPT, "m122_result": M122_RESULT,
        "m129_result": M129_RESULT, "m129_overlay": M129_OVERLAY,
        "m129_review": M129_REVIEW, "m131_receipt": M131_RECEIPT,
        "m109_script": M109_SCRIPT, "m108_script": M108_SCRIPT,
        "m105_script": M105_SCRIPT, "m40_manifest": M40_MANIFEST,
        "m72_result": M72_RESULT, "m41_result": M41_RESULT,
    }
    for label, path in frozen_paths.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)

    m122 = load_module("m132_frozen_m122", M122_SCRIPT)
    m109 = load_module("m132_frozen_m109", M109_SCRIPT)
    m108 = load_module("m132_frozen_m108", M108_SCRIPT)
    m105 = load_module("m132_frozen_m105", M105_SCRIPT)
    manifest = strict_json(M40_MANIFEST)
    m72 = strict_json(M72_RESULT)
    m41 = strict_json(M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "heldout record extent drift")

    frozen_m122 = strict_json(M122_RESULT)
    frozen_k4 = next(row for row in frozen_m122["fold_dse"]
                     if int(row["fold_sources_per_update"]) == FOLD)
    frozen_m129 = strict_json(M129_RESULT)
    frozen_m109 = strict_json(m122.M109_RESULT)
    w384 = next(row for row in frozen_m109["frontier"]
                if int(row["window_rows"]) == WINDOW_ROWS)
    fixed_baseline_service_tokens = (
        int(w384["dual_timeline_recurrence"]["fair_fixed8_baseline_cycles"])
        - int(w384["dual_timeline_recurrence"]["accumulator_commit_cycles"])
        - int(w384["dual_timeline_recurrence"]["accumulator_pipeline_flush_cycles"]))

    popcount = np.fromiter(
        (int(value).bit_count() for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    schedules = {
        "compact_k4_pwp256": m122.FoldSchedule(),
        "compact_k4_dualrow_pwp512": m122.FoldSchedule(),
    }
    totals = Counter()
    width_uses = Counter()
    starts = np.arange(0, ROWS, WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + WINDOW_ROWS, ROWS)

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, pwp256_rows, pwp512_rows, record_widths = (
            build_record_rows(m105, masks, record["operator_index"],
                              centers, widths, popcount))
        width_uses.update(record_widths)
        frozen_events, frozen_pwp = m109.build_record_rows(
            m105, m108, masks, record["operator_index"],
            centers, widths, popcount)
        require(np.array_equal(event_masks, frozen_events),
                "event reconstruction drift")
        require(np.array_equal(pwp256_rows, frozen_pwp),
                "256-bit PWP reconstruction drift")

        source_counts = popcount[event_masks]
        event_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(source_counts.sum(axis=2, dtype=np.uint16),
                      axis=1, dtype=np.uint32)), axis=1)
        folded_per_row = ((source_counts.astype(np.uint16) + FOLD - 1)
                          // FOLD).sum(axis=2, dtype=np.uint16)
        folded_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(folded_per_row, axis=1, dtype=np.uint32)), axis=1)
        pwp256_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(pwp256_rows, axis=1, dtype=np.uint32)), axis=1)
        pwp512_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(pwp512_rows, axis=1, dtype=np.uint32)), axis=1)
        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)

        for window_index, (start, end) in enumerate(zip(starts, ends)):
            events = event_prefix[:, end] - event_prefix[:, start]
            folded = folded_prefix[:, end] - folded_prefix[:, start]
            pwp256 = pwp256_prefix[:, end] - pwp256_prefix[:, start]
            pwp512 = pwp512_prefix[:, end] - pwp512_prefix[:, start]
            for partition in range(PARTITIONS):
                event_count = int(events[partition])
                group_count = int(groups[partition, window_index])
                folded_count = int(folded[partition])
                startup = int(folded_count != 0)
                correction_cycles = folded_count + startup
                schedules["compact_k4_pwp256"].descriptor(
                    window_index, partition, event_count, group_count,
                    int(pwp256[partition]), correction_cycles)
                schedules["compact_k4_dualrow_pwp512"].descriptor(
                    window_index, partition, event_count, group_count,
                    int(pwp512[partition]), correction_cycles)
                totals["events"] += event_count
                totals["k4_descriptors"] += folded_count
                totals["pwp256_tokens"] += int(pwp256[partition])
                totals["pwp512_tokens"] += int(pwp512[partition])
        print("[M132 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    require(dict(sorted(width_uses.items())) == EXPECTED_WIDTH_USES,
            "PWP width-use conservation drift")
    require(totals["events"] == int(frozen_m122["exact_work"]["events"]),
            "event conservation drift")
    require(totals["k4_descriptors"]
            == int(frozen_k4["exact_fold_event_cycles"]),
            "K4 descriptor conservation drift")
    recurrence = {
        name: schedule.result(fixed_baseline_service_tokens)
        for name, schedule in schedules.items()
    }
    compact256 = recurrence["compact_k4_pwp256"]
    compact512 = recurrence["compact_k4_dualrow_pwp512"]
    require(compact256 == frozen_m129["cycle_models"]
            ["m128_descriptor_conservative_startup"],
            "compact 256-bit replay does not reproduce M129")
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M132 analyzer changed during execution")

    payload = {
        "schema": "m132_dualrow512_pwp_compact_k4_schedule_v1",
        "status": "PASS_EXACT_HELDOUT_DUALROW512_CYCLE_DSE",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED_SHA256,
            "heldout_samples": list(range(5, 10)),
            "heldout_records": len(heldout),
        },
        "exact_work": dict(totals),
        "pwp_width_uses": dict(sorted(width_uses.items())),
        "pwp_geometry": {
            "baseline_port_bits": 256,
            "candidate_port_bits": 512,
            "baseline_cycles_by_width": {"8": 3, "9": 4,
                                          "10": 4, "11": 5},
            "candidate_cycles_by_width": {"8": 2, "9": 2,
                                           "10": 2, "11": 3},
            "candidate_requires_two_256bit_rows_per_cycle": True,
            "candidate_logical_bank_words_per_cycle": 16,
        },
        "cycle_models": recurrence,
        "comparisons": {
            "dualrow512_speedup_vs_compact256":
                compact256["candidate_cycles"] / compact512["candidate_cycles"],
            "dualrow512_cycles_removed_vs_compact256":
                compact256["candidate_cycles"] - compact512["candidate_cycles"],
            "dualrow512_fixed8_same_clock_service_island_ratio":
                compact512["same_clock_service_island_ratio"],
            "pwp_token_reduction_fraction":
                1.0 - totals["pwp512_tokens"] / totals["pwp256_tokens"],
        },
        "model_boundary": {
            "exact_heldout_width_placement_and_recurrence": True,
            "m131_compact_k4_cycle_contract": True,
            "dualrow512_pwp_rtl": False,
            "bank_conflicts_modeled": False,
            "foundry_dualrow_or_16bank_macro": False,
            "macro_area_energy": False,
            "matched_dc_frequency": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m132_dualrow512_pwp_compact_k4_schedule.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "PASS M132 compact256_cycles={} dualrow512_cycles={} "
        "module_cycle_speedup={:.9f}x fixed8_ratio={:.9f}x "
        "pwp256={} pwp512={} pwp_reduction={:.9%} "
        "dualrow512_rtl=false physical_speedup=false "
        "system_speedup=false headline=false".format(
            compact256["candidate_cycles"], compact512["candidate_cycles"],
            compact256["candidate_cycles"] / compact512["candidate_cycles"],
            compact512["same_clock_service_island_ratio"],
            totals["pwp256_tokens"], totals["pwp512_tokens"],
            1.0 - totals["pwp512_tokens"] / totals["pwp256_tokens"]),
        flush=True)


if __name__ == "__main__":
    main()
