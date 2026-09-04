#!/usr/bin/env python3
"""Exact heldout DSE for bounded PWP/correction pipeline overlap.

M132 serializes PWP512 reconstruction and K4 correction on one service lane.
M141 separates them into two engines while retaining the two descriptor banks:
PWP for descriptor N+1 may overlap correction of N, but a bank is not released
until its correction completes.  The ping-pong ownership therefore bounds the
lookahead and prevents an unbounded intermediate queue.  This is a cycle model,
not RTL, macro, frequency, physical, energy or system evidence.
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
M140_SCRIPT = HW / "system_simulator/scripts/analyze_m140_sparse_mask_k4_descriptorizer_dse.py"
M140_RESULT = HW / "results/m140_sparse_mask_k4_descriptorizer_dse_r1_20260824/m140_sparse_mask_k4_descriptorizer_dse.json"
M140_CONTRACT = HW / "contracts/m140_sparse_mask_k4_descriptorizer_dse_contract_r1_20260824.json"
M132_SCRIPT = HW / "system_simulator/scripts/analyze_m132_dualrow512_pwp_compact_k4_schedule.py"
M132_RESULT = HW / "results/m132_dualrow512_pwp_compact_k4_schedule_r1_20260824/m132_dualrow512_pwp_compact_k4_schedule.json"
M132_OVERLAY = HW / "contracts/m132_r1_independent_review_correction_overlay_r1_20260824.json"
M109_RESULT = HW / "results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/m109_r2_window_storage_dual_timeline_frontier.json"

EXPECTED_SHA256 = {
    "m140_script": "b088892f5aea327f0b15faae60883d25bc8b6c488b4c5bd67159e13c93661484",
    "m140_result": "c0f33cb793f3d01ac471ed710a44d3f1bb81b2b45421c533533698733a556508",
    "m140_contract": "08134208609615ac8432183debabededdce90083a330bca57e31662b736ba3e2",
    "m132_script": "f140b6b72559f04cdac374eaf696c3f6650b20d3b00bd580419b88494d89c952",
    "m132_result": "f74444576ec487b9b1034aced7add0da868a9dea5d4185e0a62c1e33fe1ad755",
    "m132_overlay": "82ca925af73a7fecb55c4a47d6d95fbba5eb5c22698a2c27695b6a68fbda36a9",
    "m109_result": "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
}
WEIGHT_BEATS_PER_KEY = 3


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


class BoundedOverlapSchedule:
    """Two-engine recurrence bounded by two descriptor/result banks."""

    def __init__(self, banks=2):
        require(banks >= 2, "overlap schedule requires at least two banks")
        self.banks = banks
        self.bank_free = [0] * banks
        self.producer_end = 0
        self.controller_free = 0
        self.pwp_free = 0
        self.correction_free = 0
        self.values = Counter()
        self.maximum_fill = 0
        self.maximum_pwp = 0
        self.maximum_correction = 0
        self.maximum_result_wait = 0

    def descriptor(self, window, partition, producer_tokens, groups,
                   pwp_tokens, correction_cycles):
        index = self.values["descriptors"]
        bank = index % self.banks
        if index == 0 or self.producer_end > self.bank_free[bank]:
            fill_start = self.producer_end
        else:
            fill_start = self.bank_free[bank] + 1
            self.values["bank_reacquire_boundaries"] += 1
        self.values["producer_bank_stall_cycles"] += (
            fill_start - self.producer_end)
        fill_cycles = producer_tokens + 1
        fill_end = fill_start + fill_cycles
        self.producer_end = fill_end
        self.values["descriptor_fill_cycles"] += fill_cycles
        self.values["producer_fill_payload_tokens"] += producer_tokens
        self.values["controller_dispatch_edges"] += 1
        fill_only_dispatch = fill_end + 1
        dispatch_ready = max(fill_end, self.controller_free) + 1
        self.values[
            "controller_serialization_delay_sum_vs_fill_only_dispatch"] += (
                dispatch_ready - fill_only_dispatch)

        pwp_start = max(self.pwp_free, fill_start)
        self.values["pwp_idle_cycles"] += pwp_start - self.pwp_free
        pwp_end = pwp_start + pwp_tokens
        self.pwp_free = pwp_end
        self.values["pwp_service_tokens"] += pwp_tokens

        correction = correction_cycles + WEIGHT_BEATS_PER_KEY * groups
        self.values["folded_event_cycles"] += correction_cycles
        self.values["weight_load_tokens"] += WEIGHT_BEATS_PER_KEY * groups
        self.values["correction_service_tokens"] += correction
        if correction:
            correction_start = max(
                self.correction_free, pwp_end, dispatch_ready)
            self.values["correction_idle_cycles"] += (
                correction_start - self.correction_free)
            result_wait = correction_start - pwp_end
            self.values["pwp_result_wait_cycles"] += result_wait
            self.maximum_result_wait = max(
                self.maximum_result_wait, result_wait)
            overlap_start = max(pwp_start, self.values.get(
                "previous_correction_start", 0))
            overlap_end = min(pwp_end, self.correction_free)
            if overlap_end > overlap_start:
                self.values["pwp_correction_overlap_cycles"] += (
                    overlap_end - overlap_start)
            self.values["previous_correction_start"] = correction_start
            correction_end = correction_start + correction
            self.correction_free = correction_end
            self.controller_free = correction_end
            self.bank_free[bank] = correction_end
        else:
            if dispatch_ready <= pwp_end:
                self.values["dispatch_hidden_by_pwp"] += 1
            self.values["empty_release_delay_sum_vs_fill_only_dispatch"] += (
                dispatch_ready - fill_only_dispatch)
            self.controller_free = dispatch_ready
            self.bank_free[bank] = dispatch_ready
            self.values["zero_correction_descriptors"] += 1
        if pwp_tokens == 0:
            self.values["zero_pwp_descriptors"] += 1

        self.maximum_fill = max(self.maximum_fill, fill_cycles)
        self.maximum_pwp = max(self.maximum_pwp, pwp_tokens)
        self.maximum_correction = max(self.maximum_correction, correction)
        self.values["descriptors"] += 1

        if partition == 431:
            window_ready = max(
                self.pwp_free, self.correction_free,
                self.controller_free)
            flush_and_commit = 1 + min(384, 3000 - window * 384) * 8
            self.values["window_barrier_wait_cycles"] += (
                (window_ready - self.pwp_free)
                + (window_ready - self.correction_free))
            self.values["accumulator_pipeline_flush_cycles"] += 1
            self.values["accumulator_commit_cycles"] += flush_and_commit - 1
            window_end = window_ready + flush_and_commit
            self.pwp_free = window_end
            self.correction_free = window_end
            self.controller_free = window_end

    def result(self, fixed_baseline_service_tokens, source_events,
               packed_descriptors, producer_lanes):
        candidate = max(
            self.pwp_free, self.correction_free, self.controller_free)
        common_tail = (self.values["accumulator_pipeline_flush_cycles"]
                       + self.values["accumulator_commit_cycles"])
        baseline = fixed_baseline_service_tokens + common_tail
        result = dict(self.values)
        result.update({
            "candidate_cycles": candidate,
            "controller_final_free_cycle": self.controller_free,
            "pwp_final_free_cycle": self.pwp_free,
            "correction_final_free_cycle": self.correction_free,
            "fair_fixed8_baseline_cycles": baseline,
            "same_clock_service_island_ratio": baseline / candidate,
            "headroom_to_two_x_cycles": baseline // 2 - candidate,
            "maximum_descriptor_fill_cycles": self.maximum_fill,
            "maximum_pwp_tokens": self.maximum_pwp,
            "maximum_correction_tokens": self.maximum_correction,
            "maximum_pwp_result_wait_cycles": self.maximum_result_wait,
            "source_events": int(source_events),
            "packed_k4_descriptors": int(packed_descriptors),
            "descriptorizer_lanes": int(producer_lanes),
            "interstage_buffer_bound": self.banks,
        })
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M141 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    frozen_paths = {
        "m140_script": M140_SCRIPT, "m140_result": M140_RESULT,
        "m140_contract": M140_CONTRACT, "m132_script": M132_SCRIPT,
        "m132_result": M132_RESULT, "m132_overlay": M132_OVERLAY,
        "m109_result": M109_RESULT,
    }
    for label, path in frozen_paths.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)

    m132 = load_module("m141_frozen_m132", M132_SCRIPT)
    for label, path in {
        key: getattr(m132, {
            "m122_script": "M122_SCRIPT", "m122_result": "M122_RESULT",
            "m129_result": "M129_RESULT", "m129_overlay": "M129_OVERLAY",
            "m129_review": "M129_REVIEW", "m131_receipt": "M131_RECEIPT",
            "m109_script": "M109_SCRIPT", "m108_script": "M108_SCRIPT",
            "m105_script": "M105_SCRIPT", "m40_manifest": "M40_MANIFEST",
            "m72_result": "M72_RESULT", "m41_result": "M41_RESULT",
        }[key]) for key in m132.EXPECTED_SHA256
    }.items():
        require(sha256(path) == m132.EXPECTED_SHA256[label],
                "M132 transitive input identity drift: " + label)

    frozen_m132 = strict_json(M132_RESULT)
    frozen_m140 = strict_json(M140_RESULT)
    m122 = load_module("m141_frozen_m122", m132.M122_SCRIPT)
    require(m122.WEIGHT_BEATS_PER_KEY == WEIGHT_BEATS_PER_KEY,
            "frozen weight-beats-per-key drift")
    m105 = load_module("m141_frozen_m105", m132.M105_SCRIPT)
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
    serial = m122.FoldSchedule()
    overlap_raw = BoundedOverlapSchedule(banks=2)
    overlap_k4_b2 = BoundedOverlapSchedule(banks=2)
    overlap_k4_b3 = BoundedOverlapSchedule(banks=3)
    overlap_k4_b4 = BoundedOverlapSchedule(banks=4)
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
                correction_cycles = folded_count + int(folded_count != 0)
                serial.descriptor(window_index, partition, event_count,
                                  group_count, pwp_tokens,
                                  correction_cycles)
                overlap_raw.descriptor(
                    window_index, partition, event_count, group_count,
                    pwp_tokens, correction_cycles)
                overlap_k4_b2.descriptor(
                    window_index, partition, folded_count, group_count,
                    pwp_tokens, correction_cycles)
                overlap_k4_b3.descriptor(
                    window_index, partition, folded_count, group_count,
                    pwp_tokens, correction_cycles)
                overlap_k4_b4.descriptor(
                    window_index, partition, folded_count, group_count,
                    pwp_tokens, correction_cycles)
                totals["source_events"] += event_count
                totals["packed_k4_descriptors"] += folded_count
                totals["pwp512_tokens"] += pwp_tokens
        print("[M141 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    serial_result = serial.result(fixed_baseline_service_tokens)
    raw_result = overlap_raw.result(
        fixed_baseline_service_tokens, totals["source_events"],
        totals["packed_k4_descriptors"], 0)
    k4_b2_result = overlap_k4_b2.result(
        fixed_baseline_service_tokens, totals["source_events"],
        totals["packed_k4_descriptors"], 1)
    k4_b3_result = overlap_k4_b3.result(
        fixed_baseline_service_tokens, totals["source_events"],
        totals["packed_k4_descriptors"], 1)
    k4_b4_result = overlap_k4_b4.result(
        fixed_baseline_service_tokens, totals["source_events"],
        totals["packed_k4_descriptors"], 1)
    require(serial_result == frozen_m132["cycle_models"]
            ["compact_k4_dualrow_pwp512"],
            "M132 serial baseline replay drift")
    require(raw_result["correction_service_tokens"]
            == serial_result["correction_service_tokens"]
            == k4_b2_result["correction_service_tokens"]
            == k4_b3_result["correction_service_tokens"]
            == k4_b4_result["correction_service_tokens"]
            == 124730596,
            "correction-service token conservation drift")
    require(raw_result["pwp_service_tokens"]
            == serial_result["pwp_service_tokens"]
            == k4_b2_result["pwp_service_tokens"]
            == k4_b3_result["pwp_service_tokens"]
            == k4_b4_result["pwp_service_tokens"]
            == 119447791,
            "PWP512 token conservation drift")
    require(totals["source_events"]
            == frozen_m140["exact_work"]["source_events"],
            "source-event conservation drift")
    require(totals["packed_k4_descriptors"]
            == frozen_m140["exact_work"]["packed_k4_descriptors"],
            "K4 descriptor conservation drift")
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M141 analyzer changed during execution")

    compact256_cycles = int(frozen_m132["cycle_models"]
                            ["compact_k4_pwp256"]["candidate_cycles"])
    payload = {
        "schema": "m141_bounded_pwp_correction_overlap_dse_v3",
        "status": "PASS_EXACT_HELDOUT_BOUNDED_TWO_ENGINE_BUFFER_DEPTH_DSE_R3",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED_SHA256,
            "m132_transitive_inputs_sha256": m132.EXPECTED_SHA256,
            "heldout_samples": list(range(5, 10)),
            "heldout_records": len(heldout),
        },
        "exact_work": dict(totals),
        "cycle_models": {
            "serial_m132_dualrow512": serial_result,
            "bounded_overlap_raw_source_fill": raw_result,
            "bounded_overlap_sparse_mask_k4_fill_b2": k4_b2_result,
            "bounded_overlap_sparse_mask_k4_fill_b3": k4_b3_result,
            "bounded_overlap_sparse_mask_k4_fill_b4": k4_b4_result,
        },
        "comparisons": {
            "raw_overlap_speedup_vs_serial_m132_dualrow512":
                serial_result["candidate_cycles"]
                / raw_result["candidate_cycles"],
            "k4_overlap_speedup_vs_serial_m132_dualrow512":
                serial_result["candidate_cycles"]
                / k4_b2_result["candidate_cycles"],
            "raw_overlap_speedup_vs_m132_compact256":
                compact256_cycles / raw_result["candidate_cycles"],
            "k4_overlap_speedup_vs_m132_compact256":
                compact256_cycles / k4_b2_result["candidate_cycles"],
            "k4_fill_incremental_speedup_after_overlap":
                raw_result["candidate_cycles"]
                / k4_b2_result["candidate_cycles"],
            "k4_overlap_same_clock_ratio_vs_fixed8":
                k4_b2_result["same_clock_service_island_ratio"],
            "b3_speedup_vs_b2":
                k4_b2_result["candidate_cycles"]
                / k4_b3_result["candidate_cycles"],
            "b4_speedup_vs_b3":
                k4_b3_result["candidate_cycles"]
                / k4_b4_result["candidate_cycles"],
            "b3_speedup_vs_m132_compact256":
                compact256_cycles / k4_b3_result["candidate_cycles"],
            "b4_speedup_vs_m132_compact256":
                compact256_cycles / k4_b4_result["candidate_cycles"],
        },
        "architecture_assumption": {
            "pwp_engine": "M138-class 512-bit one-cycle-return stream boundary",
            "correction_engine": "M131-class compact K4 update lane",
            "descriptor_and_result_bank_depths_dse": [2, 3, 4],
            "bank_release": "only after the owning correction completes",
            "interstage_lookahead": "bounded by bank ownership; no unbounded FIFO",
            "window_boundary": "both engines drain before one flush plus commit",
            "consumer_arithmetic_or_work_changed": False,
        },
        "model_boundary": {
            "exact_heldout_width_placement_and_recurrence": True,
            "m132_serial_baseline_exactly_replayed": True,
            "two_engine_rtl": False,
            "bounded_handoff_rtl": False,
            "foundry_macro": False,
            "matched_frequency": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m141r3_bounded_pwp_correction_overlap_dse.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "PASS M141r3 serial={} overlap_raw={} k4_b2={} k4_b3={} k4_b4={} "
        "raw_vs_serial={:.9f}x b2_vs_serial={:.9f}x "
        "raw_vs_compact256={:.9f}x b2_vs_compact256={:.9f}x "
        "b3_vs_compact256={:.9f}x b4_vs_compact256={:.9f}x "
        "b2_fixed8_ratio={:.9f}x bounded_rtl=false physical_speedup=false "
        "system_speedup=false headline=false".format(
            serial_result["candidate_cycles"], raw_result["candidate_cycles"],
            k4_b2_result["candidate_cycles"],
            k4_b3_result["candidate_cycles"],
            k4_b4_result["candidate_cycles"],
            serial_result["candidate_cycles"] / raw_result["candidate_cycles"],
            serial_result["candidate_cycles"] / k4_b2_result["candidate_cycles"],
            compact256_cycles / raw_result["candidate_cycles"],
            compact256_cycles / k4_b2_result["candidate_cycles"],
            compact256_cycles / k4_b3_result["candidate_cycles"],
            compact256_cycles / k4_b4_result["candidate_cycles"],
            k4_b2_result["same_clock_service_island_ratio"]), flush=True)


if __name__ == "__main__":
    main()
