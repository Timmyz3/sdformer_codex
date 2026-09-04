#!/usr/bin/env python3
"""Exact heldout W384 DSE for lossless same-row multi-source folding.

The frozen M109-r2 trace reconstruction supplies a 16-bit source mask for
every (partition, row, output-block).  A K-source row fold consumes up to K
set bits from one mask per cycle, reads the corresponding resident signed INT8
weight vectors, adds them lane-wise, and performs one signed19 accumulator
update.  PWP service, descriptor fill, three 256-bit loads per active key,
commit, bank reuse and all dual-timeline edge rules remain unchanged.

This is a cycle-model DSE, not RTL, foundry-SRAM, physical or system evidence.
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
M109_SCRIPT = HW / (
    "system_simulator/scripts/"
    "analyze_m109_r2_window_storage_dual_timeline_frontier.py")
M109_RESULT = HW / (
    "results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/"
    "m109_r2_window_storage_dual_timeline_frontier.json")
M115R2_RESULT = HW / (
    "results/m115r2_pwp_prefix_coefficient_width_r1_20260824/"
    "m115r2_pwp_prefix_coefficient_width.json")
M121_RECEIPT = HW / (
    "dc_handoff/runs/"
    "m121_w384_scheduler_numeric_island_vcs_r1_sealed_20260824/"
    "RUN_COMPLETE.txt")

EXPECTED_SHA256 = {
    "m109_script":
        "4eed1e1ef25cdbea0fdd40d1602d6b1eb7661b15b5ae47541c80e149fd060ada",
    "m109_result":
        "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
    "m115r2_result":
        "b0e7fbb0573473ad854ca856d5eab3eaf15af1ba79ea2ce3a958810575bc6708",
    "m121_receipt":
        "4b3e0d1bf249bff14dc18a6de05cc7ddf5bca4e2d384a7ef160650702fbee986",
}

WINDOW_ROWS = 384
ROWS = 3000
PARTITIONS = 432
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
SOURCES = 16
WEIGHT_BITS = 8
WEIGHT_PORT_BITS = 256
WEIGHT_BEATS_PER_KEY = 3
FOLDS = (1, 2, 4, 8)


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


class FoldSchedule:
    """M109-r2 edge recurrence with only correction event service replaced."""

    def __init__(self):
        self.bank_free = [0, 0]
        self.producer_end = 0
        self.controller_free = 0
        self.service_end = 0
        self.values = Counter()
        self.maximum_descriptor_fill_cycles = 0
        self.maximum_descriptor_service_tokens = 0

    def descriptor(self, window, partition, events, groups, pwp_tokens,
                   folded_event_cycles):
        index = self.values["descriptors"]
        bank = index & 1
        if index == 0 or self.producer_end > self.bank_free[bank]:
            fill_start = self.producer_end
        else:
            fill_start = self.bank_free[bank] + 1
            self.values["bank_reacquire_boundaries"] += 1
        self.values["producer_bank_stall_cycles"] += (
            fill_start - self.producer_end)
        fill_cycles = events + 1
        fill_end = fill_start + fill_cycles
        self.producer_end = fill_end
        self.values["descriptor_fill_cycles"] += fill_cycles
        self.values["controller_dispatch_edges"] += 1
        fill_only_dispatch = fill_end + 1
        dispatch_ready = max(fill_end, self.controller_free) + 1
        self.values[
            "controller_serialization_delay_sum_vs_fill_only_dispatch"] += (
                dispatch_ready - fill_only_dispatch)

        pwp_start = max(self.service_end, fill_start)
        self.values["service_idle_cycles"] += pwp_start - self.service_end
        pwp_end = pwp_start + pwp_tokens
        self.values["pwp_service_tokens"] += pwp_tokens

        correction = folded_event_cycles + WEIGHT_BEATS_PER_KEY * groups
        self.values["raw_event_tokens"] += events
        self.values["folded_event_cycles"] += folded_event_cycles
        self.values["weight_load_tokens"] += WEIGHT_BEATS_PER_KEY * groups
        self.values["correction_service_tokens"] += correction
        if correction:
            correction_start = max(pwp_end, dispatch_ready)
            if dispatch_ready <= pwp_end:
                self.values[
                    "dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
            exposed = correction_start - pwp_end
            self.values[
                "exposed_post_pwp_fill_or_dispatch_wait_cycles"] += exposed
            self.values["service_idle_cycles"] += exposed
            self.service_end = correction_start + correction
            self.bank_free[bank] = self.service_end
            self.controller_free = self.service_end
        else:
            if dispatch_ready <= pwp_end:
                self.values[
                    "dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
            self.values["empty_release_delay_sum_vs_fill_only_dispatch"] += (
                dispatch_ready - fill_only_dispatch)
            self.bank_free[bank] = dispatch_ready
            self.controller_free = dispatch_ready
            self.service_end = pwp_end
        if pwp_tokens == 0:
            self.values["zero_pwp_descriptors"] += 1

        self.maximum_descriptor_fill_cycles = max(
            self.maximum_descriptor_fill_cycles, fill_cycles)
        self.maximum_descriptor_service_tokens = max(
            self.maximum_descriptor_service_tokens, pwp_tokens + correction)
        self.values["descriptors"] += 1

        if partition == PARTITIONS - 1:
            window_ready = max(self.service_end, self.controller_free)
            self.values["service_idle_cycles"] += (
                window_ready - self.service_end)
            self.service_end = window_ready + 1
            self.values["accumulator_pipeline_flush_cycles"] += 1
            rows_here = min(WINDOW_ROWS, ROWS - window * WINDOW_ROWS)
            require(rows_here > 0, "invalid final-window row count")
            commit = rows_here * OUTPUT_BLOCKS
            self.service_end += commit
            self.values["accumulator_commit_cycles"] += commit

    def result(self, fixed_baseline_service_tokens):
        common_tail = (self.values["accumulator_pipeline_flush_cycles"]
                       + self.values["accumulator_commit_cycles"])
        baseline = fixed_baseline_service_tokens + common_tail
        require(
            self.service_end == self.values["pwp_service_tokens"]
            + self.values["correction_service_tokens"]
            + self.values["service_idle_cycles"] + common_tail,
            "fold schedule cycle conservation failed")
        result = dict(self.values)
        result.update({
            "candidate_cycles": self.service_end,
            "controller_final_free_cycle": self.controller_free,
            "fair_fixed8_baseline_cycles": baseline,
            "same_clock_service_island_ratio": baseline / self.service_end,
            "headroom_to_two_x_cycles": baseline // 2 - self.service_end,
            "maximum_descriptor_fill_cycles":
                self.maximum_descriptor_fill_cycles,
            "maximum_descriptor_service_tokens":
                self.maximum_descriptor_service_tokens,
        })
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M122 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    for label, path in {
        "m109_script": M109_SCRIPT,
        "m109_result": M109_RESULT,
        "m115r2_result": M115R2_RESULT,
        "m121_receipt": M121_RECEIPT,
    }.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)

    m109 = load_module("m122_frozen_m109", M109_SCRIPT)
    m108 = m109.load_module("m122_frozen_m108", m109.M108_R1_SCRIPT)
    m105 = m108.load_m105_module()
    manifest = m108.strict_json(m105.M40_MANIFEST)
    m72 = m108.strict_json(m105.M72_RESULT)
    m41 = m108.strict_json(m105.M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "heldout record extent drift")

    frozen_m109 = strict_json(M109_RESULT)
    w384 = next(row for row in frozen_m109["frontier"]
                if int(row["window_rows"]) == WINDOW_ROWS)
    fixed_baseline_service_tokens = (
        int(w384["dual_timeline_recurrence"]
            ["fair_fixed8_baseline_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_commit_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_pipeline_flush_cycles"]))

    popcount = np.fromiter(
        (int(value).bit_count() for value in range(1 << SOURCES)),
        dtype=np.uint8, count=1 << SOURCES)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    schedules = {fold: FoldSchedule() for fold in FOLDS}
    totals = Counter()
    fold_cycle_totals = Counter()
    source_count_histogram = Counter()

    starts = np.arange(0, ROWS, WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + WINDOW_ROWS, ROWS)
    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, pwp_rows = m109.build_record_rows(
            m105, m108, masks, record["operator_index"],
            centers, widths, popcount)
        row_source_counts = popcount[event_masks]
        values, counts = np.unique(row_source_counts, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            source_count_histogram[int(value)] += int(count)

        event_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(row_source_counts.sum(axis=2, dtype=np.uint16),
                      axis=1, dtype=np.uint32)), axis=1)
        pwp_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(pwp_rows, axis=1, dtype=np.uint32)), axis=1)
        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)

        fold_prefixes = {}
        for fold in FOLDS:
            per_row = ((row_source_counts.astype(np.uint16) + fold - 1)
                       // fold).sum(axis=2, dtype=np.uint16)
            fold_prefixes[fold] = np.concatenate((
                np.zeros((PARTITIONS, 1), dtype=np.uint32),
                np.cumsum(per_row, axis=1, dtype=np.uint32)), axis=1)

        for window_index, (start, end) in enumerate(zip(starts, ends)):
            events = event_prefix[:, end] - event_prefix[:, start]
            pwp = pwp_prefix[:, end] - pwp_prefix[:, start]
            folded = {
                fold: (fold_prefixes[fold][:, end]
                       - fold_prefixes[fold][:, start])
                for fold in FOLDS
            }
            for partition in range(PARTITIONS):
                event_count = int(events[partition])
                group_count = int(groups[partition, window_index])
                pwp_count = int(pwp[partition])
                totals["events"] += event_count
                totals["groups"] += group_count
                totals["pwp_tokens"] += pwp_count
                for fold in FOLDS:
                    folded_count = int(folded[fold][partition])
                    fold_cycle_totals[fold] += folded_count
                    schedules[fold].descriptor(
                        window_index, partition, event_count, group_count,
                        pwp_count, folded_count)
        print("[M122 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    # totals were added once per fold above only outside the fold loop.
    require(totals["events"] == int(w384["exact_work"]["events"]),
            "heldout event conservation drift")
    require(totals["groups"] == int(w384["exact_work"]["groups"]),
            "heldout group conservation drift")
    require(totals["pwp_tokens"] == int(w384["exact_work"]["pwp_tokens"]),
            "heldout PWP conservation drift")
    require(sum(source_count_histogram.values())
            == len(heldout) * PARTITIONS * ROWS * OUTPUT_BLOCKS,
            "row/block histogram extent drift")

    records = []
    k1_recurrence = None
    for fold in FOLDS:
        recurrence = schedules[fold].result(fixed_baseline_service_tokens)
        if fold == 1:
            k1_recurrence = recurrence
            frozen_recurrence = w384["dual_timeline_recurrence"]
            differences = {
                key: {"observed": recurrence.get(key), "frozen": value}
                for key, value in frozen_recurrence.items()
                if recurrence.get(key) != value
            }
            require(
                not differences,
                "K1 does not exactly reproduce every frozen M109-r2 W384 field: "
                + json.dumps(differences, sort_keys=True))
        delta_bits = WEIGHT_BITS + (fold - 1).bit_length()
        records.append({
            "fold_sources_per_update": fold,
            "signed_fold_delta_bits": delta_bits,
            "exact_fold_event_cycles": int(fold_cycle_totals[fold]),
            "event_cycle_reduction_fraction":
                1.0 - fold_cycle_totals[fold] / totals["events"],
            "accumulator_write_cycles_removed":
                totals["events"] - fold_cycle_totals[fold],
            "dual_timeline_recurrence": recurrence,
            "ratio_vs_k1_candidate_cycles":
                k1_recurrence["candidate_cycles"]
                / recurrence["candidate_cycles"],
            "hardware_contract": {
                "resident_weight_vectors_per_output_block": SOURCES,
                "resident_weight_cache_bits":
                    SOURCES * OUTPUT_LANES * WEIGHT_BITS,
                "resident_weight_cache_bytes":
                    SOURCES * OUTPUT_LANES * WEIGHT_BITS // 8,
                "logical_weight_vector_reads_per_fold_cycle_max": fold,
                "lane_addends_per_fold_cycle_max": fold,
                "lane_fold_delta_bits": delta_bits,
                "signed19_accumulator_bound_reused": True,
            },
            "admission": {
                "exact_heldout_cycle_model": True,
                "lossless_set_bit_partition_model": True,
                "rtl_vcs": fold == 1,
                "multi_read_weight_cache_rtl": False,
                "foundry_weight_macro": False,
                "physical_speedup": False,
                "system_speedup": False,
                "headline": False,
            },
        })

    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M122 analyzer changed during execution")
    payload = {
        "schema": "m122_w384_row_synchronous_source_fold_dse_result_v1",
        "status": "PASS_EXACT_HELDOUT_ROW_FOLD_DSE_K1_REPRODUCES_M109",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED_SHA256,
            "heldout_samples": list(range(5, 10)),
            "heldout_records": len(heldout),
        },
        "architecture": {
            "name": "row-synchronous resident-key source fold",
            "window_rows": WINDOW_ROWS,
            "sources": SOURCES,
            "output_blocks": OUTPUT_BLOCKS,
            "output_lanes": OUTPUT_LANES,
            "weight_precision_bits": WEIGHT_BITS,
            "weight_port_bits": WEIGHT_PORT_BITS,
            "weight_beats_per_key": WEIGHT_BEATS_PER_KEY,
            "operation": "partition each 16-bit same-row source mask into groups of at most K set bits; sum their signed INT8 weight vectors lane-wise; issue one signed19 accumulator update per group",
            "numerical_order": "integer addition only; no intermediate saturation or rounding",
        },
        "exact_work": dict(totals),
        "same_row_source_count_histogram": {
            str(key): source_count_histogram[key]
            for key in range(SOURCES + 1)
        },
        "fold_dse": records,
        "model_boundary": {
            "k1_exactly_reproduces_frozen_m109_w384": True,
            "descriptor_fill_remains_one_event_per_cycle": True,
            "pwp_service_unchanged": True,
            "three_weight_load_cycles_per_active_key_unchanged": True,
            "row_fold_selector_cycles_included": "one output group per cycle with no separate selector charge",
            "weight_cache_capacity_bits_included": SOURCES * OUTPUT_LANES * WEIGHT_BITS,
            "multi_read_weight_cache_area_timing_energy": False,
            "rtl_for_k2_k4_k8": False,
            "foundry_sram_macro": False,
            "macro_inclusive_ppa": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m122_w384_row_synchronous_source_fold_dse.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M122 " + " ".join(
        "K{}={:.9f}x/cycles{}".format(
            row["fold_sources_per_update"],
            row["dual_timeline_recurrence"]
               ["same_clock_service_island_ratio"],
            row["dual_timeline_recurrence"]["candidate_cycles"])
        for row in records), flush=True)


if __name__ == "__main__":
    main()
