#!/usr/bin/env python3
"""Independent M129 held-out work and cycle recurrence reconstruction.

This review deliberately does not import the M129 analyzer or M122's
FoldSchedule.  It reuses only the frozen upstream trace decoding helpers,
implements the recurrence locally, and cross-checks raw reconstruction against
an independent histogram algebra over the frozen M122 result.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


HW = Path(__file__).resolve().parents[2]
M122_RESULT = HW / (
    "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/"
    "m122_w384_row_synchronous_source_fold_dse.json")
M109_SCRIPT = HW / (
    "system_simulator/scripts/"
    "analyze_m109_r2_window_storage_dual_timeline_frontier.py")
M109_RESULT = HW / (
    "results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/"
    "m109_r2_window_storage_dual_timeline_frontier.json")
PRODUCTION_RESULT = HW / (
    "results/m129_row_admission_bubble_and_descriptor_cost_r1_20260824/"
    "m129_row_admission_bubble_and_descriptor_cost.json")

ROWS = 3000
WINDOW_ROWS = 384
PARTITIONS = 432
OUTPUT_BLOCKS = 8
SOURCES = 16
FOLD = 4
WEIGHT_BEATS_PER_KEY = 3

EXPECTED = {
    "m122_result": "be11341211b92d85dc42cb7b79b98a826a782765a4780e1207e7bad5368d27b2",
    "m109_script": "4eed1e1ef25cdbea0fdd40d1602d6b1eb7661b15b5ae47541c80e149fd060ada",
    "m109_result": "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
    "production_result": "2443a651675763c9e867a2186e83440c323cf20e381e7a49724d6cb0d9ab411e",
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
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot load module " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndependentSchedule(object):
    """Independent spelling of the frozen two-bank/controller recurrence."""

    def __init__(self):
        self.bank_available = [0, 0]
        self.producer_cursor = 0
        self.controller_cursor = 0
        self.service_cursor = 0
        self.counts = Counter()
        self.max_fill = 0
        self.max_service = 0

    def consume(self, window, partition, events, groups, pwp, correction_events):
        ordinal = self.counts["descriptors"]
        bank = ordinal % 2
        if ordinal == 0 or self.producer_cursor > self.bank_available[bank]:
            fill_begin = self.producer_cursor
        else:
            fill_begin = self.bank_available[bank] + 1
            self.counts["bank_reacquire_boundaries"] += 1
        self.counts["producer_bank_stall_cycles"] += (
            fill_begin - self.producer_cursor)
        fill_length = events + 1
        fill_finish = fill_begin + fill_length
        self.producer_cursor = fill_finish
        self.counts["descriptor_fill_cycles"] += fill_length
        self.counts["controller_dispatch_edges"] += 1

        fill_only_dispatch = fill_finish + 1
        dispatch = max(fill_finish, self.controller_cursor) + 1
        self.counts[
            "controller_serialization_delay_sum_vs_fill_only_dispatch"] += (
                dispatch - fill_only_dispatch)

        pwp_begin = max(self.service_cursor, fill_begin)
        self.counts["service_idle_cycles"] += pwp_begin - self.service_cursor
        pwp_finish = pwp_begin + pwp
        self.counts["pwp_service_tokens"] += pwp

        correction = correction_events + WEIGHT_BEATS_PER_KEY * groups
        self.counts["raw_event_tokens"] += events
        self.counts["folded_event_cycles"] += correction_events
        self.counts["weight_load_tokens"] += WEIGHT_BEATS_PER_KEY * groups
        self.counts["correction_service_tokens"] += correction
        if correction != 0:
            correction_begin = max(pwp_finish, dispatch)
            if dispatch <= pwp_finish:
                self.counts[
                    "dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
            wait = correction_begin - pwp_finish
            self.counts[
                "exposed_post_pwp_fill_or_dispatch_wait_cycles"] += wait
            self.counts["service_idle_cycles"] += wait
            self.service_cursor = correction_begin + correction
            self.bank_available[bank] = self.service_cursor
            self.controller_cursor = self.service_cursor
        else:
            if dispatch <= pwp_finish:
                self.counts[
                    "dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
            self.counts[
                "empty_release_delay_sum_vs_fill_only_dispatch"] += (
                    dispatch - fill_only_dispatch)
            self.bank_available[bank] = dispatch
            self.controller_cursor = dispatch
            self.service_cursor = pwp_finish
        if pwp == 0:
            self.counts["zero_pwp_descriptors"] += 1

        self.max_fill = max(self.max_fill, fill_length)
        self.max_service = max(self.max_service, pwp + correction)
        self.counts["descriptors"] += 1

        if partition == PARTITIONS - 1:
            synchronized = max(self.service_cursor, self.controller_cursor)
            self.counts["service_idle_cycles"] += (
                synchronized - self.service_cursor)
            self.service_cursor = synchronized + 1
            self.counts["accumulator_pipeline_flush_cycles"] += 1
            rows_here = min(WINDOW_ROWS, ROWS - window * WINDOW_ROWS)
            require(rows_here > 0, "invalid final window")
            commit = rows_here * OUTPUT_BLOCKS
            self.service_cursor += commit
            self.counts["accumulator_commit_cycles"] += commit

    def finish(self, fixed_baseline_service):
        common_tail = (self.counts["accumulator_pipeline_flush_cycles"]
                       + self.counts["accumulator_commit_cycles"])
        require(
            self.service_cursor == self.counts["pwp_service_tokens"]
            + self.counts["correction_service_tokens"]
            + self.counts["service_idle_cycles"] + common_tail,
            "independent cycle conservation failed")
        result = dict(self.counts)
        baseline = fixed_baseline_service + common_tail
        result.update({
            "candidate_cycles": self.service_cursor,
            "controller_final_free_cycle": self.controller_cursor,
            "fair_fixed8_baseline_cycles": baseline,
            "same_clock_service_island_ratio": baseline / self.service_cursor,
            "headroom_to_two_x_cycles": baseline // 2 - self.service_cursor,
            "maximum_descriptor_fill_cycles": self.max_fill,
            "maximum_descriptor_service_tokens": self.max_service,
        })
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing independent output overwrite")
    for label, path in {
        "m122_result": M122_RESULT,
        "m109_script": M109_SCRIPT,
        "m109_result": M109_RESULT,
        "production_result": PRODUCTION_RESULT,
    }.items():
        require(sha256(path) == EXPECTED[label], "identity drift: " + label)

    frozen_m122 = strict_json(M122_RESULT)
    production = strict_json(PRODUCTION_RESULT)
    histogram = {int(key): int(value) for key, value in
                 frozen_m122["same_row_source_count_histogram"].items()}
    histogram_work = {
        "row_blocks_total": sum(histogram.values()),
        "events": sum(count * occurrences
                      for count, occurrences in histogram.items()),
        "active_row_blocks": sum(occurrences
                                 for count, occurrences in histogram.items()
                                 if count != 0),
        "k4_descriptors": sum(((count + FOLD - 1) // FOLD) * occurrences
                              for count, occurrences in histogram.items()),
    }
    require(histogram_work["events"] ==
            int(frozen_m122["exact_work"]["events"]),
            "histogram event algebra mismatch")
    frozen_k4 = next(row for row in frozen_m122["fold_dse"]
                     if int(row["fold_sources_per_update"]) == FOLD)
    require(histogram_work["k4_descriptors"] ==
            int(frozen_k4["exact_fold_event_cycles"]),
            "histogram K4 algebra mismatch")

    m109 = load_module("m129_review_m109", M109_SCRIPT)
    m108 = m109.load_module("m129_review_m108", m109.M108_R1_SCRIPT)
    m105 = m108.load_m105_module()
    manifest = m108.strict_json(m105.M40_MANIFEST)
    m72 = m108.strict_json(m105.M72_RESULT)
    m41 = m108.strict_json(m105.M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "heldout extent drift")

    frozen_m109 = strict_json(M109_RESULT)
    w384 = next(row for row in frozen_m109["frontier"]
                if int(row["window_rows"]) == WINDOW_ROWS)
    fixed_baseline_service = (
        int(w384["dual_timeline_recurrence"]["fair_fixed8_baseline_cycles"])
        - int(w384["dual_timeline_recurrence"]["accumulator_commit_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_pipeline_flush_cycles"]))

    popcount = np.fromiter(
        (bin(value).count("1") for value in range(1 << SOURCES)),
        dtype=np.uint8, count=1 << SOURCES)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    schedules = {
        "m122_ideal_no_interface_bubble": IndependentSchedule(),
        "m125_m127_row_mask_admission": IndependentSchedule(),
        "m128_descriptor_conservative_startup": IndependentSchedule(),
    }
    totals = Counter()
    starts = np.arange(0, ROWS, WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + WINDOW_ROWS, ROWS)

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, pwp_rows = m109.build_record_rows(
            m105, m108, masks, record["operator_index"],
            centers, widths, popcount)
        source_counts = popcount[event_masks]
        event_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(source_counts.sum(axis=2, dtype=np.uint16),
                      axis=1, dtype=np.uint32)), axis=1)
        pwp_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(pwp_rows, axis=1, dtype=np.uint32)), axis=1)
        folded_per_row = ((source_counts.astype(np.uint16) + FOLD - 1)
                          // FOLD).sum(axis=2, dtype=np.uint16)
        folded_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(folded_per_row, axis=1, dtype=np.uint32)), axis=1)
        active_per_row = (source_counts != 0).sum(axis=2, dtype=np.uint16)
        active_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(active_per_row, axis=1, dtype=np.uint32)), axis=1)
        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)

        for window, (start, end) in enumerate(zip(starts, ends)):
            events = event_prefix[:, end] - event_prefix[:, start]
            pwp = pwp_prefix[:, end] - pwp_prefix[:, start]
            folded = folded_prefix[:, end] - folded_prefix[:, start]
            active = active_prefix[:, end] - active_prefix[:, start]
            for partition in range(PARTITIONS):
                event_count = int(events[partition])
                group_count = int(groups[partition, window])
                pwp_count = int(pwp[partition])
                folded_count = int(folded[partition])
                active_count = int(active[partition])
                nonempty = int(folded_count != 0)
                totals["events"] += event_count
                totals["active_row_blocks"] += active_count
                totals["k4_descriptors"] += folded_count
                totals["nonempty_partition_windows"] += nonempty
                schedules["m122_ideal_no_interface_bubble"].consume(
                    window, partition, event_count, group_count,
                    pwp_count, folded_count)
                schedules["m125_m127_row_mask_admission"].consume(
                    window, partition, event_count, group_count,
                    pwp_count, folded_count + active_count)
                schedules["m128_descriptor_conservative_startup"].consume(
                    window, partition, event_count, group_count,
                    pwp_count, folded_count + nonempty)
        print("[M129 INDEPENDENT] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    for key in ("events", "active_row_blocks", "k4_descriptors"):
        require(totals[key] == histogram_work[key],
                "raw vs histogram mismatch: " + key)
    recurrences = {name: schedule.finish(fixed_baseline_service)
                   for name, schedule in schedules.items()}
    require(recurrences == production["cycle_models"],
            "independent recurrence differs from production")
    require(totals["events"] == production["exact_work"]["events"]
            and totals["active_row_blocks"]
            == production["exact_work"]["active_row_blocks"]
            and totals["k4_descriptors"]
            == production["exact_work"]["k4_descriptors"]
            and totals["nonempty_partition_windows"]
            == production["exact_work"]["active_correction_descriptors"],
            "independent exact work differs from production")

    rowmask_cycles = recurrences[
        "m125_m127_row_mask_admission"]["candidate_cycles"]
    descriptor_cycles = recurrences[
        "m128_descriptor_conservative_startup"]["candidate_cycles"]
    ideal_cycles = recurrences[
        "m122_ideal_no_interface_bubble"]["candidate_cycles"]
    descriptor = {
        "row_mask_bits_each": 44,
        "row_mask_total_bits": totals["active_row_blocks"] * 44,
        "m128_bits_each": 53,
        "m128_total_bits": totals["k4_descriptors"] * 53,
        "m130_proposed_bits_each": 35,
        "m130_proposed_total_bits": totals["k4_descriptors"] * 35,
    }
    descriptor["m128_per_item_fraction_vs_row_mask"] = 53 / 44
    descriptor["m128_total_fraction_vs_row_mask"] = (
        descriptor["m128_total_bits"] / descriptor["row_mask_total_bits"])
    descriptor["m130_proposed_total_fraction_vs_row_mask"] = (
        descriptor["m130_proposed_total_bits"]
        / descriptor["row_mask_total_bits"])
    require(descriptor["m128_total_bits"] ==
            production["descriptor_traffic"]["m128_total_bits"],
            "M128 bit accounting mismatch")
    require(descriptor["row_mask_total_bits"] ==
            production["descriptor_traffic"]["row_mask_total_bits"],
            "row-mask bit accounting mismatch")

    payload = {
        "schema": "m129_independent_recompute_v1",
        "status": "PASS_EXACT_INDEPENDENT_RECOMPUTE",
        "histogram_algebra": histogram_work,
        "raw_trace_work": dict(totals),
        "candidate_cycles": {
            "m122_ideal": ideal_cycles,
            "m125_m127_row_mask": rowmask_cycles,
            "m128_conservative_descriptor": descriptor_cycles,
        },
        "candidate_cycle_speedup_m128_vs_row_mask":
            rowmask_cycles / descriptor_cycles,
        "cycle_charge_identities": {
            "m122_folded_event_cycles": totals["k4_descriptors"],
            "row_mask_folded_event_cycles":
                totals["k4_descriptors"] + totals["active_row_blocks"],
            "m128_folded_event_cycles":
                totals["k4_descriptors"]
                + totals["nonempty_partition_windows"],
        },
        "descriptor_traffic": descriptor,
        "identity": {
            label + "_sha256": expected for label, expected in EXPECTED.items()
        },
        "claim_boundary": {
            "module_cycle_ab_only": True,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
            "m130_35bit_status": "proposed_only",
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m129_independent_recompute.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M129 independent active={} k4={} nonempty={} cycles={}/{}/{} "
          "speedup={:.12f} physical=false system=false headline=false".format(
              totals["active_row_blocks"], totals["k4_descriptors"],
              totals["nonempty_partition_windows"], ideal_cycles,
              rowmask_cycles, descriptor_cycles,
              rowmask_cycles / descriptor_cycles), flush=True)


if __name__ == "__main__":
    main()
