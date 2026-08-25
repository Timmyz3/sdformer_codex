#!/usr/bin/env python3
"""Build an exact heldout window-size/storage dual-timeline frontier.

Only W64 has the current M106 RTL/VCS admission.  Other rows are explicit
parameterized architecture projections over the same exact heldout event,
group, PWP, flush, and commit ledger; they are not RTL or physical results.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M108_R1_SCRIPT = HW / (
    "system_simulator/scripts/analyze_m108_w64_fused_pwp_accumulator_schedule.py")
M108_R2_SCRIPT = HW / (
    "system_simulator/scripts/analyze_m108_r2_rtl_edge_fused_schedule.py")
M108_R3_RESULT = HW / (
    "results/m108_r3_dual_timeline_fused_schedule_r1_20260824/"
    "m108_r3_dual_timeline_fused_schedule.json")
M105_RESULT = HW / (
    "reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/"
    "m105_bounded_row_transpose_preflight.json")

EXPECTED_SHA256 = {
    "m108_r1_script": "4404e5825ece95fbf0a28dd580c03c7e9f34bcfa9ec12fa3b66d226a9042cbe2",
    "m108_r2_script": "8915ae225f658ac8b4e2d4ca178f870e95a45a85ba647791ead0495b2a29e7f3",
    "m108_r3_result": "d5a4d7c27a91a7735ed4481100d0db3640191357e4617e043378bb367a77dacc",
    "m105_result": "3348b6c02ad97be5b61ffb6f8d5f79578f4551e037097c4f74ac598d2842767b",
}
WINDOW_ROWS = (43, 64, 96, 128, 192, 256, 294, 384, 512, 1024, 3000)
ROWS = 3000
PARTITIONS = 432
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
PARTITION_BITS = 16
SIGNED_ACC_BITS = 24
BASELINE_SERVICE_TOKENS = 1114383288
EXPECTED_EVENTS = 188148490
EXPECTED_PWP_TOKENS = 226222255
EXPECTED_W64_GROUPS = 35140002


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
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, path)
    require(spec is not None and spec.loader is not None,
            "cannot load frozen module: " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EdgeSchedule:
    """Stateful form of the independently corrected M108-r3 recurrence."""

    def __init__(self, window_rows):
        self.window_rows = window_rows
        self.bank_free = [0, 0]
        self.producer_end = 0
        self.controller_free = 0
        self.service_end = 0
        self.values = Counter()
        self.maximum_descriptor_fill_cycles = 0
        self.maximum_descriptor_service_tokens = 0

    def descriptor(self, window, partition, events, groups, pwp_tokens):
        index = self.values["descriptors"]
        bank = index & 1
        if index == 0 or self.producer_end > self.bank_free[bank]:
            fill_start = self.producer_end
        else:
            fill_start = self.bank_free[bank] + 1
            self.values["bank_reacquire_boundaries"] += 1
        self.values["producer_bank_stall_cycles"] += fill_start - self.producer_end
        fill_cycles = events + 1
        fill_end = fill_start + fill_cycles
        self.producer_end = fill_end
        self.values["descriptor_fill_cycles"] += fill_cycles
        self.values["controller_dispatch_edges"] += 1
        fill_only_dispatch = fill_end + 1
        dispatch_ready = max(fill_end, self.controller_free) + 1
        self.values["controller_serialization_delay_sum_vs_fill_only_dispatch"] += (
            dispatch_ready - fill_only_dispatch)

        pwp_start = max(self.service_end, fill_start)
        self.values["service_idle_cycles"] += pwp_start - self.service_end
        pwp_end = pwp_start + pwp_tokens
        self.values["pwp_service_tokens"] += pwp_tokens

        correction = events + 3 * groups
        self.values["correction_service_tokens"] += correction
        if correction:
            correction_start = max(pwp_end, dispatch_ready)
            if dispatch_ready <= pwp_end:
                self.values["dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
            exposed = correction_start - pwp_end
            self.values["exposed_post_pwp_fill_or_dispatch_wait_cycles"] += exposed
            self.values["service_idle_cycles"] += exposed
            self.service_end = correction_start + correction
            self.bank_free[bank] = self.service_end
            self.controller_free = self.service_end
        else:
            if dispatch_ready <= pwp_end:
                self.values["dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
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
            self.values["service_idle_cycles"] += window_ready - self.service_end
            self.service_end = window_ready + 1
            self.values["accumulator_pipeline_flush_cycles"] += 1
            rows_here = min(self.window_rows, ROWS - window * self.window_rows)
            require(rows_here > 0, "invalid final-window row count")
            commit = rows_here * OUTPUT_BLOCKS
            self.service_end += commit
            self.values["accumulator_commit_cycles"] += commit

    def result(self):
        common_tail = (self.values["accumulator_pipeline_flush_cycles"]
                       + self.values["accumulator_commit_cycles"])
        baseline = BASELINE_SERVICE_TOKENS + common_tail
        require(self.service_end == self.values["pwp_service_tokens"]
                + self.values["correction_service_tokens"]
                + self.values["service_idle_cycles"] + common_tail,
                "edge schedule cycle conservation failed")
        result = dict(self.values)
        result.update({
            "candidate_cycles": self.service_end,
            "controller_final_free_cycle": self.controller_free,
            "fair_fixed8_baseline_cycles": baseline,
            "same_clock_service_island_ratio": baseline / float(self.service_end),
            "headroom_to_two_x_cycles": baseline // 2 - self.service_end,
            "maximum_descriptor_fill_cycles": self.maximum_descriptor_fill_cycles,
            "maximum_descriptor_service_tokens":
                self.maximum_descriptor_service_tokens,
        })
        return result


def build_record_rows(m105, m108, masks, operator, centers, widths, popcount):
    """Return natural-row correction masks and exact PWP service beats."""
    event_masks = np.zeros(
        (PARTITIONS, ROWS, OUTPUT_BLOCKS), dtype=np.uint16)
    pwp_rows = np.zeros((PARTITIONS, ROWS), dtype=np.uint16)
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
        pwp_rows[partition] = np.where(
            eligible, m108.PWP_BEATS[selected_widths], 0).sum(
                axis=1, dtype=np.uint16)
    return event_masks, pwp_rows


def window_counts(event_masks, pwp_rows, window_rows, popcount):
    starts = np.arange(0, ROWS, window_rows, dtype=np.intp)
    ends = np.minimum(starts + window_rows, ROWS)
    row_events = popcount[event_masks].sum(axis=2, dtype=np.uint16)
    event_prefix = np.concatenate((
        np.zeros((PARTITIONS, 1), dtype=np.uint32),
        np.cumsum(row_events, axis=1, dtype=np.uint32)), axis=1)
    pwp_prefix = np.concatenate((
        np.zeros((PARTITIONS, 1), dtype=np.uint32),
        np.cumsum(pwp_rows, axis=1, dtype=np.uint32)), axis=1)
    events = (event_prefix[:, ends] - event_prefix[:, starts]).astype(np.int64)
    pwp = (pwp_prefix[:, ends] - pwp_prefix[:, starts]).astype(np.int64)
    union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
    groups = popcount[union].sum(axis=2, dtype=np.uint16).astype(np.int64)
    require(events.shape == groups.shape == pwp.shape,
            "window count shape mismatch")
    return events, groups, pwp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M109 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    for label, path in {
        "m108_r1_script": M108_R1_SCRIPT,
        "m108_r2_script": M108_R2_SCRIPT,
        "m108_r3_result": M108_R3_RESULT,
        "m105_result": M105_RESULT,
    }.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)

    m108 = load_module("m109_frozen_m108", M108_R1_SCRIPT)
    m105 = m108.load_m105_module()
    manifest = m108.strict_json(m105.M40_MANIFEST)
    m72 = m108.strict_json(m105.M72_RESULT)
    m41 = m108.strict_json(m105.M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"] if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "heldout record extent drift")
    popcount = np.fromiter(
        (int(value).bit_count() for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, weight_shas, _ = m105.build_width_catalog(m72, m41)
    schedules = dict((window, EdgeSchedule(window)) for window in WINDOW_ROWS)
    exact_totals = dict((window, Counter()) for window in WINDOW_ROWS)

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, pwp_rows = build_record_rows(
            m105, m108, masks, record["operator_index"], centers, widths, popcount)
        for window in WINDOW_ROWS:
            events, groups, pwp = window_counts(
                event_masks, pwp_rows, window, popcount)
            exact_totals[window]["events"] += int(events.sum())
            exact_totals[window]["groups"] += int(groups.sum())
            exact_totals[window]["pwp_tokens"] += int(pwp.sum())
            for window_index in range(events.shape[1]):
                for partition in range(PARTITIONS):
                    schedules[window].descriptor(
                        window_index, partition,
                        int(events[partition, window_index]),
                        int(groups[partition, window_index]),
                        int(pwp[partition, window_index]))
        print("[M109 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"], record["operator_index"]),
            flush=True)

    frozen_r3 = strict_json(M108_R3_RESULT)["dual_timeline_schedule"]
    m105_result = strict_json(M105_RESULT)
    m105_groups = dict((row["window_rows"], row["active_groups_total"])
                       for row in m105_result["window_results"])
    frontier = []
    for window in WINDOW_ROWS:
        totals = exact_totals[window]
        require(totals["events"] == EXPECTED_EVENTS,
                "event conservation drift W{}".format(window))
        require(totals["pwp_tokens"] == EXPECTED_PWP_TOKENS,
                "PWP conservation drift W{}".format(window))
        if window in m105_groups:
            require(totals["groups"] == m105_groups[window],
                    "M105 group total drift W{}".format(window))
        schedule = schedules[window].result()
        if window == 64:
            require(totals["groups"] == EXPECTED_W64_GROUPS,
                    "W64 group conservation drift")
            for key in (
                    "descriptors", "descriptor_fill_cycles",
                    "producer_bank_stall_cycles", "controller_dispatch_edges",
                    "bank_reacquire_boundaries",
                    "exposed_post_pwp_fill_or_dispatch_wait_cycles",
                    "service_idle_cycles", "pwp_service_tokens",
                    "correction_service_tokens",
                    "accumulator_pipeline_flush_cycles",
                    "accumulator_commit_cycles", "candidate_cycles",
                    "fair_fixed8_baseline_cycles", "controller_final_free_cycle",
                    "controller_serialization_delay_sum_vs_fill_only_dispatch",
                    "empty_release_delay_sum_vs_fill_only_dispatch",
                    "dispatch_hidden_by_pwp_or_prior_lane_descriptors",
                    "zero_pwp_descriptors"):
                require(schedule[key] == frozen_r3[key],
                        "frozen M108-r3 mismatch W64 key=" + key)
        descriptor_bits = 2 * 128 * window * 2
        descriptor_metadata_bits_min = 314
        accumulator_bits = window * OUTPUT_BLOCKS * OUTPUT_LANES * SIGNED_ACC_BITS
        total_bits_min = descriptor_bits + descriptor_metadata_bits_min + accumulator_bits
        frontier.append({
            "window_rows": window,
            "windows_per_phase": math.ceil(ROWS / float(window)),
            "exact_work": dict(totals),
            "dual_timeline_recurrence": schedule,
            "storage_lower_bound": {
                "dual_bank_presence_plus_direction_bits": descriptor_bits,
                "descriptor_bank_metadata_bits_minimum": descriptor_metadata_bits_min,
                "single_window_signed24_accumulator_bits": accumulator_bits,
                "single_window_signed24_accumulator_bytes": accumulator_bits // 8,
                "combined_bits_before_control_ecc_macro_rounding": total_bits_min,
                "combined_bytes_ceiling_before_control_ecc_macro_rounding":
                    (total_bits_min + 7) // 8,
            },
            "admission": {
                "same_clock_dual_timeline_projection": True,
                "exact_heldout_work": True,
                "controller_geometry_vcs": window == 64,
                "full_lane_accumulator_vcs": False,
                "macro_inclusive_ppa": False,
                "physical_speedup": False,
                "system_speedup": False,
                "headline": False,
            },
        })

    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M109 analyzer changed during execution")
    payload = {
        "schema": "m109_r2_window_storage_dual_timeline_frontier_result_v1",
        "status": "PASS_EXACT_HELDOUT_WINDOW_STORAGE_DUAL_TIMELINE_FRONTIER_PORT_CUTS_REMAIN",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED_SHA256,
            "weight_payload_sha256": weight_shas,
        },
        "frontier": frontier,
        "interpretation": {
            "w64": "exact extension of the independently corrected M108-r3 dual-timeline recurrence and current M106-r2 controller geometry",
            "other_windows": "parameterized architecture projections; no corresponding RTL/VCS admission",
            "storage": "lower bound only; excludes controller/grace state, valid or epoch tags, ECC and SRAM macro rounding",
            "ratio": "same-clock precompacted service-island ratio against fixed8 with identical accumulator flush and commit charges",
        },
        "model_boundary": {
            "precompaction_schedule": False,
            "shared_weight_sram_arbitration": False,
            "full_lane_numeric_accumulator_miter": False,
            "macro_inclusive_ppa": False,
            "equal_area": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "m109_r2_window_storage_dual_timeline_frontier.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M109R2 " + " ".join(
        "W{}={:.9f}x/{}B".format(
            row["window_rows"],
            row["dual_timeline_recurrence"]["same_clock_service_island_ratio"],
            row["storage_lower_bound"][
                "combined_bytes_ceiling_before_control_ecc_macro_rounding"])
        for row in frontier), flush=True)


if __name__ == "__main__":
    main()
