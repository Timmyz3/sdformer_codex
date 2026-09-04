#!/usr/bin/env python3
"""Independent M132 width algebra, trace placement, and cycle recurrence."""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


HW = Path(__file__).resolve().parents[2]
M109_SCRIPT = HW / ("system_simulator/scripts/"
                    "analyze_m109_r2_window_storage_dual_timeline_frontier.py")
M109_RESULT = HW / ("results/"
                    "m109_r2_window_storage_dual_timeline_frontier_r1_20260824/"
                    "m109_r2_window_storage_dual_timeline_frontier.json")
M108_SCRIPT = HW / ("system_simulator/scripts/"
                    "analyze_m108_w64_fused_pwp_accumulator_schedule.py")
M105_SCRIPT = HW / ("reviews/"
                    "m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/"
                    "audit_m105_bounded_row_transpose.py")
M40_MANIFEST = HW / ("results/"
                     "m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
                     "m40_bottleneck_packed_source_manifest.json")
M72_RESULT = HW / ("results/"
                   "m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
                   "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M41_RESULT = HW / ("results/"
                   "m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/"
                   "m41_h67_ep35_bottleneck_int8_bridge.json")
M129_RESULT = HW / ("results/"
                    "m129_row_admission_bubble_and_descriptor_cost_r1_20260824/"
                    "m129_row_admission_bubble_and_descriptor_cost.json")
M132_RESULT = HW / ("results/"
                    "m132_dualrow512_pwp_compact_k4_schedule_r1_20260824/"
                    "m132_dualrow512_pwp_compact_k4_schedule.json")

EXPECTED = {
    "m109_script": "4eed1e1ef25cdbea0fdd40d1602d6b1eb7661b15b5ae47541c80e149fd060ada",
    "m109_result": "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
    "m108_script": "4404e5825ece95fbf0a28dd580c03c7e9f34bcfa9ec12fa3b66d226a9042cbe2",
    "m105_script": "5e5c07631dd8c4bb328cd234da5c04fde8eb9800d1516b3fe462124b2b661ed5",
    "m40_manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m72_result": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m129_result": "2443a651675763c9e867a2186e83440c323cf20e381e7a49724d6cb0d9ab411e",
    "m132_result": "f74444576ec487b9b1034aced7add0da868a9dea5d4185e0a62c1e33fe1ad755",
}
WIDTH_USES = {8: 11164284, 9: 32360036, 10: 13936011, 11: 1509043}
BEATS_256 = {8: 3, 9: 4, 10: 4, 11: 5}
BEATS_512 = {8: 2, 9: 2, 10: 2, 11: 3}
ROWS = 3000
PARTITIONS = 432
OUTPUT_BLOCKS = 8
WINDOW_ROWS = 384
FOLD = 4
WEIGHT_BEATS = 3


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

    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=hook, parse_constant=reject)


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot load " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndependentSchedule(object):
    def __init__(self):
        self.bank_free = [0, 0]
        self.producer = 0
        self.controller = 0
        self.service = 0
        self.values = Counter()
        self.max_fill = 0
        self.max_service = 0

    def add(self, window, partition, events, groups, pwp, correction_events):
        index = self.values["descriptors"]
        bank = index & 1
        if index == 0 or self.producer > self.bank_free[bank]:
            fill_start = self.producer
        else:
            fill_start = self.bank_free[bank] + 1
            self.values["bank_reacquire_boundaries"] += 1
        self.values["producer_bank_stall_cycles"] += fill_start - self.producer
        fill_cycles = events + 1
        fill_end = fill_start + fill_cycles
        self.producer = fill_end
        self.values["descriptor_fill_cycles"] += fill_cycles
        self.values["controller_dispatch_edges"] += 1
        fill_only_dispatch = fill_end + 1
        dispatch = max(fill_end, self.controller) + 1
        self.values[
            "controller_serialization_delay_sum_vs_fill_only_dispatch"] += (
                dispatch - fill_only_dispatch)

        pwp_start = max(self.service, fill_start)
        self.values["service_idle_cycles"] += pwp_start - self.service
        pwp_end = pwp_start + pwp
        self.values["pwp_service_tokens"] += pwp
        correction = correction_events + WEIGHT_BEATS * groups
        self.values["raw_event_tokens"] += events
        self.values["folded_event_cycles"] += correction_events
        self.values["weight_load_tokens"] += WEIGHT_BEATS * groups
        self.values["correction_service_tokens"] += correction
        if correction:
            correction_start = max(pwp_end, dispatch)
            if dispatch <= pwp_end:
                self.values[
                    "dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
            wait = correction_start - pwp_end
            self.values[
                "exposed_post_pwp_fill_or_dispatch_wait_cycles"] += wait
            self.values["service_idle_cycles"] += wait
            self.service = correction_start + correction
            self.bank_free[bank] = self.service
            self.controller = self.service
        else:
            if dispatch <= pwp_end:
                self.values[
                    "dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
            self.values[
                "empty_release_delay_sum_vs_fill_only_dispatch"] += (
                    dispatch - fill_only_dispatch)
            self.bank_free[bank] = dispatch
            self.controller = dispatch
            self.service = pwp_end
        if pwp == 0:
            self.values["zero_pwp_descriptors"] += 1
        self.max_fill = max(self.max_fill, fill_cycles)
        self.max_service = max(self.max_service, pwp + correction)
        self.values["descriptors"] += 1

        if partition == PARTITIONS - 1:
            ready = max(self.service, self.controller)
            self.values["service_idle_cycles"] += ready - self.service
            self.service = ready + 1
            self.values["accumulator_pipeline_flush_cycles"] += 1
            rows_here = min(WINDOW_ROWS, ROWS - window * WINDOW_ROWS)
            require(rows_here > 0, "invalid window")
            commit = rows_here * OUTPUT_BLOCKS
            self.service += commit
            self.values["accumulator_commit_cycles"] += commit

    def result(self, baseline_service):
        tail = (self.values["accumulator_pipeline_flush_cycles"]
                + self.values["accumulator_commit_cycles"])
        require(self.service == self.values["pwp_service_tokens"]
                + self.values["correction_service_tokens"]
                + self.values["service_idle_cycles"] + tail,
                "cycle conservation")
        baseline = baseline_service + tail
        result = dict(self.values)
        result.update({
            "candidate_cycles": self.service,
            "controller_final_free_cycle": self.controller,
            "fair_fixed8_baseline_cycles": baseline,
            "same_clock_service_island_ratio": baseline / self.service,
            "headroom_to_two_x_cycles": baseline // 2 - self.service,
            "maximum_descriptor_fill_cycles": self.max_fill,
            "maximum_descriptor_service_tokens": self.max_service,
        })
        return result


def build_rows(m105, masks, operator, centers, widths, popcount):
    events = np.zeros((PARTITIONS, ROWS, OUTPUT_BLOCKS), dtype=np.uint16)
    pwp256 = np.zeros((PARTITIONS, ROWS), dtype=np.uint16)
    pwp512 = np.zeros((PARTITIONS, ROWS), dtype=np.uint16)
    uses = Counter()
    beat256 = np.zeros(33, dtype=np.uint8)
    beat512 = np.zeros(33, dtype=np.uint8)
    for width, beats in BEATS_256.items():
        beat256[width] = beats
    for width, beats in BEATS_512.items():
        beat512[width] = beats
    for partition in range(PARTITIONS):
        values = masks[partition]
        center_values = centers[operator, partition]
        order = np.argsort(center_values, kind="stable")
        distance = popcount[np.bitwise_xor(
            values[:, None], center_values[order][None, :])]
        choice = order[distance.argmin(axis=1)]
        best_distance = distance[np.arange(ROWS), distance.argmin(axis=1)]
        best_center = center_values[choice]
        beneficial = (1 + best_distance) < popcount[values]
        delta = np.bitwise_xor(values, best_center)
        selected_width = widths[operator, partition, choice]
        eligible = beneficial[:, None] & (selected_width <= m105.CAP)
        events[partition] = np.where(
            eligible, delta[:, None], values[:, None]).astype(np.uint16)
        pwp256[partition] = np.where(
            eligible, beat256[selected_width], 0).sum(axis=1, dtype=np.uint16)
        pwp512[partition] = np.where(
            eligible, beat512[selected_width], 0).sum(axis=1, dtype=np.uint16)
        for width in WIDTH_USES:
            uses[width] += int(
                (eligible & (selected_width == width)).sum())
    return events, pwp256, pwp512, uses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing output overwrite")
    evidence = {
        "m109_script": M109_SCRIPT, "m109_result": M109_RESULT,
        "m108_script": M108_SCRIPT, "m105_script": M105_SCRIPT,
        "m40_manifest": M40_MANIFEST, "m72_result": M72_RESULT,
        "m41_result": M41_RESULT, "m129_result": M129_RESULT,
        "m132_result": M132_RESULT,
    }
    for label, path in evidence.items():
        require(sha256(path) == EXPECTED[label], "identity drift: " + label)

    algebra256 = sum(WIDTH_USES[w] * BEATS_256[w] for w in WIDTH_USES)
    algebra512 = sum(WIDTH_USES[w] * BEATS_512[w] for w in WIDTH_USES)
    require(algebra256 == 226222255, "256 token algebra")
    require(algebra512 == 119447791, "512 token algebra")

    m109 = load_module("m132_review_m109", M109_SCRIPT)
    m108 = load_module("m132_review_m108", M108_SCRIPT)
    m105 = load_module("m132_review_m105", M105_SCRIPT)
    manifest = strict_json(M40_MANIFEST)
    m72 = strict_json(M72_RESULT)
    m41 = strict_json(M41_RESULT)
    m129 = strict_json(M129_RESULT)
    m132 = strict_json(M132_RESULT)
    frozen_m109 = strict_json(M109_RESULT)
    w384 = next(row for row in frozen_m109["frontier"]
                if int(row["window_rows"]) == WINDOW_ROWS)
    baseline_service = (
        int(w384["dual_timeline_recurrence"]["fair_fixed8_baseline_cycles"])
        - int(w384["dual_timeline_recurrence"]["accumulator_commit_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_pipeline_flush_cycles"]))
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "heldout extent")
    popcount = np.fromiter((int(value).bit_count()
                            for value in range(1 << 16)),
                           dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    schedules = {"pwp256": IndependentSchedule(),
                 "pwp512": IndependentSchedule()}
    totals = Counter()
    uses = Counter()
    starts = np.arange(0, ROWS, WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + WINDOW_ROWS, ROWS)
    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, pwp256_rows, pwp512_rows, record_uses = build_rows(
            m105, masks, record["operator_index"], centers, widths, popcount)
        uses.update(record_uses)
        source_counts = popcount[event_masks]
        event_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(source_counts.sum(axis=2, dtype=np.uint16),
                      axis=1, dtype=np.uint32)), axis=1)
        folded_rows = ((source_counts.astype(np.uint16) + FOLD - 1)
                       // FOLD).sum(axis=2, dtype=np.uint16)
        folded_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(folded_rows, axis=1, dtype=np.uint32)), axis=1)
        pwp256_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(pwp256_rows, axis=1, dtype=np.uint32)), axis=1)
        pwp512_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(pwp512_rows, axis=1, dtype=np.uint32)), axis=1)
        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)
        for window, (start, end) in enumerate(zip(starts, ends)):
            event = event_prefix[:, end] - event_prefix[:, start]
            folded = folded_prefix[:, end] - folded_prefix[:, start]
            p256 = pwp256_prefix[:, end] - pwp256_prefix[:, start]
            p512 = pwp512_prefix[:, end] - pwp512_prefix[:, start]
            for partition in range(PARTITIONS):
                event_count = int(event[partition])
                folded_count = int(folded[partition])
                correction = folded_count + int(folded_count != 0)
                group_count = int(groups[partition, window])
                schedules["pwp256"].add(
                    window, partition, event_count, group_count,
                    int(p256[partition]), correction)
                schedules["pwp512"].add(
                    window, partition, event_count, group_count,
                    int(p512[partition]), correction)
                totals["events"] += event_count
                totals["k4_descriptors"] += folded_count
                totals["pwp256_tokens"] += int(p256[partition])
                totals["pwp512_tokens"] += int(p512[partition])
        print("[M132 INDEPENDENT] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    require(dict(sorted(uses.items())) == WIDTH_USES, "width uses")
    require(totals["pwp256_tokens"] == algebra256, "raw 256 tokens")
    require(totals["pwp512_tokens"] == algebra512, "raw 512 tokens")
    recurrence = {name: schedule.result(baseline_service)
                  for name, schedule in schedules.items()}
    require(recurrence["pwp256"] == m129["cycle_models"]
            ["m128_descriptor_conservative_startup"],
            "compact256 does not exactly reproduce M129")
    require(recurrence["pwp256"] == m132["cycle_models"]
            ["compact_k4_pwp256"], "M132 pwp256 recurrence")
    require(recurrence["pwp512"] == m132["cycle_models"]
            ["compact_k4_dualrow_pwp512"], "M132 pwp512 recurrence")
    c256 = recurrence["pwp256"]["candidate_cycles"]
    c512 = recurrence["pwp512"]["candidate_cycles"]
    ratio = c256 / c512
    fixed8 = recurrence["pwp512"]["same_clock_service_island_ratio"]
    require(c256 == 351479358 and c512 == 245485910, "candidate cycles")
    require(abs(ratio - 1.4317699863100086) < 1e-15, "cycle ratio")
    require(abs(fixed8 - 4.541455955659533) < 1e-15, "fixed8 ratio")

    payload = {
        "schema": "m132_independent_recompute_v1",
        "status": "PASS_EXACT_INDEPENDENT_RECOMPUTE",
        "width_uses": WIDTH_USES,
        "token_algebra": {
            "pwp256_tokens": algebra256,
            "pwp512_tokens": algebra512,
            "reduction_fraction": 1 - algebra512 / algebra256,
            "raw_trace_matches": True,
        },
        "candidate_cycles": {"compact256": c256, "dualrow512": c512},
        "comparisons": {"dualrow512_vs_compact256": ratio,
                        "fixed8_same_clock_service_island": fixed8},
        "compact256_exactly_reproduces_m129": True,
        "claim_boundary": {
            "heldout_same_clock_service_island_cycle_dse": True,
            "dualrow512_rtl": False,
            "bank_conflicts": False,
            "macro_area_energy": False,
            "frequency": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
        "identity": {label + "_sha256": value
                     for label, value in EXPECTED.items()},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m132_independent_recompute.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M132 independent tokens={}/{} cycles={}/{} ratio={:.12f} "
          "fixed8={:.12f} physical=false system=false headline=false".format(
              algebra256, algebra512, c256, c512, ratio, fixed8), flush=True)


if __name__ == "__main__":
    main()
