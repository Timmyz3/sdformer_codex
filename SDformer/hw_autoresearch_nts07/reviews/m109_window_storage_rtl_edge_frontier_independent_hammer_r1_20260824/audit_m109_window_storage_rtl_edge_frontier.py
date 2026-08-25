#!/usr/bin/env python3
"""Independent raw-work, recurrence, fairness, and storage audit for M109.

The raw decoder is loaded from the prior independent M108 hammer, never from
the M109/M108 producer analyzers.  Natural-row event/PWP work is rebuilt once
from frozen M40/M72/M41 inputs, then independently reduced at four window
sizes.  Both the published recurrence and the prior-drain-aware dual-timeline
recurrence are evaluated over the identical ordered descriptors.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

import numpy as np


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
WINDOWS_TO_AUDIT = (43, 64, 294, 384)
ROWS = 3000
PARTITIONS = 432
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
SIGNED_ACC_BITS = 24
BASELINE_EVENTS = 371461096
BASELINE_TOKENS = BASELINE_EVENTS * 3
EXPECTED_EVENTS = 188148490
EXPECTED_PWP_TOKENS = 226222255
EXPECTED_GROUPS = {
    43: 46867834,
    64: 35140002,
    294: 10395056,
    384: 8271296,
}

M109_ANALYZER = HW / "system_simulator/scripts/analyze_m109_window_storage_rtl_edge_frontier.py"
M109_CONTRACT = HW / "contracts/m109_window_storage_rtl_edge_frontier_contract_r1_20260824.json"
M109_DIR = HW / "results/m109_window_storage_rtl_edge_frontier_r1_20260824"
M109_RESULT = M109_DIR / "m109_window_storage_rtl_edge_frontier.json"
M109_RUN = M109_DIR / "RUN_COMPLETE.txt"
M109_MANIFEST = M109_DIR / "manifest.sha256"
M108_RESULT = HW / "results/m108_r2_rtl_edge_fused_schedule_r1_20260824/m108_r2_rtl_edge_fused_schedule.json"
M108_AUDIT_SCRIPT = HW / (
    "reviews/m108_r2_rtl_edge_fused_schedule_independent_hammer_r1_20260824/"
    "audit_m108_r2_rtl_edge_fused_schedule.py")
M108_AUDIT_RESULT = HW / (
    "reviews/m108_r2_rtl_edge_fused_schedule_independent_hammer_r1_20260824/"
    "m108_r2_rtl_edge_fused_schedule_independent_audit.json")
M105_RESULT = HW / (
    "reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/"
    "m105_bounded_row_transpose_preflight.json")
M40_MANIFEST = HW / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
    "m40_bottleneck_packed_source_manifest.json")
M72_RESULT = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M41_RESULT = HW / (
    "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/"
    "m41_h67_ep35_bottleneck_int8_bridge.json")
M106_RTL = HW / "rtl_m106/m106_bounded_bitmap_transpose_scheduler.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SHA = {
    M109_ANALYZER: "f4fd84d8797012cb3b3faeba599e6b2a856bdb69083b1fad23aedfa788de2637",
    M109_CONTRACT: "61b41e77d59ef5f6e881cd65aa017a8274e376d81fa23a5c2c6f099ecc9bb752",
    M109_RESULT: "9f5ce436537571ea022289b1c23c12ebe37f616f8ff19280939df769ced8d12f",
    M109_RUN: "ab443a66f30e974b65772d015a05d8b8667457ac7fbe6b8f7be6fb55ee734679",
    M109_MANIFEST: "64dce8ead730f021a55c3e292a1964680384be8388ff1f408e746bbf9a026b4b",
    M108_RESULT: "2813ea18de27ac59d45e48897f2c217a3a67828c4b17fcbf93795cab9950582a",
    M108_AUDIT_SCRIPT: "a1f29d7e0f131deee76d15cdc7cc953e90d68c1d01a600caea66e43ba60abdd2",
    M108_AUDIT_RESULT: "7db3ba30936ae505d39ac8bd1134c8877ed3c68234850d94ccdc2d1e65e7cfc7",
    M105_RESULT: "3348b6c02ad97be5b61ffb6f8d5f79578f4551e037097c4f74ac598d2842767b",
    M40_MANIFEST: "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    M72_RESULT: "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    M41_RESULT: "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    M106_RTL: "a6937765aea87269c3d38123b656c72b7ee400e36b0d634f21ab9c7dbdefc0b7",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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


def verify_manifest(path, base):
    checked = 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, name = raw.split(None, 1)
        target = Path(base) / name.strip()
        require(target.is_file(), "manifest target missing: " + str(target))
        require(sha256(target) == expected, "manifest target drift: " + str(target))
        checked += 1
    return checked


def load_independent_raw_decoder():
    spec = importlib.util.spec_from_file_location(
        "m109_prior_independent_raw_decoder", M108_AUDIT_SCRIPT)
    require(spec is not None and spec.loader is not None,
            "cannot load prior independent raw decoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_natural_rows(masks, operator, centers, widths, popcount, pwp_beats):
    event_masks = np.zeros((PARTITIONS, ROWS, OUTPUT_BLOCKS), dtype=np.uint16)
    pwp_rows = np.zeros((PARTITIONS, ROWS), dtype=np.uint16)
    baseline_events = 0
    for partition in range(PARTITIONS):
        values = masks[partition]
        center_values = centers[operator, partition]
        order = np.argsort(center_values, kind="stable")
        ordered = center_values[order]
        distance = popcount[np.bitwise_xor(values[:, None], ordered[None, :])]
        ordered_choice = distance.argmin(axis=1)
        choice = order[ordered_choice]
        best_distance = distance[np.arange(ROWS), ordered_choice]
        best_center = center_values[choice]
        population = popcount[values]
        baseline_events += int(population.sum()) * OUTPUT_BLOCKS
        beneficial = (1 + best_distance) < population
        selected_width = widths[operator, partition, choice]
        eligible = beneficial[:, None] & (selected_width <= 11)
        delta = np.bitwise_xor(values, best_center)
        event_masks[partition] = np.where(
            eligible, delta[:, None], values[:, None]).astype(np.uint16)
        pwp_rows[partition] = np.where(
            eligible, pwp_beats[selected_width], 0).sum(axis=1, dtype=np.uint16)
    return event_masks, pwp_rows, baseline_events


def reduce_window(event_masks, pwp_rows, window_rows, popcount):
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
            "window array shape mismatch")
    require(np.all(events >= groups), "event/group relation mismatch")
    require(np.all(events[groups == 0] == 0), "event in empty descriptor")
    return events, groups, pwp


class Recurrence:
    def __init__(self, window_rows, prior_drain_aware):
        self.window_rows = window_rows
        self.prior_drain_aware = prior_drain_aware
        self.bank_free = [0, 0]
        self.producer_end = 0
        self.controller_free = 0
        self.lane_end = 0
        self.values = Counter()
        self.maximum_fill = 0
        self.maximum_service = 0
        self.digest = hashlib.sha256()

    def descriptor(self, sample, operator, window, partition,
                   events, groups, pwp_tokens):
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
        if self.prior_drain_aware:
            dispatch_edge = max(fill_end, self.controller_free) + 1
            self.values["controller_serialization_delay_sum_vs_fill_only_dispatch"] += (
                dispatch_edge - fill_only_dispatch)
        else:
            dispatch_edge = fill_only_dispatch

        pwp_start = max(self.lane_end, fill_start)
        self.values["service_idle_cycles"] += pwp_start - self.lane_end
        pwp_end = pwp_start + pwp_tokens
        self.values["pwp_service_tokens"] += pwp_tokens
        if pwp_tokens == 0:
            self.values["zero_pwp_descriptors"] += 1

        correction = events + 3 * groups
        self.values["correction_service_tokens"] += correction
        if correction:
            correction_start = max(pwp_end, dispatch_edge)
            if dispatch_edge <= pwp_end:
                self.values[
                    "dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
            exposed = correction_start - pwp_end
            self.values["exposed_post_pwp_fill_or_dispatch_wait_cycles"] += exposed
            self.values["service_idle_cycles"] += exposed
            correction_end = correction_start + correction
            self.bank_free[bank] = correction_end
            self.lane_end = correction_end
            if self.prior_drain_aware:
                self.controller_free = correction_end
        else:
            if dispatch_edge <= pwp_end:
                self.values[
                    "dispatch_hidden_by_pwp_or_prior_lane_descriptors"] += 1
            self.values["empty_release_delay_sum_vs_fill_only_dispatch"] += (
                dispatch_edge - fill_only_dispatch)
            self.bank_free[bank] = dispatch_edge
            self.lane_end = pwp_end
            if self.prior_drain_aware:
                self.controller_free = dispatch_edge

        self.maximum_fill = max(self.maximum_fill, fill_cycles)
        self.maximum_service = max(self.maximum_service, pwp_tokens + correction)
        self.values["descriptors"] += 1

        if partition == PARTITIONS - 1:
            if self.prior_drain_aware:
                ready = max(self.lane_end, self.controller_free)
            else:
                ready = max(self.lane_end, self.bank_free[bank])
            self.values["service_idle_cycles"] += ready - self.lane_end
            self.lane_end = ready + 1
            self.values["accumulator_pipeline_flush_cycles"] += 1
            rows_here = min(self.window_rows, ROWS - window * self.window_rows)
            require(rows_here > 0, "invalid final-window row count")
            commit = rows_here * OUTPUT_BLOCKS
            self.values["accumulator_commit_cycles"] += commit
            self.lane_end += commit

        self.digest.update(struct.pack(
            "<BBHHIII", sample, operator, window, partition,
            events, groups, pwp_tokens))

    def result(self):
        correction = EXPECTED_EVENTS + 3 * EXPECTED_GROUPS[self.window_rows]
        common_tail = (self.values["accumulator_pipeline_flush_cycles"]
                       + self.values["accumulator_commit_cycles"])
        require(self.values["pwp_service_tokens"] == EXPECTED_PWP_TOKENS,
                "PWP conservation mismatch")
        require(self.values["correction_service_tokens"] == correction,
                "correction conservation mismatch")
        require(self.lane_end == EXPECTED_PWP_TOKENS + correction
                + self.values["service_idle_cycles"] + common_tail,
                "candidate cycle conservation mismatch")
        baseline = BASELINE_TOKENS + common_tail
        result = dict(self.values)
        result.update({
            "producer_final_cycle": self.producer_end,
            "controller_final_free_cycle": self.controller_free,
            "candidate_cycles": self.lane_end,
            "fair_fixed8_baseline_cycles": baseline,
            "same_clock_service_island_ratio": baseline / float(self.lane_end),
            "headroom_to_two_x_cycles": baseline // 2 - self.lane_end,
            "maximum_descriptor_fill_cycles": self.maximum_fill,
            "maximum_descriptor_service_tokens": self.maximum_service,
            "ordered_descriptor_sha256": self.digest.hexdigest(),
        })
        return result


def storage_lower_bound(window_rows):
    descriptor_bits = 2 * 128 * window_rows * 2
    metadata_bits = 314
    accumulator_bits = (window_rows * OUTPUT_BLOCKS
                        * OUTPUT_LANES * SIGNED_ACC_BITS)
    combined = descriptor_bits + metadata_bits + accumulator_bits
    return {
        "dual_bank_presence_plus_direction_bits": descriptor_bits,
        "descriptor_bank_metadata_bits_minimum": metadata_bits,
        "single_window_signed24_accumulator_bits": accumulator_bits,
        "single_window_signed24_accumulator_bytes": accumulator_bits // 8,
        "combined_bits_before_control_ecc_macro_rounding": combined,
        "combined_bytes_ceiling_before_control_ecc_macro_rounding":
            (combined + 7) // 8,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing independent output overwrite")
    start_sha = sha256(Path(__file__).resolve())

    identity = {}
    for path, expected in EXPECTED_SHA.items():
        actual = sha256(path)
        require(actual == expected, "identity drift {} {}".format(path, actual))
        identity[str(path.relative_to(HW))] = actual
    require(verify_manifest(M109_MANIFEST, M109_DIR) == 4,
            "M109 manifest extent drift")

    producer = strict_json(M109_RESULT)
    contract = strict_json(M109_CONTRACT)
    m108 = strict_json(M108_RESULT)
    m108_audit = strict_json(M108_AUDIT_RESULT)
    m105 = strict_json(M105_RESULT)
    m40 = strict_json(M40_MANIFEST)
    m72 = strict_json(M72_RESULT)
    m41 = strict_json(M41_RESULT)
    independent = load_independent_raw_decoder()

    require(m108_audit["p0_closure_audit"]
            ["prior_drain_in_order_dispatch_dependency_added"] is False,
            "prior M108 recurrence P0 state drift")
    rtl_text = M106_RTL.read_text(encoding="utf-8")
    require("if (!drain_active_q" in rtl_text
            and "bank_state_q[next_drain_bank_q] == BANK_READY" in rtl_text,
            "M106 prior-drain dispatch guard missing")

    popcount = np.fromiter(
        (int(value).bit_count() for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = independent.centers_array(m72)
    widths, width_histogram, weight_shas = independent.build_widths(m72, m41)
    heldout = sorted((row for row in m40["records"]
                      if row["sample_id"] in range(5, 10)),
                     key=lambda row: (row["sample_id"], row["operator_index"]))
    require([(row["sample_id"], row["operator_index"]) for row in heldout]
            == [(sample, operator) for sample in range(5, 10)
                for operator in range(4)], "heldout order drift")

    published_models = {w: Recurrence(w, False) for w in WINDOWS_TO_AUDIT}
    corrected_models = {w: Recurrence(w, True) for w in WINDOWS_TO_AUDIT}
    work = {w: Counter() for w in WINDOWS_TO_AUDIT}
    raw_baseline_events = 0

    for record_index, record in enumerate(heldout, 1):
        masks = independent.decode_masks(record)
        event_masks, pwp_rows, baseline_events = build_natural_rows(
            masks, record["operator_index"], centers, widths,
            popcount, independent.PWP_BEATS)
        raw_baseline_events += baseline_events
        for window_rows in WINDOWS_TO_AUDIT:
            events, groups, pwp = reduce_window(
                event_masks, pwp_rows, window_rows, popcount)
            work[window_rows]["events"] += int(events.sum())
            work[window_rows]["groups"] += int(groups.sum())
            work[window_rows]["pwp_tokens"] += int(pwp.sum())
            for window in range(events.shape[1]):
                for partition in range(PARTITIONS):
                    args_row = (
                        record["sample_id"], record["operator_index"],
                        window, partition,
                        int(events[partition, window]),
                        int(groups[partition, window]),
                        int(pwp[partition, window]))
                    published_models[window_rows].descriptor(*args_row)
                    corrected_models[window_rows].descriptor(*args_row)
        print("[M109 INDEPENDENT] {}/20 sample={} op={}".format(
            record_index, record["sample_id"], record["operator_index"]),
            flush=True)

    require(raw_baseline_events == BASELINE_EVENTS,
            "fixed8 baseline event reconstruction drift")
    m105_groups = {int(window): groups for window, groups in
                   m105["group_totals_scan_1_to_512"].items()}
    producer_rows = {row["window_rows"]: row for row in producer["frontier"]}
    audit_rows = []
    for window_rows in WINDOWS_TO_AUDIT:
        totals = work[window_rows]
        require(totals == Counter({
            "events": EXPECTED_EVENTS,
            "groups": EXPECTED_GROUPS[window_rows],
            "pwp_tokens": EXPECTED_PWP_TOKENS,
        }), "raw work aggregate mismatch W{} {}".format(window_rows, totals))
        require(m105_groups[window_rows] == totals["groups"],
                "M105 group total mismatch W{}".format(window_rows))
        published_schedule = published_models[window_rows].result()
        corrected_schedule = corrected_models[window_rows].result()
        require(published_schedule["ordered_descriptor_sha256"]
                == corrected_schedule["ordered_descriptor_sha256"],
                "ordered descriptor digest mismatch W{}".format(window_rows))

        row = producer_rows[window_rows]
        require(row["exact_work"] == dict(totals),
                "M109 exact work field mismatch W{}".format(window_rows))
        for key, value in row["rtl_edge_recurrence"].items():
            if isinstance(value, float):
                require(math.isclose(published_schedule[key], value,
                                     rel_tol=0.0, abs_tol=1e-15),
                        "published ratio mismatch W{} {}".format(window_rows, key))
            else:
                require(published_schedule[key] == value,
                        "published schedule mismatch W{} {}".format(window_rows, key))
        storage = storage_lower_bound(window_rows)
        require(storage == row["storage_lower_bound"],
                "storage arithmetic mismatch W{}".format(window_rows))
        require(row["windows_per_phase"]
                == int(math.ceil(ROWS / float(window_rows))),
                "windows/phase mismatch W{}".format(window_rows))
        require(published_schedule["accumulator_pipeline_flush_cycles"]
                == 20 * row["windows_per_phase"],
                "flush arithmetic mismatch W{}".format(window_rows))
        require(published_schedule["accumulator_commit_cycles"]
                == 20 * ROWS * OUTPUT_BLOCKS,
                "commit arithmetic mismatch W{}".format(window_rows))
        require(published_schedule["fair_fixed8_baseline_cycles"]
                == BASELINE_TOKENS
                + published_schedule["accumulator_pipeline_flush_cycles"]
                + published_schedule["accumulator_commit_cycles"],
                "baseline arithmetic mismatch W{}".format(window_rows))

        audit_rows.append({
            "window_rows": window_rows,
            "work": dict(totals),
            "published_recurrence_reproduction": published_schedule,
            "prior_drain_aware_dual_timeline": corrected_schedule,
            "difference": {
                "candidate_underestimate_cycles":
                    corrected_schedule["candidate_cycles"]
                    - published_schedule["candidate_cycles"],
                "ratio_overstatement":
                    published_schedule["same_clock_service_island_ratio"]
                    - corrected_schedule["same_clock_service_island_ratio"],
                "headroom_overstatement_cycles":
                    published_schedule["headroom_to_two_x_cycles"]
                    - corrected_schedule["headroom_to_two_x_cycles"],
            },
            "storage_reconstruction": storage,
            "controller_geometry_vcs": row["admission"]["controller_geometry_vcs"],
        })

    w64 = producer_rows[64]
    require(w64["rtl_edge_recurrence"] == m108["rtl_edge_schedule"],
            "W64 is not field-exact M108-r2")
    require(contract["frozen_observations"]["w64"]
            ["m108_r2_integer_field_match"] is True,
            "M109 contract W64 identity statement drift")
    require(all(producer_rows[w]["admission"]["controller_geometry_vcs"] is False
                for w in WINDOWS_TO_AUDIT if w != 64),
            "non-W64 geometry incorrectly admitted as VCS")

    audit_by_window = {row["window_rows"]: row for row in audit_rows}
    published_w294 = audit_by_window[294]["published_recurrence_reproduction"]
    published_w384 = audit_by_window[384]["published_recurrence_reproduction"]
    corrected_w294 = audit_by_window[294]["prior_drain_aware_dual_timeline"]
    corrected_w384 = audit_by_window[384]["prior_drain_aware_dual_timeline"]

    payload = {
        "schema": "m109_window_storage_rtl_edge_frontier_independent_audit_v1",
        "status": "RAW_ARITHMETIC_REPRODUCED_INHERITED_M108_CONTROLLER_SERIALIZATION_P0",
        "identity": identity,
        "m109_manifest_entries_verified": 4,
        "raw_reconstruction": {
            "heldout_records": len(heldout),
            "fixed8_baseline_events": raw_baseline_events,
            "fixed8_baseline_tokens": BASELINE_TOKENS,
            "weight_payload_sha256": weight_shas,
            "weight_width_histogram": dict(sorted(width_histogram.items())),
        },
        "audited_windows": audit_rows,
        "w64_field_exactness": {
            "m109_equals_frozen_m108_r2_all_schedule_fields": True,
            "independent_published_recurrence_equals_both": True,
            "consequence": "M109 also inherits the unresolved M108-r2 prior-drain serialization P0.",
        },
        "threshold_attack": {
            "published_w294_ratio": published_w294["same_clock_service_island_ratio"],
            "published_w294_below_2p5":
                published_w294["same_clock_service_island_ratio"] < 2.5,
            "published_w384_ratio": published_w384["same_clock_service_island_ratio"],
            "published_w384_above_2p5":
                published_w384["same_clock_service_island_ratio"] > 2.5,
            "corrected_w294_ratio": corrected_w294["same_clock_service_island_ratio"],
            "corrected_w294_below_2p5":
                corrected_w294["same_clock_service_island_ratio"] < 2.5,
            "corrected_w384_ratio": corrected_w384["same_clock_service_island_ratio"],
            "corrected_w384_above_2p5":
                corrected_w384["same_clock_service_island_ratio"] > 2.5,
            "verdict": "threshold ordering tested under both recurrences",
        },
        "baseline_fairness": {
            "raw_fixed8_events_reconstructed": True,
            "three_tokens_per_fixed8_event": True,
            "candidate_and_baseline_share_identical_commit_and_flush": True,
            "controller_and_descriptor_ingress_edges_charged_to_baseline": False,
            "verdict": "reproducible service-token denominator, not equal-controller end-to-end cycles",
        },
        "projection_boundary": {
            "w64_controller_geometry_vcs": True,
            "non_w64_controller_geometry_vcs": False,
            "non_w64_rows_are_parameterized_architecture_projections": True,
            "full_lane_accumulator_vcs": False,
            "physical_speedup": False,
            "equal_area": False,
            "system_speedup": False,
            "headline": False,
        },
        "root_cause": (
            "M109 deliberately reuses the published M108-r2 recurrence. That recurrence "
            "sets dispatch_ready to fill_end+1 but current M106 dispatch is also gated by "
            "the completion of the prior drain. The missing controller_free dependency "
            "understates some descriptor and empty-release edges."
        ),
        "docs_359_sha256_unchanged": sha256(DOC359),
        "producer_analyzer_executed": False,
        "production_files_modified": False,
    }
    require(sha256(Path(__file__).resolve()) == start_sha,
            "independent auditor changed during execution")
    require(sha256(DOC359) == EXPECTED_SHA[DOC359],
            "docs/359 changed during audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M109 raw arithmetic; inherited recurrence P0 "
          "W294={:.12f} W384={:.12f}".format(
              corrected_w294["same_clock_service_island_ratio"],
              corrected_w384["same_clock_service_island_ratio"]), flush=True)


if __name__ == "__main__":
    main()
