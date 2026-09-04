#!/usr/bin/env python3
"""Replay W64 PWP/correction service with a bounded accumulator schedule."""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M105_DIR = HW / "reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824"
M105_SCRIPT = M105_DIR / "audit_m105_bounded_row_transpose.py"
M105_RESULT = M105_DIR / "m105_bounded_row_transpose_preflight.json"
M107_RESULT = HW / (
    "results/m107_w64_pingpong_schedule_replay_r1_20260824/"
    "m107_w64_pingpong_schedule_replay.json")
M107_MANIFEST = HW / (
    "results/m107_w64_pingpong_schedule_replay_r1_20260824/manifest.sha256")

EXPECTED_SHA256 = {
    "m105_script": "5e5c07631dd8c4bb328cd234da5c04fde8eb9800d1516b3fe462124b2b661ed5",
    "m105_result": "3348b6c02ad97be5b61ffb6f8d5f79578f4551e037097c4f74ac598d2842767b",
    "m107_result": "0c613c2da4eb3e860ccc33dfbbe3fba0bb1424c19ab12af339e54178d03333db",
}
ROWS = 3000
WIN_ROWS = 64
WINDOWS_PER_PHASE = 47
PARTITIONS = 432
OUTPUT_BLOCKS = 8
EXPECTED_EVENTS = 188148490
EXPECTED_GROUPS = 35140002
EXPECTED_PWP_USES = {8: 11164284, 9: 32360036, 10: 13936011, 11: 1509043}
EXPECTED_PWP_TOKENS = 226222255
BASELINE_SERVICE_TOKENS = 1114383288
PWP_BEATS = np.zeros(33, dtype=np.uint8)
PWP_BEATS[8:12] = (3, 4, 4, 5)


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


def load_m105_module():
    spec = importlib.util.spec_from_file_location("m105_frozen_auditor", M105_SCRIPT)
    require(spec is not None and spec.loader is not None,
            "cannot load frozen M105 auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prefix_windows(per_row):
    starts = np.arange(0, ROWS, WIN_ROWS, dtype=np.intp)
    ends = np.minimum(starts + WIN_ROWS, ROWS)
    prefix = np.concatenate((
        np.zeros((per_row.shape[0], 1), dtype=np.uint32),
        np.cumsum(per_row, axis=1, dtype=np.uint32),
    ), axis=1)
    return (prefix[:, ends] - prefix[:, starts]).astype(np.int64)


def descriptor_counts(m105, masks, op, centers, widths, popcount):
    events = np.zeros((PARTITIONS, WINDOWS_PER_PHASE), dtype=np.int64)
    groups = np.zeros_like(events)
    pwp_tokens = np.zeros_like(events)
    pwp_updates = np.zeros_like(events)
    positive_events = np.zeros_like(events)
    negative_events = np.zeros_like(events)
    width_uses = Counter()
    coefficient_checks = 0

    starts = np.arange(0, ROWS, WIN_ROWS, dtype=np.intp)
    for partition in range(PARTITIONS):
        values = masks[partition]
        center_values = centers[op, partition]
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
        selected_widths = widths[op, partition, best_index]
        eligible = beneficial[:, None] & (selected_widths <= m105.CAP)

        event_masks = np.where(eligible, delta[:, None], values[:, None]).astype(np.uint16)
        negative_masks = np.where(
            eligible,
            np.bitwise_and(best_center[:, None],
                           np.bitwise_xor(values[:, None], np.uint16(0xffff))),
            np.uint16(0),
        ).astype(np.uint16)
        require(not np.any(np.bitwise_and(
            negative_masks, np.bitwise_xor(event_masks, np.uint16(0xffff)))),
            "negative direction escaped event support")
        positive_masks = np.bitwise_xor(event_masks, negative_masks)

        # Exact bit-coefficient miter.  Positive bits are disjoint from the
        # center and negative bits are a subset of it, so this bitwise form is
        # equivalent to checking center_bit + positive_bit - negative_bit for
        # all 16 sources without admitting integer-mask carries.
        center_matrix = np.broadcast_to(best_center[:, None], eligible.shape)
        require(not np.any(np.where(
            eligible, np.bitwise_and(positive_masks, center_matrix), 0)),
                "positive correction overlaps a center bit")
        require(not np.any(np.where(
            eligible,
            np.bitwise_and(
                negative_masks,
                np.bitwise_xor(center_matrix, np.uint16(0xffff))),
            0)),
            "negative correction is not a center subset")
        reconstructed = np.where(
            eligible,
            np.bitwise_and(np.bitwise_or(center_matrix, positive_masks),
                           np.bitwise_xor(negative_masks, np.uint16(0xffff))),
            positive_masks,
        ).astype(np.uint16)
        require(np.array_equal(
            reconstructed,
            np.broadcast_to(values[:, None], eligible.shape)),
            "source coefficient miter mismatch")
        coefficient_checks += int(eligible.size) * m105.PARTITION_BITS

        row_events = popcount[event_masks].sum(axis=1, dtype=np.uint16)
        row_negative = popcount[negative_masks].sum(axis=1, dtype=np.uint16)
        row_positive = popcount[positive_masks].sum(axis=1, dtype=np.uint16)
        require(np.array_equal(row_events, row_negative + row_positive),
                "event direction conservation mismatch")
        events[partition] = prefix_windows(row_events[None, :])[0]
        negative_events[partition] = prefix_windows(row_negative[None, :])[0]
        positive_events[partition] = prefix_windows(row_positive[None, :])[0]

        union = np.bitwise_or.reduceat(event_masks, starts, axis=0)
        groups[partition] = popcount[union].sum(axis=1, dtype=np.uint16)
        row_pwp_updates = eligible.sum(axis=1, dtype=np.uint8)
        row_pwp_tokens = np.where(eligible, PWP_BEATS[selected_widths], 0).sum(
            axis=1, dtype=np.uint16)
        pwp_updates[partition] = prefix_windows(row_pwp_updates[None, :])[0]
        pwp_tokens[partition] = prefix_windows(row_pwp_tokens[None, :])[0]
        for width in range(8, 12):
            width_uses[width] += int((eligible & (selected_widths == width)).sum())

    require(np.array_equal(events, positive_events + negative_events),
            "record event direction total mismatch")
    return {
        "events": events,
        "groups": groups,
        "pwp_tokens": pwp_tokens,
        "pwp_updates": pwp_updates,
        "positive_events": positive_events,
        "negative_events": negative_events,
        "width_uses": width_uses,
        "coefficient_checks": coefficient_checks,
    }


def simulate_window_major(sequence):
    bank_free = [0, 0]
    producer_end = 0
    service_end = 0
    service_idle = 0
    producer_stall = 0
    descriptor_fill_cycles = 0
    correction_tokens = 0
    pwp_tokens = 0
    commit_cycles = 0
    pipeline_flush_cycles = 0
    max_fill = 0
    max_service = 0

    for index, row in enumerate(sequence):
        bank = index & 1
        fill_start = max(producer_end, bank_free[bank])
        producer_stall += fill_start - producer_end
        fill_cycles = row["events"] + 1
        fill_end = fill_start + fill_cycles
        producer_end = fill_end

        # PWP and correction share one 256-bit service lane.  The descriptor
        # producer is independent, so its bitmap fill overlaps PWP service.
        pwp_start = max(service_end, fill_start)
        service_idle += pwp_start - service_end
        pwp_end = pwp_start + row["pwp_tokens"]
        correction_start = max(pwp_end, fill_end)
        service_idle += correction_start - pwp_end
        correction = row["events"] + 3 * row["groups"]
        correction_end = correction_start + correction
        bank_free[bank] = correction_end
        service_end = correction_end

        descriptor_fill_cycles += fill_cycles
        correction_tokens += correction
        pwp_tokens += row["pwp_tokens"]
        max_fill = max(max_fill, fill_cycles)
        max_service = max(max_service, row["pwp_tokens"] + correction)

        if row["partition"] == PARTITIONS - 1:
            # One cycle retires the final pipelined vector before the bounded
            # accumulator is committed.  Commit one 96-lane vector per
            # (valid row, output block); this cost is also charged to baseline.
            service_end += 1
            pipeline_flush_cycles += 1
            rows_here = min(WIN_ROWS, ROWS - row["window"] * WIN_ROWS)
            commit = rows_here * OUTPUT_BLOCKS
            service_end += commit
            commit_cycles += commit

    require(correction_tokens == EXPECTED_EVENTS + 3 * EXPECTED_GROUPS,
            "correction token conservation failed")
    require(pwp_tokens == EXPECTED_PWP_TOKENS,
            "PWP token conservation failed")
    common_tail = commit_cycles + pipeline_flush_cycles
    baseline_cycles = BASELINE_SERVICE_TOKENS + common_tail
    require(service_end == pwp_tokens + correction_tokens + service_idle
            + common_tail, "M108 service cycle conservation failed")
    return {
        "windows": len(sequence),
        "descriptor_fill_cycles": descriptor_fill_cycles,
        "producer_bank_stall_cycles": producer_stall,
        "service_idle_cycles": service_idle,
        "pwp_service_tokens": pwp_tokens,
        "correction_service_tokens": correction_tokens,
        "accumulator_pipeline_flush_cycles": pipeline_flush_cycles,
        "accumulator_commit_cycles": commit_cycles,
        "candidate_cycles": service_end,
        "fair_fixed8_baseline_cycles": baseline_cycles,
        "same_clock_service_island_ratio": baseline_cycles / float(service_end),
        "maximum_descriptor_fill_cycles": max_fill,
        "maximum_descriptor_service_tokens": max_service,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M108 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    require(sha256(M105_SCRIPT) == EXPECTED_SHA256["m105_script"],
            "M105 auditor identity drift")
    require(sha256(M105_RESULT) == EXPECTED_SHA256["m105_result"],
            "M105 result identity drift")
    require(sha256(M107_RESULT) == EXPECTED_SHA256["m107_result"],
            "M107 result identity drift")

    m105 = load_m105_module()
    for label, path in {
        "m40_manifest": m105.M40_MANIFEST,
        "m72_result": m105.M72_RESULT,
        "m78_result": m105.M78_RESULT,
        "m41_result": m105.M41_RESULT,
    }.items():
        require(sha256(path) == m105.EXPECTED_SHA256[label],
                "frozen source identity drift: " + label)
    manifest = strict_json(m105.M40_MANIFEST)
    m72 = strict_json(m105.M72_RESULT)
    m41 = strict_json(m105.M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"] if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]),
    )
    require(len(heldout) == 20, "heldout record extent drift")
    popcount = np.fromiter(
        (int(value).bit_count() for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, weight_shas, _ = m105.build_width_catalog(m72, m41)

    record_counts = {}
    total_width_uses = Counter()
    total_events = 0
    total_groups = 0
    total_pwp_tokens = 0
    total_pwp_updates = 0
    total_positive = 0
    total_negative = 0
    coefficient_checks = 0
    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        counts = descriptor_counts(
            m105, masks, record["operator_index"], centers, widths, popcount)
        key = (record["sample_id"], record["operator_index"])
        record_counts[key] = counts
        total_width_uses.update(counts["width_uses"])
        total_events += int(counts["events"].sum())
        total_groups += int(counts["groups"].sum())
        total_pwp_tokens += int(counts["pwp_tokens"].sum())
        total_pwp_updates += int(counts["pwp_updates"].sum())
        total_positive += int(counts["positive_events"].sum())
        total_negative += int(counts["negative_events"].sum())
        coefficient_checks += counts["coefficient_checks"]
        print("[M108 RECORD] {}/20 sample={} op={} pwp={} events={} neg={}".format(
            record_index + 1, key[0], key[1],
            int(counts["pwp_tokens"].sum()), int(counts["events"].sum()),
            int(counts["negative_events"].sum())), flush=True)

    require(total_events == EXPECTED_EVENTS, "event conservation drift")
    require(total_groups == EXPECTED_GROUPS, "group conservation drift")
    require(dict(sorted(total_width_uses.items())) == EXPECTED_PWP_USES,
            "PWP width-use conservation drift")
    require(total_pwp_tokens == EXPECTED_PWP_TOKENS,
            "PWP service-token conservation drift")
    require(total_positive + total_negative == total_events,
            "direction conservation drift")

    sequence = []
    for sample in range(5, 10):
        for op in range(4):
            counts = record_counts[(sample, op)]
            for window in range(WINDOWS_PER_PHASE):
                for partition in range(PARTITIONS):
                    sequence.append({
                        "sample": sample,
                        "operator": op,
                        "window": window,
                        "partition": partition,
                        "events": int(counts["events"][partition, window]),
                        "groups": int(counts["groups"][partition, window]),
                        "pwp_tokens": int(counts["pwp_tokens"][partition, window]),
                        "pwp_updates": int(counts["pwp_updates"][partition, window]),
                    })
    require(len(sequence) == 406080, "window-major descriptor extent drift")
    schedule = simulate_window_major(sequence)
    m107 = strict_json(M107_RESULT)
    require(m107["work_conservation"]["correction_tokens"]
            == schedule["correction_service_tokens"],
            "M107 correction ledger mismatch")
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M108 analyzer changed during execution")

    payload = {
        "schema": "m108_w64_fused_pwp_accumulator_schedule_result_v1",
        "status": "PASS_WINDOW_MAJOR_PRECOMPACTED_SERVICE_ISLAND_ACCUMULATOR_SCHEDULE",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "m105_auditor_sha256": EXPECTED_SHA256["m105_script"],
            "m105_result_sha256": EXPECTED_SHA256["m105_result"],
            "m107_result_sha256": EXPECTED_SHA256["m107_result"],
            "m107_manifest_sha256": sha256(M107_MANIFEST),
            "weight_payload_sha256": weight_shas,
        },
        "work_conservation": {
            "events": total_events,
            "positive_events": total_positive,
            "negative_events": total_negative,
            "negative_event_fraction": total_negative / float(total_events),
            "active_groups": total_groups,
            "pwp_uses_by_width": dict(sorted(total_width_uses.items())),
            "pwp_updates": total_pwp_updates,
            "pwp_service_tokens": total_pwp_tokens,
            "source_coefficient_checks": coefficient_checks,
        },
        "schedule": schedule,
        "accumulator_contract": {
            "window_rows": WIN_ROWS,
            "output_block_banks": 8,
            "lanes_per_vector": 96,
            "signed_bits_per_lane": 24,
            "single_window_bytes": WIN_ROWS * OUTPUT_BLOCKS * 96 * 3,
            "storage_implementation": "macro required; not included in logic-only RTL",
            "port_contract": "one vector read plus one vector write per accepted update; one global update result per cycle maximum",
            "same_address_next_cycle_hazard": "structurally absent: PWP results are separated by at least three service tokens; correction rows are unique within a key and key changes insert three load tokens",
            "lazy_clear_or_epoch_tag": "required to avoid a dense initialization sweep",
            "commit": "one 96-lane vector per valid row/output block, serialized and charged equally to candidate and fixed8 baseline",
            "finite_width_basis": "M41 dense INT8 bound requires 21 signed bits; 24 signed bits selected",
            "full_lane_numeric_miter": False
        },
        "precompacted_input_contract": {
            "correction_events": "M106 receives lossless precompacted event descriptors; natural-raster scan/compaction is outside this service island",
            "pwp_requests": "PWP request stream is available independently while the selected M106 descriptor bank fills",
            "shared_service_lane": "PWP beats and M104 load/event tokens are serialized",
            "shared_weight_sram_arbitration": False
        },
        "admission": {
            "exact_heldout_pwp_event_group_direction_replay": True,
            "window_major_pwp_correction_shared_lane_schedule": True,
            "accumulator_flush_and_commit_charged": True,
            "source_coefficient_miter": True,
            "full_lane_numeric_miter": False,
            "precompaction_schedule": False,
            "shared_weight_sram_schedule": False,
            "macro_inclusive_ppa": False,
            "equal_area": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M108 ratio={:.12f} candidate={} idle={} neg_fraction={:.9f} output={}".format(
        schedule["same_clock_service_island_ratio"], schedule["candidate_cycles"],
        schedule["service_idle_cycles"],
        payload["work_conservation"]["negative_event_fraction"], args.output),
        flush=True)


if __name__ == "__main__":
    main()
