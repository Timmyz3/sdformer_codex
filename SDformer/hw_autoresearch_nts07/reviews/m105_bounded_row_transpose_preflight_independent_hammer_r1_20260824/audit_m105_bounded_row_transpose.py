#!/usr/bin/env python3
"""Independent M105 natural-raster bounded-window service-token preflight.

This reconstructs heldout masks directly from the M40 packed support planes,
uses the frozen M72 centers, independently rebuilds the M78 cap11 PWP width
catalog from exact INT8 weights, and groups correction/fallback events only
inside a bounded natural-raster row window.  It is not a scheduler or a cycle,
physical, system, accuracy, or headline model.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M40_DIR = HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822"
M40_MANIFEST = M40_DIR / "m40_bottleneck_packed_source_manifest.json"
M72_RESULT = HW / (
    "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/"
    "m72_phi_kmeans_k16q16_valid825_internal_screen.json")
M78_RESULT = HW / (
    "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/"
    "m78_precision_elastic_pwp.json")
M41_DIR = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41_RESULT = M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json"

EXPECTED_SHA256 = {
    "m40_manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m72_result": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m78_result": "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
}
EXPECTED_WEIGHT_SHA256 = (
    "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
)

TIMESTEPS = 10
CHANNELS = 768
HEIGHT = 15
WIDTH = 20
ROWS = TIMESTEPS * HEIGHT * WIDTH
FEATURES = CHANNELS * 3 * 3
PARTITION_BITS = 16
PARTITIONS = FEATURES // PARTITION_BITS
PATTERNS = 16
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
CAP = 11
WINDOWS = (1, 4, 16, 43, 64, 256, 294, 1024, 3000)
SCAN_MAX = 512
BASELINE_SERVICE_CYCLES = 1114383288
EXPECTED_BASELINE_EVENTS = 371461096
EXPECTED_CORRECTION_FALLBACK_EVENTS = 188148490
EXPECTED_PWP_SERVICE_CYCLES = 226222255
EXPECTED_PWP_USES = {8: 11164284, 9: 32360036, 10: 13936011, 11: 1509043}


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


def signed_width(minimum, maximum):
    for width in range(1, 33):
        if minimum >= -(1 << (width - 1)) and maximum <= (1 << (width - 1)) - 1:
            return max(8, width)
    raise ValueError("signed width exceeds 32")


def update_counter(counter, values):
    unique, counts = np.unique(values, return_counts=True)
    counter.update(dict((int(key), int(count)) for key, count in zip(unique, counts)))


def counter_distribution(counter):
    total = sum(counter.values())
    require(total > 0, "empty distribution")
    weighted = sum(value * count for value, count in counter.items())

    def nearest_rank(percent):
        target = (percent * total + 99) // 100
        cumulative = 0
        for value in sorted(counter):
            cumulative += counter[value]
            if cumulative >= target:
                return value
        raise AssertionError("nearest rank fell through")

    return {
        "count": total,
        "minimum": min(counter),
        "maximum": max(counter),
        "mean_exact": {"numerator": weighted, "denominator": total},
        "p50_nearest_rank": nearest_rank(50),
        "p95_nearest_rank": nearest_rank(95),
        "p99_nearest_rank": nearest_rank(99),
    }


def build_width_catalog(m72, m41):
    operator_names = [row["operator"] for row in m72["operators"]]
    layers = dict((row["operator"], row) for row in m41["layers"])
    catalog = np.zeros((4, PARTITIONS, PATTERNS, OUTPUT_BLOCKS), dtype=np.uint8)
    weight_shas = []
    width_histogram = Counter()
    for op, operator in enumerate(operator_names):
        payload = next(row for row in layers[operator]["payloads"]
                       if row["role"] == "weight")
        path = M41_DIR / payload["file"]
        observed = sha256(path)
        require(observed == EXPECTED_WEIGHT_SHA256[op] == payload["sha256"],
                "weight identity drift op{}".format(op))
        weight_shas.append(observed)
        weights = np.fromfile(str(path), dtype=np.int8)
        require(weights.size == FEATURES * CHANNELS, "weight extent drift")
        weights = weights.reshape(FEATURES, CHANNELS).astype(np.int32)
        for partition, partition_row in enumerate(m72["operators"][op]["partitions"]):
            require(partition_row["partition"] == partition,
                    "M72 partition order drift")
            source = weights[partition * PARTITION_BITS:(partition + 1) * PARTITION_BITS]
            for pattern, center_hex in enumerate(partition_row["centers_hex"]):
                center = int(center_hex, 16)
                indices = [bit for bit in range(PARTITION_BITS)
                           if center & (1 << bit)]
                pwp = source[indices].sum(axis=0, dtype=np.int32)
                for block in range(OUTPUT_BLOCKS):
                    vector = pwp[block * OUTPUT_LANES:(block + 1) * OUTPUT_LANES]
                    width = signed_width(int(vector.min()), int(vector.max()))
                    catalog[op, partition, pattern, block] = width
                    width_histogram[width] += 1
        print("[M105 WIDTH] operator={}/4".format(op + 1), flush=True)
    return catalog, weight_shas, width_histogram


def decode_natural_partition_masks(record, popcount):
    path = M40_DIR / record["packed_file"]
    require(path.is_file() and sha256(path) == record["packed_file_sha256"],
            "M40 heldout packed payload drift")
    raw = path.read_bytes()
    plane_bytes = record["positive_plane_bytes"]
    require(len(raw) == 3 * plane_bytes and not any(raw[plane_bytes:2 * plane_bytes]),
            "M40 packed plane extent or sign drift")
    total_bits = TIMESTEPS * CHANNELS * HEIGHT * WIDTH
    support = np.unpackbits(np.frombuffer(raw[:plane_bytes], dtype=np.uint8),
                            bitorder="little")[:total_bits]
    support = support.reshape(TIMESTEPS, 1, CHANNELS, HEIGHT, WIDTH)[:, 0]
    require(int(support.sum()) == record["nonzero_count"],
            "M40 direct support count drift")
    padded = np.pad(support, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")
    im2col = np.empty((ROWS, FEATURES), dtype=np.uint8)
    for kernel_y in range(3):
        for kernel_x in range(3):
            slot = kernel_y * 3 + kernel_x
            values = padded[:, :, kernel_y:kernel_y + HEIGHT,
                            kernel_x:kernel_x + WIDTH]
            values = values.transpose(0, 2, 3, 1).reshape(ROWS, CHANNELS)
            im2col[:, slot::9] = values
    powers = (np.uint16(1) << np.arange(PARTITION_BITS, dtype=np.uint16))
    masks = (im2col.reshape(ROWS, PARTITIONS, PARTITION_BITS) * powers).sum(
        axis=2, dtype=np.uint16).T.copy()
    require(int(popcount[masks].sum()) == sum(record["local_nonzero_count_by_timestep"]) * 9
            - boundary_replication_loss(record, support),
            "natural im2col support accounting drift")
    return masks


def boundary_replication_loss(record, support):
    # Each source would replicate to nine output positions without padding
    # boundaries.  Count the exact invalid destinations removed at image edges.
    loss = 0
    for input_y in range(HEIGHT):
        for input_x in range(WIDTH):
            valid = 0
            for kernel_y in range(3):
                output_y = input_y - kernel_y + 1
                if output_y < 0 or output_y >= HEIGHT:
                    continue
                for kernel_x in range(3):
                    output_x = input_x - kernel_x + 1
                    if 0 <= output_x < WIDTH:
                        valid += 1
            loss += int(support[:, :, input_y, input_x].sum()) * (9 - valid)
    return loss


def centers_array(m72):
    centers = np.zeros((4, PARTITIONS, PATTERNS), dtype=np.uint16)
    for op, operator in enumerate(m72["operators"]):
        for partition, row in enumerate(operator["partitions"]):
            require(row["partition"] == partition and len(row["centers_hex"]) == PATTERNS,
                    "M72 center extent/order drift")
            values = [int(value, 16) for value in row["centers_hex"]]
            require(len(set(values)) == PATTERNS, "M72 center uniqueness drift")
            centers[op, partition] = values
    return centers


def build_record_events(masks, op, centers, widths, popcount):
    event_masks = np.zeros((PARTITIONS, ROWS, OUTPUT_BLOCKS), dtype=np.uint16)
    totals = Counter()
    pwp_width_uses = Counter()
    for partition in range(PARTITIONS):
        values = masks[partition]
        center_values = centers[op, partition]
        order = np.argsort(center_values, kind="stable")
        ordered_centers = center_values[order]
        distances = popcount[np.bitwise_xor(values[:, None], ordered_centers[None, :])]
        ordered_choice = distances.argmin(axis=1)
        best_index = order[ordered_choice]
        best_distance = distances[np.arange(ROWS), ordered_choice]
        best_center = center_values[best_index]
        population = popcount[values]
        beneficial = (1 + best_distance) < population
        delta = np.bitwise_xor(values, best_center)
        selected_widths = widths[op, partition, best_index]
        eligible = beneficial[:, None] & (selected_widths <= CAP)
        source_masks = np.where(eligible, delta[:, None], values[:, None]).astype(np.uint16)
        event_masks[partition] = source_masks

        event_count = int(popcount[source_masks].sum())
        correction_count = int((popcount[delta][:, None] * eligible).sum())
        nonbeneficial_count = int((population * (~beneficial)).sum()) * OUTPUT_BLOCKS
        cap_escape_count = int((population[:, None] *
                                (beneficial[:, None] & ~eligible)).sum())
        require(event_count == correction_count + nonbeneficial_count + cap_escape_count,
                "event class conservation drift")
        totals["correction_fallback_events"] += event_count
        totals["beneficial_correction_events"] += correction_count
        totals["nonbeneficial_bit_sparse_fallback_events"] += nonbeneficial_count
        totals["cap_escape_fallback_events"] += cap_escape_count
        totals["cap_escape_rows"] += int((beneficial[:, None] & ~eligible).sum())
        totals["assignment_rows"] += int((beneficial & eligible.any(axis=1)).sum())
        totals["baseline_events"] += int(population.sum()) * OUTPUT_BLOCKS
        for width in (8, 9, 10, 11):
            pwp_width_uses[width] += int((eligible & (selected_widths == width)).sum())
    return event_masks, totals, pwp_width_uses


def group_total(event_masks, window, popcount):
    starts = np.arange(0, ROWS, window, dtype=np.intp)
    union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
    return int(popcount[union].sum())


def update_selected(stats, event_masks, window, popcount):
    starts = np.arange(0, ROWS, window, dtype=np.intp)
    union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
    active = popcount[union].sum(axis=2).reshape(-1)
    update_counter(stats[window]["active_groups_per_window"], active)

    row_events = popcount[event_masks].sum(axis=2, dtype=np.uint16)
    prefix = np.concatenate((np.zeros((PARTITIONS, 1), dtype=np.uint32),
                             np.cumsum(row_events, axis=1, dtype=np.uint32)), axis=1)
    ends = np.minimum(starts + window, ROWS)
    window_events = (prefix[:, ends] - prefix[:, starts]).reshape(-1)
    update_counter(stats[window]["events_per_window"], window_events)

    descriptor_event_sum = 0
    descriptor_count = 0
    for bit in range(PARTITION_BITS):
        bit_rows = ((event_masks >> np.uint16(bit)) & np.uint16(1)).astype(np.uint16)
        bit_prefix = np.concatenate((
            np.zeros((PARTITIONS, 1, OUTPUT_BLOCKS), dtype=np.uint16),
            np.cumsum(bit_rows, axis=1, dtype=np.uint16)), axis=1)
        counts = bit_prefix[:, ends, :] - bit_prefix[:, starts, :]
        nonzero = counts[counts != 0]
        if nonzero.size:
            update_counter(stats[window]["events_per_active_descriptor"], nonzero)
            descriptor_event_sum += int(nonzero.sum())
            descriptor_count += int(nonzero.size)
    require(descriptor_event_sum == int(window_events.sum()),
            "descriptor event sum drift window{}".format(window))
    require(descriptor_count == int(active.sum()),
            "descriptor count drift window{}".format(window))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    paths = {
        "m40_manifest": M40_MANIFEST,
        "m72_result": M72_RESULT,
        "m78_result": M78_RESULT,
        "m41_result": M41_RESULT,
    }
    for key, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED_SHA256[key],
                "input identity drift: " + key)
    m40 = strict_json(M40_MANIFEST)
    m72 = strict_json(M72_RESULT)
    m78 = strict_json(M78_RESULT)
    m41 = strict_json(M41_RESULT)
    heldout_records = sorted(
        (row for row in m40["records"] if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout_records) == 20, "heldout record extent drift")
    require([(row["sample_id"], row["operator_index"]) for row in heldout_records]
            == [(sample, op) for sample in range(5, 10) for op in range(4)],
            "heldout natural record order drift")
    require(m72["split"]["heldout_samples_within_valid825"] == [5, 6, 7, 8, 9],
            "M72 heldout split drift")
    cap11 = next(row for row in m78["configurations"]
                 if row["signed_width_cap"] == CAP)

    popcount = np.fromiter((int(value).bit_count() for value in range(1 << 16)),
                           dtype=np.uint8, count=1 << 16)
    centers = centers_array(m72)
    widths, weight_shas, width_histogram = build_width_catalog(m72, m41)
    require(dict(sorted(width_histogram.items())) ==
            dict((int(key), value) for key, value in m78["pwp_precision"]["width_histogram"].items()),
            "independent cap width histogram drift")

    totals = Counter()
    pwp_uses = Counter()
    group_totals_by_window = Counter()
    selected = dict((window, {
        "active_groups_per_window": Counter(),
        "events_per_window": Counter(),
        "events_per_active_descriptor": Counter(),
    }) for window in WINDOWS)
    raw_payload_shas = []

    for record_index, record in enumerate(heldout_records):
        raw_payload_shas.append(record["packed_file_sha256"])
        masks = decode_natural_partition_masks(record, popcount)
        event_masks, record_totals, record_pwp = build_record_events(
            masks, record["operator_index"], centers, widths, popcount)
        totals.update(record_totals)
        pwp_uses.update(record_pwp)
        for window in range(1, SCAN_MAX + 1):
            group_totals_by_window[window] += group_total(event_masks, window, popcount)
        for window in WINDOWS:
            update_selected(selected, event_masks, window, popcount)
        print("[M105 RECORD] {}/20 sample={} op={} events={}".format(
            record_index + 1, record["sample_id"], record["operator_index"],
            record_totals["correction_fallback_events"]), flush=True)

    require(totals["baseline_events"] == EXPECTED_BASELINE_EVENTS,
            "baseline event conservation drift")
    require(totals["correction_fallback_events"] == EXPECTED_CORRECTION_FALLBACK_EVENTS,
            "candidate correction/fallback event conservation drift")
    require(dict(sorted(pwp_uses.items())) == EXPECTED_PWP_USES,
            "PWP uses-by-width drift")
    pwp_service_cycles = (pwp_uses[8] * 3 + pwp_uses[9] * 4 +
                          pwp_uses[10] * 4 + pwp_uses[11] * 5)
    require(pwp_service_cycles == EXPECTED_PWP_SERVICE_CYCLES,
            "PWP service cycle drift")
    require(totals["cap_escape_rows"] == cap11["heldout"]["block_local_escape_rows"],
            "cap11 escape row drift")
    require(totals["assignment_rows"] == cap11["heldout"]["assignment_rows"],
            "cap11 assignment row drift")

    target_group_limits = {
        "2.0": math.floor((BASELINE_SERVICE_CYCLES / 2.0 -
                           EXPECTED_PWP_SERVICE_CYCLES -
                           EXPECTED_CORRECTION_FALLBACK_EVENTS) / 3.0),
        "2.5": math.floor((BASELINE_SERVICE_CYCLES / 2.5 -
                           EXPECTED_PWP_SERVICE_CYCLES -
                           EXPECTED_CORRECTION_FALLBACK_EVENTS) / 3.0),
    }
    minimum_windows = {}
    for label, limit in target_group_limits.items():
        passing = [window for window in range(1, SCAN_MAX + 1)
                   if group_totals_by_window[window] <= limit]
        require(passing, "SCAN_MAX insufficient for {}x target".format(label))
        minimum = min(passing)
        groups = group_totals_by_window[minimum]
        correction_tokens = EXPECTED_CORRECTION_FALLBACK_EVENTS + 3 * groups
        combined = EXPECTED_PWP_SERVICE_CYCLES + correction_tokens
        minimum_windows[label] = {
            "minimum_window_rows_within_scanned_1_to_{}".format(SCAN_MAX): minimum,
            "active_groups_total": groups,
            "group_limit_for_target": limit,
            "correction_fallback_token_envelope": correction_tokens,
            "candidate_plus_existing_pwp_envelope": combined,
            "baseline_ratio": BASELINE_SERVICE_CYCLES / float(combined),
        }

    window_rows = []
    for window in WINDOWS:
        active_distribution = counter_distribution(
            selected[window]["active_groups_per_window"])
        event_distribution = counter_distribution(
            selected[window]["events_per_window"])
        descriptor_distribution = counter_distribution(
            selected[window]["events_per_active_descriptor"])
        groups = active_distribution["mean_exact"]["numerator"]
        if window <= SCAN_MAX:
            require(group_totals_by_window[window] == groups,
                    "selected active group total drift")
        correction_tokens = EXPECTED_CORRECTION_FALLBACK_EVENTS + 3 * groups
        combined = EXPECTED_PWP_SERVICE_CYCLES + correction_tokens
        require(event_distribution["mean_exact"]["numerator"] ==
                EXPECTED_CORRECTION_FALLBACK_EVENTS,
                "selected event conservation drift")
        require(descriptor_distribution["mean_exact"]["numerator"] ==
                EXPECTED_CORRECTION_FALLBACK_EVENTS,
                "selected descriptor event conservation drift")
        window_rows.append({
            "window_rows": window,
            "windows_total": active_distribution["count"],
            "active_groups_total": groups,
            "active_groups_per_window": active_distribution,
            "events_per_window": event_distribution,
            "events_per_active_descriptor": descriptor_distribution,
            "correction_fallback_events": EXPECTED_CORRECTION_FALLBACK_EVENTS,
            "three_cycle_group_load_tokens": 3 * groups,
            "one_token_per_event": EXPECTED_CORRECTION_FALLBACK_EVENTS,
            "correction_fallback_token_envelope": correction_tokens,
            "existing_cap11_pwp_service_cycles": EXPECTED_PWP_SERVICE_CYCLES,
            "candidate_plus_existing_pwp_envelope": combined,
            "baseline_service_cycles": BASELINE_SERVICE_CYCLES,
            "baseline_ratio": BASELINE_SERVICE_CYCLES / float(combined),
        })

    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "audit script changed during run")
    payload = {
        "schema": "m105_bounded_natural_raster_row_transpose_preflight_v1",
        "status": "PASS_M105_CONDITIONAL_SERVICE_TOKEN_ENVELOPE_ONLY",
        "identity": {
            "auditor_start_end_sha256": script_start_sha,
            "inputs_sha256": EXPECTED_SHA256,
            "weight_payload_sha256": weight_shas,
            "heldout_raw_mask_payload_sha256": raw_payload_shas,
        },
        "scope": {
            "source": "M40 exact heldout raw support masks plus M72 centers plus independently rebuilt M78 cap11 widths",
            "samples": [5, 6, 7, 8, 9],
            "operators": [row["operator"] for row in m72["operators"]],
            "phases": 5 * 4 * PARTITIONS,
            "partitions_per_sample_operator": PARTITIONS,
            "rows_per_phase": ROWS,
            "natural_row_order": "timestep_major_then_output_y_then_output_x",
            "window_resets_at": "each sample_operator_partition phase",
            "group_key": ["partition_local_source_0_to_15", "output_block_0_to_7"],
            "maximum_group_keys_per_window": PARTITION_BITS * OUTPUT_BLOCKS,
            "reordering": "only correction/fallback events inside each bounded raster window",
            "token_model": "three tokens to load one 96-byte INT8 weight vector plus one token per grouped event",
            "pwp_model": "existing exact cap11 shared-32B service ledger added unchanged",
        },
        "work_conservation": {
            "baseline_events": totals["baseline_events"],
            "correction_fallback_events": totals["correction_fallback_events"],
            "beneficial_correction_events": totals["beneficial_correction_events"],
            "nonbeneficial_bit_sparse_fallback_events":
                totals["nonbeneficial_bit_sparse_fallback_events"],
            "cap_escape_fallback_events": totals["cap_escape_fallback_events"],
            "cap_escape_rows": totals["cap_escape_rows"],
            "assignment_rows": totals["assignment_rows"],
            "pwp_uses_by_width": dict(sorted(pwp_uses.items())),
            "pwp_service_cycles": pwp_service_cycles,
            "events_conserved_for_every_reported_window": True,
        },
        "window_results": window_rows,
        "minimum_windows": minimum_windows,
        "group_totals_scan_1_to_{}".format(SCAN_MAX):
            dict((str(window), group_totals_by_window[window])
                 for window in range(1, SCAN_MAX + 1)),
        "hardware_risk": {
            "descriptor_key_space_per_window": 128,
            "event_direction_token_required": True,
            "destination_identity_or_offset_required_per_event": True,
            "finite_descriptor_and_event_buffer_not_implemented": True,
            "accumulator_bank_conflicts_port_cut": True,
            "finite_accumulator_width_port_cut": True,
            "dependency_and_commit_order_port_cut": True,
            "memory_address_and_shared_port_schedule_port_cut": True,
            "window_fill_drain_and_descriptor_issue_overhead_unmodeled": True,
        },
        "admission": {
            "natural_order_bounded_window_conditional_service_token_envelope": True,
            "exact_event_conservation": True,
            "scheduled_cycle_ratio": False,
            "actual_record_rtl_replay": False,
            "physical_speedup": False,
            "equal_area": False,
            "full_network_or_system_speedup": False,
            "accuracy": False,
            "date_or_best_paper_headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M105 windows={} min2x={} min2p5x={} result={}".format(
        list(WINDOWS),
        minimum_windows["2.0"]["minimum_window_rows_within_scanned_1_to_512"],
        minimum_windows["2.5"]["minimum_window_rows_within_scanned_1_to_512"],
        args.output), flush=True)


if __name__ == "__main__":
    main()
