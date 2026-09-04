#!/usr/bin/env python3
"""Independent raw and dual-timeline audit of the M108 r2 schedule.

This auditor does not import or execute the M105/M108 producer analyzers.  It
reconstructs the heldout W64 descriptor work directly from frozen M40 support
planes, M72 centers, and M41 INT8 weights.  It then reproduces the published
recurrence and compares it with an event-driven recurrence that keeps the M106
controller/drain timeline separate from the shared PWP/correction lane.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
OUTPUT = REVIEW / "m108_r2_rtl_edge_fused_schedule_independent_audit.json"

M108_CONTRACT = HW / "contracts/m108_r2_rtl_edge_fused_schedule_contract_r1_20260824.json"
M108_ANALYZER = HW / "system_simulator/scripts/analyze_m108_r2_rtl_edge_fused_schedule.py"
M108_DIR = HW / "results/m108_r2_rtl_edge_fused_schedule_r1_20260824"
M108_RESULT = M108_DIR / "m108_r2_rtl_edge_fused_schedule.json"
M108_RUN = M108_DIR / "RUN_COMPLETE.txt"
M108_MANIFEST = M108_DIR / "manifest.sha256"
M106_CONTRACT = HW / "contracts/m106_r2_standard_streaming_grace_vcs_contract_r1_20260824.json"
M106_RTL = HW / "rtl_m106/m106_bounded_bitmap_transpose_scheduler.sv"
M106_RUN = HW / "dc_handoff/runs/m106_r2_standard_streaming_grace_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
M107_AUDIT = HW / "reviews/m107_w64_pingpong_schedule_replay_independent_hammer_r1_20260824/m107_w64_pingpong_schedule_replay_independent_audit.json"
M40_DIR = HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822"
M40_MANIFEST = M40_DIR / "m40_bottleneck_packed_source_manifest.json"
M72_RESULT = HW / "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/m72_phi_kmeans_k16q16_valid825_internal_screen.json"
M78_RESULT = HW / "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/m78_precision_elastic_pwp.json"
M41_DIR = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41_RESULT = M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SHA = {
    M108_CONTRACT: "f2ce9d2ed5b8d2f6f019035f8f25b7ac7edd339a874d15eee748fbb165f0c0ac",
    M108_ANALYZER: "8915ae225f658ac8b4e2d4ca178f870e95a45a85ba647791ead0495b2a29e7f3",
    M108_RESULT: "2813ea18de27ac59d45e48897f2c217a3a67828c4b17fcbf93795cab9950582a",
    M108_RUN: "592249419fb96b6060693cd164a1a1bdccfa7bee93c0f4711a9f144dfd423ced",
    M108_MANIFEST: "2ad0aad7fad246259212b0856a2c79c1b1daff4bd175c8381155a966f33de2fa",
    M106_CONTRACT: "984ca6558ebbf3a58135e60b4aa889b7726532b8a4fc872acf7156f50d7d8196",
    M106_RTL: "a6937765aea87269c3d38123b656c72b7ee400e36b0d634f21ab9c7dbdefc0b7",
    M106_RUN: "45db2f7ae514f7afbafff93dddbd272076181e2a3f1aa6bbcb25f24f71710999",
    M107_AUDIT: "42b36d56b484cae3958d5f720f1bd64e9dc5d8b5c34bff9b088fac0e55f7d9dd",
    M40_MANIFEST: "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    M72_RESULT: "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    M78_RESULT: "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    M41_RESULT: "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
WEIGHT_SHA = (
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
WIN_ROWS = 64
WINDOWS = (ROWS + WIN_ROWS - 1) // WIN_ROWS
CAP = 11
EXPECTED_EVENTS = 188148490
EXPECTED_POSITIVE = 170591133
EXPECTED_NEGATIVE = 17557357
EXPECTED_GROUPS = 35140002
EXPECTED_PWP_UPDATES = 58969374
EXPECTED_PWP_TOKENS = 226222255
EXPECTED_PWP_USES = {8: 11164284, 9: 32360036, 10: 13936011, 11: 1509043}
EXPECTED_BASELINE_EVENTS = 371461096
BASELINE_TOKENS = EXPECTED_BASELINE_EVENTS * 3
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
        target = Path(name.strip())
        if not target.is_absolute():
            target = Path(base) / target
        require(target.is_file(), "manifest target missing: " + str(target))
        require(sha256(target) == expected, "manifest mismatch: " + str(target))
        checked += 1
    return checked


def signed_width(minimum, maximum):
    for bits in range(1, 33):
        if minimum >= -(1 << (bits - 1)) and maximum <= (1 << (bits - 1)) - 1:
            return max(8, bits)
    raise ValueError("signed width overflow")


def build_widths(m72, m41):
    layers = {row["operator"]: row for row in m41["layers"]}
    widths = np.zeros((4, PARTITIONS, PATTERNS, OUTPUT_BLOCKS), dtype=np.uint8)
    histogram = Counter()
    payload_shas = []
    for op, operator in enumerate(m72["operators"]):
        payload = next(row for row in layers[operator["operator"]]["payloads"]
                       if row["role"] == "weight")
        path = M41_DIR / payload["file"]
        observed = sha256(path)
        require(observed == payload["sha256"] == WEIGHT_SHA[op],
                "weight identity drift op{}".format(op))
        payload_shas.append(observed)
        weights = np.fromfile(str(path), dtype=np.int8)
        require(weights.size == FEATURES * CHANNELS, "weight extent drift")
        weights = weights.reshape(FEATURES, CHANNELS).astype(np.int32)
        for partition, partition_row in enumerate(operator["partitions"]):
            require(partition_row["partition"] == partition,
                    "center partition order drift")
            source = weights[partition * PARTITION_BITS:
                             (partition + 1) * PARTITION_BITS]
            for pattern, center_hex in enumerate(partition_row["centers_hex"]):
                center = int(center_hex, 16)
                active = [bit for bit in range(PARTITION_BITS)
                          if center & (1 << bit)]
                if active:
                    pwp = source[active].sum(axis=0, dtype=np.int32)
                else:
                    pwp = np.zeros(CHANNELS, dtype=np.int32)
                for block in range(OUTPUT_BLOCKS):
                    vector = pwp[block * OUTPUT_LANES:(block + 1) * OUTPUT_LANES]
                    bits = signed_width(int(vector.min()), int(vector.max()))
                    widths[op, partition, pattern, block] = bits
                    histogram[bits] += 1
    return widths, histogram, payload_shas


def centers_array(m72):
    centers = np.zeros((4, PARTITIONS, PATTERNS), dtype=np.uint16)
    for op, operator in enumerate(m72["operators"]):
        for partition, row in enumerate(operator["partitions"]):
            values = [int(value, 16) for value in row["centers_hex"]]
            require(row["partition"] == partition and len(values) == PATTERNS
                    and len(set(values)) == PATTERNS,
                    "center extent/order/uniqueness drift")
            centers[op, partition] = values
    return centers


def decode_masks(record):
    path = M40_DIR / record["packed_file"]
    require(path.is_file() and sha256(path) == record["packed_file_sha256"],
            "packed record identity drift")
    raw = path.read_bytes()
    plane_bytes = record["positive_plane_bytes"]
    require(len(raw) == 3 * plane_bytes, "packed record extent drift")
    require(not any(raw[plane_bytes:2 * plane_bytes]), "negative support plane not zero")
    total_bits = TIMESTEPS * CHANNELS * HEIGHT * WIDTH
    support = np.unpackbits(np.frombuffer(raw[:plane_bytes], dtype=np.uint8),
                            bitorder="little")[:total_bits]
    support = support.reshape(TIMESTEPS, CHANNELS, HEIGHT, WIDTH)
    require(int(support.sum()) == record["nonzero_count"], "support count drift")
    padded = np.pad(support, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")
    im2col = np.empty((ROWS, FEATURES), dtype=np.uint8)
    for ky in range(3):
        for kx in range(3):
            slot = ky * 3 + kx
            view = padded[:, :, ky:ky + HEIGHT, kx:kx + WIDTH]
            im2col[:, slot::9] = view.transpose(0, 2, 3, 1).reshape(ROWS, CHANNELS)
    powers = np.uint16(1) << np.arange(PARTITION_BITS, dtype=np.uint16)
    return (im2col.reshape(ROWS, PARTITIONS, PARTITION_BITS) * powers).sum(
        axis=2, dtype=np.uint16).T.copy()


def prefix_windows(per_row):
    starts = np.arange(0, ROWS, WIN_ROWS, dtype=np.intp)
    ends = np.minimum(starts + WIN_ROWS, ROWS)
    prefix = np.concatenate((np.zeros((per_row.shape[0], 1), dtype=np.uint32),
                             np.cumsum(per_row, axis=1, dtype=np.uint32)), axis=1)
    return (prefix[:, ends] - prefix[:, starts]).astype(np.int64)


def reconstruct_record(masks, op, centers, widths, popcount):
    events = np.zeros((PARTITIONS, WINDOWS), dtype=np.int64)
    groups = np.zeros_like(events)
    pwp_tokens = np.zeros_like(events)
    pwp_updates = np.zeros_like(events)
    positive = np.zeros_like(events)
    negative = np.zeros_like(events)
    pwp_uses = Counter()
    baseline_events = 0
    coefficient_checks = 0
    starts = np.arange(0, ROWS, WIN_ROWS, dtype=np.intp)

    for partition in range(PARTITIONS):
        values = masks[partition]
        center_values = centers[op, partition]
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
        delta = np.bitwise_xor(values, best_center)
        selected_width = widths[op, partition, choice]
        eligible = beneficial[:, None] & (selected_width <= CAP)

        event_masks = np.where(eligible, delta[:, None], values[:, None]).astype(np.uint16)
        negative_masks = np.where(
            eligible,
            np.bitwise_and(best_center[:, None],
                           np.bitwise_xor(values[:, None], np.uint16(0xffff))),
            np.uint16(0)).astype(np.uint16)
        positive_masks = np.bitwise_xor(event_masks, negative_masks)
        center_matrix = np.broadcast_to(best_center[:, None], eligible.shape)
        require(not np.any(np.where(eligible,
                                     np.bitwise_and(positive_masks, center_matrix), 0)),
                "positive correction overlaps center")
        require(not np.any(np.where(
            eligible,
            np.bitwise_and(negative_masks,
                           np.bitwise_xor(center_matrix, np.uint16(0xffff))), 0)),
                "negative correction outside center")
        reconstructed = np.where(
            eligible,
            np.bitwise_and(np.bitwise_or(center_matrix, positive_masks),
                           np.bitwise_xor(negative_masks, np.uint16(0xffff))),
            positive_masks).astype(np.uint16)
        require(np.array_equal(reconstructed,
                               np.broadcast_to(values[:, None], eligible.shape)),
                "bit coefficient miter mismatch")
        coefficient_checks += int(eligible.size) * PARTITION_BITS

        row_events = popcount[event_masks].sum(axis=1, dtype=np.uint16)
        row_negative = popcount[negative_masks].sum(axis=1, dtype=np.uint16)
        row_positive = popcount[positive_masks].sum(axis=1, dtype=np.uint16)
        require(np.array_equal(row_events, row_negative + row_positive),
                "event direction conservation mismatch")
        events[partition] = prefix_windows(row_events[None, :])[0]
        negative[partition] = prefix_windows(row_negative[None, :])[0]
        positive[partition] = prefix_windows(row_positive[None, :])[0]
        union = np.bitwise_or.reduceat(event_masks, starts, axis=0)
        groups[partition] = popcount[union].sum(axis=1, dtype=np.uint16)

        row_updates = eligible.sum(axis=1, dtype=np.uint8)
        row_pwp = np.where(eligible, PWP_BEATS[selected_width], 0).sum(
            axis=1, dtype=np.uint16)
        pwp_updates[partition] = prefix_windows(row_updates[None, :])[0]
        pwp_tokens[partition] = prefix_windows(row_pwp[None, :])[0]
        for bits in (8, 9, 10, 11):
            pwp_uses[bits] += int(np.count_nonzero(
                eligible & (selected_width == bits)))

    require(np.array_equal(events, positive + negative),
            "record direction window mismatch")
    require(np.all(events >= groups), "event/group relation mismatch")
    require(np.all(events[groups == 0] == 0), "events in empty descriptor")
    return {
        "events": events,
        "groups": groups,
        "pwp_tokens": pwp_tokens,
        "pwp_updates": pwp_updates,
        "positive": positive,
        "negative": negative,
        "pwp_uses": pwp_uses,
        "baseline_events": baseline_events,
        "coefficient_checks": coefficient_checks,
    }


def published_recurrence(sequence):
    bank_free = [0, 0]
    producer_end = 0
    service_end = 0
    totals = Counter()
    digest = hashlib.sha256()
    for index, row in enumerate(sequence):
        bank = index & 1
        if index == 0 or producer_end > bank_free[bank]:
            fill_start = producer_end
        else:
            fill_start = bank_free[bank] + 1
            totals["bank_reacquire_boundaries"] += 1
        totals["producer_stall"] += fill_start - producer_end
        fill_cycles = row["events"] + 1
        fill_end = fill_start + fill_cycles
        producer_end = fill_end
        totals["fill_cycles"] += fill_cycles
        dispatch_ready = fill_end + 1
        totals["dispatch_edges"] += 1

        pwp_start = max(service_end, fill_start)
        totals["service_idle"] += pwp_start - service_end
        pwp_end = pwp_start + row["pwp_tokens"]
        correction = row["events"] + 3 * row["groups"]
        if correction:
            correction_start = max(pwp_end, dispatch_ready)
            totals["exposed_post_pwp"] += correction_start - pwp_end
            totals["service_idle"] += correction_start - pwp_end
            correction_end = correction_start + correction
            bank_free[bank] = correction_end
            service_end = correction_end
        else:
            bank_free[bank] = dispatch_ready
            service_end = pwp_end
        if row["partition"] == PARTITIONS - 1:
            ready = max(service_end, bank_free[bank])
            totals["service_idle"] += ready - service_end
            service_end = ready + 1
            totals["flush"] += 1
            commit = min(WIN_ROWS, ROWS - row["window"] * WIN_ROWS) * OUTPUT_BLOCKS
            totals["commit"] += commit
            service_end += commit
        digest.update(struct.pack("<BBHHIII", row["sample"], row["operator"],
                                  row["window"], row["partition"], row["events"],
                                  row["groups"], row["pwp_tokens"]))
    return summarize_schedule(totals, service_end, producer_end,
                              digest.hexdigest(), "published_single_service_end")


def independent_dual_timeline(sequence):
    """M106 controller/drain and shared lane remain distinct state machines.

    A descriptor can dispatch only one controller edge after both its close and
    the prior M106 drain/release.  PWP can hide that edge on the shared lane.
    Empty descriptors release on their actual in-order dispatch edge, not merely
    on fill_end+1.  Commit occupies the shared accumulator lane but need not
    block M106 dispatch of an already filled descriptor.
    """
    bank_free = [0, 0]
    producer_end = 0
    controller_free = 0
    lane_end = 0
    totals = Counter()
    digest = hashlib.sha256()
    controller_serialization_delays = 0
    empty_release_delays = 0
    dispatch_hidden_by_pwp_or_lane = 0
    zero_pwp = 0

    for index, row in enumerate(sequence):
        bank = index & 1
        if index == 0 or producer_end > bank_free[bank]:
            fill_start = producer_end
        else:
            fill_start = bank_free[bank] + 1
            totals["bank_reacquire_boundaries"] += 1
        totals["producer_stall"] += fill_start - producer_end
        fill_cycles = row["events"] + 1
        fill_end = fill_start + fill_cycles
        producer_end = fill_end
        totals["fill_cycles"] += fill_cycles

        published_dispatch_ready = fill_end + 1
        dispatch_edge = max(fill_end, controller_free) + 1
        if dispatch_edge > published_dispatch_ready:
            controller_serialization_delays += dispatch_edge - published_dispatch_ready
        totals["dispatch_edges"] += 1

        pwp_start = max(lane_end, fill_start)
        totals["service_idle"] += pwp_start - lane_end
        pwp_end = pwp_start + row["pwp_tokens"]
        if row["pwp_tokens"] == 0:
            zero_pwp += 1
        correction = row["events"] + 3 * row["groups"]
        if correction:
            correction_start = max(pwp_end, dispatch_edge)
            if dispatch_edge <= pwp_end:
                dispatch_hidden_by_pwp_or_lane += 1
            totals["exposed_post_pwp"] += correction_start - pwp_end
            totals["service_idle"] += correction_start - pwp_end
            correction_end = correction_start + correction
            controller_free = correction_end
            lane_end = correction_end
            bank_free[bank] = correction_end
        else:
            if dispatch_edge <= pwp_end:
                dispatch_hidden_by_pwp_or_lane += 1
            empty_release_delays += dispatch_edge - published_dispatch_ready
            controller_free = dispatch_edge
            bank_free[bank] = dispatch_edge
            lane_end = pwp_end

        if row["partition"] == PARTITIONS - 1:
            ready = max(lane_end, controller_free)
            totals["service_idle"] += ready - lane_end
            lane_end = ready + 1
            totals["flush"] += 1
            commit = min(WIN_ROWS, ROWS - row["window"] * WIN_ROWS) * OUTPUT_BLOCKS
            totals["commit"] += commit
            lane_end += commit
        digest.update(struct.pack("<BBHHIII", row["sample"], row["operator"],
                                  row["window"], row["partition"], row["events"],
                                  row["groups"], row["pwp_tokens"]))

    result = summarize_schedule(totals, lane_end, producer_end,
                                digest.hexdigest(), "independent_dual_timeline")
    result.update({
        "controller_final_free_cycle": controller_free,
        "controller_serialization_delay_sum_vs_fill_only_dispatch":
            controller_serialization_delays,
        "empty_release_delay_sum_vs_fill_only_dispatch": empty_release_delays,
        "dispatch_hidden_by_pwp_or_prior_lane_descriptors":
            dispatch_hidden_by_pwp_or_lane,
        "zero_pwp_descriptors": zero_pwp,
    })
    return result


def summarize_schedule(totals, candidate, producer_end, digest, model):
    correction = EXPECTED_EVENTS + 3 * EXPECTED_GROUPS
    common_tail = totals["flush"] + totals["commit"]
    require(candidate == EXPECTED_PWP_TOKENS + correction
            + totals["service_idle"] + common_tail,
            "candidate lane conservation failed for " + model)
    baseline = BASELINE_TOKENS + common_tail
    return {
        "model": model,
        "descriptors": 20 * PARTITIONS * WINDOWS,
        "ordered_descriptor_sha256": digest,
        "descriptor_fill_cycles": totals["fill_cycles"],
        "producer_bank_stall_cycles": totals["producer_stall"],
        "producer_final_cycle": producer_end,
        "controller_dispatch_edges": totals["dispatch_edges"],
        "bank_reacquire_boundaries": totals["bank_reacquire_boundaries"],
        "exposed_post_pwp_fill_or_dispatch_wait_cycles": totals["exposed_post_pwp"],
        "service_idle_cycles": totals["service_idle"],
        "pwp_service_tokens": EXPECTED_PWP_TOKENS,
        "correction_service_tokens": correction,
        "accumulator_pipeline_flush_cycles": totals["flush"],
        "accumulator_commit_cycles": totals["commit"],
        "candidate_cycles": candidate,
        "fair_fixed8_baseline_cycles": baseline,
        "same_clock_service_island_ratio": baseline / float(candidate),
        "headroom_to_two_x_cycles": baseline // 2 - candidate,
    }


def main():
    start_sha = sha256(Path(__file__).resolve())
    observed = {}
    for path, expected in EXPECTED_SHA.items():
        actual = sha256(path)
        require(actual == expected, "identity mismatch {} {}".format(path, actual))
        observed[str(path.relative_to(HW))] = actual
    manifest_entries = verify_manifest(M108_MANIFEST, HW)
    require(manifest_entries == 8, "M108 manifest extent drift")
    require(M108_RUN.read_text(encoding="utf-8").splitlines()[0]
            == "status=PASS_M108_R2_RTL_EDGE_AWARE_PRECOMPACTED_MODULE_CYCLE_SIM",
            "M108 RUN_COMPLETE admission drift")

    contract = strict_json(M108_CONTRACT)
    published = strict_json(M108_RESULT)
    m40 = strict_json(M40_MANIFEST)
    m72 = strict_json(M72_RESULT)
    m78 = strict_json(M78_RESULT)
    m41 = strict_json(M41_RESULT)
    m107 = strict_json(M107_AUDIT)
    m106 = strict_json(M106_CONTRACT)

    popcount = np.fromiter((int(value).bit_count() for value in range(1 << 16)),
                           dtype=np.uint8, count=1 << 16)
    centers = centers_array(m72)
    widths, width_histogram, weight_shas = build_widths(m72, m41)
    require(dict(sorted(width_histogram.items()))
            == {int(k): v for k, v in m78["pwp_precision"]["width_histogram"].items()},
            "weight-derived width histogram drift")

    heldout = sorted((row for row in m40["records"]
                      if row["sample_id"] in range(5, 10)),
                     key=lambda row: (row["sample_id"], row["operator_index"]))
    require([(row["sample_id"], row["operator_index"]) for row in heldout]
            == [(sample, op) for sample in range(5, 10) for op in range(4)],
            "heldout record order drift")

    record_counts = {}
    totals = Counter()
    pwp_uses = Counter()
    for index, record in enumerate(heldout, 1):
        masks = decode_masks(record)
        counts = reconstruct_record(masks, record["operator_index"], centers,
                                    widths, popcount)
        record_counts[(record["sample_id"], record["operator_index"])] = counts
        for name in ("events", "groups", "pwp_tokens", "pwp_updates",
                     "positive", "negative"):
            totals[name] += int(counts[name].sum())
        totals["baseline_events"] += counts["baseline_events"]
        totals["coefficient_checks"] += counts["coefficient_checks"]
        pwp_uses.update(counts["pwp_uses"])
        print("[M108R2 INDEPENDENT] {}/20 sample={} op={} events={} pwp={}".format(
            index, record["sample_id"], record["operator_index"],
            int(counts["events"].sum()), int(counts["pwp_tokens"].sum())),
            flush=True)

    require(totals["events"] == EXPECTED_EVENTS, "event aggregate drift")
    require(totals["positive"] == EXPECTED_POSITIVE, "positive event drift")
    require(totals["negative"] == EXPECTED_NEGATIVE, "negative event drift")
    require(totals["groups"] == EXPECTED_GROUPS, "active group drift")
    require(totals["pwp_updates"] == EXPECTED_PWP_UPDATES, "PWP update drift")
    require(totals["pwp_tokens"] == EXPECTED_PWP_TOKENS, "PWP token drift")
    require(totals["baseline_events"] == EXPECTED_BASELINE_EVENTS,
            "fixed8 baseline event drift")
    require(dict(sorted(pwp_uses.items())) == EXPECTED_PWP_USES,
            "PWP uses-by-width drift")
    require(totals["coefficient_checks"] == 3317760000,
            "coefficient check extent drift")

    sequence = []
    empty_descriptors = 0
    for sample in range(5, 10):
        for op in range(4):
            counts = record_counts[(sample, op)]
            for window in range(WINDOWS):
                for partition in range(PARTITIONS):
                    row = {
                        "sample": sample,
                        "operator": op,
                        "window": window,
                        "partition": partition,
                        "events": int(counts["events"][partition, window]),
                        "groups": int(counts["groups"][partition, window]),
                        "pwp_tokens": int(counts["pwp_tokens"][partition, window]),
                    }
                    empty_descriptors += int(row["groups"] == 0)
                    sequence.append(row)
    require(len(sequence) == 406080 and empty_descriptors == 35309,
            "sequence extent/empty count drift")

    reproduced = published_recurrence(sequence)
    corrected = independent_dual_timeline(sequence)
    result_schedule = published["rtl_edge_schedule"]
    for field in (
        "descriptor_fill_cycles", "producer_bank_stall_cycles",
        "controller_dispatch_edges", "bank_reacquire_boundaries",
        "exposed_post_pwp_fill_or_dispatch_wait_cycles", "service_idle_cycles",
        "pwp_service_tokens", "correction_service_tokens",
        "accumulator_pipeline_flush_cycles", "accumulator_commit_cycles",
        "candidate_cycles", "fair_fixed8_baseline_cycles",
        "headroom_to_two_x_cycles",
    ):
        require(reproduced[field] == result_schedule[field],
                "published recurrence mismatch " + field)
    require(math.isclose(reproduced["same_clock_service_island_ratio"],
                         result_schedule["same_clock_service_island_ratio"],
                         rel_tol=0.0, abs_tol=1e-15), "published ratio mismatch")
    require(reproduced["ordered_descriptor_sha256"]
            == corrected["ordered_descriptor_sha256"], "order digest mismatch")

    rtl_text = M106_RTL.read_text(encoding="utf-8")
    require("if (!drain_active_q" in rtl_text
            and "bank_state_q[next_drain_bank_q] == BANK_READY" in rtl_text,
            "M106 serialized dispatch guard evidence missing")
    require("drain_active_q <= 1'b0;" in rtl_text
            and "next_drain_bank_q <= ~next_drain_bank_q;" in rtl_text,
            "M106 drain completion evidence missing")
    require(m106["cycle_boundary"]["ready_dispatch_edge_per_window"]
            .startswith("required by current RTL"), "M106 edge contract drift")

    require(reproduced["candidate_cycles"] == 521238438,
            "published candidate expected value drift")
    require(reproduced["fair_fixed8_baseline_cycles"] == 1114864228,
            "baseline expected value drift")
    require(corrected["candidate_cycles"] >= reproduced["candidate_cycles"],
            "corrected recurrence unexpectedly faster")

    payload = {
        "schema": "m108_r2_rtl_edge_fused_schedule_independent_audit_v1",
        "status": "RAW_LEDGER_AND_PUBLISHED_FORMULA_REPRODUCED_CONTROLLER_SERIALIZATION_P0_FOUND",
        "identity": observed,
        "manifest_entries_verified": manifest_entries,
        "raw_reconstruction": {
            "heldout_records": 20,
            "descriptors": len(sequence),
            "empty_descriptors": empty_descriptors,
            "events": totals["events"],
            "positive_events": totals["positive"],
            "negative_events": totals["negative"],
            "active_groups": totals["groups"],
            "correction_service_tokens": totals["events"] + 3 * totals["groups"],
            "pwp_updates": totals["pwp_updates"],
            "pwp_uses_by_width": dict(sorted(pwp_uses.items())),
            "pwp_service_tokens": totals["pwp_tokens"],
            "source_coefficient_checks": totals["coefficient_checks"],
            "fixed8_baseline_events": totals["baseline_events"],
            "fixed8_baseline_tokens": BASELINE_TOKENS,
            "weight_payload_sha256": weight_shas,
            "weight_width_histogram": dict(sorted(width_histogram.items())),
        },
        "published_recurrence_reproduction": {
            **reproduced,
            "exact_match_to_m108_r2_result": True,
        },
        "independent_dual_timeline_recurrence": corrected,
        "difference": {
            "candidate_underestimate_cycles":
                corrected["candidate_cycles"] - reproduced["candidate_cycles"],
            "ratio_overstatement":
                reproduced["same_clock_service_island_ratio"]
                - corrected["same_clock_service_island_ratio"],
            "headroom_overstatement_cycles":
                reproduced["headroom_to_two_x_cycles"]
                - corrected["headroom_to_two_x_cycles"],
            "root_cause": (
                "The published recurrence computes dispatch_ready solely as fill_end+1. "
                "Frozen M106 permits dispatch only when drain_active_q is false, so the "
                "next descriptor dispatch edge must follow both fill_end and the prior "
                "M106 drain/release. PWP can hide this edge on the shared lane, but zero-PWP "
                "and empty descriptors still require in-order controller serialization."
            ),
        },
        "baseline_fairness": {
            "raw_fixed8_events_reconstructed": True,
            "three_tokens_per_fixed8_event": True,
            "common_commit_and_flush_candidate": reproduced["accumulator_commit_cycles"]
                + reproduced["accumulator_pipeline_flush_cycles"],
            "common_commit_and_flush_baseline": reproduced["accumulator_commit_cycles"]
                + reproduced["accumulator_pipeline_flush_cycles"],
            "candidate_baseline_common_tail_symmetric": True,
            "controller_and_descriptor_ingress_edges_charged_to_baseline": False,
            "fairness_note": (
                "The frozen denominator is a service-token baseline plus the common tail, "
                "not an edge-aware fixed8 controller/ingress schedule. It is reproducible "
                "but does not establish an equal-controller end-to-end cycle comparison."
            ),
        },
        "p0_closure_audit": {
            "m107_fluid_ready_dispatch_boundary": "PARTIALLY_CLOSED",
            "fill_end_plus_one_added": True,
            "blocked_bank_reacquire_plus_one_added": True,
            "prior_drain_in_order_dispatch_dependency_added": False,
            "m107_p0_fully_closed": False,
        },
        "model_boundary": {
            "precompacted_pwp_and_correction_input": True,
            "pwp_seed_before_descriptor_correction": True,
            "one_shared_256b_service_lane": True,
            "accumulator_flush_and_commit_cycles": True,
            "actual_combined_controller_pwp_accumulator_rtl": False,
            "commercial_small_stream_cycle_miter": False,
            "shared_weight_sram_address_port_latency_contention": False,
            "precompaction_scan_buffer_or_bandwidth_schedule": False,
            "full_lane_signed24_numeric_miter": False,
            "accumulator_macro_rdw_clear_epoch_commit_proof": False,
            "macro_inclusive_ppa": False,
        },
        "admission": {
            "exact_raw_work_ledger": True,
            "published_software_formula_reproduced": True,
            "cycle_exact_current_m106_controller": False,
            "scheduled_precompacted_module_cycle_ratio": False,
            "corrected_dual_timeline_software_bound": True,
            "service_island_only": True,
            "physical_speedup": False,
            "equal_area": False,
            "macro_inclusive_ppa": False,
            "system_speedup": False,
            "headline": False,
        },
        "contract_observations": {
            "contract_status": contract["status"],
            "contract_expected_candidate_reproduced": True,
            "contract_expected_ratio_reproduced": True,
            "m107_prior_edge_ratio": m107["m106_rtl_edge_recurrence"]
                ["window_major"]["same_clock_service_island_ratio"],
        },
        "docs_359_sha256_unchanged": sha256(DOC359),
        "producer_analyzer_executed": False,
        "production_files_modified": False,
    }
    require(sha256(DOC359) == EXPECTED_SHA[DOC359], "docs/359 changed during audit")
    require(sha256(Path(__file__).resolve()) == start_sha,
            "independent auditor changed during run")
    require(not OUTPUT.exists(), "refusing independent output overwrite")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS raw and published formula; corrected candidate={} ratio={:.12f} "
          "underestimate={} output={}".format(
              corrected["candidate_cycles"],
              corrected["same_clock_service_island_ratio"],
              payload["difference"]["candidate_underestimate_cycles"], OUTPUT),
          flush=True)


if __name__ == "__main__":
    main()
