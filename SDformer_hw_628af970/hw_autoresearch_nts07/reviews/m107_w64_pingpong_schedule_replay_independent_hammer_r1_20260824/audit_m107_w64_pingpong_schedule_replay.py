#!/usr/bin/env python3
"""Independent raw-source reconstruction and cycle-boundary audit for M107.

No M105/M107 producer module is imported or executed.  The script rebuilds
cap11 W64 event/group windows from frozen M40/M72/M41 inputs, reproduces the
published fluid ping-pong recurrence, and compares it with an event-driven
recurrence that includes the dispatch and bank-reacquire edges visible in the
frozen M106 RTL.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
OUTPUT = REVIEW / "m107_w64_pingpong_schedule_replay_independent_audit.json"

M107_CONTRACT = HW / "contracts/m107_w64_pingpong_schedule_replay_contract_r1_20260824.json"
M107_ANALYZER = HW / "system_simulator/scripts/analyze_m107_w64_pingpong_schedule_replay.py"
M107_RESULT_DIR = HW / "results/m107_w64_pingpong_schedule_replay_r1_20260824"
M107_RESULT = M107_RESULT_DIR / "m107_w64_pingpong_schedule_replay.json"
M107_RUN = M107_RESULT_DIR / "RUN_COMPLETE.txt"
M107_MANIFEST = M107_RESULT_DIR / "manifest.sha256"
M105_SCRIPT = HW / "reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/audit_m105_bounded_row_transpose.py"
M105_RESULT = HW / "reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824/m105_bounded_row_transpose_preflight.json"
M106_CONTRACT = HW / "contracts/m106_w64_bounded_bitmap_transpose_vcs_contract_r1_20260824.json"
M106_RTL = HW / "rtl_m106/m106_bounded_bitmap_transpose_scheduler.sv"
M106_RUN = HW / "dc_handoff/runs/m106_w64_bounded_bitmap_transpose_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
M40_DIR = HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822"
M40_MANIFEST = M40_DIR / "m40_bottleneck_packed_source_manifest.json"
M72_RESULT = HW / "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/m72_phi_kmeans_k16q16_valid825_internal_screen.json"
M78_RESULT = HW / "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/m78_precision_elastic_pwp.json"
M41_DIR = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41_RESULT = M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SHA = {
    M107_CONTRACT: "84ce5aac54075b431988995b9ceaec592bfe21eabbefce594069ef7d3abb3002",
    M107_ANALYZER: "2c655e2e907d046761de2ce44563ae1f0505798d40fdaba5faca98c192d741b8",
    M107_RESULT: "0c613c2da4eb3e860ccc33dfbbe3fba0bb1424c19ab12af339e54178d03333db",
    M107_RUN: "83234c3e626278a0bff2cf7bd5c481a08504b7448a33802870e1fdf186027fc7",
    M107_MANIFEST: "8857341cc2263bc1d75cef3b9d6815e68b9153a8b8329ae15654c9bd05519678",
    M105_SCRIPT: "5e5c07631dd8c4bb328cd234da5c04fde8eb9800d1516b3fe462124b2b661ed5",
    M105_RESULT: "3348b6c02ad97be5b61ffb6f8d5f79578f4551e037097c4f74ac598d2842767b",
    M106_CONTRACT: "881491f58543f2c6b0b5b3c1d07d7b170cdbfb4190153a18929bdddd83a39999",
    M106_RTL: "0abc1adf612788bbfdd2f26ff847234ee7efaaa2addcc7f28f03ddac22cd68e7",
    M106_RUN: "fc118089b84ea99c1ed72077171539bd113aa85e0d07f709d35800ae23b5b1d4",
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
PWP_TOKENS = 226222255
BASELINE = 1114383288


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
    width_hist = Counter()
    observed_weight_sha = []
    for op, operator in enumerate(m72["operators"]):
        payload = next(row for row in layers[operator["operator"]]["payloads"]
                       if row["role"] == "weight")
        path = M41_DIR / payload["file"]
        observed = sha256(path)
        require(observed == payload["sha256"] == WEIGHT_SHA[op],
                "weight payload identity drift")
        observed_weight_sha.append(observed)
        weight = np.fromfile(str(path), dtype=np.int8)
        require(weight.size == FEATURES * CHANNELS, "weight extent drift")
        weight = weight.reshape(FEATURES, CHANNELS).astype(np.int32)
        for partition, partition_row in enumerate(operator["partitions"]):
            require(partition_row["partition"] == partition,
                    "partition order drift")
            source = weight[partition * PARTITION_BITS:
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
                    width_hist[bits] += 1
    return widths, dict(sorted(width_hist.items())), observed_weight_sha


def centers_array(m72):
    centers = np.zeros((4, PARTITIONS, PATTERNS), dtype=np.uint16)
    for op, operator in enumerate(m72["operators"]):
        for partition, row in enumerate(operator["partitions"]):
            require(row["partition"] == partition, "center partition order drift")
            values = [int(value, 16) for value in row["centers_hex"]]
            require(len(values) == PATTERNS and len(set(values)) == PATTERNS,
                    "center extent/uniqueness drift")
            centers[op, partition] = values
    return centers


def decode_masks(record, popcount):
    path = M40_DIR / record["packed_file"]
    require(path.is_file() and sha256(path) == record["packed_file_sha256"],
            "packed record identity drift")
    raw = path.read_bytes()
    plane_bytes = record["positive_plane_bytes"]
    require(len(raw) == 3 * plane_bytes, "packed record extent drift")
    require(not any(raw[plane_bytes:2 * plane_bytes]), "negative plane not zero")
    total_bits = TIMESTEPS * CHANNELS * HEIGHT * WIDTH
    support = np.unpackbits(np.frombuffer(raw[:plane_bytes], dtype=np.uint8),
                            bitorder="little")[:total_bits]
    support = support.reshape(TIMESTEPS, CHANNELS, HEIGHT, WIDTH)
    require(int(support.sum()) == record["nonzero_count"],
            "support population drift")
    padded = np.pad(support, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")
    im2col = np.empty((ROWS, FEATURES), dtype=np.uint8)
    for ky in range(3):
        for kx in range(3):
            slot = ky * 3 + kx
            view = padded[:, :, ky:ky + HEIGHT, kx:kx + WIDTH]
            im2col[:, slot::9] = view.transpose(0, 2, 3, 1).reshape(ROWS, CHANNELS)
    powers = np.uint16(1) << np.arange(PARTITION_BITS, dtype=np.uint16)
    masks = (im2col.reshape(ROWS, PARTITIONS, PARTITION_BITS) * powers).sum(
        axis=2, dtype=np.uint16).T.copy()
    require(int(popcount[masks].sum()) > 0, "decoded mask population zero")
    return masks


def record_windows(masks, op, centers, widths, popcount):
    event_masks = np.zeros((PARTITIONS, ROWS, OUTPUT_BLOCKS), dtype=np.uint16)
    pwp_uses = Counter()
    for partition in range(PARTITIONS):
        values = masks[partition]
        center_values = centers[op, partition]
        numeric_order = np.argsort(center_values, kind="stable")
        ordered_centers = center_values[numeric_order]
        distance = popcount[np.bitwise_xor(values[:, None],
                                           ordered_centers[None, :])]
        ordered_choice = distance.argmin(axis=1)
        choice = numeric_order[ordered_choice]
        best_distance = distance[np.arange(ROWS), ordered_choice]
        selected_center = center_values[choice]
        population = popcount[values]
        beneficial = (1 + best_distance) < population
        delta = np.bitwise_xor(values, selected_center)
        selected_width = widths[op, partition, choice]
        eligible = beneficial[:, None] & (selected_width <= CAP)
        event_masks[partition] = np.where(
            eligible, delta[:, None], values[:, None]).astype(np.uint16)
        for bits in (8, 9, 10, 11):
            pwp_uses[bits] += int(np.count_nonzero(
                eligible & (selected_width == bits)))

    padded_rows = WINDOWS * WIN_ROWS
    padded = np.pad(event_masks, ((0, 0), (0, padded_rows - ROWS), (0, 0)),
                    mode="constant")
    shaped = padded.reshape(PARTITIONS, WINDOWS, WIN_ROWS, OUTPUT_BLOCKS)
    events = popcount[shaped].sum(axis=(2, 3), dtype=np.uint32).astype(np.int64)
    union = np.bitwise_or.reduce(shaped, axis=2)
    groups = popcount[union].sum(axis=2, dtype=np.uint16).astype(np.int64)
    require(events.shape == groups.shape == (PARTITIONS, WINDOWS),
            "window shape drift")
    require(np.all(events >= groups), "event/group relation drift")
    require(np.all(events[groups == 0] == 0), "event in empty window")
    return events, groups, pwp_uses


def fluid_schedule(events, groups):
    bank_free = [0, 0]
    producer_end = 0
    drain_end = 0
    producer_stall = 0
    service_idle = 0
    empty = 0
    fill_total = 0
    token_total = 0
    drain_wait_on_fill = 0
    drain_back_to_back = 0
    for index, (event_count, group_count) in enumerate(zip(events, groups)):
        event_count = int(event_count)
        group_count = int(group_count)
        if group_count == 0:
            require(event_count == 0, "fluid event in empty window")
            empty += 1
        bank = index & 1
        fill_start = max(producer_end, bank_free[bank])
        producer_stall += fill_start - producer_end
        fill_cycles = event_count + 1
        fill_end = fill_start + fill_cycles
        producer_end = fill_end
        drain_start = max(drain_end, fill_end)
        if drain_start == drain_end:
            drain_back_to_back += 1
        if drain_start == fill_end and fill_end > drain_end:
            drain_wait_on_fill += 1
        service_idle += drain_start - drain_end
        drain_tokens = event_count + 3 * group_count
        drain_end = drain_start + drain_tokens
        bank_free[bank] = drain_end
        fill_total += fill_cycles
        token_total += drain_tokens
    require(drain_end == token_total + service_idle,
            "fluid conservation failed")
    return {
        "windows": len(events),
        "empty_windows": empty,
        "nonempty_windows": len(events) - empty,
        "descriptor_fill_cycles": fill_total,
        "correction_service_tokens": token_total,
        "producer_bank_stall_cycles": producer_stall,
        "service_idle_cycles": service_idle,
        "correction_schedule_cycles": drain_end,
        "producer_end": producer_end,
        "drain_wait_on_fill_boundaries": drain_wait_on_fill,
        "drain_back_to_back_boundaries": drain_back_to_back,
    }


def rtl_edge_schedule(events, groups):
    """Event-driven recurrence for the accept-edge latency in frozen M106 RTL.

    A READY bank is observed one edge after close and consumes one dispatch edge
    before service_valid.  If close cannot immediately switch to the other bank,
    an EMPTY bank is observed on one edge and the first new fill accept is on the
    following edge.  These are controller edges, not physical SRAM latency.
    """
    bank_free = [0, 0]
    producer_end = 0
    drain_end = 0
    producer_stall = 0
    service_idle = 0
    dispatch_cycles = 0
    reacquire_edges = 0
    token_total = 0
    empty = 0
    for index, (event_count, group_count) in enumerate(zip(events, groups)):
        event_count = int(event_count)
        group_count = int(group_count)
        if group_count == 0:
            require(event_count == 0, "edge event in empty window")
            empty += 1
        bank = index & 1
        if index == 0 or producer_end > bank_free[bank]:
            fill_start = producer_end
        else:
            fill_start = bank_free[bank] + 1
            reacquire_edges += 1
        producer_stall += fill_start - producer_end
        fill_end = fill_start + event_count + 1
        producer_end = fill_end
        dispatch_start = max(drain_end, fill_end)
        service_idle += dispatch_start - drain_end
        dispatch_cycles += 1
        drain_tokens = event_count + 3 * group_count
        drain_end = dispatch_start + 1 + drain_tokens
        bank_free[bank] = drain_end
        token_total += drain_tokens
    require(drain_end == token_total + dispatch_cycles + service_idle,
            "RTL-edge conservation failed")
    return {
        "windows": len(events),
        "empty_windows": empty,
        "nonempty_windows": len(events) - empty,
        "correction_service_tokens": token_total,
        "controller_dispatch_cycles": dispatch_cycles,
        "bank_reacquire_boundaries": reacquire_edges,
        "producer_bank_stall_cycles": producer_stall,
        "service_idle_cycles": service_idle,
        "correction_schedule_cycles": drain_end,
        "producer_end": producer_end,
    }


def schedule_summary(schedule, pwp_tokens=PWP_TOKENS):
    combined = schedule["correction_schedule_cycles"] + pwp_tokens
    return {
        **schedule,
        "pwp_tokens_serially_charged": pwp_tokens,
        "combined_service_island_cycles": combined,
        "same_clock_service_island_ratio": BASELINE / float(combined),
        "two_x_limit": BASELINE // 2,
        "headroom_to_two_x_cycles": BASELINE // 2 - combined,
        "headroom_fraction_of_combined": (BASELINE // 2 - combined) / float(combined),
    }


def main():
    start_sha = sha256(Path(__file__).resolve())
    observed = {}
    for path, expected in EXPECTED_SHA.items():
        actual = sha256(path)
        require(actual == expected, "identity mismatch {} {}".format(path, actual))
        observed[str(path.relative_to(HW))] = actual
    require(verify_manifest(M107_MANIFEST, HW) == 8,
            "M107 manifest count drift")

    contract = strict_json(M107_CONTRACT)
    result = strict_json(M107_RESULT)
    m105 = strict_json(M105_RESULT)
    m40 = strict_json(M40_MANIFEST)
    m72 = strict_json(M72_RESULT)
    m78 = strict_json(M78_RESULT)
    m41 = strict_json(M41_RESULT)
    m106 = strict_json(M106_CONTRACT)

    popcount = np.fromiter((bin(value).count("1") for value in range(1 << 16)),
                           dtype=np.uint8, count=1 << 16)
    centers = centers_array(m72)
    widths, width_hist, weight_sha = build_widths(m72, m41)
    frozen_width_hist = {int(key): value for key, value in
                         m78["pwp_precision"]["width_histogram"].items()}
    require(width_hist == frozen_width_hist, "PWP width histogram drift")

    heldout = sorted((row for row in m40["records"]
                      if row["sample_id"] in range(5, 10)),
                     key=lambda row: (row["sample_id"], row["operator_index"]))
    require([(row["sample_id"], row["operator_index"]) for row in heldout]
            == [(sample, op) for sample in range(5, 10) for op in range(4)],
            "heldout record order drift")
    records = []
    pwp_uses = Counter()
    total_events = 0
    total_groups = 0
    empty_windows = 0
    for record_index, record in enumerate(heldout, 1):
        masks = decode_masks(record, popcount)
        events, groups, record_pwp = record_windows(
            masks, record["operator_index"], centers, widths, popcount)
        records.append((events, groups))
        pwp_uses.update(record_pwp)
        total_events += int(events.sum())
        total_groups += int(groups.sum())
        empty_windows += int(np.count_nonzero(groups == 0))
        print("[M107 INDEPENDENT] {}/20 sample={} op={} events={} groups={}".format(
            record_index, record["sample_id"], record["operator_index"],
            int(events.sum()), int(groups.sum())), flush=True)

    require(total_events == 188148490, "aggregate event drift")
    require(total_groups == 35140002, "aggregate group drift")
    require(empty_windows == 35309, "empty window drift")
    require(dict(sorted(pwp_uses.items())) == {
        8: 11164284, 9: 32360036, 10: 13936011, 11: 1509043,
    }, "PWP use count drift")
    pwp_recalculated = (pwp_uses[8] * 3 + pwp_uses[9] * 4
                        + pwp_uses[10] * 4 + pwp_uses[11] * 5)
    require(pwp_recalculated == PWP_TOKENS, "PWP token drift")

    partition_events = np.concatenate([events.reshape(-1) for events, _ in records])
    partition_groups = np.concatenate([groups.reshape(-1) for _, groups in records])
    window_events = np.concatenate([events.T.reshape(-1) for events, _ in records])
    window_groups = np.concatenate([groups.T.reshape(-1) for _, groups in records])
    require(partition_events.size == window_events.size == 406080,
            "sequence extent drift")
    require(np.array_equal(np.sort(partition_events), np.sort(window_events)) and
            np.array_equal(np.sort(partition_groups), np.sort(window_groups)),
            "loop order multiset drift")
    require(int(partition_events.sum()) == int(window_events.sum()) == total_events,
            "loop event conservation drift")
    require(int(partition_groups.sum()) == int(window_groups.sum()) == total_groups,
            "loop group conservation drift")

    partition_fluid = schedule_summary(fluid_schedule(partition_events, partition_groups))
    window_fluid = schedule_summary(fluid_schedule(window_events, window_groups))
    partition_edge = schedule_summary(rtl_edge_schedule(partition_events, partition_groups))
    window_edge = schedule_summary(rtl_edge_schedule(window_events, window_groups))

    for label, independent, published in (
        ("partition_major", partition_fluid, result["partition_major_schedule"]),
        ("window_major", window_fluid, result["window_major_schedule"]),
    ):
        for field in (
            "windows", "empty_windows", "nonempty_windows",
            "correction_service_tokens", "descriptor_fill_cycles",
            "producer_bank_stall_cycles", "correction_schedule_cycles",
        ):
            require(independent[field] == published[field],
                    "published fluid mismatch {} {}".format(label, field))
        require(independent["service_idle_cycles"] == published[
            "service_idle_cycles_from_initial_fill_or_bank_not_ready"],
            "published idle mismatch " + label)
        require(independent["combined_service_island_cycles"] == published[
            "combined_service_island_cycles"], "published combined mismatch " + label)
        require(math.isclose(independent["same_clock_service_island_ratio"],
                             published["same_clock_service_island_ratio"],
                             rel_tol=0.0, abs_tol=1e-15),
                "published ratio mismatch " + label)

    w64 = next(row for row in m105["window_results"] if row["window_rows"] == 64)
    require(w64["correction_fallback_events"] == total_events and
            w64["active_groups_total"] == total_groups and
            w64["correction_fallback_token_envelope"]
            == total_events + 3 * total_groups == 293568496,
            "M105 W64 conservation drift")
    require(result["work_conservation"]["pwp_tokens"] == pwp_recalculated,
            "published PWP charge drift")

    rtl_text = M106_RTL.read_text(encoding="utf-8")
    require("bank_state_q[next_drain_bank_q] == BANK_READY" in rtl_text and
            "drain_active_q <= 1'b1;" in rtl_text,
            "M106 dispatch edge evidence missing")
    require("if (!fill_available_q)" in rtl_text and
            "fill_available_q <= 1'b1;" in rtl_text,
            "M106 reacquire edge evidence missing")
    require(m106["microarchitecture"]["empty_window_policy"]
            == "one control cycle and zero service tokens",
            "M106 empty policy drift")

    selected_reported_headroom = window_fluid["headroom_to_two_x_cycles"]
    selected_edge_headroom = window_edge["headroom_to_two_x_cycles"]
    require(selected_reported_headroom == 10731198, "reported headroom drift")
    require(selected_edge_headroom > 0, "edge-aware schedule falls below 2x")

    pwp_semantics = {
        "uses_by_width": dict(sorted(pwp_uses.items())),
        "serial_tokens": pwp_recalculated,
        "charged_exactly_once_in_combined": True,
        "not_in_correction_event_or_group_term": True,
        "dependency_order_executable": False,
        "reason": (
            "eligible PWP seeds and correction deltas are distinct work, but M107 appends "
            "one aggregate PWP term after all correction drain rather than replaying "
            "per-row seed-before-correction dependencies or shared SRAM contention"
        ),
    }

    payload = {
        "schema": "m107_w64_pingpong_schedule_replay_independent_audit_v1",
        "status": "PUBLISHED_FLUID_REPLAY_EXACT_EDGE_DISPATCH_UNDERCOUNT_FOUND_SERVICE_ISLAND_ONLY",
        "identity": observed,
        "manifest_entries_verified": 8,
        "raw_reconstruction": {
            "heldout_records": 20,
            "windows": 406080,
            "empty_windows": empty_windows,
            "nonempty_windows": 406080 - empty_windows,
            "events": total_events,
            "active_groups": total_groups,
            "correction_service_tokens": total_events + 3 * total_groups,
            "pwp_uses_by_width": dict(sorted(pwp_uses.items())),
            "pwp_tokens": pwp_recalculated,
            "weight_payload_sha256": weight_sha,
            "width_histogram": width_hist,
        },
        "published_fluid_reproduction": {
            "partition_major": partition_fluid,
            "window_major": window_fluid,
            "exact_match_to_m107_json": True,
        },
        "m106_rtl_edge_recurrence": {
            "model_boundary": (
                "always-ready service and no producer grace; adds controller READY dispatch "
                "and blocked-fill bank reacquire edges visible in frozen RTL; still excludes "
                "accumulator, shared SRAM, physical memory and PWP dependency schedule"
            ),
            "partition_major": partition_edge,
            "window_major": window_edge,
            "published_model_omits_dispatch_cycle_per_window": True,
            "published_selected_window_major_underestimate_cycles": (
                window_edge["correction_schedule_cycles"]
                - window_fluid["correction_schedule_cycles"]
            ),
            "selected_edge_aware_ratio_still_above_two": (
                window_edge["same_clock_service_island_ratio"] > 2.0
            ),
        },
        "pwp_charge_audit": pwp_semantics,
        "headroom": {
            "two_x_combined_limit": BASELINE // 2,
            "reported_partition_major_cycles": partition_fluid[
                "combined_service_island_cycles"],
            "reported_partition_major_headroom": partition_fluid[
                "headroom_to_two_x_cycles"],
            "reported_window_major_cycles": window_fluid[
                "combined_service_island_cycles"],
            "reported_window_major_headroom": selected_reported_headroom,
            "edge_aware_window_major_cycles": window_edge[
                "combined_service_island_cycles"],
            "edge_aware_window_major_headroom": selected_edge_headroom,
        },
        "port_cuts": {
            "accumulator_schedule": False,
            "accumulator_finite_width_miter": False,
            "accumulator_bank_and_rmw_replay": False,
            "shared_weight_sram_schedule": False,
            "pwp_seed_correction_dependency_replay": False,
            "physical_memory_latency": False,
        },
        "admission": {
            "exact_raw_event_group_and_pwp_reconstruction": True,
            "published_fluid_pingpong_recurrence": True,
            "cycle_exact_current_m106_controller": False,
            "edge_aware_software_bound": True,
            "service_island_only": True,
            "scheduled_accumulator": False,
            "physical_speedup": False,
            "equal_area": False,
            "macro_inclusive_ppa": False,
            "system_speedup": False,
            "headline": False,
        },
        "docs_359_sha256_unchanged": sha256(DOC359),
        "producer_analyzer_executed": False,
        "production_files_modified": False,
    }
    require(sha256(DOC359) == EXPECTED_SHA[DOC359], "docs/359 changed during audit")
    require(sha256(Path(__file__).resolve()) == start_sha,
            "independent script changed during run")
    require(not OUTPUT.exists(), "refusing independent output overwrite")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M107 independent raw reconstruction; published fluid exact; "
          "RTL-edge undercount={} selected_ratio={:.12f} scheduled=false physical=false".format(
              payload["m106_rtl_edge_recurrence"]
              ["published_selected_window_major_underestimate_cycles"],
              window_edge["same_clock_service_island_ratio"]), flush=True)


if __name__ == "__main__":
    main()
