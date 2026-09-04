#!/usr/bin/env python3
"""M484 exact-row signed-bundle/state-stationary offline DSE.

This script consumes the frozen H67 ep35 M51 bit-packed binary activations.  It
reconstructs every logical output row for eligible Conv2d and FC1 calls, applies
the already-frozen row selector (current row versus signed temporal delta), and
measures an offline destination-major N-source fold for N=1..8.  It is a
measurement/screen only: it does not model descriptor reorder construction,
RTL timing, SRAM macros, or system speedup.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED = {
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "execution_trace": "ad8d1f286c0936ce7cf42324068cfd074aeef3cf77af62890e0598b663b91bfd",
    "dual_line_trace": "2390dc3ee5f093a2c760cd53d7b9587f874767b78073da8b99f3a88b5079bd1c",
    "transactions": "dbd6630b3bec3726762270ae6c6c24b6328da7c65d6f2c6a5878be3940b4ef59",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

WIDTHS = tuple(range(1, 9))
SLOTS = (1, 2, 4, 8)
ACC_BITS = 32
WEIGHT_BITS = 8
STATE_BANKS = 8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def bits_required(count):
    return max(1, int(math.ceil(math.log2(max(2, int(count))))))


def weighted_percentile(histogram, percentile):
    total = sum(histogram.values())
    if total == 0:
        return 0
    target = int(math.ceil(total * float(percentile) / 100.0))
    running = 0
    for value in sorted(histogram):
        running += int(histogram[value])
        if running >= target:
            return int(value)
    raise RuntimeError("percentile histogram underflow")


def decode_record(record, payload_root):
    path = payload_root / record["relative_path"]
    require(path.is_file(), "missing payload {}".format(path))
    require(sha256(path) == record["file_sha256"],
            "payload SHA drift {}".format(path))
    shape = tuple(int(value) for value in record["input_shape"])
    expected_bits = int(np.prod(shape))
    require(expected_bits == int(record["input_elements"]),
            "input geometry mismatch")
    packed = np.fromfile(str(path), dtype=np.uint8)
    require(int(packed.size) == int(record["packed_bytes"]),
            "packed byte mismatch")
    unpacked = np.unpackbits(packed, bitorder="little")[:expected_bits]
    value = unpacked.reshape(shape).astype(np.bool_)
    require(int(value.sum(dtype=np.int64)) == int(record["active_elements"]),
            "active count mismatch")
    return value


def conv_geometry(record):
    name = record["name"]
    if name.endswith("preds.3.conv.0"):
        return 1, 1, 0
    return 3, 2 if ("patch_embed.conv.conv.0" in name or
                    "patch_embed.proj.conv" in name) else 1, 1


def spatial_receptive_count(mask, kernel, stride, padding, output_shape):
    """Count active input scalars in every Conv receptive row."""
    require(mask.ndim == 4, "Conv slice must be B,C,H,W")
    channel_count = mask.sum(axis=1, dtype=np.int32)
    padded = np.pad(channel_count,
                    ((0, 0), (padding, padding), (padding, padding)))
    out_h, out_w = int(output_shape[-2]), int(output_shape[-1])
    result = np.zeros((mask.shape[0], out_h, out_w), dtype=np.int32)
    for ky in range(kernel):
        for kx in range(kernel):
            result += padded[:, ky:ky + out_h * stride:stride,
                             kx:kx + out_w * stride:stride]
    return result


def selected_row_counts(record, value):
    is_conv = record["operator"] == "Conv2d"
    if is_conv and value.ndim == 4:
        value = value[:, None, :, :, :]
    require(value.ndim == 5, "unsupported exact-binary tensor rank")
    previous = np.zeros_like(value[0], dtype=np.bool_)
    selected_parts = []
    positive_parts = []
    negative_parts = []
    motion_parts = []
    local_total = 0
    transition_total = 0
    for timestep in range(value.shape[0]):
        current = value[timestep]
        positive = np.logical_and(current, np.logical_not(previous))
        negative = np.logical_and(np.logical_not(current), previous)
        if is_conv:
            kernel, stride, padding = conv_geometry(record)
            local = spatial_receptive_count(current, kernel, stride, padding,
                                             record["output_shape"])
            pos = spatial_receptive_count(positive, kernel, stride, padding,
                                           record["output_shape"])
            neg = spatial_receptive_count(negative, kernel, stride, padding,
                                           record["output_shape"])
        else:
            local = current.sum(axis=-1, dtype=np.int32)
            pos = positive.sum(axis=-1, dtype=np.int32)
            neg = negative.sum(axis=-1, dtype=np.int32)
        transition = pos + neg
        use_motion = transition < local
        selected = np.where(use_motion, transition, local).astype(np.int32)
        selected_pos = np.where(use_motion, pos, local).astype(np.int32)
        selected_neg = np.where(use_motion, neg, 0).astype(np.int32)
        require(np.array_equal(selected, selected_pos + selected_neg),
                "signed source conservation failed")
        selected_parts.append(selected.reshape(-1))
        positive_parts.append(selected_pos.reshape(-1))
        negative_parts.append(selected_neg.reshape(-1))
        motion_parts.append(use_motion.reshape(-1))
        local_total += int(local.sum(dtype=np.int64))
        transition_total += int(transition.sum(dtype=np.int64))
        previous = current
    return {
        "selected": np.concatenate(selected_parts),
        "positive": np.concatenate(positive_parts),
        "negative": np.concatenate(negative_parts),
        "motion": np.concatenate(motion_parts),
        "local_total": local_total,
        "transition_total": transition_total,
    }


def load_dual_line(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "PASS_EXACT_SOURCE_WORK":
                continue
            key = (int(row["sample_id"]), row["name"])
            entry = result.setdefault(key, {
                "selected_sources": 0,
                "positive": 0,
                "negative": 0,
                "current": 0,
                "selector_rows": 0,
                "motion_rows": 0,
            })
            fanout = int(row["output_channel_fanout"])
            selected_work = int(row["selected_work"])
            require(selected_work % fanout == 0,
                    "selected work is not divisible by fanout")
            entry["selected_sources"] += selected_work // fanout
            entry["positive"] += int(row["positive_transition_source_count"])
            entry["negative"] += int(row["negative_transition_source_count"])
            entry["current"] += int(row["current_source_count"])
            entry["selector_rows"] += int(row["selector_rows"])
            entry["motion_rows"] += int(row["motion_selected_rows"])
    return result


def load_atlif_pairs(path):
    rows = defaultdict(dict)
    sequences = set()
    samples = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample = int(row["sample_id"])
            rows[sample][int(row["call_index"])] = row
            sequences.add(row["sequence_key"])
            samples.add(sample)
    pairs = set()
    for sample, calls in rows.items():
        for index, row in calls.items():
            following = calls.get(index + 1)
            if (row["kind"] == "operator" and following is not None and
                    following["kind"] == "atlif" and
                    row["output_shape"] == following["input_shape"]):
                pairs.add((sample, row["name"]))
    return pairs, sorted(sequences), sorted(samples)


def audit_acc32_transaction_widths(transaction_path, execution_path,
                                   selected_records, atlif_pairs):
    execution = {}
    with Path(execution_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            execution[(int(row["sample_id"]), int(row["call_index"]))] = row
    phase_bytes = defaultdict(int)
    with Path(transaction_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["identity"] != "h67_ep35" or
                    row["variant"] != "motion_selector_shared_state"):
                continue
            phase_bytes[(int(row["sample_id"]), int(row["call_index"]),
                         row["phase"])] += int(row["byte_count"])
    operator_checks = 0
    atlif_pair_checks = 0
    for record in selected_records:
        sample = int(record["sample_id"])
        call = int(record["frozen_execution_call_index"])
        trace = execution[(sample, call)]
        require(trace["name"] == record["name"], "execution call/name drift")
        expected = int(trace["output_elements"]) * (ACC_BITS // 8)
        actual = phase_bytes[(sample, call, "operator_acc_write")]
        require(actual == expected,
                "M22 operator Acc32 width mismatch {} {} {}".format(
                    record["name"], actual, expected))
        operator_checks += 1
        if (record["operator"] == "Conv2d" and
                (sample, record["name"]) in atlif_pairs):
            atlif = execution[(sample, call + 1)]
            atlif_expected = int(atlif["output_elements"]) * (ACC_BITS // 8)
            for phase in ("atlif_state_read", "atlif_state_write"):
                actual = phase_bytes[(sample, call + 1, phase)]
                require(actual == atlif_expected,
                        "M22 ATLIF state width mismatch {} {} {}".format(
                            phase, actual, atlif_expected))
                atlif_pair_checks += 1
    return {
        "variant": "motion_selector_shared_state",
        "accumulator_bits_verified": ACC_BITS,
        "operator_acc_write_checks": operator_checks,
        "atlif_state_read_write_checks": atlif_pair_checks,
        "mismatches": 0,
    }


def per_record_metrics(record, selected, positive, negative, motion, paired):
    if record["operator"] == "Conv2d":
        # Conv tensors are either T,B,C,H,W or T,C,H,W.  The proj.conv call
        # is the latter and must not interpret H as the channel dimension.
        input_channel_axis = 2 if len(record["input_shape"]) == 5 else 1
        output_channel_axis = 2 if len(record["output_shape"]) == 5 else 1
        out_channels = int(record["output_shape"][output_channel_axis])
        source_width = (int(record["input_shape"][input_channel_axis]) *
                        conv_geometry(record)[0] ** 2)
    else:
        out_channels = int(record["output_shape"][-1])
        source_width = int(record["input_shape"][-1])
    rows = int(selected.size)
    nonempty = int(np.count_nonzero(selected))
    return {
        "sample_id": int(record["sample_id"]),
        "sample_key": record["sample_key"],
        "sequence_key": record["sequence_key"],
        "module_index": int(record["module_index"]),
        "name": record["name"],
        "operator": record["operator"],
        "out_channels": out_channels,
        "source_width": source_width,
        "rows": rows,
        "nonempty_rows": nonempty,
        "empty_rows": rows - nonempty,
        "selected_sources": int(selected.sum(dtype=np.int64)),
        "positive_sources": int(positive.sum(dtype=np.int64)),
        "negative_sources": int(negative.sum(dtype=np.int64)),
        "motion_selected_rows": int(np.count_nonzero(motion)),
        "row_switches_destination_major": max(0, nonempty - 1),
        "conv_atlif_pair": bool(paired),
        "selected_histogram": np.bincount(selected).astype(np.int64),
    }


def aggregate_category(name, records):
    require(records, "empty category {}".format(name))
    max_count = max(len(row["selected_histogram"]) for row in records)
    histogram = np.zeros(max_count, dtype=np.int64)
    for row in records:
        histogram[:len(row["selected_histogram"])] += row["selected_histogram"]
    return {
        "category": name,
        "records": len(records),
        "operators": len({row["name"] for row in records}),
        "samples": len({row["sample_id"] for row in records}),
        "rows": sum(row["rows"] for row in records),
        "nonempty_rows": sum(row["nonempty_rows"] for row in records),
        "empty_rows": sum(row["empty_rows"] for row in records),
        "selected_sources": sum(row["selected_sources"] for row in records),
        "positive_sources": sum(row["positive_sources"] for row in records),
        "negative_sources": sum(row["negative_sources"] for row in records),
        "motion_selected_rows": sum(row["motion_selected_rows"] for row in records),
        "row_switches_destination_major": sum(
            row["row_switches_destination_major"] for row in records),
        "record_rows": records,
        "histogram": histogram,
    }


def dse_point(category, width, slots):
    histogram = category["histogram"]
    counts = np.arange(len(histogram), dtype=np.int64)
    bundles_per_row = (counts + width - 1) // width
    bundles = int(np.dot(histogram, bundles_per_row))
    events = int(category["selected_sources"])
    nonempty = int(category["nonempty_rows"])
    padding = bundles * width - events
    wait_hist = defaultdict(int)
    full = 0
    remainder = 0
    for count, row_frequency in enumerate(histogram):
        if count == 0 or row_frequency == 0:
            continue
        full_bundles, rem = divmod(count, width)
        if full_bundles:
            wait_hist[width - 1] += int(row_frequency) * full_bundles
            full += int(row_frequency) * full_bundles
        if rem:
            wait_hist[rem - 1] += int(row_frequency)
            remainder += int(row_frequency)

    k1_resident_cycles = 0
    k8_resident_cycles = 0
    m484_cycles = 0
    resident_state_bits = 0
    weight_bits = 0
    resident_row_header_bits = 0
    header_bits = 0
    event_metadata_bits = 0
    padding_bits = 0
    for row in category["record_rows"]:
        h = row["selected_histogram"]
        c = np.arange(len(h), dtype=np.int64)
        row_bundles = int(np.dot(h, (c + width - 1) // width))
        row_events = int(row["selected_sources"])
        row_nonempty = int(row["nonempty_rows"])
        # A fair destination-resident baseline keeps the complete Acc32 vector
        # from row activation through commit.  K8 has the identical N lanes,
        # ports and fold tree as M484, so both issue ceil(events/N) groups.
        k1_resident_cycles += row_events + 2 * row_nonempty
        k8_resident_cycles += row_bundles + 2 * row_nonempty
        m484_cycles += row_bundles + 2 * row_nonempty
        vector_bits = int(row["out_channels"]) * ACC_BITS
        resident_state_bits += 2 * row_nonempty * vector_bits
        weight_bits += row_events * int(row["out_channels"]) * WEIGHT_BITS
        source_bits = bits_required(row["source_width"])
        row_id_bits = bits_required(row["rows"])
        resident_row_header_bits += row_nonempty * (
            row_id_bits + bits_required(row["source_width"] + 1))
        header_bits += row_bundles * (row_id_bits + bits_required(width + 1))
        event_metadata_bits += row_events * (source_bits + 1)
        padding_bits += (row_bundles * width - row_events) * (source_bits + 1)
        if category["category"] == "Conv->ATLIF":
            all_row_state_bits = int(row["rows"]) * vector_bits
            # The fair row-resident baseline is allowed the same direct ATLIF
            # handoff/fusion.  Only ATLIF state read/write remains in all modes.
            resident_state_bits += 2 * all_row_state_bits
            k1_resident_cycles += int(row["rows"])
            k8_resident_cycles += int(row["rows"])
            m484_cycles += int(row["rows"])

    metadata_bits = header_bits + event_metadata_bits
    k8_total_bits = (resident_state_bits + weight_bits + event_metadata_bits +
                     resident_row_header_bits)
    m484_total_bits = (resident_state_bits + weight_bits + metadata_bits +
                       padding_bits)
    k1_scaling = k1_resident_cycles / m484_cycles
    speedup = k8_resident_cycles / m484_cycles
    traffic_reduction = 1.0 - m484_total_bits / k8_total_bits
    return {
        "schedule": "offline_destination_major_oracle",
        "category": category["category"],
        "bundle_width": width,
        "finite_slots": slots,
        "records": category["records"],
        "operators": category["operators"],
        "samples": category["samples"],
        "rows": category["rows"],
        "nonempty_rows": nonempty,
        "selected_sources": events,
        "positive_sources": category["positive_sources"],
        "negative_sources": category["negative_sources"],
        "motion_selected_rows": category["motion_selected_rows"],
        "bundles": bundles,
        "full_bundles": full,
        "remainder_bundles": remainder,
        "bundle_occupancy": events / (bundles * width) if bundles else 1.0,
        "mean_sources_per_bundle": events / bundles if bundles else 0.0,
        "padding_slots": padding,
        "row_switches_baseline": category["row_switches_destination_major"],
        "row_switches_candidate": category["row_switches_destination_major"],
        "parallel_source_updates_k1_minus_k8": events - bundles,
        "state_bank_conflict_attempts_baseline": 0,
        "state_bank_conflict_rate_baseline": 0.0,
        "state_bank_conflicts_candidate": 0,
        "state_banks": STATE_BANKS,
        "pack_wait_accepted_events_p50": weighted_percentile(wait_hist, 50),
        "pack_wait_accepted_events_p95": weighted_percentile(wait_hist, 95),
        "pack_wait_accepted_events_p99": weighted_percentile(wait_hist, 99),
        "finite_slot_stall_cycles": 0,
        "finite_slot_stall_reason": (
            "offline destination-major schedule uses one live row; slot count "
            "does not model or hide descriptor-reorder construction"),
        "k1_resident_cycles": k1_resident_cycles,
        "k8_resident_cycles": k8_resident_cycles,
        "m484_signed_bundle_cycles": m484_cycles,
        "k1_to_m484_resource_scaling_speedup": k1_scaling,
        "baseline_cycles": k8_resident_cycles,
        "candidate_cycles": m484_cycles,
        "same_resource_cycle_speedup": speedup,
        "baseline_state_psum_rw_bits": resident_state_bits,
        "candidate_state_psum_rw_bits": resident_state_bits,
        "weight_bits_both_modes": weight_bits,
        "k8_resident_row_header_bits": resident_row_header_bits,
        "candidate_header_bits": header_bits,
        "candidate_event_metadata_bits": event_metadata_bits,
        "candidate_metadata_bits": metadata_bits,
        "candidate_padding_bits": padding_bits,
        "baseline_state_psum_plus_weight_bits": k8_total_bits,
        "candidate_state_psum_plus_weight_metadata_padding_bits": m484_total_bits,
        "traffic_reduction_fraction": traffic_reduction,
        "cycle_gate_1p20": speedup >= 1.20,
        "traffic_gate_30pct": traffic_reduction >= 0.30,
        "screen_gate": speedup >= 1.20 or traffic_reduction >= 0.30,
        "system_speedup_admitted": False,
        "rtl_authorized": False,
        "paper_performance_admitted": False,
    }


def online_original_order_point(category, offline_best):
    """Return a claim-safe original-layout point, never an inferred reorder."""
    if category["category"] == "FC1":
        # M51 FC1 is T,B,H,W,C in C order: all source-channel events of one
        # destination row are already adjacent.  No global reorder is needed.
        point = dict(offline_best)
        point["schedule"] = "online_original_C_order_exact"
        point["finite_slots"] = 1
        point["finite_slot_stall_reason"] = (
            "FC1 M51 C-last source channels are contiguous within each row")
        return point
    # M51 Conv is T,B,C,H,W.  Without a measured reorder frontend, the only
    # universally feasible lower bound is to emit every selected source as a
    # one-source packet.  Do not credit Conv->ATLIF fusion in this lower bound.
    point = dse_point(category, 1, 1)
    point["schedule"] = "online_original_NCHW_safe_lower_bound"
    point["bundle_width"] = int(offline_best["bundle_width"])
    point["bundles"] = point["selected_sources"]
    point["bundle_occupancy"] = 1.0 / int(offline_best["bundle_width"])
    point["mean_sources_per_bundle"] = 1.0
    point["padding_slots"] = (point["selected_sources"] *
                              (int(offline_best["bundle_width"]) - 1))
    online_padding_bits = 0
    for row in category["record_rows"]:
        source_bits = bits_required(row["source_width"])
        online_padding_bits += (row["selected_sources"] *
                                (int(offline_best["bundle_width"]) - 1) *
                                (source_bits + 1))
    point["candidate_padding_bits"] = online_padding_bits
    point["m484_signed_bundle_cycles"] = point["k8_resident_cycles"]
    point["candidate_cycles"] = point["baseline_cycles"]
    point["same_resource_cycle_speedup"] = 1.0
    point["candidate_state_psum_rw_bits"] = point["baseline_state_psum_rw_bits"]
    point["candidate_state_psum_plus_weight_metadata_padding_bits"] = (
        point["baseline_state_psum_plus_weight_bits"] +
        point["candidate_metadata_bits"] + point["candidate_padding_bits"])
    point["traffic_reduction_fraction"] = (
        1.0 - point["candidate_state_psum_plus_weight_metadata_padding_bits"] /
        point["baseline_state_psum_plus_weight_bits"])
    point["state_bank_conflict_attempts_baseline"] = 0
    point["state_bank_conflict_rate_baseline"] = 0.0
    point["state_bank_conflicts_candidate"] = 0
    point["pack_wait_accepted_events_p50"] = 0
    point["pack_wait_accepted_events_p95"] = 0
    point["pack_wait_accepted_events_p99"] = 0
    point["cycle_gate_1p20"] = False
    point["traffic_gate_30pct"] = False
    point["screen_gate"] = False
    point["finite_slot_stall_reason"] = (
        "safe lower bound emits one source per packet; no Conv NCHW reorder is credited")
    return point


def write_csv(path, rows):
    require(rows, "cannot write empty CSV")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--execution-trace", required=True, type=Path)
    parser.add_argument("--dual-line-trace", required=True, type=Path)
    parser.add_argument("--transactions", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    inputs = {
        "manifest": args.manifest,
        "execution_trace": args.execution_trace,
        "dual_line_trace": args.dual_line_trace,
        "transactions": args.transactions,
        "docs359": args.docs359,
    }
    for key, path in inputs.items():
        require(path.is_file(), "missing input {}".format(path))
        require(sha256(path) == EXPECTED[key], "SHA drift for {}".format(key))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    require(contract["schema"] ==
            "m484_row_coherent_signed_bundle_stationary_dse_contract_v1",
            "wrong M484 contract")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    require(manifest["status"] ==
            "PASS_EXACT_BINARY_INPUT_TRACE_NO_OUTPUT_OR_PERFORMANCE_CLAIM",
            "M51 manifest is not admitted exact binary capture")
    dual = load_dual_line(args.dual_line_trace)
    atlif_pairs, sequences, samples = load_atlif_pairs(args.execution_trace)
    require(len(sequences) == 1, "M484 r1 is intentionally single-sequence")

    per_records = []
    mismatch_rows = []
    selected_manifest_records = [
        row for row in manifest["records"]
        if (row["operator"] == "Conv2d" or
            (row["operator"] == "Linear" and ".mlp.fc1" in row["name"]))
    ]
    selected_payload_records = [
        row for row in selected_manifest_records
        if (args.payload_root / row["relative_path"]).is_file()
    ]
    missing_selected_records = [
        row for row in selected_manifest_records
        if not (args.payload_root / row["relative_path"]).is_file()
    ]
    require(len(selected_payload_records) == 160,
            "unexpected present M484 payload population")
    require(len(missing_selected_records) == 10 and all(
        row["name"].endswith("preds.3.conv.0")
        for row in missing_selected_records),
        "unexpected missing M484 payload population")
    transaction_width_audit = audit_acc32_transaction_widths(
        args.transactions, args.execution_trace, selected_payload_records,
        atlif_pairs)
    for index, record in enumerate(selected_payload_records, 1):
        value = decode_record(record, args.payload_root)
        selected = selected_row_counts(record, value)
        paired = (int(record["sample_id"]), record["name"]) in atlif_pairs
        metrics = per_record_metrics(
            record, selected["selected"], selected["positive"],
            selected["negative"], selected["motion"], paired)
        expected = dual[(int(record["sample_id"]), record["name"])]
        # The ordered producer uses the same row selector.  Positive/negative
        # trace columns describe the full transition stream, so only selected
        # source work/current count are equality oracles here.
        mismatch = {
            "sample_id": int(record["sample_id"]),
            "name": record["name"],
            "selected_source_delta": (metrics["selected_sources"] -
                                      expected["selected_sources"]),
            "current_source_delta": (selected["local_total"] -
                                     expected["current"]),
        }
        mismatch_rows.append(mismatch)
        require(mismatch["selected_source_delta"] == 0,
                "selected source mismatch {}".format(mismatch))
        require(mismatch["current_source_delta"] == 0,
                "current source mismatch {}".format(mismatch))
        metrics["local_sources"] = selected["local_total"]
        metrics["transition_sources"] = selected["transition_total"]
        per_records.append(metrics)
        if index % 20 == 0:
            print("processed {}/{} exact records".format(
                index, len(selected_payload_records)), flush=True)

    categories = {
        "Conv": aggregate_category(
            "Conv", [row for row in per_records if row["operator"] == "Conv2d"]),
        "Conv->ATLIF": aggregate_category(
            "Conv->ATLIF", [row for row in per_records
                             if row["operator"] == "Conv2d" and
                             row["conv_atlif_pair"]]),
        "FC1": aggregate_category(
            "FC1", [row for row in per_records
                    if row["operator"] == "Linear"]),
    }
    dse = [dse_point(category, width, slots)
           for category in categories.values()
           for width in WIDTHS for slots in SLOTS]
    best = {}
    for name in categories:
        points = [row for row in dse if row["category"] == name and
                  row["bundle_width"] == 8 and row["finite_slots"] == 1]
        require(len(points) == 1, "missing K8 decision point")
        best[name] = points[0]
    online_original = {
        name: online_original_order_point(categories[name], best[name])
        for name in categories
    }
    dse.extend(online_original.values())
    worst_window = {}
    for name, category in categories.items():
        width = int(best[name]["bundle_width"])
        sample_points = []
        for sample_id in samples:
            sample_records = [row for row in category["record_rows"]
                              if row["sample_id"] == sample_id]
            if not sample_records:
                continue
            point = dse_point(aggregate_category(name, sample_records), width, 1)
            sample_keys = sorted({row["sample_key"] for row in sample_records})
            require(len(sample_keys) == 1, "sample id/key ambiguity")
            sample_points.append({
                "sequence": sequences[0],
                "sample_id": sample_id,
                "sample_key": sample_keys[0],
                "bundle_width": width,
                "same_resource_cycle_speedup": point["same_resource_cycle_speedup"],
                "traffic_reduction_fraction": point["traffic_reduction_fraction"],
                "screen_gate": point["screen_gate"],
            })
        worst_window[name] = min(
            sample_points,
            key=lambda row: (row["same_resource_cycle_speedup"],
                             row["traffic_reduction_fraction"]))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dse_name = "m484_row_coherent_signed_bundle_stationary_dse.csv"
    record_name = "m484_per_record_exact_row_ledger.csv"
    mismatch_name = "m484_dual_line_reconciliation.csv"
    json_name = "m484_row_coherent_signed_bundle_stationary_dse.json"
    write_csv(output / dse_name, dse)
    public_records = [{key: value for key, value in row.items()
                       if key != "selected_histogram"}
                      for row in per_records]
    write_csv(output / record_name, public_records)
    write_csv(output / mismatch_name, mismatch_rows)

    result = {
        "schema": "m484_row_coherent_signed_bundle_stationary_dse_v1",
        "status": "NO_GO_VS_STRONG_K8_RESIDENT_BASELINE_NOT_ADMISSION",
        "identity": {
            "network": "H67",
            "checkpoint_path": manifest["run_context"]["checkpoint_path"],
            "checkpoint_sha256": manifest["run_context"]["checkpoint_sha256"],
            "sequences": sequences,
            "sequence_count": len(sequences),
            "samples": samples,
            "sample_count": len(samples),
            "single_sequence_limitation": True,
            "input_sha256": {key: EXPECTED[key] for key in inputs},
        },
        "population": {
            "selected_manifest_records": len(selected_manifest_records),
            "payload_records": len(selected_payload_records),
            "missing_selected_payload_records": len(missing_selected_records),
            "missing_selected_modules": sorted({
                row["name"] for row in missing_selected_records}),
            "conv_records": sum(row["operator"] == "Conv2d"
                                for row in per_records),
            "conv_atlif_pair_records": sum(
                row["operator"] == "Conv2d" and row["conv_atlif_pair"]
                for row in per_records),
            "fc1_records": sum(row["operator"] == "Linear"
                               for row in per_records),
            "dual_line_reconciliation_mismatches": sum(
                row["selected_source_delta"] != 0 or
                row["current_source_delta"] != 0
                for row in mismatch_rows),
        },
        "model": {
            "selector": "per-output-row min(current popcount, signed temporal-delta popcount); strict less selects motion",
            "baseline": "strong destination-major K8 baseline with whole Acc32 row resident from activation through commit",
            "candidate": "M484 fixed-width up-to-N signed bundle with the same resident Acc32 row",
            "same_resources": {
                "signed_source_lanes": "N in K8-resident and M484 modes",
                "reduction_tree": "identical N-input signed fold in K8-resident and M484 modes",
                "state_psum_ports": "identical resident vector activation/commit ports",
                "state_banks": STATE_BANKS,
                "accumulator_bits": ACC_BITS,
                "weight_bits": WEIGHT_BITS,
            },
            "cycle_formula": "K1 resident uses one source/cycle; K8 resident and M484 both use ceil(sources/N), plus identical row activate/commit and identical direct ATLIF handoff",
            "traffic_formula": "K8 resident and M484 both read/write Acc32 only at row boundaries and read identical INT8 weights/event metadata; M484 adds per-bundle header/padding",
            "pack_wait_unit": "accepted source events within one offline destination-major row, not wall-clock cycles",
            "finite_slot_scope": "offline destination-major consumer uses one live row; zero stalls do not prove an online reorder frontend",
            "transaction_width_audit": transaction_width_audit,
        },
        "gate": {
            "rule": "same-resource cycles >=1.20x OR state/psum+weight traffic reduction >=30%",
            "per_category_best": best,
            "per_category_online_original_order": online_original,
            "per_category_worst_window_at_best_N": worst_window,
            "all_categories_pass": all(row["screen_gate"] for row in best.values()),
            "all_online_original_order_pass": all(
                row["screen_gate"] for row in online_original.values()),
            "performance_admitted": False,
            "rtl_authorized": False,
        },
        "limitations": [
            "Only ten windows from one DSEC sequence (zurich_city_09_a).",
            "Ten preds.3.conv.0 records are listed by the M51 manifest but absent from this local handoff; Conv coverage is 60 records/six modules, not all seven manifest Conv modules.",
            "Offline destination-major order is evaluated; descriptor reorder construction, capacity and input backpressure are excluded.",
            "Cycles are an operator-local analytical schedule, not end-to-end latency, FPS or system speedup.",
            "Traffic uses logical bit counts, not CACTI SRAM macro energy or DRAMsim3 timing.",
            "Conv->ATLIF is a fused what-if only for four observed adjacent execution pairs; it is not RTL.",
            "K1-to-K8 uplift is a resource-scaling reference, not an M484 mechanism speedup.",
        ],
        "claim_boundary": {
            "system_speedup_admitted": False,
            "rtl_complete": False,
            "paper_ppa_ready": False,
            "may_claim": "exact-row offline opportunity screen on frozen H67 ep35 S10",
            "may_not_claim": "system speedup, online finite-buffer feasibility, synthesized performance, energy, or paper admission",
        },
        "files": {
            "dse_csv": dse_name,
            "per_record_csv": record_name,
            "reconciliation_csv": mismatch_name,
        },
    }
    (output / json_name).write_text(json.dumps(result, indent=2) + "\n",
                                    encoding="utf-8")
    readme = "# M484 row-coherent signed-bundle/state-stationary DSE\n\n"
    readme += "Status: **NO-GO versus the strong same-resource K8-resident baseline; no RTL or performance admission**.\n\n"
    readme += "All 160 locally present target records reconcile to the frozen dual-line selected-source ledger. "
    readme += "The population is ten windows from only `zurich_city_09_a`.\n\n"
    readme += "| Category | Decision N | K1->K8 resource scaling | M484 vs K8 cycles | M484 vs K8 traffic | Gate |\n|---|---:|---:|---:|---:|---|\n"
    for name in ("Conv", "Conv->ATLIF", "FC1"):
        row = best[name]
        readme += "| {} | {} | {:.4f}x | {:.4f}x | {:.2%} | {} |\n".format(
            name, row["bundle_width"],
            row["k1_to_m484_resource_scaling_speedup"],
            row["same_resource_cycle_speedup"],
            row["traffic_reduction_fraction"],
            "PASS" if row["screen_gate"] else "FAIL")
    readme += "\nWorst window at each category's best N:\n\n"
    for name in ("Conv", "Conv->ATLIF", "FC1"):
        row = worst_window[name]
        readme += "- {}: {} sample {} (`{}`), {:.4f}x, traffic reduction {:.2%}.\n".format(
            name, row["sequence"], row["sample_id"], row["sample_key"],
            row["same_resource_cycle_speedup"],
            row["traffic_reduction_fraction"])
    readme += "\nOnline original-order boundary:\n\n"
    for name in ("Conv", "Conv->ATLIF", "FC1"):
        row = online_original[name]
        readme += "- {}: `{}`, {:.4f}x, traffic reduction {:.2%}, gate {}.\n".format(
            name, row["schedule"], row["same_resource_cycle_speedup"],
            row["traffic_reduction_fraction"],
            "PASS" if row["screen_gate"] else "NO-GO")
    readme += "\nZero finite-slot stalls are structural to the offline destination-major schedule; they are not evidence for an online reorder frontend. Pack wait is measured in accepted events, not wall-clock cycles.\n"
    (output / "README.md").write_text(readme, encoding="utf-8")

    produced = [dse_name, record_name, mismatch_name, json_name, "README.md"]
    manifest_out = {
        "schema": "m484_row_coherent_signed_bundle_stationary_manifest_v1",
        "status": result["status"],
        "files": {name: sha256(output / name) for name in produced},
        "input_sha256": {key: EXPECTED[key] for key in inputs},
        "contract_sha256": sha256(args.contract),
        "script_sha256": sha256(Path(__file__)),
        "docs359_unchanged": sha256(args.docs359) == EXPECTED["docs359"],
    }
    manifest_name = "m484_manifest.json"
    (output / manifest_name).write_text(json.dumps(manifest_out, indent=2) + "\n",
                                        encoding="utf-8")
    produced.append(manifest_name)
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for name in sorted(produced):
            handle.write("{}  {}\n".format(sha256(output / name), name))
    seal = sha256(output / "SHA256SUMS")
    (output / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(seal), encoding="utf-8")
    (output / "RUN_COMPLETE.txt").write_text(
        "NO_GO M484 versus strong K8-resident baseline; no RTL/performance admission\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "best": {name: {
            "N": row["bundle_width"],
            "cycles": row["same_resource_cycle_speedup"],
            "traffic_reduction": row["traffic_reduction_fraction"],
            "gate": row["screen_gate"],
        } for name, row in best.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
