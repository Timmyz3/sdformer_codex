#!/usr/bin/env python3
"""M507 r4: one-shot same-resource cycle fast-kill for ExSpike APEC-G2.

Both arms are the same APEC-capable engine and reserve the same SRAM and
ports.  The baseline disables compression.  Both arms use one identical
96-lane block-accumulation dataflow: the first product initializes a block,
subsequent products accumulate locally, and each completed block is materialized
once into a banked destination slot.  Candidate common blocks use the same lane
accumulator, are serialized through a 128-B/cycle overlap scratch, and seed the
two residual destinations through synchronous block reads.  No arm uses a
per-event destination-SRAM RMW shortcut.  Logical/physical bytes, final slot
readout, weight/output banks, queue backpressure, border taps, and synchronous
tails are explicitly accounted.

This is a standalone Conv cycle model, not a full-network simulator and not a
novelty claim: APEC is direct ExSpike prior art.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def sequence_key(sample_key):
    match = re.match(r"^(.*)_([0-9]+)$", Path(sample_key).stem)
    require(match is not None, "sample key has no numeric suffix: " + sample_key)
    return match.group(1)


def load_record(manifest_dir, record):
    payload = manifest_dir / record["value_payload_file"]
    require(payload.is_file(), "missing payload: " + str(payload))
    require(sha256(payload) == record["value_payload_sha256"],
            "compressed payload SHA drift: " + str(payload))
    require(payload.stat().st_size == int(record["value_payload_compressed_bytes"]),
            "compressed payload byte drift: " + str(payload))
    raw = zlib.decompress(payload.read_bytes())
    require(len(raw) == int(record["input_content_bytes"]),
            "decoded payload byte drift: " + str(payload))
    require(hashlib.sha256(raw).hexdigest() == record["input_content_sha256"],
            "decoded payload SHA drift: " + str(payload))
    require(record["input_dtype"] == "float32", "M507 requires float32")
    shape = tuple(int(value) for value in record["shape"])
    values = np.frombuffer(raw, dtype="<f4")
    require(values.size == integer_product(shape), "decoded element drift")
    values = values.reshape(shape)
    require(bool(np.isfinite(values).all()), "NaN/Inf in M507 input")
    require(int(np.count_nonzero(values)) == int(record["nonzero_count"]),
            "nonzero-count drift")
    require(int(np.count_nonzero(values < np.float32(0.0))) ==
            int(record["negative_count"]), "negative-count drift")
    return values


def taps_for_position(y, x, height, width):
    vertical = 2 if y in (0, height - 1) else 3
    horizontal = 2 if x in (0, width - 1) else 3
    return vertical * horizontal


def union_taps_for_horizontal_pair(y, x0, x1, height, width):
    require(x1 == x0 + 1, "M507 only supports adjacent horizontal G2")
    vertical = 2 if y in (0, height - 1) else 3
    horizontal_offsets = set()
    for x in (x0, x1):
        for delta in (-1, 0, 1):
            if 0 <= x + delta < width:
                horizontal_offsets.add(delta)
    # Pairing is by source position.  The stored vector is indexed by kernel
    # offset, so the union cannot exceed the three physical horizontal taps.
    return vertical * len(horizontal_offsets)


def service_terms(event_count, taps, model):
    output_channels = int(model["output_channels"])
    lanes = int(model["compute_lanes"])
    weight_bw = int(model["weight_bytes_per_cycle"])
    weight_banks = int(model["weight_banks"])
    weight_bank_bw = int(model["weight_bank_bytes_per_cycle"])
    weight_bits = int(model["weight_bits"])
    require(weight_bits == 8, "M507 frozen model requires INT8 weights")
    operations = int(event_count) * output_channels * int(taps)
    compute_cycles = int(event_count) * int(math.ceil(
        output_channels * int(taps) / float(lanes)))
    weight_bytes = int(event_count) * output_channels * int(taps)
    weight_cycles = int(math.ceil(weight_bytes / float(weight_bw)))
    require(output_channels % weight_banks == 0,
            "output channels do not balance over weight banks")
    per_bank_bytes = (int(event_count) * int(taps) *
                      (output_channels // weight_banks))
    per_bank_cycles = int(math.ceil(per_bank_bytes / float(weight_bank_bw)))
    require(weight_bw == weight_banks * weight_bank_bw,
            "aggregate/per-bank weight bandwidth drift")
    require(weight_cycles == per_bank_cycles,
            "weight-bank mapping introduces an unmodeled conflict")
    return {
        "events": int(event_count),
        "operations": operations,
        "compute_cycles": compute_cycles,
        "weight_bytes": weight_bytes,
        "weight_transactions": int(math.ceil(weight_bytes /
                                             float(weight_bw))),
        "weight_cycles": weight_cycles,
        "weight_bank_conflict_cycles": 0,
    }


def add_terms(dst, src):
    for key in ("events", "operations", "compute_cycles", "weight_bytes",
                "weight_transactions", "weight_cycles",
                "weight_bank_conflict_cycles"):
        dst[key] += int(src[key])


def empty_terms():
    return {key: 0 for key in (
        "events", "operations", "compute_cycles", "weight_bytes",
        "weight_transactions", "weight_cycles",
        "weight_bank_conflict_cycles")}


def destination_slot_terms(taps, model):
    """One vector access through one explicitly banked destination slot."""
    output_channels = int(model["output_channels"])
    accumulator_bits = int(model["accumulator_bits"])
    banks = int(model["destination_slot_banks"])
    bank_bw = int(model["destination_slot_bank_bytes_per_cycle"])
    lanes = int(model["compute_lanes"])
    require(output_channels % banks == 0 and lanes % banks == 0,
            "output/lane values do not balance over destination banks")
    blocks = int(math.ceil(output_channels * int(taps) / float(lanes)))
    per_bank_block_bytes = int(math.ceil(
        (lanes // banks) * accumulator_bits / 8.0))
    require(per_bank_block_bytes <= bank_bw,
            "one lane block exceeds a destination bank word")
    cycles = blocks
    total_bytes = int(math.ceil(
        output_channels * int(taps) * accumulator_bits / 8.0))
    return {
        "bytes": total_bytes,
        "physical_bytes": cycles * banks * bank_bw,
        "cycles": cycles,
        "transactions": cycles * banks,
        "bank_conflict_cycles": 0,
    }



def vector_transfer_terms(taps, model):
    """Final slot readout pipelined into the common output sink."""
    slot = destination_slot_terms(taps, model)
    output_channels = int(model["output_channels"])
    accumulator_bits = int(model["accumulator_bits"])
    banks = int(model["output_banks"])
    bank_bw = int(model["output_bank_bytes_per_cycle"])
    require(output_channels % banks == 0,
            "output channels do not balance over output banks")
    total_bytes = int(math.ceil(
        output_channels * int(taps) * accumulator_bits / 8.0))
    sink_cycles = int(math.ceil(total_bytes / float(banks * bank_bw)))
    sink_transactions = sink_cycles * banks
    read_tail = int(model["destination_slot_sync_read_latency_cycles"])
    # Slot read and sink write are one streaming commit phase.  The slower
    # side plus the synchronous first-response latency determines completion.
    cycles = max(slot["cycles"], sink_cycles) + read_tail
    return {
        "bytes": total_bytes,
        "cycles": cycles,
        "transactions": sink_transactions,
        "slot_read_bytes": slot["bytes"],
        "slot_read_physical_bytes": slot["physical_bytes"],
        "slot_read_cycles": slot["cycles"] + read_tail,
        "slot_read_transactions": slot["transactions"],
        "sink_write_bytes": total_bytes,
        "sink_write_cycles": sink_cycles,
        "sink_write_transactions": sink_transactions,
        "bank_conflict_cycles": 0,
    }


def destination_block_write_terms(taps, model):
    """Materialize one completed vector into a striped destination slot."""
    taps = int(taps)
    lanes = int(model["compute_lanes"])
    output_channels = int(model["output_channels"])
    accumulator_bits = int(model["accumulator_bits"])
    banks = int(model["destination_slot_banks"])
    bank_bw = int(model["destination_slot_bank_bytes_per_cycle"])
    values = output_channels * taps
    blocks = int(math.ceil(values / float(lanes)))
    logical_bytes = int(math.ceil(values * accumulator_bits / 8.0))
    physical_bytes = 0
    write_cycles = 0
    remaining = values
    for _ in range(blocks):
        block_values = min(lanes, remaining)
        require(block_values % banks == 0,
                "lane block does not stripe evenly across destination banks")
        per_bank_bytes = int(math.ceil(
            (block_values // banks) * accumulator_bits / 8.0))
        cycles = int(math.ceil(per_bank_bytes / float(bank_bw)))
        require(cycles == 1,
                "destination bank cannot accept one completed lane block")
        write_cycles += cycles
        physical_bytes += cycles * banks * bank_bw
        remaining -= block_values
    require(remaining == 0, "lane block coverage drift")
    return {
        "blocks": blocks,
        "destination_write_logical_bytes": logical_bytes,
        "destination_write_physical_bytes": physical_bytes,
        "destination_write_cycles": write_cycles,
        "destination_write_transactions": write_cycles * banks,
        "bank_conflict_cycles": 0,
    }


def lane_block_terms(event_count, taps, model):
    """One stream through the sole time-multiplexed lane accumulator.

    A block contains at most ``compute_lanes`` output-tap values.  For a
    nonempty stream, its first product initializes the lane register in the
    normal MAC cycle; no zero-fill or destination SRAM read is required.
    """
    event_count = int(event_count)
    write = destination_block_write_terms(taps, model)
    service = service_terms(event_count, taps, model)
    require(service["compute_cycles"] == event_count * write["blocks"],
            "event-major and block-major compute cycles disagree")
    return {
        "events": event_count,
        "blocks": write["blocks"] if event_count else 0,
        "service": service,
        "first_product_seed_blocks": write["blocks"] if event_count else 0,
    }


def scratch_block_terms(taps, model):
    """Block-serialized scratch traffic for one exact common vector."""
    lanes = int(model["compute_lanes"])
    output_channels = int(model["output_channels"])
    accumulator_bits = int(model["accumulator_bits"])
    scratch_bw = int(model["scratch_bytes_per_cycle"])
    values = output_channels * int(taps)
    blocks = int(math.ceil(values / float(lanes)))
    logical_bytes = int(math.ceil(values * accumulator_bits / 8.0))
    physical_bytes = 0
    transfer_cycles = 0
    remaining = values
    for _ in range(blocks):
        block_values = min(lanes, remaining)
        block_bytes = int(math.ceil(block_values * accumulator_bits / 8.0))
        cycles = int(math.ceil(block_bytes / float(scratch_bw)))
        transfer_cycles += cycles
        physical_bytes += cycles * scratch_bw
        remaining -= block_values
    require(remaining == 0, "scratch block coverage drift")
    require(logical_bytes <= int(model["reserved_overlap_scratch_bytes"]),
            "common vector exceeds reserved overlap scratch")
    read_latency = int(model["overlap_scratch_sync_read_latency_cycles"])
    return {
        "blocks": blocks,
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "write_cycles": transfer_cycles,
        "write_transactions": transfer_cycles,
        "one_read_cycles": transfer_cycles + blocks * read_latency,
        "one_read_transactions": transfer_cycles,
        "one_read_sync_tail_cycles": blocks * read_latency,
    }


def build_resource_ledger(model):
    """Derive equal physical storage/ports for both execution modes."""
    total = int(model["common_total_sram_bytes"])
    channels = int(model["input_channels"])
    output_channels = int(model["output_channels"])
    acc_bits = int(model["accumulator_bits"])
    kernel_elements = integer_product(model["kernel"])
    bitmap_pair = int(math.ceil(2 * channels / 8.0))
    full_blocks = int(math.ceil(
        output_channels * kernel_elements / float(model["compute_lanes"])))
    scratch_cycles_per_block = int(math.ceil(
        int(model["compute_lanes"]) * acc_bits / 8.0 /
        int(model["scratch_bytes_per_cycle"])))
    overlap = (full_blocks * scratch_cycles_per_block *
               int(model["scratch_bytes_per_cycle"]))
    one_destination = (
        full_blocks * int(model["destination_slot_banks"]) *
        int(model["destination_slot_bank_bytes_per_cycle"]))
    destination = 2 * one_destination
    payload = total - bitmap_pair - overlap - destination
    require(payload >= 0, "M507 SRAM component ledger exceeds total capacity")
    capacity = {
        "pair_bitmap_bytes": bitmap_pair,
        "overlap_cache_bytes": overlap,
        "two_destination_vector_slots_bytes": destination,
        "payload_and_weight_window_bytes": payload,
    }
    require(sum(capacity.values()) == total,
            "M507 SRAM capacity does not conserve")
    ports = {
        "pair_bitmap": {
            "read_ports": 1,
            "write_ports": 0,
            "bytes_per_cycle": int(model["bitmap_bytes_per_cycle"]),
        },
        "overlap_cache": {
            "read_ports": 1,
            "write_ports": 1,
            "bytes_per_cycle": int(model["scratch_bytes_per_cycle"]),
            "synchronous_read_latency_cycles":
                int(model["overlap_scratch_sync_read_latency_cycles"]),
        },
        "destination_slots": {
            "slots": int(model["destination_slot_count"]),
            "banks_per_slot": int(model["destination_slot_banks"]),
            "read_ports_per_bank": 1,
            "write_ports_per_bank": 1,
            "bytes_per_bank_per_cycle":
                int(model["destination_slot_bank_bytes_per_cycle"]),
            "synchronous_read_latency_cycles":
                int(model["destination_slot_sync_read_latency_cycles"]),
            "completed_block_write": True,
            "event_read_modify_write": False,
            "final_readout": True,
        },
        "weight": {
            "read_banks": int(model["weight_banks"]),
            "write_banks": 0,
            "bytes_per_bank_per_cycle":
                int(model["weight_bank_bytes_per_cycle"]),
        },
        "output_sink": {
            "read_banks": 0,
            "write_banks": int(model["output_banks"]),
            "bytes_per_bank_per_cycle":
                int(model["output_bank_bytes_per_cycle"]),
        },
        "compute": {
            "accumulator_lanes": int(model["compute_lanes"]),
            "accumulator_bits": acc_bits,
            "lane_accumulator_register_bits":
                int(model["compute_lanes"]) * acc_bits,
            "instances": 1,
            "time_multiplexed_across_baseline_residual_and_common": True,
            "block_mapping": "contiguous output-tap values",
            "bank_mapping": "output_channel_mod_destination_slot_banks",
            "first_product_initializes_nonempty_block": True,
        },
        "group_queue": {
            "entries": int(model["group_queue_entries"]),
            "ready_valid_backpressure": True,
        },
    }
    require(bitmap_pair == int(model["pair_bitmap_buffer_bytes"]),
            "pair bitmap capacity drift")
    require(overlap == int(model["reserved_overlap_scratch_bytes"]),
            "overlap-cache capacity drift")
    require(destination == int(model["destination_vector_slots_bytes"]),
            "destination-vector capacity drift")
    require(int(model["destination_slot_count"]) == 2,
            "M507 requires exactly two destination slots")
    require(int(model["destination_slot_banks"]) ==
            int(model["output_banks"]),
            "destination/output bank-count drift")
    require(payload == int(model["payload_and_weight_window_bytes"]),
            "payload/weight window capacity drift")
    # The same physically derived ledger is instantiated in both modes.  The
    # baseline clock-gates, rather than reallocates, the overlap resources.
    return {
        "total_sram_bytes": total,
        "baseline": {"capacity": dict(capacity), "ports": dict(ports)},
        "candidate": {"capacity": dict(capacity), "ports": dict(ports)},
    }


def record_cycles(values, model):
    require(values.ndim == 5 and values.shape[1] == 1,
            "M507 requires T,B=1,C,H,W")
    time_steps, _, channels, height, width = values.shape
    require(channels == int(model["input_channels"]), "input-channel drift")
    require(width % 2 == 0, "M507 G2 requires even width")
    active = values != np.float32(0.0)
    left = active[..., 0::2]
    right = active[..., 1::2]
    overlap = left & right

    baseline = empty_terms()
    candidate = empty_terms()
    counters = {key: 0 for key in (
        "groups", "nonempty_groups", "overlap_groups", "baseline_events",
        "candidate_events", "exact_overlap_events",
        "bitmap_read_cycles_baseline", "bitmap_read_cycles_candidate",
        "compare_cycles", "weight_startup_cycles_baseline",
        "weight_startup_cycles_candidate",
        "weight_not_ready_stall_baseline",
        "weight_not_ready_stall_candidate",
        "scratch_write_transactions", "scratch_read_transactions",
        "scratch_write_logical_bytes", "scratch_write_physical_bytes",
        "scratch_read_logical_bytes", "scratch_read_physical_bytes",
        "scratch_serialization_stall_cycles",
        "scratch_sync_read_tail_cycles",
        "lane_first_product_seed_blocks_baseline",
        "lane_first_product_seed_blocks_candidate",
        "lane_scratch_seed_blocks_candidate",
        "destination_product_seed_outputs_baseline",
        "destination_product_seed_outputs_candidate",
        "destination_scratch_seed_outputs_candidate",
        "destination_initialization_mismatch_count",
        "destination_zero_init_bytes_baseline",
        "destination_zero_init_bytes_candidate",
        "destination_zero_init_cycles_baseline",
        "destination_zero_init_cycles_candidate",
        "destination_seed_write_bytes_candidate",
        "destination_seed_write_cycles_candidate",
        "destination_rmw_read_bytes_baseline",
        "destination_rmw_read_bytes_candidate",
        "destination_rmw_write_bytes_baseline",
        "destination_rmw_write_bytes_candidate",
        "destination_rmw_sync_tail_cycles_baseline",
        "destination_rmw_sync_tail_cycles_candidate",
        "destination_materialize_logical_bytes_baseline",
        "destination_materialize_logical_bytes_candidate",
        "destination_materialize_physical_bytes_baseline",
        "destination_materialize_physical_bytes_candidate",
        "destination_materialize_cycles_baseline",
        "destination_materialize_cycles_candidate",
        "destination_materialize_transactions_baseline",
        "destination_materialize_transactions_candidate",
        "destination_final_read_bytes_baseline",
        "destination_final_read_bytes_candidate",
        "destination_final_read_physical_bytes_baseline",
        "destination_final_read_physical_bytes_candidate",
        "destination_final_read_cycles_baseline",
        "destination_final_read_cycles_candidate",
        "destination_final_read_transactions_baseline",
        "destination_final_read_transactions_candidate",
        "output_sink_write_bytes_baseline",
        "output_sink_write_bytes_candidate",
        "output_sink_write_cycles_baseline",
        "output_sink_write_cycles_candidate",
        "output_sink_write_transactions_baseline",
        "output_sink_write_transactions_candidate",
        "destination_commit_cycles_baseline",
        "destination_commit_cycles_candidate",
        "destination_commit_transactions_baseline",
        "destination_commit_transactions_candidate",
        "weight_bank_conflict_cycles_baseline",
        "weight_bank_conflict_cycles_candidate",
        "output_bank_conflict_cycles_baseline",
        "output_bank_conflict_cycles_candidate",
        "destination_bank_conflict_cycles_baseline",
        "destination_bank_conflict_cycles_candidate",
        "group_queue_backpressure_cycles_baseline",
        "group_queue_backpressure_cycles_candidate",
        "group_queue_max_occupancy_baseline",
        "group_queue_max_occupancy_candidate",
        "bank_mapping_mismatch_count", "lane_issue_order_mismatch_count",
        "scratch_conservation_mismatch_count", "border_groups")}

    total_baseline_cycles = 0
    total_candidate_cycles = 0
    bitmap_read_cycles = int(model["bitmap_pair_read_cycles"])
    compare_cycles = int(model["exact_compare_cycles"])
    startup = int(model["weight_startup_latency_cycles"])
    destination_banks = int(model["destination_slot_banks"])
    destination_bank_bw = int(
        model["destination_slot_bank_bytes_per_cycle"])

    def empty_write():
        return {
            "blocks": 0,
            "destination_write_logical_bytes": 0,
            "destination_write_physical_bytes": 0,
            "destination_write_cycles": 0,
            "destination_write_transactions": 0,
            "bank_conflict_cycles": 0,
        }

    def empty_commit():
        return {
            "cycles": 0, "transactions": 0, "slot_read_bytes": 0,
            "slot_read_physical_bytes": 0,
            "slot_read_cycles": 0, "slot_read_transactions": 0,
            "sink_write_bytes": 0, "sink_write_cycles": 0,
            "sink_write_transactions": 0, "bank_conflict_cycles": 0,
        }

    for y in range(height):
        for pair in range(width // 2):
            x0 = pair * 2
            x1 = x0 + 1
            taps0 = taps_for_position(y, x0, height, width)
            taps1 = taps_for_position(y, x1, height, width)
            union_taps = union_taps_for_horizontal_pair(
                y, x0, x1, height, width)
            require(union_taps >= max(taps0, taps1),
                    "union-tap undercount")
            if taps0 != 9 or taps1 != 9:
                counters["border_groups"] += time_steps

            e0 = np.count_nonzero(left[:, 0, :, y, pair], axis=1)
            e1 = np.count_nonzero(right[:, 0, :, y, pair], axis=1)
            ov = np.count_nonzero(overlap[:, 0, :, y, pair], axis=1)
            for timestep in range(time_steps):
                count0 = int(e0[timestep])
                count1 = int(e1[timestep])
                common = int(ov[timestep])
                require(0 <= common <= min(count0, count1),
                        "overlap population invalid")
                live_outputs = int(count0 > 0) + int(count1 > 0)
                counters["groups"] += 1
                counters["nonempty_groups"] += int(live_outputs > 0)
                counters["overlap_groups"] += int(common > 0)

                # Baseline: two independent streams use the same sole lane
                # accumulator.  First products initialize blocks; completed
                # blocks are written once to the banked destination slots.
                b0 = lane_block_terms(count0, taps0, model)
                b1 = lane_block_terms(count1, taps1, model)
                bgroup = empty_terms()
                add_terms(bgroup, b0["service"])
                add_terms(bgroup, b1["service"])
                add_terms(baseline, bgroup)
                bw0 = destination_block_write_terms(taps0, model) \
                    if count0 else empty_write()
                bw1 = destination_block_write_terms(taps1, model) \
                    if count1 else empty_write()
                bstartup = startup if bgroup["events"] else 0
                bexec = max(bgroup["compute_cycles"],
                            bgroup["weight_cycles"] + bstartup)
                bmaterialize = int(
                    bw0["destination_write_cycles"] +
                    bw1["destination_write_cycles"])
                bout0 = vector_transfer_terms(taps0, model) \
                    if count0 else empty_commit()
                bout1 = vector_transfer_terms(taps1, model) \
                    if count1 else empty_commit()
                bcommit = int(bout0["cycles"] + bout1["cycles"])
                bgroup_cycles = (bitmap_read_cycles + bexec +
                                 bmaterialize + bcommit)
                total_baseline_cycles += bgroup_cycles

                counters["weight_startup_cycles_baseline"] += bstartup
                counters["weight_not_ready_stall_baseline"] += max(
                    0, bgroup["weight_cycles"] + bstartup -
                    bgroup["compute_cycles"])
                counters["weight_bank_conflict_cycles_baseline"] += int(
                    bgroup["weight_bank_conflict_cycles"])
                counters["lane_first_product_seed_blocks_baseline"] += int(
                    b0["first_product_seed_blocks"] +
                    b1["first_product_seed_blocks"])
                baseline_product_seed_outputs = (
                    int(count0 > 0) + int(count1 > 0))
                counters["destination_product_seed_outputs_baseline"] += int(
                    baseline_product_seed_outputs)
                counters["destination_materialize_logical_bytes_baseline"] += int(
                    bw0["destination_write_logical_bytes"] +
                    bw1["destination_write_logical_bytes"])
                counters["destination_materialize_physical_bytes_baseline"] += int(
                    bw0["destination_write_physical_bytes"] +
                    bw1["destination_write_physical_bytes"])
                counters["destination_materialize_cycles_baseline"] += (
                    bmaterialize)
                counters["destination_materialize_transactions_baseline"] += int(
                    bw0["destination_write_transactions"] +
                    bw1["destination_write_transactions"])
                counters["destination_final_read_bytes_baseline"] += int(
                    bout0["slot_read_bytes"] + bout1["slot_read_bytes"])
                counters["destination_final_read_physical_bytes_baseline"] += int(
                    bout0["slot_read_physical_bytes"] +
                    bout1["slot_read_physical_bytes"])
                counters["destination_final_read_cycles_baseline"] += int(
                    bout0["slot_read_cycles"] + bout1["slot_read_cycles"])
                counters["destination_final_read_transactions_baseline"] += int(
                    bout0["slot_read_transactions"] +
                    bout1["slot_read_transactions"])
                counters["output_sink_write_bytes_baseline"] += int(
                    bout0["sink_write_bytes"] + bout1["sink_write_bytes"])
                counters["output_sink_write_cycles_baseline"] += int(
                    bout0["sink_write_cycles"] + bout1["sink_write_cycles"])
                counters["output_sink_write_transactions_baseline"] += int(
                    bout0["sink_write_transactions"] +
                    bout1["sink_write_transactions"])
                counters["destination_commit_cycles_baseline"] += bcommit
                counters["destination_commit_transactions_baseline"] += int(
                    bout0["slot_read_transactions"] +
                    bout1["slot_read_transactions"] +
                    bout0["sink_write_transactions"] +
                    bout1["sink_write_transactions"])

                # Candidate: common, left residual and right residual use the
                # same lane accumulator in serialized phases.  Common blocks
                # are drained to scratch, then synchronously re-read to seed
                # each live destination before residual accumulation.
                r0_count = count0 - common
                r1_count = count1 - common
                cres0 = lane_block_terms(r0_count, taps0, model)
                cres1 = lane_block_terms(r1_count, taps1, model)
                ccommon = lane_block_terms(common, union_taps, model)
                cgroup = empty_terms()
                add_terms(cgroup, cres0["service"])
                add_terms(cgroup, cres1["service"])
                add_terms(cgroup, ccommon["service"])
                add_terms(candidate, cgroup)
                cstartup = startup if cgroup["events"] else 0
                cexec = max(cgroup["compute_cycles"],
                            cgroup["weight_cycles"] + cstartup)

                cw0 = destination_block_write_terms(taps0, model) \
                    if count0 else empty_write()
                cw1 = destination_block_write_terms(taps1, model) \
                    if count1 else empty_write()
                cmaterialize = int(
                    cw0["destination_write_cycles"] +
                    cw1["destination_write_cycles"])
                scratch_store = 0
                scratch_reads = 0
                scratch = None
                if common:
                    scratch = scratch_block_terms(union_taps, model)
                    require(ccommon["blocks"] == scratch["blocks"],
                            "common compute/scratch block-count drift")
                    scratch_store = int(scratch["write_cycles"])
                    scratch_reads = 2 * int(scratch["one_read_cycles"])
                    counters["scratch_write_transactions"] += int(
                        scratch["write_transactions"])
                    counters["scratch_read_transactions"] += 2 * int(
                        scratch["one_read_transactions"])
                    counters["scratch_write_logical_bytes"] += int(
                        scratch["logical_bytes"])
                    counters["scratch_write_physical_bytes"] += int(
                        scratch["physical_bytes"])
                    counters["scratch_read_logical_bytes"] += 2 * int(
                        scratch["logical_bytes"])
                    counters["scratch_read_physical_bytes"] += 2 * int(
                        scratch["physical_bytes"])
                    counters["scratch_sync_read_tail_cycles"] += 2 * int(
                        scratch["one_read_sync_tail_cycles"])
                    counters["lane_scratch_seed_blocks_candidate"] += int(
                        cw0["blocks"] + cw1["blocks"])
                    candidate_scratch_seed_outputs = (
                        int(count0 > 0) + int(count1 > 0))
                    candidate_product_seed_outputs = 0
                    counters["destination_scratch_seed_outputs_candidate"] += int(
                        candidate_scratch_seed_outputs)
                    counters["lane_first_product_seed_blocks_candidate"] += int(
                        ccommon["first_product_seed_blocks"])
                else:
                    candidate_scratch_seed_outputs = 0
                    candidate_product_seed_outputs = (
                        int(r0_count > 0) + int(r1_count > 0))
                    counters["lane_first_product_seed_blocks_candidate"] += int(
                        cres0["first_product_seed_blocks"] +
                        cres1["first_product_seed_blocks"])
                    counters["destination_product_seed_outputs_candidate"] += int(
                        candidate_product_seed_outputs)

                cout0 = vector_transfer_terms(taps0, model) \
                    if count0 else empty_commit()
                cout1 = vector_transfer_terms(taps1, model) \
                    if count1 else empty_commit()
                ccommit = int(cout0["cycles"] + cout1["cycles"])
                candidate_extra = scratch_store + scratch_reads + cmaterialize
                cgroup_cycles = (bitmap_read_cycles + compare_cycles + cexec +
                                 candidate_extra + ccommit)
                total_candidate_cycles += cgroup_cycles

                counters["weight_startup_cycles_candidate"] += cstartup
                counters["weight_not_ready_stall_candidate"] += max(
                    0, cgroup["weight_cycles"] + cstartup -
                    cgroup["compute_cycles"])
                counters["weight_bank_conflict_cycles_candidate"] += int(
                    cgroup["weight_bank_conflict_cycles"])
                counters["scratch_serialization_stall_cycles"] += int(
                    scratch_store + scratch_reads)
                counters["destination_materialize_logical_bytes_candidate"] += int(
                    cw0["destination_write_logical_bytes"] +
                    cw1["destination_write_logical_bytes"])
                counters["destination_materialize_physical_bytes_candidate"] += int(
                    cw0["destination_write_physical_bytes"] +
                    cw1["destination_write_physical_bytes"])
                counters["destination_materialize_cycles_candidate"] += (
                    cmaterialize)
                counters["destination_materialize_transactions_candidate"] += int(
                    cw0["destination_write_transactions"] +
                    cw1["destination_write_transactions"])
                counters["destination_final_read_bytes_candidate"] += int(
                    cout0["slot_read_bytes"] + cout1["slot_read_bytes"])
                counters["destination_final_read_physical_bytes_candidate"] += int(
                    cout0["slot_read_physical_bytes"] +
                    cout1["slot_read_physical_bytes"])
                counters["destination_final_read_cycles_candidate"] += int(
                    cout0["slot_read_cycles"] + cout1["slot_read_cycles"])
                counters["destination_final_read_transactions_candidate"] += int(
                    cout0["slot_read_transactions"] +
                    cout1["slot_read_transactions"])
                counters["output_sink_write_bytes_candidate"] += int(
                    cout0["sink_write_bytes"] + cout1["sink_write_bytes"])
                counters["output_sink_write_cycles_candidate"] += int(
                    cout0["sink_write_cycles"] + cout1["sink_write_cycles"])
                counters["output_sink_write_transactions_candidate"] += int(
                    cout0["sink_write_transactions"] +
                    cout1["sink_write_transactions"])
                counters["destination_commit_cycles_candidate"] += ccommit
                counters["destination_commit_transactions_candidate"] += int(
                    cout0["slot_read_transactions"] +
                    cout1["slot_read_transactions"] +
                    cout0["sink_write_transactions"] +
                    cout1["sink_write_transactions"])

                initialized_baseline = baseline_product_seed_outputs
                initialized_candidate = (
                    candidate_product_seed_outputs +
                    candidate_scratch_seed_outputs)
                if initialized_baseline != live_outputs or \
                        initialized_candidate != live_outputs:
                    counters["destination_initialization_mismatch_count"] += 1
                if common and scratch is not None:
                    if (scratch["logical_bytes"] <= 0 or
                            scratch["physical_bytes"] <
                            scratch["logical_bytes"]):
                        counters["scratch_conservation_mismatch_count"] += 1

                counters["output_bank_conflict_cycles_baseline"] += int(
                    bout0["bank_conflict_cycles"] +
                    bout1["bank_conflict_cycles"])
                counters["output_bank_conflict_cycles_candidate"] += int(
                    cout0["bank_conflict_cycles"] +
                    cout1["bank_conflict_cycles"])
                counters["destination_bank_conflict_cycles_baseline"] += int(
                    bw0["bank_conflict_cycles"] +
                    bw1["bank_conflict_cycles"] +
                    bout0["bank_conflict_cycles"] +
                    bout1["bank_conflict_cycles"])
                counters["destination_bank_conflict_cycles_candidate"] += int(
                    cw0["bank_conflict_cycles"] +
                    cw1["bank_conflict_cycles"] +
                    cout0["bank_conflict_cycles"] +
                    cout1["bank_conflict_cycles"])
                counters["group_queue_backpressure_cycles_baseline"] += max(
                    0, bgroup_cycles - 1)
                counters["group_queue_backpressure_cycles_candidate"] += max(
                    0, cgroup_cycles - 1)
                counters["group_queue_max_occupancy_baseline"] = max(
                    counters["group_queue_max_occupancy_baseline"],
                    1 if bgroup_cycles else 0)
                counters["group_queue_max_occupancy_candidate"] = max(
                    counters["group_queue_max_occupancy_candidate"],
                    1 if cgroup_cycles else 0)
                counters["bitmap_read_cycles_baseline"] += bitmap_read_cycles
                counters["bitmap_read_cycles_candidate"] += bitmap_read_cycles
                counters["compare_cycles"] += compare_cycles
                counters["baseline_events"] += count0 + count1
                counters["candidate_events"] += count0 + count1 - common
                counters["exact_overlap_events"] += common

    require(counters["candidate_events"] + counters["exact_overlap_events"] ==
            counters["baseline_events"], "event conservation failed")
    require(baseline["events"] == counters["baseline_events"],
            "baseline service-event mismatch")
    require(candidate["events"] == counters["candidate_events"],
            "candidate service-event mismatch")
    require(counters["destination_zero_init_bytes_baseline"] == 0 and
            counters["destination_zero_init_bytes_candidate"] == 0 and
            counters["destination_rmw_read_bytes_baseline"] == 0 and
            counters["destination_rmw_read_bytes_candidate"] == 0,
            "r4 lane-block dataflow unexpectedly used destination zero/RMW")
    require(destination_banks *
            destination_bank_bw >=
            int(model["compute_lanes"]) *
            int(model["accumulator_bits"]) / 8.0,
            "destination aggregate bandwidth cannot accept one lane block")
    return {
        "baseline_cycles": int(total_baseline_cycles),
        "candidate_cycles": int(total_candidate_cycles),
        "cycle_speedup": (float(total_baseline_cycles) /
                          float(total_candidate_cycles)),
        "baseline_service": baseline,
        "candidate_service": candidate,
        **counters,
    }


def analyze_cohort(name, manifest_path, expectation, model):
    manifest = strict_json(manifest_path)
    require(manifest.get("schema") == expectation["schema"],
            name + " schema drift")
    require(manifest.get("status") == expectation["status"],
            name + " status drift")
    records = manifest["records"]
    require(len(records) == int(expectation["records"]),
            name + " record-count drift")
    require({int(record["sample_id"]) for record in records} ==
            set(range(int(expectation["samples"]))), name + " sample drift")
    require({record["operator"] for record in records} ==
            set(expectation["operators"]), name + " operator drift")
    expected_pairs = {
        (sample_id, operator)
        for sample_id in range(int(expectation["samples"]))
        for operator in expectation["operators"]
    }
    observed_pairs = {
        (int(record["sample_id"]), record["operator"])
        for record in records
    }
    require(len(observed_pairs) == len(records) and
            observed_pairs == expected_pairs,
            name + " sample-by-operator Cartesian coverage drift")
    for sample_id in range(int(expectation["samples"])):
        sample_keys = {
            record["sample_key"] for record in records
            if int(record["sample_id"]) == sample_id
        }
        require(len(sample_keys) == 1,
                name + " sample key is not operator-consistent")

    rows = []
    manifest_dir = manifest_path.parent
    for record in records:
        require(list(record["shape"]) == list(expectation["shape"]),
                name + " shape drift")
        require(list(record["output_shape"]) == list(expectation["shape"]),
                name + " output-shape drift")
        geometry = record["module_geometry"]
        require(int(geometry["in_channels"]) == int(model["input_channels"]) and
                int(geometry["out_channels"]) == int(model["output_channels"]) and
                list(geometry["kernel_size"]) == list(model["kernel"]) and
                list(geometry["stride"]) == [1, 1] and
                list(geometry["padding"]) == [1, 1] and
                list(geometry["dilation"]) == [1, 1] and
                int(geometry["groups"]) == 1,
                name + " frozen Conv geometry drift")
        require(int(record["negative_count"]) == 0,
                "M507 selected trace is not positive-only")
        codebook = record["value_bit_pattern_population"]
        require(int(codebook["unique_float32_bit_patterns"]) == 2 and
                bool(codebook["full_codebook_in_manifest"]),
                "M507 requires complete two-codeword trace")
        entries = codebook["codebook"]
        require(len(entries) == 2,
                "M507 codebook layout drift")
        entry_by_bits = {entry["float32_bits_hex"]: entry for entry in entries}
        expected_codeword = expectation["operator_nonzero_codeword_bits"][
            record["operator"]]
        require(set(entry_by_bits) == {"00000000", expected_codeword},
                name + " operator codeword drift")
        require(sum(int(entry["count"]) for entry in entries) ==
                int(record["elements"]), name + " codebook count drift")
        nonzero = np.asarray(
            [int(expected_codeword, 16)], dtype="<u4").view("<f4")[0]
        require(bool(np.isfinite(nonzero)) and float(nonzero) > 0.0,
                "M507 nonzero codeword is not finite-positive")
        values = load_record(manifest_dir, record)
        value_bits = values.view("<u4")
        require(bool(np.all((value_bits == np.uint32(0)) |
                            (value_bits == np.uint32(
                                int(expected_codeword, 16))))),
                name + " decoded payload contains a third codeword")
        require(int(np.count_nonzero(value_bits == np.uint32(
                    int(expected_codeword, 16)))) ==
                int(entry_by_bits[expected_codeword]["count"]) ==
                int(record["positive_count"]),
                name + " decoded nonzero-codeword population drift")
        metrics = record_cycles(values, model)
        rows.append({
            "cohort": name,
            "sample_id": int(record["sample_id"]),
            "sample_key": record["sample_key"],
            "sequence": sequence_key(record["sample_key"]),
            "operator": record["operator"],
            **metrics,
        })
        del values

    return rows


def aggregate(rows):
    result = {
        "records": len(rows),
        "baseline_cycles": sum(row["baseline_cycles"] for row in rows),
        "candidate_cycles": sum(row["candidate_cycles"] for row in rows),
        "baseline_events": sum(row["baseline_events"] for row in rows),
        "candidate_events": sum(row["candidate_events"] for row in rows),
        "exact_overlap_events": sum(row["exact_overlap_events"] for row in rows),
        "groups": sum(row["groups"] for row in rows),
        "overlap_groups": sum(row["overlap_groups"] for row in rows),
        "weight_bytes_baseline": sum(
            row["baseline_service"]["weight_bytes"] for row in rows),
        "weight_bytes_candidate": sum(
            row["candidate_service"]["weight_bytes"] for row in rows),
        "scratch_write_transactions": sum(
            row["scratch_write_transactions"] for row in rows),
        "scratch_read_transactions": sum(
            row["scratch_read_transactions"] for row in rows),
        "scratch_serialization_stall_cycles": sum(
            row["scratch_serialization_stall_cycles"] for row in rows),
        "scratch_sync_read_tail_cycles": sum(
            row["scratch_sync_read_tail_cycles"] for row in rows),
        "destination_commit_cycles_baseline": sum(
            row["destination_commit_cycles_baseline"] for row in rows),
        "destination_commit_cycles_candidate": sum(
            row["destination_commit_cycles_candidate"] for row in rows),
        "destination_commit_transactions_baseline": sum(
            row["destination_commit_transactions_baseline"] for row in rows),
        "destination_commit_transactions_candidate": sum(
            row["destination_commit_transactions_candidate"] for row in rows),
        "compare_cycles": sum(row["compare_cycles"] for row in rows),
        "weight_not_ready_stall_baseline": sum(
            row["weight_not_ready_stall_baseline"] for row in rows),
        "weight_not_ready_stall_candidate": sum(
            row["weight_not_ready_stall_candidate"] for row in rows),
        "weight_bank_conflict_cycles_baseline": sum(
            row["weight_bank_conflict_cycles_baseline"] for row in rows),
        "weight_bank_conflict_cycles_candidate": sum(
            row["weight_bank_conflict_cycles_candidate"] for row in rows),
        "output_bank_conflict_cycles_baseline": sum(
            row["output_bank_conflict_cycles_baseline"] for row in rows),
        "output_bank_conflict_cycles_candidate": sum(
            row["output_bank_conflict_cycles_candidate"] for row in rows),
        "group_queue_backpressure_cycles_baseline": sum(
            row["group_queue_backpressure_cycles_baseline"] for row in rows),
        "group_queue_backpressure_cycles_candidate": sum(
            row["group_queue_backpressure_cycles_candidate"] for row in rows),
        "group_queue_max_occupancy_baseline": max(
            row["group_queue_max_occupancy_baseline"] for row in rows),
        "group_queue_max_occupancy_candidate": max(
            row["group_queue_max_occupancy_candidate"] for row in rows),
    }
    for field in (
            "destination_zero_init_bytes_baseline",
            "destination_zero_init_bytes_candidate",
            "destination_zero_init_cycles_baseline",
            "destination_zero_init_cycles_candidate",
            "destination_seed_write_bytes_candidate",
            "destination_seed_write_cycles_candidate",
            "destination_rmw_read_bytes_baseline",
            "destination_rmw_read_bytes_candidate",
            "destination_rmw_write_bytes_baseline",
            "destination_rmw_write_bytes_candidate",
            "destination_rmw_sync_tail_cycles_baseline",
            "destination_rmw_sync_tail_cycles_candidate",
            "scratch_write_logical_bytes",
            "scratch_write_physical_bytes",
            "scratch_read_logical_bytes",
            "scratch_read_physical_bytes",
            "lane_first_product_seed_blocks_baseline",
            "lane_first_product_seed_blocks_candidate",
            "lane_scratch_seed_blocks_candidate",
            "destination_product_seed_outputs_baseline",
            "destination_product_seed_outputs_candidate",
            "destination_scratch_seed_outputs_candidate",
            "destination_initialization_mismatch_count",
            "scratch_conservation_mismatch_count",
            "bank_mapping_mismatch_count",
            "lane_issue_order_mismatch_count",
            "destination_materialize_logical_bytes_baseline",
            "destination_materialize_logical_bytes_candidate",
            "destination_materialize_physical_bytes_baseline",
            "destination_materialize_physical_bytes_candidate",
            "destination_materialize_cycles_baseline",
            "destination_materialize_cycles_candidate",
            "destination_materialize_transactions_baseline",
            "destination_materialize_transactions_candidate",
            "destination_final_read_bytes_baseline",
            "destination_final_read_bytes_candidate",
            "destination_final_read_physical_bytes_baseline",
            "destination_final_read_physical_bytes_candidate",
            "destination_final_read_cycles_baseline",
            "destination_final_read_cycles_candidate",
            "destination_final_read_transactions_baseline",
            "destination_final_read_transactions_candidate",
            "output_sink_write_bytes_baseline",
            "output_sink_write_bytes_candidate",
            "output_sink_write_cycles_baseline",
            "output_sink_write_cycles_candidate",
            "output_sink_write_transactions_baseline",
            "output_sink_write_transactions_candidate",
            "destination_bank_conflict_cycles_baseline",
            "destination_bank_conflict_cycles_candidate"):
        result[field] = sum(int(row[field]) for row in rows)
    result["cycle_speedup"] = (float(result["baseline_cycles"]) /
                               float(result["candidate_cycles"]))
    result["event_reduction_ratio"] = (float(result["baseline_events"]) /
                                       float(result["candidate_events"]))
    return result


def reconcile_m501_cohort(m501, cohort_name, rows, aggregated,
                          per_sequence):
    cohorts = [cohort for cohort in m501["cohorts"]
               if cohort["cohort"] == cohort_name]
    require(len(cohorts) == 1, "M501 cohort is not unique: " + cohort_name)
    cohort = cohorts[0]
    selected_detail = [
        row for row in cohort["detailed"]
        if row["axis"] == "horizontal" and int(row["group_size"]) == 2
    ]
    expected_by_key = {
        (int(row["sample_id"]), row["sample_key"], row["operator"]): row
        for row in selected_detail
    }
    observed_by_key = {
        (int(row["sample_id"]), row["sample_key"], row["operator"]): row
        for row in rows
    }
    require(len(expected_by_key) == len(selected_detail) == len(rows) and
            set(expected_by_key) == set(observed_by_key),
            "M501 per-record key coverage drift: " + cohort_name)
    mismatch_count = 0
    for key, expected in expected_by_key.items():
        observed = observed_by_key[key]
        for field in ("baseline_events", "candidate_events",
                      "exact_overlap_events"):
            if int(observed[field]) != int(expected[field]):
                mismatch_count += 1
    require(mismatch_count == 0,
            "M501 per-record event ledger mismatch: " + cohort_name)

    overall_rows = [
        row for row in cohort["aggregate"]["overall"]
        if row["axis"] == "horizontal" and int(row["group_size"]) == 2
    ]
    require(len(overall_rows) == 1,
            "M501 aggregate point is not unique: " + cohort_name)
    overall = overall_rows[0]
    for field in ("baseline_events", "candidate_events",
                  "exact_overlap_events"):
        require(int(aggregated[field]) == int(overall[field]),
                "M501 aggregate event ledger mismatch: " +
                cohort_name + ":" + field)

    expected_sequences = {
        row["sequence"]: row
        for row in cohort["aggregate"]["per_sequence"]
        if row["axis"] == "horizontal" and int(row["group_size"]) == 2
    }
    observed_sequences = {row["sequence"]: row for row in per_sequence}
    require(len(expected_sequences) == len(observed_sequences) and
            set(expected_sequences) == set(observed_sequences),
            "M501 per-sequence coverage drift: " + cohort_name)
    sequence_mismatch_count = 0
    for sequence, expected in expected_sequences.items():
        observed = observed_sequences[sequence]
        for field in ("baseline_events", "candidate_events",
                      "exact_overlap_events"):
            if int(observed[field]) != int(expected[field]):
                sequence_mismatch_count += 1
    require(sequence_mismatch_count == 0,
            "M501 per-sequence event ledger mismatch: " + cohort_name)
    return {
        "cohort": cohort_name,
        "records_reconciled": len(rows),
        "sequences_reconciled": len(per_sequence),
        "per_record_field_mismatches": mismatch_count,
        "per_sequence_field_mismatches": sequence_mismatch_count,
        "aggregate_field_mismatches": 0,
    }


def write_csv(path, rows):
    fields = [
        "cohort", "sample_id", "sample_key", "sequence", "operator",
        "baseline_events", "candidate_events", "exact_overlap_events",
        "baseline_cycles", "candidate_cycles", "cycle_speedup", "groups",
        "overlap_groups", "compare_cycles", "scratch_write_transactions",
        "scratch_read_transactions", "scratch_serialization_stall_cycles",
        "scratch_sync_read_tail_cycles",
        "scratch_write_logical_bytes", "scratch_write_physical_bytes",
        "scratch_read_logical_bytes", "scratch_read_physical_bytes",
        "lane_first_product_seed_blocks_baseline",
        "lane_first_product_seed_blocks_candidate",
        "lane_scratch_seed_blocks_candidate",
        "destination_product_seed_outputs_baseline",
        "destination_product_seed_outputs_candidate",
        "destination_scratch_seed_outputs_candidate",
        "destination_initialization_mismatch_count",
        "scratch_conservation_mismatch_count",
        "destination_zero_init_bytes_baseline",
        "destination_zero_init_bytes_candidate",
        "destination_zero_init_cycles_baseline",
        "destination_zero_init_cycles_candidate",
        "destination_seed_write_bytes_candidate",
        "destination_seed_write_cycles_candidate",
        "destination_rmw_read_bytes_baseline",
        "destination_rmw_read_bytes_candidate",
        "destination_rmw_write_bytes_baseline",
        "destination_rmw_write_bytes_candidate",
        "destination_rmw_sync_tail_cycles_baseline",
        "destination_rmw_sync_tail_cycles_candidate",
        "destination_materialize_logical_bytes_baseline",
        "destination_materialize_logical_bytes_candidate",
        "destination_materialize_physical_bytes_baseline",
        "destination_materialize_physical_bytes_candidate",
        "destination_materialize_cycles_baseline",
        "destination_materialize_cycles_candidate",
        "destination_materialize_transactions_baseline",
        "destination_materialize_transactions_candidate",
        "destination_final_read_bytes_baseline",
        "destination_final_read_bytes_candidate",
        "destination_final_read_physical_bytes_baseline",
        "destination_final_read_physical_bytes_candidate",
        "destination_final_read_cycles_baseline",
        "destination_final_read_cycles_candidate",
        "destination_final_read_transactions_baseline",
        "destination_final_read_transactions_candidate",
        "output_sink_write_bytes_baseline",
        "output_sink_write_bytes_candidate",
        "output_sink_write_cycles_baseline",
        "output_sink_write_cycles_candidate",
        "output_sink_write_transactions_baseline",
        "output_sink_write_transactions_candidate",
        "destination_commit_cycles_baseline",
        "destination_commit_cycles_candidate",
        "destination_commit_transactions_baseline",
        "destination_commit_transactions_candidate",
        "weight_not_ready_stall_baseline", "weight_not_ready_stall_candidate",
        "weight_bank_conflict_cycles_baseline",
        "weight_bank_conflict_cycles_candidate",
        "output_bank_conflict_cycles_baseline",
        "output_bank_conflict_cycles_candidate",
        "destination_bank_conflict_cycles_baseline",
        "destination_bank_conflict_cycles_candidate",
        "group_queue_backpressure_cycles_baseline",
        "group_queue_backpressure_cycles_candidate",
        "group_queue_max_occupancy_baseline",
        "group_queue_max_occupancy_candidate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("\n".join(
        "{}  {}".format(sha256(output_dir / name), name)
        for name in sorted(names)) + "\n", encoding="utf-8")
    seal = output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M507 overwrite")

    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m507_h67_apec_g2_same_resource_cycle_fastkill_contract_v4" and
            contract.get("status") ==
            "LOCKED_R4_LANE_BLOCK_UNIFIED_DATAFLOW_BEFORE_ONE_SHOT_EXECUTION",
            "M507 contract identity drift")
    require(contract["inputs"]["analyzer"]["sha256"] == source_start,
            "M507 analyzer self SHA drift")

    inputs = {}
    for name, spec in contract["inputs"].items():
        path = ROOT / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M507 input SHA drift: " + name)
        inputs[name] = path
    require(inputs["analyzer"].resolve() == Path(__file__).resolve(),
            "M507 analyzer path drift")

    model = contract["cycle_model"]
    require(int(model["common_total_sram_bytes"]) == 240 * 1024,
            "M507 total SRAM budget drift")
    resource_ledger = build_resource_ledger(model)
    scratch_bytes = int(
        resource_ledger["baseline"]["capacity"]["overlap_cache_bytes"])
    require(scratch_bytes == 18432, "M507 scratch capacity drift")
    require(int(model["bitmap_pair_read_cycles"]) == int(math.ceil(
        int(model["pair_bitmap_buffer_bytes"]) /
        float(model["bitmap_bytes_per_cycle"]))),
        "M507 bitmap bandwidth/cycle drift")

    validation_rows = analyze_cohort(
        "validation_s10", inputs["m40_manifest"],
        contract["cohort_expectations"]["m40"], model)
    train_rows = analyze_cohort(
        "train_calibration_s32", inputs["m73_manifest"],
        contract["cohort_expectations"]["m73"], model)
    validation = aggregate(validation_rows)
    train = aggregate(train_rows)

    def sequence_aggregates(rows):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row["sequence"]].append(row)
        return [
            {"sequence": sequence, **aggregate(bucket)}
            for sequence, bucket in sorted(buckets.items())
        ]

    validation_per_sequence = sequence_aggregates(validation_rows)
    train_per_sequence = sequence_aggregates(train_rows)
    require(len(validation_per_sequence) == 1,
            "M507 validation sequence-count drift")
    require(len(train_per_sequence) == 18,
            "M507 train sequence-count drift")
    m501 = strict_json(inputs["m501_result"])
    m501_reconciliation = [
        reconcile_m501_cohort(
            m501, "validation_s10", validation_rows, validation,
            validation_per_sequence),
        reconcile_m501_cohort(
            m501, "train_calibration_s32", train_rows, train,
            train_per_sequence),
    ]

    envelope = contract["frozen_envelope"]
    total = int(envelope["total_cycles"])
    conv = int(envelope["four_bottleneck_conv_cycles"])
    charged_total = total - conv + conv / validation["cycle_speedup"]
    charged_sensitivity = total / charged_total
    gates = contract["hard_gates"]
    gate_results = {
        "event_conservation_zero_mismatch": (
            validation["candidate_events"] +
            validation["exact_overlap_events"] ==
            validation["baseline_events"]),
        "validation_four_conv_cycle_speedup": (
            validation["cycle_speedup"] >=
            float(gates["validation_cycle_speedup_min"])),
        "charged_envelope_sensitivity": (
            charged_sensitivity >=
            float(gates["charged_envelope_sensitivity_min"])),
        "train_worst_sequence_cycle_speedup": (
            min(row["cycle_speedup"] for row in train_per_sequence) >=
            float(gates["train_worst_sequence_cycle_speedup_min"])),
        "same_capacity_and_ports_derived": (
            resource_ledger["baseline"] == resource_ledger["candidate"] and
            sum(resource_ledger["baseline"]["capacity"].values()) ==
            int(model["common_total_sram_bytes"]) and
            "destination_slots" in
            resource_ledger["baseline"]["ports"]),
        "symmetric_destination_path": (
            validation["destination_commit_cycles_baseline"] ==
            validation["destination_commit_cycles_candidate"] and
            validation["destination_commit_transactions_baseline"] ==
            validation["destination_commit_transactions_candidate"] and
            all(
                row["destination_commit_cycles_baseline"] ==
                row["destination_commit_cycles_candidate"] and
                row["destination_commit_transactions_baseline"] ==
                row["destination_commit_transactions_candidate"] and
                row["destination_final_read_bytes_baseline"] ==
                row["destination_final_read_bytes_candidate"] and
                row["output_sink_write_bytes_baseline"] ==
                row["output_sink_write_bytes_candidate"]
                for row in validation_rows + train_rows)),
        "destination_bytes_and_ports_conserve": all(
            row["destination_zero_init_bytes_baseline"] == 0 and
            row["destination_zero_init_bytes_candidate"] == 0 and
            row["destination_rmw_read_bytes_baseline"] == 0 and
            row["destination_rmw_write_bytes_baseline"] == 0 and
            row["destination_rmw_read_bytes_candidate"] == 0 and
            row["destination_rmw_write_bytes_candidate"] == 0 and
            row["destination_materialize_logical_bytes_baseline"] ==
            row["destination_final_read_bytes_baseline"] and
            row["destination_materialize_logical_bytes_candidate"] ==
            row["destination_final_read_bytes_candidate"] and
            row["destination_materialize_physical_bytes_baseline"] ==
            row["destination_final_read_physical_bytes_baseline"] and
            row["destination_materialize_physical_bytes_candidate"] ==
            row["destination_final_read_physical_bytes_candidate"] and
            row["destination_final_read_bytes_baseline"] ==
            row["output_sink_write_bytes_baseline"] and
            row["destination_final_read_bytes_candidate"] ==
            row["output_sink_write_bytes_candidate"] and
            row["destination_initialization_mismatch_count"] == 0 and
            (row["destination_product_seed_outputs_baseline"] ==
             row["destination_product_seed_outputs_candidate"] +
             row["destination_scratch_seed_outputs_candidate"]) and
            row["destination_bank_conflict_cycles_baseline"] == 0 and
            row["destination_bank_conflict_cycles_candidate"] == 0
            for row in validation_rows + train_rows),
        "scratch_logical_and_physical_bytes_conserve": all(
            row["scratch_read_logical_bytes"] ==
            2 * row["scratch_write_logical_bytes"] and
            row["scratch_read_physical_bytes"] ==
            2 * row["scratch_write_physical_bytes"] and
            row["scratch_write_physical_bytes"] >=
            row["scratch_write_logical_bytes"] and
            row["scratch_conservation_mismatch_count"] == 0
            for row in validation_rows + train_rows),
        "locked_bank_mapping_zero_conflicts": (
            all(row["weight_bank_conflict_cycles_baseline"] == 0 and
                row["weight_bank_conflict_cycles_candidate"] == 0 and
                row["output_bank_conflict_cycles_baseline"] == 0 and
                row["output_bank_conflict_cycles_candidate"] == 0 and
                row["bank_mapping_mismatch_count"] == 0 and
                row["lane_issue_order_mismatch_count"] == 0
                for row in validation_rows + train_rows)),
        "single_entry_queue_respected": (
            all(row["group_queue_max_occupancy_baseline"] <=
                int(model["group_queue_entries"]) and
                row["group_queue_max_occupancy_candidate"] <=
                int(model["group_queue_entries"])
                for row in validation_rows + train_rows)),
    }
    all_pass = all(gate_results.values())
    verdict = ("PASS_EXSPIKE_DERIVED_SUPPORT_ONLY_NO_STANDALONE_RTL" if all_pass
               else "KILL_M501_M507_HARDWARE_LINE")

    output = {
        "schema": "m507_h67_apec_g2_same_resource_cycle_fastkill_result_v4",
        "status": verdict,
        "identity": {
            "contract": str(args.contract.resolve().relative_to(ROOT)),
            "contract_sha256": sha256(args.contract.resolve()),
            "analyzer_sha256": source_start,
            "docs359_sha256": sha256(inputs["docs359"]),
        },
        "cycle_model": model,
        "matched_resource_accounting": {
            "derived_ledger": resource_ledger,
            "baseline_mode": "same APEC-capable top, compression disabled",
            "candidate_mode": "same top, exact horizontal G2 enabled",
            "one_lane_accumulator_instance_in_both_arms": True,
            "first_product_or_scratch_response_initializes_each_block": True,
            "destination_per_event_rmw_used": False,
            "completed_block_materialization_and_final_read_charged": True,
            "destination_slots_are_banked_1r1w_sram": True,
            "candidate_only_overlap_cache_path":
                "block-serialized write plus two block-serialized synchronous reads",
        },
        "m501_event_ledger_reconciliation": m501_reconciliation,
        "validation": validation,
        "train_calibration": train,
        "validation_per_sequence": validation_per_sequence,
        "train_per_sequence": train_per_sequence,
        "charged_envelope_sensitivity": {
            "frozen_total_cycles": total,
            "frozen_four_conv_cycles": conv,
            "replacement_ratio_only": validation["cycle_speedup"],
            "charged_total_cycles_analytical": charged_total,
            "sensitivity_ratio": charged_sensitivity,
            "system_cycle_simulation": False,
        },
        "hard_gate_results": gate_results,
        "decision": {
            "all_gates_pass": all_pass,
            "verdict": verdict,
            "standalone_rtl_novelty": False,
            "direct_prior_art": "ExSpike APEC",
            "signed_analog_novelty_activated": False,
            "rtl_authorized": False,
            "next_action": (
                "Retain only as an ExSpike-cited supporting mechanism; do not "
                "implement standalone APEC RTL." if all_pass else
                "Permanently close M501/M507 and retain opportunity as a "
                "negative DSE/prior-art audit only."),
        },
        "claim_boundary": contract["claim_boundary"],
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(
        prefix=args.output_dir.name + ".staging.",
        dir=str(args.output_dir.parent)))
    result_name = "m507_h67_apec_g2_same_resource_cycle_fastkill_result_r4.json"
    csv_name = "m507_record_cycle_ledger_r4.csv"
    seq_name = "m507_train_sequence_cycle_ledger_r4.csv"
    readme_name = "README.md"
    completion_name = "RUN_COMPLETE.txt"
    (staging_dir / result_name).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(staging_dir / csv_name, validation_rows + train_rows)
    with (staging_dir / seq_name).open("w", encoding="utf-8", newline="") as handle:
        fields = ["sequence", "records", "baseline_events", "candidate_events",
                  "exact_overlap_events", "baseline_cycles", "candidate_cycles",
                  "cycle_speedup", "overlap_groups",
                  "scratch_write_transactions", "scratch_read_transactions"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in train_per_sequence:
            writer.writerow({field: row[field] for field in fields})
    (staging_dir / readme_name).write_text(
        "# M507 same-resource APEC-G2 cycle fast-kill\n\n"
        "Verdict: `{}`.\n\n"
        "- Validation four-Conv standalone cycle ratio: `{:.9f}x`.\n"
        "- Charged 620.303M-envelope sensitivity: `{:.9f}x` (analytical only).\n"
        "- Train-only 18-sequence worst cycle ratio: `{:.9f}x`.\n"
        "- Both arms reserve the same 240 KiB SRAM, including the same "
        "18 KiB block-padded 1R1W overlap scratch and two explicit 18 KiB "
        "banked 1R1W "
        "destination slots; all streams use one 96-lane block accumulator, "
        "and block materialization plus final read traffic are charged.\n"
        "- APEC is direct ExSpike prior art; this result never authorizes a "
        "standalone RTL novelty or system-speedup claim.\n".format(
            verdict, validation["cycle_speedup"], charged_sensitivity,
            min(row["cycle_speedup"] for row in train_per_sequence)),
        encoding="utf-8")
    (staging_dir / completion_name).write_text(
        verdict + "\n", encoding="utf-8")
    write_seal(staging_dir, [result_name, csv_name, seq_name, readme_name,
                             completion_name])
    manifest = strict_json(staging_dir / result_name)
    require(manifest["status"] == verdict,
            "M507 staged result status drift")
    require(sha256(staging_dir / "SHA256SUMS") ==
            (staging_dir / "SHA256SUMS.seal.sha256").read_text(
                encoding="utf-8").split()[0],
            "M507 staged seal mismatch")
    require(sha256(Path(__file__).resolve()) == source_start,
            "M507 analyzer mutated during run")
    require(not args.output_dir.exists(),
            "M507 final output appeared during staging")
    os.replace(staging_dir, args.output_dir)
    print(json.dumps({
        "status": verdict,
        "validation_cycle_speedup": validation["cycle_speedup"],
        "charged_envelope_sensitivity": charged_sensitivity,
        "train_worst_sequence_cycle_speedup": min(
            row["cycle_speedup"] for row in train_per_sequence),
        "output_dir": str(args.output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
