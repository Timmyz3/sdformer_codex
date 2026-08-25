#!/usr/bin/env python3
"""Replay frozen H67 FC2 with the exact M202 aligned-packet recurrence."""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST = "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
EXPECTED_M199_SCRIPT = "c5d40586dce632d8ce44ff1abcb237f96ebfdbea4002b336fbaded214c700425"
EXPECTED_M199_RESULT = "44861d8644113267f7265d63030d789774e6a21f9dfee56630c05acd0e40f7f7"
EXPECTED_M172 = "ae633daee1e07f16396570c1ef757c11bac7f1f72be108f4411d75f6dcb5f6d9"
EXPECTED_M192 = "39715b64890d75be7c60587d639f3e75b51e5bf38642e6b8640c761fa512f24b"
EXPECTED_M202_RTL = "eb9f42ffd4286a4f5c83436acdad30568ddd6e7d90510e725d210a9a35677354"
EXPECTED_M202_REVIEW = "f99f16bbadd2e78d100ccf8900344650f61f388606778be957bc5de64b17adaa"
EXPECTED_DOCS359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_RECORDS = 120
EXPECTED_TOKENS = 5580000
EXPECTED_EVENTS = 143894510
EXPECTED_RAW_BEATS = 36480000
EXPECTED_DESCRIPTORS = 18869376
EXPECTED_WINDOWS = 6523707


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, expected, name):
    require(sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction(numerator, denominator):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def mask_from_nonzero(nonzero):
    mask = 0
    for index, value in enumerate(nonzero):
        if value:
            mask |= 1 << index
    return mask


def nonzero_from_mask(mask, extent):
    return tuple(bool(mask & (1 << index)) for index in range(extent))


def m199_segmented_service(nonzero, depth):
    positions = [index for index, value in enumerate(nonzero) if value]
    previous = 0
    cycles = 0
    close_cycles = []
    elapsed = 0
    for target in range(depth, len(positions) + 1, depth):
        boundary = positions[target - 1] + 1
        duration = (boundary - previous + 3) // 4
        cycles += duration
        elapsed += duration
        close_cycles.append(elapsed)
        previous = boundary
    trailing = (len(nonzero) - previous + 3) // 4
    cycles += trailing
    if positions and len(positions) % depth:
        close_cycles.append(cycles)
    intervals = []
    prior = 0
    for close in close_cycles:
        intervals.append(close - prior)
        prior = close
    return intervals, cycles - (close_cycles[-1] if close_cycles else 0), cycles


def m202_service(nonzero, depth, queue_depth=8):
    """Cycle-exact M202 recurrence with an always-ready descriptor sink."""
    raw_position = 0
    arrived = 0
    emitted = 0
    queue_count = 0
    maximum_queue = 0
    cycles = 0
    close_cycles = []
    total_descriptors = sum(nonzero)
    while raw_position < len(nonzero) or queue_count:
        incoming_count = 0
        packet_extent = 0
        if raw_position < len(nonzero):
            packet_extent = min(4, len(nonzero) - raw_position)
            incoming_count = sum(
                nonzero[raw_position:raw_position + packet_extent]
            )
        fresh_mode = queue_count == 0 and incoming_count != 0
        source_count = incoming_count if fresh_mode else queue_count
        remaining_in_window = depth - (emitted % depth)
        descriptor_count = min(4, source_count, remaining_in_window)
        queue_pop = 0 if fresh_mode else descriptor_count
        fresh_pop = descriptor_count if fresh_mode else 0
        available_after_pop = queue_depth - queue_count + queue_pop
        raw_accept = raw_position < len(nonzero) \
            and incoming_count - fresh_pop <= available_after_pop
        require(raw_position >= len(nonzero) or raw_accept or queue_count,
                "M202 deadlock")
        if descriptor_count:
            emitted += descriptor_count
            if emitted % depth == 0:
                close_cycles.append(cycles + 1)
        queue_count -= queue_pop
        if raw_accept:
            raw_position += packet_extent
            arrived += incoming_count
            queue_count += incoming_count - fresh_pop
        require(0 <= queue_count <= queue_depth, "M202 queue overflow")
        maximum_queue = max(maximum_queue, queue_count)
        cycles += 1
    require(arrived == emitted == total_descriptors,
            "M202 descriptor conservation failure")
    if total_descriptors and total_descriptors % depth:
        close_cycles.append(cycles)
    intervals = []
    prior = 0
    for close in close_cycles:
        intervals.append(close - prior)
        prior = close
    trailing = cycles - (close_cycles[-1] if close_cycles else 0)
    return intervals, trailing, cycles, maximum_queue


def empty_ledger():
    return {
        "records": 0, "tokens": 0, "events": 0,
        "raw96_beats": 0, "nonzero96_descriptors": 0,
        "windows": 0, "zero_tokens": 0,
        "m199_w1_s4_f4_wall_cycles": 0,
        "m199_pair_s4_f4_wall_cycles": 0,
        "m202_w1_wall_cycles": 0,
        "m202_pair_wall_cycles": 0,
        "m199_fill_service_cycles": 0,
        "m202_fill_service_cycles": 0,
        "equal_service_tokens": 0,
        "m202_faster_service_tokens": 0,
        "m202_slower_service_tokens": 0,
        "maximum_m202_queue": 0,
    }


def merge(target, source):
    for key, value in source.items():
        if key == "maximum_m202_queue":
            target[key] = max(target[key], value)
        else:
            target[key] += value


def audit_record(record, payload_root, m172, m192, m199, chunk_tokens):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    input_width, output_width, depth = m192.STAGE_GEOMETRY[stage]
    require((shape[-1], output_shape[-1]) == (input_width, output_width),
            "FC2 geometry drift")
    output_blocks = output_width // 96
    beats_per_token = input_width // 96
    bytes_per_token = input_width // 8
    tokens = int(np.prod(shape[:-1], dtype=np.int64))
    payload = payload_root / record["relative_path"]
    require(payload.is_file() and payload.stat().st_size == record["packed_bytes"],
            "payload extent drift")
    require(sha256(payload) == record["file_sha256"], "payload SHA drift")
    raw = np.memmap(payload, dtype=np.uint8, mode="r").reshape(
        tokens, beats_per_token, bytes_per_token // beats_per_token
    )
    ledger = empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    ledger["raw96_beats"] = tokens * beats_per_token
    service_cache = {}
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        byte_bits = m172.BYTE_BITS[np.asarray(raw[start:stop])]
        bank_counts = byte_bits.sum(axis=2, dtype=np.int16)
        beat_events = bank_counts.sum(axis=2, dtype=np.int16)
        nonzero = beat_events != 0
        descriptor_count = nonzero.sum(axis=1, dtype=np.int16)
        positions = np.cumsum(nonzero, axis=1, dtype=np.int16) - 1
        row, beat = np.nonzero(nonzero)
        maximum_windows = (beats_per_token + depth - 1) // depth
        pooled = np.zeros((stop - start, maximum_windows, 8), dtype=np.int16)
        if row.size:
            window = (positions[row, beat] // depth).astype(np.intp)
            np.add.at(pooled, (row, window), bank_counts[row, beat])
        window_count = (descriptor_count.astype(np.int64) + depth - 1) // depth
        for token_offset, count_value in enumerate(window_count):
            count = int(count_value)
            mask = mask_from_nonzero(nonzero[token_offset])
            if mask not in service_cache:
                pattern = nonzero_from_mask(mask, beats_per_token)
                service_cache[mask] = (
                    m199_segmented_service(pattern, depth),
                    m202_service(pattern, depth),
                )
            segmented, measured = service_cache[mask]
            m199_fill, m199_trailing, m199_service = segmented
            m202_fill, m202_trailing, m202_service_cycles, max_queue = measured
            require(len(m199_fill) == len(m202_fill) == count,
                    "window close extent drift")
            ledger["m199_fill_service_cycles"] += m199_service
            ledger["m202_fill_service_cycles"] += m202_service_cycles
            ledger["maximum_m202_queue"] = max(
                ledger["maximum_m202_queue"], max_queue
            )
            if m202_service_cycles == m199_service:
                ledger["equal_service_tokens"] += 1
            elif m202_service_cycles < m199_service:
                ledger["m202_faster_service_tokens"] += 1
            else:
                ledger["m202_slower_service_tokens"] += 1
            if count == 0:
                ledger["zero_tokens"] += 1
                m199_wall = m199_service + 2
                m202_wall = m202_service_cycles + 2
                ledger["m199_w1_s4_f4_wall_cycles"] += m199_wall
                ledger["m199_pair_s4_f4_wall_cycles"] += m199_wall
                ledger["m202_w1_wall_cycles"] += m202_wall
                ledger["m202_pair_wall_cycles"] += m202_wall
                continue
            loads = np.asarray(pooled[token_offset, :count], dtype=np.int64)
            w1_groups = [int(value) for value in loads.max(axis=1)]
            pair_groups = []
            windows_per_job = []
            for left in range(0, count, 2):
                right = min(count, left + 2)
                pair_groups.append(int(loads[left:right].sum(axis=0).max()))
                windows_per_job.append(right - left)
            ledger["m199_w1_s4_f4_wall_cycles"] += m199.finite_wall(
                m199_fill, m199_trailing, w1_groups, [1] * count,
                output_blocks
            )
            ledger["m199_pair_s4_f4_wall_cycles"] += m199.finite_wall(
                m199_fill, m199_trailing, pair_groups, windows_per_job,
                output_blocks
            )
            ledger["m202_w1_wall_cycles"] += m199.finite_wall(
                m202_fill, m202_trailing, w1_groups, [1] * count,
                output_blocks
            )
            ledger["m202_pair_wall_cycles"] += m199.finite_wall(
                m202_fill, m202_trailing, pair_groups, windows_per_job,
                output_blocks
            )
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["nonzero96_descriptors"] += int(descriptor_count.sum())
        ledger["windows"] += int(window_count.sum())
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return stage, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m192-analyzer", required=True, type=Path)
    parser.add_argument("--m199-analyzer", required=True, type=Path)
    parser.add_argument("--m199-result", required=True, type=Path)
    parser.add_argument("--m202-rtl", required=True, type=Path)
    parser.add_argument("--m202-review", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST, "manifest drift")
    require(sha256(args.m199_result) == EXPECTED_M199_RESULT, "M199 result drift")
    require(sha256(args.m202_rtl) == EXPECTED_M202_RTL, "M202 RTL drift")
    require(sha256(args.m202_review) == EXPECTED_M202_REVIEW, "M202 review drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359, "docs359 drift")
    m172 = load_module(args.m172_analyzer, EXPECTED_M172, "m172_pinned_m203")
    m192 = load_module(args.m192_analyzer, EXPECTED_M192, "m192_pinned_m203")
    m199 = load_module(args.m199_analyzer, EXPECTED_M199_SCRIPT, "m199_pinned_m203")
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = [record for record in manifest["records"]
               if record["operator"] == "Linear" and ".mlp.fc2" in record["name"]]
    require(len(records) == EXPECTED_RECORDS, "FC2 record count drift")
    aggregate = empty_ledger()
    per_stage = defaultdict(empty_ledger)
    for ordinal, record in enumerate(records):
        stage, ledger = audit_record(
            record, args.payload_root, m172, m192, m199,
            args.chunk_tokens
        )
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print("[M203] {}/120".format(ordinal + 1), flush=True)
    require(aggregate["tokens"] == EXPECTED_TOKENS, "token identity drift")
    require(aggregate["events"] == EXPECTED_EVENTS, "event identity drift")
    require(aggregate["raw96_beats"] == EXPECTED_RAW_BEATS, "raw identity drift")
    require(aggregate["nonzero96_descriptors"] == EXPECTED_DESCRIPTORS,
            "descriptor identity drift")
    require(aggregate["windows"] == EXPECTED_WINDOWS, "window identity drift")
    with args.m199_result.open("r", encoding="utf-8") as handle:
        old = json.load(handle)
    require(aggregate["m199_w1_s4_f4_wall_cycles"]
            == old["aggregate"]["w1_s4_f4_wall_cycles"], "M199 W1 crosscheck")
    require(aggregate["m199_pair_s4_f4_wall_cycles"]
            == old["aggregate"]["pair_s4_f4_wall_cycles"], "M199 pair crosscheck")
    m199_stage_aware = aggregate["m199_pair_s4_f4_wall_cycles"] \
        - per_stage[0]["m199_pair_s4_f4_wall_cycles"] \
        + per_stage[0]["m199_w1_s4_f4_wall_cycles"]
    m202_stage_aware = aggregate["m202_pair_wall_cycles"] \
        - per_stage[0]["m202_pair_wall_cycles"] \
        + per_stage[0]["m202_w1_wall_cycles"]
    baseline = old["aggregate"]["w1_s1_f1_wall_cycles"]
    result = {
        "schema": "m203_h67_fc2_m202_rtl_semantic_replay_v1",
        "status": "PASS_EXACT_FROZEN_PAYLOAD_M202_RECURRENCE",
        "identity": {
            "analyzer_start_sha256": script_start,
            "manifest_sha256": EXPECTED_MANIFEST,
            "m199_analyzer_sha256": EXPECTED_M199_SCRIPT,
            "m199_result_sha256": EXPECTED_M199_RESULT,
            "m202_rtl_sha256": EXPECTED_M202_RTL,
            "m202_review_sha256s_sha256": EXPECTED_M202_REVIEW,
            "docs359_sha256": EXPECTED_DOCS359,
        },
        "architecture": {
            "raw_packet_alignment": 4,
            "descriptor_emit_width": 4,
            "descriptor_queue_depth": 8,
            "empty_queue_fresh_bypass": True,
            "queue_plus_fresh_coemit": False,
            "descriptor_sink_always_ready_during_fill_interval": True,
            "finite_dual_window_wall_model": True,
            "paired_window_drain_is_analytic_not_m184_rtl": True,
        },
        "aggregate": aggregate,
        "per_stage": {str(key): value for key, value in sorted(per_stage.items())},
        "comparison": {
            "baseline_s1_f1_w1_cycles": baseline,
            "m199_stage_aware_cycles": m199_stage_aware,
            "m202_stage_aware_cycles": m202_stage_aware,
            "m199_stage_aware_speed": fraction(baseline, m199_stage_aware),
            "m202_stage_aware_speed": fraction(baseline, m202_stage_aware),
            "m202_vs_m199_stage_aware_cycle_factor": fraction(
                m199_stage_aware, m202_stage_aware
            ),
            "m202_pair_cycles": aggregate["m202_pair_wall_cycles"],
            "m202_w1_cycles": aggregate["m202_w1_wall_cycles"],
            "m202_pair_over_w1_factor": fraction(
                aggregate["m202_w1_wall_cycles"],
                aggregate["m202_pair_wall_cycles"]
            ),
        },
        "claim_boundary": {
            "exact_m202_frontend_recurrence": True,
            "m202_rtl_measured_cycles": False,
            "four_descriptor_sink_rtl": False,
            "paired_window_drain_rtl": False,
            "complete_fc2": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "analyzer mutated during run")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["comparison"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
