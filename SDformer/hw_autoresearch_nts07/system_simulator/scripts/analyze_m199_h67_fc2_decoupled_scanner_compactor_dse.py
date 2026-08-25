#!/usr/bin/env python3
"""Exact decoupled raw-scanner/stable-compactor DSE for H67 FC2.

M198 tied raw scan width to descriptor write width.  M199 independently varies
S contiguous raw96 beats scanned per cycle and F stable nonzero descriptors
written per cycle.  A finite in-window reservoir absorbs bursts; descriptors
never cross a window or token boundary.  Exact causal arrivals determine fill
latency rather than max(total scan,total emit) arithmetic.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST_SHA256 = (
    "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
)
EXPECTED_M172_ANALYZER_SHA256 = (
    "ae633daee1e07f16396570c1ef757c11bac7f1f72be108f4411d75f6dcb5f6d9"
)
EXPECTED_M192_ANALYZER_SHA256 = (
    "39715b64890d75be7c60587d639f3e75b51e5bf38642e6b8640c761fa512f24b"
)
EXPECTED_M198_ANALYZER_SHA256 = (
    "1b2504f478d0edb212379dbee81c3658da80c633fe2e16e238e2307d87dca625"
)
EXPECTED_M198_RESULT_SHA256 = (
    "f93020381bc1c5d25f16d84029b3a748caec25cda9fd0bc68ce3c6110848a6cb"
)
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
EXPECTED_EVENTS = 143894510
EXPECTED_TOKENS = 5580000
EXPECTED_RAW_BEATS = 36480000
EXPECTED_DESCRIPTORS = 18869376
EXPECTED_WINDOWS = 6523707
EXPECTED_PAIR_REPLAY = 71596122
EXPECTED_M198_B2 = {
    "s1_f1": 115308018,
    "s2_f2": 97542123,
    "s4_f4": 90222444,
    "s8_f8": 87973357,
}
PIPE_POINTS = (
    (1, 1),
    (2, 1), (2, 2),
    (4, 1), (4, 2), (4, 4),
    (8, 2), (8, 4), (8, 8),
)
WIDTH = 96
BUFFERS = 2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path, expected_sha, name):
    require(sha256(path) == expected_sha, name + " identity drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def segment_service(nonzero_segment, scan_width, emit_width):
    """Return causal cycles and maximum post-emit reservoir occupancy."""
    require(scan_width >= emit_width >= 1, "illegal S/F point")
    backlog = 0
    maximum_backlog = 0
    cycles = 0
    extent = int(nonzero_segment.size)
    for start in range(0, extent, scan_width):
        stop = min(extent, start + scan_width)
        backlog += int(np.count_nonzero(nonzero_segment[start:stop]))
        backlog -= min(backlog, emit_width)
        maximum_backlog = max(maximum_backlog, backlog)
        cycles += 1
    while backlog:
        backlog -= min(backlog, emit_width)
        maximum_backlog = max(maximum_backlog, backlog)
        cycles += 1
    require(backlog == 0, "compactor drain failure")
    return cycles, maximum_backlog


def finite_wall(fill_cycles, trailing_cycles, groups,
                windows_per_job, output_blocks):
    if not fill_cycles:
        return int(trailing_cycles) + 2
    require(sum(windows_per_job) == len(fill_cycles),
            "window/job extent drift")
    fill_end = 0
    drain_end = 0
    buffer_free = [0, 0]
    window_index = 0
    cycle_index = 0
    for job, job_windows in enumerate(windows_per_job):
        slots = []
        for _unused in range(job_windows):
            slot = window_index & 1
            fill_end = max(fill_end, buffer_free[slot]) \
                + int(fill_cycles[cycle_index])
            slots.append(slot)
            window_index += 1
            cycle_index += 1
        drain_end = max(fill_end, drain_end) \
            + int(groups[job]) * int(output_blocks)
        for slot in slots:
            buffer_free[slot] = drain_end
    require(cycle_index == len(fill_cycles), "fill extent drift")
    fill_end += int(trailing_cycles)
    return max(fill_end, drain_end) + 1


def point_key(scan_width, emit_width):
    return "s{}_f{}".format(scan_width, emit_width)


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "raw96_beats": 0,
        "nonzero96_descriptors": 0,
        "windows": 0,
        "zero_tokens": 0,
        "full_pairs": 0,
        "odd_tails": 0,
        "pair_replay_cycles": 0,
    }
    for scan_width, emit_width in PIPE_POINTS:
        key = point_key(scan_width, emit_width)
        result["w1_{}_wall_cycles".format(key)] = 0
        result["pair_{}_wall_cycles".format(key)] = 0
        result["{}_maximum_post_emit_backlog".format(key)] = 0
    return result


def merge(target, source):
    for key, value in source.items():
        if key.endswith("_maximum_post_emit_backlog"):
            target[key] = max(int(target[key]), int(value))
        else:
            target[key] += int(value)


def audit_record(record, payload_root, m172, m192, chunk_tokens):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    input_width, output_width, depth = m192.STAGE_GEOMETRY[stage]
    require((shape[-1], output_shape[-1]) == (input_width, output_width),
            "FC2 geometry drift")
    output_blocks = output_width // WIDTH
    beats_per_token = input_width // WIDTH
    bytes_per_token = input_width // 8
    tokens = int(np.prod(shape[:-1], dtype=np.int64))
    payload = payload_root / record["relative_path"]
    require(payload.is_file(), "payload missing")
    require(payload.stat().st_size == record["packed_bytes"],
            "payload extent drift")
    require(sha256(payload) == record["file_sha256"], "payload SHA drift")
    raw = np.memmap(payload, dtype=np.uint8, mode="r").reshape(
        tokens, beats_per_token, bytes_per_token // beats_per_token
    )
    ledger = empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    ledger["raw96_beats"] = tokens * beats_per_token
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
        pooled = np.zeros(
            (stop - start, maximum_windows, 8), dtype=np.int16
        )
        if row.size:
            window = (positions[row, beat] // depth).astype(np.intp)
            np.add.at(pooled, (row, window), bank_counts[row, beat])
        window_count = (
            descriptor_count.astype(np.int64) + depth - 1
        ) // depth
        for token_offset, count_value in enumerate(window_count):
            count = int(count_value)
            descriptors = int(descriptor_count[token_offset])
            token_nonzero = nonzero[token_offset]
            nonzero_indices = np.flatnonzero(token_nonzero)
            segments = []
            previous_boundary = 0
            for window_index in range(count):
                entries = min(depth, descriptors - window_index * depth)
                if entries == depth:
                    boundary = int(
                        nonzero_indices[(window_index + 1) * depth - 1]
                    ) + 1
                else:
                    boundary = beats_per_token
                require(boundary > previous_boundary,
                        "nonpositive scan segment")
                segments.append(token_nonzero[previous_boundary:boundary])
                previous_boundary = boundary
            trailing_raw = beats_per_token - previous_boundary
            if count == 0:
                ledger["zero_tokens"] += 1
                for scan_width, emit_width in PIPE_POINTS:
                    key = point_key(scan_width, emit_width)
                    wall = (beats_per_token + scan_width - 1) \
                        // scan_width + 2
                    ledger["w1_{}_wall_cycles".format(key)] += wall
                    ledger["pair_{}_wall_cycles".format(key)] += wall
                continue
            loads = np.asarray(pooled[token_offset, :count], dtype=np.int64)
            w1_groups = [int(value) for value in loads.max(axis=1)]
            pair_groups = []
            windows_per_job = []
            for left in range(0, count, 2):
                right = min(count, left + 2)
                pair_groups.append(int(loads[left:right].sum(axis=0).max()))
                windows_per_job.append(right - left)
            ledger["full_pairs"] += count // 2
            ledger["odd_tails"] += count % 2
            ledger["pair_replay_cycles"] += sum(pair_groups) * output_blocks
            for scan_width, emit_width in PIPE_POINTS:
                key = point_key(scan_width, emit_width)
                fill_cycles = []
                maximum_backlog = 0
                for segment in segments:
                    cycles, backlog = segment_service(
                        segment, scan_width, emit_width
                    )
                    fill_cycles.append(cycles)
                    maximum_backlog = max(maximum_backlog, backlog)
                ledger[
                    "{}_maximum_post_emit_backlog".format(key)
                ] = max(
                    ledger["{}_maximum_post_emit_backlog".format(key)],
                    maximum_backlog,
                )
                trailing_cycles = (trailing_raw + scan_width - 1) \
                    // scan_width
                ledger["w1_{}_wall_cycles".format(key)] += finite_wall(
                    fill_cycles, trailing_cycles, w1_groups,
                    [1] * count, output_blocks
                )
                ledger["pair_{}_wall_cycles".format(key)] += finite_wall(
                    fill_cycles, trailing_cycles, pair_groups,
                    windows_per_job, output_blocks
                )
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["nonzero96_descriptors"] += int(
            descriptor_count.sum(dtype=np.int64)
        )
        ledger["windows"] += int(window_count.sum(dtype=np.int64))
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return stage, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m192-analyzer", required=True, type=Path)
    parser.add_argument("--m198-analyzer", required=True, type=Path)
    parser.add_argument("--m198-result", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    require(sha256(args.m198_analyzer) == EXPECTED_M198_ANALYZER_SHA256,
            "M198 analyzer identity drift")
    require(sha256(args.m198_result) == EXPECTED_M198_RESULT_SHA256,
            "M198 result identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359_SHA256,
            "docs359 identity drift")
    m172 = load_pinned(
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256,
        "m172_pinned_m199"
    )
    m192 = load_pinned(
        args.m192_analyzer, EXPECTED_M192_ANALYZER_SHA256,
        "m192_pinned_m199"
    )
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = [
        record for record in manifest["records"]
        if record["operator"] == "Linear" and ".mlp.fc2" in record["name"]
    ]
    require(len(records) == 120, "expected 120 FC2 records")
    aggregate = empty_ledger()
    per_stage = defaultdict(empty_ledger)
    for ordinal, record in enumerate(records):
        stage, ledger = audit_record(
            record, args.payload_root, m172, m192, args.chunk_tokens
        )
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print("[M199] {}/120".format(ordinal + 1), flush=True)
    require(aggregate["events"] == EXPECTED_EVENTS, "event identity drift")
    require(aggregate["tokens"] == EXPECTED_TOKENS, "token identity drift")
    require(aggregate["raw96_beats"] == EXPECTED_RAW_BEATS,
            "raw beat identity drift")
    require(aggregate["nonzero96_descriptors"] == EXPECTED_DESCRIPTORS,
            "descriptor identity drift")
    require(aggregate["windows"] == EXPECTED_WINDOWS,
            "window identity drift")
    require(aggregate["pair_replay_cycles"] == EXPECTED_PAIR_REPLAY,
            "pair replay identity drift")
    for key, expected in EXPECTED_M198_B2.items():
        require(aggregate["pair_{}_wall_cycles".format(key)] == expected,
                "M198 {} cross-check drift".format(key))

    legacy = aggregate["w1_s1_f1_wall_cycles"]
    points = {}
    for scan_width, emit_width in PIPE_POINTS:
        key = point_key(scan_width, emit_width)
        w1_wall = aggregate["w1_{}_wall_cycles".format(key)]
        pair_wall = aggregate["pair_{}_wall_cycles".format(key)]
        points[key] = {
            "raw_beats_per_cycle": scan_width,
            "descriptor_emits_per_cycle": emit_width,
            "raw_ingress_bits_per_cycle": scan_width * WIDTH,
            "descriptor_output_bits_per_cycle": emit_width * WIDTH,
            "maximum_post_emit_backlog_descriptors": aggregate[
                "{}_maximum_post_emit_backlog".format(key)
            ],
            "w1_wall_cycles": w1_wall,
            "pair_wall_cycles": pair_wall,
            "speed_vs_s1_f1_w1": fraction(legacy, pair_wall),
            "fusion_increment_vs_iso_pipeline_w1": fraction(
                w1_wall, pair_wall
            ),
            "w1_pipeline_speed_vs_s1_f1": fraction(legacy, w1_wall),
        }
    result = {
        "schema": "m199_h67_fc2_decoupled_scanner_compactor_dse_v1",
        "status": "PASS_EXACT_DECOUPLED_SCANNER_COMPACTOR_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m198_analyzer_sha256": EXPECTED_M198_ANALYZER_SHA256,
            "m198_result_sha256": EXPECTED_M198_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "m198_b2_crosschecks": EXPECTED_M198_B2,
        },
        "architecture": {
            "input": "contiguous raw 96-bit sn2 bitmap beats",
            "pipeline_points": [
                {"scan_width": scan_width, "emit_width": emit_width}
                for scan_width, emit_width in PIPE_POINTS
            ],
            "window_buffers": BUFFERS,
            "stable_original_beat_order": True,
            "causal_finite_reservoir": True,
            "preindexed_nonzero_oracle": False,
            "cross_window_same_cycle_fill": False,
            "cross_token_same_cycle_fill": False,
        },
        "aggregate": aggregate,
        "points": points,
        "per_stage": {
            str(stage): per_stage[stage]
            for stage in m192.STAGE_GEOMETRY
        },
        "claim_boundary": {
            "exact_raw_bitmap_cycles": True,
            "causal_stable_compactor": True,
            "finite_compactor_backlog": True,
            "integrated_rtl": False,
            "logic_only_dc": False,
            "weight_sram_response_latency": False,
            "result_backpressure": False,
            "physical_speedup": False,
            "complete_fc2": False,
            "ffn_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    require(sha256(script_path) == script_start,
            "analyzer changed during run")


if __name__ == "__main__":
    main()
