#!/usr/bin/env python3
"""Exact raw-bitmap scanner DSE for H67 FC2 token-owned pair fusion.

Unlike M197, this model does not begin with an oracle list of nonzero96
descriptors.  It consumes contiguous raw 96-bit sn2 bitmap beats, compacts up
to R nonzero beats per scan cycle and forms fixed-depth windows in original
beat order.  A conservative boundary rule forbids one scan cycle from filling
two different windows.  Full windows may drain before token scan completion;
the final partial window closes only at raw token end.

Weight SRAM response latency, result backpressure, RTL timing and physical
cost remain excluded.
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
EXPECTED_M197_RESULT_SHA256 = (
    "2174e8fe8f33ef0e57a43cad6243fa86147325a0c8dd3aa391b1c35f08ee6777"
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
SCAN_WIDTHS = (1, 2, 4, 8)
BUFFER_POINTS = (2, 3, 4)
WIDTH = 96


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


def scan_wall(segment_raw_beats, trailing_raw_beats, groups,
              windows_per_job, output_blocks, buffers, scan_width):
    """Schedule a conservative raw scanner against finite window buffers."""
    if not segment_raw_beats:
        return (int(trailing_raw_beats) + scan_width - 1) // scan_width + 2
    require(buffers >= max(windows_per_job), "insufficient buffers")
    require(sum(windows_per_job) == len(segment_raw_beats),
            "window/job extent drift")
    scan_end = 0
    drain_end = 0
    buffer_free = [0 for _ in range(buffers)]
    window_index = 0
    segment_index = 0
    for job, job_windows in enumerate(windows_per_job):
        slots = []
        for _unused in range(job_windows):
            slot = window_index % buffers
            scan_cycles = (
                int(segment_raw_beats[segment_index]) + scan_width - 1
            ) // scan_width
            scan_end = max(scan_end, buffer_free[slot]) + scan_cycles
            slots.append(slot)
            window_index += 1
            segment_index += 1
        drain_end = max(scan_end, drain_end) \
            + int(groups[job]) * int(output_blocks)
        for slot in slots:
            buffer_free[slot] = drain_end
    require(segment_index == len(segment_raw_beats),
            "segment extent drift")
    trailing_cycles = (
        int(trailing_raw_beats) + scan_width - 1
    ) // scan_width
    scan_end += trailing_cycles
    return max(scan_end, drain_end) + 1


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
    for scan_width in SCAN_WIDTHS:
        result["w1_r{}_b2_wall_cycles".format(scan_width)] = 0
        for buffers in BUFFER_POINTS:
            result[
                "pair_r{}_b{}_wall_cycles".format(scan_width, buffers)
            ] = 0
    return result


def merge(target, source):
    for key, value in source.items():
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
            nonzero_indices = np.flatnonzero(nonzero[token_offset])
            segment_raw_beats = []
            previous_boundary = 0
            for window_index in range(count):
                entries = min(
                    depth, descriptors - window_index * depth
                )
                if entries == depth:
                    final_descriptor = (window_index + 1) * depth - 1
                    boundary = int(nonzero_indices[final_descriptor]) + 1
                else:
                    boundary = beats_per_token
                require(boundary > previous_boundary,
                        "nonpositive scan segment")
                segment_raw_beats.append(boundary - previous_boundary)
                previous_boundary = boundary
            trailing = beats_per_token - previous_boundary
            require(trailing >= 0, "negative trailing scan")
            if count == 0:
                ledger["zero_tokens"] += 1
                for scan_width in SCAN_WIDTHS:
                    zero_wall = scan_wall(
                        [], beats_per_token, [], [], output_blocks,
                        2, scan_width
                    )
                    ledger[
                        "w1_r{}_b2_wall_cycles".format(scan_width)
                    ] += zero_wall
                    for buffers in BUFFER_POINTS:
                        ledger[
                            "pair_r{}_b{}_wall_cycles".format(
                                scan_width, buffers
                            )
                        ] += zero_wall
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
            for scan_width in SCAN_WIDTHS:
                ledger[
                    "w1_r{}_b2_wall_cycles".format(scan_width)
                ] += scan_wall(
                    segment_raw_beats, trailing, w1_groups,
                    [1] * count, output_blocks, 2, scan_width
                )
                for buffers in BUFFER_POINTS:
                    ledger[
                        "pair_r{}_b{}_wall_cycles".format(
                            scan_width, buffers
                        )
                    ] += scan_wall(
                        segment_raw_beats, trailing, pair_groups,
                        windows_per_job, output_blocks, buffers, scan_width
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
    parser.add_argument("--m197-result", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    require(sha256(args.m197_result) == EXPECTED_M197_RESULT_SHA256,
            "M197 result identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359_SHA256,
            "docs359 identity drift")
    m172 = load_pinned(
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256,
        "m172_pinned_m198"
    )
    m192 = load_pinned(
        args.m192_analyzer, EXPECTED_M192_ANALYZER_SHA256,
        "m192_pinned_m198"
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
        print("[M198] {}/120".format(ordinal + 1), flush=True)
    require(aggregate["events"] == EXPECTED_EVENTS, "event identity drift")
    require(aggregate["tokens"] == EXPECTED_TOKENS, "token identity drift")
    require(aggregate["raw96_beats"] == EXPECTED_RAW_BEATS,
            "raw beat identity drift")
    require(aggregate["nonzero96_descriptors"] == EXPECTED_DESCRIPTORS,
            "descriptor identity drift")
    require(aggregate["windows"] == EXPECTED_WINDOWS,
            "window identity drift")
    require(aggregate["pair_replay_cycles"] == EXPECTED_PAIR_REPLAY,
            "M195 replay cross-check drift")

    legacy_raw = aggregate["w1_r1_b2_wall_cycles"]
    points = {}
    for scan_width in SCAN_WIDTHS:
        iso_w1 = aggregate[
            "w1_r{}_b2_wall_cycles".format(scan_width)
        ]
        for buffers in BUFFER_POINTS:
            pair_wall = aggregate[
                "pair_r{}_b{}_wall_cycles".format(scan_width, buffers)
            ]
            points["r{}_b{}".format(scan_width, buffers)] = {
                "raw_beats_per_cycle": scan_width,
                "raw_ingress_bits_per_cycle": WIDTH * scan_width,
                "window_buffers": buffers,
                "w1_same_width_b2_wall_cycles": iso_w1,
                "pair_wall_cycles": pair_wall,
                "speed_vs_raw_w1_r1_b2": fraction(legacy_raw, pair_wall),
                "fusion_increment_vs_iso_width_w1_b2": fraction(
                    iso_w1, pair_wall
                ),
                "w1_scanner_width_speed_vs_r1": fraction(
                    legacy_raw, iso_w1
                ),
            }
    result = {
        "schema": "m198_h67_fc2_raw_scanner_pair_fusion_dse_v1",
        "status": "PASS_EXACT_RAW_BITMAP_SCANNER_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m192_analyzer_sha256": EXPECTED_M192_ANALYZER_SHA256,
            "m197_result_sha256": EXPECTED_M197_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "m195_pair_replay_crosscheck": EXPECTED_PAIR_REPLAY,
        },
        "architecture": {
            "input": "contiguous raw 96-bit sn2 bitmap beats",
            "scan_widths": list(SCAN_WIDTHS),
            "buffer_points": list(BUFFER_POINTS),
            "stable_original_beat_order": True,
            "preindexed_nonzero_oracle": False,
            "cross_window_same_cycle_fill": False,
            "cross_token_same_cycle_fill": False,
            "full_window_early_close": True,
            "partial_window_closes_at_raw_token_end": True,
            "drain_group_results_per_cycle": 1,
        },
        "aggregate": aggregate,
        "points": points,
        "per_stage": {
            str(stage): per_stage[stage]
            for stage in m192.STAGE_GEOMETRY
        },
        "claim_boundary": {
            "exact_raw_bitmap_scanner_cycles": True,
            "preindexed_nonzero_oracle": False,
            "cross_window_compactor": False,
            "weight_sram_response_latency": False,
            "result_backpressure": False,
            "integrated_rtl": False,
            "logic_only_dc": False,
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
