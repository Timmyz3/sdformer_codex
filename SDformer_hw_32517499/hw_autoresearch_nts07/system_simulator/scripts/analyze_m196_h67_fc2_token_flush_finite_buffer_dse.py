#!/usr/bin/env python3
"""Finite-buffer cycle DSE for M195 token-flush FC2 pair fusion.

M195's replay kernel assumes that both windows of a token-owned pair are
resident.  This analyzer charges one accepted nonzero descriptor per fill
cycle, serial group-result drain, pair readiness, odd token tails and a finite
pool of B={2,3,4} window buffers.  A fused pair occupies two buffers until its
last group/output-block result drains.  A third buffer can prefill one window
of the following pair; four buffers can ping-pong complete pairs.

There is no cross-token pairing or window reordering.  Weight-response latency,
SRAM ports, result backpressure, RTL timing and downstream commit are excluded.
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
EXPECTED_M195_RESULT_SHA256 = (
    "58732122f31635b3f958972b3f3b42252a10627d5407ef76f3b2076c2bc84d60"
)
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
EXPECTED_EVENTS = 143894510
EXPECTED_TOKENS = 5580000
EXPECTED_DESCRIPTORS = 18869376
EXPECTED_WINDOWS = 6523707
EXPECTED_W1_REPLAY = 79397844
EXPECTED_W1_B2_WALL = 97607807
EXPECTED_PAIR_REPLAY = 71596122
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


def finite_wall(entries, groups, windows_per_job, output_blocks, buffers):
    """Schedule sequential fills and drains through a circular buffer pool."""
    require(buffers >= max(windows_per_job or [1]), "insufficient buffers")
    fill_end = 0
    drain_end = 0
    buffer_free = [0 for _ in range(buffers)]
    window_index = 0
    entry_index = 0
    for job, job_windows in enumerate(windows_per_job):
        slots = []
        for _unused in range(job_windows):
            slot = window_index % buffers
            fill_end = max(fill_end, buffer_free[slot]) \
                + int(entries[entry_index])
            slots.append(slot)
            window_index += 1
            entry_index += 1
        drain_end = max(fill_end, drain_end) \
            + int(groups[job]) * int(output_blocks)
        for slot in slots:
            buffer_free[slot] = drain_end
    require(entry_index == len(entries), "window/job extent drift")
    return drain_end + 1 if entries else 2


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "nonzero96_descriptors": 0,
        "windows": 0,
        "zero_tokens": 0,
        "full_pairs": 0,
        "odd_tails": 0,
        "w1_replay_cycles": 0,
        "w1_b2_wall_cycles": 0,
        "pair_replay_cycles": 0,
    }
    for buffers in BUFFER_POINTS:
        result["pair_b{}_wall_cycles".format(buffers)] = 0
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
            if count == 0:
                ledger["zero_tokens"] += 1
                ledger["w1_b2_wall_cycles"] += 2
                for buffers in BUFFER_POINTS:
                    ledger["pair_b{}_wall_cycles".format(buffers)] += 2
                continue
            entries = [
                min(depth, descriptors - window * depth)
                for window in range(count)
            ]
            loads = np.asarray(pooled[token_offset, :count], dtype=np.int64)
            w1_groups = loads.max(axis=1)
            pair_groups = []
            windows_per_job = []
            for left in range(0, count, 2):
                right = min(count, left + 2)
                pair_groups.append(int(loads[left:right].sum(axis=0).max()))
                windows_per_job.append(right - left)
            ledger["full_pairs"] += count // 2
            ledger["odd_tails"] += count % 2
            ledger["w1_replay_cycles"] += int(
                w1_groups.sum(dtype=np.int64)
            ) * output_blocks
            ledger["pair_replay_cycles"] += sum(pair_groups) * output_blocks
            ledger["w1_b2_wall_cycles"] += finite_wall(
                entries, list(w1_groups), [1] * count, output_blocks, 2
            )
            for buffers in BUFFER_POINTS:
                ledger["pair_b{}_wall_cycles".format(buffers)] += finite_wall(
                    entries, pair_groups, windows_per_job,
                    output_blocks, buffers
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
    parser.add_argument("--m195-result", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    require(sha256(args.m195_result) == EXPECTED_M195_RESULT_SHA256,
            "M195 result identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359_SHA256,
            "docs359 identity drift")
    m172 = load_pinned(
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256,
        "m172_pinned_m196"
    )
    m192 = load_pinned(
        args.m192_analyzer, EXPECTED_M192_ANALYZER_SHA256,
        "m192_pinned_m196"
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
        print("[M196] {}/120".format(ordinal + 1), flush=True)

    require(aggregate["events"] == EXPECTED_EVENTS, "event identity drift")
    require(aggregate["tokens"] == EXPECTED_TOKENS, "token identity drift")
    require(aggregate["nonzero96_descriptors"] == EXPECTED_DESCRIPTORS,
            "descriptor identity drift")
    require(aggregate["windows"] == EXPECTED_WINDOWS,
            "window identity drift")
    require(aggregate["w1_replay_cycles"] == EXPECTED_W1_REPLAY,
            "W1 replay cross-check drift")
    require(aggregate["w1_b2_wall_cycles"] == EXPECTED_W1_B2_WALL,
            "M187 W1/B2 wall cross-check drift")
    require(aggregate["pair_replay_cycles"] == EXPECTED_PAIR_REPLAY,
            "M195 pair replay cross-check drift")

    points = {}
    for buffers in BUFFER_POINTS:
        wall = aggregate["pair_b{}_wall_cycles".format(buffers)]
        points[str(buffers)] = {
            "window_buffers": buffers,
            "wall_cycles": wall,
            "speed_vs_w1_b2_wall": fraction(EXPECTED_W1_B2_WALL, wall),
            "fill_drain_overhead_cycles": wall - EXPECTED_PAIR_REPLAY,
            "replay_to_wall_ratio": fraction(EXPECTED_PAIR_REPLAY, wall),
        }
    result = {
        "schema": "m196_h67_fc2_token_flush_finite_buffer_dse_v1",
        "status": "PASS_EXACT_FINITE_BUFFER_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m192_analyzer_sha256": EXPECTED_M192_ANALYZER_SHA256,
            "m195_result_sha256": EXPECTED_M195_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "m187_w1_b2_wall_crosscheck": EXPECTED_W1_B2_WALL,
            "m195_pair_replay_crosscheck": EXPECTED_PAIR_REPLAY,
        },
        "architecture": {
            "token_boundary_flush": True,
            "cross_token_pairing": False,
            "window_reordering": False,
            "fill_descriptors_per_cycle": 1,
            "drain_group_results_per_cycle": 1,
            "pair_occupancy_windows": 2,
            "buffer_points": list(BUFFER_POINTS),
            "stage_window_depths": {"0": 2, "1": 4, "2": 8, "3": 8},
        },
        "aggregate": aggregate,
        "points": points,
        "per_stage": {
            str(stage): {
                **per_stage[stage],
                "b2_speed_vs_w1": fraction(
                    per_stage[stage]["w1_b2_wall_cycles"],
                    per_stage[stage]["pair_b2_wall_cycles"]
                ),
                "b3_speed_vs_w1": fraction(
                    per_stage[stage]["w1_b2_wall_cycles"],
                    per_stage[stage]["pair_b3_wall_cycles"]
                ),
                "b4_speed_vs_w1": fraction(
                    per_stage[stage]["w1_b2_wall_cycles"],
                    per_stage[stage]["pair_b4_wall_cycles"]
                ),
            }
            for stage in m192.STAGE_GEOMETRY
        },
        "claim_boundary": {
            "exact_payload_finite_buffer_cycles": True,
            "weight_sram_response_latency": False,
            "result_backpressure": False,
            "integrated_rtl": False,
            "logic_only_dc": False,
            "complete_fc2": False,
            "ffn_speedup": False,
            "physical_speedup": False,
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
