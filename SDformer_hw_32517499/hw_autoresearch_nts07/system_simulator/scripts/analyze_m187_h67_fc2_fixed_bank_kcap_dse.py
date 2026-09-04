#!/usr/bin/env python3
"""Exact H67 FC2 fixed-bank arithmetic-cap DSE on frozen binary payloads.

For a closed descriptor window with eight bank populations, a K-source issue
may consume at most one item from a bank and at most K items overall.  The
minimum number of issue groups is therefore

    max(max(bank_population), ceil(sum(bank_population) / K)).

Serving the K largest remaining bank populations each cycle attains this
bound.  The script inserts those exact group counts into the pinned M179
two-window recurrence at the bounded D={2,4,8,8} point.  It screens K=1..8;
it does not include selector or accumulator RTL, SRAM response latency, power,
BN2/residual, complete-FC2 or system cycles.
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
EXPECTED_M179_ANALYZER_SHA256 = (
    "9d2dbea7779480aebde4dd9f6e4d720aafeee1fce53195f577a09958954105bc"
)
EXPECTED_M179_RESULT_SHA256 = (
    "8138b14ea0a48aed73d741eb8196ea21ec0781f7421c5df24760243bdfc47025"
)
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
EXPECTED_M179_K4_WALL = 127581198
EXPECTED_M182_K8_WALL = 97607807
WIDTH = 96
K_POINTS = tuple(range(1, 9))
STAGE_GEOMETRY = {
    0: (384, 96, 2),
    1: (768, 192, 4),
    2: (1536, 384, 8),
    3: (3072, 768, 8),
}


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


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "raw96_beats": 0,
        "nonzero96_descriptors": 0,
        "windows": 0,
        "zero_tokens": 0,
    }
    for cap in K_POINTS:
        result[f"groups_k{cap}"] = 0
        result[f"replay_cycles_k{cap}"] = 0
        result[f"wall_cycles_k{cap}"] = 0
    return result


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def audit_record(record, payload_root, m172, m179, chunk_tokens):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    input_width, output_width, depth = STAGE_GEOMETRY[stage]
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
    ledger.update({
        "records": 1,
        "tokens": tokens,
        "raw96_beats": tokens * beats_per_token,
    })
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
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["nonzero96_descriptors"] += int(
            descriptor_count.sum(dtype=np.int64)
        )
        ledger["windows"] += int(window_count.sum(dtype=np.int64))
        ledger["zero_tokens"] += int(
            np.count_nonzero(descriptor_count == 0)
        )
        total = pooled.sum(axis=2, dtype=np.int64)
        busiest = pooled.max(axis=2).astype(np.int64)
        for cap in K_POINTS:
            capacity_bound = (total + cap - 1) // cap
            groups = np.maximum(busiest, capacity_bound)
            walls = m179.finite_dual_window_wall(
                groups, descriptor_count, depth, output_blocks
            )
            group_sum = int(groups.sum(dtype=np.int64))
            ledger[f"groups_k{cap}"] += group_sum
            ledger[f"replay_cycles_k{cap}"] += group_sum * output_blocks
            ledger[f"wall_cycles_k{cap}"] += int(
                walls.sum(dtype=np.int64)
            )
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return stage, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m179-analyzer", required=True, type=Path)
    parser.add_argument("--m179-result", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    require(sha256(args.m179_result) == EXPECTED_M179_RESULT_SHA256,
            "M179 result identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359_SHA256,
            "docs359 identity drift")
    m172 = load_pinned(
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256, "m172_pinned_m187"
    )
    m179 = load_pinned(
        args.m179_analyzer, EXPECTED_M179_ANALYZER_SHA256, "m179_pinned_m187"
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
            record, args.payload_root, m172, m179, args.chunk_tokens
        )
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print(f"[M187] {ordinal + 1}/120", flush=True)
    require(aggregate["events"] == 143894510, "event identity drift")
    require(aggregate["raw96_beats"] == 36480000,
            "raw beat identity drift")
    require(aggregate["wall_cycles_k4"] == EXPECTED_M179_K4_WALL,
            "M179 bounded K4 cross-check drift")
    require(aggregate["wall_cycles_k8"] == EXPECTED_M182_K8_WALL,
            "M182 bounded K8 cross-check drift")

    points = {}
    for cap in K_POINTS:
        wall = aggregate[f"wall_cycles_k{cap}"]
        points[str(cap)] = {
            "maximum_sources_per_issue": cap,
            "weight_response_payload_bits": cap * WIDTH * 8,
            "groups": aggregate[f"groups_k{cap}"],
            "replay_cycles": aggregate[f"replay_cycles_k{cap}"],
            "wall_cycles": wall,
            "k1_over_k": fraction(aggregate["wall_cycles_k1"], wall),
            "k4_over_k": fraction(EXPECTED_M179_K4_WALL, wall),
            "k8_over_k": fraction(EXPECTED_M182_K8_WALL, wall),
        }
    result = {
        "schema": "m187_h67_fc2_fixed_bank_kcap_exact_payload_dse_v1",
        "status": "PASS_EXACT_PAYLOAD_FIXED_BANK_KCAP_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m179_analyzer_sha256": EXPECTED_M179_ANALYZER_SHA256,
            "m179_result_sha256": EXPECTED_M179_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
            "m179_k4_crosscheck_wall_cycles": EXPECTED_M179_K4_WALL,
            "m182_k8_crosscheck_wall_cycles": EXPECTED_M182_K8_WALL,
        },
        "architecture": {
            "physical_weight_banks": 8,
            "arithmetic_cap_points": list(K_POINTS),
            "maximum_one_source_per_bank_per_group": True,
            "group_formula": "max(max_bank_population,ceil(total_events/K))",
            "achieving_policy": "serve K largest remaining bank populations",
            "stage_window_depths": {"0": 2, "1": 4, "2": 8, "3": 8},
            "ping_pong_windows": 2,
            "fill_descriptors_per_cycle": 1,
            "drain_group_results_per_cycle": 1,
        },
        "aggregate_identity": {
            key: value for key, value in aggregate.items()
            if not key.startswith(("groups_k", "replay_cycles_k", "wall_cycles_k"))
        },
        "points": points,
        "per_stage": {
            str(stage): per_stage[stage] for stage in STAGE_GEOMETRY
        },
        "claim_boundary": {
            "exact_payload_finite_buffer_cycle_dse": True,
            "largest_k_selector_rtl": False,
            "k5_k6_k7_accumulator_rtl": False,
            "weight_sram_response_latency": False,
            "logic_only_dc": False,
            "power": False,
            "complete_fc2": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    require(sha256(Path(__file__).resolve()) == script_start,
            "analyzer changed during run")


if __name__ == "__main__":
    main()
