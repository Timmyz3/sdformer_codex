#!/usr/bin/env python3
"""Exact H67 FC2 DSE for hardware-simple XOR-paired four-lane reservoirs.

M179's global top-four-bank selector is cycle optimal inside each finite
descriptor window, but expensive to materialize.  This audit keeps the same
native/preindexed nonzero96 input, token directory, D={2,4,8,8} dual-window
fill/drain recurrence and K4 replay.  It replaces global sorting with four
fixed lanes.  For XOR mask m, a lane owns the pair (b, b xor m) and drains one
event per cycle from either member.  A window therefore needs exactly the
maximum paired population cycles.  Seven wiring-only XOR matchings are
reported, including the predeclared half-split mask four.

This is an exact frozen-payload cycle DSE.  It is not an RTL, PPA, complete-FC2
or system-speedup result.
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
STAGE_GEOMETRY = {
    0: (384, 96),
    1: (768, 192),
    2: (1536, 384),
    3: (3072, 768),
}
STAGE_DEPTH = {0: 2, 1: 4, 2: 8, 3: 8}
XOR_MASKS = tuple(range(1, 8))
PREDECLARED_MASK = 4
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


def xor_pairs(mask):
    unseen = set(range(8))
    pairs = []
    while unseen:
        left = min(unseen)
        right = left ^ mask
        require(right in unseen and right != left, "invalid XOR matching")
        pairs.append((left, right))
        unseen.remove(left)
        unseen.remove(right)
    require(len(pairs) == 4, "XOR matching does not have four pairs")
    return tuple(pairs)


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "raw96_beats": 0,
        "nonzero96_descriptors": 0,
        "zero_tokens": 0,
        "windows": 0,
        "optimal_groups": 0,
        "optimal_replay_cycles": 0,
        "optimal_wall_cycles": 0,
        "k8_groups": 0,
        "k8_replay_cycles": 0,
        "k8_wall_cycles": 0,
    }
    for mask in XOR_MASKS:
        result["xor{}_groups".format(mask)] = 0
        result["xor{}_replay_cycles".format(mask)] = 0
        result["xor{}_wall_cycles".format(mask)] = 0
    return result


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def audit_record(record, payload_root, m172, m179, chunk_tokens):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    require((shape[-1], output_shape[-1]) == STAGE_GEOMETRY[stage],
            "FC2 geometry drift")
    output_blocks = output_shape[-1] // WIDTH
    beats_per_token = shape[-1] // WIDTH
    bytes_per_token = shape[-1] // 8
    tokens = int(np.prod(shape[:-1], dtype=np.int64))
    depth = STAGE_DEPTH[stage]
    maximum_windows = (beats_per_token + depth - 1) // depth
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
        pooled = np.zeros(
            (stop - start, maximum_windows, 8), dtype=np.int16
        )
        if row.size:
            window = (positions[row, beat] // depth).astype(np.intp)
            np.add.at(pooled, (row, window), bank_counts[row, beat])
        window_count = (
            descriptor_count.astype(np.int64) + depth - 1
        ) // depth
        events = pooled.sum(axis=2, dtype=np.int16)
        optimal_groups = np.maximum(
            pooled.max(axis=2), (events + 3) // 4
        ).astype(np.int64)
        optimal_wall = m179.finite_dual_window_wall(
            optimal_groups, descriptor_count, depth, output_blocks
        )
        k8_groups = pooled.max(axis=2).astype(np.int64)
        k8_wall = m179.finite_dual_window_wall(
            k8_groups, descriptor_count, depth, output_blocks
        )
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["nonzero96_descriptors"] += int(
            descriptor_count.sum(dtype=np.int64)
        )
        ledger["zero_tokens"] += int(
            np.count_nonzero(descriptor_count == 0)
        )
        ledger["windows"] += int(window_count.sum(dtype=np.int64))
        ledger["optimal_groups"] += int(
            optimal_groups.sum(dtype=np.int64)
        )
        ledger["optimal_replay_cycles"] += int(
            optimal_groups.sum(dtype=np.int64)
        ) * output_blocks
        ledger["optimal_wall_cycles"] += int(
            optimal_wall.sum(dtype=np.int64)
        )
        ledger["k8_groups"] += int(k8_groups.sum(dtype=np.int64))
        ledger["k8_replay_cycles"] += int(
            k8_groups.sum(dtype=np.int64)
        ) * output_blocks
        ledger["k8_wall_cycles"] += int(
            k8_wall.sum(dtype=np.int64)
        )
        for mask in XOR_MASKS:
            pair_populations = np.stack([
                pooled[:, :, left] + pooled[:, :, right]
                for left, right in xor_pairs(mask)
            ], axis=2)
            paired_groups = pair_populations.max(axis=2).astype(np.int64)
            paired_wall = m179.finite_dual_window_wall(
                paired_groups, descriptor_count, depth, output_blocks
            )
            group_sum = int(paired_groups.sum(dtype=np.int64))
            ledger["xor{}_groups".format(mask)] += group_sum
            ledger["xor{}_replay_cycles".format(mask)] += (
                group_sum * output_blocks
            )
            ledger["xor{}_wall_cycles".format(mask)] += int(
                paired_wall.sum(dtype=np.int64)
            )
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return stage, ledger


def enrich(ledger):
    result = dict(ledger)
    for mask in XOR_MASKS:
        wall = result["xor{}_wall_cycles".format(mask)]
        result["xor{}_over_optimal_wall_ratio".format(mask)] = fraction(
            wall, result["optimal_wall_cycles"]
        )
        result["d1_k4_over_xor{}_wall_ratio".format(mask)] = fraction(
            result["d1_k4_wall_cycles"], wall
        ) if "d1_k4_wall_cycles" in result else None
    return result


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
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256, "m172_pinned_m181"
    )
    m179 = load_pinned(
        args.m179_analyzer, EXPECTED_M179_ANALYZER_SHA256, "m179_pinned_m181"
    )
    with args.m179_result.open("r", encoding="utf-8") as handle:
        m179_result = json.load(handle)
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
        print("[M181] {}/120".format(ordinal + 1), flush=True)
    require(aggregate["events"] == 143894510, "event identity drift")
    require(aggregate["raw96_beats"] == 36480000,
            "raw beat identity drift")
    expected_optimal = int(
        m179_result["stage_adaptive_selection"]["matched_k4_wall_cycles"]
    )
    require(aggregate["optimal_wall_cycles"] == expected_optimal,
            "M179 selected K4 recurrence drift")
    aggregate["d1_k4_wall_cycles"] = 144146504
    per_stage_result = {}
    for stage in STAGE_GEOMETRY:
        stage_ledger = per_stage[stage]
        stage_ledger["d1_k4_wall_cycles"] = int(
            m179_result["per_stage"][str(stage)]["wall_cycles_d1_k4"]
        )
        per_stage_result[str(stage)] = enrich(stage_ledger)
    aggregate_result = enrich(aggregate)
    global_best_mask = min(
        XOR_MASKS,
        key=lambda mask: aggregate["xor{}_wall_cycles".format(mask)],
    )
    per_stage_best = {}
    stage_adaptive_wall = 0
    for stage in STAGE_GEOMETRY:
        ledger = per_stage[stage]
        best_mask = min(
            XOR_MASKS,
            key=lambda mask: ledger["xor{}_wall_cycles".format(mask)],
        )
        best_wall = ledger["xor{}_wall_cycles".format(best_mask)]
        per_stage_best[str(stage)] = {
            "depth": STAGE_DEPTH[stage],
            "xor_mask": best_mask,
            "pairs": [list(pair) for pair in xor_pairs(best_mask)],
            "wall_cycles": best_wall,
            "paired_over_optimal_wall_ratio": fraction(
                best_wall, ledger["optimal_wall_cycles"]
            ),
        }
        stage_adaptive_wall += best_wall
    fixed_wall = aggregate[
        "xor{}_wall_cycles".format(PREDECLARED_MASK)
    ]
    result = {
        "schema": "m181_h67_fc2_xor_pair_reservoir_exact_payload_dse_v1",
        "status": "PASS_EXACT_PAYLOAD_HARDWARE_SIMPLE_PAIRING_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m179_analyzer_sha256": EXPECTED_M179_ANALYZER_SHA256,
            "m179_result_sha256": EXPECTED_M179_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
        },
        "architecture": {
            "stage_depths": [STAGE_DEPTH[stage] for stage in STAGE_GEOMETRY],
            "ping_pong_windows": 2,
            "fixed_lanes": 4,
            "banks_per_lane": 2,
            "lane_schedule": "drain one event per cycle from either fixed-pair member; no global bank sorter",
            "xor_masks_evaluated": list(XOR_MASKS),
            "predeclared_mask": PREDECLARED_MASK,
            "predeclared_pairs": [
                list(pair) for pair in xor_pairs(PREDECLARED_MASK)
            ],
            "native_or_preindexed_nonzero96_source": True,
            "token_directory_count_closes_partial_window": True,
            "posthoc_scanner": False,
        },
        "aggregate": aggregate_result,
        "per_stage": per_stage_result,
        "selection": {
            "global_best_xor_mask_in_sample": global_best_mask,
            "global_best_pairs": [
                list(pair) for pair in xor_pairs(global_best_mask)
            ],
            "global_best_wall_cycles": aggregate[
                "xor{}_wall_cycles".format(global_best_mask)
            ],
            "per_stage_best_in_sample": per_stage_best,
            "per_stage_best_wall_cycles": stage_adaptive_wall,
            "selection_holdout": False,
        },
        "predeclared_xor4": {
            "wall_cycles": fixed_wall,
            "optimal_top4_wall_cycles": expected_optimal,
            "paired_over_optimal_wall_ratio": fraction(
                fixed_wall, expected_optimal
            ),
            "d1_k4_over_paired_wall_ratio": fraction(144146504, fixed_wall),
            "optimized_k1_over_paired_wall_ratio": fraction(
                424060394, fixed_wall
            ),
        },
        "same_depth_k8_scaling": {
            "stage_depths": [
                STAGE_DEPTH[stage] for stage in STAGE_GEOMETRY
            ],
            "wall_cycles": aggregate["k8_wall_cycles"],
            "k4_top4_wall_cycles": expected_optimal,
            "k4_over_k8_wall_ratio": fraction(
                expected_optimal, aggregate["k8_wall_cycles"]
            ),
            "d1_k4_over_k8_wall_ratio": fraction(
                144146504, aggregate["k8_wall_cycles"]
            ),
            "optimized_k1_over_k8_wall_ratio": fraction(
                424060394, aggregate["k8_wall_cycles"]
            ),
            "qualification": "K8 uses the K4-selected D={2,4,8,8}; K8 depth is not independently optimized in this screen",
            "fixed_bank_lanes_remove_global_top4_bank_sort": True,
            "eight_weight_banks_and_eight_accumulator_lanes_required": True
        },
        "claim_boundary": {
            "exact_payload_finite_buffer_cycle_dse": True,
            "hardware_simple_schedule_constructive": True,
            "rtl": False,
            "producer_or_directory_rtl": False,
            "window_storage_rtl": False,
            "weight_sram_response": False,
            "arithmetic_composed": False,
            "complete_fc2": False,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "docs359_sha256_unchanged": EXPECTED_DOCS359_SHA256,
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "analyzer mutated during run")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS M181 xor4/optimal={:.6f} D1/xor4={:.6f} K1/xor4={:.6f} K1/K8={:.6f} global_best_mask={}".format(
            result["predeclared_xor4"]["paired_over_optimal_wall_ratio"]["float"],
            result["predeclared_xor4"]["d1_k4_over_paired_wall_ratio"]["float"],
            result["predeclared_xor4"]["optimized_k1_over_paired_wall_ratio"]["float"],
            result["same_depth_k8_scaling"]["optimized_k1_over_k8_wall_ratio"]["float"],
            global_best_mask,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
