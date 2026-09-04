#!/usr/bin/env python3
"""Exact H67 FC2 K8 dual-window depth DSE on frozen binary payloads.

Eight fixed bank lanes remove M180's global top-four sort: one event from each
nonempty bank can issue per source group.  For every D in {1,2,4,8,16,32}, the
window drain groups are therefore the maximum bank population.  Fill, drain,
two-buffer reuse, partial-window close and zero-token timing are identical to
the independently checked M179 recurrence.

This selects K8 window depths in sample.  It does not include the descriptor
producer, directory, physical weight banks, accumulators, power or system PPA.
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
DEPTH_POINTS = (1, 2, 4, 8, 16, 32)
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


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "raw96_beats": 0,
        "nonzero96_descriptors": 0,
        "zero_tokens": 0,
    }
    for depth in DEPTH_POINTS:
        result["windows_d{}".format(depth)] = 0
        result["groups_d{}_k8".format(depth)] = 0
        result["replay_cycles_d{}_k8".format(depth)] = 0
        result["wall_cycles_d{}_k8".format(depth)] = 0
    return result


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def enrich(ledger):
    result = dict(ledger)
    for depth in DEPTH_POINTS:
        wall = result["wall_cycles_d{}_k8".format(depth)]
        result["optimized_k1_over_d{}_k8".format(depth)] = fraction(
            424060394, wall
        )
        result["m179_k4_over_d{}_k8".format(depth)] = fraction(
            127581198, wall
        )
        result["fill_drain_overhead_d{}_k8".format(depth)] = (
            wall - result["replay_cycles_d{}_k8".format(depth)]
        )
    return result


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
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["nonzero96_descriptors"] += int(
            descriptor_count.sum(dtype=np.int64)
        )
        ledger["zero_tokens"] += int(
            np.count_nonzero(descriptor_count == 0)
        )
        for depth in DEPTH_POINTS:
            maximum_windows = (beats_per_token + depth - 1) // depth
            pooled = np.zeros(
                (stop - start, maximum_windows, 8), dtype=np.int16
            )
            if row.size:
                window = (positions[row, beat] // depth).astype(np.intp)
                np.add.at(pooled, (row, window), bank_counts[row, beat])
            groups = pooled.max(axis=2).astype(np.int64)
            walls = m179.finite_dual_window_wall(
                groups, descriptor_count, depth, output_blocks
            )
            window_count = (
                descriptor_count.astype(np.int64) + depth - 1
            ) // depth
            group_sum = int(groups.sum(dtype=np.int64))
            ledger["windows_d{}".format(depth)] += int(
                window_count.sum(dtype=np.int64)
            )
            ledger["groups_d{}_k8".format(depth)] += group_sum
            ledger["replay_cycles_d{}_k8".format(depth)] += (
                group_sum * output_blocks
            )
            ledger["wall_cycles_d{}_k8".format(depth)] += int(
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
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256, "m172_pinned_m182"
    )
    m179 = load_pinned(
        args.m179_analyzer, EXPECTED_M179_ANALYZER_SHA256, "m179_pinned_m182"
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
        print("[M182] {}/120".format(ordinal + 1), flush=True)
    require(aggregate["events"] == 143894510, "event identity drift")
    require(aggregate["raw96_beats"] == 36480000,
            "raw beat identity drift")
    # M181's same-depth K8 value is an exact cross-check at D={2,4,8,8}.
    m181_same_depth = (
        per_stage[0]["wall_cycles_d2_k8"]
        + per_stage[1]["wall_cycles_d4_k8"]
        + per_stage[2]["wall_cycles_d8_k8"]
        + per_stage[3]["wall_cycles_d8_k8"]
    )
    require(m181_same_depth == 97607807, "M181 K8 cross-check drift")
    selected = {}
    selected_wall = 0
    for stage in STAGE_GEOMETRY:
        ledger = per_stage[stage]
        ordered = sorted(
            DEPTH_POINTS,
            key=lambda depth: (
                ledger["wall_cycles_d{}_k8".format(depth)], depth
            ),
        )
        best = ordered[0]
        runner_up = ordered[1]
        best_wall = ledger["wall_cycles_d{}_k8".format(best)]
        runner_wall = ledger["wall_cycles_d{}_k8".format(runner_up)]
        selected[str(stage)] = {
            "depth": best,
            "wall_cycles": best_wall,
            "runner_up_depth": runner_up,
            "runner_up_wall_cycles": runner_wall,
            "runner_up_margin_percent":
                100.0 * (runner_wall - best_wall) / best_wall,
        }
        selected_wall += best_wall
    result = {
        "schema": "m182_h67_fc2_k8_dual_window_depth_exact_payload_dse_v1",
        "status": "PASS_EXACT_PAYLOAD_K8_DEPTH_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m179_analyzer_sha256": EXPECTED_M179_ANALYZER_SHA256,
            "m179_result_sha256": EXPECTED_M179_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
            "m181_same_depth_k8_crosscheck_wall_cycles": m181_same_depth,
        },
        "architecture": {
            "fixed_bank_lanes": 8,
            "maximum_one_source_per_bank_per_group": True,
            "global_top4_sorter": False,
            "depth_points": list(DEPTH_POINTS),
            "ping_pong_windows": 2,
            "fill_descriptors_per_cycle": 1,
            "drain_group_results_per_cycle": 1,
            "native_or_preindexed_nonzero96_source": True,
            "token_directory_count_closes_partial_window": True,
        },
        "aggregate": enrich(aggregate),
        "per_stage": {
            str(stage): enrich(per_stage[stage]) for stage in STAGE_GEOMETRY
        },
        "stage_adaptive_selection_in_sample": {
            "per_stage": selected,
            "wall_cycles": selected_wall,
            "m181_same_depth_wall_cycles": m181_same_depth,
            "m181_same_depth_over_selected": fraction(
                m181_same_depth, selected_wall
            ),
            "m179_selected_k4_wall_cycles": int(
                m179_result["stage_adaptive_selection"]
                    ["matched_k4_wall_cycles"]
            ),
            "m179_selected_k4_over_selected_k8": fraction(
                int(m179_result["stage_adaptive_selection"]
                    ["matched_k4_wall_cycles"]), selected_wall
            ),
            "optimized_k1_wall_cycles": 424060394,
            "optimized_k1_over_selected_k8": fraction(
                424060394, selected_wall
            ),
            "selection_holdout": False,
        },
        "claim_boundary": {
            "exact_payload_finite_buffer_cycle_dse": True,
            "rtl": False,
            "eight_weight_bank_response": False,
            "eight_lane_accumulator": False,
            "native_descriptor_producer": False,
            "token_directory_generation": False,
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
        "PASS M182 K1/K8={:.6f} K4/K8={:.6f} depths={}".format(
            result["stage_adaptive_selection_in_sample"]
                ["optimized_k1_over_selected_k8"]["float"],
            result["stage_adaptive_selection_in_sample"]
                ["m179_selected_k4_over_selected_k8"]["float"],
            [selected[str(stage)]["depth"] for stage in STAGE_GEOMETRY],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
