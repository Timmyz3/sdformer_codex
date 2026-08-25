#!/usr/bin/env python3
"""Exact H67 FC2 DSE for bounded dual-window cross-beat bank packing.

Each native/preindexed nonzero96 descriptor is placed into one of two ping-pong
windows.  A closed window pools at most D descriptors and emits bank-unique
groups.  Fill is one descriptor per cycle, drain is one group result per output
block cycle, and a window cannot be refilled until its prior drain finishes.
The token directory supplies descriptor count, so a partial final window is
closed without a serialized EOT.  K1 and K4 use identical windows and fill
schedule.  This is an executable finite-buffer cycle DSE, not RTL or memory PPA.
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
EXPECTED_M176_ANALYZER_SHA256 = (
    "94cf1d234f2af581f7649a06c070ba14857f513f28591cd852d5ae184d6b7456"
)
STAGE_GEOMETRY = {
    0: (384, 96),
    1: (768, 192),
    2: (1536, 384),
    3: (3072, 768),
}
WINDOW_POINTS = (1, 2, 4, 8, 16, 32)
K_POINTS = (1, 4)
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


def finite_dual_window_wall(
        window_groups, descriptor_count, window_size, output_blocks):
    """Vector edge schedule for two alternating fill/drain windows."""
    groups = np.asarray(window_groups, dtype=np.int64)
    descriptor_count = np.asarray(descriptor_count, dtype=np.int16)
    token_count, maximum_windows = groups.shape
    window_count = (
        descriptor_count.astype(np.int64) + window_size - 1
    ) // window_size
    fill_end = np.zeros(token_count, dtype=np.int64)
    drain_end = np.zeros(token_count, dtype=np.int64)
    buffer_free = [
        np.zeros(token_count, dtype=np.int64),
        np.zeros(token_count, dtype=np.int64),
    ]
    for window in range(maximum_windows):
        active = window < window_count
        entries = np.minimum(
            window_size,
            descriptor_count.astype(np.int64) - window * window_size,
        )
        entries = np.maximum(entries, 0)
        next_fill_end = np.maximum(
            fill_end, buffer_free[window & 1]
        ) + entries
        next_drain_end = np.maximum(
            next_fill_end, drain_end
        ) + groups[:, window] * output_blocks
        fill_end = np.where(active, next_fill_end, fill_end)
        drain_end = np.where(active, next_drain_end, drain_end)
        buffer_free[window & 1] = np.where(
            active, next_drain_end, buffer_free[window & 1]
        )
    return np.where(window_count > 0, drain_end + 1, 2)


def scalar_dual_window_wall(
        window_groups, descriptor_count, window_size, output_blocks):
    if descriptor_count == 0:
        return 2
    windows = (descriptor_count + window_size - 1) // window_size
    fill_end = 0
    drain_end = 0
    buffer_free = [0, 0]
    for window in range(windows):
        entries = min(window_size, descriptor_count - window * window_size)
        fill_end = max(fill_end, buffer_free[window & 1]) + entries
        drain_end = max(fill_end, drain_end) \
            + int(window_groups[window]) * output_blocks
        buffer_free[window & 1] = drain_end
    return drain_end + 1


def verify_recurrence(m176):
    rng = np.random.default_rng(179)
    cases = 0
    d1_matches = 0
    for output_blocks in (1, 2, 4, 8):
        for maximum_descriptors in range(1, 33):
            for _ in range(100):
                descriptor_count = int(
                    rng.integers(0, maximum_descriptors + 1)
                )
                descriptor_groups = rng.integers(
                    1, 13, size=descriptor_count, dtype=np.int16
                )
                for window_size in WINDOW_POINTS:
                    windows = max(
                        1,
                        (descriptor_count + window_size - 1) // window_size,
                    )
                    window_groups = np.zeros((1, windows), dtype=np.int64)
                    for window in range(windows):
                        left = window * window_size
                        right = min(descriptor_count, left + window_size)
                        window_groups[0, window] = int(
                            descriptor_groups[left:right].sum()
                        )
                    vector = int(finite_dual_window_wall(
                        window_groups,
                        np.asarray([descriptor_count]),
                        window_size,
                        output_blocks,
                    )[0])
                    scalar = scalar_dual_window_wall(
                        window_groups[0], descriptor_count,
                        window_size, output_blocks
                    )
                    require(vector == scalar, "finite recurrence mismatch")
                    cases += 1
                    if window_size == 1:
                        padded = descriptor_groups.reshape(1, -1)
                        if descriptor_count == 0:
                            reference = 2
                        else:
                            reference = int(m176.variable_wall_cycles(
                                padded,
                                np.asarray([descriptor_count]),
                                output_blocks,
                            )[0])
                        require(vector == reference,
                                "D1 versus M176 recurrence mismatch")
                        d1_matches += 1
    return cases, d1_matches


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "raw96_beats": 0,
        "nonzero96_descriptors": 0,
        "zero_tokens": 0,
    }
    for window_size in WINDOW_POINTS:
        result["window_count_d{}".format(window_size)] = 0
        for k_value in K_POINTS:
            result.update({
                "groups_d{}_k{}".format(window_size, k_value): 0,
                "replay_cycles_d{}_k{}".format(window_size, k_value): 0,
                "wall_cycles_d{}_k{}".format(window_size, k_value): 0,
            })
    return result


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def enrich(ledger):
    result = dict(ledger)
    d1_k4 = result["wall_cycles_d1_k4"]
    for window_size in WINDOW_POINTS:
        result["d1_over_d{}_wall_ratio_k4".format(window_size)] = fraction(
            d1_k4, result["wall_cycles_d{}_k4".format(window_size)]
        )
        result["k1_over_k4_wall_ratio_d{}".format(window_size)] = fraction(
            result["wall_cycles_d{}_k1".format(window_size)],
            result["wall_cycles_d{}_k4".format(window_size)],
        )
        result["fill_drain_overhead_cycles_d{}_k4".format(window_size)] = (
            result["wall_cycles_d{}_k4".format(window_size)]
            - result["replay_cycles_d{}_k4".format(window_size)]
        )
    return result


def audit_record(record, payload_root, m172, chunk_tokens):
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
    require(payload.is_file() and payload.stat().st_size == record["packed_bytes"],
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
        ledger["zero_tokens"] += int(np.count_nonzero(descriptor_count == 0))
        for window_size in WINDOW_POINTS:
            maximum_windows = (
                beats_per_token + window_size - 1
            ) // window_size
            pooled = np.zeros(
                (stop - start, maximum_windows, 8), dtype=np.int16
            )
            if row.size:
                window = (
                    positions[row, beat] // window_size
                ).astype(np.intp)
                np.add.at(pooled, (row, window), bank_counts[row, beat])
            window_events = pooled.sum(axis=2, dtype=np.int16)
            maximum_bank = pooled.max(axis=2)
            window_count = (
                descriptor_count.astype(np.int64) + window_size - 1
            ) // window_size
            ledger["window_count_d{}".format(window_size)] += int(
                window_count.sum(dtype=np.int64)
            )
            for k_value in K_POINTS:
                groups = np.maximum(
                    maximum_bank,
                    (window_events + (k_value - 1)) // k_value,
                ).astype(np.int64)
                group_sum = int(groups.sum(dtype=np.int64))
                wall = finite_dual_window_wall(
                    groups, descriptor_count, window_size, output_blocks
                )
                ledger["groups_d{}_k{}".format(
                    window_size, k_value
                )] += group_sum
                ledger["replay_cycles_d{}_k{}".format(
                    window_size, k_value
                )] += group_sum * output_blocks
                ledger["wall_cycles_d{}_k{}".format(
                    window_size, k_value
                )] += int(wall.sum(dtype=np.int64))
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return stage, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m176-analyzer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    m172 = load_pinned(
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256, "m172_pinned_m179"
    )
    m176 = load_pinned(
        args.m176_analyzer, EXPECTED_M176_ANALYZER_SHA256, "m176_pinned_m179"
    )
    recurrence_cases, d1_matches = verify_recurrence(m176)
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
            record, args.payload_root, m172, args.chunk_tokens
        )
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print("[M179] {}/120".format(ordinal + 1), flush=True)
    require(aggregate["events"] == 143894510, "event identity drift")
    require(aggregate["raw96_beats"] == 36480000,
            "raw beat identity drift")
    require(aggregate["wall_cycles_d1_k1"] == 424060394,
            "M176 D1 K1 wall identity drift")
    require(aggregate["wall_cycles_d1_k4"] == 144146504,
            "M176 D1 K4 wall identity drift")
    stage_enriched = {
        str(stage): enrich(per_stage[stage]) for stage in STAGE_GEOMETRY
    }
    selected = {}
    selected_k1 = 0
    selected_k4 = 0
    selected_replay_k4 = 0
    for stage in STAGE_GEOMETRY:
        stage_result = stage_enriched[str(stage)]
        best_window = min(
            WINDOW_POINTS,
            key=lambda value: (
                stage_result["wall_cycles_d{}_k4".format(value)], value
            ),
        )
        stage_k1 = stage_result["wall_cycles_d{}_k1".format(best_window)]
        stage_k4 = stage_result["wall_cycles_d{}_k4".format(best_window)]
        stage_replay = stage_result[
            "replay_cycles_d{}_k4".format(best_window)
        ]
        selected[str(stage)] = {
            "window_descriptors_per_buffer": best_window,
            "two_buffer_storage_descriptors": 2 * best_window,
            "k1_wall_cycles": stage_k1,
            "k4_wall_cycles": stage_k4,
            "k4_replay_cycles": stage_replay,
            "d1_over_selected_k4_wall_ratio": fraction(
                stage_result["wall_cycles_d1_k4"], stage_k4
            ),
            "matched_selected_k1_over_k4_wall_ratio": fraction(
                stage_k1, stage_k4
            ),
        }
        selected_k1 += stage_k1
        selected_k4 += stage_k4
        selected_replay_k4 += stage_replay
    result = {
        "schema": "m179_h67_fc2_dual_window_reservoir_exact_payload_dse_v1",
        "status": "PASS_EXACT_PAYLOAD_FINITE_DUAL_WINDOW_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m176_analyzer_sha256": EXPECTED_M176_ANALYZER_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
            "vector_vs_scalar_recurrence_cases": recurrence_cases,
            "d1_vs_m176_recurrence_cases": d1_matches,
            "recurrence_mismatches": 0,
        },
        "architecture": {
            "native_or_preindexed_nonzero96_source": True,
            "window_points": list(WINDOW_POINTS),
            "ping_pong_windows": 2,
            "fill_descriptors_per_cycle": 1,
            "drain_group_results_per_cycle": 1,
            "window_reuse_waits_for_prior_drain": True,
            "token_directory_count_closes_partial_window": True,
            "posthoc_scanner": False,
        },
        "aggregate": enrich(aggregate),
        "per_stage": stage_enriched,
        "stage_adaptive_selection": {
            "per_stage": selected,
            "matched_k1_wall_cycles": selected_k1,
            "matched_k4_wall_cycles": selected_k4,
            "k4_replay_cycles": selected_replay_k4,
            "d1_k4_wall_cycles": aggregate["wall_cycles_d1_k4"],
            "d1_over_selected_k4_wall_ratio": fraction(
                aggregate["wall_cycles_d1_k4"], selected_k4
            ),
            "matched_selected_k1_over_k4_wall_ratio": fraction(
                selected_k1, selected_k4
            ),
            "maximum_descriptors_per_buffer": max(
                value["window_descriptors_per_buffer"]
                for value in selected.values()
            ),
            "maximum_two_buffer_payload_bits_without_metadata":
                2 * max(
                    value["window_descriptors_per_buffer"]
                    for value in selected.values()
                ) * WIDTH,
        },
        "claim_boundary": {
            "exact_payload_finite_buffer_cycle_dse": True,
            "rtl": False,
            "producer_or_directory_rtl": False,
            "window_sram_or_register_packing": False,
            "weight_sram_response": False,
            "arithmetic_composed": False,
            "complete_fc2": False,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "docs359_sha256_unchanged":
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "analyzer mutated during run")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS M179 D1/selected K4={:.6f} matched K1/K4={:.6f} windows={}".format(
            result["stage_adaptive_selection"]["d1_over_selected_k4_wall_ratio"]["float"],
            result["stage_adaptive_selection"]["matched_selected_k1_over_k4_wall_ratio"]["float"],
            [selected[str(stage)]["window_descriptors_per_buffer"]
             for stage in STAGE_GEOMETRY],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
