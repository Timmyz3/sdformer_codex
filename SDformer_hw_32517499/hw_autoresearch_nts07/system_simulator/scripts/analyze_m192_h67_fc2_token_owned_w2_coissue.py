#!/usr/bin/env python3
"""Exact token-aware W2 FC2 bank co-issue audit on frozen H67 payloads.

M191 proved an ideal adjacent-window W2 replay opportunity but accidentally
called every window an independent context.  This successor preserves the
owning token of every window and separates two implementable contracts:

* one token update per cycle: two windows may co-issue only when they belong
  to the same token and therefore share one Acc24;
* two token updates per cycle: adjacent windows may also belong to different
  tokens, requiring two independently reduced Acc24 updates.

For W2, a cross-token batch contains one window from each token.  Its exact
one-update service is the sum of the two per-window bank maxima.  A same-token
batch may merge both bank populations and completes in the maximum combined
bank population.  The audit performs no window reordering and flushes at each
frozen FC2 record boundary.
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
EXPECTED_M191_RESULT_SHA256 = (
    "7ea3b46f78b321e319ed1b9aa0f9377c6f4fe18c583e6152c856d8b66ab7e2fb"
)
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
EXPECTED_EVENTS = 143894510
EXPECTED_TOKENS = 5580000
EXPECTED_WINDOWS = 6523707
EXPECTED_W1_CYCLES = 79397844
EXPECTED_IDEAL_W2_CYCLES = 71233088
WIDTH = 96
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
    return {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "windows": 0,
        "full_w2_batches": 0,
        "partial_w2_batches": 0,
        "same_token_w2_batches": 0,
        "cross_token_w2_batches": 0,
        "w1_replay_cycles": 0,
        "same_token_only_coissue_cycles": 0,
        "ideal_dual_token_coissue_cycles": 0,
        "cycles_requiring_second_token_update": 0,
        "replayed_event_terms": 0,
    }


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def account_pairs(ledger, pending, windows, output_blocks):
    combined = pending + windows
    complete = (len(combined) // 2) * 2
    if not complete:
        return combined
    selected = combined[:complete]
    token_ids = np.asarray([item[0] for item in selected], dtype=np.int64)
    loads = np.asarray([item[1] for item in selected], dtype=np.int64)
    token_ids = token_ids.reshape(-1, 2)
    loads = loads.reshape(-1, 2, 8)
    first_cycles = loads[:, 0, :].max(axis=1)
    second_cycles = loads[:, 1, :].max(axis=1)
    sequential = first_cycles + second_cycles
    combined_cycles = loads.sum(axis=1, dtype=np.int64).max(axis=1)
    same = token_ids[:, 0] == token_ids[:, 1]
    one_update = np.where(same, combined_cycles, sequential)
    ideal_dual = combined_cycles
    require(bool(np.all(ideal_dual <= one_update)),
            "dual-token service exceeded one-update service")
    require(bool(np.all(one_update <= sequential)),
            "same-token merge exceeded W1 service")
    scale = int(output_blocks)
    ledger["full_w2_batches"] += int(loads.shape[0])
    ledger["same_token_w2_batches"] += int(same.sum(dtype=np.int64))
    ledger["cross_token_w2_batches"] += int((~same).sum(dtype=np.int64))
    ledger["w1_replay_cycles"] += int(sequential.sum(dtype=np.int64)) * scale
    ledger["same_token_only_coissue_cycles"] += int(
        one_update.sum(dtype=np.int64)
    ) * scale
    ledger["ideal_dual_token_coissue_cycles"] += int(
        ideal_dual.sum(dtype=np.int64)
    ) * scale
    ledger["cycles_requiring_second_token_update"] += int(
        (one_update - ideal_dual).sum(dtype=np.int64)
    ) * scale
    return combined[complete:]


def flush_partial(ledger, pending, output_blocks):
    if not pending:
        return
    require(len(pending) == 1, "W2 partial extent drift")
    cycles = int(np.asarray(pending[0][1], dtype=np.int64).max())
    ledger["partial_w2_batches"] += 1
    ledger["w1_replay_cycles"] += cycles * output_blocks
    ledger["same_token_only_coissue_cycles"] += cycles * output_blocks
    ledger["ideal_dual_token_coissue_cycles"] += cycles * output_blocks


def audit_record(record, payload_root, m172, chunk_tokens):
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
    ledger["records"] = 1
    ledger["tokens"] = tokens
    pending = []
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
        windows = []
        for token_offset, count in enumerate(window_count):
            token_id = start + token_offset
            for window_index in range(int(count)):
                loads = pooled[token_offset, window_index]
                windows.append((token_id, loads))
                ledger["replayed_event_terms"] += int(
                    loads.sum(dtype=np.int64)
                ) * output_blocks
        pending = account_pairs(ledger, pending, windows, output_blocks)
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["windows"] += int(window_count.sum(dtype=np.int64))
    flush_partial(ledger, pending, output_blocks)
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return stage, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m191-result", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    require(sha256(args.m191_result) == EXPECTED_M191_RESULT_SHA256,
            "M191 result identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359_SHA256,
            "docs359 identity drift")
    m172 = load_pinned(
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256,
        "m172_pinned_m192"
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
            record, args.payload_root, m172, args.chunk_tokens
        )
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print("[M192] {}/120".format(ordinal + 1), flush=True)

    require(aggregate["events"] == EXPECTED_EVENTS, "event identity drift")
    require(aggregate["tokens"] == EXPECTED_TOKENS, "token identity drift")
    require(aggregate["windows"] == EXPECTED_WINDOWS,
            "window identity drift")
    require(aggregate["w1_replay_cycles"] == EXPECTED_W1_CYCLES,
            "W1 replay cross-check drift")
    require(aggregate["ideal_dual_token_coissue_cycles"]
            == EXPECTED_IDEAL_W2_CYCLES,
            "M191 ideal W2 cross-check drift")
    require(aggregate["replayed_event_terms"] == 412900394,
            "replayed event-term identity drift")

    one_update_cycles = aggregate["same_token_only_coissue_cycles"]
    ideal_cycles = aggregate["ideal_dual_token_coissue_cycles"]
    result = {
        "schema": "m192_h67_fc2_token_owned_w2_coissue_v1",
        "status": "PASS_EXACT_TOKEN_OWNED_W2_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m191_result_sha256": EXPECTED_M191_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
        },
        "aggregate": aggregate,
        "exact_points": {
            "w1_sequential": {
                "cycles": EXPECTED_W1_CYCLES,
                "speed_vs_w1": fraction(EXPECTED_W1_CYCLES,
                                         EXPECTED_W1_CYCLES),
            },
            "w2_same_token_fusion_one_acc24_update": {
                "cycles": one_update_cycles,
                "speed_vs_w1": fraction(EXPECTED_W1_CYCLES,
                                         one_update_cycles),
                "extra_acc24_contexts": 0,
            },
            "w2_ideal_dual_token_update": {
                "cycles": ideal_cycles,
                "speed_vs_w1": fraction(EXPECTED_W1_CYCLES, ideal_cycles),
                "incremental_speed_vs_same_token_only": fraction(
                    one_update_cycles, ideal_cycles
                ),
                "cycles_requiring_a_second_token_update": aggregate[
                    "cycles_requiring_second_token_update"
                ],
            },
        },
        "batch_identity": {
            "same_token_fraction": fraction(
                aggregate["same_token_w2_batches"],
                aggregate["full_w2_batches"]
            ),
            "cross_token_fraction": fraction(
                aggregate["cross_token_w2_batches"],
                aggregate["full_w2_batches"]
            ),
        },
        "per_stage": {
            str(stage): {
                **per_stage[stage],
                "same_token_only_speed_vs_w1": fraction(
                    per_stage[stage]["w1_replay_cycles"],
                    per_stage[stage]["same_token_only_coissue_cycles"]
                ),
                "ideal_dual_token_speed_vs_w1": fraction(
                    per_stage[stage]["w1_replay_cycles"],
                    per_stage[stage]["ideal_dual_token_coissue_cycles"]
                ),
            }
            for stage in STAGE_GEOMETRY
        },
        "architecture_decision": {
            "first_rtl_candidate": "same-token adjacent-window fusion",
            "reason": (
                "It captures token-owned bank complementarity without a "
                "second Acc24 context; cross-token batches fall back to W1."
            ),
            "dual_token_candidate": "hold until matched state/port area is charged",
        },
        "claim_boundary": {
            "exact_payload_replay_cycles": True,
            "token_identity_preserved": True,
            "finite_fill_wall_cycles": False,
            "rtl_measured_speedup": False,
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
