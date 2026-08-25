#!/usr/bin/env python3
"""Exact H67 FC2 token-flush adjacent-window fusion replay audit.

M192 preserved token IDs but kept a global W2 pairing phase across token
boundaries.  A cross-token pair then fell back to sequential service and also
shifted the pairing phase of the following token.  Real token-owned hardware
does not need that constraint: it flushes an odd tail at token completion and
starts the next token at window zero.  Consequently every full pair owns one
Acc24 and may merge its per-bank queues without a second context.

This analyzer pins M192's decoder/helpers but replaces its pairing policy.
It does not reorder windows within a token and flushes at every token boundary.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_M192_ANALYZER_SHA256 = (
    "39715b64890d75be7c60587d639f3e75b51e5bf38642e6b8640c761fa512f24b"
)
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
EXPECTED_GLOBAL_IDEAL_W2_CYCLES = 71233088
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
    ledger = m192.empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    zero_window_tokens = 0
    even_nonzero_window_tokens = 0
    odd_window_tokens = 0
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
            if count == 0:
                zero_window_tokens += 1
                continue
            if count % 2:
                odd_window_tokens += 1
            else:
                even_nonzero_window_tokens += 1
            token_id = start + token_offset
            token_windows = []
            for window_index in range(count):
                loads = pooled[token_offset, window_index]
                token_windows.append((token_id, loads))
                ledger["replayed_event_terms"] += int(
                    loads.sum(dtype=np.int64)
                ) * output_blocks
            pending = m192.account_pairs(
                ledger, [], token_windows, output_blocks
            )
            m192.flush_partial(ledger, pending, output_blocks)
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["windows"] += int(window_count.sum(dtype=np.int64))
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    require(ledger["cross_token_w2_batches"] == 0,
            "token-flush emitted a cross-token pair")
    return stage, ledger, {
        "zero_window_tokens": zero_window_tokens,
        "even_nonzero_window_tokens": even_nonzero_window_tokens,
        "odd_window_tokens": odd_window_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m192-analyzer", required=True, type=Path)
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
        "m172_pinned_m195"
    )
    m192 = load_pinned(
        args.m192_analyzer, EXPECTED_M192_ANALYZER_SHA256,
        "m192_pinned_m195"
    )
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = [
        record for record in manifest["records"]
        if record["operator"] == "Linear" and ".mlp.fc2" in record["name"]
    ]
    require(len(records) == 120, "expected 120 FC2 records")
    aggregate = m192.empty_ledger()
    per_stage = defaultdict(m192.empty_ledger)
    token_classes = defaultdict(int)
    per_stage_token_classes = defaultdict(lambda: defaultdict(int))
    for ordinal, record in enumerate(records):
        stage, ledger, classes = audit_record(
            record, args.payload_root, m172, m192, args.chunk_tokens
        )
        m192.merge(aggregate, ledger)
        m192.merge(per_stage[stage], ledger)
        for key, value in classes.items():
            token_classes[key] += int(value)
            per_stage_token_classes[stage][key] += int(value)
        print("[M195] {}/120".format(ordinal + 1), flush=True)

    require(aggregate["events"] == EXPECTED_EVENTS, "event identity drift")
    require(aggregate["tokens"] == EXPECTED_TOKENS, "token identity drift")
    require(aggregate["windows"] == EXPECTED_WINDOWS,
            "window identity drift")
    require(aggregate["w1_replay_cycles"] == EXPECTED_W1_CYCLES,
            "W1 replay cross-check drift")
    require(aggregate["same_token_only_coissue_cycles"]
            == aggregate["ideal_dual_token_coissue_cycles"],
            "token-flush one-context and ideal cycles diverged")
    require(aggregate["replayed_event_terms"] == 412900394,
            "replayed event-term identity drift")
    token_flush_cycles = aggregate["same_token_only_coissue_cycles"]
    result = {
        "schema": "m195_h67_fc2_token_flush_pair_fusion_v1",
        "status": "PASS_EXACT_TOKEN_FLUSH_PAIR_FUSION_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m192_analyzer_sha256": EXPECTED_M192_ANALYZER_SHA256,
            "m191_result_sha256": EXPECTED_M191_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
        },
        "aggregate": aggregate,
        "token_classes": dict(token_classes),
        "exact_point": {
            "token_flush_pair_fusion_cycles": token_flush_cycles,
            "speed_vs_w1": fraction(EXPECTED_W1_CYCLES, token_flush_cycles),
            "gap_vs_global_ideal_w2": fraction(
                token_flush_cycles, EXPECTED_GLOBAL_IDEAL_W2_CYCLES
            ),
            "extra_acc24_contexts": 0,
            "cross_token_pairs": 0,
        },
        "per_stage": {
            str(stage): {
                **per_stage[stage],
                "token_classes": dict(per_stage_token_classes[stage]),
                "speed_vs_w1": fraction(
                    per_stage[stage]["w1_replay_cycles"],
                    per_stage[stage]["same_token_only_coissue_cycles"]
                ),
            }
            for stage in m192.STAGE_GEOMETRY
        },
        "architecture_decision": {
            "selected": "token-owned pair fusion with odd-tail flush",
            "acc24_updates_per_cycle": 1,
            "extra_acc24_contexts": 0,
            "window_reordering": False,
            "token_boundary_flush": True,
            "cross_token_pairing": False,
        },
        "claim_boundary": {
            "exact_payload_replay_cycles": True,
            "finite_fill_wall_cycles": False,
            "integrated_rtl": False,
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
