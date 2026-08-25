#!/usr/bin/env python3
"""Exact FC2 replay-kernel DSE for bank-disjoint multi-context co-issue.

The bounded K8/K7 engines drain one descriptor window at a time.  Across a
small queue of independent accumulation contexts, however, each of the eight
weight banks can choose a different context in the same cycle.  For a batch
of C adjacent windows with bank populations p[c,b], the exact replay service
cycles per output block are

    max_b sum_c p[c,b].

This is attainable because each bank schedules its own source sequence and
emits a context tag; no bank performs more than one read per cycle.  The DSE
does not reorder windows to manufacture complementary masks.  It batches the
captured window stream in arrival order within each frozen FC2 call.

The result is a replay-kernel opportunity only.  Descriptor fill, context
SRAM ports/storage, routing, weight-response latency, RTL timing, BN2/residual,
complete-FC2 and system cycles are explicitly excluded.
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
EXPECTED_M187_RESULT_SHA256 = (
    "411e61ff9c5e0a8b4ff27e86cf15d5ae87b8ef523fc25e43b0939417d65cd201"
)
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
EXPECTED_EVENTS = 143894510
EXPECTED_TOKENS = 5580000
EXPECTED_RAW96_BEATS = 36480000
EXPECTED_WINDOWS = 6523707
EXPECTED_K1_REPLAY_CYCLES = 412900394
EXPECTED_K8_REPLAY_CYCLES = 79397844
EXPECTED_K8_WALL_CYCLES = 97607807
WIDTH = 96
CONTEXT_POINTS = (1, 2, 4, 8, 16)
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
        "replayed_event_terms": 0,
    }
    for contexts in CONTEXT_POINTS:
        result["replay_cycles_c{}".format(contexts)] = 0
        result["full_batches_c{}".format(contexts)] = 0
        result["partial_batches_c{}".format(contexts)] = 0
    return result


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def account_batches(ledger, pending, windows, output_blocks):
    for contexts in CONTEXT_POINTS:
        combined = pending[contexts] + windows
        complete = (len(combined) // contexts) * contexts
        if complete:
            batches = np.asarray(
                combined[:complete], dtype=np.int64
            ).reshape(-1, contexts, 8)
            bank_loads = batches.sum(axis=1, dtype=np.int64)
            service = bank_loads.max(axis=1)
            ledger["replay_cycles_c{}".format(contexts)] += int(
                service.sum(dtype=np.int64)
            ) * output_blocks
            ledger["full_batches_c{}".format(contexts)] += int(
                batches.shape[0]
            )
        pending[contexts] = combined[complete:]


def flush_partial_batches(ledger, pending, output_blocks):
    for contexts in CONTEXT_POINTS:
        if not pending[contexts]:
            continue
        bank_load = np.asarray(
            pending[contexts], dtype=np.int64
        ).sum(axis=0, dtype=np.int64)
        ledger["replay_cycles_c{}".format(contexts)] += int(
            bank_load.max()
        ) * output_blocks
        ledger["partial_batches_c{}".format(contexts)] += 1


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
    ledger.update({
        "records": 1,
        "tokens": tokens,
        "raw96_beats": tokens * beats_per_token,
    })
    pending = {contexts: [] for contexts in CONTEXT_POINTS}
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
        for token, count in enumerate(window_count):
            if count:
                windows.extend(pooled[token, :int(count)])
        account_batches(ledger, pending, windows, output_blocks)
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["nonzero96_descriptors"] += int(
            descriptor_count.sum(dtype=np.int64)
        )
        ledger["windows"] += int(window_count.sum(dtype=np.int64))
        if windows:
            ledger["replayed_event_terms"] += int(
                np.asarray(windows, dtype=np.int64).sum(dtype=np.int64)
            ) * output_blocks
    flush_partial_batches(ledger, pending, output_blocks)
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return stage, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m187-result", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    require(sha256(args.m187_result) == EXPECTED_M187_RESULT_SHA256,
            "M187 result identity drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359_SHA256,
            "docs359 identity drift")
    m172 = load_pinned(
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256,
        "m172_pinned_m191"
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
        print("[M191] {}/120".format(ordinal + 1), flush=True)

    require(aggregate["events"] == EXPECTED_EVENTS, "event identity drift")
    require(aggregate["tokens"] == EXPECTED_TOKENS, "token identity drift")
    require(aggregate["raw96_beats"] == EXPECTED_RAW96_BEATS,
            "raw-beat identity drift")
    require(aggregate["windows"] == EXPECTED_WINDOWS,
            "window identity drift")
    require(aggregate["replayed_event_terms"] == EXPECTED_K1_REPLAY_CYCLES,
            "replayed event-term identity drift")
    require(aggregate["replay_cycles_c1"] == EXPECTED_K8_REPLAY_CYCLES,
            "M187 K8 replay-cycle cross-check drift")

    points = {}
    measured_fill_drain_overhead = (
        EXPECTED_K8_WALL_CYCLES - EXPECTED_K8_REPLAY_CYCLES
    )
    for contexts in CONTEXT_POINTS:
        cycles = aggregate["replay_cycles_c{}".format(contexts)]
        optimistic_wall = cycles + measured_fill_drain_overhead
        points[str(contexts)] = {
            "accumulation_contexts": contexts,
            "context_tag_bits_per_bank": 0 if contexts == 1 else int(
                np.ceil(np.log2(contexts))
            ),
            "replay_cycles": cycles,
            "full_batches": aggregate["full_batches_c{}".format(contexts)],
            "partial_batches": aggregate[
                "partial_batches_c{}".format(contexts)
            ],
            "c1_over_c": fraction(
                aggregate["replay_cycles_c1"], cycles
            ),
            "k1_serial_replay_over_c": fraction(
                aggregate["replayed_event_terms"], cycles
            ),
            "weight_bank_utilization": fraction(
                aggregate["replayed_event_terms"], 8 * cycles
            ),
            "optimistic_wall_if_m187_overhead_unchanged": optimistic_wall,
            "m187_k8_wall_over_optimistic_wall": fraction(
                EXPECTED_K8_WALL_CYCLES, optimistic_wall
            ),
        }

    result = {
        "schema": "m191_h67_fc2_multicontext_bank_coissue_dse_v1",
        "status": "PASS_EXACT_PAYLOAD_REPLAY_KERNEL_DSE_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m187_result_sha256": EXPECTED_M187_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
            "m187_k8_replay_crosscheck_cycles": EXPECTED_K8_REPLAY_CYCLES,
        },
        "architecture": {
            "physical_weight_banks": 8,
            "maximum_one_read_per_bank_per_cycle": True,
            "context_points": list(CONTEXT_POINTS),
            "batching": "adjacent arrival-order windows within each FC2 call",
            "window_reordering": False,
            "exact_batch_service_formula": "max_b(sum_c(bank_population[c,b]))",
            "achievability": (
                "each bank independently drains one source per cycle and emits "
                "a context tag; contexts own independent Acc24 state"
            ),
            "stage_window_depths": {"0": 2, "1": 4, "2": 8, "3": 8},
        },
        "aggregate_identity": {
            key: value for key, value in aggregate.items()
            if not key.startswith(("replay_cycles_c", "full_batches_c",
                                   "partial_batches_c"))
        },
        "points": points,
        "per_stage": {
            str(stage): per_stage[stage] for stage in STAGE_GEOMETRY
        },
        "candidate": {
            "selected_for_first_rtl_screen": "C2",
            "selected_for_upper_dse_point": "C4",
            "reason": (
                "C2 tests the minimum dual-context tag/partial-sum machinery; "
                "C4 is the first stronger performance point but requires a "
                "matched four-context storage/port cost before admission"
            ),
        },
        "claim_boundary": {
            "exact_payload_replay_kernel_cycles": True,
            "finite_descriptor_fill_wall_cycles": False,
            "context_storage_and_ports": False,
            "bank_to_context_routing_rtl": False,
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
    require(sha256(script_path) == script_start,
            "analyzer changed during run")


if __name__ == "__main__":
    main()
