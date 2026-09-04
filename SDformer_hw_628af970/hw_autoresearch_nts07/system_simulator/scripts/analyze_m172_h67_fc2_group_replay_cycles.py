#!/usr/bin/env python3
"""Cycle-account M171 group-held FC2 replay on the frozen H67 payloads.

The model is intentionally the executable standalone frontend contract:

* one continuously presented 64-bit bitmap beat;
* one raw residual/prefetch beat;
* K={1,4} bank-unique grouping within each beat;
* one held group replayed for every Cout/96 output block;
* always-ready group and token-done consumers.

It includes scan bubbles, per-beat grouping fragmentation, zero tokens, replay
and the M171 token-done control latency.  It excludes weight SRAM response,
M169 arithmetic, accumulator context, BN2 and residual commit.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST_SHA256 = (
    "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
)
STAGE_GEOMETRY = {
    0: (384, 96),
    1: (768, 192),
    2: (1536, 384),
    3: (3072, 768),
}
K_POINTS = (1, 4)
BYTE_BITS = np.asarray(
    [[(value >> bit) & 1 for bit in range(8)] for value in range(256)],
    dtype=np.uint8,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + value)
            ),
        )


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def explicit_wall_cycles(group_counts, output_blocks):
    """Small scalar edge simulator matching M171 ready/valid recurrence."""
    groups = [int(value) for value in group_counts]
    beat_index = 0
    residual_groups = 0
    group_replays_left = 0
    active = False
    last_seen = False
    done_valid = False
    cycle = 0
    while cycle < 100000:
        if done_valid:
            return cycle
        group_final = group_replays_left == 1
        group_slot_open = group_replays_left == 0 or group_final
        residual_will_clear = (
            residual_groups == 1 and group_slot_open
        )
        scan_ready = (
            not last_seen
            and (residual_groups == 0 or residual_will_clear)
        )
        scan_accept = scan_ready and beat_index < len(groups)
        incoming_groups = groups[beat_index] if scan_accept else 0
        incoming_last = scan_accept and beat_index == len(groups) - 1
        load_residual = group_slot_open and residual_groups > 0
        load_scan = (
            group_slot_open and residual_groups == 0
            and scan_accept and incoming_groups > 0
        )

        old_empty = residual_groups == 0 and group_replays_left == 0
        if active and last_seen and old_empty:
            done_valid = True

        if group_replays_left > 0:
            if group_final:
                group_replays_left = (
                    output_blocks if (load_residual or load_scan) else 0
                )
            else:
                group_replays_left -= 1
        elif load_residual or load_scan:
            group_replays_left = output_blocks

        if residual_groups > 0:
            if load_residual:
                residual_groups -= 1
                if residual_groups == 0 and scan_accept:
                    residual_groups = incoming_groups
        elif scan_accept:
            residual_groups = (
                incoming_groups - 1 if load_scan else incoming_groups
            )

        if scan_accept:
            active = True
            last_seen = last_seen or incoming_last
            beat_index += 1
        cycle += 1
    raise RuntimeError("explicit M171 cycle simulator did not terminate")


def closed_form_wall_cycles(group_counts, output_blocks):
    """Vectorized beat-event recurrence proved against the edge simulator.

    ``accept_cycle`` is when the current scan beat is accepted.  ``slot_cycle``
    is the next cycle in which a group register can be loaded.  A nonzero beat
    accepted at/after ``slot_cycle`` loads its first group directly; otherwise
    it waits in the raw prefetch register.  All later groups from that beat are
    loaded every ``output_blocks`` cycles.  A stored beat permits its successor
    to be accepted on the same edge that its last group is extracted; a direct
    one-group beat needs the following edge because one scan port cannot accept
    two beats in one cycle.
    """
    groups = np.asarray(group_counts, dtype=np.int16)
    token_count, beats_per_token = groups.shape
    accept_cycle = np.zeros(token_count, dtype=np.int64)
    slot_cycle = np.zeros(token_count, dtype=np.int64)
    last_accept_cycle = np.zeros(token_count, dtype=np.int64)
    for beat in range(beats_per_token):
        count = groups[:, beat].astype(np.int64)
        present = count > 0
        direct = present & (accept_cycle >= slot_cycle)
        load_cycle = np.maximum(accept_cycle, slot_cycle)
        last_load_cycle = load_cycle + np.maximum(count - 1, 0) * output_blocks
        slot_cycle = np.where(
            present, last_load_cycle + output_blocks, slot_cycle
        )
        last_accept_cycle = accept_cycle
        accept_cycle = np.where(
            ~present | (direct & (count == 1)),
            accept_cycle + 1,
            last_load_cycle,
        )
    return np.maximum(last_accept_cycle, slot_cycle) + 2


def verify_closed_form():
    rng = np.random.default_rng(172)
    cases = 0
    for output_blocks in (1, 2, 4, 8):
        for beats in range(1, 13):
            for _ in range(200):
                values = rng.integers(0, 17, size=beats, dtype=np.int16)
                vector = closed_form_wall_cycles(
                    values.reshape(1, beats), output_blocks
                )
                scalar = explicit_wall_cycles(values, output_blocks)
                require(
                    int(vector[0]) == scalar,
                    "closed-form recurrence mismatch blocks={} groups={} vector={} scalar={}".format(
                        output_blocks, values.tolist(), int(vector[0]), scalar
                    ),
                )
                cases += 1
    return cases


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "input_elements": 0,
        "events": 0,
        "scan_beats": 0,
        "zero_tokens": 0,
    }
    for k_value in K_POINTS:
        result.update({
            "groups_k{}".format(k_value): 0,
            "group_replay_cycles_k{}".format(k_value): 0,
            "wall_cycles_k{}".format(k_value): 0,
            "control_and_unhidden_scan_cycles_k{}".format(k_value): 0,
        })
    return result


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def audit_record(record, payload_root, chunk_tokens):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    require(
        record["operator"] == "Linear"
        and ".mlp.fc2" in record["name"]
        and len(shape) == 5 and len(output_shape) == 5,
        "FC2 topology drift",
    )
    input_channels = shape[-1]
    output_channels = output_shape[-1]
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    require(
        stage in STAGE_GEOMETRY
        and (input_channels, output_channels) == STAGE_GEOMETRY[stage],
        "FC2 geometry drift",
    )
    require(input_channels % 64 == 0, "FC2 input not 64-bit aligned")
    output_blocks = output_channels // 96
    beats_per_token = input_channels // 64
    tokens = int(np.prod(shape[:-1], dtype=np.int64))
    payload = payload_root / record["relative_path"]
    require(payload.is_file(), "missing payload " + str(payload))
    require(payload.stat().st_size == int(record["packed_bytes"]),
            "payload size drift")
    require(sha256(payload) == record["file_sha256"], "payload SHA drift")
    packed = np.memmap(payload, dtype=np.uint8, mode="r")
    require(packed.size == tokens * input_channels // 8,
            "payload extent drift")
    packed = packed.reshape(tokens, beats_per_token, 8)

    ledger = empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    ledger["input_elements"] = int(record["input_elements"])
    ledger["scan_beats"] = tokens * beats_per_token
    event_total = 0
    zero_tokens = 0
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        rows = BYTE_BITS[np.asarray(packed[start:stop])]
        bank_counts = rows.sum(axis=2, dtype=np.int16)
        beat_events = bank_counts.sum(axis=2, dtype=np.int16)
        event_total += int(beat_events.sum(dtype=np.int64))
        zero_tokens += int(np.count_nonzero(
            beat_events.sum(axis=1, dtype=np.int32) == 0
        ))
        maximum_bank = bank_counts.max(axis=2)
        for k_value in K_POINTS:
            groups = np.maximum(
                maximum_bank,
                (beat_events + (k_value - 1)) // k_value,
            ).astype(np.int16)
            group_sum = int(groups.sum(dtype=np.int64))
            replay_cycles = group_sum * output_blocks
            wall = closed_form_wall_cycles(groups, output_blocks)
            wall_sum = int(wall.sum(dtype=np.int64))
            ledger["groups_k{}".format(k_value)] += group_sum
            ledger["group_replay_cycles_k{}".format(k_value)] += (
                replay_cycles
            )
            ledger["wall_cycles_k{}".format(k_value)] += wall_sum
            ledger[
                "control_and_unhidden_scan_cycles_k{}".format(k_value)
            ] += wall_sum - replay_cycles
    require(event_total == int(record["active_elements"]),
            "payload popcount drift")
    ledger["events"] = event_total
    ledger["zero_tokens"] = zero_tokens
    return stage, ledger


def enrich(ledger):
    result = dict(ledger)
    result["k1_over_k4_wall_cycle_ratio"] = fraction(
        result["wall_cycles_k1"], result["wall_cycles_k4"]
    )
    result["k1_over_k4_group_replay_cycle_ratio"] = fraction(
        result["group_replay_cycles_k1"],
        result["group_replay_cycles_k4"],
    )
    result["k4_wall_over_group_replay_overhead"] = fraction(
        result["wall_cycles_k4"], result["group_replay_cycles_k4"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    require(args.chunk_tokens > 0, "invalid chunk size")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    recurrence_cases = verify_closed_form()
    manifest = strict_json(args.manifest)
    records = [
        record for record in manifest["records"]
        if record["operator"] == "Linear" and ".mlp.fc2" in record["name"]
    ]
    require(len(records) == 120, "expected 120 FC2 records")

    aggregate = empty_ledger()
    per_stage = defaultdict(empty_ledger)
    for ordinal, record in enumerate(records):
        stage, ledger = audit_record(
            record, args.payload_root, args.chunk_tokens
        )
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print(
            "[M172] {}/120 sample={} stage={} module={}".format(
                ordinal + 1, record["sample_id"], stage, record["name"]
            ),
            flush=True,
        )

    require(aggregate["events"] == 143894510,
            "aggregate event population drift")
    require(aggregate["scan_beats"] == 54720000,
            "aggregate scan beat drift")
    require(aggregate["group_replay_cycles_k1"] == 412900394,
            "K1 replay cycle drift")
    result = {
        "schema": "m172_h67_fc2_group_replay_exact_payload_cycles_v1",
        "status": "PASS_EXACT_PAYLOAD_M171_FRONTEND_CYCLE_MODEL",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "analyzer_start_sha256": script_start,
            "payload_identity": "all 120 payload SHA/size/popcount checked",
            "closed_form_vs_explicit_recurrence_cases": recurrence_cases,
            "closed_form_vs_explicit_recurrence_mismatches": 0,
        },
        "cycle_contract": {
            "scan_width_bits": 64,
            "raw_bitmap_prefetch_entries": 1,
            "source_group_held_across_output_blocks": True,
            "group_consumer_always_ready": True,
            "token_done_consumer_always_ready": True,
            "K_points": list(K_POINTS),
            "included": [
                "per-beat modulo-8 bank grouping",
                "one raw beat prefetch",
                "scan bubbles and zero tokens",
                "group replay across Cout/96 blocks",
                "M171 token-done control latency",
            ],
            "excluded": [
                "weight request/response latency and macro conflicts",
                "M169 arithmetic and 2304-bit accumulator context",
                "BN2, residual and complete FC2 commit",
                "clock tree, routed timing, power and system scheduling",
            ],
        },
        "aggregate": enrich(aggregate),
        "per_stage": {
            str(stage): enrich(per_stage[stage])
            for stage in sorted(STAGE_GEOMETRY)
        },
        "claim_boundary": {
            "standalone_frontend_exact_payload_cycle_ratio": True,
            "rtl_cycle_measured": False,
            "weight_sram_response": False,
            "arithmetic_composed": False,
            "complete_fc2": False,
            "physical_speedup": False,
            "full_ffn_cycles": False,
            "system_speedup": False,
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
        "PASS M172 exact-payload M171 frontend K1/K4 wall={:.6f}x".format(
            result["aggregate"]["k1_over_k4_wall_cycle_ratio"]["float"]
        )
    )


if __name__ == "__main__":
    main()
