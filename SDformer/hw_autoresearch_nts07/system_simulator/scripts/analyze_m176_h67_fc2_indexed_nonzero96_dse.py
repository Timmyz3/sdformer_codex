#!/usr/bin/env python3
"""Audit indexed nonzero 96-bit FC2 beats on the frozen H67 payloads.

The indexed stream removes zero bitmap beats, retains the original base-row on
every nonzero beat and appends one explicit zero-payload end-of-token (EOT)
descriptor.  It uses the same K={1,4} bank-unique grouping, group hold and
output-block replay recurrence as M173.  The model therefore charges the EOT
cycle and does not assume that a previous nonzero beat can predict the future.

This is an exact-payload analytic frontend DSE.  It excludes construction of
the indexed producer, bitmap/event memory, weight SRAM, arithmetic, accumulator
context, BN2/residual and system scheduling.
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
EXPECTED_M173_RESULT_SHA256 = (
    "f9823e2fb0256fc5e98d08069f3e91c6b47b63f43fc6edf6d0a50c84b302c3d3"
)
STAGE_GEOMETRY = {
    0: (384, 96),
    1: (768, 192),
    2: (1536, 384),
    3: (3072, 768),
}
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


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def load_m172(path):
    require(sha256(path) == EXPECTED_M172_ANALYZER_SHA256,
            "M172 analyzer identity drift")
    spec = importlib.util.spec_from_file_location("m172_pinned_m176", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variable_wall_cycles(group_counts, lengths, output_blocks):
    """Vectorized M171/M174 recurrence for compact variable-length tokens."""
    groups = np.asarray(group_counts, dtype=np.int16)
    lengths = np.asarray(lengths, dtype=np.int16)
    require(groups.ndim == 2 and lengths.shape == (groups.shape[0],),
            "variable recurrence shape drift")
    require(np.all(lengths >= 1) and np.all(lengths <= groups.shape[1]),
            "invalid compact token length")
    token_count, maximum_beats = groups.shape
    accept_cycle = np.zeros(token_count, dtype=np.int64)
    slot_cycle = np.zeros(token_count, dtype=np.int64)
    last_accept_cycle = np.zeros(token_count, dtype=np.int64)
    for beat in range(maximum_beats):
        active = beat < lengths
        count = groups[:, beat].astype(np.int64)
        present = active & (count > 0)
        direct = present & (accept_cycle >= slot_cycle)
        load_cycle = np.maximum(accept_cycle, slot_cycle)
        last_load_cycle = load_cycle + np.maximum(count - 1, 0) * output_blocks
        slot_cycle = np.where(
            present, last_load_cycle + output_blocks, slot_cycle
        )
        last_accept_cycle = np.where(
            active, accept_cycle, last_accept_cycle
        )
        next_accept = np.where(
            ~present | (direct & (count == 1)),
            accept_cycle + 1,
            last_load_cycle,
        )
        accept_cycle = np.where(active, next_accept, accept_cycle)
    return np.maximum(last_accept_cycle, slot_cycle) + 2


def verify_variable_recurrence(m172):
    rng = np.random.default_rng(176)
    cases = 0
    for output_blocks in (1, 2, 4, 8):
        for maximum_beats in range(1, 13):
            for _ in range(200):
                length = int(rng.integers(1, maximum_beats + 1))
                values = rng.integers(0, 17, size=length, dtype=np.int16)
                padded = np.zeros((1, maximum_beats), dtype=np.int16)
                padded[0, :length] = values
                vector = int(variable_wall_cycles(
                    padded, np.asarray([length]), output_blocks
                )[0])
                scalar = int(m172.explicit_wall_cycles(
                    values, output_blocks
                ))
                require(vector == scalar,
                        "variable recurrence mismatch")
                cases += 1
    return cases


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "input_elements": 0,
        "raw_96bit_beats": 0,
        "nonzero_96bit_beats": 0,
        "zero_96bit_beats_elided": 0,
        "zero_tokens": 0,
        "eot_descriptors": 0,
        "indexed_descriptors": 0,
    }
    for k_value in K_POINTS:
        result.update({
            "raw_groups_k{}".format(k_value): 0,
            "indexed_groups_k{}".format(k_value): 0,
            "raw_group_replay_cycles_k{}".format(k_value): 0,
            "indexed_group_replay_cycles_k{}".format(k_value): 0,
            "raw_wall_cycles_k{}".format(k_value): 0,
            "indexed_wall_cycles_k{}".format(k_value): 0,
        })
    return result


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def enrich(ledger):
    result = dict(ledger)
    result["zero_beat_fraction"] = fraction(
        result["zero_96bit_beats_elided"], result["raw_96bit_beats"]
    )
    result["raw_over_indexed_descriptor_ratio"] = fraction(
        result["raw_96bit_beats"], result["indexed_descriptors"]
    )
    for k_value in K_POINTS:
        result["raw_over_indexed_wall_ratio_k{}".format(k_value)] = fraction(
            result["raw_wall_cycles_k{}".format(k_value)],
            result["indexed_wall_cycles_k{}".format(k_value)],
        )
    result["indexed_k1_over_k4_wall_ratio"] = fraction(
        result["indexed_wall_cycles_k1"],
        result["indexed_wall_cycles_k4"],
    )
    return result


def compact_nonzero(groups, nonzero):
    """Left-pack nonzero beats and reserve one final zero EOT descriptor."""
    token_count, beats = groups.shape
    lengths = nonzero.sum(axis=1, dtype=np.int16) + 1
    compact = np.zeros((token_count, beats + 1), dtype=np.int16)
    row, column = np.nonzero(nonzero)
    positions = np.cumsum(nonzero, axis=1, dtype=np.int16) - 1
    compact[row, positions[row, column]] = groups[row, column]
    return compact, lengths


def audit_record(record, payload_root, m172, chunk_tokens):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    require(record["operator"] == "Linear"
            and ".mlp.fc2" in record["name"]
            and len(shape) == 5 and len(output_shape) == 5,
            "FC2 topology drift")
    input_channels = shape[-1]
    output_channels = output_shape[-1]
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    require((input_channels, output_channels) == STAGE_GEOMETRY[stage],
            "FC2 geometry drift")
    require(input_channels % WIDTH == 0,
            "production FC2 input not 96-bit aligned")
    output_blocks = output_channels // 96
    beats_per_token = input_channels // WIDTH
    bytes_per_token = input_channels // 8
    tokens = int(np.prod(shape[:-1], dtype=np.int64))
    payload = payload_root / record["relative_path"]
    require(payload.is_file(), "missing payload " + str(payload))
    require(payload.stat().st_size == int(record["packed_bytes"]),
            "payload size drift")
    require(sha256(payload) == record["file_sha256"],
            "payload SHA drift")
    raw = np.memmap(payload, dtype=np.uint8, mode="r")
    require(raw.size == tokens * bytes_per_token,
            "payload extent drift")
    raw = raw.reshape(tokens, beats_per_token, WIDTH // 8)

    ledger = empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    ledger["input_elements"] = int(record["input_elements"])
    ledger["raw_96bit_beats"] = tokens * beats_per_token
    ledger["eot_descriptors"] = tokens
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        rows = m172.BYTE_BITS[np.asarray(raw[start:stop])]
        bank_counts = rows.sum(axis=2, dtype=np.int16)
        beat_events = bank_counts.sum(axis=2, dtype=np.int16)
        nonzero = beat_events != 0
        maximum_bank = bank_counts.max(axis=2)
        event_count = int(beat_events.sum(dtype=np.int64))
        nonzero_count = int(nonzero.sum(dtype=np.int64))
        ledger["events"] += event_count
        ledger["nonzero_96bit_beats"] += nonzero_count
        ledger["zero_96bit_beats_elided"] += nonzero.size - nonzero_count
        ledger["zero_tokens"] += int(np.count_nonzero(~nonzero.any(axis=1)))
        ledger["indexed_descriptors"] += nonzero_count + (stop - start)
        for k_value in K_POINTS:
            groups = np.maximum(
                maximum_bank,
                (beat_events + (k_value - 1)) // k_value,
            ).astype(np.int16)
            group_sum = int(groups.sum(dtype=np.int64))
            compact, lengths = compact_nonzero(groups, nonzero)
            compact_group_sum = int(compact.sum(dtype=np.int64))
            require(compact_group_sum == group_sum,
                    "group conservation drift")
            raw_wall = m172.closed_form_wall_cycles(groups, output_blocks)
            indexed_wall = variable_wall_cycles(
                compact, lengths, output_blocks
            )
            ledger["raw_groups_k{}".format(k_value)] += group_sum
            ledger["indexed_groups_k{}".format(k_value)] += compact_group_sum
            ledger["raw_group_replay_cycles_k{}".format(k_value)] += (
                group_sum * output_blocks
            )
            ledger["indexed_group_replay_cycles_k{}".format(k_value)] += (
                compact_group_sum * output_blocks
            )
            ledger["raw_wall_cycles_k{}".format(k_value)] += int(
                raw_wall.sum(dtype=np.int64)
            )
            ledger["indexed_wall_cycles_k{}".format(k_value)] += int(
                indexed_wall.sum(dtype=np.int64)
            )
    require(ledger["events"] == int(record["active_elements"]),
            "payload popcount drift")
    return stage, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--m173-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    require(args.chunk_tokens > 0, "invalid chunk size")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    require(sha256(args.m173_result) == EXPECTED_M173_RESULT_SHA256,
            "M173 result identity drift")
    m172 = load_m172(args.m172_analyzer)
    require(m172.verify_closed_form() == 9600,
            "M172 fixed recurrence self-check drift")
    variable_cases = verify_variable_recurrence(m172)
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with args.m173_result.open("r", encoding="utf-8") as handle:
        m173_result = json.load(handle)
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
        print("[M176] {}/120".format(ordinal + 1), flush=True)

    require(aggregate["events"] == 143894510,
            "aggregate event population drift")
    require(aggregate["raw_96bit_beats"] == 36480000,
            "raw 96-bit beat identity drift")
    require(aggregate["raw_wall_cycles_k1"] == 437234151,
            "M173 raw96 K1 wall identity drift")
    require(aggregate["raw_wall_cycles_k4"] == 157504597,
            "M173 raw96 K4 wall identity drift")
    require(aggregate["raw_wall_cycles_k1"]
            == int(m173_result["aggregate"]["96"]["K1_wall_cycles"]),
            "M173 result cross-check drift")
    require(aggregate["raw_wall_cycles_k4"]
            == int(m173_result["aggregate"]["96"]["K4_wall_cycles"]),
            "M173 result cross-check drift")

    aggregate_enriched = enrich(aggregate)
    stage_enriched = {
        str(stage): enrich(per_stage[stage]) for stage in STAGE_GEOMETRY
    }
    result = {
        "schema": "m176_h67_fc2_indexed_nonzero96_exact_payload_dse_v1",
        "status": "PASS_EXACT_PAYLOAD_INDEXED_NONZERO96_FRONTEND_DSE",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m173_result_sha256": EXPECTED_M173_RESULT_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
            "fixed_recurrence_self_check_cases": 9600,
            "variable_recurrence_self_check_cases": variable_cases,
            "recurrence_self_check_mismatches": 0,
        },
        "transport": {
            "bitmap_width_bits": 96,
            "nonzero_beats_carry_absolute_base_row": True,
            "zero_beats_elided": True,
            "explicit_zero_payload_eot_per_token": True,
            "future_nonzero_prediction_assumed": False,
            "descriptor_acceptance": "one descriptor per cycle",
        },
        "aggregate": aggregate_enriched,
        "per_stage": stage_enriched,
        "reference_raw128": m173_result["aggregate"]["128"],
        "claim_boundary": {
            "exact_payload_analytic_frontend_dse": True,
            "indexed_producer_rtl": False,
            "rtl_cycle_measured": False,
            "bitmap_or_event_memory_delivery": False,
            "weight_sram_response": False,
            "arithmetic_composed": False,
            "accumulator_context": False,
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
        "PASS M176 indexed nonzero96 ratio={:.6f} stage0={:.6f} descriptors={}/{}".format(
            result["aggregate"]["indexed_k1_over_k4_wall_ratio"]["float"],
            result["per_stage"]["0"]["indexed_k1_over_k4_wall_ratio"]["float"],
            result["aggregate"]["indexed_descriptors"],
            result["aggregate"]["raw_96bit_beats"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
