#!/usr/bin/env python3
"""Exact-payload DSE for removing serialized FC2 EOT descriptors.

M178 assumes that the ATLIF write path builds two memories concurrently:
nonzero 96-bit descriptors and a per-token directory carrying start/count and
identity.  At read time the directory is available beside the first descriptor,
so a nonzero token marks its final descriptor from count instead of consuming a
separate EOT beat.  A count-zero directory entry still takes the two-cycle empty
token completion latency.  This is a conditional module-level schedule: the
directory, ports, producer RTL and energy are not included here.
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
EXPECTED_M176_ANALYZER_SHA256 = (
    "94cf1d234f2af581f7649a06c070ba14857f513f28591cd852d5ae184d6b7456"
)
EXPECTED_M172_ANALYZER_SHA256 = (
    "ae633daee1e07f16396570c1ef757c11bac7f1f72be108f4411d75f6dcb5f6d9"
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


def compact_nonzero_without_eot(groups, nonzero):
    token_count, beats = groups.shape
    lengths = nonzero.sum(axis=1, dtype=np.int16)
    compact = np.zeros((token_count, beats), dtype=np.int16)
    row, column = np.nonzero(nonzero)
    positions = np.cumsum(nonzero, axis=1, dtype=np.int16) - 1
    compact[row, positions[row, column]] = groups[row, column]
    return compact, lengths


def directory_wall_cycles(group_counts, lengths, output_blocks):
    """M174 recurrence with count-known length; length zero is header-only."""
    groups = np.asarray(group_counts, dtype=np.int16)
    lengths = np.asarray(lengths, dtype=np.int16)
    require(groups.ndim == 2 and lengths.shape == (groups.shape[0],),
            "directory recurrence shape drift")
    require(np.all(lengths >= 0) and np.all(lengths <= groups.shape[1]),
            "invalid directory token length")
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


def verify_recurrence(m176):
    rng = np.random.default_rng(178)
    cases = 0
    for output_blocks in (1, 2, 4, 8):
        for maximum_beats in range(1, 13):
            for _ in range(200):
                length = int(rng.integers(0, maximum_beats + 1))
                values = rng.integers(0, 17, size=length, dtype=np.int16)
                padded = np.zeros((1, maximum_beats), dtype=np.int16)
                padded[0, :length] = values
                observed = int(directory_wall_cycles(
                    padded, np.asarray([length]), output_blocks
                )[0])
                if length == 0:
                    expected = 2
                else:
                    expected = int(m176.variable_wall_cycles(
                        padded, np.asarray([length]), output_blocks
                    )[0])
                require(observed == expected, "directory recurrence mismatch")
                cases += 1
    return cases


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "raw_96bit_beats": 0,
        "nonzero_96bit_descriptors": 0,
        "zero_tokens": 0,
        "explicit_eot_descriptors_removed": 0,
        "directory_entries": 0,
    }
    for k_value in K_POINTS:
        result.update({
            "groups_k{}".format(k_value): 0,
            "explicit_eot_wall_cycles_k{}".format(k_value): 0,
            "directory_fused_wall_cycles_k{}".format(k_value): 0,
        })
    return result


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def enrich(ledger):
    result = dict(ledger)
    result["serialized_descriptor_reduction_vs_explicit_eot"] = fraction(
        result["explicit_eot_descriptors_removed"],
        result["nonzero_96bit_descriptors"]
        + result["explicit_eot_descriptors_removed"],
    )
    for k_value in K_POINTS:
        result["explicit_eot_over_directory_fused_wall_ratio_k{}".format(
            k_value
        )] = fraction(
            result["explicit_eot_wall_cycles_k{}".format(k_value)],
            result["directory_fused_wall_cycles_k{}".format(k_value)],
        )
    result["directory_fused_k1_over_k4_wall_ratio"] = fraction(
        result["directory_fused_wall_cycles_k1"],
        result["directory_fused_wall_cycles_k4"],
    )
    return result


def audit_record(record, payload_root, m172, m176, chunk_tokens):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    require(record["operator"] == "Linear" and ".mlp.fc2" in record["name"],
            "FC2 identity drift")
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
        tokens, beats_per_token, WIDTH // 8
    )
    ledger = empty_ledger()
    ledger.update({
        "records": 1,
        "tokens": tokens,
        "raw_96bit_beats": tokens * beats_per_token,
        "explicit_eot_descriptors_removed": tokens,
        "directory_entries": tokens,
    })
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        rows = m172.BYTE_BITS[np.asarray(raw[start:stop])]
        bank_counts = rows.sum(axis=2, dtype=np.int16)
        beat_events = bank_counts.sum(axis=2, dtype=np.int16)
        nonzero = beat_events != 0
        maximum_bank = bank_counts.max(axis=2)
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        ledger["nonzero_96bit_descriptors"] += int(
            nonzero.sum(dtype=np.int64)
        )
        ledger["zero_tokens"] += int(np.count_nonzero(~nonzero.any(axis=1)))
        for k_value in K_POINTS:
            groups = np.maximum(
                maximum_bank,
                (beat_events + (k_value - 1)) // k_value,
            ).astype(np.int16)
            compact_explicit, lengths_explicit = m176.compact_nonzero(
                groups, nonzero
            )
            compact_fused, lengths_fused = compact_nonzero_without_eot(
                groups, nonzero
            )
            require(int(compact_explicit.sum(dtype=np.int64))
                    == int(compact_fused.sum(dtype=np.int64)),
                    "group conservation drift")
            explicit_wall = m176.variable_wall_cycles(
                compact_explicit, lengths_explicit, output_blocks
            )
            fused_wall = directory_wall_cycles(
                compact_fused, lengths_fused, output_blocks
            )
            ledger["groups_k{}".format(k_value)] += int(
                compact_fused.sum(dtype=np.int64)
            )
            ledger["explicit_eot_wall_cycles_k{}".format(k_value)] += int(
                explicit_wall.sum(dtype=np.int64)
            )
            ledger["directory_fused_wall_cycles_k{}".format(k_value)] += int(
                fused_wall.sum(dtype=np.int64)
            )
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
        args.m172_analyzer, EXPECTED_M172_ANALYZER_SHA256, "m172_pinned_m178"
    )
    m176 = load_pinned(
        args.m176_analyzer, EXPECTED_M176_ANALYZER_SHA256, "m176_pinned_m178"
    )
    require(m172.verify_closed_form() == 9600,
            "fixed recurrence self-check drift")
    recurrence_cases = verify_recurrence(m176)
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
            record, args.payload_root, m172, m176, args.chunk_tokens
        )
        merge(aggregate, ledger)
        merge(per_stage[stage], ledger)
        print("[M178] {}/120".format(ordinal + 1), flush=True)
    require(aggregate["events"] == 143894510, "event identity drift")
    require(aggregate["raw_96bit_beats"] == 36480000,
            "raw beat identity drift")
    require(aggregate["explicit_eot_wall_cycles_k4"] == 144146504,
            "M176 K4 identity drift")
    result = {
        "schema": "m178_h67_fc2_token_directory_fusion_exact_payload_dse_v1",
        "status": "PASS_EXACT_PAYLOAD_CONDITIONAL_TOKEN_DIRECTORY_DSE",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "m176_analyzer_sha256": EXPECTED_M176_ANALYZER_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
            "fixed_recurrence_cases": 9600,
            "directory_recurrence_cases": recurrence_cases,
            "recurrence_mismatches": 0,
        },
        "architecture_assumption": {
            "atlif_native_compaction": True,
            "descriptor_memory": "one write/read port, 96-bit bitmap plus 5-bit beat index",
            "token_directory": "separate start/count/tag/output-block metadata port",
            "directory_available_with_first_descriptor": True,
            "serialized_eot_descriptor": False,
            "zero_token_completion_cycles": 2,
            "posthoc_scanner": False,
        },
        "aggregate": enrich(aggregate),
        "per_stage": {
            str(stage): enrich(per_stage[stage]) for stage in STAGE_GEOMETRY
        },
        "claim_boundary": {
            "exact_payload_conditional_frontend_dse": True,
            "producer_rtl": False,
            "directory_rtl": False,
            "finite_memory_ports": False,
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
        "PASS M178 explicit/fused K4={:.6f} K1/K4={:.6f} descriptors={}/{}".format(
            result["aggregate"]["explicit_eot_over_directory_fused_wall_ratio_k4"]["float"],
            result["aggregate"]["directory_fused_k1_over_k4_wall_ratio"]["float"],
            result["aggregate"]["nonzero_96bit_descriptors"],
            result["aggregate"]["nonzero_96bit_descriptors"]
            + result["aggregate"]["explicit_eot_descriptors_removed"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
