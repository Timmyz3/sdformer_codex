#!/usr/bin/env python3
"""Exact-payload scan-width DSE for the M171 group-held FC2 frontend."""

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
WIDTH_POINTS = (64, 96, 128, 192, 384)
STAGE_GEOMETRY = {
    0: (384, 96),
    1: (768, 192),
    2: (1536, 384),
    3: (3072, 768),
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


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def load_m172_analyzer(path):
    require(sha256(path) == EXPECTED_M172_ANALYZER_SHA256,
            "M172 recurrence identity drift")
    spec = importlib.util.spec_from_file_location("m172_pinned", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def empty_ledger():
    return {
        "records": 0,
        "tokens": 0,
        "events": 0,
        "input_elements": 0,
        "scan_beats": 0,
        "K1_groups": 0,
        "K4_groups": 0,
        "K1_group_replay_cycles": 0,
        "K4_group_replay_cycles": 0,
        "K1_wall_cycles": 0,
        "K4_wall_cycles": 0,
    }


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def enrich(ledger):
    result = dict(ledger)
    result["K1_over_K4_group_replay_ratio"] = fraction(
        result["K1_group_replay_cycles"],
        result["K4_group_replay_cycles"],
    )
    result["K1_over_K4_wall_ratio"] = fraction(
        result["K1_wall_cycles"], result["K4_wall_cycles"]
    )
    result["K4_control_and_unhidden_scan_cycles"] = (
        result["K4_wall_cycles"] - result["K4_group_replay_cycles"]
    )
    return result


def audit_width(raw, tokens, input_channels, output_blocks, width, m172,
                chunk_tokens):
    require(width % 8 == 0, "scan width not byte aligned")
    bytes_per_beat = width // 8
    bytes_per_token = input_channels // 8
    beat_count = (bytes_per_token + bytes_per_beat - 1) // bytes_per_beat
    padded_bytes = beat_count * bytes_per_beat
    if padded_bytes == bytes_per_token:
        beats = raw.reshape(tokens, beat_count, bytes_per_beat)
    else:
        padded = np.pad(raw, ((0, 0), (0, padded_bytes-bytes_per_token)))
        beats = padded.reshape(tokens, beat_count, bytes_per_beat)

    ledger = empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    ledger["input_elements"] = tokens * input_channels
    ledger["scan_beats"] = tokens * beat_count
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        rows = m172.BYTE_BITS[np.asarray(beats[start:stop])]
        bank_counts = rows.sum(axis=2, dtype=np.int16)
        beat_events = bank_counts.sum(axis=2, dtype=np.int16)
        ledger["events"] += int(beat_events.sum(dtype=np.int64))
        maximum_bank = bank_counts.max(axis=2)
        for k_value in (1, 4):
            groups = np.maximum(
                maximum_bank,
                (beat_events + (k_value - 1)) // k_value,
            ).astype(np.int16)
            group_sum = int(groups.sum(dtype=np.int64))
            wall_sum = int(m172.closed_form_wall_cycles(
                groups, output_blocks
            ).sum(dtype=np.int64))
            ledger["K{}_groups".format(k_value)] += group_sum
            ledger["K{}_group_replay_cycles".format(k_value)] += (
                group_sum * output_blocks
            )
            ledger["K{}_wall_cycles".format(k_value)] += wall_sum
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--m172-analyzer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=32768)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    script_start = sha256(Path(__file__).resolve())
    m172 = load_m172_analyzer(args.m172_analyzer)
    require(m172.verify_closed_form() == 9600,
            "M172 recurrence self-check drift")
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = [
        record for record in manifest["records"]
        if record["operator"] == "Linear" and ".mlp.fc2" in record["name"]
    ]
    require(len(records) == 120, "expected 120 FC2 records")

    aggregate = {width: empty_ledger() for width in WIDTH_POINTS}
    per_stage = {
        stage: {width: empty_ledger() for width in WIDTH_POINTS}
        for stage in STAGE_GEOMETRY
    }
    for ordinal, record in enumerate(records):
        shape = [int(value) for value in record["input_shape"]]
        input_channels = shape[-1]
        output_channels = int(record["output_shape"][-1])
        tokens = int(np.prod(shape[:-1], dtype=np.int64))
        stage = int(record["name"].split(".layers.")[1].split(".")[0])
        require((input_channels, output_channels) == STAGE_GEOMETRY[stage],
                "stage geometry drift")
        payload = args.payload_root / record["relative_path"]
        require(payload.is_file(), "missing payload")
        require(payload.stat().st_size == int(record["packed_bytes"]),
                "payload size drift")
        require(sha256(payload) == record["file_sha256"],
                "payload SHA drift")
        raw = np.fromfile(payload, dtype=np.uint8).reshape(
            tokens, input_channels // 8
        )
        for width in WIDTH_POINTS:
            ledger = audit_width(
                raw, tokens, input_channels, output_channels // 96,
                width, m172, args.chunk_tokens,
            )
            require(ledger["events"] == int(record["active_elements"]),
                    "payload popcount drift")
            merge(aggregate[width], ledger)
            merge(per_stage[stage][width], ledger)
        print("[M173] {}/120".format(ordinal + 1), flush=True)

    require(aggregate[64]["K1_wall_cycles"] == 446528624,
            "M172 K1 wall identity drift")
    require(aggregate[64]["K4_wall_cycles"] == 179057955,
            "M172 K4 wall identity drift")
    require(aggregate[128]["K1_wall_cycles"] == 432951702,
            "128-bit K1 wall drift")
    require(aggregate[128]["K4_wall_cycles"] == 146423753,
            "128-bit K4 wall drift")

    result = {
        "schema": "m173_h67_fc2_scan_width_exact_payload_dse_v1",
        "status": "PASS_SELECT_128BIT_RTL_TIMING_RECOVERY_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "m172_analyzer_sha256": EXPECTED_M172_ANALYZER_SHA256,
            "analyzer_start_sha256": script_start,
            "all_120_payload_sha_size_popcount_checked": True,
            "recurrence_self_check_cases": 9600,
            "recurrence_self_check_mismatches": 0,
        },
        "scan_width_points_bits": list(WIDTH_POINTS),
        "aggregate": {
            str(width): enrich(aggregate[width])
            for width in WIDTH_POINTS
        },
        "per_stage": {
            str(stage): {
                str(width): enrich(per_stage[stage][width])
                for width in WIDTH_POINTS
            }
            for stage in STAGE_GEOMETRY
        },
        "selection": {
            "scan_width_bits": 128,
            "K1_wall_cycles": 432951702,
            "K4_wall_cycles": 146423753,
            "K1_over_K4_wall_ratio": 2.9568406295391156,
            "stage0_wall_ratio": 2.1513040700672446,
            "reason": "smallest power-of-two point that exceeds 2x in every stage and approaches 3x aggregate; 192/384 remain bandwidth and selector-depth upper points",
            "required_rtl_change": "one shared hierarchical per-bank selector, not the duplicated nested-priority M171 r1 network",
        },
        "claim_boundary": {
            "exact_payload_analytic_frontend_DSE": True,
            "rtl_cycle_measured": False,
            "dc_128bit": False,
            "bitmap_memory_128bit_delivery": False,
            "weight_sram_response": False,
            "arithmetic_composed": False,
            "complete_fc2": False,
            "physical_speedup": False,
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
        "PASS M173 scan width DSE select128 wall={:.6f}x stage0={:.6f}x".format(
            result["selection"]["K1_over_K4_wall_ratio"],
            result["selection"]["stage0_wall_ratio"],
        )
    )


if __name__ == "__main__":
    main()
