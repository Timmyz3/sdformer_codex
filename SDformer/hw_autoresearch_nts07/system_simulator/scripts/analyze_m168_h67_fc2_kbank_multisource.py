#!/usr/bin/env python3
"""Audit bank-feasible multi-source service for frozen H67 FC2 events.

The M39/M159 FC2 model services one active binary input column at a time and
updates one 96-output block.  This audit asks how many such source columns can
be retired together when weights are striped by input_channel modulo eight.
For K={1,2,4,8}, a cycle accepts at most K events globally and at most one
event from each weight bank.  For one token the exact minimum service count is

    max(max_b event_count[b], ceil(total_events / K)).

The bound is executable for independent bank queues: distribute every bank's
events over that many cycles, filling at most K distinct banks per cycle.  It
does not model the compactor, SRAM macro, accumulator tree, or routed timing.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST_SHA256 = (
    "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
)
K_POINTS = (1, 2, 4, 8)
SAMPLES = 10
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


def nearest_rank(values, percentile):
    require(values, "empty percentile population")
    ordered = sorted(int(value) for value in values)
    rank = int(math.ceil(percentile * len(ordered)))
    return ordered[max(0, rank - 1)]


def distribution(values):
    return {
        "count": len(values),
        "min": min(values),
        "p50_nearest_rank": nearest_rank(values, 0.50),
        "p95_nearest_rank": nearest_rank(values, 0.95),
        "max": max(values),
        "sum": sum(values),
    }


def empty_ledger():
    result = {
        "records": 0,
        "tokens": 0,
        "nonzero_tokens": 0,
        "input_elements": 0,
        "events": 0,
    }
    for k_value in K_POINTS:
        result["service_cycles_k{}".format(k_value)] = 0
        result["output_block_cycles_k{}".format(k_value)] = 0
    return result


def merge(target, source):
    for key, value in source.items():
        target[key] += int(value)


def audit_record(record, payload_root):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    require(
        record["operator"] == "Linear"
        and ".mlp.fc2" in record["name"]
        and len(shape) == 5
        and len(output_shape) == 5,
        "FC2 record topology drift",
    )
    require(shape[:2] == [10, 1], "FC2 T/B geometry drift")
    require(shape[:-1] == output_shape[:-1], "FC2 token geometry drift")
    input_channels = shape[-1]
    output_channels = output_shape[-1]
    require(input_channels % 8 == 0, "FC2 input not byte aligned")
    require(output_channels % 96 == 0, "FC2 output not 96-lane aligned")
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    require(
        stage in STAGE_GEOMETRY
        and (input_channels, output_channels) == STAGE_GEOMETRY[stage],
        "FC2 stage geometry drift",
    )
    payload = payload_root / record["relative_path"]
    require(payload.is_file(), "missing payload " + str(payload))
    require(payload.stat().st_size == int(record["packed_bytes"]),
            "payload size drift")
    require(sha256(payload) == record["file_sha256"], "payload SHA drift")

    tokens = 1
    for extent in shape[:-1]:
        tokens *= extent
    bytes_per_token = input_channels // 8
    packed = np.fromfile(str(payload), dtype=np.uint8)
    require(packed.size == tokens * bytes_per_token,
            "payload packed extent drift")
    packed = packed.reshape(tokens, bytes_per_token)

    # Channels are packed little-bit-first and each byte spans exactly the
    # eight modulo-8 banks.  Summing one bit position across bytes gives the
    # number of active input columns owned by that bank for the token.
    bank_counts = np.empty((tokens, 8), dtype=np.int16)
    for bank in range(8):
        bank_counts[:, bank] = np.bitwise_and(
            np.right_shift(packed, bank), 1
        ).sum(axis=1, dtype=np.int16)
    active = bank_counts.sum(axis=1, dtype=np.int32)
    require(int(active.sum()) == int(record["active_elements"]),
            "payload popcount drift")
    maximum_bank = bank_counts.max(axis=1)
    output_blocks = output_channels // 96

    ledger = empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    ledger["nonzero_tokens"] = int(np.count_nonzero(active))
    ledger["input_elements"] = int(record["input_elements"])
    ledger["events"] = int(active.sum())
    for k_value in K_POINTS:
        cycles = np.maximum(
            maximum_bank, (active + (k_value - 1)) // k_value
        )
        cycle_sum = int(cycles.sum())
        ledger["service_cycles_k{}".format(k_value)] = cycle_sum
        ledger["output_block_cycles_k{}".format(k_value)] = (
            cycle_sum * output_blocks
        )
    return stage, int(record["sample_id"]), ledger


def enrich(ledger):
    ledger = dict(ledger)
    ledger["event_density"] = (
        float(ledger["events"]) / float(ledger["input_elements"])
    )
    ledger["nonzero_token_fraction"] = (
        float(ledger["nonzero_tokens"]) / float(ledger["tokens"])
    )
    for k_value in (2, 4, 8):
        ledger["k1_over_k{}_output_block_cycle_ratio".format(k_value)] = \
            fraction(
                ledger["output_block_cycles_k1"],
                ledger["output_block_cycles_k{}".format(k_value)],
            )
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "manifest identity drift")
    manifest = strict_json(args.manifest)
    records = [
        record
        for record in manifest["records"]
        if record["operator"] == "Linear" and ".mlp.fc2" in record["name"]
    ]
    require(len(records) == 120, "expected 120 frozen FC2 records")
    require(
        sorted(set(int(record["sample_id"]) for record in records))
        == list(range(SAMPLES)),
        "FC2 sample population drift",
    )
    require(len(set(record["name"] for record in records)) == 12,
            "FC2 module population drift")

    aggregate = empty_ledger()
    by_stage = defaultdict(empty_ledger)
    by_sample = defaultdict(empty_ledger)
    for ordinal, record in enumerate(records):
        stage, sample, ledger = audit_record(record, args.payload_root)
        merge(aggregate, ledger)
        merge(by_stage[stage], ledger)
        merge(by_sample[sample], ledger)
        print(
            "[M168] {}/120 sample={} stage={} module={}".format(
                ordinal + 1, sample, stage, record["name"]
            ),
            flush=True,
        )

    require(aggregate["events"] == 143894510,
            "aggregate FC2 event population drift")
    require(aggregate["output_block_cycles_k1"] == 412900394,
            "aggregate K1 output-block cycle drift")
    require(aggregate["output_block_cycles_k4"] == 106536803,
            "aggregate K4 output-block cycle drift")
    require(aggregate["output_block_cycles_k8"] == 70657362,
            "aggregate K8 output-block cycle drift")

    sample_cycles = {}
    for k_value in K_POINTS:
        sample_cycles["k{}_output_block_cycles".format(k_value)] = \
            distribution([
                by_sample[sample]["output_block_cycles_k{}".format(k_value)]
                for sample in range(SAMPLES)
            ])

    result = {
        "schema": "m168_h67_fc2_kbank_multisource_dse_v1",
        "status": "PASS_EXACT_FROZEN_FC2_PAYLOAD_KBANK_CYCLE_BOUNDARY_RTL_OPEN",
        "identity": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "analyzer_start_sha256": script_start,
            "payload_identity": "all 120 payload SHA/size/popcount checked",
        },
        "population": {
            "samples": SAMPLES,
            "modules": 12,
            "records": 120,
            "stages": [0, 1, 2, 3],
        },
        "bank_service_contract": {
            "weight_banks": 8,
            "bank_mapping": "input_channel modulo 8",
            "per_bank_reads_per_cycle": 1,
            "candidate_sources_per_cycle": list(K_POINTS),
            "cycle_equation_per_token":
                "max(max_bank_event_count, ceil(active_event_count/K))",
            "proof_scope":
                "exact minimum for independent per-bank event queues before weight SRAM, compactor, accumulator and routing costs",
            "output_lanes": 96,
            "weight_bits_signed": 8,
            "weight_read_payload_bits_per_cycle": {
                "K1": 768,
                "K2": 1536,
                "K4": 3072,
                "K8": 6144,
            },
        },
        "aggregate": enrich(aggregate),
        "per_stage": {
            str(stage): enrich(by_stage[stage])
            for stage in sorted(STAGE_GEOMETRY)
        },
        "per_sample_cycle_distribution": sample_cycles,
        "selection": {
            "rtl_candidate": "K4",
            "reason":
                "K4 reaches 3.875660x exact output-block service reduction while requiring half the weight bandwidth and shallower reduction than K8; K8 remains a DSE upper point",
            "k4_required_weight_rows_per_cycle": 4,
            "k4_required_signed_weight_terms_per_output_lane": 4,
            "k4_requires_accumulator_exactness_miter": True,
        },
        "claim_boundary": {
            "admitted": [
                "exact frozen 120-record FC2 binary-input payload population",
                "exact modulo-8 bank occupancy and K1/K2/K4/K8 cycle equation",
                "output-block-weighted service-cycle ratios under the stated bank contract",
            ],
            "forbidden": [
                "calling the K ratios RTL, physical, FFN, network or system speedup",
                "assuming four or eight simultaneous SRAM weight-row reads without macro evidence",
                "assuming compaction, accumulator, BN2 or residual commit are free",
                "mixing this ten-sample payload population with the profile100 M39 denominator",
                "energy, FPS, PPA, external comparison, DATE headline or best-paper claim",
            ],
            "rtl": False,
            "vcs": False,
            "dc": False,
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
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "PASS M168 FC2 K-bank payload DSE K4={:.6f}x K8={:.6f}x".format(
            result["aggregate"]["k1_over_k4_output_block_cycle_ratio"]["float"],
            result["aggregate"]["k1_over_k8_output_block_cycle_ratio"]["float"],
        )
    )


if __name__ == "__main__":
    main()
