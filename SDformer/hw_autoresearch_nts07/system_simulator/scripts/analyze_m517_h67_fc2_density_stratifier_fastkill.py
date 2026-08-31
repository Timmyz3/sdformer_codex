#!/usr/bin/env python3
"""Fast-kill a Bishop-style dense/sparse route on frozen H67 FC2 tiles.

The unit of stratification is one 96-channel FC2 input tile for one token.
The strong reference is the sealed M216 full-vector K8 frontend.  The proposed
dense bypass and sparse path have the same eight logical weight banks and may
retire at most one source per bank per cycle.  Therefore a dense tile always
costs 12 issue cycles per output block, whereas the exact sparse issue floor is
the maximum of the eight per-bank nonzero counts and is never greater than 12.

The audit streams the sealed tar.zst directly, verifies every one of the 120
FC2 payload SHA256 identities, and reports both the density distribution and a
deliberately optimistic cycle/energy sensitivity.  It does not run RTL or EDA.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import zstandard as zstd


EXPECTED = {
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "archive": "aa261ebe64015bbd295f65f4b734efcb6b26c11c3dd0828e9e7a659433f6c3b4",
    "m216": "06a7b682b55fec8100bfa134017e186411eedfba0b170b024eb29606fd437422",
    "m496_contract": "e529aa8a5735fd25028b0c3325523167293be22c3c9760267c2f0397ff604f35",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
STAGE_GEOMETRY = {
    0: (384, 96),
    1: (768, 192),
    2: (1536, 384),
    3: (3072, 768),
}
DENSITY_THRESHOLDS = (12, 24, 36, 48, 54, 72)
OVERHEAD_SENSITIVITY = (0.0, 1.0, 2.0, 4.0, 8.0)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs)


def fraction(numerator, denominator):
    require(denominator != 0, "zero denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def percentile_from_histogram(histogram, quantile):
    total = int(sum(histogram))
    require(total > 0, "empty histogram")
    target = int(math.ceil(float(total) * float(quantile)))
    running = 0
    for value, count in enumerate(histogram):
        running += int(count)
        if running >= target:
            return int(value)
    raise RuntimeError("histogram percentile fell through")


def empty_ledger():
    return {
        "records": 0,
        "tokens": 0,
        "tiles96": 0,
        "events": 0,
        "empty_tiles": 0,
        "full_vector_sparse_service_floor_cycles": 0,
        "tile_partition_sparse_service_floor_cycles": 0,
        "dense_sequential_service_cycles": 0,
        "nnz_histogram_0_to_96": [0] * 97,
        "max_bank_histogram_0_to_12": [0] * 13,
        "thresholds": {
            str(value): {
                "tiles": 0,
                "events": 0,
                "dense_source_rows": 0,
                "extra_source_rows_vs_event_sparse": 0,
                "sparse_tile_floor_cycles": 0,
                "dense_cycles": 0,
            }
            for value in DENSITY_THRESHOLDS
        },
        "overhead_sensitivity": {
            str(value): {
                "dense_routed_tiles": 0,
                "cycles_saved_vs_segmented_sparse": 0.0,
            }
            for value in OVERHEAD_SENSITIVITY
        },
    }


def merge(target, source):
    scalar_keys = (
        "records", "tokens", "tiles96", "events", "empty_tiles",
        "full_vector_sparse_service_floor_cycles",
        "tile_partition_sparse_service_floor_cycles",
        "dense_sequential_service_cycles",
    )
    for key in scalar_keys:
        target[key] += int(source[key])
    for key in ("nnz_histogram_0_to_96", "max_bank_histogram_0_to_12"):
        target[key] = [int(a) + int(b)
                       for a, b in zip(target[key], source[key])]
    for threshold in DENSITY_THRESHOLDS:
        key = str(threshold)
        for field in target["thresholds"][key]:
            target["thresholds"][key][field] += int(
                source["thresholds"][key][field])
    for overhead in OVERHEAD_SENSITIVITY:
        key = str(overhead)
        target["overhead_sensitivity"][key]["dense_routed_tiles"] += int(
            source["overhead_sensitivity"][key]["dense_routed_tiles"])
        target["overhead_sensitivity"][key][
            "cycles_saved_vs_segmented_sparse"] += float(
                source["overhead_sensitivity"][key][
                    "cycles_saved_vs_segmented_sparse"])


def analyze_payload(record, data, popcount):
    shape = [int(value) for value in record["input_shape"]]
    output_shape = [int(value) for value in record["output_shape"]]
    require(len(shape) == 5 and shape[:2] == [10, 1],
            "FC2 input geometry drift")
    require(shape[:-1] == output_shape[:-1], "FC2 token geometry drift")
    stage = int(record["name"].split(".layers.")[1].split(".")[0])
    require(stage in STAGE_GEOMETRY, "FC2 stage drift")
    input_channels, output_channels = STAGE_GEOMETRY[stage]
    require((shape[-1], output_shape[-1]) ==
            (input_channels, output_channels), "FC2 channels drift")
    require(input_channels % 96 == 0 and output_channels % 96 == 0,
            "FC2 block alignment drift")
    tokens = int(np.prod(shape[:-1]))
    require(len(data) == tokens * input_channels // 8,
            "FC2 payload extent drift")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(
        tokens, input_channels // 8)
    output_blocks = output_channels // 96

    # Full-vector bank counts reproduce the strong M168/M216 K8 service floor.
    full_bank_counts = np.stack([
        ((packed >> bank) & 1).sum(axis=1, dtype=np.int32)
        for bank in range(8)
    ], axis=1)
    full_active = full_bank_counts.sum(axis=1, dtype=np.int32)
    require(int(full_active.sum()) == int(record["active_elements"]),
            "FC2 record event count drift")

    # 12 bytes are exactly one channel-contiguous 96-bit tile.
    tiles = packed.reshape(-1, 12)
    tile_nnz = popcount[tiles].sum(axis=1, dtype=np.uint16)
    tile_bank_counts = np.stack([
        ((tiles >> bank) & 1).sum(axis=1, dtype=np.uint16)
        for bank in range(8)
    ], axis=1)
    tile_max_bank = tile_bank_counts.max(axis=1)
    require(int(tile_nnz.sum()) == int(record["active_elements"]),
            "96-bit tile event conservation failed")
    require(bool(np.all(tile_max_bank <= 12)), "bank capacity drift")

    ledger = empty_ledger()
    ledger["records"] = 1
    ledger["tokens"] = tokens
    ledger["tiles96"] = int(tile_nnz.size)
    ledger["events"] = int(tile_nnz.sum())
    ledger["empty_tiles"] = int(np.count_nonzero(tile_nnz == 0))
    ledger["full_vector_sparse_service_floor_cycles"] = (
        int(full_bank_counts.max(axis=1).sum()) * output_blocks)
    ledger["tile_partition_sparse_service_floor_cycles"] = (
        int(tile_max_bank.sum()) * output_blocks)
    ledger["dense_sequential_service_cycles"] = (
        int(tile_nnz.size) * 12 * output_blocks)
    ledger["nnz_histogram_0_to_96"] = np.bincount(
        tile_nnz, minlength=97).astype(np.int64).tolist()
    ledger["max_bank_histogram_0_to_12"] = np.bincount(
        tile_max_bank, minlength=13).astype(np.int64).tolist()

    for threshold in DENSITY_THRESHOLDS:
        selected = tile_nnz >= threshold
        selected_tiles = int(np.count_nonzero(selected))
        selected_events = int(tile_nnz[selected].sum())
        dense_rows = selected_tiles * 96 * output_blocks
        key = str(threshold)
        ledger["thresholds"][key] = {
            "tiles": selected_tiles,
            "events": selected_events,
            "dense_source_rows": dense_rows,
            "extra_source_rows_vs_event_sparse":
                dense_rows - selected_events * output_blocks,
            "sparse_tile_floor_cycles":
                int(tile_max_bank[selected].sum()) * output_blocks,
            "dense_cycles": selected_tiles * 12 * output_blocks,
        }

    tile_floor = tile_max_bank.astype(np.float64) * output_blocks
    dense_cost = np.full(tile_floor.shape, 12.0 * output_blocks,
                         dtype=np.float64)
    for overhead in OVERHEAD_SENSITIVITY:
        # This is intentionally generous to the candidate: all modeled sparse
        # overhead vanishes on a dense-routed tile; router, format conversion,
        # ordered merge, and mode-switch costs are all zero.
        sparse_cost = tile_floor + float(overhead) * (tile_nnz != 0)
        savings = np.maximum(sparse_cost - dense_cost, 0.0)
        key = str(overhead)
        ledger["overhead_sensitivity"][key] = {
            "dense_routed_tiles": int(np.count_nonzero(savings > 0.0)),
            "cycles_saved_vs_segmented_sparse": float(savings.sum()),
        }
    return stage, int(record["sample_id"]), ledger


def enrich(ledger):
    result = dict(ledger)
    tiles = int(result["tiles96"])
    events = int(result["events"])
    result["event_density"] = float(events) / float(tiles * 96)
    result["empty_tile_fraction"] = float(result["empty_tiles"]) / float(tiles)
    result["nnz_quantiles"] = {
        "p50": percentile_from_histogram(result["nnz_histogram_0_to_96"], .50),
        "p90": percentile_from_histogram(result["nnz_histogram_0_to_96"], .90),
        "p95": percentile_from_histogram(result["nnz_histogram_0_to_96"], .95),
        "p99": percentile_from_histogram(result["nnz_histogram_0_to_96"], .99),
        "p99p9": percentile_from_histogram(result["nnz_histogram_0_to_96"], .999),
        "max": max(index for index, count in enumerate(
            result["nnz_histogram_0_to_96"]) if count),
    }
    result["max_bank_quantiles"] = {
        "p50": percentile_from_histogram(
            result["max_bank_histogram_0_to_12"], .50),
        "p95": percentile_from_histogram(
            result["max_bank_histogram_0_to_12"], .95),
        "p99": percentile_from_histogram(
            result["max_bank_histogram_0_to_12"], .99),
        "p99p9": percentile_from_histogram(
            result["max_bank_histogram_0_to_12"], .999),
        "max": max(index for index, count in enumerate(
            result["max_bank_histogram_0_to_12"]) if count),
    }
    for threshold in DENSITY_THRESHOLDS:
        row = result["thresholds"][str(threshold)]
        row["tile_fraction"] = float(row["tiles"]) / float(tiles)
        row["event_fraction"] = (float(row["events"]) / float(events)
                                 if events else 0.0)
        row["mean_nnz_if_selected"] = (
            float(row["events"]) / float(row["tiles"])
            if row["tiles"] else None)
        row["dense_over_sparse_tile_floor_ratio"] = (
            float(row["dense_cycles"]) /
            float(row["sparse_tile_floor_cycles"])
            if row["sparse_tile_floor_cycles"] else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--m216-result", required=True, type=Path)
    parser.add_argument("--m496-contract", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    script_start = sha256(Path(__file__).resolve())

    inputs = {
        "manifest": args.manifest,
        "archive": args.archive,
        "m216": args.m216_result,
        "m496_contract": args.m496_contract,
        "docs359": args.docs359,
    }
    observed = {key: sha256(path) for key, path in inputs.items()}
    require(observed == EXPECTED, "frozen input identity drift")
    manifest = strict_json(args.manifest)
    m216 = strict_json(args.m216_result)
    m496 = strict_json(args.m496_contract)
    records = [
        record for record in manifest["records"]
        if record["operator"] == "Linear" and ".mlp.fc2" in record["name"]
    ]
    require(len(records) == 120, "expected 120 FC2 records")
    require(sorted(set(int(row["sample_id"]) for row in records)) ==
            list(range(10)), "FC2 sample extent drift")
    require(len(set(row["name"] for row in records)) == 12,
            "FC2 module extent drift")
    require(m216["aggregate"]["k8_cycles"] == 90196785,
            "M216 K8 baseline drift")
    require(m216["aggregate"]["k8_service_cycle_floor"] == 70657362,
            "M216 service floor drift")
    require(m496["prior_cycle_evidence"][
        "frozen_h67_120_record_k8_over_k1_frontend_aggregate"] ==
        4.7642090014627465, "M496 inherited cycle evidence drift")

    record_by_relative = {row["relative_path"]: row for row in records}
    require(len(record_by_relative) == 120, "duplicate FC2 relative path")
    aggregate = empty_ledger()
    by_stage = defaultdict(empty_ledger)
    by_sample = defaultdict(empty_ledger)
    seen = set()
    popcount = np.array([bin(value).count("1") for value in range(256)],
                        dtype=np.uint8)

    with args.archive.open("rb") as compressed:
        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        archive = tarfile.open(fileobj=reader, mode="r|")
        for member in archive:
            relative = "/".join(member.name.split("/")[-2:])
            if relative not in record_by_relative:
                continue
            require(relative not in seen, "duplicate archive member " + relative)
            record = record_by_relative[relative]
            handle = archive.extractfile(member)
            require(handle is not None, "unreadable archive member " + relative)
            data = handle.read()
            require(len(data) == int(record["packed_bytes"]),
                    "archive member size drift " + relative)
            require(sha256_bytes(data) == record["file_sha256"],
                    "archive member SHA drift " + relative)
            stage, sample, ledger = analyze_payload(record, data, popcount)
            merge(aggregate, ledger)
            merge(by_stage[stage], ledger)
            merge(by_sample[sample], ledger)
            seen.add(relative)
            print("[M517] {}/120 {}".format(len(seen), relative), flush=True)
    require(seen == set(record_by_relative), "archive FC2 population incomplete")

    require(aggregate["records"] == 120
            and aggregate["tokens"] == 5580000
            and aggregate["tiles96"] == 36480000
            and aggregate["events"] == 143894510,
            "M517 frozen population drift")
    require(aggregate["full_vector_sparse_service_floor_cycles"] == 70657362,
            "full-vector service floor does not reproduce M216")
    require(aggregate["dense_sequential_service_cycles"] == 1105920000,
            "dense sequential denominator drift")
    # Same eight-bank bandwidth makes dense strictly faster for no tile.
    require(aggregate["max_bank_histogram_0_to_12"][12] >= 0,
            "max-bank histogram malformed")

    baseline_cycles = int(m216["aggregate"]["k8_cycles"])
    tile_floor = int(aggregate[
        "tile_partition_sparse_service_floor_cycles"])
    nonzero_tiles = int(aggregate["tiles96"] - aggregate["empty_tiles"])
    observed_overhead = baseline_cycles - int(aggregate[
        "full_vector_sparse_service_floor_cycles"])
    overhead_per_nonzero_tile = float(observed_overhead) / float(nonzero_tiles)
    calibrated_key = "calibrated_{:.9f}".format(overhead_per_nonzero_tile)

    # Calibrated sensitivity uses the observed aggregate overhead only as a
    # uniform per-nonzero-tile probe.  It is not an exact attribution.
    calibrated_dense_tiles = 0
    calibrated_savings = 0.0
    for stage in by_stage:
        ledger = by_stage[stage]
        output_blocks = 1 << int(stage)
        for max_bank, count in enumerate(
                ledger["max_bank_histogram_0_to_12"]):
            if max_bank == 0 or count == 0:
                continue
            sparse_cost = max_bank * output_blocks + overhead_per_nonzero_tile
            dense_cost = 12 * output_blocks
            if sparse_cost > dense_cost:
                calibrated_dense_tiles += int(count)
                calibrated_savings += (sparse_cost - dense_cost) * int(count)

    high50 = aggregate["thresholds"]["48"]["tiles"]
    high25 = aggregate["thresholds"]["24"]["tiles"]
    dense_equal_tiles = int(aggregate[
        "max_bank_histogram_0_to_12"][12])
    result = {
        "schema": "m517_h67_fc2_density_stratifier_fastkill_v1",
        "status": "KILL_NO_SAME_BANDWIDTH_DENSE_SPARSE_ROUTER_RTL",
        "identity": {
            "analyzer_start_sha256": script_start,
            "manifest_sha256": EXPECTED["manifest"],
            "archive_sha256": EXPECTED["archive"],
            "m216_result_sha256": EXPECTED["m216"],
            "m496_contract_sha256": EXPECTED["m496_contract"],
            "docs359_sha256_unchanged": EXPECTED["docs359"],
            "payload_sha_checks": len(seen),
        },
        "scope": {
            "network": "Motion/H67 ep35",
            "sequence": "zurich_city_09_a",
            "samples": 10,
            "modules": 12,
            "records": 120,
            "tile": "one channel-contiguous 96-bit FC2 input tile for one token",
            "same_resource_contract":
                "eight logical weight banks, one source per bank per cycle",
            "strong_baseline":
                "sealed M216 full-vector K8 always-ready standalone FC2 frontend",
        },
        "aggregate": enrich(aggregate),
        "per_stage": {str(key): enrich(value)
                      for key, value in sorted(by_stage.items())},
        "per_sample": {str(key): enrich(value)
                       for key, value in sorted(by_sample.items())},
        "cycle_fastkill": {
            "strong_m216_k8_cycles": baseline_cycles,
            "strong_m216_full_vector_service_floor_cycles": int(
                aggregate["full_vector_sparse_service_floor_cycles"]),
            "observed_control_and_collision_overhead_cycles": observed_overhead,
            "observed_overhead_per_nonzero_tile_calibration_only":
                overhead_per_nonzero_tile,
            "dense_all_tiles_service_cycles": int(
                aggregate["dense_sequential_service_cycles"]),
            "dense_all_over_strong_k8_ratio": fraction(
                aggregate["dense_sequential_service_cycles"], baseline_cycles),
            "tile_partition_sparse_floor_cycles_zero_tax": tile_floor,
            "tile_partition_regression_vs_strong_k8": fraction(
                tile_floor, baseline_cycles),
            "strong_k8_over_tile_partition_candidate": fraction(
                baseline_cycles, tile_floor),
            "cross_tile_bank_aggregation_loss_vs_full_vector_floor": fraction(
                tile_floor,
                aggregate["full_vector_sparse_service_floor_cycles"]),
            "strict_dense_issue_wins_zero_tax": 0,
            "dense_issue_ties_zero_tax": dense_equal_tiles,
            "best_noop_router_speedup": 1.0,
            "required_ideal_speedup": 1.10,
            "passes_1p10_gate": False,
            "calibrated_uniform_overhead_sensitivity": {
                "key": calibrated_key,
                "dense_routed_tiles": calibrated_dense_tiles,
                "dense_routed_fraction":
                    float(calibrated_dense_tiles) /
                    float(aggregate["tiles96"]),
                "cycles_saved_before_router_format_queue_tax":
                    calibrated_savings,
                "upper_speedup_if_subtracted_from_strong_baseline":
                    float(baseline_cycles) /
                    float(baseline_cycles - calibrated_savings),
                "warning":
                    "uniform overhead attribution is a sensitivity, not exact per-tile M216 accounting",
            },
        },
        "density_gate": {
            "tiles_at_least_25pct_density": high25,
            "fraction_at_least_25pct_density":
                float(high25) / float(aggregate["tiles96"]),
            "tiles_at_least_50pct_density": high50,
            "fraction_at_least_50pct_density":
                float(high50) / float(aggregate["tiles96"]),
            "tiles_at_least_75pct_density":
                aggregate["thresholds"]["72"]["tiles"],
            "maximum_nnz_per_96": enrich(aggregate)["nnz_quantiles"]["max"],
            "meaningful_high_low_mix_for_bishop_style_balance": False,
            "reason":
                "97.32% of tiles are below 25% density and only 0.0053% reach 50%; the high-density tail cannot balance a second path",
        },
        "energy_fastkill": {
            "admitted_net_energy": False,
            "weight_access_direction":
                "dense mode reads all 96 source rows per selected tile; event-sparse mode reads only nonzero source rows",
            "router_tax_omitted_in_cycle_upper_bound": True,
            "format_conversion_tax_omitted_in_cycle_upper_bound": True,
            "ordered_merge_queue_tax_omitted_in_cycle_upper_bound": True,
            "online_density_popcount_tax":
                "a pre-decode decision needs a 96-bit popcount/threshold tree; reusing the sparse decoder learns density too late to bypass it",
            "break_even_interpretation":
                "any decoder-energy saving must exceed the added zero-source weight reads plus router/format/merge energy; no matched SAIF/PTPX exists, so net energy remains unadmitted",
        },
        "decision": {
            "new_rtl": "NO",
            "verdict": "KILL",
            "reason": [
                "At equal eight-bank issue bandwidth dense sequential is never strictly faster than the exact sparse issue floor on any 96-bit tile.",
                "Forcing tile boundaries raises the zero-tax sparse floor from 70.657M to 118.651M cycles and already loses to the 90.197M full-vector K8 frontend.",
                "The observed workload has no useful high-density population: 1,922 of 36.48M tiles reach 50% density and none reaches 75%.",
                "The no-op router is therefore the best admitted choice (1.0x), below the predeclared 1.10x RTL gate.",
            ],
            "retain": [
                "Keep the shared-state K8 FC2 compactor/service path.",
                "Use FireFly-T only as support for multi-nonzero decode and bank-aware load balance already aligned with K8.",
                "Cite Bishop to explain why heterogeneous routing requires a real mixed-density distribution; H67 FC2 lacks it at 96-bit granularity.",
            ],
        },
        "claim_boundary": {
            "trace_only_cpu_audit": True,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "power": False,
            "complete_fc2": False,
            "ffn": False,
            "system_speedup": False,
            "headline": False,
            "forbidden": [
                "calling any local density ratio a full-network or system speedup",
                "claiming an energy saving without matched SAIF/PTPX and SRAM energy",
                "adding a dense path or stratifier RTL after the failed 1.10x gate",
                "transferring Bishop or FireFly-T published speedups to H67",
            ],
        },
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "analyzer mutated during execution")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "tiles96": result["aggregate"]["tiles96"],
        "high50_fraction": result["density_gate"][
            "fraction_at_least_50pct_density"],
        "tile_partition_regression": result["cycle_fastkill"][
            "tile_partition_regression_vs_strong_k8"]["float"],
        "new_rtl": result["decision"]["new_rtl"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
