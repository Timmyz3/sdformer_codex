#!/usr/bin/env python3
"""Portable M361r2 k32/k64 exact-work replay on disjoint S10."""

from __future__ import division

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path


K_VALUES = (32, 64)
Q_VALUES = (16, 32, 64, 128)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def popcount(value):
    """Python-3.6-compatible population count for non-negative masks."""
    require(value >= 0, "popcount requires a non-negative mask")
    return bin(value).count("1")


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

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=reject)


def load_module(path):
    spec = importlib.util.spec_from_file_location("m361r2_frozen_m43", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M43 unpacker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_histograms(m43, manifest, manifest_path, operators, per_sample,
                       label):
    op_index = {name: index for index, name in enumerate(operators)}
    histograms = {k: defaultdict(Counter) for k in K_VALUES}
    seen = Counter()
    for record_index, record in enumerate(manifest["records"]):
        packed = manifest_path.parent / record["packed_file"]
        values = manifest_path.parent / record["value_payload_file"]
        require(packed.is_file() and sha256(packed) ==
                record["packed_file_sha256"], label + " packed SHA drift")
        require(values.is_file() and sha256(values) ==
                record["value_payload_sha256"], label + " value SHA drift")
        operator = record["operator"]
        require(operator in op_index, label + " unexpected operator")
        sample = int(record["sample_id"])
        seen[(sample, operator)] += 1
        masks = m43.unpack_record_masks(manifest_path.parent, record)
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                for k in K_VALUES:
                    mask = (1 << k) - 1
                    partition_base = tile * (m43.TILE_BITS // k)
                    for subtile in range(m43.TILE_BITS // k):
                        value = (value256 >> (subtile * k)) & mask
                        key = ((sample, op_index[operator],
                                partition_base + subtile) if per_sample else
                               (op_index[operator], partition_base + subtile))
                        histograms[k][key][value] += 1
        print("[M361 {}] {}/{}".format(
            label, record_index + 1, len(manifest["records"])), flush=True)
    require(len(seen) == len(manifest["records"]) and
            all(value == 1 for value in seen.values()),
            label + " sample/operator uniqueness failure")
    return histograms


def build_nested_centers(counter, k, q_max):
    """Deterministic count-weighted farthest expansion on train only.

    This is an opportunity screen, not a Lloyd optimum.  Zero and one-hot
    observations are excluded because a one-PWP representation cannot strictly
    beat their exact bit-sparse cost.
    """
    eligible = [value for value in counter
                if value != 0 and popcount(value) >= 2]
    if not eligible:
        return []
    remaining = set(eligible)
    minimum_distance = {value: popcount(value) for value in eligible}
    centers = []
    while remaining and len(centers) < q_max:
        chosen = max(
            remaining,
            key=lambda value: (
                counter[value] * minimum_distance[value],
                minimum_distance[value], counter[value], -value))
        centers.append(chosen)
        remaining.remove(chosen)
        for value in remaining:
            distance = popcount(value ^ chosen)
            if distance < minimum_distance[value]:
                minimum_distance[value] = distance
    require(len(centers) == len(set(centers)), "center duplication")
    return centers


def evaluate(counter, centers):
    result = Counter()
    used = set()
    for value, count in counter.items():
        population = popcount(value)
        if centers:
            best_distance = population + 1
            best_index = 0
            for index, center in enumerate(centers):
                distance = popcount(value ^ center)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
        else:
            best_distance = population
            best_index = 0
        use_pwp = bool(centers) and 1 + best_distance < population
        candidate = 1 + best_distance if use_pwp else population
        result["vectors"] += count
        result["bit_sparse_vector_ops_per_block"] += count * population
        result["candidate_vector_ops_per_block"] += count * candidate
        result["pwp_vector_ops_per_block"] += count * int(use_pwp)
        result["correction_vector_ops_per_block"] += count * (
            best_distance if use_pwp else population)
        result["exact_pattern_hits"] += count * int(
            use_pwp and best_distance == 0)
        if use_pwp:
            used.add(best_index)
    require(result["candidate_vector_ops_per_block"] ==
            result["pwp_vector_ops_per_block"] +
            result["correction_vector_ops_per_block"],
            "exact work conservation failure")
    result["used_pwp_patterns"] = len(used)
    return result


def signed_pwp_bits(k):
    # Signed INT8 source weights: exact range [-128*k, 127*k].
    bits = 1
    while (1 << (bits - 1)) < 128 * k:
        bits += 1
    return bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M361 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m361r2_wide_partition_exact_pattern_dse_contract_v1",
            "M361r2 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M361R2_EXECUTION",
            "M361r2 contract not frozen")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, identity in contract["inputs"].items():
        path = root / identity["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"], "SHA drift for " + name)
        paths[name] = path
        identities[name] = {"path": identity["path"], "sha256": observed}

    train_path = paths["m73_train_trace_manifest"]
    runtime_path = paths["m248_runtime_trace_manifest"]
    train = strict_json(train_path)
    runtime = strict_json(runtime_path)
    parent = strict_json(paths["m338_parent"])
    require(train["status"] ==
            "PASS_M73_DSEC_TRAIN_ONLY_S32_ALL18_SEQUENCES_EXACT_H67_EP35_FOUR_BOTTLENECK_TRACE" and
            train["split_audit"]["full_train_valid825_key_overlap"] == 0,
            "M73 split/status drift")
    require(runtime["status"] ==
            "PASS_PAFT_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE" and
            runtime["cohort"]["samples"] == 10 and
            runtime["cohort"]["records"] == 40,
            "M248 runtime status drift")
    require(parent["status"] ==
            "PASS_M338_TRAIN_ONLY_NESTED_Q16_Q32_Q64_Q128_EXACT_WORK_NO_CYCLES" and
            parent["split"]["test_or_validation_data_used"] is False,
            "M338 parent drift")
    operators = tuple(runtime["cohort"]["operators"])
    require(tuple(train["cohort"]["operators"]) == operators and
            tuple(parent["geometry"]["operators"]) == operators,
            "operator order drift")
    m43 = load_module(paths["m43_support_unpacker"])
    require(m43.TILE_BITS == 256 and all(256 % k == 0 for k in K_VALUES),
            "unsupported frozen tile width")
    train_hist = collect_histograms(
        m43, train, train_path, operators, False, "TRAIN")
    runtime_hist = collect_histograms(
        m43, runtime, runtime_path, operators, True, "RUNTIME")

    geometry_rows = []
    catalog_payload = {}
    for k in K_VALUES:
        partitions = m43.TILES * (m43.TILE_BITS // k)
        train_totals = {q: Counter() for q in Q_VALUES}
        runtime_totals = {q: Counter() for q in Q_VALUES}
        runtime_working_sets = {q: [] for q in Q_VALUES}
        short_partitions = Counter()
        operator_catalogs = []
        for op_index, operator in enumerate(operators):
            partition_rows = []
            for partition in range(partitions):
                centers = build_nested_centers(
                    train_hist[k][(op_index, partition)], k, Q_VALUES[-1])
                short_partitions[Q_VALUES[-1]] += int(
                    len(centers) < Q_VALUES[-1])
                observations = {}
                for q in Q_VALUES:
                    active = centers[:min(q, len(centers))]
                    train_observation = evaluate(
                        train_hist[k][(op_index, partition)], active)
                    train_totals[q].update(train_observation)
                    runtime_aggregate = Counter()
                    for sample in range(runtime["cohort"]["samples"]):
                        runtime_observation = evaluate(
                            runtime_hist[k][(sample, op_index, partition)],
                            active)
                        runtime_aggregate.update(runtime_observation)
                        runtime_working_sets[q].append(
                            runtime_observation["used_pwp_patterns"])
                    runtime_totals[q].update(runtime_aggregate)
                    observations[str(q)] = {
                        "active_patterns": len(active),
                        "train_candidate_vector_ops_per_block":
                            train_observation[
                                "candidate_vector_ops_per_block"],
                        "runtime_candidate_vector_ops_per_block":
                            runtime_aggregate[
                                "candidate_vector_ops_per_block"],
                    }
                partition_rows.append({
                    "partition": partition,
                    "nested_patterns": [
                        format(value, "0{}x".format(k // 4))
                        for value in centers],
                    "observations": observations,
                })
            operator_catalogs.append({
                "operator": operator,
                "partitions": partition_rows,
            })
            print("[M361 CATALOG] k={} op={}/{}".format(
                k, op_index + 1, len(operators)), flush=True)

        pwp_bits = signed_pwp_bits(k)
        pwp_vector_bytes = int(math.ceil(96 * pwp_bits / 8.0))
        q_rows = []
        for q in Q_VALUES:
            train_total = train_totals[q]
            runtime_total = runtime_totals[q]
            q_rows.append({
                "q_capacity": q,
                "train_exact_vector_work_speedup":
                    train_total["bit_sparse_vector_ops_per_block"] /
                    float(train_total["candidate_vector_ops_per_block"]),
                "runtime_exact_vector_work_speedup":
                    runtime_total["bit_sparse_vector_ops_per_block"] /
                    float(runtime_total["candidate_vector_ops_per_block"]),
                "runtime_bit_sparse_vector_ops_per_block":
                    runtime_total["bit_sparse_vector_ops_per_block"],
                "runtime_candidate_vector_ops_per_block":
                    runtime_total["candidate_vector_ops_per_block"],
                "runtime_pwp_vector_ops_per_block":
                    runtime_total["pwp_vector_ops_per_block"],
                "runtime_correction_vector_ops_per_block":
                    runtime_total["correction_vector_ops_per_block"],
                "runtime_exact_pattern_hits":
                    runtime_total["exact_pattern_hits"],
                "runtime_used_patterns_mean":
                    sum(runtime_working_sets[q]) /
                    float(len(runtime_working_sets[q])),
                "runtime_used_patterns_max": max(runtime_working_sets[q]),
                "pattern_table_capacity_bytes":
                    len(operators) * partitions * q * (k // 8),
                "full_signed_pwp_capacity_bytes":
                    len(operators) * partitions * q * 8 * pwp_vector_bytes,
            })
        require(all(q_rows[index]["runtime_candidate_vector_ops_per_block"] <=
                    q_rows[index - 1]["runtime_candidate_vector_ops_per_block"]
                    for index in range(1, len(q_rows))),
                "nested runtime work monotonicity failure")
        geometry_rows.append({
            "partition_bits": k,
            "partitions_per_operator": partitions,
            "signed_int8_pwp_bits": pwp_bits,
            "signed_pwp_vector_bytes_per_output_block": pwp_vector_bytes,
            "q_rows": q_rows,
        })
        catalog_payload[str(k)] = operator_catalogs

    payload = {
        "schema": "m361r2_wide_partition_exact_pattern_dse_v1",
        "status": "PASS_M361R2_TRAIN_ONLY_K32_K64_CATALOG_DISJOINT_S10_EXACT_WORK_NO_CYCLES",
        "identity": identities,
        "split": {
            "catalog_fit": "DSEC train only M73",
            "opportunity_replay": "disjoint M248 S10 runtime trace",
            "valid825_used": False,
            "train_valid825_overlap": 0,
        },
        "algorithm": {
            "catalog": "deterministic count-weighted farthest expansion over train-observed nonzero popcount>=2 patterns",
            "lloyd_optimum_claimed": False,
            "nested_q_prefixes": True,
            "exact_arithmetic": "PWP(center) plus signed residual with strict bit-sparse fallback",
            "accuracy_loss": False,
        },
        "geometry_dse": geometry_rows,
        "catalogs": catalog_payload,
        "admission": {
            "train_only_catalog": True,
            "disjoint_runtime_exact_work": True,
            "cycle_speedup": False,
            "physical_buffer": False,
            "rtl": False,
            "energy": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "claim_boundary": (
            "M361r2 reports exact vector-operation opportunity for k32/k64 "
            "nested train-only catalogs on a disjoint S10 trace. It explicitly "
            "prices wider signed PWP precision/capacity, but matcher cycles, "
            "cache/DMA schedule, RTL, area, energy, system and headline remain "
            "unadmitted."),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m361r2_wide_partition_exact_pattern_dse_r1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    summaries = []
    for geometry in geometry_rows:
        best = max(geometry["q_rows"],
                   key=lambda row: row["runtime_exact_vector_work_speedup"])
        summaries.append("k{}q{}={:.6f}x".format(
            geometry["partition_bits"], best["q_capacity"],
            best["runtime_exact_vector_work_speedup"]))
    print("M361R2_PASS " + " ".join(summaries) +
          " cycle_admitted=false", flush=True)


if __name__ == "__main__":
    main()
