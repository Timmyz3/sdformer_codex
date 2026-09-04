#!/usr/bin/env python3
"""Screen bounded near-pattern snapping on the frozen PAFT Conv trace."""

from __future__ import division

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


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

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import pinned module: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def popcount(value):
    return bin(int(value)).count("1")


def nearest_center(value, centers):
    return min((popcount(value ^ candidate), candidate)
               for candidate in centers)


def snapped_phase_metrics(counter, centers, threshold):
    result = Counter()
    for value, count in counter.items():
        population = popcount(value)
        distance, center = nearest_center(value, centers)
        snap = population >= 2 and distance <= threshold
        candidate = 1 if snap else min(population, 1 + distance)
        result["partition_vectors"] += count
        result["dense_vector_ops_per_block"] += count * 16
        result["bit_sparse_vector_ops_per_block"] += count * population
        result["candidate_vector_ops_per_block"] += count * candidate
        result["matcher_rows"] += count * int(population >= 2)
        result["nonzero_partition_vectors"] += count * int(population != 0)
        if value != 0 and value in centers:
            result["exact_pattern_hits"] += count
        if snap:
            result["pwp_ops_per_block"] += count
            result["assignment_rows"] += count
            if distance:
                result["approximated_partition_vectors"] += count
                result["elided_correction_ops_per_block"] += count * distance
                result["approximated_hamming_bit_flips"] += count * distance
                result["approximated_used_center_{}".format(center)] += count
        elif 1 + distance < population:
            result["pwp_ops_per_block"] += count
            result["correction_ops_per_block"] += count * distance
            result["assignment_rows"] += count
        else:
            result["correction_ops_per_block"] += count * population
    require(result["pwp_ops_per_block"] +
            result["correction_ops_per_block"] ==
            result["candidate_vector_ops_per_block"],
            "snapped work conservation failure")
    return dict(result)


def weighted_quantile(histogram, quantile):
    total = sum(histogram.values())
    if total == 0:
        return None
    target = int(math.ceil(quantile * total))
    running = 0
    for value in sorted(histogram):
        running += histogram[value]
        if running >= target:
            return int(value)
    raise RuntimeError("weighted quantile fallthrough")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m280_paft_near_match_residual_elision_dse_contract_v1",
            "M280 contract schema drift")
    root = args.contract.resolve().parents[1]
    source_start = sha256(Path(__file__).resolve())
    identities = {}
    paths = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing M280 input: " + str(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "M280 SHA drift for {}: {}".format(name, observed))
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}

    m251 = load_module(paths["m251_analyzer"], "m280_pinned_m251")
    m43 = load_module(paths["m43_support_unpacker"], "m280_pinned_m43")
    m251_contract = strict_json(paths["m251_contract"])
    m251_result = strict_json(paths["m251_result"])
    catalog = strict_json(paths["m77_train_only_catalog"])
    trace = strict_json(paths["m248_paft_running_bn_trace"])
    m256 = strict_json(paths["m256_int8_result"])
    geometry = m251_contract["geometry"]
    cycle_model = m251_contract["same_resource_cycle_model"]
    thresholds = list(contract["policy"]["distance_thresholds"])
    require(thresholds == [0, 1, 2, 3, 4], "M280 threshold drift")
    require(catalog["split"]["test_or_validation_data_used"] is False and
            catalog["split"]["train_valid825_key_overlap"] == 0,
            "M280 train-only catalog admission drift")
    require(trace["status"] ==
            "PASS_PAFT_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE" and
            trace["identity"]["capture_bn_policy"] == "running" and
            trace["cohort"]["records"] == 40,
            "M280 trace admission drift")
    require(m256["identity"]["m248_source_manifest_sha256"] ==
            identities["m248_paft_running_bn_trace"]["sha256"],
            "M280 M256/trace identity mismatch")
    operator_names = trace["cohort"]["operators"]
    require([row["operator"] for row in catalog["operators"]] == operator_names,
            "M280 catalog/trace operator mismatch")
    require(geometry == {
        "samples": 10, "operators": 4, "rows_per_operator": 3000,
        "features_per_row": 6912, "partition_bits": 16,
        "partitions_per_operator": 432, "patterns_per_partition": 16,
        "output_blocks": 8, "output_lanes_per_block": 96},
        "M280 frozen geometry drift")
    require(m43.ROWS == geometry["rows_per_operator"] and
            m43.TILES * (m43.TILE_BITS // 16) == 432,
            "M280 support unpack geometry drift")

    weights = []
    for op in range(4):
        path = paths["m256_weight_o{}".format(op)]
        data = np.fromfile(str(path), dtype=np.int8)
        require(data.size == 6912 * 768,
                "M280 weight payload shape drift op{}".format(op))
        weights.append(data.reshape(6912, 768).astype(np.int32))
    require([layer["operator"] for layer in m256["layers"]] == operator_names,
            "M280 M256 layer order drift")

    trace_dir = paths["m248_paft_running_bn_trace"].parent
    op_index = {name: index for index, name in enumerate(operator_names)}
    histograms = defaultdict(Counter)
    record_audit = []
    for record_index, record in enumerate(trace["records"]):
        packed_path = trace_dir / record["packed_file"]
        require(packed_path.is_file() and
                sha256(packed_path) == record["packed_file_sha256"],
                "M280 packed payload drift")
        require(record["negative_count"] == 0,
                "M280 requires nonnegative binary support")
        masks = m43.unpack_record_masks(trace_dir, record)
        reconstructed = 0
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // 16)
                for subtile in range(m43.TILE_BITS // 16):
                    value = (value256 >> (subtile * 16)) & 0xffff
                    histograms[(record["sample_id"],
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
                    reconstructed += popcount(value)
        record_audit.append({
            "sample_id": record["sample_id"],
            "operator_index": record["operator_index"],
            "expanded_conv3x3_source_events": reconstructed
        })
        print("[M280 HIST] {}/40 sample={} op={} expanded={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"], reconstructed), flush=True)

    phases = dict((threshold, defaultdict(list)) for threshold in thresholds)
    aggregates = dict((threshold, Counter()) for threshold in thresholds)
    per_operator = dict((threshold, [Counter() for _ in operator_names])
                        for threshold in thresholds)
    lossy_pairs = Counter()
    for sample in range(10):
        for op in range(4):
            operator = catalog["operators"][op]
            require(len(operator["partitions"]) == 432,
                    "M280 catalog partition extent drift")
            for partition, row in enumerate(operator["partitions"]):
                centers = [int(item["value_hex"], 16)
                           for item in row["patterns"]]
                require(row["partition"] == partition and
                        len(centers) == 16 and len(set(centers)) == 16 and
                        all(center != 0 for center in centers),
                        "M280 center domain/order drift")
                counter = histograms[(sample, op, partition)]
                require(sum(counter.values()) == 3000,
                        "M280 phase population drift")
                for value, count in counter.items():
                    population = popcount(value)
                    distance, center = nearest_center(value, centers)
                    if population >= 2 and 1 <= distance <= thresholds[-1]:
                        lossy_pairs[(op, partition, value, center)] += count
                for threshold in thresholds:
                    metric = snapped_phase_metrics(counter, centers, threshold)
                    phases[threshold][sample].append(metric)
                    aggregates[threshold].update(dict(
                        (key, value) for key, value in metric.items()
                        if not key.startswith("approximated_used_center_")))
                    per_operator[threshold][op].update(dict(
                        (key, value) for key, value in metric.items()
                        if not key.startswith("approximated_used_center_")))

    expected_vectors = 10 * 4 * 3000 * 432
    for threshold in thresholds:
        aggregate = aggregates[threshold]
        require(aggregate["partition_vectors"] == expected_vectors,
                "M280 aggregate population drift")
        require(aggregate["pwp_ops_per_block"] +
                aggregate["correction_ops_per_block"] ==
                aggregate["candidate_vector_ops_per_block"],
                "M280 aggregate conservation drift")

    error_state = {}
    for threshold in thresholds:
        error_state[threshold] = {
            "rows": 0, "hamming": 0, "values": 0, "unchanged": 0,
            "abs_sum": 0, "square_sum": 0, "row_max_sum": 0,
            "worst_row_max": 0, "row_max_histogram": Counter()
        }
    for pair_index, ((op, partition, value, center), count) in enumerate(
            lossy_pairs.items()):
        distance = popcount(value ^ center)
        difference = np.asarray(
            [((value >> bit) & 1) - ((center >> bit) & 1)
             for bit in range(16)], dtype=np.int32)
        begin = partition * 16
        delta = np.matmul(difference, weights[op][begin:begin + 16, :])
        absolute = np.abs(delta.astype(np.int64))
        maximum = int(absolute.max())
        abs_sum = int(absolute.sum())
        square_sum = int(np.square(delta.astype(np.int64)).sum())
        unchanged = int(np.count_nonzero(delta == 0))
        for threshold in thresholds:
            if distance <= threshold:
                state = error_state[threshold]
                state["rows"] += count
                state["hamming"] += count * distance
                state["values"] += count * 768
                state["unchanged"] += count * unchanged
                state["abs_sum"] += count * abs_sum
                state["square_sum"] += count * square_sum
                state["row_max_sum"] += count * maximum
                state["worst_row_max"] = max(state["worst_row_max"], maximum)
                state["row_max_histogram"][maximum] += count
        if (pair_index + 1) % 10000 == 0:
            print("[M280 ERROR] pairs={}/{}".format(
                pair_index + 1, len(lossy_pairs)), flush=True)

    threshold_rows = []
    for threshold in thresholds:
        aggregate = aggregates[threshold]
        cycle_rows = []
        for port in cycle_model["ports"]:
            totals = Counter()
            for sample in range(10):
                replay = m251.replay_sample(phases[threshold][sample], port,
                                             cycle_model, geometry)
                for key in ("dense_cycles", "bit_sparse_cycles",
                            "candidate_cycles"):
                    totals[key] += replay[key]
                totals.update(replay["binding_phases"])
            cycle_rows.append({
                "port": port["name"],
                "weight_vector_service_cycles":
                    port["weight_vector_service_cycles"],
                "pwp_vector_service_cycles":
                    port["pwp_vector_service_cycles"],
                "dense_cycles": totals["dense_cycles"],
                "bit_sparse_cycles": totals["bit_sparse_cycles"],
                "candidate_cycles": totals["candidate_cycles"],
                "speedup_vs_dense": totals["dense_cycles"] /
                    float(totals["candidate_cycles"]),
                "speedup_vs_bit_sparse": totals["bit_sparse_cycles"] /
                    float(totals["candidate_cycles"]),
                "binding_phases": dict((name, totals[name]) for name in
                    ("compute", "matcher", "packer", "dma"))
            })
        state = error_state[threshold]
        require(state["rows"] ==
                aggregate["approximated_partition_vectors"],
                "M280 error/approximate row mismatch")
        require(state["hamming"] ==
                aggregate["approximated_hamming_bit_flips"],
                "M280 error/Hamming mismatch")
        error = {
            "approximated_partition_vectors": state["rows"],
            "approximated_fraction_all_partition_vectors":
                state["rows"] / float(expected_vectors),
            "elided_correction_ops_per_block": state["hamming"],
            "accumulator_values": state["values"],
            "unchanged_accumulator_values": state["unchanged"],
            "unchanged_accumulator_fraction":
                (state["unchanged"] / float(state["values"])
                 if state["values"] else 1.0),
            "mean_absolute_accumulator_delta":
                (state["abs_sum"] / float(state["values"])
                 if state["values"] else 0.0),
            "rms_accumulator_delta":
                (math.sqrt(state["square_sum"] / float(state["values"]))
                 if state["values"] else 0.0),
            "mean_row_maximum_absolute_accumulator_delta":
                (state["row_max_sum"] / float(state["rows"])
                 if state["rows"] else 0.0),
            "p50_row_maximum_absolute_accumulator_delta":
                weighted_quantile(state["row_max_histogram"], 0.50),
            "p95_row_maximum_absolute_accumulator_delta":
                weighted_quantile(state["row_max_histogram"], 0.95),
            "p99_row_maximum_absolute_accumulator_delta":
                weighted_quantile(state["row_max_histogram"], 0.99),
            "worst_row_maximum_absolute_accumulator_delta":
                state["worst_row_max"]
        }
        operator_rows = []
        for op, name in enumerate(operator_names):
            row = per_operator[threshold][op]
            operator_rows.append({
                "operator_index": op,
                "operator": name,
                "bit_sparse_vector_ops_per_block":
                    row["bit_sparse_vector_ops_per_block"],
                "candidate_vector_ops_per_block":
                    row["candidate_vector_ops_per_block"],
                "speedup_vs_bit_sparse":
                    row["bit_sparse_vector_ops_per_block"] /
                    float(row["candidate_vector_ops_per_block"]),
                "approximated_partition_vectors":
                    row["approximated_partition_vectors"],
                "elided_correction_ops_per_block":
                    row["elided_correction_ops_per_block"]
            })
        threshold_rows.append({
            "distance_threshold": threshold,
            "exact_work": {
                "dense_vector_ops_per_block":
                    aggregate["dense_vector_ops_per_block"],
                "bit_sparse_vector_ops_per_block":
                    aggregate["bit_sparse_vector_ops_per_block"],
                "candidate_vector_ops_per_block":
                    aggregate["candidate_vector_ops_per_block"],
                "pwp_ops_per_block": aggregate["pwp_ops_per_block"],
                "correction_ops_per_block":
                    aggregate["correction_ops_per_block"],
                "elided_correction_ops_per_block":
                    aggregate["elided_correction_ops_per_block"],
                "speedup_vs_bit_sparse":
                    aggregate["bit_sparse_vector_ops_per_block"] /
                    float(aggregate["candidate_vector_ops_per_block"])
            },
            "int8_accumulator_error": error,
            "same_resource_cycle_simulations": cycle_rows,
            "operators": operator_rows
        })

    baseline = threshold_rows[0]
    m251_work = m251_result["exact_natural_work"]
    for key in ("dense_vector_ops_per_block",
                "bit_sparse_vector_ops_per_block",
                "candidate_vector_ops_per_block",
                "pwp_ops_per_block", "correction_ops_per_block"):
        require(baseline["exact_work"][key] == m251_work[key],
                "M280 threshold-zero M251 work mismatch: " + key)
    for observed, expected in zip(
            baseline["same_resource_cycle_simulations"],
            m251_result["same_resource_cycle_simulations"]):
        require(observed["port"] == expected["port"],
                "M280 threshold-zero port mismatch")
        for key in ("dense_cycles", "bit_sparse_cycles", "candidate_cycles"):
            require(observed[key] == expected[key],
                    "M280 threshold-zero M251 cycle mismatch: " + key)

    promoted = []
    for row in threshold_rows[1:]:
        for cycle in row["same_resource_cycle_simulations"]:
            if (cycle["port"] == "WIDE144_PWP_96_WEIGHT" and
                    cycle["speedup_vs_bit_sparse"] >= 2.0):
                promoted.append(row["distance_threshold"])
    payload = {
        "schema": "m280_paft_near_match_residual_elision_dse_v1",
        "status": ("PASS_TRACE_OPPORTUNITY_REQUIRES_ACCURACY"
                   if promoted else "PASS_NO_2X_TRACE_OPPORTUNITY"),
        "identity": identities,
        "numpy_version": np.__version__,
        "scope": {
            "checkpoint": "M87 PAFT ep4",
            "bn_policy": "running",
            "samples": 10,
            "operators": operator_names,
            "partition_vectors": expected_vectors,
            "unique_lossy_value_pattern_pairs": len(lossy_pairs)
        },
        "thresholds": threshold_rows,
        "promotion": {
            "thresholds_reaching_2x_wide_conv_cycles": promoted,
            "accuracy_admitted": False,
            "next_gate": "modified-forward S10 followed by paired running-BN valid825 with absolute AEE increase <= 0.02"
        },
        "record_expansion_audit": record_audit,
        "admission": contract["claim_boundary"],
        "claim_boundary": "Exact frozen-trace isolated-Conv opportunity and exact checkpoint-INT8 pre-scale accumulator-error audit only. No snapped network forward, accuracy, RTL, energy, system speedup, PPA or headline is admitted."
    }
    require(sha256(Path(__file__).resolve()) == source_start,
            "M280 analyzer changed during execution")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m280_paft_near_match_residual_elision_dse_r1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M280_PASS status={} promoted={} rows={}".format(
        payload["status"], promoted, expected_vectors), flush=True)


if __name__ == "__main__":
    main()
