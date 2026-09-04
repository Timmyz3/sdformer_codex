#!/usr/bin/env python3
"""Replay the train-only PAFT catalog on the exact PAFT running-BN trace."""

from __future__ import division

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path


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


def load_module(path):
    spec = importlib.util.spec_from_file_location("m251_frozen_m43", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M43 support unpacker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def popcount(value):
    return bin(int(value)).count("1")


def phase_metrics(counter, centers):
    result = Counter()
    for value, count in counter.items():
        population = popcount(value)
        distance, center = min((popcount(value ^ candidate), candidate)
                               for candidate in centers)
        candidate = min(population, 1 + distance)
        result["partition_vectors"] += count
        result["dense_vector_ops_per_block"] += count * 16
        result["bit_sparse_vector_ops_per_block"] += count * population
        result["candidate_vector_ops_per_block"] += count * candidate
        result["matcher_rows"] += count * int(population >= 2)
        result["nonzero_partition_vectors"] += count * int(population != 0)
        if value != 0 and value in centers:
            result["exact_pattern_hits"] += count
        if 1 + distance < population:
            result["pwp_ops_per_block"] += count
            result["correction_ops_per_block"] += count * distance
            result["assignment_rows"] += count
            result["used_center_{}".format(center)] += count
        else:
            result["correction_ops_per_block"] += count * population
    require(result["pwp_ops_per_block"] +
            result["correction_ops_per_block"] ==
            result["candidate_vector_ops_per_block"],
            "candidate work conservation failure")
    return dict(result)


def replay_sample(phases, port, model, geometry):
    output_blocks = geometry["output_blocks"]
    weight_phase_bytes = (geometry["partition_bits"] * output_blocks *
                          model["weight_vector_bytes"])
    pwp_phase_bytes = (geometry["patterns_per_partition"] * output_blocks *
                       model["fixed_pwp_vector_bytes"])
    dram = model["dram_bytes_per_cycle"]
    baseline_load = int(math.ceil(weight_phase_bytes / float(dram)))
    candidate_load = int(math.ceil(
        (weight_phase_bytes + pwp_phase_bytes) / float(dram)))
    dense_cycles = baseline_load
    sparse_cycles = baseline_load
    candidate_cycles = candidate_load
    bindings = Counter()
    components = Counter()
    for phase_index, phase in enumerate(phases):
        weight_service = port["weight_vector_service_cycles"]
        pwp_service = port["pwp_vector_service_cycles"]
        dense_compute = (phase["dense_vector_ops_per_block"] * output_blocks *
                         weight_service)
        sparse_compute = (phase["bit_sparse_vector_ops_per_block"] *
                          output_blocks * weight_service)
        candidate_compute = (
            phase["correction_ops_per_block"] * output_blocks * weight_service +
            phase["pwp_ops_per_block"] * output_blocks * pwp_service)
        matcher = phase["matcher_rows"] + model["matcher_pipeline_cycles"]
        packer = (int(math.ceil(phase["assignment_rows"] /
                                float(model["packer_lanes"]))) +
                  model["packer_pipeline_cycles"])
        next_baseline = baseline_load if phase_index + 1 < len(phases) else 0
        next_candidate = candidate_load if phase_index + 1 < len(phases) else 0
        tail = model["compute_tail_cycles_per_partition"]
        dense_cycles += max(dense_compute, next_baseline) + tail
        sparse_cycles += max(sparse_compute, next_baseline) + tail
        candidates = ((candidate_compute, "compute"),
                      (matcher, "matcher"),
                      (packer, "packer"),
                      (next_candidate, "dma"))
        binding_cycles, binding = max(candidates)
        candidate_cycles += binding_cycles + tail
        bindings[binding] += 1
        components["candidate_compute_cycles_sum"] += candidate_compute
        components["matcher_cycles_sum"] += matcher
        components["packer_cycles_sum"] += packer
        components["next_dma_cycles_sum"] += next_candidate
    return {
        "dense_cycles": dense_cycles,
        "bit_sparse_cycles": sparse_cycles,
        "candidate_cycles": candidate_cycles,
        "speedup_vs_dense": dense_cycles / float(candidate_cycles),
        "speedup_vs_bit_sparse": sparse_cycles / float(candidate_cycles),
        "binding_phases": dict(bindings),
        "component_cycle_sums_not_additive": dict(components),
        "initial_baseline_load_cycles": baseline_load,
        "initial_candidate_load_cycles": candidate_load
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m251_paft_running_bn_train_catalog_fixed12_pwp_cycle_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    source_start = sha256(Path(__file__).resolve())
    identities = {}
    paths = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: {}".format(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "SHA drift for {}: {}".format(name, observed))
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}

    catalog = strict_json(paths["m77_train_only_catalog"])
    catalog_admission = strict_json(paths["m77_catalog_admission"])
    trace = strict_json(paths["m248_paft_running_bn_trace"])
    accuracy = strict_json(paths["m247_paft_control_accuracy"])
    geometry = contract["geometry"]
    cycle_model = contract["same_resource_cycle_model"]

    require(catalog["status"] ==
            "PASS_M77_TRAIN_ONLY_KMEANS_PAFT_CATALOG_ACCURACY_CYCLES_UNADMITTED" and
            catalog["split"]["role"] == "DSEC_TRAIN_ONLY_PAFT_CALIBRATION" and
            catalog["split"]["test_or_validation_data_used"] is False and
            catalog["split"]["train_valid825_key_overlap"] == 0,
            "M77 train-only admission drift")
    require(catalog_admission["catalog_sha256"] ==
            identities["m77_train_only_catalog"]["sha256"] and
            catalog_admission["train_only_admitted"] is True and
            catalog_admission["train_valid825_key_overlap"] == 0,
            "M77 admission-contract drift")
    require(trace["status"] ==
            "PASS_PAFT_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE" and
            trace["identity"]["capture_bn_policy"] == "running" and
            trace["cohort"]["samples"] == geometry["samples"] and
            trace["cohort"]["records"] ==
                geometry["samples"] * geometry["operators"],
            "M248 trace admission drift")
    require(accuracy["status"] ==
            "PASS_SINGLE_SEED_SMALL_POSITIVE_RUNNING_BN_DIRECTION",
            "M247 paired accuracy drift")
    operator_names = trace["cohort"]["operators"]
    require([row["operator"] for row in catalog["operators"]] == operator_names,
            "M77/M248 operator order drift")
    require(geometry["features_per_row"] ==
            geometry["partition_bits"] * geometry["partitions_per_operator"],
            "partition geometry drift")
    require(16 * 127 == 2032 and 2032 <= (1 << 11) - 1,
            "universal signed12 PWP bound drift")

    m43 = load_module(paths["m43_support_unpacker"])
    require(m43.ROWS == geometry["rows_per_operator"] and
            m43.TILES * (m43.TILE_BITS // geometry["partition_bits"]) ==
                geometry["partitions_per_operator"],
            "M43 unpack geometry drift")
    trace_dir = paths["m248_paft_running_bn_trace"].parent
    op_index = {name: index for index, name in enumerate(operator_names)}
    histograms = defaultdict(Counter)
    record_audit = []
    for record_index, record in enumerate(trace["records"]):
        packed_path = trace_dir / record["packed_file"]
        require(packed_path.is_file() and
                sha256(packed_path) == record["packed_file_sha256"],
                "M248 packed payload drift")
        require(record["negative_count"] == 0,
                "M251 fixed support unpacker requires nonnegative trace")
        masks = m43.unpack_record_masks(trace_dir, record)
        reconstructed = 0
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS //
                                          geometry["partition_bits"])
                for subtile in range(m43.TILE_BITS //
                                     geometry["partition_bits"]):
                    value = ((value256 >>
                              (subtile * geometry["partition_bits"])) & 0xffff)
                    histograms[(record["sample_id"],
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
                    reconstructed += popcount(value)
        record_audit.append({
            "sample_id": record["sample_id"],
            "operator_index": record["operator_index"],
            "operator": record["operator"],
            "input_nonzero_count": record["nonzero_count"],
            "expanded_conv3x3_source_events": reconstructed
        })
        print("[M251 HIST] {}/40 sample={} op={} expanded={}".format(
            record_index + 1, record["sample_id"], record["operator_index"],
            reconstructed), flush=True)

    sample_phases = defaultdict(list)
    aggregate = Counter()
    per_operator = [Counter() for _ in operator_names]
    for sample in range(geometry["samples"]):
        for op in range(geometry["operators"]):
            operator = catalog["operators"][op]
            require(len(operator["partitions"]) ==
                    geometry["partitions_per_operator"],
                    "catalog partition extent drift")
            for partition, row in enumerate(operator["partitions"]):
                require(row["partition"] == partition and
                        len(row["patterns"]) == geometry["patterns_per_partition"],
                        "catalog partition/pattern order drift")
                centers = [int(item["value_hex"], 16)
                           for item in row["patterns"]]
                require(len(set(centers)) == len(centers) and
                        all(value != 0 for value in centers),
                        "catalog center uniqueness/domain drift")
                phase = phase_metrics(histograms[(sample, op, partition)], centers)
                require(phase["partition_vectors"] ==
                        geometry["rows_per_operator"],
                        "phase row population drift")
                sample_phases[sample].append(phase)
                aggregate.update(dict(
                    (key, value) for key, value in phase.items()
                    if not key.startswith("used_center_")))
                per_operator[op].update(dict(
                    (key, value) for key, value in phase.items()
                    if not key.startswith("used_center_")))

    expected_vectors = (geometry["samples"] * geometry["operators"] *
                        geometry["rows_per_operator"] *
                        geometry["partitions_per_operator"])
    require(aggregate["partition_vectors"] == expected_vectors,
            "full partition-vector population drift")
    require(aggregate["pwp_ops_per_block"] +
            aggregate["correction_ops_per_block"] ==
            aggregate["candidate_vector_ops_per_block"],
            "aggregate work conservation failure")

    cycle_rows = []
    for port in cycle_model["ports"]:
        totals = Counter()
        per_sample = []
        for sample in range(geometry["samples"]):
            replay = replay_sample(sample_phases[sample], port, cycle_model,
                                   geometry)
            replay["sample_id"] = sample
            per_sample.append(replay)
            for key in ("dense_cycles", "bit_sparse_cycles", "candidate_cycles"):
                totals[key] += replay[key]
            totals.update(replay["binding_phases"])
        cycle_rows.append({
            "port": port["name"],
            "weight_vector_service_cycles": port["weight_vector_service_cycles"],
            "pwp_vector_service_cycles": port["pwp_vector_service_cycles"],
            "dense_cycles": totals["dense_cycles"],
            "bit_sparse_cycles": totals["bit_sparse_cycles"],
            "candidate_cycles": totals["candidate_cycles"],
            "speedup_vs_dense": totals["dense_cycles"] /
                                float(totals["candidate_cycles"]),
            "speedup_vs_bit_sparse": totals["bit_sparse_cycles"] /
                                     float(totals["candidate_cycles"]),
            "binding_phases": {
                "compute": totals["compute"],
                "matcher": totals["matcher"],
                "packer": totals["packer"],
                "dma": totals["dma"]
            },
            "per_sample": per_sample
        })

    operator_rows = []
    for op, name in enumerate(operator_names):
        row = per_operator[op]
        operator_rows.append({
            "operator_index": op,
            "operator": name,
            "partition_vectors": row["partition_vectors"],
            "bit_sparse_vector_ops_per_block": row[
                "bit_sparse_vector_ops_per_block"],
            "candidate_vector_ops_per_block": row[
                "candidate_vector_ops_per_block"],
            "natural_vector_op_speedup":
                row["bit_sparse_vector_ops_per_block"] /
                float(row["candidate_vector_ops_per_block"]),
            "pwp_ops_per_block": row["pwp_ops_per_block"],
            "correction_ops_per_block": row["correction_ops_per_block"],
            "exact_pattern_hits": row["exact_pattern_hits"]
        })

    natural_speedup = (aggregate["bit_sparse_vector_ops_per_block"] /
                       float(aggregate["candidate_vector_ops_per_block"]))
    pwp_payload_bytes = (geometry["operators"] *
                         geometry["partitions_per_operator"] *
                         geometry["patterns_per_partition"] *
                         geometry["output_blocks"] *
                         cycle_model["fixed_pwp_vector_bytes"])
    payload = {
        "schema": "m251_paft_running_bn_train_catalog_fixed12_pwp_cycle_v1",
        "status": "PASS_PAFT_RUNNING_BN_TRAIN_CATALOG_ISOLATED_CONV_CYCLE_MODEL",
        "identity": identities,
        "scope": {
            "checkpoint": "M87 PAFT ep4",
            "bn_policy": "running",
            "samples": geometry["samples"],
            "operators": operator_names,
            "records": len(trace["records"]),
            "partition_vectors": aggregate["partition_vectors"]
        },
        "exact_natural_work": {
            "dense_vector_ops_per_block": aggregate["dense_vector_ops_per_block"],
            "bit_sparse_vector_ops_per_block":
                aggregate["bit_sparse_vector_ops_per_block"],
            "candidate_vector_ops_per_block":
                aggregate["candidate_vector_ops_per_block"],
            "pwp_ops_per_block": aggregate["pwp_ops_per_block"],
            "correction_ops_per_block": aggregate["correction_ops_per_block"],
            "natural_vector_op_speedup_vs_bit_sparse": natural_speedup,
            "candidate_reduction_percent_vs_bit_sparse":
                100.0 * (1.0 - 1.0 / natural_speedup),
            "exact_pattern_hits": aggregate["exact_pattern_hits"],
            "operators": operator_rows
        },
        "fixed12_pwp": {
            "universal_minimum": -16 * 127,
            "universal_maximum": 16 * 127,
            "signed12_minimum": -(1 << 11),
            "signed12_maximum": (1 << 11) - 1,
            "universal_range_safe": True,
            "pattern_table_bytes": (geometry["operators"] *
                                    geometry["partitions_per_operator"] *
                                    geometry["patterns_per_partition"] * 2),
            "all_pwp_payload_bytes": pwp_payload_bytes
        },
        "same_resource_cycle_simulations": cycle_rows,
        "record_expansion_audit": record_audit,
        "algorithm_hardware_context": {
            "paired_running_bn_aee_improvement_percent":
                accuracy["hardware_decision"][
                    "paft_running_aee_improvement_percent"],
            "accuracy_strength": accuracy["hardware_decision"]["strength"],
            "paft_vs_control_hardware_gain_measured": False,
            "reason": "the exact no-PAFT control running-BN bottleneck source trace has not yet been captured"
        },
        "admission": contract["claim_boundary"],
        "claim_boundary": "Isolated four-Conv module cycle simulator on the exact PAFT-ep4 running-BN ten-sample trace and a disjoint train-only catalog. Dense, bit-sparse and fixed12-PWP rows share the declared resources and traffic model. This is not an RTL-integrated cycle match, PAFT-versus-control hardware gain, system speedup, energy, paper PPA or headline."
    }
    require(sha256(Path(__file__).resolve()) == source_start,
            "M251 analyzer changed during execution")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m251_paft_running_bn_train_catalog_fixed12_pwp_cycle_r1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M251_PASS natural={:.6f} wide_dense={:.6f} wide_sparse={:.6f} shared96_dense={:.6f} shared96_sparse={:.6f}".format(
        natural_speedup,
        cycle_rows[0]["speedup_vs_dense"],
        cycle_rows[0]["speedup_vs_bit_sparse"],
        cycle_rows[1]["speedup_vs_dense"],
        cycle_rows[1]["speedup_vs_bit_sparse"]), flush=True)


if __name__ == "__main__":
    main()
