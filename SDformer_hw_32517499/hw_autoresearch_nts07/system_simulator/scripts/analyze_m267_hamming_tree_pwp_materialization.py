#!/usr/bin/env python3
"""Prove exact Hamming-tree materialization of checkpoint-bound PWP vectors."""

import argparse
from collections import Counter
import hashlib
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
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             RuntimeError("non-finite JSON: " + value)))


def popcount(value):
    return bin(int(value)).count("1")


def minimum_tree(values):
    """Return deterministic Prim edges (parent, child, distance)."""
    visited = {0}
    edges = []
    while len(visited) != len(values):
        distance, parent, child = min(
            (popcount(values[parent] ^ values[child]), parent, child)
            for parent in visited
            for child in range(len(values)) if child not in visited)
        require(distance > 0, "duplicate pattern escaped catalog admission")
        visited.add(child)
        edges.append((parent, child, distance))
    require(len(edges) == len(values) - 1, "tree edge population drift")
    return edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m267_hamming_tree_pwp_materialization_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing frozen input: " + str(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "frozen SHA drift {}: {}".format(name, observed))
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}

    catalog = strict_json(paths["m77_train_only_catalog"])
    admission = strict_json(paths["m77_catalog_admission"])
    m251 = strict_json(paths["m251_cycle_result"])
    m256 = strict_json(paths["m256_int8_bridge"])
    geometry = contract["geometry"]
    materializer = contract["materializer"]
    require(catalog["status"] ==
            "PASS_M77_TRAIN_ONLY_KMEANS_PAFT_CATALOG_ACCURACY_CYCLES_UNADMITTED" and
            admission["catalog_sha256"] == identities["m77_train_only_catalog"]["sha256"] and
            admission["train_only_admitted"] is True,
            "M77 catalog admission drift")
    require(m251["status"] ==
            "PASS_PAFT_RUNNING_BN_TRAIN_CATALOG_ISOLATED_CONV_CYCLE_MODEL" and
            m256["status"].startswith(
                "PASS_PAFT_EP4_RUNNING_BN_FOUR_LAYER_INT8_WEIGHT_BRIDGE"),
            "upstream M251/M256 status drift")
    require(len(catalog["operators"]) == geometry["operators"] ==
            len(m256["layers"]), "operator population drift")

    direct_digest = hashlib.sha256()
    tree_digest = hashlib.sha256()
    flip_histogram = Counter()
    edge_distance_histogram = Counter()
    operator_rows = []
    global_min = 0
    global_max = 0
    full_vectors = 0
    block_vectors = 0
    mismatches = 0
    total_flips = 0
    maximum_flips = 0
    first_partition_flips = None

    for operator_index, (catalog_operator, layer) in enumerate(
            zip(catalog["operators"], m256["layers"])):
        require(catalog_operator["operator"] == layer["operator"] and
                layer["operator_index"] == operator_index,
                "catalog/weight operator identity drift")
        weight_payloads = [payload for payload in layer["payloads"]
                           if payload["role"] == "weight"]
        require(len(weight_payloads) == 1, "weight payload population drift")
        payload = weight_payloads[0]
        weight_path = paths["m256_int8_bridge"].parent / payload["file"]
        require(weight_path.is_file() and sha256(weight_path) == payload["sha256"] and
                payload["dtype"] == "signed_int8" and
                payload["layout"] == "I_KY_KX_O_C_ORDER" and
                payload["shape"] == [768, 3, 3, 768],
                "checkpoint weight payload drift")
        weights = np.fromfile(str(weight_path), dtype=np.int8).reshape(
            768 * 3 * 3, 768).astype(np.int32)
        require(weights.shape == (6912, 768), "flattened weight geometry drift")
        op_flips = 0
        op_min = 0
        op_max = 0
        require(len(catalog_operator["partitions"]) ==
                geometry["partitions_per_operator"],
                "partition population drift")
        for partition_index, partition in enumerate(
                catalog_operator["partitions"]):
            require(partition["partition"] == partition_index and
                    len(partition["patterns"]) ==
                    geometry["patterns_per_partition"],
                    "partition/pattern order drift")
            patterns = [int(item["value_hex"], 16)
                        for item in partition["patterns"]]
            require(len(set(patterns)) == 16 and
                    all(0 < value < (1 << 16) for value in patterns),
                    "pattern domain drift")
            values = [0] + patterns
            edges = minimum_tree(values)
            flips = sum(edge[2] for edge in edges)
            if first_partition_flips is None:
                first_partition_flips = flips
            flip_histogram[flips] += 1
            for _, _, distance in edges:
                edge_distance_histogram[distance] += 1
            total_flips += flips
            op_flips += flips
            maximum_flips = max(maximum_flips, flips)

            part_weights = weights[partition_index * 16:
                                   (partition_index + 1) * 16, :]
            bit_matrix = np.asarray(
                [[(pattern >> bit) & 1 for bit in range(16)]
                 for pattern in patterns], dtype=np.int32)
            direct = np.matmul(bit_matrix, part_weights)
            rebuilt = np.zeros((17, 768), dtype=np.int32)
            for parent, child, _ in edges:
                value = rebuilt[parent].copy()
                changed = values[parent] ^ values[child]
                for bit in range(16):
                    if (changed >> bit) & 1:
                        if (values[child] >> bit) & 1:
                            value += part_weights[bit]
                        else:
                            value -= part_weights[bit]
                rebuilt[child] = value
            rebuilt_patterns = rebuilt[1:]
            mismatches += int(np.count_nonzero(direct != rebuilt_patterns))
            local_min = int(direct.min())
            local_max = int(direct.max())
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)
            op_min = min(op_min, local_min)
            op_max = max(op_max, local_max)
            direct_digest.update(direct.astype("<i2", copy=False).tobytes())
            tree_digest.update(rebuilt_patterns.astype("<i2", copy=False).tobytes())
            full_vectors += 16
            block_vectors += 16 * geometry["output_blocks"]
        operator_rows.append({
            "operator_index": operator_index,
            "operator": layer["operator"],
            "partitions": len(catalog_operator["partitions"]),
            "tree_source_vector_add_subtracts":
                op_flips * geometry["output_blocks"],
            "pwp_minimum": op_min,
            "pwp_maximum": op_max,
        })

    expected_partitions = (geometry["operators"] *
                           geometry["partitions_per_operator"])
    require(sum(flip_histogram.values()) == expected_partitions and
            full_vectors == expected_partitions * 16 and
            block_vectors == full_vectors * geometry["output_blocks"],
            "proof population drift")
    require(mismatches == 0 and direct_digest.hexdigest() ==
            tree_digest.hexdigest(), "PWP reconstruction mismatch")
    require(global_min >= -(1 << 11) and global_max <= (1 << 11) - 1,
            "checkpoint PWP escaped signed12")

    output_blocks = geometry["output_blocks"]
    weight_bytes_per_partition = (geometry["source_bits_per_partition"] *
                                  output_blocks *
                                  geometry["output_lanes_per_block"])
    pwp_bytes_per_partition = (geometry["patterns_per_partition"] *
                               output_blocks *
                               geometry["output_lanes_per_block"] *
                               geometry["pwp_bits"] // 8)
    pattern_bytes_per_partition = geometry["patterns_per_partition"] * 2
    tree_bytes_per_partition = (geometry["patterns_per_partition"] *
                                materializer["tree_descriptor_bytes_per_edge"])
    fixed_total = expected_partitions * (
        weight_bytes_per_partition + pwp_bytes_per_partition +
        pattern_bytes_per_partition)
    tree_total = expected_partitions * (
        weight_bytes_per_partition + pattern_bytes_per_partition +
        tree_bytes_per_partition)
    load_cycles = int(math.ceil(
        (weight_bytes_per_partition + pattern_bytes_per_partition +
         tree_bytes_per_partition) /
        float(materializer["dram_bytes_per_cycle"])))
    max_generator_cycles = maximum_flips * output_blocks
    max_serial_materialization = load_cycles + max_generator_cycles
    require(max_serial_materialization <= 960,
            "tree materialization exceeds prior M251 next-partition envelope")
    require(all(row["binding_phases"]["compute"] == 17280 and
                row["binding_phases"]["dma"] == 0
                for row in m251["same_resource_cycle_simulations"]),
            "M251 frozen binding observation drift")

    output = {
        "schema": "m267_hamming_tree_pwp_materialization_v1",
        "status": "PASS_EXACT_PWP_DERIVED_PAYLOAD_ELISION_CYCLES_UNCHANGED",
        "identity": identities,
        "scope": {
            "checkpoint": "M87 PAFT ep4 signed INT8 bridge",
            "catalog": "M77 disjoint DSEC-train-only K16/Q16",
            "operators": geometry["operators"],
            "partitions": expected_partitions,
            "full_768_lane_pwp_vectors": full_vectors,
            "96_lane_pwp_block_vectors": block_vectors,
        },
        "exact_reconstruction": {
            "identity": "PWP(child)=PWP(parent)+sum(0_to_1 W_bit)-sum(1_to_0 W_bit)",
            "direct_sum_sha256_int16_le": direct_digest.hexdigest(),
            "tree_sum_sha256_int16_le": tree_digest.hexdigest(),
            "scalar_mismatches": mismatches,
            "pwp_minimum": global_min,
            "pwp_maximum": global_max,
            "signed12_safe": True,
        },
        "minimum_hamming_trees": {
            "algorithm": materializer["algorithm"],
            "partition_flip_count_histogram": {
                str(key): value for key, value in sorted(flip_histogram.items())},
            "edge_distance_histogram": {
                str(key): value for key, value in
                sorted(edge_distance_histogram.items())},
            "minimum_flips_per_partition": min(flip_histogram),
            "mean_flips_per_partition": total_flips / float(expected_partitions),
            "maximum_flips_per_partition": maximum_flips,
            "first_partition_flips": first_partition_flips,
            "total_96_lane_add_subtract_cycles": total_flips * output_blocks,
            "operator_rows": operator_rows,
        },
        "storage_and_dram": {
            "weight_bytes": expected_partitions * weight_bytes_per_partition,
            "fixed12_pwp_payload_bytes":
                expected_partitions * pwp_bytes_per_partition,
            "pattern_table_bytes": expected_partitions * pattern_bytes_per_partition,
            "tree_descriptor_bytes": expected_partitions * tree_bytes_per_partition,
            "fixed_pwp_total_bytes_per_catalog_pass": fixed_total,
            "tree_materialized_total_bytes_per_catalog_pass": tree_total,
            "pwp_payload_elimination_percent": 100.0,
            "total_byte_reduction_percent":
                100.0 * (1.0 - tree_total / float(fixed_total)),
        },
        "cycle_envelope": {
            "prior_m251_next_partition_dma_cycles": 960,
            "tree_weight_pattern_descriptor_load_cycles": load_cycles,
            "maximum_tree_generator_cycles": max_generator_cycles,
            "maximum_serial_load_plus_generator_cycles":
                max_serial_materialization,
            "all_partitions_within_prior_envelope": True,
            "m251_binding_phases_remain_compute": 17280,
            "m251_candidate_cycles_unchanged": {
                row["port"]: row["candidate_cycles"]
                for row in m251["same_resource_cycle_simulations"]},
            "new_speedup_admitted": False,
        },
        "hardware_innovation": (
            "PWP values are checkpoint-derived products, not independent model "
            "state.  A 17-node Hamming tree converts their off-chip storage and "
            "DMA into bounded on-chip signed weight-vector updates while reusing "
            "the existing per-partition PWP buffer."
        ),
        "admission": {
            "checkpoint_bound_exactness": True,
            "signed12": True,
            "pwp_dram_payload_elided": True,
            "hidden_under_m251_partition_envelope": True,
            "new_arithmetic_speedup": False,
            "system_speedup": False,
            "power_energy": False,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "formality": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    require(sha256(Path(__file__).resolve()) == source_start,
            "M267 analyzer changed during execution")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = (args.output_dir /
                   "m267_hamming_tree_pwp_materialization_r1.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("M267_PASS mismatch={} range=[{},{}] flips_mean={:.6f} "
          "flips_max={} bytes_save={:.6f}% envelope={}/960".format(
              mismatches, global_min, global_max,
              output["minimum_hamming_trees"]["mean_flips_per_partition"],
              maximum_flips,
              output["storage_and_dram"]["total_byte_reduction_percent"],
              max_serial_materialization), flush=True)


if __name__ == "__main__":
    main()
