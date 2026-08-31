#!/usr/bin/env python3
"""Freeze an exact q32-parent plus three-local-child catalog from M73 only.

The 32 parents are copied bit-for-bit from the sealed M430 catalog.  Every
train-observed 16-bit mask is routed once to its nearest q32 parent (lowest
parent ID wins).  Inside each disjoint route bucket, three non-parent masks
are selected by a deterministic weighted greedy minimization of the exact
dual-port issue objective.  No M40/runtime payload is accepted by this tool.
"""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


K = 16
PARENTS = 32
CHILDREN_PER_PARENT = 3
TOTAL_PATTERNS = PARENTS * (1 + CHILDREN_PER_PARENT)
PARTITIONS = 432
SAMPLES = 32
POPCOUNT = np.asarray([bin(value).count("1")
                       for value in range(1 << K)], dtype=np.uint8)


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
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hamming(a, b):
    return int(POPCOUNT[(int(a) ^ int(b)) & 0xffff])


def route_to_parents(values, parents):
    parent_array = np.asarray(parents, dtype=np.uint16)
    matrix = POPCOUNT[np.bitwise_xor(
        parent_array[:, None], values[None, :])]
    # NumPy argmin is the required lowest-parent-ID tie break.
    return matrix.argmin(axis=0), matrix


def issue_units(values, distance):
    pops = POPCOUNT[values].astype(np.int16)
    distance = distance.astype(np.int16)
    return np.where(1 + distance < pops, 1 + distance, pops)


def greedy_children(bucket_values, bucket_counts, parent, all_parents):
    """Select three unique children using weighted exact issue cost.

    Candidate tie break is ascending numeric mask.  The parent remains a
    legal local center at every greedy step.  All q32 parent masks are
    excluded so the final 32+96 catalog is globally unique.
    """
    parent_set = set(int(value) for value in all_parents)
    candidates = sorted(int(value) for value in bucket_values
                        if int(value) not in parent_set)
    require(len(candidates) >= CHILDREN_PER_PARENT,
            "parent route bucket has fewer than three non-parent masks")
    values = np.asarray(bucket_values, dtype=np.uint16)
    counts = np.asarray(bucket_counts, dtype=np.int64)
    best_distance = POPCOUNT[np.bitwise_xor(values, int(parent))]
    objectives = [int(np.dot(counts,
                             issue_units(values, best_distance)
                             .astype(np.int64)))]
    selected = []
    for _ in range(CHILDREN_PER_PARENT):
        best_key = None
        best_value = None
        best_next = None
        for candidate in candidates:
            if candidate in selected:
                continue
            candidate_distance = POPCOUNT[
                np.bitwise_xor(values, candidate)]
            next_distance = np.minimum(best_distance, candidate_distance)
            objective = int(np.dot(
                counts, issue_units(values, next_distance)
                .astype(np.int64)))
            key = (objective, candidate)
            if best_key is None or key < best_key:
                best_key = key
                best_value = candidate
                best_next = next_distance
        require(best_value is not None, "child greedy exhausted candidates")
        selected.append(best_value)
        best_distance = best_next
        objectives.append(best_key[0])
    require(len(selected) == 3 and len(set(selected)) == 3,
            "child selection extent failure")
    return selected, objectives


def evaluate_counter(counter, parents, children_by_parent):
    result = Counter()
    for original, population in counter.items():
        original = int(original) & 0xffff
        population = int(population)
        pop = int(POPCOUNT[original])
        result["source_rows"] += population
        if original == 0:
            result["zero_rows"] += population
            continue
        result["active_rows"] += population
        parent_distances = [hamming(original, value) for value in parents]
        parent_id = parent_distances.index(min(parent_distances))
        local = [parents[parent_id]] + children_by_parent[parent_id]
        distances = [hamming(original, value) for value in local]
        local_id = distances.index(min(distances))
        distance = distances[local_id]
        global_id = (parent_id if local_id == 0 else
                     PARENTS + parent_id * CHILDREN_PER_PARENT + local_id - 1)
        use_pwp = 1 + distance < pop
        correction = distance if use_pwp else pop
        result["pwp_rows"] += population * int(use_pwp)
        result["fallback_rows"] += population * int(not use_pwp)
        result["exact_pwp_rows"] += population * int(use_pwp and distance == 0)
        result["positive_residual_pwp_rows"] += population * int(
            use_pwp and distance > 0)
        result["correction_ops_per_block"] += population * correction
        result["separate_issues_per_block"] += population * (
            int(use_pwp) + correction)
        result["fused_k1_issues_per_block"] += population * (
            (max(1, correction) if use_pwp else correction))
        if use_pwp:
            result[("used_pattern", global_id)] += population
    require(result["source_rows"] ==
            result["zero_rows"] + result["active_rows"] and
            result["active_rows"] ==
            result["pwp_rows"] + result["fallback_rows"] and
            result["pwp_rows"] ==
            result["exact_pwp_rows"] +
            result["positive_residual_pwp_rows"],
            "train population conservation failure")
    return result


def static_pwp_range(weight_slice, patterns):
    bits = np.asarray([[(pattern >> bit) & 1 for bit in range(K)]
                       for pattern in patterns], dtype=np.int16)
    products = bits @ weight_slice
    require(products.shape == (TOTAL_PATTERNS, 768),
            "q128 PWP product shape drift")
    return int(products.min()), int(products.max())


def write_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(output_dir / name), name)
        for name in sorted(names)), encoding="utf-8")
    seal = output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")
    return manifest, seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M453a output overwrite")
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m453a_trainonly_hierarchical_q32x3_catalog_contract_v1" and
            contract.get("status") ==
            "FROZEN_BEFORE_M453A_TRAIN_EXECUTION_M40_FORBIDDEN",
            "M453a contract status drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M453a input SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": spec["path"],
                            "sha256": spec["sha256"]}
    require(paths["builder"].resolve() == Path(__file__).resolve() and
            identities["builder"]["sha256"] == source_start,
            "M453a builder self identity drift")

    manifest = strict_json(paths["m73_train_trace_manifest"])
    m430 = strict_json(paths["m430_catalog"])
    require(manifest["status"] ==
            "PASS_M73_DSEC_TRAIN_ONLY_S32_ALL18_SEQUENCES_EXACT_H67_EP35_FOUR_BOTTLENECK_TRACE" and
            manifest["split_audit"]["role"] ==
            "DSEC_TRAIN_ONLY_PAFT_CALIBRATION" and
            manifest["split_audit"]["selected_samples"] == SAMPLES and
            manifest["split_audit"]["selected_sequences"] == 18 and
            manifest["split_audit"]["full_train_valid825_key_overlap"] == 0,
            "M453a M73 split identity drift")
    require(m430["status"] ==
            "PASS_M430_TRAIN_ONLY_DUALAWARE_Q32_FROZEN_BEFORE_HELDOUT" and
            m430["split"]["runtime_or_validation_data_used"] is False and
            m430["geometry"]["q_capacity"] == PARENTS,
            "M453a M430 parent identity drift")

    # Verify the parent catalog's two seal layers again before using it.
    parent_dir = paths["m430_catalog"].parent
    parent_manifest = paths["m430_manifest"]
    for line in parent_manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(sha256(parent_dir / name) == expected,
                "M453a M430 inner seal mismatch: " + name)
    expected, name = paths["m430_seal"].read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(name == "SHA256SUMS" and sha256(parent_manifest) == expected,
            "M453a M430 outer seal mismatch")

    operators = tuple(manifest["cohort"]["operators"])
    require(len(operators) == 4 and
            [item["operator"] for item in m430["operators"]] ==
            list(operators), "M453a operator order drift")
    helper = load_module(paths["m423_train_helper"], "m453a_helper")
    m43 = helper.load_module(paths["m43_unpacker"])
    histograms, payload_files, payload_bytes = helper.collect_histograms(
        m43, manifest, paths["m73_train_trace_manifest"], operators)
    weights = [np.fromfile(paths["weight_o{}".format(op)], dtype=np.int8)
               .reshape(6912, 768).astype(np.int16) for op in range(4)]

    operator_payloads = []
    route_rows = []
    train_total = Counter()
    static_minimum = 1 << 30
    static_maximum = -(1 << 30)
    total_children = 0
    minimum_bucket_candidates = 1 << 30
    maximum_bucket_candidates = 0
    for op, operator in enumerate(operators):
        partitions = []
        for partition in range(PARTITIONS):
            aggregate = Counter()
            sample_counters = []
            for sample in range(SAMPLES):
                counter = histograms[(sample, op, partition)]
                require(sum(counter.values()) == 3000,
                        "M453a train phase row extent drift")
                sample_counters.append(counter)
                aggregate.update(counter)
            values = np.asarray(sorted(aggregate), dtype=np.uint16)
            counts = np.asarray([aggregate[int(value)] for value in values],
                                dtype=np.int64)
            parents = [int(value, 16) for value in
                       m430["operators"][op]["partitions"][partition]
                       ["nested_patterns"][:PARENTS]]
            require(len(parents) == PARENTS and len(set(parents)) == PARENTS,
                    "M453a q32 parent extent drift")
            parent_ids, parent_matrix = route_to_parents(values, parents)
            children_by_parent = []
            parent_audits = []
            for parent_id, parent in enumerate(parents):
                indices = np.flatnonzero(parent_ids == parent_id)
                require(len(indices) > 0, "M453a empty parent route bucket")
                bucket_values = values[indices]
                bucket_counts = counts[indices]
                child_pool_count = sum(
                    int(value) not in set(parents)
                    for value in bucket_values)
                minimum_bucket_candidates = min(minimum_bucket_candidates,
                                                child_pool_count)
                maximum_bucket_candidates = max(maximum_bucket_candidates,
                                                child_pool_count)
                children, objectives = greedy_children(
                    bucket_values, bucket_counts, parent, parents)
                children_by_parent.append(children)
                total_children += len(children)
                routed_population = int(bucket_counts.sum())
                parent_audits.append({
                    "parent_id": parent_id,
                    "parent_hex": "{:04x}".format(parent),
                    "children_hex": ["{:04x}".format(value)
                                     for value in children],
                    "routed_unique_masks": int(len(indices)),
                    "eligible_nonparent_unique_masks": child_pool_count,
                    "routed_population": routed_population,
                    "weighted_issue_objective_parent_then_each_child":
                        objectives,
                })
                route_rows.append({
                    "operator": op, "partition": partition,
                    "parent_id": parent_id,
                    "parent_hex": "{:04x}".format(parent),
                    "child0_hex": "{:04x}".format(children[0]),
                    "child1_hex": "{:04x}".format(children[1]),
                    "child2_hex": "{:04x}".format(children[2]),
                    "routed_unique_masks": len(indices),
                    "eligible_nonparent_unique_masks": child_pool_count,
                    "routed_population": routed_population,
                    "objective_parent": objectives[0],
                    "objective_child1": objectives[1],
                    "objective_child2": objectives[2],
                    "objective_child3": objectives[3],
                })
            all_children = [value for group in children_by_parent
                            for value in group]
            patterns = parents + all_children
            require(len(patterns) == TOTAL_PATTERNS and
                    len(set(patterns)) == TOTAL_PATTERNS and
                    not (set(parents) & set(all_children)),
                    "M453a global 32+96 uniqueness failure")
            minimum, maximum = static_pwp_range(
                weights[op][partition * K:(partition + 1) * K], patterns)
            require(minimum >= -2048 and maximum <= 2047,
                    "M453a signed12 PWP overflow")
            static_minimum = min(static_minimum, minimum)
            static_maximum = max(static_maximum, maximum)
            for counter in sample_counters:
                train_total.update(evaluate_counter(
                    counter, parents, children_by_parent))
            partitions.append({
                "partition": partition,
                "parent_patterns": ["{:04x}".format(value)
                                    for value in parents],
                "children_by_parent": [["{:04x}".format(value)
                                         for value in group]
                                        for group in children_by_parent],
                "flat_patterns": ["{:04x}".format(value)
                                  for value in patterns],
                "parent_route_audit": parent_audits,
            })
            if (partition + 1) % 108 == 0:
                print("[M453A TRAIN] operator={}/4 partition={}/432".format(
                    op + 1, partition + 1), flush=True)
        operator_payloads.append({"operator": operator,
                                  "partitions": partitions})

    expected_rows = SAMPLES * 4 * PARTITIONS * 3000
    require(train_total["source_rows"] == expected_rows and
            total_children == 4 * PARTITIONS * 96,
            "M453a global extent drift")
    # Counter tuple keys are audit-only and must not enter JSON.
    used_assignments = sum(value for key, value in train_total.items()
                           if isinstance(key, tuple) and
                           key[0] == "used_pattern")
    clean_total = {str(key): int(value) for key, value in train_total.items()
                   if isinstance(key, str)}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    catalog = {
        "schema": "m453a_trainonly_hierarchical_q32x3_catalog_v1",
        "status": "PASS_M453A_TRAIN_ONLY_TREE_FROZEN_BEFORE_M40",
        "identity": identities,
        "split": {
            "role": "DSEC_TRAIN_ONLY_SECONDARY_HIERARCHICAL_CATALOG",
            "selected_train_samples": SAMPLES,
            "selected_train_sequences": 18,
            "train_valid825_key_overlap": 0,
            "m40_or_validation_data_used": False,
        },
        "algorithm": {
            "parent_catalog": "M430 q32 bit-identical including order",
            "train_routing":
                "nearest q32 parent by Hamming distance; lowest parent ID tie",
            "children":
                "three unique train-observed non-parent masks per disjoint parent route bucket",
            "greedy_objective":
                "weighted exact dual issue units min(popcount(x),1+nearest-local-Hamming); ascending numeric mask tie",
            "runtime_selection":
                "first 32 parents, then selected parent plus its three children; local tie order parent,child0,child1,child2",
            "runtime_arithmetic":
                "old_psum += PWP[p] + W*(x-p), else old_psum += W*x",
            "comparisons_per_row": 35,
            "random_seed": None,
            "accuracy_loss": False,
        },
        "geometry": {
            "partition_bits": K,
            "partitions_per_operator": PARTITIONS,
            "operators": list(operators),
            "parent_capacity": PARENTS,
            "children_per_parent": CHILDREN_PER_PARENT,
            "total_pwp_capacity": TOTAL_PATTERNS,
            "output_blocks": 8,
            "shared_lanes": 96,
            "pwp_stride_bytes_per_four_output_blocks": 640,
        },
        "operators": operator_payloads,
        "admission": {
            "train_only_catalog": True,
            "m430_q32_parent_bit_identical": True,
            "children_from_train_observed_masks_only": True,
            "global_32_plus_96_unique": True,
            "exact_arithmetic_identity": True,
            "checkpoint_or_accuracy_changed": False,
            "m40_runtime_evaluated": False,
            "cycle_speedup": False,
            "selected_rtl": False,
            "synopsys": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "claim_boundary":
            "Train-only exact catalog predesign artifact. M40 was forbidden and remains unread by M453a; no heldout cycles, RTL, PPA, power, full-network or headline claim.",
    }
    catalog_path = args.output_dir / (
        "m453a_trainonly_hierarchical_q32x3_catalog_r1.json")
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) +
                            "\n", encoding="utf-8")
    audit = {
        "schema": "m453a_trainonly_hierarchical_q32x3_catalog_audit_v1",
        "status": "PASS_M453A_DOUBLE_SEAL_READY_M40_NOT_READ",
        "identity": identities,
        "payload_audit": {"files_rehashed": payload_files,
                          "bytes_rehashed": payload_bytes,
                          "mismatches": 0},
        "catalog_extent": {
            "operators": 4, "partitions_per_operator": PARTITIONS,
            "parents_per_partition": PARENTS,
            "children_per_parent": CHILDREN_PER_PARENT,
            "total_patterns_per_partition": TOTAL_PATTERNS,
            "total_children": total_children,
            "minimum_eligible_children_in_any_parent_bucket":
                minimum_bucket_candidates,
            "maximum_eligible_children_in_any_parent_bucket":
                maximum_bucket_candidates,
        },
        "train_observation": {
            **clean_total,
            "pwp_rows_with_a_used_pattern_assignment": used_assignments,
            "static_pwp_minimum": static_minimum,
            "static_pwp_maximum": static_maximum,
            "signed12_overflows": 0,
        },
        "exactness": {
            "m430_parent_mismatches": 0,
            "child_outside_train_route_bucket_mismatches": 0,
            "parent_or_child_uniqueness_mismatches": 0,
            "population_conservation_mismatches": 0,
            "arithmetic_identity": True,
            "checkpoint_changed": False,
            "accuracy_loss": False,
        },
        "heldout_gate": {
            "m40_payload_reads_so_far": 0,
            "m40_completed_evaluations_so_far": 0,
            "catalog_sealed_before_m40": True,
            "post_m40_tuning_allowed": False,
        },
        "admission": catalog["admission"],
    }
    audit_path = args.output_dir / (
        "m453a_trainonly_hierarchical_q32x3_catalog_audit_r1.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    csv_path = args.output_dir / "m453a_parent_child_train_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(route_rows[0]))
        writer.writeheader()
        writer.writerows(route_rows)
    require(source_start == sha256(Path(__file__).resolve()),
            "M453a builder changed during execution")
    _, seal = write_seal(args.output_dir, [
        catalog_path.name, audit_path.name, csv_path.name])
    print("PASS_M453A_TRAIN_ONLY_TREE patterns=128 children={} "
          "signed12=[{},{}] m40_reads=0 seal={}".format(
              total_children, static_minimum, static_maximum, sha256(seal)),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
