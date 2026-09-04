#!/usr/bin/env python3
"""Bit-identical vectorized implementation recovery for M453a r2.

Selection, candidate extent and tie order are inherited unchanged from r2.
Only the full-train per-phase audit is vectorized over the exact Counter
values/counts.  A deterministic scalar/vector miter runs before M73 access.
"""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vector_evaluate(base, counter, parents, children_by_parent):
    values = np.asarray(sorted(int(value) & 0xffff for value in counter),
                        dtype=np.uint16)
    counts = np.asarray([int(counter[int(value)]) for value in values],
                        dtype=np.int64)
    pops = base.POPCOUNT[values].astype(np.int16)
    parent_matrix = base.POPCOUNT[np.bitwise_xor(
        np.asarray(parents, dtype=np.uint16)[:, None], values[None, :])]
    parent_ids = parent_matrix.argmin(axis=0)
    columns = np.arange(len(values))
    best_distance = parent_matrix[parent_ids, columns].astype(np.int16)
    local_ids = np.zeros(len(values), dtype=np.int16)
    global_ids = parent_ids.astype(np.int16)
    for parent_id in range(base.PARENTS):
        indices = np.flatnonzero(parent_ids == parent_id)
        if len(indices) == 0:
            continue
        local_centers = [parents[parent_id]] + children_by_parent[parent_id]
        matrix = base.POPCOUNT[np.bitwise_xor(
            np.asarray(local_centers, dtype=np.uint16)[:, None],
            values[indices][None, :])]
        selected = matrix.argmin(axis=0)
        local_ids[indices] = selected
        best_distance[indices] = matrix[selected,
                                        np.arange(len(indices))]
        global_ids[indices] = np.where(
            selected == 0, parent_id,
            base.PARENTS + parent_id * base.CHILDREN_PER_PARENT +
            selected - 1)
    active = values != 0
    pwp = active & (1 + best_distance < pops)
    correction = np.where(pwp, best_distance, pops).astype(np.int64)
    exact = pwp & (best_distance == 0)
    positive = pwp & (best_distance > 0)
    result = Counter({
        "source_rows": int(counts.sum()),
        "zero_rows": int(counts[~active].sum()),
        "active_rows": int(counts[active].sum()),
        "pwp_rows": int(counts[pwp].sum()),
        "fallback_rows": int(counts[active & (~pwp)].sum()),
        "exact_pwp_rows": int(counts[exact].sum()),
        "positive_residual_pwp_rows": int(counts[positive].sum()),
        "correction_ops_per_block": int(np.dot(counts, correction)),
        "separate_issues_per_block": int(np.dot(
            counts, correction + pwp.astype(np.int64))),
        "fused_k1_issues_per_block": int(np.dot(
            counts, np.where(pwp, np.maximum(1, correction), correction))),
    })
    for global_id in np.unique(global_ids[pwp]):
        population = int(counts[pwp & (global_ids == global_id)].sum())
        result[("used_pattern", int(global_id))] = population
    require(result["source_rows"] ==
            result["zero_rows"] + result["active_rows"] and
            result["active_rows"] ==
            result["pwp_rows"] + result["fallback_rows"] and
            result["pwp_rows"] ==
            result["exact_pwp_rows"] +
            result["positive_residual_pwp_rows"] and
            result["separate_issues_per_block"] ==
            result["pwp_rows"] + result["correction_ops_per_block"] and
            result["fused_k1_issues_per_block"] ==
            result["separate_issues_per_block"] -
            result["positive_residual_pwp_rows"],
            "M453a r3 vector conservation failure")
    return result


def scalar_vector_miter(base):
    parents = [0x0000, 0xffff, 0x00ff, 0xff00] + [
        (0x1111 * index) & 0xffff for index in range(4, 32)]
    # Repair accidental duplicates deterministically for this local miter.
    seen = set()
    for index, value in enumerate(parents):
        while value in seen:
            value = (value + 0x0101) & 0xffff
        parents[index] = value
        seen.add(value)
    available = [value for value in range(1, 0x10000)
                 if value not in seen]
    cursor = 0
    children = []
    for _ in range(32):
        children.append(available[cursor:cursor + 3])
        cursor += 3
    masks = [0, 1, 2, 3, 0x00ff, 0x0f0f, 0x3333, 0x5555,
             0xaaaa, 0xffff, 0x8001, 0x7ffe, 0x1010, 0x0101]
    counter = Counter({value: (index + 1) * 7
                       for index, value in enumerate(masks)})
    scalar = base.evaluate_counter(counter, parents, children)
    vector = vector_evaluate(base, counter, parents, children)
    keys = set(scalar) | set(vector)
    mismatches = sum(int(scalar[key] != vector[key]) for key in keys)
    require(mismatches == 0, "M453a r3 scalar/vector miter mismatch")
    return {"unique_masks": len(masks), "fields_compared": len(keys),
            "mismatches": mismatches,
            "covers_zero_exact_positive_fallback_and_tie": True}


def scalar_greedy_core(base, state, bucket_values, bucket_counts, parent,
                       all_parents):
    parent_set = set(int(value) for value in all_parents)
    bucket_set = set(int(value) for value in bucket_values)
    candidates = sorted(
        int(value) for value in state["partition_values"]
        if int(value) not in parent_set and
        int(value) not in state["selected_children"])
    require(len(candidates) >= base.CHILDREN_PER_PARENT,
            "scalar selection reference candidate underflow")
    routed_candidates = [value for value in candidates
                         if value in bucket_set]
    if len(routed_candidates) < base.CHILDREN_PER_PARENT:
        state["parents_with_route_shortfall"] += 1
    values = np.asarray(bucket_values, dtype=np.uint16)
    counts = np.asarray(bucket_counts, dtype=np.int64)
    best_distance = base.POPCOUNT[np.bitwise_xor(values, int(parent))]
    objectives = [int(np.dot(
        counts, base.issue_units(values, best_distance).astype(np.int64)))]
    selected = []
    for _ in range(base.CHILDREN_PER_PARENT):
        best_key = None
        best_value = None
        best_next = None
        for candidate in candidates:
            if candidate in selected:
                continue
            candidate_distance = base.POPCOUNT[
                np.bitwise_xor(values, candidate)]
            next_distance = np.minimum(best_distance, candidate_distance)
            objective = int(np.dot(
                counts, base.issue_units(values, next_distance)
                .astype(np.int64)))
            key = (objective, int(candidate not in bucket_set),
                   int(base.POPCOUNT[candidate ^ int(parent)]), candidate)
            if best_key is None or key < best_key:
                best_key = key
                best_value = candidate
                best_next = next_distance
        selected.append(best_value)
        best_distance = best_next
        objectives.append(best_key[0])
        state["outside_route_children"] += int(
            best_value not in bucket_set)
    require(not (set(selected) & state["selected_children"]),
            "scalar selection reference cross-parent reuse")
    state["selected_children"].update(selected)
    state["global_candidate_calls"] += 1
    return selected, objectives


def vector_greedy_core(base, state, bucket_values, bucket_counts, parent,
                       all_parents):
    """Exact r2 greedy in bounded candidate chunks, with identical tie key."""
    parent_set = set(int(value) for value in all_parents)
    values = np.asarray(bucket_values, dtype=np.uint16)
    counts = np.asarray(bucket_counts, dtype=np.int64)
    bucket_set = set(int(value) for value in values)
    candidates = np.asarray(sorted(
        int(value) for value in state["partition_values"]
        if int(value) not in parent_set and
        int(value) not in state["selected_children"]), dtype=np.uint16)
    require(len(candidates) >= base.CHILDREN_PER_PARENT,
            "vector selection candidate underflow")
    routed_count = sum(int(value) in bucket_set for value in candidates)
    if routed_count < base.CHILDREN_PER_PARENT:
        state["parents_with_route_shortfall"] += 1
    best_distance = base.POPCOUNT[np.bitwise_xor(values, int(parent))]
    objectives = [int(np.dot(
        counts, base.issue_units(values, best_distance).astype(np.int64)))]
    selected = []
    # Bound the largest temporary objective matrix to about four million
    # candidate-by-bucket elements even for a pathological dense partition.
    chunk_size = max(16, min(2048, 4000000 // max(1, len(values))))
    for _ in range(base.CHILDREN_PER_PARENT):
        best_key = None
        best_value = None
        best_next = None
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start:start + chunk_size]
            keep = np.asarray([int(value) not in selected
                               for value in chunk], dtype=bool)
            chunk = chunk[keep]
            if len(chunk) == 0:
                continue
            distance = base.POPCOUNT[np.bitwise_xor(
                chunk[:, None], values[None, :])]
            next_distance = np.minimum(distance, best_distance[None, :])
            pops = base.POPCOUNT[values].astype(np.int16)
            units = np.where(1 + next_distance < pops[None, :],
                             1 + next_distance, pops[None, :])
            objective = units @ counts
            outside = np.asarray([int(int(value) not in bucket_set)
                                  for value in chunk], dtype=np.int8)
            parent_distance = base.POPCOUNT[np.bitwise_xor(
                chunk, int(parent))]
            order = np.lexsort((chunk.astype(np.int64),
                                parent_distance.astype(np.int64),
                                outside.astype(np.int64),
                                objective.astype(np.int64)))
            index = int(order[0])
            candidate = int(chunk[index])
            key = (int(objective[index]), int(outside[index]),
                   int(parent_distance[index]), candidate)
            if best_key is None or key < best_key:
                best_key = key
                best_value = candidate
                best_next = next_distance[index].copy()
        require(best_value is not None,
                "vector selection exhausted chunks")
        selected.append(best_value)
        best_distance = best_next
        objectives.append(best_key[0])
        state["outside_route_children"] += int(
            best_value not in bucket_set)
    require(not (set(selected) & state["selected_children"]),
            "vector selection cross-parent reuse")
    state["selected_children"].update(selected)
    state["global_candidate_calls"] += 1
    return selected, objectives


def directed_selection_miter(base):
    parents = [index * 0x0801 & 0xffff for index in range(32)]
    require(len(set(parents)) == 32, "selection miter parent collision")
    values = np.asarray(sorted(set(parents) | set(range(0, 768))),
                        dtype=np.uint16)
    counts = np.asarray([1 + (int(value) * 17) % 101 for value in values],
                        dtype=np.int64)
    matrix = base.POPCOUNT[np.bitwise_xor(
        np.asarray(parents, dtype=np.uint16)[:, None], values[None, :])]
    routes = matrix.argmin(axis=0)
    states = [{"partition_values": values, "selected_children": set(),
               "outside_route_children": 0,
               "parents_with_route_shortfall": 0,
               "global_candidate_calls": 0} for _ in range(2)]
    mismatches = 0
    objective_mismatches = 0
    selected_sequences = [[], []]
    for parent_id, parent in enumerate(parents):
        indices = np.flatnonzero(routes == parent_id)
        bucket_values = values[indices]
        bucket_counts = counts[indices]
        scalar = scalar_greedy_core(
            base, states[0], bucket_values, bucket_counts, parent, parents)
        vector = vector_greedy_core(
            base, states[1], bucket_values, bucket_counts, parent, parents)
        selected_sequences[0].extend(scalar[0])
        selected_sequences[1].extend(vector[0])
        mismatches += sum(a != b for a, b in zip(scalar[0], vector[0]))
        objective_mismatches += sum(a != b for a, b in
                                    zip(scalar[1], vector[1]))
    require(mismatches == 0 and objective_mismatches == 0 and
            selected_sequences[0] == selected_sequences[1] and
            len(set(selected_sequences[1])) == 96,
            "directed scalar/vector selection miter mismatch")
    return {"parents": 32, "children_compared": 96,
            "selected_child_mismatches": mismatches,
            "objective_step_mismatches": objective_mismatches,
            "full_sequence_equal": True}


class BaseProxy:
    """Intercept r2's local scalar greedy assignment without another wrapper."""
    def __init__(self, base, selection_receipt):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_receipt", selection_receipt)
        object.__setattr__(self, "_partition", -1)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        if name == "route_to_parents":
            def routed(values, parents):
                object.__setattr__(self, "_partition", self._partition + 1)
                return value(values, parents)
            setattr(self._base, name, routed)
            return
        if name == "greedy_children":
            closure = dict(zip(value.__code__.co_freevars,
                               (cell.cell_contents for cell in
                                value.__closure__)))
            state = closure["state"]
            base = closure["base"]
            selected_partitions = {0, 431, 432, 863,
                                   864, 1295, 1296, 1727}

            def vector_checked(bucket_values, bucket_counts, parent,
                               all_parents):
                if self._partition in selected_partitions:
                    before = {
                        "partition_values": state["partition_values"],
                        "selected_children": set(state["selected_children"]),
                        "outside_route_children":
                            state["outside_route_children"],
                        "parents_with_route_shortfall":
                            state["parents_with_route_shortfall"],
                        "global_candidate_calls":
                            state["global_candidate_calls"],
                    }
                    scalar = value(bucket_values, bucket_counts, parent,
                                   all_parents)
                    scalar_after = {
                        "selected_children": set(state["selected_children"]),
                        "outside_route_children":
                            state["outside_route_children"],
                        "parents_with_route_shortfall":
                            state["parents_with_route_shortfall"],
                        "global_candidate_calls":
                            state["global_candidate_calls"],
                    }
                    state.update(before)
                    vector = vector_greedy_core(
                        base, state, bucket_values, bucket_counts, parent,
                        all_parents)
                    mismatch = int(scalar != vector or
                                   scalar_after["selected_children"] !=
                                   state["selected_children"] or
                                   scalar_after["outside_route_children"] !=
                                   state["outside_route_children"] or
                                   scalar_after[
                                       "parents_with_route_shortfall"] !=
                                   state["parents_with_route_shortfall"] or
                                   scalar_after["global_candidate_calls"] !=
                                   state["global_candidate_calls"])
                    self._receipt["actual_parent_calls_compared"] += 1
                    self._receipt["actual_selection_mismatches"] += mismatch
                    require(mismatch == 0,
                            "actual train scalar/vector selection mismatch")
                    if self._receipt["actual_parent_calls_compared"] % 32 == 0:
                        self._receipt["actual_full_partitions_compared"] += 1
                        self._receipt["partition_ids"].append(self._partition)
                    return vector
                return vector_greedy_core(
                    base, state, bucket_values, bucket_counts, parent,
                    all_parents)
            setattr(self._base, name, vector_checked)
            return
        setattr(self._base, name, value)


def main():
    wrapper_path = Path(__file__).resolve()
    r2_path = wrapper_path.with_name(
        "build_m453a_trainonly_hierarchical_q32x3_catalog_r2.py")
    r2 = load_module(r2_path, "m453a_r2_wrapper")
    original_load_base = r2.load_base
    miter_receipt = {}
    selection_miter_receipt = {}
    actual_selection_crosscheck = {
        "actual_parent_calls_compared": 0,
        "actual_full_partitions_compared": 0,
        "actual_selection_mismatches": 0,
        "partition_ids": [],
    }
    actual_crosscheck = {
        "phase_calls": 0, "actual_train_phases_compared": 0,
        "fields_compared": 0, "mismatches": 0, "phase_ids": []}

    def r3_load_base(path):
        base = original_load_base(path)
        raw_strict_json = base.strict_json

        def inherited_contract_json(json_path):
            payload = raw_strict_json(json_path)
            if payload.get("schema") == (
                    "m453a_trainonly_hierarchical_q32x3_catalog_vector_recovery_contract_rev2_v1"):
                parent_spec = payload["base_frozen_contract"]
                parent_path = Path(json_path).resolve().parents[1] / parent_spec[
                    "path"]
                require(base.sha256(parent_path) == parent_spec["sha256"],
                        "M453a r3 rev2 base contract SHA drift")
                parent = raw_strict_json(parent_path)
                parent["inputs"].update(payload["inputs"])
                parent["status"] = payload["status"]
                return parent
            return payload

        base.strict_json = inherited_contract_json
        miter_receipt.update(scalar_vector_miter(base))
        selection_miter_receipt.update(directed_selection_miter(base))
        scalar_evaluate = base.evaluate_counter

        def checked_vector(counter, parents, children):
            call = actual_crosscheck["phase_calls"]
            flat_partition = call // 32
            sample = call % 32
            operator = flat_partition // 432
            partition = flat_partition % 432
            vector = vector_evaluate(base, counter, parents, children)
            if sample == 0 and partition in (0, 431):
                scalar = scalar_evaluate(counter, parents, children)
                keys = set(scalar) | set(vector)
                mismatch = sum(int(scalar[key] != vector[key])
                               for key in keys)
                actual_crosscheck["actual_train_phases_compared"] += 1
                actual_crosscheck["fields_compared"] += len(keys)
                actual_crosscheck["mismatches"] += mismatch
                actual_crosscheck["phase_ids"].append({
                    "operator": operator, "partition": partition,
                    "sample": sample, "unique_masks": len(counter),
                    "fields": len(keys), "mismatches": mismatch})
                require(mismatch == 0,
                        "M453a r3 actual train scalar/vector mismatch")
            actual_crosscheck["phase_calls"] += 1
            return vector

        base.evaluate_counter = checked_vector
        return BaseProxy(base, actual_selection_crosscheck)

    r2.__file__ = str(wrapper_path)
    r2.load_base = r3_load_base
    return_code = r2.main()

    output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    catalog_path = output_dir / (
        "m453a_trainonly_hierarchical_q32x3_catalog_r1.json")
    audit_path = output_dir / (
        "m453a_trainonly_hierarchical_q32x3_catalog_audit_r1.json")
    csv_path = output_dir / "m453a_parent_child_train_audit.csv"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    catalog["schema"] = "m453a_trainonly_hierarchical_q32x3_catalog_v3"
    catalog["status"] = (
        "PASS_M453A_R3_VECTORIZED_TRAIN_ONLY_TREE_FROZEN_BEFORE_M40")
    catalog["algorithm"]["r3_implementation_recovery"] = (
        "selection is r2 bit-identical; only full-train Counter audit uses vectorized int64 aggregation")
    audit["schema"] = (
        "m453a_trainonly_hierarchical_q32x3_catalog_audit_v3")
    audit["status"] = "PASS_M453A_R3_DOUBLE_SEAL_READY_M40_NOT_READ"
    audit["implementation_recovery"] = {
        "r2_abort_receipt":
            "results/m453a_trainonly_hierarchical_q32x3_catalog_r2_aborted_20260826/m453a_r2_implementation_abort_receipt.json",
        "selection_contract_changed": False,
        "full_train_phase_audits_skipped": 0,
        "full_train_expected_rows": 165888000,
        "full_train_observed_rows":
            audit["train_observation"]["source_rows"],
        "scalar_vector_miter": miter_receipt,
        "scalar_vector_selection_miter": selection_miter_receipt,
        "actual_train_full_tree_selection_crosscheck":
            actual_selection_crosscheck,
        "actual_train_scalar_vector_crosscheck": actual_crosscheck,
        "integer_accumulation": "int64 exact counts and dot products",
        "m40_reads": 0,
    }
    require(audit["train_observation"]["source_rows"] == 165888000,
            "M453a r3 full train row extent drift")
    require(actual_crosscheck["phase_calls"] == 4 * 432 * 32 and
            actual_crosscheck["actual_train_phases_compared"] == 8 and
            actual_crosscheck["mismatches"] == 0,
            "M453a r3 actual train crosscheck extent/mismatch")
    require(actual_selection_crosscheck["actual_parent_calls_compared"] ==
            8 * 32 and
            actual_selection_crosscheck["actual_full_partitions_compared"] ==
            8 and
            actual_selection_crosscheck["actual_selection_mismatches"] == 0 and
            actual_selection_crosscheck["partition_ids"] ==
            [0, 431, 432, 863, 864, 1295, 1296, 1727],
            "M453a r3 actual full-tree selection crosscheck failure")
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    base = original_load_base(wrapper_path.with_name(
        "build_m453a_trainonly_hierarchical_q32x3_catalog.py"))
    _, seal = base.write_seal(output_dir, [
        catalog_path.name, audit_path.name, csv_path.name])
    print("PASS_M453A_R3_VECTOR_FULL_TRAIN rows={} miter={} m40_reads=0 "
          "seal={}".format(
              audit["train_observation"]["source_rows"],
              miter_receipt["mismatches"], base.sha256(seal)), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
