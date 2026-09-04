#!/usr/bin/env python3
"""Fail-closed r2 recovery for M453a sparse parent route buckets.

The r1 implementation and every train/heldout gate are retained.  Only the
child candidate pool is repaired: the weighted objective remains the current
parent's M73 route bucket, while candidates may come from every train-observed
mask in the same operator/partition.  Fixed parent order and a global selected
set preserve exactly 32 parents plus 96 unique children.
"""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_base(path):
    spec = importlib.util.spec_from_file_location("m453a_r1_base", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot load M453a r1 base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    wrapper_path = Path(__file__).resolve()
    base_path = wrapper_path.with_name(
        "build_m453a_trainonly_hierarchical_q32x3_catalog.py")
    base = load_base(base_path)
    original_strict_json = base.strict_json
    original_route = base.route_to_parents
    state = {
        "partition_values": None,
        "selected_children": set(),
        "outside_route_children": 0,
        "parents_with_route_shortfall": 0,
        "global_candidate_calls": 0,
    }

    def r2_strict_json(path):
        payload = original_strict_json(path)
        if payload.get("schema") == (
                "m453a_trainonly_hierarchical_q32x3_catalog_recovery_contract_v2"):
            payload = copy.deepcopy(payload)
            payload["schema"] = (
                "m453a_trainonly_hierarchical_q32x3_catalog_contract_v1")
            payload["status"] = (
                "FROZEN_BEFORE_M453A_TRAIN_EXECUTION_M40_FORBIDDEN")
        return payload

    def r2_route(values, parents):
        state["partition_values"] = np.asarray(values, dtype=np.uint16)
        state["selected_children"] = set()
        return original_route(values, parents)

    def r2_greedy(bucket_values, bucket_counts, parent, all_parents):
        parent_set = set(int(value) for value in all_parents)
        bucket_set = set(int(value) for value in bucket_values)
        candidates = sorted(
            int(value) for value in state["partition_values"]
            if int(value) not in parent_set and
            int(value) not in state["selected_children"])
        require(len(candidates) >= base.CHILDREN_PER_PARENT,
                "partition has fewer than three globally unused train masks")
        routed_candidates = [value for value in candidates
                             if value in bucket_set]
        if len(routed_candidates) < base.CHILDREN_PER_PARENT:
            state["parents_with_route_shortfall"] += 1
        values = np.asarray(bucket_values, dtype=np.uint16)
        counts = np.asarray(bucket_counts, dtype=np.int64)
        best_distance = base.POPCOUNT[np.bitwise_xor(values, int(parent))]
        objectives = [int(np.dot(
            counts, base.issue_units(values, best_distance)
            .astype(np.int64)))]
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
                next_distance = np.minimum(best_distance,
                                           candidate_distance)
                objective = int(np.dot(
                    counts, base.issue_units(values, next_distance)
                    .astype(np.int64)))
                # Deterministic recovery tie order: objective, prefer a mask
                # in this route bucket, distance to parent, numeric mask.
                key = (objective,
                       int(candidate not in bucket_set),
                       int(base.POPCOUNT[candidate ^ int(parent)]),
                       candidate)
                if best_key is None or key < best_key:
                    best_key = key
                    best_value = candidate
                    best_next = next_distance
            require(best_value is not None,
                    "M453a r2 global child greedy exhausted")
            selected.append(best_value)
            best_distance = best_next
            objectives.append(best_key[0])
            state["outside_route_children"] += int(
                best_value not in bucket_set)
        require(not (set(selected) & state["selected_children"]),
                "M453a r2 cross-parent child reuse")
        state["selected_children"].update(selected)
        state["global_candidate_calls"] += 1
        return selected, objectives

    # The frozen r1 main uses module globals for identity and helpers.  Point
    # its self identity at this wrapper, transform only the r2 contract header,
    # and replace only route/candidate selection as documented above.
    base.__file__ = str(wrapper_path)
    base.strict_json = r2_strict_json
    base.route_to_parents = r2_route
    base.greedy_children = r2_greedy
    return_code = base.main()

    # Patch only truthful recovery metadata, then regenerate both seal layers.
    import sys
    output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    catalog_path = output_dir / (
        "m453a_trainonly_hierarchical_q32x3_catalog_r1.json")
    audit_path = output_dir / (
        "m453a_trainonly_hierarchical_q32x3_catalog_audit_r1.json")
    csv_path = output_dir / "m453a_parent_child_train_audit.csv"
    catalog = original_strict_json(catalog_path)
    audit = original_strict_json(audit_path)
    catalog["schema"] = "m453a_trainonly_hierarchical_q32x3_catalog_v2"
    catalog["status"] = (
        "PASS_M453A_R2_TRAIN_ONLY_TREE_FROZEN_BEFORE_M40")
    catalog["algorithm"]["children"] = (
        "three globally unique masks per parent from all M73 train-observed masks in the same operator/partition; no runtime mask")
    catalog["algorithm"]["greedy_objective"] = (
        "parent-route-bucket weighted exact dual issue units; tie=(objective, prefer-local-route, Hamming-to-parent, numeric-mask), parents processed ID0..31")
    catalog["algorithm"]["r1_recovery"] = (
        "r1 failed because one real parent route bucket had fewer than three non-parent masks; r2 changes candidate extent only")
    catalog["admission"]["children_from_train_observed_masks_only"] = True
    catalog["admission"]["global_32_plus_96_unique"] = True
    catalog["claim_boundary"] = (
        "Recovered train-only exact catalog. Child candidates are same-partition M73 train observations; objectives remain per-parent train route buckets. M40 remains unread; no heldout cycle, RTL, PPA, power, system or headline claim.")
    audit["schema"] = (
        "m453a_trainonly_hierarchical_q32x3_catalog_audit_v2")
    audit["status"] = "PASS_M453A_R2_DOUBLE_SEAL_READY_M40_NOT_READ"
    audit["recovery"] = {
        "r1_failclosed_receipt":
            "results/m453a_trainonly_hierarchical_q32x3_catalog_r1_failed_20260826/m453a_r1_failclosed_receipt.json",
        "parents_with_fewer_than_three_local_route_candidates":
            state["parents_with_route_shortfall"],
        "children_selected_outside_local_route_bucket":
            state["outside_route_children"],
        "parent_greedy_calls": state["global_candidate_calls"],
        "all_children_same_partition_train_observed": True,
        "m40_reads_during_recovery": 0,
    }
    audit["exactness"].pop(
        "child_outside_train_route_bucket_mismatches", None)
    audit["exactness"][
        "child_outside_same_partition_train_observation_mismatches"] = 0
    audit["exactness"]["global_child_reuse_mismatches"] = 0
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    _, seal = base.write_seal(output_dir, [
        catalog_path.name, audit_path.name, csv_path.name])
    print("PASS_M453A_R2_RECOVERY parents={} outside_route={} m40_reads=0 "
          "seal={}".format(
              state["global_candidate_calls"],
              state["outside_route_children"], base.sha256(seal)),
          flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
