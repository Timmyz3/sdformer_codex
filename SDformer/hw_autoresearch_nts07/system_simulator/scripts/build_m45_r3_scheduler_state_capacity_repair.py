#!/usr/bin/env python3
"""Build the ledger-only M45-r3 scheduler-state capacity repair.

M45-r3 does not change or rerun the M45-r2 transaction schedule.  It replaces
the clamped 16-entry metadata-capacity interpretation with a structurally
bounded 20-entry spatial frontier and a separate response-metadata FIFO.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m45_r3_scheduler_state_capacity_repair_contract_r1_20260823.json")
EXPECTED_CONTRACT_SHA256 = (
    "466ed800cf4bb5258573a30e1df4f3165a1215e5cc034f55c9f1ef7cd890961e")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_frontier_invariant(r2_analyzer_path):
    """Check the mandatory-up topology that bounds ready nodes by width.

    Optional left edges can only delay readiness.  Because every node below
    row zero has the node immediately above as a predecessor, two uncommitted
    ready nodes cannot coexist in one column.  The maximum ready frontier is
    therefore the 20 spatial columns.
    """
    r2 = load_module(r2_analyzer_path, "m45_r3_pinned_r2")
    r1 = r2.load_r1()
    width = int(r1.W)
    height = int(r1.H)
    rows = int(r1.ROWS_PER_T)
    require((width, height, rows) == (20, 15, 300),
            "M45-r3 spatial geometry drift")
    for parent_name in ("local_zero", "left", "up"):
        selected = [parent_name] * rows
        _indegree, children = r1.build_structural_dag(selected)
        for y in range(1, height):
            for x in range(width):
                parent = (y - 1) * width + x
                child = y * width + x
                require(child in children[parent],
                        "mandatory up edge missing for {},{}".format(y, x))
    return {
        "height": height,
        "width": width,
        "tasks_per_tile_timestep": rows,
        "mandatory_up_edge_checked_for_parent_policies": 3,
        "maximum_raw_ready_entries": width,
        "proof": ("at most one uncommitted ready node per column; optional "
                  "left dependencies cannot increase readiness")
    }


def build():
    require(sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M45-r3 contract identity drift")
    contract = read_json(CONTRACT)
    require(contract["schema"] ==
            "m45_r3_scheduler_state_capacity_repair_contract_v1",
            "M45-r3 contract schema drift")
    inputs = {}
    input_paths = {}
    for name, item in contract["inputs"].items():
        path = HW_ROOT / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "M45-r3 input identity drift: {}".format(name))
        if path.suffix == ".json":
            inputs[name] = read_json(path)
        input_paths[name] = path

    r2 = inputs["m45_r2_result"]
    review = inputs["m45_r2_independent_review"]
    replay = inputs["m45_r2_targeted_replay"]
    require(r2["status"] ==
            "PASS_M45_R2_TRANSACTION_GATES_RTL_AND_SYSTEM_UNADMITTED",
            "M45-r3 upstream transaction schedule is not GO")
    require(review["status"] ==
            "GO_ALL10_TRANSACTION_SCHEDULE_METADATA_CAPACITY_NO_GO_PENDING_R3",
            "M45-r3 independent repair request drift")
    require(review["review"] == {
        "decision": "GO_ALL10_TRANSACTION_SCHEDULE_METADATA_CAPACITY_NO_GO_PENDING_R3",
        "score_0_to_100": 86, "p0": 0, "p1": 1, "p2": 5},
        "M45-r3 review scorecard drift")
    observed = [row["raw_spatial_dag_ready_depth"]
                for row in replay["records"]]
    require(observed == [20, 20, 19, 20, 20, 20, 19, 20] and
            max(observed) == 20 and
            replay["status"] ==
            "PASS_EXACT_RECORD_REPLAY_K2_CTX8_SAMPLES_3_AND_7" and
            all(row["record_exact_match"] for row in replay["records"]),
            "M45-r3 targeted raw-ready evidence drift")
    proof = validate_frontier_invariant(input_paths["m45_r2_analyzer"])
    require(proof["maximum_raw_ready_entries"] ==
            contract["spatial_frontier_proof"]["maximum_raw_ready_entries"],
            "M45-r3 frontier proof/contract mismatch")

    repair = contract["capacity_repair"]
    require(repair["ready_descriptor_frontier_bytes"] == 20 * 64 and
            repair["response_metadata_fifo_bytes"] == 16 * 8 and
            repair["scheduler_and_fifo_bytes"] == 1280 + 128 + 4864 and
            repair["capacity_delta_bytes_vs_r2"] == 384 and
            repair["combined_local_capacity_bytes"] == 151040 and
            repair["local_capacity_headroom_bytes"] == 42688 and
            repair["combined_local_capacity_bytes"] +
            repair["local_capacity_headroom_bytes"] ==
            repair["frozen_local_residency_bytes"],
            "M45-r3 capacity arithmetic drift")
    require(r2["capacity"]["combined_local_capacity_bytes"] == 150656 and
            r2["capacity"]["fifo_storage_bytes"] == 5888,
            "M45-r3 superseded r2 capacity drift")

    by_name = dict((row["name"], row) for row in r2["configurations"])
    primary = by_name["K2_CTX8_PRIMARY"]
    inherited = {
        "population": r2["population"],
        "k1_aggregate_integrated_cycles":
            by_name["K1_CTX4_REPRODUCTION"]["aggregate_integrated_cycles"],
        "k2_ctx8_aggregate_source_only_cycles":
            primary["aggregate_source_only_cycles"],
        "k2_ctx8_aggregate_integrated_cycles":
            primary["aggregate_integrated_cycles"],
        "k2_ctx8_p95_integrated_cycles":
            primary["integrated_cycle_distribution"]["p95_nearest_rank"],
        "k2_ctx4_p95_integrated_cycles":
            by_name["K2_CTX4_CAPACITY_ABLATION"][
                "integrated_cycle_distribution"]["p95_nearest_rank"],
        "k4_ctx4_p95_integrated_cycles":
            by_name["K4_CTX4_KILLED_ABLATION"][
                "integrated_cycle_distribution"]["p95_nearest_rank"],
        "all_r2_transaction_kill_gates_pass":
            r2["kill_gates"]["all_kill_gates_pass"]
    }
    require(inherited == {
        "population": {"samples": 10, "operators": 4, "records": 40},
        "k1_aggregate_integrated_cycles": 122418024,
        "k2_ctx8_aggregate_source_only_cycles": 88269520,
        "k2_ctx8_aggregate_integrated_cycles": 95047672,
        "k2_ctx8_p95_integrated_cycles": 9681752,
        "k2_ctx4_p95_integrated_cycles": 10101880,
        "k4_ctx4_p95_integrated_cycles": 10535896,
        "all_r2_transaction_kill_gates_pass": True},
        "M45-r3 inherited transaction identity drift")

    return {
        "schema": "m45_r3_scheduler_state_capacity_repair_result_v1",
        "status": "PASS_M45_R3_LEDGER_ONLY_CAPACITY_REPAIR_RTL_SYSTEM_UNADMITTED",
        "identity": {
            "contract_sha256": sha256(CONTRACT),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "inputs_sha256": dict((name, item["sha256"])
                                   for name, item in contract["inputs"].items())
        },
        "repair_scope": contract["repair_scope"],
        "spatial_frontier_proof": proof,
        "targeted_raw_ready_entries": observed,
        "capacity": repair,
        "inherited_transaction_identity": inherited,
        "admission": {
            "transaction_cycles_unchanged": True,
            "scheduler_state_capacity_repaired": True,
            "nominal_local_headroom_admitted_at_transaction_level": True,
            "rtl_atomic_push2_or_response_fifo_occupancy_proved": False,
            "rtl_ppa_system_or_three_x_admitted": False
        },
        "claim_policy": contract["claim_policy"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite M45-r3 result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
