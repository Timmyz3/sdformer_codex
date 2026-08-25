#!/usr/bin/env python3
"""Independent hammer for the ledger-only M45-r3 capacity repair."""

from __future__ import print_function

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m45_r3_scheduler_state_capacity_repair_contract_r1_20260823.json")
BUILDER = HW_ROOT / (
    "system_simulator/scripts/build_m45_r3_scheduler_state_capacity_repair.py")
PRODUCER_VALIDATOR = HW_ROOT / (
    "system_simulator/scripts/validate_m45_r3_scheduler_state_capacity_repair.py")
RESULT = HW_ROOT / (
    "results/m45_scheduler_state_capacity_repair_r3_20260823/"
    "m45_r3_scheduler_state_capacity_repair.json")
R1_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py")

EXPECTED = {
    "contract": "466ed800cf4bb5258573a30e1df4f3165a1215e5cc034f55c9f1ef7cd890961e",
    "builder": "91a38fb867d74b52e288399e410b6ba22ca522ff6b54000a239670b8c78220b5",
    "producer_validator": "6596c84ded79db0f657eab685c3f9e6e6d6948ff66bd919ee248864e5d14cc3e",
    "result": "4e3764f58b5c8b893e9d5b71b6a27adca582aaac43378cfc610e6e1010a0ce72",
    "r1_analyzer": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
}


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
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def build_graph(height, width, left_bits, remove_up=False):
    predecessors = [set() for _ in range(height * width)]
    bit = 0
    for y in range(height):
        for x in range(width):
            node = y * width + x
            if y > 0 and not remove_up:
                predecessors[node].add((y - 1) * width + x)
            if x > 0:
                if (left_bits >> bit) & 1:
                    predecessors[node].add(node - 1)
                bit += 1
    return predecessors


def ready_nodes(predecessors, committed):
    return set(node for node, preds in enumerate(predecessors)
               if node not in committed and preds.issubset(committed))


def exhaustive_small_frontier_proof():
    height, width = 3, 4
    optional_edges = height * (width - 1)
    checked_states = 0
    maximum = 0
    for left_bits in range(1 << optional_edges):
        graph = build_graph(height, width, left_bits)
        for mask in range(1 << (height * width)):
            committed = set(index for index in range(height * width)
                            if (mask >> index) & 1)
            downward_closed = all(
                graph[node].issubset(committed)
                for node in committed)
            if not downward_closed:
                continue
            ready = ready_nodes(graph, committed)
            checked_states += 1
            maximum = max(maximum, len(ready))
            columns = [node % width for node in ready]
            require(len(columns) == len(set(columns)),
                    "more than one ready node in a column")
            require(len(ready) <= width, "frontier exceeds width")
    mutant = build_graph(height, width, 0, remove_up=True)
    mutant_initial = len(ready_nodes(mutant, set()))
    require(maximum == width and mutant_initial == height * width > width,
            "frontier proof sensitivity mutation failed")
    return {
        "height": height,
        "width": width,
        "optional_left_patterns": 1 << optional_edges,
        "reachable_downward_closed_states_checked": checked_states,
        "maximum_ready_entries": maximum,
        "missing_mandatory_up_mutant_initial_ready_entries": mutant_initial,
    }


def validate_payload(result, contract):
    require(result["schema"] ==
            "m45_r3_scheduler_state_capacity_repair_result_v1",
            "schema drift")
    require(result["status"] ==
            "PASS_M45_R3_LEDGER_ONLY_CAPACITY_REPAIR_RTL_SYSTEM_UNADMITTED",
            "status drift")
    require(result["identity"]["contract_sha256"] == EXPECTED["contract"] and
            result["identity"]["builder_sha256"] == EXPECTED["builder"],
            "embedded identity drift")
    require(result["identity"]["inputs_sha256"] ==
            dict((name, item["sha256"])
                 for name, item in contract["inputs"].items()),
            "embedded input identities drift")
    scope = result["repair_scope"]
    require(scope == contract["repair_scope"], "repair scope drift")
    require(all(scope[name] is False for name in (
        "transaction_schedule_changed", "ready_selection_changed",
        "command_timing_changed", "descriptor_backpressure_added",
        "all10_schedule_rerun_required")),
        "ledger-only conditions not satisfied")
    frontier = result["spatial_frontier_proof"]
    require(frontier["height"] == 15 and frontier["width"] == 20 and
            frontier["tasks_per_tile_timestep"] == 300 and
            frontier["maximum_raw_ready_entries"] == 20 and
            frontier["mandatory_up_edge_checked_for_parent_policies"] == 3,
            "frontier result drift")
    require(result["targeted_raw_ready_entries"] ==
            [20, 20, 19, 20, 20, 20, 19, 20],
            "targeted evidence drift")

    cap = result["capacity"]
    expected_fields = [
        "context0[2:0]", "context1_valid", "context1[2:0]",
        "bank_valid[7:0]", "destination0_valid[7:0]",
        "destination1_valid[7:0]", "destination0_subtract[7:0]",
        "destination1_subtract[7:0]", "last"]
    require(cap["response_metadata_fields"] == expected_fields,
            "response metadata field set/order drift")
    response_bits = 3 + 1 + 3 + 8 + 8 + 8 + 8 + 8 + 1
    require(response_bits == cap["response_metadata_payload_bits"] == 48,
            "response metadata bit sum drift")
    require(cap["response_metadata_aligned_bytes_per_entry"] ==
            (response_bits + 7) // 8 + 2 == 8,
            "response metadata aligned byte contract drift")
    require(cap["ready_descriptor_frontier_entries"] == 20 and
            cap["ready_descriptor_bytes_per_entry"] == 64 and
            cap["ready_descriptor_frontier_bytes"] == 20 * 64 == 1280,
            "descriptor frontier capacity drift")
    require(cap["response_metadata_fifo_entries"] == 16 and
            cap["response_metadata_fifo_bytes"] == 16 * 8 == 128,
            "response FIFO capacity drift")
    require(cap["complete_fifo_bytes_unchanged"] == 16 * (288 + 16) == 4864,
            "complete FIFO capacity drift")
    require(cap["scheduler_and_fifo_bytes"] == 1280 + 128 + 4864 == 6272,
            "scheduler/FIFO sum drift")
    require(cap["m45_r2_scheduler_and_fifo_bytes_superseded"] == 5888 and
            cap["capacity_delta_bytes_vs_r2"] == 6272 - 5888 == 384,
            "r2/r3 delta drift")
    require(cap["base_local_bytes_before_scheduler_state"] == 144768 and
            cap["combined_local_capacity_bytes"] == 144768 + 6272 == 151040,
            "combined capacity drift")
    require(cap["frozen_local_residency_bytes"] == 193728 and
            cap["local_capacity_headroom_bytes"] == 193728 - 151040 == 42688,
            "headroom drift")
    inherited = result["inherited_transaction_identity"]
    require(inherited == {
        "population": {"samples": 10, "operators": 4, "records": 40},
        "k1_aggregate_integrated_cycles": 122418024,
        "k2_ctx8_aggregate_source_only_cycles": 88269520,
        "k2_ctx8_aggregate_integrated_cycles": 95047672,
        "k2_ctx8_p95_integrated_cycles": 9681752,
        "k2_ctx4_p95_integrated_cycles": 10101880,
        "k4_ctx4_p95_integrated_cycles": 10535896,
        "all_r2_transaction_kill_gates_pass": True},
        "inherited transaction identity drift")
    admission = result["admission"]
    require(admission == {
        "transaction_cycles_unchanged": True,
        "scheduler_state_capacity_repaired": True,
        "nominal_local_headroom_admitted_at_transaction_level": True,
        "rtl_atomic_push2_or_response_fifo_occupancy_proved": False,
        "rtl_ppa_system_or_three_x_admitted": False},
        "admission boundary drift")
    forbidden = " ".join(result["claim_policy"]["forbidden"])
    require(all(token in forbidden for token in
                ("RTL", "response metadata", "power", "system", "3x")),
            "forbidden claim boundary weakened")


def mutation_matrix(canonical, contract):
    rejected = []

    def run(name, mutate):
        item = copy.deepcopy(canonical)
        mutate(item)
        try:
            validate_payload(item, contract)
        except (ValueError, KeyError, TypeError):
            rejected.append(name)
            return
        raise ValueError("independent validator accepted attack: {}".format(name))

    run("frontier_19_entries", lambda d: d["capacity"].__setitem__(
        "ready_descriptor_frontier_entries", 19))
    run("descriptor_63_bytes", lambda d: d["capacity"].__setitem__(
        "ready_descriptor_bytes_per_entry", 63))
    run("response_payload_47_bits", lambda d: d["capacity"].__setitem__(
        "response_metadata_payload_bits", 47))
    run("response_context1_id_removed", lambda d: d["capacity"][
        "response_metadata_fields"].remove("context1[2:0]"))
    run("scheduler_sum_minus_one", lambda d: d["capacity"].__setitem__(
        "scheduler_and_fifo_bytes", 6271))
    run("combined_capacity_minus_one", lambda d: d["capacity"].__setitem__(
        "combined_local_capacity_bytes", 151039))
    run("headroom_plus_one", lambda d: d["capacity"].__setitem__(
        "local_capacity_headroom_bytes", 42689))
    run("ready_selection_changed_but_no_rerun", lambda d: d[
        "repair_scope"].__setitem__("ready_selection_changed", True))
    run("inherited_p95_corruption", lambda d: d[
        "inherited_transaction_identity"].__setitem__(
            "k2_ctx8_p95_integrated_cycles", 9681751))
    run("atomic_push2_improperly_proved", lambda d: d["admission"].__setitem__(
        "rtl_atomic_push2_or_response_fifo_occupancy_proved", True))
    run("three_x_improperly_admitted", lambda d: d["admission"].__setitem__(
        "rtl_ppa_system_or_three_x_admitted", True))
    run("forbidden_claims_erased", lambda d: d["claim_policy"].__setitem__(
        "forbidden", []))
    with tempfile.TemporaryDirectory(prefix="m45_r3_json_attack_") as tempdir:
        for name, raw in (("duplicate_json_key", '{"x":1,"x":2}\n'),
                          ("nan_json_constant", '{"x":NaN}\n')):
            path = Path(tempdir) / (name + ".json")
            path.write_text(raw, encoding="utf-8")
            try:
                read_json(path)
            except ValueError:
                rejected.append(name)
    require(len(rejected) == 14, "mutation rejection count drift")
    return rejected


def producer_validator_gaps(canonical):
    spec = importlib.util.spec_from_file_location(
        "m45_r3_producer_validator_gap", str(PRODUCER_VALIDATOR))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mutations = []
    item = copy.deepcopy(canonical)
    item["capacity"]["response_metadata_payload_bits"] = 47
    mutations.append(("response_payload_47_bits", item))
    item = copy.deepcopy(canonical)
    item["claim_policy"]["forbidden"] = []
    mutations.append(("forbidden_claims_erased", item))
    accepted = []
    with tempfile.TemporaryDirectory(prefix="m45_r3_producer_gap_") as tempdir:
        for name, payload in mutations:
            path = Path(tempdir) / (name + ".json")
            path.write_text(json.dumps(payload, sort_keys=True) + "\n",
                            encoding="utf-8")
            try:
                module.validate(path, require_canonical_sha=False)
            except ValueError:
                continue
            accepted.append(name)
    return accepted


def build():
    for name, path in (("contract", CONTRACT), ("builder", BUILDER),
                       ("producer_validator", PRODUCER_VALIDATOR),
                       ("result", RESULT), ("r1_analyzer", R1_ANALYZER)):
        require(sha256(path) == EXPECTED[name], "anchor drift: {}".format(name))
    contract = read_json(CONTRACT)
    result = read_json(RESULT)
    for name, item in contract["inputs"].items():
        path = HW_ROOT / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "upstream identity drift: {}".format(name))
    validate_payload(result, contract)
    source = R1_ANALYZER.read_text(encoding="utf-8")
    require("if y > 0:" in source and
            'if x > 0 and selected_parent[spatial] == "left":' in source,
            "frozen DAG topology source evidence drift")
    exhaustive = exhaustive_small_frontier_proof()
    attacks = mutation_matrix(result, contract)
    producer_gaps = producer_validator_gaps(result)
    require(set(producer_gaps) ==
            set(("response_payload_47_bits", "forbidden_claims_erased")),
            "producer validator gap characterization drift")
    return {
        "schema": "m45_r3_scheduler_state_capacity_independent_hammer_review_v1",
        "date": "2026-08-23",
        "status": "GO_LEDGER_ONLY_SCHEDULER_STATE_CAPACITY_REPAIR",
        "review": {
            "decision": "GO_LEDGER_ONLY_SCHEDULER_STATE_CAPACITY_REPAIR",
            "score_0_to_100": 94,
            "p0": 0,
            "p1": 0,
            "p2": 4,
        },
        "anchors": {
            "contract": sha256(CONTRACT),
            "builder": sha256(BUILDER),
            "producer_validator": sha256(PRODUCER_VALIDATOR),
            "canonical_result": sha256(RESULT),
            "r1_analyzer": sha256(R1_ANALYZER),
            "independent_reviewer": sha256(Path(__file__).resolve()),
        },
        "candidate_modified_by_reviewer": False,
        "frontier_proof": {
            "actual_height": 15,
            "actual_width": 20,
            "mandatory_up_implies_at_most_one_ready_uncommitted_node_per_column": True,
            "optional_left_only_adds_a_predecessor": True,
            "universal_actual_geometry_bound_entries": 20,
            "targeted_observed_depths": [20, 20, 19, 20, 20, 20, 19, 20],
            "independent_exhaustive_small_geometry": exhaustive,
        },
        "capacity_reconstruction": {
            "ready_frontier": {"entries": 20, "bytes_per_entry": 64,
                               "bytes": 1280},
            "response_metadata": {"payload_bits": 48,
                                  "aligned_bytes_per_entry": 8,
                                  "entries": 16, "bytes": 128},
            "complete_vector_fifo_bytes": 4864,
            "scheduler_and_fifo_bytes": 6272,
            "base_local_bytes": 144768,
            "combined_local_capacity_bytes": 151040,
            "frozen_local_residency_bytes": 193728,
            "local_capacity_headroom_bytes": 42688,
            "all_byte_arithmetic_exact": True,
        },
        "ledger_only_legality": {
            "upstream_r2_exact_sha_pinned": True,
            "transaction_schedule_changed": False,
            "ready_selection_changed": False,
            "command_timing_changed": False,
            "descriptor_backpressure_added": False,
            "identical_lowest_16_window_preserved": True,
            "all10_rerun_required_for_this_capacity_only_repair": False,
            "future_rerun_trigger": "any implementation that cannot insert newly-ready descriptors without schedule-visible backpressure or changes selection/command timing",
        },
        "adversarial_matrix": {
            "tested": len(attacks),
            "rejected_by_independent_validator": len(attacks),
            "rejected_attacks": attacks,
            "producer_validator_known_accepted_mutations": producer_gaps,
        },
        "findings": {
            "p0": [],
            "p1": [],
            "p2": [
                {
                    "id": "P2_FRONTIER_INSERT_BANDWIDTH_NOT_PROVED",
                    "detail": "The 20-entry capacity bound is structural, but K2 may make two successors ready together. Multi-insert descriptor generation/write bandwidth is not scheduled or implemented; a single-write implementation would trigger an all10 rerun.",
                },
                {
                    "id": "P2_RESPONSE_METADATA_OCCUPANCY_NOT_EVENT_LEDGERED",
                    "detail": "The 48-bit payload and 16-entry byte capacity are conservative and separate, but response metadata enqueue/dequeue occupancy and atomic push2 remain explicitly unproved.",
                },
                {
                    "id": "P2_DESCRIPTOR_AND_RESPONSE_STORAGE_NOT_RTL_MAPPED",
                    "detail": "The 64-byte descriptor and 8-byte aligned response entry are byte contracts, not a synthesized register/SRAM field map, port configuration, timing, area, or energy result.",
                },
                {
                    "id": "P2_PRODUCER_VALIDATOR_CLAIM_AND_FIELD_GAPS",
                    "detail": "The producer validator accepts a corrupted response payload-bit count and erased forbidden-claim list when canonical SHA checking is disabled. The independent validator rejects both; canonical data itself is unchanged.",
                },
            ],
        },
        "repair_disposition": {
            "r2_metadata_capacity_p1_closed": True,
            "reason": "the clamped ready metric is no longer used as FIFO occupancy; the universal 20-entry frontier is separately provisioned and response metadata has its own conservative storage",
            "next_gate": "RTL/VCS must prove multi-insert frontier behavior, true response-metadata occupancy, atomic push2, and same-cycle pop/push before any RTL-cycle or physical claim",
        },
        "claim_boundary": "GO covers only the exact-SHA ledger-only scheduler-state capacity repair: a 20-entry frontier, separately accounted response metadata and complete-vector FIFOs, 151040 total transaction-level local bytes, and 42688 nominal headroom. RTL implementation/cycles, response occupancy proof, atomic push2 proof, SRAM timing, PPA, power, energy, system speedup, 3x, external comparisons, DATE headline, and best-paper claims remain forbidden.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite review")
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
