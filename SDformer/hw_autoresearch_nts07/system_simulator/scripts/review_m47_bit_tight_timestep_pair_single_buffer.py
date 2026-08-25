#!/usr/bin/env python3
"""Independent hammer for the M47-r1 conservative timestep-pair ledger."""

from __future__ import print_function

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m47_bit_tight_timestep_pair_single_buffer_contract_r1_20260823.json")
BUILDER = HW_ROOT / (
    "system_simulator/scripts/build_m47_bit_tight_timestep_pair_single_buffer.py")
PRODUCER_VALIDATOR = HW_ROOT / (
    "system_simulator/scripts/validate_m47_bit_tight_timestep_pair_single_buffer.py")
RESULT = HW_ROOT / (
    "results/m47_bit_tight_timestep_pair_single_buffer_r1_20260823/"
    "m47_bit_tight_timestep_pair_single_buffer.json")
M45_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py")

EXPECTED = {
    "contract": "64319c18c4ac8d2bfe925c39e6d867638c803cbe2546f9781187881ec97c234d",
    "builder": "469fec2d14619d6d71c7103a48fe742233fbb54258813c661ba1f0fdb4110d7e",
    "producer_validator": "57060840ecd0b2076ec2ca05fe096a7e9d0b0f73757ab2a8adb192476b199612",
    "result": "dc42df25567ad49be863586a3e287c9137a9f41470e83a5ef95bf125aa1734ed",
    "m45_analyzer": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
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


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return {"numerator": numerator, "denominator": denominator}


def distribution(values):
    ordered = sorted(values)
    require(ordered, "empty distribution")

    def nr(percent):
        return ordered[(percent * len(ordered) + 99) // 100 - 1]
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean_exact": fraction(sum(ordered), len(ordered)),
        "p50_nearest_rank": nr(50),
        "p95_nearest_rank": nr(95),
        "p99_nearest_rank": nr(99),
    }


def find_config(result, name):
    rows = [row for row in result["configurations"] if row["name"] == name]
    require(len(rows) == 1, "configuration identity drift")
    return rows[0]


def validate_payload(result, contract, inputs):
    require(result["schema"] ==
            "m47_bit_tight_timestep_pair_single_buffer_result_v1",
            "schema drift")
    require(result["status"] ==
            "PASS_M47_CONSERVATIVE_CAPACITY_TRAFFIC_AND_CONDITIONAL_3X_GATE_RTL_SYSTEM_UNADMITTED",
            "status drift")
    require(result["identity"]["contract_sha256"] == EXPECTED["contract"] and
            result["identity"]["builder_sha256"] == EXPECTED["builder"],
            "embedded identity drift")
    require(result["identity"]["inputs_sha256"] ==
            dict((name, item["sha256"])
                 for name, item in contract["inputs"].items()),
            "embedded input SHA drift")
    geometry = contract["frozen_geometry"]
    policy = contract["storage_policy"]
    require(result["architecture"]["loop_order"] == geometry["loop_order"] and
            result["architecture"]["resident_frames"] == 2 and
            result["architecture"]["weight_buffers"] == 1 and
            result["architecture"]["no_free_prefetch"] is True,
            "architecture policy drift")

    vector = (96 * 19 + 7) // 8
    frame = (300 * 96 * 19 + 7) // 8
    parent = (20 * 96 * 19 + 7) // 8
    require((vector, frame, parent) == (228, 68400, 4560),
            "signed19 packing arithmetic drift")
    components = {
        "single_weight_tile_buffer_bytes": 24576,
        "bit_tight_parent_line_bytes": parent,
        "support_line_bytes": 20 * 256 // 8,
        "eight_contexts_bytes": 8 * (vector + 64),
        "two_bit_tight_frames_bytes": 2 * frame,
        "ready_frontier_bytes": 20 * 64,
        "response_metadata_fifo_bytes": 16 * 8,
        "complete_fifo_bytes": 16 * (vector + 16),
    }
    capacity = result["capacity"]
    require(capacity["bit_tight_vector_bytes"] == vector and
            capacity["bit_tight_frame_bytes"] == frame and
            capacity["components"] == components,
            "capacity component drift")
    combined = sum(components.values())
    require(combined == 174224 and
            capacity["combined_local_capacity_bytes"] == combined and
            capacity["frozen_local_residency_bytes"] == 193728 and
            capacity["local_capacity_headroom_bytes"] == 193728 - combined == 19504 and
            capacity["minimum_required_headroom_bytes"] == 16384 and
            capacity["double_weight_buffer_permitted"] is False and
            capacity["external_accumulator_spill_permitted"] is False,
            "combined capacity/headroom drift")
    require(capacity["local_capacity_headroom_bytes"] -
            capacity["minimum_required_headroom_bytes"] == 3120,
            "capacity-gate margin drift")

    k2 = find_config(inputs["m45_r2_result"], "K2_CTX8_PRIMARY")
    loads = 4 * 8 * 5 * 27
    weight_bytes = loads * 24576
    load_cycles = weight_bytes // 64
    require((loads, weight_bytes, load_cycles) ==
            (4320, 106168320, 1658880),
            "weight load arithmetic drift")
    traffic = result["weight_traffic"]
    require(traffic == {
        "weight_tile_loads_per_sample": loads,
        "weight_tile_bytes": 24576,
        "candidate_bytes_per_sample": weight_bytes,
        "m45_r2_bytes_per_sample": 212336640,
        "reduction_fraction": fraction(weight_bytes, 212336640),
        "exact_two_x_reduction": True,
        "serialized_load_cycles_per_sample": load_cycles,
        "weight_dma_bytes_per_cycle": 64},
        "weight traffic ledger drift")
    require(k2["traffic_bytes_per_sample"]["weight_dma"] == 2 * weight_bytes,
            "M45/M47 traffic bridge drift")

    inherited = dict((row["sample_id"], row) for row in k2["per_sample"])
    upper = result["conservative_cycle_upper_bound"]
    require([row["sample_id"] for row in upper["per_sample"]] == list(range(10)),
            "cycle sample identity/order drift")
    rebuilt = []
    inherited_wait = []
    deduplicated = []
    for row in upper["per_sample"]:
        source = inherited[row["sample_id"]]
        require(row["inherited_m45_k2_ctx8_integrated_cycles"] ==
                source["integrated_cycles"], "inherited cycles drift")
        require(row["inherited_m45_weight_dma_wait_cycles_not_subtracted"] ==
                source["weight_dma_wait_cycles"] == 12288,
                "inherited weight-wait disclosure drift")
        require(row["serialized_single_buffer_weight_load_cycles_added"] ==
                load_cycles, "serialized load addition drift")
        value = source["integrated_cycles"] + load_cycles
        require(row["conservative_integrated_cycle_upper_bound"] == value,
                "per-sample upper bound drift")
        rebuilt.append(value)
        inherited_wait.append(source["weight_dma_wait_cycles"])
        deduplicated.append(source["integrated_cycles"] -
                            source["weight_dma_wait_cycles"] + load_cycles)
    require(upper["distribution"] == distribution(rebuilt) and
            upper["aggregate_cycles"] == sum(rebuilt) == 111636472,
            "upper-bound aggregate/distribution drift")
    require(upper["distribution"]["p95_nearest_rank"] == 11340632 and
            distribution(deduplicated)["p95_nearest_rank"] == 11328344,
            "upper-bound p95/slack drift")
    require(all(value == 12288 for value in inherited_wait) and
            all(rebuilt[index] - deduplicated[index] == 12288
                for index in range(10)),
            "conservative double-count slack drift")
    require(upper["m45_r2_inherited_distribution"] ==
            k2["integrated_cycle_distribution"],
            "inherited distribution drift")
    require(upper["maximum_p95_gate_cycles"] == 15495075 and
            upper["p95_margin_to_gate_cycles"] == 4154443 and
            upper["new_address_timed_pair_schedule_replayed"] is False,
            "p95 gate/qualification drift")

    m42 = inputs["m42_result"]
    frozen = m42["frozen_resource_model"]
    fixed = frozen["fixed_compute_reference_cycles"]
    outside = frozen["outside_four_bottleneck_model_cycles"]
    late = frozen["fixed_late_scale_plus_frontend_cycles"]
    p95 = distribution(rebuilt)["p95_nearest_rank"]
    denominator = outside + late + p95
    model = result["conditional_frozen_compute_model"]
    require((fixed, outside, late, denominator) ==
            (620868243, 188824491, 2636515, 202801638),
            "M42 denominator bridge drift")
    require(model["conditional_compute_speedup"] == fraction(fixed, denominator) and
            model["conditional_total_cycles"] == denominator and
            model["candidate_p95_cycle_upper_bound"] == p95 and
            fixed >= 3 * denominator and
            fixed - 3 * denominator == 12463329 and
            model["conditional_compute_speedup_decimal"] ==
            float(fixed) / denominator and
            model["three_x_crossing_in_conditional_model"] is True and
            model["three_x_product_cycle_ceiling"] == 15495075 and
            model["product_cycle_headroom_to_three_x"] == 4154443 and
            model["system_or_end_to_end_speedup_admitted"] is False,
            "conditional M42 model drift")
    require(result["admission"] == {
        "bit_exact_capacity_admitted": True,
        "weight_traffic_reduction_admitted": True,
        "conservative_cycle_upper_bound_admitted": True,
        "conditional_frozen_compute_three_x_admitted": True,
        "new_address_timed_pair_schedule_admitted": False,
        "rtl_vcs_synopsys_admitted": False,
        "full_network_or_system_speedup_admitted": False,
        "ppa_power_energy_admitted": False,
        "date_headline_or_best_paper_admitted": False},
        "admission boundary drift")
    require(result["claim_policy"] == contract["claim_policy"],
            "claim policy drift")
    return {
        "components": components,
        "load_count": loads,
        "weight_bytes": weight_bytes,
        "load_cycles": load_cycles,
        "upper_values": rebuilt,
        "deduplicated_values": deduplicated,
        "conditional_denominator": denominator,
    }


def mutation_matrix(canonical, contract, inputs):
    rejected = []

    def run(name, mutate):
        item = copy.deepcopy(canonical)
        mutate(item)
        try:
            validate_payload(item, contract, inputs)
        except (ValueError, KeyError, TypeError):
            rejected.append(name)
            return
        raise ValueError("independent validator accepted attack: {}".format(name))

    run("vector_bytes_229", lambda d: d["capacity"].__setitem__(
        "bit_tight_vector_bytes", 229))
    run("frame_bytes_minus_one", lambda d: d["capacity"].__setitem__(
        "bit_tight_frame_bytes", 68399))
    run("second_frame_removed", lambda d: d["capacity"]["components"].__setitem__(
        "two_bit_tight_frames_bytes", 68400))
    run("second_weight_buffer_added", lambda d: d["capacity"].__setitem__(
        "double_weight_buffer_permitted", True))
    run("frontier_shrunk_to_16", lambda d: d["capacity"]["components"].__setitem__(
        "ready_frontier_bytes", 1024))
    run("complete_fifo_padded_vector", lambda d: d["capacity"]["components"].__setitem__(
        "complete_fifo_bytes", 4864))
    run("load_count_2160", lambda d: d["weight_traffic"].__setitem__(
        "weight_tile_loads_per_sample", 2160))
    run("weight_bytes_minus_tile", lambda d: d["weight_traffic"].__setitem__(
        "candidate_bytes_per_sample", 106168320 - 24576))
    run("serialized_cycles_minus_one", lambda d: d["weight_traffic"].__setitem__(
        "serialized_load_cycles_per_sample", 1658879))
    run("subtract_inherited_wait", lambda d: d[
        "conservative_cycle_upper_bound"]["per_sample"][0].__setitem__(
            "conservative_integrated_cycle_upper_bound", 11283216 - 12288))
    run("p95_minus_one", lambda d: d["conservative_cycle_upper_bound"][
        "distribution"].__setitem__("p95_nearest_rank", 11340631))
    run("conditional_denominator_minus_one", lambda d: d[
        "conditional_frozen_compute_model"].__setitem__(
            "conditional_total_cycles", 202801637))
    run("address_timed_replay_promoted", lambda d: d[
        "conservative_cycle_upper_bound"].__setitem__(
            "new_address_timed_pair_schedule_replayed", True))
    run("system_speedup_promoted", lambda d: d["conditional_frozen_compute_model"].__setitem__(
        "system_or_end_to_end_speedup_admitted", True))
    run("best_paper_promoted", lambda d: d["admission"].__setitem__(
        "date_headline_or_best_paper_admitted", True))
    run("forbidden_claims_erased", lambda d: d["claim_policy"].__setitem__(
        "forbidden", []))
    with tempfile.TemporaryDirectory(prefix="m47_json_attack_") as tempdir:
        for name, raw in (("duplicate_json_key", '{"x":1,"x":2}\n'),
                          ("nan_json_constant", '{"x":NaN}\n')):
            path = Path(tempdir) / (name + ".json")
            path.write_text(raw, encoding="utf-8")
            try:
                read_json(path)
            except ValueError:
                rejected.append(name)
    require(len(rejected) == 18, "attack rejection count drift")
    return rejected


def deterministic_builder_check(canonical):
    spec = importlib.util.spec_from_file_location(
        "m47_independent_builder_check", str(BUILDER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_result() == canonical


def build():
    for name, path in (("contract", CONTRACT), ("builder", BUILDER),
                       ("producer_validator", PRODUCER_VALIDATOR),
                       ("result", RESULT), ("m45_analyzer", M45_ANALYZER)):
        require(sha256(path) == EXPECTED[name], "anchor drift: {}".format(name))
    contract = read_json(CONTRACT)
    canonical = read_json(RESULT)
    inputs = {}
    for name, item in contract["inputs"].items():
        path = HW_ROOT / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "input drift: {}".format(name))
        inputs[name] = read_json(path)
    rebuilt = validate_payload(canonical, contract, inputs)
    attacks = mutation_matrix(canonical, contract, inputs)
    require(deterministic_builder_check(canonical),
            "direct deterministic builder mismatch")
    source = M45_ANALYZER.read_text(encoding="utf-8")
    require("module.ALLOW_TEMPORAL_PARENT = False" in source and
            "for timestep in range(T):" in source and
            "for tile in range(TILES):" in source and
            "end_cycle = max(now" in source and
            'counts["final_accumulator_writes"] == ROWS_PER_T' in source,
            "M45 additivity/drain source evidence drift")
    return {
        "schema": "m47_r1_independent_hammer_review_v1",
        "date": "2026-08-23",
        "status": "GO_M47_R1_CONSERVATIVE_LEDGER_ONLY",
        "review": {
            "decision": "GO_M47_R1_CONSERVATIVE_LEDGER_ONLY",
            "score_0_to_100": 92,
            "p0": 0,
            "p1": 0,
            "p2": 5,
        },
        "anchors": {
            "contract": sha256(CONTRACT),
            "builder": sha256(BUILDER),
            "producer_validator": sha256(PRODUCER_VALIDATOR),
            "canonical_result": sha256(RESULT),
            "m45_analyzer": sha256(M45_ANALYZER),
            "independent_reviewer": sha256(Path(__file__).resolve()),
        },
        "candidate_modified_by_reviewer": False,
        "determinism": {
            "direct_builder_reproduces_canonical_object": True,
            "producer_validator_rerun_sha_passed": True,
        },
        "capacity_reconstruction": {
            "signed19_vector_bits": 96 * 19,
            "signed19_vector_bytes": 228,
            "signed19_frame_bits": 300 * 96 * 19,
            "signed19_frame_bytes": 68400,
            "components": rebuilt["components"],
            "combined_local_capacity_bytes": 174224,
            "local_capacity_headroom_bytes": 19504,
            "headroom_margin_above_16KiB_gate_bytes": 3120,
            "all_bit_and_byte_arithmetic_exact": True,
        },
        "weight_reconstruction": {
            "loads_expression": "4 operators * 8 output blocks * 5 timestep pairs * 27 feature tiles",
            "weight_tile_loads_per_sample": rebuilt["load_count"],
            "weight_bytes_per_sample": rebuilt["weight_bytes"],
            "serialized_load_cycles_per_sample": rebuilt["load_cycles"],
            "m45_weight_bytes_per_sample": 212336640,
            "traffic_reduction": fraction(106168320, 212336640),
            "exact_two_x_traffic_reduction": True,
        },
        "upper_bound_reconstruction": {
            "per_sample_cycles": rebuilt["upper_values"],
            "aggregate_cycles": sum(rebuilt["upper_values"]),
            "p95_nearest_rank": distribution(rebuilt["upper_values"])[
                "p95_nearest_rank"],
            "p95_is_maximum_for_ten_samples": True,
            "inherited_weight_wait_cycles_per_sample_not_subtracted": 12288,
            "deduplicated_weight_wait_p95": distribution(
                rebuilt["deduplicated_values"])["p95_nearest_rank"],
            "canonical_upper_bound_slack_vs_deduplicated_model_cycles": 12288,
            "conservative_if_nonweight_services_are_permutation_invariant": True,
            "permutation_basis": [
                "M45 disables temporal parents",
                "each tile/timestep scheduler owns fresh port calendars and drains all 300 commits",
                "two independent frames preserve each timestep final accumulator",
                "single-buffer loads are fully serialized and no inherited weight wait is subtracted",
            ],
        },
        "conditional_model_reconstruction": {
            "fixed_compute_reference_cycles": 620868243,
            "outside_four_bottleneck_model_cycles": 188824491,
            "fixed_late_scale_plus_frontend_cycles": 2636515,
            "candidate_p95_cycles": 11340632,
            "conditional_total_cycles": rebuilt["conditional_denominator"],
            "conditional_speedup": fraction(620868243, 202801638),
            "margin_over_exact_3x_numerator_cycles": 12463329,
            "three_x_conditional_model_crossing": True,
            "system_speedup_admitted": False,
        },
        "adversarial_matrix": {
            "tested": len(attacks),
            "rejected": len(attacks),
            "rejected_attacks": attacks,
        },
        "findings": {
            "p0": [],
            "p1": [],
            "p2": [
                {
                    "id": "P2_NO_NEW_ADDRESS_TIMED_PAIR_REPLAY",
                    "detail": "The upper bound is a conservative composition over M45 records, not a replay of the reordered pair loop with explicit addresses, queue occupancy, or contention.",
                },
                {
                    "id": "P2_BIT_TIGHT_MEMORY_PORT_AND_PACKING_UNPROVED",
                    "detail": "Capacity assumes one 1,824-bit signed19 vector service quantum. Packing/unpacking, macro width/banking, read-modify-write, timing, area, and energy are not implemented or measured.",
                },
                {
                    "id": "P2_PAIR_CONTROL_STATE_NOT_ITEMIZED",
                    "detail": "Frame ownership, pair/tile/weight-buffer tags, valid bits, and loop-control registers are not individually itemized. The 3,120-byte gate margin is ample for ordinary control, but the 174,224-byte figure is a datapath-buffer ledger, not whole-RTL sequential area.",
                },
                {
                    "id": "P2_SECOND_TIMESTEP_INPUT_AVAILABILITY_UNSCHEDULED",
                    "detail": "The pair loop assumes both timestep source tiles are available while one weight tile is resident. Upstream production, source SRAM traffic, and system scheduling are outside this ledger.",
                },
                {
                    "id": "P2_CONDITIONAL_3X_IS_TIGHT_AND_NOT_SYSTEM_SPEEDUP",
                    "detail": "The 3.061x ratio uses the frozen M42 compute denominator and ten windows from one sequence. It is neither measured nor full-network/system performance and has no cross-sequence tail evidence.",
                },
            ],
        },
        "next_gate": [
            "Implement/replay the exact timestep-pair address schedule with two frame identities, single-buffer load barriers, source availability, frontier/FIFO occupancy, and no-free-prefetch assertions.",
            "Miter bit-tight signed19 frame reads/writes and pair outputs against the frozen integer oracle.",
            "Then run exact-SHA VCS/SVA and Synopsys plus SRAM macro timing/power before promoting any measured cycle, PPA, energy, or system claim.",
        ],
        "claim_boundary": "GO admits only the SHA-bound bit-tight buffer byte ledger, exact half weight traffic, conservative all-ten cycle upper bound, and conditional M42 frozen-compute 3x crossing. It does not admit a new address-timed pair replay, RTL/VCS/Synopsys behavior, SRAM timing, PPA, power, energy, full-network/system speedup, external comparison, DATE headline, or best-paper claim.",
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
