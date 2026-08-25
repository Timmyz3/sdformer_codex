#!/usr/bin/env python3
"""Build the SHA-bound M47 conservative timestep-pair capacity/cycle ledger."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_REL = "contracts/m47_bit_tight_timestep_pair_single_buffer_contract_r1_20260823.json"
RESULT_REL = "results/m47_bit_tight_timestep_pair_single_buffer_r1_20260823/m47_bit_tight_timestep_pair_single_buffer.json"
EXPECTED_CONTRACT_SHA256 = "64319c18c4ac8d2bfe925c39e6d867638c803cbe2546f9781187881ec97c234d"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with path.open("r") as handle:
        return json.load(handle)


def exact_ceil_bits_to_bytes(elements, bits):
    return (elements * bits + 7) // 8


def distribution(values):
    require(values, "empty distribution")
    ordered = sorted(values)
    count = len(ordered)

    def nearest_rank(percent):
        rank = int(math.ceil(percent * count))
        return ordered[max(0, rank - 1)]

    return {
        "count": count,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean_exact": {"numerator": sum(ordered), "denominator": count},
        "p50_nearest_rank": nearest_rank(0.50),
        "p95_nearest_rank": nearest_rank(0.95),
        "p99_nearest_rank": nearest_rank(0.99),
    }


def load_inputs():
    contract_path = HW_ROOT / CONTRACT_REL
    require(sha256_path(contract_path) == EXPECTED_CONTRACT_SHA256,
            "M47 contract SHA mismatch")
    contract = read_json(contract_path)
    loaded = {}
    input_shas = {}
    for name, identity in sorted(contract["inputs"].items()):
        path = HW_ROOT / identity["path"]
        actual = sha256_path(path)
        require(actual == identity["sha256"],
                "input SHA mismatch for {}".format(name))
        loaded[name] = read_json(path)
        input_shas[name] = actual
    return contract, loaded, input_shas


def select_configuration(m45_result, name):
    matches = [entry for entry in m45_result["configurations"]
               if entry.get("name") == name]
    require(len(matches) == 1, "expected one {} configuration".format(name))
    return matches[0]


def build_result():
    contract, inputs, input_shas = load_inputs()
    geometry = contract["frozen_geometry"]
    storage_policy = contract["storage_policy"]
    gates = contract["gates"]
    m42 = inputs["m42_result"]
    m42_review = inputs["m42_independent_review"]
    m45_r2 = inputs["m45_r2_result"]
    m45_r3 = inputs["m45_r3_capacity_repair"]
    m45_r3_review = inputs["m45_r3_independent_review"]

    require(m42["status"].startswith("PASS_M42_"), "M42 result is not admitted")
    require(m42_review["status"] == "GO_M42_R1_EXACT_HEADROOM_GATE_ONLY",
            "M42 independent verdict mismatch")
    require(m45_r2["status"].startswith("PASS_M45_R2_"),
            "M45-r2 result is not admitted")
    require(m45_r2["kill_gates"]["all_kill_gates_pass"] is True,
            "M45-r2 kill gates are not all passing")
    require(m45_r3["status"].startswith("PASS_M45_R3_"),
            "M45-r3 repair is not admitted")
    require(m45_r3_review["review"]["decision"] ==
            "GO_LEDGER_ONLY_SCHEDULER_STATE_CAPACITY_REPAIR",
            "M45-r3 independent verdict mismatch")
    require(m45_r3_review["review"]["score_0_to_100"] == 94,
            "unexpected M45-r3 review score")

    k2 = select_configuration(m45_r2, "K2_CTX8_PRIMARY")
    require(k2["destination_fanout_k"] == geometry["destination_fanout_k"],
            "K2 fanout mismatch")
    require(k2["resident_contexts"] == geometry["resident_contexts"],
            "K2 context count mismatch")
    require(len(k2["per_sample"]) == 10, "M45 K2 population must be all ten")

    lanes = geometry["output_lanes"]
    accum_bits = geometry["accumulator_signed_bits"]
    vector_bytes = exact_ceil_bits_to_bytes(lanes, accum_bits)
    frame_bytes = exact_ceil_bits_to_bytes(
        geometry["spatial_tasks_per_frame"] * lanes, accum_bits)
    parent_line_bytes = exact_ceil_bits_to_bytes(20 * lanes, accum_bits)
    require(vector_bytes == storage_policy["vector_bytes"], "vector byte mismatch")
    require(frame_bytes == storage_policy["frame_bytes"], "frame byte mismatch")
    require(parent_line_bytes == storage_policy["parent_line_bytes"],
            "parent-line byte mismatch")

    context_bytes = geometry["resident_contexts"] * (
        vector_bytes + storage_policy["context_metadata_bytes_per_entry"])
    ready_frontier_bytes = (storage_policy["ready_frontier_entries"] *
                            storage_policy["ready_descriptor_bytes_per_entry"])
    response_metadata_bytes = (
        storage_policy["response_metadata_entries"] *
        storage_policy["response_metadata_aligned_bytes_per_entry"])
    complete_fifo_bytes = storage_policy["complete_fifo_entries"] * (
        vector_bytes + storage_policy["complete_tag_control_bytes_per_entry"])
    storage_parts = {
        "single_weight_tile_buffer_bytes": geometry["weight_tile_bytes"],
        "bit_tight_parent_line_bytes": parent_line_bytes,
        "support_line_bytes": storage_policy["support_line_bytes"],
        "eight_contexts_bytes": context_bytes,
        "two_bit_tight_frames_bytes": storage_policy["resident_frames"] * frame_bytes,
        "ready_frontier_bytes": ready_frontier_bytes,
        "response_metadata_fifo_bytes": response_metadata_bytes,
        "complete_fifo_bytes": complete_fifo_bytes,
    }
    combined_capacity = sum(storage_parts.values())
    capacity_headroom = geometry["local_residency_bytes"] - combined_capacity
    require(combined_capacity == 174224, "unexpected combined capacity")
    require(capacity_headroom >= gates["minimum_local_capacity_headroom_bytes"],
            "M47 capacity headroom gate failed")

    tile_loads = (geometry["operators"] * geometry["output_blocks"] *
                  geometry["timestep_pairs"] * geometry["feature_tiles"])
    weight_bytes = tile_loads * geometry["weight_tile_bytes"]
    require(weight_bytes % geometry["weight_dma_bytes_per_cycle"] == 0,
            "weight load does not divide into integral cycles")
    serialized_load_cycles = weight_bytes // geometry["weight_dma_bytes_per_cycle"]
    inherited_weight_bytes = k2["traffic_bytes_per_sample"]["weight_dma"]
    traffic_reduction_num = inherited_weight_bytes - weight_bytes
    traffic_reduction_den = inherited_weight_bytes
    require(2 * weight_bytes == inherited_weight_bytes,
            "M47 weight traffic is not exactly half M45")

    per_sample = []
    for sample in sorted(k2["per_sample"], key=lambda entry: entry["sample_id"]):
        inherited_cycles = sample["integrated_cycles"]
        upper_cycles = inherited_cycles + serialized_load_cycles
        per_sample.append({
            "sample_id": sample["sample_id"],
            "inherited_m45_k2_ctx8_integrated_cycles": inherited_cycles,
            "inherited_m45_weight_dma_wait_cycles_not_subtracted":
                sample["weight_dma_wait_cycles"],
            "serialized_single_buffer_weight_load_cycles_added":
                serialized_load_cycles,
            "conservative_integrated_cycle_upper_bound": upper_cycles,
        })
    upper_distribution = distribution([
        sample["conservative_integrated_cycle_upper_bound"] for sample in per_sample
    ])
    inherited_distribution = k2["integrated_cycle_distribution"]
    require(upper_distribution["p95_nearest_rank"] ==
            inherited_distribution["p95_nearest_rank"] + serialized_load_cycles,
            "constant load barrier did not preserve percentile ordering")
    require(upper_distribution["p95_nearest_rank"] <=
            gates["maximum_p95_candidate_cycles"],
            "M47 conservative p95 misses the 3x product-cycle gate")

    frozen = m42["frozen_resource_model"]
    fixed_reference = frozen["fixed_compute_reference_cycles"]
    outside = frozen["outside_four_bottleneck_model_cycles"]
    late_frontend = frozen["fixed_late_scale_plus_frontend_cycles"]
    p95_candidate = upper_distribution["p95_nearest_rank"]
    conditional_denominator = outside + late_frontend + p95_candidate
    minimum_speedup = gates["minimum_conditional_compute_speedup"]
    require(fixed_reference * minimum_speedup["denominator"] >=
            conditional_denominator * minimum_speedup["numerator"],
            "M47 conditional 3x gate failed")
    three_x_product_ceiling = m42["target_gates"][2][
        "maximum_executable_product_cycles_required"]["numerator"]
    require(m42["target_gates"][2][
        "maximum_executable_product_cycles_required"]["denominator"] == 1,
        "M42 3x product ceiling denominator changed")

    return {
        "schema": "m47_bit_tight_timestep_pair_single_buffer_result_v1",
        "status": "PASS_M47_CONSERVATIVE_CAPACITY_TRAFFIC_AND_CONDITIONAL_3X_GATE_RTL_SYSTEM_UNADMITTED",
        "identity": {
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "builder_sha256": sha256_path(Path(__file__).resolve()),
            "inputs_sha256": input_shas,
        },
        "architecture": {
            "name": "BIT_TIGHT_DUAL_TIMESTEP_SINGLE_WEIGHT_BUFFER_K2_CTX8",
            "loop_order": geometry["loop_order"],
            "destination_fanout_k": geometry["destination_fanout_k"],
            "resident_contexts": geometry["resident_contexts"],
            "resident_frames": storage_policy["resident_frames"],
            "weight_buffers": storage_policy["weight_buffers"],
            "no_free_prefetch": True,
        },
        "capacity": {
            "accumulator_signed_bits": accum_bits,
            "bit_tight_vector_bytes": vector_bytes,
            "bit_tight_frame_bytes": frame_bytes,
            "components": storage_parts,
            "combined_local_capacity_bytes": combined_capacity,
            "frozen_local_residency_bytes": geometry["local_residency_bytes"],
            "local_capacity_headroom_bytes": capacity_headroom,
            "minimum_required_headroom_bytes": gates[
                "minimum_local_capacity_headroom_bytes"],
            "double_weight_buffer_permitted": False,
            "external_accumulator_spill_permitted": False,
        },
        "weight_traffic": {
            "weight_tile_loads_per_sample": tile_loads,
            "weight_tile_bytes": geometry["weight_tile_bytes"],
            "candidate_bytes_per_sample": weight_bytes,
            "m45_r2_bytes_per_sample": inherited_weight_bytes,
            "reduction_fraction": {
                "numerator": traffic_reduction_num,
                "denominator": traffic_reduction_den,
            },
            "exact_two_x_reduction": True,
            "serialized_load_cycles_per_sample": serialized_load_cycles,
            "weight_dma_bytes_per_cycle": geometry["weight_dma_bytes_per_cycle"],
        },
        "conservative_cycle_upper_bound": {
            "construction": contract["cycle_policy"]["upper_bound_construction"],
            "per_sample": per_sample,
            "distribution": upper_distribution,
            "aggregate_cycles": sum(
                sample["conservative_integrated_cycle_upper_bound"]
                for sample in per_sample),
            "m45_r2_inherited_distribution": inherited_distribution,
            "maximum_p95_gate_cycles": gates["maximum_p95_candidate_cycles"],
            "p95_margin_to_gate_cycles":
                gates["maximum_p95_candidate_cycles"] - p95_candidate,
            "new_address_timed_pair_schedule_replayed": False,
        },
        "conditional_frozen_compute_model": {
            "qualification": "M42_CONDITIONAL_MODEL_NOT_FULL_NETWORK_OR_SYSTEM_SPEEDUP",
            "fixed_compute_reference_cycles": fixed_reference,
            "outside_four_bottleneck_model_cycles": outside,
            "fixed_late_scale_plus_frontend_cycles": late_frontend,
            "candidate_p95_cycle_upper_bound": p95_candidate,
            "conditional_total_cycles": conditional_denominator,
            "conditional_compute_speedup": {
                "numerator": fixed_reference,
                "denominator": conditional_denominator,
            },
            "conditional_compute_speedup_decimal":
                float(fixed_reference) / conditional_denominator,
            "three_x_crossing_in_conditional_model": True,
            "three_x_product_cycle_ceiling": three_x_product_ceiling,
            "product_cycle_headroom_to_three_x":
                three_x_product_ceiling - p95_candidate,
            "system_or_end_to_end_speedup_admitted": False,
        },
        "admission": {
            "bit_exact_capacity_admitted": True,
            "weight_traffic_reduction_admitted": True,
            "conservative_cycle_upper_bound_admitted": True,
            "conditional_frozen_compute_three_x_admitted": True,
            "new_address_timed_pair_schedule_admitted": False,
            "rtl_vcs_synopsys_admitted": False,
            "full_network_or_system_speedup_admitted": False,
            "ppa_power_energy_admitted": False,
            "date_headline_or_best_paper_admitted": False,
        },
        "claim_policy": contract["claim_policy"],
    }


def write_result(result, output_path, force):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    require(force or not output_path.exists(),
            "refusing to overwrite existing result without --force")
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary = output_path.with_name(output_path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("x") as handle:
        handle.write(payload)
    os.replace(str(temporary), str(output_path))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HW_ROOT / RESULT_REL))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    result = build_result()
    write_result(result, output, args.force)
    print("PASS M47 result {} {}".format(output, sha256_path(output)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("FAIL M47 {}".format(error), file=sys.stderr)
        sys.exit(1)
