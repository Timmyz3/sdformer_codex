#!/usr/bin/env python3
"""Build the M25 resource-bounded tiling and cycle-claim envelope.

M22/M23 transport ticks are deliberately never consumed as cycles.  M23 is
used only for frozen physical capacity and byte/service ledgers.  Cycle terms
come from the cycle-defined M4/M7/M21 evidence named by the input contract.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_div(numerator, denominator):
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def align_up(value, alignment):
    return ceil_div(value, alignment) * alignment


def load_checked(root, item):
    path = Path(item["path"])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise ValueError("missing frozen input: {}".format(path))
    observed = sha256(path)
    if observed != item["sha256"]:
        raise ValueError(
            "frozen input SHA mismatch: {} expected={} observed={}".format(
                path, item["sha256"], observed
            )
        )
    return path


def packed_atlif_service(transactions, lanes, slots):
    """Cycle-defined discrete service; rounding remains per invocation."""
    if lanes <= 0 or slots <= 0:
        raise ValueError("lanes and slots must be positive")
    service = 0
    by_steps = {}
    live = 0
    for row in transactions:
        if row.get("deployment_dead_result", "False").lower() == "true":
            continue
        steps = int(row["temporal_steps"])
        elements = int(row["elements_per_frame"])
        dense_macs = int(row["dense_macs_per_frame"])
        if steps <= 0 or slots % steps or elements % steps:
            raise ValueError("unsupported ATLIF invocation {}".format(row.get("name")))
        neurons = elements // steps
        if dense_macs != neurons * steps * steps:
            raise ValueError("ATLIF dense-MAC identity mismatch")
        groups_per_command = slots // steps
        cycles = ceil_div(neurons, lanes * groups_per_command) * steps
        service += cycles
        live += 1
        bucket = by_steps.setdefault(
            str(steps), {"invocations": 0, "neurons": 0, "service_cycles": 0}
        )
        bucket["invocations"] += 1
        bucket["neurons"] += neurons
        bucket["service_cycles"] += cycles
    if not live:
        raise ValueError("no live ATLIF invocation")
    return {"lanes": lanes, "multipliers": lanes * slots,
            "service_cycles": service, "by_temporal_steps": by_steps}


def flat_multiplier_service_lower_bound(transactions, multipliers):
    """Arithmetic-only lower bound for exactly ``multipliers`` INT8 units.

    This is not an executable ATLIF schedule because temporal recurrence and
    slot routing are ignored.  It is useful only to prove whether an ideal
    exactly-96-multiplier fabric could reach 2x before RTL is attempted.
    """
    if multipliers <= 0:
        raise ValueError("multipliers must be positive")
    service = 0
    for row in transactions:
        if row.get("deployment_dead_result", "False").lower() == "true":
            continue
        service += ceil_div(int(row["dense_macs_per_frame"]), multipliers)
    return {
        "multipliers": multipliers,
        "service_cycles_lower_bound": service,
        "status": "ARITHMETIC_ONLY_LOWER_BOUND_NOT_EXECUTABLE_ATLIF_SCHEDULE",
    }


def resident_state_by_identity(m7):
    result = {}
    for item in m7["identities"]:
        bits = int(item["resident_peak"]["total_bits"])
        if bits % 8:
            raise ValueError("resident state is not byte aligned")
        result[item["identity"]] = {
            "resident_state_bytes": bits // 8,
            "contexts": int(item["resident_peak"]["contexts"]),
            "global_materialized_bytes": (
                int(item["global_materialized_peak_per_sample"]["total_bits"]) // 8
            ),
        }
    return result


def fixed_resident_footprint(m4_cycle, m4_descriptor, m21):
    if m4_cycle["architecture"]["availability_mode"] != "temporal_fenced":
        raise ValueError("M4 must be temporal-fenced")
    descriptor = int(m4_descriptor["architecture"]["max_descriptor_buffer_bytes"])
    accumulator = ceil_div(
        int(m4_descriptor["architecture"]["accumulator_state_bits"]), 8
    )
    fifo = ceil_div(
        int(m21["configurations"]["tiles1_fifo4"]
            ["dual_line_required_fifo_payload_bits_at_configured_or_observed_maximum"]),
        8,
    )
    moments = ceil_div(int(m21["summary"]["maximum_moment_state_bits"]), 8)
    weight_response = ceil_div(
        int(m4_descriptor["architecture"]["weight_response_width_bits"]), 8
    )
    max_bn_channels = 16 * 96
    sideband_requested = 4096
    dma_requested = 4096
    fields = {
        "m4_descriptor_bytes": descriptor,
        "m4_accumulator_bytes": accumulator,
        "m4_weight_response_register_bytes": weight_response,
        "m21_fifo4_payload_bytes": fifo,
        "m21_maximum_moment_state_bytes": moments,
        "m21_fifo_sideband_and_snapshot_reserve_requested_bytes": sideband_requested,
        "m21_fifo_sideband_and_snapshot_reserve_bytes": align_up(sideband_requested, 96),
        "dynamic_bn_scale_offset_q32_bytes": max_bn_channels * 2 * 4,
        "normalization_packet_scratch_bytes": 96 * 4,
        "dma_replay_control_reserve_requested_bytes": dma_requested,
        "dma_replay_control_reserve_bytes": align_up(dma_requested, 96),
        "physical_row_alignment_bytes": 96,
    }
    allocated_fields = [
        "m4_descriptor_bytes", "m4_accumulator_bytes",
        "m4_weight_response_register_bytes", "m21_fifo4_payload_bytes",
        "m21_maximum_moment_state_bytes",
        "m21_fifo_sideband_and_snapshot_reserve_bytes",
        "dynamic_bn_scale_offset_q32_bytes", "normalization_packet_scratch_bytes",
        "dma_replay_control_reserve_bytes",
    ]
    fields["total_bytes"] = sum(fields[name] for name in allocated_fields)
    if fields["total_bytes"] % 96:
        raise ValueError("fixed resident footprint is not 96B row aligned")
    return fields


def exact_cohorts(rows, contexts):
    """Recover the same exact row-aligned C4 cohorts admitted by M7."""
    physical = {}
    by_geometry = {}
    for row in rows:
        physical_key = (
            row["sample_id"], row["sequence_key"], row["name"], row["operator"],
            row["operator_call_index"], row["row_id"], row["weight_group"],
        )
        geometry = (
            int(row["source_width"]), int(row["chunks_per_row"]),
            int(row["output_channel_fanout"]),
        )
        item = physical.setdefault(
            physical_key, {"geometry": geometry, "steps": set(), "chunks": {}}
        )
        if item["geometry"] != geometry:
            raise ValueError("physical row geometry changed")
        step = int(row["temporal_step"])
        item["steps"].add(step)
        item["chunks"].setdefault(step, set()).add(int(row["chunk_index"]))
        geometry_key = (
            row["sample_id"], row["sequence_key"], row["name"], row["operator"],
            row["operator_call_index"], row["weight_group"], row["source_width"],
            row["chunks_per_row"], row["output_channel_fanout"],
        )
        by_geometry.setdefault(geometry_key, set()).add(physical_key)
    for key, item in physical.items():
        steps = sorted(item["steps"])
        if steps != list(range(len(steps))):
            raise ValueError("non-contiguous temporal steps {}".format(key))
        chunks = item["geometry"][1]
        for step in steps:
            if item["chunks"][step] != set(range(chunks)):
                raise ValueError("incomplete row chunks {}".format(key))
    cohorts = []
    for geometry_key in sorted(by_geometry):
        keys = sorted(by_geometry[geometry_key], key=lambda key: int(key[5]))
        for start in range(0, len(keys), contexts):
            group = keys[start:start + contexts]
            if len(group) != contexts:
                raise ValueError("partial C4 cohort is outside the M7 contract")
            source_widths = [physical[key]["geometry"][0] for key in group]
            fanouts = [physical[key]["geometry"][2] for key in group]
            if len(set(fanouts)) != 1:
                raise ValueError("mixed fanout within exact cohort")
            cohorts.append({
                "cohort_id": len(cohorts),
                "sample_id": group[0][0],
                "sequence_key": group[0][1],
                "name": group[0][2],
                "operator": group[0][3],
                "operator_call_index": group[0][4],
                "weight_group": group[0][6],
                "row_ids": [int(key[5]) for key in group],
                "contexts": contexts,
                "source_widths": source_widths,
                "fanout": fanouts[0],
                "temporal_steps": len(physical[group[0]]["steps"]),
            })
    if not cohorts:
        raise ValueError("no exact cohorts")
    return cohorts


def plan_exact_cohorts(identity, budget_kib, cohorts, fixed, slots):
    """Pack only complete 96-channel row-aligned slices of an exact C4 cohort."""
    budget = int(budget_kib) * 1024
    fixed_bytes = int(fixed["total_bytes"])
    available = budget - fixed_bytes
    if available <= 0:
        raise ValueError("fixed resident footprint exceeds SRAM budget")
    plan = []
    maximum_full_payload = 0
    maximum_full_allocated = 0
    maximum_tiles = 0
    for cohort in cohorts:
        activation_payload_bytes = ceil_div(sum(cohort["source_widths"]), 8)
        activation_bytes = align_up(activation_payload_bytes, 96)
        lane_slices = []
        for lane_start in range(0, int(cohort["fanout"]), 96):
            width = min(96, int(cohort["fanout"]) - lane_start)
            # Acc32 destination plus ten Q24 ATLIF partial slots per context.
            lane_payload_bytes = int(cohort["contexts"]) * width * (4 + slots * 3)
            lane_allocated_bytes = align_up(lane_payload_bytes, 96)
            lane_slices.append(
                (lane_start, lane_start + width, lane_payload_bytes, lane_allocated_bytes)
            )
        full_payload = activation_payload_bytes + sum(item[2] for item in lane_slices)
        full_allocated = activation_bytes + sum(item[3] for item in lane_slices)
        maximum_full_payload = max(maximum_full_payload, full_payload)
        maximum_full_allocated = max(maximum_full_allocated, full_allocated)
        current = []
        current_bytes = activation_bytes
        cohort_tiles = []
        for lane_slice in lane_slices:
            if current and current_bytes + lane_slice[3] > available:
                cohort_tiles.append((current, current_bytes))
                current = []
                current_bytes = activation_bytes
            if current_bytes + lane_slice[3] > available:
                raise ValueError(
                    "one row-aligned lane slice does not fit {} {}KiB".format(
                        identity, budget_kib
                    )
                )
            current.append(lane_slice)
            current_bytes += lane_slice[3]
        if current:
            cohort_tiles.append((current, current_bytes))
        maximum_tiles = max(maximum_tiles, len(cohort_tiles))
        barrier_key = "{}|{}|{}|{}|{}".format(
            cohort["sample_id"], cohort["sequence_key"], cohort["name"],
            cohort["operator_call_index"], cohort["weight_group"]
        )
        for tile_index, tile in enumerate(cohort_tiles):
            slices, state_bytes = tile
            plan.append({
                "identity": identity,
                "budget_kib": int(budget_kib),
                "cohort_id": int(cohort["cohort_id"]),
                "sample_id": cohort["sample_id"],
                "sequence_key": cohort["sequence_key"],
                "name": cohort["name"],
                "operator": cohort["operator"],
                "operator_call_index": cohort["operator_call_index"],
                "weight_group": cohort["weight_group"],
                "row_ids": ";".join(str(value) for value in cohort["row_ids"]),
                "temporal_steps": int(cohort["temporal_steps"]),
                "barrier_key": barrier_key,
                "tile_index": tile_index,
                "tile_count": len(cohort_tiles),
                "lane_start": slices[0][0],
                "lane_end_exclusive": slices[-1][1],
                "lane_slice_count": len(slices),
                "activation_payload_bytes": activation_payload_bytes,
                "activation_state_bytes": activation_bytes,
                "lane_payload_bytes": sum(item[2] for item in slices),
                "lane_allocated_bytes": sum(item[3] for item in slices),
                "tile_payload_bytes": activation_payload_bytes + sum(
                    item[2] for item in slices
                ),
                "tile_state_bytes": state_bytes,
                "fixed_resident_bytes": fixed_bytes,
                "maximum_simultaneous_bytes": fixed_bytes + state_bytes,
                "dependency": (
                    "first_pass_tile_{}->first_pass_tile_{}".format(
                        tile_index - 1, tile_index
                    ) if tile_index else "operator_begin->first_pass_tile_0"
                ),
                "phase_boundary_dependency": (
                    "first_pass_tile_{}->same_identity_phase_boundary_once".format(
                        len(cohort_tiles) - 1
                    ) if tile_index == len(cohort_tiles) - 1 else "NOT_FINAL_TILE"
                ),
                "second_pass_dependency": (
                    "same_identity_phase_boundary_once->second_pass_tile_0" if tile_index == 0
                    else "second_pass_tile_{}->second_pass_tile_{}".format(
                        tile_index - 1, tile_index
                    )
                ),
                "barrier_release": (
                    "second_pass_tile_{}->operator_barrier_release".format(tile_index)
                    if tile_index == len(cohort_tiles) - 1 else "NOT_FINAL_TILE"
                ),
                "barrier_crossing": 0,
            })
    if any(item["maximum_simultaneous_bytes"] > budget for item in plan):
        raise AssertionError("exact row-aligned plan exceeded SRAM budget")
    if any(item["tile_state_bytes"] % 96 for item in plan):
        raise AssertionError("exact tile state is not 96B row aligned")
    return {
        "identity": identity,
        "budget_kib": int(budget_kib),
        "status": "LEGAL_FOR_FROZEN_NON_ATTENTION_C4_COHORTS",
        "qualification": (
            "H67 abstract attention and Local5 missing attention are outside this exact plan"
        ),
        "budget_bytes": budget,
        "fixed_resident_bytes": fixed_bytes,
        "available_tile_state_bytes": available,
        "exact_cohorts": len(cohorts),
        "plan_records": len(plan),
        "maximum_full_cohort_payload_bytes": maximum_full_payload,
        "maximum_full_cohort_allocated_bytes": maximum_full_allocated,
        "maximum_tiles_per_cohort": maximum_tiles,
        "maximum_simultaneous_bytes": max(
            item["maximum_simultaneous_bytes"] for item in plan
        ),
        "dynamic_bn_barrier_crossings": sum(item["barrier_crossing"] for item in plan),
        "extra_source_activation_reads_upper_bound": max(0, maximum_tiles - 1),
    }, plan


def m21_operator_barrier_schedule(m21, bubbles_per_lane_tile, exact_cohort_rows):
    rows = [
        row for row in m21["selected_operator_rows"]
        if row["configuration"] == "tiles1_fifo4"
    ]
    if len(rows) != 13:
        raise ValueError("expected 13 M21 FIFO4 operator rows")
    fanout_by_name = {}
    for cohort in exact_cohort_rows:
        fanout_by_name.setdefault(cohort["name"], set()).add(int(cohort["fanout"]))
    schedule = []
    for row in rows:
        name = row["operator"]
        fanouts = fanout_by_name.get(name, set())
        if len(fanouts) != 1:
            raise ValueError("M21 operator has missing or mixed exact fanout {}".format(name))
        fanout = next(iter(fanouts))
        lane_tiles = ceil_div(fanout, 96)
        incremental = (
            int(row["source_plus_moment_makespan_cycles"])
            - int(row["source_cycles_without_stalls"])
        )
        if incremental != (
                int(row["producer_stall_cycles"]) + int(row["barrier_drain_cycles"])):
            raise ValueError("M21 phase-1 incremental identity failed")
        schedule.append({
            "operator": name,
            "operator_call_index": int(row["operator_call_index"]),
            "exact_frozen_fanout": fanout,
            "lane_tile_count": lane_tiles,
            "fifo4_producer_stall_cycles": int(row["producer_stall_cycles"]),
            "fifo4_barrier_drain_cycles": int(row["barrier_drain_cycles"]),
            "fifo4_phase1_incremental_cycles": incremental,
            "registered_result_bubble_cycles": lane_tiles * bubbles_per_lane_tile,
            "coefficient_generation_count": 1,
            "operator_barrier_crossings": 0,
            "schedule": (
                "clear moments; execute every first-pass capacity tile while moments persist; "
                "after the final tile retire {} lane tiles x six registered subtiles; generate "
                "coefficients once; execute every second-pass normalization/ATLIF tile; release "
                "this operator barrier"
            ).format(lane_tiles),
        })
    return schedule


def memory_cycles(payload_bytes, bandwidth_gbps, frequency_mhz):
    # Decimal GB/s, consistent with the published 64 GB/s interface convention.
    bytes_per_cycle = float(bandwidth_gbps) * 1000.0 / float(frequency_mhz)
    return int(math.ceil(float(payload_bytes) / bytes_per_cycle))


def proposal_compute(m7, line, atlif_service_cycles, m21_phase1_incremental_cycles,
                     registered_bubbles,
                     effective_m4_speed=None):
    env = m7["system_envelope"]
    variant = env["variants"][line]
    speed = float(variant["effective_m4_speedup_vs_local_p1"])
    if effective_m4_speed is not None:
        speed = float(effective_m4_speed)
    m4_eligible = int(env["m4_profiled_eligible_cycles"])
    fixed_operator = (
        int(env["noneligible_operator_cycles"])
        + int(env["qk_projection_cycles_frozen_unprofiled"])
    )
    attention = int(env["rqtb_attention_cycles"])
    accelerated = int(math.ceil(float(m4_eligible) / speed))
    total = fixed_operator + accelerated + int(atlif_service_cycles) + attention
    total += int(m21_phase1_incremental_cycles) + int(registered_bubbles)
    return {
        "noneligible_plus_qk_cycles": fixed_operator,
        "m4_profiled_eligible_cycles": m4_eligible,
        "effective_m4_speed": speed,
        "accelerated_m4_cycles": accelerated,
        "atlif_service_cycles": int(atlif_service_cycles),
        "rqtb_attention_cycles": attention,
        "m21_fifo4_phase1_incremental_cycles": int(m21_phase1_incremental_cycles),
        "m21_registered_result_bubble_cycles": int(registered_bubbles),
        "compute_cycles": total,
    }


def two_x_requirements(m7, line, atlif_service, m21_phase1_incremental_cycles,
                       registered_bubbles,
                       effective_m4_speed=None):
    env = m7["system_envelope"]
    target = float(env["fixed_baseline_cycles"]) / 2.0
    fixed = (
        int(env["noneligible_operator_cycles"])
        + int(env["qk_projection_cycles_frozen_unprofiled"])
        + int(env["rqtb_attention_cycles"])
        + int(m21_phase1_incremental_cycles)
        + int(registered_bubbles)
    )
    eligible = int(env["m4_profiled_eligible_cycles"])
    current_speed = float(env["variants"][line]["effective_m4_speedup_vs_local_p1"])
    if effective_m4_speed is not None:
        current_speed = float(effective_m4_speed)
    service = int(atlif_service["service_cycles"])
    denominator = target - fixed - service
    required_m4 = "IMPOSSIBLE_EVEN_INFINITE_M4"
    if denominator > 0:
        required_m4 = float(eligible) / denominator
    accelerated = int(math.ceil(float(eligible) / current_speed))
    allowed_atlif = int(math.floor(target - fixed - accelerated))
    required_atlif_speed = "IMPOSSIBLE_WITH_CURRENT_M4"
    if allowed_atlif > 0:
        required_atlif_speed = float(service) / float(allowed_atlif)
    amdahl_infinite_m4_atlif = float(env["fixed_baseline_cycles"]) / float(fixed)
    return {
        "target_cycles_for_2x": target,
        "required_m4_speed_at_this_discrete_atlif_point": required_m4,
        "maximum_atlif_cycles_at_current_m4_speed": allowed_atlif,
        "required_atlif_speedup_at_current_m4_speed": required_atlif_speed,
        "infinite_m4_and_atlif_amdahl_speedup": amdahl_infinite_m4_atlif,
    }


def adapter_matrix(config):
    common = {
        "frequency_mhz": config["frequency_mhz"],
        "dram_bandwidth_gbps": config["headline_bandwidth_gbps"],
        "sram_budgets_kib": config["sram_budgets_kib"],
        "int8_multiplier_equivalent_budget": config["logic_budget"]["int8_multipliers"],
        "workload_identity": "H67_ep35_and_Local5_ep44_ordered_trace",
    }
    return [
        dict(common, adapter="Fixed", cycle="INTERNAL_ACTIVITY_MODEL_CYCLE_DEFINED",
             memory="M23_LOCAL_BOUNDARY_LEDGER", logic="96_MAC_DECLARED_NOT_SYNTHESIS_MATCHED",
             accuracy="FROZEN_CHECKPOINT", status="PARTIAL_NOT_PAPER_COMPARABLE"),
        dict(common, adapter="RQTB", cycle="INTERNAL_ATTENTION_ANCHOR_ONLY_PLUS_FIXED_STACK",
             memory="M23_LOCAL_BOUNDARY_LEDGER", logic="SAME_96_MAC_DECLARED_NOT_SYNTHESIS_MATCHED",
             accuracy="FROZEN_CHECKPOINT", status="PARTIAL_NOT_PAPER_COMPARABLE"),
        dict(common, adapter="M25_L8", cycle="DISCRETE_L8_SERVICE_MODEL_80_MULTIPLIERS",
             memory="M23_PLUS_LEGAL_TILE_REPLAY", logic="M4_M21_AREA_NOT_MATCHED_TO_FIXED",
             accuracy="FROZEN_OPERATOR_IDENTITIES", status="PARTIAL_UNDER_ATLIF_MULTIPLIER_BUDGET"),
        dict(common, adapter="M25_L10", cycle="DISCRETE_L10_SERVICE_MODEL_REQUIRES_RTL_DSE",
             memory="M23_PLUS_LEGAL_TILE_REPLAY", logic="100_MULTIPLIERS_EXCEEDS_96_BUDGET",
             accuracy="FROZEN_OPERATOR_IDENTITIES", status="FAIL_SAME_LOGIC_BUDGET"),
        dict(common, adapter="Prosperity", cycle="MISSING_SAME_WORKLOAD_ADAPTER",
             memory="MISSING_SAME_ADDRESS_TRACE_ADAPTER", logic="MISSING_SAME_PPA_BUDGET_ADAPTER",
             accuracy="MISSING_SAME_CHECKPOINT_ADAPTER", status="NO_GO_EXTERNAL_COMPARISON"),
        dict(common, adapter="Phi-like", cycle="MISSING_SAME_WORKLOAD_ADAPTER",
             memory="MISSING_SAME_ADDRESS_TRACE_ADAPTER", logic="MISSING_SAME_PPA_BUDGET_ADAPTER",
             accuracy="MISSING_SAME_CHECKPOINT_ADAPTER", status="NO_GO_EXTERNAL_COMPARISON"),
    ]


def build(contract_path, output):
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    repo_root = contract_path.resolve().parents[2]
    inputs = {}
    paths = {}
    for name, item in contract["inputs"].items():
        path = load_checked(repo_root, item)
        paths[name] = path
        if path.suffix == ".csv":
            with path.open(encoding="utf-8") as handle:
                inputs[name] = list(csv.DictReader(handle))
        else:
            inputs[name] = json.loads(path.read_text(encoding="utf-8"))

    m23 = inputs["m23"]
    m21 = inputs["m21"]
    m4_cycle = inputs["m4_stateful_cycles"]
    m4_descriptor = inputs["m4_descriptor_cycles"]
    m7 = inputs["m7"]
    transactions = inputs["atlif_transactions"]
    if m23["claim_boundary"]["forbidden"][1].find("system cycles") < 0:
        raise ValueError("M23 cycle claim boundary changed")
    if m21["configurations"]["tiles1_fifo4"]["local_maximum_resident_packets"] != 4:
        raise ValueError("M21 one-slice/FIFO4 identity changed")

    fixed = fixed_resident_footprint(m4_cycle, m4_descriptor, m21)
    states = resident_state_by_identity(m7)
    config = contract["uniform_resource_contract"]
    tilings = []
    exact_rows = {
        "H67": inputs["h67_tile_records"],
        "Local5": inputs["local5_tile_records"],
    }
    exact_by_identity = {
        identity: exact_cohorts(exact_rows[identity], 4)
        for identity in sorted(exact_rows)
    }
    exact_plan = []
    for identity in sorted(exact_by_identity):
        for budget in config["sram_budgets_kib"]:
            summary, rows = plan_exact_cohorts(
                identity, budget, exact_by_identity[identity], fixed,
                int(config["atlif_slots"])
            )
            if summary["maximum_full_cohort_payload_bytes"] != states[identity][
                    "resident_state_bytes"]:
                raise ValueError("exact cohort peak no longer matches M7 {}".format(identity))
            if identity == "Local5":
                summary["status"] = (
                    "LEGAL_FOR_FROZEN_NON_ATTENTION_C4_COHORTS_"
                    "FULL_SYSTEM_CAPACITY_UNKNOWN"
                )
            else:
                summary["status"] = (
                    "LEGAL_FOR_FROZEN_C4_COHORTS_"
                    "ABSTRACT_ATTENTION_PHYSICAL_CAPACITY_UNKNOWN"
                )
            tilings.append(summary)
            exact_plan.extend(rows)

    discrete = {}
    for lanes in config["atlif_lane_points"]:
        point = packed_atlif_service(transactions, int(lanes), int(config["atlif_slots"]))
        if int(lanes) == 16:
            proof = "L16_SERVICE_MODEL_PLUS_EXISTING_L16_CHECKPOINT_VCS_FUNCTIONAL_PROOF"
        else:
            proof = "DSE_ONLY_REQUIRES_RTL_VCS_DC"
        point["evidence_status"] = proof
        point["same_96_multiplier_budget"] = point["multipliers"] <= 96
        point["distance_from_96_multiplier_budget"] = point["multipliers"] - 96
        discrete[str(lanes)] = point

    lane16_expected = int(
        m7["system_envelope"]["variants"]["local"]["stream_points"][0]
        ["atlif_stream_service_cycles"]
    )
    if discrete["16"]["service_cycles"] != lane16_expected:
        raise ValueError("discrete L16 ATLIF service no longer matches M7")
    flat96 = flat_multiplier_service_lower_bound(
        transactions, int(config["logic_budget"]["int8_multipliers"])
    )

    correction_contract = contract["m21_registered_result_correction"]
    lane_tile_count = int(correction_contract["lane_tile_count"])
    derived_lane_tiles = sum(
        int(value) for value in correction_contract["lane_tiles_by_operator_group"].values()
    )
    if lane_tile_count != derived_lane_tiles:
        raise ValueError("M21 lane-tile correction derivation does not sum")
    bubbles_per_tile = int(
        correction_contract["cycles_per_lane_tile"]
    )
    bubbles = lane_tile_count * bubbles_per_tile
    m21_fifo4 = m21["configurations"]["tiles1_fifo4"]
    operator_barriers = m21_operator_barrier_schedule(
        m21, bubbles_per_tile, exact_by_identity["H67"]
    )
    if sum(row["lane_tile_count"] for row in operator_barriers) != lane_tile_count:
        raise ValueError("per-operator M21 lane-tile schedule does not sum")
    if sum(row["registered_result_bubble_cycles"] for row in operator_barriers) != bubbles:
        raise ValueError("per-operator registered-result correction does not sum")
    m21_incremental = {
        "local": (
            int(m21_fifo4["local_source_plus_moment_makespan_cycles"])
            - int(m21_fifo4["local_source_cycles_without_stalls"])
        ),
        "hybrid": (
            int(m21_fifo4["hybrid_source_plus_moment_makespan_cycles"])
            - int(m21_fifo4["hybrid_source_cycles_without_stalls"])
        ),
    }
    if sum(row["fifo4_phase1_incremental_cycles"] for row in operator_barriers) \
            != m21_incremental["local"]:
        raise ValueError("per-operator Local M21 phase-1 increment does not sum")
    m21_correction = {
        "implemented_architecture": "ONE_ARITHMETIC_SLICE_FIFO4",
        "dse_only_not_implemented": "THREE_ARITHMETIC_SLICES_FIFO40",
        "lane_tile_count_contract": lane_tile_count,
        "lane_tiles_by_operator_group": correction_contract["lane_tiles_by_operator_group"],
        "registered_result_cycles_per_lane_tile": bubbles_per_tile,
        "registered_result_bubble_cycles": bubbles,
        "per_operator_barrier_schedule": operator_barriers,
        "local_fifo4_phase1_incremental_cycles": m21_incremental["local"],
        "hybrid_fifo4_phase1_incremental_cycles": m21_incremental["hybrid"],
        "local_total_incremental_cycles_added_to_m7": m21_incremental["local"] + bubbles,
        "hybrid_total_incremental_cycles_added_to_m7": m21_incremental["hybrid"] + bubbles,
        "local_payload_only_cycles_before_correction": int(
            m21_fifo4["local_payload_only_region_cycles"]
        ),
        "local_payload_only_cycles_after_correction": int(
            m21_fifo4["local_payload_only_region_cycles"]
        ) + bubbles,
        "hybrid_payload_only_cycles_after_correction": int(
            m21_fifo4["hybrid_payload_only_region_cycles"]
        ) + bubbles,
        "composition_boundary": (
            "add only FIFO4 phase-1 producer stall plus barrier drain and the registered-result "
            "bubbles to M7. Do not add M21 source-without-stalls or phase-2 replay because those "
            "overlap M4 source and ATLIF terms already present in M7"
        ),
    }

    compute = {}
    requirements = {}
    # M7-v3's Motion envelope used the pre-state-RMW M4 wall cycles.  Rebind
    # both lines to the later temporal-fenced stateful evidence.  The numerator
    # stays the same Local P1 population so Local and Motion compare against an
    # identical baseline.
    m4_local_h67 = m4_cycle["variants"]["local"]["per_identity"]["H67"]
    m4_hybrid_h67 = m4_cycle["variants"]["hybrid"]["per_identity"]["H67"]
    common_p1_cycles = int(m4_local_h67["p1_sparse_wall_cycles"])
    effective_m4_speeds = {
        "local": float(common_p1_cycles) / float(m4_local_h67["stateful_wall_cycles"]),
        "hybrid": float(common_p1_cycles) / float(m4_hybrid_h67["stateful_wall_cycles"]),
    }
    for line in ("local", "hybrid"):
        compute[line] = {}
        requirements[line] = {}
        for lanes in config["atlif_lane_points"]:
            key = str(lanes)
            compute[line][key] = proposal_compute(
                m7, line, discrete[key]["service_cycles"],
                m21_incremental[line], bubbles,
                effective_m4_speeds[line]
            )
            compute[line][key]["speedup_vs_fixed_compute"] = (
                float(m7["system_envelope"]["fixed_baseline_cycles"])
                / compute[line][key]["compute_cycles"]
            )
            requirements[line][key] = two_x_requirements(
                m7, line, discrete[key], m21_incremental[line], bubbles,
                effective_m4_speeds[line]
            )
        compute[line]["flat96_lower_bound"] = proposal_compute(
            m7, line, flat96["service_cycles_lower_bound"],
            m21_incremental[line], bubbles,
            effective_m4_speeds[line]
        )
        compute[line]["flat96_lower_bound"]["speedup_vs_fixed_compute"] = (
            float(m7["system_envelope"]["fixed_baseline_cycles"])
            / compute[line]["flat96_lower_bound"]["compute_cycles"]
        )
        requirements[line]["flat96_lower_bound"] = two_x_requirements(
            m7, line, {"service_cycles": flat96["service_cycles_lower_bound"]},
            m21_incremental[line], bubbles, effective_m4_speeds[line]
        )

    # M23 has ten ordered samples.  Use bytes/sample only; no M23 tick is a cycle.
    samples = int(contract["m23_ordered_samples"])
    h67_variants = m23["identities"]["h67_ep35"]["variants"]
    payload_by_variant = {}
    activation_input_per_sample = {}
    for variant in ("local_line", "motion_selector_shared_state",
                    "motion_selector_explicit_copy"):
        payload_by_variant[variant] = ceil_div(
            int(h67_variants[variant]["transport_and_bank_service"]["dram_payload_bytes"]),
            samples,
        )
        activation_input_per_sample[variant] = ceil_div(
            int(h67_variants[variant]["lifetime_payload_bytes_by_category"]
                ["activation_input"]), samples
        )

    sensitivity = []
    fixed_compute = int(m7["system_envelope"]["fixed_baseline_cycles"])
    for tiling in tilings:
        if tiling["identity"] != "H67" or not tiling["status"].startswith("LEGAL"):
            continue
        replay = int(tiling["extra_source_activation_reads_upper_bound"])
        for line, memory_variant in (
                ("local", "local_line"),
                ("motion_shared", "motion_selector_shared_state"),
                ("motion_copy", "motion_selector_explicit_copy")):
            compute_line = "local" if line == "local" else "hybrid"
            for lanes in config["atlif_lane_points"]:
                proposal = int(compute[compute_line][str(lanes)]["compute_cycles"])
                for bandwidth in config["dram_bandwidth_sweep_gbps"]:
                    base_bytes = int(payload_by_variant["local_line"])
                    prop_bytes_lower = int(payload_by_variant[memory_variant])
                    prop_bytes_upper = prop_bytes_lower + replay * int(
                        activation_input_per_sample[memory_variant]
                    )
                    base_mem = memory_cycles(
                        base_bytes, bandwidth, config["frequency_mhz"]
                    )
                    prop_mem_lower = memory_cycles(
                        prop_bytes_lower, bandwidth, config["frequency_mhz"]
                    )
                    prop_mem_upper = memory_cycles(
                        prop_bytes_upper, bandwidth, config["frequency_mhz"]
                    )
                    sensitivity.append({
                        "identity": "H67",
                        "line": line,
                        "memory_variant": memory_variant,
                        "budget_kib": tiling["budget_kib"],
                        "tile_count": tiling["maximum_tiles_per_cohort"],
                        "atlif_lanes": int(lanes),
                        "atlif_multipliers": discrete[str(lanes)]["multipliers"],
                        "bandwidth_gbps": int(bandwidth),
                        "fixed_compute_cycles": fixed_compute,
                        "proposal_compute_cycles": proposal,
                        "baseline_dram_payload_bytes_per_sample": base_bytes,
                        "proposal_dram_payload_bytes_per_sample_lower": prop_bytes_lower,
                        "proposal_dram_payload_bytes_per_sample_replay_upper": prop_bytes_upper,
                        "baseline_memory_ideal_cycles": base_mem,
                        "proposal_memory_ideal_cycles_lower": prop_mem_lower,
                        "proposal_memory_ideal_cycles_replay_upper": prop_mem_upper,
                        "serialized_speedup_lower_replay_upper": (
                            float(fixed_compute + base_mem) /
                            float(proposal + prop_mem_upper)
                        ),
                        "serialized_speedup_no_extra_replay": (
                            float(fixed_compute + base_mem) /
                            float(proposal + prop_mem_lower)
                        ),
                        "perfect_overlap_speedup_lower_replay_upper": (
                            float(max(fixed_compute, base_mem)) /
                            float(max(proposal, prop_mem_upper))
                        ),
                        "perfect_overlap_speedup_no_extra_replay": (
                            float(max(fixed_compute, base_mem)) /
                            float(max(proposal, prop_mem_lower))
                        ),
                        "same_logic_96_multiplier_status": (
                            "UNDER_OR_EQUAL_ATLIF_ONLY" if discrete[str(lanes)]["multipliers"] <= 96
                            else "EXCEEDS_ATLIF_MULTIPLIER_BUDGET"
                        ),
                        "claim": "IDEAL_BANDWIDTH_SENSITIVITY_NOT_DRAMSIM3_OR_MEASURED_SYSTEM_CYCLES",
                    })

    local_attention = m23["identities"]["local_ep44"]["attention"]
    local_status = {
        "tiling_capacity": "CONDITIONAL_NON_ATTENTION_C4_PLAN_ONLY",
        "full_system_cycles": "UNKNOWN",
        "speedup": "UNKNOWN",
        "attention_status": local_attention["status"],
        "minimum_missing_module_calls": local_attention["minimum_missing_module_calls"],
        "reason": (
            "Local5 attention is unknown nonzero and can change tile count; no zero-cost or "
            "full-system capacity substitution is permitted"
        ),
    }
    h67_attention = m23["identities"]["h67_ep35"]["attention"]

    adapters = adapter_matrix(config)
    best_implemented_point = compute["local"]["16"]
    best_equal_under_point = compute["local"]["8"]
    exact96_arithmetic_lower_bound = compute["local"]["flat96_lower_bound"]
    conclusions = {
        "crosses_2x_compute_at_existing_l16": (
            best_implemented_point["speedup_vs_fixed_compute"] >= 2.0
        ),
        "crosses_2x_at_discrete_l8_under_96_atlif_multiplier_budget": (
            best_equal_under_point["speedup_vs_fixed_compute"] >= 2.0
        ),
        "crosses_2x_at_ideal_flat_exact96_arithmetic_lower_bound": (
            exact96_arithmetic_lower_bound["speedup_vs_fixed_compute"] >= 2.0
        ),
        "paper_headline_over_2x_status": "NO_GO",
        "paper_headline_blockers": [
            "L16 uses 160 ATLIF multipliers versus the declared 96-MAC baseline",
            "L8 is below the ATLIF multiplier budget but does not cross 2x",
            "L10 is the nearest discrete point at or above 96 but uses 100 multipliers and is DSE-only",
            "even the non-executable arithmetic-only exact-96 lower bound does not cross 2x",
            "M4 plus M21 logic area is not matched to Fixed/RQTB",
            "M23 is not DRAMsim3 timing and H67 attention traffic is an abstract lower bound",
            "Local5 attention is unknown nonzero",
            "Prosperity and Phi-like same-workload adapters are missing",
        ],
        "highest_value_next_rtl": {
            "name": "M25A_EXACT96_RANK3_FACTORIZED_T10_CO_DESIGN",
            "objective": (
                "algorithm-hardware co-design a trained rank-3 factorized T10 transform "
                "(theoretical 60 versus 100 MAC per neuron), time-shared on exactly 96 arithmetic "
                "lanes with M4; add a barrier-indexed tile DMA/replay controller and one-entry "
                "M21 result snapshot queue"
            ),
            "why": (
                "the ideal exact-96 arithmetic lower bound still misses 2x, so lane sharing "
                "alone is insufficient. A frozen checkpoint audit reports weak diagonal/rank-2 "
                "structure but substantially stronger rank-4 energy, making trained rank-3 a "
                "candidate work-reduction point rather than an exact transform claim"
            ),
            "cycle_target": {
                "current_exact96_arithmetic_lower_bound_cycles": flat96[
                    "service_cycles_lower_bound"
                ],
                "maximum_atlif_cycles_for_2x_at_current_local_m4": requirements[
                    "local"
                ]["flat96_lower_bound"]["maximum_atlif_cycles_at_current_m4_speed"],
                "required_additional_atlif_speedup": requirements["local"][
                    "flat96_lower_bound"
                ]["required_atlif_speedup_at_current_m4_speed"],
            },
            "required_proof": (
                "fine-tune and accuracy admission first; freeze rank factors and all 105 ATLIF "
                "matrix identities; only then build VCS numeric RTL, DC area-constrained A/B, "
                "and address-timed SRAM/DRAM schedule; do not credit 60/100 as speedup before "
                "those gates and do not implement three-slice/FIFO40 first"
            ),
        },
        "rank3_candidate_not_credited": {
            "status": "UNSEALED_REMOTE_CHECKPOINT_AUDIT_DESIGN_CANDIDATE_ONLY",
            "matrix_population": {"total": 105, "T10": 45, "T2": 60},
            "T10_rank2_energy_min_median_max": [0.4714, 0.6303, 0.8524],
            "T10_rank4_energy_min_median_max": [0.7736, 0.8594, 0.9542],
            "T10_diagonal_energy_median": 0.0968,
            "interpretation": (
                "diagonal and existing low-rank approximations are not exact; trained rank-3 "
                "requires accuracy fine-tuning and receives zero cycle/speedup credit in M25"
            ),
        },
    }

    payload = {
        "schema": "m25_resource_bounded_tiled_cycle_architecture_v1",
        "status": "PASS_FROZEN_C4_TILING_AND_CYCLE_ENVELOPE_HEADLINE_NO_GO",
        "uniform_resource_contract": config,
        "claim_boundary": {
            "permitted": [
                "legal SRAM-bounded spill/replay tiling for frozen non-attention C4 cohorts",
                "M4/M7/M21 cycle-defined compute envelopes and explicit correction",
                "ideal-bandwidth serialized/perfect-overlap sensitivity",
                "same-interface baseline gap matrix and 2x requirement equations",
            ],
            "forbidden": [
                "calling any M22/M23 transport or bank-service tick a cycle",
                "claiming DRAMsim3 timing, FPS, energy, or paper PPA",
                "claiming L16 versus 96-MAC as same logic resource",
                "claiming Local5 attention is zero or reporting Local5 system speedup",
                "claiming three-slice/FIFO40 is implemented RTL",
                "direct numeric speedup versus Prosperity or Phi without adapters",
            ],
        },
        "source_evidence": {
            name: {"path": str(paths[name]), "sha256": sha256(paths[name])}
            for name in sorted(paths)
        },
        "boundary_materialized_working_set": {
            "h67_allocator_capacity_bytes": int(
                h67_variants["local_line"]["allocation"]["allocator_capacity_bytes"]
            ),
            "h67_peak_live_aligned_bytes": int(
                h67_variants["local_line"]["allocation"]["peak_live_aligned_bytes"]
            ),
            "classification": "DRAM_WORKING_SET_REQUIRES_TILING_NOT_ON_CHIP_SRAM",
        },
        "fixed_resident_footprint": fixed,
        "legal_tilings": tilings,
        "exact_row_aligned_plan": {
            "records": len(exact_plan),
            "cohort_identity_fields": [
                "sample_id", "sequence_key", "name", "operator_call_index",
                "weight_group", "row_ids"
            ],
            "lane_alignment": 96,
            "dynamic_bn_barrier_crossings": sum(
                item["barrier_crossing"] for item in exact_plan
            ),
            "qualification": (
                "exact for frozen M4/ATLIF C4 tile records; H67 abstract attention and "
                "Local5 missing attention are outside the plan"
            ),
        },
        "m21_registered_result_correction": m21_correction,
        "atlif_discrete_points": discrete,
        "atlif_exact96_arithmetic_lower_bound": flat96,
        "compute_envelopes": compute,
        "effective_m4_speed_rebind": {
            "baseline": "H67_LOCAL_P1_SPARSE_CYCLE_POPULATION",
            "local": effective_m4_speeds["local"],
            "hybrid_motion_stateful": effective_m4_speeds["hybrid"],
            "reason": "charge the later M4 temporal-fenced one-cycle Motion RMW evidence",
        },
        "two_x_requirements": requirements,
        "bandwidth_sram_sensitivity": sensitivity,
        "attention_completeness": {
            "H67": {
                "cycle_anchor": "M7_RQTB_CYCLE_DEFINED",
                "physical_traffic": h67_attention["status"],
                "unmodeled_physical_bytes": h67_attention["unmodeled_physical_bytes"],
            },
            "Local5": local_status,
        },
        "baseline_adapter_matrix": adapters,
        "conclusions": conclusions,
        "generator": {
            "script_sha256": sha256(Path(__file__)),
            "contract_sha256": sha256(contract_path),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "m25_resource_bounded_tiled_cycles.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (output / "m25_tiling_schedule.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["identity", "budget_kib", "status", "budget_bytes",
                  "fixed_resident_bytes", "maximum_full_cohort_payload_bytes",
                  "maximum_full_cohort_allocated_bytes",
                  "exact_cohorts", "plan_records", "maximum_tiles_per_cohort",
                  "available_tile_state_bytes", "maximum_simultaneous_bytes",
                  "dynamic_bn_barrier_crossings", "qualification"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tilings)
    with (output / "m25_exact_row_aligned_plan.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exact_plan[0].keys()))
        writer.writeheader()
        writer.writerows(exact_plan)
    with (output / "m25_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity[0].keys()))
        writer.writeheader()
        writer.writerows(sensitivity)
    with (output / "m25_adapter_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adapters[0].keys()))
        writer.writeheader()
        writer.writerows(adapters)

    report = [
        "# M25 resource-bounded tiling + cycle architecture\n\n",
        "Status: **headline >2x NO-GO**. Exact row-aligned tiling is proven for frozen "
        "non-attention C4 cohorts, but current >2x compute exists only at L16/160 ATLIF "
        "multipliers and is not a same-96-MAC comparison.\n\n",
        "| identity | SRAM | fixed resident | cohort state | tiles | legal |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for item in tilings:
        report.append("| {identity} | {budget_kib} KiB | {fixed_resident_bytes} B | "
                      "{maximum_full_cohort_allocated_bytes} B | {maximum_tiles_per_cohort} | "
                      "{status} |\n".format(**item))
    report.extend([
        "\n| line | lanes | multipliers | compute cycles | vs Fixed compute | evidence |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ])
    for line in ("local", "hybrid"):
        for lanes in config["atlif_lane_points"]:
            point = compute[line][str(lanes)]
            report.append(
                "| {} | {} | {} | {} | {:.6f}x | {} |\n".format(
                    line, lanes, discrete[str(lanes)]["multipliers"],
                    point["compute_cycles"], point["speedup_vs_fixed_compute"],
                    discrete[str(lanes)]["evidence_status"]
                )
            )
        lower = compute[line]["flat96_lower_bound"]
        report.append(
            "| {} | flat96 | 96 | {} | {:.6f}x | ARITHMETIC LOWER BOUND; NOT EXECUTABLE |\n".format(
                line, lower["compute_cycles"], lower["speedup_vs_fixed_compute"]
            )
        )
    report.extend([
        "\nM21 is charged as the implemented one-slice/FIFO4 architecture with Local/Hybrid "
        "phase-1 increments {}/{} cycles plus {} registered-result bubbles ({} lane tiles x {} "
        "cycles, bound per operator). Three-slice/FIFO40 remains DSE-only. M23 ticks are not "
        "cycles. Local5 full-system capacity and speedup are UNKNOWN because attention is "
        "missing nonzero.\n\n".format(
            m21_incremental["local"], m21_incremental["hybrid"], bubbles,
            lane_tile_count, bubbles_per_tile
        ),
        "The next candidate is trained rank-3 factorized T10 on an exactly-96 resource-shared "
        "ATLIF/M4 lane cluster, plus a barrier-indexed tile replay controller and one-entry M21 "
        "result snapshot queue. It receives zero M25 speed credit: fine-tune/accuracy admission "
        "must precede VCS numeric RTL, area-constrained DC A/B, and address-timed memory "
        "simulation.\n",
    ])
    (output / "REPORT.md").write_text("".join(report), encoding="utf-8")

    artifact_files = [json_path, output / "m25_tiling_schedule.csv",
                      output / "m25_exact_row_aligned_plan.csv",
                      output / "m25_sensitivity.csv", output / "m25_adapter_matrix.csv",
                      output / "REPORT.md"]
    manifest = {
        "schema": "m25_output_manifest_v1",
        "files": [{"path": item.name, "bytes": item.stat().st_size,
                   "sha256": sha256(item)} for item in artifact_files],
        "claim": "RESOURCE_AND_CYCLE_ENVELOPE_NOT_PAPER_SYSTEM_SPEEDUP",
    }
    (output / "m25_output_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PASS: wrote {}".format(json_path))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.contract, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
