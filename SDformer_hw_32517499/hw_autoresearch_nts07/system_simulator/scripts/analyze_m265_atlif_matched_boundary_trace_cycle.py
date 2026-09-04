#!/usr/bin/env python3
"""Seal M258 corrections and execute a matched-boundary ATLIF cycle model."""

import argparse
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-finite JSON token: {}".format(token))

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=reject)


def exact_fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def ceil_div(numerator, denominator):
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def lcm(*values):
    result = 1
    for value in values:
        result = result * int(value) // math.gcd(result, int(value))
    return result


def mask_fraction(mask):
    return exact_fraction(Fraction(sum(mask), len(mask)))


def validate_mask(mask, name):
    require(type(mask) is list and mask, "empty mask: {}".format(name))
    require(all(type(bit) is int and bit in (0, 1) for bit in mask),
            "mask is not exact 0/1: {}".format(name))
    require(sum(mask) > 0, "mask has no service slot: {}".format(name))


def load_trace(trace_path, boundary):
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    t10 = [row for row in rows if row["kind"] == "atlif" and
           int(row["temporal_steps"] or 0) == boundary["temporal_steps"]]
    require(len(rows) == 1840 and len(t10) == 450,
            "execution/T10 record population drift")

    sample_maps = []
    sample_keys = []
    for sample_id in range(10):
        sample = sorted(
            [row for row in t10 if int(row["sample_id"]) == sample_id],
            key=lambda row: int(row["call_index"]))
        require(len(sample) == 45, "T10 context count drift")
        mapping = []
        for ordinal, row in enumerate(sample):
            denominator = (boundary["temporal_steps"] *
                           boundary["lanes_per_tile"])
            elements = int(row["output_elements"])
            require(elements % denominator == 0,
                    "nonintegral T10 tile population")
            tiles = elements // denominator
            require(0 < tiles < (1 << 24),
                    "tile ordinal collides with context tag")
            mapping.append({
                "ordinal": ordinal,
                "call_index": int(row["call_index"]),
                "name": row["name"],
                "output_elements": elements,
                "tiles": tiles,
                "first_tag": ordinal << 24,
                "last_tag": (ordinal << 24) + tiles - 1,
            })
        sample_maps.append(mapping)
        sample_keys.append(sample[0]["sample_key"])

    reference = [(row["name"], row["tiles"]) for row in sample_maps[0]]
    require(all([(row["name"], row["tiles"]) for row in mapping] == reference
                for mapping in sample_maps),
            "ordered T10 context map differs across samples")
    contexts = sample_maps[0]
    require(all(contexts[index]["last_tag"] < contexts[index + 1]["first_tag"]
                for index in range(len(contexts) - 1)),
            "cross-context tag intervals overlap")
    total_tiles = sum(row["tiles"] for row in contexts)
    total_beats = total_tiles * boundary["result_beats_per_tile"]
    output_scalars = sum(row["output_elements"] for row in contexts)
    maximum_tag = max(row["last_tag"] for row in contexts)
    require(total_tiles == 7318350 and total_beats == 36591750 and
            output_scalars == 1170936000 and maximum_tag == 738658303,
            "frozen trace population drift")
    return {
        "execution_records": len(rows),
        "t10_records": len(t10),
        "samples": 10,
        "sample_keys": sample_keys,
        "ordered_contexts_per_inference": len(contexts),
        "ordered_context_map_identical_across_samples": True,
        "factor_tiles_per_inference": total_tiles,
        "five_beat_tile_results_per_inference": total_tiles,
        "ordered_result_beats_per_inference": total_beats,
        "output_scalars_per_inference": output_scalars,
        "raw_input_beats_per_inference":
            total_tiles * boundary["raw_input_beats_per_tile"],
        "tag_encoding": "(context_ordinal << 24) | tile_ordinal",
        "actual_maximum_tag": maximum_tag,
        "tag_intervals_disjoint": True,
        "context_map": contexts,
    }


def metric_snapshot(state):
    keys = (
        "tiles_loaded", "stage1_started", "stage1_done",
        "stage2_started", "stage2_done", "result_pushes",
        "result_departures", "raw_accepts", "producer_stalls",
        "result_blocked_cycles", "ingress_mask_wait_cycles",
        "ingress_capacity_wait_cycles", "stage1_active_cycles",
        "stage2_active_cycles",
    )
    return {key: state[key] for key in keys}


def control_key(state, cycle, mask_period):
    return (
        cycle % mask_period,
        state["input_beat"],
        state["raw_ready"],
        state["stage1_remaining"],
        state["intermediate_waiting"],
        state["intermediate_reserved"],
        state["stage2_remaining"],
        state["product_valid"],
        state["fifo_count"],
    )


def safe_fast_forward_repetitions(state, previous, tile_count):
    guards = {
        "tiles_loaded": (tile_count, 48),
        "stage1_done": (tile_count, 48),
        "result_pushes": (tile_count * 5, 240),
        "result_departures": (tile_count * 5, 240),
    }
    if state["architecture"] == "rank3":
        guards["stage2_done"] = (tile_count, 48)
    repetitions = []
    for key, (target, guard) in guards.items():
        delta = state[key] - previous["metrics"][key]
        if delta > 0:
            repetitions.append((target - guard - state[key]) // delta)
        else:
            require(state[key] <= target, "counter overflow: {}".format(key))
    if not repetitions:
        return 0
    return max(0, min(repetitions))


def simulate_context(architecture, tile_count, start_cycle, masks, common,
                     architecture_spec):
    """Cycle-accurate deterministic control model with periodic-state skipping.

    The FIFO is registered: a push into an empty FIFO cannot depart in the same
    cycle.  A full FIFO may push when a simultaneous pop supplies credit.  Raw
    banks remain owned through stage1 (or the dense Fixed issue schedule), and
    rank-3 intermediate banks are reserved before stage1 starts and released
    only when stage2 emits its fifth beat.
    """
    config_mask, ingress_mask, result_mask = masks
    for name, mask in zip(("config", "ingress", "result"), masks):
        validate_mask(mask, name)
    mask_period = lcm(len(config_mask), len(ingress_mask), len(result_mask))
    cycle = int(start_cycle)
    config_beats = ceil_div(
        architecture_spec["configuration_payload_bits"],
        common["configuration_bits_per_beat"])
    config_remaining = config_beats
    config_wait_cycles = 0
    while config_remaining:
        if config_mask[cycle % len(config_mask)]:
            config_remaining -= 1
        else:
            config_wait_cycles += 1
        cycle += 1
    config_end_cycle = cycle

    state = {
        "architecture": architecture,
        "input_beat": 0,
        "raw_ready": 0,
        "stage1_remaining": 0,
        "intermediate_waiting": 0,
        "intermediate_reserved": 0,
        "stage2_remaining": 0,
        "product_valid": 0,
        "fifo_count": 0,
        "fifo_peak": 0,
        "tiles_loaded": 0,
        "stage1_started": 0,
        "stage1_done": 0,
        "stage2_started": 0,
        "stage2_done": 0,
        "result_pushes": 0,
        "result_departures": 0,
        "raw_accepts": 0,
        "producer_stalls": 0,
        "result_blocked_cycles": 0,
        "ingress_mask_wait_cycles": 0,
        "ingress_capacity_wait_cycles": 0,
        "stage1_active_cycles": 0,
        "stage2_active_cycles": 0,
    }
    seen = {}
    fast_forward_cycles = 0
    explicit_cycles = 0
    total_result_beats = tile_count * common["result_beats_per_tile"]

    while True:
        complete = (
            state["tiles_loaded"] == tile_count and
            state["stage1_done"] == tile_count and
            (architecture == "fixed" or
             state["stage2_done"] == tile_count) and
            state["result_departures"] == total_result_beats and
            state["result_pushes"] == total_result_beats and
            state["input_beat"] == 0 and state["raw_ready"] == 0 and
            state["stage1_remaining"] == 0 and
            state["intermediate_waiting"] == 0 and
            state["intermediate_reserved"] == 0 and
            state["stage2_remaining"] == 0 and
            state["product_valid"] == 0 and
            state["fifo_count"] == 0)
        if complete:
            release_cycle = cycle
            cycle += common["release_cycles_per_context"]
            break

        far_from_tail = (
            state["tiles_loaded"] < tile_count - 48 and
            state["stage1_done"] < tile_count - 48 and
            state["result_departures"] < total_result_beats - 240)
        if architecture == "rank3":
            far_from_tail = (far_from_tail and
                             state["stage2_done"] < tile_count - 48)
        if far_from_tail:
            key = control_key(state, cycle, mask_period)
            if key in seen:
                previous = seen[key]
                cycle_delta = cycle - previous["cycle"]
                require(cycle_delta > 0 and cycle_delta % mask_period == 0,
                        "illegal periodic-state recurrence")
                repetitions = safe_fast_forward_repetitions(
                    state, previous, tile_count)
                if repetitions > 0:
                    current_metrics = metric_snapshot(state)
                    for metric, value in current_metrics.items():
                        delta = value - previous["metrics"][metric]
                        require(delta >= 0, "nonmonotonic metric")
                        state[metric] += delta * repetitions
                    skipped = cycle_delta * repetitions
                    cycle += skipped
                    fast_forward_cycles += skipped
                    seen.clear()
                    continue
            else:
                seen[key] = {
                    "cycle": cycle,
                    "metrics": metric_snapshot(state),
                }

        result_ready = bool(result_mask[cycle % len(result_mask)])
        result_fire = state["fifo_count"] > 0 and result_ready
        if state["fifo_count"] > 0 and not result_ready:
            state["result_blocked_cycles"] += 1

        # A waiting tile may start only from state owned at the beginning of the
        # cycle.  A fifth input beat or stage1 result cannot bypass registers.
        if architecture == "fixed":
            if state["stage1_remaining"] == 0 and state["raw_ready"] > 0:
                state["raw_ready"] -= 1
                state["stage1_remaining"] = architecture_spec[
                    "issue_cycles_per_tile"]
                state["stage1_started"] += 1
        else:
            if (state["stage2_remaining"] == 0 and
                    state["intermediate_waiting"] > 0):
                state["intermediate_waiting"] -= 1
                state["stage2_remaining"] = architecture_spec[
                    "stage2_cycles_per_tile"]
                state["stage2_started"] += 1
            if (state["stage1_remaining"] == 0 and state["raw_ready"] > 0 and
                    state["intermediate_reserved"] <
                    architecture_spec["intermediate_banks"]):
                state["raw_ready"] -= 1
                state["stage1_remaining"] = architecture_spec[
                    "stage1_cycles_per_tile"]
                state["intermediate_reserved"] += 1
                state["stage1_started"] += 1

        # Input capacity includes the partially filled bank, the ready queue,
        # and the raw bank owned by an active dense/stage1 issue schedule.
        raw_slots_used = (
            (1 if state["input_beat"] else 0) + state["raw_ready"] +
            (1 if state["stage1_remaining"] else 0))
        require(0 <= raw_slots_used <= common["raw_input_banks"],
                "raw bank occupancy overflow")
        ingress_available = bool(
            ingress_mask[cycle % len(ingress_mask)])
        if state["tiles_loaded"] < tile_count:
            # Once beat0 reserves a bank, beats1..4 continue filling that same
            # bank and do not require another free-bank credit.
            can_accept_input = (state["input_beat"] > 0 or
                                raw_slots_used < common["raw_input_banks"])
            if can_accept_input:
                if ingress_available:
                    state["raw_accepts"] += 1
                    state["input_beat"] += 1
                    if state["input_beat"] == common["raw_input_beats_per_tile"]:
                        state["input_beat"] = 0
                        state["raw_ready"] += 1
                        state["tiles_loaded"] += 1
                else:
                    state["ingress_mask_wait_cycles"] += 1
            elif ingress_available:
                state["ingress_capacity_wait_cycles"] += 1

        producer_wants = False
        if architecture == "fixed" and state["stage1_remaining"]:
            state["stage1_active_cycles"] += 1
            if state["stage1_remaining"] <= common["result_beats_per_tile"]:
                producer_wants = True
            else:
                state["stage1_remaining"] -= 1
        elif architecture == "rank3":
            if state["stage1_remaining"]:
                state["stage1_active_cycles"] += 1
                state["stage1_remaining"] -= 1
                if state["stage1_remaining"] == 0:
                    state["stage1_done"] += 1
                    state["intermediate_waiting"] += 1
            if state["stage2_remaining"]:
                state["stage2_active_cycles"] += 1
                producer_wants = True

        fifo_credit = (state["fifo_count"] <
                       common["result_fifo_depth_beats"] or result_fire)
        if architecture == "fixed":
            producer_push = producer_wants and fifo_credit
            if producer_wants and not fifo_credit:
                state["producer_stalls"] += 1
            if producer_push:
                state["result_pushes"] += 1
                state["stage1_remaining"] -= 1
                if state["stage1_remaining"] == 0:
                    state["stage1_done"] += 1
        else:
            # Frozen M37 has one elastic product register between issue and the
            # result FIFO.  The current product may move to the FIFO while a new
            # stage2 beat replaces it in the same cycle.
            producer_push = bool(state["product_valid"] and fifo_credit)
            product_stage_ready = bool(
                not state["product_valid"] or fifo_credit)
            stage2_issue = bool(producer_wants and product_stage_ready)
            if producer_wants and not product_stage_ready:
                state["producer_stalls"] += 1
            if producer_push:
                state["result_pushes"] += 1
            if stage2_issue:
                state["stage2_remaining"] -= 1
                if state["stage2_remaining"] == 0:
                    state["stage2_done"] += 1
                    state["intermediate_reserved"] -= 1
            state["product_valid"] = int(
                stage2_issue or
                (state["product_valid"] and not producer_push))

        if result_fire:
            state["result_departures"] += 1
            state["fifo_count"] -= 1
        if producer_push:
            state["fifo_count"] += 1
        state["fifo_peak"] = max(state["fifo_peak"], state["fifo_count"])
        require(0 <= state["fifo_count"] <=
                common["result_fifo_depth_beats"], "result FIFO overflow")
        require(0 <= state["intermediate_waiting"] <=
                state["intermediate_reserved"] <=
                (architecture_spec.get("intermediate_banks", 0)
                 if architecture == "rank3" else 0),
                "intermediate bank ownership overflow")
        require(state["product_valid"] in (0, 1),
                "product register validity drift")
        require(state["result_departures"] <= state["result_pushes"] <=
                total_result_beats, "result beat accounting overflow")
        cycle += 1
        explicit_cycles += 1
        require(explicit_cycles < 5000000,
                "cycle recurrence failed to accelerate")

    require(state["raw_accepts"] ==
            tile_count * common["raw_input_beats_per_tile"],
            "raw input beat loss/duplication")
    require(state["result_pushes"] == total_result_beats and
            state["result_departures"] == total_result_beats,
            "result beat loss/duplication")
    require(state["stage1_started"] == tile_count and
            state["stage1_done"] == tile_count,
            "stage1/fixed tile loss/duplication")
    if architecture == "rank3":
        require(state["stage2_started"] == tile_count and
                state["stage2_done"] == tile_count,
                "rank3 stage2 tile loss/duplication")
    return cycle, {
        "cycles": cycle - start_cycle,
        "configuration_beats": config_beats,
        "configuration_cycles": config_end_cycle - start_cycle,
        "configuration_wait_cycles": config_wait_cycles,
        "run_and_drain_cycles": release_cycle - config_end_cycle,
        "release_cycles": common["release_cycles_per_context"],
        "raw_input_beats": state["raw_accepts"],
        "result_beats": state["result_departures"],
        "producer_stall_cycles": state["producer_stalls"],
        "result_blocked_cycles": state["result_blocked_cycles"],
        "ingress_mask_wait_cycles": state["ingress_mask_wait_cycles"],
        "ingress_capacity_wait_cycles":
            state["ingress_capacity_wait_cycles"],
        "stage1_active_cycles": state["stage1_active_cycles"],
        "stage2_active_cycles": state["stage2_active_cycles"],
        "maximum_result_fifo_occupancy": state["fifo_peak"],
        "fast_forwarded_cycles": fast_forward_cycles,
        "explicitly_simulated_cycles": explicit_cycles,
        "drained_before_release": True,
    }


def simulate_inference(architecture, contexts, masks, common,
                       architecture_spec):
    cycle = 0
    rows = []
    totals = {
        "configuration_beats": 0,
        "configuration_cycles": 0,
        "configuration_wait_cycles": 0,
        "run_and_drain_cycles": 0,
        "release_cycles": 0,
        "raw_input_beats": 0,
        "result_beats": 0,
        "producer_stall_cycles": 0,
        "result_blocked_cycles": 0,
        "ingress_mask_wait_cycles": 0,
        "ingress_capacity_wait_cycles": 0,
        "stage1_active_cycles": 0,
        "stage2_active_cycles": 0,
        "fast_forwarded_cycles": 0,
        "explicitly_simulated_cycles": 0,
    }
    fifo_peak = 0
    for context in contexts:
        context_start = cycle
        cycle, row = simulate_context(
            architecture, context["tiles"], cycle, masks, common,
            architecture_spec)
        row.update({
            "ordinal": context["ordinal"],
            "name": context["name"],
            "tiles": context["tiles"],
            "start_cycle": context_start,
            "next_context_cycle": cycle,
        })
        rows.append(row)
        for key in totals:
            totals[key] += row[key]
        fifo_peak = max(fifo_peak, row["maximum_result_fifo_occupancy"])
    require(all(rows[index]["next_context_cycle"] ==
                rows[index + 1]["start_cycle"]
                for index in range(len(rows) - 1)),
            "cross-context execution overlap")
    require(all(row["drained_before_release"] for row in rows),
            "context released before drain")
    totals.update({
        "module_cycles": cycle,
        "contexts": len(contexts),
        "context_cycle_minimum": min(row["cycles"] for row in rows),
        "context_cycle_maximum": max(row["cycles"] for row in rows),
        "maximum_result_fifo_occupancy": fifo_peak,
        "cross_context_execution_overlap": False,
        "all_contexts_drained_before_release": True,
    })
    return totals


def m258_correction_overlay(m258, m261):
    corrected_profiles = []
    require(len(m258["cycle_sensitivity"]) ==
            len(m261["independent_cycle_recompute"]),
            "M258/M261 profile count drift")
    for original, audited in zip(m258["cycle_sensitivity"],
                                 m261["independent_cycle_recompute"]):
        require(original["profile"] == audited["profile"] and
                original["serial_cycles"] == audited["serial_cycles"] and
                original["candidate_cycles"] == audited["candidate_cycles"],
                "M258 core cycle table differs from M261")
        corrected_profiles.append({
            "profile": original["profile"],
            "serial_cycles_unchanged": original["serial_cycles"],
            "candidate_cycles_unchanged": original["candidate_cycles"],
            "module_speedup_unchanged": original["module_speedup"],
            "revoked_field": {
                "name": "candidate_producer_stall_lower_bound",
                "value": original["candidate_producer_stall_lower_bound"],
            },
            "replacement_field": {
                "name": "candidate_exact_registered_fifo_producer_stalls",
                "value": audited[
                    "candidate_exact_registered_fifo_producer_stalls"],
            },
            "revoked_minus_replacement": original[
                "candidate_producer_stall_lower_bound"] - audited[
                    "candidate_exact_registered_fifo_producer_stalls"],
        })
    population = m261["trace_population"]
    require(population["actual_maximum_tag"] == 738658303 and
            population["producer_reported_maximum_tag"] == 739119103,
            "M261 maximum-tag correction drift")
    return {
        "schema": "m265_m258_correction_overlay_v1",
        "status": "PASS_CORRECTION_OVERLAY_M258_CORE_CYCLES_UNCHANGED",
        "supersedes_fields_only": [
            "population.maximum_tag",
            "cycle_sensitivity[*].candidate_producer_stall_lower_bound",
            "README five-beat result population wording",
            "admission.configuration_and_result_backpressure_sensitivity",
        ],
        "maximum_tag": {
            "revoked_conservative_composite":
                population["producer_reported_maximum_tag"],
            "actual_maximum_over_context_intervals":
                population["actual_maximum_tag"],
            "difference": population[
                "producer_reported_minus_actual_maximum_tag"],
            "all_context_tile_counts_less_than_2pow24": True,
            "tag_intervals_disjoint": True,
        },
        "producer_stall_correction": corrected_profiles,
        "readme_wording": {
            "revoked": "36,591,750 five-beat results",
            "replacement": (
                "7,318,350 five-beat tile results, comprising 36,591,750 "
                "ordered result beats"),
        },
        "admission_correction": {
            "revoked": "configuration_and_result_backpressure_sensitivity",
            "replacement": "fixed configuration/release barrier plus result-ready sensitivity",
            "reason": "M258 did not sweep configuration bandwidth or ready",
        },
        "unchanged": {
            "core_serial_candidate_cycle_table_exact": True,
            "core_speedup_table_exact": True,
            "isolated_atlif_only": True,
            "system_speedup": False,
            "headline": False,
        },
    }


def write_csv(path, rows):
    fields = [
        "scenario", "config_mask", "ingress_mask", "result_mask",
        "fixed_module_cycles", "rank3_module_cycles", "module_speedup",
        "fixed_config_beats", "rank3_config_beats",
        "fixed_producer_stall_cycles", "rank3_producer_stall_cycles",
        "fixed_ingress_capacity_wait_cycles",
        "rank3_ingress_capacity_wait_cycles", "fixed_fifo_peak",
        "rank3_fifo_peak", "system_speedup_admitted", "headline_admitted",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "scenario": row["scenario"],
                "config_mask": row["masks"]["config"],
                "ingress_mask": row["masks"]["ingress"],
                "result_mask": row["masks"]["result"],
                "fixed_module_cycles": row["fixed"]["module_cycles"],
                "rank3_module_cycles": row["rank3"]["module_cycles"],
                "module_speedup": "{:.12f}".format(row["module_speedup"]),
                "fixed_config_beats": row["fixed"]["configuration_beats"],
                "rank3_config_beats": row["rank3"]["configuration_beats"],
                "fixed_producer_stall_cycles":
                    row["fixed"]["producer_stall_cycles"],
                "rank3_producer_stall_cycles":
                    row["rank3"]["producer_stall_cycles"],
                "fixed_ingress_capacity_wait_cycles":
                    row["fixed"]["ingress_capacity_wait_cycles"],
                "rank3_ingress_capacity_wait_cycles":
                    row["rank3"]["ingress_capacity_wait_cycles"],
                "fixed_fifo_peak":
                    row["fixed"]["maximum_result_fifo_occupancy"],
                "rank3_fifo_peak":
                    row["rank3"]["maximum_result_fifo_occupancy"],
                "system_speedup_admitted": "false",
                "headline_admitted": "false",
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing to overwrite output directory")
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m265_atlif_matched_boundary_trace_cycle_contract_v1",
            "contract schema drift")
    root = contract_path.parents[1]

    identities = {}
    loaded = {}
    resolved = {}
    for name, spec in contract["inputs"].items():
        require(set(spec) == {"path", "sha256"}, "input schema drift")
        path = root / spec["path"]
        require(path.is_file(), "missing input: {}".format(path))
        observed = sha256(path)
        require(observed == spec["sha256"], "SHA drift: {}".format(name))
        identities[name] = {"path": spec["path"], "sha256": observed}
        resolved[name] = path
        if path.suffix == ".json":
            loaded[name] = strict_json(path)

    manifest_text = resolved["trace_manifest"].read_text(encoding="utf-8")
    require(identities["execution_trace"]["sha256"] in manifest_text and
            "execution_trace.csv" in manifest_text,
            "raw execution trace is not manifest-bound")
    require(identities["protected_docs359"]["sha256"] ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "protected docs/359 drift")
    require(loaded["m261_review"]["score"] == 89 and
            loaded["m261_review"]["severity_counts"] ==
            {"P0": 0, "P1": 2, "P2": 5}, "M261 review drift")
    require(loaded["m37_independent_review"]["status"] ==
            "PASS_INDEPENDENT_HAMMER_STANDALONE_LOGIC_ONLY",
            "M37 independent review drift")
    require(loaded["m38_reachable_cycle_model"][
        "finite_reachable_state_audit"]["reachable_states"] == 669,
        "M38 reachable-state census drift")

    common = contract["common_boundary"]
    fixed_spec = contract["fixed_t10"]
    rank3_spec = contract["rank3_candidate"]
    population = load_trace(resolved["execution_trace"], common)
    total_tiles = population["factor_tiles_per_inference"]
    m25 = loaded["m25_fixed_cycle_model"]
    m26 = loaded["m26_rank3_arithmetic"]
    rank3_rows = [row for row in m26["candidates"]
                  if int(row["rank"]) == 3]
    require(len(rank3_rows) == 1, "M26 rank-3 candidate count drift")
    rank3 = rank3_rows[0]
    require(m25["atlif_exact96_arithmetic_lower_bound"][
        "service_cycles_lower_bound"] == 128020500,
        "M25 exact96 ATLIF lower bound drift")
    require(rank3["factor_tiles"] == total_tiles and
            rank3["factorized_macs"] == total_tiles * 960 and
            rank3["factorized_modules"] == 45,
            "M26 rank-3 population/arithmetic drift")
    require(total_tiles * fixed_spec["dense_products_per_tile"] ==
            11709360000, "Fixed T10 dense product population drift")
    require(fixed_spec["issue_cycles_per_tile"] ==
            ceil_div(fixed_spec["dense_products_per_tile"],
                     fixed_spec["signed_int8_multiplier_slots"]),
            "Fixed tile-closed issue schedule drift")
    require(rank3_spec["stage1_products_per_tile"] == 480 and
            rank3_spec["stage2_products_per_tile"] == 480 and
            rank3_spec["stage1_cycles_per_tile"] == 5 and
            rank3_spec["stage2_cycles_per_tile"] == 5 and
            rank3_spec["stage2_product_pipeline_cycles"] == 1 and
            rank3_spec["stage2_product_register_entries"] == 1,
            "rank-3 complete stage schedule drift")

    m31_text = resolved["m31_serial_rank3_rtl"].read_text(encoding="utf-8")
    m37_text = resolved["m37_stage2_rtl"].read_text(encoding="utf-8")
    require("input_port0" in m31_text and "input_port1" in m31_text and
            "T10_STAGE1" in m31_text and "T10_STAGE2" in m31_text,
            "M31 raw/stage interface drift")
    require("input_intermediate" in m37_text and
            "config_term_valid" in m37_text and
            "config_term_shift" in m37_text,
            "M37 stage2 sidecar interface drift")

    correction = m258_correction_overlay(
        loaded["m258_result"], loaded["m261_recompute"])
    masks_by_name = contract["periodic_masks"]
    for name, mask in masks_by_name.items():
        validate_mask(mask, name)
    sweep = []
    for scenario in contract["pressure_scenarios"]:
        masks = (
            masks_by_name[scenario["config"]],
            masks_by_name[scenario["ingress"]],
            masks_by_name[scenario["result"]],
        )
        fixed = simulate_inference(
            "fixed", population["context_map"], masks, common, fixed_spec)
        candidate = simulate_inference(
            "rank3", population["context_map"], masks, common, rank3_spec)
        require(fixed["raw_input_beats"] == candidate["raw_input_beats"] ==
                population["raw_input_beats_per_inference"],
                "matched raw ingress beat accounting drift")
        require(fixed["result_beats"] == candidate["result_beats"] ==
                population["ordered_result_beats_per_inference"],
                "matched result beat accounting drift")
        speedup = Fraction(fixed["module_cycles"],
                           candidate["module_cycles"])
        sweep.append({
            "scenario": scenario["name"],
            "masks": {
                "config": scenario["config"],
                "ingress": scenario["ingress"],
                "result": scenario["result"],
                "config_ready_fraction_exact": mask_fraction(masks[0]),
                "ingress_ready_fraction_exact": mask_fraction(masks[1]),
                "result_ready_fraction_exact": mask_fraction(masks[2]),
            },
            "fixed": fixed,
            "rank3": candidate,
            "module_speedup_exact": exact_fraction(speedup),
            "module_speedup": float(speedup),
            "system_speedup_admitted": False,
            "headline_admitted": False,
        })

    ideal = next(row for row in sweep if row["scenario"] == "IDEAL")
    context_count = population["ordered_contexts_per_inference"]
    expected_ideal_fixed = (
        total_tiles * fixed_spec["issue_cycles_per_tile"] +
        context_count * 12)
    expected_ideal_rank3 = (
        total_tiles * rank3_spec["stage2_cycles_per_tile"] +
        context_count * 19)
    require(ideal["fixed"]["module_cycles"] == expected_ideal_fixed,
            "ideal Fixed closed-form reconciliation failed")
    require(ideal["rank3"]["module_cycles"] == expected_ideal_rank3,
            "ideal rank3 closed-form reconciliation failed")
    t2_exact96_cycles = rank3["dense_fallback_macs"] // 96
    require(rank3["dense_fallback_macs"] % 96 == 0 and
            t2_exact96_cycles == 6048000,
            "T2 exact96 arithmetic population drift")
    fixed_t10_arithmetic_lower_bound_cycles = (
        m25["atlif_exact96_arithmetic_lower_bound"][
            "service_cycles_lower_bound"] - t2_exact96_cycles)
    rank3_steady_state_core_cycles = total_tiles * 5
    tile_closed_fixed_core_cycles = total_tiles * 17
    require(fixed_t10_arithmetic_lower_bound_cycles == 121972500 and
            rank3_steady_state_core_cycles == 36591750 and
            tile_closed_fixed_core_cycles == 124411950,
            "steady/core fairness crosscheck drift")
    arithmetic_crosscheck_speedup = Fraction(
        fixed_t10_arithmetic_lower_bound_cycles,
        rank3_steady_state_core_cycles)
    tile_rounding_factor = Fraction(
        tile_closed_fixed_core_cycles,
        fixed_t10_arithmetic_lower_bound_cycles)
    result = {
        "schema": "m265_atlif_matched_boundary_trace_cycle_v1",
        "status": "PASS_MATCHED_RAW_CONFIG_RESULT_TRACE_EXECUTABLE_ANALYTICAL_MODULE_CYCLES",
        "identity": {
            "contract": {"path": str(contract_path),
                         "sha256": sha256(contract_path)},
            "inputs": identities,
            "analyzer": {"path": str(Path(__file__).resolve()),
                         "sha256": sha256(Path(__file__).resolve())},
        },
        "population": population,
        "matched_boundary": {
            "common": common,
            "fixed": fixed_spec,
            "rank3_candidate": rank3_spec,
            "same_raw_input_boundary": True,
            "same_configuration_bus_and_ready": True,
            "same_result_sink_and_ready": True,
            "candidate_includes_stage1_and_stage2": True,
            "m37_sidecar_alone_counted_as_candidate": False,
            "contexts_drain_before_release": True,
            "cross_context_execution": False,
        },
        "pressure_sweep": sweep,
        "fairness_crosschecks": {
            "ideal_closed_form_reconciliation": {
                "fixed_formula": "17*tiles + 12*contexts",
                "rank3_formula": "5*tiles + 19*contexts",
                "fixed_cycles": expected_ideal_fixed,
                "rank3_cycles": expected_ideal_rank3,
                "passes": True,
            },
            "m25_exact96_t10_arithmetic_lower_bound_cycles":
                fixed_t10_arithmetic_lower_bound_cycles,
            "tile_closed_fixed_core_cycles": tile_closed_fixed_core_cycles,
            "tile_closed_rounding_over_m25_arithmetic_lower_bound_exact":
                exact_fraction(tile_rounding_factor),
            "tile_closed_rounding_over_m25_arithmetic_lower_bound":
                float(tile_rounding_factor),
            "rank3_steady_state_core_cycles": rank3_steady_state_core_cycles,
            "m25_lower_bound_to_rank3_steady_core_speedup_exact":
                exact_fraction(arithmetic_crosscheck_speedup),
            "m25_lower_bound_to_rank3_steady_core_speedup":
                float(arithmetic_crosscheck_speedup),
            "interpretation": (
                "The 3.40x ideal trace module ratio uses a legal tile-closed "
                "17-cycle Fixed issue schedule. M25's cross-tile packed exact96 "
                "arithmetic lower bound gives a separate 3.333x steady-core "
                "crosscheck. Neither is an area-matched RTL or system claim."),
        },
        "decision": {
            "direction": "KEEP_COMPLETE_RANK3_DECOUPLED_ATLIF_FOR_RTL_INTEGRATION_GATE",
            "ideal_module_cycles": {
                "fixed": ideal["fixed"]["module_cycles"],
                "rank3": ideal["rank3"]["module_cycles"],
            },
            "ideal_isolated_module_speedup": ideal["module_speedup"],
            "bottleneck_rule": (
                "rank3 tends toward one five-beat tile per five cycles only "
                "when raw ingress and the 48-bit result sink both sustain it; "
                "the full pressure table is mandatory"),
            "next_gate": (
                "integrate stage1 plus M37-class CSD stage2 in one RTL top, "
                "then run VCS and area-matched DC before throughput/area claims"),
        },
        "claim_boundary": contract["claim_boundary"],
        "claim_boundary_text": (
            "Exact deterministic cycles for an isolated trace-projected T10 "
            "ATLIF analytical control model over the frozen 45-context H67 "
            "population. Fixed and rank3 start at the same raw 5x256-bit tile "
            "ingress, share a 256-bit config-bandwidth/ready contract, and end "
            "at the same registered 48-bit result sink. Candidate stage1 and "
            "stage2 are both charged. This is not integrated RTL, area matched, "
            "trained-rank3 accuracy, energy, system speedup, paper PPA or a "
            "headline claim."),
    }

    output_dir.mkdir(parents=True)
    correction_path = output_dir / "m258_correction_overlay_r1.json"
    correction_path.write_text(
        json.dumps(correction, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    result_path = output_dir / "m265_atlif_matched_boundary_trace_cycle_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    write_csv(output_dir / "m265_pressure_sweep.csv", sweep)
    readme = """# M265 matched-boundary ATLIF module cycles

M265 first overlays four corrections on frozen M258 without changing M258's
core cycle table: the actual maximum tag is 738,658,303; the exact registered-
FIFO producer-stall counts replace the invalid lower-bound wording; M258 only
had a fixed config/release barrier, not config sensitivity; and the population
is 7,318,350 five-beat tile results comprising 36,591,750 ordered result beats.

The new model compares a tile-closed exact-96 Fixed T10 schedule against a
complete rank-3 candidate.  Both consume the same five raw 256-bit beats per
tile, use the same 256-bit config bus and ready trace, and emit the same five
48-bit registered-FIFO result beats.  Rank-3 explicitly executes five-cycle
stage1 and five-cycle CSD stage2 on distinct resources; M37 alone is never
counted as the candidate.  Every one of 45 contexts drains before release.

The JSON and CSV contain the ideal point plus isolated result, ingress, config,
and joint periodic-pressure sweeps.  Ideal matched-boundary cycles are
124,412,490 versus 36,592,605 (`3.399935x`).  The independent M25 exact-96
cross-tile arithmetic lower-bound comparison is `3.333333x`; the difference is
the explicit 17-cycle tile closure.  All speedups are isolated ATLIF analytical
module-cycle ratios.  They are not system speedup, throughput/area, trained
rank-3 accuracy, energy, paper PPA, or headline claims.  Integrated RTL and an
area-matched Synopsys comparison remain future gates.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print("M265_PASS ideal_fixed={} ideal_rank3={} ideal_module={:.6f}x".format(
        ideal["fixed"]["module_cycles"],
        ideal["rank3"]["module_cycles"], ideal["module_speedup"]))


if __name__ == "__main__":
    main()
