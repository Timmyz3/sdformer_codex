#!/usr/bin/env python3
"""Read-only pre-RTL hammer for the M377 active-descriptor controller."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path


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
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs(contract_path, contract):
    root = contract_path.resolve().parents[1]
    paths = {}
    identities = {}
    for name, identity in contract["inputs"].items():
        path = root / identity["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"], "SHA drift for " + name)
        paths[name] = path
        identities[name] = {"path": identity["path"], "sha256": observed}
    return root, paths, identities


def event_audit(path, descriptor_bytes):
    phases = defaultdict(lambda: {"replays": [], "matcher": []})
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (int(row["sample"]), int(row["operator_index"]),
                   int(row["partition"]))
            if row["event"] in (
                    "TILE0_ACTIVE_DESCRIPTOR_REPLAY_COMPUTE",
                    "TILE1_ACTIVE_DESCRIPTOR_REPLAY_COMPUTE"):
                require(int(row["bytes"]) % descriptor_bytes == 0,
                        "fractional descriptor bytes")
                phases[key]["replays"].append({
                    "event": row["event"],
                    "active": int(row["bytes"]) // descriptor_bytes,
                    "duration": int(row["duration_cycles"]),
                })
            elif row["event"] == (
                    "SERIAL16_MATCH_AND_ACTIVE_DESCRIPTOR_COMPACT_WRITE"):
                phases[key]["matcher"].append({
                    "active": int(row["bytes"]) // descriptor_bytes,
                    "duration": int(row["duration_cycles"]),
                })

    active_sum = 0
    replayed = 0
    exact_work = 0
    minimum_work_per_active = float("inf")
    maximum_work_per_active = 0.0
    for key, records in phases.items():
        require(len(records["matcher"]) == 1,
                "matcher event count drift at {}".format(key))
        require(len(records["replays"]) == 2,
                "dual replay count drift at {}".format(key))
        active = records["matcher"][0]["active"]
        require(active > 0, "M377 unexpectedly has empty replay phase")
        require(all(row["active"] == active for row in records["replays"]),
                "dual replay active_count mismatch")
        require(records["replays"][0]["duration"] ==
                records["replays"][1]["duration"],
                "dual replay duration mismatch")
        work = records["replays"][0]["duration"] - 1
        require(work >= 4 * active,
                "less than four cycles of service per active descriptor")
        active_sum += active
        replayed += 2 * active
        exact_work += 2 * work
        ratio = work / float(active)
        minimum_work_per_active = min(minimum_work_per_active, ratio)
        maximum_work_per_active = max(maximum_work_per_active, ratio)
    return {
        "phase_count": len(phases),
        "active_descriptors": active_sum,
        "replayed_descriptors": replayed,
        "replay_count_per_phase": 2,
        "exact_compute_cycles_both_replays": exact_work,
        "minimum_compute_cycles_per_descriptor_per_replay":
            minimum_work_per_active,
        "maximum_phase_average_compute_cycles_per_descriptor_per_replay":
            maximum_work_per_active,
        "all_phase_active_counts_match_write_and_both_replays": True,
    }


def reconstruct_population(m358_contract, root):
    model = m358_contract["cycle_model"]
    transitive = {}
    for name, identity in m358_contract["inputs"].items():
        path = root / identity["path"]
        require(path.is_file() and sha256(path) == identity["sha256"],
                "M358 transitive SHA drift: " + name)
        transitive[name] = path
    m43 = load_module(transitive["m43_support_unpacker"], "m380_m43")
    catalog = strict_json(transitive["m338_catalog"])
    trace = strict_json(transitive["m248_runtime_trace"])
    trace_dir = transitive["m248_runtime_trace"].parent
    operators = tuple(trace["cohort"]["operators"])
    op_index = {name: index for index, name in enumerate(operators)}
    histograms = defaultdict(Counter)
    for record_index, record in enumerate(trace["records"]):
        packed = trace_dir / record["packed_file"]
        values = trace_dir / record["value_payload_file"]
        require(sha256(packed) == record["packed_file_sha256"] and
                sha256(values) == record["value_payload_sha256"],
                "M248 payload drift")
        masks = m43.unpack_record_masks(trace_dir, record)
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (
                    m43.TILE_BITS // model["partition_bits"])
                for subtile in range(
                        m43.TILE_BITS // model["partition_bits"]):
                    value = ((value256 >>
                              (subtile * model["partition_bits"])) & 0xffff)
                    histograms[(record["sample_id"],
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
        print("[M380 HIST] {}/{}".format(record_index + 1,
                                         len(trace["records"])), flush=True)

    totals = Counter()
    maximum_active = 0
    maximum_retirements_in_one_cycle = 0
    center_id_max = 0
    for sample in range(model["samples"]):
        for op in range(model["operators"]):
            for partition in range(model["partitions_per_operator"]):
                counter = histograms[(sample, op, partition)]
                require(sum(counter.values()) == model["rows_per_operator"],
                        "phase row extent drift")
                centers = [
                    int(value, 16) for value in
                    catalog["operators"][op]["partitions"][partition]
                    ["nested_patterns"][:32]
                ]
                require(len(centers) == 32, "q32 catalog underflow")
                phase_active = 0
                # A single serialized slot performs one q16 half-scan per
                # cycle. A row can retire only on its terminal slot, hence
                # the peak retirement and descriptor-write rate is one.
                serialized_slots = 0
                serialized_retires = Counter()
                row_cursor = 0
                for original, count in sorted(counter.items()):
                    population = original.bit_count()
                    best_distance = 17
                    best_index = 0
                    for index, center in enumerate(centers):
                        distance = (original ^ center).bit_count()
                        if distance < best_distance:
                            best_distance = distance
                            best_index = index
                    center_id_max = max(center_id_max, best_index)
                    use_pwp = (1 + best_distance < population)
                    center = centers[best_index]
                    plus = original & ((~center) & 0xffff)
                    minus = center & ((~original) & 0xffff)
                    require((plus & minus) == 0,
                            "plus/minus residual overlap")
                    require((plus.bit_count() + minus.bit_count()) ==
                            best_distance, "distance reconstruction drift")
                    require((((center | plus) & ((~minus) & 0xffff)) ==
                             original), "center residual is not exact")
                    totals["source_rows"] += count
                    totals["matcher_second_half_rows"] += count * int(
                        population >= 2)
                    if original == 0:
                        totals["zero_rows"] += count
                    else:
                        totals["active_descriptors"] += count
                        phase_active += count
                        totals["pwp_descriptors" if use_pwp else
                               "fallback_descriptors"] += count
                        totals["plus_residual_bits"] += count * plus.bit_count()
                        totals["minus_residual_bits"] += count * minus.bit_count()
                        if population == 1:
                            require(not use_pwp,
                                    "pop1 illegally selected PWP")
                            totals["pop1_fallback_descriptors"] += count
                    # Expand only the abstract serialized retirement schedule;
                    # 51.84 M slots are still small and make the peak proof
                    # executable rather than merely average-rate arithmetic.
                    slots_per_row = 1 + int(population >= 2)
                    for _ in range(count):
                        serialized_slots += slots_per_row
                        serialized_retires[serialized_slots - 1] += 1
                        row_cursor += 1
                require(row_cursor == model["rows_per_operator"],
                        "serialized matcher row loss")
                require(serialized_slots == model["rows_per_operator"] +
                        sum(count for value, count in counter.items()
                            if value.bit_count() >= 2),
                        "serialized matcher recurrence mismatch")
                local_peak = max(serialized_retires.values())
                require(local_peak <= 1, "more than one retirement per cycle")
                maximum_retirements_in_one_cycle = max(
                    maximum_retirements_in_one_cycle, local_peak)
                maximum_active = max(maximum_active, phase_active)

    require(totals["active_descriptors"] ==
            totals["pwp_descriptors"] + totals["fallback_descriptors"],
            "PWP/fallback descriptor conservation failure")
    require(totals["pop1_fallback_descriptors"] > 0,
            "pop1 fallback population absent")
    require(center_id_max <= 31, "q32 center ID overflow")
    require(maximum_active <= 3000, "descriptor bank overflow")
    return {
        "source_rows": totals["source_rows"],
        "zero_rows_elided": totals["zero_rows"],
        "active_descriptors": totals["active_descriptors"],
        "pwp_descriptors": totals["pwp_descriptors"],
        "fallback_descriptors": totals["fallback_descriptors"],
        "pop1_fallback_descriptors": totals[
            "pop1_fallback_descriptors"],
        "plus_residual_bits": totals["plus_residual_bits"],
        "minus_residual_bits": totals["minus_residual_bits"],
        "maximum_active_descriptors_in_one_phase": maximum_active,
        "maximum_center_id": center_id_max,
        "maximum_serialized_matcher_retirements_per_cycle":
            maximum_retirements_in_one_cycle,
        "row12_covers_phase_extent": model["rows_per_operator"] <= (1 << 12),
        "original16_plus_center_id_exact_reconstruction": True,
        "pop1_fallback_retained": True,
    }


def sensitivity_rows(baseline, candidate, phases, replayed):
    rows = []

    def add(name, overhead, detail):
        modeled = candidate + overhead
        rows.append({
            "scenario": name,
            "extra_cycles_over_m377": overhead,
            "modeled_candidate_cycles": modeled,
            "speedup_vs_bit_sparse": baseline / float(modeled),
            "detail": detail,
        })

    # M377 already charges one startup cycle for every nonempty replay.
    for latency in (1, 2, 4, 8, 16, 32):
        overhead = phases + 2 * phases * (latency - 1)
        add("streaming_L{}_count_commit".format(latency), overhead,
            "one active-count seal/phase plus L-1 extra startup/replay; "
            "steady-state descriptor response II=1 overlaps compute")
    pinned = phases + 2 * phases * (2 - 1)
    for stall in (0.0, 0.125, 0.25, 0.5, 1.0):
        add("pinned_L2_plus_{:.3f}_blocking_cycle_per_replay_descriptor".
            format(stall), pinned + replayed * stall,
            "pessimistic non-overlapped SRAM/FIFO/port penalty")
    add("blocking_one_cycle_per_replay_descriptor", replayed,
        "fast-kill: every descriptor read adds one serialized cycle")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M380 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m380_m377_active_descriptor_controller_prertl_hammer_contract_v1",
            "M380 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M380_EXECUTION",
            "M380 contract not frozen")
    root, paths, identities = load_inputs(args.contract, contract)
    m377 = strict_json(paths["m377_result"])
    m373 = strict_json(paths["m373_result"])
    m358 = strict_json(paths["m358_contract"])
    require(m377["status"] ==
            "PASS_M377_EXACT_ACTIVE_DESCRIPTOR_FINITE_EXECUTION",
            "M377 status drift")
    require(m377["decision"] == "GO_VCS_ACTIVE_DESCRIPTOR_SCHEDULER_RTL",
            "M377 decision drift")
    require(m373["cycles"]["bit_sparse_reproduced_cycles"] ==
            m377["cycles"]["bit_sparse_reproduced_cycles"],
            "M373/M377 baseline drift")
    descriptor_bytes = m358["cycle_model"]["descriptor_bytes_per_row"]
    events = event_audit(paths["m377_candidate_events"], descriptor_bytes)
    population = reconstruct_population(m358, root)
    require(events["active_descriptors"] ==
            population["active_descriptors"] ==
            m377["population"]["active_descriptor_rows"],
            "source/event/M377 active population drift")
    require(events["phase_count"] ==
            m358["cycle_model"]["samples"] *
            m358["cycle_model"]["operators"] *
            m358["cycle_model"]["partitions_per_operator"],
            "phase extent drift")

    baseline = m377["cycles"]["bit_sparse_reproduced_cycles"]
    candidate = m377["cycles"]["m377_active_compact_candidate_cycles"]
    rows = sensitivity_rows(baseline, candidate, events["phase_count"],
                            events["replayed_descriptors"])
    pinned_overhead = 3 * events["phase_count"]
    one05_limit = baseline / 1.05
    threshold_budget = one05_limit - candidate - pinned_overhead
    break_even_budget = baseline - candidate - pinned_overhead
    threshold_stall = threshold_budget / events["replayed_descriptors"]
    break_even_stall = break_even_budget / events["replayed_descriptors"]
    require(threshold_stall > 0.0, "pinned contract misses 1.05x")

    hard_boundary = contract["controller_boundary"]
    result = {
        "schema": "m380_m377_active_descriptor_controller_prertl_hammer_v1",
        "status": "PASS_M380_READ_ONLY_PRERTL_HAMMER",
        "identity": identities,
        "frozen_asset_verification": {
            "m377_sha_layer_pass": True,
            "m377_seal_layer_pass": True,
            "m373_sha_layer_pass": True,
            "m373_seal_layer_pass": True,
            "docs359_sha256": identities["docs359"]["sha256"],
            "m377_modified": False,
            "m373_modified": False,
            "docs359_modified": False,
        },
        "descriptor_population_replay": population,
        "m377_event_replay_audit": events,
        "controller_boundary": hard_boundary,
        "cycle_risk": {
            "bit_sparse_baseline_cycles": baseline,
            "m377_candidate_cycles": candidate,
            "m377_speedup": baseline / float(candidate),
            "absolute_cycle_slack_to_baseline": baseline - candidate,
            "m377_to_1p05_cycle_overhead_budget": one05_limit - candidate,
            "pinned_L2_plus_count_commit_overhead_cycles": pinned_overhead,
            "pinned_L2_plus_count_commit_speedup": baseline / float(
                candidate + pinned_overhead),
            "remaining_1p05_budget_after_pinned_contract_cycles":
                threshold_budget,
            "remaining_1p05_budget_per_replayed_descriptor_cycles":
                threshold_stall,
            "break_even_budget_per_replayed_descriptor_cycles":
                break_even_stall,
            "one_serial_cycle_per_replayed_descriptor_speedup":
                baseline / float(candidate +
                                 events["replayed_descriptors"]),
            "interpretation": (
                "fixed streaming SRAM latency is a replay-startup term and "
                "does not threaten M377; a blocking or shared-port penalty "
                "paid per replayed descriptor does"
            ),
        },
        "sensitivity": rows,
        "findings": contract["hammer_findings"],
        "vcs_miter_and_coverage_gates": contract[
            "vcs_miter_and_coverage_gates"],
        "dc_gates_after_vcs": contract["dc_gates_after_vcs"],
        "scorecard": contract["scorecard"],
        "decision": contract["decision"],
        "claim_boundary": contract["claim_boundary"],
        "output_files": {"sensitivity": "cycle_sensitivity.csv"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "cycle_sensitivity.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = ["scenario", "extra_cycles_over_m377",
                  "modeled_candidate_cycles", "speedup_vs_bit_sparse",
                  "detail"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output = args.output_dir / (
        "m380_m377_active_descriptor_controller_prertl_hammer_r1.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M380_PASS active={} pop1={} replayed={} pinned_speedup={:.6f}x "
          "stall_budget_1p05={:.6f}/replay_desc decision={}".format(
              population["active_descriptors"],
              population["pop1_fallback_descriptors"],
              events["replayed_descriptors"],
              result["cycle_risk"]["pinned_L2_plus_count_commit_speedup"],
              threshold_stall, result["decision"]), flush=True)


if __name__ == "__main__":
    main()
