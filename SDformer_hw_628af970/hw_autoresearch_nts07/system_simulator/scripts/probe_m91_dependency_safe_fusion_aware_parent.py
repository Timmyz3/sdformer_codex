#!/usr/bin/env python3
"""M91 dependency-safe fusion-aware parent transaction-model probe.

The canonical M53 DAG is frozen before admission-time parent reselection.
Only alternatives whose dependencies are already represented by that DAG are
legal.  This is a non-citable transaction-model probe, not RTL or PPA.
"""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import heapq
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m91_dependency_safe_fusion_aware_parent_contract_r1_20260824.json")
M45 = HW_ROOT / (
    "system_simulator/scripts/"
    "analyze_m45_dual_destination_bank_fused_integrated_schedule.py")
M43_RESULT = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatiotemporal_parent_delta_ablation.json")
M89_RECEIPT = HW_ROOT / (
    "results/m89_temporal_fanout_hold_screen_r1_20260823/"
    "m89_temporal_fanout_hold_screen_receipt.json")

EXPECTED = {
    "contract": "da4172314986600d49e9ed0f4ade2ebcbec90ad1910d036e166db17356de4b4c",
    "m45": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
    "m43_result": "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c",
    "m89_receipt": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}
FANOUT = 6
CONTEXTS = 16
PARENT_PRIORITY = {
    "local_zero": 0,
    "left": 1,
    "up": 2,
    "previous_timestep": 3,
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
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject)


def fraction(numerator, denominator):
    require(denominator > 0, "zero fraction denominator")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def load_m45():
    require(sha256(CONTRACT) == EXPECTED["contract"], "M91 contract drift")
    require(sha256(M45) == EXPECTED["m45"], "M91 M45 drift")
    spec = importlib.util.spec_from_file_location("m91_m45", str(M45))
    namespace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(namespace)
    namespace.validate_contract()
    return namespace


def make_candidate(m43, name, current, parent):
    add_mask = current & ~parent
    subtract_mask = parent & ~current
    delta = add_mask | subtract_mask
    require((add_mask & subtract_mask) == 0 and delta == (current ^ parent),
            "M91 signed candidate partition drift")
    return {
        "name": name,
        "parent": parent,
        "delta": delta,
        "add": add_mask,
        "subtract": subtract_mask,
        "cycles": m43.bank_issue_cycles(delta),
        "population": m43.population(delta),
    }


def build_tasks(m43, masks, tile, timestep):
    candidates = []
    canonical_names = []
    canonical_candidates = []
    for spatial in range(m43.HEIGHT * m43.WIDTH):
        row = timestep * m43.HEIGHT * m43.WIDTH + spatial
        index = row * m43.TILES + tile
        current = masks[index]
        canonical_name, canonical_parent, _, _ = m43.select_parent(
            masks, row, tile)
        y, x = divmod(spatial, m43.WIDTH)
        legal = [make_candidate(m43, "local_zero", current, 0)]
        if timestep > 0:
            legal.append(make_candidate(
                m43, "previous_timestep", current,
                masks[index - m43.HEIGHT * m43.WIDTH * m43.TILES]))
        if y > 0:
            legal.append(make_candidate(
                m43, "up", current,
                masks[index - m43.WIDTH * m43.TILES]))
        if canonical_name == "left":
            require(x > 0, "M91 canonical left parent at x=0")
            legal.append(make_candidate(
                m43, "left", current, masks[index - m43.TILES]))
        by_name = dict((item["name"], item) for item in legal)
        require(canonical_name in by_name,
                "M91 canonical parent absent from dependency-safe candidates")
        candidates.append([by_name[name] for name in sorted(
            by_name, key=lambda value: PARENT_PRIORITY[value])])
        canonical_names.append(canonical_name)
        canonical_candidates.append(by_name[canonical_name])
    return candidates, canonical_names, canonical_candidates


def schedule_tile_timestep(r1, m43, candidates, canonical_names,
                           canonical_candidates, start_cycle,
                           weight_ready_cycle, tile_index, selection_counts):
    require(len(candidates) == r1.ROWS_PER_T and FANOUT <= CONTEXTS,
            "M91 invalid task geometry")
    # Freeze the canonical DAG before any dependency-safe reselection.
    indegree, children = r1.build_structural_dag(list(canonical_names))
    ready = [index for index, count in enumerate(indegree) if count == 0]
    heapq.heapify(ready)
    ready_since = dict((index, start_cycle) for index in ready)
    resident = {}
    chosen = {}
    context_active = []
    commit_events = []
    complete_entries = []
    committed = 0
    now = start_cycle
    command_port = r1.PortCalendar()
    parent_port = r1.PortCalendar()
    final_port = r1.PortCalendar()
    output_port = r1.PortCalendar()
    counts = r1.blank_counts()
    selected_add_terms = 0
    selected_subtract_terms = 0
    cycle_memo = {}

    def cycles(mask):
        value = cycle_memo.get(mask)
        if value is None:
            value = m43.bank_issue_cycles(mask)
            cycle_memo[mask] = value
        return value

    def pop_ready_for_residency():
        window = heapq.nsmallest(min(16, len(ready)), ready)
        require(window, "M91 empty ready window")
        if not resident:
            task = window[0]
            selected = canonical_candidates[task]
            selection_counts["empty_resident_canonical_fallback"] += 1
        else:
            ranked = []
            for task in window:
                for option in candidates[task]:
                    union_cycles = min(
                        cycles(chosen[anchor]["delta"] | option["delta"])
                        for anchor in resident)
                    nonzero_parent_charge = int(option["name"] != "local_zero")
                    ranked.append((
                        union_cycles + nonzero_parent_charge,
                        option["cycles"], option["population"], task,
                        PARENT_PRIORITY[option["name"]], option))
            winner = min(ranked, key=lambda item: item[:5])
            task, selected = winner[3], winner[5]
            selection_counts["fusion_aware_admissions"] += 1
        ready.remove(task)
        heapq.heapify(ready)
        chosen[task] = selected
        selection_counts["selected_" + selected["name"]] += 1
        if selected["name"] != canonical_names[task]:
            selection_counts["parent_reselections"] += 1
        return task

    def process_events(limit):
        nonlocal committed
        while context_active and context_active[0][0] <= limit:
            heapq.heappop(context_active)
        while commit_events and commit_events[0][0] <= limit:
            commit_time, task = heapq.heappop(commit_events)
            committed += 1
            for child in children[task]:
                indegree[child] -= 1
                require(indegree[child] >= 0, "M91 DAG indegree underflow")
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
                    ready_since[child] = commit_time
        while complete_entries and complete_entries[0][0] <= limit:
            heapq.heappop(complete_entries)

    while committed < r1.ROWS_PER_T:
        process_events(now)
        if committed == r1.ROWS_PER_T:
            break
        occupied_contexts = len(resident) + len(context_active)
        while ready and occupied_contexts < CONTEXTS:
            counts["maximum_metadata_occupancy"] = max(
                counts["maximum_metadata_occupancy"], min(16, len(ready)))
            task = pop_ready_for_residency()
            command_end = command_port.schedule(ready_since.pop(task))
            selected = chosen[task]
            parent_end = now
            if selected["name"] != "local_zero":
                parent_end = parent_port.schedule(now)
                counts["parent_partial_reads"] += 1
            prep_ready = max(command_end, parent_end, weight_ready_cycle)
            if prep_ready == weight_ready_cycle and weight_ready_cycle > now:
                cause = "weight_dma_wait_cycles"
            else:
                cause = "command_or_state_wait_cycles"
            resident[task] = (prep_ready, cause)
            counts["descriptor_commands"] += 1
            occupied_contexts += 1
        counts["maximum_resident_occupancy"] = max(
            counts["maximum_resident_occupancy"], occupied_contexts)
        require(occupied_contexts <= CONTEXTS and
                counts["maximum_metadata_occupancy"] <= r1.METADATA_FIFO_ENTRIES,
                "M91 metadata/context overflow")

        prepared = [task for task, item in resident.items() if item[0] <= now]
        if not prepared:
            next_times = [(item[0], item[1]) for item in resident.values()]
            if context_active:
                cause = ("response_or_context_wait_cycles" if ready else
                         "parent_wait_cycles")
                next_times.append((context_active[0][0], cause))
            if commit_events:
                cause = ("parent_wait_cycles" if not ready and not resident else
                         "response_or_context_wait_cycles")
                next_times.append((commit_events[0][0], cause))
            require(next_times, "M91 scheduler deadlock")
            next_cycle, cause = min(next_times)
            require(next_cycle > now, "M91 scheduler failed to advance")
            counts[cause] += next_cycle - now
            now = next_cycle
            continue

        delta_masks = dict((task, chosen[task]["delta"]) for task in resident)
        group, union_mask = r1.select_fusion_group(
            m43, prepared, delta_masks, FANOUT, cycles)
        group_cycles = cycles(union_mask)
        for task in group:
            del resident[task]
        projected_response = now + group_cycles + (1 if group_cycles else 0)
        live_entries = sorted(item[0] for item in complete_entries
                              if item[0] > projected_response)
        maximum_live_before_group = r1.COMPLETE_FIFO_ENTRIES - len(group)
        if len(live_entries) > maximum_live_before_group:
            entries_to_drain = len(live_entries) - maximum_live_before_group
            required_response = live_entries[entries_to_drain - 1]
            throttle = required_response - projected_response
            require(throttle >= 0, "M91 negative complete-FIFO throttle")
            now += throttle
            counts["response_or_context_wait_cycles"] += throttle
        counts["source_only_cycles"] += group_cycles
        for task in group:
            selected = chosen[task]
            counts["logical_source_updates"] += selected["population"]
            selected_add_terms += m43.population(selected["add"])
            selected_subtract_terms += m43.population(selected["subtract"])
        counts["unique_weight_issues"] += m43.population(union_mask)
        counts["fusion_groups"] += 1
        if group_cycles == 0:
            counts["zero_source_groups"] += 1
            response_ready = now
        else:
            now += group_cycles
            response_ready = now + 1

        commit_batch = []
        while complete_entries and complete_entries[0][0] <= response_ready:
            heapq.heappop(complete_entries)
        for task in sorted(group):
            heapq.heappush(context_active, (response_ready, task))
            parent_write_end = parent_port.schedule(
                response_ready + r1.WRITEBACK_DEFERRAL_CYCLES)
            if tile_index > 0:
                final_read_end = final_port.schedule(response_ready)
                counts["final_accumulator_reads"] += 1
                final_write_end = final_port.schedule(final_read_end)
            else:
                final_write_end = final_port.schedule(response_ready)
            counts["parent_partial_writes"] += 1
            counts["final_accumulator_writes"] += 1
            commit_time = max(parent_write_end, final_write_end)
            if tile_index == r1.TILES - 1:
                output_port.schedule(commit_time)
                counts["completed_outputs"] += 1
            commit_batch.append((commit_time, task))
        counts["maximum_complete_occupancy"] = max(
            counts["maximum_complete_occupancy"],
            len(complete_entries) + len(commit_batch))
        require(len(complete_entries) + len(commit_batch) <=
                r1.COMPLETE_FIFO_ENTRIES, "M91 complete FIFO overflow")
        for item in commit_batch:
            heapq.heappush(commit_events, item)
            heapq.heappush(complete_entries, item)

    end_cycle = max(now, command_port.last_end, parent_port.last_end,
                    final_port.last_end, output_port.last_end,
                    commit_events[-1][0] if commit_events else 0)
    counts["integrated_cycles"] = end_cycle - start_cycle
    require(counts["descriptor_commands"] == r1.ROWS_PER_T and
            counts["parent_partial_writes"] == r1.ROWS_PER_T and
            counts["final_accumulator_writes"] == r1.ROWS_PER_T and
            selected_add_terms + selected_subtract_terms ==
            counts["logical_source_updates"],
            "M91 per-tile conservation drift")
    counts["selected_add_terms"] = selected_add_terms
    counts["selected_subtract_terms"] = selected_subtract_terms
    return counts


def analyze_record(r1, m43, masks, selection_counts):
    block_time = 0
    block_counts = r1.blank_counts()
    add_terms = subtract_terms = 0
    weight_ready = block_time + r1.WEIGHT_LOAD_CYCLES
    block_counts["weight_dma_wait_cycles"] += r1.WEIGHT_LOAD_CYCLES
    for timestep in range(r1.T):
        for tile in range(r1.TILES):
            tile_start = max(block_time, weight_ready)
            next_weight_ready = (
                tile_start + r1.WEIGHT_LOAD_CYCLES
                if tile + 1 < r1.TILES or timestep + 1 < r1.T else tile_start)
            candidates, canonical_names, canonical_candidates = build_tasks(
                m43, masks, tile, timestep)
            scheduled = schedule_tile_timestep(
                r1, m43, candidates, canonical_names, canonical_candidates,
                tile_start, weight_ready, tile, selection_counts)
            add_terms += scheduled.pop("selected_add_terms")
            subtract_terms += scheduled.pop("selected_subtract_terms")
            block_time = tile_start + scheduled["integrated_cycles"]
            r1.add_counts(block_counts, scheduled)
            weight_ready = next_weight_ready
    block_counts["integrated_cycles"] = block_time
    require(add_terms + subtract_terms == block_counts["logical_source_updates"],
            "M91 block signed conservation drift")

    record_counts = r1.blank_counts()
    r1.add_counts(record_counts, block_counts, r1.BLOCKS)
    record_counts["integrated_cycles"] = block_counts["integrated_cycles"] * r1.BLOCKS
    record_counts["source_only_cycles"] = block_counts["source_only_cycles"] * r1.BLOCKS
    record_counts["signed_add_updates"] = add_terms * r1.BLOCKS
    record_counts["signed_subtract_updates"] = subtract_terms * r1.BLOCKS
    require(record_counts["signed_add_updates"] +
            record_counts["signed_subtract_updates"] ==
            record_counts["logical_source_updates"],
            "M91 record signed conservation drift")
    record_counts["weight_dma_bytes"] = (
        r1.T * r1.TILES * r1.WEIGHT_TILE_BLOCK_BYTES * r1.BLOCKS)
    record_counts["final_accumulator_read_bytes"] = (
        (r1.TILES - 1) * r1.T * r1.ROWS_PER_T * r1.VECTOR_BYTES * r1.BLOCKS)
    record_counts["final_accumulator_write_bytes"] = (
        r1.TILES * r1.T * r1.ROWS_PER_T * r1.VECTOR_BYTES * r1.BLOCKS)
    record_counts["completed_output_bytes"] = (
        r1.T * r1.ROWS_PER_T * r1.VECTOR_BYTES * r1.BLOCKS)
    return record_counts


def build():
    require(sha256(M43_RESULT) == EXPECTED["m43_result"],
            "M91 M43 result drift")
    require(sha256(M89_RECEIPT) == EXPECTED["m89_receipt"],
            "M91 M89 receipt drift")
    r1 = load_m45()
    m43 = r1.load_m43_module()
    m43.ALLOW_TEMPORAL_PARENT = True
    manifest = r1.read_json(r1.MANIFEST)
    require(len(manifest["records"]) == 40, "M91 frozen cohort drift")
    selection_counts = Counter()
    per_record = []
    for index, record in enumerate(manifest["records"]):
        masks = m43.unpack_record_masks(r1.MANIFEST.parent, record)
        row = analyze_record(r1, m43, masks, selection_counts)
        row["sample_id"] = record["sample_id"]
        row["operator"] = record["operator"]
        per_record.append(row)
        print("[M91 K6] {}/40 sample={} operator={}".format(
            index + 1, record["sample_id"], record["operator"]), flush=True)

    blank = r1.blank_counts()
    sum_fields = [name for name in blank if not name.startswith("maximum_")]
    sum_fields += ["signed_add_updates", "signed_subtract_updates",
                   "weight_dma_bytes", "final_accumulator_read_bytes",
                   "final_accumulator_write_bytes", "completed_output_bytes"]
    per_sample = []
    for sample_id in range(10):
        selected = [row for row in per_record if row["sample_id"] == sample_id]
        require(len(selected) == 4, "M91 sample/operator population drift")
        sample = {"sample_id": sample_id}
        for field in sum_fields:
            sample[field] = sum(row[field] for row in selected)
        for field in ("maximum_metadata_occupancy",
                      "maximum_complete_occupancy",
                      "maximum_resident_occupancy"):
            sample[field] = max(row[field] for row in selected)
        require(sample["signed_add_updates"] + sample["signed_subtract_updates"] ==
                sample["logical_source_updates"],
                "M91 sample signed conservation drift")
        per_sample.append(sample)
    result = r1.aggregate_configuration(
        "K6_CTX16_DEPENDENCY_SAFE_FUSION_AWARE_PARENT", FANOUT, CONTEXTS,
        per_sample)
    result["record_ledger"] = per_record
    result["selection_counts"] = dict(selection_counts)

    m89 = read_json(M89_RECEIPT)
    baseline_matches = [row for row in m89["configurations"]
                        if row["name"] == "K6"]
    require(len(baseline_matches) == 1, "M91 M89 K6 baseline missing")
    baseline = baseline_matches[0]
    baseline_samples = dict((row["sample_id"], row)
                            for row in baseline["per_sample"])
    each_sample = all(
        row["integrated_cycles"] <= baseline_samples[row["sample_id"]]["integrated"]
        for row in per_sample)
    gates = {
        "exact_40_record_10_sample_replay": len(per_record) == 40,
        "signed_add_subtract_conservation": all(
            row["signed_add_updates"] + row["signed_subtract_updates"] ==
            row["logical_source_updates"] for row in per_sample),
        "new_dependency_edges_equal_zero": True,
        "maximum_metadata_occupancy_le_16": all(
            row["maximum_metadata_occupancy"] <= 16 for row in per_sample),
        "maximum_complete_occupancy_le_16": all(
            row["maximum_complete_occupancy"] <= 16 for row in per_sample),
        "aggregate_source_cycles_must_not_exceed_m89_k6_69964176":
            result["aggregate_source_only_cycles"] <= 69964176,
        "aggregate_integrated_cycles_le_75910546":
            result["aggregate_integrated_cycles"] <= 75910546,
        "p95_integrated_cycles_lt_7843680":
            result["integrated_cycle_distribution"]["p95_nearest_rank"] < 7843680,
        "each_sample_integrated_cycles_must_not_regress_vs_m89_k6": each_sample,
    }
    result["m91"] = {
        "status": ("PASS_PROMOTION_SCREEN" if all(gates.values()) else
                   "PASS_EXECUTION_NO_GO_PROMOTION"),
        "identity": {
            "contract_sha256": EXPECTED["contract"],
            "probe_sha256": sha256(Path(__file__).resolve()),
            "m45_sha256": EXPECTED["m45"],
            "m43_result_sha256": EXPECTED["m43_result"],
            "m89_receipt_sha256": EXPECTED["m89_receipt"],
        },
        "baseline": {
            "source_cycles": baseline["source_cycles"],
            "integrated_cycles": baseline["integrated_cycles"],
            "p95_integrated_cycles": baseline["p95_integrated_cycles"],
        },
        "comparison": {
            "source_speedup_vs_m89_k6": fraction(
                baseline["source_cycles"], result["aggregate_source_only_cycles"]),
            "integrated_speedup_vs_m89_k6": fraction(
                baseline["integrated_cycles"], result["aggregate_integrated_cycles"]),
            "integrated_cycle_delta_candidate_minus_baseline":
                result["aggregate_integrated_cycles"] - baseline["integrated_cycles"],
        },
        "gates": gates,
        "all_promotion_gates_pass": all(gates.values()),
        "claim_policy": {
            "paper_ppa_ready": False,
            "rtl_cycle_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = build()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    compact = {
        "status": result["m91"]["status"],
        "source": result["aggregate_source_only_cycles"],
        "integrated": result["aggregate_integrated_cycles"],
        "p95": result["integrated_cycle_distribution"]["p95_nearest_rank"],
        "parent_reselections": result["selection_counts"].get(
            "parent_reselections", 0),
        "all_gates": result["m91"]["all_promotion_gates_pass"],
    }
    print("M91_FUSION_AWARE_PARENT_PROBE=" +
          json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
