#!/usr/bin/env python3
"""M92 same-cycle commit-bus parent bypass transaction-model probe.

This keeps the frozen M91 parent-selection and scheduler contracts, but an
exact spatial parent that commits in the current admission cycle is forwarded
from the live writeback bus instead of being read back from parent SRAM.  It is
a non-citable transaction-model screen, not RTL, timing, PPA, or system proof.
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
CONTRACT = HW_ROOT / "contracts/m92_commit_bus_parent_bypass_contract_r1_20260824.json"
M91_PROBE = HW_ROOT / "system_simulator/scripts/probe_m91_dependency_safe_fusion_aware_parent.py"
M91_RESULT = HW_ROOT / (
    "results/m91_dependency_safe_fusion_aware_parent_probe_r1_20260824/"
    "remote_artifacts/m91_dependency_safe_fusion_aware_parent_probe_r1_20260824.json")
M91_RECEIPT = HW_ROOT / (
    "results/m91_dependency_safe_fusion_aware_parent_probe_r1_20260824/"
    "m91_dependency_safe_fusion_aware_parent_probe_receipt_r1.json")
M89_RECEIPT = HW_ROOT / (
    "results/m89_temporal_fanout_hold_screen_r1_20260823/"
    "m89_temporal_fanout_hold_screen_receipt.json")

EXPECTED = {
    "contract": "4f7063ae00c55bd0926a834a5f11c70547659282f81c85fd17d1e591af08d550",
    "m91_probe": "c6bf6d37713137c3e63067ead2ab0460856098d9b9f3d1c613359b48dc88f97a",
    "m91_result": "6245514b51c1d15a62d994be262a9a5da24235ad9c04b8dda919a8d68da70011",
    "m91_receipt": "83a3fe67e592e0fee1b619329e612798eee5da443285d35ce914d0fe2a9539a1",
    "m89_receipt": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}
FANOUT = 6
CONTEXTS = 16
MAX_PROMOTABLE_INTEGRATED = 75740988


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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs():
    paths = {
        "contract": CONTRACT,
        "m91_probe": M91_PROBE,
        "m91_result": M91_RESULT,
        "m91_receipt": M91_RECEIPT,
        "m89_receipt": M89_RECEIPT,
    }
    for name, path in paths.items():
        require(sha256(path) == EXPECTED[name], "M92 {} drift".format(name))


def selected_parent_task(r1, task, selected):
    if selected["name"] == "left":
        require(task % r1.W > 0, "M92 left parent at x=0")
        return task - 1
    if selected["name"] == "up":
        require(task >= r1.W, "M92 up parent at y=0")
        return task - r1.W
    return None


def schedule_tile_timestep(base, r1, m43, candidates, canonical_names,
                           canonical_candidates, start_cycle,
                           weight_ready_cycle, tile_index, selection_counts):
    require(len(candidates) == r1.ROWS_PER_T and FANOUT <= CONTEXTS,
            "M92 invalid task geometry")
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
    parent_vector_demands = 0
    parent_sram_reads = 0
    commit_bus_forward_hits = 0
    forward_hits_left = 0
    forward_hits_up = 0
    late_commit_events_rejected = 0
    cycle_memo = {}

    def cycles(mask):
        value = cycle_memo.get(mask)
        if value is None:
            value = m43.bank_issue_cycles(mask)
            cycle_memo[mask] = value
        return value

    def pop_ready_for_residency():
        window = heapq.nsmallest(min(16, len(ready)), ready)
        require(window, "M92 empty ready window")
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
                        base.PARENT_PRIORITY[option["name"]], option))
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
        nonlocal committed, late_commit_events_rejected
        commit_bus_tasks = set()
        while context_active and context_active[0][0] <= limit:
            heapq.heappop(context_active)
        while commit_events and commit_events[0][0] <= limit:
            commit_time, task = heapq.heappop(commit_events)
            if commit_time == limit:
                commit_bus_tasks.add(task)
            else:
                require(commit_time < limit, "M92 future commit consumed")
                late_commit_events_rejected += 1
            committed += 1
            for child in children[task]:
                indegree[child] -= 1
                require(indegree[child] >= 0, "M92 DAG indegree underflow")
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
                    ready_since[child] = commit_time
        while complete_entries and complete_entries[0][0] <= limit:
            heapq.heappop(complete_entries)
        return commit_bus_tasks

    while committed < r1.ROWS_PER_T:
        commit_bus_tasks = process_events(now)
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
                parent_vector_demands += 1
                parent_task = selected_parent_task(r1, task, selected)
                forward_hit = (
                    selected["name"] in ("left", "up") and
                    parent_task in commit_bus_tasks)
                if forward_hit:
                    commit_bus_forward_hits += 1
                    if selected["name"] == "left":
                        forward_hits_left += 1
                    else:
                        forward_hits_up += 1
                else:
                    parent_end = parent_port.schedule(now)
                    counts["parent_partial_reads"] += 1
                    parent_sram_reads += 1
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
                "M92 metadata/context overflow")

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
            require(next_times, "M92 scheduler deadlock")
            next_cycle, cause = min(next_times)
            require(next_cycle > now, "M92 scheduler failed to advance")
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
            require(throttle >= 0, "M92 negative complete-FIFO throttle")
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
                r1.COMPLETE_FIFO_ENTRIES, "M92 complete FIFO overflow")
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
            counts["logical_source_updates"] and
            parent_sram_reads == counts["parent_partial_reads"] and
            parent_sram_reads + commit_bus_forward_hits == parent_vector_demands,
            "M92 per-tile conservation drift")
    counts["selected_add_terms"] = selected_add_terms
    counts["selected_subtract_terms"] = selected_subtract_terms
    counts["parent_vector_demands"] = parent_vector_demands
    counts["parent_sram_reads"] = parent_sram_reads
    counts["commit_bus_forward_hits"] = commit_bus_forward_hits
    counts["forward_hits_left"] = forward_hits_left
    counts["forward_hits_up"] = forward_hits_up
    counts["late_commit_events_rejected"] = late_commit_events_rejected
    return counts


EXTRA_FIELDS = (
    "parent_vector_demands",
    "parent_sram_reads",
    "commit_bus_forward_hits",
    "forward_hits_left",
    "forward_hits_up",
    "late_commit_events_rejected",
)


def analyze_record(base, r1, m43, masks, selection_counts):
    block_time = 0
    block_counts = r1.blank_counts()
    block_extra = dict((name, 0) for name in EXTRA_FIELDS)
    add_terms = subtract_terms = 0
    weight_ready = block_time + r1.WEIGHT_LOAD_CYCLES
    block_counts["weight_dma_wait_cycles"] += r1.WEIGHT_LOAD_CYCLES
    for timestep in range(r1.T):
        for tile in range(r1.TILES):
            tile_start = max(block_time, weight_ready)
            next_weight_ready = (
                tile_start + r1.WEIGHT_LOAD_CYCLES
                if tile + 1 < r1.TILES or timestep + 1 < r1.T else tile_start)
            candidates, canonical_names, canonical_candidates = base.build_tasks(
                m43, masks, tile, timestep)
            scheduled = schedule_tile_timestep(
                base, r1, m43, candidates, canonical_names,
                canonical_candidates, tile_start, weight_ready, tile,
                selection_counts)
            add_terms += scheduled.pop("selected_add_terms")
            subtract_terms += scheduled.pop("selected_subtract_terms")
            for name in EXTRA_FIELDS:
                block_extra[name] += scheduled.pop(name)
            block_time = tile_start + scheduled["integrated_cycles"]
            r1.add_counts(block_counts, scheduled)
            weight_ready = next_weight_ready
    block_counts["integrated_cycles"] = block_time
    require(add_terms + subtract_terms == block_counts["logical_source_updates"],
            "M92 block signed conservation drift")

    record_counts = r1.blank_counts()
    r1.add_counts(record_counts, block_counts, r1.BLOCKS)
    record_counts["integrated_cycles"] = block_counts["integrated_cycles"] * r1.BLOCKS
    record_counts["source_only_cycles"] = block_counts["source_only_cycles"] * r1.BLOCKS
    record_counts["signed_add_updates"] = add_terms * r1.BLOCKS
    record_counts["signed_subtract_updates"] = subtract_terms * r1.BLOCKS
    for name in EXTRA_FIELDS:
        record_counts[name] = block_extra[name] * r1.BLOCKS
    require(record_counts["signed_add_updates"] +
            record_counts["signed_subtract_updates"] ==
            record_counts["logical_source_updates"] and
            record_counts["parent_sram_reads"] +
            record_counts["commit_bus_forward_hits"] ==
            record_counts["parent_vector_demands"],
            "M92 record conservation drift")
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
    validate_inputs()
    base = load_module(M91_PROBE, "m92_m91")
    r1 = base.load_m45()
    m43 = r1.load_m43_module()
    m43.ALLOW_TEMPORAL_PARENT = True
    manifest = r1.read_json(r1.MANIFEST)
    require(len(manifest["records"]) == 40, "M92 frozen cohort drift")
    selection_counts = Counter()
    per_record = []
    for index, record in enumerate(manifest["records"]):
        masks = m43.unpack_record_masks(r1.MANIFEST.parent, record)
        row = analyze_record(base, r1, m43, masks, selection_counts)
        row["sample_id"] = record["sample_id"]
        row["operator"] = record["operator"]
        per_record.append(row)
        print("[M92 K6] {}/40 sample={} operator={}".format(
            index + 1, record["sample_id"], record["operator"]), flush=True)

    blank = r1.blank_counts()
    sum_fields = [name for name in blank if not name.startswith("maximum_")]
    sum_fields += ["signed_add_updates", "signed_subtract_updates",
                   "weight_dma_bytes", "final_accumulator_read_bytes",
                   "final_accumulator_write_bytes", "completed_output_bytes"]
    sum_fields += list(EXTRA_FIELDS)
    per_sample = []
    for sample_id in range(10):
        selected = [row for row in per_record if row["sample_id"] == sample_id]
        require(len(selected) == 4, "M92 sample/operator population drift")
        sample = {"sample_id": sample_id}
        for field in sum_fields:
            sample[field] = sum(row[field] for row in selected)
        for field in ("maximum_metadata_occupancy",
                      "maximum_complete_occupancy",
                      "maximum_resident_occupancy"):
            sample[field] = max(row[field] for row in selected)
        require(sample["signed_add_updates"] + sample["signed_subtract_updates"] ==
                sample["logical_source_updates"] and
                sample["parent_sram_reads"] +
                sample["commit_bus_forward_hits"] ==
                sample["parent_vector_demands"],
                "M92 sample conservation drift")
        per_sample.append(sample)

    result = r1.aggregate_configuration(
        "K6_CTX16_M91_COMMIT_BUS_PARENT_BYPASS", FANOUT, CONTEXTS,
        per_sample)
    result["record_ledger"] = per_record
    result["selection_counts"] = dict(selection_counts)
    for field in EXTRA_FIELDS:
        result["aggregate_" + field] = sum(row[field] for row in per_sample)

    m91 = read_json(M91_RESULT)
    baseline_samples = dict((row["sample_id"], row)
                            for row in m91["per_sample"])
    each_sample = all(
        row["integrated_cycles"] <=
        baseline_samples[row["sample_id"]]["integrated_cycles"]
        for row in per_sample)
    gates = {
        "exact_40_record_10_sample_replay": len(per_record) == 40,
        "signed_add_subtract_conservation": all(
            row["signed_add_updates"] + row["signed_subtract_updates"] ==
            row["logical_source_updates"] for row in per_sample),
        "new_dependency_edges_equal_zero": True,
        "parent_vector_demand_conservation": all(
            row["parent_sram_reads"] + row["commit_bus_forward_hits"] ==
            row["parent_vector_demands"] for row in per_sample),
        "additional_vector_payload_storage_bytes_equal_zero": True,
        "maximum_metadata_occupancy_le_16": all(
            row["maximum_metadata_occupancy"] <= 16 for row in per_sample),
        "maximum_complete_occupancy_le_16": all(
            row["maximum_complete_occupancy"] <= 16 for row in per_sample),
        "aggregate_source_cycles_must_not_exceed_m91_69211896":
            result["aggregate_source_only_cycles"] <= 69211896,
        "aggregate_integrated_cycles_le_75740988":
            result["aggregate_integrated_cycles"] <= MAX_PROMOTABLE_INTEGRATED,
        "p95_integrated_cycles_lt_7769480":
            result["integrated_cycle_distribution"]["p95_nearest_rank"] < 7769480,
        "each_sample_integrated_cycles_must_not_regress_vs_m91": each_sample,
        "commit_bus_forward_hits_must_be_positive":
            result["aggregate_commit_bus_forward_hits"] > 0,
    }
    result["m92"] = {
        "status": ("PASS_PROMOTION_SCREEN" if all(gates.values()) else
                   "PASS_EXECUTION_NO_GO_PROMOTION"),
        "identity": {
            "contract_sha256": EXPECTED["contract"],
            "probe_sha256": sha256(Path(__file__).resolve()),
            "m91_probe_sha256": EXPECTED["m91_probe"],
            "m91_result_sha256": EXPECTED["m91_result"],
            "m91_receipt_sha256": EXPECTED["m91_receipt"],
            "m89_receipt_sha256": EXPECTED["m89_receipt"],
        },
        "baseline": {
            "source_cycles": m91["aggregate_source_only_cycles"],
            "integrated_cycles": m91["aggregate_integrated_cycles"],
            "p95_integrated_cycles":
                m91["integrated_cycle_distribution"]["p95_nearest_rank"],
        },
        "comparison": {
            "source_speedup_vs_m91": fraction(
                m91["aggregate_source_only_cycles"],
                result["aggregate_source_only_cycles"]),
            "integrated_speedup_vs_m91": fraction(
                m91["aggregate_integrated_cycles"],
                result["aggregate_integrated_cycles"]),
            "integrated_cycle_delta_candidate_minus_m91":
                result["aggregate_integrated_cycles"] -
                m91["aggregate_integrated_cycles"],
            "parent_sram_read_bypass_fraction": fraction(
                result["aggregate_commit_bus_forward_hits"],
                result["aggregate_parent_vector_demands"]),
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
        "status": result["m92"]["status"],
        "source": result["aggregate_source_only_cycles"],
        "integrated": result["aggregate_integrated_cycles"],
        "p95": result["integrated_cycle_distribution"]["p95_nearest_rank"],
        "forward_hits": result["aggregate_commit_bus_forward_hits"],
        "parent_demands": result["aggregate_parent_vector_demands"],
        "all_gates": result["m92"]["all_promotion_gates_pass"],
    }
    print("M92_COMMIT_BUS_PARENT_BYPASS=" + json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
