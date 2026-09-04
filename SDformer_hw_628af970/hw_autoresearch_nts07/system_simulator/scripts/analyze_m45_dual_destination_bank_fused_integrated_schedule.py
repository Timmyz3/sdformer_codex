#!/usr/bin/env python3
"""M45 exact all-ten dual-destination bank-fused transaction scheduler.

The source scheduler is exact for the frozen algorithm below.  It is a
transaction-level hardware schedule, not RTL-measured timing or system speedup.
"""

from __future__ import print_function

import argparse
import hashlib
import heapq
import importlib.util
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / (
    "contracts/m45_dual_destination_bank_fused_integrated_schedule_contract_r1_20260823.json")
EXPECTED_CONTRACT_SHA256 = (
    "fbe559775f0262f7671fb790ebbaa452e2a01d68e5a44670c61003f4751b69e8")
M43_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py")
MANIFEST = HW_ROOT / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
    "m40_bottleneck_packed_source_manifest.json")
M40_RESULT = HW_ROOT / (
    "results/m40_conflict_aware_event_schedule_r3_20260822/"
    "m40_conflict_aware_event_schedule.json")
M41_RESULT = HW_ROOT / (
    "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/"
    "m41_h67_ep35_bottleneck_int8_bridge.json")
M42_RESULT = HW_ROOT / (
    "results/m42_real_work_headroom_gate_r1_20260823/"
    "m42_real_work_headroom_gate.json")
M43_RESULT = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatial_parent_delta_schedule_final.json")
M43_REVIEW = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_r1_independent_hammer_review.json")

T = 10
C = 768
H = 15
W = 20
ROWS_PER_T = H * W
FEATURES = C * 9
TILE_BITS = 256
TILES = FEATURES // TILE_BITS
BANKS = 8
LANES = 96
BLOCKS = C // LANES
ACC_BYTES = 3
VECTOR_BYTES = LANES * ACC_BYTES
WEIGHT_DMA_BYTES_PER_CYCLE = 64
WEIGHT_TILE_BLOCK_BYTES = TILE_BITS * LANES
WEIGHT_LOAD_CYCLES = WEIGHT_TILE_BLOCK_BYTES // WEIGHT_DMA_BYTES_PER_CYCLE
METADATA_FIFO_ENTRIES = 16
COMPLETE_FIFO_ENTRIES = 16
METADATA_FIFO_ENTRY_BYTES = 64
METADATA_FIFO_BYTES = METADATA_FIFO_ENTRIES * METADATA_FIFO_ENTRY_BYTES
COMPLETE_FIFO_VECTOR_BYTES = VECTOR_BYTES
COMPLETE_FIFO_TAG_CONTROL_BYTES = 16
COMPLETE_FIFO_ENTRY_BYTES = (COMPLETE_FIFO_VECTOR_BYTES +
                             COMPLETE_FIFO_TAG_CONTROL_BYTES)
COMPLETE_FIFO_BYTES = COMPLETE_FIFO_ENTRIES * COMPLETE_FIFO_ENTRY_BYTES
FIFO_BYTES = METADATA_FIFO_BYTES + COMPLETE_FIFO_BYTES
LOCAL_SCRATCH_BYTES = 56960
TIMESTEP_ACC_BYTES = ROWS_PER_T * VECTOR_BYTES
BASE_LOCAL_BYTES = LOCAL_SCRATCH_BYTES + TIMESTEP_ACC_BYTES
COMBINED_LOCAL_BYTES = BASE_LOCAL_BYTES + FIFO_BYTES
FROZEN_LOCAL_RESIDENCY_BYTES = 193728
LOCAL_CAPACITY_HEADROOM_BYTES = FROZEN_LOCAL_RESIDENCY_BYTES - COMBINED_LOCAL_BYTES
WRITEBACK_DEFERRAL_CYCLES = 0
CONFIGURATIONS = (
    ("K1_CTX4_REPRODUCTION", 1, 4),
    ("K2_CTX4_PRIMARY", 2, 4),
    ("K4_CTX4_ABLATION", 4, 4),
    ("K2_CTX2_SWEEP", 2, 2),
    ("K2_CTX8_SWEEP", 2, 8),
)


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


def load_m43_module():
    spec = importlib.util.spec_from_file_location("m45_pinned_m43", M43_ANALYZER)
    require(spec is not None and spec.loader is not None,
            "cannot import pinned M43 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ALLOW_TEMPORAL_PARENT = False
    return module


def validate_contract():
    require(sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M45 contract identity drift")
    contract = read_json(CONTRACT)
    require(contract["schema"] ==
            "m45_dual_destination_bank_fused_integrated_schedule_contract_v1",
            "M45 contract schema drift")
    for name, item in contract["inputs"].items():
        path = HW_ROOT / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "M45 upstream identity drift: {}".format(name))
    service = contract["service_model"]
    require(service["base_local_capacity_bytes"] == BASE_LOCAL_BYTES == 143360,
            "M45 base local capacity contract drift")
    require(service["metadata_fifo_storage_bytes"] == METADATA_FIFO_BYTES == 1024,
            "M45 metadata FIFO capacity contract drift")
    require(service["complete_fifo_storage_bytes"] == COMPLETE_FIFO_BYTES == 4864,
            "M45 complete FIFO capacity contract drift")
    require(service["fifo_storage_bytes"] == FIFO_BYTES == 5888,
            "M45 FIFO capacity contract drift")
    require(service["combined_local_capacity_bytes"] ==
            COMBINED_LOCAL_BYTES == 149248,
            "M45 local capacity contract drift")
    require(service["local_capacity_headroom_bytes"] ==
            LOCAL_CAPACITY_HEADROOM_BYTES == 44480,
            "M45 local capacity headroom contract drift")
    return contract


class PortCalendar(object):
    """One-operation-per-cycle calendar supporting future reservations."""

    def __init__(self):
        self.occupied = set()
        self.last_end = 0
        self.operations = 0

    def schedule(self, ready_cycle):
        cycle = int(ready_cycle)
        while cycle in self.occupied:
            cycle += 1
        self.occupied.add(cycle)
        self.last_end = max(self.last_end, cycle + 1)
        self.operations += 1
        return cycle + 1


def fraction(numerator, denominator):
    require(denominator > 0, "zero fraction denominator")
    return {"numerator": numerator, "denominator": denominator}


def distribution(values):
    ordered = sorted(values)
    require(ordered, "empty distribution")

    def nr(percent):
        rank = (percent * len(ordered) + 99) // 100
        return ordered[rank - 1]
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean_exact": fraction(sum(ordered), len(ordered)),
        "p50_nearest_rank": nr(50),
        "p95_nearest_rank": nr(95),
        "p99_nearest_rank": nr(99),
    }


def blank_counts():
    return {
        "source_only_cycles": 0,
        "integrated_cycles": 0,
        "logical_source_updates": 0,
        "unique_weight_issues": 0,
        "descriptor_commands": 0,
        "parent_partial_reads": 0,
        "parent_partial_writes": 0,
        "final_accumulator_reads": 0,
        "final_accumulator_writes": 0,
        "completed_outputs": 0,
        "fusion_groups": 0,
        "zero_source_groups": 0,
        "parent_wait_cycles": 0,
        "command_or_state_wait_cycles": 0,
        "response_or_context_wait_cycles": 0,
        "weight_dma_wait_cycles": 0,
        "fusion_hold_wait_cycles": 0,
        "late_join_groups": 0,
        "maximum_metadata_occupancy": 0,
        "maximum_complete_occupancy": 0,
        "maximum_resident_occupancy": 0,
    }


def add_counts(destination, source, multiplier=1):
    for name in destination:
        if name.startswith("maximum_"):
            destination[name] = max(destination[name], source[name])
        else:
            destination[name] += source[name] * multiplier


def build_tile_timestep_tasks(m43, masks, tile, timestep):
    delta_masks = []
    selected_parent = []
    add_terms = subtract_terms = 0
    for spatial in range(ROWS_PER_T):
        row = timestep * ROWS_PER_T + spatial
        name, parent, add_mask, subtract_mask = m43.select_parent(
            masks, row, tile)
        require(name in ("local_zero", "left", "up"),
                "temporal parent leaked into M45 primary")
        delta = add_mask | subtract_mask
        require((add_mask & subtract_mask) == 0 and
                delta == (masks[row * TILES + tile] ^ parent),
                "M45 signed delta partition drift")
        delta_masks.append(delta)
        selected_parent.append(name)
        add_terms += m43.population(add_mask)
        subtract_terms += m43.population(subtract_mask)
    return delta_masks, selected_parent, add_terms, subtract_terms


def build_structural_dag(selected_parent):
    indegree = [0] * ROWS_PER_T
    children = [[] for _ in range(ROWS_PER_T)]
    for spatial in range(ROWS_PER_T):
        y, x = divmod(spatial, W)
        if x > 0 and selected_parent[spatial] == "left":
            indegree[spatial] += 1
            children[spatial - 1].append(spatial)
        if y > 0:
            indegree[spatial] += 1
            children[spatial - W].append(spatial)
    return indegree, children


def select_fusion_group(m43, prepared, delta_masks, fanout_k, cycle_cache):
    seed = min(prepared)
    group = [seed]
    union = delta_masks[seed]
    while len(group) < fanout_k:
        choices = [item for item in prepared if item not in group]
        if not choices:
            break
        union_cycles = cycle_cache(union)
        ranked = []
        for candidate in choices:
            candidate_mask = delta_masks[candidate]
            fused = union | candidate_mask
            fused_cycles = cycle_cache(fused)
            saved = (union_cycles + cycle_cache(candidate_mask) - fused_cycles)
            ranked.append((-saved, fused_cycles, candidate, fused))
        winner = min(ranked)
        group.append(winner[2])
        union = winner[3]
    return group, union


def delayed_pair_cycles(m43, first_mask, second_mask, join_delay):
    """Exact duration when the second destination joins after 1-2 cycles.

    Each bank uses the pre-join slots only for first-destination sources absent
    from the second mask.  No weight response is replayed.
    """
    require(join_delay in (1, 2), "invalid K2 late-join delay")
    duration = 0
    for bank_mask in m43.BANK_MASKS:
        first_unique = m43.population((first_mask & ~second_mask) & bank_mask)
        union_count = m43.population((first_mask | second_mask) & bank_mask)
        bank_duration = (join_delay + union_count -
                         min(join_delay, first_unique))
        duration = max(duration, bank_duration)
    return duration


def schedule_tile_timestep(m43, delta_masks, selected_parent, fanout_k,
                           context_capacity, start_cycle, weight_ready_cycle,
                           tile_index):
    require(1 <= fanout_k <= context_capacity <= 8,
            "invalid fanout/context geometry")
    indegree, children = build_structural_dag(selected_parent)
    ready = [index for index, count in enumerate(indegree) if count == 0]
    heapq.heapify(ready)
    ready_since = dict((index, start_cycle) for index in ready)
    resident = {}
    context_active = []
    commit_events = []
    complete_entries = []
    committed = 0
    now = start_cycle
    command_port = PortCalendar()
    parent_port = PortCalendar()
    final_port = PortCalendar()
    output_port = PortCalendar()
    counts = blank_counts()
    cycle_memo = {}

    def cycles(mask):
        value = cycle_memo.get(mask)
        if value is None:
            value = m43.bank_issue_cycles(mask)
            cycle_memo[mask] = value
        return value

    def pop_ready_for_residency():
        window = heapq.nsmallest(min(16, len(ready)), ready)
        require(window, "empty ready selection window")
        if fanout_k == 1 or not resident:
            selected = window[0]
        else:
            ranked = []
            for candidate in window:
                best_saved = -1
                best_fused = None
                for anchor in resident:
                    fused = delta_masks[anchor] | delta_masks[candidate]
                    fused_cycles = cycles(fused)
                    saved = (cycles(delta_masks[anchor]) +
                             cycles(delta_masks[candidate]) - fused_cycles)
                    if saved > best_saved or (saved == best_saved and
                                               (best_fused is None or
                                                fused_cycles < best_fused)):
                        best_saved = saved
                        best_fused = fused_cycles
                ranked.append((-best_saved, best_fused, candidate))
            selected = min(ranked)[2]
        ready.remove(selected)
        heapq.heapify(ready)
        return selected

    def process_events(limit):
        nonlocal committed
        while context_active and context_active[0][0] <= limit:
            heapq.heappop(context_active)
        while commit_events and commit_events[0][0] <= limit:
            commit_time, task = heapq.heappop(commit_events)
            committed += 1
            for child in children[task]:
                indegree[child] -= 1
                require(indegree[child] >= 0, "DAG indegree underflow")
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
                    ready_since[child] = commit_time
        while complete_entries and complete_entries[0][0] <= limit:
            heapq.heappop(complete_entries)

    while committed < ROWS_PER_T:
        process_events(now)
        if committed == ROWS_PER_T:
            break
        occupied_contexts = len(resident) + len(context_active)
        while ready and occupied_contexts < context_capacity:
            counts["maximum_metadata_occupancy"] = max(
                counts["maximum_metadata_occupancy"], min(16, len(ready)))
            task = pop_ready_for_residency()
            command_end = command_port.schedule(ready_since.pop(task))
            parent_end = now
            if selected_parent[task] != "local_zero":
                parent_end = parent_port.schedule(now)
                counts["parent_partial_reads"] += 1
            prep_ready = max(command_end, parent_end,
                             weight_ready_cycle)
            if prep_ready == weight_ready_cycle and weight_ready_cycle > now:
                cause = "weight_dma_wait_cycles"
            elif prep_ready == command_end:
                cause = "command_or_state_wait_cycles"
            else:
                cause = "command_or_state_wait_cycles"
            resident[task] = (prep_ready, cause)
            counts["descriptor_commands"] += 1
            occupied_contexts += 1
        counts["maximum_resident_occupancy"] = max(
            counts["maximum_resident_occupancy"], occupied_contexts)
        require(occupied_contexts <= context_capacity and
                counts["maximum_metadata_occupancy"] <= METADATA_FIFO_ENTRIES,
                "metadata/context capacity overflow")

        prepared = [task for task, item in resident.items() if item[0] <= now]
        if not prepared:
            next_times = []
            for task, item in resident.items():
                next_times.append((item[0], item[1]))
            if context_active:
                cause = ("response_or_context_wait_cycles" if ready else
                         "parent_wait_cycles")
                next_times.append((context_active[0][0], cause))
            if commit_events:
                cause = ("parent_wait_cycles" if not ready and not resident else
                         "response_or_context_wait_cycles")
                next_times.append((commit_events[0][0], cause))
            require(next_times, "M45 scheduler deadlock")
            next_cycle, cause = min(next_times)
            require(next_cycle > now, "M45 scheduler failed to advance")
            counts[cause] += next_cycle - now
            now = next_cycle
            continue

        late_join = None
        if fanout_k == 2 and len(prepared) == 1:
            seed = prepared[0]
            future = []
            for candidate, item in resident.items():
                delay = item[0] - now
                if candidate != seed and delay in (1, 2):
                    fused = delta_masks[seed] | delta_masks[candidate]
                    future.append((
                        delayed_pair_cycles(m43, delta_masks[seed],
                                            delta_masks[candidate], delay),
                        -cycles(delta_masks[seed]) - cycles(delta_masks[candidate]) +
                        cycles(fused), candidate, fused, delay))
            if future:
                winner = min(future)
                late_join = ([seed, winner[2]], winner[3], winner[0])

        if fanout_k == 4 and len(prepared) < fanout_k:
            future_resident = [item[0] for item in resident.values()
                               if item[0] > now]
            if future_resident:
                next_cycle = min(future_resident)
                if next_cycle - now <= 2:
                    counts["fusion_hold_wait_cycles"] += next_cycle - now
                    now = next_cycle
                    continue

        if late_join is not None:
            group, union_mask, group_cycles = late_join
            counts["late_join_groups"] += 1
        else:
            group, union_mask = select_fusion_group(
                m43, prepared, delta_masks, fanout_k, cycles)
            group_cycles = cycles(union_mask)
        for task in group:
            del resident[task]
        projected_response = now + group_cycles + (1 if group_cycles else 0)
        live_entries = sorted(item[0] for item in complete_entries
                              if item[0] > projected_response)
        maximum_live_before_group = COMPLETE_FIFO_ENTRIES - len(group)
        if len(live_entries) > maximum_live_before_group:
            entries_to_drain = len(live_entries) - maximum_live_before_group
            required_response = live_entries[entries_to_drain - 1]
            throttle = required_response - projected_response
            require(throttle >= 0, "negative complete-FIFO throttle")
            now += throttle
            counts["response_or_context_wait_cycles"] += throttle
        counts["source_only_cycles"] += group_cycles
        counts["logical_source_updates"] += sum(
            m43.population(delta_masks[task]) for task in group)
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
                response_ready + WRITEBACK_DEFERRAL_CYCLES)
            if tile_index > 0:
                final_read_end = final_port.schedule(response_ready)
                counts["final_accumulator_reads"] += 1
                final_write_end = final_port.schedule(final_read_end)
            else:
                final_write_end = final_port.schedule(response_ready)
            counts["parent_partial_writes"] += 1
            counts["final_accumulator_writes"] += 1
            commit_time = max(parent_write_end, final_write_end)
            if tile_index == TILES - 1:
                output_port.schedule(commit_time)
                counts["completed_outputs"] += 1
            commit_batch.append((commit_time, task))
        counts["maximum_complete_occupancy"] = max(
            counts["maximum_complete_occupancy"],
            len(complete_entries) + len(commit_batch))
        require(len(complete_entries) + len(commit_batch) <= COMPLETE_FIFO_ENTRIES,
                "complete FIFO overflow")
        for item in commit_batch:
            heapq.heappush(commit_events, item)
            heapq.heappush(complete_entries, item)

    end_cycle = max(now, command_port.last_end, parent_port.last_end,
                    final_port.last_end, output_port.last_end,
                    commit_events[-1][0] if commit_events else 0)
    counts["integrated_cycles"] = end_cycle - start_cycle
    require(counts["descriptor_commands"] == ROWS_PER_T and
            counts["parent_partial_writes"] == ROWS_PER_T and
            counts["final_accumulator_writes"] == ROWS_PER_T,
            "M45 per-tile context conservation drift")
    return counts


def analyze_record(m43, masks, expected_m43_record, fanout_k,
                   context_capacity):
    block_time = 0
    block_counts = blank_counts()
    add_terms = subtract_terms = 0
    # One 86.4-KB accumulator frame remains resident for a whole timestep.
    # Therefore every timestep replays all 27 weight tiles.  Tile n+1 loads in
    # the second buffer while tile n computes.
    weight_ready = block_time + WEIGHT_LOAD_CYCLES
    block_counts["weight_dma_wait_cycles"] += WEIGHT_LOAD_CYCLES
    for timestep in range(T):
        for tile in range(TILES):
            tile_start = max(block_time, weight_ready)
            next_weight_ready = (
                tile_start + WEIGHT_LOAD_CYCLES
                if tile + 1 < TILES or timestep + 1 < T else tile_start)
            delta_masks, selected_parent, adds, subtracts = (
                build_tile_timestep_tasks(m43, masks, tile, timestep))
            add_terms += adds
            subtract_terms += subtracts
            scheduled = schedule_tile_timestep(
                m43, delta_masks, selected_parent, fanout_k,
                context_capacity, tile_start,
                weight_ready, tile)
            block_time = tile_start + scheduled["integrated_cycles"]
            add_counts(block_counts, scheduled)
            weight_ready = next_weight_ready
    block_counts["integrated_cycles"] = block_time
    require(add_terms + subtract_terms == block_counts["logical_source_updates"],
            "M45 signed source/update conservation drift")
    # All eight output blocks have the identical source/control schedule.
    record_counts = blank_counts()
    add_counts(record_counts, block_counts, BLOCKS)
    record_counts["integrated_cycles"] = block_counts["integrated_cycles"] * BLOCKS
    record_counts["source_only_cycles"] = block_counts["source_only_cycles"] * BLOCKS
    require(record_counts["logical_source_updates"] ==
            (add_terms + subtract_terms) * BLOCKS,
            "M45 output-block conservation drift")
    if fanout_k == 1:
        require(record_counts["source_only_cycles"] ==
                expected_m43_record["parent_delta_p8_l96_source_issue_cycles"],
                "K1 does not reproduce M43 record source cycles")
    record_counts["signed_add_updates"] = add_terms * BLOCKS
    record_counts["signed_subtract_updates"] = subtract_terms * BLOCKS
    record_counts["weight_dma_bytes"] = (
        T * TILES * WEIGHT_TILE_BLOCK_BYTES * BLOCKS)
    record_counts["final_accumulator_read_bytes"] = (
        (TILES - 1) * T * ROWS_PER_T * VECTOR_BYTES * BLOCKS)
    record_counts["final_accumulator_write_bytes"] = (
        TILES * T * ROWS_PER_T * VECTOR_BYTES * BLOCKS)
    record_counts["completed_output_bytes"] = (
        T * ROWS_PER_T * VECTOR_BYTES * BLOCKS)
    return record_counts


def analyze_configuration(m43, manifest, m43_records, fanout_k,
                          context_capacity):
    per_record = []
    trace_dir = MANIFEST.parent
    for index, record in enumerate(manifest["records"]):
        key = (record["sample_id"], record["operator"])
        require(key in m43_records, "M43 record identity mismatch")
        masks = m43.unpack_record_masks(trace_dir, record)
        row = analyze_record(m43, masks, m43_records[key], fanout_k,
                             context_capacity)
        row["sample_id"] = record["sample_id"]
        row["operator"] = record["operator"]
        per_record.append(row)
        print("[M45 K{} C{}] {}/40 sample={} operator={}".format(
            fanout_k, context_capacity, index + 1, record["sample_id"],
            record["operator"]))
    per_sample = []
    sum_fields = [name for name in blank_counts()
                  if not name.startswith("maximum_")]
    sum_fields += ["signed_add_updates", "signed_subtract_updates",
                   "weight_dma_bytes", "final_accumulator_read_bytes",
                   "final_accumulator_write_bytes", "completed_output_bytes"]
    for sample_id in range(10):
        selected = [row for row in per_record if row["sample_id"] == sample_id]
        require(len(selected) == 4, "M45 per-sample operator drift")
        sample = {"sample_id": sample_id}
        for name in sum_fields:
            sample[name] = sum(row[name] for row in selected)
        for name in ("maximum_metadata_occupancy", "maximum_complete_occupancy"):
            sample[name] = max(row[name] for row in selected)
        sample["integrated_over_source_only"] = fraction(
            sample["integrated_cycles"] - sample["source_only_cycles"],
            sample["source_only_cycles"])
        sample["parent_wait_fraction"] = fraction(
            sample["parent_wait_cycles"], sample["integrated_cycles"])
        per_sample.append(sample)
    return per_record, per_sample


def aggregate_configuration(name, fanout_k, context_capacity, per_sample):
    source_values = [row["source_only_cycles"] for row in per_sample]
    integrated_values = [row["integrated_cycles"] for row in per_sample]
    total_source = sum(source_values)
    total_integrated = sum(integrated_values)
    total_parent_wait = sum(row["parent_wait_cycles"] for row in per_sample)
    return {
        "name": name,
        "destination_fanout_k": fanout_k,
        "resident_contexts": context_capacity,
        "qualification": ("PRIMARY_DUAL_DESTINATION" if name ==
                          "K2_CTX4_PRIMARY" else
                          "COUNTERFACTUAL_ABLATION_ONLY" if fanout_k == 4 else
                          "REPRODUCTION_OR_CONTEXT_SWEEP"),
        "source_only_cycle_distribution": distribution(source_values),
        "integrated_cycle_distribution": distribution(integrated_values),
        "aggregate_source_only_cycles": total_source,
        "aggregate_integrated_cycles": total_integrated,
        "aggregate_integrated_over_source_only": fraction(
            total_integrated - total_source, total_source),
        "aggregate_parent_wait_fraction": fraction(
            total_parent_wait, total_integrated),
        "aggregate_logical_source_updates": sum(
            row["logical_source_updates"] for row in per_sample),
        "aggregate_unique_weight_issues": sum(
            row["unique_weight_issues"] for row in per_sample),
        "aggregate_fusion_groups": sum(row["fusion_groups"] for row in per_sample),
        "traffic_bytes_per_sample": {
            "weight_dma": per_sample[0]["weight_dma_bytes"],
            "final_accumulator_read": per_sample[0]["final_accumulator_read_bytes"],
            "final_accumulator_write": per_sample[0]["final_accumulator_write_bytes"],
            "completed_output": per_sample[0]["completed_output_bytes"],
        },
        "per_sample": per_sample,
    }


def build():
    contract = validate_contract()
    manifest = read_json(MANIFEST)
    m43_result = read_json(M43_RESULT)
    m43_review = read_json(M43_REVIEW)
    m41 = read_json(M41_RESULT)
    m42 = read_json(M42_RESULT)
    require(manifest["cohort"]["records"] == 40 and
            manifest["cohort"]["samples"] == 10,
            "M45 cohort drift")
    require(m43_review["status"] ==
            "GO_EXACT_SOURCE_BANK_SCHEDULE_AND_CAPACITY_GATES_ONLY",
            "M43 independent review is not GO")
    require(m41["quantization_contract"]["weight_payload_layout"].startswith(
            "I_KY_KX_O"), "M41 weight layout drift")
    m43_records = dict(((row["sample_id"], row["operator"]), row)
                       for row in m43_result["records"])
    require(len(m43_records) == 40, "M43 record population drift")
    m43 = load_m43_module()
    configurations = []
    for name, fanout_k, contexts in CONFIGURATIONS:
        records, samples = analyze_configuration(
            m43, manifest, m43_records, fanout_k, contexts)
        config = aggregate_configuration(name, fanout_k, contexts, samples)
        config["records"] = records
        configurations.append(config)

    by_name = dict((row["name"], row) for row in configurations)
    k1 = by_name["K1_CTX4_REPRODUCTION"]
    primary = by_name["K2_CTX4_PRIMARY"]
    ctx8 = by_name["K2_CTX8_SWEEP"]
    require(k1["aggregate_source_only_cycles"] == 116376872,
            "K1 aggregate does not reproduce M43")
    maximum_product_cycles = 15495075
    require(m42["target_gates"][2]["target_compute_speedup"] ==
            {"numerator": 3, "denominator": 1}, "M42 3x gate order drift")
    kill = contract["kill_gates"]
    primary_overhead_pass = all(
        (sample["integrated_cycles"] - sample["source_only_cycles"]) *
        kill["maximum_integrated_over_source_only_fraction"]["denominator"] <=
        sample["source_only_cycles"] *
        kill["maximum_integrated_over_source_only_fraction"]["numerator"]
        for sample in primary["per_sample"])
    parent_wait_pass = all(
        sample["parent_wait_cycles"] *
        kill["maximum_parent_wait_fraction"]["denominator"] <=
        sample["integrated_cycles"] *
        kill["maximum_parent_wait_fraction"]["numerator"]
        for sample in primary["per_sample"])
    k2_reduction_num = (k1["aggregate_integrated_cycles"] -
                        primary["aggregate_integrated_cycles"])
    k2_reduction_den = k1["aggregate_integrated_cycles"]
    k2_reduction_pass = (
        k2_reduction_num *
        kill["minimum_k2_primary_reduction_vs_k1_fraction"]["denominator"] >=
        k2_reduction_den *
        kill["minimum_k2_primary_reduction_vs_k1_fraction"]["numerator"])
    primary_p95 = primary["integrated_cycle_distribution"]["p95_nearest_rank"]
    ctx8_p95 = ctx8["integrated_cycle_distribution"]["p95_nearest_rank"]
    ctx8_improvement = max(0, primary_p95 - ctx8_p95)
    ctx8_saturation_pass = (
        ctx8_improvement *
        kill["maximum_ctx8_p95_improvement_over_ctx4_fraction"]["denominator"] <=
        primary_p95 *
        kill["maximum_ctx8_p95_improvement_over_ctx4_fraction"]["numerator"])
    p95_gate_pass = primary_p95 <= maximum_product_cycles
    gates = {
        "k1_reproduces_m43_116376872": True,
        "primary_all_samples_integrated_over_source_only_at_most_10pct":
            primary_overhead_pass,
        "primary_all_samples_parent_wait_at_most_5pct": parent_wait_pass,
        "k2_primary_integrated_reduction_vs_k1": fraction(
            k2_reduction_num, k2_reduction_den),
        "k2_primary_reduction_at_least_15pct": k2_reduction_pass,
        "ctx8_p95_improvement_vs_ctx4": fraction(ctx8_improvement, primary_p95),
        "ctx8_p95_improvement_at_most_3pct": ctx8_saturation_pass,
        "m42_three_x_maximum_product_cycles": maximum_product_cycles,
        "k2_primary_p95_integrated_cycles": primary_p95,
        "k2_primary_p95_below_m42_product_gate": p95_gate_pass,
        "all_kill_gates_pass": all((primary_overhead_pass, parent_wait_pass,
                                    k2_reduction_pass, ctx8_saturation_pass,
                                    p95_gate_pass)),
        "three_x_target_crossing_admitted": False,
        "reason": ("the M42 comparison is an arithmetic product gate; external "
                   "memory-system timing, RTL and same-resource integration "
                   "remain unproved"),
    }
    return {
        "schema": "m45_dual_destination_bank_fused_integrated_schedule_result_v1",
        "status": ("PASS_M45_TRANSACTION_SCHEDULE_KILL_GATES_RTL_AND_SYSTEM_UNADMITTED"
                   if gates["all_kill_gates_pass"] else
                   "NO_GO_M45_ONE_OR_MORE_TRANSACTION_SCHEDULE_KILL_GATES_FAILED"),
        "identity": {
            "contract_sha256": sha256(CONTRACT),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "upstream_sha256": dict((name, item["sha256"])
                                    for name, item in contract["inputs"].items()),
        },
        "architecture": {
            "name": "DUAL_DESTINATION_BANK_FUSED_PARENT_DELTA",
            "primary_destination_fanout_k": 2,
            "primary_resident_contexts": 4,
            "source_banks": BANKS,
            "output_lanes": LANES,
            "weight_response_latency_cycles": 1,
            "metadata_fifo_entries": METADATA_FIFO_ENTRIES,
            "complete_fifo_entries": COMPLETE_FIFO_ENTRIES,
            "loop_order": contract["fixed_schedule"]["loop_order"],
            "spatial_safety_DAG": contract["fixed_schedule"]["spatial_safety_DAG"],
            "group_policy": contract["fixed_schedule"]["group_policy"],
            "signed_source_fusion": contract["fixed_schedule"]["signed_source_fusion"],
        },
        "capacity": {
            "m43_local_scratch_bytes": LOCAL_SCRATCH_BYTES,
            "single_timestep_final_accumulator_bytes": TIMESTEP_ACC_BYTES,
            "base_local_bytes_before_fifos": BASE_LOCAL_BYTES,
            "metadata_fifo_descriptor_bytes_per_entry": METADATA_FIFO_ENTRY_BYTES,
            "metadata_fifo_storage_bytes": METADATA_FIFO_BYTES,
            "complete_fifo_vector_storage_bytes_per_entry": COMPLETE_FIFO_VECTOR_BYTES,
            "complete_fifo_tag_control_bytes_per_entry": COMPLETE_FIFO_TAG_CONTROL_BYTES,
            "complete_fifo_storage_bytes": COMPLETE_FIFO_BYTES,
            "fifo_storage_bytes": FIFO_BYTES,
            "combined_local_bytes": COMBINED_LOCAL_BYTES,
            "combined_local_bytes_reconciled": COMBINED_LOCAL_BYTES == 149248,
            "frozen_local_residency_bytes": FROZEN_LOCAL_RESIDENCY_BYTES,
            "local_capacity_headroom_bytes": LOCAL_CAPACITY_HEADROOM_BYTES,
            "timestep_frame_resident_across_all_27_tiles": True,
            "partial_frame_spill_permitted": False,
            "external_accumulator_backing_required": False,
        },
        "population": {"samples": 10, "operators": 4, "records": 40},
        "configurations": configurations,
        "kill_gates": gates,
        "qualification": {
            "exact": [
                "all 40 frozen M40 support records and M43 spatial parents",
                "K1 M43 source-cycle reproduction",
                "deterministic K2/K4 bank-source union and logical update conservation",
                "left/up safety DAG and selected-parent state reads",
                "transaction-level command, state-port, response, output and weight-DMA schedule",
                "modeled local-capacity and byte traffic"
            ],
            "not_admitted": [
                "RTL or VCS measured cycles and integer output equivalence",
                "local accumulator and weight SRAM macro/interconnect timing",
                "SRAM/DRAM energy, PPA or power",
                "full-network or end-to-end speedup",
                "external comparison, DATE headline or best-paper claim"
            ]
        },
        "claim_policy": contract["claim_policy"],
    }


def write_output(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite M45 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_output(args.output, build())
    print(args.output)


if __name__ == "__main__":
    main()
