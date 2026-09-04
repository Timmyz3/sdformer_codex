#!/usr/bin/env python3
"""Replay the pre-sealed M453a exact hierarchy once on secondary M40.

This is deliberately a non-pristine, fixed-hardware ablation.  It prices the
q32 early parent matcher, a dedicated three-way child stage, 128 PWP slots,
actual-used sparse PWP DMA, current-slot overflow, exact K1 separate issue and
the separately VCS-verified M451 K1 fused opportunity.  It never tunes the
catalog from M40.
"""

import argparse
from array import array
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path


K = 16
PARENTS = 32
CHILDREN = 3
PATTERNS = 128
PARTITIONS = 432
SAMPLES = 10
POPCOUNT = tuple(bin(value).count("1") for value in range(1 << K))
DISTANCE_FIELDS = tuple("distance{}_rows".format(value)
                        for value in range(K + 1))
ORDERED_LEDGER_FIELDS = (
    "sample", "operator", "operator_name", "partition", "run_index",
    "source_row_start", "source_row_count", "source_row_end_exclusive",
    "original_mask_hex", "selected_global_id", "parent_id", "child_slot",
    "selected_center_hex", "selected_distance", "path",
    "pwp_correction_ops_per_row", "fallback_source_ops_per_row",
    "separate_issues_per_block_per_row",
    "fused_issues_per_block_per_row")
PHASE_METRIC_FIELDS = (
    "source_rows", "zero_rows", "active_rows", "pwp_rows",
    "fallback_rows", "exact_pwp_rows", "positive_residual_pwp_rows",
    "pwp_correction_ops", "fallback_source_ops",
    "correction_ops_per_block", "separate_issues_per_block",
    "fused_k1_issues_per_block", "parent_selected_rows",
    "child_selected_rows", "used_pwp_patterns",
    "used_parent_pwp_patterns", "used_child_pwp_patterns",
    "used_center_runs", "triangle_child_comparisons_potential",
    "triangle_child_comparisons_gated",
    "triangle_child_comparisons_executed",
    "triangle_selection_mismatches", "q32_parent_matcher_cycles",
    "child_matcher_pipeline_latency_cycles", "hierarchical_matcher_cycles",
    "actual_pwp_bytes_per_tile", "actual_slot_bytes",
    "current_slot_overflow") + DISTANCE_FIELDS
CENTER_LEDGER_FIELDS = (
    "sample", "operator", "operator_name", "partition",
    "global_center_id", "parent_id", "child_slot", "center_hex",
    "parent_hex", "parent_child_hamming", "selected_rows", "pwp_rows",
    "exact_pwp_rows", "positive_residual_pwp_rows", "fallback_rows",
    "pwp_correction_ops", "fallback_source_ops",
    "separate_issues_per_block", "fused_k1_issues_per_block",
    "selected_hamming_flip_terms", "pwp_residual_flip_terms") + \
    DISTANCE_FIELDS


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
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_runs(indices):
    ordered = sorted(indices)
    return (0 if not ordered else
            1 + sum(current != previous + 1
                    for previous, current in zip(ordered, ordered[1:])))


def verify_double_seal(manifest_path, seal_path, label):
    """Recompute every inner entry and then the outer manifest seal."""
    output_dir = manifest_path.parent
    entries = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        require(line and "  " in line,
                label + " malformed inner manifest line")
        expected, name = line.split("  ", 1)
        entry_path = Path(name)
        if not entry_path.is_absolute():
            entry_path = output_dir / entry_path
        require(entry_path.is_file() and sha256(entry_path) == expected,
                label + " inner seal mismatch: " + name)
        entries += 1
    require(entries > 0, label + " empty inner manifest")
    require(seal_path.is_file(), label + " outer seal missing")
    expected, name = seal_path.read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(Path(name).name == manifest_path.name and
            sha256(manifest_path) == expected,
            label + " outer seal mismatch")
    return {
        "entries": entries,
        "inner_manifest_sha256": sha256(manifest_path),
        "outer_seal_file_sha256": sha256(seal_path),
    }


def analyze_phase(counter, partition_catalog, model):
    parents = [int(value, 16) for value in
               partition_catalog["parent_patterns"]]
    children = [[int(value, 16) for value in group]
                for group in partition_catalog["children_by_parent"]]
    flat = [int(value, 16) for value in
            partition_catalog["flat_patterns"]]
    require(len(parents) == PARENTS and len(children) == PARENTS and
            all(len(group) == CHILDREN for group in children) and
            len(flat) == PATTERNS and len(set(flat)) == PATTERNS and
            flat[:PARENTS] == parents and
            flat[PARENTS:] == [value for group in children for value in group],
            "M453b catalog geometry drift")
    result = Counter()
    used = set()
    center_ledger = defaultdict(Counter)
    descriptor_by_mask = {}
    zero_descriptor = {
            "selected_global_id": -1,
            "parent_id": -1,
            "child_slot": -1,
            "selected_center_hex": "0000",
            "selected_distance": 0,
            "path": "zero",
            "pwp_correction_ops_per_row": 0,
            "fallback_source_ops_per_row": 0,
            "separate_issues_per_block_per_row": 0,
            "fused_issues_per_block_per_row": 0,
    }
    reconstruction_mismatches = 0
    for original, population in counter.items():
        original = int(original) & 0xffff
        population = int(population)
        pop = POPCOUNT[original]
        result["source_rows"] += population
        if original == 0:
            result["zero_rows"] += population
            descriptor_by_mask[0] = dict(zero_descriptor)
            continue
        result["active_rows"] += population
        parent_distances = [POPCOUNT[original ^ center]
                            for center in parents]
        parent_id = parent_distances.index(min(parent_distances))
        local = [parents[parent_id]] + children[parent_id]
        local_distances = [POPCOUNT[original ^ center]
                           for center in local]
        local_id = local_distances.index(min(local_distances))
        center = local[local_id]
        distance = local_distances[local_id]
        global_id = (parent_id if local_id == 0 else
                     PARENTS + parent_id * CHILDREN + local_id - 1)
        require(flat[global_id] == center,
                "M453b flat/local ID mapping drift")
        # Read-only exact triangle-bound gate.  Equality is safely gated
        # because the frozen tie order favors parent/earlier child.
        triangle_best = parent_distances[parent_id]
        triangle_local_id = 0
        for child_slot, child in enumerate(children[parent_id]):
            radius = POPCOUNT[parents[parent_id] ^ child]
            lower_bound = abs(parent_distances[parent_id] - radius)
            result["triangle_child_comparisons_potential"] += population
            if lower_bound >= triangle_best:
                result["triangle_child_comparisons_gated"] += population
                continue
            result["triangle_child_comparisons_executed"] += population
            child_distance = POPCOUNT[original ^ child]
            if child_distance < triangle_best:
                triangle_best = child_distance
                triangle_local_id = child_slot + 1
        result["triangle_selection_mismatches"] += population * int(
            triangle_best != distance or triangle_local_id != local_id)
        # Exact signed residual conservation: additions and removals are the
        # disjoint XOR terms and reconstruct the original mask exactly.
        added = original & (~center & 0xffff)
        removed = center & (~original & 0xffff)
        reconstructed = (center | added) & (~removed & 0xffff)
        reconstruction_mismatches += population * int(
            reconstructed != original or
            POPCOUNT[added] + POPCOUNT[removed] != distance)
        use_pwp = 1 + distance < pop
        correction = distance if use_pwp else pop
        result["pwp_rows"] += population * int(use_pwp)
        result["fallback_rows"] += population * int(not use_pwp)
        result["exact_pwp_rows"] += population * int(
            use_pwp and distance == 0)
        result["positive_residual_pwp_rows"] += population * int(
            use_pwp and distance > 0)
        result["pwp_correction_ops"] += population * distance * int(use_pwp)
        result["fallback_source_ops"] += population * pop * int(not use_pwp)
        result["correction_ops_per_block"] += population * correction
        result["separate_issues_per_block"] += population * (
            int(use_pwp) + correction)
        # M451: exact PWP uses one issue; positive-residual PWP folds the
        # first correction into that issue; fallback remains popcount issues.
        fused = (max(1, correction) if use_pwp else correction)
        result["fused_k1_issues_per_block"] += population * fused
        result["parent_selected_rows"] += population
        result["child_selected_rows"] += population * int(local_id != 0)
        result["distance{}_rows".format(distance)] += population
        entry = center_ledger[global_id]
        entry["selected_rows"] += population
        entry["exact_pwp_rows"] += population * int(
            use_pwp and distance == 0)
        entry["positive_residual_pwp_rows"] += population * int(
            use_pwp and distance > 0)
        entry["fallback_rows"] += population * int(not use_pwp)
        entry["pwp_correction_ops"] += population * distance * int(use_pwp)
        entry["fallback_source_ops"] += population * pop * int(not use_pwp)
        entry["separate_issues_per_block"] += population * (
            int(use_pwp) + correction)
        entry["fused_k1_issues_per_block"] += population * fused
        entry["distance{}_rows".format(distance)] += population
        entry["selected_hamming_flip_terms"] += population * distance
        if use_pwp:
            used.add(global_id)
            entry["pwp_rows"] += population
            entry["pwp_residual_flip_terms"] += population * distance
        descriptor_by_mask[original] = {
            "selected_global_id": global_id,
            "parent_id": parent_id,
            "child_slot": (-1 if local_id == 0 else local_id - 1),
            "selected_center_hex": "{:04x}".format(center),
            "selected_distance": distance,
            "path": ("exact_pwp" if use_pwp and distance == 0 else
                     "positive_residual_pwp" if use_pwp else "fallback"),
            "pwp_correction_ops_per_row": distance if use_pwp else 0,
            "fallback_source_ops_per_row": pop if not use_pwp else 0,
            "separate_issues_per_block_per_row":
                int(use_pwp) + correction,
            "fused_issues_per_block_per_row": fused,
        }
    result["used_pwp_patterns"] = len(used)
    result["used_center_runs"] = count_runs(used)
    result["reconstruction_mismatches"] = reconstruction_mismatches
    result["q32_early_extra_prefix_tasks"] = sum(
        population for original, population in counter.items()
        if POPCOUNT[int(original) & 0xffff] >= 2 and
        (int(original) & 0xffff) not in set(parents[:16]))
    result["q32_parent_matcher_cycles"] = (
        model["rows_per_phase"] +
        result["q32_early_extra_prefix_tasks"] +
        model["q32_parent_pipeline_overhead_cycles"])
    result["child_matcher_pipeline_latency_cycles"] = (
        model["child_matcher_pipeline_latency_cycles"])
    result["hierarchical_matcher_cycles"] = (
        result["q32_parent_matcher_cycles"] +
        result["child_matcher_pipeline_latency_cycles"])
    result["actual_pwp_bytes_per_tile"] = (
        result["used_pwp_patterns"] * model["pwp_stride_bytes"])
    result["actual_slot_bytes"] = (
        model["hierarchical_config_bytes"] +
        model["weight_bytes_per_tile"] +
        result["actual_pwp_bytes_per_tile"])
    result["current_slot_overflow"] = int(
        result["actual_slot_bytes"] > model["current_tile_slot_bytes"])
    result["used_parent_pwp_patterns"] = sum(
        int(global_id < PARENTS) for global_id in used)
    result["used_child_pwp_patterns"] = sum(
        int(global_id >= PARENTS) for global_id in used)
    require(result["source_rows"] == model["rows_per_phase"] and
            result["source_rows"] ==
            result["zero_rows"] + result["active_rows"] and
            result["active_rows"] ==
            result["pwp_rows"] + result["fallback_rows"] and
            result["pwp_rows"] ==
            result["exact_pwp_rows"] +
            result["positive_residual_pwp_rows"] and
            result["correction_ops_per_block"] ==
            result["pwp_correction_ops"] +
            result["fallback_source_ops"] and
            result["separate_issues_per_block"] ==
            result["pwp_rows"] + result["correction_ops_per_block"] and
            result["fused_k1_issues_per_block"] ==
            result["separate_issues_per_block"] -
            result["positive_residual_pwp_rows"] and
            reconstruction_mismatches == 0 and
            result["triangle_child_comparisons_potential"] ==
            result["active_rows"] * CHILDREN and
            result["triangle_child_comparisons_potential"] ==
            result["triangle_child_comparisons_gated"] +
            result["triangle_child_comparisons_executed"] and
            result["triangle_selection_mismatches"] == 0,
            "M453b phase exactness/conservation failure")
    ledger_rows = []
    for global_id in sorted(center_ledger):
        parent_id = (global_id if global_id < PARENTS else
                     (global_id - PARENTS) // CHILDREN)
        child_slot = (-1 if global_id < PARENTS else
                      (global_id - PARENTS) % CHILDREN)
        parent_child_hamming = (0 if child_slot < 0 else
            POPCOUNT[parents[parent_id] ^ children[parent_id][child_slot]])
        entry = center_ledger[global_id]
        ledger_rows.append({
            "global_center_id": global_id,
            "parent_id": parent_id,
            "child_slot": child_slot,
            "center_hex": "{:04x}".format(flat[global_id]),
            "parent_hex": "{:04x}".format(parents[parent_id]),
            "parent_child_hamming": parent_child_hamming,
            "selected_rows": entry["selected_rows"],
            "pwp_rows": entry["pwp_rows"],
            "exact_pwp_rows": entry["exact_pwp_rows"],
            "positive_residual_pwp_rows":
                entry["positive_residual_pwp_rows"],
            "fallback_rows": entry["fallback_rows"],
            "pwp_correction_ops": entry["pwp_correction_ops"],
            "fallback_source_ops": entry["fallback_source_ops"],
            "separate_issues_per_block":
                entry["separate_issues_per_block"],
            "fused_k1_issues_per_block":
                entry["fused_k1_issues_per_block"],
            "selected_hamming_flip_terms":
                entry["selected_hamming_flip_terms"],
            "pwp_residual_flip_terms":
                entry["pwp_residual_flip_terms"],
            **{"distance{}_rows".format(distance):
               entry["distance{}_rows".format(distance)]
               for distance in range(K + 1)},
        })
    require(len(descriptor_by_mask) == len(counter),
            "M453b descriptor map does not cover phase masks")
    return dict(result), ledger_rows, descriptor_by_mask


def replay_sample(phases, mode, model):
    require(mode in ("separate", "m451_fused_opportunity"),
            "unknown M453b replay mode")
    time = 0
    components = Counter()
    maximum_actual_slot = 0
    overflow_phases = 0
    for phase in phases:
        config_data = int(math.ceil(
            model["hierarchical_config_bytes"] /
            float(model["dram_bytes_per_cycle"])))
        time += (config_data + model["dma_command_setup_cycles"] +
                 phase["hierarchical_matcher_cycles"] +
                 model["bitmap_seal_cycles"])
        components["config_data"] += config_data
        components["config_command"] += model["dma_command_setup_cycles"]
        components["q32_parent_matcher"] += phase[
            "q32_parent_matcher_cycles"]
        components["child_matcher_pipeline_latency"] += phase[
            "child_matcher_pipeline_latency_cycles"]
        components["bitmap_seal"] += model["bitmap_seal_cycles"]
        maximum_actual_slot = max(maximum_actual_slot,
                                  phase["actual_slot_bytes"])
        overflow_phases += phase["current_slot_overflow"]
        if phase["active_rows"] == 0:
            time += model["tail_cycles"]
            components["tail"] += model["tail_cycles"]
            continue
        tile_bytes = (model["weight_bytes_per_tile"] +
                      phase["actual_pwp_bytes_per_tile"])
        require(model["hierarchical_config_bytes"] + tile_bytes <=
                model["expanded_q128_tile_slot_bytes"],
                "M453b expanded q128 slot overflow")
        require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                "M453b tile DMA alignment drift")
        tile_data = tile_bytes // model["dram_bytes_per_cycle"]
        tile_commands = 1 + phase["used_center_runs"]
        tile_dma = (tile_data + tile_commands *
                    model["dma_command_setup_cycles"])
        issue_key = ("separate_issues_per_block" if mode == "separate"
                     else "fused_k1_issues_per_block")
        work = model["output_blocks_per_tile"] * phase[issue_key]
        replay0 = work + model["descriptor_sram_latency_cycles"]
        replay1 = work + model["descriptor_sram_latency_cycles"]
        time += tile_dma
        tile0_end = time + replay0
        tile1_dma_end = time + tile_dma
        tile1_start = max(tile0_end, tile1_dma_end)
        tile1_exposed = max(0, tile1_dma_end - tile0_end)
        time = tile1_start + replay1 + model["tail_cycles"]
        components["tile0_dma_data"] += tile_data
        components["tile0_dma_commands"] += (
            tile_commands * model["dma_command_setup_cycles"])
        components["tile1_dma_exposed"] += tile1_exposed
        components["replay0"] += replay0
        components["replay1"] += replay1
        components["active_compute"] += 2 * work
        components["descriptor_sram_startup"] += (
            2 * model["descriptor_sram_latency_cycles"])
        components["tail"] += model["tail_cycles"]
        components["actual_pwp_dram_bytes"] += (
            phase["actual_pwp_bytes_per_tile"] * 2)
        components["weight_dram_bytes"] += (
            model["weight_bytes_per_tile"] * 2)
        components["used_pwp_pattern_slots_across_tiles"] += (
            phase["used_pwp_patterns"] * 2)
    time += model["commit_cycles_per_sample"]
    components["commit"] += model["commit_cycles_per_sample"]
    return {"cycles": int(time), "components": dict(components),
            "maximum_actual_slot_bytes": maximum_actual_slot,
            "current_slot_overflow_phases": overflow_phases}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_ordered_runs(writer, ordered_masks, descriptor_by_mask,
                       sample, operator, operator_name, partition):
    """Write an exact run-length reconstruction of source-row issue order."""
    require(len(ordered_masks) > 0, "M453b empty ordered phase")
    summary = Counter()
    start = 0
    run_index = 0
    while start < len(ordered_masks):
        original = int(ordered_masks[start])
        end = start + 1
        while end < len(ordered_masks) and int(ordered_masks[end]) == original:
            end += 1
        count = end - start
        require(original in descriptor_by_mask,
                "M453b ordered mask missing descriptor")
        descriptor = descriptor_by_mask[original]
        writer.writerow({
            "sample": sample,
            "operator": operator,
            "operator_name": operator_name,
            "partition": partition,
            "run_index": run_index,
            "source_row_start": start,
            "source_row_count": count,
            "source_row_end_exclusive": end,
            "original_mask_hex": "{:04x}".format(original),
            **descriptor,
        })
        summary["runs"] += 1
        summary["source_rows"] += count
        summary[descriptor["path"] + "_rows"] += count
        summary["pwp_correction_ops"] += (
            count * descriptor["pwp_correction_ops_per_row"])
        summary["fallback_source_ops"] += (
            count * descriptor["fallback_source_ops_per_row"])
        summary["separate_issues_per_block"] += (
            count * descriptor["separate_issues_per_block_per_row"])
        summary["fused_k1_issues_per_block"] += (
            count * descriptor["fused_issues_per_block_per_row"])
        if descriptor["path"] != "zero":
            summary["distance{}_rows".format(
                descriptor["selected_distance"])] += count
        start = end
        run_index += 1
    require(summary["source_rows"] == len(ordered_masks),
            "M453b ordered run coverage drift")
    return summary


def write_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(output_dir / name), name)
        for name in sorted(names)), encoding="utf-8")
    seal = output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")
    return manifest, seal


def run_prefreeze_selfcheck(args, root, contract, catalog, model,
                            seal_results, h1, source_start):
    """Exercise the frozen ledgers without opening the M40 manifest/payload."""
    partition_catalog = catalog["operators"][0]["partitions"][0]
    flat = [int(value, 16) for value in partition_catalog["flat_patterns"]]
    ordered = array("H")
    ordered.extend([0] * 300)
    for center in flat[:16]:
        ordered.extend([center] * 20)
    for bit in range(K):
        ordered.extend([(1 << bit)] * 20)
    remaining = model["rows_per_phase"] - len(ordered)
    require(remaining > 0, "M453b micro construction overflow")
    state = 0x453b
    for _ in range(remaining):
        state = (state * 25173 + 13849) & 0xffff
        ordered.append(state)
    counter = Counter(ordered)
    phase, center_rows, descriptors = analyze_phase(
        counter, partition_catalog, model)
    require(phase["exact_pwp_rows"] > 0 and
            phase["positive_residual_pwp_rows"] > 0 and
            phase["fallback_rows"] > 0 and
            phase["child_selected_rows"] > 0,
            "M453b micro did not cover required paths")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ordered_name = "m453b_final_freeze_micro_ordered_runs.csv"
    ordered_path = args.output_dir / ordered_name
    with ordered_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ORDERED_LEDGER_FIELDS)
        writer.writeheader()
        ordered_summary = write_ordered_runs(
            writer, ordered, descriptors, 0, 0,
            catalog["operators"][0]["operator"], 0)
    require(ordered_summary["source_rows"] == phase["source_rows"] and
            ordered_summary["pwp_correction_ops"] ==
            phase["pwp_correction_ops"] and
            ordered_summary["fallback_source_ops"] ==
            phase["fallback_source_ops"] and
            ordered_summary["separate_issues_per_block"] ==
            phase["separate_issues_per_block"] and
            ordered_summary["fused_k1_issues_per_block"] ==
            phase["fused_k1_issues_per_block"] and
            all(ordered_summary[field] == phase.get(field, 0)
                for field in DISTANCE_FIELDS),
            "M453b micro ordered ledger conservation drift")
    center_summary = Counter()
    for row in center_rows:
        for key in ("selected_rows", "exact_pwp_rows",
                    "positive_residual_pwp_rows", "fallback_rows",
                    "pwp_correction_ops", "fallback_source_ops",
                    "separate_issues_per_block",
                    "fused_k1_issues_per_block") + DISTANCE_FIELDS:
            center_summary[key] += row[key]
    require(center_summary["selected_rows"] == phase["active_rows"] and
            center_summary["exact_pwp_rows"] == phase["exact_pwp_rows"] and
            center_summary["positive_residual_pwp_rows"] ==
            phase["positive_residual_pwp_rows"] and
            center_summary["fallback_rows"] == phase["fallback_rows"] and
            center_summary["pwp_correction_ops"] ==
            phase["pwp_correction_ops"] and
            center_summary["fallback_source_ops"] ==
            phase["fallback_source_ops"] and
            center_summary["separate_issues_per_block"] ==
            phase["separate_issues_per_block"] and
            center_summary["fused_k1_issues_per_block"] ==
            phase["fused_k1_issues_per_block"] and
            all(center_summary[field] == phase.get(field, 0)
                for field in DISTANCE_FIELDS),
            "M453b micro center ledger conservation drift")
    receipt = {
        "schema": "m453b_final_freeze_micro_receipt_v1",
        "status": "PASS_M453B_FINAL_FREEZE_MICRO_M40_NOT_READ",
        "identity": {
            "analyzer_sha256": source_start,
            "final_contract_path": str(args.contract.resolve().relative_to(root)),
            "final_contract_sha256": sha256(args.contract),
            "docs359_sha256": sha256(root / "docs/359_DATE终局冻结_20260813.md"),
        },
        "h1_authorization": {
            "status": h1["status"],
            "score": h1["score"],
            "p0": len(h1["findings"]["P0"]),
            "p1": len(h1["findings"]["P1"]),
            "p2": len(h1["findings"]["P2"]),
            "authorization": h1["decision"]["authorization"],
        },
        "upstream_double_seals_recomputed": seal_results,
        "directed_micro": {
            "source_rows": phase["source_rows"],
            "active_rows": phase["active_rows"],
            "exact_pwp_rows": phase["exact_pwp_rows"],
            "positive_residual_pwp_rows":
                phase["positive_residual_pwp_rows"],
            "fallback_rows": phase["fallback_rows"],
            "pwp_correction_ops": phase["pwp_correction_ops"],
            "fallback_source_ops": phase["fallback_source_ops"],
            "separate_issues_per_block":
                phase["separate_issues_per_block"],
            "fused_k1_issues_per_block":
                phase["fused_k1_issues_per_block"],
            "center_ledger_rows": len(center_rows),
            "ordered_run_rows": ordered_summary["runs"],
            "ordered_source_rows": ordered_summary["source_rows"],
            "triangle_selection_mismatches":
                phase["triangle_selection_mismatches"],
            "reconstruction_mismatches":
                phase["reconstruction_mismatches"],
            "center_and_ordered_conservation_mismatches": 0,
            "distance_histogram": {
                field: phase.get(field, 0) for field in DISTANCE_FIELDS},
        },
        "execution": {
            "m40_manifest_read": False,
            "m40_payload_reads": 0,
            "m40_completed_phase_replays": 0,
            "final_contract_frozen": True,
        },
        "claim_boundary": {
            "final_freeze_micro_only": True,
            "heldout_cycles": False,
            "resource_normalized_speedup": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    receipt_name = "m453b_final_freeze_micro_receipt_r1.json"
    receipt_path = args.output_dir / receipt_name
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    require(source_start == sha256(Path(__file__).resolve()),
            "M453b analyzer changed during final-freeze micro")
    _, seal = write_seal(args.output_dir, [ordered_name, receipt_name])
    print("{} rows={} ordered_runs={} seal={}".format(
        receipt["status"], phase["source_rows"], ordered_summary["runs"],
        sha256(seal)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefreeze-self-check", action="store_true",
                        help="exercise ledgers without reading M40")
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M453b output overwrite")
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m453b_h67_hierarchical_q32x3_secondary_replay_contract_v1" and
            contract.get("status") ==
            "FROZEN_AFTER_M453A_DOUBLE_SEAL_SECONDARY_M40_ONCE",
            "M453b contract status drift")
    root = args.contract.resolve().parents[1]
    paths = {name: root / spec["path"]
             for name, spec in contract["inputs"].items()}
    identities = {name: {"path": spec["path"],
                         "sha256": spec["sha256"]}
                  for name, spec in contract["inputs"].items()}
    for name, spec in contract["inputs"].items():
        if args.prefreeze_self_check and name == "m40_trace":
            continue
        path = paths[name]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M453b input SHA drift: " + name)
    require(paths["analyzer"].resolve() == Path(__file__).resolve() and
            identities["analyzer"]["sha256"] == source_start,
            "M453b analyzer self identity drift")
    seal_results = {
        "m453a": verify_double_seal(
            paths["m453a_manifest"], paths["m453a_seal"], "M453a"),
        "m453a_h1": verify_double_seal(
            paths["m453a_h1_manifest"], paths["m453a_h1_seal"],
            "M453a H1"),
        "m430": verify_double_seal(
            paths["m430_manifest"], paths["m430_seal"], "M430"),
        "m451": verify_double_seal(
            paths["m451_manifest"], paths["m451_seal"], "M451"),
        "m455": verify_double_seal(
            paths["m455_manifest"], paths["m455_seal"], "M455"),
        "m457": verify_double_seal(
            paths["m457_manifest"], paths["m457_seal"], "M457"),
    }

    catalog = strict_json(paths["m453a_catalog"])
    train_audit = strict_json(paths["m453a_audit"])
    h1 = strict_json(paths["m453a_h1_receipt"])
    m430 = strict_json(paths["m430_result"])
    m451 = strict_json(paths["m451_receipt"])
    m455 = strict_json(paths["m455_dc_receipt"])
    m457 = strict_json(paths["m457_review"])
    require(catalog["status"] ==
            "PASS_M453A_R3_VECTORIZED_TRAIN_ONLY_TREE_FROZEN_BEFORE_M40" and
            catalog["schema"] ==
            "m453a_trainonly_hierarchical_q32x3_catalog_v3" and
            train_audit["status"] ==
            "PASS_M453A_R3_DOUBLE_SEAL_READY_M40_NOT_READ" and
            train_audit["schema"] ==
            "m453a_trainonly_hierarchical_q32x3_catalog_audit_v3" and
            train_audit["heldout_gate"]["m40_payload_reads_so_far"] == 0 and
            train_audit["heldout_gate"]["post_m40_tuning_allowed"] is False,
            "M453b train seal/heldout gate drift")
    require(h1["status"] ==
            "PASS_GO_EXACTLY_ONE_FIXED_M453B_SECONDARY_REPLAY" and
            h1["schema"] == "m453a_h1_independent_audit_receipt_r1" and
            h1["score"] == 100 and h1["decision"]["go"] is True and
            h1["decision"]["authorization"] ==
            "exactly one fixed M453b secondary M40 replay" and
            h1["immutability"]["m40_payload_reads"] == 0 and
            h1["immutability"]["m453b_executions"] == 0 and
            all(len(h1["findings"][severity]) == 0
                for severity in ("P0", "P1", "P2")),
            "M453b H1 authorization drift")
    require(m430["comparisons"]["strong_zero_cycles"] ==
            contract["comparisons"]["strong_zero_cycles"] and
            m430["comparisons"]["m430_catalog_dual_cycles"] ==
            contract["comparisons"]["m430_cycles"],
            "M453b M430 comparison drift")
    require(m451["status"] ==
            "PASS_M451_EXACT_K1_FUSED_PWP_CORRECTION_DIRECTED_VCS" and
            m451["claim_boundary"]["cycle_opportunity_only"] is True and
            m451["claim_boundary"]["resource_normalized_speedup"] is False,
            "M453b M451 receipt boundary drift")
    require(m455["status"] == "PASS_M455_M451_STANDALONE_3NS_DC" and
            m455["comparison"]
            ["standalone_adapter_opportunity_throughput_per_area_ratio"] < 1.0 and
            m457["status"] ==
            "PASS_RAW_STANDALONE_DC_BUT_NO_GO_PERFORMANCE_FORMALITY_AND_PT" and
            m457["decision"]["m451_date_performance_mainline"] ==
            "NO_GO_NEGATIVE_STANDALONE_OPPORTUNITY_PER_AREA",
            "M453b M455/M457 resource-kill boundary drift")
    model = contract["cycle_model"]
    require(model["expanded_q128_tile_slot_bytes"] ==
            model["hierarchical_config_bytes"] +
            model["weight_bytes_per_tile"] +
            PATTERNS * model["pwp_stride_bytes"],
            "M453b q128 capacity equation drift")
    if args.prefreeze_self_check:
        return run_prefreeze_selfcheck(
            args, root, contract, catalog, model, seal_results, h1,
            source_start)

    require(paths["m40_trace"].is_file() and
            sha256(paths["m40_trace"]) ==
            contract["inputs"]["m40_trace"]["sha256"],
            "M453b M40 manifest SHA drift")
    trace = strict_json(paths["m40_trace"])
    require(trace["identity"]["checkpoint_sha256"] ==
            contract["paper_identity"]["checkpoint_sha256"] and
            trace["identity"]["bn_policy"] ==
            contract["paper_identity"]["bn_policy"],
            "M453b paper identity drift")
    operators = tuple(trace["cohort"]["operators"])
    operator_index = {name: index for index, name in enumerate(operators)}
    require([item["operator"] for item in catalog["operators"]] ==
            list(operators), "M453b operator order drift")
    m43 = load_module(paths["m43_unpacker"], "m453b_m43")
    trace_dir = paths["m40_trace"].parent
    phases = defaultdict(list)
    aggregate = Counter()
    center_aggregate = Counter()
    ordered_aggregate = Counter()
    payload_files = 0
    payload_bytes = 0
    packed_unpack_rereads = 0
    phase_count = 0
    center_ledger_row_count = 0
    seen_record_keys = set()
    maximum_runtime_used_patterns = 0
    maximum_actual_slot_bytes = 0
    current_slot_overflow_phases = 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    phase_path = args.output_dir / "m453b_per_phase_secondary_replay.csv"
    center_path = args.output_dir / (
        "m453b_per_phase_selected_center_materialization_ledger.csv")
    ordered_path = args.output_dir / (
        "m453b_ordered_selected_id_descriptor_runs.csv")
    with phase_path.open("w", encoding="utf-8", newline="") as phase_handle, \
            center_path.open("w", encoding="utf-8", newline="") as center_handle, \
            ordered_path.open("w", encoding="utf-8", newline="") as ordered_handle:
        phase_writer = csv.DictWriter(
            phase_handle,
            fieldnames=("sample", "operator", "operator_name", "partition") +
            PHASE_METRIC_FIELDS)
        center_writer = csv.DictWriter(
            center_handle, fieldnames=CENTER_LEDGER_FIELDS)
        ordered_writer = csv.DictWriter(
            ordered_handle, fieldnames=ORDERED_LEDGER_FIELDS)
        phase_writer.writeheader()
        center_writer.writeheader()
        ordered_writer.writeheader()
        for record_index, record in enumerate(trace["records"]):
            sample = int(record["sample_id"])
            operator_name = record["operator"]
            require(operator_name in operator_index,
                    "M453b unknown trace operator")
            op = operator_index[operator_name]
            require(0 <= sample < SAMPLES and (sample, op) not in
                    seen_record_keys, "M453b duplicate/out-of-range record")
            seen_record_keys.add((sample, op))
            counters = [Counter() for _ in range(PARTITIONS)]
            ordered_masks = [array("H") for _ in range(PARTITIONS)]
            for key, hash_key in (("packed_file", "packed_file_sha256"),
                                  ("value_payload_file",
                                   "value_payload_sha256")):
                path = trace_dir / record[key]
                require(path.is_file() and sha256(path) == record[hash_key],
                        "M453b M40 payload identity drift")
                payload_files += 1
                payload_bytes += path.stat().st_size
            masks = m43.unpack_record_masks(trace_dir, record)
            packed_unpack_rereads += 1
            for source_row in range(m43.ROWS):
                base = source_row * m43.TILES
                for tile in range(m43.TILES):
                    value256 = masks[base + tile]
                    for subtile in range(16):
                        partition = tile * 16 + subtile
                        value = (value256 >> (subtile * 16)) & 0xffff
                        counters[partition][value] += 1
                        ordered_masks[partition].append(value)
            for partition in range(PARTITIONS):
                require(len(ordered_masks[partition]) ==
                        model["rows_per_phase"],
                        "M453b ordered phase extent drift")
                phase, center_rows, descriptors = analyze_phase(
                    counters[partition],
                    catalog["operators"][op]["partitions"][partition],
                    model)
                phases[sample].append(phase)
                aggregate.update(phase)
                phase_count += 1
                maximum_runtime_used_patterns = max(
                    maximum_runtime_used_patterns,
                    phase["used_pwp_patterns"])
                maximum_actual_slot_bytes = max(
                    maximum_actual_slot_bytes, phase["actual_slot_bytes"])
                current_slot_overflow_phases += phase["current_slot_overflow"]
                phase_writer.writerow({
                    "sample": sample,
                    "operator": op,
                    "operator_name": operator_name,
                    "partition": partition,
                    **{key: phase.get(key, 0)
                       for key in PHASE_METRIC_FIELDS},
                })
                phase_center = Counter()
                for row in center_rows:
                    center_writer.writerow({
                        "sample": sample,
                        "operator": op,
                        "operator_name": operator_name,
                        "partition": partition,
                        **row,
                    })
                    center_ledger_row_count += 1
                    for key in ("selected_rows", "exact_pwp_rows",
                                "positive_residual_pwp_rows", "fallback_rows",
                                "pwp_correction_ops", "fallback_source_ops",
                                "separate_issues_per_block",
                                "fused_k1_issues_per_block") + DISTANCE_FIELDS:
                        phase_center[key] += row[key]
                        center_aggregate[key] += row[key]
                ordered_summary = write_ordered_runs(
                    ordered_writer, ordered_masks[partition], descriptors,
                    sample, op, operator_name, partition)
                ordered_aggregate.update(ordered_summary)
                require(
                    phase_center["selected_rows"] == phase["active_rows"] and
                    phase_center["exact_pwp_rows"] ==
                    phase["exact_pwp_rows"] and
                    phase_center["positive_residual_pwp_rows"] ==
                    phase["positive_residual_pwp_rows"] and
                    phase_center["fallback_rows"] == phase["fallback_rows"] and
                    phase_center["pwp_correction_ops"] ==
                    phase["pwp_correction_ops"] and
                    phase_center["fallback_source_ops"] ==
                    phase["fallback_source_ops"] and
                    phase_center["separate_issues_per_block"] ==
                    phase["separate_issues_per_block"] and
                    phase_center["fused_k1_issues_per_block"] ==
                    phase["fused_k1_issues_per_block"] and
                    ordered_summary["source_rows"] == phase["source_rows"] and
                    ordered_summary["zero_rows"] == phase["zero_rows"] and
                    ordered_summary["exact_pwp_rows"] ==
                    phase["exact_pwp_rows"] and
                    ordered_summary["positive_residual_pwp_rows"] ==
                    phase["positive_residual_pwp_rows"] and
                    ordered_summary["fallback_rows"] == phase["fallback_rows"] and
                    ordered_summary["pwp_correction_ops"] ==
                    phase["pwp_correction_ops"] and
                    ordered_summary["fallback_source_ops"] ==
                    phase["fallback_source_ops"] and
                    ordered_summary["separate_issues_per_block"] ==
                    phase["separate_issues_per_block"] and
                    ordered_summary["fused_k1_issues_per_block"] ==
                    phase["fused_k1_issues_per_block"] and
                    all(phase_center[field] == phase.get(field, 0) and
                        ordered_summary[field] == phase.get(field, 0)
                        for field in DISTANCE_FIELDS),
                    "M453b phase center/ordered ledger conservation drift")
            print("[M453B RECORD] {}/{} sample={} op={}".format(
                record_index + 1, len(trace["records"]), sample,
                operator_name), flush=True)
    require(seen_record_keys ==
            set((sample, op) for sample in range(SAMPLES)
                for op in range(len(operators))) and
            phase_count == 17280 and
            center_ledger_row_count > 0 and
            aggregate["source_rows"] == 51840000 and
            aggregate["reconstruction_mismatches"] == 0 and
            aggregate["triangle_selection_mismatches"] == 0 and
            center_aggregate["selected_rows"] == aggregate["active_rows"] and
            ordered_aggregate["source_rows"] == aggregate["source_rows"] and
            ordered_aggregate["zero_rows"] == aggregate["zero_rows"] and
            all(center_aggregate[key] == aggregate[key]
                for key in ("exact_pwp_rows",
                            "positive_residual_pwp_rows", "fallback_rows",
                            "pwp_correction_ops", "fallback_source_ops",
                            "separate_issues_per_block",
                            "fused_k1_issues_per_block")) and
            all(ordered_aggregate[key] == aggregate[key]
                for key in ("exact_pwp_rows",
                            "positive_residual_pwp_rows", "fallback_rows",
                            "pwp_correction_ops", "fallback_source_ops",
                            "separate_issues_per_block",
                            "fused_k1_issues_per_block")) and
            all(center_aggregate[field] == aggregate.get(field, 0) and
                ordered_aggregate[field] == aggregate.get(field, 0)
                for field in DISTANCE_FIELDS),
            "M453b aggregate extent/exactness drift")

    replays = {}
    for mode in ("separate", "m451_fused_opportunity"):
        samples = [replay_sample(phases[sample], mode, model)
                   for sample in range(SAMPLES)]
        components = Counter()
        for row in samples:
            components.update(row["components"])
        replays[mode] = {
            "cycles": sum(row["cycles"] for row in samples),
            "components": dict(components),
            "maximum_actual_slot_bytes": max(
                row["maximum_actual_slot_bytes"] for row in samples),
            "current_slot_overflow_phases": sum(
                row["current_slot_overflow_phases"] for row in samples),
        }
    require(replays["separate"]["components"]["active_compute"] -
            replays["m451_fused_opportunity"]["components"]["active_compute"] ==
            aggregate["positive_residual_pwp_rows"] *
            model["output_blocks"],
            "M453b M451 fused saving conservation drift")

    strong = contract["comparisons"]["strong_zero_cycles"]
    m430_cycles = contract["comparisons"]["m430_cycles"]
    separate_cycles = replays["separate"]["cycles"]
    fused_cycles = replays["m451_fused_opportunity"]["cycles"]
    separate_vs_m430 = m430_cycles / float(separate_cycles)
    fused_vs_strong = strong / float(fused_cycles)
    thresholds = contract["decision_rule"]
    pass_separate = separate_vs_m430 >= thresholds[
        "minimum_separate_speedup_vs_m430"]
    pass_fused = fused_vs_strong >= thresholds[
        "minimum_fused_speedup_vs_strong_zero"]
    # The historical fused threshold is retained but resource-killed by the
    # sealed M455/M457 standalone result.  It cannot promote matcher RTL.
    go_m461 = pass_separate

    result = {
        "schema": "m453b_h67_hierarchical_q32x3_secondary_replay_v1",
        "status": ("PASS_M453B_GO_M461_MATCHED_RESOURCE_SCREEN" if
                   go_m461 else
                   "PASS_M453B_TREE_SEPARATE_BELOW_GATE_NO_GO"),
        "identity": identities,
        "paper_identity": contract["paper_identity"],
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "secondary_fixed_hardware_ablation": {
            "pristine_heldout": False,
            "m40_previously_consumed_by_upstream_milestones": True,
            "catalog_frozen_and_double_sealed_before_this_replay": True,
            "post_m40_catalog_or_cycle_model_tuning": False,
            "completed_17280_phase_replays_this_milestone": 1,
            "one_fixed_secondary_replay": True,
            "exact_once_physical_reader": False,
            "payload_hash_file_reads": payload_files,
            "packed_payload_unpack_rereads": packed_unpack_rereads,
            "physical_access_disclosure":
                "80 payload hash reads plus 40 packed-file unpack rereads; this is one fixed phase replay, not exact-once physical I/O",
        },
        "catalog_and_matcher": {
            "q32_parent_bit_identical_to_m430": True,
            "children_per_parent": CHILDREN,
            "total_pwp_capacity": PATTERNS,
            "maximum_logical_comparisons_per_active_row": 35,
            "parent_matcher":
                "M430 serial16 q32 early-zero-stop task stream",
            "child_matcher":
                "three parallel 16-bit comparators, II1, two-cycle pipeline drain charged after parent stage",
            "matcher_stages_overlap_across_rows": True,
            "child_stage_throughput_not_slower_than_parent_stage": True,
            "q32_parent_matcher_cycles":
                aggregate["q32_parent_matcher_cycles"],
            "child_matcher_pipeline_latency_cycles":
                aggregate["child_matcher_pipeline_latency_cycles"],
            "hierarchical_matcher_cycles":
                aggregate["hierarchical_matcher_cycles"],
        },
        "triangle_bound_child_comparator_gating_diagnostic": {
            "formula":
                "gate child when abs(H(x,parent)-H(parent,child)) >= current_best",
            "fixed_child_order": True,
            "equality_gate_safe_due_to_earlier_center_tie": True,
            "potential_comparisons":
                aggregate["triangle_child_comparisons_potential"],
            "executed_comparisons":
                aggregate["triangle_child_comparisons_executed"],
            "gated_comparisons":
                aggregate["triangle_child_comparisons_gated"],
            "gated_fraction":
                aggregate["triangle_child_comparisons_gated"] /
                float(aggregate["triangle_child_comparisons_potential"]),
            "selection_mismatches":
                aggregate["triangle_selection_mismatches"],
            "changes_main_matcher_cycles": False,
            "energy_or_clock_gating_opportunity_only": True,
        },
        "population": {
            key: aggregate[key] for key in (
                "source_rows", "zero_rows", "active_rows", "pwp_rows",
                "fallback_rows", "exact_pwp_rows",
                "positive_residual_pwp_rows", "child_selected_rows",
                "pwp_correction_ops", "fallback_source_ops",
                "correction_ops_per_block", "separate_issues_per_block",
                "fused_k1_issues_per_block")
        },
        "selected_distance_histogram": {
            field: aggregate[field] for field in DISTANCE_FIELDS},
        "cycles": {
            "strong_zero": strong,
            "m430_q32_separate": m430_cycles,
            "m453_tree_separate": separate_cycles,
            "m453_tree_m451_fused_opportunity": fused_cycles,
        },
        "comparisons": {
            "tree_separate_speedup_vs_strong_zero":
                strong / float(separate_cycles),
            "tree_separate_speedup_vs_m430": separate_vs_m430,
            "tree_fused_speedup_vs_strong_zero": fused_vs_strong,
            "tree_fused_speedup_vs_m430":
                m430_cycles / float(fused_cycles),
        },
        "component_ledgers": replays,
        "pwp_capacity_dma_and_cache": {
            "pwp_stride_bytes_per_four_output_blocks":
                model["pwp_stride_bytes"],
            "static_q128_pwp_bytes_per_tile":
                PATTERNS * model["pwp_stride_bytes"],
            "static_q128_pwp_bytes_two_tiles":
                2 * PATTERNS * model["pwp_stride_bytes"],
            "hierarchical_config_bytes":
                model["hierarchical_config_bytes"],
            "expanded_q128_tile_slot_bytes_each":
                model["expanded_q128_tile_slot_bytes"],
            "two_expanded_slots_bytes":
                2 * model["expanded_q128_tile_slot_bytes"],
            "current_tile_slot_bytes_each":
                model["current_tile_slot_bytes"],
            "additional_two_slot_capacity_bytes":
                2 * (model["expanded_q128_tile_slot_bytes"] -
                     model["current_tile_slot_bytes"]),
            "current_slot_actual_working_set_capacity_patterns":
                (model["current_tile_slot_bytes"] -
                 model["hierarchical_config_bytes"] -
                 model["weight_bytes_per_tile"]) //
                model["pwp_stride_bytes"],
            "maximum_runtime_used_patterns_per_phase":
                maximum_runtime_used_patterns,
            "maximum_actual_slot_bytes": maximum_actual_slot_bytes,
            "current_slot_overflow_phases": current_slot_overflow_phases,
            "actual_used_pwp_dram_bytes_separate":
                replays["separate"]["components"]
                ["actual_pwp_dram_bytes"],
            "full_q128_capacity_not_all_dma_loaded_each_phase": True,
            "expanded_storage_area_power_charged_in_cycle_result": False,
        },
        "immutable_m461_ledgers": {
            "phase_csv": phase_path.name,
            "center_csv": center_path.name,
            "ordered_selected_id_descriptor_runs_csv": ordered_path.name,
            "phase_rows": phase_count,
            "center_rows": center_ledger_row_count,
            "ordered_run_rows": ordered_aggregate["runs"],
            "ordered_source_rows": ordered_aggregate["source_rows"],
            "ordered_reconstruction":
                "per phase, expand each run from source_row_start for source_row_count using original_mask_hex and the bound selected-ID/descriptor/path fields",
            "ordered_phase_source_row_extent": model["rows_per_phase"],
            "center_and_ordered_population_issue_distance_mismatches": 0,
            "materializer_or_fold_timing_window_claim": False,
            "main_cycle_model_changed_by_ledgers": False,
        },
        "exactness_gates": {
            "upstream_inner_and_outer_seals_recomputed": seal_results,
            "m40_payload_files_rehashed": payload_files,
            "m40_payload_bytes_rehashed": payload_bytes,
            "m40_packed_unpack_rereads": packed_unpack_rereads,
            "payload_sha_mismatches": 0,
            "parent_or_child_identity_mismatches": 0,
            "mask_reconstruction_mismatches":
                aggregate["reconstruction_mismatches"],
            "population_conservation_mismatches": 0,
            "fallback_pwp_correction_conservation_mismatches": 0,
            "m451_k1_saving_conservation_mismatches": 0,
            "triangle_bound_selection_mismatches":
                aggregate["triangle_selection_mismatches"],
            "sealed_center_materialization_ledger_rows":
                center_ledger_row_count,
            "sealed_ordered_descriptor_run_rows":
                ordered_aggregate["runs"],
            "sealed_ordered_source_rows":
                ordered_aggregate["source_rows"],
            "center_ledger_conservation_mismatches": 0,
            "ordered_ledger_conservation_mismatches": 0,
            "arithmetic_exact": True,
            "accuracy_loss": False,
        },
        "resource_boundary": {
            "matcher_new_three_parallel_child_comparators": True,
            "matcher_rtl_dc_ptpx": False,
            "q128_cache_expansion_required": True,
            "expanded_sram_macro_area_frequency_power": False,
            "m451_distinct_160B_pwp_plus_96B_correction_reads": True,
            "m451_new_96_lane_signed_preadder": True,
            "m451_standalone_directed_vcs": True,
            "m451_standalone_opportunity_per_area":
                m455["comparison"]
                ["standalone_adapter_opportunity_throughput_per_area_ratio"],
            "m451_resource_killed_by_m457": True,
            "tree_plus_m451_integration_rtl": False,
            "resource_normalized_speedup": False,
        },
        "decision": {
            "tree_separate_threshold":
                thresholds["minimum_separate_speedup_vs_m430"],
            "tree_separate_threshold_pass": pass_separate,
            "tree_fused_threshold":
                thresholds["minimum_fused_speedup_vs_strong_zero"],
            "tree_fused_threshold_pass": pass_fused,
            "tree_fused_threshold_disposition":
                "RESOURCE_KILLED_DIAGNOSTIC_BY_M455_M457",
            "matcher_rtl": "NO_GO_PENDING_MATCHED_RESOURCE_SCREEN",
            "next": ("GO_M461_EXACT_MATERIALIZATION_AND_MATCHED_RESOURCE_SCREEN"
                     if go_m461 else "NO_GO_TREE_SEPARATE_BELOW_1P10"),
            "reason": ("Separate stored-q128 tree passes the predeclared 1.10x screen; only M461 exact materialization and matched q128-cache/matcher resource pricing may proceed. M451 fused is resource-killed and cannot promote RTL." if go_m461 else
                       "Separate stored-q128 tree misses 1.10x versus M430; do not proceed to matcher RTL or materialization hardware."),
            "cycle_speedup_admitted": False,
            "resource_normalized_speedup": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    result_path = args.output_dir / (
        "m453b_h67_hierarchical_q32x3_secondary_replay_r1.json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(source_start == sha256(Path(__file__).resolve()),
            "M453b analyzer changed during replay")
    _, seal = write_seal(args.output_dir, [phase_path.name,
                                           center_path.name,
                                           ordered_path.name,
                                           result_path.name])
    print("{} strong={} m430={} tree_sep={} tree_fused={} "
          "sep_vs_m430={:.9f}x fused_vs_strong={:.9f}x overflow_phases={} "
          "next={} matcher_rtl={} seal={}".format(
              result["status"], strong, m430_cycles, separate_cycles,
              fused_cycles, separate_vs_m430, fused_vs_strong,
              result["pwp_capacity_dma_and_cache"]
              ["current_slot_overflow_phases"],
              result["decision"]["next"],
              result["decision"]["matcher_rtl"], sha256(seal)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
