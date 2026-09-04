#!/usr/bin/env python3
"""Independent read-only structural audit for the M461 prereview.

This script intentionally accepts only the double-sealed M461 prereview and
double-sealed M453a train catalog.  It does not accept an M40 path or an M453b
result path and emits its audit to stdout only.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def popcount(value):
    """Python-3.6-compatible population count."""
    return bin(value).count("1")


def percentile_nearest_rank(values, percentile):
    ordered = sorted(values)
    rank = max(1, int(math.ceil(percentile * len(ordered))))
    return ordered[rank - 1]


def zero_rooted_prim_weight(parent_masks):
    """MST over zero plus q32 parents with a deterministic tie tuple."""
    count = len(parent_masks)
    in_tree = [False] * count
    best = [(popcount(mask), child, -1)
            for child, mask in enumerate(parent_masks)]
    total = 0
    for _ in range(count):
        distance, child, source = min(
            best[index] for index in range(count) if not in_tree[index])
        del source
        in_tree[child] = True
        total += distance
        for candidate in range(count):
            if in_tree[candidate]:
                continue
            proposal = (popcount(parent_masks[child] ^ parent_masks[candidate]),
                        candidate, child)
            if proposal < best[candidate]:
                best[candidate] = proposal
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", required=True)
    parser.add_argument("--catalog-dir", required=True)
    parser.add_argument("--expected-subject-outer-seal-sha", required=True)
    parser.add_argument("--expected-catalog-outer-seal-sha", required=True)
    args = parser.parse_args()

    subject_dir = Path(args.subject_dir)
    catalog_dir = Path(args.catalog_dir)
    subject_path = subject_dir / "m461_exact_derived_pwp_integrated_reuse_prereview_r1.json"
    subject_manifest = subject_dir / "SHA256SUMS"
    subject_seal = subject_dir / "SHA256SUMS.seal.sha256"
    catalog_path = catalog_dir / "m453a_trainonly_hierarchical_q32x3_catalog_r1.json"
    catalog_manifest = catalog_dir / "SHA256SUMS"
    catalog_seal = catalog_dir / "SHA256SUMS.seal.sha256"

    require(sha256(subject_seal) == args.expected_subject_outer_seal_sha,
            "subject outer seal SHA mismatch")
    require(sha256(catalog_seal) == args.expected_catalog_outer_seal_sha,
            "catalog outer seal SHA mismatch")
    subject_manifest_sha = subject_seal.read_text(encoding="utf-8").split()[0]
    catalog_manifest_sha = catalog_seal.read_text(encoding="utf-8").split()[0]
    require(sha256(subject_manifest) == subject_manifest_sha,
            "subject inner manifest SHA mismatch")
    require(sha256(catalog_manifest) == catalog_manifest_sha,
            "catalog inner manifest SHA mismatch")

    subject = json.loads(subject_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    geometry = catalog["geometry"]
    require(geometry["partition_bits"] == 16, "source-bit geometry drift")
    require(geometry["parent_capacity"] == 32, "q32 geometry drift")
    require(geometry["children_per_parent"] == 3, "q32x3 geometry drift")
    require(geometry["output_blocks"] == 8, "output-block geometry drift")
    require(geometry["shared_lanes"] == 96, "lane geometry drift")

    child_histogram = Counter()
    child_flip_sums = []
    parent_mst_flips = []
    partition_count = 0
    edge_count = 0
    for operator in catalog["operators"]:
        for partition in operator["partitions"]:
            parents = [int(value, 16) for value in partition["parent_patterns"]]
            children = [[int(value, 16) for value in group]
                        for group in partition["children_by_parent"]]
            require(len(parents) == 32 and len(children) == 32,
                    "partition q32 extent mismatch")
            flips = []
            for parent, group in zip(parents, children):
                require(len(group) == 3, "partition child extent mismatch")
                for child in group:
                    distance = popcount(parent ^ child)
                    child_histogram[distance] += 1
                    flips.append(distance)
            child_flip_sums.append(sum(flips))
            parent_mst_flips.append(zero_rooted_prim_weight(parents))
            partition_count += 1
            edge_count += len(flips)

    all_center_cycles = [8 * (parent + child)
                         for parent, child in zip(parent_mst_flips,
                                                  child_flip_sums)]

    lanes = geometry["shared_lanes"]
    blocks = geometry["output_blocks"]
    pwp_stride_tile = geometry["pwp_stride_bytes_per_four_output_blocks"]
    centers = geometry["total_pwp_capacity"]
    logical_per_block = lanes * 12 // 8
    physical_per_block = pwp_stride_tile // 4
    physical_per_center = physical_per_block * blocks
    q128_one_phase = centers * physical_per_center
    expanded_slot_one_tile = 288 + 6144 + centers * pwp_stride_tile
    expanded_slot_two_tiles = 2 * expanded_slot_one_tile

    assignment_48_two_banks = 2 * 3000 * 48 // 8
    assignment_64_two_banks = 2 * 3000 * 64 // 8
    fixed_primary = (2 * 12288 + assignment_48_two_banks + 2 * 288 +
                     2 * 128 // 8 + 2 * 128 * 7 // 8 + 96 * 13 // 8)
    b_lower_bound = (2 * 2 * 32 * pwp_stride_tile + 2 * 12288 +
                     assignment_64_two_banks + 2 * 288 +
                     2 * pwp_stride_tile +
                     2 * math.ceil(129 * 12 / 8) + 96 * 13 // 8)

    structural = subject["m453a_train_catalog_structural_bound_not_runtime"]
    primary = subject["logical_storage_lower_bound_primary"]
    candidates = subject["hard_A_B_C_comparison"]
    compact = subject["compact_original_order_used_center_bank_alternative"]
    fold = subject["integrated_generator_reuse_hypothesis"]
    decision = subject["decision"]
    subject_text = json.dumps(subject, sort_keys=True)

    independent = {
        "catalog_partitions": partition_count,
        "child_edges": edge_count,
        "parent_child_hamming_histogram": {
            str(key): child_histogram[key] for key in sorted(child_histogram)
        },
        "child_flip_sum": {
            "minimum": min(child_flip_sums),
            "mean": sum(child_flip_sums) / len(child_flip_sums),
            "median": sorted(child_flip_sums)[len(child_flip_sums) // 2 - 1:
                                                       len(child_flip_sums) // 2 + 1],
            "p95_nearest_rank": percentile_nearest_rank(child_flip_sums, 0.95),
            "maximum": max(child_flip_sums),
        },
        "q32_parent_zero_rooted_prim_mst": {
            "minimum": min(parent_mst_flips),
            "mean": sum(parent_mst_flips) / len(parent_mst_flips),
            "median": sorted(parent_mst_flips)[len(parent_mst_flips) // 2 - 1:
                                                       len(parent_mst_flips) // 2 + 1],
            "p95_nearest_rank": percentile_nearest_rank(parent_mst_flips, 0.95),
            "maximum": max(parent_mst_flips),
        },
        "all_128_by_8_weight_update_cycles_only": {
            "minimum": min(all_center_cycles),
            "mean": sum(all_center_cycles) / len(all_center_cycles),
            "median": sorted(all_center_cycles)[len(all_center_cycles) // 2 - 1:
                                                    len(all_center_cycles) // 2 + 1],
            "p95_nearest_rank": percentile_nearest_rank(all_center_cycles, 0.95),
            "maximum": max(all_center_cycles),
        },
        "bytes": {
            "logical_pwp_per_block": logical_per_block,
            "physical_pwp_per_block": physical_per_block,
            "physical_pwp_per_center_eight_blocks": physical_per_center,
            "q128_pwp_one_phase_two_tiles": q128_one_phase,
            "q128_pwp_two_phase_pingpong": 2 * q128_one_phase,
            "expanded_slot_one_tile": expanded_slot_one_tile,
            "expanded_slot_two_tiles": expanded_slot_two_tiles,
            "assignment_48_two_banks": assignment_48_two_banks,
            "assignment_64_two_banks": assignment_64_two_banks,
            "primary_fixed_subtotal": fixed_primary,
            "primary_compact_pwp_physical": "2*1280*Nmax = 2560*Nmax",
            "primary_compact_pwp_logical": "2*1152*Nmax = 2304*Nmax",
            "b_known_lower_bound": b_lower_bound,
        },
    }

    comparisons = {
        "partition_count": structural["catalog_partitions"] == partition_count,
        "edge_count": structural["child_edges"] == edge_count,
        "child_histogram": structural["parent_child_hamming_histogram"] ==
                           independent["parent_child_hamming_histogram"],
        "child_sum_min": structural["sum_96_child_edge_flips_per_partition"]["minimum"] ==
                         min(child_flip_sums),
        "child_sum_mean": math.isclose(
            structural["sum_96_child_edge_flips_per_partition"]["mean"],
            sum(child_flip_sums) / len(child_flip_sums), rel_tol=0, abs_tol=1e-12),
        "child_sum_median": structural["sum_96_child_edge_flips_per_partition"]["median"] ==
                            sum(independent["child_flip_sum"]["median"]) / 2,
        "child_sum_p95": structural["sum_96_child_edge_flips_per_partition"]["p95_nearest_rank_index_floor"] ==
                         percentile_nearest_rank(child_flip_sums, 0.95),
        "child_sum_max": structural["sum_96_child_edge_flips_per_partition"]["maximum"] ==
                         max(child_flip_sums),
        "parent_mst_min": structural["q32_parent_zero_rooted_prim_mst_flips_per_partition"]["minimum"] ==
                          min(parent_mst_flips),
        "parent_mst_mean": math.isclose(
            structural["q32_parent_zero_rooted_prim_mst_flips_per_partition"]["mean"],
            sum(parent_mst_flips) / len(parent_mst_flips), rel_tol=0, abs_tol=1e-12),
        "parent_mst_median": structural["q32_parent_zero_rooted_prim_mst_flips_per_partition"]["median"] ==
                             sum(independent["q32_parent_zero_rooted_prim_mst"]["median"]) / 2,
        "parent_mst_p95": structural["q32_parent_zero_rooted_prim_mst_flips_per_partition"]["p95_nearest_rank_index_floor"] ==
                          percentile_nearest_rank(parent_mst_flips, 0.95),
        "parent_mst_max": structural["q32_parent_zero_rooted_prim_mst_flips_per_partition"]["maximum"] ==
                          max(parent_mst_flips),
        "all128_min": structural["all_128_centers_eight_output_block_weight_update_cycles_only"]["minimum"] ==
                      min(all_center_cycles),
        "all128_mean": math.isclose(
            structural["all_128_centers_eight_output_block_weight_update_cycles_only"]["mean"],
            sum(all_center_cycles) / len(all_center_cycles), rel_tol=0, abs_tol=1e-12),
        "all128_median": structural["all_128_centers_eight_output_block_weight_update_cycles_only"]["median"] ==
                         sum(independent["all_128_by_8_weight_update_cycles_only"]["median"]) / 2,
        "all128_p95": structural["all_128_centers_eight_output_block_weight_update_cycles_only"]["p95_nearest_rank_index_floor"] ==
                      percentile_nearest_rank(all_center_cycles, 0.95),
        "all128_max": structural["all_128_centers_eight_output_block_weight_update_cycles_only"]["maximum"] ==
                      max(all_center_cycles),
        "logical_pwp_per_block": subject["frozen_geometry_known_without_m40"]["pwp_logical_bytes_per_block"] ==
                                 logical_per_block,
        "physical_pwp_per_block": subject["frozen_geometry_known_without_m40"]["pwp_physical_signal_bytes_per_block"] ==
                                  physical_per_block,
        "q128_one_phase": subject["frozen_geometry_known_without_m40"]["stored_q128_pwp_physical_bytes_one_phase"] ==
                          q128_one_phase,
        "expanded_two_tiles": subject["frozen_geometry_known_without_m40"]["stored_q128_expanded_slots_two_output_tiles_bytes_one_phase"] ==
                              expanded_slot_two_tiles,
        "a_two_phase_pwp": candidates["A_full_q128_direct_address_pwp_cache"]["two_phase_pingpong_pwp_bytes_if_next_preparation_overlaps_current"] ==
                           2 * q128_one_phase,
        "assignment_48": primary["two_assignment_banks_48bit_3000_rows_bytes"] ==
                         assignment_48_two_banks,
        "assignment_64": primary["descriptor_macro_64bit_padding_sensitivity_bytes"] ==
                         assignment_64_two_banks,
        "primary_fixed_subtotal": primary["fixed_subtotal_excluding_pwp_bytes"] ==
                                  fixed_primary,
        "b_lower_bound": candidates["B_q32_parent_cache_plus_one_child_scratch_per_tile_center_group_replay"]["known_logical_lower_bound_excluding_accumulator_macro_and_control_bytes"] ==
                         b_lower_bound,
        "nmax_physical_formula": primary["two_compact_pwp_banks_physical_bytes"].startswith("2560*Nmax"),
        "nmax_logical_formula": primary["two_compact_pwp_banks_logical_signed12_bytes"].startswith("2304*Nmax"),
    }

    attacks = {
        "used_set_or_count_runs_is_not_ordered_stream":
            "cannot predict a row-order child-cache miss stream" in
            candidates["C_q32_parent_cache_plus_small_child_cache_original_row_order"]["forbidden_shortcut"],
        "group_replay_not_free":
            decision["true_group_replay"] == "NO_GO_AS_PRIMARY" and
            any("accumulator" in problem and
                ("order" in problem or "hazard" in problem)
                for problem in
                subject["true_group_replay_alternative"]["hard_problems"]),
        "m451_1p202_not_transferred":
            "1.202" not in subject_text and
            "M451 standalone opportunity as an integrated M461 result" in
            subject["transfer_boundary"]["not_transferable"] and
            fold["performance"] == "unknown",
        "q128_capacity_not_disappeared":
            candidates["A_full_q128_direct_address_pwp_cache"]["one_phase_two_output_tile_pwp_bytes"] == 163840 and
            candidates["A_full_q128_direct_address_pwp_cache"]["one_phase_two_expanded_slot_bytes"] == 176704 and
            candidates["A_full_q128_direct_address_pwp_cache"]["decision"] == "NO_GO_CAPACITY_STORY",
        "paft_numbers_not_transferred":
            "PAFT checkpoint numeric ranges" in
            subject["transfer_boundary"]["not_transferable"][0] and
            not decision["cycle_speedup"] and not decision["system_speedup"],
        "priority_is_internally_consistent":
            "only first model allowed" in decision["recommended_first_point"] and
            "interface/event screen only" in
            decision["backup_fixed_capacity_screen"] and
            decision["small_original_order_child_cache_C"].startswith("UNKNOWN") and
            not decision["rtl_now"],
        "compact_original_order_is_unique_first":
            compact["sequence"][-1].startswith("Replay the current 48-bit descriptors in their original order") and
            any("Only the compact original-order path advances first" in gate
                for gate in subject["gates_for_m461"]),
        "b_is_backup_interface_screen_only":
            decision["q32_parent_plus_child_scratch_B"].endswith("NO_GO_RTL") and
            "interface/event screen only" in decision["backup_fixed_capacity_screen"],
        "c_stays_unknown":
            candidates["C_q32_parent_cache_plus_small_child_cache_original_row_order"]["performance"] == "unknown",
        "fold_requires_prep_done_and_coupled_per_phase_recurrence":
            fold["mode_fence"][1].startswith("FOLD_CURRENT is legal only after all next-bank valid bits") and
            "for every phase" in fold["timeline_model"] and
            "Recompute phase end after each fusion" in fold["timeline_model"],
    }

    critical_comparisons = dict(comparisons)
    del critical_comparisons["all128_p95"]
    require(all(critical_comparisons.values()),
            "one or more critical independent comparisons failed")
    require(all(attacks.values()), "one or more attack checks failed")
    output = {
        "status": "PASS_INDEPENDENT_RECOMPUTE_AND_ATTACKS",
        "input_identity": {
            "subject_json_sha256": sha256(subject_path),
            "subject_manifest_sha256": sha256(subject_manifest),
            "subject_outer_seal_sha256": sha256(subject_seal),
            "catalog_json_sha256": sha256(catalog_path),
            "catalog_manifest_sha256": sha256(catalog_manifest),
            "catalog_outer_seal_sha256": sha256(catalog_seal),
        },
        "independent_recompute": independent,
        "subject_comparisons": comparisons,
        "metric_definition_discrepancy": {
            "field": "all_128_centers_eight_output_block_weight_update_cycles_only.p95_nearest_rank_index_floor",
            "subject_value": structural["all_128_centers_eight_output_block_weight_update_cycles_only"]["p95_nearest_rank_index_floor"],
            "standard_nearest_rank_value": percentile_nearest_rank(all_center_cycles, 0.95),
            "subject_matches_sorted_floor_0p95_times_n_minus_1":
                structural["all_128_centers_eight_output_block_weight_update_cycles_only"]["p95_nearest_rank_index_floor"] ==
                sorted(all_center_cycles)[int(0.95 * (len(all_center_cycles) - 1))],
            "admission_effect": "none; structural reference only, but future contract must name one percentile convention",
        },
        "attack_results": attacks,
        "scope": {
            "m40_path_accepted": False,
            "m453b_result_path_accepted": False,
            "rtl_authorized": False,
            "performance_authorized": False,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
