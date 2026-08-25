#!/usr/bin/env python3
"""Fail-closed independent audit of the M122 W384 source-fold DSE."""

import hashlib
import json
import math
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
OUTPUT = REVIEW / "m122_w384_row_synchronous_source_fold_independent_audit.json"

PATHS = {
    "analyzer": HW / "system_simulator/scripts/analyze_m122_w384_row_synchronous_source_fold.py",
    "result": HW / "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/m122_w384_row_synchronous_source_fold_dse.json",
    "result_manifest": HW / "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/SHA256SUMS.complete_r1.txt",
    "contract": HW / "contracts/m122_w384_row_synchronous_source_fold_dse_contract_r1_20260824.json",
    "m109_analyzer": HW / "system_simulator/scripts/analyze_m109_r2_window_storage_dual_timeline_frontier.py",
    "m109_result": HW / "results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/m109_r2_window_storage_dual_timeline_frontier.json",
    "m115r2_result": HW / "results/m115r2_pwp_prefix_coefficient_width_r1_20260824/m115r2_pwp_prefix_coefficient_width.json",
    "m121_receipt": HW / "dc_handoff/runs/m121_w384_scheduler_numeric_island_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m123_contract": HW / "contracts/m123_w384_signed19_forwarding_accumulator_vcs_contract_r1_20260824.json",
    "m123_receipt": HW / "dc_handoff/runs/m123_w384_signed19_forwarding_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m109_independent_review": HW / "reviews/m109_r2_window_storage_dual_timeline_frontier_independent_hammer_r1_20260824/m109_r2_window_storage_dual_timeline_frontier_independent_hammer_review.json",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}

EXPECTED = {
    "analyzer": "ecf2ae43e1282ac483b6832f5a21af6d1b6259c3595eb6150e840f0dc7a55cd3",
    "result": "be11341211b92d85dc42cb7b79b98a826a782765a4780e1207e7bad5368d27b2",
    "result_manifest": "1150b8529d41d1068ba8bc71bfdc4d36f5e64858e48e7c5a01882f8589509ea8",
    "contract": "80618e84c3b513d9c0064e200fac89e92481e4acc2cd9045170f7cc55f460a0f",
    "m109_analyzer": "4eed1e1ef25cdbea0fdd40d1602d6b1eb7661b15b5ae47541c80e149fd060ada",
    "m109_result": "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
    "m115r2_result": "b0e7fbb0573473ad854ca856d5eab3eaf15af1ba79ea2ce3a958810575bc6708",
    "m121_receipt": "4b3e0d1bf249bff14dc18a6de05cc7ddf5bca4e2d384a7ef160650702fbee986",
    "m123_contract": "63432933d974b277453545118ac02f5d8a803987f8102982e56ee70177eb3f87",
    "m123_receipt": "736b989529d1ca6b83bcb705fb87f9f381efb3f7f0809811fda3630006bbc0a8",
    "m109_independent_review": "423a53a9d65cc274dad2deedad8e41f28afe08178506f31f234624ccb0e24f9f",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

FOLDS = (1, 2, 4, 8)
SOURCES = 16
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
WEIGHT_BITS = 8
ROWS = 3000
WINDOW_ROWS = 384
PARTITIONS = 432
RECORDS = 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant " + raw)

    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def receipt(path):
    output = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            require(key not in output, "duplicate receipt key " + key)
            output[key] = value
    return output


def popcount(value):
    return bin(value).count("1")


def greedy_partition(mask, fold):
    """Independent executable definition: select/clear lowest set bits."""
    remaining = mask
    groups = []
    while remaining:
        group = 0
        for source in range(SOURCES):
            bit = 1 << source
            if remaining & bit:
                group |= bit
                if popcount(group) == fold:
                    break
        require(group != 0, "selector made no progress")
        groups.append(group)
        remaining &= ~group
    return groups


def exhaustive_mask_proof():
    cases = 0
    group_totals = {fold: 0 for fold in FOLDS}
    for mask in range(1 << SOURCES):
        for fold in FOLDS:
            groups = greedy_partition(mask, fold)
            expected_groups = math.ceil(popcount(mask) / fold)
            require(len(groups) == expected_groups,
                    "fold cycle count mismatch")
            union = 0
            source_visits = [0] * SOURCES
            numeric_reference = 0
            numeric_grouped = 0
            for source in range(SOURCES):
                if mask & (1 << source):
                    numeric_reference += -128 if source == 0 else source * 7 - 53
            for group in groups:
                require(popcount(group) <= fold,
                        "group exceeds fold width")
                require((union & group) == 0, "source duplicated across groups")
                union |= group
                for source in range(SOURCES):
                    if group & (1 << source):
                        source_visits[source] += 1
                        numeric_grouped += (-128 if source == 0
                                            else source * 7 - 53)
            require(union == mask, "fold partition lost a source")
            for source in range(SOURCES):
                require(source_visits[source] == ((mask >> source) & 1),
                        "source multiplicity mismatch")
            require(numeric_grouped == numeric_reference,
                    "integer fold numerical conservation mismatch")
            group_totals[fold] += len(groups)
            cases += 1

    # Counterexample to a plausible but incorrect implementation that repeats
    # the priority result without clearing it between fold cycles.
    counterexample_mask = 0x001F
    first = greedy_partition(counterexample_mask, 4)[0]
    broken_groups = [first, first]
    broken_union = broken_groups[0] | broken_groups[1]
    broken_visits = sum(popcount(group) for group in broken_groups)
    require(broken_union != counterexample_mask and broken_visits == 8,
            "negative selector counterexample did not trigger")
    return {
        "mask_fold_cases": cases,
        "aggregate_groups_over_all_65536_masks": group_totals,
        "loss_or_duplication": 0,
        "integer_numeric_mismatches": 0,
        "negative_repeated_priority_counterexample": {
            "mask_hex": "0x001f",
            "fold": 4,
            "broken_groups_hex": ["0x000f", "0x000f"],
            "lost_source_mask_hex": "0x0010",
            "duplicated_source_visits": 4,
        },
    }


def main():
    self_start = sha256(Path(__file__).resolve())
    observed = {}
    for label, path in PATHS.items():
        actual = sha256(path)
        require(actual == EXPECTED[label],
                "identity mismatch {} {}".format(label, actual))
        observed[str(path.relative_to(HW))] = actual

    result = strict_json(PATHS["result"])
    contract = strict_json(PATHS["contract"])
    m109 = strict_json(PATHS["m109_result"])
    m115r2 = strict_json(PATHS["m115r2_result"])
    m109_review = strict_json(PATHS["m109_independent_review"])
    m121 = receipt(PATHS["m121_receipt"])
    m123 = receipt(PATHS["m123_receipt"])

    require(result["identity"]["analyzer_start_end_sha256"]
            == EXPECTED["analyzer"], "result analyzer identity drift")
    require(contract["frozen_identity"]["analyzer_sha256"]
            == EXPECTED["analyzer"]
            and contract["frozen_identity"]["result_sha256"]
            == EXPECTED["result"], "contract identity drift")
    require(result["identity"]["heldout_samples"] == [5, 6, 7, 8, 9]
            and result["identity"]["heldout_records"] == RECORDS,
            "heldout extent drift")

    w384 = next(row for row in m109["frontier"]
                if int(row["window_rows"]) == WINDOW_ROWS)
    records = {int(row["fold_sources_per_update"]): row
               for row in result["fold_dse"]}
    require(set(records) == set(FOLDS), "fold extent drift")
    frozen_recurrence = w384["dual_timeline_recurrence"]
    k1_recurrence = records[1]["dual_timeline_recurrence"]
    field_comparison = {}
    for key, expected in frozen_recurrence.items():
        observed_value = k1_recurrence.get(key)
        require(observed_value == expected,
                "K1 M109 recurrence mismatch " + key)
        field_comparison[key] = {
            "m109": expected,
            "m122_k1": observed_value,
            "equal": True,
        }
    require(len(field_comparison) == len(frozen_recurrence),
            "K1 field comparison incomplete")

    histogram = {int(key): int(value) for key, value
                 in result["same_row_source_count_histogram"].items()}
    require(set(histogram) == set(range(SOURCES + 1)),
            "histogram key extent drift")
    histogram_extent = sum(histogram.values())
    histogram_sources = sum(source_count * count
                            for source_count, count in histogram.items())
    require(histogram_extent
            == RECORDS * PARTITIONS * ROWS * OUTPUT_BLOCKS
            == 207360000, "histogram extent mismatch")
    require(histogram_sources == result["exact_work"]["events"]
            == contract["exact_work"]["events"] == 188148490,
            "histogram source/event conservation mismatch")

    fold_recomputation = {}
    for fold in FOLDS:
        event_cycles = sum(math.ceil(source_count / fold) * count
                           for source_count, count in histogram.items())
        consecutive_same_address_pairs = sum(
            max(math.ceil(source_count / fold) - 1, 0) * count
            for source_count, count in histogram.items())
        masks_requiring_multiple_updates = sum(
            count for source_count, count in histogram.items()
            if source_count > fold)
        row = records[fold]
        require(event_cycles == row["exact_fold_event_cycles"],
                "fold event cycle mismatch K{}".format(fold))
        require(row["accumulator_write_cycles_removed"]
                == histogram_sources - event_cycles,
                "removed write count mismatch K{}".format(fold))
        require(abs(row["event_cycle_reduction_fraction"]
                    - (1.0 - event_cycles / histogram_sources)) < 1e-15,
                "fold reduction mismatch K{}".format(fold))
        fold_recomputation[str(fold)] = {
            "event_cycles": event_cycles,
            "accumulator_write_cycles_removed": histogram_sources - event_cycles,
            "masks_requiring_multiple_same_address_updates":
                masks_requiring_multiple_updates,
            "consecutive_same_address_pairs_if_grouped_per_mask":
                consecutive_same_address_pairs,
            "interlock_only_event_cycles_lower_bound":
                event_cycles + consecutive_same_address_pairs,
        }

    exact = result["exact_work"]
    require(exact == {"events": 188148490, "groups": 8271296,
                      "pwp_tokens": 226222255}, "exact work drift")
    load_tokens = 3 * exact["groups"]
    require(load_tokens == 24813888, "weight load token mismatch")
    descriptors = RECORDS * PARTITIONS * math.ceil(ROWS / WINDOW_ROWS)
    require(descriptors == 69120, "descriptor count mismatch")
    descriptor_fill = exact["events"] + descriptors
    commit_cycles = RECORDS * ROWS * OUTPUT_BLOCKS
    flush_cycles = RECORDS * math.ceil(ROWS / WINDOW_ROWS)
    require(descriptor_fill == 188217610 and commit_cycles == 480000
            and flush_cycles == 160, "fill/commit/flush arithmetic mismatch")

    cycle_ledgers = {}
    for fold in FOLDS:
        recurrence = records[fold]["dual_timeline_recurrence"]
        correction = records[fold]["exact_fold_event_cycles"] + load_tokens
        require(recurrence["weight_load_tokens"] == load_tokens
                and recurrence["correction_service_tokens"] == correction,
                "correction/load ledger mismatch K{}".format(fold))
        require(recurrence["pwp_service_tokens"] == exact["pwp_tokens"],
                "PWP ledger mismatch K{}".format(fold))
        require(recurrence["descriptor_fill_cycles"] == descriptor_fill
                and recurrence["accumulator_commit_cycles"] == commit_cycles
                and recurrence["accumulator_pipeline_flush_cycles"]
                == flush_cycles, "fill/tail ledger mismatch K{}".format(fold))
        conserved = (recurrence["pwp_service_tokens"]
                     + recurrence["correction_service_tokens"]
                     + recurrence["service_idle_cycles"]
                     + recurrence["accumulator_commit_cycles"]
                     + recurrence["accumulator_pipeline_flush_cycles"])
        require(conserved == recurrence["candidate_cycles"],
                "candidate cycle conservation mismatch K{}".format(fold))
        cycle_ledgers[str(fold)] = {
            "candidate_cycles": recurrence["candidate_cycles"],
            "conserved_component_sum": conserved,
            "correction_service_tokens": correction,
            "weight_load_tokens": load_tokens,
            "pwp_tokens": exact["pwp_tokens"],
            "descriptor_fill_cycles_overlapped_timeline": descriptor_fill,
            "commit_cycles_no_final_window_padding": commit_cycles,
            "flush_cycles": flush_cycles,
        }

    fixed8_tokens = records[4]["dual_timeline_recurrence"][
        "fair_fixed8_baseline_cycles"] - commit_cycles - flush_cycles
    require(fixed8_tokens == 1114383288
            and fixed8_tokens % 3 == 0
            and fixed8_tokens // 3 == 371461096,
            "fixed8 denominator mismatch")
    k1_cycles = k1_recurrence["candidate_cycles"]
    k4 = records[4]["dual_timeline_recurrence"]
    k4_ratio = k4["fair_fixed8_baseline_cycles"] / k4["candidate_cycles"]
    k4_vs_k1 = k1_cycles / k4["candidate_cycles"]
    require(abs(k4_ratio - 3.1725369008459166) < 1e-15
            and abs(k4_vs_k1 - 1.2512657845537327) < 1e-15,
            "K4 ratio mismatch")
    require(m109_review["projection_and_claim_boundary"]
            ["system_speedup"] is False
            and m109_review["baseline_fairness"]
            ["controller_and_descriptor_ingress_edges_charged_to_baseline"]
            is False,
            "inherited M109 claim boundary drift")

    mask_proof = exhaustive_mask_proof()
    require(m115r2["checkpoint"]["mathematical_candidate_signed_bits"] == 19
            and m115r2["prefix_coefficient_proof"]
            ["maximum_absolute_prefix_coefficient"] == 1,
            "signed19 mathematical basis drift")
    require(records[4]["signed_fold_delta_bits"] == 10
            and records[8]["signed_fold_delta_bits"] == 11,
            "fold delta width drift")

    per_block_bits = SOURCES * OUTPUT_LANES * WEIGHT_BITS
    per_block_bytes = per_block_bits // 8
    all_blocks_bits = OUTPUT_BLOCKS * per_block_bits
    all_blocks_bytes = all_blocks_bits // 8
    k4_read_payload_bits_per_cycle = 4 * OUTPUT_LANES * WEIGHT_BITS
    replicated_single_read_bytes = 4 * all_blocks_bytes
    require(per_block_bits == 12288 and per_block_bytes == 1536
            and all_blocks_bits == 98304 and all_blocks_bytes == 12288
            and k4_read_payload_bits_per_cycle == 3072
            and replicated_single_read_bytes == 49152,
            "cache/read geometry mismatch")
    for fold, delta_bits in ((1, 8), (2, 9), (4, 10), (8, 11)):
        hw_contract = records[fold]["hardware_contract"]
        require(hw_contract["resident_weight_cache_bits"] == per_block_bits
                and hw_contract["resident_weight_cache_bytes"] == per_block_bytes
                and hw_contract["resident_weight_vectors_per_output_block"]
                == SOURCES and hw_contract["lane_fold_delta_bits"] == delta_bits,
                "published cache/delta geometry drift K{}".format(fold))

    require(m121["heldout_trace_duplicate_retry_escape_replay"] == "false"
            and m121["module_cycle_projection_admitted"] == "false",
            "M121 boundary drift")
    require(m123["same_address_chain_length"] == "16"
            and m123["same_address_accept_pairs"] == "15"
            and m123["same_address_macro_reads_suppressed"] == "15"
            and m123["scheduled_cycle_ratio"] == "false",
            "post-contract M123 forwarding evidence drift")

    output = {
        "schema": "m122_w384_row_synchronous_source_fold_independent_audit_v1",
        "status": "PASS_CYCLE_ARITHMETIC_WITH_CACHE_AND_PHYSICAL_BOUNDARIES",
        "identity": observed,
        "k1_m109_reproduction": {
            "all_frozen_m109_w384_fields_equal": True,
            "field_count": len(field_comparison),
            "field_comparison": field_comparison,
            "m122_additional_ledger_fields": sorted(
                set(k1_recurrence) - set(frozen_recurrence)),
        },
        "heldout_histogram_recomputation": {
            "extent": histogram_extent,
            "source_events": histogram_sources,
            "folds": fold_recomputation,
        },
        "exhaustive_mask_fold_proof": mask_proof,
        "cycle_ledger": {
            "records": RECORDS,
            "windows_per_record": math.ceil(ROWS / WINDOW_ROWS),
            "descriptors": descriptors,
            "groups": exact["groups"],
            "weight_load_tokens": load_tokens,
            "pwp_tokens": exact["pwp_tokens"],
            "descriptor_fill_cycles": descriptor_fill,
            "commit_cycles": commit_cycles,
            "commit_uses_actual_3000_rows_not_8x384_padding": True,
            "flush_cycles": flush_cycles,
            "folds": cycle_ledgers,
        },
        "k4_ratio_audit": {
            "fixed8_raw_events": 371461096,
            "fixed8_service_tokens": fixed8_tokens,
            "fixed8_plus_shared_commit_flush_cycles":
                k4["fair_fixed8_baseline_cycles"],
            "k4_candidate_cycles": k4["candidate_cycles"],
            "ratio_vs_fixed8_service_island": k4_ratio,
            "incremental_candidate_speedup_vs_k1": k4_vs_k1,
            "event_cycle_reduction_vs_k1":
                records[4]["event_cycle_reduction_fraction"],
            "equal_controller_end_to_end_baseline": False,
            "full_network_or_system_speedup": False,
        },
        "cache_and_datapath_geometry": {
            "weight_cache_per_output_block": {
                "vectors": SOURCES,
                "vector_bits": OUTPUT_LANES * WEIGHT_BITS,
                "bits": per_block_bits,
                "bytes": per_block_bytes,
            },
            "eight_simultaneously_resident_output_blocks": {
                "bits": all_blocks_bits,
                "bytes": all_blocks_bytes,
                "factor_vs_published_per_block_entry": 8,
            },
            "shared_single_block_cache_alternative": {
                "bytes": per_block_bytes,
                "requires_strict_block_phased_service_and_lifetime_proof": True,
                "proven_by_current_dse": False,
            },
            "k4_logical_read_ports": 4,
            "k4_read_payload_bits_per_cycle": k4_read_payload_bits_per_cycle,
            "single_read_replication_upper_bound_bytes":
                replicated_single_read_bytes,
            "k4_lane_adder_tree": {
                "signed9_stage_adders": 192,
                "signed10_stage_adders": 96,
                "selectors": "four 16-to-1 signed-INT8 selections per lane",
            },
            "selector_area_timing_energy_modeled": False,
            "adder_area_timing_energy_modeled": False,
            "multi_read_cache_area_timing_energy_modeled": False,
        },
        "rmw_hazard_audit": {
            "k4_masks_requiring_multiple_same_address_updates":
                fold_recomputation["4"]
                ["masks_requiring_multiple_same_address_updates"],
            "k4_consecutive_same_address_pairs_if_grouped_per_mask":
                fold_recomputation["4"]
                ["consecutive_same_address_pairs_if_grouped_per_mask"],
            "m121_directed_numeric_downstream_backpressure_cycles": 0,
            "m121_is_k4_fold_integration": False,
            "post_contract_m123_same_address_chain_vcs": True,
            "m123_is_k4_selector_cache_adder_integration": False,
        },
        "claim_boundary": {
            "exact_heldout_ideal_cycle_dse": True,
            "k4_rtl_vcs": False,
            "multi_read_cache_physical": False,
            "foundry_weight_macro": False,
            "macro_inclusive_ppa": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
        "self_sha256_at_start": self_start,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    require(sha256(Path(__file__).resolve()) == self_start,
            "audit script changed during execution")
    print("PASS M122 independent source-fold audit")


if __name__ == "__main__":
    main()
