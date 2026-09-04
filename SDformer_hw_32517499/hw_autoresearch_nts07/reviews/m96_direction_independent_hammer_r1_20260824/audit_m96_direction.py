#!/usr/bin/env python3
"""Independent M89 K6-C16 bottleneck and M96-direction audit.

The script reads frozen JSON ledgers only.  It does not execute or modify any
producer, scheduler, contract, result, receipt, or paper document.
"""

from __future__ import print_function

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


EXPECTED = {
    "m43": "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c",
    "m45": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
    "m53": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m55": "9639903ea82e90b1a8403ff0bee66b01ec732ee6baa11d275ec2725e0a4d531b",
    "m89": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
    "m90": "358d1f170b535704ddd032fe33ccc8b6f5492e5134a52f29370a24d69ccf3b09",
    "m91": "83a3fe67e592e0fee1b619329e612798eee5da443285d35ce914d0fe2a9539a1",
    "m92": "4af6c4aa18f7ba012a70321b9ea96c7f244a8472d755f81e644bbdaf2e3456dc",
    "m93_raw": "7345e006f052bf00520800f9fbf8d2792747a2686cdb0e15c61d9703f1cac7e9",
    "m93_receipt": "b07fa6872a1eebe5f98db07c5e6502902030a32d33a480cc898cddbcecd536a9",
    "m94_raw": "a871355741e310508045a047da62659e718f237c6716a7e2fbd2a0be67d7f9a4",
    "m94_receipt": "37b0c4a95939dc3ddad9738840257447c199765b939845938e6a939d506c2eb8",
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

    def pairs(pairs_list):
        result = {}
        for key, value in pairs_list:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject, object_pairs_hook=pairs)


def ratio(numerator, denominator):
    require(denominator > 0, "zero denominator")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    hw = Path(args.hw_root).resolve()
    paths = {
        "m43": hw / "results/m43_tile_resident_parent_delta_schedule_r1_20260823/m43_spatiotemporal_parent_delta_ablation.json",
        "m45": hw / "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py",
        "m53": hw / "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/m53_adaptive_temporal_parent_k4_ctx16_dse.json",
        "m55": hw / "results/m55_h67_full_network_dual_parent_opportunity_r1_20260823/m55_h67_full_network_dual_parent_opportunity_result_r1.json",
        "m89": hw / "results/m89_temporal_fanout_hold_screen_r1_20260823/m89_temporal_fanout_hold_screen_receipt.json",
        "m90": hw / "results/m90_window_parent_integrated_probe_r1_20260823/m90_window_parent_integrated_probe_receipt.json",
        "m91": hw / "results/m91_dependency_safe_fusion_aware_parent_probe_r1_20260824/m91_dependency_safe_fusion_aware_parent_probe_receipt_r1.json",
        "m92": hw / "results/m92_commit_bus_parent_bypass_probe_r1_20260824/m92_commit_bus_parent_bypass_probe_receipt_r1.json",
        "m93_raw": hw / "results/m93_dual_descriptor_packet_issue_probe_r1_20260824/remote_artifacts/m93_dual_descriptor_packet_issue_probe_r1_20260824.json",
        "m93_receipt": hw / "results/m93_dual_descriptor_packet_issue_probe_r1_20260824/m93_dual_descriptor_packet_issue_probe_receipt_r1.json",
        "m94_raw": hw / "results/m94_critical_first_fusion_seed_probe_r1_20260824/remote_artifacts/m94_critical_first_fusion_seed_probe_r1_20260824.json",
        "m94_receipt": hw / "results/m94_critical_first_fusion_seed_probe_r1_20260824/m94_critical_first_fusion_seed_probe_receipt_r1.json",
    }
    identities = {}
    for name, path in paths.items():
        require(path.is_file(), "missing {}".format(path))
        identities[name] = sha256(path)
        require(identities[name] == EXPECTED[name],
                "{} SHA drift".format(name))

    m43 = read_json(paths["m43"])
    m53 = read_json(paths["m53"])
    m55 = read_json(paths["m55"])
    m89 = read_json(paths["m89"])
    m90 = read_json(paths["m90"])
    m91 = read_json(paths["m91"])
    m92 = read_json(paths["m92"])
    m93 = read_json(paths["m93_raw"])
    m93_receipt = read_json(paths["m93_receipt"])
    m94 = read_json(paths["m94_raw"])
    m94_receipt = read_json(paths["m94_receipt"])

    w1_list = [row for row in m93["configurations"]
               if row["descriptor_issue"]["issue_width"] == 1]
    require(len(w1_list) == 1, "M93 W1 configuration population drift")
    w1 = w1_list[0]
    k6_list = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(k6_list) == 1, "M89 K6 population drift")
    k6 = k6_list[0]
    require(w1["aggregate_integrated_cycles"] ==
            k6["integrated_cycles"] == 76677320,
            "M89/M93 integrated reproduction drift")
    require(w1["aggregate_source_only_cycles"] ==
            k6["source_cycles"] == 69964176,
            "M89/M93 source reproduction drift")
    require(w1["integrated_cycle_distribution"]["p95_nearest_rank"] ==
            k6["p95_integrated_cycles"] == 7843680,
            "M89/M93 p95 reproduction drift")

    samples = w1["per_sample"]
    require(len(samples) == 10, "M89 sample population drift")
    sum_fields = (
        "source_only_cycles", "integrated_cycles", "unique_weight_issues",
        "logical_source_updates", "fusion_groups", "zero_source_groups",
        "command_or_state_wait_cycles", "parent_wait_cycles",
        "response_or_context_wait_cycles", "weight_dma_wait_cycles",
        "descriptor_commands", "parent_partial_reads",
        "parent_partial_writes", "final_accumulator_reads",
        "final_accumulator_writes", "completed_outputs",
    )
    totals = dict((field, sum(row[field] for row in samples))
                  for field in sum_fields)
    require(totals["source_only_cycles"] ==
            w1["aggregate_source_only_cycles"], "source sum drift")
    require(totals["integrated_cycles"] ==
            w1["aggregate_integrated_cycles"], "integrated sum drift")
    require(totals["unique_weight_issues"] ==
            w1["aggregate_unique_weight_issues"], "issue sum drift")
    require(totals["logical_source_updates"] ==
            w1["aggregate_logical_source_updates"], "logical sum drift")
    require(totals["fusion_groups"] ==
            w1["aggregate_fusion_groups"], "group sum drift")

    records = w1["record_ledger"]["records"]
    require(len(records) == 40, "M89 record ledger population drift")
    per_operator = collections.defaultdict(collections.Counter)
    for row in records:
        for field in sum_fields:
            per_operator[row["operator"]][field] += row[field]
    operator_rows = []
    for name, values in sorted(
            per_operator.items(),
            key=lambda item: -item[1]["integrated_cycles"]):
        source = values["source_only_cycles"]
        integrated = values["integrated_cycles"]
        lower = (values["unique_weight_issues"] + 7) // 8
        operator_rows.append({
            "operator": name,
            "integrated_cycles": integrated,
            "source_cycles": source,
            "non_source_cycles": integrated - source,
            "source_share": ratio(source, integrated),
            "unique_weight_issues": values["unique_weight_issues"],
            "global_eight_bank_work_lower_bound": lower,
            "optimistic_bank_or_group_slack": source - lower,
            "fusion_groups": values["fusion_groups"],
            "zero_source_groups": values["zero_source_groups"],
            "command_or_state_wait_cycles":
                values["command_or_state_wait_cycles"],
            "parent_wait_cycles": values["parent_wait_cycles"],
            "response_or_context_wait_cycles":
                values["response_or_context_wait_cycles"],
        })

    total = totals["integrated_cycles"]
    source = totals["source_only_cycles"]
    overhead = total - source
    global_bank_lower = (totals["unique_weight_issues"] + 7) // 8
    optimistic_slack = source - global_bank_lower
    wait_sum = (totals["command_or_state_wait_cycles"] +
                totals["parent_wait_cycles"] +
                totals["response_or_context_wait_cycles"] +
                totals["weight_dma_wait_cycles"])
    residual = overhead - wait_sum
    require(global_bank_lower == 52029080 and optimistic_slack == 17935096,
            "global bank bound drift")
    require(totals["zero_source_groups"] == 832696,
            "zero-source group count drift")
    require(totals["command_or_state_wait_cycles"] == 2624272 and
            totals["response_or_context_wait_cycles"] == 2011048 and
            totals["parent_wait_cycles"] == 1947448 and
            totals["weight_dma_wait_cycles"] == 122880 and residual == 7496,
            "integrated overhead decomposition drift")

    per_sample_bank_bounds = []
    for row in samples:
        lower = (row["unique_weight_issues"] + 7) // 8
        per_sample_bank_bounds.append({
            "sample_id": row["sample_id"],
            "source_cycles": row["source_only_cycles"],
            "unique_weight_issues": row["unique_weight_issues"],
            "global_eight_bank_work_lower_bound": lower,
            "optimistic_bank_or_group_slack":
                row["source_only_cycles"] - lower,
        })

    conv2 = [row for row in operator_rows if "conv2" in row["operator"]]
    require(len(conv2) == 2, "conv2 operator population drift")
    conv2_zero = sum(row["zero_source_groups"] for row in conv2)
    conv2_non_source = sum(row["non_source_cycles"] for row in conv2)
    conv2_command = sum(row["command_or_state_wait_cycles"] for row in conv2)
    conv2_response = sum(row["response_or_context_wait_cycles"] for row in conv2)
    conv2_parent = sum(row["parent_wait_cycles"] for row in conv2)
    require(conv2_zero == 832288 and conv2_non_source == 6001352,
            "conv2 localization drift")

    def elimination_bound(cycles):
        return {
            "affected_cycles": cycles,
            "fraction_of_integrated": ratio(cycles, total),
            "optimistic_integrated_cycles_if_fully_eliminated": total - cycles,
            "optimistic_amdahl_speedup": ratio(total, total - cycles),
        }

    bottlenecks = [
        {
            "rank": 1,
            "name": "finite_bank_fused_source_issue",
            "ledger": {
                "source_cycles": source,
                "integrated_share": ratio(source, total),
                "logical_source_updates": totals["logical_source_updates"],
                "unique_weight_issues_after_fusion":
                    totals["unique_weight_issues"],
                "fusion_reuse_fraction": ratio(
                    totals["logical_source_updates"] -
                    totals["unique_weight_issues"],
                    totals["logical_source_updates"]),
                "fusion_groups": totals["fusion_groups"],
            },
            "absolute_zero_work_amdahl_bound": elimination_bound(source),
            "global_eight_bank_work_bound": {
                "qualification": "optimistic global conservation bound; exact per-group ceil(popcount/8) must be recomputed before promotion",
                "lower_bound_source_cycles": global_bank_lower,
                "maximum_removable_bank_or_group_slack": optimistic_slack,
                "lower_bound_integrated_cycles": overhead + global_bank_lower,
                "optimistic_integrated_speedup": ratio(
                    total, overhead + global_bank_lower),
            },
            "minimum_hardware": {
                "new_sram_ports": 0,
                "new_weight_capacity_bytes": 0,
                "new_runtime_state": "2-bit compile-time-selected bank-hash mode per operator, or hardwire the selected hash",
                "logic": "3-bit XOR/add bank decoder and offline weight-bank row permutation",
            },
            "existing_simulator_reuse": "yes; freeze M89 parent/order/group unions and replace global_feature_index_mod_8 bank masks with reversible skew-hash masks",
        },
        {
            "rank": 2,
            "name": "descriptor_command_and_state_wait",
            "ledger": {
                "wait_cycles": totals["command_or_state_wait_cycles"],
                "descriptor_commands": totals["descriptor_commands"],
                "zero_source_groups": totals["zero_source_groups"],
                "conv2_share_of_zero_groups": ratio(
                    conv2_zero, totals["zero_source_groups"]),
                "conv2_command_wait_cycles": conv2_command,
            },
            "full_wait_elimination_bound": elimination_bound(
                totals["command_or_state_wait_cycles"]),
            "frozen_prior_probe": {
                "m93_width2_integrated_cycles": 76649696,
                "actual_cycle_save": 27624,
                "actual_speedup": ratio(total, 76649696),
                "command_wait_save": 62880,
                "packet_lane_utilization": 0.5836527511120927,
                "regressed_samples": 2,
                "decision": "KILL_WIDE_DESCRIPTOR_PACKET",
            },
            "minimum_hardware": {
                "straight_width2": "second 64-byte descriptor lane, 128-byte packet/dequeue path and dual context allocation",
                "skip_variant": "zero-union alias/copy token plus parent-root tag state; payload correctness and final-accumulator materialization remain unresolved",
            },
            "existing_simulator_reuse": "yes; M93 already supports width 1/2/4, while a zero-union elision hook can be added around group_cycles==0",
        },
        {
            "rank": 3,
            "name": "response_context_and_final_accumulator_wait",
            "ledger": {
                "wait_cycles": totals["response_or_context_wait_cycles"],
                "final_accumulator_reads": totals["final_accumulator_reads"],
                "final_accumulator_writes": totals["final_accumulator_writes"],
                "final_accumulator_operations":
                    totals["final_accumulator_reads"] +
                    totals["final_accumulator_writes"],
                "conv2_response_wait_cycles": conv2_response,
                "conv2_non_source_cycles": conv2_non_source,
            },
            "full_wait_elimination_bound": elimination_bound(
                totals["response_or_context_wait_cycles"]),
            "minimum_hardware": {
                "new_capacity_bytes": 0,
                "port_or_bank_change": "split the existing 86400-byte timestep accumulator by task parity into two independently scheduled single-port banks, or use one 1R1W macro",
                "control_state": "one bank bit in the task tag and dual completion arbitration",
            },
            "existing_simulator_reuse": "yes; replace final_port with two parity-indexed PortCalendar instances and retain the frozen source/group schedule",
        },
    ]

    require(m43["aggregate"]["local_p8_l96_source_issue_cycles"] ==
            141484880 and
            m43["aggregate"]["parent_delta_p8_l96_source_issue_cycles"] ==
            113347744, "M43 source ledger drift")
    k4_temporal = [row for row in m53["configuration_summaries"]
                   if row["name"] == "K4_CTX16_TEMPORAL"]
    require(len(k4_temporal) == 1 and
            k4_temporal[0]["aggregate_source_only_cycles"] == 68847096,
            "M53 K4 temporal drift")

    negative_screens = {
        "m43_local_to_parent_delta_unfused_source_cycle_save": 28137136,
        "m43_local_to_parent_delta_unfused_reduction": ratio(
            28137136, 141484880),
        "m43_parent_delta_unfused_to_m89_k6_fused_source_save":
            113347744 - source,
        "m43_parent_delta_unfused_to_m89_k6_fused_reduction": ratio(
            113347744 - source, 113347744),
        "m55_full_network_zero_over_dual_source_work":
            m55["aggregate"]["opportunity_ratios_not_speedup"][
                "zero_over_dual_source_work"],
        "m55_is_cycle_or_speedup": False,
        "m90_k6_window_parent_integrated_regression":
            m90["comparisons"]["k6_window64_vs_m89_k6_temporal"][
                "integrated_cycle_regression"],
        "m91_integrated_save_vs_m89":
            -m91["comparison"][
                "integrated_cycle_delta_candidate_minus_baseline"],
        "m91_cycles_above_own_gate":
            m91["comparison"]["cycles_above_promotion_limit"],
        "m91_depends_on_coupled_scheduler": True,
        "m92_integrated_save_vs_m91":
            -m92["comparison_vs_m91"][
                "integrated_cycle_delta_candidate_minus_baseline"],
        "m92_source_regression_vs_m91":
            m92["comparison_vs_m91"][
                "source_cycle_delta_candidate_minus_baseline"],
        "m93_width2_integrated_save_vs_m89":
            -m93_receipt["width2_comparison_vs_m89_k6"][
                "integrated_cycle_delta"],
        "m94_critical_first_integrated_regression":
            m94_receipt["critical_first_comparison_vs_m89_k6"][
                "integrated_cycle_delta"],
        "m94_sparse_first_integrated_regression":
            [row for row in m94_receipt["configurations"]
             if row["policy"] == "sparse_first"][0]["integrated_cycles"] -
            total,
    }
    require(negative_screens["m90_k6_window_parent_integrated_regression"] ==
            664160 and negative_screens["m91_integrated_save_vs_m89"] ==
            746504 and negative_screens["m91_cycles_above_own_gate"] == 20270 and
            negative_screens["m92_integrated_save_vs_m91"] == 79632 and
            negative_screens["m92_source_regression_vs_m91"] == 58184 and
            negative_screens["m93_width2_integrated_save_vs_m89"] == 27624 and
            negative_screens["m94_critical_first_integrated_regression"] ==
            157288 and
            negative_screens["m94_sparse_first_integrated_regression"] ==
            204176, "negative-screen comparison drift")

    audit = {
        "schema": "m96_direction_independent_audit_v1",
        "status": "PASS_UNIQUE_M96_DIRECTION_SELECTED",
        "identity": {
            "all_exact_sha_match": True,
            "sha256": identities,
        },
        "m89_k6_ctx16_ledger": {
            "records": 40,
            "samples": 10,
            "integrated_cycles": total,
            "source_cycles": source,
            "non_source_cycles": overhead,
            "source_share": ratio(source, total),
            "p95_integrated_cycles": 7843680,
            "unique_weight_issues": totals["unique_weight_issues"],
            "logical_source_updates": totals["logical_source_updates"],
            "fusion_groups": totals["fusion_groups"],
            "zero_source_groups": totals["zero_source_groups"],
            "command_or_state_wait_cycles":
                totals["command_or_state_wait_cycles"],
            "response_or_context_wait_cycles":
                totals["response_or_context_wait_cycles"],
            "parent_wait_cycles": totals["parent_wait_cycles"],
            "weight_dma_wait_cycles": totals["weight_dma_wait_cycles"],
            "unclassified_calendar_residual_cycles": residual,
            "descriptor_commands": totals["descriptor_commands"],
            "parent_partial_reads": totals["parent_partial_reads"],
            "parent_partial_writes": totals["parent_partial_writes"],
            "final_accumulator_reads": totals["final_accumulator_reads"],
            "final_accumulator_writes": totals["final_accumulator_writes"],
        },
        "per_operator": operator_rows,
        "per_sample_global_bank_bounds": per_sample_bank_bounds,
        "conv2_localization": {
            "zero_source_groups": conv2_zero,
            "fraction_of_all_zero_source_groups": ratio(
                conv2_zero, totals["zero_source_groups"]),
            "non_source_cycles": conv2_non_source,
            "fraction_of_all_non_source_cycles": ratio(
                conv2_non_source, overhead),
            "command_wait_cycles": conv2_command,
            "response_wait_cycles": conv2_response,
            "parent_wait_cycles": conv2_parent,
        },
        "ranked_bottlenecks": bottlenecks,
        "negative_and_boundary_screens": negative_screens,
        "unique_m96_recommendation": {
            "name": "M96_FIXED_GROUP_REVERSIBLE_BANK_SKEW_WEIGHT_PACKING_PROBE",
            "baseline": "exact M89 K6-C16, 40 records and 10 samples",
            "freeze": [
                "parent selection",
                "DAG and task admission order",
                "K6 group membership and union masks",
                "descriptor width=1 and 16 contexts",
                "weight bytes, rows, ports, lanes and clock assumptions"
            ],
            "variants": {
                "H0": "bank=b",
                "H1": "bank=b XOR row[2:0]",
                "H2": "bank=(b+row[2:0]) mod 8",
                "H3": "bank=(b+3*row[2:0]) mod 8"
            },
            "selection_scope": "one fixed two-bit hash mode per operator across all samples; no per-group or per-sample adaptation",
            "data_layout": "offline permute each eight-entry weight row across the existing eight banks; row depth and total payload remain unchanged",
            "runtime_hardware": "two configuration bits per operator plus 3-bit XOR/add bank decode; zero extra SRAM ports and zero extra weight bytes",
            "simulator_reuse": "instrument the existing M53/M89 group builder once to seal ordered union masks, then replay bank_issue_cycles under H0-H3 without changing groups or calendars except source completion times",
            "predeclared_promotion_gates": {
                "exact_40_record_10_sample_replay": True,
                "group_identity_and_union_mask_conservation": True,
                "weight_payload_and_dma_bytes_unchanged": True,
                "new_sram_ports_equal_zero": True,
                "new_weight_capacity_bytes_equal_zero": True,
                "each_sample_source_cycles_must_not_regress": True,
                "each_sample_integrated_cycles_must_not_regress": True,
                "aggregate_source_cycles_le_69614355": True,
                "aggregate_integrated_cycles_le_76293933": True,
                "p95_integrated_cycles_lt_7843680": True
            },
            "gate_meaning": "at least 0.5 percent aggregate source and integrated improvement versus M89 K6 before RTL",
            "claim_boundary": "transaction-model bank-layout screen only; not RTL, PPA, energy, full-network or system speedup"
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    log_lines = [
        "status=PASS_UNIQUE_M96_DIRECTION_SELECTED",
        "baseline_integrated_cycles={}".format(total),
        "baseline_source_cycles={}".format(source),
        "source_share={:.12f}".format(float(source) / float(total)),
        "unique_weight_issues={}".format(totals["unique_weight_issues"]),
        "global_eight_bank_work_lower_bound={}".format(global_bank_lower),
        "optimistic_bank_or_group_slack={}".format(optimistic_slack),
        "optimistic_bank_bound_speedup={:.12f}".format(
            float(total) / float(overhead + global_bank_lower)),
        "command_wait_cycles={}".format(
            totals["command_or_state_wait_cycles"]),
        "response_wait_cycles={}".format(
            totals["response_or_context_wait_cycles"]),
        "parent_wait_cycles={}".format(totals["parent_wait_cycles"]),
        "zero_source_groups={}".format(totals["zero_source_groups"]),
        "conv2_zero_source_groups={}".format(conv2_zero),
        "recommended=M96_FIXED_GROUP_REVERSIBLE_BANK_SKEW_WEIGHT_PACKING_PROBE",
        "new_sram_ports=0",
        "new_weight_capacity_bytes=0",
    ]
    Path(args.log).write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "integrated": total,
        "source": source,
        "bank_slack_upper": optimistic_slack,
        "m96": audit["unique_m96_recommendation"]["name"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
