#!/usr/bin/env python3
"""Independent, raw-ledger M92 commit-bus parent-bypass audit.

This validator never imports or executes the M92 producer.  It verifies exact
identities, rebuilds sample and aggregate counters from the sealed raw ledgers,
and statically audits the producer's forwarding semantics and cost omissions.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_SHA256 = {
    "contract": "4f7063ae00c55bd0926a834a5f11c70547659282f81c85fd17d1e591af08d550",
    "probe": "7b6a16350dee495555eca3bb4034c39599a682bcf87a70cf21b5d8e5ffb271d8",
    "raw": "fafff857c3ac3c769e05d52a75fa352fb562c4e9a74874a11710cbb2b99cca3f",
    "r1_failure_log": "2caed0770fef0b6fe69a8c1de2994e2cc3bccf94e95ebd00af8c9033bb92bd11",
    "r2_complete_log": "8823ea307c65745e656fde6533e3e28528932492dc60262efc843a2628df0a18",
    "receipt": "4af6c4aa18f7ba012a70321b9ea96c7f244a8472d755f81e644bbdaf2e3456dc",
    "m91_probe": "c6bf6d37713137c3e63067ead2ab0460856098d9b9f3d1c613359b48dc88f97a",
    "m91_raw": "6245514b51c1d15a62d994be262a9a5da24235ad9c04b8dda919a8d68da70011",
    "m91_receipt": "83a3fe67e592e0fee1b619329e612798eee5da443285d35ce914d0fe2a9539a1",
    "m89_receipt": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
    "m45_analyzer": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
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
    def reject_constant(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def reject_duplicate(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject_constant,
                      object_pairs_hook=reject_duplicate)


def nearest_rank(values, percent):
    ordered = sorted(values)
    require(ordered, "empty distribution")
    rank = (percent * len(ordered) + 99) // 100
    return ordered[rank - 1]


def exact_fraction(numerator, denominator):
    require(denominator != 0, "zero denominator")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def require_text(text, needle, label):
    require(needle in text, "static source clause missing: {}".format(label))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    hw = Path(args.hw_root).resolve()
    paths = {
        "contract": hw / "contracts/m92_commit_bus_parent_bypass_contract_r1_20260824.json",
        "probe": hw / "system_simulator/scripts/probe_m92_commit_bus_parent_bypass.py",
        "raw": hw / "results/m92_commit_bus_parent_bypass_probe_r1_20260824/remote_artifacts/m92_commit_bus_parent_bypass_probe_r1_20260824.json",
        "r1_failure_log": hw / "results/m92_commit_bus_parent_bypass_probe_r1_20260824/remote_artifacts/m92_commit_bus_parent_bypass_probe_r1_20260824.log",
        "r2_complete_log": hw / "results/m92_commit_bus_parent_bypass_probe_r1_20260824/remote_artifacts/m92_commit_bus_parent_bypass_probe_r2_20260824.log",
        "receipt": hw / "results/m92_commit_bus_parent_bypass_probe_r1_20260824/m92_commit_bus_parent_bypass_probe_receipt_r1.json",
        "m91_probe": hw / "system_simulator/scripts/probe_m91_dependency_safe_fusion_aware_parent.py",
        "m91_raw": hw / "results/m91_dependency_safe_fusion_aware_parent_probe_r1_20260824/remote_artifacts/m91_dependency_safe_fusion_aware_parent_probe_r1_20260824.json",
        "m91_receipt": hw / "results/m91_dependency_safe_fusion_aware_parent_probe_r1_20260824/m91_dependency_safe_fusion_aware_parent_probe_receipt_r1.json",
        "m89_receipt": hw / "results/m89_temporal_fanout_hold_screen_r1_20260823/m89_temporal_fanout_hold_screen_receipt.json",
        "m45_analyzer": hw / "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py",
    }
    observed_sha = {}
    for name, path in paths.items():
        require(path.is_file(), "missing input {}: {}".format(name, path))
        observed_sha[name] = sha256(path)
        require(observed_sha[name] == EXPECTED_SHA256[name],
                "{} SHA drift".format(name))

    contract = read_json(paths["contract"])
    raw = read_json(paths["raw"])
    receipt = read_json(paths["receipt"])
    m91 = read_json(paths["m91_raw"])
    m91_receipt = read_json(paths["m91_receipt"])
    m89_receipt = read_json(paths["m89_receipt"])
    probe_text = paths["probe"].read_text(encoding="utf-8")
    m91_probe_text = paths["m91_probe"].read_text(encoding="utf-8")
    m45_text = paths["m45_analyzer"].read_text(encoding="utf-8")
    r1_log = paths["r1_failure_log"].read_text(encoding="utf-8")
    r2_log = paths["r2_complete_log"].read_text(encoding="utf-8")

    # Exact failure/complete lineage.
    require("AttributeError: module 'm91_m45' has no attribute 'WIDTH'" in r1_log,
            "R1 fail-closed WIDTH error missing")
    require(r2_log.count("[M92 K6]") == 40,
            "R2 record-completion marker count drift")
    require("M92_COMMIT_BUS_PARENT_BYPASS=" in r2_log and
            '"status": "PASS_EXECUTION_NO_GO_PROMOTION"' in r2_log,
            "R2 completion status missing")

    records = raw["record_ledger"]
    samples = raw["per_sample"]
    m91_records = m91["record_ledger"]
    m91_samples = m91["per_sample"]
    require(len(records) == len(m91_records) == 40, "record count drift")
    require(len(samples) == len(m91_samples) == 10, "sample count drift")
    require(sorted(row["sample_id"] for row in samples) == list(range(10)),
            "M92 sample IDs drift")
    require(sorted(row["sample_id"] for row in m91_samples) == list(range(10)),
            "M91 sample IDs drift")

    additive_fields = (
        "source_only_cycles", "integrated_cycles", "parent_wait_cycles",
        "command_or_state_wait_cycles", "response_or_context_wait_cycles",
        "weight_dma_wait_cycles", "logical_source_updates",
        "unique_weight_issues", "fusion_groups", "signed_add_updates",
        "signed_subtract_updates", "parent_vector_demands",
        "parent_sram_reads", "commit_bus_forward_hits", "forward_hits_left",
        "forward_hits_up", "late_commit_events_rejected",
    )
    maximum_fields = (
        "maximum_metadata_occupancy", "maximum_complete_occupancy",
        "maximum_resident_occupancy",
    )
    sample_map = dict((row["sample_id"], row) for row in samples)
    m91_sample_map = dict((row["sample_id"], row) for row in m91_samples)
    operator_counts = {}
    for sample_id in range(10):
        selected = [row for row in records if row["sample_id"] == sample_id]
        selected_m91 = [row for row in m91_records
                        if row["sample_id"] == sample_id]
        require(len(selected) == len(selected_m91) == 4,
                "sample {} does not have four records".format(sample_id))
        require(len(set(row["operator"] for row in selected)) == 4,
                "sample {} operator uniqueness drift".format(sample_id))
        operator_counts[str(sample_id)] = 4
        for field in additive_fields:
            if field in sample_map[sample_id]:
                require(sum(row[field] for row in selected) ==
                        sample_map[sample_id][field],
                        "M92 record-to-sample drift {} sample {}".format(
                            field, sample_id))
            if field in m91_sample_map[sample_id]:
                require(sum(row[field] for row in selected_m91) ==
                        m91_sample_map[sample_id][field],
                        "M91 record-to-sample drift {} sample {}".format(
                            field, sample_id))
        for field in maximum_fields:
            require(max(row[field] for row in selected) ==
                    sample_map[sample_id][field],
                    "M92 record-to-sample max drift {} sample {}".format(
                        field, sample_id))
            require(max(row[field] for row in selected_m91) ==
                    m91_sample_map[sample_id][field],
                    "M91 record-to-sample max drift {} sample {}".format(
                        field, sample_id))

    for row in records + samples:
        require(row["parent_sram_reads"] + row["commit_bus_forward_hits"] ==
                row["parent_vector_demands"],
                "parent demand conservation drift")
        require(row["forward_hits_left"] + row["forward_hits_up"] ==
                row["commit_bus_forward_hits"],
                "left/up hit conservation drift")
        require(row["signed_add_updates"] + row["signed_subtract_updates"] ==
                row["logical_source_updates"],
                "signed add/subtract conservation drift")

    aggregate = {}
    for field in additive_fields:
        aggregate[field] = sum(row[field] for row in samples)
    require(aggregate["source_only_cycles"] ==
            raw["aggregate_source_only_cycles"] == 69270080,
            "M92 source aggregate drift")
    require(aggregate["integrated_cycles"] ==
            raw["aggregate_integrated_cycles"] == 75851184,
            "M92 integrated aggregate drift")
    require(aggregate["parent_vector_demands"] ==
            raw["aggregate_parent_vector_demands"] == 11805832,
            "M92 parent demand aggregate drift")
    require(aggregate["parent_sram_reads"] ==
            raw["aggregate_parent_sram_reads"] == 10402176,
            "M92 parent read aggregate drift")
    require(aggregate["commit_bus_forward_hits"] ==
            raw["aggregate_commit_bus_forward_hits"] == 1403656,
            "M92 hit aggregate drift")
    require(aggregate["forward_hits_left"] ==
            raw["aggregate_forward_hits_left"] == 980856,
            "M92 left aggregate drift")
    require(aggregate["forward_hits_up"] ==
            raw["aggregate_forward_hits_up"] == 422800,
            "M92 up aggregate drift")
    require(aggregate["late_commit_events_rejected"] ==
            raw["aggregate_late_commit_events_rejected"] == 21070272,
            "M92 late-event aggregate drift")

    m91_aggregate = {}
    for field in ("source_only_cycles", "integrated_cycles",
                  "parent_wait_cycles", "command_or_state_wait_cycles",
                  "response_or_context_wait_cycles", "weight_dma_wait_cycles",
                  "logical_source_updates", "unique_weight_issues",
                  "fusion_groups"):
        m91_aggregate[field] = sum(row[field] for row in m91_samples)
    require(m91_aggregate["source_only_cycles"] ==
            m91["aggregate_source_only_cycles"] == 69211896,
            "M91 source aggregate drift")
    require(m91_aggregate["integrated_cycles"] ==
            m91["aggregate_integrated_cycles"] == 75930816,
            "M91 integrated aggregate drift")

    p95 = nearest_rank([row["integrated_cycles"] for row in samples], 95)
    m91_p95 = nearest_rank(
        [row["integrated_cycles"] for row in m91_samples], 95)
    require(p95 == raw["integrated_cycle_distribution"]["p95_nearest_rank"] ==
            7760888, "M92 p95 drift")
    require(m91_p95 ==
            m91["integrated_cycle_distribution"]["p95_nearest_rank"] ==
            7769480, "M91 p95 drift")

    delta_fields = ("source_only_cycles", "integrated_cycles",
                    "parent_wait_cycles")
    per_sample_deltas = []
    for sample_id in range(10):
        candidate = sample_map[sample_id]
        baseline = m91_sample_map[sample_id]
        per_sample_deltas.append({
            "sample_id": sample_id,
            "source_cycles": (candidate["source_only_cycles"] -
                              baseline["source_only_cycles"]),
            "integrated_cycles": (candidate["integrated_cycles"] -
                                  baseline["integrated_cycles"]),
            "parent_wait_cycles": (candidate["parent_wait_cycles"] -
                                   baseline["parent_wait_cycles"]),
        })
    require(per_sample_deltas ==
            receipt["per_sample_deltas_candidate_minus_m91"],
            "per-sample receipt delta drift")
    require(all(row["source_cycles"] > 0 for row in per_sample_deltas),
            "not every sample regresses source cycles")
    require(all(row["integrated_cycles"] < 0 for row in per_sample_deltas),
            "not every sample improves integrated cycles")

    hit_fraction = exact_fraction(
        aggregate["commit_bus_forward_hits"],
        aggregate["parent_vector_demands"])
    left_fraction = exact_fraction(
        aggregate["forward_hits_left"],
        aggregate["commit_bus_forward_hits"])
    up_fraction = exact_fraction(
        aggregate["forward_hits_up"],
        aggregate["commit_bus_forward_hits"])
    parent_wait_saved = (m91_aggregate["parent_wait_cycles"] -
                         aggregate["parent_wait_cycles"])
    source_delta = (aggregate["source_only_cycles"] -
                    m91_aggregate["source_only_cycles"])
    integrated_delta = (aggregate["integrated_cycles"] -
                        m91_aggregate["integrated_cycles"])
    target = contract["predeclared_promotion_gates"][
        "maximum_promotable_integrated_cycles"]
    gate_shortfall = aggregate["integrated_cycles"] - target
    require(hit_fraction["decimal"] ==
            raw["m92"]["comparison"][
                "parent_sram_read_bypass_fraction"]["decimal"],
            "hit fraction drift")
    require(parent_wait_saved == 34968 and source_delta == 58184 and
            integrated_delta == -79632 and gate_shortfall == 110196,
            "headline delta drift")

    m89_k6 = [row for row in m89_receipt["configurations"]
              if row["name"] == "K6"]
    require(len(m89_k6) == 1 and
            m89_k6[0]["integrated_cycles"] == 76677320,
            "M89 K6 baseline drift")
    combined_vs_m89 = exact_fraction(
        m89_k6[0]["integrated_cycles"] - aggregate["integrated_cycles"],
        m89_k6[0]["integrated_cycles"])
    require(combined_vs_m89["decimal"] > 0.01,
            "M92 combined M89 reduction no longer exceeds one percent")
    require(m91_receipt["gates"][
        "aggregate_integrated_cycles_le_75910546"] is False,
        "M91 unexpectedly promoted")
    require(raw["m92"]["gates"][
        "aggregate_source_cycles_must_not_exceed_m91_69211896"] is False and
        raw["m92"]["gates"][
            "aggregate_integrated_cycles_le_75740988"] is False and
        raw["m92"]["all_promotion_gates_pass"] is False,
        "M92 gate decision drift")

    # Static semantics.  These checks intentionally distinguish tag/calendar
    # modeling from physical vector-payload transport.
    static = {
        "same_cycle_commit_time_equal_now_only": all((
            require_text(probe_text, "if commit_time == limit:", "same-cycle equality"),
            require_text(probe_text, "commit_bus_tasks.add(task)", "same-cycle task publish"),
            require_text(probe_text, "require(commit_time < limit", "late event rejection"),
        )),
        "left_parent_task_is_task_minus_1": require_text(
            probe_text, "return task - 1", "left parent mapping"),
        "up_parent_task_is_task_minus_W": require_text(
            probe_text, "return task - r1.W", "up parent mapping"),
        "only_left_up_eligible": require_text(
            probe_text, 'selected["name"] in ("left", "up")',
            "left/up eligibility"),
        "previous_timestep_not_forward_eligible":
            'selected["name"] in ("left", "up")' in probe_text and
            'selected["name"] in ("left", "up", "previous_timestep")'
            not in probe_text,
        "hit_skips_parent_port_schedule": all((
            require_text(probe_text, "if forward_hit:", "hit branch"),
            require_text(probe_text, "parent_end = parent_port.schedule(now)",
                         "miss read scheduling"),
        )),
        "canonical_dag_frozen_before_reselection": require_text(
            probe_text,
            "r1.build_structural_dag(list(canonical_names))",
            "frozen canonical DAG"),
        "left_candidate_requires_canonical_left": require_text(
            m91_probe_text, 'if canonical_name == "left":',
            "dependency-safe left candidate"),
        "up_edge_present_in_frozen_dag": require_text(
            m45_text, "if y > 0:\n            indegree[spatial] += 1",
            "unconditional spatial up DAG edge"),
        "model_stores_only_commit_time_and_task": require_text(
            probe_text, "commit_batch.append((commit_time, task))",
            "tag-only commit event"),
        "model_declares_zero_payload_storage_gate": require_text(
            probe_text,
            '"additional_vector_payload_storage_bytes_equal_zero": True',
            "zero payload storage gate"),
        "new_dependency_gate_is_literal_not_dynamic_counter": require_text(
            probe_text, '"new_dependency_edges_equal_zero": True',
            "new dependency literal gate"),
        "explicit_commit_bus_payload_model": False,
        "explicit_commit_bus_capacity_or_fanout_calendar": False,
        "explicit_tag_compare_or_vector_route_latency": False,
    }

    vector_bytes = 96 * 3
    vector_bits = vector_bytes * 8
    require("LANES = 96" in m45_text and "ACC_BYTES = 3" in m45_text and
            "VECTOR_BYTES = LANES * ACC_BYTES" in m45_text,
            "parent vector geometry drift")

    audit = {
        "schema": "m92_commit_bus_parent_bypass_independent_audit_v1",
        "status": "PASS_RECOMPUTATION_NO_GO_CONFIRMED",
        "identity": {
            "all_exact_sha_match": True,
            "sha256": observed_sha,
            "r1_failure_is_fail_closed_width_interface_typo": True,
            "r2_completion_markers": 40,
        },
        "independent_raw_recomputation": {
            "records": len(records),
            "samples": len(samples),
            "records_per_sample": operator_counts,
            "aggregate_source_cycles": aggregate["source_only_cycles"],
            "aggregate_integrated_cycles": aggregate["integrated_cycles"],
            "p95_nearest_rank": p95,
            "m91_source_cycles": m91_aggregate["source_only_cycles"],
            "m91_integrated_cycles": m91_aggregate["integrated_cycles"],
            "m91_p95_nearest_rank": m91_p95,
            "per_sample_delta_candidate_minus_m91": per_sample_deltas,
            "all_samples_source_regress": True,
            "all_samples_integrated_improve": True,
            "signed_conservation": True,
        },
        "bypass_accounting": {
            "parent_vector_demands": aggregate["parent_vector_demands"],
            "parent_sram_reads": aggregate["parent_sram_reads"],
            "commit_bus_forward_hits": aggregate["commit_bus_forward_hits"],
            "demand_equals_reads_plus_hits": True,
            "hit_fraction": hit_fraction,
            "forward_hits_left": aggregate["forward_hits_left"],
            "forward_hits_up": aggregate["forward_hits_up"],
            "left_share_of_hits": left_fraction,
            "up_share_of_hits": up_fraction,
            "late_commit_events_rejected":
                aggregate["late_commit_events_rejected"],
            "parent_vector_bytes": vector_bytes,
            "parent_vector_bits": vector_bits,
            "modeled_parent_read_payload_bytes_avoided":
                aggregate["commit_bus_forward_hits"] * vector_bytes,
        },
        "cycle_coupling": {
            "parent_wait_cycle_delta": -parent_wait_saved,
            "source_cycle_delta": source_delta,
            "integrated_cycle_delta": integrated_delta,
            "command_or_state_wait_delta":
                (aggregate["command_or_state_wait_cycles"] -
                 m91_aggregate["command_or_state_wait_cycles"]),
            "response_or_context_wait_delta":
                (aggregate["response_or_context_wait_cycles"] -
                 m91_aggregate["response_or_context_wait_cycles"]),
            "fusion_group_delta": (aggregate["fusion_groups"] -
                                   m91_aggregate["fusion_groups"]),
            "unique_weight_issue_delta":
                (aggregate["unique_weight_issues"] -
                 m91_aggregate["unique_weight_issues"]),
            "logical_source_update_delta":
                (aggregate["logical_source_updates"] -
                 m91_aggregate["logical_source_updates"]),
            "parent_wait_cycles_saved_per_hit":
                float(parent_wait_saved) /
                float(aggregate["commit_bus_forward_hits"]),
            "hits_per_parent_wait_cycle_saved":
                float(aggregate["commit_bus_forward_hits"]) /
                float(parent_wait_saved),
        },
        "promotion_gate": {
            "maximum_promotable_integrated_cycles": target,
            "candidate_integrated_cycles": aggregate["integrated_cycles"],
            "cycles_above_gate": gate_shortfall,
            "required_cycle_save_vs_m91":
                m91_aggregate["integrated_cycles"] - target,
            "actual_cycle_save_vs_m91": -integrated_delta,
            "integrated_reduction_vs_m91": exact_fraction(
                -integrated_delta, m91_aggregate["integrated_cycles"]),
            "source_regression_vs_m91": exact_fraction(
                source_delta, m91_aggregate["source_only_cycles"]),
            "m92_self_gates_pass": False,
            "combined_reduction_vs_m89_k6": combined_vs_m89,
            "combined_m89_comparison_is_not_a_promotion_override": True,
            "no_go_is_correct": True,
        },
        "static_semantics": static,
        "claim_boundary": {
            "transaction_opportunity_only": True,
            "additional_payload_storage_modeled_bytes": 0,
            "new_dependency_edges_modeled": 0,
            "physical_zero_storage_proven": False,
            "commit_bus_width_fanout_and_timing_charged": False,
            "rtl_correctness": False,
            "rtl_cycle_speedup": False,
            "paper_ppa_ready": False,
            "system_speedup": False,
            "headline": False,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    log_lines = [
        "status=PASS_RECOMPUTATION_NO_GO_CONFIRMED",
        "exact_sha_files={}".format(len(observed_sha)),
        "records=40",
        "samples=10",
        "source_cycles=69270080",
        "integrated_cycles=75851184",
        "p95=7760888",
        "parent_demands=11805832",
        "parent_reads=10402176",
        "forward_hits=1403656",
        "hit_fraction={:.12f}".format(hit_fraction["decimal"]),
        "forward_left=980856",
        "forward_up=422800",
        "parent_wait_delta=-34968",
        "source_delta=58184",
        "integrated_delta=-79632",
        "cycles_above_gate=110196",
        "commit_bus_vector_bits=2304",
        "physical_bus_cost_charged=false",
        "promotion=NO_GO",
    ]
    Path(args.log).write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "records": 40,
        "samples": 10,
        "hit_fraction": hit_fraction["decimal"],
        "cycles_above_gate": gate_shortfall,
        "no_go": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
