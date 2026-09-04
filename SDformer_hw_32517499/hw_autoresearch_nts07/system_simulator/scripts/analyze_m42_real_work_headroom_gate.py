#!/usr/bin/env python3
from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from fractions import Fraction


HW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONTRACT = os.path.join(
    HW_ROOT, "contracts", "m42_real_work_headroom_gate_contract_r1_20260823.json")


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def no_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_constant(value):
    raise AuditError("nonstandard JSON number: %s" % value)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=no_duplicate_pairs,
                         parse_constant=reject_constant)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def exact_int(value, name):
    require(type(value) is int, "%s must be an exact JSON integer" % name)
    return value


def exact_bool(value, name):
    require(type(value) is bool, "%s must be an exact JSON boolean" % name)
    return value


def exact_fraction(value, name):
    require(type(value) is dict, "%s must be an object" % name)
    require(set(value.keys()) == set(["numerator", "denominator"]),
            "%s fraction keys drift" % name)
    numerator = exact_int(value["numerator"], name + ".numerator")
    denominator = exact_int(value["denominator"], name + ".denominator")
    require(denominator > 0, "%s denominator must be positive" % name)
    return Fraction(numerator, denominator)


def fraction_json(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def resolve_pinned(identity, name):
    entry = identity[name]
    require(type(entry) is dict, "identity.%s must be an object" % name)
    require(set(entry.keys()) == set(["path", "sha256"]),
            "identity.%s keys drift" % name)
    relative = entry["path"]
    expected = entry["sha256"]
    require(type(relative) is str and relative and not os.path.isabs(relative),
            "identity.%s path must be nonempty relative string" % name)
    path = os.path.realpath(os.path.join(HW_ROOT, relative))
    root_real = os.path.realpath(HW_ROOT) + os.sep
    require(path.startswith(root_real), "identity.%s escapes hardware root" % name)
    require(os.path.isfile(path) and not os.path.islink(path),
            "identity.%s must resolve to a regular non-symlink file" % name)
    require(type(expected) is str and len(expected) == 64,
            "identity.%s SHA must be a 64-character string" % name)
    actual = sha256_file(path)
    require(actual == expected, "identity.%s SHA mismatch" % name)
    return path, actual


def select_local_m35_row(m39):
    rows = m39["conditional_dse"]["four_bottleneck_rows"]
    matches = [row for row in rows
               if row.get("line") == "Local"
               and row.get("late_scale_implementation")
               == "M35_parallel_complement_CSD_sidecar"]
    require(len(matches) == 1, "expected exactly one Local M35 row")
    return matches[0]


def validate_review(review):
    require(review.get("schema") == "m40a_r3_independent_hammer_review_v1",
            "M40 independent review schema drift")
    require(review.get("status") == "GO_M40A_R3_EXACT_TRACE_AND_ALGEBRA_ONLY",
            "M40 independent review is not GO for exact trace/algebra")
    rerun = review["mandatory_rereview_passes"]["independent_raw_rebuild"]
    require(exact_int(rerun["float32_values_bit_mitered"],
                      "review.float32_values_bit_mitered") == 92160000,
            "independent value population drift")
    require(exact_int(rerun["float32_value_bit_mismatches"],
                      "review.float32_value_bit_mismatches") == 0,
            "independent bit miter is not exact")


def build_result(contract):
    require(contract.get("schema") == "m42_real_work_headroom_gate_contract_v1",
            "contract schema drift")
    identity = contract["identity"]
    m39_path, m39_sha = resolve_pinned(identity, "m39_result")
    m40_path, m40_sha = resolve_pinned(identity, "m40_result")
    review_path, review_sha = resolve_pinned(identity, "m40_independent_review")
    m39 = load_json(m39_path)
    m40 = load_json(m40_path)
    review = load_json(review_path)
    validate_review(review)

    require(m39.get("status")
            == "PASS_M39_R3_CURRENT_ANCHORS_CONDITIONAL_BOTTLENECK_DSE_ONLY",
            "M39 status drift")
    require(exact_bool(m39["admission"]["system_speedup_admitted"],
                       "m39.system_speedup_admitted") is False,
            "M39 unexpectedly admits system speedup")
    m40_status = m40.get("status")
    require(type(m40_status) is str
            and m40_status.startswith("PASS_M40A_EXACT_AMPLITUDE_CODEBOOK"),
            "M40 status drift")
    require(exact_bool(m40["admission"]["system_speedup_admitted"],
                       "m40.system_speedup_admitted") is False,
            "M40 unexpectedly admits system speedup")

    frozen = contract["frozen_model"]
    row = select_local_m35_row(m39)
    fixed = exact_int(m39["conditional_dse"]["fixed_compute_reference_cycles"],
                      "fixed_compute_reference_cycles")
    before_model = exact_int(row["m38_model_substituted_ideal_before_scope_cycles"],
                             "m38_model_before_scope_cycles")
    before_scope = exact_int(row["before_cycles"], "before_scope_cycles")
    event_work = exact_int(row["replacement"]["conditional_m4_projected_event_cycles"],
                           "conditional_event_work_quanta")
    late = exact_int(row["replacement"]["late_scale_cycles"], "late_scale_cycles")
    frontend = exact_int(row["replacement"]["proportional_frontend_control_cycles"],
                         "frontend_control_cycles")
    replacement_total = exact_int(row["replacement"]["total_cycles"],
                                  "conditional_replacement_total_cycles")
    issue_width = exact_int(frozen["event_engine_issue_width"],
                            "contract.frozen_model.event_engine_issue_width")
    output_lanes = exact_int(frozen["event_engine_output_lanes"],
                             "contract.frozen_model.event_engine_output_lanes")
    peak_adds = exact_int(frozen["event_engine_peak_product_adds_per_cycle"],
                          "contract.frozen_model.event_engine_peak_product_adds_per_cycle")
    require(issue_width == 8 and output_lanes == 96
            and peak_adds == issue_width * output_lanes,
            "P8-L96 event-engine geometry drift")
    observed = {
        "fixed_compute_reference_cycles": fixed,
        "m38_model_before_scope_cycles": before_model,
        "four_bottleneck_before_cycles": before_scope,
        "conditional_event_work_quanta": event_work,
        "late_scale_cycles": late,
        "frontend_control_cycles": frontend,
        "conditional_replacement_total_cycles": replacement_total,
    }
    for key, value in observed.items():
        require(exact_int(frozen[key], "contract.frozen_model.%s" % key) == value,
                "frozen M39 value drift: %s" % key)
    require(replacement_total == event_work + late + frontend,
            "M39 replacement total is not conserved")

    distributions = m40["real_source_trace"][
        "exact_work_lower_bound_distribution_by_line"]
    local = distributions["Local"]
    motion = distributions["Motion"]
    local_mean = exact_fraction(local["mean_exact"], "Local.mean_exact")
    motion_mean = exact_fraction(motion["mean_exact"], "Motion.mean_exact")
    local_max = Fraction(exact_int(local["maximum"], "Local.maximum"), 1)
    motion_max = Fraction(exact_int(motion["maximum"], "Motion.maximum"), 1)
    require(exact_int(local["count"], "Local.count") == 10,
            "Local sample count drift")
    require(exact_int(motion["count"], "Motion.count") == 10,
            "Motion sample count drift")
    require(local["p95_nearest_rank"] == local["maximum"],
            "Local p95/max relation drift")
    require(motion["p95_nearest_rank"] == motion["maximum"],
            "Motion p95/max relation drift")
    require(motion_mean > local_mean, "pure Motion is no longer worse than Local")

    outside_local = Fraction(before_model - before_scope, 1)
    overhead_local = Fraction(late + frontend, 1)
    local_mean_total = outside_local + overhead_local + local_mean
    local_max_total = outside_local + overhead_local + local_max

    motion_row = [candidate for candidate in
                  m39["conditional_dse"]["four_bottleneck_rows"]
                  if candidate.get("line") == "Motion"
                  and candidate.get("late_scale_implementation")
                  == "M35_parallel_complement_CSD_sidecar"]
    require(len(motion_row) == 1, "expected exactly one Motion M35 row")
    motion_row = motion_row[0]
    motion_before = exact_int(
        motion_row["m38_model_substituted_ideal_before_scope_cycles"],
        "motion.m38_model_before_scope_cycles")
    motion_before_scope = exact_int(motion_row["before_cycles"],
                                    "motion.before_scope_cycles")
    motion_overhead = Fraction(
        exact_int(motion_row["replacement"]["late_scale_cycles"],
                  "motion.late_scale_cycles")
        + exact_int(motion_row["replacement"]["proportional_frontend_control_cycles"],
                    "motion.frontend_control_cycles"), 1)
    outside_motion = Fraction(motion_before - motion_before_scope, 1)
    motion_mean_total = outside_motion + motion_overhead + motion_mean
    motion_max_total = outside_motion + motion_overhead + motion_max

    gates = []
    for index, target_json in enumerate(contract["target_speedups"]):
        target = exact_fraction(target_json, "target_speedups[%d]" % index)
        require(target > 1, "target speedup must exceed one")
        total_ceiling = Fraction(fixed, 1) / target
        replacement_budget = total_ceiling - outside_local
        product_budget = replacement_budget - overhead_local
        require(product_budget > 0, "target leaves no positive product budget")
        gates.append({
            "target_compute_speedup": fraction_json(target),
            "total_cycle_ceiling_sensitivity": fraction_json(total_ceiling),
            "maximum_replacement_total_sensitivity": fraction_json(
                replacement_budget),
            "maximum_executable_product_cycles_required": fraction_json(
                product_budget),
            "required_effective_source_issue_width_from_local_mean": fraction_json(
                local_mean / product_budget),
            "required_effective_source_issue_width_from_local_p95": fraction_json(
                local_max / product_budget),
            "issue_width_peak": issue_width,
            "peak_issue_width_margin_from_local_mean": fraction_json(
                Fraction(issue_width, 1) / (local_mean / product_budget)),
            "conditional_model_headroom_cycles": fraction_json(
                replacement_budget - Fraction(replacement_total, 1)),
            "real_executable_schedule_admitted": False,
            "target_crossing_admitted": False,
        })

    return {
        "schema": "m42_real_work_headroom_gate_result_v1",
        "status": "PASS_M42_EXACT_PERFORMANCE_BUDGETS_ONLY_REAL_EXECUTABLE_SCHEDULE_PENDING",
        "identity": {
            "contract_sha256": sha256_file(DEFAULT_CONTRACT),
            "m39_result_sha256": m39_sha,
            "m40_result_sha256": m40_sha,
            "m40_independent_review_sha256": review_sha,
        },
        "frozen_resource_model": {
            "line": "Local",
            "fixed_compute_reference_cycles": fixed,
            "outside_four_bottleneck_model_cycles": outside_local.numerator,
            "fixed_late_scale_plus_frontend_cycles": overhead_local.numerator,
            "conditional_projected_event_work_quanta": event_work,
            "conditional_projected_replacement_total_cycles": replacement_total,
            "event_engine_issue_width": issue_width,
            "event_engine_output_lanes": output_lanes,
            "event_engine_peak_product_adds_per_cycle": peak_adds,
            "conditional_projected_compute_speedup": fraction_json(
                Fraction(fixed, row["conditional_cycles_after_scope_substitution"])),
        },
        "independently_reviewed_real_work": {
            "qualification": "PRODUCT_COUNT_DIV_96_LOWER_BOUND_WORK_QUANTA_NOT_EXECUTABLE_CYCLES",
            "local_mean": fraction_json(local_mean),
            "local_p95": fraction_json(local_max),
            "local_p99": fraction_json(local_max),
            "motion_mean": fraction_json(motion_mean),
            "motion_p95": fraction_json(motion_max),
            "motion_over_local_mean": fraction_json(motion_mean / local_mean),
            "pure_motion_is_worse_on_this_cohort": True,
        },
        "non_executable_diagnostic_envelopes": {
            "local_uncoalesced_mean_total": fraction_json(local_mean_total),
            "local_uncoalesced_mean_compute_speedup": fraction_json(
                Fraction(fixed, 1) / local_mean_total),
            "local_uncoalesced_p95_total": fraction_json(local_max_total),
            "local_uncoalesced_p95_compute_speedup": fraction_json(
                Fraction(fixed, 1) / local_max_total),
            "motion_uncoalesced_mean_total": fraction_json(motion_mean_total),
            "motion_uncoalesced_mean_compute_speedup": fraction_json(
                Fraction(fixed, 1) / motion_mean_total),
            "motion_uncoalesced_p95_total": fraction_json(motion_max_total),
            "motion_uncoalesced_p95_compute_speedup": fraction_json(
                Fraction(fixed, 1) / motion_max_total),
            "conditional_model_implied_effective_source_issue_width": fraction_json(
                local_mean / Fraction(event_work, 1)),
            "executable_or_system_metric_admitted": False,
        },
        "target_gates": gates,
        "admission": {
            "exact_budget_math_admitted": True,
            "m39_m40_identity_chain_admitted": True,
            "real_executable_schedule_admitted": False,
            "target_2p5_crossing_admitted": False,
            "target_2p7_crossing_admitted": False,
            "target_3p0_crossing_admitted": False,
            "system_speedup_admitted": False,
            "rtl_synopsys_ppa_power_energy_admitted": False,
            "headline_or_best_paper_admitted": False,
        },
        "required_next_gate": {
            "description": "Use exact INT8 weights, physical addresses, finite banks, accumulator ownership and memory service to produce executable Local cycles per sample.",
            "must_report": [
                "mean/p95/p99 product-engine cycles",
                "bank conflicts, queue stalls and utilization",
                "weight and accumulator SRAM/DRAM traffic",
                "integer output miter and quantized accuracy",
                "same-resource baseline cycles"
            ]
        },
        "claim_boundary": contract["claim_policy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract_path = os.path.realpath(args.contract)
    require(contract_path == os.path.realpath(DEFAULT_CONTRACT),
            "only the canonical M42 contract is accepted")
    require(not os.path.lexists(args.output), "refusing to overwrite output")
    contract = load_json(contract_path)
    result = build_result(contract)
    output_parent = os.path.dirname(os.path.abspath(args.output))
    if output_parent and not os.path.isdir(output_parent):
        os.makedirs(output_parent)
    with open(args.output, "x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(result["status"])


if __name__ == "__main__":
    try:
        main()
    except (AuditError, KeyError, TypeError, ValueError, OSError) as error:
        print("M42_AUDIT_FAIL: %s" % error)
        raise SystemExit(2)
