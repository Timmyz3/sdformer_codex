#!/usr/bin/env python3
"""Fail-closed static audit for the M37-r10 area-recovery candidate.

This checker proves only source topology and arithmetic consistency.  It does
not compile RTL and deliberately cannot admit VCS, DC, STA, or Formality.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import re
import sys


EXPECTED_CONTRACT = (
    "hw_autoresearch_nts07/contracts/"
    "m37_r10_area_recovery_arch_contract_r1_20260822.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_strict(path):
    def reject_duplicate(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key: {0}".format(key))
            value[key] = item
        return value

    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate)


def require_exact_int(value, expected, name, failures):
    if type(value) is not int or value != expected:
        failures.append(
            "{0}: expected exact int {1}, got {2!r}".format(
                name, expected, value
            )
        )


def require_exact_bool(value, expected, name, failures):
    if type(value) is not bool or value is not expected:
        failures.append(
            "{0}: expected exact bool {1}, got {2!r}".format(
                name, expected, value
            )
        )


def resolve(root, relative):
    return os.path.realpath(os.path.join(root, relative))


def audit(root, contract_path, rtl_path, enforce_candidate_hash):
    failures = []
    contract = load_json_strict(contract_path)
    with open(rtl_path, "r") as handle:
        rtl = handle.read()

    if contract.get("schema") != "m37_r10_area_recovery_arch_contract_v1":
        failures.append("contract schema mismatch")
    require_exact_bool(contract.get("headline_admitted"), False,
                       "headline_admitted", failures)
    require_exact_bool(
        contract.get("area_admission", {}).get(
            "static_equation_is_not_a_cell_area_prediction"
        ), True, "static_equation_is_not_a_cell_area_prediction", failures
    )

    resource = contract.get("resource_equation", {})
    terms = resource.get("terms", {})
    expected_terms = {
        "active_bank_and_compute_control": 7,
        "config_and_protocol_flags": 2,
        "done_state": 49,
        "fifo_state": 2365,
        "input_bank_payload_and_tags": 864,
        "layer_static_phase_table_and_threshold": 864,
        "product_stage_data_bias_tag_beat_valid": 1828,
    }
    for name, expected in expected_terms.items():
        require_exact_int(terms.get(name), expected,
                          "resource_equation.terms.{0}".format(name), failures)
    calculated_state = sum(expected_terms.values())
    require_exact_int(
        resource.get("candidate_architectural_state_bits_before_optimization"),
        calculated_state,
        "candidate_architectural_state_bits_before_optimization", failures
    )
    require_exact_int(
        resource.get("r8_architectural_state_bits_before_optimization"),
        5931, "r8_architectural_state_bits_before_optimization", failures
    )
    require_exact_int(resource.get("delta_bits"), 48, "delta_bits", failures)

    architecture = contract.get("architecture", {})
    phase = architecture.get("phase_bundle", {})
    bank = architecture.get("active_bank_mux", {})
    require_exact_int(phase.get("packed_output_bits"), 168,
                      "phase_bundle.packed_output_bits", failures)
    require_exact_int(phase.get("table_bits"), 840,
                      "phase_bundle.table_bits", failures)
    require_exact_int(phase.get("two_to_one_mux_bit_equivalents_upper_bound"),
                      672, "phase_bundle.mux_upper_bound", failures)
    require_exact_int(bank.get("two_to_one_mux_bit_equivalents"), 432,
                      "active_bank_mux.mux_bits", failures)
    require_exact_int(
        architecture.get(
            "selector_total_two_to_one_mux_bit_equivalents_upper_bound"
        ), 1104, "selector_total_mux_upper_bound", failures
    )

    area = contract.get("area_admission", {})
    r8_area = area.get("r8_fresh_logic_only_cell_area_um2")
    area_cap = area.get("candidate_area_maximum_um2")
    if type(r8_area) is not float or abs(r8_area - 63671.579642) > 1e-9:
        failures.append("r8 exact area anchor mismatch")
    if type(area_cap) is not float or abs(area_cap - (63671.579642 * 1.10)) > 1e-6:
        failures.append("area cap is not exact r8 area times 1.10")

    required_snippets = [
        "logic [(PHASES*PHASE_BUNDLE_W)-1:0] phase_table_q;",
        "logic [(INTERMEDIATES*IN_W)-1:0] intermediate_bank0_q;",
        "logic [(INTERMEDIATES*IN_W)-1:0] intermediate_bank1_q;",
        "product_bias_pair_q <= phase_bias_pair_comb;",
        "assign uses_integer_multiplier = 1'b0;",
        "phase_table_q[0 +: PHASE_BUNDLE_W]",
        "phase_table_q[168 +: PHASE_BUNDLE_W]",
        "phase_table_q[336 +: PHASE_BUNDLE_W]",
        "phase_table_q[504 +: PHASE_BUNDLE_W]",
        "phase_table_q[672 +: PHASE_BUNDLE_W]",
    ]
    for snippet in required_snippets:
        if snippet not in rtl:
            failures.append("missing required RTL structure: {0}".format(snippet))

    phase_arms = re.findall(
        r"3'd([0-4])\s*:\s*phase_bundle_comb\s*=\s*phase_table_q\[",
        rtl
    )
    if sorted(phase_arms) != ["0", "1", "2", "3", "4"]:
        failures.append("phase case arms must be exactly 0,1,2,3,4")
    if len(re.findall(r"phase_bundle_comb\s*=\s*phase_table_q\[", rtl)) != 5:
        failures.append("phase bundle selector must have exactly five table arms")

    forbidden_patterns = {
        "r9 equality-expanded selector": r"selected_coefficient\s*==",
        "r9 30-coefficient scan": r"coefficient_index\s*<\s*COEFFICIENTS",
        "unpacked stored valid descriptors": r"term_valid_q\s*\[",
        "unpacked stored negative descriptors": r"term_negative_q\s*\[",
        "unpacked stored shifts": r"term_shift_q\s*\[",
        "unpacked stored biases": r"bias_q\s*\[",
        "dynamic selected row": r"selected_row",
        "runtime coefficient selector": r"selected_coefficient",
        "runtime intermediate selector": r"selected_intermediate",
        "nonzero multiplier flag": r"uses_integer_multiplier\s*=\s*1'b1",
        "multiply-by-rank runtime index": r"selected_row\s*\*\s*RANK",
    }
    for name, pattern in forbidden_patterns.items():
        if re.search(pattern, rtl):
            failures.append("forbidden RTL structure: {0}".format(name))

    if rtl.count("signed_power_term(") != 5:
        failures.append(
            "expected one function definition plus four generated term calls; got {0}"
            .format(rtl.count("signed_power_term("))
        )
    if rtl.count("output_event(") != 2:
        failures.append(
            "expected one function definition plus one generated result call; got {0}"
            .format(rtl.count("output_event("))
        )

    candidate = contract.get("r10_candidate", {})
    candidate_path = resolve(root, candidate.get("path", ""))
    if enforce_candidate_hash and os.path.realpath(rtl_path) == candidate_path:
        actual_sha = sha256_file(rtl_path)
        if actual_sha != candidate.get("sha256"):
            failures.append("canonical r10 candidate SHA mismatch")

    for anchor_name, anchor in contract.get("frozen_anchors", {}).items():
        anchor_path = resolve(root, anchor.get("path", ""))
        if not os.path.isfile(anchor_path):
            failures.append("missing frozen anchor: {0}".format(anchor_name))
        elif sha256_file(anchor_path) != anchor.get("sha256"):
            failures.append("frozen anchor SHA mismatch: {0}".format(anchor_name))

    result = {
        "candidate_rtl_sha256": sha256_file(rtl_path),
        "contract_sha256": sha256_file(contract_path),
        "failures": failures,
        "headline_admitted": False,
        "resource_equation": {
            "active_and_phase_selector_mux_bit_upper_bound": 1104,
            "candidate_state_bits": calculated_state,
            "r8_state_bits": 5931,
            "state_delta_bits": 48,
        },
        "scope": "STATIC_SOURCE_ARCHITECTURE_ONLY",
        "status": (
            "PASS_M37_R10_STATIC_AREA_RECOVERY_ARCHITECTURE_ONLY"
            if not failures else
            "FAIL_M37_R10_STATIC_AREA_RECOVERY_ARCHITECTURE"
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--contract", default=EXPECTED_CONTRACT)
    parser.add_argument("--rtl", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-enforce-candidate-hash", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_root = os.path.realpath(os.path.join(script_dir, "../../.."))
    root = os.path.realpath(args.root or default_root)
    contract_path = resolve(root, args.contract)
    contract = load_json_strict(contract_path)
    rtl_relative = args.rtl or contract["r10_candidate"]["path"]
    rtl_path = resolve(root, rtl_relative)
    result = audit(root, contract_path, rtl_path,
                   not args.no_enforce_candidate_hash)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
