#!/usr/bin/env python3
"""Fail-closed M37-r11 static evidence audit.

There is deliberately no candidate/contract/root override.  The canonical RTL
realpath and SHA are checked on every run before semantic checks.  This script
admits source evidence only; it cannot admit compilation, PPA, or equivalence.
"""

from __future__ import print_function

import hashlib
import json
import os
import re
import sys


SCRIPT_REL = (
    "hw_autoresearch_nts07/dc_handoff/scripts/"
    "audit_m37_r11_area_recovery_evidence.py"
)
CONTRACT_REL = (
    "hw_autoresearch_nts07/contracts/"
    "m37_r11_area_recovery_evidence_contract_r1_20260822.json"
)
PIN_REL = (
    "hw_autoresearch_nts07/contracts/"
    "m37_r11_evidence_pin_r1_20260822.json"
)
RTL_REL = (
    "hw_autoresearch_nts07/rtl_m37_r10/"
    "qfit_atlif_csd_reconstruct_t10.sv"
)
TEST_REL = (
    "hw_autoresearch_nts07/system_simulator/tests/"
    "test_m37_r11_area_recovery_evidence.py"
)
RESULT_REL = (
    "hw_autoresearch_nts07/results/"
    "m37_r11_area_recovery_evidence_r1_20260822/"
    "m37_r11_area_recovery_evidence.json"
)
README_REL = (
    "hw_autoresearch_nts07/results/"
    "m37_r11_area_recovery_evidence_r1_20260822/README.md"
)
R10_REVIEW_REL = (
    "hw_autoresearch_nts07/results/"
    "m37_r10_independent_hammer_review_20260822/"
    "m37_r10_independent_hammer_review.json"
)
EXPECTED_RTL_SHA = (
    "f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd"
)


def repository_root():
    script_path = os.path.realpath(__file__)
    root = os.path.realpath(os.path.join(os.path.dirname(script_path), "../../.."))
    expected_script = os.path.realpath(os.path.join(root, SCRIPT_REL))
    if script_path != expected_script:
        raise ValueError("auditor is not running from its canonical contained path")
    return root


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_strict(path):
    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: {0}".format(key))
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("non-finite JSON constant: {0}".format(value))

    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate,
                         parse_constant=reject_constant)


def canonical_contained_path(root, relative, expected_relative, failures, label):
    if type(relative) is not str:
        failures.append("{0}: path is not exact string".format(label))
        return None
    if relative != expected_relative:
        failures.append("{0}: path is not canonical".format(label))
        return None
    if os.path.isabs(relative) or os.path.normpath(relative) != relative:
        failures.append("{0}: path is absolute or non-normal".format(label))
        return None
    path = os.path.realpath(os.path.join(root, relative))
    prefix = root + os.sep
    if path != root and not path.startswith(prefix):
        failures.append("{0}: realpath escapes repository".format(label))
        return None
    return path


def type_strict_equal(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is list:
        return len(value) == len(expected) and all(
            type_strict_equal(left, right)
            for left, right in zip(value, expected)
        )
    if type(expected) is dict:
        return set(value.keys()) == set(expected.keys()) and all(
            type_strict_equal(value[key], expected[key]) for key in expected
        )
    return value == expected


def exact(value, expected, label, failures):
    if not type_strict_equal(value, expected):
        failures.append(
            "{0}: expected exact {1} {2!r}, got {3!r}".format(
                label, type(expected).__name__, expected, value
            )
        )


def nested(mapping, keys):
    value = mapping
    for key in keys:
        if type(value) is not dict or key not in value:
            return None
        value = value[key]
    return value


def validate_contract(contract, root):
    failures = []
    if type(contract) is not dict:
        return ["contract root is not object"]
    exact(contract.get("schema"),
          "m37_r11_area_recovery_evidence_contract_v1",
          "contract.schema", failures)
    exact(nested(contract, ["candidate", "path"]), RTL_REL,
          "candidate.path", failures)
    exact(nested(contract, ["candidate", "sha256"]), EXPECTED_RTL_SHA,
          "candidate.sha256", failures)
    canonical_contained_path(
        root, nested(contract, ["candidate", "path"]), RTL_REL,
        failures, "candidate"
    )
    exact(nested(contract, ["evidence_identity",
                            "alternate_candidate_path_allowed"]),
          False, "alternate_candidate_path_allowed", failures)
    exact(nested(contract, ["evidence_identity",
                            "candidate_realpath_must_be_contained"]),
          True, "candidate_realpath_must_be_contained", failures)
    exact(nested(contract, ["evidence_identity",
                            "candidate_sha_check_unconditional"]),
          True, "candidate_sha_check_unconditional", failures)
    exact(nested(contract, ["evidence_identity", "pin_manifest_path"]),
          PIN_REL, "pin_manifest_path", failures)
    exact(nested(contract, ["claim_boundary", "headline_admitted"]),
          False, "headline_admitted", failures)
    exact(nested(contract, ["claim_boundary", "physical_shared_mux_proven"]),
          False, "physical_shared_mux_proven", failures)
    exact(nested(contract, ["claim_boundary", "source_selector_factoring_only"]),
          True, "source_selector_factoring_only", failures)
    exact(nested(contract, ["area_gate", "physical_result_available"]),
          False, "physical_result_available", failures)
    exact(nested(contract, ["area_gate", "r8_logic_only_cell_area_um2"]),
          63671.579642, "r8_logic_only_cell_area_um2", failures)
    exact(nested(contract, ["area_gate", "candidate_to_r8_maximum_ratio"]),
          1.1, "candidate_to_r8_maximum_ratio", failures)
    exact(nested(contract, ["area_gate", "candidate_maximum_um2"]),
          70038.737606, "candidate_maximum_um2", failures)
    exact(nested(contract, ["resource_equation", "candidate_state_bits"]),
          5979, "candidate_state_bits", failures)
    exact(nested(contract, ["resource_equation", "r8_state_bits"]),
          5931, "r8_state_bits", failures)
    exact(nested(contract, ["resource_equation", "state_delta_bits"]),
          48, "state_delta_bits", failures)
    exact(nested(contract, ["resource_equation",
                            "explicit_source_selector_mux_bit_equivalents_upper_bound"]),
          1104, "source_selector_upper_bound", failures)
    exact(nested(contract, ["resource_equation",
                            "physical_area_predicted_by_equation"]),
          False, "physical_area_predicted_by_equation", failures)
    exact(nested(contract, ["semantic_invariants", "phase_bundle_offsets"]),
          [0, 168, 336, 504, 672], "phase_bundle_offsets", failures)
    exact(nested(contract, ["semantic_invariants",
                            "descriptor_bundle_order_lsb_to_msb"]),
          ["valid_24", "negative_24", "shift_72", "bias_48"],
          "descriptor_bundle_order", failures)
    exact(nested(contract, ["semantic_invariants",
                            "issue_product_driver_count_in_source"]),
          5, "issue_product_driver_count", failures)
    exact(nested(contract, ["semantic_invariants",
                            "valid_and_negative_sources_are_distinct"]),
          True, "valid_and_negative_sources_are_distinct", failures)
    return failures


def compact(text):
    return re.sub(r"\s+", " ", text).strip()


def semantic_failures(rtl):
    failures = []
    source = compact(rtl)

    bias_expression = (
        "$signed(product_bias_pair_q[ "
        "(result_row_group*ACC_W) +: ACC_W])"
    )
    if source.count(bias_expression) != 1:
        failures.append("SEM_BIAS_ROW_MAPPING")
    if "1-result_row_group" in source or "1 - result_row_group" in source:
        failures.append("SEM_BIAS_ROW_SWAP")

    beat_assignments = re.findall(
        r"product_beat_q\s*<=\s*([^;]+);", rtl
    )
    if [compact(item) for item in beat_assignments] != ["'0", "phase_cycle_q"]:
        failures.append("SEM_PRODUCT_BEAT_CAPTURE")

    retire = "bank_valid_q[active_bank_q] <= 1'b0;"
    replace = "bank_valid_q[input_bank] <= 1'b1;"
    if source.count(retire) != 1 or source.count(replace) != 1:
        failures.append("SEM_PHASE4_REPLACEMENT_ASSIGNMENTS")
    elif source.index(retire) >= source.index(replace):
        failures.append("SEM_PHASE4_REPLACEMENT_ORDER")
    if "bank_valid_q[input_bank] <= 1'b0;" in source:
        failures.append("SEM_PHASE4_REPLACEMENT_CLEARED")

    arms = re.findall(
        r"3'd([0-4])\s*:\s*phase_bundle_comb\s*=\s*"
        r"phase_table_q\[(\d+)\s*\+:\s*PHASE_BUNDLE_W\]", rtl
    )
    expected_arms = [(str(index), str(index * 168)) for index in range(5)]
    if arms != expected_arms:
        failures.append("SEM_PHASE_TABLE_SELECTOR_ORDER")

    load_expression = (
        "phase_table_q[(config_phase*PHASE_BUNDLE_W) "
        "+: PHASE_BUNDLE_W] <= {config_bias["
        "(config_phase*PHASE_BIAS_W) +: PHASE_BIAS_W], "
        "config_term_shift[(config_phase*PHASE_SHIFT_W) "
        "+: PHASE_SHIFT_W], config_term_negative[ "
        "(config_phase*PHASE_VALID_W) +: PHASE_VALID_W], "
        "config_term_valid[(config_phase*PHASE_VALID_W) "
        "+: PHASE_VALID_W]};"
    )
    if source.count(load_expression) != 1:
        failures.append("SEM_PHASE_TABLE_LOAD_PACKING")
    if "PHASES-1-config_phase" in source \
            or "PHASES - 1 - config_phase" in source:
        failures.append("SEM_PHASE_TABLE_REVERSED")

    valid_loads = len(re.findall(
        r"config_term_valid\s*\[\s*\(config_phase\*PHASE_VALID_W\)", rtl
    ))
    negative_loads = len(re.findall(
        r"config_term_negative\s*\[\s*\(config_phase\*PHASE_VALID_W\)", rtl
    ))
    if valid_loads != 1 or negative_loads != 1:
        failures.append("SEM_VALID_NEGATIVE_DISTINCT_LOADS")

    product_lhs = re.findall(
        r"(?m)^\s*issue_product_comb\s*\[([^\]]+)\]\s*=", rtl
    )
    if product_lhs != ["PRODUCT"] * 5:
        failures.append("SEM_ISSUE_PRODUCT_DRIVER_SET")
    if re.search(r"assign\s+issue_product_comb\s*\[", rtl) \
            or re.search(r"issue_product_comb\s*\[[^\]]+\]\s*<=", rtl):
        failures.append("SEM_ISSUE_PRODUCT_EXTRA_DRIVER_FORM")

    forbidden_r9 = [
        r"selected_coefficient\s*==",
        r"coefficient_index\s*<\s*COEFFICIENTS",
        r"term_valid_q\s*\[",
        r"term_negative_q\s*\[",
        r"term_shift_q\s*\[",
        r"bias_q\s*\[",
    ]
    if any(re.search(pattern, rtl) for pattern in forbidden_r9):
        failures.append("SEM_DYNAMIC_OR_EQUALITY_EXPANDED_STORAGE")
    if source.count("assign uses_integer_multiplier = 1'b0;") != 1:
        failures.append("SEM_ZERO_MULTIPLIER_FLAG")

    return sorted(set(failures))


def validate_pin(pin, root):
    failures = []
    if type(pin) is not dict:
        return ["pin root is not object"]
    exact(pin.get("schema"), "m37_r11_evidence_pin_v1",
          "pin.schema", failures)
    exact(pin.get("headline_admitted"), False,
          "pin.headline_admitted", failures)
    exact(pin.get("physical_shared_mux_proven"), False,
          "pin.physical_shared_mux_proven", failures)
    artifacts = pin.get("artifacts")
    if type(artifacts) is not dict:
        return failures + ["pin.artifacts is not object"]
    expected = {
        "candidate_rtl": RTL_REL,
        "contract": CONTRACT_REL,
        "auditor": SCRIPT_REL,
        "tests": TEST_REL,
        "result": RESULT_REL,
        "readme": README_REL,
        "r10_independent_nogo_review": R10_REVIEW_REL,
    }
    if sorted(artifacts.keys()) != sorted(expected.keys()):
        failures.append("pin artifact key set mismatch")
    for name, expected_relative in expected.items():
        artifact = artifacts.get(name)
        if type(artifact) is not dict:
            failures.append("pin artifact is not object: {0}".format(name))
            continue
        relative = artifact.get("path")
        path = canonical_contained_path(
            root, relative, expected_relative, failures,
            "pin.artifacts.{0}".format(name)
        )
        digest = artifact.get("sha256")
        if type(digest) is not str or not re.match(r"^[0-9a-f]{64}$", digest):
            failures.append("pin SHA is not lowercase sha256: {0}".format(name))
        elif path is not None:
            if not os.path.isfile(path):
                failures.append("pin artifact missing: {0}".format(name))
            elif sha256_file(path) != digest:
                failures.append("pin artifact SHA mismatch: {0}".format(name))
    return failures


def audit():
    failures = []
    root = repository_root()
    contract_path = canonical_contained_path(
        root, CONTRACT_REL, CONTRACT_REL, failures, "canonical contract"
    )
    pin_path = canonical_contained_path(
        root, PIN_REL, PIN_REL, failures, "canonical pin"
    )
    rtl_path = canonical_contained_path(
        root, RTL_REL, RTL_REL, failures, "canonical candidate"
    )
    if failures:
        return {"failures": failures,
                "status": "FAIL_M37_R11_CANONICAL_PATH_IDENTITY"}

    contract = load_json_strict(contract_path)
    pin = load_json_strict(pin_path)
    failures.extend(validate_contract(contract, root))
    failures.extend(validate_pin(pin, root))

    actual_rtl_sha = sha256_file(rtl_path)
    if actual_rtl_sha != EXPECTED_RTL_SHA:
        failures.append("candidate SHA mismatch against auditor constant")
    if actual_rtl_sha != nested(contract, ["candidate", "sha256"]):
        failures.append("candidate SHA mismatch against contract")
    with open(rtl_path, "r") as handle:
        rtl = handle.read()
    failures.extend(semantic_failures(rtl))

    result = {
        "candidate_rtl_sha256": actual_rtl_sha,
        "contract_sha256": sha256_file(contract_path),
        "failures": sorted(set(failures)),
        "headline_admitted": False,
        "physical_shared_mux_proven": False,
        "pin_sha256": sha256_file(pin_path),
        "scope": "CANONICAL_PINNED_STATIC_SOURCE_EVIDENCE_ONLY",
        "source_equations": {
            "candidate_state_bits": 5979,
            "explicit_source_selector_mux_bit_upper_bound": 1104,
            "r8_state_bits": 5931,
            "state_delta_bits": 48,
        },
        "status": (
            "PASS_M37_R11_CANONICAL_PINNED_STATIC_SOURCE_EVIDENCE_ONLY"
            if not failures else "FAIL_M37_R11_STATIC_EVIDENCE"
        ),
    }
    return result


def main():
    if len(sys.argv) != 1:
        sys.stderr.write(
            "M37-r11 auditor accepts no path, root, SHA, or enforcement overrides\n"
        )
        return 2
    try:
        result = audit()
    except (IOError, OSError, TypeError, ValueError) as error:
        result = {
            "failures": ["audit exception: {0}".format(error)],
            "headline_admitted": False,
            "physical_shared_mux_proven": False,
            "status": "FAIL_M37_R11_STATIC_EVIDENCE_EXCEPTION",
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result.get("failures") else 1


if __name__ == "__main__":
    sys.exit(main())
