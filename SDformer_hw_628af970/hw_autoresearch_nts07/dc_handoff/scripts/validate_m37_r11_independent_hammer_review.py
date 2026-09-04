#!/usr/bin/env python3
"""Independent Synopsys-free hammer for the M37-r11 static admission.

This validator treats every r11 input as untrusted.  It pins exact bytes,
re-runs the canonical unit suite, reproduces the prior six attacks, adds path
and JSON attacks, and distinguishes lexical-parser weakness from an actual
canonical-SHA substitution.  A pass admits only the exact-SHA VCS gate.
"""

from __future__ import print_function

import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
PATHS = {
    "pin": "hw_autoresearch_nts07/contracts/m37_r11_evidence_pin_r1_20260822.json",
    "contract": "hw_autoresearch_nts07/contracts/m37_r11_area_recovery_evidence_contract_r1_20260822.json",
    "auditor": "hw_autoresearch_nts07/dc_handoff/scripts/audit_m37_r11_area_recovery_evidence.py",
    "tests": "hw_autoresearch_nts07/system_simulator/tests/test_m37_r11_area_recovery_evidence.py",
    "result": "hw_autoresearch_nts07/results/m37_r11_area_recovery_evidence_r1_20260822/m37_r11_area_recovery_evidence.json",
    "readme": "hw_autoresearch_nts07/results/m37_r11_area_recovery_evidence_r1_20260822/README.md",
    "r10_review": "hw_autoresearch_nts07/results/m37_r10_independent_hammer_review_20260822/m37_r10_independent_hammer_review.json",
    "rtl": "hw_autoresearch_nts07/rtl_m37_r10/qfit_atlif_csd_reconstruct_t10.sv",
}
EXPECTED_SHA256 = {
    "pin": "9410b3418a001b84cfe035b9ebe9fef6190284db916d0ef6c7b1d806d46a09b4",
    "contract": "ac842a11559aa3e35c50d665026b942074176093a032eb3a48bf817703b69735",
    "auditor": "7f1f77f59de4e0f52b7a83bb55b6a461b6539bafe7aa1edcc6b364d5db884a10",
    "tests": "ae0c2b6454419041425d93204bc85de29abb5b3d45f4f26e42e38a7e81baddc4",
    "result": "ffd2b8cc19113c23cf552aea1c9fda7d32facd37dd9965a46e66aad638966dc8",
    "readme": "3bcf652fc13a83998b1119ad5a3ef2d168f84947abdc5d5efa83f985d5a8c890",
    "r10_review": "3779f8d52c45ebff8ad0bf991db00a72c42fb7514518390be844fcedf6cbdd26",
    "rtl": "f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd",
}


def full_path(relative):
    if type(relative) is not str or os.path.isabs(relative) \
            or os.path.normpath(relative) != relative:
        raise ValueError("non-canonical review path: {0!r}".format(relative))
    path = os.path.realpath(os.path.join(ROOT, relative))
    if os.path.commonpath([ROOT, path]) != ROOT:
        raise ValueError("review path escapes repository: {0}".format(relative))
    return path


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

    def reject_constant(value):
        raise ValueError("non-finite JSON constant: {0}".format(value))

    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate,
                         parse_constant=reject_constant)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mutate_once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError("mutation anchor count is not one: {0!r}".format(old))
    return text.replace(old, new, 1)


def strict_loader_rejects(auditor, content):
    descriptor, path = tempfile.mkstemp(prefix="m37_r11_json_", suffix=".json")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
        try:
            auditor.load_json_strict(path)
        except ValueError:
            return True
        return False
    finally:
        os.unlink(path)


def main():
    failures = []
    observed_sha256 = {}
    modes = {}
    for name, relative in PATHS.items():
        path = full_path(relative)
        if not os.path.isfile(path):
            failures.append("missing reviewed artifact: {0}".format(name))
            continue
        observed_sha256[name] = sha256_file(path)
        modes[name] = oct(os.stat(path).st_mode & 0o777)
        if observed_sha256[name] != EXPECTED_SHA256[name]:
            failures.append("reviewed SHA drift: {0}".format(name))
    if failures:
        print(json.dumps({"failures": failures,
                          "status": "FAIL_R11_REVIEW_INPUT_IDENTITY"},
                         indent=2, sort_keys=True))
        return 1

    auditor = load_module("m37_r11_milestone_auditor", full_path(PATHS["auditor"]))
    test_module = load_module("m37_r11_milestone_tests", full_path(PATHS["tests"]))
    contract = load_json_strict(full_path(PATHS["contract"]))
    pin = load_json_strict(full_path(PATHS["pin"]))
    result = load_json_strict(full_path(PATHS["result"]))
    r10_review = load_json_strict(full_path(PATHS["r10_review"]))
    with open(full_path(PATHS["rtl"]), "r") as handle:
        rtl = handle.read()

    # Independently run all canonical tests, rather than trusting the recorded
    # 15/15 count in the result JSON.
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
    stream = io.StringIO()
    canonical_run = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    canonical_count = canonical_run.testsRun
    canonical_failures = len(canonical_run.failures) + len(canonical_run.errors)
    if canonical_count != 15 or canonical_failures != 0:
        failures.append("canonical unit suite is not an independently clean 15/15")

    canonical_audit = auditor.audit()
    if canonical_audit.get("failures") != [] or not str(
            canonical_audit.get("status", "")).startswith("PASS_M37_R11"):
        failures.append("canonical milestone audit does not pass")

    # The pin is not trusted: independently require its exact key set, paths,
    # hashes, and the exact reviewed NO-GO ancestry.
    pin_expected = {
        "candidate_rtl": "rtl",
        "contract": "contract",
        "auditor": "auditor",
        "tests": "tests",
        "result": "result",
        "readme": "readme",
        "r10_independent_nogo_review": "r10_review",
    }
    artifacts = pin.get("artifacts", {})
    if set(artifacts) != set(pin_expected):
        failures.append("pin artifact key set drift")
    for pin_name, review_name in pin_expected.items():
        artifact = artifacts.get(pin_name, {})
        if artifact.get("path") != PATHS[review_name]:
            failures.append("pin path mismatch: {0}".format(pin_name))
        if artifact.get("sha256") != EXPECTED_SHA256[review_name]:
            failures.append("pin SHA mismatch: {0}".format(pin_name))
    if r10_review.get("review_verdict") != "NO_GO_CURRENT_STATIC_EVIDENCE_ADMISSION" \
            or r10_review.get("p1_count") != 1:
        failures.append("r10 independent NO-GO ancestry drift")

    if contract.get("candidate", {}).get("path") != PATHS["rtl"] \
            or contract.get("candidate", {}).get("sha256") != EXPECTED_SHA256["rtl"]:
        failures.append("contract does not bind exact candidate")
    if contract.get("claim_boundary", {}).get("headline_admitted") is not False \
            or contract.get("claim_boundary", {}).get("physical_shared_mux_proven") is not False:
        failures.append("contract claim boundary drift")
    if result.get("headline_admitted") is not False \
            or result.get("claim_boundary", {}).get("physical_shared_mux_proven") is not False:
        failures.append("result claim boundary drift")

    # Independently derive the default-shape storage and explicit source-mux
    # equations.  These remain source equations, not mapped-area predictors.
    state_terms = [840 + 24, 2, 2 * (384 + 48), 2 + 1 + 1 + 3,
                   96 * 18 + 48 + 48 + 3 + 1,
                   16 * (48 + 3 + 48 + 48) + 4 + 4 + 5, 1 + 48]
    candidate_state_bits = sum(state_terms)
    r8_state_bits = 5931
    selector_mux_bits = 168 * 4 + 384 + 48
    if candidate_state_bits != 5979 or candidate_state_bits - r8_state_bits != 48:
        failures.append("independent state equation mismatch")
    if selector_mux_bits != 1104:
        failures.append("independent explicit-selector equation mismatch")

    # Reproduce the six exact r10 attacks.  R11 must reject each at its static
    # semantic layer, independent of the stronger canonical SHA gate.
    prior_mutations = {
        "bias_row_swap": (
            "(result_row_group*ACC_W) +: ACC_W",
            "((1-result_row_group)*ACC_W) +: ACC_W",
        ),
        "product_beat_off_by_one": (
            "product_beat_q <= phase_cycle_q;",
            "product_beat_q <= phase_cycle_q + 1'b1;",
        ),
        "same_cycle_replacement_clear": (
            "bank_valid_q[input_bank] <= 1'b1;",
            "bank_valid_q[input_bank] <= 1'b0;",
        ),
        "phase_table_load_reverse": (
            "phase_table_q[(config_phase*PHASE_BUNDLE_W)",
            "phase_table_q[((PHASES-1-config_phase)*PHASE_BUNDLE_W)",
        ),
        "valid_loaded_from_negative": (
            "config_term_valid[(config_phase*PHASE_VALID_W)",
            "config_term_negative[(config_phase*PHASE_VALID_W)",
        ),
        "additional_issue_product_driver": (
            "endmodule\n\n`default_nettype wire",
            "always_comb begin issue_product_comb[0] = 18'sd0; end\n"
            "endmodule\n\n`default_nettype wire",
        ),
    }
    prior_admission_rejected = []
    prior_semantic_rejected = []
    prior_semantic_parser_only = []
    for name, pair in prior_mutations.items():
        mutated = mutate_once(rtl, pair[0], pair[1])
        if hashlib.sha256(mutated.encode("utf-8")).hexdigest() \
                != EXPECTED_SHA256["rtl"]:
            prior_admission_rejected.append(name)
        else:
            failures.append("prior mutation retained canonical SHA: {0}".format(name))
        if auditor.semantic_failures(mutated):
            prior_semantic_rejected.append(name)
        else:
            prior_semantic_parser_only.append(name)

    rejected_admission_attacks = []
    cli = subprocess.Popen(
        ["/usr/bin/python3.6", full_path(PATHS["auditor"]),
         "--rtl", "/tmp/alternate.sv"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    cli_stdout, cli_stderr = cli.communicate()
    if cli.returncode == 2 and cli_stdout == b"" and b"accepts no path" in cli_stderr:
        rejected_admission_attacks.append("alternate_cli_rtl_path")
    else:
        failures.append("alternate CLI RTL path was not fail-closed")

    relative_failures = []
    if auditor.canonical_contained_path(
            ROOT, "../escape", "../escape", relative_failures, "attack") is None \
            and relative_failures:
        rejected_admission_attacks.append("relative_path_escape")
    else:
        failures.append("relative path escape was not rejected")

    with tempfile.TemporaryDirectory(prefix="m37_r11_symlink_root_") as symlink_root:
        with tempfile.NamedTemporaryFile(prefix="m37_r11_outside_", delete=False) as outside:
            outside_path = outside.name
        link_path = os.path.join(symlink_root, "candidate.sv")
        try:
            os.symlink(outside_path, link_path)
            symlink_failures = []
            selected = auditor.canonical_contained_path(
                symlink_root, "candidate.sv", "candidate.sv",
                symlink_failures, "symlink_attack",
            )
            if selected is None and any("escapes" in item for item in symlink_failures):
                rejected_admission_attacks.append("candidate_symlink_escape")
            else:
                failures.append("candidate symlink escape was not rejected")
        finally:
            os.unlink(outside_path)

    if auditor.validate_contract(pin, ROOT) and auditor.validate_pin(contract, ROOT):
        rejected_admission_attacks.append("contract_pin_object_swap")
    else:
        failures.append("contract/pin object swap did not fail both validators")
    swapped_contract = json.loads(json.dumps(contract))
    swapped_contract["candidate"]["sha256"] = "0" * 64
    if auditor.validate_contract(swapped_contract, ROOT):
        rejected_admission_attacks.append("contract_candidate_sha_swap")
    else:
        failures.append("contract candidate SHA swap was not rejected")
    swapped_pin = json.loads(json.dumps(pin))
    swapped_pin["artifacts"]["candidate_rtl"]["sha256"] = "0" * 64
    if auditor.validate_pin(swapped_pin, ROOT):
        rejected_admission_attacks.append("pin_candidate_sha_swap")
    else:
        failures.append("pin candidate SHA swap was not rejected")

    for label, payload in (
            ("duplicate_json_key", '{"schema":"a","schema":"b"}'),
            ("json_nan", '{"value":NaN}'),
            ("json_positive_infinity", '{"value":Infinity}')):
        if strict_loader_rejects(auditor, payload):
            rejected_admission_attacks.append(label)
        else:
            failures.append("strict JSON loader accepted: {0}".format(label))
    bool_contract = json.loads(json.dumps(contract))
    bool_contract["resource_equation"]["candidate_state_bits"] = True
    bool_contract["semantic_invariants"]["phase_bundle_offsets"][0] = False
    bool_failures = auditor.validate_contract(bool_contract, ROOT)
    if any("candidate_state_bits" in item for item in bool_failures) \
            and any("phase_bundle_offsets" in item for item in bool_failures):
        rejected_admission_attacks.append("bool_as_int_scalar_and_nested")
    else:
        failures.append("bool-as-int contract mutation was not rejected")

    # The semantic helper is still a lexical regex check and can be spoofed by
    # comments/dead code/macros.  Record this as defense-in-depth weakness.  It
    # does not substitute the canonical candidate because any changed bytes fail
    # the unconditional f947... SHA before VCS admission.
    parser_spoofs = {}
    parser_spoofs["comment_spoof_bias_xor"] = mutate_once(
        rtl,
        "$signed(product_bias_pair_q[\n                        "
        "(result_row_group*ACC_W) +: ACC_W]),",
        "/* $signed(product_bias_pair_q[\n                        "
        "(result_row_group*ACC_W) +: ACC_W]) */\n                    "
        "$signed(product_bias_pair_q[\n                        "
        "((result_row_group ^ 1'b1)*ACC_W) +: ACC_W]),",
    )
    parser_spoofs["dead_code_spoof_product_beat"] = mutate_once(
        rtl, "product_beat_q <= phase_cycle_q;",
        "product_beat_q /* active mutation */ <= phase_cycle_q + 1'b1;\n"
        "                    if (1'b0) product_beat_q <= phase_cycle_q;",
    )
    selector_spoof = rtl
    for phase in range(5):
        old = "3'd{0}: phase_bundle_comb = phase_table_q[{1} +: PHASE_BUNDLE_W];".format(
            phase, phase * 168)
        new = "3'd{0}: phase_bundle_comb = phase_table_q[((4-{0})*168) +: PHASE_BUNDLE_W]; // {1}".format(
            phase, old)
        selector_spoof = mutate_once(selector_spoof, old, new)
    parser_spoofs["comment_spoof_reversed_phase_selector"] = selector_spoof
    parser_spoofs["macro_hidden_duplicate_driver"] = mutate_once(
        rtl, "endmodule\n\n`default_nettype wire",
        "`define M37_EVIL_PRODUCT issue_product_comb\n"
        "always_comb begin : evil_driver\n"
        "    `M37_EVIL_PRODUCT[0] = 18'sd0;\n"
        "end\n`undef M37_EVIL_PRODUCT\n"
        "endmodule\n\n`default_nettype wire",
    )
    accepted_parser_spoofs = []
    for name, mutated in parser_spoofs.items():
        if not auditor.semantic_failures(mutated):
            accepted_parser_spoofs.append(name)
        else:
            failures.append("expected lexical parser weakness no longer reproduces: {0}".format(name))
        mutated_sha = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
        if mutated_sha == EXPECTED_SHA256["rtl"]:
            failures.append("mutated parser-spoof source retained canonical SHA: {0}".format(name))
    accepted_parser_spoofs.extend(prior_semantic_parser_only)

    findings = [
        {
            "id": "P2_LEXICAL_SEMANTIC_CHECKS_ARE_SPOOFABLE",
            "severity": "P2",
            "finding": "Comment, dead-code, and macro token tricks can fool semantic_failures in isolation. They cannot replace the admitted candidate because audit() unconditionally checks the canonical contained RTL against f947... first.",
            "evidence": accepted_parser_spoofs,
            "repair_gate": "Use parser/preprocessor-aware structural checks as defense in depth; retain exact external SHA binding for every downstream VCS run.",
        },
        {
            "id": "P2_PIN_IS_NOT_A_SELF_AUTHENTICATING_TRUST_ROOT",
            "severity": "P2",
            "finding": "The separate pin closes child artifacts but cannot authenticate its own bytes; mode 0444 is mutable by the owner. This independent validator supplies the external exact-byte anchor.",
            "repair_gate": "Downstream runners must pin both this validator and its deterministic review output, not rely on file mode or the r11 pin alone.",
        },
        {
            "id": "P2_VCS_FUNCTIONAL_LEGALITY_NOT_YET_PROVEN",
            "severity": "P2",
            "finding": "Static source review cannot prove elaboration legality, generated-driver treatment, cycle behavior, backpressure, or equivalence.",
            "repair_gate": "Run only the exact f947... candidate under the frozen 245-tile VCS/SVA workload and independently seal the transcript and executable inputs.",
        },
        {
            "id": "P2_AREA_AND_PHYSICAL_SHARING_UNPROVEN",
            "severity": "P2",
            "finding": "The 5,979-bit and 1,104 selector-bit equations exclude descriptor checking, CSD logic, replication, fanout, FIFO muxing, and timing repair; they do not establish mapped area or a unique physical mux.",
            "repair_gate": "No DC/STA/PPA admission until a successful exact-SHA VCS milestone is independently hammered; later DC must meet 70,038.737606 um2 and zero-multiplier gates.",
        },
    ]
    review = {
        "accepted_parser_only_spoof_mutations": accepted_parser_spoofs,
        "canonical_tests": {"failed": canonical_failures,
                            "passed": canonical_count},
        "claim_boundary": {
            "permitted": [
                "exact f9474151... candidate identity and reviewed source structure",
                "default-parameter 5979-bit architectural-state equation",
                "source-level 1104 two-to-one selector mux-bit upper-bound equation",
                "launch of an exact-SHA frozen VCS/SVA gate only",
            ],
            "forbidden": [
                "DC STA mapped area frequency or multiplier-resource admission",
                "Formality equivalence",
                "power energy full-system performance or speedup",
                "a unique physical shared mux",
                "paper headline or PPA claim",
            ],
        },
        "direct_prior_mutations_admission_rejected": sorted(prior_admission_rejected),
        "direct_prior_mutations_semantic_layer_rejected": sorted(prior_semantic_rejected),
        "failures": failures,
        "file_modes_observed_additional_protection_only": modes,
        "findings": findings,
        "headline_admitted": False,
        "independent_equations": {
            "candidate_state_bits": candidate_state_bits,
            "explicit_selector_mux_bit_equivalents": selector_mux_bits,
            "r8_state_bits": r8_state_bits,
            "state_delta_bits": candidate_state_bits - r8_state_bits,
        },
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": len(findings),
        "rejected_admission_attacks": sorted(rejected_admission_attacks),
        "review_score_0_to_100": 92,
        "review_verdict": "GO_EXACT_SHA_VCS_ONLY_NO_DC_PPA_ADMISSION",
        "reviewed_sha256": observed_sha256,
        "schema": "m37_r11_independent_hammer_review_v1",
        "status": "PASS_INDEPENDENT_HAMMER_GO_EXACT_SHA_VCS_ONLY" if not failures else "FAIL_INDEPENDENT_HAMMER",
    }
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
