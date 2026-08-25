#!/usr/bin/env python3
"""Independent, Synopsys-free hammer review for the M37-r10 source candidate.

The script intentionally does not admit simulation, synthesis, timing, formal,
power, or system claims.  It locks the exact reviewed bytes, independently
recomputes the packed mapping/state equations, and demonstrates critical
mutations that the milestone's own static auditor currently accepts.
"""

from __future__ import print_function

import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
PATHS = {
    "rtl": "hw_autoresearch_nts07/rtl_m37_r10/qfit_atlif_csd_reconstruct_t10.sv",
    "contract": "hw_autoresearch_nts07/contracts/m37_r10_area_recovery_arch_contract_r1_20260822.json",
    "auditor": "hw_autoresearch_nts07/dc_handoff/scripts/audit_m37_r10_area_recovery_arch.py",
    "tests": "hw_autoresearch_nts07/system_simulator/tests/test_m37_r10_area_recovery_arch.py",
    "result": "hw_autoresearch_nts07/results/m37_r10_area_recovery_arch_r1_20260822/m37_r10_area_recovery_architecture.json",
    "readme": "hw_autoresearch_nts07/results/m37_r10_area_recovery_arch_r1_20260822/README.md",
    "r8_rtl": "hw_autoresearch_nts07/evidence_snapshots/m37_r8_ab7d73a6_20260822/qfit_atlif_csd_reconstruct_t10.sv",
    "r8_receipt": "hw_autoresearch_nts07/contracts/m37_output_receipt_r3_20260822.json",
    "r9_rtl": "hw_autoresearch_nts07/rtl_m37/qfit_atlif_csd_reconstruct_t10.sv",
    "r9_receipt": "hw_autoresearch_nts07/contracts/m37_output_receipt_r4_20260822.json",
}
EXPECTED_SHA256 = {
    "rtl": "f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd",
    "contract": "5825c1c752db2b09d15c06b862e95eeed12312a30a0375f374571762278f2747",
    "auditor": "a4a8aa3d4308f26ceeaa2cf60b3cd27076caf06e8d9584cc6276cfdaf0081021",
    "tests": "1b733bf690d9d7ec17a3318561cd356f942b73c7240ed09411da7b7d162bb65a",
    "result": "2f5260e1d1a6e5e0ebe62892827c25e73a250191eb783fd9774d988e4707745b",
    "readme": "578f6a3f6ed4122c1fcd58542991c63716152e57bcc09c2e5d4a52483bb8bf73",
    "r8_rtl": "ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd",
    "r8_receipt": "363fb61d2838b6379a065dd8eb23b6219441cfb8ed70164766f07d8469e95d97",
    "r9_rtl": "a5f42567fc5262a99152ef04699c9062cbedc70075c0a91397ce8d00dc4397ed",
    "r9_receipt": "7ba9b180705cbc61bc8188e09935ca9cdd86edddd13b5adef0053332941993c1",
}


def full_path(relative):
    path = os.path.realpath(os.path.join(ROOT, relative))
    if os.path.commonpath([ROOT, path]) != ROOT:
        raise ValueError("path escapes repository: {0}".format(relative))
    return path


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
        raise ValueError("non-finite JSON number: {0}".format(value))

    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate,
                         parse_constant=reject_constant)


def compact(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_descriptor_checker(text, stop_marker):
    start = text.index("always_comb begin : validate_all_descriptors")
    stop = text.index(stop_marker, start)
    return compact(text[start:stop])


def load_milestone_auditor():
    path = full_path(PATHS["auditor"])
    spec = importlib.util.spec_from_file_location("m37_r10_milestone_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def auditor_accepts_mutation(auditor, rtl, old, new):
    if old not in rtl:
        raise AssertionError("review mutation source not found: {0}".format(old))
    mutated = rtl.replace(old, new, 1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sv", delete=False) as handle:
        handle.write(mutated)
        temporary = handle.name
    try:
        result = auditor.audit(
            ROOT, full_path(PATHS["contract"]), temporary, True
        )
        return result.get("failures") == []
    finally:
        os.unlink(temporary)


def main():
    failures = []
    observed_sha256 = {}
    for name, relative in PATHS.items():
        path = full_path(relative)
        if not os.path.isfile(path):
            failures.append("missing reviewed artifact: {0}".format(name))
            continue
        observed_sha256[name] = sha256_file(path)
        if observed_sha256[name] != EXPECTED_SHA256[name]:
            failures.append("reviewed SHA drift: {0}".format(name))

    if failures:
        print(json.dumps({"failures": failures,
                          "status": "FAIL_REVIEW_INPUT_IDENTITY"},
                         indent=2, sort_keys=True))
        return 1

    contract = load_json_strict(full_path(PATHS["contract"]))
    result = load_json_strict(full_path(PATHS["result"]))
    with open(full_path(PATHS["rtl"]), "r") as handle:
        rtl = handle.read()
    with open(full_path(PATHS["r8_rtl"]), "r") as handle:
        r8_rtl = handle.read()

    if type(contract.get("headline_admitted")) is not bool \
            or contract.get("headline_admitted") is not False:
        failures.append("headline_admitted is not exact false")
    if contract.get("r10_candidate", {}).get("path") != PATHS["rtl"]:
        failures.append("candidate path is not the reviewed canonical path")
    if contract.get("r10_candidate", {}).get("sha256") != EXPECTED_SHA256["rtl"]:
        failures.append("contract candidate SHA is not the reviewed SHA")
    for key in ("candidate_rtl", "contract", "auditor", "negative_tests"):
        artifact = result.get("artifacts", {}).get(key, {})
        translated = {"candidate_rtl": "rtl", "negative_tests": "tests"}.get(key, key)
        if artifact.get("path") != PATHS[translated]:
            failures.append("result artifact path drift: {0}".format(key))
        if artifact.get("sha256") != EXPECTED_SHA256[translated]:
            failures.append("result artifact SHA drift: {0}".format(key))

    # Independently derive every flattened configuration mapping.  This proves
    # bijective coverage for the intended default T=10/RANK=3/TERMS=4 shape.
    valid_bits = []
    shift_bits = []
    bias_bits = []
    products = []
    inputs = []
    for phase in range(5):
        for row_group in range(2):
            row = phase * 2 + row_group
            bias_bits.extend(range(row * 24, row * 24 + 24))
            for rank in range(3):
                coefficient = row * 3 + rank
                desc = row_group * 3 + rank
                for term in range(4):
                    valid_bits.append((phase * 24 + desc * 4 + term,
                                       coefficient * 4 + term))
                    for bit in range(3):
                        shift_bits.append((phase * 72 + desc * 12 + term * 3 + bit,
                                           coefficient * 12 + term * 3 + bit))
                for lane in range(16):
                    product = (row_group * 16 + lane) * 3 + rank
                    intermediate = rank * 16 + lane
                    products.append(product)
                    inputs.append((product, intermediate))
    if any(source != destination for source, destination in valid_bits):
        failures.append("valid/negative packed mapping is not identity")
    if any(source != destination for source, destination in shift_bits):
        failures.append("shift packed mapping is not identity")
    if sorted(bias_bits) != list(range(240)):
        failures.append("bias packed mapping is not a 240-bit bijection")
    if sorted(products) != sorted(list(range(96)) * 5):
        failures.append("generated product indices do not cover 0..95 per phase")
    if any(intermediate != (product % 3) * 16 + ((product // 3) % 16)
           for product, intermediate in inputs):
        failures.append("rank/lane intermediate mapping mismatch")

    # Lock the exact source expressions that connect the independently derived
    # mapping to the reviewed RTL bytes.
    source = compact(rtl)
    required_source = [
        "phase_table_q[(config_phase*PHASE_BUNDLE_W) +: PHASE_BUNDLE_W] <= {config_bias[(config_phase*PHASE_BIAS_W) +: PHASE_BIAS_W], config_term_shift[(config_phase*PHASE_SHIFT_W) +: PHASE_SHIFT_W], config_term_negative[ (config_phase*PHASE_VALID_W) +: PHASE_VALID_W], config_term_valid[(config_phase*PHASE_VALID_W) +: PHASE_VALID_W]};",
        "localparam int DESC = (output_row_group*RANK)+rank;",
        "localparam int PRODUCT = ((output_row_group*LANES+lane)*RANK) +rank;",
        "localparam int INPUT = (rank*LANES)+lane;",
        "product_tag_q <= active_tag_comb; product_beat_q <= phase_cycle_q; product_bias_pair_q <= phase_bias_pair_comb;",
        "$signed(product_bias_pair_q[ (result_row_group*ACC_W) +: ACC_W])",
        "bank_valid_q[active_bank_q] <= 1'b0;",
        "bank_valid_q[input_bank] <= 1'b1;",
    ]
    for snippet in required_source:
        if snippet not in source:
            failures.append("reviewed semantic source expression missing: {0}".format(snippet))

    arms = re.findall(
        r"3'd([0-4])\s*:\s*phase_bundle_comb\s*=\s*phase_table_q\[(\d+)\s*\+:\s*PHASE_BUNDLE_W\]",
        rtl,
    )
    if arms != [(str(phase), str(phase * 168)) for phase in range(5)]:
        failures.append("phase selector arm/order/offset mismatch")
    if source.index("bank_valid_q[active_bank_q] <= 1'b0;") >= \
            source.index("bank_valid_q[input_bank] <= 1'b1;"):
        failures.append("same-cycle replacement assignment does not follow retirement")

    # The descriptor checker is intentionally still present and is byte-for-byte
    # source-equivalent to r8 after whitespace normalization.  It is therefore
    # not a new r10 delta, but it is combinational logic omitted from the state
    # and selector equations and must remain inside the DC area boundary.
    if extract_descriptor_checker(rtl, "// One shared 168-bit") != \
            extract_descriptor_checker(r8_rtl, "assign result_valid"):
        failures.append("descriptor checker drifted from the frozen r8 source")

    expected_state_terms = {
        "layer_static_phase_table_and_threshold": 840 + 24,
        "config_and_protocol_flags": 2,
        "input_bank_payload_and_tags": 2 * (384 + 48),
        "active_bank_and_compute_control": 2 + 1 + 1 + 3,
        "product_stage_data_bias_tag_beat_valid": 96 * 18 + 48 + 48 + 3 + 1,
        "fifo_state": 16 * (48 + 3 + 48 + 48) + 4 + 4 + 5,
        "done_state": 1 + 48,
    }
    contract_terms = contract.get("resource_equation", {}).get("terms", {})
    for name, value in expected_state_terms.items():
        if type(contract_terms.get(name)) is not int or contract_terms.get(name) != value:
            failures.append("state term mismatch: {0}".format(name))
    candidate_state = sum(expected_state_terms.values())
    if candidate_state != 5979:
        failures.append("independent state equation did not equal 5979")
    phase_mux = 168 * (5 - 1)
    bank_mux = 384 + 48
    if phase_mux + bank_mux != 1104:
        failures.append("independent explicit selector equation did not equal 1104")

    # Reproduce the auditor's evidence-path and semantic-blindness failures.
    auditor = load_milestone_auditor()
    mutations = {
        "alternate_path_skips_candidate_sha_and_swaps_bias_rows": (
            "(result_row_group*ACC_W) +: ACC_W",
            "((1-result_row_group)*ACC_W) +: ACC_W",
        ),
        "product_beat_pipeline_misaligned": (
            "product_beat_q <= phase_cycle_q;",
            "product_beat_q <= phase_cycle_q + 1'b1;",
        ),
        "same_cycle_replacement_cleared": (
            "bank_valid_q[input_bank] <= 1'b1;",
            "bank_valid_q[input_bank] <= 1'b0;",
        ),
        "phase_table_load_reversed": (
            "phase_table_q[(config_phase*PHASE_BUNDLE_W)",
            "phase_table_q[((PHASES-1-config_phase)*PHASE_BUNDLE_W)",
        ),
        "descriptor_valid_loaded_from_negative": (
            "config_term_valid[(config_phase*PHASE_VALID_W)",
            "config_term_negative[(config_phase*PHASE_VALID_W)",
        ),
        "generated_product_extra_driver": (
            "endmodule\n\n`default_nettype wire",
            "always_comb begin issue_product_comb[0] = 18'sd0; end\nendmodule\n\n`default_nettype wire",
        ),
    }
    accepted_attacks = []
    for name, mutation in mutations.items():
        if auditor_accepts_mutation(auditor, rtl, mutation[0], mutation[1]):
            accepted_attacks.append(name)
        else:
            failures.append("expected auditor bypass no longer reproduces: {0}".format(name))

    findings = [
        {
            "id": "P1_EVIDENCE_PATH_AND_SEMANTIC_MUTATION_BYPASS",
            "severity": "P1",
            "finding": "The milestone auditor enforces the candidate SHA only when the supplied RTL realpath already equals the contract path. An alternate --rtl path skips SHA enforcement, and six critical semantic mutations are accepted.",
            "evidence": accepted_attacks,
            "repair_gate": "Require canonical contained contract/RTL paths unconditionally, pin the contract SHA in a separate admission, and reject packing, bias-row, beat-pipeline, replacement-order, and multi-driver mutations.",
        },
        {
            "id": "P2_COMPILER_AND_GENERATED_DRIVER_LEGALITY_UNPROVEN",
            "severity": "P2",
            "finding": "The 96 generated always_comb blocks appear to drive disjoint constant array elements, but static text cannot establish VCS/elaboration/DC legality or absence of tool-specific multiple-driver treatment.",
            "repair_gate": "Fresh exact-SHA VCS compile and frozen functional/SVA workload before any functional admission.",
        },
        {
            "id": "P2_AREA_HYPOTHESIS_EXCLUDES_COMBINATIONAL_AND_FANOUT_COST",
            "severity": "P2",
            "finding": "The 5,979-bit state equation is exact for default parameters and the 1,104 mux-bit count is exact only for two explicit source selectors. The unchanged 30-descriptor legality checker, CSD networks, high-fanout replication/buffering, FIFO muxing, and timing repair are not predicted.",
            "repair_gate": "Fresh exact-SHA 3 ns DC/STA under the frozen r8 boundary must meet the 70,038.737606 um2 cap; report hierarchy and high-fanout/replication evidence.",
        },
        {
            "id": "P2_SHARED_BUS_MUX_IS_SOURCE_TOPOLOGY_ONLY",
            "severity": "P2",
            "finding": "One 168-bit case statement is a valid source-level factoring, not proof that mapping preserves a unique physical mux rather than replicating it across 96 consumers.",
            "repair_gate": "Treat uniqueness as a source claim only until mapped netlist structure and area reports are independently audited.",
        },
    ]
    review = {
        "accepted_adversarial_mutations": accepted_attacks,
        "architectural_assessment": "CONDITIONALLY_PLAUSIBLE_EXACT_SOURCE_CANDIDATE",
        "candidate_rtl_sha256": observed_sha256["rtl"],
        "claim_boundary": {
            "permitted": [
                "exact reviewed RTL source structure",
                "default-parameter 5979-bit architectural-state equation",
                "source-level explicit-selector upper bound of 1104 two-to-one mux-bit equivalents",
                "packing and pipeline mapping by independent static derivation",
            ],
            "forbidden": [
                "compiled or simulated functional equivalence",
                "synthesis legality timing mapped area or multiplier resources",
                "Formality equivalence",
                "power energy full-network performance speedup or paper headline",
                "a unique physical shared mux",
            ],
        },
        "failures": failures,
        "findings": findings,
        "headline_admitted": False,
        "independent_equations": {
            "candidate_state_bits": candidate_state,
            "r8_state_bits": 5931,
            "state_delta_bits": candidate_state - 5931,
            "explicit_selector_mux_bit_equivalents": phase_mux + bank_mux,
        },
        "review_score_0_to_100": 82,
        "review_verdict": "NO_GO_CURRENT_STATIC_EVIDENCE_ADMISSION",
        "reviewed_sha256": observed_sha256,
        "schema": "m37_r10_independent_hammer_review_v1",
        "status": "PASS_INDEPENDENT_HAMMER_REPRODUCED_NO_GO_FINDINGS" if not failures else "FAIL_INDEPENDENT_REVIEW_VALIDATION",
    }
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
