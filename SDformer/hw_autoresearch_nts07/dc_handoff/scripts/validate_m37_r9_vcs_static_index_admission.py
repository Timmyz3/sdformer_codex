#!/usr/bin/env python3
"""Validate the independent M37-r9 VCS/static-index-only admission."""

import argparse
import copy
import difflib
import hashlib
import importlib.util
import json
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
RUN_DIR = pathlib.Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
    "dc_handoff/runs/m37_csd_reconstruct_t10_vcs_r9_20260822"
)

EXPECTED = {
    "receipt": "7ba9b180705cbc61bc8188e09935ca9cdd86edddd13b5adef0053332941993c1",
    "contract": "1d8644e3e964bdbb83bf02fc51f41a4669ca21ad6eeb61d9a62a451026d82b77",
    "r9_rtl": "a5f42567fc5262a99152ef04699c9062cbedc70075c0a91397ce8d00dc4397ed",
    "r8_rtl": "ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd",
    "runner": "67e273e8b901100950aefc92e93a97a6b252f77586f6ae008de8419aabb03bf0",
    "r9_auditor": "f8045d9ff9dddf26202e2d3cde0997fe8cd7f89f5d58dacd170da9e6ef802aa9",
    "r8_auditor": "6fcf221ac018e38283723b687852e1809941aabdbbfa031dd812da14113cc856",
    "r8_admission": "f133b96a458686e17f94ecf52c26db3c9b753ef7145f4b396a9f047acfda0fa2",
    "r8_validator": "7be9c7e5bba4ffb0fb972be948019dce5354362bc1da4e8d3e68057b0c4cce07",
    "r8_provenance": "f7b88ceafe4447ad7dc1abb11751bead49d3170293ffec1ea6f521aac0c99f99",
    "r8_ledger": "01dc86fcda8ba3627e2de27fbab26866ca794b0e3e8da05d6fbd563cf72364a3",
    "math_contract": "790c8a6e7d0fafacf5fcf64b1f4cb106d12fdb93464d68b8592ce5b14125d144",
    "math_result": "9b5b080aeb198d54df92ab6bd21741dfb5e05cbe24a2b81fe5d39843e82d47e6",
    "input_manifest": "cf2edf71c1cb618ec485af730f315aaebd23e36f29f48f6e09129f35c5dab081",
    "output_manifest": "eaaf6000ff46c3f9e01bfd3525b7bb7403c23841896073601bda3a4c2418d9a6",
    "local_seal": "de6e657000be6fd1b143386c12478f9c98197847b19a26c469569445e4dd918a",
    "compile_log": "4a30c753b28663304ef6cf333a1a8a97f387dc6c2ffbca5cd56651f3ecddf093",
    "sim_log": "583448f590a6cf368333d61a5e1e19e43bb31eb060c8d7a3687a00a357b0b8e3",
    "vectors": "2d58455e5b9bbf4b15450649f6259a6216c3ff8dbcb1097e90439c3c067e1627",
    "source_report": "db142db38e35c8230ba15e0c7ea6c48674ba29f2e3799481c4a7eddbe5ae4cad",
    "runner_status": "0f24a483ddce0493b18d2218bd1d935114f429f5e99cf80929e48e9ce2c832e4",
    "r8_validation_report": "33b3e545560cfe0feb2b251f8b548b3422f075236fa96ec8cba4017045002e2d",
    "canonical_diff": "48f1df8d0d30a1a72219fbd25899ce5de311ff676657c4b453a19cb649fb1037",
}

EXPECTED_SCHEMA = "m37_r9_independent_vcs_static_index_admission_v1"
EXPECTED_STATUS = "PASS_EXACT_M37_R9_VCS_STATIC_INDEX_ONLY"


class ValidationFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationFailure(message)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_keys(payload, expected, label):
    require(isinstance(payload, dict) and set(payload) == set(expected),
            "{} key population drift".format(label))


def read_json_no_duplicates(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook)


def resolve(raw):
    path = pathlib.Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "module import failed: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_manifest_text(text, expected_count):
    rows = []
    seen = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None,
                "malformed manifest line {}".format(line_number))
        digest, raw = match.groups()
        require(raw not in seen, "duplicate manifest target")
        seen.add(raw)
        rows.append((digest, raw))
    require(len(rows) == expected_count, "manifest entry-count drift")
    return rows


def verify_manifest(path, expected_sha, expected_count, base, expected_entries):
    path = pathlib.Path(path)
    require(path.is_file() and sha256(path) == expected_sha,
            "manifest identity drift: {}".format(path.name))
    rows = parse_manifest_text(path.read_text(encoding="utf-8"), expected_count)
    actual = {raw: digest for digest, raw in rows}
    require(actual == expected_entries, "manifest population drift")
    for raw, digest in actual.items():
        target = pathlib.Path(raw)
        if not target.is_absolute():
            target = pathlib.Path(base) / target
        require(target.is_file() and sha256(target) == digest,
                "manifest live target drift: {}".format(raw))
    return actual


def blank_comments_and_strings(text):
    output = []
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "code" and char == "/" and nxt == "/":
            output.extend("  ")
            index += 2
            state = "line"
        elif state == "code" and char == "/" and nxt == "*":
            output.extend("  ")
            index += 2
            state = "block"
        elif state == "code" and char == '"':
            output.append(" ")
            index += 1
            state = "string"
        elif state == "line":
            output.append("\n" if char == "\n" else " ")
            index += 1
            if char == "\n":
                state = "code"
        elif state == "block" and char == "*" and nxt == "/":
            output.extend("  ")
            index += 2
            state = "code"
        elif state == "block":
            output.append("\n" if char == "\n" else " ")
            index += 1
        elif state == "string" and char == "\\" and nxt:
            output.extend("  ")
            index += 2
        elif state == "string" and char == '"':
            output.append(" ")
            index += 1
            state = "code"
        elif state == "string":
            output.append("\n" if char == "\n" else " ")
            index += 1
        else:
            output.append(char)
            index += 1
    require(state in ("code", "line"), "unterminated source comment/string")
    return "".join(output)


def validate_r9_static_intent(source):
    cleaned = blank_comments_and_strings(source)
    required = {
        "bounded bias select": r"if\s*\(\s*selected_row\s*==\s*row_index\s*\)",
        "bounded phase select": r"if\s*\(\s*phase_cycle_q\s*==\s*phase_index\s*\)",
        "bounded coefficient select":
        r"if\s*\(\s*selected_coefficient\s*==\s*coefficient_index\s*\)",
        "shift-add rank index":
        r"selected_coefficient\s*=\s*\(\s*selected_row\s*<<\s*1\s*\)"
        r"\s*\+\s*selected_row\s*\+\s*rank_index\s*;",
    }
    for label, pattern in required.items():
        require(len(re.findall(pattern, cleaned)) == 1,
                "{} population drift".format(label))
    forbidden = {
        "dynamic bias index": r"bias_q\s*\[\s*selected_row\s*\]",
        "dynamic term-valid index":
        r"term_valid_q\s*\[\s*selected_coefficient\s*\]",
        "dynamic term-negative index":
        r"term_negative_q\s*\[\s*selected_coefficient\s*\]",
        "dynamic term-shift index":
        r"term_shift_q\s*\[\s*selected_coefficient\s*\]",
        "dynamic intermediate index":
        r"intermediate_bank_q\s*\[[^\]]+\]\s*\[\s*selected_intermediate\s*\]",
        "Formality mismatch filter": r"set_mismatch_message_filter",
    }
    for label, pattern in forbidden.items():
        require(re.search(pattern, cleaned) is None,
                "forbidden {} present".format(label))
    dimensions = (
        r"logic\s+term_valid_q\s*\[0:COEFFICIENTS-1\]\[0:TERMS-1\]",
        r"logic\s+term_negative_q\s*\[0:COEFFICIENTS-1\]\[0:TERMS-1\]",
        r"logic\s+\[2:0\]\s+term_shift_q\s*\[0:COEFFICIENTS-1\]\[0:TERMS-1\]",
        r"logic\s+signed\s+\[ACC_W-1:0\]\s+bias_q\s*\[0:T-1\]",
        r"intermediate_bank_q\s*\n\s*\[0:1\]\[0:INTERMEDIATES-1\]",
    )
    for pattern in dimensions:
        require(len(re.findall(pattern, cleaned)) == 1,
                "unpadded array dimension drift")
    require(re.search(r"\[(?:0:)?(?:15|31|63)\]", cleaned) is None,
            "candidate padded array dimension present")
    return cleaned


def canonical_diff(r8_lines, r9_lines):
    text = "\n".join(difflib.unified_diff(
        r8_lines, r9_lines, fromfile="r8", tofile="r9", lineterm="")) + "\n"
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_static_delta(r8_source, r9_source):
    require(hashlib.sha256(r8_source.encode("utf-8")).hexdigest() ==
            EXPECTED["r8_rtl"], "r8 source SHA drift")
    require(hashlib.sha256(r9_source.encode("utf-8")).hexdigest() ==
            EXPECTED["r9_rtl"], "r9 source SHA drift")
    r8_lines = r8_source.splitlines()
    r9_lines = r9_source.splitlines()
    matcher = difflib.SequenceMatcher(None, r8_lines, r9_lines, autojunk=False)
    changed = [row for row in matcher.get_opcodes() if row[0] != "equal"]
    require(changed == [
        ("delete", 284, 285, 284, 284),
        ("delete", 286, 287, 285, 285),
        ("replace", 353, 364, 351, 370),
        ("replace", 390, 415, 396, 442)],
        "r8-to-r9 changed-region drift")
    diff_text, diff_sha = canonical_diff(r8_lines, r9_lines)
    require(diff_sha == EXPECTED["canonical_diff"],
            "r8-to-r9 canonical diff drift")
    require(len(r8_lines) == 465 and len(r9_lines) == 492,
            "r8/r9 source line population drift")
    require("integer selected_lane;" in r8_source
            and "integer selected_intermediate;" in r8_source
            and "integer selected_lane;" not in r9_source
            and "integer selected_intermediate;" not in r9_source,
            "removed dynamic-index variable drift")
    validate_r9_static_intent(r9_source)

    witnesses = 0
    rows = set()
    coefficients = set()
    intermediates = set()
    for phase in range(5):
        for output_index in range(32):
            old_row = (phase * 2) + (output_index // 16)
            matched_rows = [row for row in range(10) if old_row == row]
            matched_phases = [item for item in range(5) if phase == item]
            require(matched_rows == [old_row] and matched_phases == [phase],
                    "static phase/row selector is not one-hot")
            rows.add(old_row)
            for rank_index in range(3):
                old_coefficient = old_row * 3 + rank_index
                matched_coefficients = [item for item in range(30)
                                        if old_coefficient == item]
                old_intermediate = (rank_index * 16) + (output_index % 16)
                new_intermediate = (rank_index * 16) + (output_index % 16)
                require(matched_coefficients == [old_coefficient]
                        and old_intermediate == new_intermediate,
                        "static coefficient/intermediate selector mismatch")
                coefficients.add(old_coefficient)
                intermediates.add(old_intermediate)
                witnesses += 1
    require(rows == set(range(10)) and coefficients == set(range(30))
            and intermediates == set(range(48)) and witnesses == 480,
            "reachable index-domain witness population drift")
    return {
        "canonical_diff_sha256": diff_sha,
        "changed_regions": len(changed),
        "reachable_index_selection_witnesses": witnesses,
        "rows_covered": len(rows), "coefficients_covered": len(coefficients),
        "intermediates_covered": len(intermediates),
        "padding_detected": False, "formality_filter_detected": False,
    }


def validate_sim_text(sim):
    exact_lines = (
        "M37_SVA_BOUND=1",
        "SIMULATOR=Synopsys VCS",
        "ASSERTIONS=enabled",
        "M37_RANDOM_SEED=0x4d370203",
        "M37_PASS total_tiles=245 nominal_tiles=96 dut_unique_signed_input_coefficient_product_pairs=65536 product_miters=117600 bit_miters=39200 arithmetic_issues=1225 no_data_multiplier=1",
        "M37_UNIQUENESS unique_tile_payloads=96 unique_expected_product_fingerprints=96 unique_expected_bitmaps=96 consecutive_identical=0 nominal_unique_signed_inputs=256",
        "M37_FLOW conditional_standalone_accept_ii5_matches=69 phase4_chain_accepts=220 max_fifo=16 fifo_full_cycles=249 full_pop_push=116 stalls=1001/147 done_with_fifo_pending=245",
        "M37_CONFIG config_load_release_reload=15/15/14 release_reject_busy_fifo_input=599/599/599/571 live_pin_perturbations=96 legal_zero_min_max=1",
        "M37_ILLEGAL illegal_matrix=210/210 illegal_classes=30,30,30,30,30,30,30",
        "M37_DIVERSITY generic_saturation=80/96 diversity=19740/19460",
    )
    for line in exact_lines:
        require(sim.splitlines().count(line) == 1,
                "simulation exact metric/marker drift")
    threshold = re.findall(r"^M37_THRESHOLD .+$", sim, re.MULTILINE)
    require(threshold == [
        "M37_THRESHOLD index=0 value=-8388608 equal=48 just_below_raw=16 positive_saturation=16 negative_saturation=32",
        "M37_THRESHOLD index=1 value=-12345 equal=16 just_below_raw=16 positive_saturation=16 negative_saturation=16",
        "M37_THRESHOLD index=2 value=0 equal=112 just_below_raw=16 positive_saturation=16 negative_saturation=16",
        "M37_THRESHOLD index=3 value=12345 equal=16 just_below_raw=16 positive_saturation=16 negative_saturation=16",
        "M37_THRESHOLD index=4 value=8388607 equal=32 just_below_raw=16 positive_saturation=16 negative_saturation=16"],
        "threshold metric population drift")
    require("Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64" in sim,
            "VCS version marker absent")
    covers = [int(value) for value in re.findall(
        r",\s*2758 attempts,\s*([0-9]+) match$", sim, re.MULTILINE)]
    require(covers == [220, 1271, 249, 117, 245, 571, 133, 210],
            "SVA cover vector drift")
    require(re.search(
        r"failed at|Offending|assertion[^\n]*(?:fail|error)|"
        r"(?:^|[^A-Za-z])(?:Error|Fatal)(?:[^A-Za-z]|$)",
        sim, re.IGNORECASE | re.MULTILINE) is None,
        "simulation failure signature present")
    return covers


def validate_source_report(report):
    require(report.splitlines().count(
        "status=PASS_M37_R9_STATIC_INDEX_SOURCE_AUDIT") == 1,
        "source report status drift")
    expected = {
        "canonical_star_token_count": "43",
        "data_multiplication_token_count": "0",
        "runtime_non_power_of_two_control_multiplication_token_count": "0",
        "rank3_shift_add_count": "1",
        "bounded_bias_row_select_count": "1",
        "bounded_runtime_phase_select_count": "1",
        "bounded_coefficient_select_count": "1",
        "dynamic_bias_array_index_count": "0",
        "dynamic_term_array_index_count": "0",
        "dynamic_intermediate_array_index_count": "0",
        "padding_used": "false",
        "formality_message_filter_used": "false",
    }
    observed = {}
    for line in report.splitlines():
        if "=" in line and not line.startswith(("line=", "counterexample=")):
            key, value = line.split("=", 1)
            observed[key] = value
    for key, value in expected.items():
        require(observed.get(key) == value,
                "source report metric drift: {}".format(key))
    counterexamples = re.findall(
        r"^counterexample=(\S+) result=REJECT detail=.+$", report,
        re.MULTILINE)
    require(counterexamples == [
        "hidden_data_a_times_b_no_spaces",
        "hidden_control_selected_row_times_space_rank",
        "comment_shift_add_signature_real_multiply",
        "dynamic_bias_selected_row_oob",
        "dynamic_term_selected_coefficient_oob",
        "dynamic_selected_intermediate_oob"],
        "source report counterexample population drift")
    return counterexamples


def validate_payload(payload):
    exact_keys(payload, {
        "schema", "status", "date", "review", "anchors", "observed",
        "independent_delta_audit", "admitted", "claim_boundary", "validator"},
        "M37-r9 independent admission")
    require(payload["schema"] == EXPECTED_SCHEMA, "admission schema drift")
    require(payload["status"] == EXPECTED_STATUS, "admission status drift")
    require(payload["date"] == "2026-08-22", "admission date drift")
    require(payload["review"] == {
        "independent_of_r9_implementation": True,
        "score_0_to_100": 94,
        "p0": 0, "p1": 1, "p2": 2,
        "go": "EXACT_CURRENT_R9_STANDALONE_VCS_AND_STATIC_INDEX_SOURCE_INTENT_ONLY",
        "nogo": "ALL_STATE_OR_FOUR_STATE_EQUIVALENCE_DC_STA_FORMALITY_PHYSICAL_ZERO_MULTIPLIER_PPA_POWER_ENERGY_SYSTEM_HEADLINE",
        "p1_findings": [
            "r8_to_r9_all_state_and_four_state_equivalence_is_not_proven_without_Formality"],
        "p2_findings": [
            "r9_RTL_is_live_exact_SHA_bound_source_not_an_immutable_revision_snapshot",
            "implementation_source_auditor_dynamic_forgeries_can_fail_on_line_ledger_before_semantic_diagnostic"]},
        "review decision drift")
    require(payload["admitted"] == {
        "standalone_r9_vcs_functional": True,
        "exact_sha_bound_static_index_source_delta": True,
        "reachable_integer_index_selection_equivalence": True,
        "same_frozen_vector_bytes_as_r8": True,
        "all_state_or_four_state_equivalence": False,
        "physical_zero_multiplier": False,
        "dc": False, "sta": False, "formality": False, "ppa": False,
        "power": False, "energy": False, "system": False, "headline": False},
        "admission claim boundary drift")
    require(payload["claim_boundary"] == {
        "permitted": "exact current r9 SHA, standalone frozen-workload Synopsys VCS/SVA behavior, reachable integer index-selection equivalence to r8, and static-index source intent",
        "forbidden": "all-state or four-state equivalence, successful Formality, physical zero-multiplier structure, DC/STA/PPA/power/energy, integrated M31/M38 behavior, memory or full-system speedup, and any DATE headline"},
        "claim boundary text drift")
    return True


def validate_external(payload):
    anchors = payload["anchors"]
    exact_keys(anchors, {
        "receipt", "contract", "rtl", "runner", "source_auditor",
        "r8_source_auditor", "r8_snapshot", "r8_admission", "r8_validator",
        "input_manifest", "output_manifest", "run_local_seal",
        "r8_validation_report"}, "M37-r9 anchors")
    expected_pairs = {
        "receipt": ["hw_autoresearch_nts07/contracts/m37_output_receipt_r4_20260822.json", EXPECTED["receipt"]],
        "contract": ["hw_autoresearch_nts07/contracts/m37_csd_reconstruct_t10_vcs_contract_r4_20260822.json", EXPECTED["contract"]],
        "rtl": ["hw_autoresearch_nts07/rtl_m37/qfit_atlif_csd_reconstruct_t10.sv", EXPECTED["r9_rtl"]],
        "runner": ["hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m37_csd_reconstruct_t10_r9_sva.sh", EXPECTED["runner"]],
        "source_auditor": ["hw_autoresearch_nts07/dc_handoff/scripts/audit_m37_r9_source_intent.py", EXPECTED["r9_auditor"]],
        "r8_source_auditor": ["hw_autoresearch_nts07/dc_handoff/scripts/audit_m37_r8_source_intent.py", EXPECTED["r8_auditor"]],
        "r8_snapshot": ["hw_autoresearch_nts07/evidence_snapshots/m37_r8_ab7d73a6_20260822/qfit_atlif_csd_reconstruct_t10.sv", EXPECTED["r8_rtl"]],
        "r8_admission": ["hw_autoresearch_nts07/contracts/m37_r8_independent_vcs_source_intent_admission_r1_20260822.json", EXPECTED["r8_admission"]],
        "r8_validator": ["hw_autoresearch_nts07/dc_handoff/scripts/validate_m37_r8_vcs_source_intent_admission.py", EXPECTED["r8_validator"]],
    }
    for name, pair in expected_pairs.items():
        require(anchors[name] == pair, "{} admission anchor drift".format(name))
        require(sha256(resolve(pair[0])) == pair[1],
                "{} live SHA drift".format(name))
    require((resolve(anchors["receipt"][0]).stat().st_mode & 0o777) == 0o444,
            "r9 receipt mode drift")
    require(anchors["input_manifest"] == [str(RUN_DIR / "input_sha256.txt"), EXPECTED["input_manifest"], 15]
            and anchors["output_manifest"] == [str(RUN_DIR / "output_sha256.txt"), EXPECTED["output_manifest"], 5]
            and anchors["run_local_seal"] == [str(RUN_DIR / "run_local_seal.sha256"), EXPECTED["local_seal"], 3]
            and anchors["r8_validation_report"] == [str(RUN_DIR / "r8_admission_validation.txt"), EXPECTED["r8_validation_report"]],
            "run anchor drift")

    receipt = read_json_no_duplicates(resolve(anchors["receipt"][0]))
    exact_keys(receipt, {
        "schema", "status", "date", "advances", "contract", "math_anchor",
        "r8_historical_admission", "r9_delta", "files", "vcs_run", "observed",
        "claim_boundary", "review_required", "headline_admitted"},
        "M37-r9 receipt")
    require(receipt["schema"] == "m37_output_receipt_v4"
            and receipt["status"] ==
            "PASS_R9_STATIC_BOUNDED_INDEX_VCS_SVA_PENDING_INDEPENDENT_HAMMER_NO_DC_OR_FORMALITY_CLAIM",
            "r9 receipt schema/status drift")
    require(receipt["contract"] == {"path": anchors["contract"][0],
                                    "sha256": anchors["contract"][1]},
            "r9 receipt contract drift")
    require(receipt["r9_delta"] == {
        "old_rtl_revision": "r8", "old_rtl_sha256": EXPECTED["r8_rtl"],
        "new_rtl_revision": "r9", "new_rtl_sha256": EXPECTED["r9_rtl"],
        "implementation": "bounded static row phase and coefficient loops with equality selection plus direct compile-time rank-lane intermediate subscript",
        "dynamic_bias_coefficient_intermediate_array_indices_remaining": 0,
        "padding_used": False, "formality_message_filter_used": False,
        "r9_source_counterexamples_rejected": 6},
        "r9 receipt delta drift")
    require(receipt["claim_boundary"] == {
        "permitted_before_independent_hammer": "exact-SHA r9 static-index source intent and standalone frozen-workload VCS/SVA regression only",
        "r8_fresh_dc_zero_multiplier_result": "HISTORICAL_R8_ONLY_NOT_R9_EVIDENCE",
        "r8_fresh_formality_result": "FAIL_FMR_ELAB_147_REFERENCE_NOT_LINKED_DO_NOT_CITE_AS_CLOSED",
        "r9_dc": False, "r9_sta": False, "r9_formality": False,
        "r9_ppa": False, "r9_power": False, "r9_energy": False,
        "r9_system": False, "headline": False},
        "r9 receipt claim boundary drift")
    require(receipt["review_required"] is True
            and receipt["headline_admitted"] is False,
            "r9 receipt review/headline boundary drift")
    for name, pair in receipt["files"].items():
        require(isinstance(pair, list) and len(pair) == 2
                and sha256(resolve(pair[0])) == pair[1],
                "r9 receipt source drift: {}".format(name))
    require(receipt["files"]["rtl"] == anchors["rtl"]
            and receipt["files"]["runner"] == anchors["runner"]
            and receipt["files"]["r9_source_auditor"] == anchors["source_auditor"]
            and receipt["files"]["r8_base_source_auditor"] == anchors["r8_source_auditor"],
            "r9 receipt source anchor mismatch")

    contract = read_json_no_duplicates(resolve(anchors["contract"][0]))
    require(contract["contract"] == "m37_csd_reconstruct_t10_vcs_contract_r4"
            and contract["r9_delta"]["old_rtl_sha256"] == EXPECTED["r8_rtl"]
            and contract["r9_delta"]["new_rtl_sha256"] == EXPECTED["r9_rtl"]
            and contract["r9_delta"]["padding_used"] is False
            and contract["r9_delta"]["formality_message_filter_used"] is False
            and contract["claim_boundary"]["r9_dc_sta_formality_ppa_power_energy_system_headline_admitted"] is False,
            "r9 contract boundary drift")

    expected_inputs = {}
    for pair in receipt["files"].values():
        prefix = "hw_autoresearch_nts07/"
        require(pair[0].startswith(prefix), "r9 receipt source root drift")
        expected_inputs[pair[0][len(prefix):]] = pair[1]
    expected_inputs.update({
        "contracts/m37_csd_reconstruct_t10_vcs_contract_r4_20260822.json": EXPECTED["contract"],
        "contracts/m37_phase_decoupled_csd_reconstruct_input_contract_r2_20260822.json": EXPECTED["math_contract"],
        "results/m37_phase_decoupled_csd_reconstruct_r2_20260822/m37_phase_decoupled_csd_reconstruct.json": EXPECTED["math_result"],
        "contracts/m37_r8_independent_vcs_source_intent_admission_r1_20260822.json": EXPECTED["r8_admission"],
        "dc_handoff/scripts/validate_m37_r8_vcs_source_intent_admission.py": EXPECTED["r8_validator"],
        "evidence_snapshots/m37_r8_ab7d73a6_20260822/qfit_atlif_csd_reconstruct_t10.sv": EXPECTED["r8_rtl"],
        "evidence_snapshots/m37_r8_ab7d73a6_20260822/README.provenance.txt": EXPECTED["r8_provenance"],
        "evidence_snapshots/m37_r8_ab7d73a6_20260822/snapshot_contents.sha256": EXPECTED["r8_ledger"],
    })
    require(len(expected_inputs) == 15, "r9 expected input population drift")
    verify_manifest(RUN_DIR / "input_sha256.txt", EXPECTED["input_manifest"],
                    15, HW_ROOT, expected_inputs)
    expected_outputs = {
        str(RUN_DIR / "compile.log"): EXPECTED["compile_log"],
        str(RUN_DIR / "sim.log"): EXPECTED["sim_log"],
        str(RUN_DIR / "vectors.txt"): EXPECTED["vectors"],
        str(RUN_DIR / "r9_source_intent_audit.txt"): EXPECTED["source_report"],
        str(RUN_DIR / "runner_status.txt"): EXPECTED["runner_status"],
    }
    verify_manifest(RUN_DIR / "output_sha256.txt", EXPECTED["output_manifest"],
                    5, RUN_DIR, expected_outputs)
    expected_local = {
        "input_sha256.txt": EXPECTED["input_manifest"],
        "output_sha256.txt": EXPECTED["output_manifest"],
        "runner_status.txt": EXPECTED["runner_status"],
    }
    verify_manifest(RUN_DIR / "run_local_seal.sha256", EXPECTED["local_seal"],
                    3, RUN_DIR, expected_local)
    require(sha256(RUN_DIR / "r8_admission_validation.txt") ==
            EXPECTED["r8_validation_report"]
            and "M37_R8_VCS_SOURCE_INTENT_ADMISSION_VALID=1" in
            (RUN_DIR / "r8_admission_validation.txt").read_text(),
            "r8 validation report drift")

    compile_text = (RUN_DIR / "compile.log").read_text(errors="replace")
    require("Chronologic VCS" in compile_text,
            "VCS compiler marker absent")
    require(re.search(r"(?:^|[^A-Za-z])(?:warning|error|fatal)(?:[^A-Za-z]|$)",
                      compile_text, re.IGNORECASE | re.MULTILINE) is None,
            "compile warning/error/fatal signature present")
    sim_text = (RUN_DIR / "sim.log").read_text(errors="replace")
    covers = validate_sim_text(sim_text)
    source_report = (RUN_DIR / "r9_source_intent_audit.txt").read_text()
    source_counterexamples = validate_source_report(source_report)
    runner_status = (RUN_DIR / "runner_status.txt").read_text()
    require(runner_status ==
            "status=PASS_R9_STATIC_INDEX_VCS_SVA_PENDING_INDEPENDENT_HAMMER_NO_DC_CLAIM\n"
            "review_required=true\nheadline_admitted=false\n"
            "r9_rtl_sha256={}\n"
            "r8_dc_zero_multiplier_state=HISTORICAL_R8_ONLY_NOT_R9_EVIDENCE\n"
            "r8_formality_state=FAIL_FMR_ELAB_147_DO_NOT_CITE_AS_CLOSED\n"
            "r9_dc_sta_formality_ppa_power_energy_system_admitted=false\n".format(EXPECTED["r9_rtl"]),
            "runner status boundary drift")

    assertions = resolve(receipt["files"]["assertions"][0]).read_text()
    require(len(re.findall(r"^\s*assert\s+property", assertions, re.MULTILINE)) == 21
            and len(re.findall(r"^\s*cover\s+property", assertions, re.MULTILINE)) == 8,
            "r9 SVA property population drift")
    r8_source = resolve(anchors["r8_snapshot"][0]).read_text()
    r9_source = resolve(anchors["rtl"][0]).read_text()
    delta = validate_static_delta(r8_source, r9_source)
    runner_text = resolve(anchors["runner"][0]).read_text()
    require("set_mismatch_message_filter" not in runner_text,
            "Formality mismatch filter present in r9 runner")

    auditor = load_module(resolve(anchors["source_auditor"][0]),
                          "m37_r9_source_auditor_independent_replay")
    base = auditor.load_base_auditor(resolve(anchors["r8_source_auditor"][0]))
    auditor.configure_base(base)
    _, stars = auditor.audit_text(base, r9_source)
    replayed = auditor.run_counterexamples(base, r9_source)
    require(len(stars) == 43 and len(replayed) == 6
            and all("result=REJECT" in item for item in replayed),
            "r9 source auditor replay drift")
    require(source_counterexamples == [item.split()[0].split("=", 1)[1]
                                        for item in replayed],
            "source counterexample replay/report mismatch")

    observed = payload["observed"]
    require(observed == {
        "seed": "0x4d370203", "tiles": 245, "nominal_tiles": 96,
        "unique_nominal_payloads_products_bitmaps": [96, 96, 96],
        "signed_inputs": 256, "input_coefficient_pairs": 65536,
        "product_miters": 117600, "bit_miters": 39200,
        "arithmetic_issues": 1225,
        "illegal_accept_reject": [210, 210],
        "illegal_classes": [30, 30, 30, 30, 30, 30, 30],
        "sva_cover_matches": covers,
        "vectors_sha256": EXPECTED["vectors"],
        "input_output_local_manifest_counts": [15, 5, 3]},
        "independent observed ledger drift")
    require(payload["independent_delta_audit"] == dict(delta,
        source_audit_replay_star_tokens=43,
        source_audit_replay_counterexamples=6,
        independent_counterexamples=11),
        "independent delta audit drift")
    return {"covers": covers, "delta": delta}


def run_independent_counterexamples(payload):
    rejected = []
    for name, mutator in (
            ("forged_status", lambda item: item.update(status="PASS_DC")),
            ("forged_dc_claim", lambda item: item["admitted"].update(dc=True)),
            ("forged_receipt_sha", lambda item: item["anchors"]["receipt"].__setitem__(1, "0" * 64))):
        forged = copy.deepcopy(payload)
        mutator(forged)
        try:
            validate_payload(forged)
            if name == "forged_receipt_sha":
                validate_external(forged)
        except (ValidationFailure, OSError, ValueError, KeyError):
            rejected.append(name)
            continue
        raise ValidationFailure("{} admission counterexample accepted".format(name))

    input_manifest = (RUN_DIR / "input_sha256.txt").read_text()
    manifest_cases = {
        "forged_manifest_hash": input_manifest.replace(input_manifest[:64], "0" * 64, 1),
        "duplicate_manifest_target": input_manifest + input_manifest.splitlines()[0] + "\n",
        "truncated_manifest": "\n".join(input_manifest.splitlines()[:-1]) + "\n",
    }
    for name, forged in manifest_cases.items():
        try:
            rows = parse_manifest_text(forged, 15)
            for digest, raw in rows:
                target = HW_ROOT / raw
                require(target.is_file() and sha256(target) == digest,
                        "forged manifest target SHA")
        except (ValidationFailure, OSError):
            rejected.append(name)
            continue
        raise ValidationFailure("{} counterexample accepted".format(name))

    sim = (RUN_DIR / "sim.log").read_text(errors="replace")
    for name, forged in (
            ("forged_sim_metric", sim.replace("total_tiles=245", "total_tiles=999", 1)),
            ("forged_sim_error", sim + "\nError: forged failure\n")):
        try:
            validate_sim_text(forged)
        except ValidationFailure:
            rejected.append(name)
            continue
        raise ValidationFailure("{} counterexample accepted".format(name))

    source = resolve(payload["anchors"]["rtl"][0]).read_text()
    static_intermediate = """[(rank_index*LANES)
                                                                + (output_index
                                                                    % LANES)]"""
    dynamic_cases = {
        "dynamic_bias_index": source.replace("bias_q[row_index]", "bias_q[selected_row]", 1),
        "dynamic_term_index": source.replace(
            "term_valid_q[\n                                                            coefficient_index]",
            "term_valid_q[selected_coefficient]", 1),
        "dynamic_intermediate_index": source.replace(
            static_intermediate, "[selected_intermediate]", 1),
    }
    for name, forged in dynamic_cases.items():
        require(forged != source, "cannot construct {}".format(name))
        try:
            validate_r9_static_intent(forged)
        except ValidationFailure:
            rejected.append(name)
            continue
        raise ValidationFailure("{} source counterexample accepted".format(name))
    require(len(rejected) == 11, "independent counterexample population drift")
    return rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("admission", type=pathlib.Path)
    args = parser.parse_args()
    payload = read_json_no_duplicates(args.admission)
    validate_payload(payload)
    validator_pair = payload["validator"]
    require(validator_pair == [
        "hw_autoresearch_nts07/dc_handoff/scripts/validate_m37_r9_vcs_static_index_admission.py",
        sha256(pathlib.Path(__file__).resolve())],
        "independent validator self-identity drift")
    validate_external(payload)
    rejected = run_independent_counterexamples(payload)
    print("M37_R9_VCS_STATIC_INDEX_ADMISSION_VALID=1 status={}".format(
        EXPECTED_STATUS))
    print("score=94 p0=0 p1=1 p2=2 manifests=input15/output5/local3 "
          "source_delta_witnesses=480 independent_counterexamples={}".format(
              len(rejected)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationFailure, OSError, ValueError, KeyError) as error:
        print("M37_R9_VCS_STATIC_INDEX_ADMISSION_VALID=0 detail={}".format(error),
              file=sys.stderr)
        raise SystemExit(1)
