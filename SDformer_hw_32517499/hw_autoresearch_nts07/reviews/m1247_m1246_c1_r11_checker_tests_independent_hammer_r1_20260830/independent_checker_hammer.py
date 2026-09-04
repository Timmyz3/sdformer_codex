#!/usr/bin/env python3
"""Independent source-only hammer for the M1246 R11 checker.

This program reads local source, hashes files, runs the declared Python tests,
and applies in-memory mutations.  It never writes candidate source or invokes
VCS, simv, EDA, GPU, or remote work.
"""

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERIF = ROOT / "verif_m1232r11_c1_common_charge_protocol"

PATHS = {
    "r11_tb": VERIF / "tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv",
    "checker": VERIF / "check_m1232r11_source.py",
    "tests": VERIF / "test_m1232r11_source.py",
    "r10_tb": ROOT / "verif_m1226r10_c1_common_charge_protocol/tb_m1226r10_m1162_common_charge_protocol_unit_delay_r10.sv",
    "m1246_contract": ROOT / "contracts/m1246_m1242_m1239_c1_r11_checker_source_contract_r1_20260830.json",
    "m1242_review": ROOT / "reviews/m1242_m1239_c1_r11_checker_hardening_independent_hammer_r1_20260830/review.json",
    "m528": ROOT / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
    "m935": ROOT / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv",
    "m1162": ROOT / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv",
    "sva_r3": ROOT / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv",
    "docs359": ROOT / "docs/359_DATE终局冻结_20260813.md",
}

EXPECTED = {
    "r11_tb": "850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776",
    "checker": "154860a16dfa3e2175653e81c14db645da3718af2c8d659c35299d80248e68fd",
    "tests": "de89c87210e8782d38b84b8202d229a418ebb153583a02043f4080e25aac4605",
    "r10_tb": "f2df09cf6177f1dcb48e7eae24bedfe914a9222d417eee9d08a11d0a1d89c14b",
    "m1246_contract": "7f956c2343a596da25dc8658a79e6f50da462370fa3ddd4c7b4a650ab8c6c88d",
    "m1242_review": "47f1dee909721610ee5c0baeb139bddc7d11dce6a5347e0296d64409544e955c",
    "m528": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "m935": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "m1162": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    "sva_r3": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_M1246_TESTS = {
    "test_canonical_structure_passes",
    "test_claim_inflation_is_rejected",
    "test_normal_row_mutation_is_rejected",
    "test_r10_normal_mutation_is_rejected",
    "test_random_core_ready_posedge_race_is_rejected",
    "test_random_extra_response_posedge_is_rejected",
    "test_random_hold_assignment_immediate_override_is_rejected",
    "test_random_hold_loop_comment_decoy_is_rejected",
    "test_random_hold_loop_string_decoy_is_rejected",
    "test_random_hold_loop_zero_trip_is_rejected",
    "test_random_post_response_oracle_removal_is_rejected",
    "test_random_post_retirement_edge_removal_is_rejected",
    "test_random_ready_retirement_removal_is_rejected",
    "test_random_request_window_comment_decoy_is_rejected",
    "test_random_request_window_disable_is_rejected",
    "test_random_request_window_immediate_override_is_rejected",
    "test_random_request_window_string_decoy_is_rejected",
    "test_random_response_stability_removal_is_rejected",
    "test_random_state_dump_removal_is_rejected",
    "test_random_sva_mask_injection_is_rejected",
    "test_random_tuple_retirement_removal_is_rejected",
    "test_tuple_helper_early_core_ready_release_is_rejected",
    "test_workload_count_mutation_is_rejected",
    "test_zero_sva_gate_removal_is_rejected",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checker():
    spec = importlib.util.spec_from_file_location("m1246_checker", PATHS["checker"])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_double_seal(directory: Path, manifest_name: str) -> bool:
    manifest = directory / manifest_name
    outer = directory / (manifest_name + ".seal.sha256")
    if not manifest.is_file() or not outer.is_file():
        return False
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        if sha(directory / name.strip()) != digest:
            return False
    digest, name = outer.read_text().split(None, 1)
    return name.strip() == manifest_name and sha(manifest) == digest


def reference_strip(text: str) -> str:
    """Independent SV comment/string blanker used as a bounded oracle."""
    result: List[str] = []
    pos = 0
    mode = "code"
    while pos < len(text):
        char = text[pos]
        following = text[pos + 1] if pos + 1 < len(text) else ""
        if mode == "code":
            if char == "/" and following == "/":
                result.extend([" ", " "])
                pos += 2
                mode = "line"
            elif char == "/" and following == "*":
                result.extend([" ", " "])
                pos += 2
                mode = "block"
            elif char == '"':
                result.append(" ")
                pos += 1
                mode = "string"
            else:
                result.append(char)
                pos += 1
        elif mode == "line":
            result.append("\n" if char == "\n" else " ")
            pos += 1
            if char == "\n":
                mode = "code"
        elif mode == "block":
            if char == "*" and following == "/":
                result.extend([" ", " "])
                pos += 2
                mode = "code"
            else:
                result.append("\n" if char == "\n" else " ")
                pos += 1
        else:
            if char == "\\" and following:
                result.extend([" ", "\n" if following == "\n" else " "])
                pos += 2
            elif char == '"':
                result.append(" ")
                pos += 1
                mode = "code"
            else:
                result.append("\n" if char == "\n" else " ")
                pos += 1
    return "".join(result)


def extract_task(text: str, name: str) -> Optional[str]:
    match = re.search(
        r"    task automatic " + re.escape(name) + r"\b.*?^    endtask$",
        text, flags=re.MULTILINE | re.DOTALL)
    return None if match is None else match.group(0)


def replace_in_task(checker, text: str, task_name: str,
                    old: str, new: str) -> str:
    task = checker.extract_task(text, task_name)
    if task is None or old not in task:
        raise AssertionError("mutation anchor absent in %s: %s" %
                             (task_name, old))
    changed = task.replace(old, new, 1)
    return text.replace(task, changed, 1)


def insert_before_task_end(checker, text: str, task_name: str,
                           statement: str) -> str:
    task = checker.extract_task(text, task_name)
    if task is None:
        raise AssertionError("task absent: " + task_name)
    changed = task.replace("    endtask", "        " + statement + "\n    endtask", 1)
    return text.replace(task, changed, 1)


def main() -> int:
    errors: List[str] = []
    hashes: Dict[str, str] = {}
    for name, path in PATHS.items():
        if not path.is_file():
            errors.append("missing %s: %s" % (name, path))
            continue
        hashes[name] = sha(path)
        if hashes[name] != EXPECTED[name]:
            errors.append("%s SHA mismatch: %s" % (name, hashes[name]))

    author_dir = ROOT / "reviews/m1246_m1242_c1_r11_checker_hardening_source_author_r1_20260830"
    author_seal_ok = verify_double_seal(author_dir, "SHA256SUMS")
    contract_seal_ok = verify_double_seal(
        PATHS["m1246_contract"].parent,
        PATHS["m1246_contract"].name + ".sha256")
    if not author_seal_ok:
        errors.append("M1246 author double seal does not verify")
    if not contract_seal_ok:
        errors.append("M1246 contract double seal does not verify")

    checker = load_checker()
    r11 = PATHS["r11_tb"].read_text()
    r10 = PATHS["r10_tb"].read_text()

    child_env = dict(os.environ)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    checker_run = subprocess.run(
        [sys.executable, str(PATHS["checker"])], cwd=str(ROOT), env=child_env,
        universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    tests_run = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", str(PATHS["tests"])],
        cwd=str(ROOT), env=child_env, universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    observed_tests = {
        line.split()[0] for line in tests_run.stderr.splitlines()
        if line.startswith("test_")
    }
    test_count = sum(1 for line in tests_run.stderr.splitlines()
                     if line.startswith("test_"))
    all_test_names_ok = observed_tests == EXPECTED_M1246_TESTS
    if checker_run.returncode != 0:
        errors.append("canonical checker failed")
    if tests_run.returncode != 0 or test_count != 24:
        errors.append("declared 24-test suite did not pass exactly")
    if not all_test_names_ok:
        errors.append("declared M1246 test-name inventory changed")

    canonical_stripped = checker.strip_sv_comments_and_strings(r11)
    reference_stripped = reference_strip(r11)
    offset_preservation_ok = (
        len(canonical_stripped) == len(r11)
        and [i for i, c in enumerate(canonical_stripped) if c == "\n"]
        == [i for i, c in enumerate(r11) if c == "\n"])
    reference_agreement_ok = canonical_stripped == reference_stripped
    if not offset_preservation_ok:
        errors.append("checker comment/string stripper changes offsets/newlines")
    if not reference_agreement_ok:
        errors.append("checker stripper differs from independent reference")

    lexical_probe = (
        "code0; /* enable = 1;\nrepeat (hold_cycles); */ code1;\n"
        "$display(\"escaped \\\" enable = 1; // still text\"); code2;\n"
        "// code3;\ncode4;\n")
    lexical_probe_ok = (
        checker.strip_sv_comments_and_strings(lexical_probe)
        == reference_strip(lexical_probe)
        and len(checker.strip_sv_comments_and_strings(lexical_probe))
        == len(lexical_probe)
        and "enable = 1" not in checker.strip_sv_comments_and_strings(lexical_probe)
        and "repeat (hold_cycles)" not in checker.strip_sv_comments_and_strings(lexical_probe)
        and "code0" in checker.strip_sv_comments_and_strings(lexical_probe)
        and "code1" in checker.strip_sv_comments_and_strings(lexical_probe)
        and "code2" in checker.strip_sv_comments_and_strings(lexical_probe)
        and "code4" in checker.strip_sv_comments_and_strings(lexical_probe))
    if not lexical_probe_ok:
        errors.append("block-comment/escaped-string lexical probe failed")

    random_task = extract_task(reference_stripped, "random_legal_transaction") or ""
    window_writes = list(re.finditer(
        r"\brandom_request_window_active\s*(?:<=|=)\s*[^;]+;", random_task))
    enables = list(re.finditer(
        r"\brandom_request_window_active\s*=\s*1'b1\s*;", random_task))
    disables = list(re.finditer(
        r"\brandom_request_window_active\s*=\s*1'b0\s*;", random_task))
    hold_writes = list(re.finditer(
        r"(?:\bhold_cycles\s*(?:=|<=|\+=|-=|\*=|/=|%=|\+\+|--)"
        r"|(?:\+\+|--)\s*hold_cycles\b)", random_task))
    hold_positive = list(re.finditer(
        r"\bhold_cycles\s*=\s*1\s*\+\s*prng_q\[9:7\]\s*;", random_task))
    hold_repeats = list(re.finditer(
        r"\brepeat\s*\(\s*hold_cycles\s*\)\s*begin", random_task))
    force_pos = random_task.find("force_request(first")
    fire_loop_pos = random_task.find("while ((weight_fire_count != w0 + 1")
    ready_retire_pos = random_task.find(
        "weight_req_ready = 1'b0;\n"
        "            psum_req_ready = 1'b0;\n"
        "            random_request_window_active = 1'b0;")
    backpressure_pos = random_task.find("force dut.core_issue_data_ready = 1'b0;")
    response_drive_pos = random_task.find("if (index[0]) begin", backpressure_pos)
    canonical_order_ok = (
        len(window_writes) == 2 and len(enables) == 1 and len(disables) == 1
        and len(hold_writes) == 1 and len(hold_positive) == 1
        and len(hold_repeats) == 1 and force_pos >= 0 and fire_loop_pos >= 0
        and ready_retire_pos >= 0 and backpressure_pos >= 0
        and response_drive_pos >= 0
        and hold_positive[0].start() < enables[0].start() < force_pos
        < fire_loop_pos < ready_retire_pos < disables[0].start()
        < backpressure_pos < response_drive_pos < hold_repeats[0].start())
    if not canonical_order_ok:
        errors.append("independent canonical task-boundary/order audit failed")

    mutants: Dict[str, str] = {}
    mutants["window_block_comment_decoy"] = replace_in_task(
        checker, r11, "random_legal_transaction",
        "random_request_window_active = 1'b1;",
        "/* random_request_window_active = 1'b1; */\n"
        "            random_request_window_active = 1'b0;")
    mutants["window_escaped_string_decoy"] = replace_in_task(
        checker, r11, "random_legal_transaction",
        "random_request_window_active = 1'b1;",
        "$display(\"escaped \\\" random_request_window_active = 1'b1;\");\n"
        "            random_request_window_active = 1'b0;")
    mutants["window_same_line_immediate_override"] = replace_in_task(
        checker, r11, "random_legal_transaction",
        "random_request_window_active = 1'b1;",
        "random_request_window_active = 1'b1; random_request_window_active = 1'b0;")
    window_other_task = replace_in_task(
        checker, r11, "random_legal_transaction",
        "random_request_window_active = 1'b1;",
        "random_request_window_active = 1'b0;")
    mutants["window_different_task_decoy"] = insert_before_task_end(
        checker, window_other_task, "reset_dut",
        "random_request_window_active = 1'b1;")

    mutants["hold_block_comment_decoy"] = replace_in_task(
        checker, r11, "random_legal_transaction",
        "hold_cycles = 1 + prng_q[9:7];",
        "/* hold_cycles = 1 + prng_q[9:7]; */ hold_cycles = 0;")
    mutants["hold_escaped_string_decoy"] = replace_in_task(
        checker, r11, "random_legal_transaction",
        "hold_cycles = 1 + prng_q[9:7];",
        "$display(\"escaped \\\" hold_cycles = 1 + prng_q[9:7];\"); hold_cycles = 0;")
    mutants["hold_same_line_immediate_override"] = replace_in_task(
        checker, r11, "random_legal_transaction",
        "hold_cycles = 1 + prng_q[9:7];",
        "hold_cycles = 1 + prng_q[9:7]; hold_cycles = 0;")
    hold_other_task = replace_in_task(
        checker, r11, "random_legal_transaction",
        "hold_cycles = 1 + prng_q[9:7];",
        "hold_cycles = 0;")
    mutants["hold_different_task_decoy"] = insert_before_task_end(
        checker, hold_other_task, "reset_dut",
        "hold_cycles = 1 + prng_q[9:7];")

    mutation_errors: Dict[str, List[str]] = {}
    accepted_in_error: List[str] = []
    for name, mutant in sorted(mutants.items()):
        mutation_errors[name] = checker.audit_text(mutant, r10)
        if not mutation_errors[name]:
            accepted_in_error.append(name)
    if accepted_in_error:
        errors.append("nearby mutations accepted: " + ", ".join(accepted_in_error))

    post_hashes = {name: sha(path) for name, path in PATHS.items()
                   if path.is_file()}
    immutable_ok = post_hashes == hashes
    if not immutable_ok:
        errors.append("frozen/source hash changed during read-only hammer")

    result = {
        "schema": "m1247_m1246_c1_r11_checker_tests_independent_hammer_r1_v1",
        "status": ("PASS_M1247_RELEASE_AUTHORING_GO"
                   if not errors else "FAIL_M1247_NO_GO_RELEASE_AUTHORING"),
        "errors": errors,
        "hashes": hashes,
        "seal_audit": {
            "m1246_author_double_seal_verified": author_seal_ok,
            "m1246_contract_double_seal_verified": contract_seal_ok,
        },
        "declared_test_suite": {
            "checker_exit_code": checker_run.returncode,
            "unittest_exit_code": tests_run.returncode,
            "tests_run": test_count,
            "all_24_passed": tests_run.returncode == 0 and test_count == 24,
            "exact_test_name_inventory": all_test_names_ok,
        },
        "independent_lexical_audit": {
            "canonical_reference_agreement": reference_agreement_ok,
            "offset_and_newline_preservation": offset_preservation_ok,
            "block_comment_and_escaped_string_probe": lexical_probe_ok,
            "task_boundary_and_statement_order": canonical_order_ok,
            "window_executable_writes": len(window_writes),
            "window_enables": len(enables),
            "window_retire_disables": len(disables),
            "hold_executable_writes": len(hold_writes),
            "positive_hold_assignments": len(hold_positive),
            "hold_repeats": len(hold_repeats),
        },
        "independent_nearby_mutations": {
            "mutations_run": len(mutants),
            "accepted_in_error": accepted_in_error,
            "all_rejected": not accepted_in_error,
            "checker_errors": mutation_errors,
        },
        "source_immutability": {
            "pre_and_post_hashes_equal": immutable_ok,
            "candidate_tb_mutated": False,
            "dut_or_sva_mutated": False,
        },
        "authorization": {
            "fresh_disjoint_release_authoring": not errors,
            "vcs": False,
            "simv": False,
            "eda": False,
            "gpu": False,
            "remote": False,
            "automatic_retry": False,
        },
        "claims": {
            "checker_hammer_only": True,
            "functional_vcs_verified": False,
            "timing_verified": False,
            "cycles_measured": False,
            "speedup": False,
            "ppa": False,
            "energy": False,
            "system_speedup": False,
            "headline": False,
            "paper_admission": False,
        },
    }
    import json
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
