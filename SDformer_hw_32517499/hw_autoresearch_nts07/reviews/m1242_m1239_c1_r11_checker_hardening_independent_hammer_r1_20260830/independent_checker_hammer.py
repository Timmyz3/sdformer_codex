#!/usr/bin/env python3
"""Independent in-memory hammer for the M1239 R11 source checker.

The hammer reads and hashes local source, runs the declared unittest suite,
and feeds mutations to ``audit_text`` in memory.  It never writes candidate
source and never invokes VCS, simv, EDA, GPU, or remote work.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERIF = ROOT / "verif_m1232r11_c1_common_charge_protocol"

PATHS = {
    "r11_tb": VERIF / "tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv",
    "checker": VERIF / "check_m1232r11_source.py",
    "tests": VERIF / "test_m1232r11_source.py",
    "r10_tb": ROOT / "verif_m1226r10_c1_common_charge_protocol/tb_m1226r10_m1162_common_charge_protocol_unit_delay_r10.sv",
    "m1239_contract": ROOT / "contracts/m1239_m1235_m1232_c1_r11_checker_hardening_source_contract_r1_20260830.json",
    "m1235_review": ROOT / "reviews/m1235_m1232_c1_r11_source_independent_hammer_r1_20260830/review.json",
    "m528": ROOT / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
    "m935": ROOT / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv",
    "m1162": ROOT / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv",
    "sva_r3": ROOT / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv",
    "docs359": ROOT / "docs/359_DATE终局冻结_20260813.md",
}

EXPECTED = {
    "r11_tb": "850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776",
    "checker": "ccec195091bd79d8d24008ac9b1d4b2e6259a7c38b51cb695a17bff2678d5a94",
    "tests": "56c279d71e7fcf5350166f8e31dca010d2635de1aaf414df6c0d36c68e0b9f36",
    "r10_tb": "f2df09cf6177f1dcb48e7eae24bedfe914a9222d417eee9d08a11d0a1d89c14b",
    "m1239_contract": "1f33891748c30307868a5a765e7a410d4887872e780c55fbb75f9f3e34407a61",
    "m1235_review": "404d40e31da9aede75a51ef92858fcbfee51bcd77767c1666c31f37f12158d59",
    "m528": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "m935": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "m1162": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    "sva_r3": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PRESERVED_M1232_TESTS = {
    "test_canonical_structure_passes",
    "test_claim_inflation_is_rejected",
    "test_normal_row_mutation_is_rejected",
    "test_r10_normal_mutation_is_rejected",
    "test_random_core_ready_posedge_race_is_rejected",
    "test_random_extra_response_posedge_is_rejected",
    "test_random_post_retirement_edge_removal_is_rejected",
    "test_random_ready_retirement_removal_is_rejected",
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
    spec = importlib.util.spec_from_file_location("m1239_checker", PATHS["checker"])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_in_task(checker, text: str, old: str, new: str) -> str:
    task = checker.extract_task(text, "random_legal_transaction")
    if task is None or old not in task:
        raise AssertionError("mutation anchor absent: " + old)
    return text.replace(task, task.replace(old, new, 1), 1)


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

    author_dir = ROOT / "reviews/m1239_m1235_c1_r11_checker_hardening_source_author_r1_20260830"
    author_seal_ok = verify_double_seal(author_dir, "SHA256SUMS")
    contract_seal_ok = verify_double_seal(
        PATHS["m1239_contract"].parent,
        PATHS["m1239_contract"].name + ".sha256",
    )
    if not author_seal_ok:
        errors.append("M1239 author double seal does not verify")
    if not contract_seal_ok:
        errors.append("M1239 contract double seal does not verify")

    checker = load_checker()
    r11 = PATHS["r11_tb"].read_text()
    r10 = PATHS["r10_tb"].read_text()

    checker_run = subprocess.run(
        [sys.executable, str(PATHS["checker"])], cwd=ROOT,
        universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    tests_run = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", str(PATHS["tests"])],
        cwd=ROOT, universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    observed_test_names = {
        line.split()[0] for line in tests_run.stderr.splitlines()
        if line.startswith("test_")
    }
    preserved_tests_ok = PRESERVED_M1232_TESTS.issubset(observed_test_names)
    test_count = sum(1 for line in tests_run.stderr.splitlines()
                     if line.startswith("test_"))
    if checker_run.returncode != 0:
        errors.append("canonical checker failed")
    if tests_run.returncode != 0 or test_count != 18:
        errors.append("declared 18-test suite did not pass exactly")
    if not preserved_tests_ok:
        errors.append("one or more original 15 M1232 tests disappeared")

    direct = {
        "random_request_window_disabled": replace_in_task(
            checker, r11,
            "random_request_window_active = 1'b1;",
            "random_request_window_active = 1'b0;",
        ),
        "post_retire_response_count_oracle_removed": replace_in_task(
            checker, r11,
            "|| response_accept_count != response0 + 1\n"
            "                    || weight_req_valid",
            "|| weight_req_valid",
        ),
        "hold_cycles_repeat_zero": replace_in_task(
            checker, r11, "repeat (hold_cycles) begin", "repeat (0) begin",
        ),
    }
    nearby = {
        "window_enable_immediately_overridden": replace_in_task(
            checker, r11,
            "random_request_window_active = 1'b1;",
            "random_request_window_active = 1'b1;\n"
            "            random_request_window_active = 1'b0;",
        ),
        "positive_hold_assignment_immediately_overridden": replace_in_task(
            checker, r11,
            "hold_cycles = 1 + prng_q[9:7];",
            "hold_cycles = 1 + prng_q[9:7];\n"
            "            hold_cycles = 0;",
        ),
        "window_enable_token_only_in_comment": replace_in_task(
            checker, r11,
            "random_request_window_active = 1'b1;",
            "// random_request_window_active = 1'b1;\n"
            "            random_request_window_active = 1'b0;",
        ),
        "positive_hold_loop_token_only_in_comment": replace_in_task(
            checker, r11,
            "repeat (hold_cycles) begin",
            "// repeat (hold_cycles) begin\n            repeat (0) begin",
        ),
    }
    direct_raw = {name: checker.audit_text(mutant, r10)
                  for name, mutant in direct.items()}
    nearby_raw = {name: checker.audit_text(mutant, r10)
                  for name, mutant in nearby.items()}
    direct_accepted = [name for name, reasons in direct_raw.items()
                       if not reasons]
    nearby_accepted = [name for name, reasons in nearby_raw.items()
                       if not reasons]
    if direct_accepted:
        errors.append("required direct mutations accepted: %s" % direct_accepted)

    release_allowed = not errors and not nearby_accepted
    status = (
        "PASS_RELEASE_AUTHORING_ALLOWED"
        if release_allowed else
        "NO_GO_RELEASE_AUTHORING__CHECKER_TOKEN_DECOY_HARDENING_REQUIRED"
    )
    result = {
        "schema": "m1242_m1239_c1_r11_checker_hardening_independent_hammer_r1_v1",
        "status": status,
        "score": 88 if not errors else 82,
        "p0_count": 0,
        "p1_count": 1 if nearby_accepted else 0,
        "p2_count": 0,
        "mechanical_audit_pass": not errors,
        "errors": errors,
        "bindings": hashes,
        "seal_audit": {
            "m1239_author_double_seal_verified": author_seal_ok,
            "m1239_contract_double_seal_verified": contract_seal_ok,
        },
        "declared_test_suite": {
            "exit_code": tests_run.returncode,
            "tests_run": test_count,
            "all_18_passed": tests_run.returncode == 0 and test_count == 18,
            "original_15_test_names_preserved": preserved_tests_ok,
        },
        "independent_mutation_audit": {
            "direct_mutations_run": len(direct),
            "direct_mutations_accepted_in_error": direct_accepted,
            "direct_checker_errors": direct_raw,
            "nearby_mutations_run": len(nearby),
            "nearby_mutations_accepted_in_error": nearby_accepted,
            "nearby_checker_errors": nearby_raw,
        },
        "finding": (
            "M1239 closes the three named M1235 holes, but audit_text still "
            "accepts destructive immediate overrides and comment-token decoys; "
            "the checker is lexical rather than execution-order aware."
        ),
        "authorization": {
            "checker_tests_only_repair": True,
            "candidate_tb_mutation": False,
            "release_authoring": release_allowed,
            "vcs": False,
            "simv": False,
            "eda": False,
            "gpu": False,
            "remote": False,
            "rtl_mutation": False,
            "sva_mutation": False,
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
            "paper_admission": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
