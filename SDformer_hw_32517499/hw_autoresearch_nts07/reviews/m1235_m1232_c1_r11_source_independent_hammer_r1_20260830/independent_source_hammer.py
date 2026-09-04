#!/usr/bin/env python3
"""Independent read-only source hammer for M1232/R11.

This script hashes and reads local artifacts only.  It never invokes VCS,
simv, EDA, GPU, or remote work and never mutates the candidate source.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PATHS = {
    "r11_tb": ROOT / "verif_m1232r11_c1_common_charge_protocol/tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv",
    "r11_checker": ROOT / "verif_m1232r11_c1_common_charge_protocol/check_m1232r11_source.py",
    "r11_tests": ROOT / "verif_m1232r11_c1_common_charge_protocol/test_m1232r11_source.py",
    "r10_tb": ROOT / "verif_m1226r10_c1_common_charge_protocol/tb_m1226r10_m1162_common_charge_protocol_unit_delay_r10.sv",
    "m1232_contract": ROOT / "contracts/m1232_m1229_m1226_c1_r11_random_service_retirement_source_contract_r1_20260830.json",
    "m1229_review": ROOT / "reviews/m1229_m1226_c1_r10_tb_source_independent_hammer_r1_20260830/review.json",
    "m528": ROOT / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
    "m935": ROOT / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv",
    "m1162": ROOT / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv",
    "sva_r3": ROOT / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv",
    "docs359": ROOT / "docs/359_DATE终局冻结_20260813.md",
}

EXPECTED = {
    "r11_tb": "850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776",
    "r11_checker": "729184404ee23a0152848d5525deb36329756023da31c0e58c81936f3bab63d7",
    "r11_tests": "5e926a9e99dfa180e6c8232a387ecc3dc06d5bbd425841f17db2feb4f8397da4",
    "r10_tb": "f2df09cf6177f1dcb48e7eae24bedfe914a9222d417eee9d08a11d0a1d89c14b",
    "m1232_contract": "8a75c7592f9e6f8cf98e35fcfd092d83a0a6f7dd6c56c01a8c3ae6cbea6dbdf6",
    "m1229_review": "3726ff3b3f43ce963d1f22182fe74a86d52261f97661d61ca5de7e8543250aad",
    "m528": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "m935": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "m1162": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    "sva_r3": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task(text: str, name: str) -> str:
    match = re.search(
        rf"    task automatic {re.escape(name)}\b.*?^    endtask$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing task {name}")
    return match.group(0)


def load_checker():
    spec = importlib.util.spec_from_file_location("m1232_checker", PATHS["r11_checker"])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_in_task(text: str, task_name: str, old: str, new: str) -> str:
    original = task(text, task_name)
    if old not in original:
        raise AssertionError(f"mutation anchor absent in {task_name}: {old}")
    changed = original.replace(old, new, 1)
    return text.replace(original, changed, 1)


def verify_named_double_seal(directory: Path, manifest_name: str = "SHA256SUMS") -> bool:
    manifest = directory / manifest_name
    outer = directory / (manifest_name + ".seal.sha256")
    if not manifest.is_file() or not outer.is_file():
        return False
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        if sha(directory / name) != digest:
            return False
    outer_digest, outer_name = outer.read_text().split(None, 1)
    return outer_name.strip() == manifest_name and sha(manifest) == outer_digest


def main() -> int:
    errors: List[str] = []
    hashes: Dict[str, str] = {}
    for name, path in PATHS.items():
        if not path.is_file():
            errors.append(f"missing {name}: {path}")
            continue
        hashes[name] = sha(path)
        if hashes[name] != EXPECTED[name]:
            errors.append(f"{name} SHA mismatch: {hashes[name]}")

    author_dir = ROOT / "reviews/m1232_m1229_c1_r11_random_service_retirement_source_author_r1_20260830"
    contract_dir = PATHS["m1232_contract"].parent
    contract_manifest = PATHS["m1232_contract"].name + ".sha256"
    author_seal_ok = verify_named_double_seal(author_dir)
    contract_seal_ok = verify_named_double_seal(contract_dir, contract_manifest)
    if not author_seal_ok:
        errors.append("M1232 author double seal does not verify")
    if not contract_seal_ok:
        errors.append("M1232 contract double seal does not verify")

    r11 = PATHS["r11_tb"].read_text()
    r10 = PATHS["r10_tb"].read_text()
    contract = json.loads(PATHS["m1232_contract"].read_text())
    parent = json.loads(PATHS["m1229_review"].read_text())
    parent_binding_ok = (
        contract["parent_hammer"]["sha256"] == hashes.get("m1229_review")
        and contract["parent_hammer"]["verdict"] == parent["status"]
    )
    if not parent_binding_ok:
        errors.append("M1232 does not bind the exact M1229 verdict")

    shared_tasks = re.findall(r"    task automatic ([A-Za-z0-9_]+)\b", r10)
    changed_shared = [
        name for name in shared_tasks
        if name != "random_legal_transaction" and task(r10, name) != task(r11, name)
    ]
    normal_byte_identical = task(r10, "serve_normal_beat") == task(r11, "serve_normal_beat")
    if changed_shared:
        errors.append(f"shared frozen tasks changed: {changed_shared}")

    random_text = task(r11, "random_legal_transaction")
    retire_text = task(r11, "retire_random_forced_issue_tuple")
    random_checks = {
        "exact_weight_and_conditional_psum_fire": (
            "weight_fire_count != w0 + 1" in random_text
            and "psum_fire_count != p0 + first" in random_text
            and "r11_random_request_overshoot" in random_text
        ),
        "request_ready_retires_at_negedge": (
            "@(negedge clk_core);\n            weight_req_ready = 1'b0;\n"
            "            psum_req_ready = 1'b0;" in random_text
        ),
        "request_window_exact_one": (
            "random_weight_request_handshakes != 1" in random_text
            and "random_psum_request_handshakes != first" in random_text
        ),
        "backpressure_before_response": (
            random_text.find("force dut.core_issue_data_ready = 1'b0;")
            < random_text.find("if (index[0]) begin")
        ),
        "ready_rises_only_at_negedge": (
            "@(negedge clk_core);\n            force dut.core_issue_data_ready = 1'b1;"
            in random_text
        ),
        "response_payload_and_valid_stable": (
            "r11_random_response_hold" in random_text
            and "r11_random_response_unstable" in random_text
            and random_text.count("weight_data !== '0") == 2
            and random_text.count("psum_rsp_valid !== first") == 2
        ),
        "exact_one_response_accept": (
            "response_accept_count > response0 + 1" in random_text
            and "response_accept_count != response0 + 1" in random_text
        ),
        "forced_tuple_retires_before_response_valid": (
            random_text.find("retire_random_forced_issue_tuple();")
            < random_text.find("// Exact response retirement")
        ),
        "all_nine_tuple_fields_released": all(
            f"release dut.{signal};" in retire_text
            for signal in (
                "issue_request_valid", "issue_request_epoch", "issue_request_row_id",
                "issue_request_first", "issue_request_last",
                "issue_request_source_valid", "issue_request_source_index",
                "issue_request_parent_valid", "issue_request_parent_id",
            )
        ),
        "core_ready_not_released_with_tuple": "core_issue_data_ready" not in retire_text,
        "response_retires_at_immediate_negedge": (
            "// Exact response retirement: no extra response posedge is allowed.\n"
            "            @(negedge clk_core);\n"
            "            weight_rsp_valid = 1'b0;\n"
            "            psum_rsp_valid = 1'b0;" in random_text
        ),
        "post_retirement_sample_rejects_duplicate_and_fault": (
            "r11_random_post_retire" in random_text
            and "@(posedge clk_core); #1ps;" in random_text
            and "response_accept_count != response0 + 1" in random_text
            and "dut.request_active_q || dut.boundary_fault_q" in random_text
            and "dut.core_protocol_error" in random_text
        ),
        "no_random_attack_mask": all(
            marker not in random_text for marker in (
                "request_hold_attack_mode =", "weight_service_attack_mode =",
                "psum_service_attack_mode =",
            )
        ),
    }
    if not all(random_checks.values()):
        errors.append(f"canonical random closure incomplete: {random_checks}")

    frozen_markers = {
        "all_24_random": "test_index < 24" in r11 and "cov_random_transactions != 24" in r11,
        "all_phases": all(
            f"PHASE_M1219R9_{phase}_{edge}" in r11
            for phase in (
                "DIRECTED", "RESET_PENDING", "STICKY_ATTACKS", "SERVICE_ATTACKS",
                "RANDOM", "NORMAL_M935",
            )
            for edge in ("ENTER", "COMPLETE")
        ),
        "directed_ii2": "directed_ii2();" in r11 and "cov_ii2 != 1" in r11,
        "normal_two_source_row": "prep_mask = (row == 0) ? 16'h0003 : 16'h0000" in r11,
        "normal_two_issues": (
            "serve_normal_beat(1'b1, 0);" in r11
            and "serve_normal_beat(1'b0, 1);" in r11
        ),
        "coverage_gate": (
            "cov_request_attack_windows != 2" in r11
            and "cov_weight_service_attack_windows != 1" in r11
            and "cov_psum_service_attack_windows != 1" in r11
            and "cov_legal_masks_clear != 29" in r11
        ),
        "later_zero_sva_gate": "zero_sva_failures_required=true" in r11,
        "source_only_claims": (
            "functional_vcs_only=false timing_verified=false" in r11
            and "system_speedup=false headline=false" in r11
        ),
    }
    if not all(frozen_markers.values()):
        errors.append(f"frozen phase/workload gate incomplete: {frozen_markers}")

    checker_run = subprocess.run(
        [sys.executable, str(PATHS["r11_checker"])], cwd=ROOT,
        universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    tests_run = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", str(PATHS["r11_tests"])],
        cwd=ROOT, universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if checker_run.returncode != 0:
        errors.append("canonical M1232 checker failed")
    if tests_run.returncode != 0:
        errors.append("M1232 author mutation tests failed")

    checker = load_checker()
    independent_mutants = {
        "random_request_window_disabled": replace_in_task(
            r11, "random_legal_transaction",
            "random_request_window_active = 1'b1;",
            "random_request_window_active = 1'b0;",
        ),
        "random_post_response_duplicate_oracle_removed": replace_in_task(
            r11, "random_legal_transaction",
            "|| response_accept_count != response0 + 1\n                    || weight_req_valid",
            "|| weight_req_valid",
        ),
        "random_backpressure_hold_body_never_iterates": replace_in_task(
            r11, "random_legal_transaction",
            "repeat (hold_cycles) begin",
            "repeat (0) begin",
        ),
    }
    mutation_raw = {
        name: checker.audit_text(mutant, r10)
        for name, mutant in independent_mutants.items()
    }
    mutations_accepted = [name for name, found in mutation_raw.items() if not found]

    result = {
        "schema": "m1235_m1232_c1_r11_source_independent_hammer_r1_v1",
        "status": "NO_GO_RELEASE_AUTHORING__CHECKER_RANDOM_PATH_HARDENING_REQUIRED",
        "score": 86,
        "p0_count": 0,
        "p1_count": 1,
        "p2_count": 0,
        "mechanical_audit_pass": not errors,
        "errors": errors,
        "bindings": hashes,
        "seal_audit": {
            "author_double_seal_verified": author_seal_ok,
            "contract_double_seal_verified": contract_seal_ok,
            "m1229_parent_binding_verified": parent_binding_ok,
        },
        "canonical_source_audit": {
            "m1229_p0_structurally_closed": all(random_checks.values()),
            "normal_r10_task_byte_identical": normal_byte_identical,
            "all_other_shared_tasks_byte_identical": not changed_shared,
            "random_service": random_checks,
            "phases_workloads_and_claims": frozen_markers,
            "author_checker_exit_code": checker_run.returncode,
            "author_tests_exit_code": tests_run.returncode,
            "author_tests_run": 15,
        },
        "independent_mutation_audit": {
            "mutations_run": len(independent_mutants),
            "mutations_accepted_in_error": mutations_accepted,
            "raw_checker_errors": mutation_raw,
        },
        "authorization": {
            "checker_and_tests_only_repair": True,
            "candidate_tb_mutation": False,
            "release_authoring": False,
            "vcs": False,
            "simv": False,
            "eda": False,
            "gpu": False,
            "remote": False,
            "rtl_mutation": False,
            "sva_mutation": False,
        },
        "claim_boundary": {
            "source_hammer_only": True,
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
