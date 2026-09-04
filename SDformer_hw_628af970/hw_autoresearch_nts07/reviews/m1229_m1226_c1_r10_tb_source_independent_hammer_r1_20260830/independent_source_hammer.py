#!/usr/bin/env python3
"""Read-only independent source hammer for M1226/R10.

The audit deliberately does not compile or launch VCS.  It binds the exact
source/seal identities, checks that the inherited random workload is unchanged,
recounts the already-sealed R9 SVA failures, and probes the M1226 checker for
random-service semantic blind spots.
"""

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import List


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PATHS = {
    "r10_tb": ROOT / "verif_m1226r10_c1_common_charge_protocol/tb_m1226r10_m1162_common_charge_protocol_unit_delay_r10.sv",
    "r10_checker": ROOT / "verif_m1226r10_c1_common_charge_protocol/check_m1226r10_source.py",
    "r10_tests": ROOT / "verif_m1226r10_c1_common_charge_protocol/test_m1226r10_source.py",
    "r9_tb": ROOT / "verif_m1219r9_c1_common_charge_protocol/tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9.sv",
    "m1225_review": ROOT / "reviews/m1225_m1221_c1_r9_vcs_failure_forensic_r1_20260830/review.json",
    "m1226_contract": ROOT / "contracts/m1226_m1225_m1221_c1_r10_tb_service_boundary_repair_source_contract_r1_20260830.json",
    "m528": ROOT / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
    "m935": ROOT / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv",
    "m1162": ROOT / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv",
    "sva_r3": ROOT / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv",
    "docs359": ROOT / "docs/359_DATE终局冻结_20260813.md",
    "r9_sim_log": ROOT / "results/m1221_m1219r9_m1162_c1_common_charge_protocol_unit_delay_vcs_r9_20260830.failed_or_incomplete.983909.quarantine/sim.log",
}

EXPECTED = {
    "r10_tb": "f2df09cf6177f1dcb48e7eae24bedfe914a9222d417eee9d08a11d0a1d89c14b",
    "r10_checker": "708703b01babf9bcfc9915e72874d2167f2ef7f45cac3d4276ab8d541bfaf0e2",
    "r10_tests": "bb351955023c0bfcd273a8c48c2833090b2ca521ef1aeaaf9e39ae5a0279c535",
    "r9_tb": "9666e086c69ecda4670622e063e9d54c89f94f2c77cd5eb012da54ca23492a75",
    "m1225_review": "5cd5c859b0069348456a23037cf3b4495a06f6305a45f5bcd4fb74912cf3d668",
    "m528": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "m935": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "m1162": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    "sva_r3": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "r9_sim_log": "90e44f850115fe81a22ef5224b1544c5d8150cf43ae47fafa0687f33bdb756a7",
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
    spec = importlib.util.spec_from_file_location("m1226_checker", PATHS["r10_checker"])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_in_task(text: str, task_name: str, old: str, new: str) -> str:
    original = task(text, task_name)
    if old not in original:
        raise AssertionError(f"mutation anchor absent in {task_name}")
    mutated = original.replace(old, new, 1)
    return text.replace(original, mutated, 1)


def main() -> int:
    errors: List[str] = []
    hashes = {name: sha(path) for name, path in PATHS.items()}
    for name, expected in EXPECTED.items():
        if hashes[name] != expected:
            errors.append(f"{name} SHA mismatch: {hashes[name]}")

    m1225 = json.loads(PATHS["m1225_review"].read_text())
    contract = json.loads(PATHS["m1226_contract"].read_text())
    if contract["parent_forensic"]["sha256"] != hashes["m1225_review"]:
        errors.append("M1226 parent-forensic hash is not bound to M1225")
    if contract["parent_forensic"]["verdict"] != m1225["verdict"]:
        errors.append("M1226 parent-forensic verdict differs from M1225")

    r9_text = PATHS["r9_tb"].read_text()
    r10_text = PATHS["r10_tb"].read_text()
    random_equal = task(r9_text, "random_legal_transaction") == task(
        r10_text, "random_legal_transaction"
    )
    if not random_equal:
        errors.append("R9 and R10 random_legal_transaction are not byte-identical")

    sim_lines = PATHS["r9_sim_log"].read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(sim_lines) if "PHASE_M1219R9_RANDOM_ENTER" in line)
    stop = next(i for i, line in enumerate(sim_lines) if "PHASE_M1219R9_RANDOM_COMPLETE" in line)
    random_lines = sim_lines[start : stop + 1]
    random_failures = {
        "ap_weight_request_hold": sum(".ap_weight_request_hold: started" in line for line in random_lines),
        "ap_weight_response_hold": sum(".ap_weight_response_hold: started" in line for line in random_lines),
        "ap_psum_response_hold": sum(".ap_psum_response_hold: started" in line for line in random_lines),
    }
    random_failure_total = sum(random_failures.values())
    if random_failures != {
        "ap_weight_request_hold": 11,
        "ap_weight_response_hold": 11,
        "ap_psum_response_hold": 0,
    }:
        errors.append(f"unexpected sealed random failure profile {random_failures}")

    checker = load_checker()
    checker_canonical_errors = checker.audit_text(r10_text)
    if checker_canonical_errors:
        errors.append(f"M1226 checker rejects canonical source: {checker_canonical_errors}")

    # The independent mutations change only inherited random legal service.
    # A release-quality checker with an all-workload/SVA=0 claim must reject
    # both; the M1226 normal-only checker currently accepts them.
    ready_mutant = replace_in_task(
        r10_text,
        "random_legal_transaction",
        "weight_req_ready = 1'b0;\n            psum_req_ready = 1'b0;\n            random_request_window_active = 1'b0;",
        "weight_req_ready = 1'b1;\n            psum_req_ready = 1'b1;\n            random_request_window_active = 1'b0;",
    )
    response_mutant = replace_in_task(
        r10_text,
        "random_legal_transaction",
        "            @(negedge clk_core);\n            weight_rsp_valid = 1'b0;\n            psum_rsp_valid = 1'b0;",
        "            @(posedge clk_core); #1ps;\n            @(negedge clk_core);\n            weight_rsp_valid = 1'b0;\n            psum_rsp_valid = 1'b0;",
    )
    mutation_results = {
        "random_request_ready_not_retired": checker.audit_text(ready_mutant),
        "random_extra_response_posedge": checker.audit_text(response_mutant),
    }
    checker_random_blind_spots = [
        name for name, audit_errors in mutation_results.items() if not audit_errors
    ]
    if len(checker_random_blind_spots) != 2:
        errors.append(
            "expected both independent random-service mutations to expose checker blind spots"
        )

    # The normal repair itself has the requested exact-retirement anchors.
    normal_text = task(r10_text, "serve_normal_beat")
    normal_gate_checks = {
        "request_single_fire": "normal_request_overshoot" in normal_text,
        "request_ready_retire": "normal_request_retire" in normal_text,
        "response_exact_accept": "normal_response_overshoot" in normal_text,
        "response_immediate_negedge_retire": (
            "// Exact-accept retirement: no extra posedge is permitted here.\n"
            "            @(negedge clk_core);" in normal_text
        ),
        "beat_boundary": "normal_wrapper_retire" in normal_text,
        "state_dump": "dump_r9_liveness_state" in normal_text,
    }
    if not all(normal_gate_checks.values()):
        errors.append(f"normal repair gate incomplete: {normal_gate_checks}")

    result = {
        "schema": "m1229_m1226_c1_r10_tb_source_independent_hammer_r1_v1",
        "status": "NO_GO_RELEASE_AUTHORING__ADDITIVE_RANDOM_SERVICE_REPAIR_REQUIRED",
        "score": 58,
        "p0_count": 1,
        "p1_count": 1,
        "p2_count": 0,
        "mechanical_audit_pass": not errors,
        "errors": errors,
        "bindings": hashes,
        "m1225_binding_verified": True,
        "author_seals_verified_externally": True,
        "normal_repair": normal_gate_checks,
        "random_regression": {
            "r9_r10_task_byte_identical": random_equal,
            "sealed_r9_random_phase_sva_failures": random_failures,
            "sealed_r9_random_phase_sva_failure_total": random_failure_total,
            "zero_unmasked_sva_execution_gate_not_yet_plausibly_satisfied": True,
        },
        "checker_mutation_audit": {
            "canonical_errors": checker_canonical_errors,
            "mutations_accepted_in_error": checker_random_blind_spots,
            "raw_results": mutation_results,
        },
        "authorization": {
            "release_authoring": False,
            "vcs": False,
            "simv": False,
            "eda": False,
            "gpu": False,
            "remote": False,
            "rtl_mutation": False,
            "additive_tb_source_repair": True,
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
