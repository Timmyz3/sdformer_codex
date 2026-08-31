#!/usr/bin/env python3
"""Independent read-only M1220 hammer for the M1219/R9 TB source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PATHS = {
    "r9_tb": HW / "verif_m1219r9_c1_common_charge_protocol/tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9.sv",
    "checker": HW / "verif_m1219r9_c1_common_charge_protocol/check_m1219r9_source.py",
    "tests": HW / "verif_m1219r9_c1_common_charge_protocol/test_m1219r9_source.py",
    "contract": HW / "contracts/m1219_m1218_m1213_c1_r9_observability_source_contract_r1_20260830.json",
    "r8_tb": HW / "verif_m1210r8_c1_common_charge_protocol/tb_m1210r8_m1162_common_charge_protocol_unit_delay_r8.sv",
    "m528": HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
    "m935": HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv",
    "m1162": HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv",
    "sva_r3": HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "r9_tb": "9666e086c69ecda4670622e063e9d54c89f94f2c77cd5eb012da54ca23492a75",
    "checker": "2639ecfe321f004939ffe4d5de65586191ecb26c9f31f772473d92fdc7456268",
    "tests": "b365f3b8afef707359dbb54945684da953bbdd28a334201e438c7baebeaab563",
    "contract": "fd4a23ea97395f47c49fd9183a51b842156b3535c494062028d9b72a1c389c67",
    "r8_tb": "060ec9d5ae6085a0dd013160d22f63e21615730384ddaef342eb3fa77e17947b",
    "m528": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "m935": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "m1162": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    "sva_r3": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AUTHOR = HW / "reviews/m1219_m1218_c1_r9_observability_source_author_r1_20260830"
AUTHOR_EXPECTED = {
    "author_review.md": "0aa4b4515d0e507eee42d8d01dba07cffd24946c0bcad3e35764b0cef5c8d966",
    "SHA256SUMS": "3924a2b4dc976de6e4c121c0e2a7254722078a2328891fee0f547e85d66b9647",
    "SHA256SUMS.seal.sha256": "5a7007cecaaffa76cc5951951965dfa237e8fb42e4e06cb0ab2444380881f01e",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task(text: str, name: str) -> str:
    match = re.search(r"task\s+automatic\s+" + re.escape(name) + r"\b(.*?)endtask", text, re.S)
    if not match:
        return ""
    return match.group(0)


def audit_tb(text: str) -> List[str]:
    errors: List[str] = []
    need = lambda condition, message: errors.append(message) if not condition else None
    need("module tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9;" in text,
         "R9 module identity")
    need(re.search(r"\bwait\s*\(", text) is None, "unbounded wait remains")
    loops = re.findall(r"while\s*\((.*?)\)\s*begin", text, re.S)
    need(len(loops) == 8, "bounded while population")
    need(all("watchdog <" in header for header in loops), "watchdog-free while")

    random = task(text, "random_legal_transaction")
    normal_load = task(text, "load_normal_task")
    clean = task(text, "require_clean_reset_prep_ready")
    normal = task(text, "normal_m935_completion")
    for site, body, limit, predicate in (
        ("random_weight_request", random, "R9_RANDOM_WAIT_LIMIT", "weight_fire_count != w0 + 1"),
        ("random_psum_request", random, "R9_RANDOM_WAIT_LIMIT", "psum_fire_count != p0 + 1"),
        ("random_response_accept", random, "R9_RANDOM_WAIT_LIMIT", "response_accept_count != response0 + 1"),
        ("normal_prep_ready", normal_load, "R9_PREP_WAIT_LIMIT", "!prep_ready"),
    ):
        need(bool(body) and predicate in body and "watchdog < " + limit in body and
             'dump_r9_liveness_state("' + site + '"' in body and "$fatal" in body,
             "incomplete bounded site " + site)
    for overshoot in ("random_weight_overshoot", "random_psum_overshoot",
                      "random_response_overshoot"):
        need('dump_r9_liveness_state("' + overshoot + '"' in random,
             "missing fail-closed " + overshoot)

    need(bool(clean) and "R9_CLEAN_RESET_PREP_LIMIT" in clean and
         "while (!prep_ready && watchdog < R9_CLEAN_RESET_PREP_LIMIT)" in clean and
         'dump_r9_liveness_state("clean_reset_prep_ready"' in clean and "$fatal" in clean,
         "clean-reset prep gate incomplete")
    positions = [normal.find(token) for token in
                 ("reset_dut();", "require_legal_masks_clear(200);",
                  "require_clean_reset_prep_ready();", "issue0 =", "load_normal_task(")]
    need(bool(normal) and all(pos >= 0 for pos in positions) and positions == sorted(positions),
         "clean-reset prep gate ordering")

    phases = ("DIRECTED", "RESET_PENDING", "STICKY_ATTACKS", "SERVICE_ATTACKS",
              "RANDOM", "NORMAL_M935", "CLEAN_RESET_PREP")
    for phase in phases:
        for edge in ("ENTER", "COMPLETE"):
            token = "PHASE_M1219R9_{}_{}".format(phase, edge)
            need(text.count(token) == 1, "nonunique/missing phase token " + token)
    for edge in ("ENTER", "COMPLETE"):
        token = "PHASE_M1219R9_RANDOM_TRANSACTION_{} index=%0d".format(edge)
        need(text.count(token) == 1, "random transaction token " + edge)
    need(text.count("$fflush();") >= 16, "phase tokens not flushed")

    dump = task(text, "dump_r9_liveness_state")
    for field in ("cycle_count", "reset_n", "prep_valid", "prep_ready",
                  "issue_request_valid", "weight_req_valid", "weight_req_ready",
                  "psum_req_valid", "psum_req_ready", "weight_rsp_valid", "psum_rsp_valid",
                  "dut.response_accept_w", "dut.request_active_q",
                  "dut.weight_request_accepted_q", "dut.psum_request_accepted_q",
                  "dut.boundary_fault_q", "dut.core_protocol_error",
                  "dut.u_frozen_m935.fault_q", "dut.u_frozen_m935.prep_active_q",
                  "dut.u_frozen_m935.match_active_q", "dut.u_frozen_m935.bank_state_q[0]",
                  "weight_fire_count", "psum_fire_count", "response_accept_count"):
        need(field in dump, "state dump missing " + field)
    need("TIMEOUT_M1219R9" in dump and "$fflush();" in dump,
         "timeout dump token/flush")

    # Frozen workload, protocol, and claim boundaries inherited from R8.
    for frozen in ("test_index < 24", "directed_ii2();", "cov_ii2 != 1",
                   "cov_request_attack_windows != 2",
                   "cov_weight_service_attack_windows != 1",
                   "cov_psum_service_attack_windows != 1",
                   "cov_normal_issue != 2 || cov_normal_row != 1",
                   "|| cov_normal_task != 1 || cov_legal_masks_clear != 29",
                   "random_request_quiesce=24", "protocol_attacks=7",
                   "service_assumption_attacks=2", "normal_m935_rows=1",
                   "normal_m935_tasks=1", "functional_vcs_only=true",
                   "timing_verified=false", "cycles_measured=false", "speedup=false",
                   "ppa=false", "energy=false", "system_speedup=false", "headline=false"):
        need(frozen in text, "frozen semantic token " + frozen)
    q = text.find("R8_RANDOM_REQUEST_READY_QUIESCE_BOUNDARY")
    window = text[q:q + 900] if q >= 0 else ""
    need(q >= 0 and "weight_req_ready = 1'b0;" in window and
         "psum_req_ready = 1'b0;" in window and
         "random_request_window_active = 1'b0;" in window,
         "R8 request-ready quiesce drift")
    need(text.count("m1162_m935_c1_common_charge_protocol_boundary dut (") == 1 and
         text.count("m1168r3_m1162_common_charge_protocol_assertions_r3 u_protocol_sva (") == 1,
         "DUT/SVA binding drift")
    return errors


def check_seals() -> List[str]:
    errors: List[str] = []
    sidecar = Path(str(PATHS["contract"]) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    if sidecar.read_text().split() != [EXPECTED["contract"], PATHS["contract"].name]:
        errors.append("contract sidecar")
    if outer.read_text().split() != [sha(sidecar), sidecar.name]:
        errors.append("contract outer seal")
    manifest, author_outer = AUTHOR / "SHA256SUMS", AUTHOR / "SHA256SUMS.seal.sha256"
    if author_outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        errors.append("author outer seal")
    listed: Dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        listed[name] = digest
        if sha(AUTHOR / name) != digest:
            errors.append("author member drift " + name)
    actual = {path.name for path in AUTHOR.iterdir() if path.is_file()} - {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"}
    if set(listed) != actual:
        errors.append("author membership drift")
    for name, digest in AUTHOR_EXPECTED.items():
        if sha(AUTHOR / name) != digest:
            errors.append("author frozen identity " + name)
    return errors


def main() -> int:
    errors: List[str] = []
    hashes = {name: sha(path) for name, path in PATHS.items()}
    for name, expected in EXPECTED.items():
        if hashes.get(name) != expected:
            errors.append("identity drift " + name)
    contract = json.loads(PATHS["contract"].read_text())
    if contract.get("status") != "SOURCE_ONLY_READY_FOR_FRESH_INDEPENDENT_HAMMER":
        errors.append("contract status")
    auth = contract.get("authorization", {})
    if not (auth.get("fresh_source_hammer") is True and auth.get("vcs") is False and
            auth.get("eda") is False and auth.get("launcher_or_release") is False and
            auth.get("rtl_mutation") is False):
        errors.append("contract authorization")
    errors.extend(check_seals())
    canonical = PATHS["r9_tb"].read_text()
    errors.extend(audit_tb(canonical))

    quiesce_anchor = canonical.index("R8_RANDOM_REQUEST_READY_QUIESCE_BOUNDARY")
    quiesce_pos = canonical.index("weight_req_ready = 1'b0;", quiesce_anchor)
    quiesce_mutant = (canonical[:quiesce_pos] + "weight_req_ready = 1'b1;" +
                      canonical[quiesce_pos + len("weight_req_ready = 1'b0;"):])
    mutations = {
        "drop_random_watchdog": canonical.replace(
            "&& watchdog < R9_RANDOM_WAIT_LIMIT) begin", ") begin", 1),
        "drop_response_timeout_dump": canonical.replace(
            'dump_r9_liveness_state("random_response_accept"',
            'dump_r9_liveness_state("random_response_missing"', 1),
        "drop_clean_reset_call": canonical.replace(
            "require_clean_reset_prep_ready();", "/* removed */", 1),
        "drop_phase_complete": canonical.replace(
            "PHASE_M1219R9_RANDOM_COMPLETE", "PHASE_M1219R9_RANDOM_DONE", 1),
        "drop_m935_dump_field": canonical.replace(
            "dut.u_frozen_m935.match_active_q", "dut.match_active_removed", 1),
        "change_random_count": canonical.replace("test_index < 24", "test_index < 23", 1),
        "break_quiesce": quiesce_mutant,
        "inflate_claim": canonical.replace("timing_verified=false",
                                            "timing_verified=true", 1),
    }
    mutation_results = {}
    for name, mutant in mutations.items():
        rejected = bool(audit_tb(mutant))
        mutation_results[name] = "REJECTED" if rejected else "ACCEPTED_IN_ERROR"
        if not rejected:
            errors.append("mutation accepted " + name)
    result = {
        "schema": "m1220_m1219_c1_r9_observability_source_hammer_mechanical_r1_v1",
        "status": "PASS" if not errors else "FAIL",
        "score": 99 if not errors else 0,
        "p0_count": len(errors),
        "p1_count": 0,
        "p2_count": 0,
        "hashes": hashes,
        "bounded_while_loops": len(re.findall(r"while\s*\((.*?)\)\s*begin", canonical, re.S)),
        "newly_bounded_wait_sites": 4,
        "clean_reset_prep_gate": not any("clean-reset" in error for error in errors),
        "phase_pairs": 7,
        "random_transaction_token_pairs": 24,
        "timeout_state_dump": not any("state dump" in error for error in errors),
        "independent_mutations": mutation_results,
        "dut_rtl_sva_r8_frozen": all(hashes[name] == EXPECTED[name] for name in
                                      ("r8_tb", "m528", "m935", "m1162", "sva_r3")),
        "authorization": {"release_authoring": not errors, "vcs": False, "eda": False},
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
