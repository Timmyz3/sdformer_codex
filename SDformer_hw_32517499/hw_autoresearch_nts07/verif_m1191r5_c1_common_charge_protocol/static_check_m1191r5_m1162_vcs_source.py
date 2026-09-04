#!/usr/bin/env python3
"""Fail-closed source-only forensic/checker for M1191/R5; invokes no EDA."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path


HW = Path(__file__).resolve().parents[1]
TB = HW / "verif_m1191r5_c1_common_charge_protocol/tb_m1191r5_m1162_common_charge_protocol_unit_delay_r5.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1191r5_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
R4_ATTEMPT = HW / "results/.m1187_m1168r3_m1162_c1_common_charge_protocol_vcs_r4_attempt_consumed/identity.txt"
R4_Q = HW / "results/m1187_m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r4_20260830.failed_or_incomplete.3580131.quarantine"

EXPECTED = {
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    R4_ATTEMPT: "617b1a1521cf0c0c4e9ffbb240290bcc4a942da785cf3b9e58333803565820a8",
    R4_Q / "sim.log": "bbb1a50e96ea26abae5f1978d8cfa72ca052c32bd7167c9ca40cc6c3e6fad747",
    R4_Q / "SHA256SUMS": "22e2149b3ae6ff446e72f0bd788b3cf20fcc06970525947f5769df1391ffc709",
    R4_Q / "SHA256SUMS.seal.sha256": "6974c63a969b426924088b4d4b1be398b401958c1f1855c3beb7be37a1b6d05e",
}

checks = 0
mutations = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_recursive(directory: Path) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "R4 quarantine absent")
    outer = (directory / "SHA256SUMS.seal.sha256").read_text().split()
    require(outer == [sha(directory / "SHA256SUMS"), "SHA256SUMS"], "R4 outer seal")
    listed: dict[str, str] = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "unsafe R4 manifest")
        listed[name] = digest
    actual: set[str] = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        base = Path(root)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            member = base / name
            rel = member.relative_to(directory).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or member.is_symlink():
                continue
            if stat.S_ISREG(member.lstat().st_mode):
                actual.add(rel)
    require(actual == set(listed), "R4 quarantine membership drift")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, "R4 member drift: " + name)


def task_body(text: str, name: str) -> str:
    match = re.search(r"task automatic " + re.escape(name) + r";(.*?)endtask", text, re.S)
    require(match is not None, "missing task " + name)
    return match.group(1)


def validate(tb: str, sva: str) -> None:
    require("module tb_m1191r5_m1162_common_charge_protocol_unit_delay_r5" in tb, "R5 module")
    require("m1168r3_m1162_common_charge_protocol_assertions_r3 u_protocol_sva" in tb, "frozen R3 SVA")
    require("m1168r3_service_assumption_checker u_service_checker" in tb, "independent checker")
    require(sva.count("assert property") == 16, "16 assertions")
    require(sva.count("cover property") == 6, "six covers")
    for token in (
        "ap_weight_request_hold", "ap_psum_request_hold", "ap_weight_no_reissue",
        "ap_psum_no_reissue", "ap_nonfirst_never_requests_psum",
        "ap_core_valid_requires_requests", "ap_weight_ready_is_atomic",
        "ap_psum_ready_is_first_atomic", "ap_no_lone_weight_consume",
        "ap_no_lone_psum_consume", "ap_core_backpressure_atomic",
        "ap_weight_response_hold", "ap_psum_response_hold",
        "ap_boundary_fault_sticky", "ap_reset_clears_transaction",
        "ap_no_consecutive_response_accept", "cp_weight_first", "cp_psum_first",
        "cp_nonfirst", "cp_response_skew_weight", "cp_response_skew_psum", "cp_ii2"):
        require(token in sva, "property absent: " + token)
    require(sva.count("|| request_hold_attack_mode") == 2, "request-mask scope")
    require(sva.count("!weight_service_attack_mode") == 1, "weight-mask scope")
    require(sva.count("!psum_service_attack_mode") == 1, "psum-mask scope")
    require(tb.count("request_hold_attack_mode = 1'b1;") == 2, "two request attacks")
    require(tb.count("weight_service_attack_mode = 1'b1;") == 1, "one weight attack")
    require(tb.count("psum_service_attack_mode = 1'b1;") == 1, "one psum attack")
    service = task_body(tb, "service_assumption_attacks")
    require("weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b0;" in service,
            "weight skew isolation")
    require("weight_rsp_valid = 1'b0; psum_rsp_valid = 1'b1;" in service,
            "psum skew isolation")
    require("force dut.core_issue_data_ready" not in service,
            "service attack must not force M935 ready")
    require(service.count("dut.boundary_fault_q || dut.core_protocol_error") == 2,
            "both service oracles require clean boundary/core")
    require(service.count("protocol_error") >= 4, "both service oracles require clean protocol")
    require("@(posedge clk_core);\n            @(negedge clk_core);\n            if (!weight_service_fault" in service,
            "race-free weight sample")
    require("@(posedge clk_core);\n            @(negedge clk_core);\n            if (!psum_service_fault" in service,
            "race-free psum sample")
    require("cov_legal_masks_clear != 29" in tb, "29 legal regressions")
    require("cov_request_attack_windows != 2" in tb, "two request windows")
    require("cov_weight_service_attack_windows != 1" in tb
            and "cov_psum_service_attack_windows != 1" in tb, "two service windows")
    require("directed_random=24" in tb and "protocol_attacks=7" in tb
            and "service_assumption_attacks=2" in tb and "ii=2" in tb
            and "normal_m935_rows=1" in tb and "normal_m935_tasks=1" in tb,
            "regression minima")
    require("PASS_M1191R5_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE" in tb,
            "R5 pass token")
    for token in ("service_skew_isolated=1", "boundary_fault=0", "core_fault=0",
                  "functional_vcs_only=true", "timing_verified=false",
                  "cycles_measured=false", "speedup=false", "ppa=false",
                  "energy=false", "system_speedup=false", "headline=false"):
        require(token in tb, "claim token " + token)


def reject(tb: str, sva: str, label: str, target: str, old: str, new: str) -> None:
    global mutations
    source = tb if target == "tb" else sva
    require(old in source, "mutation anchor " + label)
    changed = source.replace(old, new, 1)
    try:
        validate(changed, sva) if target == "tb" else validate(tb, changed)
    except AssertionError:
        mutations += 1
        return
    raise AssertionError("mutation accepted: " + label)


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift: " + str(path))
    verify_recursive(R4_Q)
    sim = (R4_Q / "sim.log").read_text(errors="replace")
    require("compile+sim" not in sim, "unexpected synthetic R4 marker")
    require("weight service mutation boundary misclassified weight=1 psum=0 protocol=1" in sim,
            "R4 failure forensic")
    require("at time 354000 ps" in sim and "PASS_M1168R3" not in sim, "R4 failed, not passed")
    wrapper = WRAPPER.read_text()
    m935 = M935.read_text()
    require("protocol_error = core_protocol_error || boundary_fault_q;" in wrapper,
            "protocol composition")
    require("if (issue_data_valid && !issue_request_valid)" in m935,
            "frozen M935 context-consistency fault")
    require("response_valid_and_payload_hold_until_ready" not in wrapper,
            "wrapper contains no hidden service checker")
    tb, sva = TB.read_text(), SVA.read_text()
    validate(tb, sva)
    lines = [line.strip() for line in FILELIST.read_text().splitlines() if line.strip()]
    require(len(lines) == 6 and len(set(lines)) == 6, "filelist cardinality")
    for item in lines:
        require(Path(item).is_file(), "filelist member absent: " + item)
    for args in (
        ("tb", "weight_peer_present", "weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b0;",
         "weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b1;"),
        ("tb", "psum_peer_present", "weight_rsp_valid = 1'b0; psum_rsp_valid = 1'b1;",
         "weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b1;"),
        ("tb", "clean_core_removed", "|| dut.boundary_fault_q || dut.core_protocol_error)",
         "|| dut.boundary_fault_q)"),
        ("tb", "weight_attack_removed", "weight_service_attack_mode = 1'b1;",
         "weight_service_attack_mode = 1'b0;"),
        ("tb", "psum_attack_removed", "psum_service_attack_mode = 1'b1;",
         "psum_service_attack_mode = 1'b0;"),
        ("tb", "normal_m935_removed", "normal_m935_tasks=1", "normal_m935_tasks=0"),
        ("sva", "assert_removed", "ap_psum_request_hold: assert property",
         "ap_psum_request_hold: assume property"),
        ("sva", "cover_removed", "cp_ii2: cover property", "cp_ii2: assert property"),
    ):
        reject(tb, sva, args[1], args[0], args[2], args[3])
    print(json.dumps({
        "schema": "m1191r5_m1162_vcs_source_static_check_v1",
        "status": "PASS_SOURCE_ONLY_R4_FORENSIC_AND_R5_SKEW_ISOLATED_SERVICE_ORACLE__FRESH_HAMMER_AND_RELEASE_REQUIRED__NO_EDA",
        "checks_passed": checks,
        "mutations_rejected": mutations,
        "r4_compile_elab_link_passed": True,
        "r4_sim_failed": True,
        "r4_attempt_reusable": False,
        "r4_failure_class": "SERVICE_ATTACK_HARNESS_CREATED_ARTIFICIAL_M935_CONTEXT_INCONSISTENCY",
        "rtl_modified": False,
        "assertions_preserved": 16,
        "covers_preserved": 6,
        "protocol_attacks": 7,
        "service_assumption_attacks": 2,
        "deterministic_random_transactions": 24,
        "normal_m935_rows": 1,
        "normal_m935_tasks": 1,
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "source_sha256": {str(path.relative_to(HW)): sha(path)
                          for path in (TB, SVA, FILELIST, WRAPPER, M935)},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
