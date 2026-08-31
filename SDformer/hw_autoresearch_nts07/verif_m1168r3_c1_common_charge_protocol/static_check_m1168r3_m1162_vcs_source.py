#!/usr/bin/env python3
"""Static fail-closed check of the additive M1168R3 package; no EDA."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from pathlib import Path


HW = Path(__file__).resolve().parents[1]
TB = HW / "verif_m1168r3_c1_common_charge_protocol/tb_m1168r3_m1162_common_charge_protocol_unit_delay_r3.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
R2_ATTEMPT = HW / "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed/identity.txt"
R2_Q = HW / "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830.failed_or_incomplete.3284331.quarantine"

EXPECTED = {
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    R2_ATTEMPT: "dde2eca905affe76a5e5a74966fe2502bc1fb82364493ee314819264d8bd75ca",
    R2_Q / "compile.log": "1ee670031e192b1d2e894183e851cf6b65dd86319ba1224d76ea938dbf979de4",
    R2_Q / "sim.log": "fbcc88d9893be34d3aa5bbf3cb49936cc4c1f5f24d0eab1eb797e3039bd657c3",
    R2_Q / "SHA256SUMS": "f3926823e62535facb13a369d78f0d13489be90494ac2cb1ea192885e412ecb9",
    R2_Q / "SHA256SUMS.seal.sha256": "c147c7e8a6ff7d523aa96f159e715967054ce0d4d19f699f7f8ed4daef8f9989",
}

checks = 0
mutations = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recursive(directory: Path, outer_expected: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "quarantine absent")
    require(sha(directory / "SHA256SUMS.seal.sha256") == outer_expected, "outer identity")
    require((directory / "SHA256SUMS.seal.sha256").read_text().split() ==
            [sha(directory / "SHA256SUMS"), "SHA256SUMS"], "outer content")
    listed = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "unsafe manifest")
        listed[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        rel = member.relative_to(directory).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        # VCS emits tool-internal symlinks.  The frozen runner's recursive seal
        # intentionally covers every regular evidence byte and ignores those
        # generated links; reproduce that exact policy here.
        if stat.S_ISREG(mode):
            actual.add(rel)
    require(actual == set(listed), "quarantine membership")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, "quarantine drift: " + name)


def validate(tb: str, sva: str) -> None:
    require("module tb_m1168r3_m1162_common_charge_protocol_unit_delay_r3" in tb, "R3 TB")
    require("m1168r3_m1162_common_charge_protocol_assertions_r3 u_protocol_sva" in tb, "R3 SVA instance")
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
    require(sva.count("|| request_hold_attack_mode") == 2, "request mask scope")
    require(sva.count("!weight_service_attack_mode") == 1, "weight mask scope")
    require(sva.count("!psum_service_attack_mode") == 1, "psum mask scope")
    require("disable iff (!reset_n || weight_service_attack_mode" not in sva, "broad weight disable")
    require("disable iff (!reset_n || psum_service_attack_mode" not in sva, "broad psum disable")
    require(tb.count("request_hold_attack_mode = 1'b1;") == 2, "two request windows")
    require(tb.count("weight_service_attack_mode = 1'b1;") == 1, "one weight window")
    require(tb.count("psum_service_attack_mode = 1'b1;") == 1, "one psum window")
    require("cov_legal_masks_clear != 29" in tb and
            "require_legal_masks_clear(100 + index)" in tb, "legal mask regression")
    require("cov_request_attack_windows != 2" in tb, "request window coverage")
    require("cov_weight_service_attack_windows != 1" in tb and
            "cov_psum_service_attack_windows != 1" in tb, "service window coverage")
    require("@(posedge clk_core);\n            @(negedge clk_core);\n            if (!weight_service_fault" in tb,
            "race-free weight sample")
    require("@(posedge clk_core);\n            @(negedge clk_core);\n            if (!psum_service_fault" in tb,
            "race-free psum sample")
    require("protocol_error)" in tb and "dut_fault_claim=0" in tb, "classification boundary")
    require("directed_random=24" in tb and "protocol_attacks=7" in tb and
            "service_assumption_attacks=2" in tb and "ii=2" in tb and
            "normal_m935_rows=1" in tb and "normal_m935_tasks=1" in tb, "regression minima")
    require("PASS_M1168R3_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE" in tb, "R3 pass token")
    for token in ("functional_vcs_only=true", "timing_verified=false", "cycles_measured=false",
                  "speedup=false", "ppa=false", "energy=false", "system_speedup=false",
                  "headline=false"):
        require(token in tb, "claim boundary: " + token)
    force_body = re.search(r"task automatic force_request\((.*?)endtask", tb, re.S)
    require(force_body is not None and force_body.group(1).count("force dut.") == 10, "force staging")
    for formal in ("epoch", "row", "first", "last", "source"):
        require(re.search(r"force\s+dut\.[^;]+?=\s*" + formal + r"\s*;", force_body.group(1)) is None,
                "automatic force RHS")


def reject_mutation(tb: str, sva: str, label: str, target: str, old: str, new: str) -> None:
    global mutations
    original = tb if target == "tb" else sva
    require(old in original, "mutation anchor: " + label)
    changed = original.replace(old, new, 1)
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
    recursive(R2_Q, EXPECTED[R2_Q / "SHA256SUMS.seal.sha256"])
    compile_log = (R2_Q / "compile.log").read_text(errors="replace")
    sim_log = (R2_Q / "sim.log").read_text(errors="replace")
    require("CPU time:" in compile_log and "to compile +" in compile_log
            and "to elab +" in compile_log and "to link" in compile_log, "R2 compile/elab/link proof")
    require("Error-[" not in compile_log, "R2 compile error")
    require("ap_psum_request_hold: started at 283500ps failed at 286500ps" in sim_log,
            "R2 assertion forensic")
    require("weight service mutation boundary was misclassified" in sim_log, "R2 service forensic")
    require("PASS_M1168R2" not in sim_log, "R2 falsely passed")

    tb = TB.read_text()
    sva = SVA.read_text()
    validate(tb, sva)
    lines = [x.strip() for x in FILELIST.read_text().splitlines() if x.strip()]
    require(len(lines) == 6 and len(set(lines)) == 6, "filelist cardinality")
    for item in lines:
        require(Path(item).is_file(), "missing filelist member: " + item)

    for args in (
        ("tb", "legal_mask_gate", "cov_legal_masks_clear != 29", "cov_legal_masks_clear != 0"),
        ("tb", "request_window_count", "cov_request_attack_windows != 2", "cov_request_attack_windows != 0"),
        ("tb", "weight_window_removed", "weight_service_attack_mode = 1'b1;", "weight_service_attack_mode = 1'b0;"),
        ("tb", "psum_window_removed", "psum_service_attack_mode = 1'b1;", "psum_service_attack_mode = 1'b0;"),
        ("tb", "weight_race", "@(negedge clk_core);\n            if (!weight_service_fault", "#1ps;\n            if (!weight_service_fault"),
        ("tb", "psum_race", "@(negedge clk_core);\n            if (!psum_service_fault", "#1ps;\n            if (!psum_service_fault"),
        ("tb", "normal_m935_removed", "normal_m935_tasks=1", "normal_m935_tasks=0"),
        ("sva", "request_mask_broad", "|| request_hold_attack_mode", "|| 1'b1"),
        ("sva", "weight_mask_removed", "!weight_service_attack_mode", "1'b1"),
        ("sva", "psum_mask_removed", "!psum_service_attack_mode", "1'b1"),
        ("sva", "assert_removed", "ap_psum_request_hold: assert property", "ap_psum_request_hold: assume property"),
        ("sva", "cover_removed", "cp_ii2: cover property", "cp_ii2: assert property"),
    ):
        reject_mutation(tb, sva, args[1], args[0], args[2], args[3])

    print(json.dumps({
        "schema": "m1168r3_m1162_vcs_source_static_check_r3_v1",
        "status": "PASS_SOURCE_ONLY_FORENSICS_AND_R3_NEGATIVE_TEST_ISOLATION__FRESH_HAMMER_REQUIRED__NO_EDA",
        "checks_passed": checks,
        "mutations_rejected": mutations,
        "r2_compile_elab_link_passed": True,
        "r2_sim_failed": True,
        "r2_failure_class": "NEGATIVE_TEST_ISOLATION_AND_SERVICE_MONITOR_SAMPLING_BOUNDARY",
        "rtl_fault_proven": False,
        "assertions_preserved": 16,
        "covers_preserved": 6,
        "legal_mask_clear_regressions": 29,
        "protocol_attacks": 7,
        "service_assumption_attacks": 2,
        "deterministic_random_transactions": 24,
        "normal_m935_rows": 1,
        "normal_m935_tasks": 1,
        "r2_attempt_reusable": False,
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "source_sha256": {str(p.relative_to(HW)): sha(p) for p in (TB, SVA, FILELIST)},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
