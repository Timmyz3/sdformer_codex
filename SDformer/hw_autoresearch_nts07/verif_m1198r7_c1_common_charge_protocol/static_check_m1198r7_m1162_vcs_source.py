#!/usr/bin/env python3
"""Fail-closed R7 source gate repairing M1194 P1. Invokes no EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

sys.dont_write_bytecode = True


HW = Path(__file__).resolve().parents[1]
TB = HW / "verif_m1193r6_c1_common_charge_protocol/tb_m1193r6_m1162_common_charge_protocol_unit_delay_r6.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1198r7_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1194 = HW / "reviews/m1194_m1193_c1_r6_service_call_closure_source_hammer_r1_20260830"
BASE = M1194 / "hammer_m1194.py"
R4_ATTEMPT = HW / "results/.m1187_m1168r3_m1162_c1_common_charge_protocol_vcs_r4_attempt_consumed/identity.txt"
R4_FAILED = HW / "results/m1187_m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r4_20260830.failed_or_incomplete.3580131.quarantine/RUN_FAILED_OR_INCOMPLETE.txt"
R5_TB = HW / "verif_m1191r5_c1_common_charge_protocol/tb_m1191r5_m1162_common_charge_protocol_unit_delay_r5.sv"

EXPECTED = {
    TB: "0fcc2138ef5d716735eea01dee25a148a5223b1d6adf1e3b2fa464341fbf1345",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    BASE: "d9456931be035ff020146750ca678b626d29b424c011dabbee4c94cf4b93190c",
    M1194 / "review.json": "06082abefe7e7333f9ec6e41f2890e17a892fa3016740315cf46fb7aaaa085c9",
    M1194 / "SHA256SUMS": "32894feaeba31d406bcd29c5e75de40f803ba1e76765421fde9353774fbab14b",
    M1194 / "SHA256SUMS.seal.sha256": "2b9692802dcf5c278980e3f748159ab1a64fe6490af7fec2b3d98d9bf6858f52",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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


def verify_sealed_dir(directory: Path) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "M1194 sealed directory")
    sums = directory / "SHA256SUMS"
    seal = directory / "SHA256SUMS.seal.sha256"
    require(seal.read_text().split() == [sha(sums), "SHA256SUMS"], "M1194 outer seal")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "unsafe M1194 manifest")
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
    require(actual == set(listed), "M1194 manifest membership")
    for name, digest in listed.items():
        require(sha(directory / name) == digest, "M1194 member drift " + name)


def load_base():
    spec = importlib.util.spec_from_file_location("m1194_strong_validator", BASE)
    require(spec is not None and spec.loader is not None, "load M1194 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_MODULE = load_base()


def service_body(text: str) -> str:
    found = BASE_MODULE.tasks(text)
    require("service_assumption_attacks" in found, "service root")
    return BASE_MODULE.strip_comments_and_strings(found["service_assumption_attacks"])


def validate(tb: str, sva: str) -> None:
    # M1194's independent validator recognizes task calls in both legal forms
    # at every statement position, requires the exact closure/force multiset,
    # and exact-matches both full service fault oracles.
    BASE_MODULE.independent_validate(tb, sva)
    require("module tb_m1193r6_m1162_common_charge_protocol_unit_delay_r6" in tb,
            "frozen clean R6 TB module")
    require("m1168r3_m1162_common_charge_protocol_assertions_r3 u_protocol_sva" in tb,
            "frozen R3 SVA")
    require("m1168r3_service_assumption_checker u_service_checker" in tb,
            "independent service checker")
    service = service_body(tb)
    require(service.count("weight_service_attack_mode = 1'b1;") == 1,
            "one weight service attack window")
    require(service.count("psum_service_attack_mode = 1'b1;") == 1,
            "one psum service attack window")
    require(tb.count("request_hold_attack_mode = 1'b1;") == 2,
            "two request attacks")
    require("@(posedge clk_core);\n            @(negedge clk_core);\n"
            "            if (!weight_service_fault" in service,
            "race-free exact weight oracle sample")
    require("@(posedge clk_core);\n            @(negedge clk_core);\n"
            "            if (!psum_service_fault" in service,
            "race-free exact psum oracle sample")
    for token in ("service_skew_isolated=1", "reachable_core_ready_force=0",
                  "boundary_fault=0", "core_fault=0", "functional_vcs_only=true",
                  "timing_verified=false", "cycles_measured=false", "speedup=false",
                  "ppa=false", "energy=false", "system_speedup=false", "headline=false"):
        require(token in tb, "claim token " + token)


def rejected(tb: str, sva: str, target: str, old: str, new: str, label: str) -> None:
    global mutations
    source = tb if target == "tb" else sva
    require(old in source, "mutation anchor " + label)
    changed = source.replace(old, new, 1)
    try:
        validate(changed, sva) if target == "tb" else validate(tb, changed)
    except AssertionError:
        mutations += 1
        return
    raise AssertionError("mutation accepted " + label)


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift " + str(path))
    verify_sealed_dir(M1194)
    m1194_review = json.loads((M1194 / "review.json").read_text())
    require(m1194_review["status"] ==
            "FAIL_SOURCE_CONTRACT__DO_NOT_AUTHOR_RELEASE__NO_VCS_NO_EDA",
            "M1194 fail authority")
    require(m1194_review["p1_finding"]["id"] ==
            "M1194-P1-AUTHOR-CHECKER-BYPASS", "M1194 P1 identity")
    require(R4_ATTEMPT.is_file() and R4_FAILED.is_file(), "R4 failed namespace frozen")
    require(R5_TB.is_file() and sha(R5_TB) != EXPECTED[TB], "R5 non-reuse")
    lines = [line.strip() for line in FILELIST.read_text().splitlines() if line.strip()]
    require(len(lines) == 6 and len(set(lines)) == 6, "R7 filelist cardinality")
    for member in lines:
        require(Path(member).is_file(), "R7 filelist member absent " + member)
    require(lines[-1] == str(TB), "R7 reuses exact clean sealed R6 TB")
    tb, sva = TB.read_text(), SVA.read_text()
    validate(tb, sva)

    marker = "// R6_SERVICE_NO_CORE_READY_FORCE_BOUNDARY"
    require(marker in tb, "marker")
    service_anchor = "    task automatic service_assumption_attacks;"
    require(service_anchor in tb, "service anchor")
    before_marker, after_marker = tb.split(marker, 1)
    force_removed = before_marker + marker + after_marker.replace(
        "            force dut.issue_request_parent_id = 6'b0;\n", "", 1)
    mutations_to_test: dict[str, str] = {
        "helper_alias_to_generic": tb.replace(
            "force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
            "force_request(1'b1, 1'b0, 16'h7301", 1),
        "alias_force": tb.replace(marker,
            "alias r7_ready_alias = dut.core_issue_data_ready;\n"
            "            force r7_ready_alias = 1'b1;", 1),
        "one_request_force_removed": force_removed,
        "weight_peer_oracle_relaxed": tb.replace(
            "!weight_service_fault || psum_service_fault || protocol_error",
            "!weight_service_fault || (1'b0 && psum_service_fault) || protocol_error", 1),
        "psum_peer_oracle_relaxed": tb.replace(
            "!psum_service_fault || weight_service_fault || protocol_error",
            "!psum_service_fault || (1'b0 && weight_service_fault) || protocol_error", 1),
        "protocol_oracle_relaxed": tb.replace(
            "!weight_service_fault || psum_service_fault || protocol_error",
            "!weight_service_fault || psum_service_fault || (1'b0 && protocol_error)", 1),
        "weight_peer_present": tb.replace(
            "weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b0;",
            "weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b1;", 1),
        "psum_peer_present": tb.replace(
            "weight_rsp_valid = 1'b0; psum_rsp_valid = 1'b1;",
            "weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b1;", 1),
        "weight_attack_mask_removed": tb.replace(
            "weight_service_attack_mode = 1'b1;",
            "weight_service_attack_mode = 1'b0;", 1),
        "psum_attack_mask_removed": tb.replace(
            "psum_service_attack_mode = 1'b1;",
            "psum_service_attack_mode = 1'b0;", 1),
        "normal_m935_removed": tb.replace("normal_m935_tasks=1", "normal_m935_tasks=0", 1),
    }
    hidden = (
        "    task automatic r7_hidden_core_force;\n"
        "        begin force dut.core_issue_data_ready = 1'b1; end\n"
        "    endtask\n\n" + service_anchor)
    mutations_to_test["indirect_helper"] = tb.replace(service_anchor, hidden, 1).replace(
        "reset_dut();\n            @(negedge clk_core);\n            force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
        "reset_dut();\n            r7_hidden_core_force();\n            @(negedge clk_core);\n"
        "            force_request_no_core_ready(1'b1, 1'b0, 16'h7301", 1)
    bare = (
        "    task automatic r7_bare_core_force;\n"
        "        begin force dut.core_issue_data_ready = 1'b1; end\n"
        "    endtask\n\n" + service_anchor)
    mutations_to_test["bare_task_call_bypass"] = tb.replace(service_anchor, bare, 1).replace(
        "force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
        "r7_bare_core_force;\n            force_request_no_core_ready(1'b1, 1'b0, 16'h7301", 1)
    same = (
        "    task automatic r7_same_line_core_force;\n"
        "        begin force dut.core_issue_data_ready = 1'b1; end\n"
        "    endtask\n\n" + service_anchor)
    mutations_to_test["same_line_call_bypass"] = tb.replace(service_anchor, same, 1).replace(
        "reset_dut();\n            @(negedge clk_core);\n            force_request_no_core_ready(1'b1, 1'b0, 16'h7301",
        "reset_dut(); r7_same_line_core_force();\n            @(negedge clk_core);\n"
        "            force_request_no_core_ready(1'b1, 1'b0, 16'h7301", 1)
    for label, changed in mutations_to_test.items():
        try:
            validate(changed, sva)
        except AssertionError:
            global mutations
            mutations += 1
        else:
            raise AssertionError("R7 accepted mutation " + label)
    rejected(tb, sva, "sva", "ap_psum_request_hold: assert property",
             "ap_psum_request_hold: assume property", "assert_removed")
    rejected(tb, sva, "sva", "cp_ii2: cover property",
             "cp_ii2: assert property", "cover_removed")
    require(mutations == 16, "all 16 mutations rejected")
    print(json.dumps({
        "schema": "m1198r7_m1162_vcs_source_static_check_v1",
        "status": "PASS_R7_SOURCE_ONLY_M1194_P1_CLOSED__FRESH_DIFFERENT_AUTHOR_HAMMER_AND_RELEASE_REQUIRED__NO_EDA",
        "checks_passed": checks,
        "mutations_rejected": mutations,
        "m1192_p0_closed": True,
        "m1194_p1_closed": True,
        "task_call_forms_detected": ["helper(...) anywhere", "helper; anywhere"],
        "service_closure": sorted(BASE_MODULE.closure(tb)),
        "service_force_multiset_exact_nine": True,
        "service_oracles_exact": True,
        "r6_tb_reused_exactly": True,
        "rtl_modified": False,
        "sva_modified": False,
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
        "source_sha256": {str(path.relative_to(HW)): sha(path)
                          for path in (TB, SVA, FILELIST, WRAPPER, M935, BASE)},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
