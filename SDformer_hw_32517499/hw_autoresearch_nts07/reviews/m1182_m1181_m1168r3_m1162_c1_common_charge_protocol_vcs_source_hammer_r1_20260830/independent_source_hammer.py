#!/usr/bin/env python3
"""Fresh, source-only M1182 hammer for M1181/M1168R3.

This script reads sealed source and failure evidence only.  It must never invoke
VCS, simv, another EDA executable, or a license client.
"""
from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_author_receipt_r1_20260830"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1168r3_m1162_c1_common_charge_protocol_exact_sha_r3.sh"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
TB = HW / "verif_m1168r3_c1_common_charge_protocol/tb_m1168r3_m1162_common_charge_protocol_unit_delay_r3.sv"
STATIC = HW / "verif_m1168r3_c1_common_charge_protocol/static_check_m1168r3_m1162_vcs_source.py"
FILELIST = HW / "dc_handoff/filelists/date_m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
R2_ATTEMPT = HW / "results/.m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed/identity.txt"
R2_Q = HW / "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830.failed_or_incomplete.3284331.quarantine"
FUTURE_RELEASE = HW / "contracts/m1183_m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_launch_release_r3_20260830.json"


checks = 0
mutations = 0


def need(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_manifest(directory: Path) -> tuple[str, str]:
    need(directory.is_dir() and not directory.is_symlink(), f"missing sealed directory: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink(), "manifest absent/nonregular")
    need(outer.is_file() and not outer.is_symlink(), "outer absent/nonregular")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer content mismatch")
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = Path(name)
        need(name not in listed and not path.is_absolute() and ".." not in path.parts,
             "unsafe/duplicate manifest member")
        listed[name] = digest
    actual: set[str] = set()
    for member in directory.rglob("*"):
        rel = member.relative_to(directory).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        if stat.S_ISREG(member.lstat().st_mode):
            actual.add(rel)
    need(actual == set(listed), f"manifest membership mismatch {directory}")
    for name, digest in listed.items():
        need(sha(directory / name) == digest, f"sealed member drift: {name}")
    return sha(manifest), sha(outer)


def verify_leaf_seal(path: Path) -> None:
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(manifest.read_text().split() == [sha(path), path.name], "leaf manifest mismatch")
    need(outer.read_text().split() == [sha(manifest), manifest.name], "leaf outer mismatch")


ASSERTS = (
    "ap_weight_request_hold", "ap_psum_request_hold", "ap_weight_no_reissue",
    "ap_psum_no_reissue", "ap_nonfirst_never_requests_psum",
    "ap_core_valid_requires_requests", "ap_weight_ready_is_atomic",
    "ap_psum_ready_is_first_atomic", "ap_no_lone_weight_consume",
    "ap_no_lone_psum_consume", "ap_core_backpressure_atomic",
    "ap_weight_response_hold", "ap_psum_response_hold",
    "ap_boundary_fault_sticky", "ap_reset_clears_transaction",
    "ap_no_consecutive_response_accept",
)
COVERS = (
    "cp_weight_first", "cp_psum_first", "cp_nonfirst",
    "cp_response_skew_weight", "cp_response_skew_psum", "cp_ii2",
)


def property_body(sva: str, label: str, kind: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s*{kind}\s+property\s*\((.*?)\);", sva, re.S)
    need(match is not None, f"missing {kind} property {label}")
    return match.group(1)


def validate_sources(tb: str, sva: str, runner: str) -> None:
    need(sva.count("assert property") == 16, "assertion count")
    need(sva.count("cover property") == 6, "cover count")
    for name in ASSERTS:
        property_body(sva, name, "assert")
    for name in COVERS:
        property_body(sva, name, "cover")

    # Only the two upstream hold properties may see the request-attack mask.
    request_users = [name for name in ASSERTS if "request_hold_attack_mode" in property_body(sva, name, "assert")]
    need(request_users == ["ap_weight_request_hold", "ap_psum_request_hold"], "request mask broadened")
    need(sva.count("|| request_hold_attack_mode") == 2, "request mask not narrow")
    for name in request_users:
        body = property_body(sva, name, "assert")
        need("disable iff (!reset_n" in body and "|| request_hold_attack_mode" in body,
             "request hold attack mask shape")
        need("1'b1" not in body, "request property permanently disabled")

    weight_users = [name for name in ASSERTS if "weight_service_attack_mode" in property_body(sva, name, "assert")]
    psum_users = [name for name in ASSERTS if "psum_service_attack_mode" in property_body(sva, name, "assert")]
    need(weight_users == ["ap_weight_response_hold"], "weight mask broadened")
    need(psum_users == ["ap_psum_response_hold"], "psum mask broadened")
    need(property_body(sva, "ap_weight_response_hold", "assert").count("!weight_service_attack_mode") == 1,
         "weight mask missing/permanent")
    need(property_body(sva, "ap_psum_response_hold", "assert").count("!psum_service_attack_mode") == 1,
         "psum mask missing/permanent")
    need("disable iff (!reset_n || weight_service_attack_mode" not in sva, "weight broad disable")
    need("disable iff (!reset_n || psum_service_attack_mode" not in sva, "psum broad disable")

    # The service checker is structurally separate from DUT protocol_error and
    # samples service holds with sticky bits at the next posedge.
    checker = re.search(r"module m1168r3_service_assumption_checker\s*\((.*?)endmodule", sva, re.S)
    need(checker is not None, "independent service checker missing")
    cbody = checker.group(1)
    need("protocol_error" not in cbody and "boundary_fault" not in cbody,
         "service checker coupled to DUT classification")
    need("always_ff @(posedge clk_core or negedge reset_n)" in cbody, "checker edge")
    need(cbody.count("service_fault <= 1'b1") == 2, "two sticky service faults")
    need("weight_hold_q" in cbody and "psum_hold_q" in cbody,
         "per-service history missing")

    need(tb.count("request_hold_attack_mode = 1'b1;") == 2, "request attack windows")
    need(tb.count("weight_service_attack_mode = 1'b1;") == 1, "weight attack windows")
    need(tb.count("psum_service_attack_mode = 1'b1;") == 1, "psum attack windows")
    need(tb.count("request_hold_attack_mode = 1'b0;") >= 3, "request mask not cleared")
    need(tb.count("weight_service_attack_mode = 1'b0;") >= 2, "weight mask not cleared")
    need(tb.count("psum_service_attack_mode = 1'b0;") >= 2, "psum mask not cleared")
    need("@(posedge clk_core);\n            @(negedge clk_core);\n            if (!weight_service_fault" in tb,
         "weight service sample is not after NBA")
    need("@(posedge clk_core);\n            @(negedge clk_core);\n            if (!psum_service_fault" in tb,
         "psum service sample is not after NBA")
    need("if (!weight_service_fault || psum_service_fault || protocol_error)" in tb,
         "weight classification boundary relaxed")
    need("if (!psum_service_fault || weight_service_fault || protocol_error)" in tb,
         "psum classification boundary relaxed")

    # 29 legal transactions = four directed + 24 deterministic random + one
    # frozen-M935 normal task.  Each calls the same all-three-masks-low gate.
    gate = re.search(r"task automatic require_legal_masks_clear.*?endtask", tb, re.S)
    need(gate is not None, "legal mask gate missing")
    need("request_hold_attack_mode || weight_service_attack_mode" in gate.group(0)
         and "|| psum_service_attack_mode" in gate.group(0), "legal mask gate weakened")
    need("cov_legal_masks_clear = cov_legal_masks_clear + 1" in gate.group(0), "legal gate uncounted")
    for case in (1, 2, 3, 4, 200):
        need(f"require_legal_masks_clear({case});" in tb, f"legal case {case} unguarded")
    need("require_legal_masks_clear(100 + index);" in tb, "random cases unguarded")
    need("test_index < 24" in tb and "cov_random_transactions != 24" in tb,
         "24-case deterministic regression weakened")
    need("cov_legal_masks_clear != 29" in tb, "29-case all-masks-low gate weakened")

    # Seven DUT faults remain classified as protocol faults; the two service
    # attacks explicitly forbid that classification.
    sticky = re.search(r"task automatic sticky_fault_attacks.*?endtask", tb, re.S)
    need(sticky is not None and sticky.group(0).count("require_sticky_protocol_fault(") == 7,
         "seven DUT attacks missing")
    need("cov_request_attack_windows != 2" in tb, "two request attacks not closed")
    need("cov_weight_service_attack_windows != 1" in tb, "weight service attack not closed")
    need("cov_psum_service_attack_windows != 1" in tb, "psum service attack not closed")
    need("protocol_attacks=7" in tb and "service_assumption_attacks=2" in tb,
         "attack claim counters weakened")
    need("directed_random=24" in tb and "legal_masks_clear=29" in tb,
         "PASS regression counts weakened")

    # II=2 and frozen-M935 completion remain executable checks, not comments.
    ii2 = re.search(r"task automatic directed_ii2.*?endtask", tb, re.S)
    need(ii2 is not None and "second_accept_cycle - first_accept_cycle != 2" in ii2.group(0),
         "II=2 executable check missing")
    need("cp_ii2: cover property" in sva and "ii=2" in tb, "II=2 coverage/claim missing")
    normal = re.search(r"task automatic normal_m935_completion.*?endtask", tb, re.S)
    need(normal is not None and "count_issue_accepts != issue0 + 2" in normal.group(0),
         "M935 two-beat completion missing")
    need("row_complete_count != row0 + 1" in normal.group(0)
         and "task_done_count != done0 + 1" in normal.group(0), "M935 row/task closure missing")
    need("normal_m935_rows=1" in tb and "normal_m935_tasks=1" in tb, "M935 claim weakened")

    for token in ("functional_vcs_only=true", "timing_verified=false", "cycles_measured=false",
                  "speedup=false", "ppa=false", "energy=false", "system_speedup=false",
                  "headline=false"):
        need(re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", tb) is not None,
             f"claim boundary missing {token}")

    # Runner must remain inert before a separate exact-hash release and use a
    # fresh R3 namespace once released.
    need('[[ $# -eq 0 ]]' in runner, "runner accepts arguments")
    for env in ("M1168R3_EXPECTED_RELEASE_SHA256", "M1168R3_EXPECTED_HAMMER_REVIEW_SHA256",
                "M1168R3_EXPECTED_HAMMER_OUTER_SHA256"):
        need(runner.count(env) >= 2, f"runner exact env missing {env}")
    need(str(HERE.relative_to(HW)) in runner, "runner hammer path drift")
    need("m1183_m1182_m1181_m1168r3" in runner, "separate release path absent")
    need("PASS_M1182_M1181_M1168R3_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE" in runner,
         "hammer status gate absent")
    need("AUTHORIZE_EXACTLY_ONE_M1168R3_FUNCTIONAL_VCS_ATTEMPT" in runner,
         "release status gate absent")
    need("'vcs_compiles':1,'simv_runs':1,'all_other_eda_runs':0" in runner,
         "one compile/one sim release cardinality absent")
    need("m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830" in runner,
         "fresh R3 result absent")
    need(".m1168r3_m1162_c1_common_charge_protocol_vcs_r3_attempt_consumed" in runner,
         "fresh R3 attempt absent")
    need("m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed" in runner,
         "consumed R2 evidence not bound")
    need('[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]]' in runner,
         "fresh namespace gate absent")
    need(runner.count('"${VCS_BIN}" -full64') == 1, "compile cardinality")
    need(runner.count("./simv -no_save") == 1, "simv cardinality")
    need("functional_vcs_verified':True" in runner and "timing_verified':False" in runner,
         "functional-only receipt boundary")
    need("speedup':False" in runner and "system_speedup':False" in runner,
         "runner speedup boundary")


def rejected(tb: str, sva: str, runner: str, label: str) -> None:
    global mutations
    try:
        validate_sources(tb, sva, runner)
    except AssertionError:
        mutations += 1
        return
    raise AssertionError(f"mutation accepted: {label}")


def mutate(text: str, old: str, new: str, label: str) -> str:
    need(old in text, f"mutation anchor absent: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    need(not FUTURE_RELEASE.exists(), "separate release was pre-created before hammer")
    c = json.loads(CONTRACT.read_text())
    verify_leaf_seal(CONTRACT)
    author_manifest_sha, author_outer_sha = safe_manifest(AUTHOR)
    author = json.loads((AUTHOR / "review.json").read_text())
    need(author["status"] == "PASS_M1181_M1168R3_SOURCE_ONLY_FORENSICS_AND_NEGATIVE_TEST_ISOLATION__FRESH_M1182_HAMMER_REQUIRED__NO_VCS_NO_EDA",
         "author status")
    need(author["execution_audit"] == {"runner_invocations": 0, "vcs_compiles": 0,
         "simv_runs": 0, "all_eda_runs": 0, "license_queries": 0,
         "attempts_consumed": 0, "results_created": 0}, "author execution audit")
    need(c["status"] == "SOURCE_READY_FOR_FRESH_M1182_HAMMER__NO_VCS_RELEASE",
         "contract status")
    need(c["release"] is False and c["launch_now"] is False and c["source_only"] is True,
         "source-only contract")
    need(c["forensics"]["rtl_fault_proven"] is False
         and c["forensics"]["normal_path_assertion_relaxed"] is False,
         "R2 failure was falsely promoted to RTL fault/relaxation")
    need(c["fresh_namespaces"]["old_r2_namespace_reuse_forbidden"] is True,
         "old namespace reuse not forbidden")
    need(c["execution_audit"] == {"runner_invocations": 0, "vcs_compiles": 0,
         "simv_runs": 0, "all_eda_runs": 0, "license_queries": 0,
         "attempts_consumed": 0, "results_created": 0}, "contract execution audit")

    identities = {
        RUNNER: c["identity"]["runner_sha256"], WRAPPER: c["identity"]["wrapper_sha256"],
        M935: c["identity"]["m935_sha256"], PARENT: c["identity"]["parent_macro_wrapper_sha256"],
        SVA: c["identity"]["sva_r3_sha256"], TB: c["identity"]["testbench_r3_sha256"],
        STATIC: c["identity"]["static_checker_r3_sha256"],
        FILELIST: c["identity"]["filelist_r3_sha256"], DOCS359: c["identity"]["docs359_sha256"],
        R2_ATTEMPT: c["forensics"]["r2_attempt_identity_sha256"],
        R2_Q / "compile.log": c["forensics"]["r2_compile_log_sha256"],
        R2_Q / "sim.log": c["forensics"]["r2_sim_log_sha256"],
        R2_Q / "SHA256SUMS": c["forensics"]["r2_quarantine_manifest_sha256"],
        R2_Q / "SHA256SUMS.seal.sha256": c["forensics"]["r2_quarantine_outer_seal_file_sha256"],
    }
    for path, digest in identities.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             f"identity drift: {path}")
    q_manifest_sha, q_outer_sha = safe_manifest(R2_Q)
    need(q_manifest_sha == c["forensics"]["r2_quarantine_manifest_sha256"], "R2 manifest identity")
    need(q_outer_sha == c["forensics"]["r2_quarantine_outer_seal_file_sha256"], "R2 outer identity")

    compile_log = (R2_Q / "compile.log").read_text(errors="replace")
    sim_log = (R2_Q / "sim.log").read_text(errors="replace")
    need("CPU time:" in compile_log and "to compile +" in compile_log
         and "to elab +" in compile_log and "to link" in compile_log,
         "R2 compile/elaborate/link did not pass")
    need("Error-[" not in compile_log, "R2 had a compiler error")
    need("ap_psum_request_hold: started at 283500ps failed at 286500ps" in sim_log,
         "R2 request-attack failure forensic")
    need("weight service mutation boundary was misclassified" in sim_log,
         "R2 service-sampling failure forensic")
    need("PASS_M1168R2" not in sim_log, "R2 false PASS")
    need((R2_Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text().startswith("status=FAILED_OR_INCOMPLETE"),
         "R2 quarantine status")

    # Execute only the Python static checker.  This is not EDA and does not
    # consume the exact runner, VCS, simv, or a license.
    proc = subprocess.run([sys.executable, "-I", str(STATIC)], cwd=HW,
                          text=True, capture_output=True, check=False)
    need(proc.returncode == 0, "author static checker failed")
    static_output = json.loads(proc.stdout)
    need(static_output["status"] == "PASS_SOURCE_ONLY_FORENSICS_AND_R3_NEGATIVE_TEST_ISOLATION__FRESH_HAMMER_REQUIRED__NO_EDA",
         "static status")
    need(static_output["checks_passed"] >= 600 and static_output["mutations_rejected"] >= 12,
         "author static strength")

    tb = TB.read_text()
    sva = SVA.read_text()
    runner = RUNNER.read_text()
    validate_sources(tb, sva, runner)

    source_mutations = [
        ("request_permanent", tb, mutate(sva, "|| request_hold_attack_mode", "|| 1'b1", "request_permanent"), runner),
        ("request_mask_removed", tb, mutate(sva, "|| request_hold_attack_mode", "|| 1'b0", "request_removed"), runner),
        ("request_mask_broadened", tb, mutate(sva, "disable iff (!reset_n)\n        request_active && weight_request_accepted", "disable iff (!reset_n || request_hold_attack_mode)\n        request_active && weight_request_accepted", "request_broaden"), runner),
        ("weight_mask_permanent", tb, mutate(sva, "!weight_service_attack_mode", "1'b0", "weight_permanent"), runner),
        ("psum_mask_permanent", tb, mutate(sva, "!psum_service_attack_mode", "1'b0", "psum_permanent"), runner),
        ("weight_mask_broadened", tb, mutate(sva, "disable iff (!reset_n)\n        boundary_fault |=>", "disable iff (!reset_n || weight_service_attack_mode)\n        boundary_fault |=>", "weight_broaden"), runner),
        ("service_checker_coupled", tb, mutate(sva, "input  logic reset_n,", "input  logic reset_n,\n    input logic protocol_error,", "checker_coupled"), runner),
        ("assert_removed", tb, mutate(sva, "ap_psum_request_hold: assert property", "ap_psum_request_hold: assume property", "assert_removed"), runner),
        ("cover_removed", tb, mutate(sva, "cp_ii2: cover property", "cp_ii2: assert property", "cover_removed"), runner),
        ("weight_sample_race", mutate(tb, "@(posedge clk_core);\n            @(negedge clk_core);\n            if (!weight_service_fault", "@(posedge clk_core);\n            #1ps;\n            if (!weight_service_fault", "weight_race"), sva, runner),
        ("psum_sample_race", mutate(tb, "@(posedge clk_core);\n            @(negedge clk_core);\n            if (!psum_service_fault", "@(posedge clk_core);\n            #1ps;\n            if (!psum_service_fault", "psum_race"), sva, runner),
        ("weight_class_relaxed", mutate(tb, "if (!weight_service_fault || psum_service_fault || protocol_error)", "if (!weight_service_fault || psum_service_fault)", "weight_class"), sva, runner),
        ("psum_class_relaxed", mutate(tb, "if (!psum_service_fault || weight_service_fault || protocol_error)", "if (!psum_service_fault || weight_service_fault)", "psum_class"), sva, runner),
        ("legal_gate_relaxed", mutate(tb, "|| psum_service_attack_mode)", ")", "legal_gate"), sva, runner),
        ("legal_count_relaxed", mutate(tb, "cov_legal_masks_clear != 29", "cov_legal_masks_clear != 0", "legal_count"), sva, runner),
        ("directed_legal_guard_removed", mutate(tb, "require_legal_masks_clear(1);", "/* removed */", "directed_guard"), sva, runner),
        ("random_legal_guard_removed", mutate(tb, "require_legal_masks_clear(100 + index);", "/* removed */", "random_guard"), sva, runner),
        ("m935_legal_guard_removed", mutate(tb, "require_legal_masks_clear(200);", "/* removed */", "m935_guard"), sva, runner),
        ("random_count_relaxed", mutate(tb, "cov_random_transactions != 24", "cov_random_transactions != 0", "random_count"), sva, runner),
        ("request_windows_relaxed", mutate(tb, "cov_request_attack_windows != 2", "cov_request_attack_windows != 0", "request_windows"), sva, runner),
        ("weight_window_removed", mutate(tb, "weight_service_attack_mode = 1'b1;", "weight_service_attack_mode = 1'b0;", "weight_window"), sva, runner),
        ("psum_window_removed", mutate(tb, "psum_service_attack_mode = 1'b1;", "psum_service_attack_mode = 1'b0;", "psum_window"), sva, runner),
        ("dut_attacks_relaxed", mutate(tb, "protocol_attacks=7", "protocol_attacks=0", "dut_attacks"), sva, runner),
        ("service_attacks_relaxed", mutate(tb, "service_assumption_attacks=2", "service_assumption_attacks=0", "service_attacks"), sva, runner),
        ("ii2_relaxed", mutate(tb, "second_accept_cycle - first_accept_cycle != 2", "second_accept_cycle - first_accept_cycle != 1", "ii2"), sva, runner),
        ("m935_row_relaxed", mutate(tb, "normal_m935_rows=1", "normal_m935_rows=0", "m935_row"), sva, runner),
        ("m935_task_relaxed", mutate(tb, "normal_m935_tasks=1", "normal_m935_tasks=0", "m935_task"), sva, runner),
        ("speed_claim", mutate(tb, "speedup=false", "speedup=true", "speed_claim"), sva, runner),
    ]
    for label, mtb, msva, mrunner in source_mutations:
        rejected(mtb, msva, mrunner, label)

    runner_mutations = [
        ("old_result_namespace", "m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830", "m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830"),
        ("old_attempt_namespace", ".m1168r3_m1162_c1_common_charge_protocol_vcs_r3_attempt_consumed", ".m1168r2_m1162_c1_common_charge_protocol_vcs_r2_attempt_consumed"),
        ("hammer_status_relaxed", "PASS_M1182_M1181_M1168R3_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE", "PASS"),
        ("release_status_relaxed", "AUTHORIZE_EXACTLY_ONE_M1168R3_FUNCTIONAL_VCS_ATTEMPT", "AUTHORIZE"),
        ("release_cardinality_relaxed", "'vcs_compiles':1,'simv_runs':1,'all_other_eda_runs':0", "'vcs_compiles':2,'simv_runs':1,'all_other_eda_runs':0"),
        ("fresh_gate_removed", '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${WORK}" ]]', "true"),
        ("extra_compile", '"${VCS_BIN}" -full64', '"${VCS_BIN}" -full64\n"${VCS_BIN}" -full64'),
        ("extra_sim", "./simv -no_save", "./simv -no_save\n./simv -no_save"),
        ("timing_claim", "timing_verified':False", "timing_verified':True"),
    ]
    for label, old, new in runner_mutations:
        rejected(tb, sva, mutate(runner, old, new, label), label)

    # Fresh R3 namespaces are still absent.  A consumed R2 attempt and sealed
    # quarantine cannot authorize or be reused by this hammer.
    for path in (
        HW / c["fresh_namespaces"]["attempt"],
        HW / c["fresh_namespaces"]["result"],
    ):
        need(not path.exists(), f"fresh namespace already consumed: {path}")
    work_prefix = c["fresh_namespaces"]["work_prefix"].split("/")[-1]
    quarantine_prefix = c["fresh_namespaces"]["quarantine_prefix"].split("/")[-1]
    need(not any((HW / "results").glob(work_prefix + "*")), "R3 work namespace reused")
    need(not any((HW / "results").glob(quarantine_prefix + "*")), "R3 quarantine namespace reused")

    output = {
        "schema": "m1182_m1181_m1168r3_vcs_source_hammer_output_r1_v1",
        "status": "PASS_SOURCE_ONLY",
        "checks_passed": checks,
        "mutations_rejected": mutations,
        "author_static_checks": static_output["checks_passed"],
        "author_static_mutations": static_output["mutations_rejected"],
        "r2_forensics": {
            "compile_elab_link_passed": True,
            "sim_failed": True,
            "request_attack_failure_seen": True,
            "service_sampling_failure_seen": True,
            "rtl_fault_proven": False,
            "attempt_reusable": False,
        },
        "r3": {
            "assertions": 16, "covers": 6, "request_attack_windows": 2,
            "weight_service_attack_windows": 1, "psum_service_attack_windows": 1,
            "legal_masks_clear": 29, "protocol_attacks": 7,
            "service_assumption_attacks": 2, "random_legal_transactions": 24,
            "minimum_completed_issue_ii": 2, "normal_m935_rows": 1,
            "normal_m935_tasks": 1, "independent_service_checker": True,
            "after_nba_negedge_sampling": True, "fresh_namespace": True,
        },
        "source_sha256": {
            "contract": sha(CONTRACT), "runner": sha(RUNNER), "sva": sha(SVA),
            "testbench": sha(TB), "static_checker": sha(STATIC), "filelist": sha(FILELIST),
            "author_review": sha(AUTHOR / "review.json"),
            "author_manifest": author_manifest_sha, "author_outer": author_outer_sha,
            "docs359": sha(DOCS359),
        },
        "execution_audit": {"vcs": 0, "simv": 0, "eda": 0, "license_queries": 0},
    }
    (HERE / "hammer_output.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    (HERE / "static_checker_output.json").write_text(json.dumps(static_output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
