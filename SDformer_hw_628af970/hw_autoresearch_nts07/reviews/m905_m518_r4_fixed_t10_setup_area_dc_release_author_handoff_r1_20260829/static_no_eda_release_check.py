#!/usr/bin/env python3
"""M905 author-side static closure.  This file never invokes a tool runner."""

from __future__ import print_function

import copy
import glob
import hashlib
import json
import math
import os
import re


HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RELEASE = "contracts/m905_m518_r4_fixed_t10_setup_area_dc_launch_release_r1_20260829.json"
RUNNER = "dc_handoff/scripts/run_dc_m518_r4_per_point_setup_area_exact_sha.sh"
TCL = "dc_handoff/scripts/run_dc_m518_r3_per_point_setup_area.tcl"
FILELIST = "dc_handoff/filelists/date_m518_matched_fixed_rank3_logic_only_dc.f"
SDC = "dc_handoff/constraints/date_m289_m273r2_logic_only_3ns_fanout24.sdc"
RTL = "rtl_m518/m518_matched_fixed_t10_atlif.sv"
CONTRACT = "contracts/m518_r4_per_point_setup_area_dc_contract_r1_20260828.json"
ADMISSION = "contracts/m518_r4_fixed_setup_area_dc_launch_admission_r1_20260828.json"
M904 = "reviews/m904_c3_atlif_psn_physical_closure_first_principles_audit_r1_20260829"
VCS = "reviews/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_r1_20260827"
SPEC = "reviews/m518_atlif_fixed_baseline_spec_r1_20260827"
DOCS359 = "docs/359_DATE终局冻结_20260813.md"
CANONICAL = "dc_handoff/runs/m518_r4_fixed_setup_area_logic_only_dc_3p000ns_r1_20260828"
ATTEMPT = "dc_handoff/runs/.m518_r4_fixed_setup_area_attempt_consumed"
FINAL = "reviews/m906_m905_m518_r4_fixed_t10_setup_area_dc_final_launch_hammer_r1_20260829"

EXPECTED = {
    RELEASE: "b1d0e7515a7d12a9918033385bb2b2b7e4981cde1cd438f1aca4c285aa50c2d6",
    RUNNER: "5240712aeaf5dd3b50d68fb29389b1be5d27ba0611c7c50b9d744185c63a00c8",
    TCL: "8f189fc861722f4d5e9005c9301cd01d4fd3c515f5942287df0c58e8e00119e6",
    FILELIST: "bd4454fdb4c86c5ead9e56bf61447dc637916b5258ab5ad8382499a3dfba6b00",
    SDC: "73030f70b27909c1f8100bbc02af75c77fed246908027980912afd6499beb6e3",
    RTL: "8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6",
    CONTRACT: "fab51d46ddabff5254943cd1646be107f3fa173447f26cfd3f863b3657e65b5f",
    ADMISSION: "72e08fc809c149608f1b0701facc1dd41b433547dd6f36fe7e0f35ce1159bcb9",
    M904 + "/review.json": "84dd875ef72410d28c2541c3708a2f9c43969803aac5f12700a0f34a68f60aa0",
    M904 + "/SHA256SUMS": "2c24f6c5a2239bbc5b70d47fe7f31b081b4861804f33acd926532172802f4521",
    M904 + "/SHA256SUMS.seal.sha256": "7ada212aa81240594bc80daa933875df0e5f729261fdc42fde5296629adbe1fc",
    VCS + "/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_verdict_r1.json": "513c5d916859b0f48b9ffeced6853ad89a8ace5ea6a9b264baf05d1ed1966665",
    VCS + "/SHA256SUMS": "76aa238d8ab7feb864a33c5320da2b37acaddb91f952e2635ce80c1de7f7e3c0",
    VCS + "/SHA256SUMS.seal.sha256": "55c661095245364b4f76645f05f48e3a4901129c28d1918e22e6c582d8fd0dcb",
    SPEC + "/m518_atlif_fixed_baseline_spec_r1_20260827.json": "a4b57569d86dca3f0f906565d9b5f7be97335946ac91e38a536d73dca3f2bee1",
    SPEC + "/SHA256SUMS": "177851f6d773c78366382b1cd1e3a64d6e47e06edab0c0fd7c732ba2fdf63d74",
    SPEC + "/SHA256SUMS.seal.sha256": "1a06765ec9bf602cbd2e4b5bda938360713e91a9befa65e1b68aff7e29974bb0",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def path(relative):
    return os.path.join(ROOT, relative)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(relative):
    digest = hashlib.sha256()
    with open(path(relative), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def reject_duplicate(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_load_bytes(payload):
    return json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicate,
                      parse_constant=reject_constant)


def strict_load(relative):
    with open(path(relative), "rb") as handle:
        return strict_load_bytes(handle.read())


def parse_manifest(relative):
    result = []
    with open(path(relative), "r") as handle:
        for line in handle:
            line = line.rstrip("\n")
            require(re.match(r"^[0-9a-f]{64}  \S", line) is not None,
                    "malformed manifest line: " + relative)
            expected, member = line.split("  ", 1)
            result.append((expected, member))
    require(result, "empty manifest: " + relative)
    return result


def verify_file_seal(relative):
    base = os.path.basename(relative)
    directory = os.path.dirname(relative)
    member = relative + ".sha256"
    outer = member + ".seal.sha256"
    rows = parse_manifest(member)
    require(rows == [(sha(relative), base)], "file member seal drift: " + relative)
    outer_rows = parse_manifest(outer)
    require(outer_rows == [(sha(member), os.path.basename(member))],
            "file outer seal drift: " + relative)


def verify_dir_seal(relative):
    manifest = relative + "/SHA256SUMS"
    outer = manifest + ".seal.sha256"
    seen = set()
    for expected, member in parse_manifest(manifest):
        member = member[2:] if member.startswith("./") else member
        target = relative + "/" + member
        require(target not in seen, "duplicate sealed member: " + target)
        seen.add(target)
        require(os.path.isfile(path(target)) and not os.path.islink(path(target)),
                "missing/nonregular sealed member: " + target)
        require(sha(target) == expected, "sealed member SHA drift: " + target)
    outer_rows = parse_manifest(outer)
    require(outer_rows == [(sha(manifest), "SHA256SUMS")],
            "directory outer seal drift: " + relative)


def validate_release(doc):
    require(doc["schema"] ==
            "m905_m518_r4_fixed_t10_setup_area_dc_launch_release_v1",
            "release schema")
    require(doc["status"] ==
            "AUTHORIZED_ONE_M905_M518_R4_FIXED_T10_SETUP_AREA_DC_ATTEMPT",
            "release status")
    require(doc["launch_now"] is True, "launch_now")
    auth = doc["authorization"]
    require(auth["max_attempts"] == 1 and auth["run_dc"] is True,
            "one DC attempt")
    for key in ("run_vcs", "run_formality", "run_pt", "run_ptpx",
                "run_saif", "run_remote", "run_paired_comparison",
                "run_second_c3_point"):
        require(auth[key] is False, "forbidden authorization: " + key)
    fixed = doc["fixed_point_identity"]
    require(fixed["selector_name"] == "M518_R4_POINT" and
            fixed["selector_value"] == "fixed" and
            fixed["top"] == "m518_matched_fixed_t10_atlif",
            "fixed selector/top")
    require(fixed["rtl_sha256"] == EXPECTED[RTL], "fixed RTL identity")
    require(fixed["canonical_result_path"] == CANONICAL and
            fixed["attempt_path"] == ATTEMPT, "unique population identity")
    require(fixed["result_absent_at_authoring"] is True and
            fixed["attempt_absent_at_authoring"] is True and
            fixed["authoring_consumes_attempt"] is False,
            "author population boundary")
    frozen = doc["frozen_execution_identity"]
    require(frozen["runner_sha256"] == EXPECTED[RUNNER] and
            frozen["tcl_sha256"] == EXPECTED[TCL] and
            frozen["filelist_sha256"] == EXPECTED[FILELIST] and
            frozen["sdc_sha256"] == EXPECTED[SDC] and
            frozen["source_contract_sha256"] == EXPECTED[CONTRACT] and
            frozen["fixed_point_admission_sha256"] == EXPECTED[ADMISSION],
            "frozen execution identity")
    require(frozen["inherited_generic_runner_is_narrowed_by_exact_selector_and_fixed_admission"] is True and
            frozen["second_point_admission_bound_by_this_release"] is False and
            frozen["paired_admission_bound_by_this_release"] is False,
            "fixed-only authority")
    require(doc["fixed_functional_authority"]["vcs_review_sha256"] ==
            EXPECTED[VCS + "/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_verdict_r1.json"],
            "VCS authority")
    require(doc["m904_authority"]["review_sha256"] ==
            EXPECTED[M904 + "/review.json"], "M904 authority")
    flow = doc["flow_contract"]
    require(flow["clock_period_ns"] == 3.0 and
            flow["clock_network"] == "ideal" and
            flow["wireload"] == "ZeroWireload" and
            flow["logic_only"] is True and flow["macro_count"] == 0,
            "3ns logic-only flow")
    require(flow["compile_ultra_count"] == 1 and
            flow["incremental_compile_count"] == 0 and
            flow["hold_fix_command_count"] == 0 and
            flow["hold_only_optimization_count"] == 0,
            "single compile/no hold fix")
    require(flow["precompile_TIM_209_required"] == 0 and
            flow["precompile_OPT_150_required"] == 0 and
            flow["setup_met_required"] is True,
            "precompile/setup gate")
    require(flow["hold_scope"] == "diagnostic_only_not_closed_at_dc" and
            flow["hold_report_generated"] is False,
            "hold diagnostic boundary")
    artifacts = set(flow["required_artifacts"])
    for required in ("reports/area.rpt", "reports/qor.rpt",
                     "reports/timing_setup.rpt",
                     "netlist/m518_matched_fixed_t10_atlif_mapped.v",
                     "netlist/m518_matched_fixed_t10_atlif_mapped.sdc",
                     "netlist/m518_matched_fixed_t10_atlif.ddc",
                     "netlist/m518_matched_fixed_t10_atlif.svf"):
        require(required in artifacts, "missing artifact gate: " + required)
    pre = doc["prelaunch_gates"]
    require(pre["fresh_independent_final_release_hammer_required"] is True and
            pre["active_c2_three_axis_one_shot_must_be_terminal"] is True and
            pre["fresh_full_shared_host_collision_check_required"] is True and
            pre["runner_fresh_three_sample_resource_preflight_required"] is True and
            pre["runner_runtime_monitor_and_final_ack_required"] is True and
            pre["result_and_attempt_must_remain_absent"] is True and
            pre["final_reviewer_must_not_execute_command"] is True,
            "prelaunch gates")
    command = doc["post_pass_command_contract"]
    require(command["selector"] == "M518_R4_POINT=fixed" and
            command["caller_pin_runner_sha256"].endswith(EXPECTED[RUNNER]) and
            command["caller_pin_fixed_admission_sha256"].endswith(EXPECTED[ADMISSION]) and
            command["root_may_invoke_once_only_after_final_hammer_and_live_gates"] is True,
            "post-PASS exact command")
    receipt = doc["author_execution_receipt"]
    for key, expected in (("runner_production_invocations", 0), ("dc_runs", 0),
                          ("vcs_runs", 0), ("formality_runs", 0),
                          ("pt_runs", 0), ("ptpx_runs", 0),
                          ("saif_runs", 0), ("license_queries", 0),
                          ("remote_runs", 0)):
        require(receipt[key] == expected, "author no-work: " + key)
    require(receipt["attempt_or_result_created"] is False and
            receipt["docs359_modified"] is False, "author no-work files")
    claims = doc["claim_boundary"]
    require(claims["release_authored"] is True, "release-authored claim")
    for key, value in claims.items():
        if key != "release_authored":
            require(value is False, "premature claim: " + key)
    require(doc["docs359_sha256"] == EXPECTED[DOCS359], "docs359 pin")


def validate_source_text():
    with open(path(RUNNER), "r") as handle:
        runner = handle.read()
    with open(path(TCL), "r") as handle:
        tcl = handle.read()
    with open(path(FILELIST), "r") as handle:
        filelist = handle.read()
    require('M518_R4_POINT:-}" == fixed' in runner,
            "runner lacks fixed selector")
    require("m518_r4_top=m518_matched_fixed_t10_atlif" in runner and
            "m518_r4_admission=contracts/m518_r4_fixed_setup_area_dc_launch_admission_r1_20260828.json" in runner,
            "runner fixed mapping")
    require("M518_R4_EXPECTED_DC_RUNNER_SHA256" in runner and
            "M518_R4_EXPECTED_POINT_ADMISSION_SHA256" in runner,
            "runner caller SHA pins")
    require("m518_r4_preflight_commit_kib=67108864" in runner and
            "m518_r4_runtime_soft_commit_kib=50331648" in runner and
            "m518_r4_runtime_hard_commit_kib=41943040" in runner,
            "runner resource gates")
    require("PASS_FINAL_GATE_ACK" in runner and
            "resource_preflight_external_collisions.tsv" in runner,
            "runner collision/final ACK gates")
    commands = re.findall(r"^\s*compile_ultra\s*$", tcl, re.MULTILINE)
    require(len(commands) == 1, "exactly one compile_ultra command")
    require(re.search(r"compile[^\n]*-incremental", tcl) is None,
            "incremental compile command")
    require("set_fix_hold" not in tcl, "hold-fix command")
    require("TIM-209" in tcl and "OPT-150" in tcl and
            "FAIL_PRECOMPILE_STRUCTURAL_OR_TIMING_GATE" in tcl,
            "TIM/OPT precompile gate")
    for fragment in ("report_qor", "report_area -hierarchy",
                     "report_timing -delay_type max", "write_file -format verilog",
                     "write_sdc", "write -format ddc", "set_svf"):
        require(fragment in tcl, "Tcl artifact command: " + fragment)
    require("report_timing -delay_type min" not in tcl,
            "hold report must remain absent")
    require(filelist.splitlines()[0] == RTL,
            "Fixed RTL must be first frozen corpus member")


def mutation_attacks(original):
    attacks = []
    def add(path_keys, value):
        mutated = copy.deepcopy(original)
        cursor = mutated
        for key in path_keys[:-1]:
            cursor = cursor[key]
        cursor[path_keys[-1]] = value
        attacks.append(mutated)
    add(["status"], "AUTHORIZED_TWO_POINTS")
    add(["launch_now"], False)
    add(["authorization", "max_attempts"], 2)
    add(["authorization", "run_dc"], False)
    add(["authorization", "run_vcs"], True)
    add(["authorization", "run_pt"], True)
    add(["authorization", "run_paired_comparison"], True)
    add(["authorization", "run_second_c3_point"], True)
    add(["fixed_point_identity", "selector_value"], "other")
    add(["fixed_point_identity", "top"], "m273_integrated_rank3_atlif")
    add(["fixed_point_identity", "rtl_sha256"], "0" * 64)
    add(["frozen_execution_identity", "runner_sha256"], "1" * 64)
    add(["frozen_execution_identity", "fixed_point_admission_sha256"], "2" * 64)
    add(["frozen_execution_identity", "second_point_admission_bound_by_this_release"], True)
    add(["flow_contract", "clock_period_ns"], 2.0)
    add(["flow_contract", "compile_ultra_count"], 2)
    add(["flow_contract", "hold_fix_command_count"], 1)
    add(["flow_contract", "precompile_TIM_209_required"], 1)
    add(["prelaunch_gates", "active_c2_three_axis_one_shot_must_be_terminal"], False)
    add(["post_pass_command_contract", "selector"], "M518_R4_POINT=rank3")
    add(["author_execution_receipt", "dc_runs"], 1)
    add(["claim_boundary", "area"], True)
    add(["claim_boundary", "system_speedup"], True)
    add(["docs359_sha256"], "f" * 64)
    rejected = 0
    for mutated in attacks:
        try:
            validate_release(mutated)
        except (AssertionError, KeyError, TypeError, ValueError):
            rejected += 1
    require(rejected == len(attacks), "semantic mutation escaped")
    return len(attacks)


def json_parser_attacks():
    rejected = 0
    for payload in (b'{"x":1,"x":2}',
                    b'{"launch_now":true,"launch_now":false}',
                    b'{"authorization":{"run_dc":true,"run_dc":false}}'):
        try:
            strict_load_bytes(payload)
        except ValueError:
            rejected += 1
    require(rejected == 3, "duplicate key accepted")
    rejected = 0
    for payload in (b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'):
        try:
            strict_load_bytes(payload)
        except ValueError:
            rejected += 1
    require(rejected == 3, "non-finite JSON accepted")
    return 3, 3


def assert_population_absence():
    require(not os.path.lexists(path(CANONICAL)), "canonical result populated")
    require(not os.path.lexists(path(ATTEMPT)), "attempt populated")
    require(not os.path.lexists(path(FINAL)), "final hammer output populated")
    require(glob.glob(path(CANONICAL + ".failed_or_incomplete.*.quarantine")) == [],
            "fixed quarantine populated")


def main():
    assert_population_absence()
    for relative, expected in EXPECTED.items():
        require(os.path.isfile(path(relative)) and not os.path.islink(path(relative)),
                "missing/nonregular/symlink: " + relative)
        require(sha(relative) == expected, "SHA drift: " + relative)
    verify_file_seal(RELEASE)
    verify_file_seal(ADMISSION)
    verify_dir_seal(M904)
    verify_dir_seal(VCS)
    verify_dir_seal(SPEC)
    release = strict_load(RELEASE)
    m904 = strict_load(M904 + "/review.json")
    vcs = strict_load(VCS + "/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_verdict_r1.json")
    validate_release(release)
    require(m904["status"] == release["m904_authority"]["status"] and
            m904["one_and_only_one_next_action"]["success_scope"] ==
            "raw Fixed setup/area-only point requiring an independent result hammer",
            "M904 next-action authority")
    require(vcs["status"] == release["fixed_functional_authority"]["vcs_status"],
            "VCS status authority")
    validate_source_text()
    semantic_count = mutation_attacks(release)
    duplicate_count, nonfinite_count = json_parser_attacks()
    assert_population_absence()
    print("PASS_M905_FIXED_ONLY_RELEASE_STATIC_NO_EDA")
    print("semantic_mutation_attacks=%d" % semantic_count)
    print("duplicate_key_attacks=%d" % duplicate_count)
    print("nonfinite_attacks=%d" % nonfinite_count)
    print("runner_production_invocations=0")
    print("dc_runs=0")
    print("vcs_runs=0")
    print("license_queries=0")
    print("attempt_or_result_created=false")
    print("docs359_sha256=" + sha(DOCS359))


if __name__ == "__main__":
    main()
