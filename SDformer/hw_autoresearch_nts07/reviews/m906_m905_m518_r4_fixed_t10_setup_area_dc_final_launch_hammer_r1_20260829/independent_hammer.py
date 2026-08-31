#!/usr/bin/env python3
"""Fresh independent no-EDA final hammer for the M905 Fixed-only release.

The program is deliberately Python-3.6 compatible.  It performs only static
reads, hashes, strict JSON parsing, in-memory mutation attacks, and shell
syntax checking.  It never invokes the released runner, an EDA executable, a
license command, or a remote command, and it never creates an attempt/result.
"""

from __future__ import print_function

import copy
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys


HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

RELEASE = "contracts/m905_m518_r4_fixed_t10_setup_area_dc_launch_release_r1_20260829.json"
HANDOFF = "reviews/m905_m518_r4_fixed_t10_setup_area_dc_release_author_handoff_r1_20260829"
REQUEST = "reviews/m906_m905_m518_r4_fixed_t10_setup_area_dc_final_launch_hammer_REQUEST_r1_20260829"
M904 = "reviews/m904_c3_atlif_psn_physical_closure_first_principles_audit_r1_20260829"
VCS = "reviews/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_r1_20260827"
SPEC = "reviews/m518_atlif_fixed_baseline_spec_r1_20260827"
CANDIDATE_HAMMER = "reviews/m572_m518_r4_per_point_dc_launch_admission_candidate_hammer_r1_20260828"

RUNNER = "dc_handoff/scripts/run_dc_m518_r4_per_point_setup_area_exact_sha.sh"
TCL = "dc_handoff/scripts/run_dc_m518_r3_per_point_setup_area.tcl"
FILELIST = "dc_handoff/filelists/date_m518_matched_fixed_rank3_logic_only_dc.f"
SDC = "dc_handoff/constraints/date_m289_m273r2_logic_only_3ns_fanout24.sdc"
FIXED_RTL = "rtl_m518/m518_matched_fixed_t10_atlif.sv"
RANK3_RTL = "rtl_m273/m273_integrated_rank3_atlif.sv"
CONTRACT = "contracts/m518_r4_per_point_setup_area_dc_contract_r1_20260828.json"
ADMISSION = "contracts/m518_r4_fixed_setup_area_dc_launch_admission_r1_20260828.json"
CANDIDATE = "contracts/m572_m518_r4_fixed_setup_area_dc_launch_admission_candidate_r1_20260828.json"
DOCS359 = "docs/359_DATE终局冻结_20260813.md"

M904_JSON = M904 + "/review.json"
VCS_JSON = VCS + "/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_verdict_r1.json"
SPEC_JSON = SPEC + "/m518_atlif_fixed_baseline_spec_r1_20260827.json"
HANDOFF_JSON = HANDOFF + "/handoff.json"
REQUEST_JSON = REQUEST + "/request.json"
CANDIDATE_HAMMER_JSON = CANDIDATE_HAMMER + "/review.json"

CANONICAL = "dc_handoff/runs/m518_r4_fixed_setup_area_logic_only_dc_3p000ns_r1_20260828"
ATTEMPT = "dc_handoff/runs/.m518_r4_fixed_setup_area_attempt_consumed"

TOOL_PATHS = {
    "/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell":
        "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    "/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell":
        "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    "/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec":
        "bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391",
    "/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db":
        "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    "/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db":
        "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
}

EXPECTED = {
    RELEASE: "b1d0e7515a7d12a9918033385bb2b2b7e4981cde1cd438f1aca4c285aa50c2d6",
    HANDOFF_JSON: "bbb06d657378d50edbcbfe29d1f8f145234fcd080b0c94447493a409fe81620d",
    REQUEST_JSON: "1aa3c3f4b2ebd573e6d87f249e315b8b5205705a74f3b8f367987f1d6b27b464",
    RUNNER: "5240712aeaf5dd3b50d68fb29389b1be5d27ba0611c7c50b9d744185c63a00c8",
    TCL: "8f189fc861722f4d5e9005c9301cd01d4fd3c515f5942287df0c58e8e00119e6",
    FILELIST: "bd4454fdb4c86c5ead9e56bf61447dc637916b5258ab5ad8382499a3dfba6b00",
    SDC: "73030f70b27909c1f8100bbc02af75c77fed246908027980912afd6499beb6e3",
    FIXED_RTL: "8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6",
    RANK3_RTL: "11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d",
    CONTRACT: "fab51d46ddabff5254943cd1646be107f3fa173447f26cfd3f863b3657e65b5f",
    ADMISSION: "72e08fc809c149608f1b0701facc1dd41b433547dd6f36fe7e0f35ce1159bcb9",
    CANDIDATE: "e83e2a47319a5fca165fb918adfb64659d1d968022aa946c52e8788bd5aa82a4",
    CANDIDATE_HAMMER_JSON: "df459336391ead6372999de1e68b78439fdd5e225662646b64761dc10c389e3b",
    M904_JSON: "84dd875ef72410d28c2541c3708a2f9c43969803aac5f12700a0f34a68f60aa0",
    VCS_JSON: "513c5d916859b0f48b9ffeced6853ad89a8ace5ea6a9b264baf05d1ed1966665",
    SPEC_JSON: "a4b57569d86dca3f0f906565d9b5f7be97335946ac91e38a536d73dca3f2bee1",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_DIR_SEALS = {
    HANDOFF: (
        "5a14d01ad980b0bdcee354cd0c9b211d3ac75f74e12c39ed502fd4ac224ac196",
        "ed9e2741de350a42345ee81851aced24383b39314c90eb3ef94d765964275692"),
    REQUEST: (
        "1102080b247b3af99250ef9d8ecd7df0adc7bea95333c614fa14315724b0eb64",
        "c2f53906cbb31f7afd6b08d30662e3eda380da0850dd4574e9ea26d10ab0f388"),
    M904: (
        "2c24f6c5a2239bbc5b70d47fe7f31b081b4861804f33acd926532172802f4521",
        "7ada212aa81240594bc80daa933875df0e5f729261fdc42fde5296629adbe1fc"),
    VCS: (
        "76aa238d8ab7feb864a33c5320da2b37acaddb91f952e2635ce80c1de7f7e3c0",
        "55c661095245364b4f76645f05f48e3a4901129c28d1918e22e6c582d8fd0dcb"),
    SPEC: (
        "177851f6d773c78366382b1cd1e3a64d6e47e06edab0c0fd7c732ba2fdf63d74",
        "1a06765ec9bf602cbd2e4b5bda938360713e91a9befa65e1b68aff7e29974bb0"),
    CANDIDATE_HAMMER: (
        "97f75d77dd12517fb971ae37fd3d9e7aa25a6e8f4fedd84cc2f8ebfab1c165c0",
        "52b29d31cb89ce632f012acfbccadacdea090a98a2ef5ccc35970473aa350788"),
}


def path(relative):
    return os.path.join(ROOT, relative)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha_abs(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha(relative):
    return sha_abs(path(relative))


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
    rows = []
    with open(path(relative), "r") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            require(re.match(r"^[0-9a-f]{64}  \S", line) is not None,
                    "malformed manifest: " + relative)
            expected, member = line.split("  ", 1)
            rows.append((expected, member))
    require(rows, "empty manifest: " + relative)
    return rows


def verify_file_seal(relative):
    member = relative + ".sha256"
    outer = member + ".seal.sha256"
    require(parse_manifest(member) == [(sha(relative), os.path.basename(relative))],
            "payload seal drift: " + relative)
    require(parse_manifest(outer) == [(sha(member), os.path.basename(member))],
            "outer seal drift: " + relative)


def verify_dir_seal(relative):
    manifest = relative + "/SHA256SUMS"
    outer = manifest + ".seal.sha256"
    seen = set()
    for expected, member in parse_manifest(manifest):
        if member.startswith("./"):
            member = member[2:]
        require(member not in seen and member not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
                "duplicate/reserved sealed member: " + member)
        seen.add(member)
        target = relative + "/" + member
        require(os.path.isfile(path(target)) and not os.path.islink(path(target)),
                "missing/nonregular/symlink member: " + target)
        require(sha(target) == expected, "sealed member drift: " + target)
    require(parse_manifest(outer) == [(sha(manifest), "SHA256SUMS")],
            "directory outer seal drift: " + relative)
    expected_manifest, expected_outer = EXPECTED_DIR_SEALS[relative]
    require(sha(manifest) == expected_manifest, "manifest identity drift: " + relative)
    require(sha(outer) == expected_outer, "outer identity drift: " + relative)


def typed_int(value, expected):
    return type(value) is int and value == expected


def validate_release(doc):
    require(doc["schema"] == "m905_m518_r4_fixed_t10_setup_area_dc_launch_release_v1",
            "release schema")
    require(doc["status"] == "AUTHORIZED_ONE_M905_M518_R4_FIXED_T10_SETUP_AREA_DC_ATTEMPT",
            "release status")
    require(doc["launch_now"] is True, "launch_now")
    auth = doc["authorization"]
    require(typed_int(auth["max_attempts"], 1) and auth["run_dc"] is True,
            "one Fixed DC attempt")
    for key in ("run_vcs", "run_formality", "run_pt", "run_ptpx",
                "run_saif", "run_remote", "run_paired_comparison",
                "run_second_c3_point"):
        require(auth[key] is False, "forbidden release authorization: " + key)

    fixed = doc["fixed_point_identity"]
    require(fixed["selector_name"] == "M518_R4_POINT" and
            fixed["selector_value"] == "fixed" and
            fixed["top"] == "m518_matched_fixed_t10_atlif",
            "not exact Fixed selector/top")
    require(fixed["rtl_path"] == FIXED_RTL and
            fixed["rtl_sha256"] == EXPECTED[FIXED_RTL], "Fixed RTL binding")
    require(fixed["canonical_result_path"] == CANONICAL and
            fixed["attempt_path"] == ATTEMPT, "Fixed population identity")
    require(fixed["result_absent_at_authoring"] is True and
            fixed["attempt_absent_at_authoring"] is True and
            fixed["authoring_consumes_attempt"] is False, "author no-attempt boundary")

    frozen = doc["frozen_execution_identity"]
    pairs = (("runner_sha256", RUNNER), ("tcl_sha256", TCL),
             ("filelist_sha256", FILELIST), ("sdc_sha256", SDC),
             ("source_contract_sha256", CONTRACT),
             ("fixed_point_admission_sha256", ADMISSION))
    for key, filename in pairs:
        require(frozen[key] == EXPECTED[filename], "release frozen SHA: " + key)
    require(frozen["inherited_generic_runner_is_narrowed_by_exact_selector_and_fixed_admission"] is True,
            "generic runner is not explicitly narrowed")
    require(frozen["second_point_admission_bound_by_this_release"] is False and
            frozen["paired_admission_bound_by_this_release"] is False,
            "rank3/paired bound by release")

    functional = doc["fixed_functional_authority"]
    require(functional["vcs_review_sha256"] == EXPECTED[VCS_JSON] and
            functional["vcs_status"] ==
            "PASS_DIRECTED_FIXED_T10_VCS_BEHAVIOR__DC_PPA_SYSTEM_HEADLINE_NOT_ADMITTED",
            "Fixed r11 VCS authority")
    require(typed_int(functional["directed_issue_cycles_per_tile"], 17) and
            functional["directed_service_formula"] == "17*N+12",
            "Fixed directed service authority")
    require(doc["m904_authority"]["review_sha256"] == EXPECTED[M904_JSON] and
            doc["m904_authority"]["required_next_action_selector"] == "fixed" and
            doc["m904_authority"]["matched_ppa"] is False,
            "M904 Fixed-only authority")

    flow = doc["flow_contract"]
    require(typed_int(flow["technology_nm"], 28) and
            type(flow["clock_period_ns"]) is float and flow["clock_period_ns"] == 3.0 and
            flow["clock_network"] == "ideal" and flow["wireload"] == "ZeroWireload" and
            flow["logic_only"] is True and typed_int(flow["macro_count"], 0),
            "3ns 28nm ideal/ZeroWireload logic-only flow")
    for key, expected in (("compile_ultra_count", 1),
                          ("incremental_compile_count", 0),
                          ("hold_fix_command_count", 0),
                          ("hold_only_optimization_count", 0),
                          ("precompile_TIM_209_required", 0),
                          ("precompile_OPT_150_required", 0)):
        require(typed_int(flow[key], expected), "flow integer contract: " + key)
    require(flow["setup_met_required"] is True and
            flow["hold_scope"] == "diagnostic_only_not_closed_at_dc" and
            flow["hold_report_generated"] is False, "setup/hold boundary")

    # Seven citable raw-result objects, plus one separate structured predicate.
    seven = ["reports/area.rpt", "reports/qor.rpt", "reports/timing_setup.rpt",
             "netlist/m518_matched_fixed_t10_atlif_mapped.v",
             "netlist/m518_matched_fixed_t10_atlif_mapped.sdc",
             "netlist/m518_matched_fixed_t10_atlif.ddc",
             "netlist/m518_matched_fixed_t10_atlif.svf"]
    artifacts = flow["required_artifacts"]
    require(len(artifacts) == 8 and all(item in artifacts for item in seven) and
            "reports/structured_postcompile_gate.rpt" in artifacts,
            "seven outputs plus structured gate")

    pre = doc["prelaunch_gates"]
    for key in ("fresh_independent_final_release_hammer_required",
                "active_c2_three_axis_one_shot_must_be_terminal",
                "fresh_full_shared_host_collision_check_required",
                "runner_fresh_three_sample_resource_preflight_required",
                "runner_runtime_monitor_and_final_ack_required",
                "result_and_attempt_must_remain_absent",
                "final_reviewer_must_not_execute_command"):
        require(pre[key] is True, "prelaunch gate: " + key)

    post = doc["post_pass_command_contract"]
    require(post["selector"] == "M518_R4_POINT=fixed" and
            post["caller_pin_runner_sha256"] ==
            "M518_R4_EXPECTED_DC_RUNNER_SHA256=" + EXPECTED[RUNNER] and
            post["caller_pin_fixed_admission_sha256"] ==
            "M518_R4_EXPECTED_POINT_ADMISSION_SHA256=" + EXPECTED[ADMISSION] and
            post["environment_isolation_required"] is True and
            post["root_may_invoke_once_only_after_final_hammer_and_live_gates"] is True,
            "post-PASS exact command contract")

    receipt = doc["author_execution_receipt"]
    for key in ("runner_production_invocations", "dc_runs", "vcs_runs",
                "formality_runs", "pt_runs", "ptpx_runs", "saif_runs",
                "license_queries", "remote_runs"):
        require(typed_int(receipt[key], 0), "author execution: " + key)
    require(receipt["attempt_or_result_created"] is False and
            receipt["docs359_modified"] is False, "author changed protected state")
    for key, value in doc["claim_boundary"].items():
        if key == "release_authored":
            require(value is True, "release-authored boundary")
        else:
            require(value is False, "premature release claim: " + key)
    require(doc["docs359_sha256"] == EXPECTED[DOCS359], "docs359 release pin")


def validate_chain(release, handoff, request, contract, admission, m904, vcs, spec):
    require(handoff["schema"] == "m905_m518_r4_fixed_t10_setup_area_dc_release_author_handoff_v1" and
            handoff["status"] == "PASS_AUTHOR_M905_FIXED_ONLY_INERT_RELEASE__FRESH_FINAL_HAMMER_REQUIRED__NO_EDA",
            "handoff schema/status")
    require(handoff["release"]["payload_sha256"] == EXPECTED[RELEASE] and
            handoff["release"]["inert_until_fresh_final_hammer_pass100"] is True,
            "handoff release binding")
    require(handoff["frozen_fixed_source"]["selector"] == "M518_R4_POINT=fixed" and
            handoff["frozen_fixed_source"]["top"] == "m518_matched_fixed_t10_atlif",
            "handoff Fixed binding")
    require(handoff["flow_boundary"]["required_outputs"] ==
            ["area", "qor", "setup", "mapped_verilog", "mapped_sdc", "ddc", "svf"],
            "handoff seven-output list")
    require(handoff["prelaunch_boundary"]["active_c2_one_shot_must_be_terminal"] is True and
            handoff["prelaunch_boundary"]["fresh_shared_host_collision_and_resource_gates_required"] is True,
            "handoff C2/live gates")

    require(request["schema"] ==
            "m906_m905_m518_r4_fixed_t10_setup_area_dc_final_launch_hammer_request_v1" and
            request["status"] ==
            "REQUEST_FRESH_INDEPENDENT_M905_FIXED_ONLY_DC_FINAL_LAUNCH_HAMMER__NO_EDA",
            "request schema/status")
    require(request["review_target"]["release_payload_sha256"] == EXPECTED[RELEASE] and
            request["review_target"]["score_required"] == 100 and
            request["review_target"]["severity_counts_required"] ==
            {"p0": 0, "p1": 0, "p2": 0}, "request target")
    require(request["frozen_source_chain"]["selector"] == "M518_R4_POINT=fixed" and
            request["frozen_source_chain"]["top"] == "m518_matched_fixed_t10_atlif",
            "request Fixed-only selector")
    require(request["post_pass_command_contract"]["command_must_wait_for_active_c2_one_shot_terminal"] is True and
            request["post_pass_command_contract"]["command_must_not_be_executed_by_final_reviewer"] is True,
            "request C2/no-execution boundary")
    for key, value in request["request_authorization"].items():
        if key in ("run_static_no_eda_checks", "run_mutation_attacks_in_memory"):
            require(value is True, "requested static authority")
        elif key == "max_eda_attempts_authorized_by_request":
            require(typed_int(value, 0), "request attempt authority")
        else:
            require(value is False, "request authorized execution: " + key)

    require(contract["status"] == "AUTHOR_SOURCE_ONLY__FRESH_STATIC_REVIEW_REQUIRED__NO_LAUNCH_ADMISSION" and
            contract["authorization"]["launch_now"] is False and
            contract["authorization"]["run_dc"] is False,
            "source contract remains inert")
    require(contract["identity"]["runner_sha256"] == EXPECTED[RUNNER] and
            contract["identity"]["tcl_sha256"] == EXPECTED[TCL] and
            contract["identity"]["fixed_top"] == "m518_matched_fixed_t10_atlif",
            "source contract identity")
    require(contract["setup_area_flow"] == {
        "clock_period_ns": 3, "clock_network": "ideal", "wireload": "ZeroWireload",
        "compile_ultra_count": 1, "incremental_compile_count": 0,
        "hold_fix_command_count": 0, "hold_only_optimization_count": 0,
        "hold_report_generated": False, "hold_not_closed_at_dc": True,
        "hold_or_full_sta_claim": False, "setup_and_area_only": True,
        "macro_count": 0, "logic_only": True}, "contract setup/area flow")
    require(contract["paired_comparison_admission_schema"]["created_in_this_source_package"] is False and
            contract["paired_comparison_admission_schema"]["comparison_claim_before_admission"] is False,
            "paired comparison not admitted")

    require(admission["status"] == "AUTHORIZED_ONE_M518_R4_FIXED_SETUP_AREA_DC_ATTEMPT" and
            admission["point"] == "fixed" and admission["top"] == "m518_matched_fixed_t10_atlif",
            "Fixed admission")
    require(typed_int(admission["authorization"]["max_attempts"], 1) and
            admission["authorization"]["run_dc"] is True and
            admission["authorization"]["run_paired_comparison"] is False,
            "Fixed admission authorization")
    require(admission["identity"]["runner_sha256"] == EXPECTED[RUNNER] and
            admission["identity"]["contract_sha256"] == EXPECTED[CONTRACT] and
            admission["identity"]["candidate_sha256"] == EXPECTED[CANDIDATE] and
            admission["identity"]["candidate_hammer_review_sha256"] == EXPECTED[CANDIDATE_HAMMER_JSON],
            "Fixed admission chain")
    require(admission["paired_boundary"]["paired_comparison_authorized"] is False and
            admission["live_execution_gates"]["fresh_shared_host_collision_check_required"] is True and
            admission["live_execution_gates"]["fresh_resource_preflight_required"] is True,
            "admission paired/live boundary")

    require(m904["status"] == release["m904_authority"]["status"] and
            m904["score_out_of_100"] == 100 and
            m904["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0},
            "M904 authority status")
    next_action = m904["one_and_only_one_next_action"]
    require(next_action["ordinal"] == 1 and
            next_action["success_scope"] ==
            "raw Fixed setup/area-only point requiring an independent result hammer" and
            "M518_R4_POINT=fixed" in next_action["action"] and
            "do not launch rank3" in next_action["action"], "M904 next action")
    require(m904["current_r4_physical_state"]["matched_ppa"] is False and
            m904["claim_boundary"]["paper_ppa_ready"] is False,
            "M904 open physical boundary")

    require(vcs["status"] == release["fixed_functional_authority"]["vcs_status"] and
            vcs["vcs"]["compile_rc"] == 0 and vcs["vcs"]["sim_rc"] == 0 and
            vcs["vcs"]["numeric_mismatches"] == 0 and
            vcs["cycle_anchors"]["issue_cycles_per_tile"] == 17,
            "Fixed r11 directed VCS")
    require(vcs["claim_boundary"]["dc"] is False and
            vcs["claim_boundary"]["ppa"] is False and
            vcs["claim_boundary"]["headline"] is False,
            "VCS claim boundary")
    require(spec["decision"]["proposed_top"] == "m518_matched_fixed_t10_atlif" and
            spec["decision"]["proposed_rtl_path"] == FIXED_RTL and
            spec["matched_anchor"]["fixed_formula"] == "17*N+12" and
            spec["matched_anchor"]["area_matched"] is False,
            "Fixed baseline spec")


def validate_source_text():
    runner = open(path(RUNNER), "r").read()
    tcl = open(path(TCL), "r").read()
    sdc = open(path(SDC), "r").read()
    filelist = [line.strip() for line in open(path(FILELIST), "r") if line.strip()]

    require('[[ "${M518_R4_POINT:-}" == fixed || "${M518_R4_POINT:-}" == rank3 ]]' in runner,
            "runner selector parser drift")
    require("m518_r4_top=m518_matched_fixed_t10_atlif" in runner and
            "m518_r4_admission=contracts/m518_r4_fixed_setup_area_dc_launch_admission_r1_20260828.json" in runner,
            "runner Fixed mapping")
    require("M518_R4_EXPECTED_DC_RUNNER_SHA256" in runner and
            "M518_R4_EXPECTED_POINT_ADMISSION_SHA256" in runner,
            "caller SHA pins")
    require("m518_r4_preflight_commit_kib=67108864" in runner and
            "m518_r4_runtime_soft_commit_kib=50331648" in runner and
            "m518_r4_runtime_hard_commit_kib=41943040" in runner and
            "m518_r4_mem_available_kib=134217728" in runner and
            "m518_r4_swap_free_kib=33554432" in runner,
            "resource thresholds")
    require("for m518_r4_sample in 1 2 3" in runner and
            "external_eda_collision_immediate" in runner and
            "runtime_final_gate_ack.txt" in runner and
            "PASS_FINAL_GATE_ACK" in runner,
            "three-sample/collision/runtime-final gates")
    require("dc_shell:*|dc_shell-t:*|fm_shell:*|pt_shell:*|vcs:*|vcs1:*|vlogan:*|simv:*|common_shell_ex*:common_shell_exec" in runner,
            "full-host collision namespace")

    compile_lines = re.findall(r"^\s*compile_ultra\s*$", tcl, re.MULTILINE)
    require(len(compile_lines) == 1, "not exactly one compile_ultra")
    require(re.search(r"^\s*compile[^\n]*-incremental", tcl, re.MULTILINE) is None,
            "incremental compile command")
    require("set_fix_hold" not in tcl and
            re.search(r"report_timing[^\n]*-delay_type\s+min", tcl) is None,
            "hold closure/report leaked into setup-only point")
    require("TIM-209" in tcl and "OPT-150" in tcl and
            "FAIL_PRECOMPILE_STRUCTURAL_OR_TIMING_GATE__EXPLICIT_EXIT36__NO_COMPILE" in tcl,
            "precompile fail-closed gate")
    require("set_wire_load_model -name ZeroWireload" in tcl and
            "set_propagated_clock" not in tcl and
            "report_timing -delay_type max" in tcl,
            "ideal ZeroWireload setup flow")
    require("set clock_period_ns 3.000" in sdc and
            "create_clock -name core_clk -period $clock_period_ns" in sdc,
            "3ns SDC")

    seven_commands = ("report_area -hierarchy", "report_qor",
                      "report_timing -delay_type max", "write_file -format verilog",
                      "write_sdc", "write -format ddc", "set_svf")
    for command in seven_commands:
        require(command in tcl, "missing seven-output producer: " + command)
    for predicate in ("structured_postcompile_gate.rpt", "check_design_ok=1",
                      "check_timing_ok=1", "dc_bit_level_port_count=1175",
                      "Number of macros/black boxes:               0",
                      "slack (MET)", "slack (VIOLATED)",
                      "m518_r4_complete=1"):
        require(predicate in runner, "missing production predicate: " + predicate)
    publish = runner.index('mv -T "${m518_r4_work}" "${m518_r4_canonical}"')
    require(publish > runner.index("slack (VIOLATED)") and
            publish > runner.index("runtime_final_gate_ack.txt") and
            publish > runner.index("structured_postcompile_gate.rpt"),
            "canonical publication occurs before predicates")
    require(filelist == [FIXED_RTL, RANK3_RTL], "frozen two-file corpus")

    shell_check = subprocess.Popen(["/usr/bin/bash", "-n", path(RUNNER)])
    require(shell_check.wait() == 0, "runner bash syntax")


def mutation_attacks(release):
    mutations = []

    def add(keys, value):
        candidate = copy.deepcopy(release)
        cursor = candidate
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        mutations.append(candidate)

    add(["schema"], "other")
    add(["status"], "AUTHORIZED_TWO_POINTS")
    add(["launch_now"], False)
    add(["authorization", "max_attempts"], 2)
    add(["authorization", "max_attempts"], True)
    add(["authorization", "run_dc"], False)
    add(["authorization", "run_vcs"], True)
    add(["authorization", "run_formality"], True)
    add(["authorization", "run_pt"], True)
    add(["authorization", "run_ptpx"], True)
    add(["authorization", "run_saif"], True)
    add(["authorization", "run_remote"], True)
    add(["authorization", "run_paired_comparison"], True)
    add(["authorization", "run_second_c3_point"], True)
    add(["fixed_point_identity", "selector_value"], "rank3")
    add(["fixed_point_identity", "top"], "m273_integrated_rank3_atlif")
    add(["fixed_point_identity", "rtl_sha256"], "0" * 64)
    add(["frozen_execution_identity", "runner_sha256"], "1" * 64)
    add(["frozen_execution_identity", "fixed_point_admission_sha256"], "2" * 64)
    add(["frozen_execution_identity", "second_point_admission_bound_by_this_release"], True)
    add(["frozen_execution_identity", "paired_admission_bound_by_this_release"], True)
    add(["flow_contract", "clock_period_ns"], 2.0)
    add(["flow_contract", "clock_period_ns"], 3)
    add(["flow_contract", "wireload"], "Enclosed")
    add(["flow_contract", "macro_count"], 1)
    add(["flow_contract", "compile_ultra_count"], 2)
    add(["flow_contract", "compile_ultra_count"], True)
    add(["flow_contract", "incremental_compile_count"], 1)
    add(["flow_contract", "hold_fix_command_count"], 1)
    add(["flow_contract", "hold_report_generated"], True)
    add(["flow_contract", "precompile_TIM_209_required"], 1)
    add(["flow_contract", "precompile_OPT_150_required"], 1)
    add(["flow_contract", "required_artifacts"], ["reports/area.rpt"])
    add(["prelaunch_gates", "active_c2_three_axis_one_shot_must_be_terminal"], False)
    add(["prelaunch_gates", "fresh_full_shared_host_collision_check_required"], False)
    add(["prelaunch_gates", "runner_fresh_three_sample_resource_preflight_required"], False)
    add(["prelaunch_gates", "runner_runtime_monitor_and_final_ack_required"], False)
    add(["post_pass_command_contract", "selector"], "M518_R4_POINT=rank3")
    add(["post_pass_command_contract", "caller_pin_runner_sha256"], "bad")
    add(["author_execution_receipt", "dc_runs"], 1)
    add(["claim_boundary", "area"], True)
    add(["claim_boundary", "system_speedup"], True)
    add(["docs359_sha256"], "f" * 64)

    rejected = 0
    for candidate in mutations:
        try:
            validate_release(candidate)
        except (AssertionError, KeyError, TypeError, ValueError):
            rejected += 1
    require(rejected == len(mutations), "release mutation escaped")
    return len(mutations)


def parser_attacks():
    duplicate = (b'{"x":1,"x":2}',
                 b'{"launch_now":true,"launch_now":false}',
                 b'{"authorization":{"run_dc":true,"run_dc":false}}')
    nonfinite = (b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}')
    for payload in duplicate + nonfinite:
        try:
            strict_load_bytes(payload)
        except ValueError:
            continue
        raise AssertionError("malformed JSON attack accepted")
    return len(duplicate), len(nonfinite)


def verify_all_json_strict():
    files = set(EXPECTED.keys())
    files.update((HANDOFF_JSON, REQUEST_JSON, M904_JSON, VCS_JSON, SPEC_JSON,
                  CONTRACT, ADMISSION, CANDIDATE, CANDIDATE_HAMMER_JSON, RELEASE))
    for directory in (HANDOFF, REQUEST, M904, VCS, SPEC, CANDIDATE_HAMMER):
        files.update(os.path.relpath(name, ROOT) for name in
                     glob.glob(path(directory) + "/**/*.json", recursive=True))
    json_files = sorted(item for item in files if item.endswith(".json"))
    for filename in json_files:
        strict_load(filename)
    return len(json_files)


def assert_static_populations():
    require(not os.path.lexists(path(CANONICAL)), "Fixed canonical result already exists")
    require(not os.path.lexists(path(ATTEMPT)), "Fixed attempt already consumed")
    require(glob.glob(path(CANONICAL + ".failed_or_incomplete.*.quarantine")) == [],
            "Fixed failure quarantine already exists")
    require(glob.glob(path("dc_handoff/runs/.m518_r4_fixed_setup_area_work.*")) == [],
            "Fixed work population already exists")
    require(glob.glob(path("dc_handoff/runs/.m518_r4_fixed_preflight.*.staging")) == [],
            "Fixed preflight population already exists")


def main():
    assert_static_populations()
    for filename, expected in EXPECTED.items():
        require(os.path.isfile(path(filename)) and not os.path.islink(path(filename)),
                "missing/nonregular/symlink identity: " + filename)
        require(sha(filename) == expected, "identity drift: " + filename)
    for filename, expected in TOOL_PATHS.items():
        require(os.path.isfile(filename) and sha_abs(filename) == expected,
                "tool/library identity drift: " + filename)

    for filename in (RELEASE, CONTRACT, ADMISSION, CANDIDATE):
        verify_file_seal(filename)
    for directory in EXPECTED_DIR_SEALS:
        verify_dir_seal(directory)

    json_count = verify_all_json_strict()
    release = strict_load(RELEASE)
    handoff = strict_load(HANDOFF_JSON)
    request = strict_load(REQUEST_JSON)
    contract = strict_load(CONTRACT)
    admission = strict_load(ADMISSION)
    m904 = strict_load(M904_JSON)
    vcs = strict_load(VCS_JSON)
    spec = strict_load(SPEC_JSON)
    validate_release(release)
    validate_chain(release, handoff, request, contract, admission, m904, vcs, spec)
    validate_source_text()
    mutation_count = mutation_attacks(release)
    duplicate_count, nonfinite_count = parser_attacks()
    assert_static_populations()

    result = {
        "status": "PASS_M906_INDEPENDENT_STATIC_NO_EDA",
        "python_version": sys.version.split()[0],
        "strict_json_files": json_count,
        "release_mutations_rejected": mutation_count,
        "duplicate_key_attacks_rejected": duplicate_count,
        "nonfinite_attacks_rejected": nonfinite_count,
        "seven_raw_outputs_plus_structured_gate": True,
        "exact_selector": "M518_R4_POINT=fixed",
        "rank3_or_paired_authorized": False,
        "c2_terminal_required_before_root_invocation": True,
        "fresh_resource_collision_gates_required": True,
        "runner_production_invocations": 0,
        "eda_runs": 0,
        "license_queries": 0,
        "attempt_or_result_created": False,
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
