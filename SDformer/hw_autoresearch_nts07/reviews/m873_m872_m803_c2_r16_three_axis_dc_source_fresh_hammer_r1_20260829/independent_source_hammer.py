#!/usr/bin/env python3
"""Independent no-EDA source hammer for M872/M803 C2 R16.

This reviewer never invokes an EDA executable, simulator, license query,
remote command, or the production branch of the runner.
"""

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "dc_handoff/scripts/run_dc_m872_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh"
CONTRACT = ROOT / "contracts/m872_m803_c2_r16_channel_split_three_axis_dc_source_only_contract_r1_20260829.json"
CANDIDATE = ROOT / "contracts/m872_m803_c2_r16_channel_split_three_axis_dc_launch_candidate_source_only_r1_20260829.json"
FILELIST = ROOT / "dc_handoff/filelists/date_m803_c2_r16_channel_split_three_axis_logic_only_dc.f"
TCL = ROOT / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
TOP = ROOT / "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv"
AUTHOR_TEST = ROOT / "verif_m872/test_m872_m803_three_axis_dc_source_closure.py"
CANONICAL = ROOT / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
ATTEMPT = ROOT / "dc_handoff/runs/.m872_m803_c2_r16_channel_split_three_axis_dc_attempt_consumed"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def strict_load(path):
    return json.loads(Path(path).read_bytes().decode("utf-8"),
                      object_pairs_hook=unique_object)


def verify_tree(path):
    path = Path(path)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"], cwd=str(path),
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"],
                          cwd=str(path), stdout=subprocess.DEVNULL)


def verify_file_seal(path):
    path = Path(path)
    subprocess.check_call(["sha256sum", "-c", path.name + ".sha256"],
                          cwd=str(path.parent), stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", path.name + ".sha256.seal.sha256"],
                          cwd=str(path.parent), stdout=subprocess.DEVNULL)


def validate_candidate(contract, candidate):
    required_identity = {
        "dc_actual_exec_path", "dc_actual_exec_sha256", "dc_filelist_path",
        "dc_filelist_sha256", "dc_runner_path", "dc_runner_sha256",
        "dc_shell_path", "dc_shell_sha256", "dc_tcl_path", "dc_tcl_sha256",
        "dc_wrapper_path", "dc_wrapper_sha256", "docs359_path",
        "docs359_sha256", "fast_lib_path", "fast_lib_sha256", "lmutil_path",
        "lmutil_sha256", "r5_final_failure_review_outer_seal_file_sha256",
        "r5_final_failure_review_path", "r5_quarantine_outer_seal_file_sha256",
        "r5_quarantine_path", "r5_static_review_outer_seal_file_sha256",
        "r5_static_review_path", "r5_vcs_result_outer_seal_file_sha256",
        "r5_vcs_result_path", "r5_vcs_review_outer_seal_file_sha256",
        "r5_vcs_review_path", "r6_static_review_outer_seal_file_sha256",
        "r6_static_review_path", "r7_disqualified_review_outer_seal_file_sha256",
        "r7_disqualified_review_path", "recovery_contract_path",
        "recovery_contract_sha256", "sdc_path", "sdc_sha256", "slow_lib_path",
        "slow_lib_sha256", "snps_license_file_path", "snps_license_file_sha256",
    }
    require(set(candidate["identity"]) == required_identity,
            "candidate identity keys are not closed")
    require(candidate["status"] ==
            "READY_FOR_FRESH_M872_M803_C2_R16_THREE_AXIS_DC_SOURCE_HAMMER__NO_EDA_AUTHORIZED",
            "candidate status drift")
    require(candidate["launch_now"] is False, "candidate launch_now not false")
    require(candidate["authorization"] == {
        "max_attempts": 1, "run_dc": True, "run_formality": False,
        "run_pt": False, "run_ptpx": False, "run_remote": False,
        "run_vcs": False,
    }, "typed authorization drift")
    plan = candidate["three_axis_pre_attempt_plan"]
    require(plan == contract["three_axis_pre_attempt_plan"],
            "candidate/contract axis plan mismatch")
    require(plan["point_order"] == ["k1", "k8", "k1x8"], "axis order drift")
    require(plan["tim209_required_each_axis"] == 0, "TIM-209 gate drift")
    require(plan["opt150_required_each_axis"] == 0, "OPT-150 gate drift")
    require(plan["all_three_axes_same_attempt_required"] is True,
            "same-attempt requirement drift")
    require(plan["partial_axis_or_cross_attempt_reuse_citable"] is False,
            "partial/cross-attempt result became citable")
    require(candidate["identity"]["dc_runner_sha256"] == sha(RUNNER),
            "runner SHA binding drift")
    require(candidate["identity"]["recovery_contract_sha256"] == sha(CONTRACT),
            "contract SHA binding drift")
    require(candidate["identity"]["dc_filelist_sha256"] == sha(FILELIST),
            "filelist SHA binding drift")
    require(candidate["identity"]["docs359_sha256"] == sha(ROOT / candidate["identity"]["docs359_path"]),
            "docs359 SHA binding drift")
    require(candidate["m800_failure_authority"] == contract["m800_failure_authority"],
            "M800 authority mismatch")
    require(candidate["m803_vcs_authority"] == contract["m803_vcs_authority"],
            "M867/R25 authority mismatch")


def mutation_rejected(contract, candidate, mutator):
    changed = copy.deepcopy(candidate)
    mutator(changed)
    try:
        validate_candidate(contract, changed)
    except (RuntimeError, KeyError, TypeError):
        return True
    return False


def check_tcl(text):
    tokens = [
        "analyze -format sverilog",
        "elaborate $design_name",
        'check_design > "$output_dir/reports/check_design_precompile.rpt"',
        "redirect $precompile_timing_report {check_timing}",
        "if {$precompile_tim209_count != 0 || $precompile_opt150_count != 0}",
        "\n    compile_ultra\n",
    ]
    locations = [text.index(token) for token in tokens]
    require(locations == sorted(locations) and len(set(locations)) == len(locations),
            "Tcl analyze/elaborate/design/timing/loop-gate/compile order drift")
    require(text.count("\n    compile_ultra\n") == 1,
            "Tcl compile_ultra count is not one")
    require("exit 36" in text and "TIM-209=$precompile_tim209_count" in text and
            "OPT-150=$precompile_opt150_count" in text,
            "Tcl explicit loop failure gate incomplete")
    for item in [
        "write_file -format verilog", "write_sdc", "write -format ddc",
        "set_svf", "report_area -hierarchy", "report_qor",
        "report_timing -delay_type max",
    ]:
        require(item in text, "Tcl required output producer missing: %s" % item)


def run_no_eda_full_path(expected_runner_sha):
    root = tempfile.mkdtemp(prefix="m878_fullpath.", dir="/tmp")
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
        "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
        "M872_M803_DC_NO_EDA_FULL_PATH_SELF_TEST": "1",
        "M872_M803_DC_FULL_PATH_SELF_TEST_ROOT": root,
        "M872_M803_DC_EXPECTED_DC_RUNNER_SHA256": expected_runner_sha,
        "M872_M803_DC_EXPECTED_DC_LAUNCH_ADMISSION_SHA256": sha(CANDIDATE),
    }
    completed = subprocess.run([str(RUNNER)], cwd=str(ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return root, completed


def main():
    require(not CANONICAL.exists(), "canonical result exists")
    require(not ATTEMPT.exists(), "attempt sentinel exists")
    require(not list((ROOT / "dc_handoff/runs").glob(".m872_m803_c2_r16_channel_split_three_axis_dc_work.*")),
            "work identity exists")
    require(not list((ROOT / "dc_handoff/runs").glob("m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829.failed_or_incomplete.*")),
            "quarantine identity exists")

    require(sha(RUNNER) == "2f4b8592b01df4a74bd2180f2b0947e4fefe5ba5a5cc4f5a2b4f00673725d327",
            "runner SHA drift")
    require(sha(CONTRACT) == "76f333b7652ce1ba562b23a3262154062b01bcb5eb5e176c2bfce2d1c28812a6",
            "contract SHA drift")
    require(sha(CANDIDATE) == "2ed67244518c2275e244dd47ad67f7ded775e8fc2ce79689f15acbaddaa94181",
            "candidate SHA drift")
    require(sha(ROOT / "docs/359_DATE\u7ec8\u5c40\u51bb\u7ed3_20260813.md") ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "docs359 SHA drift")
    verify_file_seal(RUNNER)
    verify_file_seal(CONTRACT)
    verify_file_seal(CANDIDATE)
    for path in [
        ROOT / "reviews/m872_m803_c2_r16_three_axis_dc_source_author_handoff_r1_20260829",
        ROOT / "reviews/m873_m872_m803_c2_r16_three_axis_dc_source_fresh_hammer_REQUEST_r1_20260829",
        ROOT / "reviews/m800_m519_r15_k8_tim209_failure_hammer_r1_20260828",
        ROOT / "reviews/m867_m859_c2_r25_shared_whitelist_vcs_result_hammer_r1_20260829",
        ROOT / "results/m859_c2_r25_shared_whitelist_vcs_r1_20260829",
    ]:
        verify_tree(path)

    contract = strict_load(CONTRACT)
    candidate = strict_load(CANDIDATE)
    validate_candidate(contract, candidate)
    require(len(contract["exact_files"]) == 17, "exact_files count drift")
    for relative, expected in contract["exact_files"].items():
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(),
                "exact file absent/not-regular/symlink: %s" % relative)
        require(sha(path) == expected, "exact file SHA drift: %s" % relative)

    expected_rtl = [
        "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv",
        "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
        "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
        "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
        "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
        "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
        "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
        "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
        "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv",
        "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
    ]
    actual_rtl = [line.strip() for line in FILELIST.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    require(actual_rtl == expected_rtl and len(actual_rtl) == len(set(actual_rtl)),
            "filelist membership/order/duplicate closure failed")
    for relative in actual_rtl:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(),
                "filelist member not regular nonsymlink: %s" % relative)

    top = TOP.read_text()
    bindings = [
        ("if (ARCH_MODE == 0) begin : g_k1", "m519_fc2_k1_registered_release_8bank_raw4_acc24"),
        ("else if (ARCH_MODE == 1) begin : g_k8", "m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24"),
        ("else if (ARCH_MODE == 2) begin : g_k1x8", "m519_fc2_k1x8_registered_release_raw4_acc24"),
    ]
    positions = []
    for branch, module in bindings:
        b = top.index(branch)
        m = top.index(module, b)
        positions.append((b, m))
    require(positions[0][0] < positions[1][0] < positions[2][0], "ARCH branch order drift")
    runner_text = RUNNER.read_text()
    require(re.search(r"m872_m803_dc_run_point\s+k1\s+0\s*\n.*?m872_m803_dc_run_point\s+k8\s+1\s*\n.*?m872_m803_dc_run_point\s+k1x8\s+2", runner_text, re.S),
            "runner axis invocation order/binding drift")

    definitions = {m.group(1): runner_text.count("\n", 0, m.start()) + 1
                   for m in re.finditer(r"^(m872_m803_dc_[a-z0-9_]+)\(\)\s*\{", runner_text, re.M)}
    require(len(definitions) == 36, "runner function population drift")
    for match in re.finditer(r"(?<![A-Za-z0-9_])(m872_m803_dc_[a-z0-9_]+)(?=[ \t\n])", runner_text):
        name = match.group(1)
        line = runner_text.count("\n", 0, match.start()) + 1
        if name in definitions and line != definitions[name]:
            require(definitions[name] < line, "function used before definition: %s" % name)
    check_tcl(TCL.read_text())
    for token in [
        "artifact_count=7",
        "mapped_verilog mapped_sdc ddc svf area_report qor_report setup_timing_report",
        "m872_m803_dc_run_point k1 0", "m872_m803_dc_run_point k8 1",
        "m872_m803_dc_run_point k1x8 2", "TIM-209=0", "OPT-150=0",
    ]:
        require(token in runner_text, "runner hard gate missing: %s" % token)

    mutation_count = 0
    raw_duplicate_cases = [
        b'{"status":"A","status":"B"}',
        b'{"authorization":{"run_dc":true,"run_dc":false}}',
        b'{"identity":{"dc_runner_sha256":"a","dc_runner_sha256":"b"}}',
    ]
    for payload in raw_duplicate_cases:
        try:
            json.loads(payload.decode(), object_pairs_hook=unique_object)
        except ValueError:
            mutation_count += 1
        else:
            raise RuntimeError("duplicate JSON mutation accepted")
    mutations = [
        lambda c: c["identity"].pop("dc_runner_sha256"),
        lambda c: c["identity"].update({"unknown_identity_key": "x"}),
        lambda c: c["identity"].update({"dc_runner_sha256": "0" * 64}),
        lambda c: c["identity"].update({"recovery_contract_sha256": "0" * 64}),
        lambda c: c["identity"].update({"dc_filelist_sha256": "0" * 64}),
        lambda c: c["identity"].update({"docs359_sha256": "0" * 64}),
        lambda c: c["three_axis_pre_attempt_plan"].update({"point_order": ["k8", "k1", "k1x8"]}),
        lambda c: c["three_axis_pre_attempt_plan"].update({"tim209_required_each_axis": 1}),
        lambda c: c["three_axis_pre_attempt_plan"].update({"opt150_required_each_axis": 1}),
        lambda c: c["three_axis_pre_attempt_plan"].update({"partial_axis_or_cross_attempt_reuse_citable": True}),
        lambda c: c["m800_failure_authority"].update({"review_sha256": "0" * 64}),
        lambda c: c["m803_vcs_authority"].update({"m867_review_sha256": "0" * 64}),
    ]
    for mutator in mutations:
        require(mutation_rejected(contract, candidate, mutator), "semantic mutation accepted")
        mutation_count += 1

    bad_tcl = TCL.read_text().replace("\n    compile_ultra\n", "\ncompile_ultra\n", 1)
    try:
        check_tcl(bad_tcl)
    except (RuntimeError, ValueError):
        mutation_count += 1
    else:
        raise RuntimeError("compile-order Tcl mutation accepted")

    # Execute author test in the system Python 3.6 and a second independent,
    # newer interpreter.  This host has Python 3.12 rather than 3.10.
    for executable in ["/usr/libexec/platform-python3.6", "/usr/bin/python3.12"]:
        subprocess.check_call([executable, str(AUTHOR_TEST)], cwd=str(ROOT),
                              stdout=subprocess.DEVNULL)

    artifact_root = tempfile.mkdtemp(prefix="m878_artifact.", dir="/tmp")
    artifact_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "M872_M803_DC_ARTIFACT_GATE_NO_EDA_SELF_TEST": "1",
        "M872_M803_DC_ARTIFACT_GATE_SELF_TEST_ROOT": artifact_root,
    }
    subprocess.check_call([str(RUNNER)], cwd=str(ROOT), env=artifact_env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    artifact_text = (Path(artifact_root) / "ARTIFACT_GATE_SELF_TEST_PASS.txt").read_text()
    for token in ["positive_cases=1", "negative_cases=25", "deleted_cases=7",
                  "zero_byte_cases=7", "leaf_symlink_cases=7",
                  "partial_publish_cases=1", "ancestor_symlink_cases=1",
                  "path_escape_cases=1", "postreceipt_mutation_cases=1",
                  "final_manifest_positive_cases=1"]:
        require(token in artifact_text, "artifact self-test token absent: %s" % token)

    full_root, full = run_no_eda_full_path(sha(RUNNER))
    require(full.returncode == 0, "full no-EDA path failed")
    full_text = (Path(full_root) / "FULL_PATH_PASS.txt").read_text()
    for token in ["preflight_started=false", "attempt_consumed=false",
                  "dc_shell_started=false",
                  "three_axis_source_plan=PASS_K1_M803K8_K1X8_TIM209_OPT150_PRECOMPILE_GATE"]:
        require(token in full_text, "full no-EDA path token absent: %s" % token)

    wrong_root, wrong = run_no_eda_full_path("0" * 64)
    require(wrong.returncode == 3, "wrong runner SHA did not return 3")
    receipts = list(Path(wrong_root).glob("m872_m803_dc_pre_attempt_shell_failure.*.receipt"))
    require(len(receipts) == 1, "wrong-SHA failure receipt population drift")
    verify_tree(receipts[0])
    failure = (receipts[0] / "FAILURE.txt").read_text()
    require("exit_code=3" in failure and "attempt_consumed=false" in failure,
            "wrong-SHA failure receipt semantic drift")

    require(not CANONICAL.exists() and not ATTEMPT.exists(),
            "canonical/attempt appeared during no-EDA hammer")
    require(not list((ROOT / "dc_handoff/runs").glob(".m872_m803_c2_r16_channel_split_three_axis_dc_work.*")),
            "work identity appeared during no-EDA hammer")
    require(not list((ROOT / "dc_handoff/runs").glob("m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829.failed_or_incomplete.*")),
            "quarantine identity appeared during no-EDA hammer")

    for temporary in [artifact_root, full_root, wrong_root]:
        shutil.rmtree(temporary)

    # The executable gates are closed, but the new M872/R16 contract still
    # contains current-successor obligations named as R15.  These are not the
    # legitimate historical r15_* provenance keys: they are prospective
    # instructions that falsely name the already-consumed predecessor attempt.
    stale_contract_phrases = [
        "Run R15 without exact SNPSLMD_LICENSE_FILE and LM_LICENSE_FILE values",
        "Consume the R15 attempt if license preflight is unreachable",
        "Create launch_now=true release or run any EDA from the R15 source-only author package",
        "manufacture HOME in the R15 runner",
        "under the R15 attempt",
    ]
    contract_text = CONTRACT.read_text()
    for phrase in stale_contract_phrases:
        require(phrase in contract_text, "expected stale-label evidence moved: %s" % phrase)
    require("all_three_axes_must_rerun_under_one_r15_attempt" in CANDIDATE.read_text(),
            "expected stale candidate attempt label moved")
    require("Fresh independent R15 source/static hammer" in CANDIDATE.read_text(),
            "expected stale candidate next-gate label moved")

    print("FAIL98_M872_M803_C2_R16_THREE_AXIS_DC_SOURCE_FRESH_HAMMER__RETURN_TO_AUTHOR")
    print("p0=0 p1=0 p2=1")
    print("exact_files=17 rtl_files=12 functions=36")
    print("python36=PASS python312_as_modern_crosscheck=PASS python310_unavailable_on_host=true")
    print("strict_duplicate_negatives=3 semantic_mutations=%d" % mutation_count)
    print("artifact_positive=1 artifact_negatives=25 seven_artifacts_each_axis=true")
    print("full_no_eda_path=PASS wrong_runner_sha=PASS_RC3_DOUBLE_SEALED")
    print("axis_bindings=K1_ARCH0_M803K8_ARCH1_K1X8_ARCH2")
    print("tim209_opt150_precompile_gate=PASS")
    print("canonical_absent=true attempt_absent=true quarantine_absent=true")
    print("dc_runs=0 vcs_runs=0 license_queries=0 remote_runs=0")
    print("p2_id=P2_STALE_R15_CURRENT_SUCCESSOR_TERMINOLOGY")


if __name__ == "__main__":
    main()
