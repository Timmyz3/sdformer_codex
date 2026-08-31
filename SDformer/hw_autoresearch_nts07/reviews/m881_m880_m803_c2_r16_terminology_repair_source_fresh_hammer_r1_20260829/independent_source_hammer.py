#!/usr/bin/env python3
"""Fresh independent, strictly no-EDA hammer of the M880/M803 R16 repair."""

import copy
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "dc_handoff/scripts/run_dc_m880_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh"
OLD_RUNNER = ROOT / "dc_handoff/scripts/run_dc_m872_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh"
CONTRACT = ROOT / "contracts/m880_m803_c2_r16_channel_split_three_axis_dc_source_only_contract_r1_20260829.json"
OLD_CONTRACT = ROOT / "contracts/m872_m803_c2_r16_channel_split_three_axis_dc_source_only_contract_r1_20260829.json"
CANDIDATE = ROOT / "contracts/m880_m803_c2_r16_channel_split_three_axis_dc_launch_candidate_source_only_r1_20260829.json"
OLD_CANDIDATE = ROOT / "contracts/m872_m803_c2_r16_channel_split_three_axis_dc_launch_candidate_source_only_r1_20260829.json"
AUTHOR_TEST = ROOT / "verif_m880/test_m880_m803_three_axis_dc_source_closure.py"
FILELIST = ROOT / "dc_handoff/filelists/date_m803_c2_r16_channel_split_three_axis_logic_only_dc.f"
TCL = ROOT / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
TOP = ROOT / "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv"
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


def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def strict_load(path):
    return json.loads(Path(path).read_bytes().decode("utf-8"),
                      object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


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


def assert_absent():
    require(not CANONICAL.exists(), "canonical result exists")
    require(not ATTEMPT.exists(), "attempt sentinel exists")
    require(not list((ROOT / "dc_handoff/runs").glob(
        ".m872_m803_c2_r16_channel_split_three_axis_dc_work.*")),
        "work population exists")
    require(not list((ROOT / "dc_handoff/runs").glob(
        "m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829.failed_or_incomplete.*")),
        "quarantine population exists")


def check_multi_python_strict_json(paths):
    program = r'''
import json, sys
def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate")
        out[key] = value
    return out
def nonfinite(value):
    raise ValueError("nonfinite")
def load(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=nonfinite)
for name in sys.argv[1:]:
    with open(name, "rb") as handle:
        load(handle.read())
for raw in [b'{"x":1,"x":2}',
            b'{"authorization":{"run_dc":true,"run_dc":false}}',
            b'{"identity":{"sha":"a","sha":"b"}}',
            b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}']:
    try:
        load(raw)
    except ValueError:
        pass
    else:
        raise SystemExit(91)
print("STRICT_JSON_PASS")
'''
    for executable in ["/usr/libexec/platform-python3.6",
                       "/opt/anaconda3/envs/pytorch310/bin/python3.10"]:
        completed = subprocess.run([executable, "-c", program] +
                                   [str(p) for p in paths],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        require(completed.returncode == 0 and
                b"STRICT_JSON_PASS" in completed.stdout,
                "strict JSON cross-interpreter failure: %s" % executable)


def normalized_new_runner():
    text = RUNNER.read_text()
    replacements = [
        ("contracts/m880_m803_c2_r16_channel_split_three_axis_dc_source_only_contract_r1_20260829.json",
         "contracts/m872_m803_c2_r16_channel_split_three_axis_dc_source_only_contract_r1_20260829.json"),
        ("contracts/m882_m880_m803_c2_r16_channel_split_three_axis_dc_launch_admission_r1_20260829.json",
         "contracts/m874_m872_m803_c2_r16_channel_split_three_axis_dc_launch_admission_r1_20260829.json"),
        ("contracts/m880_m803_c2_r16_channel_split_three_axis_dc_launch_candidate_source_only_r1_20260829.json",
         "contracts/m872_m803_c2_r16_channel_split_three_axis_dc_launch_candidate_source_only_r1_20260829.json"),
        ("AUTHORIZED_ONE_M880_M803_C2_R16_CHANNEL_SPLIT_THREE_AXIS_LOGIC_ONLY_DC_ATTEMPT_R1",
         "AUTHORIZED_ONE_M872_M803_C2_R16_CHANNEL_SPLIT_THREE_AXIS_LOGIC_ONLY_DC_ATTEMPT_R1"),
        ("READY_FOR_FRESH_M880_M803_C2_R16_TERMINOLOGY_REPAIR_THREE_AXIS_DC_SOURCE_HAMMER__NO_EDA_AUTHORIZED",
         "READY_FOR_FRESH_M872_M803_C2_R16_THREE_AXIS_DC_SOURCE_HAMMER__NO_EDA_AUTHORIZED"),
        ("AUTHOR_M880_M803_C2_R16_TERMINOLOGY_REPAIR_THREE_AXIS_DC_SOURCE_ONLY_COMPLETE__FRESH_HAMMER_REQUIRED__NO_EDA_AUTHORIZED",
         "AUTHOR_M872_M803_C2_R16_THREE_AXIS_DC_SOURCE_ONLY_COMPLETE__FRESH_HAMMER_REQUIRED__NO_EDA_AUTHORIZED"),
        ("dc_handoff/scripts/run_dc_m880_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh",
         "dc_handoff/scripts/run_dc_m872_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh"),
    ]
    for new, old in replacements:
        text = text.replace(new, old)
    text = text.replace(
        '\ndef reject_nonfinite(value):\n    raise ValueError("non-finite JSON constant: %s" % value)\n', '')
    text = text.replace(
        'json.loads(handle.read().decode("utf-8"), object_pairs_hook=unique_object,\n'
        '               parse_constant=reject_nonfinite)',
        'json.loads(handle.read().decode("utf-8"), object_pairs_hook=unique_object)')
    return text


def validate_candidate(contract, candidate):
    expected_identity = {
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
    require(set(candidate["identity"]) == expected_identity,
            "candidate identity key closure failed")
    require(candidate["status"] ==
            "READY_FOR_FRESH_M880_M803_C2_R16_TERMINOLOGY_REPAIR_THREE_AXIS_DC_SOURCE_HAMMER__NO_EDA_AUTHORIZED",
            "candidate status drift")
    require(candidate["launch_now"] is False, "launch_now drift")
    require(candidate["authorization"] == {
        "max_attempts": 1, "run_dc": True, "run_formality": False,
        "run_pt": False, "run_ptpx": False, "run_remote": False,
        "run_vcs": False}, "typed authorization drift")
    plan = candidate["three_axis_pre_attempt_plan"]
    require(plan == contract["three_axis_pre_attempt_plan"], "plan mismatch")
    require(plan["point_order"] == ["k1", "k8", "k1x8"], "axis order drift")
    require(plan["tim209_required_each_axis"] == 0, "TIM-209 drift")
    require(plan["opt150_required_each_axis"] == 0, "OPT-150 drift")
    require(plan["all_three_axes_same_attempt_required"] is True,
            "same attempt drift")
    require(plan["partial_axis_or_cross_attempt_reuse_citable"] is False,
            "partial result citable drift")
    identity = candidate["identity"]
    require(identity["dc_runner_sha256"] == sha(RUNNER), "runner SHA drift")
    require(identity["recovery_contract_sha256"] == sha(CONTRACT),
            "contract SHA drift")
    require(identity["dc_filelist_sha256"] == sha(FILELIST), "filelist SHA drift")
    require(identity["docs359_sha256"] == sha(ROOT / identity["docs359_path"]),
            "docs359 SHA drift")
    for key in ["m800_failure_authority", "m803_vcs_authority",
                "m873_terminology_repair_provenance"]:
        require(candidate[key] == contract[key], "%s mismatch" % key)
    require(candidate["m800_failure_authority"]["review_sha256"] ==
            "fc7ee6da76f953789296ea7acc5fd0abeaa721b7581f17be6cfaf3b755ae05ac",
            "M800 authority drift")
    require(candidate["m803_vcs_authority"]["m867_review_sha256"] ==
            "3ef4ccc8103e632ab189e9aa99276e449205b60c48b29069fe94ac48846032d3",
            "M867 authority drift")
    provenance = candidate["m873_terminology_repair_provenance"]
    require(provenance["review_sha256"] ==
            "cd78fd6ddc343c120f6ff64f02d53a2a1908ccfbb3595e51165366fc3d59416c",
            "M873 authority drift")
    require(provenance["p2_id"] ==
            "P2_STALE_R15_CURRENT_SUCCESSOR_TERMINOLOGY",
            "M873 P2 identity drift")
    unique = candidate["unique_attempt"]
    require(unique["all_three_axes_must_rerun_under_one_m872_m803_r16_attempt"]
            is True, "M872/M803 R16 same-attempt key absent")
    require("all_three_axes_must_rerun_under_one_r15_attempt" not in unique,
            "stale R15 same-attempt key present")
    require(candidate["required_next_gate"].startswith(
        "Fresh independent M880/M803 R16 terminology-repair source/static hammer"),
        "fresh next-gate terminology drift")
    forbidden = "\n".join(contract["forbidden"])
    repaired = [
        "Run the current M872/M803 R16 attempt without exact SNPSLMD_LICENSE_FILE",
        "Consume the current M872/M803 R16 attempt if license preflight is unreachable",
        "from the M880/M803 R16 source-only author package",
        "manufacture HOME in the current M872/M803 R16 runner",
        "under the current M872/M803 R16 attempt",
    ]
    stale = [
        "Run R15 without exact SNPSLMD_LICENSE_FILE and LM_LICENSE_FILE values",
        "Consume the R15 attempt if license preflight is unreachable",
        "from the R15 source-only author package",
        "manufacture HOME in the R15 runner",
        "under the R15 attempt",
    ]
    for phrase in repaired:
        require(phrase in forbidden, "repaired forbidden term absent: %s" % phrase)
    for phrase in stale:
        require(phrase not in forbidden, "stale prospective term remains: %s" % phrase)


def mutation_rejected(contract, candidate, mutator):
    changed = copy.deepcopy(candidate)
    mutator(changed)
    try:
        validate_candidate(contract, changed)
    except (RuntimeError, KeyError, TypeError):
        return True
    return False


def run_full_path(expected_sha):
    root = tempfile.mkdtemp(prefix="m881_fullpath.", dir="/tmp")
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
        "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
        "M872_M803_DC_NO_EDA_FULL_PATH_SELF_TEST": "1",
        "M872_M803_DC_FULL_PATH_SELF_TEST_ROOT": root,
        "M872_M803_DC_EXPECTED_DC_RUNNER_SHA256": expected_sha,
        "M872_M803_DC_EXPECTED_DC_LAUNCH_ADMISSION_SHA256": sha(CANDIDATE),
    }
    completed = subprocess.run([str(RUNNER)], cwd=str(ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return Path(root), completed


def main():
    assert_absent()
    expected = {
        RUNNER: "3f5553cac5ccd61e87fe7e76bb5febc988c429ee5f36be7f23953879e402212e",
        CONTRACT: "70c65ee56e8147de242081376e3da3cd73ac7b39ee0520aaaa7a8942808f6ee4",
        CANDIDATE: "941f38419acb013ea2804dc88a25e607b35846a4855a7cf2cac950a1f7fafec2",
        AUTHOR_TEST: "192f97eb00fb1238ce511e7b7ff74dd4fe8e935b561ba1649717b62be145b5f7",
        ROOT / "docs/359_DATE终局冻结_20260813.md":
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    }
    for path, digest in expected.items():
        require(sha(path) == digest, "target SHA drift: %s" % path)
    for path in [RUNNER, CONTRACT, CANDIDATE]:
        verify_file_seal(path)

    trees = [
        ROOT / "reviews/m880_m803_c2_r16_terminology_repair_source_author_handoff_r1_20260829",
        ROOT / "reviews/m881_m880_m803_c2_r16_terminology_repair_source_fresh_hammer_REQUEST_r1_20260829",
        ROOT / "reviews/m873_m872_m803_c2_r16_three_axis_dc_source_fresh_hammer_r1_20260829",
        ROOT / "reviews/m872_m803_c2_r16_three_axis_dc_source_author_handoff_r1_20260829",
        ROOT / "reviews/m800_m519_r15_k8_tim209_failure_hammer_r1_20260828",
        ROOT / "reviews/m867_m859_c2_r25_shared_whitelist_vcs_result_hammer_r1_20260829",
        ROOT / "results/m859_c2_r25_shared_whitelist_vcs_r1_20260829",
    ]
    for tree in trees:
        verify_tree(tree)

    handoff = trees[0] / "handoff.json"
    request = trees[1] / "request.json"
    require(sha(handoff) ==
            "04f8f7a639f141a414159028640bc817b37a4e72533da71bdec4a050c807eb9b",
            "handoff SHA drift")
    require(sha(request) ==
            "cb0e5f9a1bcde1463c8e5babf59abef16ddd362b59e08c89a45a4bdc7260fb20",
            "request SHA drift")
    cited_json = [CONTRACT, CANDIDATE, OLD_CONTRACT, OLD_CANDIDATE, handoff,
                   request, trees[2] / "review.json", trees[3] / "handoff.json",
                   trees[4] / "review.json", trees[5] / "review.json",
                   trees[6] / "m859_c2_r25_shared_whitelist_vcs_receipt_r1.json"]
    check_multi_python_strict_json(cited_json)

    contract = strict_load(CONTRACT)
    candidate = strict_load(CANDIDATE)
    old_contract = strict_load(OLD_CONTRACT)
    old_candidate = strict_load(OLD_CANDIDATE)
    validate_candidate(contract, candidate)

    # The fresh runner changes only identity bindings and strict nonfinite JSON
    # rejection.  Normalizing exactly those edits must reproduce M872 bytewise.
    normalized = normalized_new_runner()
    if normalized != OLD_RUNNER.read_text():
        diff = "".join(difflib.unified_diff(
            OLD_RUNNER.read_text().splitlines(True), normalized.splitlines(True),
            fromfile="M872", tofile="normalized-M880"))
        raise RuntimeError("runner production behavior drift:\n" + diff[:4000])

    # Historical R15 provenance remains semantic history.  The current-label
    # repair is deliberately excluded from this historical subset comparison.
    for key in ["m800_failure_authority", "r15_atomic_artifact_gate_repair_provenance"]:
        require(contract[key] == old_contract[key],
                "contract historical R15 provenance drift: %s" % key)
        require(candidate[key] == old_candidate[key],
                "candidate historical R15 provenance drift: %s" % key)
    require(contract["m800_failure_authority"]["r15_attempt_consumed"] is True,
            "historical R15 attempt-consumed fact absent")

    require(len(contract["exact_files"]) == 17, "exact_files count drift")
    for relative, digest in contract["exact_files"].items():
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(),
                "exact file is absent/nonregular/symlink: %s" % relative)
        require(sha(path) == digest, "exact file hash drift: %s" % relative)
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
    rtl = [line.strip() for line in FILELIST.read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    require(rtl == expected_rtl and len(rtl) == len(set(rtl)),
            "12-file RTL closure drift")
    top = TOP.read_text()
    positions = [top.index("if (ARCH_MODE == 0) begin : g_k1"),
                 top.index("else if (ARCH_MODE == 1) begin : g_k8"),
                 top.index("else if (ARCH_MODE == 2) begin : g_k1x8")]
    require(positions == sorted(positions), "ARCH branch order drift")
    for branch, module in [
        (positions[0], "m519_fc2_k1_registered_release_8bank_raw4_acc24"),
        (positions[1], "m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24"),
        (positions[2], "m519_fc2_k1x8_registered_release_raw4_acc24")]:
        require(top.index(module, branch) > branch, "ARCH binding drift")

    runner_text = RUNNER.read_text()
    require(re.search(r"m872_m803_dc_run_point\s+k1\s+0\s*\n.*?m872_m803_dc_run_point\s+k8\s+1\s*\n.*?m872_m803_dc_run_point\s+k1x8\s+2",
                      runner_text, re.S), "runner axis binding/order drift")
    definitions = {m.group(1): runner_text.count("\n", 0, m.start()) + 1
                   for m in re.finditer(
                       r"^(m872_m803_dc_[a-z0-9_]+)\(\)\s*\{", runner_text, re.M)}
    require(len(definitions) == 36, "runner function population drift")
    tcl = TCL.read_text()
    ordered = ["analyze -format sverilog", "elaborate $design_name",
               'check_design > "$output_dir/reports/check_design_precompile.rpt"',
               "redirect $precompile_timing_report {check_timing}",
               "if {$precompile_tim209_count != 0 || $precompile_opt150_count != 0}",
               "\n    compile_ultra\n"]
    indexes = [tcl.index(token) for token in ordered]
    require(indexes == sorted(indexes) and tcl.count("\n    compile_ultra\n") == 1,
            "Tcl precompile gate/order drift")
    require("exit 36" in tcl and "TIM-209=$precompile_tim209_count" in tcl and
            "OPT-150=$precompile_opt150_count" in tcl,
            "TIM-209/OPT-150 gate drift")

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
        lambda c: c["three_axis_pre_attempt_plan"].update({"all_three_axes_same_attempt_required": False}),
        lambda c: c["three_axis_pre_attempt_plan"].update({"partial_axis_or_cross_attempt_reuse_citable": True}),
        lambda c: c["m800_failure_authority"].update({"review_sha256": "0" * 64}),
        lambda c: c["m803_vcs_authority"].update({"m867_review_sha256": "0" * 64}),
        lambda c: c["m873_terminology_repair_provenance"].update({"review_sha256": "0" * 64}),
        lambda c: c["unique_attempt"].update({"all_three_axes_must_rerun_under_one_r15_attempt": True}),
        lambda c: c.update({"required_next_gate": "Fresh independent R15 source/static hammer"}),
        lambda c: c["authorization"].update({"run_dc": False}),
        lambda c: c.update({"launch_now": True}),
        lambda c: c.update({"status": "READY_BUT_WRONG"}),
    ]
    for index, mutator in enumerate(mutations):
        require(mutation_rejected(contract, candidate, mutator),
                "semantic mutation accepted: %d" % index)

    stale_contract = copy.deepcopy(contract)
    stale_candidate = copy.deepcopy(candidate)
    stale_candidate["unique_attempt"].pop(
        "all_three_axes_must_rerun_under_one_m872_m803_r16_attempt")
    stale_candidate["unique_attempt"][
        "all_three_axes_must_rerun_under_one_r15_attempt"] = True
    stale_candidate["required_next_gate"] = "Fresh independent R15 source/static hammer"
    stale_contract["forbidden"][15] = "Run R15 without exact SNPSLMD_LICENSE_FILE and LM_LICENSE_FILE values"
    stale_contract["forbidden"][16] = "Consume the R15 attempt if license preflight is unreachable"
    stale_contract["forbidden"][20] = "Create launch_now=true release or run any EDA from the R15 source-only author package"
    stale_contract["forbidden"][27] = "Set synthesize inherit or otherwise manufacture HOME in the R15 runner"
    stale_contract["forbidden"][30] = "Run fewer than all three axes under the R15 attempt"
    require(mutation_rejected(stale_contract, stale_candidate, lambda c: None),
            "aggregate stale-R15 reinjection accepted")

    subprocess.check_call(["bash", "-n", str(RUNNER)])
    # The complete author closure is not trusted as a conclusion, but is
    # replayed as an executable test vector on both required interpreters.
    for executable in ["/usr/libexec/platform-python3.6",
                       "/opt/anaconda3/envs/pytorch310/bin/python3.10"]:
        completed = subprocess.run([executable, str(AUTHOR_TEST)], cwd=str(ROOT),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(completed.returncode == 0 and
                b"PASS_M880_M803_C2_R16_TERMINOLOGY_REPAIR_SOURCE_CLOSURE" in completed.stdout,
                "full no-EDA author closure failed under %s: %s" %
                (executable, completed.stderr.decode("utf-8", "replace")[-1000:]))

    artifact_root = Path(tempfile.mkdtemp(prefix="m881_artifact.", dir="/tmp"))
    full_root = wrong_root = None
    try:
        artifact_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "M872_M803_DC_ARTIFACT_GATE_NO_EDA_SELF_TEST": "1",
            "M872_M803_DC_ARTIFACT_GATE_SELF_TEST_ROOT": str(artifact_root),
        }
        completed = subprocess.run([str(RUNNER)], cwd=str(ROOT), env=artifact_env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(completed.returncode == 0, "artifact self-test failed")
        artifact = (artifact_root / "ARTIFACT_GATE_SELF_TEST_PASS.txt").read_text()
        for token in ["positive_cases=1", "negative_cases=25",
                      "deleted_cases=7", "zero_byte_cases=7",
                      "leaf_symlink_cases=7", "partial_publish_cases=1",
                      "ancestor_symlink_cases=1", "path_escape_cases=1",
                      "postreceipt_mutation_cases=1", "final_manifest_positive_cases=1"]:
            require(token in artifact, "artifact token absent: %s" % token)

        full_root, completed = run_full_path(sha(RUNNER))
        require(completed.returncode == 0, "full no-EDA candidate/contract path failed")
        full = (full_root / "FULL_PATH_PASS.txt").read_text()
        for token in ["preflight_started=false", "attempt_consumed=false",
                      "dc_shell_started=false",
                      "three_axis_source_plan=PASS_K1_M803K8_K1X8_TIM209_OPT150_PRECOMPILE_GATE"]:
            require(token in full, "full-path token absent: %s" % token)

        wrong_root, completed = run_full_path("0" * 64)
        require(completed.returncode == 3, "wrong-runner-SHA did not return 3")
        receipts = list(wrong_root.glob("m872_m803_dc_pre_attempt_shell_failure.*.receipt"))
        require(len(receipts) == 1, "wrong-SHA receipt population drift")
        verify_tree(receipts[0])
        failure = (receipts[0] / "FAILURE.txt").read_text()
        require("exit_code=3" in failure and "attempt_consumed=false" in failure,
                "wrong-SHA receipt semantics drift")
    finally:
        for temporary in [artifact_root, full_root, wrong_root]:
            if temporary is not None and temporary.exists():
                shutil.rmtree(str(temporary))

    assert_absent()
    print("PASS100_M880_M803_C2_R16_TERMINOLOGY_REPAIR_SOURCE_FRESH_HAMMER")
    print("p0=0 p1=0 p2=0")
    print("exact_files=17 rtl_files=12 arch_bindings=0/1/2 functions=36")
    print("python36_full_no_eda=PASS python310_full_no_eda=PASS")
    print("duplicate_negatives=3 nonfinite_negatives=3 semantic_mutations=19 stale_r15_aggregate=1")
    print("artifact_positive=1 artifact_negatives=25 wrong_runner_sha=PASS_RC3_DOUBLE_SEALED")
    print("canonical_absent=true attempt_absent=true work_absent=true quarantine_absent=true")
    print("dc_runs=0 vcs_runs=0 license_queries=0 remote_runs=0")


if __name__ == "__main__":
    main()
