#!/usr/bin/env python3
"""Independent no-EDA final-release hammer for M882/M880/M803 C2 R16."""

from __future__ import print_function

import copy
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
REL = "contracts/m882_m880_m803_c2_r16_channel_split_three_axis_dc_launch_admission_r1_20260829.json"
CAND = "contracts/m880_m803_c2_r16_channel_split_three_axis_dc_launch_candidate_source_only_r1_20260829.json"
CONTRACT = "contracts/m880_m803_c2_r16_channel_split_three_axis_dc_source_only_contract_r1_20260829.json"
RUNNER = "dc_handoff/scripts/run_dc_m880_m803_c2_r16_channel_split_three_axis_exact_sha_r1.sh"
HANDOFF_DIR = "reviews/m885_m882_m803_c2_r16_true_release_author_handoff_r1_20260829"
REQUEST_DIR = "reviews/m886_m882_m803_c2_r16_final_release_hammer_REQUEST_r1_20260829"
M881_DIR = "reviews/m881_m880_m803_c2_r16_terminology_repair_source_fresh_hammer_r1_20260829"
M880_DIR = "reviews/m880_m803_c2_r16_terminology_repair_source_author_handoff_r1_20260829"
M881_REQUEST_DIR = "reviews/m881_m880_m803_c2_r16_terminology_repair_source_fresh_hammer_REQUEST_r1_20260829"
OUTPUT_DIR = "reviews/m886_m882_m803_c2_r16_final_release_hammer_r1_20260829"
CANONICAL = "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
ATTEMPT = "dc_handoff/runs/.m872_m803_c2_r16_channel_split_three_axis_dc_attempt_consumed"
TCL = "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"

EXPECTED = {
    REL: "c565f9d59dc0c65cab9b4e4d70f8624e69c2c1d1ac0a3ec5a93fe0901f878b0c",
    REL + ".sha256": "056e0bed1dacaa852be6368f119719ba3b1c7a9db044b6b50681b4a612505170",
    REL + ".sha256.seal.sha256": "1f6a466ae3907bef2ae6358daa59786a9996312c123a71d876d8ba3060c395c5",
    HANDOFF_DIR + "/handoff.json": "aa1b093b427891493ede52a68affb76abee1e685df1e0e60b15e3df865f38803",
    HANDOFF_DIR + "/SHA256SUMS": "acea190ff8e23e243e964d552c538f58018e3a48b1589b2b5fb426441f7dbdf0",
    HANDOFF_DIR + "/SHA256SUMS.seal.sha256": "3f98d8b1f81f3f3d591f74ae6e492dd1967557acc0ed293e8c7645a2df59bfcf",
    REQUEST_DIR + "/request.json": "a6355e8d50db1609ea582c651e9052a233567034fc88b80952a211ae55d041aa",
    REQUEST_DIR + "/SHA256SUMS": "f269816f88d57b93dc57e408c80177b4510bf4250f065aeb8ab31c5e519a3488",
    REQUEST_DIR + "/SHA256SUMS.seal.sha256": "9d015a81f4eaaba1d1881644f99740b0614930571135ab6f20487b7c8557c6e9",
    RUNNER: "3f5553cac5ccd61e87fe7e76bb5febc988c429ee5f36be7f23953879e402212e",
    CONTRACT: "70c65ee56e8147de242081376e3da3cd73ac7b39ee0520aaaa7a8942808f6ee4",
    CAND: "941f38419acb013ea2804dc88a25e607b35846a4855a7cf2cac950a1f7fafec2",
    M881_DIR + "/review.json": "1c0ba000f182fe5184a870c11ceccb48b76723314d85263c65452948e62a548d",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def path(relative):
    return os.path.join(ROOT, relative)


def sha(relative):
    digest = hashlib.sha256()
    with open(path(relative), "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key: %s" % key)
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("nonfinite: %s" % value)


def strict_load(relative):
    with open(path(relative), "rb") as handle:
        return json.loads(handle.read().decode("utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=reject_nonfinite)


def verify_tree(relative):
    directory = path(relative)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"], cwd=directory,
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"],
                          cwd=directory, stdout=subprocess.DEVNULL)


def verify_file(relative):
    base = os.path.basename(relative)
    directory = os.path.dirname(path(relative))
    subprocess.check_call(["sha256sum", "-c", base + ".sha256"], cwd=directory,
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", base + ".sha256.seal.sha256"],
                          cwd=directory, stdout=subprocess.DEVNULL)


def validate_release(release, candidate, contract, m881):
    require(type(release) is dict, "release type")
    require(type(release.get("launch_now")) is bool and release["launch_now"] is True,
            "release launch")
    require(release.get("status") ==
            "AUTHORIZED_ONE_M880_M803_C2_R16_CHANNEL_SPLIT_THREE_AXIS_LOGIC_ONLY_DC_ATTEMPT_R1",
            "release status")
    require(release.get("authorization") == {
        "max_attempts": 1, "run_dc": True, "run_formality": False,
        "run_pt": False, "run_ptpx": False, "run_remote": False,
        "run_vcs": False}, "closed typed authorization")
    require(type(release["authorization"]["max_attempts"]) is int,
            "max attempts bool/int confusion")
    binding = release["candidate_binding"]
    require(binding["candidate_sha256"] == EXPECTED[CAND], "candidate SHA pin")
    require(binding["candidate_launch_now"] is False, "candidate launch pin")
    require(candidate["launch_now"] is False, "candidate must remain inert")
    require(candidate["status"] == binding["candidate_exact_status"],
            "candidate status pin")
    require(len(binding["preserved_semantic_sections"]) == 18,
            "preserved section population")
    for key in binding["preserved_semantic_sections"]:
        require(release[key] == candidate[key], "section mismatch: %s" % key)
    require(release["three_axis_pre_attempt_plan"] ==
            contract["three_axis_pre_attempt_plan"], "contract plan mismatch")
    plan = release["three_axis_pre_attempt_plan"]
    require(plan["point_order"] == ["k1", "k8", "k1x8"], "axis order")
    require(plan["k1_binding"] == "frozen M519 ARCH_MODE=0 fairness baseline",
            "K1 ARCH_MODE=0")
    require(plan["k8_binding"] == "M803 channel-split ARCH_MODE=1 candidate",
            "K8 ARCH_MODE=1")
    require(plan["k1x8_binding"] ==
            "frozen M519 ARCH_MODE=2 equal-bandwidth fairness baseline",
            "K1x8 ARCH_MODE=2")
    require(plan["tim209_required_each_axis"] == 0 and
            type(plan["tim209_required_each_axis"]) is int, "TIM-209 gate")
    require(plan["opt150_required_each_axis"] == 0 and
            type(plan["opt150_required_each_axis"]) is int, "OPT-150 gate")
    require(plan["all_three_axes_same_attempt_required"] is True,
            "same-attempt gate")
    require(plan["partial_axis_or_cross_attempt_reuse_citable"] is False,
            "cross-attempt reuse")
    source = release["source_static_hammer_binding"]
    require(source["review_sha256"] == EXPECTED[M881_DIR + "/review.json"],
            "M881 SHA pin")
    require(source["exact_status"] == m881["status"] and
            source["score_out_of_100"] == 100 and
            source["p0_p1_p2"] == [0, 0, 0], "M881 verdict pin")
    require(release["identity"]["dc_runner_sha256"] == EXPECTED[RUNNER],
            "runner SHA pin")
    require(release["identity"]["recovery_contract_sha256"] == EXPECTED[CONTRACT],
            "contract SHA pin")
    gate = release["release_authorization"]
    require(gate["inert_until_fresh_final_hammer_pass100"] is True,
            "fresh final gate")
    require(gate["exactly_one_no_args_dc_attempt_subject_to_final_hammer_and_runner_live_gates"]
            is True, "one no-args gate")
    require(gate["runner_status_license_gate_before_attempt_required"] is True,
            "license-before-attempt gate")
    require(gate["runtime_resource_and_collision_identity_required"] is True,
            "runtime resource/collision gate")
    for key in ("dc_completed", "area", "setup_timing", "hold_closed", "power",
                "energy", "ppa", "throughput_per_area", "paper_ppa_ready",
                "complete_fc2", "system_speedup", "headline", "system"):
        require(release["claim_boundary"][key] is False, "claim opened: %s" % key)


def expect_rejected(base_release, candidate, contract, m881, mutate, label):
    changed = copy.deepcopy(base_release)
    mutate(changed)
    try:
        validate_release(changed, candidate, contract, m881)
    except (RuntimeError, KeyError, TypeError):
        return
    raise RuntimeError("release mutation accepted: %s" % label)


def assert_absence():
    require(not os.path.lexists(path(CANONICAL)), "canonical exists")
    require(not os.path.lexists(path(ATTEMPT)), "attempt exists")
    require(glob.glob(path("dc_handoff/runs/.m872_m803_c2_r16_channel_split_three_axis_dc_work.*")) == [],
            "work population")
    require(glob.glob(path(CANONICAL + ".failed_or_incomplete.*.quarantine")) == [],
            "quarantine population")


def main():
    assert_absence()
    for relative, expected in EXPECTED.items():
        require(os.path.isfile(path(relative)) and not os.path.islink(path(relative)),
                "missing/nonregular/symlink: %s" % relative)
        require(sha(relative) == expected, "SHA drift: %s" % relative)
    for relative in (REL, CONTRACT, CAND):
        verify_file(relative)
    for relative in (HANDOFF_DIR, REQUEST_DIR, M881_DIR, M880_DIR,
                     M881_REQUEST_DIR):
        verify_tree(relative)

    release = strict_load(REL)
    candidate = strict_load(CAND)
    contract = strict_load(CONTRACT)
    m881 = strict_load(M881_DIR + "/review.json")
    handoff = strict_load(HANDOFF_DIR + "/handoff.json")
    request = strict_load(REQUEST_DIR + "/request.json")
    validate_release(release, candidate, contract, m881)

    require(handoff["release"]["sha256"] == EXPECTED[REL], "handoff release pin")
    require(handoff["release"]["manifest_file_sha256"] == EXPECTED[REL + ".sha256"],
            "handoff sidecar pin")
    require(handoff["release"]["outer_seal_file_sha256"] ==
            EXPECTED[REL + ".sha256.seal.sha256"], "handoff outer pin")
    require(handoff["execution_audit"]["attempts_consumed"] == 0,
            "handoff attempt claim")
    require(request["review_target"]["release_sha256"] == EXPECTED[REL],
            "request release pin")
    require(request["release_author_chain"]["handoff_json_sha256"] ==
            EXPECTED[HANDOFF_DIR + "/handoff.json"], "request handoff pin")
    require(request["release_author_chain"]["handoff_manifest_file_sha256"] ==
            EXPECTED[HANDOFF_DIR + "/SHA256SUMS"], "request handoff manifest pin")
    require(request["release_author_chain"]["handoff_outer_seal_file_sha256"] ==
            EXPECTED[HANDOFF_DIR + "/SHA256SUMS.seal.sha256"], "request handoff outer pin")
    require(request["request_authorization"] == {
        "run_explicit_no_eda_selftest": True,
        "run_runner_production_path": False, "run_dc": False,
        "run_vcs": False, "run_formality": False, "run_pt": False,
        "run_ptpx": False, "run_remote": False,
        "query_license_server": False, "execute_future_command_now": False,
        "max_eda_attempts_authorized_by_request": 0},
        "request closed authorization")

    mutations = [
        (lambda x: x.update({"launch_now": False}), "launch false"),
        (lambda x: x.update({"status": "AUTHORIZED_WRONG"}), "wrong status"),
        (lambda x: x["authorization"].update({"max_attempts": True}), "bool/int"),
        (lambda x: x["authorization"].update({"max_attempts": 2}), "two attempts"),
        (lambda x: x["authorization"].update({"run_dc": False}), "dc false"),
        (lambda x: x["authorization"].update({"run_vcs": True}), "vcs true"),
        (lambda x: x["authorization"].update({"extra": False}), "extra authorization"),
        (lambda x: x["candidate_binding"].update({"candidate_sha256": "0" * 64}), "candidate SHA"),
        (lambda x: x["candidate_binding"].update({"candidate_launch_now": True}), "candidate launch"),
        (lambda x: x["source_static_hammer_binding"].update({"review_sha256": "0" * 64}), "M881 SHA"),
        (lambda x: x["source_static_hammer_binding"].update({"score_out_of_100": True}), "score bool/int"),
        (lambda x: x["three_axis_pre_attempt_plan"].update({"point_order": ["k8", "k1", "k1x8"]}), "axis order"),
        (lambda x: x["three_axis_pre_attempt_plan"].update({"k8_binding": "ARCH_MODE=0"}), "K8 binding"),
        (lambda x: x["three_axis_pre_attempt_plan"].update({"tim209_required_each_axis": 1}), "TIM-209"),
        (lambda x: x["three_axis_pre_attempt_plan"].update({"opt150_required_each_axis": 1}), "OPT-150"),
        (lambda x: x["three_axis_pre_attempt_plan"].update({"all_three_axes_same_attempt_required": False}), "same attempt"),
        (lambda x: x["three_axis_pre_attempt_plan"].update({"partial_axis_or_cross_attempt_reuse_citable": True}), "cross reuse"),
        (lambda x: x["release_authorization"].update({"inert_until_fresh_final_hammer_pass100": False}), "fresh hammer"),
        (lambda x: x["release_authorization"].update({"runner_status_license_gate_before_attempt_required": False}), "license gate"),
        (lambda x: x["release_authorization"].update({"runtime_resource_and_collision_identity_required": False}), "resource gate"),
        (lambda x: x["claim_boundary"].update({"ppa": True}), "PPA claim"),
        (lambda x: x["claim_boundary"].update({"system_speedup": True}), "speedup claim"),
    ]
    for mutate, label in mutations:
        expect_rejected(release, candidate, contract, m881, mutate, label)

    raw_negatives = [
        b'{"x":1,"x":2}', b'{"launch_now":true,"launch_now":false}',
        b'{"authorization":{},"authorization":{}}', b'{"x":NaN}',
        b'{"x":Infinity}', b'{"x":-Infinity}']
    for raw in raw_negatives:
        try:
            json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                       parse_constant=reject_nonfinite)
        except ValueError:
            pass
        else:
            raise RuntimeError("strict JSON negative accepted")

    runner_text = open(path(RUNNER), "r").read()
    require("m872_m803_dc_final_admission=" + REL in runner_text,
            "runner release path")
    require("m872_m803_dc_expected_admission_status=" + release["status"] in runner_text,
            "runner release status")
    require("M872_M803_DC_EXPECTED_DC_RUNNER_SHA256" in runner_text and
            "M872_M803_DC_EXPECTED_DC_LAUNCH_ADMISSION_SHA256" in runner_text,
            "caller pin gates")
    require(re.search(r"m872_m803_dc_run_point\s+k1\s+0\s*\n.*?"
                      r"m872_m803_dc_run_point\s+k8\s+1\s*\n.*?"
                      r"m872_m803_dc_run_point\s+k1x8\s+2", runner_text, re.S),
            "runtime ARCH order")
    require("m872_m803_dc_license_preflight" in runner_text and
            runner_text.index("m872_m803_dc_license_preflight") <
            runner_text.index("mv -T \"${m872_m803_dc_work}/.attempt_staging\""),
            "license-before-attempt static order")
    require("m872_m803_dc_resource_snapshot" in runner_text and
            "m872_m803_dc_runtime_monitor" in runner_text and
            "status=PASS_FINAL_GATE_ACK" in runner_text,
            "mature resource/runtime/final-ack gate")
    require("m872_m803_dc_verify_live_artifact_receipts" in runner_text and
            "m872_m803_dc_verify_axis_artifact_manifest" in runner_text and
            "postreceipt" in runner_text.lower(), "atomic artifact gate")
    subprocess.check_call(["bash", "-n", path(RUNNER)])

    tcl_text = open(path(TCL), "r").read()
    tokens = ["analyze -format sverilog", "elaborate $design_name",
              'check_design > "$output_dir/reports/check_design_precompile.rpt"',
              "redirect $precompile_timing_report {check_timing}",
              "if {$precompile_tim209_count != 0 || $precompile_opt150_count != 0}",
              "\n    compile_ultra\n"]
    indexes = [tcl_text.index(token) for token in tokens]
    require(indexes == sorted(indexes), "Tcl precompile ordering")
    require(tcl_text.count("\n    compile_ultra\n") == 1, "compile_ultra count")

    future = request["future_command_under_review"]
    argv = shlex.split(future)
    require(argv[-1] == path(RUNNER) and argv.count(path(RUNNER)) == 1,
            "future runner/no-args command")
    require("M872_M803_DC_EXPECTED_DC_RUNNER_SHA256=" + EXPECTED[RUNNER] in argv,
            "future runner pin")
    require("M872_M803_DC_EXPECTED_DC_LAUNCH_ADMISSION_SHA256=" + EXPECTED[REL] in argv,
            "future release pin")

    source_hammer = path(M881_DIR + "/independent_source_hammer.py")
    for executable in ("/usr/libexec/platform-python3.6",
                       "/opt/anaconda3/envs/pytorch310/bin/python3.10"):
        completed = subprocess.run([executable, source_hammer], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(completed.returncode == 0 and
                b"PASS100_M880_M803_C2_R16_TERMINOLOGY_REPAIR_SOURCE_FRESH_HAMMER" in completed.stdout,
                "source hammer replay failed: %s" % executable)

    assert_absence()
    print("PASS100_M882_M880_M803_C2_R16_FINAL_RELEASE_HAMMER")
    print("python=%s" % sys.version.split()[0])
    print("p0=0 p1=0 p2=0")
    print("source_hammer_python36=PASS source_hammer_python310=PASS")
    print("preserved_sections=18 release_mutations=22 duplicate_nonfinite_negatives=6")
    print("axes=K1_ARCH0,M803_K8_ARCH1,K1x8_ARCH2 tim209=0 opt150=0")
    print("canonical_attempt_work_quarantine_absent=true")
    print("dc_runs=0 vcs_runs=0 license_queries=0 remote_runs=0")


if __name__ == "__main__":
    main()
