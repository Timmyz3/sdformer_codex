#!/usr/bin/env python3
"""Fresh no-EDA final-launch hammer for the schema-repaired M892 C1 point."""

from __future__ import print_function

import copy
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT = "reviews/m892_m528_r21_macro_aware_product_schema_repair_dc_final_launch_hammer_r1_20260829"
RELEASE = "contracts/m892_m528_r21_macro_aware_product_schema_repair_dc_launch_release_r1_20260829.json"
CANDIDATE = "contracts/m892_m528_r21_macro_aware_product_schema_repair_dc_launch_candidate_source_only_r1_20260829.json"
CONTRACT = "contracts/m892_m528_r21_macro_aware_product_dc_schema_repair_source_only_contract_r1_20260829.json"
RUNNER = "dc_handoff/scripts/run_dc_m892_m528_r21_macro_aware_product_schema_repair_exact_sha_r1.sh"
SOURCE_TEST = "verif_m528_dw1rw/test_m892_m528_r21_macro_dc_schema_repair_source_closure.py"
SOURCE_HAMMER = "reviews/m895_m892_m528_r21_macro_aware_product_schema_repair_dc_source_fresh_hammer_r1_20260829/independent_source_hammer.py"
M895 = "reviews/m895_m892_m528_r21_macro_aware_product_schema_repair_dc_source_fresh_hammer_r1_20260829"
M891 = "reviews/m891_m884_macro_dc_release_author_preflight_failure_audit_r1_20260829"
M885 = "reviews/m885_m884_m528_r21_macro_aware_product_dc_source_fresh_hammer_r1_20260829"
HANDOFF = "reviews/m897_m892_m528_r21_macro_aware_product_schema_repair_dc_release_author_handoff_r1_20260829"
REQUEST = "reviews/m898_m897_m892_m528_r21_macro_aware_product_schema_repair_dc_final_launch_hammer_REQUEST_r1_20260829"
SOURCE_HANDOFF = "reviews/m892_m528_r21_macro_aware_product_schema_repair_dc_source_author_handoff_r1_20260829"
SOURCE_REQUEST = "reviews/m895_m892_m528_r21_macro_aware_product_schema_repair_dc_source_hammer_REQUEST_r1_20260829"
M884_HANDOFF = "reviews/m884_m528_r21_macro_aware_product_dc_source_author_handoff_r1_20260829"
M885_REQUEST = "reviews/m885_m884_m528_r21_macro_aware_product_dc_source_hammer_REQUEST_r1_20260829"
DOCS359 = "docs/359_DATE终局冻结_20260813.md"
TCL = "dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_candidate.tcl"
SDC = "dc_handoff/constraints/date_m884_m528_r21_macro_aware_product_3ns.sdc"
FILELIST = "dc_handoff/filelists/date_m884_m528_r21_macro_aware_product_dc.f"
ADAPTER = "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
CANONICAL = "dc_handoff/runs/m892_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829"
ATTEMPT = "dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_attempt_consumed"
LOCK = "dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_launch_lock"

PYTHONS = [
    "/usr/libexec/platform-python3.6",
    "/opt/anaconda3/envs/pytorch310/bin/python3.10",
]

EXPECTED = {
    RELEASE: "992b11895783939f932cc45311d07f61d7738e0a800499d12b5b99bdd7bb06ca",
    RELEASE + ".sha256": "4e78ef689a0941b55a1842043148bbb27bf8ea17abc583fadd81d3cd0a1bcee6",
    RELEASE + ".sha256.seal.sha256": "1fae935e71fe0d64ae5f40aa68bc3cb7975f4c9fb77ce44b13410994cc6ba307",
    CANDIDATE: "79f4b0a6d3d16c7977166823eb318fd00a1670d2f67f2f58e4439caad26ad1c0",
    CONTRACT: "5b5ec1ecb8fa75299bd32b5776759a3921dfc7329e27a3d48a545c0a23e1267d",
    RUNNER: "a0c07f8740a830d7a3e99ae1bf6dd2f3f55c4f77102c7b6a0eeb1746694d5d9f",
    SOURCE_TEST: "419ad48854b5b987100bad0914b2fb1fbaf1a989f14f45d5d523ca3fc769f611",
    M895 + "/review.json": "01e0aa82b044a488c83337acb34e32f572c84667c31706764f8fca37e053e665",
    M895 + "/SHA256SUMS": "6e01c1d6c5d35ea021019557a82a651aed138ea0036ae52597a6f28804732fd3",
    M895 + "/SHA256SUMS.seal.sha256": "c46f359ced4927234cd43f10dbb3bf41320f1bdba636043679336750cea78095",
    M891 + "/review.json": "883829d8017b2656161d5e3f7f2300c38ad214cc308dbcc06f761b3b875a8792",
    M891 + "/SHA256SUMS": "b71788863c24a335c257b38bf0c66dc385039ff6df41ff749b2d46e6d631c073",
    M891 + "/SHA256SUMS.seal.sha256": "5dfc669879034caa016332e7553347949b8ee82dfe8761700c4be6eddafe7f20",
    M885 + "/review.json": "607b3898c05ce816b25f8cff26ffe01991d603db5e106707e2b7f8dc80d91b95",
    HANDOFF + "/handoff.json": "403b07f2784975d3c0fbeb8d0dfc0977942879921eddd915992ac463fc4390d9",
    HANDOFF + "/SHA256SUMS": "24d351b30c52063341662aaab989d92345ea3f31c1d16dac584a148a2ff11f73",
    HANDOFF + "/SHA256SUMS.seal.sha256": "b04b8c4da4a02dbcbe04671db672c3299f5d5e433361e8effe73f6c54bf2c247",
    REQUEST + "/request.json": "80988918ac56ebe4f75d15110540fa5d207281b48e36197849833030b1394f26",
    REQUEST + "/SHA256SUMS": "5897003cdc00ed307a1e0769bca20e8d464d13df90eac795b9fe3c74e661271a",
    REQUEST + "/SHA256SUMS.seal.sha256": "a620c52a6380cb8e57c0ff596fb007b9cbc28566f8905c47dfbf1d09e41fe9b2",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def strict_load(relative):
    with open(path(relative), "rb") as handle:
        return json.loads(handle.read().decode("utf-8"),
                          object_pairs_hook=unique_object,
                          parse_constant=reject_nonfinite)


def strict_load_bytes(payload):
    return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def verify_file_seal(relative):
    full = path(relative)
    base = os.path.basename(full)
    subprocess.check_call(["sha256sum", "-c", base + ".sha256"],
                          cwd=os.path.dirname(full), stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", base + ".sha256.seal.sha256"],
                          cwd=os.path.dirname(full), stdout=subprocess.DEVNULL)


def verify_tree(relative):
    full = path(relative)
    require(os.path.isdir(full) and not os.path.islink(full),
            "missing/symlink evidence tree: %s" % relative)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"], cwd=full,
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=full,
                          stdout=subprocess.DEVNULL)


def assert_no_execution_population():
    for relative in (CANONICAL, ATTEMPT, LOCK):
        require(not os.path.lexists(path(relative)), "execution population: %s" % relative)
    require(glob.glob(path("dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_work.*")) == [],
            "work population")
    require(glob.glob(path(CANONICAL + ".failed_or_incomplete.*")) == [],
            "quarantine population")


def assert_prepublication_output():
    entries = sorted(os.listdir(path(OUTPUT)))
    require(entries == ["independent_final_hammer.py"],
            "fixed output was populated before publication: %s" % entries)


def closed_false_claims(claims):
    for key in ("fair_K_zero_bit", "throughput_per_mm2", "speedup",
                "system_speedup", "system", "power", "energy", "ppa",
                "physical_route", "paper_ppa_ready", "headline"):
        require(claims[key] is False, "claim escaped: %s" % key)


def validate_contract(contract):
    require(type(contract) is dict, "contract type")
    require(set(contract) == {
        "authorization", "claim_boundary", "date", "docs359_sha256",
        "exact_files", "fairness", "foundry_views", "frozen_authorities",
        "future_release_chain", "physical_point", "schema", "status",
        "tool_identity"}, "contract top-level keys")
    require(contract["schema"] ==
            "m892_m528_r21_macro_aware_product_dc_source_only_contract_v1",
            "contract schema")
    require(contract["status"] ==
            "SOURCE_ONLY_M892_M528_R21_MACRO_AWARE_PRODUCT_DC__FRESH_HAMMER_REQUIRED__NO_EDA_AUTHORIZED",
            "contract status")
    require(contract["authorization"] == {
        "author_ran_eda": False, "run_dc_now": False,
        "run_formality_now": False, "run_pt_now": False,
        "run_ptpx_now": False, "run_remote_now": False,
        "run_saif_now": False, "run_vcs_now": False}, "contract authorization")
    require(contract["fairness"] == {
        "candidate_point_only": True, "fair_K_zero_bit": False,
        "zero_rtl_baseline_present": False, "bit_rtl_baseline_present": False},
        "contract fairness")
    point = contract["physical_point"]
    require(point["candidate"] == "M528 R21 product-capture only", "candidate object")
    require(type(point["clock_period_ns"]) is float and point["clock_period_ns"] == 3.0,
            "3 ns clock")
    require(point["ideal_clock"] is True and point["wireload"] == "ZeroWireload",
            "clock/wireload")
    require(point["compile_define"] == "SYNTHESIS", "compile define")
    require(point["macro_cell"] == "TS1N28HPCPHVTB128X128M4S" and
            type(point["macro_count"]) is int and point["macro_count"] == 9,
            "nine foundry macros")
    require(point["macro_slow_fast_min_pair"] is True, "macro min pair")
    require(point["tim209_required"] == 0 and type(point["tim209_required"]) is int,
            "TIM-209 gate")
    require(point["opt150_required"] == 0 and type(point["opt150_required"]) is int,
            "OPT-150 gate")
    require(point["setup_must_be_met"] is True and point["hold_diagnostic_only"] is True,
            "timing claim boundary")
    require(point["mapped_outputs"] == ["Verilog", "SDC", "DDC", "SVF"],
            "mapped artifacts")
    require(point["all_storage_foundry_macro_mapped"] is False,
            "full-storage claim boundary")
    views = contract["foundry_views"]
    for prefix in ("std_slow", "std_fast", "macro_slow", "macro_fast"):
        require(os.path.isfile(views[prefix + "_path"]), "foundry view absent: %s" % prefix)
        require(hashlib.sha256(open(views[prefix + "_path"], "rb").read()).hexdigest() ==
                views[prefix + "_sha256"], "foundry view SHA: %s" % prefix)
    require("ssg0p9v125c" in views["std_slow_path"] and
            "ffg1p05vm40c" in views["std_fast_path"] and
            "ssg0p9v125c" in views["macro_slow_path"] and
            "ffg1p05vm40c" in views["macro_fast_path"], "slow/fast corner names")
    closed_false_claims(contract["claim_boundary"])


def validate_release(release, candidate, contract, m895):
    require(type(release) is dict, "release type")
    require(set(release) == {
        "authorization", "claim_boundary", "date", "docs359_sha256",
        "fairness", "frozen_authorities", "future_release_chain", "identity",
        "launch_now", "prospective_attempt", "schema", "status"},
        "release top-level keys")
    require(release["schema"] ==
            "m892_m528_r21_macro_aware_product_dc_launch_release_v1",
            "release schema")
    require(release["status"] ==
            "AUTHORIZED_ONE_M892_M528_R21_MACRO_AWARE_PRODUCT_DC_ATTEMPT",
            "release status")
    require(type(release["launch_now"]) is bool and release["launch_now"] is True,
            "release launch")
    require(release["authorization"] == {
        "max_attempts": 1, "run_dc": True, "run_formality": False,
        "run_pt": False, "run_ptpx": False, "run_remote": False,
        "run_saif": False, "run_vcs": False}, "closed release authorization")
    require(type(release["authorization"]["max_attempts"]) is int,
            "max attempts bool/int")
    require(candidate["launch_now"] is False and
            candidate["authorization"]["max_attempts"] == 0 and
            candidate["authorization"]["run_dc"] is False, "candidate inertness")
    require(release["identity"] == candidate["identity"], "candidate identity drift")
    require(release["identity"]["runner_sha256"] == EXPECTED[RUNNER], "runner pin")
    require(release["identity"]["source_contract_sha256"] == EXPECTED[CONTRACT],
            "contract pin")
    require(release["prospective_attempt"] == candidate["prospective_attempt"],
            "prospective attempt drift")
    require(release["prospective_attempt"] == {
        "attempt_absent_at_authoring": True, "candidate": "M528 R21 product-capture only",
        "canonical_unique": True, "clock_period_ns": 3.0,
        "failure_quarantine_unique": True,
        "macro_cell": "TS1N28HPCPHVTB128X128M4S", "macro_count": 9,
        "result_absent_at_authoring": True}, "prospective contract")
    require(release["fairness"] == candidate["fairness"] == contract["fairness"],
            "fairness drift")
    require(release["claim_boundary"] == candidate["claim_boundary"],
            "claim drift")
    closed_false_claims(release["claim_boundary"])
    frozen = release["frozen_authorities"]
    require(frozen["m892_candidate_sha256"] == EXPECTED[CANDIDATE], "candidate pin")
    require(frozen["m892_source_closure_test_sha256"] == EXPECTED[SOURCE_TEST],
            "source test pin")
    require(frozen["m895_source_review_sha256"] == EXPECTED[M895 + "/review.json"],
            "M895 pin")
    require(frozen["m895_source_manifest_file_sha256"] == EXPECTED[M895 + "/SHA256SUMS"] and
            frozen["m895_source_outer_seal_file_sha256"] ==
            EXPECTED[M895 + "/SHA256SUMS.seal.sha256"], "M895 seal pins")
    require(m895["verdict"] == "PASS" and m895["score_out_of_100"] == 100 and
            [m895["p0_count"], m895["p1_count"], m895["p2_count"]] == [0, 0, 0],
            "M895 score/severity")
    future = release["future_release_chain"]
    require(future["final_review_path"] == OUTPUT + "/review.json" and
            future["final_review_sha_caller_pinned"] is True and
            future["release_binds_candidate_sha"] is True and
            future["release_binds_source_hammer_sha"] is True,
            "future final gate")
    require(future["source_hammer_review_path"] == M895 + "/review.json" and
            future["source_hammer_review_sha256"] == EXPECTED[M895 + "/review.json"],
            "source hammer chain")
    require(release["docs359_sha256"] == EXPECTED[DOCS359], "docs359 release pin")


def reject_mutation(base, validator, mutator, label):
    trial = copy.deepcopy(base)
    mutator(trial)
    try:
        validator(trial)
    except (KeyError, RuntimeError, TypeError):
        return
    raise RuntimeError("mutation accepted: %s" % label)


def validate_handoff_request(handoff, request):
    require(handoff["release"]["sha256"] == EXPECTED[RELEASE], "handoff release pin")
    require(handoff["release"]["manifest_file_sha256"] == EXPECTED[RELEASE + ".sha256"],
            "handoff release manifest pin")
    require(handoff["release"]["outer_seal_file_sha256"] ==
            EXPECTED[RELEASE + ".sha256.seal.sha256"], "handoff release outer pin")
    require(handoff["release"]["launch_now"] is True and
            handoff["release"]["inert_until_fresh_final_hammer_pass100"] is True,
            "handoff inertness")
    require(handoff["physical_contract"]["macro_count"] == 9 and
            handoff["physical_contract"]["clock_period_ns"] == 3.0 and
            handoff["physical_contract"]["macro_slow_fast_min_pair"] is True and
            handoff["physical_contract"]["stdcell_slow_fast_min_pair"] is True,
            "handoff physical contract")
    require(handoff["physical_contract"]["tim209_required"] == 0 and
            handoff["physical_contract"]["opt150_required"] == 0 and
            handoff["physical_contract"]["mapped_outputs"] ==
            ["Verilog", "SDC", "DDC", "SVF"], "handoff artifact gates")
    require(handoff["physical_contract"]["fair_K_zero_bit"] is False,
            "handoff fairness")
    require(handoff["execution_audit"]["attempts_consumed"] == 0 and
            handoff["execution_audit"]["dc_runs"] == 0 and
            handoff["execution_audit"]["license_queries"] == 0,
            "handoff no-execution audit")
    target = request["review_target"]
    require(target["release_sha256"] == EXPECTED[RELEASE] and
            target["release_manifest_file_sha256"] == EXPECTED[RELEASE + ".sha256"] and
            target["release_outer_seal_file_sha256"] ==
            EXPECTED[RELEASE + ".sha256.seal.sha256"], "request release pins")
    require(target["expected_output_path"] == OUTPUT and
            target["expected_review_schema"] ==
            "m892_m528_r21_macro_aware_product_dc_final_launch_hammer_v1" and
            target["expected_review_status"] == "PASS100_M892_FINAL_LAUNCH_HAMMER",
            "request output contract")
    chain = request["release_author_chain"]
    require(chain["handoff_json_sha256"] == EXPECTED[HANDOFF + "/handoff.json"] and
            chain["handoff_manifest_file_sha256"] == EXPECTED[HANDOFF + "/SHA256SUMS"] and
            chain["handoff_outer_seal_file_sha256"] ==
            EXPECTED[HANDOFF + "/SHA256SUMS.seal.sha256"], "request handoff pins")
    require(request["request_authorization"] == {
        "execute_future_command_now": False,
        "max_eda_attempts_authorized_by_request": 0,
        "query_license": False, "run_dc": False, "run_formality": False,
        "run_pt": False, "run_ptpx": False, "run_release_no_eda_checks": True,
        "run_remote": False, "run_runner_production_path": False,
        "run_saif": False, "run_source_no_eda_selftests": True,
        "run_vcs": False}, "request closed authorization")
    require(request["post_pass_only_command_contract"] == {
        "caller_pins_final_review_sha": True, "caller_pins_release_sha": True,
        "caller_pins_runner_sha": True, "command_must_not_be_executed_by_final_reviewer": True,
        "max_attempts": 1, "no_args_runner": True,
        "run_dc": True,
        "runner_rechecks_live_resource_collision_license_identity_population_macro_and_artifact_gates": True,
    }, "post-pass command contract")


def runner_static_checks(contract, release):
    subprocess.check_call(["bash", "-n", path(RUNNER)])
    runner = open(path(RUNNER), "r").read()
    tcl = open(path(TCL), "r").read()
    sdc = open(path(SDC), "r").read()
    filelist = open(path(FILELIST), "r").read()
    adapter = open(path(ADAPTER), "r").read()
    require("m892_release=" + RELEASE in runner and
            "m892_final_review=" + OUTPUT + "/review.json" in runner,
            "runner release/final coordinates")
    require("M892_EXPECTED_DC_RUNNER_SHA256" in runner and
            "M892_EXPECTED_DC_ADMISSION_SHA256" in runner and
            "M892_EXPECTED_DC_FINAL_REVIEW_SHA256" in runner,
            "three caller SHA pins")
    final_gate = runner.index('.schema == "m892_m528_r21_macro_aware_product_dc_final_launch_hammer_v1"')
    resource_gate = runner.index('mkdir "${m892_lock}"')
    license_gate = runner.index('"${m892_lmutil}" lmstat')
    attempt_gate = runner.index('mkdir "${m892_attempt}"')
    dc_gate = runner.index('"${m892_dc}" -f')
    require(final_gate < resource_gate < license_gate < attempt_gate < dc_gate,
            "final/resource/license/attempt/DC ordering")
    require('.status == "PASS100_M892_FINAL_LAUNCH_HAMMER"' in runner and
            '.score_100 == 100' in runner and
            '.severity_counts == {"p0":0,"p1":0,"p2":0}' in runner and
            '.decision.exactly_one_dc_attempt_authorized == true' in runner,
            "final predicate")
    require('MACRO_SLOW_DB="${m892_macro_slow}" MACRO_FAST_DB="${m892_macro_fast}"' in runner,
            "macro slow/fast env")
    require('STD_SLOW_DB="${m892_std_slow}" STD_FAST_DB="${m892_std_fast}"' in runner,
            "std slow/fast env")
    require("macro_count_pre=9" in runner and "macro_count_post=9" in runner and
            "TIM-209=0" in runner and "OPT-150=0" in runner,
            "runtime physical gates")
    for artifact in (
            "reports/timing_setup.rpt", "reports/timing_hold_diagnostic.rpt",
            "reports/precompile_loop_gate.rpt", "reports/macro_binding_audit.txt",
            "_mapped.v", "_mapped.sdc", ".ddc", ".svf"):
        require(artifact in runner, "artifact gate absent: %s" % artifact)
    require("analyze -format sverilog -define SYNTHESIS" in tcl and
            "set_min_library $std_slow_db -min_version $std_fast_db" in tcl and
            "set_min_library $macro_slow_db -min_version $macro_fast_db" in tcl,
            "Tcl libraries/define")
    require("set_wire_load_model -name ZeroWireload" in tcl and
            "set expected_macro_count 9" in tcl and
            "hold_diagnostic_only=true" in tcl, "Tcl physical semantics")
    pre = tcl.index('redirect "$output_dir/reports/check_timing_precompile.rpt"')
    loop = tcl.index("if {$pre_tim209 != 0 || $pre_opt150 != 0}")
    compile_index = tcl.index("compile_ultra -no_autoungroup")
    require(pre < loop < compile_index and tcl.count("compile_ultra -no_autoungroup") == 1,
            "precompile loop gate ordering")
    require("create_clock" in sdc and ("3.000" in sdc or "3.0" in sdc), "SDC 3 ns")
    require("m528_dw1rw_parent_scratch_9x128_macro.sv" in filelist and
            adapter.count("TS1N28HPCPHVTB128X128M4S") >= 1, "macro adapter binding")
    require(contract["physical_point"]["macro_count"] == release["prospective_attempt"]["macro_count"],
            "release/contract macro count")


def run_exact_m895_production_predicate():
    scratch = tempfile.mkdtemp(prefix="m898_m895_production.", dir="/tmp")
    try:
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "M892_NO_EDA_FULL_PATH_SELFTEST": "1",
            "M892_NO_EDA_PRODUCTION_SCHEMA_SELFTEST": "1",
            "M892_NO_EDA_SELFTEST_ROOT": scratch,
            "M892_NO_EDA_SOURCE_REVIEW_FIXTURE": path(M895 + "/review.json"),
            "M892_EXPECTED_NO_EDA_SOURCE_REVIEW_SHA256": EXPECTED[M895 + "/review.json"],
            "M892_EXPECTED_DC_RUNNER_SHA256": EXPECTED[RUNNER],
            "M892_EXPECTED_DC_ADMISSION_SHA256": EXPECTED[CANDIDATE],
        }
        completed = subprocess.run([path(RUNNER)], cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(completed.returncode == 0,
                "exact M895 production predicate failed: %s" %
                completed.stderr.decode("utf-8", errors="replace"))
        marker = os.path.join(scratch, "PRODUCTION_SCHEMA_PASS.txt")
        require(os.path.isfile(marker), "production marker missing")
        text = open(marker, "r").read()
        require("status=PASS_M892_PRODUCTION_M885_SCHEMA_PATH_NO_EDA" in text and
                "source_review_schema=score_out_of_100_plus_p0_p1_p2" in text and
                "attempt_consumed=false" in text and
                "license_query_started=false" in text and
                "dc_shell_started=false" in text, "production boundary marker")
    finally:
        shutil.rmtree(scratch)


def run_cross_python_full():
    receipts = []
    for executable in PYTHONS:
        version = subprocess.check_output([executable, "--version"],
                                          stderr=subprocess.STDOUT).decode().strip()
        source = subprocess.run([executable, path(SOURCE_TEST)], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(source.returncode == 0 and b"PASS M892 source closure" in source.stdout,
                "source closure failed under %s: %s" %
                (executable, source.stderr.decode("utf-8", errors="replace")))
        hammer = subprocess.run([executable, path(SOURCE_HAMMER)], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(hammer.returncode == 0,
                "M895 source hammer replay failed under %s: %s" %
                (executable, hammer.stderr.decode("utf-8", errors="replace")))
        parsed = json.loads(hammer.stdout.decode("utf-8"),
                            object_pairs_hook=unique_object,
                            parse_constant=reject_nonfinite)
        require(parsed["status"] ==
                "PASS_M895_M892_FRESH_INDEPENDENT_SOURCE_HAMMER_NO_EDA" and
                parsed["production_positive"] == 1 and
                len(parsed["production_negatives"]) == 12 and
                parsed["semantic_negatives"] == 21 and
                parsed["canonical_count"] == 0 and parsed["attempt_count"] == 0 and
                parsed["work_count"] == 0 and parsed["quarantine_count"] == 0 and
                parsed["license_queries"] == 0 and parsed["eda_runs"] == 0,
                "M895 source hammer receipt")
        run_exact_m895_production_predicate()
        assert_no_execution_population()
        receipts.append({
            "version": version,
            "source_receipt": source.stdout.decode().strip(),
            "source_hammer_production_negatives": len(parsed["production_negatives"]),
            "source_hammer_semantic_negatives": parsed["semantic_negatives"],
            "exact_m895_production_predicate": "PASS",
        })
    return receipts


def main():
    assert_no_execution_population()
    assert_prepublication_output()
    for relative, expected in EXPECTED.items():
        require(os.path.isfile(path(relative)) and not os.path.islink(path(relative)),
                "missing/nonregular/symlink: %s" % relative)
        require(sha(relative) == expected, "SHA drift: %s" % relative)
    for relative in (RELEASE, CANDIDATE, CONTRACT, RUNNER, SOURCE_TEST):
        verify_file_seal(relative)
    for relative in (M895, M891, M885, HANDOFF, REQUEST, SOURCE_HANDOFF,
                     SOURCE_REQUEST, M884_HANDOFF, M885_REQUEST):
        verify_tree(relative)
    json_documents = [
        RELEASE, CANDIDATE, CONTRACT, M895 + "/review.json", M891 + "/review.json",
        M885 + "/review.json", HANDOFF + "/handoff.json", REQUEST + "/request.json",
        SOURCE_HANDOFF + "/handoff.json", SOURCE_REQUEST + "/request.json",
        M884_HANDOFF + "/handoff.json", M885_REQUEST + "/request.json",
    ]
    documents = [strict_load(relative) for relative in json_documents]
    release, candidate, contract, m895, m891, m885, handoff, request = documents[:8]
    validate_contract(contract)
    validate_release(release, candidate, contract, m895)
    validate_handoff_request(handoff, request)
    require(m891["status"] ==
            "PASS_FAILURE_AUDIT__M884_RELEASE_NOT_AUTHORED__SOURCE_REVIEW_SCHEMA_MISMATCH__ADDITIVE_RUNNER_SOURCE_REPAIR_REQUIRED" and
            m891["decision"]["m884_known_defective_command_must_not_execute"] is True,
            "M891 defective M884 boundary")
    require(m885["score_out_of_100"] == 100 and
            [m885["p0_count"], m885["p1_count"], m885["p2_count"]] == [0, 0, 0],
            "M885 predecessor score")

    duplicate_nonfinite = [
        b'{"x":1,"x":2}', b'{"launch_now":true,"launch_now":false}',
        b'{"authorization":{},"authorization":{}}', b'{"x":NaN}',
        b'{"x":Infinity}', b'{"x":-Infinity}',
    ]
    for payload in duplicate_nonfinite:
        try:
            strict_load_bytes(payload)
        except ValueError:
            pass
        else:
            raise RuntimeError("strict JSON negative accepted")

    release_mutations = [
        (lambda x: x.update({"launch_now": False}), "launch"),
        (lambda x: x.update({"status": "WRONG"}), "status"),
        (lambda x: x["authorization"].update({"max_attempts": True}), "bool-int"),
        (lambda x: x["authorization"].update({"max_attempts": 2}), "two-attempt"),
        (lambda x: x["authorization"].update({"run_dc": False}), "dc-false"),
        (lambda x: x["authorization"].update({"run_vcs": True}), "vcs"),
        (lambda x: x["authorization"].update({"run_formality": True}), "formality"),
        (lambda x: x["authorization"].update({"run_pt": True}), "pt"),
        (lambda x: x["authorization"].update({"run_ptpx": True}), "ptpx"),
        (lambda x: x["authorization"].update({"run_saif": True}), "saif"),
        (lambda x: x["authorization"].update({"run_remote": True}), "remote"),
        (lambda x: x["identity"].update({"runner_sha256": "0" * 64}), "runner-sha"),
        (lambda x: x["identity"].update({"source_contract_sha256": "0" * 64}), "contract-sha"),
        (lambda x: x["frozen_authorities"].update({"m892_candidate_sha256": "0" * 64}), "candidate-sha"),
        (lambda x: x["frozen_authorities"].update({"m895_source_review_sha256": "0" * 64}), "source-sha"),
        (lambda x: x["future_release_chain"].update({"final_review_sha_caller_pinned": False}), "final-pin"),
        (lambda x: x["future_release_chain"].update({"source_hammer_review_sha256": "0" * 64}), "hammer-sha"),
        (lambda x: x["prospective_attempt"].update({"macro_count": 8}), "macro-count"),
        (lambda x: x["prospective_attempt"].update({"clock_period_ns": 2.9}), "clock"),
        (lambda x: x["prospective_attempt"].update({"canonical_unique": False}), "namespace"),
        (lambda x: x["fairness"].update({"fair_K_zero_bit": True}), "fairness"),
        (lambda x: x["claim_boundary"].update({"speedup": True}), "speedup"),
        (lambda x: x["claim_boundary"].update({"energy": True}), "energy"),
        (lambda x: x["claim_boundary"].update({"ppa": True}), "ppa"),
        (lambda x: x.update({"unknown": False}), "unknown-key"),
    ]
    release_validator = lambda value: validate_release(value, candidate, contract, m895)
    for mutator, label in release_mutations:
        reject_mutation(release, release_validator, mutator, label)

    contract_mutations = [
        (lambda x: x["physical_point"].update({"macro_count": 8}), "contract-macro"),
        (lambda x: x["physical_point"].update({"clock_period_ns": 2.5}), "contract-clock"),
        (lambda x: x["physical_point"].update({"tim209_required": 1}), "contract-tim209"),
        (lambda x: x["physical_point"].update({"opt150_required": 1}), "contract-opt150"),
        (lambda x: x["physical_point"].update({"hold_diagnostic_only": False}), "contract-hold"),
        (lambda x: x["physical_point"].update({"mapped_outputs": ["Verilog"]}), "contract-artifacts"),
        (lambda x: x["fairness"].update({"fair_K_zero_bit": True}), "contract-fairness"),
        (lambda x: x["claim_boundary"].update({"system_speedup": True}), "contract-system"),
    ]
    for mutator, label in contract_mutations:
        reject_mutation(contract, validate_contract, mutator, label)

    runner_static_checks(contract, release)
    cross_python = run_cross_python_full()
    assert_no_execution_population()
    assert_prepublication_output()
    require(sha(DOCS359) == EXPECTED[DOCS359], "docs359 changed")
    print("PASS100_M892_FINAL_LAUNCH_HAMMER")
    print("python=%s" % sys.version.split()[0])
    print("p0=0 p1=0 p2=0")
    print("strict_json_documents=%d duplicate_nonfinite_negatives=6" % len(json_documents))
    print("release_mutations=25 contract_mutations=8")
    print("python36_full_no_eda=PASS python310_full_no_eda=PASS")
    print("source_hammer_python36=PASS source_hammer_python310=PASS")
    print("exact_m895_production_predicate_runs=2")
    print("source_artifact_negatives_per_interpreter=7")
    print("source_production_schema_negatives_per_interpreter=8")
    print("source_independent_production_negatives_per_interpreter=12")
    print("source_semantic_negatives_per_interpreter=21")
    print("macro_count=9 clock_period_ns=3.0 tim209=0 opt150=0")
    print("std_slow_fast=true macro_slow_fast=true artifacts=Verilog,SDC,DDC,SVF")
    print("fair_K_zero_bit=false all_speedup_ppa_system_energy_claims=false")
    print("canonical_attempt_work_quarantine_lock_absent=true")
    print("dc_runs=0 vcs_runs=0 license_queries=0 remote_runs=0")
    print("docs359_sha256=%s" % EXPECTED[DOCS359])
    require(len(cross_python) == 2, "cross-python receipt population")


if __name__ == "__main__":
    main()
