#!/usr/bin/env python3
from __future__ import print_function

import copy
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
RELEASE = "contracts/m892_m528_r21_macro_aware_product_schema_repair_dc_launch_release_r1_20260829.json"
CANDIDATE = "contracts/m892_m528_r21_macro_aware_product_schema_repair_dc_launch_candidate_source_only_r1_20260829.json"
CONTRACT = "contracts/m892_m528_r21_macro_aware_product_dc_schema_repair_source_only_contract_r1_20260829.json"
RUNNER = "dc_handoff/scripts/run_dc_m892_m528_r21_macro_aware_product_schema_repair_exact_sha_r1.sh"
SOURCE_TEST = "verif_m528_dw1rw/test_m892_m528_r21_macro_dc_schema_repair_source_closure.py"
M895 = "reviews/m895_m892_m528_r21_macro_aware_product_schema_repair_dc_source_fresh_hammer_r1_20260829"
M891 = "reviews/m891_m884_macro_dc_release_author_preflight_failure_audit_r1_20260829"
M885 = "reviews/m885_m884_m528_r21_macro_aware_product_dc_source_fresh_hammer_r1_20260829"
DOCS359 = "docs/359_DATE终局冻结_20260813.md"
FINAL_REVIEW_DIR = "reviews/m892_m528_r21_macro_aware_product_schema_repair_dc_final_launch_hammer_r1_20260829"
CANONICAL = "dc_handoff/runs/m892_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829"
ATTEMPT = "dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_attempt_consumed"
LOCK = "dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_launch_lock"

EXPECTED = {
    RELEASE: "992b11895783939f932cc45311d07f61d7738e0a800499d12b5b99bdd7bb06ca",
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
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: %s" % key)
        out[key] = value
    return out


def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def strict_load_bytes(payload):
    return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def strict_load(relative):
    with open(path(relative), "rb") as handle:
        return strict_load_bytes(handle.read())


def verify_file_seal(relative):
    full = path(relative)
    directory = os.path.dirname(full)
    base = os.path.basename(full)
    subprocess.check_call(["sha256sum", "-c", base + ".sha256"],
                          cwd=directory, stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", base + ".sha256.seal.sha256"],
                          cwd=directory, stdout=subprocess.DEVNULL)


def verify_dir_seal(relative):
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"],
                          cwd=path(relative), stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"],
                          cwd=path(relative), stdout=subprocess.DEVNULL)


def validate_release(release, candidate, contract, m895):
    require(type(release) is dict, "release type")
    require(set(release) == {
        "authorization", "claim_boundary", "date", "docs359_sha256",
        "fairness", "frozen_authorities", "future_release_chain", "identity",
        "launch_now", "prospective_attempt", "schema", "status",
    }, "release top-level keys")
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
        "run_saif": False, "run_vcs": False,
    }, "release authorization")
    require(type(release["authorization"]["max_attempts"]) is int,
            "max attempts type")
    require(release["identity"] == candidate["identity"], "identity drift")
    require(release["prospective_attempt"] == candidate["prospective_attempt"],
            "prospective attempt drift")
    require(release["fairness"] == candidate["fairness"] == contract["fairness"],
            "fairness drift")
    require(release["claim_boundary"] == candidate["claim_boundary"],
            "claim boundary drift")
    require(release["docs359_sha256"] == EXPECTED[DOCS359], "docs359 pin")
    for key, value in candidate["frozen_authorities"].items():
        require(release["frozen_authorities"].get(key) == value,
                "predecessor authority drift: %s" % key)
    frozen = release["frozen_authorities"]
    require(frozen["m892_candidate_sha256"] == EXPECTED[CANDIDATE],
            "candidate SHA pin")
    require(frozen["m892_source_closure_test_sha256"] == EXPECTED[SOURCE_TEST],
            "source test SHA pin")
    require(frozen["m895_source_review_sha256"] == EXPECTED[M895 + "/review.json"],
            "M895 review pin")
    require(frozen["m895_source_manifest_file_sha256"] ==
            EXPECTED[M895 + "/SHA256SUMS"], "M895 manifest pin")
    require(frozen["m895_source_outer_seal_file_sha256"] ==
            EXPECTED[M895 + "/SHA256SUMS.seal.sha256"], "M895 outer pin")
    require(m895["status"] ==
            "PASS100_M892_SCHEMA_REPAIR_SOURCE_FRESH_INDEPENDENT_HAMMER__NO_EDA",
            "M895 status")
    require(m895["verdict"] == "PASS" and
            m895["score_out_of_100"] == 100 and
            [m895["p0_count"], m895["p1_count"], m895["p2_count"]] == [0, 0, 0],
            "M895 score/severity")
    chain = release["future_release_chain"]
    require(set(chain) == {
        "final_review_path", "final_review_sha_caller_pinned",
        "release_binds_candidate_sha", "release_binds_source_hammer_sha",
        "source_hammer_review_path", "source_hammer_review_sha256",
    }, "future release chain keys")
    require(chain["final_review_path"] == FINAL_REVIEW_DIR + "/review.json",
            "final review coordinate")
    require(chain["final_review_sha_caller_pinned"] is True,
            "caller final SHA gate")
    require(chain["release_binds_candidate_sha"] is True and
            chain["release_binds_source_hammer_sha"] is True,
            "release binding gate")
    require(chain["source_hammer_review_path"] == M895 + "/review.json" and
            chain["source_hammer_review_sha256"] == EXPECTED[M895 + "/review.json"],
            "source hammer chain")
    require(release["prospective_attempt"]["macro_count"] == 9,
            "nine macro gate")
    require(release["prospective_attempt"]["clock_period_ns"] == 3.0,
            "3 ns gate")
    require(release["fairness"]["fair_K_zero_bit"] is False,
            "fairness claim")
    for key in ("fair_K_zero_bit", "throughput_per_mm2", "speedup",
                "system_speedup", "system", "power", "energy", "ppa",
                "paper_ppa_ready", "headline", "macro_linked_dc_result"):
        require(release["claim_boundary"][key] is False,
                "claim opened: %s" % key)


def reject_mutation(base, candidate, contract, m895, mutator):
    trial = copy.deepcopy(base)
    mutator(trial)
    try:
        validate_release(trial, candidate, contract, m895)
    except (KeyError, RuntimeError, TypeError):
        return
    raise RuntimeError("release mutation accepted")


def assert_absence():
    for relative in (CANONICAL, ATTEMPT, LOCK, FINAL_REVIEW_DIR):
        require(not os.path.lexists(path(relative)), "population exists: %s" % relative)
    require(glob.glob(path("dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_work.*")) == [],
            "work population")
    require(glob.glob(path(CANONICAL + ".failed_or_incomplete.*.quarantine")) == [],
            "quarantine population")


def run_exact_m895_production_predicate_no_eda():
    scratch = tempfile.mkdtemp(prefix="m897_m895_positive.", dir="/tmp")
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
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        require(completed.returncode == 0,
                "exact M895 production predicate no-EDA failed: %s" %
                completed.stderr.decode("utf-8", errors="replace"))
        marker = os.path.join(scratch, "PRODUCTION_SCHEMA_PASS.txt")
        require(os.path.isfile(marker), "M895 production marker absent")
        with open(marker, "r") as handle:
            text = handle.read()
        require("status=PASS_M892_PRODUCTION_M885_SCHEMA_PATH_NO_EDA" in text,
                "production predicate marker status")
        require("attempt_consumed=false" in text and
                "license_query_started=false" in text and
                "dc_shell_started=false" in text,
                "M895 positive crossed no-EDA boundary")
    finally:
        shutil.rmtree(scratch)


def main():
    assert_absence()
    for relative, expected in EXPECTED.items():
        require(os.path.isfile(path(relative)) and not os.path.islink(path(relative)),
                "missing/nonregular/symlink: %s" % relative)
        require(sha(relative) == expected, "SHA drift: %s" % relative)
    for relative in (RELEASE, CANDIDATE, CONTRACT, RUNNER, SOURCE_TEST):
        verify_file_seal(relative)
    for relative in (M895, M891, M885):
        verify_dir_seal(relative)
    release = strict_load(RELEASE)
    candidate = strict_load(CANDIDATE)
    contract = strict_load(CONTRACT)
    m895 = strict_load(M895 + "/review.json")
    m891 = strict_load(M891 + "/review.json")
    validate_release(release, candidate, contract, m895)
    require(m891["status"] ==
            "PASS_FAILURE_AUDIT__M884_RELEASE_NOT_AUTHORED__SOURCE_REVIEW_SCHEMA_MISMATCH__ADDITIVE_RUNNER_SOURCE_REPAIR_REQUIRED",
            "M891 status")
    require(m891["decision"]["m884_known_defective_command_must_not_execute"] is True,
            "M891 failure boundary")
    duplicate_negatives = [
        b'{"x":1,"x":2}', b'{"launch_now":true,"launch_now":false}',
        b'{"authorization":{},"authorization":{}}',
    ]
    nonfinite_negatives = [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}']
    for payload in duplicate_negatives + nonfinite_negatives:
        try:
            strict_load_bytes(payload)
        except ValueError:
            pass
        else:
            raise RuntimeError("strict JSON negative accepted")
    mutations = [
        lambda x: x.update({"launch_now": False}),
        lambda x: x.update({"status": "WRONG"}),
        lambda x: x["authorization"].update({"max_attempts": True}),
        lambda x: x["authorization"].update({"max_attempts": 2}),
        lambda x: x["authorization"].update({"run_dc": False}),
        lambda x: x["authorization"].update({"run_vcs": True}),
        lambda x: x["identity"].update({"runner_sha256": "0" * 64}),
        lambda x: x["frozen_authorities"].update({"m892_candidate_sha256": "0" * 64}),
        lambda x: x["frozen_authorities"].update({"m895_source_review_sha256": "0" * 64}),
        lambda x: x["future_release_chain"].update({"final_review_sha_caller_pinned": False}),
        lambda x: x["future_release_chain"].update({"source_hammer_review_sha256": "0" * 64}),
        lambda x: x["prospective_attempt"].update({"macro_count": 8}),
        lambda x: x["prospective_attempt"].update({"clock_period_ns": 2.9}),
        lambda x: x["fairness"].update({"fair_K_zero_bit": True}),
        lambda x: x["claim_boundary"].update({"speedup": True}),
        lambda x: x["claim_boundary"].update({"ppa": True}),
        lambda x: x["claim_boundary"].update({"energy": True}),
        lambda x: x.update({"unknown": False}),
    ]
    for mutator in mutations:
        reject_mutation(release, candidate, contract, m895, mutator)
    run_exact_m895_production_predicate_no_eda()
    assert_absence()
    print("PASS_M897_M892_INERT_RELEASE_AUTHOR_NO_EDA")
    print("python=%s" % sys.version.split()[0])
    print("strict_json_positive_documents=6")
    print("duplicate_key_negatives=3")
    print("nonfinite_negatives=3")
    print("release_semantic_mutations=18")
    print("exact_m895_production_predicate_no_eda=1")
    print("macro_count=9")
    print("clock_period_ns=3.0")
    print("fair_K_zero_bit=false")
    print("canonical_attempt_work_quarantine_lock_absent=true")
    print("final_hammer_output_absent=true")
    print("eda_runs=0")
    print("license_queries=0")
    print("remote_runs=0")


if __name__ == "__main__":
    main()
