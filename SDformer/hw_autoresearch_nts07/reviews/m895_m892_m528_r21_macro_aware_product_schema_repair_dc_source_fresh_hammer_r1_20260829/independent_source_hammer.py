#!/usr/bin/env python3
"""Fresh independent, no-EDA source hammer for the M892 schema repair."""

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
RUNNER = ROOT / "dc_handoff/scripts/run_dc_m892_m528_r21_macro_aware_product_schema_repair_exact_sha_r1.sh"
OLD_RUNNER = ROOT / "dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_exact_sha_r1.sh"
CONTRACT = ROOT / "contracts/m892_m528_r21_macro_aware_product_dc_schema_repair_source_only_contract_r1_20260829.json"
OLD_CONTRACT = ROOT / "contracts/m884_m528_r21_macro_aware_product_dc_source_only_contract_r1_20260829.json"
CANDIDATE = ROOT / "contracts/m892_m528_r21_macro_aware_product_schema_repair_dc_launch_candidate_source_only_r1_20260829.json"
OLD_CANDIDATE = ROOT / "contracts/m884_m528_r21_macro_aware_product_dc_launch_candidate_source_only_r1_20260829.json"
AUTHOR_TEST = ROOT / "verif_m528_dw1rw/test_m892_m528_r21_macro_dc_schema_repair_source_closure.py"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"
HANDOFF = ROOT / "reviews/m892_m528_r21_macro_aware_product_schema_repair_dc_source_author_handoff_r1_20260829"
REQUEST = ROOT / "reviews/m895_m892_m528_r21_macro_aware_product_schema_repair_dc_source_hammer_REQUEST_r1_20260829"
M885 = ROOT / "reviews/m885_m884_m528_r21_macro_aware_product_dc_source_fresh_hammer_r1_20260829"
M891 = ROOT / "reviews/m891_m884_macro_dc_release_author_preflight_failure_audit_r1_20260829"
RUNS = ROOT / "dc_handoff/runs"
CANONICAL = RUNS / "m892_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829"
ATTEMPT = RUNS / ".m892_m528_r21_macro_aware_product_dc_attempt_consumed"
LOCK = RUNS / ".m892_m528_r21_macro_aware_product_dc_launch_lock"


EXPECTED = {
    RUNNER: "a0c07f8740a830d7a3e99ae1bf6dd2f3f55c4f77102c7b6a0eeb1746694d5d9f",
    CONTRACT: "5b5ec1ecb8fa75299bd32b5776759a3921dfc7329e27a3d48a545c0a23e1267d",
    CANDIDATE: "79f4b0a6d3d16c7977166823eb318fd00a1670d2f67f2f58e4439caad26ad1c0",
    AUTHOR_TEST: "419ad48854b5b987100bad0914b2fb1fbaf1a989f14f45d5d523ca3fc769f611",
    HANDOFF / "handoff.json": "e3f0004d56973791e094cd3ca8ffd6221a842f826aa83a224ad51da0e8173049",
    HANDOFF / "SHA256SUMS": "f9e5ce3c0c560b04b1790cbc6210140e86861e367fdde792df83486a3d8ed726",
    HANDOFF / "SHA256SUMS.seal.sha256": "af733b34c5f4a3232d715694cd5b72aaef29a7b8bfd328fc2cb1896840e427db",
    REQUEST / "request.json": "7eda675e66197a7f47d7ed7f40cc91633160ee8dabc45dc7c718731bb799b314",
    REQUEST / "SHA256SUMS": "52184755eafea31c083a412ce925e66326492c4966efecdb569d9800df40546e",
    REQUEST / "SHA256SUMS.seal.sha256": "98a15cd3260033bf6b9ab4e8b6155e316dbd4561d4811ea8faf2e9bbad0bff0e",
    M885 / "review.json": "607b3898c05ce816b25f8cff26ffe01991d603db5e106707e2b7f8dc80d91b95",
    M885 / "SHA256SUMS": "7e8c08587529b574049e2dd5e43bdd9f205bf9cf8e5dbf42397ed1cce6dd3497",
    M885 / "SHA256SUMS.seal.sha256": "df48b418dd8c73b4f0e2920517c3144f158a900baa23c38727b0ea4cc53b1c59",
    M891 / "review.json": "883829d8017b2656161d5e3f7f2300c38ad214cc308dbcc06f761b3b875a8792",
    M891 / "SHA256SUMS": "b71788863c24a335c257b38bf0c66dc385039ff6df41ff749b2d46e6d631c073",
    M891 / "SHA256SUMS.seal.sha256": "5dfc669879034caa016332e7553347949b8ee82dfe8761700c4be6eddafe7f20",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PYTHONS = [
    "/usr/libexec/platform-python3.6",
    "/opt/anaconda3/envs/pytorch310/bin/python3.10",
]


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
    require(path.is_dir() and not path.is_symlink(), "bad evidence tree: " + str(path))
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


def assert_population_absent():
    require(not CANONICAL.exists(), "canonical exists")
    require(not ATTEMPT.exists(), "attempt exists")
    require(not LOCK.exists(), "launch lock exists")
    require(not list(RUNS.glob(".m892_m528_r21_macro_aware_product_dc_work.*")),
            "work population exists")
    require(not list(RUNS.glob(
        "m892_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829.failed_or_incomplete.*")),
        "quarantine population exists")


def closed_false_claims(claims):
    for key in ["fair_K_zero_bit", "throughput_per_mm2", "speedup",
                "system_speedup", "system", "power", "energy", "ppa",
                "physical_route", "paper_ppa_ready", "headline"]:
        require(claims[key] is False, "claim escaped: " + key)


def validate_identity_and_semantics():
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), "missing/symlink: " + str(path))
        require(sha(path) == expected, "SHA drift: " + str(path))
    for path in [RUNNER, CONTRACT, CANDIDATE, AUTHOR_TEST]:
        verify_file_seal(path)
    for tree in [HANDOFF, REQUEST, M885, M891]:
        verify_tree(tree)

    contract = strict_load(CONTRACT)
    old_contract = strict_load(OLD_CONTRACT)
    candidate = strict_load(CANDIDATE)
    old_candidate = strict_load(OLD_CANDIDATE)
    m885 = strict_load(M885 / "review.json")
    m891 = strict_load(M891 / "review.json")

    require(m885["verdict"] == "PASS" and m885["score_out_of_100"] == 100,
            "M885 positive authority drift")
    require([m885["p0_count"], m885["p1_count"], m885["p2_count"]] == [0, 0, 0],
            "M885 severity drift")
    require(m891["verdict"] == "PASS_FAILURE_AUDIT" and
            m891["score_out_of_100"] == 100 and
            m891["decision"]["m884_known_defective_command_must_not_execute"] is True,
            "M891 failure authority drift")

    # All hardware, foundry, resource, and claim semantics are byte-equivalent
    # to M884; only successor identity and source-review evidence are additive.
    for key in ["authorization", "claim_boundary", "fairness", "foundry_views",
                "physical_point", "tool_identity", "docs359_sha256"]:
        require(contract[key] == old_contract[key], "M884 semantic drift: " + key)
    common_files = set(contract["exact_files"]) & set(old_contract["exact_files"])
    require(len(common_files) == 8, "unexpected common exact-file population")
    for key in common_files:
        require(contract["exact_files"][key] == old_contract["exact_files"][key],
                "common exact file drift: " + key)
    common_auth = set(contract["frozen_authorities"]) & set(old_contract["frozen_authorities"])
    for key in common_auth:
        require(contract["frozen_authorities"][key] == old_contract["frozen_authorities"][key],
                "common authority drift: " + key)
    require(set(contract["frozen_authorities"]) - set(old_contract["frozen_authorities"]) == {
        "m885_source_review_path", "m885_source_review_sha256",
        "m885_source_manifest_file_sha256", "m885_source_outer_seal_file_sha256",
        "m891_failure_review_path", "m891_failure_review_sha256",
        "m891_failure_manifest_file_sha256", "m891_failure_outer_seal_file_sha256",
    }, "unexpected additive authority set")
    for key in ["authorization", "claim_boundary", "fairness", "docs359_sha256"]:
        require(candidate[key] == old_candidate[key], "candidate semantic drift: " + key)
    closed_false_claims(contract["claim_boundary"])
    closed_false_claims(candidate["claim_boundary"])
    require(contract["fairness"] == {
        "candidate_point_only": True, "fair_K_zero_bit": False,
        "zero_rtl_baseline_present": False, "bit_rtl_baseline_present": False,
    }, "fairness boundary drift")

    # Reproducible normalized runner diff: the remaining hunks are only new
    # M885/M891 bindings, no-EDA schema exercise, and the repaired predicate.
    normalized = RUNNER.read_text()
    replacements = [
        ("M892", "M884"), ("m892", "m884"),
        ("run_dc_m884_m528_r21_macro_aware_product_schema_repair_exact_sha_r1",
         "run_dc_m884_m528_r21_macro_aware_product_exact_sha_r1"),
        ("m884_m528_r21_macro_aware_product_dc_schema_repair_source_only_contract",
         "m884_m528_r21_macro_aware_product_dc_source_only_contract"),
        ("m884_m528_r21_macro_aware_product_schema_repair_dc_launch_candidate_source_only",
         "m884_m528_r21_macro_aware_product_dc_launch_candidate_source_only"),
        ("m884_m528_r21_macro_aware_product_schema_repair_dc_launch_release",
         "m884_m528_r21_macro_aware_product_dc_launch_release"),
        ("m884_m528_r21_macro_aware_product_schema_repair_dc_final_launch_hammer",
         "m884_m528_r21_macro_aware_product_dc_final_launch_hammer"),
        ("test_m884_m528_r21_macro_dc_schema_repair_source_closure",
         "test_m884_m528_r21_macro_dc_source_closure"),
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    diff = "".join(difflib.unified_diff(
        OLD_RUNNER.read_text().splitlines(True), normalized.splitlines(True),
        fromfile=str(OLD_RUNNER.relative_to(ROOT)), tofile="M892_NORMALIZED"))
    require(hashlib.sha256(diff.encode("utf-8")).hexdigest() ==
            "fe758e9d84297aaf578003b34e42a9be5576cfc5ae2b61175cb56da72de12498",
            "normalized M884-to-M892 runner diff drift")
    require(".score_out_of_100 == 100" in diff and
            "[.p0_count,.p1_count,.p2_count] == [0,0,0]" in diff and
            "m884_production_schema_selftest" in diff,
            "schema repair absent from normalized diff")
    return contract, candidate, len(diff.splitlines())


def cross_python_author_closure():
    outputs = []
    for executable in PYTHONS:
        completed = subprocess.run([executable, str(AUTHOR_TEST)], cwd=str(ROOT),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(completed.returncode == 0 and
                b"PASS M892 source closure" in completed.stdout,
                "source closure failed under %s: %s" %
                (executable, completed.stderr.decode("utf-8", errors="replace")))
        outputs.append({
            "version": subprocess.check_output([executable, "--version"],
                                                stderr=subprocess.STDOUT).decode().strip(),
            "receipt": completed.stdout.decode().strip(),
        })
        assert_population_absent()
    return outputs


def run_production_fixture(root, fixture=None):
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "M892_NO_EDA_FULL_PATH_SELFTEST": "1",
        "M892_NO_EDA_PRODUCTION_SCHEMA_SELFTEST": "1",
        "M892_NO_EDA_SELFTEST_ROOT": str(root),
        "M892_EXPECTED_DC_RUNNER_SHA256": EXPECTED[RUNNER],
        "M892_EXPECTED_DC_ADMISSION_SHA256": EXPECTED[CANDIDATE],
    }
    if fixture is not None:
        env["M892_NO_EDA_SOURCE_REVIEW_FIXTURE"] = str(fixture)
        env["M892_EXPECTED_NO_EDA_SOURCE_REVIEW_SHA256"] = sha(fixture)
    return subprocess.run([str(RUNNER)], cwd=str(ROOT), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def independent_production_suite():
    tmp = Path(tempfile.mkdtemp(prefix="m895_m892_production.", dir="/tmp"))
    try:
        positive = run_production_fixture(tmp)
        require(positive.returncode == 0, "exact M885 production path failed")
        marker = (tmp / "PRODUCTION_SCHEMA_PASS.txt").read_text()
        require("PASS_M892_PRODUCTION_M885_SCHEMA_PATH_NO_EDA" in marker and
                "attempt_consumed=false" in marker and
                "license_query_started=false" in marker and
                "dc_shell_started=false" in marker,
                "positive marker boundary drift")
        (tmp / "PRODUCTION_SCHEMA_PASS.txt").unlink()

        base = strict_load(M885 / "review.json")
        fixtures = []
        old = copy.deepcopy(base)
        for field in ["score_out_of_100", "p0_count", "p1_count", "p2_count"]:
            old.pop(field)
        old["score_100"] = 100
        old["severity_counts"] = {"p0": 0, "p1": 0, "p2": 0}
        fixtures.append(("old_fields", json.dumps(old, sort_keys=True) + "\n"))
        for field in ["score_out_of_100", "p0_count", "p1_count", "p2_count"]:
            missing = copy.deepcopy(base)
            missing.pop(field)
            fixtures.append(("missing_" + field, json.dumps(missing, sort_keys=True) + "\n"))
        for field in ["p0_count", "p1_count", "p2_count"]:
            nonzero = copy.deepcopy(base)
            nonzero[field] = 1
            fixtures.append((field + "_nonzero", json.dumps(nonzero, sort_keys=True) + "\n"))
        canonical = json.dumps(base, sort_keys=True)
        fixtures.append(("duplicate", canonical.replace(
            '"score_out_of_100": 100',
            '"score_out_of_100": 100, "score_out_of_100": 99', 1) + "\n"))
        for name, literal in [("nan", "NaN"), ("infinity", "Infinity"),
                              ("minus_infinity", "-Infinity")]:
            fixtures.append((name, canonical.replace(
                '"score_out_of_100": 100',
                '"score_out_of_100": ' + literal, 1) + "\n"))

        rejected = []
        for name, payload in fixtures:
            fixture = tmp / (name + ".json")
            fixture.write_text(payload)
            completed = run_production_fixture(tmp, fixture)
            require(completed.returncode != 0, "negative escaped: " + name)
            require(not (tmp / "PRODUCTION_SCHEMA_PASS.txt").exists(),
                    "negative produced positive marker: " + name)
            assert_population_absent()
            rejected.append(name)
        require(len(rejected) == 12, "negative count drift")
        return rejected
    finally:
        shutil.rmtree(str(tmp))


def semantic_negative_suite(contract, candidate):
    def validate(c, a):
        require(c["authorization"] == contract["authorization"], "contract auth")
        require(c["fairness"] == contract["fairness"], "contract fairness")
        closed_false_claims(c["claim_boundary"])
        require(a["authorization"] == candidate["authorization"], "candidate auth")
        require(a["fairness"] == candidate["fairness"], "candidate fairness")
        closed_false_claims(a["claim_boundary"])
        require(a["launch_now"] is False, "launch escaped")
        require(a["identity"]["runner_sha256"] == EXPECTED[RUNNER], "runner identity")
        require(a["identity"]["source_contract_sha256"] == EXPECTED[CONTRACT],
                "contract identity")

    mutations = []
    for key in ["run_dc_now", "run_vcs_now", "run_formality_now", "run_pt_now",
                "run_ptpx_now", "run_saif_now", "run_remote_now"]:
        mutations.append(lambda c, a, key=key: c["authorization"].__setitem__(key, True))
    for key in ["fair_K_zero_bit", "speedup", "system_speedup", "energy", "ppa",
                "throughput_per_mm2", "paper_ppa_ready", "headline"]:
        mutations.append(lambda c, a, key=key: c["claim_boundary"].__setitem__(key, True))
    mutations += [
        lambda c, a: a.__setitem__("launch_now", True),
        lambda c, a: a["authorization"].__setitem__("run_dc", True),
        lambda c, a: a["authorization"].__setitem__("max_attempts", 1),
        lambda c, a: a["fairness"].__setitem__("fair_K_zero_bit", True),
        lambda c, a: a["identity"].__setitem__("runner_sha256", "0" * 64),
        lambda c, a: a["identity"].__setitem__("source_contract_sha256", "0" * 64),
    ]
    rejected = 0
    for mutation in mutations:
        c = copy.deepcopy(contract)
        a = copy.deepcopy(candidate)
        mutation(c, a)
        try:
            validate(c, a)
        except RuntimeError:
            rejected += 1
    require(rejected == len(mutations) == 21, "semantic negative escaped")
    return rejected


def main():
    assert_population_absent()
    contract, candidate, normalized_diff_lines = validate_identity_and_semantics()
    python_receipts = cross_python_author_closure()
    production_negatives = independent_production_suite()
    semantic_negatives = semantic_negative_suite(contract, candidate)
    assert_population_absent()
    require(sha(DOCS359) == EXPECTED[DOCS359], "docs359 changed")
    summary = {
        "status": "PASS_M895_M892_FRESH_INDEPENDENT_SOURCE_HAMMER_NO_EDA",
        "python_receipts": python_receipts,
        "production_positive": 1,
        "production_negatives": production_negatives,
        "semantic_negatives": semantic_negatives,
        "normalized_runner_diff_lines": normalized_diff_lines,
        "canonical_count": 0,
        "attempt_count": 0,
        "work_count": 0,
        "quarantine_count": 0,
        "license_queries": 0,
        "eda_runs": 0,
    }
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
