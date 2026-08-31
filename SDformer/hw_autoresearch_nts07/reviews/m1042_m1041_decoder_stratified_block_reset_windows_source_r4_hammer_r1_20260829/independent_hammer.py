#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent receipt-blind M1042 hammer for the M1041 r4 source.

This is synthetic/source-only.  It never opens decoder payloads and never
invokes EDA, GPU, remote, or a real window execution path.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/analyze_m1041_decoder_stratified_block_reset_windows_source_r4.py"
CHECKER = HW / "system_simulator/scripts/check_m1041_decoder_stratified_block_reset_windows_source_r4.py"
TESTS = HW / "system_simulator/tests/test_m1041_decoder_stratified_block_reset_windows_source_r4.py"
CONTRACT = HW / "contracts/m1041_decoder_stratified_block_reset_windows_source_r4_contract_r1_20260829.json"
R3 = HW / "system_simulator/scripts/analyze_m1034_decoder_stratified_block_reset_windows_source_r3.py"
R2 = HW / "system_simulator/scripts/analyze_m1023_decoder_stratified_block_reset_windows_source_r2.py"
NEGATIVE = HW / "reviews/m1035_m1034_decoder_stratified_block_reset_windows_source_r3_hammer_r1_20260829"
RECEIPT = HW / "reviews/m1041_decoder_stratified_block_reset_windows_source_r4_receipt_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "09a289835e55313f7dbe06a46064a18bf9ff0caa718cb4efd2ae34097dd4456f",
    "checker": "7f681f750af175d76872bdbbef8c1a48b6982d2f5e8935c71eb92137144e68b9",
    "tests": "6074a40de3e55643cb29ba2f1e6251d56a13e7d465c43d1a790d1960b0954f62",
    "contract": "a578b3b356a0035e702abdf8e5d3227c45fb78fc8a705dcf1eb12426c93b313c",
    "r3": "155ebe3e19cb42e42afe3f26358f0598e8d33bad9558f450237cffc53eb4691a",
    "r2": "8e9ce843499cbcfdfe1856e5f829218e0329cd299ce25d1ba93e3b45cd74d2b2",
    "negative_review": "9ba3ef12a07e9508da661c99bc9c6b9088e3da2a41635acc7e3f6743962756d8",
    "negative_manifest": "ea65841f6f6690d95c335283d97cd01e3959b66b29121920376031b580b94921",
    "negative_outer": "89bb73cbd916621c9b6fd4d58e2dac2eab85c4aae651c683ea4645f21c6ae126",
    "receipt_review": "4af840abb97584b4aca2b71798c8f46916e99bfbd407d084dc03b5c309b19937",
    "receipt_manifest": "af0af82df07ffdc5dbe87ed668f384645c2e308e4e6ea35737596ff8052e8ee9",
    "receipt_outer": "075c2b0efe2b216002d4561384be0cb6b074516d7ce88dd5b82e946342c499e9",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def verify_flat(directory: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) ==
            (review_sha, manifest_sha, outer_sha),
            "sealed identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "malformed manifest: " + directory.name)
        expected, name = fields
        name = name.lstrip("./*")
        target = directory / name
        require(name and name not in listed and target.is_file() and
                not target.is_symlink() and sha(target) == expected,
                "sealed member drift: " + directory.name + "/" + name)
        listed.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(listed == actual, "sealed exact-set drift: " + directory.name)
    require(outer.read_text(encoding="utf-8").split() ==
            [manifest_sha, "SHA256SUMS"],
            "outer seal drift: " + directory.name)


def load_source():
    require(sha(SOURCE) == EXPECTED["source"], "M1041 source drift")
    spec = importlib.util.spec_from_file_location("m1042_m1041_under_hammer", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1041")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def accepted(module, value):
    try:
        module.validate_publication_envelope(value)
    except (RuntimeError, TypeError, ValueError, OverflowError):
        return False
    return True


def rejection(module, value):
    try:
        module.validate_publication_envelope(value)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        return str(error)
    raise RuntimeError("attack was accepted")


def high_result(module):
    return module.estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 1000,
        "candidate_cycles": [1, 100, 1, 100, 1, 100, 1, 100],
        "baseline_cycles": [100, 1, 100, 1, 100, 1, 100, 1],
    }])


def precise_result(module):
    return module.estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 8,
        "candidate_cycles": [10] * 8,
        "baseline_cycles": [20] * 8,
    }])


def main():
    for path, key in ((SOURCE, "source"), (CHECKER, "checker"),
                      (TESTS, "tests"), (CONTRACT, "contract"),
                      (R3, "r3"), (R2, "r2"), (DOC359, "docs359")):
        require(path.is_file() and not path.is_symlink(), "missing/nonregular: " + key)
        require(sha(path) == EXPECTED[key], "identity drift: " + key)
    verify_flat(NEGATIVE, EXPECTED["negative_review"],
                EXPECTED["negative_manifest"], EXPECTED["negative_outer"])
    verify_flat(RECEIPT, EXPECTED["receipt_review"],
                EXPECTED["receipt_manifest"], EXPECTED["receipt_outer"])

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    negative = json.loads((NEGATIVE / "review.json").read_text(encoding="utf-8"))
    receipt = json.loads((RECEIPT / "review.json").read_text(encoding="utf-8"))
    require(negative["status"] ==
            "FAIL_M1035_M1034_R3_RECURSIVE_VALUE_SHAPE_HOLE__BLOCK_EXECUTION_RELEASE" and
            negative["p0_count"] == 1 and
            negative["authorization"]["author_r4_source_repair"] is True and
            negative["authorization"]["real_window_execution"] is False,
            "M1035 negative authority drift")
    require(contract["status"] == "R4_SOURCE_ONLY__NO_REAL_WINDOW_EXECUTION" and
            contract["launch_now"] is False and
            receipt["status"] ==
            "PASS_M1041_R4_SOURCE_ONLY__M1042_INDEPENDENT_HAMMER_REQUIRED" and
            receipt["claim_boundary"]["execution_release_authorized"] is False and
            all(contract["claim_boundary"][key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "real_window_execution_authorized",
                 "eda_gpu_remote_used")), "source-only boundary drift")

    module = load_source()
    high = high_result(module)
    precise = precise_result(module)
    require(high["state"] == "HARD_STOP_ABOVE_10_PERCENT" and
            high["point_estimates"] is None and accepted(module, high),
            "canonical hard-stop invalid")
    require(precise["state"] == "CANDIDATE_AT_MOST_5_PERCENT" and
            precise["admission"]["point_estimate_admitted"] is True and
            precise["admission"]["paper_citable"] is False and
            accepted(module, precise), "canonical candidate invalid")

    # Replay the exact eleven escapes independently; every one must now fail
    # because of its semantic key before a permissive container can hide it.
    m1035 = []
    attacks = []
    for key in ("cycle", "mean", "sum", "estimate", "speedup", "FPS",
                "throughput", "latency", "time"):
        attack = copy.deepcopy(high)
        attack["bounds"]["candidate_total_cycles_ci95"] = {
            key: 50.5, "reported_bounds": [1.0, 100.0]}
        attacks.append(("bounds_nested_" + key, attack))
    attack = copy.deepcopy(high)
    attack["uncertainty"]["t_critical"] = {
        "latency_cycles": 99.0, "t_critical": 2.365}
    attacks.append(("uncertainty_nested_latency", attack))
    attack = copy.deepcopy(high)
    attack["coverage"]["strata"][0]["sample_blocks"] = {
        "cycle_sum": 404.0, "count": 8}
    attacks.append(("coverage_nested_cycle_sum", attack))
    require(len(attacks) == 11, "M1035 replay cardinality drift")
    for name, attack in attacks:
        message = rejection(module, attack)
        require("semantic point key forbidden at depth" in message,
                "M1035 attack rejected for wrong reason: " + name)
        m1035.append({"attack": name, "rejected": True, "reason": message})

    semantic_aliases = []
    for alias in ("runtimeEstimate", "cycle_sums", "latencies",
                  "THROUGHPUT", "meanValues", "fps", "speedups",
                  "executionCycles"):
        attack = copy.deepcopy(high)
        attack["bounds"]["candidate_total_cycles_ci95"] = {
            "nested": {"deeper": {alias: 1.0}}}
        message = rejection(module, attack)
        require("semantic point key forbidden at depth" in message,
                "deep alias rejected for wrong reason: " + alias)
        semantic_aliases.append(alias)

    bounds_attacks = []
    for value in ([1.0], [1.0, 2.0, 3.0], [[1.0], 2.0],
                  {"lo": 1.0, "hi": 2.0}, (1.0, 2.0),
                  [False, 2.0], [1.0, math.nan], [1.0, math.inf],
                  [2.0, 1.0]):
        attack = copy.deepcopy(high)
        attack["bounds"]["baseline_total_cycles_ci95"] = value
        bounds_attacks.append(rejection(module, attack))
    require(len(bounds_attacks) == 9, "bounds attack cardinality drift")

    uncertainty_attacks = []
    for key, value in (("t_critical", False), ("t_critical", [2.365]),
                       ("t_critical", {"value": 2.365}),
                       ("t_critical", math.nan), ("t_critical", math.inf),
                       ("t_critical", 0.0),
                       ("maximum_relative_halfwidth", -0.1)):
        attack = copy.deepcopy(high)
        attack["uncertainty"][key] = value
        uncertainty_attacks.append(rejection(module, attack))
    attack = copy.deepcopy(high)
    attack["uncertainty"]["maximum_relative_halfwidth"] += 0.01
    uncertainty_attacks.append(rejection(module, attack))
    require(len(uncertainty_attacks) == 8, "uncertainty attack cardinality drift")

    coverage_attacks = []
    for key, value in (("population_blocks", True),
                       ("population_blocks", 1000.0),
                       ("population_blocks", 0),
                       ("sample_blocks", False), ("sample_blocks", 8.0),
                       ("sample_blocks", [8]), ("sample_blocks", 0),
                       ("sample_blocks", 1001),
                       ("finite_population_fraction", math.nan),
                       ("finite_population_fraction", math.inf),
                       ("finite_population_fraction", 0.0),
                       ("finite_population_fraction", 1.1),
                       ("finite_population_fraction", 0.5)):
        attack = copy.deepcopy(high)
        attack["coverage"]["strata"][0][key] = value
        coverage_attacks.append(rejection(module, attack))
    attack = copy.deepcopy(high)
    attack["coverage"]["strata"][0]["stratum"] = "UNKNOWN"
    coverage_attacks.append(rejection(module, attack))
    attack = copy.deepcopy(high)
    attack["coverage"]["strata"].append(copy.deepcopy(
        attack["coverage"]["strata"][0]))
    coverage_attacks.append(rejection(module, attack))
    require(len(coverage_attacks) == 15, "coverage attack cardinality drift")

    points_attacks = []
    for value in (False, [80.0], math.nan, math.inf, 0.0, -1.0):
        attack = copy.deepcopy(precise)
        attack["point_estimates"]["candidate_total_cycles"] = value
        points_attacks.append(rejection(module, attack))
    require(len(points_attacks) == 6, "point attack cardinality drift")

    status_attacks = []
    state_cases = (
        (high, "status", "POINT_CANDIDATE_FOR_LATER_INDEPENDENT_RELEASE"),
        (high, "action", "NONE"),
        (precise, "status", "DIAGNOSTIC_POINT_NOT_ADMITTED_CI95_5_TO_10_PERCENT"),
        (precise, "action", "ADAPT_SAMPLE_BY_VARIANCE_BELOW_CAP"),
        (precise, "admitted", False),
        (precise, "paper", True),
    )
    for original, kind, value in state_cases:
        attack = copy.deepcopy(original)
        if kind == "status":
            attack["status"] = value
        elif kind == "action":
            attack["admission"]["adaptive_action"] = value
        elif kind == "admitted":
            attack["admission"]["point_estimate_admitted"] = value
        else:
            attack["admission"]["paper_citable"] = value
        status_attacks.append(rejection(module, attack))
    require(len(status_attacks) == 6, "status attack cardinality drift")

    # Schema aliases, extra keys, bool-as-number, and non-exact containers.
    schema_attacks = []
    attack = copy.deepcopy(high); attack["cycle_sum"] = 1.0
    schema_attacks.append(rejection(module, attack))
    attack = copy.deepcopy(high); attack["bounds"]["runtime"] = [1.0, 2.0]
    schema_attacks.append(rejection(module, attack))
    attack = copy.deepcopy(high); attack["identity"]["metric_alias"] = "x"
    schema_attacks.append(rejection(module, attack))
    attack = copy.deepcopy(high); attack["coverage"]["strata"] = []
    schema_attacks.append(rejection(module, attack))
    attack = copy.deepcopy(high); attack["state"] = "UNKNOWN"
    schema_attacks.append(rejection(module, attack))
    require(len(schema_attacks) == 5, "schema attack cardinality drift")

    # Independently prove synthetic self-tests cannot touch real transaction
    # surfaces even if a future indirect call tries to do so.
    touched = []
    def forbidden(*args, **kwargs):
        touched.append((args, kwargs))
        raise RuntimeError("REAL_PAYLOAD_FORBIDDEN_BY_M1042")
    original_prefix = module.M946.prefix_transactions
    original_real_prefix = module.M890.real_prefix_transactions
    module.M946.prefix_transactions = forbidden
    module.M890.real_prefix_transactions = forbidden
    try:
        self_test = module.self_test()
    finally:
        module.M946.prefix_transactions = original_prefix
        module.M890.real_prefix_transactions = original_real_prefix
    require(not touched and self_test["m1035_attack_count"] == 11 and
            self_test["real_payload_opened"] is False and
            self_test["real_window_execution"] is False,
            "source self-test touched real payload surface")

    unit = subprocess.run(
        [sys.executable, str(TESTS)], cwd=str(HW), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(unit.returncode == 0 and "Ran 14 tests" in unit.stdout and
            "OK" in unit.stdout, "author unit tests failed")
    checker = subprocess.run(
        [sys.executable, str(CHECKER), "--check"], cwd=str(HW), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(checker.returncode == 0 and
            "PASS_M1041_R4_SOURCE_STATIC_CHECK__NO_REAL_EXECUTION" in checker.stdout,
            "author static checker failed")

    require(module.deterministic_select is module.BASE.deterministic_select and
            module.block_reset_transactions is module.BASE.block_reset_transactions and
            module.paired_replay is module.BASE.paired_replay,
            "M1034 selector/reset function identity drift")
    selector_bound_rejected = False
    try:
        module.deterministic_select([{"block_id": "x"}], "COMPUTE_REGULAR", 33)
    except RuntimeError:
        selector_bound_rejected = True
    require(selector_bound_rejected, "selector >32 accepted")
    body = module.M890.synthetic_transactions(448)
    spec = module.WindowSpec("m1042-reset", "D0", "COMMIT_TAIL", 1)
    pair_r4 = module.paired_replay(body, body, spec)
    pair_r3 = module.BASE.paired_replay(body, body, spec)
    require(pair_r4["paired_reset_exact_equal"] is True and
            pair_r4["paired_reset_semantics_sha256"] ==
            pair_r3["paired_reset_semantics_sha256"],
            "frozen reset semantics drift")

    return {
        "schema": "m1042_m1041_decoder_r4_source_hammer_v1",
        "status": "PASS_M1042_M1041_R4_INDEPENDENT_SOURCE_HAMMER__GO_EXECUTION_RELEASE_SOURCE_ONLY",
        "verdict": "GO_WRITE_SEPARATE_EXECUTION_RELEASE__DO_NOT_EXECUTE_FROM_M1041",
        "score_out_of_100": 99,
        "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "identity": {
            "source_sha256": sha(SOURCE), "checker_sha256": sha(CHECKER),
            "tests_sha256": sha(TESTS), "contract_sha256": sha(CONTRACT),
            "m1034_r3_sha256": sha(R3), "m1023_r2_sha256": sha(R2),
            "receipt_review_sha256": sha(RECEIPT / "review.json"),
            "receipt_manifest_sha256": sha(RECEIPT / "SHA256SUMS"),
            "receipt_outer_sha256": sha(RECEIPT / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha(DOC359),
        },
        "independent_attacks": {
            "m1035_exact_escaping_attacks_rejected": m1035,
            "m1035_count": len(m1035),
            "deep_semantic_aliases_rejected": semantic_aliases,
            "deep_semantic_alias_count": len(semantic_aliases),
            "bounds_shape_range_attacks_rejected": len(bounds_attacks),
            "uncertainty_shape_range_attacks_rejected": len(uncertainty_attacks),
            "coverage_type_range_identity_attacks_rejected": len(coverage_attacks),
            "point_value_attacks_rejected": len(points_attacks),
            "state_status_action_boundary_attacks_rejected": len(status_attacks),
            "schema_alias_extra_key_attacks_rejected": len(schema_attacks),
            "total_attack_cases_rejected": (len(m1035) + len(semantic_aliases) +
                len(bounds_attacks) + len(uncertainty_attacks) +
                len(coverage_attacks) + len(points_attacks) +
                len(status_attacks) + len(schema_attacks)),
        },
        "positive": {
            "canonical_hard_stop_valid": True,
            "canonical_candidate_valid_but_not_paper_citable": True,
            "author_unittest": "14/14 PASS",
            "author_static_checker": "PASS",
            "m1035_all_rejected_for_semantic_reason": True,
            "real_payload_surface_tripwire_count": len(touched),
            "r3_selector_reset_function_identity": True,
            "selector_above_32_rejected": True,
            "reset_semantics_sha256": pair_r4["paired_reset_semantics_sha256"],
        },
        "authorization": {
            "write_separate_execution_release_source": True,
            "write_execution_runner": False,
            "execute_real_windows": False,
            "eda_gpu_remote": False,
        },
        "scope": {
            "synthetic_only": True, "real_payload_opened": False,
            "real_window_execution": False, "eda_gpu_remote_used": False,
        },
        "claim_boundary": {
            "paper_citable": False, "decoder_complete": False,
            "table_a_row": False, "system_speedup": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
