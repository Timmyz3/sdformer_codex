#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1035 hammer. Synthetic/source-only; never opens real payloads."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/analyze_m1034_decoder_stratified_block_reset_windows_source_r3.py"
CHECKER = HW / "system_simulator/scripts/check_m1034_decoder_stratified_block_reset_windows_source_r3.py"
TESTS = HW / "system_simulator/tests/test_m1034_decoder_stratified_block_reset_windows_source_r3.py"
CONTRACT = HW / "contracts/m1034_decoder_stratified_block_reset_windows_source_r3_contract_r1_20260829.json"
R2 = HW / "system_simulator/scripts/analyze_m1023_decoder_stratified_block_reset_windows_source_r2.py"
R1 = HW / "system_simulator/scripts/analyze_m1014_decoder_stratified_block_reset_windows_source.py"
M1024 = HW / "reviews/m1024_m1023_decoder_stratified_block_reset_windows_source_r2_hammer_r1_20260829"
RECEIPT = HW / "reviews/m1034_decoder_stratified_block_reset_windows_source_r3_receipt_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "155ebe3e19cb42e42afe3f26358f0598e8d33bad9558f450237cffc53eb4691a",
    "checker": "794d16b7fbd27e3ab632258aa01eafa15b24e0c8820031827548cff659fbd51e",
    "tests": "5145b5112a63e9cea5ef888b91f73dd45096ecd339c37c795cd1b0a959fd76ee",
    "contract": "a91d80679f58bea11582e375b209460a450191791738ba11547e43c16df045d3",
    "r2": "8e9ce843499cbcfdfe1856e5f829218e0329cd299ce25d1ba93e3b45cd74d2b2",
    "r1": "c1fb987bd6d9921286fd9c53f3c9374d9c4779d9b3617946ab9b3d7ab11e2c64",
    "m1024_review": "8eeb6469629cbb9c30ba2bc6c3fb9f4e7a4fa1f533a8116d8210827620a528ec",
    "m1024_manifest": "b289ad10d619b97391abfb0549ef4f8b53e3431a1a1812faba5b52db36732620",
    "m1024_outer": "b438a6d2fb5238a646b10a55157eb7b5aa9cbaf7005dffc5aa00de349f9ba0db",
    "receipt_review": "be7ad97aa19f60789dce5a475452c1c8692fe8bcd12e3a048edf7ac7fe0c3997",
    "receipt_manifest": "4ddb8d5c7bef8bd2c756354530f444e2e3ecf0861013bd3a91a38f0adc1d6e19",
    "receipt_outer": "0c2c72572519374f52a39a7e92d90bacb70de08d6c5a82cdb0906fe764713531",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def verify_flat(directory, review_sha, manifest_sha, outer_sha):
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) ==
            (review_sha, manifest_sha, outer_sha), "sealed identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("./*")
        require(name not in listed and (directory / name).is_file() and
                not (directory / name).is_symlink() and sha(directory / name) == expected,
                "sealed member drift: " + directory.name + "/" + name)
        listed.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(listed == actual and outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "sealed exact-set/outer drift: " + directory.name)


def load_source():
    require(sha(SOURCE) == EXPECTED["source"], "M1034 source drift")
    spec = importlib.util.spec_from_file_location("m1035_m1034_under_hammer", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_scalar(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def strong_hard_stop_shape(value):
    """Independent recursive value-shape and numeric-leaf audit."""
    require(value["state"] == "HARD_STOP_ABOVE_10_PERCENT" and
            value["point_estimates"] is None, "not a hard-stop envelope")
    for name, interval in value["bounds"].items():
        require(isinstance(interval, list) and len(interval) == 2 and
                all(finite_scalar(item) for item in interval) and interval[0] <= interval[1],
                "bound is not a flat finite two-scalar interval: " + name)
    require(all(finite_scalar(item) for item in value["uncertainty"].values()),
            "uncertainty is not flat finite scalar")
    for row in value["coverage"]["strata"]:
        require(isinstance(row["stratum"], str) and
                isinstance(row["population_blocks"], int) and
                not isinstance(row["population_blocks"], bool) and
                isinstance(row["sample_blocks"], int) and
                not isinstance(row["sample_blocks"], bool) and
                finite_scalar(row["finite_population_fraction"]),
                "coverage row value shape drift")
    require(isinstance(value["identity"]["metric"], str), "identity metric type drift")
    allowed = []
    for path, number in load_source()._walk_numeric_paths(value):
        ok = (path.startswith("bounds.") and path.rsplit(".", 1)[-1] in ("0", "1")) or \
             path.startswith("uncertainty.") or \
             (path.startswith("coverage.strata.") and path.rsplit(".", 1)[-1] in
              ("population_blocks", "sample_blocks", "finite_population_fraction"))
        require(ok and math.isfinite(number), "forbidden numeric leaf: " + path)
        allowed.append(path)
    return allowed


def raw_envelope(module, width):
    return {
        "candidate_total_cycles_estimate": 100.0,
        "candidate_ci95": [100.0 * (1 - width), 100.0 * (1 + width)],
        "baseline_total_cycles_estimate": 120.0,
        "baseline_ci95": [120.0 * (1 - width), 120.0 * (1 + width)],
        "paired_speedup_estimate": 1.2,
        "paired_speedup_ci95": [1.2 * (1 - width), 1.2 * (1 + width)],
        "t_critical": 2.365,
        "metric": "block-reset executable schedule cycles",
        "strata": [{"stratum": "COMPUTE_REGULAR", "population_blocks": 64,
                    "sample_blocks": 8, "finite_population_fraction": 0.125,
                    "candidate_mean_cycles": 12.5, "baseline_mean_cycles": 15.0}],
    }


def author_accepts(module, value):
    try:
        module.validate_publication_envelope(value)
    except RuntimeError:
        return False
    return True


def main():
    for path, key in ((SOURCE, "source"), (CHECKER, "checker"), (TESTS, "tests"),
                      (CONTRACT, "contract"), (R2, "r2"), (R1, "r1"),
                      (DOC359, "docs359")):
        require(sha(path) == EXPECTED[key], "identity drift: " + key)
    verify_flat(M1024, EXPECTED["m1024_review"], EXPECTED["m1024_manifest"],
                EXPECTED["m1024_outer"])
    verify_flat(RECEIPT, EXPECTED["receipt_review"], EXPECTED["receipt_manifest"],
                EXPECTED["receipt_outer"])
    contract = json.loads(CONTRACT.read_text())
    receipt = json.loads((RECEIPT / "review.json").read_text())
    m1024 = json.loads((M1024 / "review.json").read_text())
    require(m1024["status"] == "FAIL_M1024_M1023_R2_SOURCE_HAMMER__BLOCK_EXECUTION_RELEASE" and
            m1024["p0_count"] == 1 and
            m1024["authorization"]["r3_ci_redaction_repair"] is True and
            m1024["authorization"]["execution_release_authorized"] is False,
            "M1024 negative authority drift")
    require(receipt["status"] == "PASS_M1034_R3_SOURCE_ONLY__M1035_INDEPENDENT_HAMMER_REQUIRED" and
            contract["launch_now"] is False and
            all(contract["claim_boundary"][key] is False for key in
                ("paper_citable", "real_window_execution_authorized", "eda_gpu_remote_used")),
            "M1034 source-only boundary drift")
    module = load_source()

    high = module.estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 1000,
        "candidate_cycles": [1, 100, 1, 100, 1, 100, 1, 100],
        "baseline_cycles": [100, 1, 100, 1, 100, 1, 100, 1],
    }])
    require(author_accepts(module, high), "author rejects its canonical hard stop")
    hard_stop_allowed_paths = strong_hard_stop_shape(high)

    old_leaks = []
    for key in ("candidate_mean_cycles", "baseline_mean_cycles"):
        attack = copy.deepcopy(high)
        attack["coverage"]["strata"][0][key] = 50.5
        old_leaks.append({"key": key, "author_rejected": not author_accepts(module, attack)})
    require(all(item["author_rejected"] for item in old_leaks),
            "M1024 original strata leak survived")

    direct_confusion = []
    for key in ("cycle", "mean", "sum", "estimate", "speedup", "FPS",
                "throughput", "latency", "time"):
        attack = copy.deepcopy(high)
        attack["identity"][key] = 7.0
        direct_confusion.append({"key": key,
                                 "author_rejected": not author_accepts(module, attack)})
    require(all(item["author_rejected"] for item in direct_confusion),
            "direct confusion key survived")

    # P0: exact outer keys are insufficient because allowed container values
    # have no recursive shape/type contract. These numeric point values hide
    # under legal keys and pass the author's validator.
    nested = []
    attacks = []
    for key in ("cycle", "mean", "sum", "estimate", "speedup", "FPS",
                "throughput", "latency", "time"):
        attack = copy.deepcopy(high)
        attack["bounds"]["candidate_total_cycles_ci95"] = {
            key: 50.5, "reported_bounds": [1.0, 100.0]}
        attacks.append(("bounds_nested_" + key, attack))
    attack = copy.deepcopy(high)
    attack["uncertainty"]["t_critical"] = {"latency_cycles": 99.0,
                                              "t_critical": 2.365}
    attacks.append(("uncertainty_nested_latency", attack))
    attack = copy.deepcopy(high)
    attack["coverage"]["strata"][0]["sample_blocks"] = {
        "cycle_sum": 404.0, "count": 8}
    attacks.append(("coverage_nested_cycle_sum", attack))
    for name, attack in attacks:
        accepted = author_accepts(module, attack)
        independent_rejected = False
        try:
            strong_hard_stop_shape(attack)
        except RuntimeError:
            independent_rejected = True
        nested.append({"attack": name, "author_accepted": accepted,
                       "independent_rejected": independent_rejected})
    escaping = [item for item in nested if item["author_accepted"]]
    require(escaping and all(item["independent_rejected"] for item in escaping),
            "recursive shape probe did not reproduce P0")

    diagnostic = module.publication_projection(raw_envelope(module, 0.06))
    candidate = module.publication_projection(raw_envelope(module, 0.04))
    require(diagnostic["state"] == "DIAGNOSTIC_5_TO_10_PERCENT" and
            diagnostic["admission"]["point_estimate_admitted"] is False and
            diagnostic["admission"]["paper_citable"] is False,
            "5-10 diagnostic semantics drift")
    require(candidate["state"] == "CANDIDATE_AT_MOST_5_PERCENT" and
            candidate["admission"]["point_estimate_admitted"] is True and
            candidate["admission"]["paper_citable"] is False,
            "at-most-5 candidate semantics drift")

    require(module.deterministic_select is module.BASE.deterministic_select and
            module.block_reset_transactions is module.BASE.block_reset_transactions and
            module.paired_replay is module.BASE.paired_replay,
            "r2 selector/reset function identity drift")
    selector_bound_rejected = False
    try:
        module.deterministic_select([{"block_id": "x"}], "COMPUTE_REGULAR", 33)
    except RuntimeError:
        selector_bound_rejected = True
    require(selector_bound_rejected, "selector >32 accepted")
    body = module.M890.synthetic_transactions(448)
    spec = module.WindowSpec("m1035-reset", "D0", "COMMIT_TAIL", 1)
    pair_r3 = module.paired_replay(body, body, spec)
    pair_r2 = module.BASE.paired_replay(body, body, spec)
    require(pair_r3["paired_reset_exact_equal"] is True and
            pair_r3["paired_reset_semantics_sha256"] ==
            pair_r2["paired_reset_semantics_sha256"],
            "r2 reset semantic SHA reuse drift")

    return {
        "schema": "m1035_m1034_decoder_r3_source_hammer_v1",
        "status": "FAIL_M1035_M1034_R3_RECURSIVE_VALUE_SHAPE_HOLE__BLOCK_EXECUTION_RELEASE",
        "verdict": "NO_GO_EXECUTION_RELEASE__AUTHOR_R4_VALUE_SHAPE_REPAIR",
        "score_out_of_100": 88,
        "p0_count": 1, "p1_count": 0, "p2_count": 0,
        "identity": {"source_sha256": sha(SOURCE), "checker_sha256": sha(CHECKER),
                     "tests_sha256": sha(TESTS), "contract_sha256": sha(CONTRACT),
                     "m1023_r2_sha256": sha(R2), "m1014_r1_sha256": sha(R1),
                     "docs359_sha256": sha(DOC359)},
        "positive": {"m1024_original_leaks_rejected": old_leaks,
                     "direct_confusion_keys_rejected": direct_confusion,
                     "canonical_hard_stop_allowed_numeric_paths": hard_stop_allowed_paths,
                     "diagnostic_5_to_10_paper_citable": False,
                     "candidate_at_most_5_paper_citable": False,
                     "selector_above_32_rejected": True,
                     "r2_selector_reset_function_identity": True,
                     "r2_reset_semantics_sha256": pair_r3["paired_reset_semantics_sha256"]},
        "p0": [{
            "id": "P0_ALLOWED_CONTAINER_VALUE_SHAPE_NOT_RECURSIVELY_TYPED",
            "finding": "The hard-stop validator fixes outer key sets but does not constrain the recursive value shape of legal bounds/uncertainty/coverage keys. Point cycle/mean/sum/estimate/speedup/FPS/throughput/latency/time values can be nested under a legal key and pass validation.",
            "escaping_attacks": escaping,
            "required_repair": "R4 must require each bound to be a flat finite length-2 scalar interval, every uncertainty leaf to be a finite scalar, coverage value types/ranges to be exact scalars, and then independently reject semantic point keys at every nested depth. Add all M1035 attacks to tests."
        }],
        "authorization": {"author_r4_source_repair": True,
                          "write_execution_release": False,
                          "write_execution_runner": False,
                          "real_window_execution": False,
                          "eda_gpu_remote": False},
        "scope": {"synthetic_only": True, "real_payload_opened": False,
                  "real_window_execution": False, "eda_gpu_remote_used": False},
        "claim_boundary": {"paper_citable": False, "decoder_complete": False,
                           "table_a_row": False, "system_speedup": False}
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
