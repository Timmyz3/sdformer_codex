#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Exact-contract, source-only successor to the failed M1356 C2 launch gate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


HW = Path(__file__).resolve().parents[1]
M1356_CHECKER = HW / "verif_m1356_c2_activity_final_launch/static_check_m1356_c2_activity_final_launch_source.py"
M1356_CHECKER_SHA256 = "da2b9bc657c9a6555fda85a63266821f6080ebac1c4ab3c5ac297f53aad0065a"
M1356_TEST = HW / "verif_m1356_c2_activity_final_launch/test_m1356_c2_activity_final_launch_source.py"
M1356_TEST_SHA256 = "f09f4e86d8fc4942fd5aaf66b86d65e0caa37c6663350dad5e5807f40dea551d"
M1356_CONTRACT = HW / "contracts/m1356_c2_mapped_activity_vcs_saif_final_launch_authority_source_contract_r1_20260831.json"
M1356_CONTRACT_SHA256 = "546798fa9b634fcaf479e4b494887cc5fd1b6d2cbf5663794a5ee7e9cf4d38c8"
M1357 = HW / "reviews/m1357_m1356_c2_mapped_activity_vcs_saif_final_launch_authority_blind_hammer_r1_20260831"
M1357_REVIEW_SHA256 = "ebc327cd48acf8eb3dda8f096174d00c496b73011878ffe300a43525d057a834"
M1357_MANIFEST_SHA256 = "29878e72de4102c967f04ccf67c0ccdfd7a2c65257082f5be8b081ccef793977"
M1357_OUTER_SHA256 = "b45ae666cec7f72dd9eb053a4260711beadf84f8b10aa80db2aa750b3f41c4fd"

CHECKER = Path(__file__).resolve()
TEST = HW / "verif_m1361_c2_activity_final_launch_exact/test_m1361_c2_activity_final_launch_exact_source.py"
CONTRACT = HW / "contracts/m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_author_r1_20260831"
FUTURE_BLIND = HW / "reviews/m1362_m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_blind_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CLAIMS = ("functional_vcs_verified", "production_saif", "ptpx", "power",
          "energy", "performance", "system_speedup", "paper_ppa_ready", "headline")
EXACT_CLAIMS = {key: False for key in CLAIMS}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    need(path.is_file() and not path.is_symlink(), "JSON absent/nonregular")
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError("nonfinite JSON token: " + token)))
    need(type(value) is dict, "JSON root must be object")
    return value


def load_m1356():
    need(sha(M1356_CHECKER) == M1356_CHECKER_SHA256, "M1356 checker drift")
    spec = importlib.util.spec_from_file_location("m1361_sealed_m1356", M1356_CHECKER)
    need(spec is not None and spec.loader is not None, "cannot load M1356")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = load_m1356()


def verify_dir(root: Path, review_sha: str, manifest_sha: str, outer_sha: str) -> dict[str, Any]:
    need(root.is_dir() and not root.is_symlink(), "sealed directory invalid")
    review, manifest, outer = root / "review.json", root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    need(sha(review) == review_sha and sha(manifest) == manifest_sha and sha(outer) == outer_sha,
         "sealed directory exact SHA drift")
    need(outer.read_text(encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
         "outer seal content drift")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, "manifest row field count")
        digest, name = fields; name = name.lstrip("*"); rel = Path(name)
        need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
             not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "manifest row invalid")
        member = root / rel
        need(member.is_file() and not member.is_symlink() and sha(member) == digest,
             "manifest member drift: " + name)
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "sealed directory exact population drift")
    return strict_json(review)


def verify_m1357() -> dict[str, Any]:
    review = verify_dir(M1357, M1357_REVIEW_SHA256, M1357_MANIFEST_SHA256, M1357_OUTER_SHA256)
    need(review.get("status") == "FAIL_DO_NOT_LAUNCH__ADDITIVE_SOURCE_SUCCESSOR_REQUIRED" and
         review.get("p0_count") == 1 and
         review.get("fresh_hammer", {}).get("attacks") == 34 and
         review.get("fresh_hammer", {}).get("false_negatives") == 30 and
         review.get("authorization", {}) == {
             "additive_source_successor": True, "launch": False,
             "license_query": False, "vcs": False, "simv": False,
             "saif": False, "ptpx": False, "eda": False,
             "automatic_retry": False} and
         review.get("claim_boundary") == EXACT_CLAIMS,
         "M1357 verdict/authorization drift")
    return review


def expected_contract() -> dict[str, Any]:
    return {
        "schema": "m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_r1_v1",
        "status": "SOURCE_ONLY__M1357_REPAIRED__FRESH_M1362_BLIND_REQUIRED",
        "date": "2026-08-31",
        "purpose": "Additive exact-set/value successor to M1356 after M1357 found 30 false negatives in one-shot, resource, receipt and authorization contract semantics.",
        "identity": {
            "runner": str(OLD.RUNNER.relative_to(HW)),
            "runner_sha256": OLD.RUNNER_SHA256,
            "checker": str(CHECKER.relative_to(HW)),
            "checker_sha256": sha(CHECKER),
            "test": str(TEST.relative_to(HW)),
            "test_sha256": sha(TEST),
            "m1350_checker_sha256": OLD.M1350_CHECKER_SHA256,
            "m1350_contract_sha256": OLD.M1350_CONTRACT_SHA256,
            "m1353_review_sha256": OLD.M1353_REVIEW_SHA256,
            "m1353_manifest_sha256": OLD.M1353_MANIFEST_SHA256,
            "m1353_outer_file_sha256": OLD.M1353_OUTER_SHA256,
            "m1356_checker_sha256": M1356_CHECKER_SHA256,
            "m1356_test_sha256": M1356_TEST_SHA256,
            "m1356_contract_sha256": M1356_CONTRACT_SHA256,
            "m1357_review_sha256": M1357_REVIEW_SHA256,
            "m1357_manifest_sha256": M1357_MANIFEST_SHA256,
            "m1357_outer_file_sha256": M1357_OUTER_SHA256,
            "ucli_sha256": OLD.UCLI_SHA256,
        },
        "failed_predecessor": {
            "source": "M1356",
            "blind_review": "M1357",
            "attacks": 34,
            "false_negatives": 30,
            "repair": "exact top-level and nested key/value equality",
        },
        "one_shot": {
            "attempt_namespace": "results/.m1344_c2_headline_mapped_production_activity_vcs_attempt_consumed",
            "result_namespace": "results/m1344_c2_headline_mapped_production_activity_vcs_r1_20260831",
            "attempt_fresh_during_authoring": True,
            "attempt_published_exactly_once_no_replace": True,
            "automatic_retry": False,
            "maximum_vcs_compiles_after_future_authorization": 2,
            "maximum_simv_runs_after_future_authorization": 10,
        },
        "resource_fail_close": {
            "same_uid_blocked_processes": [
                "vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
                "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec", "common_shell_exe"],
            "collision_gate_before_license": True,
            "collision_gate_after_license_before_attempt": True,
            "memory_and_commit_headroom_before_attempt": True,
            "namespace_residue_before_attempt": "reject",
        },
        "receipt_contract": {
            "paths": ["failure", "attempt", "success"],
            "identity_sha_keys_each": list(OLD.M.IDENTITY_KEYS),
            "identity_key_order_exact": True,
            "active_value_expressions_exact": True,
            "failure": {
                "automatic_retry": False,
                "canonical_result": False,
                "raw_private_build_citable": False,
            },
            "attempt": {
                "automatic_retry": False,
                "maximum_vcs_compiles": 2,
                "maximum_simv_runs": 10,
            },
            "success": {
                "attempt_consumed": True,
                "vcs_compiles": 2,
                "simv_runs": 10,
                "automatic_retry": False,
                "claim_boundary_exact": True,
            },
        },
        "future_blind": {
            "path": str(FUTURE_BLIND.relative_to(HW)),
            "must_be_absent_during_authoring": True,
            "fresh_different_author": True,
            "zero_false_negatives_required": True,
        },
        "authorization": {
            "source_authoring": True,
            "source_only_tests": True,
            "different_author_blind_hammer": True,
            "launch_authorized": False,
            "license_query": False,
            "vcs": False,
            "simv": False,
            "saif": False,
            "eda": False,
            "automatic_retry": False,
        },
        "claim_boundary": dict(EXACT_CLAIMS),
        "protected_files": {
            "docs359": {"path": str(DOCS359.relative_to(HW)), "sha256": DOCS359_SHA256},
            "ucli": {"path": str(OLD.UCLI.relative_to(HW)), "sha256": OLD.UCLI_SHA256},
        },
    }


def validate_contract(skip_author: bool = False) -> dict[str, Any]:
    contract = strict_json(CONTRACT)
    need(contract == expected_contract(), "M1361 contract exact-set/value drift")
    if not skip_author:
        review = verify_dir(AUTHOR, sha(AUTHOR / "review.json"),
                            sha(AUTHOR / "SHA256SUMS"), sha(AUTHOR / "SHA256SUMS.seal.sha256"))
        need(review.get("status") == "PASS_M1361_EXACT_SOURCE_AUTHOR__FRESH_M1362_BLIND_REQUIRED" and
             review.get("bindings") == {
                 "checker_sha256": sha(CHECKER), "test_sha256": sha(TEST),
                 "contract_sha256": sha(CONTRACT),
                 "m1357_review_sha256": M1357_REVIEW_SHA256,
                 "m1357_manifest_sha256": M1357_MANIFEST_SHA256,
                 "m1357_outer_file_sha256": M1357_OUTER_SHA256} and
             review.get("authorization", {}).get("launch_authorized") is False and
             review.get("claim_boundary") == EXACT_CLAIMS,
             "M1361 author seal binding drift")
    return contract


def validate_common(skip_author: bool = False) -> dict[str, Any]:
    need(sha(M1356_TEST) == M1356_TEST_SHA256 and sha(M1356_CONTRACT) == M1356_CONTRACT_SHA256,
         "M1356 test/contract drift")
    inherited = OLD.validate_common(skip_author=False)
    failed = verify_m1357()
    contract = validate_contract(skip_author=skip_author)
    need(sha(DOCS359) == DOCS359_SHA256 and sha(OLD.UCLI) == OLD.UCLI_SHA256,
         "protected file drift")
    return {
        "m1356_inherited": inherited,
        "m1357_false_negatives_repaired": failed["fresh_hammer"]["false_negatives"],
        "contract_top_level_keys": len(contract),
        "one_shot_keys": len(contract["one_shot"]),
        "resource_keys": len(contract["resource_fail_close"]),
        "receipt_keys": len(contract["receipt_contract"]),
        "claim_keys": len(contract["claim_boundary"]),
        "launch_authorized": contract["authorization"]["launch_authorized"],
    }


def validate_future(mode: str) -> dict[str, Any]:
    need(mode == "source_absent", "author source supports source_absent only")
    need(not os.path.lexists(str(FUTURE_BLIND)), "future M1362 blind authority residue")
    return {"mode": mode, "future_blind_absent": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent",), required=True)
    parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args()
    common = validate_common(skip_author=args.skip_author)
    future = validate_future(args.mode)
    print(json.dumps({
        "schema": "m1361_c2_final_launch_exact_source_check_r1_v1",
        "status": "PASS_M1361_EXACT_SOURCE_ABSENT__FRESH_M1362_BLIND_REQUIRED__NO_EDA",
        "common": common, "future": future,
        "launch_authorized": False, "license_queries": 0, "vcs_runs": 0,
        "simv_runs": 0, "saif_runs": 0, "ptpx_runs": 0, "eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
