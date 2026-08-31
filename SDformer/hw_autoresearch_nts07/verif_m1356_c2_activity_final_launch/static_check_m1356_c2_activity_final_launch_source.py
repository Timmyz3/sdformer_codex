#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only final-launch authority gate for the mapped C2 VCS/SAIF runner."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


HW = Path(__file__).resolve().parents[1]
ROOT = HW.parent
M1350_CHECKER = HW / "verif_m1350_c2_activity_release/static_check_m1350_c2_activity_vcs_release_source.py"
M1350_CHECKER_SHA256 = "d904cd74b716fa3277dd4067dace0b33e66f5abf31bb2b947edf2bfd97ad5d34"
M1350_TEST = HW / "verif_m1350_c2_activity_release/test_m1350_c2_activity_vcs_release_source.py"
M1350_TEST_SHA256 = "a293be8afe42880970722d35b646d3b303d8ebfe89af08a9cb0f4d411f640096"
M1350_CONTRACT = HW / "contracts/m1350_c2_mapped_activity_vcs_release_source_contract_r1_20260831.json"
M1350_CONTRACT_SHA256 = "47cb426b2283187b7648e44b90c12d03b59eabed4ad9ed4e573e22d27fbc43b9"
M1353 = HW / "reviews/m1353_m1350_c2_mapped_activity_vcs_release_source_blind_hammer_r1_20260831"
M1353_REVIEW_SHA256 = "97c54ad6a3dfd4c4f731617cd8d6151dc54d5658607711ad0fc3fad81a8e23e2"
M1353_MANIFEST_SHA256 = "5d63ac5e550d47512b06c14bef73aff2a1fbc91b997789d3327591aefbba1bfe"
M1353_OUTER_SHA256 = "de14855dad968d2f6c3267536ce51b81f83c74f3f9017067482e1427a71710ca"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1344_c2_headline_mapped_production_activity_one_shot_exact_sha.sh"
RUNNER_SHA256 = "4d6081a094d4a865c23c42c9b2b0fc2644ee9d7f65ae401f838663d550daa4f0"
UCLI = HW / "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"
UCLI_SHA256 = "c90153dfd58ff4e653852a54b31ad3b19cb8fabd993e15c21d9071b555cbebc1"
CHECKER = Path(__file__).resolve()
TEST = HW / "verif_m1356_c2_activity_final_launch/test_m1356_c2_activity_final_launch_source.py"
CONTRACT = HW / "contracts/m1356_c2_mapped_activity_vcs_saif_final_launch_authority_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1356_c2_mapped_activity_vcs_saif_final_launch_authority_source_author_r1_20260831"
FUTURE_BLIND = HW / "reviews/m1357_m1356_c2_mapped_activity_vcs_saif_final_launch_authority_blind_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CLAIMS = ("functional_vcs_verified", "production_saif", "ptpx", "power",
          "energy", "performance", "system_speedup", "paper_ppa_ready", "headline")
EXACT_CLAIMS = {key: False for key in CLAIMS}
ATTEMPT = HW / "results/.m1344_c2_headline_mapped_production_activity_vcs_attempt_consumed"
RESULT = HW / "results/m1344_c2_headline_mapped_production_activity_vcs_r1_20260831"


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
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AssertionError("nonfinite JSON token: " + token)))
    need(type(value) is dict, "JSON root must be object")
    return value


def load_m1350():
    need(sha(M1350_CHECKER) == M1350_CHECKER_SHA256, "M1350 checker drift")
    spec = importlib.util.spec_from_file_location("m1356_sealed_m1350", M1350_CHECKER)
    need(spec is not None and spec.loader is not None, "cannot load M1350 checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_m1350()


def verify_m1353() -> dict[str, Any]:
    review, manifest, outer = M1353 / "review.json", M1353 / "SHA256SUMS", M1353 / "SHA256SUMS.seal.sha256"
    need(sha(review) == M1353_REVIEW_SHA256 and sha(manifest) == M1353_MANIFEST_SHA256 and
         sha(outer) == M1353_OUTER_SHA256, "M1353 exact seal drift")
    need(outer.read_text(encoding="utf-8").split() == [M1353_MANIFEST_SHA256, "SHA256SUMS"],
         "M1353 outer seal content drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and name not in listed,
             "M1353 manifest row invalid")
        path = (M1353 / name).resolve()
        need(path == HW or HW in path.parents, "M1353 manifest escapes hardware root")
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "M1353 manifest member drift")
        listed.add(name)
    review_json = strict_json(review)
    need(review_json.get("status") == "PASS_DIFFERENT_AUTHOR_SOURCE_BLIND" and
         review_json.get("verdict") == "GO_M1350_RELEASE_AUTHORING__NO_LAUNCH_AUTHORIZATION" and
         review_json.get("fresh_hammer", {}).get("false_negatives") == 0 and
         review_json.get("authorization", {}).get("release_authoring") is True and
         review_json.get("authorization", {}).get("launch") is False and
         review_json.get("claim_boundary") == EXACT_CLAIMS,
         "M1353 verdict/claim boundary drift")
    return review_json


def ast_dict(node: ast.AST, label: str) -> dict[str, ast.AST]:
    need(isinstance(node, ast.Dict), label + " must be dict literal")
    result = {}
    for key, value in zip(node.keys, node.values):
        need(isinstance(key, ast.Constant) and type(key.value) is str and key.value not in result,
             label + " key invalid/duplicate")
        result[key.value] = value
    return result


def audit_receipts(runner: str) -> dict[str, Any]:
    parsed = M.validate_runner_receipts(runner)
    failure_fmt, _ = M.extract_shell_receipt(
        runner, "FAILED_OR_INCOMPLETE", '"${FAILURE_STAGE}/RUN_FAILED_OR_INCOMPLETE.txt"')
    attempt_fmt, _ = M.extract_shell_receipt(
        runner, "M1344_ATTEMPT_CONSUMED", '"${ATTEMPT_STAGE}/attempt.txt"')
    failure_fields = [line[:-3] for line in failure_fmt.split(r"\n") if line.endswith("=%s")]
    attempt_fields = [line[:-3] for line in attempt_fmt.split(r"\n") if line.endswith("=%s")]
    need(failure_fields == ["phase", "return_code", "compile_count", "sim_count"] +
         list(M.IDENTITY_KEYS), "failure receipt exact active fields drift")
    need(attempt_fields == list(M.IDENTITY_KEYS), "attempt receipt exact active fields drift")
    need("automatic_retry=false" in failure_fmt and "canonical_result=false" in failure_fmt and
         "raw_private_build_citable=false" in failure_fmt,
         "failure fail-closed claims drift")
    need("automatic_retry=false" in attempt_fmt and "maximum_vcs_compiles=2" in attempt_fmt and
         "maximum_simv_runs=10" in attempt_fmt, "attempt one-shot claims drift")
    code = M.extract_success_python(runner); tree = ast.parse(code)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and
                   len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and
                   node.targets[0].id == "d"]
    need(len(assignments) == 1, "success receipt assignment drift")
    receipt = ast_dict(assignments[0].value, "success receipt")
    boundary = ast_dict(receipt.get("claim_boundary"), "success claim boundary")
    need(list(boundary) == list(CLAIMS) and all(isinstance(boundary[key], ast.Constant) and
         boundary[key].value is False for key in CLAIMS), "success exact claims drift")
    one_shot = ast_dict(receipt.get("one_shot"), "success one-shot")
    expected_one_shot = {"attempt_consumed": True, "vcs_compiles": 2,
                         "simv_runs": 10, "automatic_retry": False}
    need(set(one_shot) == set(expected_one_shot) and all(
         isinstance(one_shot[key], ast.Constant) and one_shot[key].value == value
         for key, value in expected_one_shot.items()), "success one-shot drift")
    return {"parsed": parsed, "failure_fields": len(failure_fields),
            "attempt_fields": len(attempt_fields), "success_claims": len(boundary)}


def audit_collision_and_namespaces(runner: str) -> dict[str, Any]:
    blocked = "blocked={'vcs','vcs1','vlogan','simv','dc_shell','dc_shell-t','pt_shell','fm_shell','icc2_shell','common_shell_exec','common_shell_exe'}"
    need(runner.count(blocked) == 1 and "same-UID EDA collision" in runner,
         "same-UID EDA collision set drift")
    first = 'phase="RESOURCE_PREFLIGHT"\ncollision_gate\nresource_gate\nphase="LICENSE_PREFLIGHT"'
    second = '|| fail "license preflight failed"\ncollision_gate\n\nphase="ATTEMPT_CONSUME"'
    need(first in runner and second in runner, "collision/resource gates do not dominate attempt")
    need(runner.count('publish_no_replace "${ATTEMPT_STAGE}" "${ATTEMPT}"') == 1,
         "attempt publication cardinality drift")
    namespaces = M.OLD.namespaces()
    need(namespaces[0] == ATTEMPT and namespaces[1] == RESULT and len(namespaces) == 5 and
         len(set(namespaces)) == 5, "one-shot namespace identity drift")
    need(all(not os.path.lexists(str(path)) for path in namespaces),
         "one-shot namespace already consumed/resident")
    return {"blocked_tools": 11, "namespace_count": len(namespaces),
            "attempt_fresh": True, "collision_before_attempt": True}


def validate_contract(skip_author: bool = False) -> dict[str, Any]:
    contract = strict_json(CONTRACT)
    need(contract.get("schema") == "m1356_c2_mapped_activity_vcs_saif_final_launch_authority_source_r1_v1" and
         contract.get("status") == "SOURCE_ONLY__M1353_BOUND__FRESH_M1357_BLIND_REQUIRED",
         "M1356 contract schema/status drift")
    need(contract.get("identity") == {
        "runner": str(RUNNER.relative_to(HW)), "runner_sha256": RUNNER_SHA256,
        "checker": str(CHECKER.relative_to(HW)), "checker_sha256": sha(CHECKER),
        "test": str(TEST.relative_to(HW)), "test_sha256": sha(TEST),
        "m1350_checker_sha256": M1350_CHECKER_SHA256,
        "m1350_contract_sha256": M1350_CONTRACT_SHA256,
        "m1353_review_sha256": M1353_REVIEW_SHA256,
        "m1353_manifest_sha256": M1353_MANIFEST_SHA256,
        "m1353_outer_file_sha256": M1353_OUTER_SHA256,
        "ucli_sha256": UCLI_SHA256}, "M1356 identity binding drift")
    need(contract.get("claim_boundary") == EXACT_CLAIMS and
         contract.get("authorization", {}).get("launch_authorized") is False and
         contract.get("authorization", {}).get("different_author_blind_hammer") is True and
         contract.get("authorization", {}).get("license_query") is False and
         contract.get("authorization", {}).get("vcs") is False and
         contract.get("authorization", {}).get("simv") is False and
         contract.get("authorization", {}).get("saif") is False and
         contract.get("authorization", {}).get("eda") is False,
         "author-stage authorization/claims lifted")
    need(contract.get("future_blind", {}).get("path") == str(FUTURE_BLIND.relative_to(HW)) and
         contract.get("future_blind", {}).get("must_be_absent_during_authoring") is True,
         "future blind namespace drift")
    if not skip_author:
        M.verify_dir(AUTHOR, sha(AUTHOR / "review.json"),
                     sha(AUTHOR / "SHA256SUMS"),
                     sha(AUTHOR / "SHA256SUMS.seal.sha256"))
        author = strict_json(AUTHOR / "review.json")
        need(author.get("status") == "PASS_M1356_SOURCE_AUTHOR__FRESH_M1357_BLIND_REQUIRED" and
             author.get("bindings") == {"checker_sha256": sha(CHECKER),
                                         "test_sha256": sha(TEST),
                                         "contract_sha256": sha(CONTRACT)} and
             author.get("authorization", {}).get("launch_authorized") is False and
             author.get("claim_boundary") == EXACT_CLAIMS,
             "M1356 author seal binding drift")
    return contract


def validate_common(skip_author: bool = False) -> dict[str, Any]:
    need(sha(M1350_TEST) == M1350_TEST_SHA256 and sha(M1350_CONTRACT) == M1350_CONTRACT_SHA256 and
         sha(RUNNER) == RUNNER_SHA256 and sha(UCLI) == UCLI_SHA256 and
         sha(DOCS359) == DOCS359_SHA256, "sealed source identity drift")
    inherited = M.validate_common(skip_author=False)
    m1353 = verify_m1353()
    runner = RUNNER.read_text(encoding="utf-8")
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, check=False)
    need(syntax.returncode == 0, "runner bash syntax failed")
    receipts = audit_receipts(runner)
    resources = audit_collision_and_namespaces(runner)
    contract = validate_contract(skip_author=skip_author)
    return {"m1350_inherited": inherited, "m1353_false_negatives":
            m1353["fresh_hammer"]["false_negatives"], "receipts": receipts,
            "resources": resources, "launch_authorized":
            contract["authorization"]["launch_authorized"]}


def validate_future(mode: str) -> dict[str, Any]:
    need(mode == "source_absent", "author source supports source_absent only")
    need(not os.path.lexists(str(FUTURE_BLIND)), "future M1357 blind authority residue")
    return {"mode": mode, "future_blind_absent": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent",), required=True)
    parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args()
    common = validate_common(skip_author=args.skip_author)
    future = validate_future(args.mode)
    print(json.dumps({
        "schema": "m1356_c2_final_launch_authority_source_check_r1_v1",
        "status": "PASS_M1356_SOURCE_ABSENT__FRESH_M1357_BLIND_REQUIRED__NO_EDA",
        "common": common, "future": future, "launch_authorized": False,
        "license_queries": 0, "vcs_runs": 0, "simv_runs": 0,
        "saif_runs": 0, "eda_runs": 0, "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
