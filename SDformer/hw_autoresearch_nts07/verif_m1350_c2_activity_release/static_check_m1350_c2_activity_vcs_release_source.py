#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1350 additive, no-EDA source gate for the sealed M1344 C2 runner."""
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
M1344_CHECKER = HW / "verif_m1344_c2_activity_release/static_check_m1344_c2_activity_vcs_release_source.py"
M1344_CHECKER_SHA256 = "fc3c89040ec4ec3ecb9b8fcb10e8734df918f81615e2c8e6c52108445e56f3bb"
M1344_TEST = HW / "verif_m1344_c2_activity_release/test_m1344_c2_activity_vcs_release_source.py"
M1344_TEST_SHA256 = "32d5499a2956da6cccf139db4850cf35a4e901086fee26331bfbad5df8ecef43"
M1336_TEST = HW / "verif_m1336_c2_activity_release/test_m1336_c2_activity_vcs_release_source.py"
M1334_TEST = HW / "system_simulator/tests/test_m1334_c2_headline_mapped_production_activity_source.py"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1344_c2_headline_mapped_production_activity_one_shot_exact_sha.sh"
RUNNER_SHA256 = "4d6081a094d4a865c23c42c9b2b0fc2644ee9d7f65ae401f838663d550daa4f0"
M1344_CONTRACT = HW / "contracts/m1344_c2_headline_mapped_production_activity_vcs_release_source_contract_r1_20260831.json"
M1344_CONTRACT_SHA256 = "0fb605913b9d779bf493811d3d6498ed466254d40aa4847493a6150d2bc8af1b"
M1348 = HW / "reviews/m1348_m1344_c2_mapped_activity_vcs_release_source_blind_hammer_r1_20260831"
M1348_REVIEW_SHA256 = "ca28112f97b2a0cd47bedb40d707f83c5e96dcbf05941f5cf5421f96516c7815"
M1348_MANIFEST_SHA256 = "2a1569a9485296631244052cb5f8455c18e2b2d583b6176b852592c7edfd2eab"
M1348_OUTER_SHA256 = "a836fa9441dc9c38d8eda8e20eb9f5fbd686d6709e632ee5704272e1c702b538"
CHECKER = Path(__file__).resolve()
TEST = HW / "verif_m1350_c2_activity_release/test_m1350_c2_activity_vcs_release_source.py"
CONTRACT = HW / "contracts/m1350_c2_mapped_activity_vcs_release_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1350_c2_mapped_activity_vcs_release_source_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CLAIMS = ("functional_vcs_verified", "production_saif", "ptpx", "power",
          "energy", "performance", "system_speedup", "paper_ppa_ready", "headline")
EXACT_CLAIM_BOUNDARY = {key: False for key in CLAIMS}
IDENTITY_KEYS = (
    "runner_sha256", "source_contract_sha256",
    "source_hammer_review_sha256", "source_hammer_manifest_sha256",
    "source_hammer_outer_file_sha256", "launch_release_sha256",
    "final_hammer_review_sha256", "final_hammer_manifest_sha256",
    "final_hammer_outer_file_sha256",
)

SHELL_EXPRESSIONS = (
    '"$(sha "${RUNNER}")"',
    '"$(sha "${SOURCE_CONTRACT}")"',
    '"${M1344_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256}"',
    '"${M1344_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256}"',
    '"${M1344_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256}"',
    '"${M1344_EXPECTED_LAUNCH_RELEASE_SHA256}"',
    '"${M1344_EXPECTED_FINAL_HAMMER_REVIEW_SHA256}"',
    '"${M1344_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256}"',
    '"${M1344_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256}"',
)
SUCCESS_EXPRESSIONS = {
    "runner_sha256": "sha(runner)",
    "source_contract_sha256": "sha(contract)",
    "source_hammer_review_sha256": "sha(source_hammer/'review.json')",
    "source_hammer_manifest_sha256": "sha(source_hammer/'SHA256SUMS')",
    "source_hammer_outer_file_sha256": "sha(source_hammer/'SHA256SUMS.seal.sha256')",
    "launch_release_sha256": "sha(release)",
    "final_hammer_review_sha256": "sha(final_hammer/'review.json')",
    "final_hammer_manifest_sha256": "sha(final_hammer/'SHA256SUMS')",
    "final_hammer_outer_file_sha256": "sha(final_hammer/'SHA256SUMS.seal.sha256')",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def load_m1344():
    need(sha(M1344_CHECKER) == M1344_CHECKER_SHA256, "sealed M1344 checker drift")
    spec = importlib.util.spec_from_file_location("m1350_sealed_m1344_checker", M1344_CHECKER)
    need(spec is not None and spec.loader is not None, "cannot import M1344 checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = load_m1344()


def strict_json_text(text: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(
        text, object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError("nonfinite JSON token: " + token)))
    need(type(value) is dict, "JSON root must be object")
    return value


def strict_json(path: Path) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), "JSON authority absent/nonregular")
    return strict_json_text(path.read_text(encoding="utf-8"))


def verify_dir(root: Path, review_sha: str, manifest_sha: str, outer_sha: str) -> None:
    need(root.is_dir() and not root.is_symlink(), "sealed directory invalid")
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    need(sha(root / "review.json") == review_sha and sha(manifest) == manifest_sha and
         sha(outer) == outer_sha, "sealed authority exact SHA drift")
    need(outer.read_text().split() == [manifest_sha, "SHA256SUMS"], "outer seal drift")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        rel = Path(name)
        need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
             not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "manifest row invalid")
        member = root / rel
        need(member.is_file() and not member.is_symlink() and sha(member) == digest,
             "manifest member drift: " + name)
        listed[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == set(listed), "sealed directory population drift")


def exact_claim_boundary(document: dict[str, Any], label: str) -> None:
    need(document.get("claim_boundary") == EXACT_CLAIM_BOUNDARY,
         label + " claim_boundary must be exact nine-key all-false object")


def validate_future_strict(mode: str, paths: dict[str, Path] | None = None,
                           expected: dict[str, str] | None = None) -> dict[str, Any]:
    paths = OLD.future_paths() if paths is None else paths
    if mode == "source_absent":
        return OLD.validate_future(mode, paths, expected)
    need(mode == "runtime_present", "unknown strict future mode")
    source = strict_json(paths["source_hammer"] / "review.json")
    release = strict_json(paths["launch_release"])
    final = strict_json(paths["final_hammer"] / "review.json")
    exact_claim_boundary(source, "source hammer")
    exact_claim_boundary(release, "launch release")
    exact_claim_boundary(final, "final hammer")
    result = OLD.validate_future(mode, paths, expected)
    return dict(result, strict_json=True, exact_claim_boundaries=True)


def normalize_shell(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\\\s*\n", " ", text)).strip()


def extract_shell_receipt(runner: str, status: str, target: str) -> tuple[str, str]:
    marker = "printf 'status=" + status
    need(runner.count(marker) == 1, status + " active printf cardinality drift")
    start = runner.index(marker)
    format_start = start + len("printf '")
    format_end = runner.find("' \\", format_start)
    need(format_end >= 0, status + " format terminator absent")
    target_marker = '>' + target
    need(runner.count(target_marker) == 1, status + " target cardinality drift")
    target_at = runner.index(target_marker, format_end)
    arguments = runner[format_end + 3:target_at]
    return runner[format_start:format_end], arguments


def validate_shell_receipt(runner: str, status: str, target: str,
                           prefix_arguments: tuple[str, ...]) -> dict[str, Any]:
    fmt, arguments = extract_shell_receipt(runner, status, target)
    lines = fmt.split(r"\n")
    active_fields = [line[:-3] for line in lines if line.endswith("=%s")]
    for key in IDENTITY_KEYS:
        need(active_fields.count(key) == 1, status + " identity field count drift: " + key)
    need([key for key in active_fields if key in IDENTITY_KEYS] == list(IDENTITY_KEYS),
         status + " identity order/set drift")
    expected_arguments = prefix_arguments + SHELL_EXPRESSIONS
    need(normalize_shell(arguments) == " ".join(expected_arguments),
         status + " active value-expression list drift")
    return {"status": status, "identity_keys": list(IDENTITY_KEYS),
            "active_value_expressions": len(expected_arguments)}


def ast_dict(node: ast.AST, label: str) -> dict[str, ast.AST]:
    need(isinstance(node, ast.Dict), label + " must be an active dict literal")
    result = {}
    for key, value in zip(node.keys, node.values):
        need(isinstance(key, ast.Constant) and isinstance(key.value, str),
             label + " has nonliteral key")
        need(key.value not in result, label + " has duplicate active key: " + key.value)
        result[key.value] = value
    return result


def extract_success_python(runner: str) -> str:
    marker = '"${WORK}/candidate/m1344_receipt.json" <<\'PY\''
    need(runner.count(marker) == 1, "success receipt heredoc cardinality drift")
    marker_at = runner.index(marker)
    line_start = runner.rfind("\n", 0, marker_at) + 1
    command_start = runner.rfind('\n"${PYTHON}" -I - "${RUNNER}"', 0, line_start + 1)
    need(command_start >= 0 and runner[command_start + 1] != "#",
         "success receipt active command absent")
    code_start = runner.index("\n", marker_at) + 1
    code_end = runner.find("\nPY\n", code_start)
    need(code_end >= 0, "success receipt heredoc terminator absent")
    return runner[code_start:code_end]


def validate_success_receipt(runner: str) -> dict[str, Any]:
    code = extract_success_python(runner)
    tree = ast.parse(code)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and
                   len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and
                   node.targets[0].id == "d"]
    need(len(assignments) == 1, "success active receipt assignment cardinality drift")
    receipt = ast_dict(assignments[0].value, "success receipt")
    identity = ast_dict(receipt.get("identity"), "success identity")
    need(set(identity) == set(IDENTITY_KEYS) and list(identity) == list(IDENTITY_KEYS),
         "success identity exact key/order drift")
    for key, expression in SUCCESS_EXPRESSIONS.items():
        expected = ast.parse(expression, mode="eval").body
        need(ast.dump(identity[key], include_attributes=False) ==
             ast.dump(expected, include_attributes=False),
             "success active value expression drift: " + key)
    writes = [node for node in tree.body if isinstance(node, ast.Expr) and
              isinstance(node.value, ast.Call) and
              isinstance(node.value.func, ast.Attribute) and
              isinstance(node.value.func.value, ast.Name) and
              node.value.func.value.id == "out" and node.value.func.attr == "write_text"]
    need(len(writes) == 1, "success active output writer cardinality drift")
    return {"status": "success", "identity_keys": list(identity),
            "active_value_expressions": len(identity)}


def validate_runner_receipts(runner: str) -> dict[str, Any]:
    failure = validate_shell_receipt(
        runner, "FAILED_OR_INCOMPLETE",
        '"${FAILURE_STAGE}/RUN_FAILED_OR_INCOMPLETE.txt"',
        ('"${phase}"', '"${rc}"', '"${compile_count}"', '"${sim_count}"'))
    attempt = validate_shell_receipt(
        runner, "M1344_ATTEMPT_CONSUMED", '"${ATTEMPT_STAGE}/attempt.txt"', tuple())
    success = validate_success_receipt(runner)
    need(failure["identity_keys"] == attempt["identity_keys"] == success["identity_keys"] ==
         list(IDENTITY_KEYS), "three receipt identity sets differ")
    return {"failure": failure, "attempt": attempt, "success": success,
            "receipts": 3, "identities_per_receipt": 9}


def validate_common(skip_author: bool = False) -> dict[str, Any]:
    need(sha(RUNNER) == RUNNER_SHA256 and sha(M1344_CONTRACT) == M1344_CONTRACT_SHA256 and
         sha(M1344_TEST) == M1344_TEST_SHA256, "sealed M1344 source family drift")
    verify_dir(M1348, M1348_REVIEW_SHA256, M1348_MANIFEST_SHA256, M1348_OUTER_SHA256)
    m1348 = strict_json(M1348 / "review.json")
    need(m1348["status"] == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED" and
         m1348["false_negative_count"] == 7, "M1348 failure authority drift")
    inherited = OLD.validate_common(skip_author=False)
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, check=False)
    need(syntax.returncode == 0, "sealed runner bash syntax failed")
    receipts = validate_runner_receipts(RUNNER.read_text(encoding="utf-8"))
    contract = strict_json(CONTRACT)
    need(contract["schema"] == "m1350_c2_mapped_activity_vcs_release_source_contract_r1_v1" and
         contract["status"] == "M1350_SOURCE_READY__FRESH_DIFFERENT_AUTHOR_BLIND_REQUIRED",
         "M1350 contract schema/status drift")
    need(contract["identity"] == {
        "runner_sha256": RUNNER_SHA256,
        "checker": str(CHECKER.relative_to(HW)), "checker_sha256": sha(CHECKER),
        "test": str(TEST.relative_to(HW)), "test_sha256": sha(TEST)},
        "M1350 source identity drift")
    exact_claim_boundary(contract, "source contract")
    need(contract["failed_predecessor"] == {
        "review_sha256": M1348_REVIEW_SHA256,
        "manifest_sha256": M1348_MANIFEST_SHA256,
        "outer_file_sha256": M1348_OUTER_SHA256,
        "false_negative_count": 7}, "M1348 binding drift")
    if not skip_author:
        OLD.OLD.verify_dir(AUTHOR)
        author = strict_json(AUTHOR / "review.json")
        need(author["status"] == "PASS_M1350_SOURCE_AUTHOR__DIFFERENT_AUTHOR_BLIND_REQUIRED" and
             author["bindings"] == {"checker_sha256": sha(CHECKER),
                                      "test_sha256": sha(TEST),
                                      "contract_sha256": sha(CONTRACT)},
             "M1350 author binding drift")
        exact_claim_boundary(author, "source author")
    return {"inherited_checks": inherited, "receipt_parser": receipts,
            "m1348_bound": True, "strict_json": True, "claim_boundary_exact": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_absent",), required=True)
    parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args()
    common = validate_common(skip_author=args.skip_author)
    future = validate_future_strict(args.mode)
    print(json.dumps({
        "schema": "m1350_c2_activity_vcs_release_source_check_r1_v1",
        "status": "PASS_M1350_SOURCE_ABSENT__NO_EDA__BLIND_HAMMER_REQUIRED",
        "common": common, "future": future,
        "license_queries": 0, "vcs_runs": 0, "simv_runs": 0, "saif_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
