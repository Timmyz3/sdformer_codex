#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, no-EDA blind hammer for M1350."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ROOT = HW.parent
CHECKER = HW / "verif_m1350_c2_activity_release/static_check_m1350_c2_activity_vcs_release_source.py"
TEST = HW / "verif_m1350_c2_activity_release/test_m1350_c2_activity_vcs_release_source.py"
CONTRACT = HW / "contracts/m1350_c2_mapped_activity_vcs_release_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1350_c2_mapped_activity_vcs_release_source_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = "/opt/anaconda3/envs/pytorch310/bin/python3.10"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1353_blind_target", CHECKER)
T = load("m1353_bound_m1344_tests", M.M1344_TEST)


def run_test(path: Path, expected: int) -> dict:
    env = dict(os.environ); env["PYTHONDONTWRITEBYTECODE"] = "1"
    run = subprocess.run([PYTHON, "-B", str(path)], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, check=False)
    return {"returncode": run.returncode, "expected": expected,
            "passed": run.returncode == 0 and f"Ran {expected} tests" in run.stdout and
                      "OK" in run.stdout}


def update_expected(fixture, document: str) -> None:
    if document == "source":
        root = fixture.paths["source_hammer"]
        T.seal_dir(root)
        fixture.expected["M1344_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256"] = M.sha(root / "review.json")
        fixture.expected["M1344_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256"] = M.sha(root / "SHA256SUMS")
        fixture.expected["M1344_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256"] = M.sha(root / "SHA256SUMS.seal.sha256")
    elif document == "release":
        path = fixture.paths["launch_release"]
        T.sidecar(path)
        fixture.expected["M1344_EXPECTED_LAUNCH_RELEASE_SHA256"] = M.sha(path)
    else:
        root = fixture.paths["final_hammer"]
        T.seal_dir(root)
        fixture.expected["M1344_EXPECTED_FINAL_HAMMER_REVIEW_SHA256"] = M.sha(root / "review.json")
        fixture.expected["M1344_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256"] = M.sha(root / "SHA256SUMS")
        fixture.expected["M1344_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256"] = M.sha(root / "SHA256SUMS.seal.sha256")


def document_path(fixture, document: str) -> Path:
    if document == "source":
        return fixture.paths["source_hammer"] / "review.json"
    if document == "release":
        return fixture.paths["launch_release"]
    return fixture.paths["final_hammer"] / "review.json"


def json_attack(document: str, mutate: Callable[[str], str]) -> bool:
    fixture = T.RuntimeFixture()
    try:
        path = document_path(fixture, document)
        path.write_text(mutate(path.read_text(encoding="utf-8")), encoding="utf-8")
        update_expected(fixture, document)
        try:
            M.validate_future_strict("runtime_present", fixture.paths, fixture.expected)
            return False
        except Exception:
            return True
    finally:
        fixture.close()


def semantic_json_attack(document: str, mutate: Callable[[dict], None]) -> bool:
    def transform(text: str) -> str:
        value = json.loads(text)
        mutate(value)
        return json.dumps(value, sort_keys=True)
    return json_attack(document, transform)


def success_code_bounds(runner: str) -> tuple[int, int, str]:
    code = M.extract_success_python(runner)
    start = runner.index(code)
    return start, start + len(code), code


def delete_identity(runner: str, receipt: str, key: str) -> str:
    if receipt in ("failure", "attempt"):
        status = "FAILED_OR_INCOMPLETE" if receipt == "failure" else "M1344_ATTEMPT_CONSUMED"
        marker = "printf 'status=" + status
        start = runner.index(marker); end = runner.index("' \\", start)
        segment = runner[start:end]
        token = key + r"=%s\n"
        assert segment.count(token) == 1
        runner = runner[:start] + segment.replace(token, "", 1) + runner[end:]
        return runner + ("\n# inactive %s\n: '%s=%%s'\n"
                         "if false; then printf '%s=%%s\\n' dead; fi\n" %
                         (key, key, key))
    start, end, code = success_code_bounds(runner)
    expression = M.SUCCESS_EXPRESSIONS[key]
    token = (",'%s':%s" if key == M.IDENTITY_KEYS[-1] else "'%s':%s,") % (key, expression)
    assert code.count(token) == 1
    code = code.replace(token, "", 1)
    code = ("dead_string = '%s'\nif False:\n    d = {'identity': {'%s': 'dead'}}\n" %
            (key, key)) + code
    return runner[:start] + code + runner[end:]


def duplicate_identity(runner: str, receipt: str) -> str:
    key = M.IDENTITY_KEYS[0]
    if receipt in ("failure", "attempt"):
        status = "FAILED_OR_INCOMPLETE" if receipt == "failure" else "M1344_ATTEMPT_CONSUMED"
        start = runner.index("printf 'status=" + status); end = runner.index("' \\", start)
        segment = runner[start:end]
        token = key + r"=%s\n"
        return runner[:start] + segment.replace(token, token + token, 1) + runner[end:]
    start, end, code = success_code_bounds(runner)
    token = "'%s':%s," % (key, M.SUCCESS_EXPRESSIONS[key])
    return runner[:start] + code.replace(token, token + token, 1) + runner[end:]


def alias_identity(runner: str, receipt: str) -> str:
    key = M.IDENTITY_KEYS[1]
    if receipt in ("failure", "attempt"):
        status = "FAILED_OR_INCOMPLETE" if receipt == "failure" else "M1344_ATTEMPT_CONSUMED"
        start = runner.index("printf 'status=" + status); end = runner.index("' \\", start)
        segment = runner[start:end]
        return runner[:start] + segment.replace(key + "=%s", key + "_alias=%s", 1) + runner[end:]
    start, end, code = success_code_bounds(runner)
    return runner[:start] + code.replace("'%s':" % key, "'%s_alias':" % key, 1) + runner[end:]


def wrong_expression(runner: str, receipt: str, key: str) -> str:
    if receipt in ("failure", "attempt"):
        status = "FAILED_OR_INCOMPLETE" if receipt == "failure" else "M1344_ATTEMPT_CONSUMED"
        start = runner.index("printf 'status=" + status); end = runner.index(">", runner.index("' \\", start))
        segment = runner[start:end]
        good = M.SHELL_EXPRESSIONS[M.IDENTITY_KEYS.index(key)]
        bad = '"${M1344_EXPECTED_RUNNER_SHA256}"' if key != "runner_sha256" else '"${M1344_EXPECTED_LAUNCH_RELEASE_SHA256}"'
        assert good in segment
        return runner[:start] + segment.replace(good, bad, 1) + runner[end:]
    start, end, code = success_code_bounds(runner)
    good = M.SUCCESS_EXPRESSIONS[key]
    bad = "sha(release)" if key != "launch_release_sha256" else "sha(runner)"
    return runner[:start] + code.replace("'%s':%s" % (key, good),
                                         "'%s':%s" % (key, bad), 1) + runner[end:]


def wrong_branch(runner: str, receipt: str) -> str:
    key = M.IDENTITY_KEYS[2]
    mutant = wrong_expression(runner, receipt, key)
    if receipt == "success":
        start, end, code = success_code_bounds(mutant)
        dead = "if False:\n    d = {'identity': {%r: %s}}\n" % (key, M.SUCCESS_EXPRESSIONS[key])
        return mutant[:start] + dead + code + mutant[end:]
    return mutant + "\nif false; then printf '%s=%%s\\n' \"${M1344_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256}\"; fi\n" % key


def receipt_rejects(mutant: str) -> bool:
    try:
        M.validate_runner_receipts(mutant)
        return False
    except Exception:
        return True


def main() -> int:
    replay = {
        "m1350": run_test(TEST, 36),
        "m1344": run_test(M.M1344_TEST, 12),
        "m1336": run_test(M.M1336_TEST, 10),
        "m1334": run_test(M.M1334_TEST, 12),
    }
    attacks: dict[str, bool] = {}
    canonical = M.RUNNER.read_text(encoding="utf-8")

    for document in ("source", "release", "final"):
        attacks[f"json_{document}_duplicate_status"] = json_attack(
            document, lambda text: text.replace('"status":', '"status":"DUP","status":', 1))
        attacks[f"json_{document}_duplicate_claim"] = json_attack(
            document, lambda text: text.replace('"functional_vcs_verified":',
                                                '"functional_vcs_verified":false,"functional_vcs_verified":', 1))
        attacks[f"json_{document}_nonfinite"] = json_attack(
            document, lambda text: text.replace('"functional_vcs_verified": false',
                                                '"functional_vcs_verified": NaN', 1))
        attacks[f"claim_{document}_extra_false"] = semantic_json_attack(
            document, lambda value: value["claim_boundary"].update({"launch_authorized": False}))
        attacks[f"claim_{document}_extra_true"] = semantic_json_attack(
            document, lambda value: value["claim_boundary"].update({"launch_authorized": True}))
        attacks[f"claim_{document}_missing"] = semantic_json_attack(
            document, lambda value: value["claim_boundary"].pop("headline"))
        attacks[f"claim_{document}_true"] = semantic_json_attack(
            document, lambda value: value["claim_boundary"].update({"performance": True}))
        attacks[f"claim_{document}_alias"] = semantic_json_attack(
            document, lambda value: value["claim_boundary"].update(
                {"system_speed_up": value["claim_boundary"].pop("system_speedup")}))

    for receipt in ("failure", "attempt", "success"):
        for key in M.IDENTITY_KEYS:
            attacks[f"delete_{receipt}_{key}_comment_string_dead"] = receipt_rejects(
                delete_identity(canonical, receipt, key))
            attacks[f"wrong_expression_{receipt}_{key}"] = receipt_rejects(
                wrong_expression(canonical, receipt, key))
        attacks[f"duplicate_{receipt}_identity"] = receipt_rejects(
            duplicate_identity(canonical, receipt))
        attacks[f"alias_{receipt}_identity"] = receipt_rejects(
            alias_identity(canonical, receipt))
        attacks[f"wrong_branch_{receipt}"] = receipt_rejects(
            wrong_branch(canonical, receipt))

    author_manifest = sha(AUTHOR / "SHA256SUMS")
    author_outer = sha(AUTHOR / "SHA256SUMS.seal.sha256")
    source_absent = subprocess.run(
        [PYTHON, "-B", str(CHECKER), "--mode", "source_absent"], cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    checks = {
        "all_replays_pass": all(row["passed"] for row in replay.values()),
        "all_fresh_attacks_rejected": all(attacks.values()),
        "source_absent_self_check_pass": source_absent.returncode == 0,
        "author_manifest_sha256": author_manifest ==
            "58de33d8226317a1367094261dca955c689b4464cfd896701796e3602b981a43",
        "author_outer_file_sha256": author_outer ==
            "451019948ac13265be5d5600b9caec8a3d035fda00eded778a891b0b8880b804",
        "docs359_unchanged": sha(DOCS359) == M.DOCS359_SHA256,
    }
    false_negatives = [name for name, rejected in attacks.items() if not rejected]
    result = {
        "schema": "m1353_m1350_c2_release_source_blind_hammer_r1_v1",
        "verdict": "PASS_SOURCE_BLIND__RELEASE_AUTHOR_MAY_PROCEED" if
                   all(checks.values()) and not false_negatives else
                   "FAIL_DO_NOT_AUTHORIZE_RELEASE",
        "replay": replay,
        "fresh_attacks": {"total": len(attacks), "rejected": sum(attacks.values()),
                          "false_negatives": false_negatives, "results": attacks},
        "checks": checks,
        "target": {"checker_sha256": sha(CHECKER), "test_sha256": sha(TEST),
                   "contract_sha256": sha(CONTRACT),
                   "author_review_sha256": sha(AUTHOR / "review.json"),
                   "author_manifest_sha256": author_manifest,
                   "author_outer_file_sha256": author_outer},
        "execution": {"license_queries": 0, "launches": 0, "vcs": 0,
                      "simv": 0, "saif": 0, "eda": 0},
        "claim_boundary": dict(M.EXACT_CLAIM_BOUNDARY),
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
