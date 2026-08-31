#!/usr/bin/env python3
"""Different-author, source-only blind hammer for sealed M1325."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                 "capture_m1325_motion_ep34_runtime_projection_successor_r1.py")
TEST = HW / "tests/test_m1325_motion_ep34_runtime_projection_successor.py"
CONTRACT = HW / ("contracts/m1325_motion_ep34_runtime_projection_successor_"
                 "source_contract_r1_20260831.json")
AUTHOR = HW / ("reviews/m1325_motion_ep34_runtime_projection_successor_"
               "source_author_r1_20260831")
EXPECTED = {
    SOURCE: "d3aba86b9003f1ee3cba2b1f81ff02ab8b43e7f5ca7bd56a18ba1c265ab76000",
    TEST: "c670f385eb16542acccb876ca61090b468859c3ceb6d8e9a6edc01f01a49d262",
    CONTRACT: "b2460c1aced88961d1b8418b7a7de326b2056228c44f9e2228acb2ca38f7a3b2",
    AUTHOR / "SHA256SUMS":
        "9b5ca9f00042f48b252d2718deaa75bb6373f3aeb0dc2ffd334188640f8db8a7",
    AUTHOR / "SHA256SUMS.seal.sha256":
        "754ffddf5bb768082d6b5a04b4f9512b5d03a914db564d9a7c600e829e6e313c",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def function(tree, name):
    rows = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(rows) == 1, "function count drift: " + name)
    return rows[0]


def call_names(node):
    output = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        try:
            output.append(ast.unparse(item.func))
        except Exception:
            pass
    return output


def main():
    checks = []
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), "identity member missing/symlink")
        require(sha(path) == expected, "identity SHA drift: " + str(path))
    checks.append("exact_source_test_contract_author_graph")

    manifest = AUTHOR / "SHA256SUMS"
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        member = AUTHOR / name.lstrip("*")
        require(sha(member) == digest, "author member drift")
    require((AUTHOR / "SHA256SUMS.seal.sha256").read_text().split() ==
            [sha(manifest), "SHA256SUMS"], "author outer seal drift")
    checks.append("author_recursive_seal")

    completed = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "unittest", "-q", str(TEST)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(completed.returncode == 0 and "OK" in completed.stdout, "author regression failed")
    checks.append("author_10_of_10")

    spec = importlib.util.spec_from_file_location("m1326_blind_m1325", SOURCE)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(module.build_runtime_contract(module.strict_json(module.M1313_CONTRACT))["capture"] ==
            {"attention_windows_per_call": 100}, "capture100 drift")
    checks.append("capture100_exact")

    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    identity_calls = call_names(function(tree, "validate_identity_and_project"))
    require("M1319.validate_exact_m1313_m1314" in identity_calls,
            "M1325 no longer calls exact M1319 validator")
    require(identity_calls.index("M1319.validate_exact_m1313_m1314") <
            identity_calls.index("build_runtime_contract"),
            "runtime built before exact identity validation")
    m1319_tree = ast.parse(Path(module.M1319.__file__).read_text())
    require("M1249.validate_production_launch" in
            call_names(function(m1319_tree, "validate_exact_m1313_m1314")),
            "M1319 launch validator call drift")
    m1249_tree = ast.parse(Path(module.M1319.M1249.__file__).read_text())
    require("ensure_fresh_namespaces" in
            call_names(function(m1249_tree, "validate_production_launch")),
            "M1249 freshness call drift")
    fresh = function(m1249_tree, "ensure_fresh_namespaces")
    fresh_text = ast.unparse(fresh)
    require(all(name in fresh_text for name in
                ("CANONICAL_RESULT", "CANONICAL_ATTEMPT", "CANONICAL_LOG")),
            "old three-namespace freshness gate drift")
    checks.append("static_old_namespace_freshness_call_chain")

    forensic = module.verify_m1324_forensic()
    require(forensic["failed_execution_evidence"]["attempt_token"] ==
            "M1249_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE", "old attempt evidence drift")
    require(forensic["authorization"]["old_M1249_attempt_reuse"] is False,
            "forensic unexpectedly permits old attempt reuse")
    checks.append("sealed_old_attempt_consumed_and_forbidden")

    observed = {}
    def fail_before_projection(*_args, **_kwargs):
        observed["old_attempt"] = str(module.M1319.M1249.CANONICAL_ATTEMPT)
        observed["new_attempt"] = str(module.CANONICAL_ATTEMPT)
        raise module.M1319.M1319Error("BLIND_OLD_M1249_FRESHNESS_GATE")
    try:
        with mock.patch.object(module.M1319, "validate_exact_m1313_m1314",
                               side_effect=fail_before_projection):
            module.validate_identity_and_project()
    except module.M1319.M1319Error as error:
        require(str(error) == "BLIND_OLD_M1249_FRESHNESS_GATE", "dynamic gate drift")
    else:
        raise AssertionError("identity projection unexpectedly bypassed gate")
    require(observed["old_attempt"] != observed["new_attempt"] and
            "m1249" in observed["old_attempt"] and "m1325" in observed["new_attempt"],
            "old/new attempt namespace observation drift")
    checks.append("dynamic_projection_calls_validator_before_new_namespace")

    source_text = SOURCE.read_text()
    require("--source-self-check" in source_text and "--run" not in source_text and
            "consume_attempt()" not in source_text, "source-only CLI boundary drift")
    checks.append("cli_inert_no_attempt_consumer")

    print(json.dumps({
        "schema": "m1326_m1325_runtime_projection_blind_hammer_output_r1_v1",
        "status": "FAIL_DO_NOT_CITE__OLD_M1249_FRESHNESS_GATE_STILL_LIVE",
        "checks_passed": checks,
        "checks": len(checks),
        "p0": {
            "call_chain": "M1325.validate_identity_and_project -> M1319.validate_exact_m1313_m1314 -> M1249.validate_production_launch -> ensure_fresh_namespaces",
            "old_attempt_consumed": True,
            "old_attempt_reuse_authorized": False,
            "failure_point": "before build_runtime_contract",
        },
        "authorization": {"production_release": False, "remote": False, "gpu": False},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
