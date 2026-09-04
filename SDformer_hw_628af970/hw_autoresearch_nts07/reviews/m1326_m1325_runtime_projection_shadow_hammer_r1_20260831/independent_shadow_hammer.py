#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author shadow hammer for sealed source-only M1325."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                 "capture_m1325_motion_ep34_runtime_projection_successor_r1.py")
TEST = HW / "tests/test_m1325_motion_ep34_runtime_projection_successor.py"
CONTRACT = HW / "contracts/m1325_motion_ep34_runtime_projection_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1325_motion_ep34_runtime_projection_successor_source_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

EXPECTED = {
    "source": "d3aba86b9003f1ee3cba2b1f81ff02ab8b43e7f5ca7bd56a18ba1c265ab76000",
    "test": "c670f385eb16542acccb876ca61090b468859c3ceb6d8e9a6edc01f01a49d262",
    "contract": "b2460c1aced88961d1b8418b7a7de326b2056228c44f9e2228acb2ca38f7a3b2",
    "author_receipt": "1408d0f625ed37861575ea63f874d46c5dc176db7c61a99caee5278890f244ec",
    "author_manifest": "9b5ca9f00042f48b252d2718deaa75bb6373f3aeb0dc2ffd334188640f8db8a7",
    "author_outer": "754ffddf5bb768082d6b5a04b4f9512b5d03a914db564d9a7c600e829e6e313c",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(root.is_dir() and not root.is_symlink(), "author root missing/symlink")
    require(sha256(manifest) == manifest_sha and sha256(outer) == outer_sha,
            "author seal SHA drift")
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n",
            "author outer content drift")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = PurePosixPath(relative)
        require(path.parts and not path.is_absolute() and ".." not in path.parts and
                relative not in rows, "unsafe/duplicate manifest member")
        rows[relative] = digest
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and path.name not in {
                        "SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sorted(rows) == actual, "author member population drift")
    for relative, digest in rows.items():
        member = root / relative
        require(member.is_file() and not member.is_symlink() and sha256(member) == digest,
                "author member drift")
    return rows


def load_source():
    spec = importlib.util.spec_from_file_location("m1326_sealed_m1325", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1325")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def function_ast(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(found) == 1, name + " function count mismatch")
    return found[0]


def call_chains(function: ast.FunctionDef) -> set[str]:
    output = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        cursor = node.func
        parts = []
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            parts.append(cursor.id)
            output.add(".".join(reversed(parts)))
    return output


def main() -> int:
    for label, path in (("source", SOURCE), ("test", TEST), ("contract", CONTRACT),
                        ("docs359", DOCS359)):
        require(path.is_file() and not path.is_symlink() and sha256(path) == EXPECTED[label],
                label + " identity drift")
    author_rows = verify_seal(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    require(author_rows.get("receipt.json") == EXPECTED["author_receipt"],
            "author receipt member mismatch")

    baseline = subprocess.run([str(PYTHON), "-I", str(TEST)], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(baseline.returncode == 0 and "Ran 10 tests" in baseline.stdout and
            baseline.stdout.rstrip().endswith("OK"), "author tests not 10/10 PASS")
    module = load_source()
    passed = []

    require(module.frozen_m1227_direct_contract_keys() == {
        "contract_path", "capture", "cohort", "output"},
        "frozen direct key audit mismatch")
    nested = ast.parse("contract['capture']['attention_windows_per_call']").body[0].value
    require(module._chain(nested) == (
        "contract", "capture", "attention_windows_per_call"),
        "nested AST chain was missed")
    computed = ast.parse("contract[key]").body[0].value
    require(module._chain(computed) is None, "dynamic AST key unexpectedly admitted")
    passed.append("AST_direct_and_nested_key_controls")

    m1227_source = Path(module.M1227.__file__).resolve()
    run_capture = function_ast(m1227_source, "run_capture")
    names = [node for node in ast.walk(run_capture)
             if isinstance(node, ast.Name) and node.id == "contract"]
    require(len(names) == 4, "independent M1227 contract-name use count drift")
    passed.append("independent_frozen_AST_has_no_hidden_contract_get_or_alias")

    exact_m1313 = module.strict_json(module.M1313_CONTRACT)
    runtime = module.build_runtime_contract(exact_m1313)
    require(set(runtime) == {"contract_path", "capture", "cohort", "output"} and
            runtime["capture"] == {"attention_windows_per_call": 100} and
            runtime["output"] == {"path": str(module.CANONICAL_RESULT.relative_to(ROOT))},
            "runtime projection mismatch")
    passed.append("capture100_and_new_output_projection")

    old_paths = {
        module.M1319.M1249.CANONICAL_RESULT,
        module.M1319.M1249.CANONICAL_ATTEMPT,
        module.M1319.M1249.CANONICAL_LOG,
    }
    new_paths = {module.CANONICAL_RESULT, module.CANONICAL_ATTEMPT, module.CANONICAL_LOG}
    require(len(new_paths) == 3 and not (new_paths & old_paths),
            "M1325 namespaces are not disjoint from M1249")
    passed.append("new_namespace_constants_disjoint")

    m1325_projection_calls = call_chains(function_ast(SOURCE, "validate_identity_and_project"))
    m1319_source = Path(module.M1319.__file__).resolve()
    m1319_validation_calls = call_chains(function_ast(
        m1319_source, "validate_exact_m1313_m1314"))
    m1249_source = Path(module.M1319.M1249.__file__).resolve()
    m1249_validation_calls = call_chains(function_ast(m1249_source, "validate_production_launch"))
    require("M1319.validate_exact_m1313_m1314" in m1325_projection_calls and
            "M1249.validate_production_launch" in m1319_validation_calls and
            "ensure_fresh_namespaces" in m1249_validation_calls,
            "old namespace validation call graph changed")

    injected = module.M1319.M1319Error("M1249 attempt namespace is not fresh")
    with mock.patch.object(module.M1319, "validate_exact_m1313_m1314",
                           side_effect=injected) as validator:
        try:
            module.validate_identity_and_project()
        except module.M1319.M1319Error as error:
            require("M1249 attempt namespace is not fresh" in str(error),
                    "old namespace failure was transformed")
        else:
            raise HammerError("old M1249 namespace failure was bypassed")
    validator.assert_called_once_with(module.M1313_CONTRACT, module.M1314_ENTRY)

    delegate_source = ast.unparse(function_ast(SOURCE, "delegate_for_future_release"))
    require("M1319.M1249.CANONICAL_RESULT" in delegate_source and
            "M1319.M1249.CANONICAL_ATTEMPT" not in delegate_source and
            "M1319.M1249.CANONICAL_LOG" not in delegate_source,
            "delegate namespace rewrite expectation changed")

    cli = subprocess.run([str(PYTHON), "-I", str(SOURCE), "--run"], cwd=ROOT,
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         check=False)
    require(cli.returncode != 0 and "unrecognized arguments" in cli.stdout,
            "production CLI unexpectedly admitted")
    passed.append("CLI_inert_and_no_direct_attempt_consumer")

    output = {
        "schema": "m1326_m1325_runtime_projection_shadow_hammer_r1_v1",
        "status": "FAIL_DO_NOT_CITE__M1327_FRESH_NAMESPACE_VALIDATOR_REQUIRED",
        "source_authority": {
            "source_sha256": sha256(SOURCE), "test_sha256": sha256(TEST),
            "contract_sha256": sha256(CONTRACT),
            "author_receipt_sha256": sha256(AUTHOR / "receipt.json"),
            "author_manifest_sha256": sha256(AUTHOR / "SHA256SUMS"),
            "author_outer_file_sha256": sha256(AUTHOR / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha256(DOCS359),
        },
        "independence": {"different_author": True},
        "author_tests": {"passed": True, "count": 10, "output": baseline.stdout},
        "passed_checks": passed,
        "fatal_finding": {
            "id": "F1_OLD_M1249_FRESHNESS_GATE_REUSED",
            "severity": "P0_RELEASE_BLOCKER",
            "call_graph": [
                "M1325.validate_identity_and_project",
                "M1319.validate_exact_m1313_m1314",
                "M1249.validate_production_launch",
                "M1249.ensure_fresh_namespaces",
            ],
            "effect": "A previously consumed M1249 attempt/result/log rejects identity projection before the new M1325 namespaces can be used.",
            "why_new_constants_do_not_fix_it": "M1325 rewrites only M1249.CANONICAL_RESULT later inside delegate_for_future_release; old freshness is checked earlier and old attempt/log are never projected.",
            "required_m1327_repair": "Add a narrowly copied identity-only validator that preserves M1319 extended-identity and M1313/M1314 checks but omits all old M1249 result/attempt/log freshness and consumption; the future release alone must validate and consume the three new M1325 namespaces exactly once.",
        },
        "authorization": {
            "m1325_release_authoring": False,
            "m1327_source_authoring": True,
            "production": False, "remote": False, "gpu": False,
        },
        "claim_boundary": {
            "source_only": True, "capture": False, "attempt_consumed": False,
            "production": False, "cycles": False, "speedup": False,
            "system_speedup": False, "energy": False, "ppa": False,
        },
    }
    (OUT / "hammer_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "author_test_output.txt").write_text(baseline.stdout, encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
