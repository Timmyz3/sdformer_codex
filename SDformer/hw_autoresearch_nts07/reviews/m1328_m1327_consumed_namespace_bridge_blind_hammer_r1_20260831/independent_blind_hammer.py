#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author blind hammer for sealed source-only M1327."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                 "capture_m1327_motion_ep34_consumed_namespace_bridge_r1.py")
TEST = HW / "tests/test_m1327_motion_ep34_consumed_namespace_bridge.py"
CONTRACT = HW / "contracts/m1327_motion_ep34_consumed_namespace_bridge_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1327_motion_ep34_consumed_namespace_bridge_source_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

EXPECTED = {
    "source": "2ab5024a11a81f7bb3ed75956114cc95e07dbe0782328414f2bd3c79342c3ac9",
    "test": "3c37f5cd2dbf7611e2c984bcad61d65a2492004db44fcc85b20aa678c7fc1dcf",
    "contract": "03aca58a422bdfd080b82ea79429948bfb6a04ef4ba1b3b4d6e52e2f75214330",
    "author_receipt": "e5d3d58653d5dac54e6d3e07d6a905dce70243dac29a38b7e2791318a6f031bc",
    "author_manifest": "55b8bd6ce0177637dfa1889b6d4291d32d50bf545a69af1c7a06f133f278fda0",
    "author_outer": "14a82a5ffd941bd1533906b2488d8202cd6caf93d903f7fbaf354dc9a5665000",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64, "malformed manifest")
        relative = PurePosixPath(parts[1])
        require(relative.parts and not relative.is_absolute() and ".." not in relative.parts and
                parts[1] not in rows, "unsafe/duplicate manifest member")
        rows[parts[1]] = parts[0]
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and path.name not in {
                        "SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sorted(rows) == actual, "author recursive members drift")
    for relative, digest in rows.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and sha256(path) == digest,
                "author member drift: " + relative)
    return rows


def load_source():
    spec = importlib.util.spec_from_file_location("m1328_sealed_m1327", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1327")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expect_reject(module, function, label: str) -> str:
    try:
        function()
    except module.M1327Error as error:
        return str(error)
    except Exception as error:
        raise HammerError(label + " raised wrong exception " + type(error).__name__) from error
    raise HammerError(label + " was accepted")


def patch_old(module, attempt: Path, result: Path, log: Path):
    return (
        mock.patch.object(module.M1249, "CANONICAL_ATTEMPT", attempt),
        mock.patch.object(module.M1249, "CANONICAL_RESULT", result),
        mock.patch.object(module.M1249, "CANONICAL_LOG", log),
    )


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
    checks = []

    require(module.verify_m1326_failure()["verdict"] == "NO_GO_M1325_PRODUCTION_RELEASE",
            "M1326 failure authority mismatch")
    forensic = module.M1325.verify_m1324_forensic()
    require(forensic["failed_execution_evidence"]["attempt_sha256"] ==
            sha256(module.FORENSIC_ATTEMPT), "forensic attempt authority mismatch")
    checks.append("sealed_failure_authorities_exact")

    with tempfile.TemporaryDirectory(prefix="m1328_old_") as temp_name:
        temp = Path(temp_name)
        result = temp / "old.result"
        log = temp / "old.log"
        patchers = patch_old(module, module.FORENSIC_ATTEMPT, result, log)
        for patcher in patchers:
            patcher.start()
        try:
            state = module.verify_old_consumed_failure_state()
            require(state == {
                "status": "PASS_EXACT_OLD_CONSUMED_FAILURE_STATE",
                "old_attempt_sha256": sha256(module.FORENSIC_ATTEMPT),
                "old_attempt_read_only": True, "old_result_absent": True,
                "old_canonical_log_absent": True, "sealed_temp_log_zero": True,
            }, "exact consumed state projection drift")
        finally:
            for patcher in reversed(patchers):
                patcher.stop()
        checks.append("sealed_real_consumed_attempt_copy_passes")

        missing = temp / "missing.attempt"
        patchers = patch_old(module, missing, result, log)
        for patcher in patchers: patcher.start()
        try:
            expect_reject(module, module.verify_old_consumed_failure_state,
                          "missing old attempt")
        finally:
            for patcher in reversed(patchers): patcher.stop()

        writable = temp / "writable.attempt"
        writable.write_bytes(module.M1249.ATTEMPT_TOKEN.encode("ascii")); writable.chmod(0o600)
        patchers = patch_old(module, writable, result, log)
        for patcher in patchers: patcher.start()
        try:
            expect_reject(module, module.verify_old_consumed_failure_state,
                          "writable old attempt")
        finally:
            for patcher in reversed(patchers): patcher.stop()

        wrong = temp / "wrong.attempt"
        wrong.write_bytes(b"wrong\n"); wrong.chmod(0o400)
        patchers = patch_old(module, wrong, result, log)
        for patcher in patchers: patcher.start()
        try:
            expect_reject(module, module.verify_old_consumed_failure_state,
                          "wrong old attempt")
        finally:
            for patcher in reversed(patchers): patcher.stop()

        symlink = temp / "symlink.attempt"
        symlink.symlink_to(module.FORENSIC_ATTEMPT)
        patchers = patch_old(module, symlink, result, log)
        for patcher in patchers: patcher.start()
        try:
            expect_reject(module, module.verify_old_consumed_failure_state,
                          "symlink old attempt")
        finally:
            for patcher in reversed(patchers): patcher.stop()
        checks.append("missing_writable_wrong_symlink_attempts_rejected")

        for occupied, label in ((result, "old result"), (log, "old canonical log")):
            occupied.write_text("occupied", encoding="utf-8")
            patchers = patch_old(module, module.FORENSIC_ATTEMPT, result, log)
            for patcher in patchers: patcher.start()
            try:
                expect_reject(module, module.verify_old_consumed_failure_state, label)
            finally:
                for patcher in reversed(patchers): patcher.stop()
                occupied.unlink()
        checks.append("old_result_and_log_presence_rejected")

        original = module.M1249.ensure_fresh_namespaces
        patchers = patch_old(module, module.FORENSIC_ATTEMPT, result, log)
        for patcher in patchers: patcher.start()
        try:
            with module.old_consumed_freshness_bridge():
                require(module.M1249.ensure_fresh_namespaces is
                        module.verify_old_consumed_failure_state,
                        "bridge did not patch exact callback")
                nested_rejected = False
                try:
                    with module.old_consumed_freshness_bridge():
                        pass
                except module.M1327Error:
                    nested_rejected = True
                require(nested_rejected, "nested callback patch was accepted")
            require(module.M1249.ensure_fresh_namespaces is original,
                    "normal bridge did not restore callback")
            try:
                with module.old_consumed_freshness_bridge():
                    raise RuntimeError("injected")
            except RuntimeError:
                pass
            require(module.M1249.ensure_fresh_namespaces is original,
                    "exception bridge did not restore callback")
        finally:
            for patcher in reversed(patchers): patcher.stop()
        checks.append("exact_one_callback_normal_exception_nested_restore")

        source_tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        assignments = []
        for node in ast.walk(source_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "ensure_fresh_namespaces":
                        assignments.append(ast.unparse(node))
        require(len(assignments) == 2 and
                any("verify_old_consumed_failure_state" in row for row in assignments) and
                any("original" in row for row in assignments),
                "source patches more or less than one callback/restore")
        checks.append("AST_exact_patch_and_restore_only")

        exact_contract = module.strict_json(module.M1313_CONTRACT)
        binding = {
            "policy": {"frozen": True}, "verified_samples": ["identity"],
            "identity": {"m1319_projection":
                "extended7_verified_then_frozen_keyset_temporarily_extended"},
            "selection": {"exact": True}, "checkpoint_path": Path("checkpoint"),
            "config_path": Path("config"),
        }
        observed = []
        def unchanged_validator(path, entry):
            observed.append((path, copy.deepcopy(entry), module.M1249.ensure_fresh_namespaces()))
            return copy.deepcopy(exact_contract), binding
        patchers = patch_old(module, module.FORENSIC_ATTEMPT, result, log)
        for patcher in patchers: patcher.start()
        try:
            with mock.patch.object(module.M1319, "validate_exact_m1313_m1314",
                                   side_effect=unchanged_validator) as validator:
                runtime, returned_binding = module.validate_identity_and_project()
            validator.assert_called_once_with(module.M1313_CONTRACT, module.M1314_ENTRY)
            require(returned_binding is binding and binding["identity"]["m1319_projection"].startswith(
                    "extended7_verified"), "M1319 binding was copied/mutated")
            require(observed[0][2]["status"] == "PASS_EXACT_OLD_CONSUMED_FAILURE_STATE",
                    "unchanged validator did not invoke real bridge callback")
            require(module.M1249.ensure_fresh_namespaces is original,
                    "identity projection leaked callback patch")
        finally:
            for patcher in reversed(patchers): patcher.stop()
        require(set(runtime) == {"contract_path", "capture", "cohort", "output"} and
                runtime["capture"] == {"attention_windows_per_call": 100} and
                runtime["output"] == {"path": str(module.CANONICAL_RESULT.relative_to(ROOT))},
                "runtime four-key projection drift")
        checks.append("M1319_binding_unchanged_and_runtime_four_keys")

    old_namespaces = {module.M1249.CANONICAL_RESULT, module.M1249.CANONICAL_ATTEMPT,
                      module.M1249.CANONICAL_LOG}
    new_namespaces = {module.CANONICAL_RESULT, module.CANONICAL_ATTEMPT,
                      module.CANONICAL_LOG}
    require(len(new_namespaces) == 3 and not new_namespaces & old_namespaces,
            "new namespaces overlap old M1249")
    with tempfile.TemporaryDirectory(prefix="m1328_new_") as temp_name:
        paths = [Path(temp_name) / name for name in ("result", "attempt", "log")]
        with mock.patch.object(module, "CANONICAL_RESULT", paths[0]), \
             mock.patch.object(module, "CANONICAL_ATTEMPT", paths[1]), \
             mock.patch.object(module, "CANONICAL_LOG", paths[2]):
            module.require_fresh_m1327_namespaces()
            paths[2].write_text("occupied", encoding="utf-8")
            expect_reject(module, module.require_fresh_m1327_namespaces,
                          "occupied new log")
    checks.append("new_namespaces_disjoint_and_fresh_fail_closed")

    child_binding = {
        "policy": {}, "verified_samples": [], "identity": {}, "selection": {},
        "checkpoint_path": Path("checkpoint"), "config_path": Path("config"),
    }
    original_result = module.M1249.CANONICAL_RESULT
    captured = []
    def capture(contract, binding_arg, substrate=None):
        captured.append((copy.deepcopy(contract), binding_arg, substrate,
                         module.M1249.CANONICAL_RESULT))
        return module.M1249.CANONICAL_RESULT
    with mock.patch.object(module.M1249, "run_capture", side_effect=capture):
        output = module.delegate_for_future_release(
            module.build_runtime_contract(module.strict_json(module.M1313_CONTRACT)),
            child_binding, object())
    require(output == module.CANONICAL_RESULT and captured[0][3] == module.CANONICAL_RESULT and
            captured[0][1] is child_binding and module.M1249.CANONICAL_RESULT is original_result,
            "new output propagation/restoration failed")
    checks.append("new_output_propagates_and_restores")

    cli = subprocess.run([str(PYTHON), "-I", str(SOURCE), "--run"], cwd=ROOT,
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         check=False)
    require(cli.returncode != 0 and "unrecognized arguments" in cli.stdout,
            "production CLI unexpectedly admitted")
    checks.append("CLI_inert_no_attempt_consumer")

    output = {
        "schema": "m1328_m1327_consumed_namespace_bridge_blind_hammer_r1_v1",
        "status": "PASS_M1328_M1327_SOURCE_HAMMER__MINIMAL_RELEASE_AUTHORING_ALLOWED",
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
        "blind_checks": checks,
        "authorization": {
            "minimal_release_authoring": True, "production_execution": False,
            "remote": False, "gpu": False,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
