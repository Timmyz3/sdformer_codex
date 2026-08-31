#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local-only different-author blind hammer for M1489; no remote side effects."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
REVIEW = Path(__file__).resolve().parent
SOURCE = HW / "scripts/run_m1489_m1485_m1434_export_alias_bootstrap.py"
TEST = HW / "tests/test_run_m1489_m1485_m1434_export_alias_bootstrap.py"
M1485_TEST = HW / "tests/test_run_m1485_m1480_nested_m1233_config_compat_one_shot.py"
EXPECTED = {
    SOURCE: "98b167ed763312b33866550f43c44c01657b60046c40849fd4780da30d04a48e",
    TEST: "c7fac4aa45e96b40d66a5348534177986808d3cd5f85acd87272a528fc36f14d",
}
ALIASES = {
    "PROFILE_SOURCE_SHA256":
        "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    "ATLIF_OVERLAY_SOURCE_SHA256":
        "d9ee7e172f941a53ad1c031b0d5cdbbf7819f521c807e5bc54001a80c41b57f3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load():
    spec = importlib.util.spec_from_file_location("m1490_bound_m1489", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M1489")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load()


def rejected(thunk) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def aliases_absent(target=None) -> bool:
    target = M.M1434 if target is None else target
    return all(not hasattr(target, name) for name in ALIASES)


def alias_values(target=None) -> dict[str, object]:
    target = M.M1434 if target is None else target
    return {name: getattr(target, name, None) for name in ALIASES}


def namespace_state() -> dict[str, tuple[bool, int | None, int | None]]:
    state = {}
    for name in ("CANONICAL_RESULT", "CANONICAL_ATTEMPT", "CANONICAL_LOG"):
        path = getattr(M.M1485.M1480.M1475.M1458, name)
        exists = os.path.lexists(str(path))
        stat_value = path.lstat() if exists else None
        state[name] = (exists,
                       None if stat_value is None else stat_value.st_mtime_ns,
                       None if stat_value is None else stat_value.st_size)
    return state


def main() -> int:
    checks: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []

    def check(name, passed, category):
        checks.append({"check": name, "category": category, "pass": bool(passed)})

    def attack(name, thunk, category, cleanup=None):
        caught = rejected(thunk)
        clean = True if cleanup is None else bool(cleanup())
        accepted = caught and clean
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "cleanup": clean,
                        "false_negative": not accepted})

    native_source = subprocess.run(
        [sys.executable, str(SOURCE), "--source-self-check"], cwd=ROOT,
        text=True, capture_output=True, check=False)
    check("native_source_self_check", native_source.returncode == 0 and
          native_source.stdout.strip() ==
          "PASS_M1489_SOURCE_SELF_CHECK__NO_REMOTE_NO_GPU_NO_ATTEMPT" and
          not native_source.stderr.strip(), "tests")
    native_tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(M1485_TEST), str(TEST)],
        cwd=ROOT, text=True, capture_output=True, check=False)
    check("native_author_tests_22", native_tests.returncode == 0 and
          "22 passed" in native_tests.stdout, "tests")

    for path, expected in EXPECTED.items():
        check("sha_" + path.name, path.is_file() and not path.is_symlink() and
              sha256(path) == expected, "identity")
    check("m1485_source_exact", M.M1485_SOURCE.is_file() and
          not M.M1485_SOURCE.is_symlink() and
          sha256(M.M1485_SOURCE) == M.M1485_SOURCE_SHA256, "identity")
    check("exact_runtime_m1434_object",
          M.M1434 is M.M1485.M1480.M1475.M1458.M1434, "identity")
    check("sealed_m1349_digest_source", all(
          type(getattr(M.M1434.M1349, name)) is str and
          getattr(M.M1434.M1349, name) == value
          for name, value in ALIASES.items()), "predecessor")
    check("aliases_initially_absent", aliases_absent(), "freshness")

    with M.export_digest_aliases():
        check("exact_alias_export", alias_values() == ALIASES, "lifecycle")
    check("normal_exit_deletes_aliases", aliases_absent(), "lifecycle")

    original_namespace = namespace_state()
    capture = {}

    def preflight_probe():
        capture["preflight_module"] = M.M1434
        capture["preflight_aliases"] = alias_values()

    with mock.patch.object(M.M1485, "remote_preflight", side_effect=preflight_probe):
        M.remote_preflight()
    check("remote_preflight_exact_delegate", capture == {
        "preflight_module": M.M1485.M1480.M1475.M1458.M1434,
        "preflight_aliases": ALIASES}, "delegation")
    check("remote_preflight_cleanup", aliases_absent(), "lifecycle")

    sentinel = ROOT / ".m1490_inert_delegate_sentinel"
    temp_log = ROOT / ".m1490_inert_temporary_log"

    def execute_probe(path):
        capture["execute_path"] = path
        capture["execute_aliases"] = alias_values()
        return sentinel

    with mock.patch.object(M.M1485, "execute_once", side_effect=execute_probe):
        delegated_result = M.execute_once(temp_log)
    check("execute_once_exact_delegate", delegated_result == sentinel and
          capture.get("execute_path") == temp_log and
          capture.get("execute_aliases") == ALIASES, "delegation")
    check("execute_once_cleanup", aliases_absent(), "lifecycle")
    check("namespace_unchanged_after_mocked_delegation",
          namespace_state() == original_namespace, "ownership")

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    owned_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    check("no_result_attempt_log_symbols",
          not ({"CANONICAL_RESULT", "CANONICAL_ATTEMPT", "CANONICAL_LOG",
                "consume_attempt"} & owned_names), "ownership")
    check("no_m1489_namespace_attributes",
          all(not hasattr(M, name) for name in
              ("CANONICAL_RESULT", "CANONICAL_ATTEMPT", "CANONICAL_LOG")),
          "ownership")

    for name in ALIASES:
        for value_label, value in (("exact_value", ALIASES[name]),
                                   ("wrong_value", "0" * 64),
                                   ("wrong_type", 7)):
            def preinstalled(name=name, value=value):
                setattr(M.M1434, name, value)
                try:
                    M.validate_bootstrap()
                finally:
                    if hasattr(M.M1434, name):
                        delattr(M.M1434, name)
            attack("preinstalled_" + name + "_" + value_label,
                   preinstalled, "preinstalled", aliases_absent)

    for name, exact in ALIASES.items():
        def missing_predecessor(name=name):
            with mock.patch.object(M.M1434.M1349, name, create=True) as patched:
                delattr(M.M1434.M1349, name)
                M.validate_bootstrap()
        attack("missing_m1349_" + name, missing_predecessor, "predecessor",
               aliases_absent)
        for label, replacement in (("wrong_type", 7),
                                   ("wrong_value", "f" * 64)):
            def changed_predecessor(name=name, replacement=replacement):
                with mock.patch.object(M.M1434.M1349, name, replacement):
                    M.validate_bootstrap()
            attack("m1349_" + name + "_" + label, changed_predecessor,
                   "predecessor", aliases_absent)

    def exception_body():
        with M.export_digest_aliases():
            raise RuntimeError("m1490 injected body exception")
    attack("body_exception", exception_body, "restoration", aliases_absent)

    for name in ALIASES:
        for label, replacement in (("wrong_type", 7),
                                   ("wrong_value", "e" * 64)):
            def tamper(name=name, replacement=replacement):
                with M.export_digest_aliases():
                    setattr(M.M1434, name, replacement)
            attack("tamper_" + name + "_" + label, tamper, "tamper",
                   aliases_absent)

        def delete_inside(name=name):
            with M.export_digest_aliases():
                delattr(M.M1434, name)
        attack("tamper_delete_" + name, delete_inside, "tamper", aliases_absent)

    def delegated_preflight_exception():
        with mock.patch.object(M.M1485, "remote_preflight",
                               side_effect=RuntimeError("delegate exception")):
            M.remote_preflight()
    attack("preflight_delegate_exception", delegated_preflight_exception,
           "restoration", aliases_absent)

    def delegated_execute_exception():
        with mock.patch.object(M.M1485, "execute_once",
                               side_effect=RuntimeError("delegate exception")):
            M.execute_once(temp_log)
    attack("execute_delegate_exception", delegated_execute_exception,
           "restoration", aliases_absent)

    real_m1434 = M.M1485.M1480.M1475.M1458.M1434
    decoy = SimpleNamespace(M1349=real_m1434.M1349)

    def different_module_object():
        def exact_delegate_probe():
            if aliases_absent(real_m1434):
                raise M.M1489Error("aliases absent on exact M1485 M1434 object")
        with mock.patch.object(M, "M1434", decoy), \
                mock.patch.object(M.M1485, "remote_preflight",
                                  side_effect=exact_delegate_probe):
            M.remote_preflight()
    attack("different_m1434_module_object", different_module_object,
           "module_identity", lambda: aliases_absent(real_m1434) and
           aliases_absent(decoy))
    check("exact_module_restored_after_decoy",
          M.M1434 is real_m1434 and aliases_absent(real_m1434), "module_identity")
    check("namespace_unchanged_after_attacks",
          namespace_state() == original_namespace, "ownership")

    false_negatives = sum(int(row["false_negative"]) for row in attacks)
    failed_checks = sum(int(not row["pass"]) for row in checks)
    categories = {}
    for row in attacks:
        item = categories.setdefault(row["category"],
                                     {"attacks": 0, "rejected": 0,
                                      "false_negatives": 0})
        item["attacks"] += 1
        item["rejected"] += int(row["rejected"])
        item["false_negatives"] += int(row["false_negative"])
    result = {
        "schema": "m1490_m1489_m1434_export_alias_bootstrap_blind_hammer_output_r1_v1",
        "check_count": len(checks), "failed_checks": failed_checks,
        "attack_count": len(attacks), "false_negatives": false_negatives,
        "checks": checks, "attacks": attacks, "attack_categories": categories,
        "native_source_self_check_stdout": native_source.stdout,
        "native_source_self_check_stderr": native_source.stderr,
        "native_pytest_stdout": native_tests.stdout,
        "native_pytest_stderr": native_tests.stderr,
        "execution": {"ssh": 0, "remote_preflight": 0, "remote_runs": 0,
                      "real_gpu_queries": 0, "capture_runs": 0,
                      "production_attempts_consumed": 0,
                      "controller_signals": 0, "controller_restores": 0,
                      "eda_runs": 0},
        "verdict": "PASS" if failed_checks == 0 and false_negatives == 0
                   else "FAIL_DO_NOT_CITE",
    }
    (REVIEW / "hammer_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("check_count", "failed_checks", "attack_count",
                       "false_negatives", "verdict")}, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
