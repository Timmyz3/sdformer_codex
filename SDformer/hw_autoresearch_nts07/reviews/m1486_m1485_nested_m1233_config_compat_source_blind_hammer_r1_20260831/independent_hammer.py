#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local-only different-author blind hammer for M1485; no remote side effects."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
REVIEW = Path(__file__).resolve().parent
SOURCE = HW / "scripts/run_m1485_m1480_nested_m1233_config_compat_one_shot.py"
TEST = HW / "tests/test_run_m1485_m1480_nested_m1233_config_compat_one_shot.py"
M1480_TEST = HW / "tests/test_run_m1480_m1475_exact_type_config_compat_one_shot.py"
M1475_TEST = HW / "tests/test_run_m1475_m1458_config_content_compat_one_shot.py"
CONTRACT = HW / (
    "contracts/m1485_m1480_nested_m1233_config_compat_source_contract_r1_20260831.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    SOURCE: "d9779f52bd6342898b26f14b05f8052888fd81cb35d73d10168319ade6d8db9a",
    TEST: "7ff297bfc5a16e3dc01b2bac089d216fb5a899a5acae889ba9f072734da4510c",
    CONTRACT: "44e8d98a5b3d997a16bdac158936e27e95eb4f66787602abc0c78edbd7aa7e2e",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load():
    spec = importlib.util.spec_from_file_location("m1486_bound_m1485", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M1485")
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


def main() -> int:
    checks = []
    attacks = []

    def check(name, passed, category):
        checks.append({"check": name, "category": category, "pass": bool(passed)})

    def attack(name, thunk, category):
        caught = rejected(thunk)
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    # M1485 correctly requires its future M1486 directory to be absent during
    # the source-author self-check.  Move this not-yet-sealed review directory
    # aside on the same filesystem for the duration of that native CLI only,
    # and restore it unconditionally before producing any evidence.
    staged_review = REVIEW.parent / (
        ".m1486_native_source_check_stage." + str(os.getpid()))
    if staged_review.exists() or staged_review.is_symlink():
        raise RuntimeError("native source-check staging path already exists")
    REVIEW.rename(staged_review)
    try:
        native_source = subprocess.run(
            ["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(SOURCE),
             "--source-self-check"], cwd=ROOT, text=True,
            capture_output=True, check=False)
    finally:
        if REVIEW.exists() or REVIEW.is_symlink():
            raise RuntimeError("canonical M1486 path occupied during restoration")
        staged_review.rename(REVIEW)
    check("native_source_self_check", native_source.returncode == 0 and
          native_source.stdout.strip() ==
          "PASS_M1485_SOURCE_SELF_CHECK__NO_REMOTE_NO_GPU_NO_ATTEMPT" and
          not native_source.stderr.strip(), "tests")

    for path, expected in EXPECTED.items():
        check("sha_" + path.name, path.is_file() and sha(path) == expected, "identity")
    policy = M.strict_json(CONTRACT)
    check("contract_status", policy.get("status") ==
          "SOURCE_ONLY__NESTED_M1233_CONFIG_CONTENT_COMPAT__M1486_REQUIRED__NO_LAUNCH",
          "identity")
    check("contract_source", policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)}, "identity")
    check("contract_test", policy.get("test") == {
        "path": str(TEST.relative_to(ROOT)), "sha256": sha(TEST)}, "identity")
    check("contract_no_launch", policy.get("claim_boundary", {}).get("launch") is False,
          "authority")
    check("m1480_pin", sha(M.M1480_SOURCE) == M.M1480_SOURCE_SHA256 and
          sha(M.M1480_TEST) == M.M1480_TEST_SHA256 and
          sha(M.M1480_CONTRACT) == M.M1480_CONTRACT_SHA256, "predecessor")
    predecessor = M.M1480.M1475.M1458.verify_double_seal(
        M.M1483, M.M1483_REVIEW_SHA256, M.M1483_MANIFEST_SHA256,
        M.M1483_OUTER_SHA256)
    check("m1483_pin", predecessor.get("status") ==
          "PASS_M1483_M1480_EXACT_TYPE_CONFIG_COMPAT_FINAL_LAUNCH", "predecessor")
    check("future_release_absent", not os.path.lexists(str(M.RELEASE)), "freshness")
    check("future_final_absent", not os.path.lexists(str(M.FINAL)), "freshness")

    native = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "pytest", "-q",
         str(M1475_TEST), str(M1480_TEST), str(TEST)], cwd=ROOT, text=True,
        capture_output=True, check=False)
    check("native_author_tests_43", native.returncode == 0 and
          "43 passed" in native.stdout, "tests")

    config = dict(M.M1480.M1475.FROZEN_CONFIG_ENTITY)
    original_ext = M.ORIGINAL_EXTENDED_EXACT_IDENTITY
    original_frozen = M.ORIGINAL_M1233_EXACT_IDENTITY
    delegated = []

    def ext_original(value, label):
        delegated.append(("extended", label))
        raise M.M1485Error("extended original")

    def frozen_original(value, label):
        delegated.append(("frozen", label))
        raise M.M1485Error("frozen original")

    with mock.patch.object(M, "ORIGINAL_EXTENDED_EXACT_IDENTITY", ext_original), \
            mock.patch.object(M, "ORIGINAL_M1233_EXACT_IDENTITY", frozen_original), \
            mock.patch.object(M.M1319, "exact_extended_identity", ext_original), \
            mock.patch.object(M.FROZEN_M1233, "exact_identity", frozen_original), \
            mock.patch.object(M.M1480.M1475, "verify_configuration_content_identity",
                              side_effect=lambda value: dict(value)):
        with M.dual_configuration_compatibility():
            check("both_real_objects_patched",
                  M.M1319.exact_extended_identity is not ext_original and
                  M.FROZEN_M1233.exact_identity is not frozen_original,
                  "dual_patch")
            check("extended_selected_configuration_accept",
                  M.M1319.exact_extended_identity(config, "selected configuration") == config,
                  "dual_patch")
            check("nested_selected_configuration_accept",
                  M.FROZEN_M1233.exact_identity(config, "selected configuration") == config,
                  "dual_patch")
            attack("checkpoint_delegates_original", lambda:
                   M.FROZEN_M1233.exact_identity(config, "selected checkpoint"), "label")
            attack("profile_delegates_original", lambda:
                   M.FROZEN_M1233.exact_identity(config, "selected profile"), "label")
            for label in ("Selected configuration", "selected configuration ",
                          "selected_configuration", "selected configuration\x00"):
                attack("label_spoof_" + repr(label), lambda label=label:
                       M.FROZEN_M1233.exact_identity(config, label), "label")
        check("both_real_objects_restored",
              M.M1319.exact_extended_identity is ext_original and
              M.FROZEN_M1233.exact_identity is frozen_original, "restore")
    check("checkpoint_profile_original_observed",
          delegated[:2] == [("frozen", "selected checkpoint"),
                            ("frozen", "selected profile")], "label")
    check("module_originals_unchanged",
          M.M1319.exact_extended_identity is original_ext and
          M.FROZEN_M1233.exact_identity is original_frozen, "restore")

    expected = M.M1480.M1475.FROZEN_CONFIG_ENTITY
    for field, frozen in expected.items():
        changed = dict(expected); changed.pop(field)
        attack("config_missing_" + field,
               lambda changed=changed: M.verify_frozen_config_entity_exact_type(changed),
               "config_shape")
        changed = dict(expected)
        if type(frozen) is int:
            changed[field] = float(frozen)
        else:
            changed[field] = True
        attack("config_type_" + field,
               lambda changed=changed: M.verify_frozen_config_entity_exact_type(changed),
               "config_type")
        changed = dict(expected)
        changed[field] = frozen + 1 if type(frozen) is int else frozen + ".drift"
        attack("config_value_" + field,
               lambda changed=changed: M.verify_frozen_config_entity_exact_type(changed),
               "config_value")
    changed = dict(expected); changed["extra"] = 0
    attack("config_extra_key", lambda:
           M.verify_frozen_config_entity_exact_type(changed), "config_shape")
    for malformed in (None, [], True, 1, "config"):
        attack("config_nonmapping_" + type(malformed).__name__,
               lambda malformed=malformed:
               M.verify_frozen_config_entity_exact_type(malformed), "config_shape")

    payload = b"m1486-exact-config" * 400
    with tempfile.TemporaryDirectory(prefix="m1486_cfg_") as raw:
        root = Path(raw)
        path = root / "config.yml"
        path.write_bytes(payload)
        entity = dict(expected)
        entity.update({"absolute_path": str(path), "size_bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest()})
        patch = dict(CONFIG_PATH=path, CONFIG_ABSOLUTE=str(path),
                     CONFIG_SIZE=len(payload), CONFIG_SHA256=entity["sha256"],
                     FROZEN_CONFIG_ENTITY=dict(entity))
        with mock.patch.multiple(M.M1480.M1475, **patch):
            check("regular_exact_content_accept",
                  M.M1480.M1475.verify_configuration_content_identity(entity) == entity,
                  "file_identity")
            wrong = dict(entity); wrong["absolute_path"] += ".wrong"
            attack("wrong_path", lambda:
                   M.M1480.M1475.verify_configuration_content_identity(wrong),
                   "file_identity")
            relative = dict(entity); relative["absolute_path"] = "config.yml"
            attack("relative_path", lambda:
                   M.M1480.M1475.verify_configuration_content_identity(relative),
                   "file_identity")
            path.unlink()
            attack("missing_file", lambda:
                   M.M1480.M1475.verify_configuration_content_identity(entity),
                   "file_identity")
            target = root / "target"; target.write_bytes(payload); path.symlink_to(target)
            attack("symlink_file", lambda:
                   M.M1480.M1475.verify_configuration_content_identity(entity),
                   "file_identity")
            path.unlink(); path.mkdir()
            attack("nonregular_file", lambda:
                   M.M1480.M1475.verify_configuration_content_identity(entity),
                   "file_identity")
            path.rmdir(); path.write_bytes(payload + b"size")
            attack("size_drift", lambda:
                   M.M1480.M1475.verify_configuration_content_identity(entity),
                   "file_identity")
            path.write_bytes(payload[:-1] + b"X")
            attack("sha_drift", lambda:
                   M.M1480.M1475.verify_configuration_content_identity(entity),
                   "file_identity")
            path.write_bytes(payload)
            def racing_sha(_path):
                path.write_bytes(payload + b"race")
                return entity["sha256"]
            with mock.patch.object(M.M1480.M1475, "sha256", side_effect=racing_sha):
                attack("stat_race", lambda:
                       M.M1480.M1475.verify_configuration_content_identity(entity),
                       "file_identity")

    attack("preinstalled_extended", lambda:
           _preinstalled(M, "extended"), "reentrancy")
    attack("preinstalled_nested", lambda:
           _preinstalled(M, "nested"), "reentrancy")
    attack("reentrant_context", lambda: _reentrant(M), "reentrancy")
    try:
        with M.dual_configuration_compatibility():
            raise RuntimeError("body exception")
    except RuntimeError:
        pass
    check("exception_restores_both",
          M.M1319.exact_extended_identity is original_ext and
          M.FROZEN_M1233.exact_identity is original_frozen, "restore")
    attack("inner_tamper_rejected", lambda: _inner_tamper(M), "restore")
    check("inner_tamper_restores_both",
          M.M1319.exact_extended_identity is original_ext and
          M.FROZEN_M1233.exact_identity is original_frozen, "restore")

    exact_false = {"launch": False, "runs": 0, "automatic_retry": False,
                   "controller_restore": False}
    exact_true = {"launch": True, "runs": 1, "automatic_retry": False,
                  "controller_restore": False}
    M.exact_authorization(dict(exact_false), False)
    M.exact_authorization(dict(exact_true), True)
    check("future_authorization_exact_accept", True, "future_authority")
    for expected_launch, base in ((False, exact_false), (True, exact_true)):
        for field, replacement in (("launch", int(expected_launch)),
                                   ("runs", bool(base["runs"])),
                                   ("runs", float(base["runs"])),
                                   ("automatic_retry", 0),
                                   ("controller_restore", 0)):
            changed = dict(base); changed[field] = replacement
            attack("authority_type_" + str(expected_launch) + "_" + field +
                   "_" + type(replacement).__name__, lambda changed=changed,
                   expected_launch=expected_launch:
                   M.exact_authorization(changed, expected_launch), "future_authority")
        for field in base:
            changed = dict(base); changed.pop(field)
            attack("authority_missing_" + str(expected_launch) + "_" + field,
                   lambda changed=changed, expected_launch=expected_launch:
                   M.exact_authorization(changed, expected_launch), "future_authority")
        changed = dict(base); changed["extra"] = False
        attack("authority_extra_" + str(expected_launch), lambda changed=changed,
               expected_launch=expected_launch:
               M.exact_authorization(changed, expected_launch), "future_authority")

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
        "schema": "m1486_m1485_nested_m1233_config_compat_blind_hammer_output_r1_v1",
        "check_count": len(checks), "failed_checks": failed_checks,
        "attack_count": len(attacks), "false_negatives": false_negatives,
        "checks": checks, "attacks": attacks, "attack_categories": categories,
        "native_source_self_check_stdout": native_source.stdout,
        "native_source_self_check_stderr": native_source.stderr,
        "native_pytest_stdout": native.stdout, "native_pytest_stderr": native.stderr,
        "execution": {"ssh": 0, "remote_runs": 0, "real_gpu_queries": 0,
                      "capture_runs": 0, "production_attempts_consumed": 0,
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


def _preinstalled(module, which):
    target = module.M1319 if which == "extended" else module.FROZEN_M1233
    name = "exact_extended_identity" if which == "extended" else "exact_identity"
    old = getattr(target, name)
    setattr(target, name, lambda value, label: value)
    try:
        with module.dual_configuration_compatibility():
            pass
    finally:
        setattr(target, name, old)


def _reentrant(module):
    with module.dual_configuration_compatibility():
        with module.dual_configuration_compatibility():
            pass


def _inner_tamper(module):
    with module.dual_configuration_compatibility():
        module.M1319.exact_extended_identity = lambda value, label: value
        module.FROZEN_M1233.exact_identity = lambda value, label: value


if __name__ == "__main__":
    raise SystemExit(main())
