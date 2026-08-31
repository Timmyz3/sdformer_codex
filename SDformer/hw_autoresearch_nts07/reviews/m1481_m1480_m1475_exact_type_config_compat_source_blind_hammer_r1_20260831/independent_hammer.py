#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local-only different-author blind hammer for M1480.

This program is forbidden from SSH, GPU access, capture, attempt consumption,
controller operations, and EDA.  Every mutation is performed against local
temporary authority/configuration fixtures.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
REVIEW = Path(__file__).resolve().parent
SOURCE = HW / "scripts/run_m1480_m1475_exact_type_config_compat_one_shot.py"
TEST = HW / "tests/test_run_m1480_m1475_exact_type_config_compat_one_shot.py"
M1475_TEST = HW / "tests/test_run_m1475_m1458_config_content_compat_one_shot.py"
CONTRACT = HW / (
    "contracts/m1480_m1475_exact_type_config_compat_source_contract_r1_20260831.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1475_SOURCE = HW / "scripts/run_m1475_m1458_config_content_compat_one_shot.py"
M1475_CONTRACT = HW / (
    "contracts/m1475_m1458_config_content_compat_source_contract_r1_20260831.json")
M1458_SOURCE = HW / (
    "scripts/run_m1458_m1434_motion_ep34_live93_production_one_shot.py")
M1476 = HW / (
    "reviews/m1476_m1475_m1458_config_content_compat_source_blind_hammer_"
    "r1_20260831")

EXPECTED = {
    SOURCE: "3a0235f91d8d6acd4c94168b3b611cb53504f50e3843580c09bc1673042df4ce",
    TEST: "dea2bc2cb3851a40462f5200b423c623331aa20abc054debc8e2ea661fc99ea3",
    CONTRACT: "c4ec0a4792a7647c46614652147de6999d2dce0c6c55d5d46a88798e12ad90e4",
    M1475_SOURCE: "2a5104b79e0d6563d8145a1e4ba136c9a2a047963d66d08a5d6b0bde93c5ac06",
    M1475_TEST: "25de303df2883dc450080d7f57c1f64047a349c32610f555cea990d5553ac10b",
    M1475_CONTRACT: "9cb1fd126621f85f7ab6ba4e7c960687ea19c49c85c901e8335a12108d4ab7b2",
    M1458_SOURCE: "e81c20056dd261619f88884f2f097c9b594887927121d9e599a4f89185d33154",
    M1476 / "review.json": "013308a83ca8f9732f9c600562c49d5ff15cb3b35fc65a3ec58230b396d0bd70",
    M1476 / "SHA256SUMS": "d3f541b213d2a5efe0b2ef9224d3ba9977f4245d7039a4b28ce7b1f3bfa12c1d",
    M1476 / "SHA256SUMS.seal.sha256": "4f2a7138200de5059ca43031b613a2dc8fb52801340e7497af21309b8cada727",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load():
    spec = importlib.util.spec_from_file_location("m1481_bound_m1480", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M1480")
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


def selected_entity(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "absolute_path": str(path),
        "size_bytes": len(payload),
        "mtime_ns": 1788081356000000000,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "device": 194,
        "inode": 26561699333,
        "mode": 33152,
    }


def valid_authorities(release: Path):
    runner_sha = sha(SOURCE)
    nonlaunch = {
        "launch": False, "runs": 0, "automatic_retry": False,
        "controller_restore": False,
    }
    launch = {
        "launch": True, "runs": 1, "automatic_retry": False,
        "controller_restore": False,
    }
    blind = {
        "status": "PASS_M1480_EXACT_TYPE_CONFIG_COMPAT_SOURCE",
        "authorization": dict(nonlaunch),
        "bindings": {"runner_sha256": runner_sha},
    }
    release_value = {
        "status": "AUTHORIZE_ONE_M1480_EXACT_TYPE_CONFIG_COMPAT_M1458_ATTEMPT",
        "runner_sha256": runner_sha,
        "m1475_runner_sha256": M.M1475_SOURCE_SHA256,
        "result": str(M.M1475.M1458.CANONICAL_RESULT.relative_to(M.ROOT)),
        "attempt": str(M.M1475.M1458.CANONICAL_ATTEMPT.relative_to(M.ROOT)),
        "log": str(M.M1475.M1458.CANONICAL_LOG.relative_to(M.ROOT)),
        "authorization": dict(launch),
    }
    release.write_text(json.dumps(release_value), encoding="utf-8")
    final = {
        "status": "PASS_M1483_M1480_EXACT_TYPE_CONFIG_COMPAT_FINAL_LAUNCH",
        "authorization": dict(launch),
        "bindings": {"release_sha256": sha(release)},
    }
    values = {name: "0" * 64 for name in M.ENV_BINDINGS}
    return release_value, blind, final, values


def validate_synthetic(release: Path, release_value: dict[str, Any],
                       blind: dict[str, Any], final: dict[str, Any],
                       values: dict[str, str]) -> None:
    release.write_text(json.dumps(release_value), encoding="utf-8")
    final["bindings"]["release_sha256"] = sha(release)
    with mock.patch.object(M, "RELEASE", release), mock.patch.object(
            M.M1475.M1458, "verify_double_seal", side_effect=[blind, final]):
        M.validate_future_authorities(values)


def main() -> int:
    checks: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []

    def check(name: str, passed: bool, category: str) -> None:
        checks.append({"check": name, "category": category, "pass": bool(passed)})

    def attack(name: str, thunk, category: str) -> None:
        caught = rejected(thunk)
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    for path, expected in EXPECTED.items():
        check("sha_" + path.name, path.is_file() and sha(path) == expected, "identity")

    policy = M.strict_json(CONTRACT)
    check("contract_status", policy.get("status") ==
          "SOURCE_ONLY__M1475_EXACT_TYPE_SUCCESSOR__M1481_REQUIRED__NO_LAUNCH",
          "identity")
    check("contract_source_identity", policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)}, "identity")
    check("contract_test_identity", policy.get("test") == {
        "path": str(TEST.relative_to(ROOT)), "sha256": sha(TEST)}, "identity")
    check("contract_no_launch", policy.get("claim_boundary", {}).get("launch") is False,
          "authority")
    check("future_release_absent", not os.path.lexists(str(M.RELEASE)), "freshness")
    check("future_final_absent", not os.path.lexists(str(M.FINAL)), "freshness")

    test_run = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "pytest", "-q",
         str(M1475_TEST), str(TEST)], cwd=ROOT, text=True, capture_output=True,
        check=False)
    check("author_tests_26", test_run.returncode == 0 and
          "26 passed" in test_run.stdout, "tests")

    m1476 = M.M1475.M1458.verify_double_seal(
        M1476, EXPECTED[M1476 / "review.json"], EXPECTED[M1476 / "SHA256SUMS"],
        EXPECTED[M1476 / "SHA256SUMS.seal.sha256"])
    check("m1476_exact_failure_pin", m1476.get("status") ==
          "FAIL_DO_NOT_CITE__M1475_FINAL_AUTHORITY_TYPE_CONFUSION" and
          m1476.get("p0_count") == 1 and
          m1476.get("authorization", {}).get("launch") is False and
          m1476.get("authorization", {}).get("remote_preflight") is False,
          "predecessor_failure")

    check("m1458_result_namespace_unchanged",
          M.M1475.M1458.CANONICAL_RESULT.name.startswith("m1458_m1434_"),
          "namespace")
    check("m1458_attempt_namespace_unchanged",
          M.M1475.M1458.CANONICAL_ATTEMPT.name.startswith(".m1458_m1434_"),
          "namespace")
    check("m1458_log_namespace_unchanged",
          M.M1475.M1458.CANONICAL_LOG.name.startswith(".m1458_m1434_"),
          "namespace")

    payload = b"m1481-config-identity" * 311
    with tempfile.TemporaryDirectory(prefix="m1481_cfg_") as raw:
        path = Path(raw) / "config.yml"
        path.write_bytes(payload)
        entity = selected_entity(path, payload)
        with mock.patch.multiple(
                M.M1475, CONFIG_PATH=path, CONFIG_ABSOLUTE=str(path),
                CONFIG_SIZE=len(payload), CONFIG_SHA256=hashlib.sha256(payload).hexdigest(),
                FROZEN_CONFIG_ENTITY=dict(entity)):
            observed = M.M1475.verify_configuration_content_identity(dict(entity))
            check("m1475_recreated_entity_compatibility_unchanged",
                  observed == entity, "compatibility")
            changed = dict(entity); changed["inode"] = int(entity["inode"]) + 1
            attack("m1475_selection_inode_still_exact", lambda:
                   M.M1475.verify_configuration_content_identity(changed), "compatibility")
            path.write_bytes(payload + b"drift")
            attack("m1475_observed_content_still_exact", lambda:
                   M.M1475.verify_configuration_content_identity(entity), "compatibility")
            path.write_bytes(payload)

            calls: list[str] = []
            def strict_original(value, label):
                calls.append(label)
                raise M.M1475.M1475Error("original verifier " + label)
            with mock.patch.object(M.M1475, "ORIGINAL_EXACT_EXTENDED_IDENTITY",
                                   strict_original), mock.patch.object(
                                       M.M1475.M1319, "exact_extended_identity",
                                       strict_original):
                with M.M1475.configuration_content_compatibility():
                    check("selected_configuration_only_compatibility",
                          M.M1475.M1319.exact_extended_identity(
                              entity, "selected configuration") == entity,
                          "compatibility")
                    attack("checkpoint_uses_original", lambda:
                           M.M1475.M1319.exact_extended_identity(
                               entity, "selected checkpoint"), "identity")
                    attack("profile_uses_original", lambda:
                           M.M1475.M1319.exact_extended_identity(
                               entity, "selected profile"), "identity")
                check("compatibility_context_restored",
                      M.M1475.M1319.exact_extended_identity is strict_original,
                      "compatibility")
            check("checkpoint_profile_original_labels_observed",
                  calls == ["selected checkpoint", "selected profile"], "identity")

    exact_launch = {
        "launch": True, "runs": 1, "automatic_retry": False,
        "controller_restore": False,
    }
    exact_nonlaunch = {
        "launch": False, "runs": 0, "automatic_retry": False,
        "controller_restore": False,
    }
    M.exact_authorization(dict(exact_launch), True)
    M.exact_authorization(dict(exact_nonlaunch), False)
    check("exact_launch_and_nonlaunch_accept", True, "authority")

    old_false_negatives = (
        ("m1476_launch_int_1", "launch", 1),
        ("m1476_runs_bool_true", "runs", True),
        ("m1476_runs_float_1", "runs", 1.0),
        ("m1476_retry_int_0", "automatic_retry", 0),
        ("m1476_restore_int_0", "controller_restore", 0),
    )
    for name, field, value in old_false_negatives:
        changed = dict(exact_launch); changed[field] = value
        attack(name, lambda changed=changed: M.exact_authorization(changed, True),
               "m1476_regression")

    for field in exact_launch:
        changed = dict(exact_launch); changed[field] = "1" if field in {
            "launch", "runs"} else "false"
        attack("string_" + field, lambda changed=changed:
               M.exact_authorization(changed, True), "string_type")
        changed = dict(exact_launch); changed.pop(field)
        attack("missing_" + field, lambda changed=changed:
               M.exact_authorization(changed, True), "shape")
    changed = dict(exact_launch); changed["extra"] = False
    attack("extra_authorization_field", lambda:
           M.exact_authorization(changed, True), "shape")
    for malformed in (None, [], "authorization", 1, True):
        attack("nonmapping_" + type(malformed).__name__, lambda malformed=malformed:
               M.exact_authorization(malformed, True), "shape")

    with tempfile.TemporaryDirectory(prefix="m1481_auth_") as raw:
        release = Path(raw) / "release.json"
        release_value, blind, final, values = valid_authorities(release)
        validate_synthetic(release, json.loads(json.dumps(release_value)),
                           json.loads(json.dumps(blind)),
                           json.loads(json.dumps(final)), values)
        check("synthetic_exact_three_authorities_accept", True, "authority")

        for where in ("blind", "release", "final"):
            for name, field, value in old_false_negatives:
                cr = json.loads(json.dumps(release_value))
                cb = json.loads(json.dumps(blind))
                cf = json.loads(json.dumps(final))
                target = cb if where == "blind" else cr if where == "release" else cf
                target["authorization"][field] = value
                attack(where + "_" + name, lambda cr=cr, cb=cb, cf=cf:
                       validate_synthetic(release, cr, cb, cf, values),
                       "three_authority_exact_type")

        for where in ("blind", "release", "final"):
            for field in exact_launch:
                cr = json.loads(json.dumps(release_value))
                cb = json.loads(json.dumps(blind))
                cf = json.loads(json.dumps(final))
                target = cb if where == "blind" else cr if where == "release" else cf
                target["authorization"].pop(field)
                attack(where + "_missing_" + field, lambda cr=cr, cb=cb, cf=cf:
                       validate_synthetic(release, cr, cb, cf, values),
                       "three_authority_shape")
            cr = json.loads(json.dumps(release_value))
            cb = json.loads(json.dumps(blind))
            cf = json.loads(json.dumps(final))
            target = cb if where == "blind" else cr if where == "release" else cf
            target["authorization"]["extra"] = False
            attack(where + "_extra_authorization", lambda cr=cr, cb=cb, cf=cf:
                   validate_synthetic(release, cr, cb, cf, values),
                   "three_authority_shape")

        for field in ("result", "attempt", "log"):
            cr = json.loads(json.dumps(release_value)); cr[field] += ".replace"
            attack("m1458_" + field + "_replacement", lambda cr=cr:
                   validate_synthetic(release, cr, json.loads(json.dumps(blind)),
                                      json.loads(json.dumps(final)), values), "namespace")

    false_negatives = sum(bool(row["false_negative"]) for row in attacks)
    failed_checks = sum(not bool(row["pass"]) for row in checks)
    categories: dict[str, dict[str, int]] = {}
    for row in attacks:
        item = categories.setdefault(str(row["category"]),
                                     {"attacks": 0, "rejected": 0,
                                      "false_negatives": 0})
        item["attacks"] += 1
        item["rejected"] += int(bool(row["rejected"]))
        item["false_negatives"] += int(bool(row["false_negative"]))
    result = {
        "schema": "m1481_m1480_exact_type_config_compat_blind_hammer_output_r1_v1",
        "check_count": len(checks), "failed_checks": failed_checks,
        "attack_count": len(attacks), "false_negatives": false_negatives,
        "checks": checks, "attacks": attacks, "attack_categories": categories,
        "execution": {"ssh": 0, "remote_runs": 0, "real_gpu_queries": 0,
                      "capture_runs": 0, "production_attempts_consumed": 0,
                      "controller_signals": 0, "controller_restores": 0,
                      "eda_runs": 0},
        "verdict": ("PASS" if failed_checks == 0 and false_negatives == 0 else
                    "FAIL_DO_NOT_CITE"),
    }
    (REVIEW / "hammer_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("check_count", "failed_checks", "attack_count",
                       "false_negatives", "verdict")}, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
