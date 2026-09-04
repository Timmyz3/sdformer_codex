#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, local-only M1483 final launch hammer.

The hammer never invokes the remote preflight or execution entry points.  It
uses local temporary regular files and double-sealed authority fixtures to
exercise M1480's native validation functions.  SSH, GPU access, capture,
attempt creation, controller signalling/restoration, and EDA are forbidden.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
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
RELEASE = HW / (
    "contracts/m1482_m1480_m1475_exact_type_config_compat_launch_release_r1_20260831.json")
M1481 = HW / (
    "reviews/m1481_m1480_m1475_exact_type_config_compat_source_blind_hammer_"
    "r1_20260831")
M1476 = HW / (
    "reviews/m1476_m1475_m1458_config_content_compat_source_blind_hammer_"
    "r1_20260831")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "3a0235f91d8d6acd4c94168b3b611cb53504f50e3843580c09bc1673042df4ce",
    TEST: "dea2bc2cb3851a40462f5200b423c623331aa20abc054debc8e2ea661fc99ea3",
    CONTRACT: "c4ec0a4792a7647c46614652147de6999d2dce0c6c55d5d46a88798e12ad90e4",
    RELEASE: "5f458009e15e759e29b54d9306ade72ba74cd927bc62e0cf1c4ca49513fb1697",
    M1481 / "review.json": "5e804c304a827d1b89af284828c2b68c8dadfdd7c687958fc4a6fbc3e8127f96",
    M1481 / "SHA256SUMS": "92698ce0f0376bda9be7348d79e8ab0c3266b5d3df7b2465424bca82192c635c",
    M1481 / "SHA256SUMS.seal.sha256": "702a4a00f8ae00e621b417b4f07a0068cc55f2deed71586d9d2cd4293804e2cf",
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
    spec = importlib.util.spec_from_file_location("m1483_bound_m1480", SOURCE)
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def seal_review(directory: Path, value: dict[str, Any]) -> tuple[str, str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    review = directory / "review.json"
    write_json(review, value)
    review_sha = sha(review)
    manifest = directory / "SHA256SUMS"
    manifest.write_text(f"{review_sha}  review.json\n", encoding="utf-8")
    manifest_sha = sha(manifest)
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{manifest_sha}  SHA256SUMS\n", encoding="utf-8")
    return review_sha, manifest_sha, sha(outer)


def final_value(release: Path) -> dict[str, Any]:
    return {
        "status": "PASS_M1483_M1480_EXACT_TYPE_CONFIG_COMPAT_FINAL_LAUNCH",
        "bindings": {"release_sha256": sha(release)},
        "authorization": {
            "launch": True, "runs": 1, "automatic_retry": False,
            "controller_restore": False,
        },
    }


def values_for(final_dir: Path, release: Path) -> dict[str, str]:
    return {
        "M1480_EXPECTED_RUNNER_SHA256": sha(SOURCE),
        "M1480_EXPECTED_BLIND_REVIEW_SHA256": sha(M1481 / "review.json"),
        "M1480_EXPECTED_BLIND_MANIFEST_SHA256": sha(M1481 / "SHA256SUMS"),
        "M1480_EXPECTED_BLIND_OUTER_SHA256": sha(
            M1481 / "SHA256SUMS.seal.sha256"),
        "M1480_EXPECTED_RELEASE_SHA256": sha(release),
        "M1480_EXPECTED_FINAL_REVIEW_SHA256": sha(final_dir / "review.json"),
        "M1480_EXPECTED_FINAL_MANIFEST_SHA256": sha(final_dir / "SHA256SUMS"),
        "M1480_EXPECTED_FINAL_OUTER_SHA256": sha(
            final_dir / "SHA256SUMS.seal.sha256"),
    }


def native_validate(release: Path, final_dir: Path) -> None:
    values = values_for(final_dir, release)
    with mock.patch.object(M, "RELEASE", release), mock.patch.object(
            M, "FINAL", final_dir):
        M.validate_future_authorities(values)


def mutate_release(base: dict[str, Any], final_root: Path, name: str,
                   mutate) -> None:
    release = final_root / (name + ".json")
    value = copy.deepcopy(base)
    mutate(value)
    write_json(release, value)
    final_dir = final_root / (name + "_final")
    seal_review(final_dir, final_value(release))
    native_validate(release, final_dir)


def selected_entity(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "absolute_path": str(path), "size_bytes": len(payload),
        "mtime_ns": 1788081356000000000,
        "sha256": hashlib.sha256(payload).hexdigest(), "device": 194,
        "inode": 26561699333, "mode": 33152,
    }


class IntSubclass(int):
    pass


class DictSubclass(dict):
    pass


def main() -> int:
    checks: list[dict[str, Any]] = []
    attacks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, category: str) -> None:
        checks.append({"check": name, "category": category, "pass": bool(passed)})

    def attack(name: str, thunk, category: str) -> None:
        caught = rejected(thunk)
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    # Immutable identities and native author tests.
    for path, expected in EXPECTED.items():
        check("sha_" + path.name + "_" + str(len(checks)),
              path.is_file() and not path.is_symlink() and sha(path) == expected,
              "identity")
    tests = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "pytest", "-q",
         str(M1475_TEST), str(TEST)], cwd=ROOT, text=True, capture_output=True,
        check=False)
    check("native_author_tests_26", tests.returncode == 0 and
          "26 passed" in tests.stdout, "native_tests")
    M.validate_source_contract()
    check("native_source_contract_validation", True, "native_validation")

    release_value = M.strict_json(RELEASE)
    check("release_strict_json_root", type(release_value) is dict, "release")
    check("release_exact_sha", sha(RELEASE) == EXPECTED[RELEASE], "release")
    check("release_final_absent_at_authoring",
          release_value.get("final_gate", {}).get("present_at_release_authoring") is False,
          "freshness")
    check("release_no_execution", release_value.get("release_author_execution") == {
        "ssh": False, "remote_preflight": False, "remote_runs": 0,
        "real_gpu_queries": 0, "capture_runs": 0,
        "production_attempts_consumed": 0, "controller_signals": 0,
        "controller_restores": 0}, "release")

    # Verify the predecessor authorities natively and test unsealing.
    blind = M.M1475.M1458.verify_double_seal(
        M1481, EXPECTED[M1481 / "review.json"], EXPECTED[M1481 / "SHA256SUMS"],
        EXPECTED[M1481 / "SHA256SUMS.seal.sha256"])
    check("native_m1481_double_seal", blind.get("status") ==
          "PASS_M1480_EXACT_TYPE_CONFIG_COMPAT_SOURCE", "native_validation")
    failed = M.M1475.M1458.verify_double_seal(
        M1476, EXPECTED[M1476 / "review.json"], EXPECTED[M1476 / "SHA256SUMS"],
        EXPECTED[M1476 / "SHA256SUMS.seal.sha256"])
    check("native_m1476_failure_pin", failed.get("status") ==
          "FAIL_DO_NOT_CITE__M1475_FINAL_AUTHORITY_TYPE_CONFUSION" and
          failed.get("p0_count") == 1, "native_validation")

    with tempfile.TemporaryDirectory(prefix=".m1483_seal_attacks_",
                                     dir=HW / "results") as raw:
        temp = Path(raw)
        for source_dir, label, expected_review, expected_manifest, expected_outer in (
            (M1476, "m1476", EXPECTED[M1476 / "review.json"],
             EXPECTED[M1476 / "SHA256SUMS"],
             EXPECTED[M1476 / "SHA256SUMS.seal.sha256"]),
            (M1481, "m1481", EXPECTED[M1481 / "review.json"],
             EXPECTED[M1481 / "SHA256SUMS"],
             EXPECTED[M1481 / "SHA256SUMS.seal.sha256"]),
        ):
            for member in ("review.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"):
                target = temp / (label + "_" + member.replace(".", "_"))
                shutil.copytree(source_dir, target)
                with (target / member).open("ab") as stream:
                    stream.write(b"M1483_MUTATION")
                attack(label + "_unseal_" + member, lambda target=target,
                       er=expected_review, em=expected_manifest, eo=expected_outer:
                       M.M1475.M1458.verify_double_seal(target, er, em, eo),
                       "sealed_predecessor")

    # Exact Python scalar and mapping types, including subclass confusion.
    exact_launch = {"launch": True, "runs": 1, "automatic_retry": False,
                    "controller_restore": False}
    exact_nonlaunch = {"launch": False, "runs": 0, "automatic_retry": False,
                       "controller_restore": False}
    M.exact_authorization(exact_launch, True)
    M.exact_authorization(exact_nonlaunch, False)
    check("exact_authorizations_accept", True, "authorization")
    scalar_attacks = (
        ("launch_int", "launch", 1), ("launch_float", "launch", 1.0),
        ("runs_bool", "runs", True), ("runs_float", "runs", 1.0),
        ("runs_int_subclass", "runs", IntSubclass(1)),
        ("retry_int", "automatic_retry", 0),
        ("retry_float", "automatic_retry", 0.0),
        ("restore_int", "controller_restore", 0),
        ("restore_float", "controller_restore", 0.0),
    )
    for name, field, item in scalar_attacks:
        changed = dict(exact_launch); changed[field] = item
        attack("direct_" + name, lambda changed=changed:
               M.exact_authorization(changed, True), "exact_type")
    attack("authorization_dict_subclass", lambda:
           M.exact_authorization(DictSubclass(exact_launch), True), "exact_type")
    for field in exact_launch:
        changed = dict(exact_launch); changed.pop(field)
        attack("direct_missing_" + field, lambda changed=changed:
               M.exact_authorization(changed, True), "authorization_shape")
    changed = dict(exact_launch); changed["extra"] = False
    attack("direct_extra_field", lambda:
           M.exact_authorization(changed, True), "authorization_shape")
    for item in (None, [], (), "authority", 1, True, 1.0):
        attack("direct_nonmapping_" + type(item).__name__, lambda item=item:
               M.exact_authorization(item, True), "authorization_shape")

    # Native M1480 future-authority path with real M1481 + real M1482 and a
    # temporary exact M1483 seal.  Remote/capture functions are hard-bombed.
    with tempfile.TemporaryDirectory(prefix=".m1483_native_",
                                     dir=HW / "results") as raw:
        temp = Path(raw)
        valid_final = temp / "valid_final"
        seal_review(valid_final, final_value(RELEASE))
        with mock.patch.object(M.M1475.M1458, "remote_preflight",
                               side_effect=AssertionError("remote forbidden")), \
             mock.patch.object(M.M1475.M1458, "execute_once",
                               side_effect=AssertionError("capture forbidden")):
            native_validate(RELEASE, valid_final)
        check("native_real_m1481_real_m1482_future_authority_validation", True,
              "native_validation")

        # External SHA binding must fail closed.
        values = values_for(valid_final, RELEASE)
        for field in values:
            altered = dict(values); altered[field] = "0" * 64
            attack("external_sha_" + field, lambda altered=altered:
                   M.external_bindings(altered), "external_binding")
        malformed = dict(values); malformed["M1480_EXPECTED_RELEASE_SHA256"] = "ABC"
        attack("external_sha_malformed", lambda:
               M.external_bindings(malformed), "external_binding")

        # Every semantic M1482 field and M1458 namespace is re-bound to a
        # fresh final seal so rejection cannot be attributed to a stale final
        # release digest.
        mutations = [
            ("release_status", lambda v: v.__setitem__("status", "PASS")),
            ("release_runner_sha", lambda v: v.__setitem__("runner_sha256", "0" * 64)),
            ("release_m1475_sha", lambda v: v.__setitem__("m1475_runner_sha256", "0" * 64)),
            ("result_namespace", lambda v: v.__setitem__("result", v["result"] + ".replace")),
            ("attempt_namespace", lambda v: v.__setitem__("attempt", v["attempt"] + ".replace")),
            ("log_namespace", lambda v: v.__setitem__("log", v["log"] + ".replace")),
            ("release_auth_launch_false", lambda v: v["authorization"].__setitem__("launch", False)),
            ("release_auth_launch_int", lambda v: v["authorization"].__setitem__("launch", 1)),
            ("release_auth_runs_zero", lambda v: v["authorization"].__setitem__("runs", 0)),
            ("release_auth_runs_bool", lambda v: v["authorization"].__setitem__("runs", True)),
            ("release_auth_runs_float", lambda v: v["authorization"].__setitem__("runs", 1.0)),
            ("release_auth_retry_true", lambda v: v["authorization"].__setitem__("automatic_retry", True)),
            ("release_auth_retry_int", lambda v: v["authorization"].__setitem__("automatic_retry", 0)),
            ("release_auth_restore_true", lambda v: v["authorization"].__setitem__("controller_restore", True)),
            ("release_auth_restore_int", lambda v: v["authorization"].__setitem__("controller_restore", 0)),
            ("release_auth_extra", lambda v: v["authorization"].__setitem__("extra", False)),
            ("release_auth_missing", lambda v: v["authorization"].pop("runs")),
        ]
        for name, mutate in mutations:
            attack(name, lambda name=name, mutate=mutate:
                   mutate_release(release_value, temp, name, mutate),
                   "release_semantics")

        # Semantic M1481 changes remain rejected even when freshly re-sealed
        # and passed with their new exact digest triplet.
        for name, mutate in (
            ("m1481_status", lambda v: v.__setitem__("status", "PASS")),
            ("m1481_runner_binding", lambda v: v["bindings"].__setitem__(
                "runner_sha256", "0" * 64)),
            ("m1481_launch_true", lambda v: v["authorization"].__setitem__(
                "launch", True)),
            ("m1481_runs_bool", lambda v: v["authorization"].__setitem__(
                "runs", False)),
            ("m1481_retry_int", lambda v: v["authorization"].__setitem__(
                "automatic_retry", 0)),
            ("m1481_restore_int", lambda v: v["authorization"].__setitem__(
                "controller_restore", 0)),
        ):
            candidate = temp / name
            value = M.strict_json(M1481 / "review.json")
            mutate(value)
            br, bm, bo = seal_review(candidate, value)
            vals = values_for(valid_final, RELEASE)
            vals["M1480_EXPECTED_BLIND_REVIEW_SHA256"] = br
            vals["M1480_EXPECTED_BLIND_MANIFEST_SHA256"] = bm
            vals["M1480_EXPECTED_BLIND_OUTER_SHA256"] = bo
            attack(name, lambda candidate=candidate, vals=vals:
                   _validate_with_blind(candidate, valid_final, vals),
                   "blind_semantics")

        # Semantic final authority attacks, each freshly double-sealed.
        for name, mutate in (
            ("final_status", lambda v: v.__setitem__("status", "PASS")),
            ("final_release_sha", lambda v: v["bindings"].__setitem__(
                "release_sha256", "0" * 64)),
            ("final_launch_false", lambda v: v["authorization"].__setitem__(
                "launch", False)),
            ("final_launch_int", lambda v: v["authorization"].__setitem__(
                "launch", 1)),
            ("final_runs_bool", lambda v: v["authorization"].__setitem__(
                "runs", True)),
            ("final_runs_float", lambda v: v["authorization"].__setitem__(
                "runs", 1.0)),
            ("final_retry_int", lambda v: v["authorization"].__setitem__(
                "automatic_retry", 0)),
            ("final_restore_int", lambda v: v["authorization"].__setitem__(
                "controller_restore", 0)),
            ("final_auth_extra", lambda v: v["authorization"].__setitem__(
                "extra", False)),
        ):
            candidate = temp / name
            value = final_value(RELEASE); mutate(value)
            seal_review(candidate, value)
            attack(name, lambda candidate=candidate:
                   native_validate(RELEASE, candidate), "final_semantics")

    # Configuration compatibility scope: exact frozen selection remains exact;
    # observed entity may differ only in inode/mode/mtime and must stay a stable
    # regular file with the fixed path, size and content SHA.
    payload = b"m1483 exact config payload\n" * 251
    with tempfile.TemporaryDirectory(prefix="m1483_config_") as raw:
        temp = Path(raw)
        path = temp / "config.yml"
        path.write_bytes(payload)
        entity = selected_entity(path, payload)
        patches = dict(CONFIG_PATH=path, CONFIG_ABSOLUTE=str(path),
                       CONFIG_SIZE=len(payload),
                       CONFIG_SHA256=hashlib.sha256(payload).hexdigest(),
                       FROZEN_CONFIG_ENTITY=dict(entity))
        with mock.patch.multiple(M.M1475, **patches):
            check("configuration_content_exact_accept",
                  M.M1475.verify_configuration_content_identity(dict(entity)) == entity,
                  "configuration")
            for name, field, item in (
                ("selection_path", "absolute_path", str(path) + ".other"),
                ("selection_size", "size_bytes", len(payload) + 1),
                ("selection_sha", "sha256", "0" * 64),
                ("selection_mode", "mode", 33188),
                ("selection_inode", "inode", 26561699334),
            ):
                changed = dict(entity); changed[field] = item
                attack("config_" + name, lambda changed=changed:
                       M.M1475.verify_configuration_content_identity(changed),
                       "configuration_selection")
            attack("config_selection_wrong_type", lambda:
                   M.M1475.verify_configuration_content_identity([]),
                   "configuration_selection")
            path.write_bytes(payload + b"x")
            attack("config_observed_size_and_sha", lambda:
                   M.M1475.verify_configuration_content_identity(entity),
                   "configuration_observation")
            path.write_bytes(payload)
            real_sha = M.M1475.sha256
            def unstable_sha(target: Path) -> str:
                result = real_sha(target)
                before = target.stat().st_mtime_ns
                os.utime(target, ns=(before + 1_000_000_000,
                                     before + 1_000_000_000))
                return result
            with mock.patch.object(M.M1475, "sha256", side_effect=unstable_sha):
                attack("config_changes_while_hashing", lambda:
                       M.M1475.verify_configuration_content_identity(entity),
                       "configuration_stability")
            path.write_bytes(payload)

        symlink = temp / "symlink.yml"
        symlink.symlink_to(path)
        symlink_entity = selected_entity(symlink, payload)
        with mock.patch.multiple(M.M1475, CONFIG_PATH=symlink,
                                 CONFIG_ABSOLUTE=str(symlink),
                                 CONFIG_SIZE=len(payload),
                                 CONFIG_SHA256=hashlib.sha256(payload).hexdigest(),
                                 FROZEN_CONFIG_ENTITY=symlink_entity):
            attack("config_symlink_type", lambda:
                   M.M1475.verify_configuration_content_identity(symlink_entity),
                   "configuration_type")
        directory = temp / "directory.yml"; directory.mkdir()
        directory_entity = selected_entity(directory, b"")
        with mock.patch.multiple(M.M1475, CONFIG_PATH=directory,
                                 CONFIG_ABSOLUTE=str(directory), CONFIG_SIZE=0,
                                 CONFIG_SHA256=hashlib.sha256(b"").hexdigest(),
                                 FROZEN_CONFIG_ENTITY=directory_entity):
            attack("config_directory_type", lambda:
                   M.M1475.verify_configuration_content_identity(directory_entity),
                   "configuration_type")

    # Scope and namespace invariants, with bomb sentinels proving the hammer
    # itself did not enter any remote/capture path.
    check("m1458_result_namespace_exact",
          str(M.M1475.M1458.CANONICAL_RESULT.relative_to(ROOT)) ==
          release_value["result"], "namespace")
    check("m1458_attempt_namespace_exact",
          str(M.M1475.M1458.CANONICAL_ATTEMPT.relative_to(ROOT)) ==
          release_value["attempt"], "namespace")
    check("m1458_log_namespace_exact",
          str(M.M1475.M1458.CANONICAL_LOG.relative_to(ROOT)) ==
          release_value["log"], "namespace")
    check("namespace_owner_m1458",
          release_value.get("one_shot_policy", {}).get("namespace_owner") == "M1458",
          "namespace")
    check("one_shot_exact",
          release_value.get("one_shot_policy", {}).get("attempt_create") == "O_EXCL" and
          release_value.get("one_shot_policy", {}).get("runs") == 1 and
          release_value.get("one_shot_policy", {}).get("automatic_retry") is False and
          release_value.get("one_shot_policy", {}).get("controller_restore") is False,
          "one_shot")

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
        "schema": "m1483_m1480_exact_type_config_compat_final_hammer_output_r1_v1",
        "check_count": len(checks), "failed_checks": failed_checks,
        "attack_count": len(attacks), "false_negatives": false_negatives,
        "checks": checks, "attacks": attacks, "attack_categories": categories,
        "execution": {"ssh": 0, "remote_preflight": 0, "remote_runs": 0,
                      "real_gpu_queries": 0, "capture_runs": 0,
                      "production_attempts_consumed": 0, "controller_signals": 0,
                      "controller_restores": 0, "eda_runs": 0},
        "verdict": "PASS" if failed_checks == 0 and false_negatives == 0 else
                   "FAIL_DO_NOT_CITE",
        "pytest_stdout": tests.stdout.strip(),
        "pytest_stderr": tests.stderr.strip(),
    }
    write_json(REVIEW / "hammer_output.json", result)
    print(json.dumps({key: result[key] for key in
                      ("check_count", "failed_checks", "attack_count",
                       "false_negatives", "verdict")}, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def _validate_with_blind(blind_dir: Path, final_dir: Path,
                         values: dict[str, str]) -> None:
    with mock.patch.object(M, "BLIND", blind_dir), mock.patch.object(
            M, "FINAL", final_dir):
        M.validate_future_authorities(values)


if __name__ == "__main__":
    raise SystemExit(main())
