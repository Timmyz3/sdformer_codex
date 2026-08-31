#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local-only different-author blind hammer for M1475.

No SSH, real GPU access, capture, production attempt, controller operation, or
EDA operation is permitted by this program.  All file and authority mutations
use temporary local fixtures.
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
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
REVIEW = Path(__file__).resolve().parent
SOURCE = HW / "scripts/run_m1475_m1458_config_content_compat_one_shot.py"
TEST = HW / "tests/test_run_m1475_m1458_config_content_compat_one_shot.py"
CONTRACT = HW / (
    "contracts/m1475_m1458_config_content_compat_source_contract_r1_20260831.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1458_SOURCE = HW / "scripts/run_m1458_m1434_motion_ep34_live93_production_one_shot.py"
M1461 = HW / (
    "reviews/m1461_m1458_m1434_motion_ep34_live93_production_runner_source_"
    "blind_hammer_r1_20260831")
M1462 = HW / (
    "contracts/m1462_m1458_m1434_motion_ep34_live93_production_launch_"
    "release_r1_20260831.json")
M1463 = HW / (
    "reviews/m1463_m1462_m1458_m1434_motion_ep34_live93_production_final_"
    "launch_hammer_r1_20260831")

EXPECTED = {
    SOURCE: "2a5104b79e0d6563d8145a1e4ba136c9a2a047963d66d08a5d6b0bde93c5ac06",
    TEST: "25de303df2883dc450080d7f57c1f64047a349c32610f555cea990d5553ac10b",
    CONTRACT: "9cb1fd126621f85f7ab6ba4e7c960687ea19c49c85c901e8335a12108d4ab7b2",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1458_SOURCE: "e81c20056dd261619f88884f2f097c9b594887927121d9e599a4f89185d33154",
    M1461 / "review.json": "43f7a91567325570a30bc27eeda6516839691a5c1efd749185a086d36e2c4d58",
    M1461 / "SHA256SUMS": "6bbb45f9103e069e453ce212b7bdeba4e75e2624b7609df618acfea6d40aae0d",
    M1461 / "SHA256SUMS.seal.sha256": "60cba22e1f6de76ba93d3e1a5730314f413b4b81c3558f452d7a911f511c3343",
    M1462: "bd56146574ad5919f326dbe87ccb1dca5da9e06c7e6471412aeaa037a6d0c88f",
    M1463 / "review.json": "50af875678603940ff3789a516ab27aa1b89842f8d1a31b01c7320c442d2dcc4",
    M1463 / "SHA256SUMS": "bed9e82d88c097d4e1fff8f90f84c69a1acbd86044964205f8acaf5d6bac138e",
    M1463 / "SHA256SUMS.seal.sha256": "6effc85b4ca3350b907eb12ac083e62f9414aa6ecf30fcafb83ca4a76ad332cf",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load():
    spec = importlib.util.spec_from_file_location("m1476_bound_m1475", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M1475")
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


def config_scope(path: Path, payload: bytes):
    entity = selected_entity(path, payload)
    return mock.patch.multiple(
        M, CONFIG_PATH=path, CONFIG_ABSOLUTE=str(path), CONFIG_SIZE=len(payload),
        CONFIG_SHA256=hashlib.sha256(payload).hexdigest(),
        FROZEN_CONFIG_ENTITY=dict(entity)), entity


def valid_authorities(release: Path):
    runner_sha = sha(SOURCE)
    release_value = {
        "status": "AUTHORIZE_ONE_M1475_CONFIG_CONTENT_COMPAT_M1458_ATTEMPT",
        "runs": 1,
        "automatic_retry": False,
        "controller_restore": False,
        "runner_sha256": runner_sha,
        "m1458_runner_sha256": M.M1458_SOURCE_SHA256,
        "result": str(M.M1458.CANONICAL_RESULT.relative_to(M.ROOT)),
        "attempt": str(M.M1458.CANONICAL_ATTEMPT.relative_to(M.ROOT)),
        "log": str(M.M1458.CANONICAL_LOG.relative_to(M.ROOT)),
    }
    release.write_text(json.dumps(release_value), encoding="utf-8")
    blind = {
        "status": "PASS_M1475_CONFIG_CONTENT_COMPAT_SOURCE",
        "authorization": {"launch": False},
        "bindings": {"runner_sha256": runner_sha},
    }
    final = {
        "status": "PASS_M1478_M1475_CONFIG_CONTENT_COMPAT_FINAL_LAUNCH",
        "authorization": {
            "launch": True, "runs": 1, "automatic_retry": False,
            "controller_restore": False,
        },
        "bindings": {"release_sha256": sha(release)},
    }
    values = {name: "0" * 64 for name in M.ENV_BINDINGS}
    return release_value, blind, final, values


def validate_synthetic(release: Path, release_value, blind, final, values):
    release.write_text(json.dumps(release_value), encoding="utf-8")
    final["bindings"]["release_sha256"] = sha(release)
    with mock.patch.object(M, "RELEASE", release), mock.patch.object(
            M.M1458, "verify_double_seal", side_effect=[blind, final]):
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
          "SOURCE_ONLY__CONFIG_CONTENT_IDENTITY_COMPAT__M1476_REQUIRED__NO_LAUNCH",
          "identity")
    check("contract_no_launch", policy.get("claim_boundary", {}).get("launch") is False,
          "authority")
    check("future_release_absent", not os.path.lexists(str(M.RELEASE)), "freshness")
    check("future_final_absent", not os.path.lexists(str(M.FINAL)), "freshness")

    test_run = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "pytest", "-q", str(TEST)],
        cwd=ROOT, text=True, capture_output=True, check=False)
    check("author_tests_13", test_run.returncode == 0 and "13 passed" in test_run.stdout,
          "tests")

    payload = b"a" * 6481
    with tempfile.TemporaryDirectory(prefix="m1476_cfg_") as raw:
        root = Path(raw)
        path = root / "config.yml"
        path.write_bytes(payload)
        scope, entity = config_scope(path, payload)
        with scope:
            before = path.lstat()
            accepted = M.verify_configuration_content_identity(dict(entity))
            check("content_exact_accepts_recreated_entity_metadata",
                  accepted == entity and
                  (before.st_dev, before.st_ino, before.st_mode, before.st_mtime_ns) !=
                  (entity["device"], entity["inode"], entity["mode"], entity["mtime_ns"]),
                  "compatibility")
            for field, value in (
                    ("absolute_path", "/tmp/wrong"), ("size_bytes", 6480),
                    ("mtime_ns", 1), ("sha256", "0" * 64), ("device", 1),
                    ("inode", 1), ("mode", stat.S_IFREG | 0o644)):
                changed = dict(entity); changed[field] = value
                attack("selection_" + field, lambda changed=changed:
                       M.verify_configuration_content_identity(changed), "selection")

            path.write_bytes(b"b" * len(payload))
            attack("observed_sha", lambda: M.verify_configuration_content_identity(entity),
                   "observed")
            path.write_bytes(payload[:-1])
            attack("observed_size", lambda: M.verify_configuration_content_identity(entity),
                   "observed")
            path.write_bytes(payload)

            original_sha = M.sha256
            def racing_sha(item):
                value = original_sha(item)
                item.write_bytes(payload + b"race")
                return value
            with mock.patch.object(M, "sha256", side_effect=racing_sha):
                attack("observed_hash_race", lambda:
                       M.verify_configuration_content_identity(entity), "observed")
            path.write_bytes(payload)

        wrong = root / "wrong.yml"; wrong.write_bytes(payload)
        with mock.patch.multiple(M, CONFIG_PATH=wrong, CONFIG_ABSOLUTE=str(path),
                                 CONFIG_SIZE=len(payload),
                                 CONFIG_SHA256=hashlib.sha256(payload).hexdigest(),
                                 FROZEN_CONFIG_ENTITY=dict(entity)):
            attack("observed_path", lambda:
                   M.verify_configuration_content_identity(entity), "observed")

        link = root / "link.yml"; link.symlink_to(path)
        link_entity = selected_entity(link, payload)
        with mock.patch.multiple(M, CONFIG_PATH=link, CONFIG_ABSOLUTE=str(link),
                                 CONFIG_SIZE=len(payload),
                                 CONFIG_SHA256=hashlib.sha256(payload).hexdigest(),
                                 FROZEN_CONFIG_ENTITY=link_entity):
            attack("observed_symlink", lambda:
                   M.verify_configuration_content_identity(link_entity), "observed")

        directory = root / "directory"; directory.mkdir()
        dir_entity = selected_entity(directory, payload)
        with mock.patch.multiple(M, CONFIG_PATH=directory, CONFIG_ABSOLUTE=str(directory),
                                 CONFIG_SIZE=len(payload),
                                 CONFIG_SHA256=hashlib.sha256(payload).hexdigest(),
                                 FROZEN_CONFIG_ENTITY=dir_entity):
            attack("observed_nonregular", lambda:
                   M.verify_configuration_content_identity(dir_entity), "observed")

        path.write_bytes(payload)
        scope, entity = config_scope(path, payload)
        with scope:
            calls = []
            def strict_original(value, label):
                calls.append(label)
                raise M.M1475Error("strict original " + label)
            with mock.patch.object(M, "ORIGINAL_EXACT_EXTENDED_IDENTITY", strict_original), \
                    mock.patch.object(M.M1319, "exact_extended_identity", strict_original):
                with M.configuration_content_compatibility():
                    check("exact_label_selected_configuration",
                          M.M1319.exact_extended_identity(entity,
                              "selected configuration") == entity, "label")
                    for label in ("selected checkpoint", "selected profile",
                                  "configuration", "selected configuration ",
                                  "Selected configuration"):
                        attack("label_" + label.replace(" ", "_"), lambda label=label:
                               M.M1319.exact_extended_identity(entity, label), "label")
                check("context_restored_normal",
                      M.M1319.exact_extended_identity is strict_original, "context")
                try:
                    with M.configuration_content_compatibility():
                        raise RuntimeError("synthetic")
                except RuntimeError:
                    pass
                check("context_restored_exception",
                      M.M1319.exact_extended_identity is strict_original, "context")
            check("checkpoint_profile_used_original",
                  "selected checkpoint" in calls and "selected profile" in calls,
                  "label")

    with tempfile.TemporaryDirectory(prefix="m1476_auth_") as raw:
        release = Path(raw) / "release.json"
        release_value, blind, final, values = valid_authorities(release)
        validate_synthetic(release, dict(release_value), dict(blind),
                           json.loads(json.dumps(final)), values)
        check("synthetic_exact_authority_accepts", True, "authority")

        for field in ("result", "attempt", "log"):
            changed = dict(release_value); changed[field] = changed[field] + ".replace"
            attack("release_" + field + "_replacement", lambda changed=changed:
                   validate_synthetic(release, changed, dict(blind),
                                      json.loads(json.dumps(final)), values), "namespace")

        authority_mutations = (
            ("release_runs_bool", "release", "runs", True),
            ("release_runs_float", "release", "runs", 1.0),
            ("release_retry_int", "release", "automatic_retry", 0),
            ("release_restore_int", "release", "controller_restore", 0),
            ("blind_launch_int", "blind", "launch", 0),
            ("final_launch_int", "final", "launch", 1),
            ("final_runs_bool", "final", "runs", True),
            ("final_runs_float", "final", "runs", 1.0),
            ("final_retry_int", "final", "automatic_retry", 0),
            ("final_restore_int", "final", "controller_restore", 0),
        )
        for name, where, field, value in authority_mutations:
            changed_release = dict(release_value)
            changed_blind = json.loads(json.dumps(blind))
            changed_final = json.loads(json.dumps(final))
            if where == "release": changed_release[field] = value
            elif where == "blind": changed_blind["authorization"][field] = value
            else: changed_final["authorization"][field] = value
            attack(name, lambda cr=changed_release, cb=changed_blind, cf=changed_final:
                   validate_synthetic(release, cr, cb, cf, values), "authority")

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
        "schema": "m1476_m1475_config_content_compat_blind_hammer_output_r1_v1",
        "check_count": len(checks), "failed_checks": failed_checks,
        "attack_count": len(attacks), "false_negatives": false_negatives,
        "checks": checks, "attacks": attacks, "attack_categories": categories,
        "execution": {"ssh": 0, "real_gpu": 0, "capture": 0,
                      "production_attempt": 0, "controller_operation": 0,
                      "eda": 0},
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
