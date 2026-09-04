#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent no-GPU/no-EDA source+result hammer for M1501/M1458."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import unittest
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / (
    "scripts/hammer_m1501_m1458_motion_ep34_live93_capture_result_"
    "safe_audit_source.py")
TEST = HW / (
    "tests/test_hammer_m1501_m1458_motion_ep34_live93_capture_result_"
    "safe_audit_source.py")
CONTRACT = HW / (
    "contracts/m1501_m1458_motion_ep34_live93_capture_result_safe_"
    "audit_source_contract_r1_20260831.json")
RESULT = HW / (
    "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_"
    "s40_r1_20260831")
PRODUCTION_LOG = HW / (
    "results/.m1458_m1434_motion_ep34_live93_unified_hardware_capture_"
    "s40_r1_20260831.production.log")
PRODUCTION_ATTEMPT = HW / (
    "results/.m1458_m1434_motion_ep34_live93_unified_hardware_capture_"
    "s40_r1_20260831.attempt_consumed")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    "source": "0c271bba3dfa57940b0ebe5a2ddf980d15f058b5ea25244aec5ead77d8146c83",
    "test": "0a0b2b5b58ccd8ae59f774b616a00510ffd99a636a794ad74f1dbb234c4f45b2",
    "contract": "e458cbe50c79a1faf659ed8329657978e6bcad7f0efb2fe91c3f016bc4a29dfb",
    "result_manifest": "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
    "result_outer": "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
    "checkpoint": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "config": "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39",
    "profile": "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_POPULATION = {
    "ordered": 9880, "payload": 640, "retained": 320,
    "attention": 480, "execution": 7360, "operator": 79,
    "atlif": 93, "forensic_snapshots": 40,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError("missing " + label) from error
    if (not stat.S_ISREG(mode) or path.is_symlink()
            or sha(path) != digest):
        raise RuntimeError(label + " identity drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise RuntimeError("duplicate JSON key")
            output[key] = value
        return output
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("JSON not regular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON")))
    if type(value) is not dict:
        raise RuntimeError("JSON root")
    return value


def load(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    regular_exact(path, digest, name + " after import")
    return module


def verify_top_seal() -> None:
    if not RESULT.is_dir() or RESULT.is_symlink():
        raise RuntimeError("canonical result root invalid")
    manifest = RESULT / "SHA256SUMS"
    outer = RESULT / "SHA256SUMS.seal.sha256"
    regular_exact(manifest, PINS["result_manifest"], "result manifest")
    regular_exact(outer, PINS["result_outer"], "result outer")
    if outer.read_text().split() != [PINS["result_manifest"], "SHA256SUMS"]:
        raise RuntimeError("result outer content drift")


def validate_output(value: dict[str, Any]) -> None:
    if value.get("status") != "PASS_M1501_M1458_EP34_LIVE93_CAPTURE_RESULT":
        raise RuntimeError("M1501 result status")
    if value.get("predecessor_status") != (
            "PASS_M1455_M1434_EP34_LIVE93_CAPTURE_RESULT"):
        raise RuntimeError("delegated result status")
    if value.get("population") != EXPECTED_POPULATION:
        raise RuntimeError("population drift")
    if value.get("identity") != {
            "checkpoint_sha256": PINS["checkpoint"],
            "config_sha256": PINS["config"],
            "profile_sha256": PINS["profile"]}:
        raise RuntimeError("result identity drift")
    if value.get("seal") != {
            "manifest_sha256": PINS["result_manifest"],
            "outer_file_sha256": PINS["result_outer"]}:
        raise RuntimeError("result seal drift")
    if value.get("audit_adapter", {}).get("all_mismatch_counts_zero") is not True:
        raise RuntimeError("checkpoint mismatch audit")
    if value.get("attention_adapter", {}).get("records") != 480:
        raise RuntimeError("attention adapter population")


def production_log_boundary() -> dict[str, Any]:
    if not os.path.lexists(PRODUCTION_LOG):
        return {
            "path": str(PRODUCTION_LOG.relative_to(ROOT)),
            "available_locally": False,
            "status_asserted": False,
            "status": "UNAVAILABLE_NOT_ASSERTED",
            "boundary": "Canonical result content is independently PASS, but the remote M1458 production log and attempt token were not transferred to this local workspace. M1512 does not infer or fabricate production-log PASS.",
            "local_attempt_token_available": os.path.lexists(PRODUCTION_ATTEMPT),
        }
    value = strict_json(PRODUCTION_LOG)
    if (value.get("schema") != "m1458_m1434_ep34_live93_production_log_r1_v1"
            or value.get("status") != "PASS"
            or value.get("automatic_retry") is not False
            or value.get("canonical_result_promotion_permitted") is not True):
        raise RuntimeError("local production log exists but is not exact PASS")
    return {"path": str(PRODUCTION_LOG.relative_to(ROOT)),
            "available_locally": True, "status_asserted": True,
            "status": "PASS", "sha256": sha(PRODUCTION_LOG),
            "local_attempt_token_available": os.path.lexists(PRODUCTION_ATTEMPT)}


def main() -> int:
    checks: list[dict[str, Any]] = []
    attacks: list[dict[str, Any]] = []
    def check(name: str, value: bool) -> None:
        checks.append({"check": name, "pass": bool(value)})
    def attack(name: str, value: dict[str, Any]) -> None:
        try:
            validate_output(value)
            rejected = False
        except BaseException:
            rejected = True
        attacks.append({"attack": name, "rejected": rejected,
                        "false_negative": not rejected})

    regular_exact(SOURCE, PINS["source"], "M1501 source")
    regular_exact(TEST, PINS["test"], "M1501 test")
    regular_exact(CONTRACT, PINS["contract"], "M1501 contract")
    regular_exact(DOCS359, PINS["docs359"], "docs359")
    verify_top_seal()
    check("exact_source_test_contract", True)
    check("exact_result_manifest_outer", True)

    M1501 = load("m1512_bound_m1501", SOURCE, PINS["source"])
    TEST_MODULE = load("m1512_bound_m1501_tests", TEST, PINS["test"])
    M1501.validate_source_policy()
    check("m1501_source_policy", True)

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(TEST_MODULE)
    replay = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    check("author_tests_17", replay.testsRun == 17 and
          not replay.failures and not replay.errors)

    result = M1501.validate_result(RESULT)
    validate_output(result)
    check("full_capture_validator", True)

    manifest = strict_json(RESULT / "manifest.json")
    selected = manifest["identity"]["selection"]["selected"]
    if {"checkpoint_sha256": selected["checkpoint"]["sha256"],
        "config_sha256": selected["configuration"]["sha256"],
        "profile_sha256": selected["profile"]["sha256"]} != result["identity"]:
        raise RuntimeError("selected identity/result mismatch")
    if manifest.get("status") != (
            "CAPTURE_COMPLETE__FRESH_M1434_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM"):
        raise RuntimeError("capture manifest status drift")
    check("selected_identity_and_manifest", True)

    mutations: dict[str, dict[str, Any]] = {}
    for key in EXPECTED_POPULATION:
        value = copy.deepcopy(result)
        value["population"][key] += 1
        mutations["population_" + key] = value
    for key in ("checkpoint_sha256", "config_sha256", "profile_sha256"):
        value = copy.deepcopy(result)
        value["identity"][key] = "0" * 64
        mutations["identity_" + key] = value
    for key in ("manifest_sha256", "outer_file_sha256"):
        value = copy.deepcopy(result)
        value["seal"][key] = "0" * 64
        mutations["seal_" + key] = value
    value = copy.deepcopy(result); value["status"] = "FAIL"
    mutations["status"] = value
    value = copy.deepcopy(result); value["predecessor_status"] = "FAIL"
    mutations["predecessor_status"] = value
    value = copy.deepcopy(result)
    value["audit_adapter"]["all_mismatch_counts_zero"] = False
    mutations["audit_mismatch"] = value
    value = copy.deepcopy(result); value["attention_adapter"]["records"] = 479
    mutations["attention_population"] = value
    for name, value in mutations.items():
        attack(name, value)

    log = production_log_boundary()
    check("production_log_boundary_explicit", True)
    p0 = sum(not item["rejected"] for item in attacks)
    p1 = sum(not item["pass"] for item in checks)
    output = {
        "schema": "m1512_m1501_m1458_ep34_capture_source_result_hammer_output_r1_v1",
        "status": "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT"
                  if p0 == 0 and p1 == 0 else "FAIL_CLOSED_DO_NOT_CITE",
        "checks": checks, "attacks": attacks,
        "summary": {"checks_passed": sum(item["pass"] for item in checks),
                    "checks_total": len(checks),
                    "attacks_rejected": sum(item["rejected"] for item in attacks),
                    "attacks_total": len(attacks),
                    "p0_count": p0, "p1_count": p1,
                    "author_tests_run": replay.testsRun,
                    "author_test_failures": len(replay.failures) + len(replay.errors)},
        "result": result,
        "production_log": log,
        "claim_boundary": {
            "capture_content_validated": True,
            "production_log_pass_validated": log["status_asserted"],
            "paper_result": False, "cycles": False, "speedup": False,
            "energy": False, "ppa": False, "system_speedup": False,
            "headline": False},
        "execution": {"remote": 0, "gpu": 0, "capture": 0,
                      "controller_signal": 0, "eda": 0},
    }
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0 if p0 == 0 and p1 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
