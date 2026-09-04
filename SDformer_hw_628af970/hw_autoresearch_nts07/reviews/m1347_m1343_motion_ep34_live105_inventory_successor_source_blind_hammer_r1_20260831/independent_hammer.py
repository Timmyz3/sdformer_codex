#!/usr/bin/env python3
"""Different-author, source-only hammer for M1343.  Never runs a model/GPU."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1343_motion_ep34_live105_inventory_successor_r1.py")
TEST = ROOT / "hw_autoresearch_nts07/tests/test_m1343_motion_ep34_live105_inventory_successor.py"
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1343_motion_ep34_live105_inventory_successor_source_contract_r1_20260831.json")
REMOTE = HERE / "remote_cpu_inventory.json"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
AUTHOR = ROOT / (
    "hw_autoresearch_nts07/reviews/"
    "m1343_motion_ep34_live105_inventory_successor_source_author_r1_20260831")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m1347_target_m1343", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inventory(module):
    result = {}
    for category, count in module.R1.EXPECTED_STATIC_COUNTS.items():
        stem = "atlif" if category == "atlif" else category
        result[category] = [f"unit.{stem}.{index:03d}" for index in range(count)]
    return result


def verify_author_seal() -> bool:
    for line in (AUTHOR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if sha(ROOT / name) != digest:
            return False
    seal_digest, seal_name = (AUTHOR / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").strip().split("  ", 1)
    return seal_name == "SHA256SUMS" and sha(AUTHOR / seal_name) == seal_digest


def main() -> int:
    module = load_source()
    remote = json.loads(REMOTE.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source_text = SOURCE.read_text(encoding="utf-8")
    checks = {}

    checks["author_double_seal"] = verify_author_seal()
    checks["docs359_unchanged"] = sha(DOCS359) == module.DOCS359_SHA256
    checks["m1329_failure_forensic"] = True
    try:
        module.verify_m1329_failure()
    except Exception:
        checks["m1329_failure_forensic"] = False

    author_tests = subprocess.run(
        [sys.executable, str(TEST)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    checks["author_tests_12_of_12"] = (
        author_tests.returncode == 0 and "Ran 12 tests" in author_tests.stdout)

    self_check = subprocess.run(
        [sys.executable, str(SOURCE), "--source-self-check"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    checks["self_check_rejects_own_sealed_contract"] = (
        self_check.returncode != 0 and "M1343 test identity mismatch" in self_check.stdout)
    checks["self_check_root_cause_extra_test_keys"] = (
        set(contract["test"]) == {"path", "sha256", "passed", "failed"})

    checks["remote_identity_exact"] = (
        remote["identity"]["checkpoint_sha256"] == module.CHECKPOINT_SHA256 and
        remote["identity"]["config_sha256"] == module.CONFIG_SHA256 and
        remote["identity"]["profile_source_sha256"] ==
        sha(ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py") and
        remote["identity"]["atlif_overlay_source_sha256"] ==
        sha(ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py"))
    checks["remote_count105_snv0"] = (
        remote["inventory"]["atlif_count"] == 105 and
        remote["inventory"]["sn_v_count"] == 0 and
        remote["load_audit"] == {"missing": 0, "unexpected": 0})
    measured_names = remote["atlif_names"]
    measured_digest = hashlib.sha256(
        ("\n".join(measured_names) + "\n").encode()).hexdigest()
    checks["remote_complete_name_list_105_sorted_unique"] = (
        len(measured_names) == 105 and measured_names == sorted(measured_names) and
        len(set(measured_names)) == 105 and not any(".sn_v" in name for name in measured_names))
    checks["remote_name_list_recomputes_declared_digest"] = (
        measured_digest == remote["inventory"]["atlif_names_sha256"] ==
        "6a616f164625e3516bd2410f82d5f577c547c43a15b3bb2a5c4065add8a94cb7")
    checks["remote_name_digest_mismatch_detected"] = (
        remote["inventory"]["atlif_names_sha256"] != module.EXPECTED_ATLIF_NAMES_SHA256)

    base = inventory(module)
    old_digest = module.EXPECTED_ATLIF_NAMES_SHA256
    module.EXPECTED_ATLIF_NAMES_SHA256 = module.inventory_digest(base["atlif"])
    try:
        checks["canonical_order_is_explicitly_set_based"] = (
            module.expected_live105_inventory({k: list(reversed(v)) for k, v in base.items()})
            == {k: sorted(v) for k, v in base.items()})
        renamed = {k: list(v) for k, v in base.items()}
        renamed["atlif"][0] = "unit.atlif.renamed"
        try:
            module.expected_live105_inventory(renamed)
            checks["name_mutation_rejected"] = False
        except module.M1343Error:
            checks["name_mutation_rejected"] = True

        injected = {k: list(v) for k, v in base.items()}
        injected["atlif"][0] = "unit.attn.sn_v.spiking_neuron"
        module.EXPECTED_ATLIF_NAMES_SHA256 = module.inventory_digest(injected["atlif"])
        try:
            module.expected_live105_inventory(injected)
            checks["sn_v_injection_rejected_even_with_matching_digest"] = False
        except module.M1343Error:
            checks["sn_v_injection_rejected_even_with_matching_digest"] = True
    finally:
        module.EXPECTED_ATLIF_NAMES_SHA256 = old_digest

    before = (module.R1.DEAD_SN_V, module.R1.EXPECTED_LIVE_COUNTS,
              module.R1.expected_live_inventory, module.R1.validate_snapshot_population,
              module.R1.final_validate_and_seal, module.M1249.CANONICAL_RESULT)
    try:
        with module.patched_live105_capture_chain():
            raise RuntimeError("context-attack")
    except RuntimeError:
        pass
    after = (module.R1.DEAD_SN_V, module.R1.EXPECTED_LIVE_COUNTS,
             module.R1.expected_live_inventory, module.R1.validate_snapshot_population,
             module.R1.final_validate_and_seal, module.M1249.CANONICAL_RESULT)
    checks["context_exception_restores_all_six_globals"] = before == after

    with mock.patch.object(module.os.path, "lexists", return_value=True):
        try:
            module.require_fresh_namespaces()
            checks["namespace_collision_rejected"] = False
        except module.M1343Error:
            checks["namespace_collision_rejected"] = True
    checks["new_namespaces_do_not_reuse_m1327"] = (
        module.CANONICAL_RESULT != module.M1327.CANONICAL_RESULT and
        "m1343_" in module.CANONICAL_ATTEMPT.name and
        "m1343_" in module.CANONICAL_LOG.name)

    good_runtime = {"contract_path": "old", "capture": {"attention_windows_per_call": 100},
                    "cohort": {"samples": ["sealed"]}, "output": {"path": "old"}}
    bad_binding = {"identity": {"checkpoint_sha256": "0" * 64,
                                "config_sha256": module.CONFIG_SHA256}}
    with mock.patch.object(module.M1327, "validate_identity_and_project",
                           return_value=(good_runtime, bad_binding)):
        try:
            module.build_runtime()
            checks["checkpoint_binding_mutation_rejected"] = False
        except module.M1343Error:
            checks["checkpoint_binding_mutation_rejected"] = True

    checks["source_only_no_production_cli"] = (
        "--run" not in source_text and "O_CREAT | os.O_EXCL" not in source_text and
        contract["production_authorized"] is False)
    checks["patch_surface_does_not_replace_payload_attention_or_cohort"] = (
        "M1249.CANONICAL_RESULT = CANONICAL_RESULT" in source_text and
        "M1249.CANONICAL_ATTEMPT" not in source_text and
        "M1249.CANONICAL_LOG" not in source_text and
        "runtime[\"capture\"] == {\"attention_windows_per_call\": 100}" in source_text)

    all_hammer_checks = all(checks.values())
    finding = {
        "schema": "m1347_m1343_blind_hammer_output_r1_v1",
        "verdict": "FAIL_SOURCE__DO_NOT_AUTHORIZE_RELEASE",
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "all_hammer_checks": all_hammer_checks,
        "checks": checks,
        "p0": [
            {
                "id": "P0_REAL_ATLIF_NAME_DIGEST_MISMATCH",
                "expected": module.EXPECTED_ATLIF_NAMES_SHA256,
                "observed_remote_cpu": remote["inventory"]["atlif_names_sha256"],
                "impact": "real ep34 writer attach is fail-closed before capture"
            },
            {
                "id": "P0_SOURCE_SELF_CHECK_CANNOT_PASS",
                "observed": "M1343 test identity mismatch",
                "impact": "sealed source contract cannot authorize even source self-check"
            }
        ],
        "author_tests": {
            "returncode": author_tests.returncode,
            "passed": 12,
            "limitation": "synthetic ATLIF inventory and monkey-patched expected digest"
        },
        "remote_execution": remote["execution"],
        "production_authorized": False
    }
    print(json.dumps(finding, indent=2, sort_keys=True))
    return 0 if all_hammer_checks else 2


if __name__ == "__main__":
    raise SystemExit(main())
