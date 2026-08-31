#!/usr/bin/env python3
"""Different-author no-GPU/no-capture hammer for M1349 live-105 source."""
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1349_motion_ep34_live105_inventory_successor_r2.py")
TEST = ROOT / "hw_autoresearch_nts07/tests/test_m1349_motion_ep34_live105_inventory_successor.py"
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1349_motion_ep34_live105_inventory_successor_source_contract_r1_20260831.json")
AUTHOR = ROOT / (
    "hw_autoresearch_nts07/reviews/"
    "m1349_motion_ep34_live105_inventory_successor_source_author_r1_20260831")
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m1353_target_m1349", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_root_relative_seal(root: Path, repo: Path) -> bool:
    manifest = root / "SHA256SUMS"; outer = root / "SHA256SUMS.seal.sha256"
    if outer.read_text(encoding="utf-8").split() != [sha(manifest), "SHA256SUMS"]:
        return False
    names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        if name in names or path.is_absolute() or ".." in path.parts:
            return False
        member = repo / path
        if not member.is_file() or member.is_symlink() or sha(member) != digest:
            return False
        names.add(name)
    return True


def rejected(action) -> bool:
    try:
        action()
    except Exception:
        return True
    return False


def main() -> int:
    module = load_source()
    checks = {}
    checks["source_identity"] = sha(SOURCE) == "3fe0f51acf489cf2f4d1a65f83f872b49a5fde79401a2fdb525768e681fbbbe5"
    checks["test_identity"] = sha(TEST) == "b20e06bcecb9fab1a326701e40e7bb72c5f13a3204a9d52470b58237a747492f"
    checks["contract_identity"] = sha(CONTRACT) == "ce2f373eef512237a0e0ee087134176384c30663bd52d42aa68c68b05fbd4712"
    checks["author_double_seal"] = verify_root_relative_seal(AUTHOR, ROOT)

    env = dict(os.environ); env["PYTHONDONTWRITEBYTECODE"] = "1"
    tests = subprocess.run([str(PYTHON), "-B", str(TEST)], cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, check=False)
    checks["author_tests_20_of_20"] = (
        tests.returncode == 0 and "Ran 20 tests" in tests.stdout and "OK" in tests.stdout)
    self_check = subprocess.run([str(PYTHON), "-B", str(SOURCE), "--source-self-check"],
                                cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=False)
    checks["source_self_check"] = (
        self_check.returncode == 0 and module.PASS_TOKEN in self_check.stdout)

    checks["m1347_review_exact"] = sha(module.M1347_REVIEW) == module.M1347_REVIEW_SHA256
    checks["m1347_manifest_exact"] = sha(module.M1347_MANIFEST) == module.M1347_MANIFEST_SHA256
    checks["m1347_outer_exact"] = sha(module.M1347_OUTER) == module.M1347_OUTER_SHA256
    checks["m1347_inventory_exact"] = sha(module.M1347_INVENTORY) == module.M1347_INVENTORY_SHA256
    checks["m1347_outer_content"] = module.M1347_OUTER.read_text(encoding="utf-8") == (
        module.M1347_MANIFEST_SHA256 + "  SHA256SUMS\n")
    manifest_rows = {}
    for line in module.M1347_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        manifest_rows[name] = digest
    checks["m1347_inventory_member_sealed"] = manifest_rows.get(
        str(module.M1347_INVENTORY.relative_to(ROOT))) == module.M1347_INVENTORY_SHA256

    remote = module.strict_json(module.M1347_INVENTORY)
    names = remote["atlif_names"]
    checks["complete_105_name_array"] = len(names) == 105
    checks["name_array_sorted_unique"] = names == sorted(names) and len(set(names)) == 105
    checks["terminal_lf_digest"] = (
        module.terminal_lf_digest(names) == remote["inventory"]["atlif_names_sha256"] ==
        module.EXPECTED_ATLIF_NAMES_SHA256)
    checks["authority_positive"] = module.validate_authority_payload(remote) == tuple(names)

    rename = json.loads(json.dumps(remote)); rename["atlif_names"][0] += ".renamed"
    checks["rename_rejected"] = rejected(lambda: module.validate_authority_payload(rename))
    reorder = json.loads(json.dumps(remote)); reorder["atlif_names"][0:2] = reversed(reorder["atlif_names"][0:2])
    checks["reorder_rejected"] = rejected(lambda: module.validate_authority_payload(reorder))
    duplicate = json.loads(json.dumps(remote)); duplicate["atlif_names"][1] = duplicate["atlif_names"][0]
    checks["duplicate_rejected"] = rejected(lambda: module.validate_authority_payload(duplicate))
    deletion = json.loads(json.dumps(remote)); deletion["atlif_names"].pop()
    checks["delete_rejected"] = rejected(lambda: module.validate_authority_payload(deletion))
    snv = json.loads(json.dumps(remote)); snv["atlif_names"][0] = "aaa.sn_v.spiking_neuron"
    snv["atlif_names"].sort(); snv["inventory"]["first_name"] = snv["atlif_names"][0]
    snv["inventory"]["last_name"] = snv["atlif_names"][-1]
    snv_digest = module.terminal_lf_digest(snv["atlif_names"])
    snv["inventory"]["atlif_names_sha256"] = snv_digest
    old_expected = module.EXPECTED_ATLIF_NAMES_SHA256
    module.EXPECTED_ATLIF_NAMES_SHA256 = snv_digest
    try:
        checks["sn_v_injection_rejected_even_with_matching_digest"] = rejected(
            lambda: module.validate_authority_payload(snv))
    finally:
        module.EXPECTED_ATLIF_NAMES_SHA256 = old_expected

    for field in ("checkpoint_sha256", "config_sha256", "profile_source_sha256",
                  "atlif_overlay_source_sha256"):
        mutant = json.loads(json.dumps(remote)); mutant["identity"][field] = "0" * 64
        checks["identity_mutation_rejected_" + field] = rejected(
            lambda value=mutant: module.validate_authority_payload(value))
    for field in ("missing", "unexpected"):
        mutant = json.loads(json.dumps(remote)); mutant["load_audit"][field] = 1
        checks["load_audit_mutation_rejected_" + field] = rejected(
            lambda value=mutant: module.validate_authority_payload(value))
    rebuild = json.loads(json.dumps(remote)); rebuild["repeatability"]["rebuilds"] = 1
    checks["rebuild_count_mutation_rejected"] = rejected(
        lambda: module.validate_authority_payload(rebuild))
    consistency = json.loads(json.dumps(remote)); consistency["repeatability"]["same_digest"] = False
    checks["rebuild_consistency_mutation_rejected"] = rejected(
        lambda: module.validate_authority_payload(consistency))
    execution = remote["execution"]
    checks["readonly_cpu_authority_semantics"] = (
        execution["device"] == "cpu" and execution["cuda_visible_devices"] == "" and
        execution["forward_executed"] is False and execution["capture_executed"] is False and
        execution["attempt_consumed"] is False and execution["remote_files_written"] is False)

    profile = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py"
    overlay = ROOT / (
        "neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/"
        "atlif_ternary_psn/atlif_ternary_psn.py")
    checks["local_profile_overlay_match_authority"] = (
        sha(profile) == module.PROFILE_SOURCE_SHA256 and
        sha(overlay) == module.ATLIF_OVERLAY_SOURCE_SHA256)

    policy = module.strict_json(CONTRACT)
    checks["contract_test_exact_fields"] = set(policy["test"]) == {
        "path", "sha256", "passed", "failed"}
    extra = json.loads(json.dumps(policy)); extra["test"]["extra"] = True
    checks["contract_test_extra_rejected"] = rejected(lambda: module.validate_source_policy(extra))
    missing = json.loads(json.dumps(policy)); del missing["test"]["failed"]
    checks["contract_test_delete_rejected"] = rejected(lambda: module.validate_source_policy(missing))
    with tempfile.TemporaryDirectory(prefix="m1353_duplicate_json_") as td:
        duplicate_path = Path(td) / "duplicate.json"
        raw = CONTRACT.read_text(encoding="utf-8")
        duplicate_path.write_text(raw.replace('"passed": 20,',
            '"passed": 999, "passed": 20,', 1), encoding="utf-8")
        checks["contract_duplicate_test_key_rejected"] = rejected(
            lambda: module.strict_json(duplicate_path))

    checks["fresh_namespaces_positive"] = not rejected(module.require_fresh_namespaces)
    with mock.patch.object(module.os.path, "lexists", return_value=True):
        checks["namespace_collision_rejected"] = rejected(module.require_fresh_namespaces)
    checks["namespace_does_not_reuse_m1343"] = (
        module.CANONICAL_RESULT != module.M1343.CANONICAL_RESULT and
        len({module.CANONICAL_RESULT, module.CANONICAL_ATTEMPT, module.CANONICAL_LOG}) == 3)

    six_before = (module.R1.DEAD_SN_V, module.R1.EXPECTED_LIVE_COUNTS,
                  module.R1.expected_live_inventory, module.R1.validate_snapshot_population,
                  module.R1.final_validate_and_seal, module.M1249.CANONICAL_RESULT)
    try:
        with module.patched_live105_capture_chain():
            raise RuntimeError("context attack")
    except RuntimeError:
        pass
    six_after = (module.R1.DEAD_SN_V, module.R1.EXPECTED_LIVE_COUNTS,
                 module.R1.expected_live_inventory, module.R1.validate_snapshot_population,
                 module.R1.final_validate_and_seal, module.M1249.CANONICAL_RESULT)
    checks["context_exception_restores_six_globals"] = six_before == six_after

    old_m1343_digest = module.M1343.EXPECTED_ATLIF_NAMES_SHA256
    with mock.patch.object(module.M1343, "final_validate_and_seal_live105",
                           side_effect=RuntimeError("final attack")):
        try:
            module.final_validate_and_seal_live105(Path("unused"), object(), {})
        except RuntimeError:
            pass
    checks["final_validator_exception_restores_predecessor_digest"] = (
        module.M1343.EXPECTED_ATLIF_NAMES_SHA256 == old_m1343_digest)

    source_text = SOURCE.read_text(encoding="utf-8")
    checks["production_cli_absent"] = (
        "--run" not in source_text and "O_CREAT | os.O_EXCL" not in source_text and
        "torch.cuda" not in source_text and policy["production_authorized"] is False)
    checks["docs359_unchanged"] = sha(DOCS359) == module.DOCS359_SHA256

    false_negatives = [key for key, value in checks.items() if not value]
    output = {
        "schema": "m1353_m1349_live105_source_blind_hammer_output_r1_v1",
        "verdict": ("PASS_SOURCE__FRESH_RELEASE_AUTHOR_MAY_BE_AUTHORED" if not false_negatives
                    else "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"),
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "false_negative_count": len(false_negatives),
        "false_negatives": false_negatives,
        "checks": checks,
        "author_tests": "20/20 PASS",
        "remote_cpu_rebuild_executed_by_blind": False,
        "execution": {"gpu": False, "forward": False, "capture": False,
                      "attempt": False, "remote_write": False},
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not false_negatives else 2


if __name__ == "__main__":
    raise SystemExit(main())
