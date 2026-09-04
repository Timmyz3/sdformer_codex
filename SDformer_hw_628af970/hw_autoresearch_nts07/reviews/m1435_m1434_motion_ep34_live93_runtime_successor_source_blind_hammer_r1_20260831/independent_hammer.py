#!/usr/bin/env python3
"""Different-author source-only hammer for the M1434 live-93 successor.

This program deliberately performs no SSH, GPU work, forward, capture,
attempt consumption, controller signal, EDA invocation, or production launch.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1434_motion_ep34_live93_runtime_successor_r1.py")
TEST = ROOT / "hw_autoresearch_nts07/tests/test_m1434_motion_ep34_live93_runtime_successor.py"
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1434_motion_ep34_live93_runtime_successor_source_contract_r1_20260831.json")
AUTHOR = ROOT / (
    "hw_autoresearch_nts07/reviews/"
    "m1434_motion_ep34_live93_runtime_successor_source_author_r1_20260831")
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

SOURCE_SHA = "b28c8507f077b754048fc54afd9fe04900dac854b273df2ba1981fa5f892b6ed"
TEST_SHA = "b05e122be8d3fb61b001be648ff8980a7341c2a19d29c401a9dc62ff5bafb8c2"
CONTRACT_SHA = "5e92af7c080f417fd94f190ce90c064a19fd70c02cfbd8fb6a2ad03d6f12e75e"
AUTHOR_REVIEW_SHA = "7342c61780a02634caebb5e5e1e2576f86a2406c2811a52bbccf47205051fe38"
AUTHOR_MANIFEST_SHA = "396c13afb70dffbc98090ee90a6994fb859d5cad18e1c826695945a9acc1001c"
AUTHOR_OUTER_FILE_SHA = "bf5852329c3d1febd48d7c4478046be33a78a6b3398e3497fd23da5d4f73b2bc"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
STATIC_DIGEST = "6a616f164625e3516bd2410f82d5f577c547c43a15b3bb2a5c4065add8a94cb7"
DEAD_DIGEST = "2f8e84f85d412008c45a67012da61d1ef7d248456aa64e8925af8aa57e6076a9"
LIVE_DIGEST = "f2dfcedab9ebe77b30b32d84bc38a2b1ea6511b0b3b359feb81a118ad2de252e"
H60_SHA = "0f77f66dbd331daa77a284199cda33125a1959a005b6f4d592e2e6cda5317187"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal_lf(names) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m1435_target_m1434", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def strict_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError("nonfinite " + token)))


def verify_root_relative_seal(root: Path) -> bool:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if not manifest.is_file() or manifest.is_symlink():
        return False
    if not outer.is_file() or outer.is_symlink():
        return False
    if outer.read_text(encoding="utf-8") != sha(manifest) + "  SHA256SUMS\n":
        return False
    names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2:
            return False
        digest, name = fields
        rel = Path(name)
        if name in names or rel.is_absolute() or ".." in rel.parts:
            return False
        member = ROOT / rel
        try:
            mode = member.lstat().st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(mode) or member.is_symlink() or sha(member) != digest:
            return False
        names.add(name)
    return bool(names)


def rejected(action) -> bool:
    try:
        action()
    except Exception:
        return True
    return False


def main() -> int:
    module = load_source()
    checks = {}
    checks["source_identity"] = sha(SOURCE) == SOURCE_SHA
    checks["test_identity"] = sha(TEST) == TEST_SHA
    checks["contract_identity"] = sha(CONTRACT) == CONTRACT_SHA
    checks["docs359_identity"] = sha(DOCS359) == DOCS359_SHA
    checks["author_review_identity"] = sha(AUTHOR / "review.json") == AUTHOR_REVIEW_SHA
    checks["author_manifest_identity"] = sha(AUTHOR / "SHA256SUMS") == AUTHOR_MANIFEST_SHA
    checks["author_outer_file_identity"] = sha(AUTHOR / "SHA256SUMS.seal.sha256") == AUTHOR_OUTER_FILE_SHA
    checks["author_double_seal_all_members"] = verify_root_relative_seal(AUTHOR)

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    tests = subprocess.run([str(PYTHON), "-B", str(TEST)], cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, check=False)
    checks["tests_22_of_22"] = (
        tests.returncode == 0 and "Ran 22 tests" in tests.stdout and "OK" in tests.stdout)
    self_check = subprocess.run(
        [str(PYTHON), "-B", str(SOURCE), "--source-self-check"], cwd=ROOT,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False)
    checks["source_self_check"] = (
        self_check.returncode == 0 and module.PASS_TOKEN in self_check.stdout)

    policy = strict_json(CONTRACT)
    module.verify_predecessors()
    checks["source_policy_positive"] = module.validate_source_policy(policy) == policy
    checks["failure_observation_positive"] = module.validate_failure_observation(policy) == (
        policy["m1400_failure_observation"])

    static_policy = module.R1.strict_json(module.R1.SOURCE_CONTRACT)
    static_inventory = module.R1.frozen_non_atlif_inventory(static_policy)
    static_inventory["atlif"] = list(module.M1349.EXPECTED_ATLIF_NAMES)
    static_names = static_inventory["atlif"]
    independent_dead = tuple(
        "sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.attn."
        "sn2_q.spiking_neuron".format(stage, block)
        for stage, blocks in enumerate((2, 2, 6, 2))
        for block in range(blocks))
    independent_live = sorted(set(static_names) - set(independent_dead))
    independent_live_inventory = {
        key: sorted(value) for key, value in static_inventory.items()}
    independent_live_inventory["atlif"] = independent_live
    checks["static_category_counts_exact"] = (
        {key: len(value) for key, value in static_inventory.items()} ==
        module.R1.EXPECTED_STATIC_COUNTS)
    checks["static_total_259"] = sum(map(len, static_inventory.values())) == 259
    checks["static_atlif_105"] = len(static_names) == 105
    checks["static_atlif_sorted_unique"] = (
        static_names == sorted(static_names) and len(set(static_names)) == 105)
    checks["static_atlif_digest_independent"] = terminal_lf(static_names) == STATIC_DIGEST
    checks["dead_tuple_exact_12"] = independent_dead == module.DEAD_SN2_Q and len(independent_dead) == 12
    checks["dead_tuple_digest_independent"] = terminal_lf(independent_dead) == DEAD_DIGEST
    checks["dead_tuple_is_static_subset"] = set(independent_dead) < set(static_names)
    checks["live_atlif_93"] = len(independent_live) == 93
    checks["live_atlif_sorted_unique"] = (
        independent_live == sorted(independent_live) and len(set(independent_live)) == 93)
    checks["live_atlif_digest_independent"] = terminal_lf(independent_live) == LIVE_DIGEST
    checks["live_total_247"] = sum(map(len, independent_live_inventory.values())) == 247
    checks["projection_only_removes_dead12"] = (
        set(static_names) - set(independent_live) == set(independent_dead) and
        all(independent_live_inventory[key] == sorted(static_inventory[key])
            for key in static_inventory if key != "atlif"))
    projected = module.expected_live93_inventory(copy.deepcopy(static_inventory))
    checks["implementation_projection_matches_independent"] = projected == independent_live_inventory

    rows = [
        {"global_sample_id": 0, "category": category, "name": name}
        for category, names in independent_live_inventory.items() for name in names]
    audit = module.audit_with_live93(rows, projected, [0])
    checks["positive_call_matrix_247"] = (
        audit["status"] == "PASS" and audit["records"] == 247 and
        audit["expected_records"] == 247 and audit["dead_modules"] == 12)
    checks["ordered_records_40x247"] = 40 * len(rows) == module.EXPECTED_ORDERED_RECORDS == 9880
    checks["attention_population_40x12"] = module.EXPECTED_ATTENTION == 480
    checks["payload_population_40x16"] = module.EXPECTED_PAYLOAD == 640

    dead_called = rows + [{"global_sample_id": 0, "category": "atlif",
                           "name": independent_dead[0]}]
    dead_audit = module.audit_with_live93(dead_called, projected, [0])
    checks["dead_called_rejected"] = (
        dead_audit["status"] == "FAIL" and
        any(item.startswith("dead_module_fired:") for item in dead_audit["errors"]))
    missing = module.audit_with_live93(rows[:-1], projected, [0])
    checks["missing_live_rejected"] = (
        missing["status"] == "FAIL" and any(item.startswith("call_count:") for item in missing["errors"]))
    duplicate = module.audit_with_live93(rows + [dict(rows[0])], projected, [0])
    checks["duplicate_live_rejected"] = (
        duplicate["status"] == "FAIL" and any(item.endswith(":2") for item in duplicate["errors"]))
    wrong_rows = copy.deepcopy(rows)
    wrong_rows[0]["category"] = "attention"
    wrong = module.audit_with_live93(wrong_rows, projected, [0])
    checks["wrong_category_rejected"] = (
        wrong["status"] == "FAIL" and
        any(item.startswith("unexpected_name_or_category:") for item in wrong["errors"]))

    def inventory_attack(kind: str) -> bool:
        value = copy.deepcopy(static_inventory)
        if kind == "rename":
            value["atlif"][value["atlif"].index(independent_dead[0])] += ".renamed"
        elif kind == "delete_replace":
            value["atlif"].remove(independent_dead[0])
            value["atlif"].append("fake.sn_q.spiking_neuron")
            value["atlif"].sort()
        elif kind == "extra_dead":
            live_name = next(name for name in value["atlif"] if name not in independent_dead)
            value["atlif"][value["atlif"].index(live_name)] = live_name + ".sn2_q.spiking_neuron"
            value["atlif"].sort()
        elif kind == "reorder":
            value["atlif"][0:2] = reversed(value["atlif"][0:2])
        return rejected(lambda: module.expected_live93_inventory(value))

    for kind in ("rename", "delete_replace", "extra_dead", "reorder"):
        checks["dead_set_" + kind + "_rejected"] = inventory_attack(kind)

    observed = policy["m1400_failure_observation"]
    expected_errors = ["call_count:0:{}:0".format(name) for name in independent_dead] + [
        "record_count:247:259"]
    checks["failure_hash_pins_exact"] = (
        observed["attempt_sha256"] == "0c17499d2fec3a58272af40c22186f5537873d2c061da9c13ff2acceff968e37" and
        observed["production_log_sha256"] == "45fe9d6054b3475a47baa3ef693d1a941e82e14077954feb27bdd20fa6b40ba5" and
        observed["failed_json_sha256"] == "225fec4fce4d246a0deb851406ef44bdb3dbe657f56d94356b80888f7178849b" and
        observed["sample0_snapshot_manifest_sha256"] == "d9dbd76ff51711518bdfeca9463c9829d4ba1e03de56e449b50b1d11ed58f23c" and
        observed["sample0_ordered_sha256"] == "82e52f5fc69a53a2ff0ddf525adcc9981791564967470f8970c3a49e91fe3298")
    checks["failure_sample0_exact_old_projection"] = observed["sample0_call_audit"] == {
        "status": "FAIL", "samples": 1, "live_modules_per_sample": 259,
        "records": 247, "expected_records": 259, "dead_modules": 0,
        "errors": expected_errors}
    checks["failure_sample0_category_projection"] = (
        observed["sample0_category_counts"] == {
            "c1_conv3x3": 4, "decoder_convtranspose": 4, "atlif": 93,
            "fc1": 12, "fc2": 12, "patch_embed": 8, "batch_norm": 78,
            "qkv": 24, "attention": 12} and
        sum(observed["sample0_category_counts"].values()) == 247)
    checks["failure_exactly_dead12_plus_total"] = observed["sample0_call_audit"]["errors"] == expected_errors
    checks["failure_controller_remains_stopped"] = observed["controller_after_failure"] == {
        "pid": 3804343, "state": "T"}
    checks["failure_no_retry_no_restore"] = (
        observed["automatic_retry"] is False and observed["controller_restore_permitted"] is False)

    observation_mutations = {
        "records": lambda value: value["sample0_call_audit"].__setitem__("records", 248),
        "expected_records": lambda value: value["sample0_call_audit"].__setitem__("expected_records", 247),
        "live_modules": lambda value: value["sample0_call_audit"].__setitem__("live_modules_per_sample", 247),
        "dead_modules": lambda value: value["sample0_call_audit"].__setitem__("dead_modules", 12),
        "errors_missing": lambda value: value["sample0_call_audit"]["errors"].pop(0),
        "errors_reordered": lambda value: value["sample0_call_audit"]["errors"].reverse(),
        "category": lambda value: value["sample0_category_counts"].__setitem__("atlif", 94),
        "attempt_hash": lambda value: value.__setitem__("attempt_sha256", "0" * 64),
        "ordered_hash": lambda value: value.__setitem__("sample0_ordered_sha256", "0" * 64),
        "retry": lambda value: value.__setitem__("automatic_retry", True),
        "restore": lambda value: value.__setitem__("controller_restore_permitted", True),
        "controller": lambda value: value["controller_after_failure"].__setitem__("state", "S"),
        "staging": lambda value: value.__setitem__("staging_basename", "old_namespace"),
    }
    for name, mutate in observation_mutations.items():
        mutant = copy.deepcopy(policy)
        mutate(mutant["m1400_failure_observation"])
        checks["failure_observation_mutation_rejected_" + name] = rejected(
            lambda value=mutant: module.validate_failure_observation(value))

    bsa_text = module.BSA_SOURCE.read_text(encoding="utf-8")
    begin = bsa_text.index('elif cfg.mode in {"h60", "tx_sc_k_mag_no_carrier_shiftmax"}:')
    end = bsa_text.index('elif cfg.mode in {"h82",', begin)
    h60_branch = bsa_text[begin:end]
    checks["h60_source_identity"] = sha(module.BSA_SOURCE) == H60_SHA
    checks["h60_branch_gate_multiply"] = "attn = k_orig.mul(gate)" in h60_branch
    checks["h60_branch_bypasses_sn2_q"] = "self.sn2_q(" not in h60_branch
    checks["h60_validator_positive"] = not rejected(module.validate_h60_bypass_source)

    namespaces = policy["new_namespaces"]
    checks["new_namespace_exact_projection"] = namespaces == {
        "result": str(module.CANONICAL_RESULT.relative_to(ROOT)),
        "attempt": str(module.CANONICAL_ATTEMPT.relative_to(ROOT)),
        "log": str(module.CANONICAL_LOG.relative_to(ROOT))}
    checks["new_namespace_three_distinct"] = len(set(namespaces.values())) == 3
    checks["new_namespace_all_m1434"] = all("m1434_" in Path(value).name for value in namespaces.values())
    checks["new_namespace_not_old_m1349"] = (
        module.CANONICAL_RESULT != module.M1349.CANONICAL_RESULT and
        module.CANONICAL_RESULT != module.M1249.CANONICAL_RESULT)
    checks["fresh_namespace_positive"] = not rejected(module.require_fresh_namespaces)
    with mock.patch.object(module.os.path, "lexists", return_value=True):
        checks["namespace_collision_rejected"] = rejected(module.require_fresh_namespaces)
    old_namespace = copy.deepcopy(policy)
    old_namespace["new_namespaces"]["result"] = str(module.M1349.CANONICAL_RESULT.relative_to(ROOT))
    checks["old_namespace_reuse_rejected"] = rejected(lambda: module.validate_source_policy(old_namespace))

    checks["source_no_launch_authority"] = (
        policy["launch_authorized"] is False and policy["runs"] == 0 and
        policy["automatic_retry"] is False)
    checks["claim_boundary_source_only"] = policy["claim_boundary"] == {
        "source_and_tests_only": True, "different_author_blind_required": True,
        "gpu": False, "forward": False, "capture": False, "attempt": False,
        "remote": False, "controller_signal": False,
        "controller_restore": False, "production_release": False,
        "hardware_result": False}
    source_text = SOURCE.read_text(encoding="utf-8")
    checks["source_has_no_production_cli"] = 'add_argument("--run"' not in source_text
    checks["source_has_no_gpu_ssh_subprocess_control"] = all(token not in source_text for token in (
        "torch.cuda", "subprocess", "os.kill", "SIGCONT", "paramiko", "ssh "))

    originals = (
        module.R1.DEAD_SN_V, module.R1.EXPECTED_LIVE_COUNTS,
        module.R1.expected_live_inventory, module.R1.validate_snapshot_population,
        module.R1.final_validate_and_seal, module.M1249.CANONICAL_RESULT)
    try:
        with module.patched_live93_capture_chain():
            raise RuntimeError("restoration attack")
    except RuntimeError:
        pass
    restored = (
        module.R1.DEAD_SN_V, module.R1.EXPECTED_LIVE_COUNTS,
        module.R1.expected_live_inventory, module.R1.validate_snapshot_population,
        module.R1.final_validate_and_seal, module.M1249.CANONICAL_RESULT)
    checks["context_finally_restores_six_globals"] = originals == restored

    old_dead = module.R1.DEAD_SN_V
    with mock.patch.object(module.R1, "audit_call_matrix", side_effect=RuntimeError("audit attack")):
        try:
            module.audit_with_live93([], projected, [0])
        except RuntimeError:
            pass
    checks["audit_finally_restores_dead_set"] = module.R1.DEAD_SN_V == old_dead

    with mock.patch.object(module.M1249, "run_capture", side_effect=RuntimeError("delegate attack")):
        runtime = {"contract_path": "old", "capture": {"attention_windows_per_call": 100},
                   "cohort": {"samples": ["sealed"]}, "output": {"path": "old"}}
        binding = {"identity": {"checkpoint_sha256": module.CHECKPOINT_SHA256,
                                "config_sha256": module.CONFIG_SHA256}}
        with mock.patch.object(module.M1349, "build_runtime", return_value=(runtime, binding)):
            expected_runtime, expected_binding = module.build_runtime()
            try:
                module.delegate_for_future_release(expected_runtime, expected_binding, object())
            except RuntimeError:
                pass
    restored_after_delegate = (
        module.R1.DEAD_SN_V, module.R1.EXPECTED_LIVE_COUNTS,
        module.R1.expected_live_inventory, module.R1.validate_snapshot_population,
        module.R1.final_validate_and_seal, module.M1249.CANONICAL_RESULT)
    checks["delegate_failure_restores_six_globals"] = originals == restored_after_delegate

    false_negatives = [key for key, value in checks.items() if not value]
    output = {
        "schema": "m1435_m1434_live93_source_blind_hammer_output_r1_v1",
        "verdict": ("PASS_SOURCE__FRESH_RELEASE_AUTHOR_MAY_BE_AUTHORED"
                    if not false_negatives else
                    "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"),
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "false_negative_count": len(false_negatives),
        "false_negatives": false_negatives,
        "checks": checks,
        "tests": "22/22 PASS" if checks["tests_22_of_22"] else "FAIL",
        "source_self_check": "PASS" if checks["source_self_check"] else "FAIL",
        "failure_evidence_boundary": {
            "remote_reaccessed": False,
            "authority": "exact hashes and sample-0 observation sealed into committed M1434 contract",
            "static_modules": 259, "static_atlif": 105,
            "observed_live_modules": 247, "observed_live_atlif": 93,
            "dead_sn2_q": 12, "ordered_records_40_samples": 9880,
        },
        "execution": {
            "ssh": False, "gpu": False, "forward": False, "capture": False,
            "attempt": False, "controller_signal": False, "controller_restore": False,
            "eda": False,
        },
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not false_negatives else 2


if __name__ == "__main__":
    raise SystemExit(main())
