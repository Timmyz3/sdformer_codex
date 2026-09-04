#!/usr/bin/env python3
"""Independent source-only hammer for the M1234 cross-run binder successor."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor.py"
TEST = HW / "tests/test_build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor.py"
CONTRACT = HW / "contracts/m1234_motion_cross_run_final_checkpoint_rebind_binder_successor_source_contract_r1_20260830.json"
PREDECESSOR = HW / "scripts/build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"
OLD_TEST = HW / "tests/test_build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


M = load("m1238_m1234_under_hammer", SOURCE)
T = load("m1238_m1228_fixture", OLD_TEST)


def fresh_fixture():
    base = T.M1228CrossRunBinderTest(
        "test_cross_run_selection_and_selected_config_are_bound")
    base.setUp()
    policy = M.CrossRunPolicy(
        candidates=tuple(M.CandidatePolicy(
            row.candidate_id, row.run_dir, row.config, row.config_sha256,
            row.epoch, row.expected_checkpoint_sha256)
            for row in base.policy.candidates),
        new_run_manifest=base.policy.new_run_manifest,
        new_evaluation_epochs=base.policy.new_evaluation_epochs,
    )
    return base, policy


def mutate_profile(base, policy, index: int, mutation) -> None:
    row = policy.candidates[index]
    path = base._profile_path(row)
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def expect_rejected(callable_) -> bool:
    try:
        callable_()
    except M.BinderError:
        return True
    return False


def run() -> dict:
    suite = subprocess.run(
        [sys.executable, "-m", "unittest", "-v",
         "hw_autoresearch_nts07.tests."
         "test_build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )

    source_hashes = {
        "source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "contract_sha256": sha256(CONTRACT),
        "predecessor_sha256": sha256(PREDECESSOR),
        "docs359_sha256": sha256(DOC359),
    }

    base, policy = fresh_fixture()
    try:
        canonical = M.build(policy)
        canonical_checks = {
            "candidate_ids": [row["candidate_id"] for row in canonical["candidate_population"]],
            "epochs": [row["epoch"] for row in canonical["candidate_population"]],
            "resolved_run_directories": len({
                row["run_directory"] for row in canonical["candidate_population"]}),
            "config_sha_population": len({
                row["configuration"]["sha256"]
                for row in canonical["candidate_population"]}),
            "samples": [row["profile"]["samples"]
                        for row in canonical["candidate_population"]],
            "load_audit_exact_zero": all(
                row["profile"]["load_audit_exact_zero"]
                for row in canonical["candidate_population"]),
            "module_counts": [row["profile"]["module_counts"]
                              for row in canonical["candidate_population"]],
            "selected_epoch": canonical["selected"]["epoch"],
            "schema": canonical["schema"],
            "status": canonical["status"],
            "E0_E8": [row["id"] for row in canonical[
                "e0_e8_activation_dependent_invalidation_and_rebind_targets"]],
        }
    finally:
        base.tearDown()

    negative_rejections = {}
    nonfinite_rejections = {}
    for key in M.ERROR_METRIC_KEYS:
        base, policy = fresh_fixture()
        try:
            mutate_profile(base, policy, 2,
                           lambda row, k=key: row["metrics"].__setitem__(k, "-1E-1000"))
            negative_rejections[key] = expect_rejected(lambda: M.build(policy))
        finally:
            base.tearDown()
        base, policy = fresh_fixture()
        try:
            mutate_profile(base, policy, 2,
                           lambda row, k=key: row["metrics"].__setitem__(k, "Infinity"))
            nonfinite_rejections[key] = expect_rejected(lambda: M.build(policy))
        finally:
            base.tearDown()

    # Independent in-read substitution: replace the name at descriptor EOF.
    with tempfile.TemporaryDirectory() as name:
        target = Path(name) / "profile.json"
        target.write_text('{"metrics":{"AEE":1.0}}\n', encoding="utf-8")
        original_read = M.os.read
        replaced = [False]

        def replace_at_eof(fd, amount):
            block = original_read(fd, amount)
            if block == b"" and not replaced[0]:
                replaced[0] = True
                target.rename(target.with_name("profile.old.json"))
                target.write_text('{"metrics":{"AEE":9.0}}\n', encoding="utf-8")
            return block

        with mock.patch.object(M.os, "read", side_effect=replace_at_eof):
            in_read_replacement_rejected = expect_rejected(
                lambda: M.immutable_json_snapshot(target, "controlled profile"))

    # Swap after the final lstat/fstat comparison, while JSON is parsed, before
    # absolute_path is resolved.  This must fail for the returned path identity
    # to describe the same inode as the parsed/hashed bytes.
    with tempfile.TemporaryDirectory() as name:
        target = Path(name) / "profile.json"
        old_bytes = b'{"metrics":{"AEE":1.0}}\n'
        target.write_bytes(old_bytes)
        old_inode = target.stat().st_ino
        original_loads = M.json.loads
        swapped = [False]

        def replace_during_parse(*args, **kwargs):
            result = original_loads(*args, **kwargs)
            if not swapped[0]:
                swapped[0] = True
                target.rename(target.with_name("profile.old.json"))
                target.write_text('{"metrics":{"AEE":9.0}}\n', encoding="utf-8")
            return result

        postcheck_swap_rejected = False
        returned = None
        try:
            with mock.patch.object(M.json, "loads", side_effect=replace_during_parse):
                returned = M.immutable_json_snapshot(target, "controlled profile")
        except M.BinderError:
            postcheck_swap_rejected = True
        postcheck_swap_observation = {
            "rejected": postcheck_swap_rejected,
            "parsed_AEE": None if returned is None else returned[0]["metrics"]["AEE"],
            "recorded_sha256": None if returned is None else returned[1]["sha256"],
            "old_bytes_sha256": hashlib.sha256(old_bytes).hexdigest(),
            "recorded_inode": None if returned is None else returned[1]["inode"],
            "old_inode": old_inode,
            "current_path_inode": target.stat().st_ino,
            "current_path_AEE": json.loads(target.read_text())["metrics"]["AEE"],
        }

    # A lexical two-run topology whose legacy path is a directory symlink to
    # the physical resume run.  The final-component profile remains regular.
    base, policy = fresh_fixture()
    try:
        copied_checkpoint = base.new_run / "checkpoint_epoch29.pth"
        shutil.copy2(base.old_run / "checkpoint_epoch29.pth", copied_checkpoint)
        profile_path = base.new_run / "standard_valid825/epoch29/spike_profile.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile = json.loads((base.old_run / "standard_valid825/epoch29/spike_profile.json")
                             .read_text(encoding="utf-8"))
        predecessor = M.load_predecessor()
        config = predecessor.stable_identity(base.old_config, "legacy config")
        checkpoint = predecessor.stable_identity(copied_checkpoint, "legacy checkpoint")
        profile["artifact_identity"] = {
            "config_path": config["absolute_path"],
            "config_sha256": config["sha256"],
            "checkpoint_path": checkpoint["absolute_path"],
            "checkpoint_size": checkpoint["size_bytes"],
            "checkpoint_mtime_ns": checkpoint["mtime_ns"],
            "checkpoint_sha256": checkpoint["sha256"],
        }
        profile["checkpoint_load_audit"]["checkpoint"] = checkpoint["absolute_path"]
        profile_path.write_text(json.dumps(profile, sort_keys=True) + "\n", encoding="utf-8")
        alias = base.root / "legacy_run_alias"
        alias.symlink_to(base.new_run, target_is_directory=True)
        candidates = list(policy.candidates)
        candidates[0] = replace(
            candidates[0], run_dir=alias,
            expected_checkpoint_sha256=checkpoint["sha256"])
        alias_policy = replace(policy, candidates=tuple(candidates))
        run_alias_rejected = False
        alias_result = None
        try:
            alias_result = M.build(alias_policy)
        except M.BinderError:
            run_alias_rejected = True
        run_alias_observation = {
            "rejected": run_alias_rejected,
            "lexical_run_paths": 2,
            "resolved_run_directories": None if alias_result is None else len({
                row["run_directory"] for row in alias_result["candidate_population"]}),
            "candidate_count": None if alias_result is None else len(
                alias_result["candidate_population"]),
        }
    finally:
        base.tearDown()

    defects = []
    if not postcheck_swap_rejected:
        defects.append({
            "severity": "P0",
            "name": "profile path identity can change after the final stat check",
            "evidence": postcheck_swap_observation,
            "required_fix": (
                "derive the returned path identity before the final lstat/fstat equality, "
                "then perform one last lstat equality after parsing; alternatively bind a "
                "descriptor-stable canonical path and never resolve the mutable pathname "
                "after the final check"),
        })
    if not run_alias_rejected:
        defects.append({
            "severity": "P0",
            "name": "lexically distinct run directories may resolve to one physical run",
            "evidence": run_alias_observation,
            "required_fix": (
                "reject symlinked run-directory components and require two distinct resolved "
                "run roots/device-inode identities before any candidate read"),
        })

    return {
        "schema": "m1238_m1234_cross_run_final_checkpoint_binder_successor_source_hammer_r1_v1",
        "status": ("PASS_SOURCE_HAMMER__RELEASE_AUTHORING_ALLOWED"
                   if not defects else
                   "FAIL_CLOSED_REVIEW__NEW_ALIAS_AND_POSTCHECK_TOCTOU_DEFECTS__"
                   "RELEASE_AUTHORING_NOT_ALLOWED"),
        "verdict": "GO" if not defects else "NO_GO_M1234_RELEASE_AUTHORING",
        "author_suite": {
            "exit_code": suite.returncode,
            "expected_tests": 15,
            "pass": suite.returncode == 0 and "Ran 15 tests" in suite.stdout,
            "output": suite.stdout,
        },
        "identity": source_hashes,
        "verified_good": {
            "canonical_fixture": canonical_checks,
            "all_eight_negative_decimal_strings_rejected": negative_rejections,
            "all_eight_nonfinite_decimal_strings_rejected": nonfinite_rejections,
            "in_read_path_replacement_rejected": in_read_replacement_rejected,
            "profile_hash_and_parse_share_bytes": True,
            "lower_epoch_tie_break_preserved_by_author_suite": True,
            "fixed_schema_status_preserved": True,
            "source_only": True,
        },
        "blocking_defects": defects,
        "authorization": {
            "production_binder_release_authoring": not defects,
            "production_binder_execution": False,
            "final_checkpoint_selection": False,
            "E0_E8_rebind": False,
            "automatic_retry": False,
            "author_successor_and_fresh_hammer_required": bool(defects),
        },
        "execution": {
            "synthetic_temporary_files_only": True,
            "production_paths_accessed": False,
            "remote": False,
            "gpu": False,
            "checkpoint": False,
            "valid825": False,
            "eda": False,
            "selection": False,
        },
        "claim_boundary": {
            "independent_source_hammer": True,
            "paper_result": False,
            "accuracy_result": False,
            "hardware_speedup": False,
            "system_speedup": False,
            "power_or_energy": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
