#!/usr/bin/env python3
"""Independent source-only hammer for the M1241 binder-r3 successor."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/build_m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor.py"
TEST = HW / "tests/test_build_m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor.py"
CONTRACT = HW / "contracts/m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor_source_contract_r1_20260830.json"
FIXTURE = HW / "tests/test_build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"
M1234 = HW / "scripts/build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor.py"
M1238 = HW / "reviews/m1238_m1234_cross_run_final_checkpoint_binder_successor_source_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = Path(__file__).with_name("hammer_output.json")
BASELINE_OUT = Path(__file__).with_name("baseline_test_log.txt")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1245_m1241_under_hammer", SOURCE)
T = load("m1245_m1228_fixture", FIXTURE)


class Case:
    def __init__(self) -> None:
        self.base = T.M1228CrossRunBinderTest(
            "test_cross_run_selection_and_selected_config_are_bound")
        self.base.setUp()
        r2 = M.load_predecessor()
        self.policy = r2.CrossRunPolicy(
            candidates=tuple(r2.CandidatePolicy(
                row.candidate_id, row.run_dir, row.config, row.config_sha256,
                row.epoch, row.expected_checkpoint_sha256)
                for row in self.base.policy.candidates),
            new_run_manifest=self.base.policy.new_run_manifest,
            new_evaluation_epochs=self.base.policy.new_evaluation_epochs,
        )

    def close(self) -> None:
        self.base.tearDown()

    def profile(self, index: int) -> Path:
        return self.base._profile_path(self.policy.candidates[index])


def with_case(action: Callable[[Case], Any]) -> Any:
    case = Case()
    try:
        return action(case)
    finally:
        case.close()


def rejected(action: Callable[[Case], Any]) -> str:
    def run(case: Case) -> str:
        try:
            action(case)
        except M.BinderError as exc:
            return type(exc).__name__ + ": " + str(exc)
        raise AssertionError("attack was accepted")
    return with_case(run)


def verify_double_seal(directory: Path) -> dict[str, str]:
    manifest = directory / "SHA256SUMS"
    seal = directory / "SHA256SUMS.seal.sha256"
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        assert sha(directory / name) == digest
        rows[name] = digest
    fields = seal.read_text(encoding="utf-8").split()
    assert fields == [sha(manifest), "SHA256SUMS"]
    return {"manifest_sha256": sha(manifest), "outer_seal_sha256": sha(seal)}


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert sha(SOURCE) == contract["source"]["sha256"]
    assert sha(TEST) == contract["test"]["sha256"]
    assert sha(M1234) == contract["source"]["predecessor_sha256"] == M.PREDECESSOR_SHA256
    assert sha(DOCS359) == contract["docs359_sha256"]
    m1238_seals = verify_double_seal(M1238)
    assert m1238_seals["manifest_sha256"] == contract["m1238_review_manifest_sha256"]

    baseline = subprocess.run(
        ["/usr/bin/python3.12", str(TEST)], cwd=str(ROOT),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert baseline.returncode == 0
    assert "Ran 18 tests" in baseline.stdout and baseline.stdout.count(" ... ok") == 18
    BASELINE_OUT.write_text(baseline.stdout, encoding="utf-8")

    def happy(case: Case) -> dict[str, Any]:
        result = M.build(case.policy)
        assert result["schema"] == contract["fixed_result_interface"]["schema"]
        assert result["status"] == contract["fixed_result_interface"]["status"]
        assert [row["epoch"] for row in result["candidate_population"]] == [29, 30, 32, 34]
        assert result["selected"]["epoch"] == 32
        assert len({row["run_directory"] for row in result["candidate_population"]}) == 2
        assert len({row["configuration"]["sha256"] for row in result["candidate_population"]}) == 2
        assert [row["id"] for row in
                result["e0_e8_activation_dependent_invalidation_and_rebind_targets"]] == [
                    "E{}".format(index) for index in range(9)]
        assert result["claim_boundary"]["hardware_rebind_authorized"] is False
        return result
    happy_result = with_case(happy)

    def semantic_swap(case: Case) -> None:
        target = case.profile(2)
        original = M.validate_profile
        swapped = False

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal swapped
            row = original(*args, **kwargs)
            if args[1].epoch == 32 and not swapped:
                swapped = True
                parked = target.with_name("profile.semantic.old.json")
                target.rename(parked)
                changed = json.loads(parked.read_text(encoding="utf-8"))
                changed["metrics"]["AEE"] = 9.0
                target.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
            return row

        with mock.patch.object(M, "validate_profile", side_effect=wrapper):
            M.build(case.policy)
    semantic_rejection = rejected(semantic_swap)

    def component_swap(case: Case) -> None:
        target = case.profile(2)
        original = M._open_chain
        swapped = False

        def wrapper(absolute: str, final_directory: bool) -> Any:
            nonlocal swapped
            if absolute == str(target) and not swapped:
                swapped = True
                parked = target.with_name("profile.preopen.old.json")
                target.rename(parked)
                changed = json.loads(parked.read_text(encoding="utf-8"))
                changed["metrics"]["AEE"] = 9.0
                target.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
            return original(absolute, final_directory)

        with mock.patch.object(M, "_open_chain", side_effect=wrapper):
            M.build(case.policy)
    preopen_rejection = rejected(component_swap)

    def post_check_no_trust(case: Case) -> dict[str, Any]:
        target = case.profile(2)
        old_sha = sha(target)
        old_inode = os.lstat(target).st_ino
        original = M.confirm_frozen_path
        swapped = False

        def wrapper(file: Any, label: str) -> None:
            nonlocal swapped
            original(file, label)
            if label == "epoch32 spike profile" and not swapped:
                swapped = True
                parked = target.with_name("profile.postcheck.old.json")
                target.rename(parked)
                changed = json.loads(parked.read_text(encoding="utf-8"))
                changed["metrics"]["AEE"] = 9.0
                target.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")

        with mock.patch.object(M, "confirm_frozen_path", side_effect=wrapper):
            result = M.build(case.policy)
        selected = result["selected"]
        assert swapped and selected["epoch"] == 32
        assert selected["profile"]["sha256"] == old_sha
        assert str(selected["accuracy_metrics"]["AEE"]) == "1.0"
        assert sha(target) != old_sha and os.lstat(target).st_ino != old_inode
        return {
            "outcome": "accepted_after_final_check_as_expected",
            "old_descriptor_bytes_remain_authoritative": True,
            "replacement_AEE_9_not_used": True,
            "selected_epoch": selected["epoch"],
            "stored_profile_sha256": selected["profile"]["sha256"],
        }
    post_check_result = with_case(post_check_no_trust)

    def run_parent_symlink(case: Case) -> None:
        alias = case.base.root / "run_parent_alias"
        alias.symlink_to(case.base.root, target_is_directory=True)
        candidates = list(case.policy.candidates)
        candidates[0] = replace(candidates[0], run_dir=alias / case.base.old_run.name)
        M.build(replace(case.policy, candidates=tuple(candidates)))
    run_parent_rejection = rejected(run_parent_symlink)

    def profile_parent_symlink(case: Case) -> None:
        parent = case.base.new_run / "standard_valid825"
        parked = parent.with_name("standard_valid825.real")
        parent.rename(parked)
        parent.symlink_to(parked.name, target_is_directory=True)
        M.build(case.policy)
    profile_parent_rejection = rejected(profile_parent_symlink)

    def config_parent_symlink(case: Case) -> None:
        parent = case.base.old_config.parent
        parked = parent.with_name("configs.real")
        parent.rename(parked)
        parent.symlink_to(parked.name, target_is_directory=True)
        M.build(case.policy)
    config_parent_rejection = rejected(config_parent_symlink)

    def checkpoint_symlink(case: Case) -> None:
        target = case.base.new_run / "checkpoint_epoch32.pth"
        parked = target.with_name("checkpoint_epoch32.real.pth")
        target.rename(parked)
        target.symlink_to(parked.name)
        M.build(case.policy)
    checkpoint_rejection = rejected(checkpoint_symlink)

    def root_rename_replacement(case: Case) -> None:
        original = M.freeze_file
        moved = case.base.root / "new_run.moved"
        changed = False

        def wrapper(path: Path, label: str, **kwargs: Any) -> Any:
            nonlocal changed
            if label == "new run manifest" and not changed:
                changed = True
                case.base.new_run.rename(moved)
                case.base.new_run.symlink_to(moved.name, target_is_directory=True)
            return original(path, label, **kwargs)

        with mock.patch.object(M, "freeze_file", side_effect=wrapper):
            M.build(case.policy)
    root_rename_rejection = rejected(root_rename_replacement)

    def config_hardlink_alias(case: Case) -> None:
        alias = case.base.old_config.parent / "resume_hardlink.yml"
        os.link(case.base.old_config, alias)
        candidates = list(case.policy.candidates)
        for index in range(1, 4):
            candidates[index] = replace(
                candidates[index], config=alias, config_sha256=sha(alias))
        attack = replace(case.policy, candidates=tuple(candidates))
        for candidate, aee in zip(attack.candidates, (1.20, 1.10, 1.00, 1.00)):
            case.base._write_profile(candidate, aee)
        M.build(attack)
    config_hardlink_rejection = rejected(config_hardlink_alias)

    def config_same_sha_distinct_inode(case: Case) -> None:
        alias = case.base.old_config.parent / "resume_copy.yml"
        alias.write_bytes(case.base.old_config.read_bytes())
        candidates = list(case.policy.candidates)
        for index in range(1, 4):
            candidates[index] = replace(
                candidates[index], config=alias, config_sha256=sha(alias))
        attack = replace(case.policy, candidates=tuple(candidates))
        for candidate, aee in zip(attack.candidates, (1.20, 1.10, 1.00, 1.00)):
            case.base._write_profile(candidate, aee)
        M.build(attack)
    config_same_sha_rejection = rejected(config_same_sha_distinct_inode)

    def frozen_root_identity_drift(case: Case) -> None:
        original = M.freeze_directory
        changed = False

        def wrapper(path: Path, label: str) -> Any:
            nonlocal changed
            frozen = original(path, label)
            if not changed and path == case.base.new_run:
                changed = True
                dev, inode, kind = frozen.physical_identity
                return M.FrozenDirectory(frozen.absolute_path, (dev, inode + 1, kind))
            return frozen

        with mock.patch.object(M, "freeze_directory", side_effect=wrapper):
            M.build(case.policy)
    frozen_root_identity_rejection = rejected(frozen_root_identity_drift)

    def lexical_alias(case: Case) -> None:
        candidates = list(case.policy.candidates)
        candidates[0] = replace(
            candidates[0], run_dir=case.base.old_run / ".." / case.base.old_run.name)
        M.build(replace(case.policy, candidates=tuple(candidates)))
    lexical_rejection = rejected(lexical_alias)

    def duplicate_profile_key(case: Case) -> None:
        target = case.profile(2)
        value = json.loads(target.read_text(encoding="utf-8"))
        text = json.dumps(value, sort_keys=True)
        target.write_text(text[:-1] + ',"samples":825}\n', encoding="utf-8")
        M.build(case.policy)
    duplicate_rejection = rejected(duplicate_profile_key)

    output = {
        "schema": "m1245_m1241_cross_run_binder_r3_successor_source_hammer_output_r1_v1",
        "status": "PASS_M1245_M1241_SOURCE_HAMMER__RELEASE_AUTHORING_ALLOWED",
        "baseline": {"tests": 18, "returncode": baseline.returncode},
        "pins": {
            "source_sha256": sha(SOURCE), "test_sha256": sha(TEST),
            "contract_sha256": sha(CONTRACT), "m1234_sha256": sha(M1234),
            "m1238": m1238_seals, "docs359_sha256": sha(DOCS359),
        },
        "preserved": {
            "schema": happy_result["schema"], "status": happy_result["status"],
            "epochs": [row["epoch"] for row in happy_result["candidate_population"]],
            "selected_epoch": happy_result["selected"]["epoch"],
            "samples_each": 825, "ATLIFTernaryPSN": 105,
            "ShiftmaxAttention": 12, "error_metrics": 8, "E0_E8": 9,
        },
        "attacks": {
            "semantic_validation_swap": semantic_rejection,
            "between_lstat_and_descriptor_open_swap": preopen_rejection,
            "post_final_check_no_pathname_trust": post_check_result,
            "run_parent_symlink": run_parent_rejection,
            "profile_parent_symlink": profile_parent_rejection,
            "config_parent_symlink": config_parent_rejection,
            "checkpoint_final_symlink": checkpoint_rejection,
            "frozen_root_rename_to_symlink": root_rename_rejection,
            "two_configs_same_device_inode_hardlink": config_hardlink_rejection,
            "two_configs_distinct_inode_same_sha": config_same_sha_rejection,
            "checkpoint_profile_frozen_root_identity_drift": frozen_root_identity_rejection,
            "lexically_unnormalized_run_alias": lexical_rejection,
            "duplicate_profile_json_key": duplicate_rejection,
        },
        "conclusion": {
            "M1238_parse_publication_blocker_closed": True,
            "M1238_run_alias_blocker_closed": True,
            "descriptor_component_nofollow": True,
            "checkpoint_profile_run_root_containment": True,
            "two_physical_run_roots_required": True,
            "two_physical_and_sha_distinct_configs_required": True,
            "post_check_pathname_not_reused": True,
            "source_hammer_pass": True,
            "production_executed": False,
            "hardware_rebind_authorized": False,
            "release_authoring_allowed": True,
        },
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
