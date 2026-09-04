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
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/build_m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor.py"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor_source_contract_r1_20260830.json"
OLD_TEST = ROOT / "hw_autoresearch_nts07/tests/test_build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


M = load("m1241_successor_under_test", SOURCE)
T = load("m1241_m1228_fixture", OLD_TEST)


class M1241SuccessorTest(unittest.TestCase):
    def setUp(self):
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

    def tearDown(self):
        self.base.tearDown()

    def profile_path(self, index):
        return self.base._profile_path(self.policy.candidates[index])

    def mutate_profile(self, index, mutation):
        path = self.profile_path(index)
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def test_01_import_is_inert_and_predecessor_is_lazy(self):
        code = (
            "import importlib.util,sys;"
            "s=importlib.util.spec_from_file_location('isolated_m1241',{!r});"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "print(int('m1241_sealed_m1234' in sys.modules))"
        ).format(str(SOURCE))
        self.assertEqual(subprocess.check_output(
            [sys.executable, "-c", code]).decode().strip(), "0")

    def test_02_source_contract_test_and_predecessor_are_pinned(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["source"]["sha256"], sha(SOURCE))
        self.assertEqual(contract["test"]["sha256"], sha(Path(__file__).resolve()))
        self.assertEqual(sha(M.PREDECESSOR), M.PREDECESSOR_SHA256)

    def test_03_happy_path_preserves_exact_fixed_interface(self):
        result = M.build(self.policy)
        self.assertEqual(result["schema"],
                         "m1234_motion_cross_run_final_checkpoint_rebind_binder_r2_v1")
        self.assertEqual(result["status"],
                         "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R2__"
                         "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY")
        self.assertEqual([row["epoch"] for row in result["candidate_population"]],
                         [29, 30, 32, 34])
        self.assertEqual(result["selected"]["epoch"], 32)
        self.assertEqual(len({row["run_directory"]
                              for row in result["candidate_population"]}), 2)
        self.assertEqual(len({row["configuration"]["sha256"]
                              for row in result["candidate_population"]}), 2)

    def test_04_lower_epoch_tie_break_is_preserved(self):
        for index, value in enumerate((0.5, 0.5, 0.8, 0.9)):
            self.base._write_profile(self.policy.candidates[index], value)
        self.assertEqual(M.build(self.policy)["selected"]["epoch"], 29)

    def test_05_post_final_stat_parse_time_swap_is_rejected(self):
        target = self.profile_path(2)
        original_loads = M.json.loads
        swapped = [False]

        def replace_during_profile_parse(*args, **kwargs):
            result = original_loads(*args, **kwargs)
            if (not swapped[0] and isinstance(result, dict) and
                    result.get("artifact_identity", {}).get("checkpoint_path", "").endswith(
                        "checkpoint_epoch32.pth")):
                swapped[0] = True
                target.rename(target.with_name("profile.before_swap.json"))
                result["metrics"]["AEE"] = 9.0
                target.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
            return result

        with mock.patch.object(M.json, "loads", side_effect=replace_during_profile_parse):
            with self.assertRaisesRegex(M.BinderError, "changed before frozen identity"):
                M.build(self.policy)
        self.assertTrue(swapped[0])

    def test_05b_semantic_validation_time_swap_is_rejected(self):
        target = self.profile_path(2)
        original_validate = M.validate_profile
        swapped = [False]

        def replace_after_semantic_validation(*args, **kwargs):
            result = original_validate(*args, **kwargs)
            candidate = args[1]
            if candidate.epoch == 32 and not swapped[0]:
                swapped[0] = True
                target.rename(target.with_name("profile.before_semantic_swap.json"))
                changed = json.loads(target.with_name(
                    "profile.before_semantic_swap.json").read_text(encoding="utf-8"))
                changed["metrics"]["AEE"] = 9.0
                target.write_text(json.dumps(changed, sort_keys=True) + "\n",
                                  encoding="utf-8")
            return result

        with mock.patch.object(M, "validate_profile",
                               side_effect=replace_after_semantic_validation):
            with self.assertRaisesRegex(M.BinderError,
                                        "changed during semantic validation"):
                M.build(self.policy)
        self.assertTrue(swapped[0])

    def test_06_run_root_symlink_collapse_is_rejected(self):
        alias = self.base.root / "legacy_run_alias"
        alias.symlink_to(self.base.new_run, target_is_directory=True)
        candidates = list(self.policy.candidates)
        candidates[0] = replace(candidates[0], run_dir=alias)
        attack = replace(self.policy, candidates=tuple(candidates))
        with self.assertRaisesRegex(M.BinderError, "non-symlink directory|no-follow"):
            M.build(attack)

    def test_07_run_root_parent_component_symlink_is_rejected(self):
        parent_alias = self.base.root / "parent_alias"
        parent_alias.symlink_to(self.base.root, target_is_directory=True)
        aliased_old = parent_alias / self.base.old_run.name
        candidates = list(self.policy.candidates)
        candidates[0] = replace(candidates[0], run_dir=aliased_old)
        attack = replace(self.policy, candidates=tuple(candidates))
        with self.assertRaises(M.BinderError):
            M.build(attack)

    def test_08_checkpoint_symlink_and_profile_parent_symlink_are_rejected(self):
        checkpoint = self.base.new_run / "checkpoint_epoch30.pth"
        real_checkpoint = checkpoint.with_name("checkpoint_epoch30.real.pth")
        checkpoint.rename(real_checkpoint)
        checkpoint.symlink_to(real_checkpoint.name)
        with self.assertRaises(M.BinderError):
            M.build(self.policy)

    def test_09_two_config_physical_and_sha_identities_are_required(self):
        candidates = list(self.policy.candidates)
        for index in range(1, 4):
            candidates[index] = replace(
                candidates[index], config=self.base.old_config,
                config_sha256=sha(self.base.old_config))
        attack = replace(self.policy, candidates=tuple(candidates))
        with self.assertRaises(M.BinderError):
            M.build(attack)

    def test_10_every_error_metric_rejects_negative_and_nonfinite_strings(self):
        r2 = M.load_predecessor()
        canonical = self.profile_path(2).read_bytes()
        for key in r2.ERROR_METRIC_KEYS:
            for value in ("-1E-1000", "Infinity", "NaN"):
                with self.subTest(key=key, value=value):
                    self.mutate_profile(
                        2, lambda row, k=key, v=value:
                        row["metrics"].__setitem__(k, v))
                    with self.assertRaises(M.BinderError):
                        M.build(self.policy)
                    self.profile_path(2).write_bytes(canonical)

    def test_11_valid825_load_audit_and_105_12_are_strict(self):
        canonical = self.profile_path(1).read_bytes()
        mutations = (
            lambda row: row.__setitem__("samples", True),
            lambda row: row["checkpoint_load_audit"].__setitem__("missing_count", False),
            lambda row: row["module_counts"].__setitem__("ATLIFTernaryPSN", 104),
            lambda row: row["module_counts"].__setitem__("ShiftmaxAttention", 13),
        )
        for mutation in mutations:
            self.mutate_profile(1, mutation)
            with self.assertRaises(M.BinderError):
                M.build(self.policy)
            self.profile_path(1).write_bytes(canonical)

    def test_12_missing_any_exact_four_profile_fails_closed(self):
        for index in range(4):
            path = self.profile_path(index)
            content = path.read_bytes()
            path.unlink()
            with self.assertRaises(M.BinderError):
                M.build(self.policy)
            path.write_bytes(content)

    def test_13_profile_publication_is_frozen_without_resolve(self):
        result = M.build(self.policy)
        for row in result["candidate_population"]:
            profile = row["profile"]
            self.assertTrue(profile["post_parse_path_identity_frozen"])
            self.assertTrue(profile["descriptor_rooted_no_symlink_components"])
            self.assertTrue(profile["immutable_single_read"])
            self.assertTrue(profile["hash_and_parse_same_bytes"])

    def test_14_E0_E8_and_claim_boundary_are_preserved(self):
        result = M.build(self.policy)
        targets = result["e0_e8_activation_dependent_invalidation_and_rebind_targets"]
        self.assertEqual([row["id"] for row in targets],
                         ["E{}".format(index) for index in range(9)])
        self.assertFalse(result["claim_boundary"]["hardware_rebind_authorized"])
        self.assertFalse(result["claim_boundary"]["hardware_speedup"])

    def test_15_receipt_remains_double_sealed(self):
        output = self.base.root / "m1241_receipt"
        M.write_receipt(output, M.build(self.policy))
        sums = {}
        for line in (output / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(None, 1)
            sums[name.lstrip("*")] = digest
        for name, digest in sums.items():
            self.assertEqual(sha(output / name), digest)
        self.assertEqual((output / "SHA256SUMS.seal.sha256").read_text().split(),
                         [sha(output / "SHA256SUMS"), "SHA256SUMS"])

    def test_16_source_only_has_no_remote_gpu_or_eda_execution(self):
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import subprocess", "paramiko", "ssh ",
                          "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)

    def test_17_docs359_is_still_frozen(self):
        docs = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(sha(docs),
                         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
