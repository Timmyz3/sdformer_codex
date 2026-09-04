from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor.py"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1234_motion_cross_run_final_checkpoint_rebind_binder_successor_source_contract_r1_20260830.json"
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


M = load("m1234_successor_under_test", SOURCE)
T = load("m1234_m1228_fixture", OLD_TEST)


class M1234SuccessorTest(unittest.TestCase):
    def setUp(self):
        self.base = T.M1228CrossRunBinderTest(
            "test_cross_run_selection_and_selected_config_are_bound")
        self.base.setUp()
        self.policy = M.CrossRunPolicy(
            candidates=tuple(M.CandidatePolicy(
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
            "s=importlib.util.spec_from_file_location('isolated_m1234',{!r});"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "print(int('m1234_sealed_m1228' in sys.modules))"
        ).format(str(SOURCE))
        self.assertEqual(subprocess.check_output(
            [sys.executable, "-c", code]).decode().strip(), "0")

    def test_02_source_contract_and_predecessor_are_pinned(self):
        policy = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(policy["source"]["sha256"], sha(SOURCE))
        self.assertEqual(policy["test"]["sha256"], sha(Path(__file__).resolve()))
        self.assertEqual(sha(M.PREDECESSOR), M.PREDECESSOR_SHA256)

    def test_03_happy_path_exact4_selection_and_fixed_interface(self):
        result = M.build(self.policy)
        self.assertEqual(result["schema"],
                         "m1234_motion_cross_run_final_checkpoint_rebind_binder_r2_v1")
        self.assertEqual(result["status"],
                         "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R2__"
                         "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY")
        self.assertEqual(len(result["candidate_population"]), 4)
        self.assertEqual(result["selected"]["candidate_id"], "resume_ep32")
        self.assertEqual(result["selected"]["checkpoint"]["sha256"], sha(
            self.base.new_run / "checkpoint_epoch32.pth"))
        self.assertEqual(result["selected"]["configuration"]["sha256"], sha(
            self.base.new_config))

    def test_04_lower_epoch_tie_break_is_preserved(self):
        for index, value in enumerate((0.5, 0.5, 0.8, 0.9)):
            self.base._write_profile(self.policy.candidates[index], value)
        self.assertEqual(M.build(self.policy)["selected"]["epoch"], 29)

    def test_05_missing_each_of_exact_four_profiles_fails_closed(self):
        for index in range(4):
            path = self.profile_path(index)
            content = path.read_bytes()
            path.unlink()
            with self.subTest(index=index):
                with self.assertRaises(M.BinderError):
                    M.build(self.policy)
            path.write_bytes(content)

    def test_06_two_run_and_two_config_topology_is_exact(self):
        attacks = (
            replace(self.policy, candidates=self.policy.candidates[:-1]),
            replace(self.policy, candidates=(self.policy.candidates[0],) + tuple(
                replace(row, run_dir=self.base.old_run) for row in self.policy.candidates[1:])),
            replace(self.policy, candidates=(self.policy.candidates[0],) + tuple(
                replace(row, config=self.base.old_config) for row in self.policy.candidates[1:])),
        )
        for attack in attacks:
            with self.assertRaises(M.BinderError):
                M.build(attack)

    def test_07_valid825_load_audit_and_module_counts_remain_strict(self):
        mutations = (
            lambda row: row.__setitem__("samples", True),
            lambda row: row["checkpoint_load_audit"].__setitem__("missing_count", False),
            lambda row: row["module_counts"].__setitem__("ATLIFTernaryPSN", 104),
            lambda row: row["module_counts"].__setitem__("extra", 1),
        )
        path = self.profile_path(1)
        canonical = path.read_bytes()
        for mutation in mutations:
            self.mutate_profile(1, mutation)
            with self.assertRaises(M.BinderError):
                M.build(self.policy)
            path.write_bytes(canonical)

    def test_08_every_error_metric_rejects_negative(self):
        path = self.profile_path(2)
        canonical = path.read_bytes()
        for key in M.ERROR_METRIC_KEYS:
            with self.subTest(key=key):
                self.mutate_profile(2, lambda row, k=key: row["metrics"].__setitem__(k, -0.001))
                with self.assertRaises(M.BinderError):
                    M.build(self.policy)
                path.write_bytes(canonical)

    def test_09_every_error_metric_rejects_nonfinite(self):
        path = self.profile_path(3)
        canonical = path.read_bytes()
        for key in M.ERROR_METRIC_KEYS:
            with self.subTest(key=key):
                self.mutate_profile(3, lambda row, k=key: row["metrics"].__setitem__(k, float("nan")))
                with self.assertRaises(M.BinderError):
                    M.build(self.policy)
                path.write_bytes(canonical)

    def test_10_controlled_path_replacement_during_read_is_rejected(self):
        target = self.profile_path(2)
        original_fstat = M.os.fstat
        calls = [0]

        def replace_on_second_fstat(descriptor):
            descriptor_path = Path("/proc/self/fd/{}".format(descriptor))
            points_to_target = False
            try:
                points_to_target = descriptor_path.resolve() == target.resolve()
            except FileNotFoundError:
                pass
            if points_to_target:
                calls[0] += 1
            if points_to_target and calls[0] == 2:
                backup = target.with_name("old_profile.json")
                target.rename(backup)
                altered = json.loads(backup.read_text(encoding="utf-8"))
                altered["metrics"]["AEE"] = 9.0
                target.write_text(json.dumps(altered) + "\n", encoding="utf-8")
            return original_fstat(descriptor)

        with mock.patch.object(M.os, "fstat", side_effect=replace_on_second_fstat):
            with self.assertRaisesRegex(M.BinderError, "changed during immutable read"):
                M.build(self.policy)

    def test_11_profile_sha_and_metrics_come_from_same_bytes(self):
        result = M.build(self.policy)
        for row in result["candidate_population"]:
            profile = row["profile"]
            self.assertTrue(profile["immutable_single_read"])
            self.assertTrue(profile["hash_and_parse_same_bytes"])
            self.assertEqual(profile["sha256"], sha(Path(profile["absolute_path"])))

    def test_12_profile_symlink_is_rejected(self):
        path = self.profile_path(1)
        real = path.with_name("real.json")
        path.rename(real)
        path.symlink_to(real.name)
        with self.assertRaises(M.BinderError):
            M.build(self.policy)

    def test_13_receipt_interface_and_double_seal(self):
        output = self.base.root / "m1234_receipt"
        result = M.build(self.policy)
        M.write_receipt(output, result)
        selected = json.loads((output / "final_checkpoint_selection.json").read_text())
        self.assertEqual(selected["schema"],
                         "m1234_motion_cross_run_final_checkpoint_rebind_binder_r2_v1")
        rows = {}
        for line in (output / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(None, 1)
            rows[name.lstrip("*")] = digest
        for name, digest in rows.items():
            self.assertEqual(sha(output / name), digest)
        self.assertEqual((output / "SHA256SUMS.seal.sha256").read_text().split(),
                         [sha(output / "SHA256SUMS"), "SHA256SUMS"])

    def test_14_E0_E8_invalidation_targets_are_preserved(self):
        targets = M.build(self.policy)[
            "e0_e8_activation_dependent_invalidation_and_rebind_targets"]
        self.assertEqual([row["id"] for row in targets],
                         ["E{}".format(index) for index in range(9)])
        self.assertTrue(all("otherwise invalidate and regenerate" in row["reuse_rule"]
                            for row in targets))

    def test_15_source_only_does_not_execute_remote_gpu_or_eda(self):
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import subprocess", "paramiko", "ssh ",
                          "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertFalse(contract["claim_boundary"]["production_authorized"])
        self.assertFalse(contract["claim_boundary"]["checkpoint_selected_now"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
