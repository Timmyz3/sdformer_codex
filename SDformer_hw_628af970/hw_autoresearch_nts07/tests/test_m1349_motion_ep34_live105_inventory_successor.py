import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1349_motion_ep34_live105_inventory_successor_r2.py")
SPEC = importlib.util.spec_from_file_location("m1349_live105", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class Live105AuthoritySuccessorTest(unittest.TestCase):
    def authority(self):
        return M.strict_json(M.M1347_INVENTORY)

    def inventory(self):
        result = {}
        for category, count in M.R1.EXPECTED_STATIC_COUNTS.items():
            if category == "atlif":
                result[category] = list(self.authority()["atlif_names"])
            else:
                result[category] = [f"unit.{category}.{index:03d}" for index in range(count)]
        return result

    def test_01_sealed_real_list_is_directly_consumed(self):
        names = M.verify_m1347_failure()
        self.assertEqual(names, tuple(self.authority()["atlif_names"]))
        self.assertEqual(len(names), 105)

    def test_02_terminal_lf_digest_is_exact(self):
        names = self.authority()["atlif_names"]
        self.assertEqual(M.terminal_lf_digest(names), M.EXPECTED_ATLIF_NAMES_SHA256)

    def test_03_valid_real_inventory_has_259_live_modules(self):
        live = M.expected_live105_inventory(self.inventory())
        self.assertEqual(len(live["atlif"]), 105)
        self.assertEqual(sum(map(len, live.values())), 259)

    def test_04_rename_rejected(self):
        inventory = self.inventory()
        inventory["atlif"][0] += ".renamed"
        with self.assertRaisesRegex(M.M1349Error, "sealed ordered"):
            M.expected_live105_inventory(inventory)

    def test_05_reorder_rejected(self):
        inventory = self.inventory()
        inventory["atlif"][0], inventory["atlif"][1] = (
            inventory["atlif"][1], inventory["atlif"][0])
        with self.assertRaisesRegex(M.M1349Error, "sealed ordered"):
            M.expected_live105_inventory(inventory)

    def test_06_duplicate_rejected(self):
        remote = self.authority()
        remote["atlif_names"][1] = remote["atlif_names"][0]
        with self.assertRaisesRegex(M.M1349Error, "sorted|unique|digest"):
            M.validate_authority_payload(remote)

    def test_07_deleted_name_rejected(self):
        remote = self.authority()
        remote["atlif_names"].pop()
        with self.assertRaisesRegex(M.M1349Error, "105 strings"):
            M.validate_authority_payload(remote)

    def test_08_snv_zero_and_load_exact(self):
        remote = self.authority()
        self.assertEqual(remote["inventory"]["sn_v_count"], 0)
        self.assertEqual(remote["load_audit"], {"missing": 0, "unexpected": 0})

    def test_09_two_or_more_rebuilds_required(self):
        remote = self.authority()
        remote["repeatability"]["rebuilds"] = 1
        with self.assertRaisesRegex(M.M1349Error, "two identical"):
            M.validate_authority_payload(remote)

    def test_10_checkpoint_config_profile_overlay_bound(self):
        remote = self.authority()
        remote["identity"]["profile_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(M.M1349Error, "binding"):
            M.validate_authority_payload(remote)

    def test_11_review_sha_mutation_rejected(self):
        with self.assertRaisesRegex(M.M1349Error, "SHA mismatch"):
            M.regular_exact(M.M1347_REVIEW, "0" * 64, "mutated review")

    def test_12_outer_seal_sha_mutation_rejected(self):
        with self.assertRaisesRegex(M.M1349Error, "SHA mismatch"):
            M.regular_exact(M.M1347_OUTER, "f" * 64, "mutated outer seal")

    def test_13_contract_test_extra_key_rejected(self):
        policy = M.strict_json(M.SOURCE_CONTRACT)
        policy["test"]["extra"] = True
        with self.assertRaisesRegex(M.M1349Error, "test schema"):
            M.validate_source_policy(policy)

    def test_14_contract_exact_test_projection_passes(self):
        policy = M.validate_source_policy()
        self.assertEqual(set(policy["test"]), {"path", "sha256", "passed", "failed"})

    def test_15_patch_context_changes_only_six_targets(self):
        before = (M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
                  M.R1.expected_live_inventory, M.R1.validate_snapshot_population,
                  M.R1.final_validate_and_seal, M.M1249.CANONICAL_RESULT)
        with M.patched_live105_capture_chain():
            self.assertEqual(M.R1.DEAD_SN_V, ())
            self.assertIs(M.R1.expected_live_inventory, M.expected_live105_inventory)
            self.assertEqual(M.M1249.CANONICAL_RESULT, M.CANONICAL_RESULT)
        after = (M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
                 M.R1.expected_live_inventory, M.R1.validate_snapshot_population,
                 M.R1.final_validate_and_seal, M.M1249.CANONICAL_RESULT)
        self.assertEqual(before, after)

    def test_16_patch_context_restores_after_exception(self):
        before = (M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
                  M.R1.expected_live_inventory, M.R1.validate_snapshot_population,
                  M.R1.final_validate_and_seal, M.M1249.CANONICAL_RESULT)
        with self.assertRaisesRegex(RuntimeError, "attack"):
            with M.patched_live105_capture_chain():
                raise RuntimeError("attack")
        after = (M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
                 M.R1.expected_live_inventory, M.R1.validate_snapshot_population,
                 M.R1.final_validate_and_seal, M.M1249.CANONICAL_RESULT)
        self.assertEqual(before, after)

    def test_17_namespaces_are_fresh_and_collisions_reject(self):
        M.require_fresh_namespaces()
        with mock.patch.object(M.os.path, "lexists", return_value=True):
            with self.assertRaisesRegex(M.M1349Error, "not fresh"):
                M.require_fresh_namespaces()

    def test_18_checkpoint_binding_mutation_rejected(self):
        runtime = {"contract_path": "old", "capture": {"attention_windows_per_call": 100},
                   "cohort": {"samples": ["sealed"]}, "output": {"path": "old"}}
        binding = {"identity": {"checkpoint_sha256": "0" * 64,
                                "config_sha256": M.CONFIG_SHA256}}
        with mock.patch.object(M.M1327, "validate_identity_and_project",
                               return_value=(runtime, binding)):
            with self.assertRaisesRegex(M.M1349Error, "identity drift"):
                M.build_runtime()

    def test_19_runtime_preserves_capture_and_cohort(self):
        runtime = {"contract_path": "old", "capture": {"attention_windows_per_call": 100},
                   "cohort": {"samples": ["sealed"]}, "output": {"path": "old"}}
        binding = {"identity": {"checkpoint_sha256": M.CHECKPOINT_SHA256,
                                "config_sha256": M.CONFIG_SHA256}}
        with mock.patch.object(M.M1327, "validate_identity_and_project",
                               return_value=(runtime, binding)):
            projected, rebound = M.build_runtime()
        self.assertEqual(projected["capture"], runtime["capture"])
        self.assertEqual(projected["cohort"], runtime["cohort"])
        self.assertEqual(rebound, binding)

    def test_20_source_is_strictly_nonproduction(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("--run", text)
        self.assertNotIn("O_CREAT | os.O_EXCL", text)
        self.assertNotIn("torch.cuda", text)
        self.assertEqual(M.EXPECTED_RETAINED, 320)
        self.assertEqual(M.EXPECTED_ATTENTION, 480)
        self.assertEqual(M.EXPECTED_PAYLOAD, 640)


if __name__ == "__main__":
    unittest.main()
