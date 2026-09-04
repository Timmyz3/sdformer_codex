import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2] / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1343_motion_ep34_live105_inventory_successor_r1.py")
SPEC = importlib.util.spec_from_file_location("m1343_live105", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class Live105SuccessorTest(unittest.TestCase):
    def inventory(self):
        result = {}
        for category, count in M.R1.EXPECTED_STATIC_COUNTS.items():
            if category == "atlif":
                result[category] = ["unit.atlif.%03d" % i for i in range(count)]
            else:
                result[category] = ["unit.%s.%03d" % (category, i) for i in range(count)]
        return result

    def with_fake_digest(self, inventory):
        original = M.EXPECTED_ATLIF_NAMES_SHA256
        M.EXPECTED_ATLIF_NAMES_SHA256 = M.inventory_digest(inventory["atlif"])
        self.addCleanup(setattr, M, "EXPECTED_ATLIF_NAMES_SHA256", original)

    def test_01_live105_inventory_is_all_259_modules(self):
        inventory = self.inventory()
        self.with_fake_digest(inventory)
        live = M.expected_live105_inventory(inventory)
        self.assertEqual(len(live["atlif"]), 105)
        self.assertEqual(sum(map(len, live.values())), 259)

    def test_02_atlif_count_mutation_rejected(self):
        inventory = self.inventory()
        self.with_fake_digest(inventory)
        inventory["atlif"].pop()
        with self.assertRaisesRegex(M.M1343Error, "count drift"):
            M.expected_live105_inventory(inventory)

    def test_03_atlif_name_digest_mutation_rejected(self):
        inventory = self.inventory()
        self.with_fake_digest(inventory)
        inventory["atlif"][0] = "unit.atlif.changed"
        with self.assertRaisesRegex(M.M1343Error, "name-set SHA"):
            M.expected_live105_inventory(inventory)

    def test_04_snv_name_rejected_even_if_digest_matches(self):
        inventory = self.inventory()
        inventory["atlif"][0] = "unit.attn.sn_v.spiking_neuron"
        self.with_fake_digest(inventory)
        with self.assertRaisesRegex(M.M1343Error, "sn_v"):
            M.expected_live105_inventory(inventory)

    def test_05_non_atlif_count_mutation_rejected(self):
        inventory = self.inventory()
        self.with_fake_digest(inventory)
        inventory["fc1"].pop()
        with self.assertRaisesRegex(M.M1343Error, "count drift"):
            M.expected_live105_inventory(inventory)

    def test_06_patch_context_changes_only_inventory_validators_and_result(self):
        before = (M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
                  M.R1.expected_live_inventory, M.R1.validate_snapshot_population,
                  M.R1.final_validate_and_seal, M.M1249.CANONICAL_RESULT)
        with M.patched_live105_capture_chain():
            self.assertEqual(M.R1.DEAD_SN_V, ())
            self.assertEqual(M.R1.EXPECTED_LIVE_COUNTS, M.R1.EXPECTED_STATIC_COUNTS)
            self.assertIs(M.R1.expected_live_inventory, M.expected_live105_inventory)
            self.assertIs(M.R1.validate_snapshot_population,
                          M.validate_snapshot_population_live105)
            self.assertIs(M.R1.final_validate_and_seal, M.final_validate_and_seal_live105)
            self.assertEqual(M.M1249.CANONICAL_RESULT, M.CANONICAL_RESULT)
        after = (M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
                 M.R1.expected_live_inventory, M.R1.validate_snapshot_population,
                 M.R1.final_validate_and_seal, M.M1249.CANONICAL_RESULT)
        self.assertEqual(before, after)

    def test_07_patch_context_restores_after_exception(self):
        original = M.R1.expected_live_inventory
        with self.assertRaisesRegex(RuntimeError, "attack"):
            with M.patched_live105_capture_chain():
                raise RuntimeError("attack")
        self.assertIs(M.R1.expected_live_inventory, original)
        self.assertEqual(len(M.R1.DEAD_SN_V), 12)

    def test_08_failed_m1329_attempt_and_log_are_exact(self):
        M.verify_m1329_failure()
        self.assertEqual(M.sha256(M.FAILED_ATTEMPT), M.FAILED_ATTEMPT_SHA256)
        self.assertEqual(M.sha256(M.FAILED_TEMP_LOG), M.FAILED_TEMP_LOG_SHA256)

    def test_09_runtime_rekeys_only_contract_and_output(self):
        old_runtime = {"contract_path": "old.json",
                       "capture": {"attention_windows_per_call": 100},
                       "cohort": {"samples": ["sealed"]},
                       "output": {"path": "old-result"}}
        old_binding = {"identity": {"checkpoint_sha256": M.CHECKPOINT_SHA256,
                                    "config_sha256": M.CONFIG_SHA256}}
        with mock.patch.object(M.M1327, "validate_identity_and_project",
                               return_value=(old_runtime, old_binding)):
            runtime, binding = M.build_runtime()
        self.assertEqual(set(runtime), {"contract_path", "capture", "cohort", "output"})
        self.assertEqual(runtime["capture"], {"attention_windows_per_call": 100})
        self.assertEqual(runtime["output"],
                         {"path": str(M.CANONICAL_RESULT.relative_to(M.ROOT))})
        self.assertEqual(binding["identity"]["checkpoint_sha256"], M.CHECKPOINT_SHA256)
        self.assertEqual(binding["identity"]["config_sha256"], M.CONFIG_SHA256)

    def test_10_source_has_live105_constants_and_no_production_cli(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("EXPECTED_ORDERED_RECORDS = 10360", text)
        self.assertIn("EXPECTED_ATLIF_ROWS = 105", text)
        self.assertNotIn("os.O_CREAT | os.O_EXCL, 0o400", text)
        self.assertNotIn("--run", text)

    def test_11_new_namespaces_are_distinct_and_fresh(self):
        M.require_fresh_namespaces()
        self.assertEqual(len({M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG}), 3)

    def test_12_predecessor_globals_remain_original_outside_context(self):
        self.assertEqual(len(M.R1.DEAD_SN_V), 12)
        self.assertEqual(M.R1.EXPECTED_LIVE_COUNTS["atlif"], 93)
        self.assertNotEqual(M.R1.expected_live_inventory, M.expected_live105_inventory)


if __name__ == "__main__":
    unittest.main()
