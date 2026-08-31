import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1434_motion_ep34_live93_runtime_successor_r1.py")
SPEC = importlib.util.spec_from_file_location("m1434_live93", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class Live93RuntimeSuccessorTest(unittest.TestCase):
    def static_inventory(self):
        policy = M.R1.strict_json(M.R1.SOURCE_CONTRACT)
        result = M.R1.frozen_non_atlif_inventory(policy)
        result["atlif"] = list(M.M1349.EXPECTED_ATLIF_NAMES)
        return result

    def live_inventory(self):
        return M.expected_live93_inventory(self.static_inventory())

    def sample_rows(self):
        return [
            {"global_sample_id": 0, "category": category, "name": name}
            for category, names in self.live_inventory().items() for name in names
        ]

    def test_01_static105_live93_exact(self):
        live = self.live_inventory()
        self.assertEqual(len(M.M1349.EXPECTED_ATLIF_NAMES), 105)
        self.assertEqual(len(live["atlif"]), 93)
        self.assertEqual(sum(map(len, live.values())), 247)

    def test_02_dead_sn2q_exact_set_and_digest(self):
        self.assertEqual(len(M.DEAD_SN2_Q), 12)
        self.assertEqual(M.terminal_lf_digest(M.DEAD_SN2_Q), M.DEAD_SN2_Q_SHA256)
        self.assertTrue(all(".attn.sn2_q.spiking_neuron" in n for n in M.DEAD_SN2_Q))

    def test_03_live93_digest_exact(self):
        self.assertEqual(M.terminal_lf_digest(self.live_inventory()["atlif"]),
                         M.LIVE_ATLIF_SHA256)

    def test_04_real_sample0_failure_observation_bound(self):
        observed = M.validate_failure_observation()
        self.assertEqual(observed["sample0_call_audit"]["records"], 247)
        self.assertEqual(observed["sample0_call_audit"]["errors"],
                         M.expected_failure_errors())

    def test_05_sample0_forensic_summary_replays_pass247(self):
        self.assertEqual(M.replay_sample0_forensic_summary(), {
            "status": "PASS", "errors": [], "samples": 1,
            "live_modules_per_sample": 247, "records": 247,
            "expected_records": 247, "dead_modules": 12,
        })

    def test_06_sample0_exact_call_matrix_replays_pass(self):
        audit = M.audit_with_live93(self.sample_rows(), self.live_inventory(), [0])
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["records"], 247)

    def test_07_dead_called_attack_fails(self):
        rows = self.sample_rows() + [{
            "global_sample_id": 0, "category": "atlif", "name": M.DEAD_SN2_Q[0]}]
        audit = M.audit_with_live93(rows, self.live_inventory(), [0])
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any(x.startswith("dead_module_fired:") for x in audit["errors"]))

    def test_08_missing_live_attack_fails(self):
        audit = M.audit_with_live93(self.sample_rows()[:-1], self.live_inventory(), [0])
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any(x.startswith("call_count:") for x in audit["errors"]))

    def test_09_duplicate_live_attack_fails(self):
        rows = self.sample_rows()
        audit = M.audit_with_live93(rows + [dict(rows[0])], self.live_inventory(), [0])
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any(x.endswith(":2") for x in audit["errors"]))

    def test_10_wrong_category_attack_fails(self):
        rows = self.sample_rows()
        rows[0] = dict(rows[0], category="attention")
        audit = M.audit_with_live93(rows, self.live_inventory(), [0])
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any(x.startswith("unexpected_name_or_category:")
                            for x in audit["errors"]))

    def test_11_dead_set_rename_rejected(self):
        inventory = self.static_inventory()
        index = inventory["atlif"].index(M.DEAD_SN2_Q[0])
        inventory["atlif"][index] += ".renamed"
        with self.assertRaisesRegex(M.M1434Error, "authority|dead sn2_q"):
            M.expected_live93_inventory(inventory)

    def test_12_dead_set_deletion_rejected(self):
        inventory = self.static_inventory()
        inventory["atlif"].remove(M.DEAD_SN2_Q[0])
        inventory["atlif"].append("fake.sn_q.spiking_neuron")
        inventory["atlif"].sort()
        with self.assertRaisesRegex(M.M1434Error, "authority|dead sn2_q"):
            M.expected_live93_inventory(inventory)

    def test_13_dead_set_extra_rejected(self):
        inventory = self.static_inventory()
        live_name = next(n for n in inventory["atlif"] if n not in M.DEAD_SN2_Q)
        inventory["atlif"][inventory["atlif"].index(live_name)] = (
            live_name + ".sn2_q.spiking_neuron")
        inventory["atlif"].sort()
        with self.assertRaisesRegex(M.M1434Error, "authority|dead sn2_q"):
            M.expected_live93_inventory(inventory)

    def test_14_failure_observation_mutation_rejected(self):
        policy = M.strict_json(M.SOURCE_CONTRACT)
        policy["m1400_failure_observation"]["sample0_call_audit"]["records"] = 248
        with self.assertRaisesRegex(M.M1434Error, "failure observation"):
            M.validate_failure_observation(policy)

    def test_15_h60_source_is_pinned_and_bypasses_sn2q(self):
        M.validate_h60_bypass_source()

    def test_16_patch_applies_exact_runtime_projection(self):
        old = (M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
               M.R1.expected_live_inventory, M.M1249.CANONICAL_RESULT)
        with M.patched_live93_capture_chain():
            self.assertEqual(M.R1.DEAD_SN_V, M.DEAD_SN2_Q)
            self.assertEqual(M.R1.EXPECTED_LIVE_COUNTS["atlif"], 93)
            self.assertIs(M.R1.expected_live_inventory, M.expected_live93_inventory)
            self.assertEqual(M.M1249.CANONICAL_RESULT, M.CANONICAL_RESULT)
        self.assertEqual((M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
                          M.R1.expected_live_inventory, M.M1249.CANONICAL_RESULT), old)

    def test_17_patch_restores_every_global_after_failure(self):
        old = (M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
               M.R1.expected_live_inventory, M.R1.validate_snapshot_population,
               M.R1.final_validate_and_seal, M.M1249.CANONICAL_RESULT)
        with self.assertRaisesRegex(RuntimeError, "attack"):
            with M.patched_live93_capture_chain():
                raise RuntimeError("attack")
        self.assertEqual((M.R1.DEAD_SN_V, M.R1.EXPECTED_LIVE_COUNTS,
                          M.R1.expected_live_inventory, M.R1.validate_snapshot_population,
                          M.R1.final_validate_and_seal, M.M1249.CANONICAL_RESULT), old)

    def test_18_audit_helper_restores_dead_set_after_failure(self):
        old = M.R1.DEAD_SN_V
        with mock.patch.object(M.R1, "audit_call_matrix", side_effect=RuntimeError("audit")):
            with self.assertRaisesRegex(RuntimeError, "audit"):
                M.audit_with_live93([], self.live_inventory(), [0])
        self.assertEqual(M.R1.DEAD_SN_V, old)

    def test_19_runtime_rebinds_only_contract_and_output(self):
        old_runtime = {"contract_path": "old",
                       "capture": {"attention_windows_per_call": 100},
                       "cohort": {"samples": ["sealed"]},
                       "output": {"path": "old"}}
        binding = {"identity": {"checkpoint_sha256": M.CHECKPOINT_SHA256,
                                "config_sha256": M.CONFIG_SHA256}}
        with mock.patch.object(M.M1349, "build_runtime",
                               return_value=(old_runtime, binding)):
            runtime, rebound = M.build_runtime()
        self.assertEqual(runtime["capture"], old_runtime["capture"])
        self.assertEqual(runtime["cohort"], old_runtime["cohort"])
        self.assertEqual(rebound, binding)
        self.assertEqual(runtime["output"]["path"],
                         str(M.CANONICAL_RESULT.relative_to(M.ROOT)))

    def test_20_fresh_namespaces_and_collisions(self):
        M.require_fresh_namespaces()
        with mock.patch.object(M.os.path, "lexists", return_value=True):
            with self.assertRaisesRegex(M.M1434Error, "not fresh"):
                M.require_fresh_namespaces()

    def test_21_source_has_no_production_cli_or_control_action(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--run"', text)
        self.assertNotIn("os.kill", text)
        self.assertNotIn("SIGCONT", text)
        self.assertNotIn("torch.cuda", text)
        self.assertNotIn("subprocess", text)

    def test_22_population_arithmetic(self):
        self.assertEqual(40 * 247, 9880)
        self.assertEqual(M.EXPECTED_ORDERED_RECORDS, 9880)
        self.assertEqual(M.EXPECTED_ATTENTION, 480)
        self.assertEqual(M.EXPECTED_PAYLOAD, 640)


if __name__ == "__main__":
    unittest.main()
