#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock


HW = Path(__file__).resolve().parents[2]
DRIVER = HW / "system_simulator/scripts/execute_m1076_decoder_exact_bool_repair.py"
RUNNER = HW / "system_simulator/scripts/run_m1078_m1076_decoder_exact_bool_pilot_one_shot.sh"
CONTRACT = HW / "contracts/m1076_decoder_exact_bool_repair_contract_r1_20260830.json"
SPEC = importlib.util.spec_from_file_location("m1076_under_test", DRIVER)
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def rejected(call):
    try:
        call()
    except (RuntimeError, TypeError, ValueError, OSError):
        return True
    return False


class M1076ExactBoolTests(unittest.TestCase):
    def test_exact_tree_rejects_bool_int_alias_at_arbitrary_depth(self):
        expected = {"a": [{"b": [{"flag": False, "count": 0},
                                   {"flag": True, "count": 1}]}]}
        mutations = ((0, "flag", 0), (0, "count", False),
                     (1, "flag", 1), (1, "count", True))
        for row, key, replacement in mutations:
            value = copy.deepcopy(expected)
            value["a"][0]["b"][row][key] = replacement
            self.assertTrue(rejected(lambda value=value: M.exact_tree(value, expected)))

    def test_contract_rejects_bool_int_aliases_in_every_schema_region(self):
        attacks = [
            lambda x: x.__setitem__("launch_now", 0),
            lambda x: x["workload"].__setitem__("sample_id", False),
            lambda x: x["sampling"].__setitem__("source_census", True),
            lambda x: x["sampling"].__setitem__("selection_before_replay", 1),
            lambda x: x["pre_attempt"].__setitem__(
                "canonical_attempt_before_payload_validation", 1),
            lambda x: x["post_attempt"].__setitem__("failure_quarantine", 1),
            lambda x: x["output"].__setitem__("bool_int_alias_allowed", 0),
            lambda x: x["claim_boundary"].__setitem__("paper_citable", 0),
            lambda x: x["frozen_payload"]["selected_records"][0].__setitem__(
                "module_index", False),
        ]
        for attack in attacks:
            value = json.loads(CONTRACT.read_text(encoding="utf-8")); attack(value)
            self.assertTrue(rejected(lambda value=value: M.validate_contract(value)))

    def test_payload_receipt_rejects_top_and_deep_bool_int_aliases(self):
        context = M.synthetic_context(); receipt = M.make_payload_receipt(context)
        attacks = [
            lambda x: x.__setitem__("payload_members_verified", 1),
            lambda x: x.__setitem__("post_attempt", 1),
            lambda x: x.__setitem__("paper_citable", 0),
            lambda x: x.__setitem__("d1_scheduled", 0),
            lambda x: x["payload"]["selected_records"][0].__setitem__(
                "sample_id", False),
            lambda x: x["payload"]["selected_records"][1].__setitem__(
                "module_index", True),
        ]
        for attack in attacks:
            value = copy.deepcopy(receipt); attack(value)
            self.assertTrue(rejected(lambda value=value:
                                     M.validate_payload_receipt(value, context)))

    def test_canonical_context_rejects_nested_bool_int_aliases(self):
        context = M.synthetic_context()
        for key, replacement in (("d1_scheduled", 0), ("paper_citable", 0)):
            value = copy.deepcopy(context); value[key] = replacement
            self.assertTrue(rejected(lambda value=value:
                M.validate_canonical_context(value, context)))
        value = copy.deepcopy(context)
        value["payload"]["selected_records"][2]["sample_id"] = False
        self.assertTrue(rejected(lambda: M.validate_canonical_context(value, context)))

    def test_raw_binding_rejects_int_bool_substitution(self):
        context = M.synthetic_context()
        raw = {"layers": [{"layer": row["layer"],
            "record_identity": M.expected_raw_record(row),
            "verified_payload_member_sha256": row["payload_member_sha256"]}
            for row in context["payload"]["selected_records"]]}
        self.assertTrue(M.bind_raw_records(raw, context))
        for field, replacement in (("sample_id", False), ("module_index", True),
                                   ("timestep", False)):
            value = copy.deepcopy(raw)
            value["layers"][1]["record_identity"][field] = replacement
            self.assertTrue(rejected(lambda value=value:
                                     M.bind_raw_records(value, context)))

    def test_result_validator_rejects_bool_int_at_arbitrary_depth(self):
        rows = []
        for selected in M.synthetic_context()["payload"]["selected_records"]:
            rows.append({"layer": selected["layer"],
                "record_identity": M.expected_raw_record(selected),
                "verified_payload_member_sha256": selected["payload_member_sha256"],
                "selection_identity_sha256": "1" * 64,
                "block_population_index_sha256": "2" * 64,
                "transaction_assignment_census_sha256": "3" * 64,
                "generated_compressed_transactions": 7,
                "assigned_compressed_transactions": 7,
                "coverage": [], "source_census_cycles": {"candidate": 9,
                    "baseline": 9}, "cycle_ci_envelope": {}, "windows": [],
                "exact_mismatch_count": 0})
        raw = {"layers": rows}
        expected = M.make_result(raw, "4" * 64, "5" * 64, "6" * 64)
        attacks = [
            lambda x: x.__setitem__("d1_scheduled", 0),
            lambda x: x.__setitem__("total_window_count", False),
            lambda x: x["claim_boundary"].__setitem__("paper_citable", 0),
            lambda x: x["layers"][0].__setitem__(
                "generated_compressed_transactions", True),
            lambda x: x["layers"][2].__setitem__("exact_mismatch_count", False),
        ]
        for attack in attacks:
            value = copy.deepcopy(expected); attack(value)
            self.assertTrue(rejected(lambda value=value: M.validate_result(
                value, raw, "4" * 64, "5" * 64, "6" * 64)))

    def test_assemble_and_publish_both_call_recursive_validators(self):
        source = DRIVER.read_text(encoding="utf-8")
        completed = source[source.index("def validate_completed_work("):
                           source.index("def assemble(")]
        self.assertIn("validate_canonical_context", source)
        self.assertIn("validate_payload_receipt", source)
        self.assertIn("validate_raw(raw, context)", completed)
        self.assertIn("validate_result(result, raw", completed)
        assemble = source[source.index("def assemble("):source.index("def publish(")]
        publish = source[source.index("def publish("):source.index("def quarantine(")]
        self.assertIn("validate_completed_work", assemble)
        self.assertIn("validate_completed_work", publish)
        self.assertEqual(set(inspect.signature(M.publish).parameters),
            {"work", "result", "attempt", "runner", "contract_sha", "authority"})

    def test_m1060_identity_attacks_remain_rejected(self):
        output = M.M1060.self_test()
        self.assertEqual(output["identity_attacks_rejected"],
            ["all_fake_sha", "nonexistent_path", "relabel_and_rehash", "raw_relabel"])

    def test_pre_attempt_does_not_call_full_payload_verifier(self):
        with mock.patch.object(M.M785, "verify_sealed_directory",
                side_effect=RuntimeError("payload verifier tripwire")):
            output = M.validate_pre_attempt_source(CONTRACT, RUNNER)
        self.assertEqual(output["status"],
            "PASS_M1076_PREATTEMPT_SOURCE_WITH_ZERO_PAYLOAD_MEMBER_ACCESS")
        self.assertIs(output["payload_members_opened"], False)
        self.assertIs(output["payload_members_statted"], False)
        self.assertIs(output["payload_members_hashed"], False)

    def test_wrong_contract_pin_cannot_consume_attempt(self):
        attempt = M.RESULTS / M.ATTEMPT_NAME
        self.assertFalse(attempt.exists())
        self.assertTrue(rejected(lambda: M.consume_attempt(
            attempt, RUNNER, "0" * 64, {"synthetic": "authority"})))
        self.assertFalse(attempt.exists())

    def test_wrong_namespaces_and_direct_run_are_rejected(self):
        for role, name in (("attempt", ".m1078_wrong_attempt"),
                           ("result", "m1078_wrong_result"),
                           ("work", ".m1078_wrong_work"),
                           ("quarantine", "m1078_wrong_quarantine")):
            path = (M.RESULTS / name).resolve()
            self.assertTrue(rejected(lambda path=path, role=role:
                                     M.safe_path(path, role)))
        work = M.RESULTS / ("." + M.RESULT_NAME + ".work.m1076test")
        if work.exists(): shutil.rmtree(work)
        work.mkdir(mode=0o700)
        try:
            self.assertTrue(rejected(lambda: M.run_pilot(
                (M.RESULTS / M.ATTEMPT_NAME).resolve(), work.resolve(), RUNNER,
                M.sha256(CONTRACT), {"synthetic": "authority"})))
        finally:
            shutil.rmtree(work)

    def test_runner_order_and_context_arguments(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertLess(source.index("--consume-attempt"),
                        source.index("--validate-payload-after-attempt"))
        self.assertLess(source.index("--validate-payload-after-attempt"),
                        source.index("--run-pilot"))
        self.assertLess(source.index("--run-pilot"), source.index("--assemble"))
        self.assertLess(source.index("--assemble"), source.index("--publish"))
        for marker in ("--assemble", "--publish"):
            section = source[source.index(marker):]
            self.assertIn("--attempt", section)
            self.assertIn("--runner", section)
            self.assertIn("--expected-contract-sha", section)


if __name__ == "__main__":
    unittest.main()
