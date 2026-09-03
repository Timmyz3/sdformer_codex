#!/usr/bin/env python3
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HW = Path(__file__).resolve().parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1862_c2_k8_case0_mapped_fault_xz_diagnostic_source.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C = load(CHECKER, "m1862_diagnostic_source_checker")


class M1862DiagnosticSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = C.source_map()

    def mutated(self, name, old, new):
        self.assertIn(old, self.texts[name])
        return {name: self.texts[name].replace(old, new, 1)}

    def assert_rejected(self, overrides):
        overrides = dict(overrides)
        changed = [name for name in overrides
                   if name != "contract" and name in C.PATHS]
        if changed:
            contract = json.loads(overrides.get("contract", self.texts["contract"]))
            for name in changed:
                rel = C.PATHS[name].relative_to(HW).as_posix()
                contract["source_files"][rel] = hashlib.sha256(
                    overrides[name].encode()).hexdigest()
            overrides["contract"] = json.dumps(contract, sort_keys=True)
        with self.assertRaises((C.CheckFailure, SyntaxError, ValueError)):
            C.check(overrides)

    def test_01_actual_source_passes(self):
        result = C.check()
        self.assertEqual(result["status"],
                         "PASS_M1862_DIAGNOSTIC_SUCCESSOR_SOURCE_STATIC")
        self.assertFalse(result["launch_authorized"])

    # The exact twelve M1857 synchronized-inventory escapes.
    def test_02_rejects_main_authority_call_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "        release_sha = verify_authority()",
            '        release_sha = "0" * 64'))

    def test_03_rejects_attempt_namespace_retarget(self):
        self.assert_rejected(self.mutated(
            "runner",
            "results/.m1862_c2_k8_case0_mapped_fault_xz_diagnostic_attempt_consumed",
            "results/.m1845_c2_fresh_mapped_production_energy_attempt_consumed"))

    def test_04_rejects_result_namespace_retarget(self):
        self.assert_rejected(self.mutated(
            "runner", "results/m1862_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902",
            "results/m1845_c2_fresh_mapped_production_energy_r1_20260902"))

    def test_05_rejects_first_freshness_removal(self):
        self.assert_rejected(self.mutated(
            "runner",
            "        namespaces_fresh()\n        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)",
            "        # first freshness removed\n        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)"))

    def test_06_rejects_second_freshness_removal(self):
        self.assert_rejected(self.mutated(
            "runner",
            "        collision_gate()\n        namespaces_fresh()\n        ATTEMPT.mkdir()",
            "        collision_gate()\n        # second freshness removed\n        ATTEMPT.mkdir()"))

    def test_07_rejects_global_lock_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)",
            "        # global queue lock removed"))

    def test_08_rejects_local_lock_removal(self):
        self.assert_rejected(self.mutated(
            "runner",
            "        fcntl.flock(local_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
            "        # local lock removed"))

    def test_09_rejects_runtime_collision_removal(self):
        self.assert_rejected(self.mutated(
            "runner", "    CHECK.validate_sources()\n    collision_gate()\n    env = {",
            "    CHECK.validate_sources()\n    # runtime collision removed\n    env = {"))

    def test_10_rejects_result_claim_erasure(self):
        self.assert_rejected(self.mutated(
            "runner", '            "claim_boundary": CHECK.CLAIMS})',
            '            "claim_boundary": {}})'))

    def test_11_rejects_first_stop_removal(self):
        self.assert_rejected(self.mutated(
            "tb", "            $finish;", "            ;"))

    def test_12_rejects_first_token_value_disconnect(self):
        self.assert_rejected(self.mutated(
            "tb", "edge_name, core.protocol_error);",
            "edge_name, 1'bx);"))

    def test_13_rejects_contract_mapped_path_drift(self):
        value = json.loads(self.texts["contract"])
        value["exact_diagnostic_identity"]["mapped_netlist"] = "evil.v"
        self.assert_rejected({"contract": json.dumps(value)})

    # Additional structural/AST and unique-count gates.
    def test_14_rejects_authority_call_in_wrong_function(self):
        attack = self.texts["runner"].replace(
            "        release_sha = verify_authority()",
            '        release_sha = "0" * 64', 1)
        attack += "\ndef decoy():\n    return verify_authority()\n"
        self.assert_rejected({"runner": attack})

    def test_15_rejects_second_compile(self):
        self.assert_rejected(self.mutated(
            "runner", "        run(compile_command(), WORK, WORK / \"compile.log\", 7200)",
            "        run(compile_command(), WORK, WORK / \"compile.log\", 7200)\n"
            "        run(compile_command(), WORK, WORK / \"compile2.log\", 7200)"))

    def test_16_rejects_second_sim(self):
        self.assert_rejected(self.mutated(
            "runner", '        run(["./simv", "-lca", "+M979_CASE=0"], WORK,',
            '        run(["./simv", "-lca", "+M979_CASE=0"], WORK,\n'
            '            WORK / "extra.log", 1800)\n'
            '        run(["./simv", "-lca", "+M979_CASE=0"], WORK,'))

    def test_17_rejects_case_change(self):
        self.assert_rejected(self.mutated(
            "runner", '"+M979_CASE=0"', '"+M979_CASE=1"'))

    def test_18_rejects_ucli(self):
        self.assert_rejected(self.mutated(
            "runner", '"+M979_CASE=0"', '"+M979_CASE=0", "+M979_UCLI_SAIF"'))

    def test_19_rejects_axis_change(self):
        self.assert_rejected(self.mutated(
            "filelist", "+define+M1831_AXIS_K8", "+define+M1831_AXIS_K1X8"))

    def test_20_rejects_weakened_case_equality(self):
        self.assert_rejected(self.mutated(
            "tb", "(value === 1'b0) || (value === 1'b1)",
            "(value == 1'b0) || (value == 1'b1)"))

    def test_21_rejects_internal_tap_decision(self):
        self.assert_rejected(self.mutated(
            "tb", "if (!is_binary(core.protocol_error))",
            "if (!is_binary(mapped_protocol_error_q_tap))"))

    def test_22_rejects_failure_namespace_retarget(self):
        self.assert_rejected(self.mutated(
            "runner", "results/m1862_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902.failed_or_incomplete.quarantine",
            "results/m1845_c2_fresh_mapped_production_energy_r1_20260902.failed_or_incomplete.quarantine"))

    def test_23_rejects_queue_namespace_retarget(self):
        self.assert_rejected(self.mutated(
            "runner", "/tmp/date_dual_synopsys_same_uid_eda_queue.lock",
            "/tmp/private_queue.lock"))

    def test_24_rejects_parser_removal(self):
        self.assert_rejected(self.mutated(
            "runner", '        result = CHECK.validate_diagnostic_log(WORK / "diagnostic.log")',
            "        result = {}"))

    def test_25_rejects_publication_retarget(self):
        self.assert_rejected(self.mutated(
            "runner", "        publish_no_replace(STAGE, RESULT)",
            "        publish_no_replace(STAGE, FAILURE)"))

    def test_26_rejects_wrong_endpoint_token_value(self):
        self.assert_rejected(self.mutated(
            "tb", "edge_name, bank, endpoint_fault[bank]);",
            "edge_name, bank, 1'bx);"))

    def test_27_rejects_contract_filelist_path_drift(self):
        value = json.loads(self.texts["contract"])
        value["exact_diagnostic_identity"]["filelist"] = "evil.f"
        self.assert_rejected({"contract": json.dumps(value)})

    def test_28_rejects_contract_claim_promotion(self):
        value = json.loads(self.texts["contract"])
        value["claim_boundary"]["power"] = True
        self.assert_rejected({"contract": json.dumps(value)})

    def test_29_rejects_old_m1858_release_label(self):
        self.assert_rejected({"runner": self.texts["runner"] + "\n# M1858\n"})

    def test_30_log_parser_accepts_bound_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.log"
            path.write_text(
                "M1862_SAMPLE time_ps=30001 sample=9 edge=negedge\n"
                "M1862_AUX q=x\n"
                "M1862_FIRST_NONBINARY time_ps=30001 edge=negedge "
                "name=protocol_error value=x\n")
            row = C.validate_diagnostic_log(path)
            self.assertEqual(row["name"], "protocol_error")
            self.assertEqual(row["value"], "x")

    def test_31_log_parser_rejects_missing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.log"
            path.write_text(
                "M1862_SAMPLE time_ps=30001 sample=9 edge=negedge\n"
                "M1862_AUX q=x\n"
                "M1862_FIRST_NONBINARY time_ps=30001 edge=negedge "
                "name=protocol_error\n")
            with self.assertRaises(C.CheckFailure):
                C.validate_diagnostic_log(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
