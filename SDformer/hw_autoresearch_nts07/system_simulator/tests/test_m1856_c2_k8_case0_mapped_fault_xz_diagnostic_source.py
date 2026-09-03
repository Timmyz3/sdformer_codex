#!/usr/bin/env python3
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HW = Path(__file__).resolve().parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_source.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C = load(CHECKER, "m1856_diagnostic_source_checker")


class M1856DiagnosticSourceTests(unittest.TestCase):
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
        with self.assertRaises((C.CheckFailure, SyntaxError)):
            C.check(overrides)

    def test_01_actual_source_passes(self):
        result = C.check()
        self.assertEqual(result["status"], "PASS_M1856_DIAGNOSTIC_SOURCE_STATIC")
        self.assertFalse(result["launch_authorized"])
        self.assertFalse(result["paper_claim"])

    def test_02_rejects_axis_change(self):
        self.assert_rejected(self.mutated(
            "filelist", "+define+M1831_AXIS_K8", "+define+M1831_AXIS_K1X8"))

    def test_03_rejects_mapped_identity_drift(self):
        value = json.loads(self.texts["contract"])
        value["exact_diagnostic_identity"]["mapped_netlist_sha256"] = "0" * 64
        self.assert_rejected({"contract": json.dumps(value)})

    def test_04_rejects_second_compile(self):
        self.assert_rejected({"runner": self.texts["runner"] +
            '\nrun(compile_command(), WORK, WORK / "compile2.log", 7200)\n'})

    def test_05_rejects_second_sim(self):
        self.assert_rejected({"runner": self.texts["runner"] +
            '\nrun(["./simv", "-lca", "+M979_CASE=0"], WORK, WORK / "sim2.log", 1800)\n'})

    def test_06_rejects_case_change(self):
        self.assert_rejected(self.mutated(
            "runner", '"+M979_CASE=0"', '"+M979_CASE=1"'))

    def test_07_rejects_ucli(self):
        self.assert_rejected({"runner": self.texts["runner"] + '\n# "+M979_UCLI_SAIF"\n'})

    def test_08_rejects_ptpx(self):
        self.assert_rejected({"runner": self.texts["runner"] +
            "\n# /opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell\n"})

    def test_09_rejects_retry_enable(self):
        self.assert_rejected(self.mutated(
            "runner", '"automatic_retry": False', '"automatic_retry": True'))

    def test_10_rejects_missing_review_authority(self):
        self.assert_rejected(self.mutated(
            "runner", 'pin("M1856_EXPECTED_M1857_REVIEW_SHA256")', '"0" * 64'))

    def test_11_rejects_missing_release_authority(self):
        self.assert_rejected(self.mutated(
            "runner", 'pin("M1856_EXPECTED_M1858_RELEASE_SHA256")', '"0" * 64'))

    def test_12_rejects_posedge_monitor_removal(self):
        self.assert_rejected(self.mutated(
            "tb", 'print_and_localize("posedge")', '// posedge removed'))

    def test_13_rejects_negedge_monitor_removal(self):
        self.assert_rejected(self.mutated(
            "tb", 'print_and_localize("negedge")', '// negedge removed'))

    def test_14_rejects_case_equality_weakening(self):
        self.assert_rejected(self.mutated(
            "tb", "(value === 1'b0) || (value === 1'b1)",
            "(value == 1'b0) || (value == 1'b1)"))

    def test_15_rejects_public_protocol_print_removal(self):
        self.assert_rejected(self.mutated(
            "tb", "M1856_BIT name=protocol_error", "M1856_BIT name=hidden"))

    def test_16_rejects_public_stale_print_removal(self):
        self.assert_rejected(self.mutated(
            "tb", "M1856_BIT name=stale_response_seen", "M1856_BIT name=hidden"))

    def test_17_rejects_endpoint_print_removal(self):
        self.assert_rejected(self.mutated(
            "tb", "M1856_BIT name=endpoint_fault[%0d]", "M1856_BIT endpoint_hidden"))

    def test_18_rejects_internal_tap_as_decision(self):
        self.assert_rejected({"tb": self.texts["tb"] +
            "\nalways @(*) if (!is_binary(mapped_service_fault_q_tap)) $finish;\n"})

    def test_19_rejects_old_aggregate_monitor(self):
        self.assert_rejected({"filelist": self.texts["filelist"] +
            "\n/path/m1831_c2_registered_public_fault_production_assertions.sv\n"})

    def test_20_rejects_production_tb(self):
        self.assert_rejected({"filelist": self.texts["filelist"] +
            "\n/path/tb_m1831_c2_fresh_mapped_production_energy.sv\n"})

    def test_21_rejects_contract_claim_promotion(self):
        value = json.loads(self.texts["contract"])
        value["claim_boundary"]["power"] = True
        self.assert_rejected({"contract": json.dumps(value)})

    def test_22_rejects_contract_m1845_retry(self):
        value = json.loads(self.texts["contract"])
        value["claim_boundary"]["m1845_retry"] = True
        self.assert_rejected({"contract": json.dumps(value)})

    def test_23_rejects_contract_second_sim_budget(self):
        value = json.loads(self.texts["contract"])
        value["future_execution_budget"]["simv_runs_exact"] = 2
        self.assert_rejected({"contract": json.dumps(value)})

    def test_24_rejects_duplicate_contract_key(self):
        attack = self.texts["contract"].replace(
            '"milestone": "M1856",',
            '"milestone": "M1856",\n  "milestone": "M1845",', 1)
        self.assert_rejected({"contract": attack})

    def test_25_log_parser_accepts_one_precise_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.log"
            path.write_text(
                "M1856_SAMPLE time_ps=30001 sample=9 edge=negedge\n"
                "M1856_AUX mapped_protocol_error_q=x\n"
                "M1856_FIRST_NONBINARY time_ps=30001 edge=negedge "
                "name=protocol_error value=x\n")
            row = C.validate_diagnostic_log(path)
            self.assertEqual(row["name"], "protocol_error")

    def test_26_log_parser_rejects_two_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.log"
            token = ("M1856_SAMPLE time_ps=30001 sample=9 edge=negedge\n"
                     "M1856_AUX q=x\n"
                     "M1856_FIRST_NONBINARY time_ps=30001 edge=negedge "
                     "name=protocol_error value=x\n")
            path.write_text(token + token)
            with self.assertRaises(C.CheckFailure):
                C.validate_diagnostic_log(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
