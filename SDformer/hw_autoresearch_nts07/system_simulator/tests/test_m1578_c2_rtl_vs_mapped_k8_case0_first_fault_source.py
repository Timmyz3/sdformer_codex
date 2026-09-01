#!/usr/bin/env python3
"""Mutation tests for the source-only M1578 diagnostic gate."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = (Path(__file__).resolve().parents[2] / "dc_handoff/scripts/"
          "run_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.py")
SPEC = importlib.util.spec_from_file_location("m1578_source_gate", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class M1578SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = M.TB.read_text(encoding="utf-8")
        cls.contract = json.loads(M.CONTRACT.read_text(encoding="utf-8"))

    def rejects(self, function):
        with self.assertRaises(M.Failure):
            function()

    def test_description_and_static_baseline(self):
        description = M.describe()
        self.assertEqual(description["case"], "exact M979 K8 case0")
        self.assertEqual(description["execution"]["vcs_compiles"], 0)
        result = M.static_check()
        self.assertEqual(result["status"],
                         "PASS_SOURCE_ONLY_READY_FOR_INDEPENDENT_HAMMER__NO_TOOL_RUN")
        self.assertFalse(result["claim"])
        self.assertEqual(result["tb"]["dut_instances"], 2)

    def test_comment_cannot_replace_active_tap(self):
        token = "mapped_internal_fault_taps"
        attacked = self.tb.replace(token, "mapped_taps_removed")
        attacked += "\n// " + token + "\n"
        self.rejects(lambda: M.verify_tb_text(attacked))

    def test_x_to_zero_coercion_rejected(self):
        attacked = self.tb.replace('else tri = "X"', 'else tri = "0"')
        self.rejects(lambda: M.verify_tb_text(attacked))

    def test_single_dut_rejected(self):
        attacked = self.tb.replace(") rtl_dut (", ") removed_rtl_dut (")
        self.rejects(lambda: M.verify_tb_text(attacked))

    def test_first_difference_or_event_removed_rejected(self):
        attacked = self.tb.replace("FIRST_RTL_MAPPED_DIFFERENCE",
                                   "REMOVED_FIRST_DIFFERENCE")
        self.rejects(lambda: M.verify_tb_text(attacked))
        attacked = self.tb.replace("source=%s/%s", "source_removed=%s/%s")
        self.rejects(lambda: M.verify_tb_text(attacked))

    def test_prohibited_mechanisms_rejected(self):
        for injection in ("force rtl_protocol_error = 1'b0;",
                          "$stop;", "initial $display(\"UCLI\");",
                          "initial $display(\"SAIF\");"):
            self.rejects(lambda injection=injection:
                         M.verify_tb_text(self.tb + "\n" + injection + "\n"))

    def test_claim_promotions_rejected(self):
        for field in ("paper_citable", "rtl_pass", "mapped_pass", "power",
                      "ppa", "system_speedup", "headline"):
            attacked = copy.deepcopy(self.contract)
            attacked["claim_boundary"][field] = True
            self.rejects(lambda attacked=attacked: M.validate_contract_obj(attacked))
        attacked = copy.deepcopy(self.contract)
        attacked["execution"]["simv_runs"] = 1
        self.rejects(lambda: M.validate_contract_obj(attacked))

    def test_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m1578_duplicate_json.") as root:
            path = Path(root) / "bad.json"
            path.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
            self.rejects(lambda: M.strict_json(path))

    def test_runner_has_no_execution_primitive(self):
        source = SOURCE.read_text(encoding="utf-8")
        for token in ("subprocess", "os.system", "Popen(", "execv(",
                      "vcs -", "./simv"):
            self.assertNotIn(token, source)
        self.assertNotIn("--run", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
