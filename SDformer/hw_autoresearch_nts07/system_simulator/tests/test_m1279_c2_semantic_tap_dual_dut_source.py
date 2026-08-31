from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" /
          "check_m1279_c2_semantic_tap_dual_dut_source.py")
SPEC = importlib.util.spec_from_file_location("m1279_checker", SCRIPT)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1279SourceTests(unittest.TestCase):
    def test_01_baseline_source_and_contract(self):
        receipt = M.run_checks()
        self.assertEqual(receipt["status"],
                         "PASS_M1279_SOURCE_ONLY__NO_EXECUTION_AUTHORIZED")
        self.assertEqual(receipt["tap_topology"]["tap_count"], 13)
        self.assertEqual(receipt["dual_dut"]["dual_dut_instances"], 2)
        self.assertEqual(receipt["real_tool_calls"], 0)

    def test_02_all_seven_clones_are_functionally_frozen(self):
        rows = M.check_clone_equivalence()
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(len(row["successor_sha256"]) == 64 for row in rows))

    def test_03_non_tap_functional_mutation_is_visible(self):
        successor_name, frozen_name = M.CLONES[0]
        successor = (M.HW / successor_name).read_text(encoding="utf-8")
        frozen = (M.HW / frozen_name).read_text(encoding="utf-8")
        attacked = successor.replace("raw_ready", "raw_ready_attacked", 1)
        self.assertNotEqual(M.functional_normal_form(attacked),
                            M.functional_normal_form(frozen))

    def test_04_exact_thirteen_tap_set(self):
        topology = M.check_tap_topology()
        self.assertEqual(tuple(topology["tap_names"]), M.TAPS)
        self.assertFalse(topology["hierarchical_binding"])
        self.assertFalse(topology["functional_fanin_from_taps"])

    def test_05_endpoint_unknown_request_is_fail_closed(self):
        endpoint = (M.HW / "dc_handoff/tb/m1279_valid_qualified_scalar_bank_endpoint.sv").read_text(encoding="utf-8")
        self.assertIn("mem_req_ready = 1'b0", endpoint)
        self.assertIn("endpoint_protocol_fault_now = 1'b1", endpoint)
        attacked = endpoint.replace("mem_req_ready = 1'b0", "mem_req_ready = inner_req_ready", 1)
        self.assertNotIn("mem_req_ready = 1'b0", attacked)

    def test_06_prohibited_mechanisms_absent(self):
        endpoint = (M.HW / "dc_handoff/tb/m1279_valid_qualified_scalar_bank_endpoint.sv").read_text(encoding="utf-8")
        tb = (M.HW / "dc_handoff/tb/tb_m1279_c2_dual_dut_valid_qualified_endpoint.sv").read_text(encoding="utf-8")
        subject = M.strip_comments(endpoint + "\n" + tb)
        for pattern in (r"\bforce\b", r"\brelease\b", r"\+?initreg",
                        r"\bcasex\b", r"\bcasez\b", r"set_case_analysis",
                        r"=\s*1'b[xXzZ]"):
            self.assertIsNone(re.search(pattern, subject, flags=re.I))
        self.assertIsNotNone(re.search(r"\bforce\b", subject + "\nforce a=0;"))

    def test_07_atomic_dual_dut_window_is_executable_source(self):
        row = M.check_endpoint_and_tb()
        self.assertEqual(row["semantic_taps_per_dut"], 13)
        self.assertEqual(row["atomic_bitmap_bits"], 32)
        self.assertEqual(row["window_cycles"], 128)
        self.assertEqual(row["filelist_members"], 11)

    def test_08_contract_duplicate_key_and_claim_elevation_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m1279_test_") as temp:
            duplicate = Path(temp) / "duplicate.json"
            duplicate.write_text('{"schema":1,"schema":2}', encoding="utf-8")
            with self.assertRaises(M.Failure):
                M.strict_json(duplicate)

            data = M.strict_json(M.CONTRACT)
            data["claim_boundary"]["power"] = True
            elevated = Path(temp) / "elevated.json"
            elevated.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            with mock.patch.object(M, "CONTRACT", elevated):
                with self.assertRaises(M.Failure):
                    M.check_contract()


if __name__ == "__main__":
    unittest.main()
