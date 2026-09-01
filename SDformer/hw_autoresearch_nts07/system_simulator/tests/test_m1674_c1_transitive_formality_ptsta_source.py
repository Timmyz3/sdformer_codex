#!/usr/bin/env python3
from __future__ import print_function

import hashlib
import json
import re
import unittest
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1674_m1665_c1_transitive_formality_ptsta_exact_closed_one_shot.sh"
RTL_FM = HW / "dc_handoff/scripts/run_formality_m1674_c1_rtl_to_m993_transitive.tcl"
GATE_FM = HW / "dc_handoff/scripts/run_formality_m1674_c1_m993_to_m1665_gate_to_gate.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1674_c1_m1665_slowmax_fastmin.tcl"
CONTRACT = HW / "contracts/m1674_m1665_c1_transitive_formality_ptsta_source_contract_r1_20260901.json"
RESULT = HW / "dc_handoff/runs/m1674_m1665_c1_transitive_formality_ptsta_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1674_m1665_c1_transitive_formality_ptsta_attempt_consumed"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestM1674Source(unittest.TestCase):
    def test_source_files_and_exact_hashes(self):
        data = json.loads(CONTRACT.read_text())
        expected = {
            "dc_handoff/scripts/run_m1674_m1665_c1_transitive_formality_ptsta_exact_closed_one_shot.sh": RUNNER,
            "dc_handoff/scripts/run_formality_m1674_c1_rtl_to_m993_transitive.tcl": RTL_FM,
            "dc_handoff/scripts/run_formality_m1674_c1_m993_to_m1665_gate_to_gate.tcl": GATE_FM,
            "dc_handoff/scripts/run_ptsta_m1674_c1_m1665_slowmax_fastmin.tcl": PT_TCL,
            "system_simulator/tests/test_m1674_c1_transitive_formality_ptsta_source.py": Path(__file__).resolve(),
        }
        self.assertEqual(set(data["source_files"]), set(expected))
        for rel, path in expected.items():
            self.assertTrue(path.is_file())
            self.assertEqual(data["source_files"][rel], sha(path))

    def test_authoring_did_not_consume_attempt(self):
        self.assertFalse(RESULT.exists())
        self.assertFalse(ATTEMPT.exists())
        data = json.loads(CONTRACT.read_text())
        self.assertEqual(data["authorization_now"], {
            "formality_runs": 0, "pt_runs": 0, "dc_runs": 0,
            "vcs_runs": 0, "ptpx_runs": 0, "gpu_runs": 0,
            "remote_runs": 0, "attempts_created": 0,
        })

    def test_exact_future_tool_count_and_order(self):
        text = RUNNER.read_text()
        rtl_call = '"${FM_SHELL}" -f "${RTL_TO_M993_TCL}"'
        gate_call = '"${FM_SHELL}" -f "${GATE_TO_GATE_TCL}"'
        pt_call = '"${PT_SHELL}" -f "${PT_TCL}"'
        self.assertEqual(text.count(rtl_call), 1)
        self.assertEqual(text.count(gate_call), 1)
        self.assertEqual(text.count(pt_call), 1)
        self.assertLess(text.index(rtl_call), text.index(gate_call))
        self.assertLess(text.index(gate_call), text.index(pt_call))
        attempt = 'mkdir "${ATTEMPT}"'
        self.assertLess(text.index('[[ -z "$(same_uid_eda)" ]] || exit 4'), text.index(attempt))
        self.assertLess(text.index('for feature in Formality PrimeTime'), text.index(attempt))
        self.assertLess(text.index(attempt), text.index(rtl_call))

    def test_release_is_mandatory_and_exact_closed(self):
        text = RUNNER.read_text()
        for token in (
            "M1674_EXPECTED_RUNNER_SHA256",
            "M1674_EXPECTED_RELEASE_SHA256",
            "AUTHORIZE_ONE_M1674_C1_TRANSITIVE_FORMALITY_PTSTA_ATTEMPT",
            "formality_runs':2",
            "pt_runs':1",
            "dc_runs':0",
            "ptpx_runs':0",
            "fresh_result_hammer_required",
        ):
            self.assertIn(token, text)
        self.assertIn("verify_dir_seal \"${HAMMER_DIR}\"", text)
        self.assertIn("verify_file_seal \"${RELEASE}\"", text)

    def test_transitive_equivalence_chain_is_not_direct_svf_misuse(self):
        rtl = RTL_FM.read_text()
        gate = GATE_FM.read_text()
        self.assertIn("M1674_M993_SVF", rtl)
        self.assertNotIn("M1674_M1665", rtl)
        self.assertIn("read_sverilog -r $rtl_files", rtl)
        self.assertIn("read_verilog -i $mapped_netlist", rtl)
        self.assertIn("M1674_M993_MAPPED_NETLIST", gate)
        self.assertIn("M1674_M1665_MAPPED_NETLIST", gate)
        self.assertNotIn("set_svf", gate)
        for text in (rtl, gate):
            self.assertIn("read_db -technology_library $macro_slow_db", text)
            self.assertIn("report_unmatched_points", text)
            self.assertIn("report_failing_points", text)
            self.assertIn("report_aborted_points", text)
            self.assertIn("report_unverified_points", text)

    def test_pt_is_independent_macro_aware_max_min(self):
        text = PT_TCL.read_text()
        for token in (
            "read_verilog $mapped_netlist",
            "read_sdc $mapped_sdc",
            "set_min_library $std_slow_db -min_version $std_fast_db",
            "set_min_library $macro_slow_db -min_version $macro_fast_db",
            "-delay_type max",
            "-delay_type min",
            "macro_count != 9",
            "setup_slack < 0.0 || $hold_slack < 0.0",
            "parasitics=none_no_read_parasitics_command",
            "pt_eco=false",
        ):
            self.assertIn(token, text)
        forbidden = re.compile(r"(^|\n)\s*(set_false_path|set_multicycle_path|set_min_delay|set_max_delay|set_disable_timing|set_case_analysis)\b")
        self.assertIsNone(forbidden.search(text))

    def test_frozen_sdc_and_macro_contract(self):
        data = json.loads(CONTRACT.read_text())
        point = data["frozen_timing_point"]
        self.assertEqual(point["clock_period_ns"], 3.0)
        self.assertEqual(point["setup_uncertainty_ns"], 0.2)
        self.assertEqual(point["hold_uncertainty_ns"], 0.05)
        self.assertEqual(point["macro_count"], 9)
        self.assertEqual(point["macro_cell"], "TS1N28HPCPHVTB128X128M4S")
        self.assertEqual(point["timing_exception_counts"], {
            "false_path": 0, "multicycle_path": 0, "min_delay": 0,
            "max_delay": 0, "disabled_timing_arc": 0,
            "case_analysis": 0,
        })
        self.assertTrue(point["ideal_clock"])
        self.assertEqual(point["wireload"], "ZeroWireload")


if __name__ == "__main__":
    unittest.main()
