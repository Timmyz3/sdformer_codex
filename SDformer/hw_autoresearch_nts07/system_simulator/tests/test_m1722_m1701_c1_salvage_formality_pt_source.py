#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1722_m1701_c1_salvage_formality_pt_one_shot.py"
FM_TCL = HW / "dc_handoff/scripts/run_formality_m1722_c1_m1665_to_m1701_gate_to_gate.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1722_c1_m1701_slowmax_fastmin.tcl"
CONTRACT = HW / "contracts/m1722_m1701_c1_salvage_formality_pt_source_contract_r1_20260901.json"
SPEC = importlib.util.spec_from_file_location("m1722_runner", str(RUNNER))
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class M1722Tests(unittest.TestCase):
    def test_01_m1714_salvage_and_quarantine_seals_are_exact(self):
        M.verify_seal(M.M1701, M.FIXED_SHA["m1701_manifest"], M.FIXED_SHA["m1701_outer"])
        M.verify_seal(M.M1714, M.FIXED_SHA["m1714_manifest"], M.FIXED_SHA["m1714_outer"])
        self.assertEqual(M.strict_json(M.M1714 / "review.json")["status"],
                         "PASS_SALVAGE_CANDIDATE_ONLY")

    def test_02_m1665_reference_and_m1701_four_artifacts_are_bound(self):
        M.verify_seal(M.M1665, M.FIXED_SHA["m1665_manifest"], M.FIXED_SHA["m1665_outer"])
        for path, key in ((M.M1665_NETLIST, "m1665_netlist"),
                          (M.M1665_SDC, "m1665_sdc"),
                          (M.M1665_DDC, "m1665_ddc"),
                          (M.M1665_SVF, "m1665_svf"),
                          (M.M1701_NETLIST, "m1701_netlist"),
                          (M.M1701_SDC, "m1701_sdc"),
                          (M.M1701_DDC, "m1701_ddc"),
                          (M.M1701_SVF, "m1701_svf")):
            self.assertEqual(M.sha(path), M.FIXED_SHA[key])

    def test_03_formality_is_gate_to_gate_in_the_right_direction(self):
        text = FM_TCL.read_text()
        self.assertIn("M1722_M1665_REFERENCE_NETLIST", text)
        self.assertIn("M1722_M1701_IMPLEMENTATION_NETLIST", text)
        self.assertLess(text.index("read_verilog -r $reference_netlist"),
                        text.index("read_verilog -i $implementation_netlist"))
        for token in ("match", "set verification_succeeded [verify]",
                      "report_unmatched_points", "report_failing_points",
                      "report_aborted_points", "report_unverified_points"):
            self.assertIn(token, text)

    def test_04_prime_time_is_independent_slowmax_fastmin(self):
        text = PT_TCL.read_text()
        for token in ("set_min_library $std_slow_db -min_version $std_fast_db",
                      "set_min_library $macro_slow_db -min_version $macro_fast_db",
                      "-max ssg0p9v125c", "-min ffg1p05vm40c",
                      "read_sdc $mapped_sdc", "-delay_type max",
                      "-delay_type min", "$macro_count != 9",
                      "abs($clock_period - 3.000)"):
            self.assertIn(token, text)

    def test_05_exact_gui_allowlist_and_other_fatals_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "tool.log"
            log.write_text("normal\n" + M.GUI_ALLOW + "\nnormal\n")
            self.assertEqual(M.scan_tool_log(log), 1)
            log.write_text("normal\n")
            self.assertEqual(M.scan_tool_log(log), 0)
            attacks = (
                "Error: another failure\n", "Fatal: crash\n",
                "Error : failed\n", "Fatal : failed\n",
                "**Error: failed\n", "**Fatal: failed\n",
                "Info:Error: failed\n", "Info:Fatal: failed\n",
                "LINK-5 failed\n", "unresolved reference x\n",
                "unresolved foo\n", "unresolved module foo\n",
                "unresolved black box foo\n", "unable to resolve x\n",
                "timing loop found\n", "loop detected\n",
                "feedback loop detected\n",
                "combinational logic loop detected\n",
                "prefix " + M.GUI_ALLOW + " suffix\n",
                " " + M.GUI_ALLOW + "\n", "\t" + M.GUI_ALLOW + "\n",
                M.GUI_ALLOW + " \n", M.GUI_ALLOW + "\t\n",
                "X" + M.GUI_ALLOW + "\n",
            )
            for attack in attacks:
                log.write_text(attack)
                with self.assertRaises(M.Failure):
                    M.scan_tool_log(log)
            log.write_text(M.GUI_ALLOW + "\n" + M.GUI_ALLOW + "\n")
            with self.assertRaises(M.Failure):
                M.scan_tool_log(log)

    def test_06_constraints_area_and_macro_gate(self):
        area = M.verify_inputs()
        self.assertAlmostEqual(area, 166514.312080)
        self.assertLessEqual(area, M.AREA_CEILING)
        self.assertEqual(M.M1701_NETLIST.read_text(errors="replace").count(
            "TS1N28HPCPHVTB128X128M4S"), 9)

    def test_07_execution_order_and_budget(self):
        text = RUNNER.read_text()
        main = text[text.index("def main()") :]
        tokens = ('state["phase"] = "ATTEMPT_CONSUME"', "ATTEMPT.mkdir()",
                  'state["phase"] = "FORMALITY"',
                  "run_tool([str(FM)", 'state["phase"] = "PRIMETIME"',
                  "run_tool([str(PT)")
        cursor = 0
        for token in tokens:
            position = main.find(token, cursor)
            self.assertGreaterEqual(position, 0, token)
            cursor = position + len(token)
        self.assertEqual(text.count('state["formality_runs"] += 1'), 1)
        self.assertEqual(text.count('state["pt_runs"] += 1'), 1)
        self.assertNotIn("dc_shell", main)

    def test_08_shared_lock_and_per_tool_collision_gate(self):
        text = RUNNER.read_text()
        self.assertEqual(str(M.LOCK), "/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
        self.assertIn("fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)", text)
        run_body = text[text.index("def run_tool(") : text.index("def read_machine")]
        self.assertLess(run_body.index("collision_gate()"),
                        run_body.index("subprocess.run("))

    def test_09_fresh_authority_and_namespaces_are_absent(self):
        for path in (M.M1723, M.M1724, Path(str(M.M1724) + ".sha256"),
                     Path(str(M.M1724) + ".sha256.seal.sha256"),
                     M.RESULT, M.ATTEMPT, M.FAILURE):
            self.assertFalse(os.path.lexists(path), str(path))
        with self.assertRaisesRegex(M.Failure, "authority absent"):
            M.verify_authority()

    def test_10_source_contract_identity_and_claims(self):
        contract = M.strict_json(CONTRACT)
        M.verify_contract_sources(contract)
        self.assertEqual(contract["claim_boundary"], M.CLAIMS)
        self.assertTrue(all(value is False for value in M.CLAIMS.values()))

    def test_11_no_retry_failure_and_candidate_boundary(self):
        text = RUNNER.read_text()
        self.assertIn('"automatic_retry": False', text)
        self.assertIn("PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER", text)
        self.assertIn('"m1701_quarantine_modified_or_promoted": False', text)
        self.assertNotIn("paper_citable_now", text)

    def test_12_source_only_no_execution(self):
        contract = M.strict_json(CONTRACT)
        execution = contract["author_execution"]
        self.assertTrue(execution["source_only"])
        for key in ("license_queries", "formality_runs", "pt_runs", "dc_runs",
                    "attempts_created", "results_created", "release_created"):
            self.assertEqual(execution[key], 0 if key != "release_created" else False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
