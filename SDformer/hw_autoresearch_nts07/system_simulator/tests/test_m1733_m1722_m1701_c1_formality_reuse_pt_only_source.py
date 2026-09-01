#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1733_m1722_m1701_c1_formality_reuse_pt_only_one_shot.py"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1733_c1_m1701_slowmax_fastmin.tcl"
CONTRACT = HW / "contracts/m1733_m1722_m1701_c1_formality_reuse_pt_only_source_contract_r1_20260901.json"
SPEC = importlib.util.spec_from_file_location("m1733_runner", str(RUNNER))
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class M1733Tests(unittest.TestCase):
    def test_01_frozen_candidates_and_salvage_authority_are_exact(self):
        M.verify_seal(M.M1701, M.FIXED_SHA["m1701_manifest"], M.FIXED_SHA["m1701_outer"])
        M.verify_seal(M.M1665, M.FIXED_SHA["m1665_manifest"], M.FIXED_SHA["m1665_outer"])
        M.verify_seal(M.M1714, M.FIXED_SHA["m1714_manifest"], M.FIXED_SHA["m1714_outer"])
        self.assertEqual(M.strict_json(M.M1714 / "review.json")["status"],
                         "PASS_SALVAGE_CANDIDATE_ONLY")

    def test_02_m1722_failed_campaign_is_double_sealed_and_pt_never_ran(self):
        M.verify_seal(M.M1722_FAILURE, M.FIXED_SHA["m1722_failure_manifest"],
                      M.FIXED_SHA["m1722_failure_outer"])
        M.verify_seal(M.M1722_ATTEMPT, M.FIXED_SHA["m1722_attempt_manifest"],
                      M.FIXED_SHA["m1722_attempt_outer"])
        failure = M.strict_json(M.M1722_FAILURE / "failure.json")
        self.assertEqual(failure["phase"], "FORMALITY")
        self.assertEqual(failure["formality_runs"], 1)
        self.assertEqual(failure["pt_runs"], 0)
        self.assertTrue(failure["attempt_consumed"])
        self.assertFalse(failure["automatic_retry"])

    def test_03_m1722_formality_pass_is_exact_and_semantically_closed(self):
        proof = M.verify_m1722_formality_reuse()
        self.assertEqual(proof["passing_compare_points"], 16549)
        self.assertEqual(proof["macro_instances_per_side"], 9)
        self.assertEqual(proof["log_allowlist"],
                         {"gui": 0, "matched_header": 1,
                          "source_echo": 1})

    def test_04_prime_time_is_independent_slowmax_fastmin(self):
        text = PT_TCL.read_text()
        for token in ("set_min_library $std_slow_db -min_version $std_fast_db",
                      "set_min_library $macro_slow_db -min_version $macro_fast_db",
                      "-max ssg0p9v125c", "-min ffg1p05vm40c",
                      "read_sdc $mapped_sdc", "-delay_type max",
                      "-delay_type min", "$macro_count != 9",
                      "abs($clock_period - 3.000)"):
            self.assertIn(token, text)

    def test_05_exact_allowlists_and_every_variant_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "tool.log"
            log.write_text("normal\n" + M.GUI_ALLOW + "\nnormal\n")
            self.assertEqual(M.scan_tool_log(log),
                             {"gui": 1, "matched_header": 0,
                              "source_echo": 0})
            log.write_text(M.MATCHED_HEADER_ALLOW + "\n")
            with self.assertRaises(M.Failure):
                M.scan_tool_log(log)
            self.assertEqual(M.scan_tool_log(log, allow_matched_header=True),
                             {"gui": 0, "matched_header": 1,
                              "source_echo": 0})
            log.write_text(M.FORMALITY_TCL_ECHO_ALLOW + "\n")
            with self.assertRaises(M.Failure):
                M.scan_tool_log(log, allow_matched_header=True)
            self.assertEqual(M.scan_tool_log(
                log, allow_matched_header=True,
                exact_source_echo_allow=(M.FORMALITY_TCL_ECHO_ALLOW,)),
                {"gui": 0, "matched_header": 0, "source_echo": 1})
            variants = (
                " " + M.MATCHED_HEADER_ALLOW, "\t" + M.MATCHED_HEADER_ALLOW,
                M.MATCHED_HEADER_ALLOW + " ", M.MATCHED_HEADER_ALLOW + "\t",
                "prefix " + M.MATCHED_HEADER_ALLOW,
                M.MATCHED_HEADER_ALLOW + " suffix",
            )
            for attack in variants:
                log.write_text(attack + "\n")
                with self.assertRaises(M.Failure):
                    M.scan_tool_log(log, allow_matched_header=True)
            attacks = (
                "Error: another failure", "Fatal: crash", "Error : failed",
                "**Error: failed", "Info:Fatal: failed", "LINK-5 failed",
                "unresolved reference x", "unable to resolve x",
                "timing loop found", "feedback loop detected", "(TIM-209)",
                "(OPT-150)", "prefix " + M.GUI_ALLOW, " " + M.GUI_ALLOW,
                M.GUI_ALLOW + " ", "Error failed", "ERROR failed",
                "Fatal crash", "FATAL crash", "error occurred",
                "fatal occurred", "Error- failed", "Fatal failure",
            )
            for attack in attacks:
                log.write_text(attack + "\n")
                with self.assertRaises(M.Failure):
                    M.scan_tool_log(log, allow_matched_header=True)
            for exact_line, allow_header in ((M.GUI_ALLOW, False),
                                             (M.MATCHED_HEADER_ALLOW, True)):
                log.write_text(exact_line + "\n" + exact_line + "\n")
                with self.assertRaises(M.Failure):
                    M.scan_tool_log(log, allow_matched_header=allow_header)
            for echo_attack in (M.FORMALITY_TCL_ECHO_ALLOW + " ",
                                " " + M.FORMALITY_TCL_ECHO_ALLOW,
                                "prefix " + M.FORMALITY_TCL_ECHO_ALLOW,
                                M.FORMALITY_TCL_ECHO_ALLOW + " suffix",
                                M.FORMALITY_TCL_ECHO_ALLOW.replace("M1722", "M1723")):
                log.write_text(echo_attack + "\n")
                with self.assertRaises(M.Failure):
                    M.scan_tool_log(log, allow_matched_header=True,
                                    exact_source_echo_allow=(M.FORMALITY_TCL_ECHO_ALLOW,))
            log.write_text(M.FORMALITY_TCL_ECHO_ALLOW + "\n" +
                           M.FORMALITY_TCL_ECHO_ALLOW + "\n")
            with self.assertRaises(M.Failure):
                M.scan_tool_log(log, allow_matched_header=True,
                                exact_source_echo_allow=(M.FORMALITY_TCL_ECHO_ALLOW,))
            accepted_non_diagnostics = (
                "0 errors", "No error", "No errors detected",
                "Summary: 0 fatal diagnostics",
            )
            log.write_text("\n".join(accepted_non_diagnostics) + "\n")
            self.assertEqual(M.scan_tool_log(log, allow_matched_header=True),
                             {"gui": 0, "matched_header": 0,
                              "source_echo": 0})

    def test_06_pt_error_echoes_are_exact_tcl_source_and_pt_scope_only(self):
        tcl_lines = set(PT_TCL.read_text().splitlines())
        self.assertEqual(len(M.PT_TCL_ECHO_ALLOW), 5)
        self.assertTrue(set(M.PT_TCL_ECHO_ALLOW).issubset(tcl_lines))
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "pt.log"
            log.write_text("\n".join(M.PT_TCL_ECHO_ALLOW) + "\n")
            with self.assertRaises(M.Failure):
                M.scan_tool_log(log)
            self.assertEqual(M.scan_tool_log(
                log, exact_source_echo_allow=M.PT_TCL_ECHO_ALLOW),
                {"gui": 0, "matched_header": 0, "source_echo": 5})
            for line in M.PT_TCL_ECHO_ALLOW:
                for attack in (line + " ", " " + line,
                               "prefix " + line, line + " suffix",
                               line.replace("M1733", "M1732")):
                    log.write_text(attack + "\n")
                    with self.assertRaises(M.Failure):
                        M.scan_tool_log(
                            log, exact_source_echo_allow=M.PT_TCL_ECHO_ALLOW)
            log.write_text(M.PT_TCL_ECHO_ALLOW[0] + "\n" +
                           M.PT_TCL_ECHO_ALLOW[0] + "\n")
            with self.assertRaises(M.Failure):
                M.scan_tool_log(log, exact_source_echo_allow=M.PT_TCL_ECHO_ALLOW)

    def test_07_constraints_area_and_macro_gate(self):
        area = M.verify_inputs()
        self.assertAlmostEqual(area, 166514.312080)
        self.assertLessEqual(area, M.AREA_CEILING)
        self.assertEqual(M.M1701_NETLIST.read_text(errors="replace").count(
            "TS1N28HPCPHVTB128X128M4S"), 9)

    def test_08_execution_order_is_pt_only_and_budget_is_zero_one_zero(self):
        text = RUNNER.read_text()
        main = text[text.index("def main()") :]
        tokens = ('verify_m1722_formality_reuse()',
                  'state["phase"] = "ATTEMPT_CONSUME"', "ATTEMPT.mkdir()",
                  'state["phase"] = "PRIMETIME"', "run_tool([str(PT)")
        cursor = 0
        for token in tokens:
            position = main.find(token, cursor)
            self.assertGreaterEqual(position, 0, token)
            cursor = position + len(token)
        self.assertNotIn('state["phase"] = "FORMALITY"', main)
        self.assertNotIn("run_tool([str(FM)", main)
        self.assertNotIn("fm_shell", main)
        self.assertEqual(text.count('state["formality_runs"] += 1'), 0)
        self.assertEqual(text.count('state["pt_runs"] += 1'), 1)
        self.assertIn('"formality_runs": 0, "pt_runs": 1, "dc_runs": 0', text)
        self.assertNotIn("dc_shell", main)

    def test_09_shared_lock_collision_and_no_formality_license_query(self):
        text = RUNNER.read_text()
        self.assertEqual(str(M.LOCK), "/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
        self.assertIn("fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)", text)
        self.assertIn('for feature in ("PrimeTime",):', text)
        self.assertNotIn('for feature in ("Formality", "PrimeTime"):', text)
        run_body = text[text.index("def run_tool(") : text.index("def read_machine")]
        self.assertLess(run_body.index("collision_gate()"), run_body.index("subprocess.run("))

    def test_10_fresh_authority_and_namespaces_are_absent(self):
        for path in (M.M1734, M.M1735, Path(str(M.M1735) + ".sha256"),
                     Path(str(M.M1735) + ".sha256.seal.sha256"),
                     M.RESULT, M.ATTEMPT, M.FAILURE):
            self.assertFalse(os.path.lexists(path), str(path))
        with self.assertRaisesRegex(M.Failure, "authority absent"):
            M.verify_authority()

    def test_11_source_contract_identity_claims_and_reuse_boundary(self):
        contract = M.strict_json(CONTRACT)
        M.verify_contract_sources(contract)
        self.assertEqual(contract["claim_boundary"], M.CLAIMS)
        self.assertTrue(all(value is False for value in M.CLAIMS.values()))
        self.assertEqual(contract["future_execution"]["formality_runs"], 0)
        self.assertEqual(contract["future_execution"]["pt_runs"], 1)
        self.assertFalse(contract["m1722_formality_reuse"]["rerun_formality"])

    def test_12_no_retry_no_promotion_and_candidate_boundary(self):
        text = RUNNER.read_text()
        self.assertIn('"automatic_retry": False', text)
        self.assertIn("PT_ONLY_CANDIDATE_PENDING", text)
        self.assertIn('"m1701_quarantine_modified_or_promoted": False', text)
        self.assertNotIn("paper_citable_now", text)

    def test_13_source_only_author_executed_nothing(self):
        execution = M.strict_json(CONTRACT)["author_execution"]
        self.assertTrue(execution["source_only"])
        for key in ("license_queries", "formality_runs", "pt_runs", "dc_runs",
                    "attempts_created", "results_created"):
            self.assertEqual(execution[key], 0)
        self.assertFalse(execution["release_created"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
