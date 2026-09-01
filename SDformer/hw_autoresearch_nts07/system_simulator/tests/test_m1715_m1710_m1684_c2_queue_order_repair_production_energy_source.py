#!/usr/bin/env python3
"""Dual-runtime, mutation-heavy no-EDA author tests for M1715."""
from __future__ import print_function

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1715_source_checker", str(CHECKER))
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class M1715Tests(unittest.TestCase):
    def test_01_m1710_failure_exact_and_no_retry(self):
        failed = M.verify_m1710_failure()
        self.assertFalse(failed["attempt_consumed"])
        self.assertEqual(failed["phase"], "SOURCE_CHAIN")
        self.assertEqual(set(failed["counts"].values()), {0})
        self.assertFalse(failed["automatic_retry"])
        for path in (M.M1710_ATTEMPT, M.M1710_RESULT, M.M1710_PRIVATE):
            self.assertFalse(os.path.lexists(path), str(path))

    def test_02_exact_six_sources_and_campaign_geometry(self):
        contract = M.strict_json(M.OLD.M1684_CONTRACT)
        mapping = dict((row["path"], row["sha256"])
                       for row in contract["source_files"])
        self.assertEqual(len(M.DIRECT_SOURCES), 6)
        for rel, path in zip(M.DIRECT_REL, M.DIRECT_SOURCES):
            self.assertEqual(M.sha(path), mapping[rel])
        text = M.RUNNER.read_text()
        self.assertGreaterEqual(text.count('for axis in ("k8", "k1x8"):'), 3)
        self.assertGreaterEqual(text.count("for case_id in range(5):"), 2)
        for token in ('"vcs_compiles": 2', '"simv_runs": 10',
                      '"saif_files": 10', '"ptpx_runs": 10'):
            self.assertIn(token, text)

    def test_03_blocking_flock_precedes_first_collision_and_rebind(self):
        M.validate_queue_source()
        text = M.RUNNER.read_text()
        main = text[text.index("def main("):]
        lock = main.index("fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)")
        collision = main.index("collision_gate()", lock)
        rebind = main.index("runtime_bind_execution_sources()", collision)
        attempt = main.index("ATTEMPT.mkdir()")
        self.assertLess(lock, collision)
        self.assertLess(collision, rebind)
        self.assertLess(rebind, attempt)
        self.assertNotIn("collision_gate()", main[:lock])
        self.assertNotIn("LOCK_NB", text)

    def test_04_queue_order_mutations_rejected(self):
        text = M.RUNNER.read_text()
        blocking = "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)"
        mutations = (
            text.replace(blocking, blocking + " | fcntl.LOCK_NB", 1),
            text.replace(blocking,
                         "collision_gate()\n        " + blocking, 1),
            text.replace("collision_gate()\n        state[\"phase\"] = \"POST_LOCK_RUNTIME_REBIND\"",
                         "state[\"phase\"] = \"POST_LOCK_RUNTIME_REBIND\"", 1),
            text.replace("runtime_bind_execution_sources()\n        forbidden_release_namespaces_absent()",
                         "pass", 1),
        )
        for changed in mutations:
            with self.assertRaises(RuntimeError):
                M.validate_queue_source(changed)

    def test_05_ancestry_and_prelaunch_rescan_preserved(self):
        text = M.RUNNER.read_text()
        self.assertIn("if cursor == runner_pid:", text)
        self.assertIn("if pid in ancestry:", text)
        run = text[text.index("def run("):text.index("def result_identity")]
        self.assertIn('Path(command[0]).name in {"vcs", "pt_shell"}', run)
        self.assertLess(run.index("collision_gate()"), run.index("subprocess.run("))

    def test_06_active_force_scanner_rejects_only_active_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sv = root / "x.sv"
            sv.write_text("// force dut.q = 0;\nmodule x; endmodule\n")
            self.assertFalse(M.active_force_present(sv))
            sv.write_text("module x; initial force dut.q = 0; endmodule\n")
            self.assertTrue(M.active_force_present(sv))
            tcl = root / "x.tcl"
            tcl.write_text('# force ignored\nset x "force ignored"\nrun\n')
            self.assertFalse(M.active_force_present(tcl))
            tcl.write_text("if {1} { force dut/q 0 }\n")
            self.assertTrue(M.active_force_present(tcl))

    def test_07_all_six_execution_sources_force_and_initreg_clean(self):
        for path in M.DIRECT_SOURCES:
            self.assertFalse(M.active_force_present(path), str(path))
            self.assertNotIn("initreg", path.read_text().lower())

    def test_08_runtime_energy_math_unchanged(self):
        rows = []
        for axis, power in (("k8", 2.0), ("k1x8", 4.0)):
            for case_id in range(5):
                rows.append({"axis": axis, "case": case_id,
                             "cycles": M.AXES[axis]["cycles"][case_id],
                             "accepted_sources": M.EVENTS[case_id],
                             "total_mw": power})
        metrics = M.aggregate_metrics(rows)
        self.assertAlmostEqual(
            metrics["equal_bandwidth_cycle_speedup_k8_vs_k1x8"],
            1945.0 / 1913.0)
        self.assertEqual(metrics["axes"]["k8"]["accepted_sources"], 261)

    def test_09_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON"):
                M.strict_json(path)
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(RuntimeError, "nonfinite JSON"):
                M.strict_json(path)

    def test_10_complete_source_contract_no_execution(self):
        value = M.validate_sources()
        self.assertEqual(value["status"], "PASS_M1715_SOURCE_ONLY_NO_EDA")
        self.assertTrue(value["blocking_flock_before_collision"])
        self.assertFalse(value["prelock_collision_scan"])
        self.assertEqual(value["postlock_runtime_rebinds_before_attempt"], 2)
        self.assertEqual(value["runtime_bound_execution_sources"], 6)
        self.assertTrue(value["m1710_retry_forbidden"])
        self.assertTrue(all(item is False
                            for item in value["claim_boundary"].values()))

    def test_11_lexists_and_postlock_rebinds_preserved(self):
        text = M.RUNNER.read_text()
        main = text[text.index("def main("):]
        lock = main.index("fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)")
        attempt = main.index("ATTEMPT.mkdir()")
        window = main[lock:attempt]
        self.assertGreaterEqual(window.count("runtime_bind_execution_sources()"), 2)
        self.assertGreaterEqual(window.count("forbidden_release_namespaces_absent()"), 2)
        self.assertGreaterEqual(text.count("os.path.lexists(path)"), 3)

    def test_12_runtime_gate_mutations_rejected(self):
        text = M.RUNNER.read_text()
        mutations = (
            text.replace("if os.path.lexists(path):", "if False:", 1),
            text.replace("DIRECT_EXECUTION_PATHS = {",
                         "DIRECT_EXECUTION_PATHS = set() or {", 1),
            text.replace("verify_m1710_pre_attempt_failure()", "pass", 1),
        )
        for changed in mutations:
            with self.assertRaises(RuntimeError):
                M.validate_queue_source(changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
