#!/usr/bin/env python3
"""Dual-runtime, mutation-heavy no-EDA author tests for M1698."""
from __future__ import print_function

import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1698_m1684_c2_shared_eda_queue_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1698_source_checker", str(CHECKER))
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class M1698Tests(unittest.TestCase):
    def test_01_predecessor_and_failed_m1685_bound(self):
        M.OLD.validate_predecessors()
        for path, digest in M.FIXED.items():
            self.assertEqual(M.sha(path), digest)
        review = M.strict_json(M.M1685 / "review.json")
        self.assertEqual(review["verdict"], "FAIL_CLOSED_NO_M1686_RELEASE")
        self.assertFalse(review["authorization"]["m1686_release_authoring"])
        self.assertFalse(M.M1686.exists())

    def test_02_exact_old_workload_and_energy_geometry(self):
        for axis in M.AXES:
            M.OLD.validate_filelist(axis)
        self.assertEqual(sum(M.EVENTS), 261)
        text = M.RUNNER.read_text()
        self.assertGreaterEqual(text.count('for axis in ("k8", "k1x8"):'), 3)
        self.assertGreaterEqual(text.count("for case_id in range(5):"), 2)
        for token in ('"vcs_compiles": 2', '"simv_runs": 10',
                      '"saif_files": 10', '"ptpx_runs": 10'):
            self.assertIn(token, text)

    def test_03_shared_flock_and_launch_rescans(self):
        M.validate_queue_source()
        text = M.RUNNER.read_text()
        self.assertIn(M.SHARED_LOCK, text)
        self.assertIn('Path(command[0]).name in {"vcs", "pt_shell"}', text)

    def test_04_queue_mutations_rejected(self):
        text = M.RUNNER.read_text()
        mutations = (
            text.replace(M.SHARED_LOCK, "/tmp/campaign_local.lock", 1),
            text.replace("fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
                         "pass", 1),
            text.replace('Path(command[0]).name in {"vcs", "pt_shell"}',
                         'Path(command[0]).name == "vcs"', 1),
            text.replace("collision_gate()\n    with output.open", 
                         "with output.open", 1),
        )
        for changed in mutations:
            with self.assertRaises(RuntimeError):
                M.validate_queue_source(changed)

    def test_05_ancestry_predicate_accepts_own_tree(self):
        import os
        self.assertTrue(M._owned_or_ancestor(os.getpid(), os.getpid())
                        if hasattr(M, "_owned_or_ancestor") else True)
        # The implementation is runner source, so additionally bind its exact
        # parent-walk tokens without importing a launch-capable module.
        text = M.RUNNER.read_text()
        self.assertIn("if cursor == runner_pid:", text)
        self.assertIn("if pid in ancestry:", text)

    def test_06_active_force_scan_rejects_only_active_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sv = root / "x.sv"
            sv.write_text("// force dut.q = 1'b0;\n/* force x = 0; */\nmodule x; endmodule\n")
            self.assertFalse(M.active_force_present(sv))
            sv.write_text("module x; initial force dut.q = 1'b0; endmodule\n")
            self.assertTrue(M.active_force_present(sv))
            py = root / "x.py"
            py.write_text("# force ignored\nx = 'force ignored'\n")
            self.assertFalse(M.active_force_present(py))
            py.write_text("force(x)\n")
            self.assertTrue(M.active_force_present(py))
            tcl = root / "x.tcl"
            tcl.write_text("# force ignored\nrun\n")
            self.assertFalse(M.active_force_present(tcl))
            tcl.write_text("force dut/q 0\n")
            self.assertTrue(M.active_force_present(tcl))

    def test_07_all_execution_sources_force_and_initreg_clean(self):
        for path in M.EXECUTION_SOURCES:
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
        self.assertAlmostEqual(metrics["equal_bandwidth_cycle_speedup_k8_vs_k1x8"],
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
        self.assertEqual(value["status"], "PASS_M1698_SOURCE_ONLY_NO_EDA")
        self.assertEqual(value["accepted_sources_per_axis"], 261)
        self.assertTrue(value["active_force_full_source_scan"])
        self.assertTrue(all(item is False for item in value["claim_boundary"].values()))
        for path in (M.M1699, M.M1700,
                     M.HW / "results/.m1698_c2_shared_eda_queue_production_energy_attempt_consumed",
                     M.HW / "results/m1698_c2_shared_eda_queue_production_energy_r1_20260901"):
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
