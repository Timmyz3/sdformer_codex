#!/usr/bin/env python3
"""Dual-runtime mutation-heavy source-only tests for M1730."""
from __future__ import print_function

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1730_m1715_c2_vcs_proxy_repair_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1730_source_checker", str(CHECKER))
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class M1730Tests(unittest.TestCase):
    def test_01_m1715_consumed_failure_exact(self):
        failed = M.verify_m1715_failure()
        self.assertTrue(failed["attempt_consumed"])
        self.assertEqual(failed["phase"], "COMPILE_k8")
        self.assertEqual(failed["error"], "KeyboardInterrupt")
        self.assertEqual(failed["counts"], {
            "vcs_compiles": 1, "simv_runs": 0,
            "saif_files": 0, "ptpx_runs": 0})
        self.assertFalse(failed["automatic_retry"])
        self.assertFalse(os.path.lexists(M.M1715_RESULT))

    def test_02_exact_proxy_scope_and_order(self):
        M.validate_proxy_source()
        text = M.RUNNER.read_text()
        main = text[text.index("def main("):]
        self.assertLess(main.index("compile_proxy = capture_exact_compile_proxy_from_launch()"),
                        main.index("preflight_compile_proxy(compile_proxy)"))
        self.assertLess(main.index("preflight_compile_proxy(compile_proxy)"),
                        main.index("ATTEMPT.mkdir()"))
        self.assertEqual(text.count("vcs_compile_proxy=compile_proxy"), 1)

    def test_03_proxy_tuple_mutations_rejected(self):
        text = M.RUNNER.read_text()
        mutations = []
        for key in M.PROXY_KEYS:
            mutations.append(text.replace('    "' + key + '":',
                                          '    "BROKEN_' + key + '":', 1))
        mutations.extend((
            text.replace("http://127.0.0.1:7897", "http://127.0.0.1:7898", 1),
            text.replace('"localhost,127.0.0.1,::1"', '"localhost"', 1),
            text.replace('PROXY_PORT = 7897', 'PROXY_PORT = 7898', 1),
            text.replace('PROXY_HOST = "127.0.0.1"', 'PROXY_HOST = "localhost"', 1),
        ))
        for changed in mutations:
            with self.assertRaises(RuntimeError):
                M.validate_proxy_source(changed)

    def test_04_proxy_gate_and_order_mutations_rejected(self):
        text = M.RUNNER.read_text()
        mutations = (
            text.replace("os.environ.get(name)", "os.getenv(name)", 1),
            text.replace("captured = {}", "captured = os.environ.copy()", 1),
            text.replace("if value != EXPECTED_PROXY[name]:", "if False:", 1),
            text.replace("if proxy != EXPECTED_PROXY:", "if False:", 1),
            text.replace("socket.create_connection(", "socket.socket(", 1),
            text.replace("if vcs_compile_proxy is not None:", "if True:", 1),
            text.replace("if vcs_compile_proxy != EXPECTED_PROXY:", "if False:", 1),
            text.replace("value.update(vcs_compile_proxy)", "value.update(os.environ)", 1),
            text.replace("preflight_compile_proxy(compile_proxy)\n",
                         "preflight_compile_proxy(compile_proxy) if False else None\n", 1),
            text.replace("vcs_compile_proxy=compile_proxy", "vcs_compile_proxy=None", 1),
            text.replace("M1715_PRIVATE is deliberately neither sealed nor inspected for claims",
                         "M1715 private evidence may be cited", 1),
            text.replace("verify_m1715_consumed_failure()\n", "pass\n", 1),
        )
        for changed in mutations:
            with self.assertRaises(RuntimeError):
                M.validate_proxy_source(changed)

    def test_05_proxy_leak_mutations_rejected(self):
        text = M.RUNNER.read_text()
        sim_anchor = '"M1684_SAIF_FILE": str(saif)}),\n                    timeout=1200'
        pt_anchor = '"REGISTERED_FAULT_PUBLIC_ZERO": "true"}),\n                    timeout=3600'
        checker_anchor = 'cwd=HW, env=clean_env({}), timeout=180'
        mutations = (
            text.replace(sim_anchor,
                         '"M1684_SAIF_FILE": str(saif)}, vcs_compile_proxy=compile_proxy),\n                    timeout=1200', 1),
            text.replace(pt_anchor,
                         '"REGISTERED_FAULT_PUBLIC_ZERO": "true"}, vcs_compile_proxy=compile_proxy),\n                    timeout=3600', 1),
            text.replace(checker_anchor,
                         'cwd=HW, env=clean_env({}, vcs_compile_proxy=compile_proxy), timeout=180', 1),
        )
        for changed in mutations:
            self.assertNotEqual(changed, text)
            with self.assertRaises(RuntimeError):
                M.validate_proxy_source(changed)

    def test_06_budget_geometry_and_no_retry(self):
        text = M.RUNNER.read_text()
        self.assertGreaterEqual(text.count('for axis in ("k8", "k1x8"):'), 3)
        self.assertGreaterEqual(text.count("for case_id in range(5):"), 2)
        for token in ('"vcs_compiles": 2', '"simv_runs": 10',
                      '"saif_files": 10', '"ptpx_runs": 10'):
            self.assertIn(token, text)
        self.assertIn('"automatic_retry": False', text)

    def test_07_runtime_math_unchanged(self):
        rows = []
        for axis, power in (("k8", 2.0), ("k1x8", 4.0)):
            for case_id in range(5):
                rows.append({"axis": axis, "case": case_id,
                             "cycles": M.AXES[axis]["cycles"][case_id],
                             "accepted_sources": M.EVENTS[case_id],
                             "total_mw": power})
        metrics = M.aggregate_metrics(rows)
        self.assertAlmostEqual(
            metrics["equal_bandwidth_cycle_speedup_k8_vs_k1x8"], 1945.0 / 1913.0)
        self.assertEqual(metrics["axes"]["k8"]["accepted_sources"], 261)

    def test_08_complete_source_contract_no_execution(self):
        value = M.validate_sources()
        self.assertEqual(value["status"], "PASS_M1730_SOURCE_ONLY_NO_EDA")
        self.assertEqual(value["proxy_scope"], "VCS_COMPILE_ONLY")
        self.assertTrue(value["proxy_tcp_preflight_before_attempt"])
        self.assertFalse(value["private_forensic_tree_citable"])
        self.assertTrue(value["m1715_retry_forbidden"])
        self.assertTrue(all(item is False for item in value["claim_boundary"].values()))

    def test_09_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON"):
                M.strict_json(path)
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(RuntimeError, "nonfinite JSON"):
                M.strict_json(path)

    def test_10_future_authority_and_result_absent(self):
        for path in (M.M1731, M.M1732, Path(str(M.M1732) + ".sha256"),
                     Path(str(M.M1732) + ".sha256.seal.sha256"),
                     M.HW / "results/.m1730_c2_vcs_proxy_repair_production_energy_attempt_consumed",
                     M.HW / "results/m1730_c2_vcs_proxy_repair_production_energy_r1_20260901"):
            self.assertFalse(os.path.lexists(path), str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
