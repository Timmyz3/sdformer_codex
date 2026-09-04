#!/opt/anaconda3/bin/python3
"""CPU-only source tests for M2217.  No simulator, EDA, GPU, or license use."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SELECTOR = HW / "system_simulator/scripts/select_m2217_ep34_tsbg_matched_power_windows.py"
PARSER = HW / "system_simulator/scripts/parse_m2217_ep34_tsbg_matched_power.py"
RUNNER = HW / "dc_handoff/scripts/run_m2217_ep34_tsbg_matched_power_one_shot.py"
SELECTION = HW / "tb_m2018/fixtures/m2217_ep34_tsbg_matched_power_windows.json"
TB = HW / "tb_m2018/tb_m2217_m2018_tsbg_matched_native_saif_power.sv"
UCLI = HW / "dc_handoff/scripts/m2217_m2018_single_dut_native_saif.ucli.tcl"
VCS_F = HW / "dc_handoff/filelists/tcasii_m2217_m2018_single_dut_native_saif_vcs.f"
DC_TCL = HW / "dc_handoff/scripts/run_dc_m2217_m2018_matched_power_axis.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m2217_m2018_matched_power_window.tcl"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestM2217(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = load(SELECTOR, "m2217_selector_test")
        cls.parser = load(PARSER, "m2217_parser_test")
        cls.selection = json.loads(SELECTION.read_text())
        cls.tb = TB.read_text()
        cls.runner = RUNNER.read_text()
        cls.ucli = UCLI.read_text()
        cls.dc = DC_TCL.read_text()
        cls.pt = PT_TCL.read_text()

    def test_01_frozen_population_and_pre_power_selection(self):
        rebuilt = self.selector.select()
        self.assertEqual(rebuilt["selections"], self.selection["selections"])
        self.assertEqual(rebuilt["population"]["rows"], 2880)
        self.assertFalse(rebuilt["representative_filter"]
                         ["uses_cycles_power_energy_or_ppa"])
        self.assertEqual({row["stratum"] for row in rebuilt["selections"]},
                         {"low", "median", "high"})
        self.assertEqual(len({row["sequence"] for row in rebuilt["selections"]}), 3)

    def test_02_descriptor_slice_shas_and_identity(self):
        lines = self.selector.LOW_MEMH.read_text().splitlines()
        for row in self.selection["selections"]:
            self.assertEqual(self.selector.descriptor_sha(lines, row["global_slot"]),
                             row["descriptor_text_sha256"])
            self.assertEqual(row["descriptor_word_count"], 192)
            self.assertGreater(row["ordinary"]["accepted_bank_requests"], 0)
        self.assertEqual(sha(DOC359),
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")

    def test_03_single_dut_compile_time_axis(self):
        self.assertEqual(len(re.findall(
            r"m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend\s*#\s*\(", self.tb)), 1)
        self.assertIn(".SCHEDULE_MODE(SCHEDULE_MODE)", self.tb)
        self.assertIn("`M2217_SCHEDULE_MODE", self.tb)
        self.assertIn(") dut_axis (", self.tb)
        self.assertNotRegex(self.tb, r"dut_(?:base|tsbg)|selective_bank|post_read")
        rows = [line for line in VCS_F.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        self.assertEqual(len(rows), 5)

    def test_04_ucli_single_scope_and_reset_order(self):
        scope = "tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis"
        effective = [line.strip() for line in self.ucli.splitlines()
                     if line.strip() and not line.startswith("#")]
        self.assertEqual(effective.count("power " + scope), 1)
        joined = "\n".join(effective)
        order = [joined.index(token) for token in (
            "power -enable", "run", "power -disable",
            "M2217_PREHISTORY_SAIF_FILE", "power -reset",
            "action=measurement_enable", "action=second_run_returned",
            "M2217_MEASUREMENT_SAIF_FILE")]
        self.assertEqual(order, sorted(order))
        self.assertNotIn("dut_base", joined); self.assertNotIn("dut_tsbg", joined)

    def test_05_runtime_ledger_has_all_six_points(self):
        for row in self.selection["selections"]:
            for mode, key in ((0, "ordinary"), (1, "tsbg")):
                self.assertIn(f"expected_cycles={row[key]['cycles']}", self.tb)
                self.assertIn(f"expected_reads={row[key]['accepted_bank_requests']}", self.tb)
        self.assertIn("req_sum!=expected_reads || rsp_sum!=expected_reads", self.tb)
        self.assertIn("arithmetic mismatch", self.tb)
        self.assertIn("second_axis=0", self.tb)

    def test_06_runner_exact_budget_and_no_retry(self):
        tree = ast.parse(self.runner)
        self.assertIn('"vcs_compiles": 2', self.runner)
        self.assertIn('"simv_runs": 6', self.runner)
        self.assertIn('"dc_runs": 2', self.runner)
        self.assertIn('"ptpx_runs": 6', self.runner)
        self.assertIn('"measurement_saif_files": 6', self.runner)
        self.assertIn('"diagnostic_saif_files": 6', self.runner)
        self.assertIn('"automatic_retry": False', self.runner)
        self.assertNotIn("shutil.rmtree(ATTEMPT", self.runner)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertTrue(calls)
        self.assertIn("for axis, mode in AXES.items():", self.runner)
        self.assertIn("for stratum in STRATA:", self.runner)

    def test_07_fresh_mapping_and_matched_pt_constraints(self):
        for token in ("saif_map -start", "compile_ultra", "SCHEDULE_MODE=>$mode",
                      "ZeroWireload", "clock_period_ns=3.0"):
            self.assertIn(token, self.dc)
        for token in ("tt0p9v25c", "ZeroWireload", "read_saif -strip_path",
                      "ann_pct < 95.0", "toggle_pct < 20.0",
                      "check_power succeeded", "report_power -unit mW"):
            self.assertIn(token, self.pt)
        self.assertIn("weight_sram_dynamic_energy_in_ptpx=false", self.pt)

    def test_08_parser_logic_sram_total_and_weights(self):
        static = self.parser.static_check()
        self.assertEqual(static["status"], "PASS_M2217_STATIC_PARSER")
        model = static["sram_model"]
        self.assertEqual((model["capacity_bytes"], model["macro_count"]),
                         (294912, 16))
        self.assertTrue(model["identical_capacity_area_and_leakage_both_axes"])
        source = PARSER.read_text()
        for token in ("logic_energy_nj", "sram_dynamic_energy_nj",
                      "sram_leakage_energy_nj", "total_energy_nj",
                      "fixed_population_tercile_weights"):
            self.assertIn(token, source)
        self.assertIn('"full_network": False', source)
        self.assertIn('"silicon": False', source)

    def test_09_mutations_fail_conceptually(self):
        mutations = {
            "dual_dut": self.tb + "\nm2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #() dut_tsbg();",
            "bad_weights": json.dumps({**self.selection,
                "aggregate_weights": {"low": [0, 1], "median": [0, 1], "high": [1, 1]}}),
            "scope_pollution": self.ucli.replace(".dut_axis", ""),
            "no_reset": self.ucli.replace("power -reset", "puts no_reset"),
            "no_annotation_gate": self.pt.replace("ann_pct < 95.0", "ann_pct < 0.0"),
        }
        self.assertGreaterEqual(mutations["dual_dut"].count(
            "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #"), 2)
        self.assertNotEqual(json.loads(mutations["bad_weights"])["aggregate_weights"],
                            self.selection["aggregate_weights"])
        self.assertNotIn("power tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis",
                         mutations["scope_pollution"])
        self.assertNotIn("power -reset", mutations["no_reset"])
        self.assertNotIn("ann_pct < 95.0", mutations["no_annotation_gate"])


if __name__ == "__main__":
    unittest.main()
