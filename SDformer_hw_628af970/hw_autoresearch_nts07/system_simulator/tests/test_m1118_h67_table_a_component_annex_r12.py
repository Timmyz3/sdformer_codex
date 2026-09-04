#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1118 r12 fail-closed source tests; no EDA, GPU, remote, or production."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
HW = REPO_ROOT / "hw_autoresearch_nts07"
BUILDER = HW / "system_simulator/scripts/build_m1118_h67_table_a_component_annex_r12.py"
CONFIG = HW / "system_simulator/config/m1118_h67_table_a_component_annex_r12_20260830.json"


def load_module():
    spec = importlib.util.spec_from_file_location("m1118_test_subject", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


M = load_module()


class M1118Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory(prefix="m1118_component_annex_")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, value):
        path = Path(self.temp.name) / "config.json"
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False), encoding="utf-8")
        return path

    def row(self, key):
        return self.config["additive_component_rows"][key]

    def reject(self, pattern=""):
        context = self.assertRaises(M.AnnexError) if not pattern else self.assertRaisesRegex(M.AnnexError, pattern)
        return context

    def test_01_canonical_three_components_zero_system_rows(self):
        result = M.build(CONFIG)
        self.assertEqual(set(result["component_annex"]), {M.C1, M.C2, M.C3})
        self.assertEqual(result["component_annex_row_count"], 3)
        self.assertEqual(result["full_system_table_a_production_rows"], 0)
        self.assertFalse(result["system_speedup_admitted"])
        self.assertFalse(result["power_or_energy_admitted"])
        self.assertFalse(result["final_checkpoint_bound"])
        self.assertFalse(result["paper_ppa_ready"])

    def test_02_c1_exact_raw_cpu_projection(self):
        row = M.build(CONFIG)["component_annex"][M.C1]
        metric = row["raw_cpu_same_ledger_metrics"]
        self.assertEqual(metric["candidate_cycles"], 434242823)
        self.assertEqual(metric["strongest_zero_cycles"], 763908050)
        self.assertEqual(metric["candidate_vs_strongest_zero_x"], "1.7591725401987818")
        self.assertEqual((metric["capacity_ledger_bytes"], metric["capacity_budget_bytes"],
                          metric["capacity_margin_bytes"]), (214912, 245760, 30848))
        self.assertTrue(row["claim_boundary"]["raw_cpu_same_ledger_component_speedup"])
        for key in ("rtl_cycles", "rtl_speedup", "mapped_gate", "final_checkpoint_bound",
                    "full_network", "decoder_complete", "system_speedup", "power", "energy"):
            self.assertFalse(row["claim_boundary"][key])

    def test_03_c3_exact_setup_area_projection(self):
        row = M.build(CONFIG)["component_annex"][M.C3]
        dc = row["dc_setup_area"]
        self.assertEqual(dc["cell_area_um2"], "62433.503388")
        self.assertEqual(dc["minimum_reported_setup_slack_ns"], "+0.0003")
        self.assertEqual(dc["clock_period_ns"], "3.000")
        self.assertTrue(row["claim_boundary"]["setup_area_citable"])
        for key in ("hold_closed", "pt_sta_completed", "macro_inclusive", "throughput",
                    "speedup", "system", "power", "energy", "paper_ppa_ready"):
            self.assertFalse(row["claim_boundary"][key])

    def test_04_duplicate_key_rejected(self):
        path = Path(self.temp.name) / "duplicate.json"
        path.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
        with self.reject("duplicate JSON key"):
            M.build(path)

    def test_05_nan_rejected(self):
        path = Path(self.temp.name) / "nan.json"
        path.write_text('{"schema":NaN}\n', encoding="utf-8")
        with self.reject("nonfinite JSON"):
            M.build(path)

    def test_06_c1_speedup_mutation_rejected(self):
        self.row(M.C1)["raw_cpu_same_ledger_metrics"]["candidate_vs_strongest_zero_x"] = "2.0"
        with self.reject("C1 metrics"):
            M.build(self.write(self.config))

    def test_07_c1_rtl_claim_escalation_rejected(self):
        self.row(M.C1)["claim_boundary"]["rtl_speedup"] = True
        with self.reject("C1 claim boundary"):
            M.build(self.write(self.config))

    def test_08_c1_final_checkpoint_escalation_rejected(self):
        self.row(M.C1)["claim_boundary"]["final_checkpoint_bound"] = True
        with self.reject("C1 claim boundary"):
            M.build(self.write(self.config))

    def test_09_c1_mapped_claim_escalation_rejected(self):
        self.row(M.C1)["claim_boundary"]["mapped_gate"] = True
        with self.reject("C1 claim boundary"):
            M.build(self.write(self.config))

    def test_10_c1_capacity_mutation_rejected(self):
        self.row(M.C1)["raw_cpu_same_ledger_metrics"]["capacity_ledger_bytes"] = 200000
        with self.reject("C1 metrics"):
            M.build(self.write(self.config))

    def test_11_c3_second_row_area_mutation_rejected(self):
        self.row(M.C3)["dc_setup_area"]["cell_area_um2"] = "1.0"
        with self.reject("C3 DC metrics"):
            M.build(self.write(self.config))

    def test_12_c3_second_row_speedup_escalation_rejected(self):
        self.row(M.C3)["claim_boundary"]["speedup"] = True
        with self.reject("C3 claim boundary"):
            M.build(self.write(self.config))

    def test_13_c3_second_row_hold_escalation_rejected(self):
        self.row(M.C3)["claim_boundary"]["hold_closed"] = True
        with self.reject("C3 claim boundary"):
            M.build(self.write(self.config))

    def test_14_c3_second_row_power_escalation_rejected(self):
        self.row(M.C3)["claim_boundary"]["power"] = True
        with self.reject("C3 claim boundary"):
            M.build(self.write(self.config))

    def test_15_full_system_row_escalation_rejected(self):
        self.config["admission_boundary"]["table_a_full_system_production_rows"] = 1
        with self.reject("admission boundary"):
            M.build(self.write(self.config))

    def test_16_extra_additive_row_rejected(self):
        self.config["additive_component_rows"]["fake"] = copy.deepcopy(self.row(M.C1))
        with self.reject("exactly two additive rows"):
            M.build(self.write(self.config))

    def test_17_c2_predecessor_authority_mutation_rejected(self):
        self.config["sealed_component_annex_r11"]["hammer_review_sha256"] = "0" * 64
        with self.reject("M910 config authority"):
            M.build(self.write(self.config))

    def test_18_c1_authority_mutation_rejected(self):
        self.row(M.C1)["authority"]["review_sha256"] = "0" * 64
        with self.reject("C1 authority"):
            M.build(self.write(self.config))

    def test_19_c3_authority_mutation_rejected(self):
        self.row(M.C3)["authority"]["outer_seal_file_sha256"] = "0" * 64
        with self.reject("C3 authority"):
            M.build(self.write(self.config))

    def test_20_unknown_field_rejected(self):
        self.row(M.C3)["claim_boundary"]["novel_speedup"] = True
        with self.reject("C3 claim boundary"):
            M.build(self.write(self.config))

    def test_21_docs359_identity_mutation_rejected(self):
        self.config["protected_file"]["sha256"] = "0" * 64
        with self.reject("protected-file"):
            M.build(self.write(self.config))


if __name__ == "__main__":
    unittest.main(verbosity=2)
