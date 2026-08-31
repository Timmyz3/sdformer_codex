#!/usr/bin/env python3
"""Static fail-closed tests for M910; no EDA, GPU, remote or license access."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
BUILDER = HW_ROOT / "system_simulator/scripts/build_m910_h67_table_a_component_annex_r11.py"
CONFIG = HW_ROOT / "system_simulator/config/m910_h67_table_a_component_annex_r11_20260829.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load("m910_builder_test", BUILDER)


class M910Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory(prefix="m910_component_annex.")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, value):
        path = Path(self.temp.name) / "config.json"
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False), encoding="utf-8")
        return path

    def row(self):
        return self.config["component_rows"][M.ROW_ID]

    def test_01_canonical_admits_one_component_and_zero_system_rows(self):
        result = M.build(CONFIG)
        self.assertEqual(result["production_component_row_count"], 1)
        self.assertEqual(result["full_system_table_a_production_rows"], 0)
        self.assertFalse(result["system_speedup_admitted"])
        self.assertFalse(result["power_or_energy_admitted"])
        self.assertFalse(result["paper_ppa_ready"])
        self.assertFalse(result["paper_headline_admitted"])

    def test_02_exact_metrics_survive_projection(self):
        row = M.build(CONFIG)["component_annex"][M.ROW_ID]
        self.assertEqual(row["dc_setup_area"]["axes"]["k1"]["cell_area_um2"],
                         "124620.173180")
        self.assertEqual(row["dc_setup_area"]["axes"]["k8"]["cell_area_um2"],
                         "131086.241193")
        self.assertEqual(row["dc_setup_area"]["axes"]["k1x8"]["cell_area_um2"],
                         "585479.153645")
        fair = row["directed_equal_bandwidth_metrics"]
        self.assertEqual((fair["k8_sum_cycles"], fair["k1x8_sum_cycles"]),
                         (1913, 1945))
        self.assertEqual(fair["fair_cycle_speedup_x"], "1.01672765")
        self.assertEqual(fair["fair_throughput_per_mm2_x"], "4.541077998")
        self.assertEqual(fair["logic_cell_area_saving_percent"], "77.6104")

    def test_03_unpinned_authority_rejects(self):
        self.row()["authority"]["review_sha256"] = "0" * 64
        with self.assertRaisesRegex(M.AnnexError, "authority"):
            M.build(self.write(self.config))

    def test_04_area_mutation_rejects(self):
        self.row()["dc_setup_area"]["axes"]["k8"]["cell_area_um2"] = "1.0"
        with self.assertRaisesRegex(M.AnnexError, "axis"):
            M.build(self.write(self.config))

    def test_05_cycle_mutation_rejects(self):
        self.row()["directed_equal_bandwidth_metrics"]["k8_sum_cycles"] = 1912
        with self.assertRaisesRegex(M.AnnexError, "fair metric"):
            M.build(self.write(self.config))

    def test_06_system_speedup_escalation_rejects(self):
        self.row()["claim_boundary"]["system_speedup"] = True
        with self.assertRaisesRegex(M.AnnexError, "claim boundary"):
            M.build(self.write(self.config))

    def test_07_macro_inclusive_escalation_rejects(self):
        self.row()["claim_boundary"]["macro_inclusive"] = True
        with self.assertRaisesRegex(M.AnnexError, "claim boundary"):
            M.build(self.write(self.config))

    def test_08_k8_vs_single_k1_headline_rejects(self):
        self.row()["claim_boundary"]["k8_vs_single_k1_performance_headline"] = True
        with self.assertRaisesRegex(M.AnnexError, "claim boundary"):
            M.build(self.write(self.config))

    def test_09_full_system_row_escalation_rejects(self):
        self.config["admission_boundary"]["table_a_full_system_production_rows"] = 1
        with self.assertRaisesRegex(M.AnnexError, "admission boundary"):
            M.build(self.write(self.config))

    def test_10_second_self_authored_row_rejects(self):
        self.config["component_rows"]["fake"] = copy.deepcopy(self.row())
        with self.assertRaisesRegex(M.AnnexError, "exactly the pinned"):
            M.build(self.write(self.config))

    def test_11_unknown_field_rejects(self):
        self.row()["paper_ready"] = True
        with self.assertRaisesRegex(M.AnnexError, "fields differ"):
            M.build(self.write(self.config))

    def test_12_protected_sha_rejects(self):
        self.config["protected_file"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(M.AnnexError, "protected-file"):
            M.build(self.write(self.config))


if __name__ == "__main__":
    unittest.main(verbosity=2)
