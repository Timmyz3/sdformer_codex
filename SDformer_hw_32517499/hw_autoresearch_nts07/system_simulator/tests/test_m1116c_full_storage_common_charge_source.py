#!/usr/bin/env python3
"""Bounded source-only tests for M1116C; no VCS, EDA, or full replay."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parents[1] / "verif_m1116c_c1_full_storage_boundary/static_check_m1116c_full_storage_common_charge_source.py"
SPEC = importlib.util.spec_from_file_location("m1116c_source_checker_tested", CHECKER)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1116CSourceTest(unittest.TestCase):
    def test_exact_mapping(self):
        value = M.parse_mapping()
        self.assertEqual(value["totals"]["represented_bytes"], 214_912)
        self.assertEqual(value["totals"]["margin_bytes"], 30_848)

    def test_only_nine_internal_parent_macros(self):
        value = M.parse_mapping()["totals"]
        self.assertEqual(value["physical_macro_count"], 9)
        self.assertEqual(value["internal_macro_bytes"], 18_432)

    def test_external_common_charge_is_not_physical_macro(self):
        value = M.parse_mapping()
        external = [row for row in value["rows"]
                    if row["placement"] == "identical_external_common_charge"]
        self.assertEqual(sum(row["bytes"] for row in external), 196_480)
        self.assertTrue(all(row["physical_macro_count"] == 0 for row in external))

    def test_mapping_gap_is_rejected(self):
        text = M.MAPPING.read_text().replace(
            "psum_store|18432|141311|122880", "psum_store|18433|141311|122879")
        with tempfile.TemporaryDirectory(prefix="m1116c_gap_") as tmpdir:
            path = Path(tmpdir) / "bad.tsv"
            path.write_text(text)
            with self.assertRaisesRegex(RuntimeError, "gap/overlap"):
                M.parse_mapping(path)

    def test_wrapper_binds_live_services(self):
        value = M.check_wrapper()
        self.assertTrue(value["live_weight_service"])
        self.assertTrue(value["live_psum_read_write_service"])
        self.assertEqual(value["direct_macro_instances"], 0)

    def test_synthesis_only_filelist(self):
        value = M.check_filelist()
        self.assertTrue(value["synthesis_only"])
        self.assertEqual(value["tb_sva_attack_members"], 0)

    def test_zero_exception_3ns_sdc(self):
        value = M.check_sdc()
        self.assertEqual(value["clock_period_ns"], 3.0)
        self.assertFalse(any(value["timing_exception_counts"].values()))

    def test_tcl_derives_mapping_without_dummy_macro_target(self):
        value = M.check_tcl()
        self.assertTrue(value["mapping_derived"])
        self.assertFalse(value["literal_93_or_105_macro_target"])


if __name__ == "__main__":
    unittest.main()
