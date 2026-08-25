import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / (
    "hw_autoresearch_nts07/dc_handoff/scripts/"
    "audit_m31_r4_dc_reports.py"
)
SPEC = importlib.util.spec_from_file_location("m31dcaudit", str(SCRIPT))
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditM31R4DcReportsTest(unittest.TestCase):
    def make_run(self, directory):
        run = Path(directory) / "run"
        reports = run / "reports"
        netlist = run / "netlist"
        reports.mkdir(parents=True)
        netlist.mkdir()
        (reports / "qor.rpt").write_text(
            "Design : qfit_atlif_unified_t10_t2_stream_core\n"
            "  Hierarchical Cell Count: 97\n"
            "  Leaf Cell Count: 96\n"
            "  Macro Count: 0\n"
            "  Net Area: 0.000000\n"
        )
        (reports / "area.rpt").write_text(
            "Number of cells: 193\n"
            "Number of combinational cells: 96\n"
            "Number of sequential cells: 0\n"
            "Number of macros/black boxes: 0\n"
            "Net Interconnect area: undefined  (Wire load has zero net area)\n"
            "Total cell area: 100.000000\n"
        )
        (reports / "clocks.rpt").write_text(
            "core_clk 3.00 {0 1.5} f {clk_core}\n"
        )
        rows = [
            "stage=postcompile", "pool_count=1", "leaf_count=96",
            "pool_path=u_mul_pool",
        ]
        rows.extend(
            "leaf=u_mul_pool/u{} ref=qfit_signed_int8_mul_leaf_{} "
            "mapped_cells=1 mapped_area=1.0".format(index, index)
            for index in range(96)
        )
        rows.extend([
            "pool_external_leaf_count=0", "empty_mapped_leaf_count=0",
            "status=PASS_EXACT_ONE_POOL_96_LEAVES",
        ])
        (reports / "m31_resource_audit_postcompile.rpt").write_text(
            "\n".join(rows) + "\n"
        )
        (reports / "references_postcompile.rpt").write_text("mapped cells\n")
        (reports / "timing_setup.rpt").write_text(" slack (MET) 0.0010\n")
        (reports / "timing_hold.rpt").write_text(" slack (MET) 0.0020\n")
        (run / "dc.log").write_text("DC complete\n")
        (netlist / (AUDIT.DESIGN + "_mapped.v")).write_text("module m; endmodule\n")
        (netlist / (AUDIT.DESIGN + ".svf")).write_text("svf\n")
        return run

    def test_positive_strict_cell_and_physical_model_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = AUDIT.build(self.make_run(directory))
            cells = result["cell_accounting"]
            self.assertEqual(cells["total_cell_instances_including_hierarchy"], 193)
            self.assertEqual(cells["hierarchical_cell_instances"], 97)
            self.assertEqual(cells["leaf_mapped_cell_instances"], 96)
            self.assertEqual(
                result["physical_assumptions"]["clock_network_model"],
                "IDEAL_UNPROPAGATED",
            )
            self.assertEqual(
                result["physical_assumptions"]["interconnect_area_model"],
                "ZERO_WIRE_LOAD",
            )

    def test_total_hierarchical_leaf_conflation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            area = run / "reports/area.rpt"
            area.write_text(area.read_text().replace(
                "Number of cells: 193", "Number of cells: 96"))
            with self.assertRaisesRegex(ValueError, "accounting drift"):
                AUDIT.build(run)

    def test_propagated_clock_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            clocks = run / "reports/clocks.rpt"
            clocks.write_text(clocks.read_text().replace(
                "{0 1.5} f", "{0 1.5} fp"))
            with self.assertRaisesRegex(ValueError, "propagated"):
                AUDIT.build(run)

    def test_empty_multiplier_leaf_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            resource = run / "reports/m31_resource_audit_postcompile.rpt"
            resource.write_text(resource.read_text().replace(
                "mapped_cells=1 mapped_area=1.0",
                "mapped_cells=0 mapped_area=0.0", 1))
            with self.assertRaisesRegex(ValueError, "empty or external"):
                AUDIT.build(run)


if __name__ == "__main__":
    unittest.main()
