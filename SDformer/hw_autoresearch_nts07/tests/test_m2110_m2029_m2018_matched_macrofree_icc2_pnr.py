import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / "system_simulator/scripts/parse_m2110_m2029_m2018_matched_macrofree_icc2_pnr.py"
SPEC = importlib.util.spec_from_file_location("m2110_parser", PARSER)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def make_axis(root, axis, area, hold=0.001):
    reports = root / "reports"
    output = root / "output"
    reports.mkdir(parents=True)
    output.mkdir()
    ports = "".join(f"p{i:04d}\n" for i in range(4551))
    (reports / "ports_sorted.txt").write_text(ports)
    port_sha = MOD.sha256(reports / "ports_sorted.txt")
    facts = {
        "status": MOD.PASS_TOKEN,
        "axis": axis,
        "top": "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend",
        "public_port_count": "4551",
        "input_master_count": "94",
        "unresolved_reference_count": "0",
        "logical_physical_mismatch_count": "0",
        "routing_layer_gate_count": "9",
        "via_layer_gate_count": "8",
        "route_check_return": "1",
        "pre_placement_check_return": "1",
        "pre_clock_check_return": "1",
        "pre_route_check_return": "1",
        "die_bbox_um": "0,0,800,800",
        "core_bbox_um": "40,40,760,760",
        "floorplan_policy": "fixed_die_core_800_720um_v1",
        "pin_policy": "sorted_four_side_round_robin_exact_location_v1",
        "route_layers": "M2:M8",
        "cts_cell_policy": "CKBD_and_CKND_only_v1",
        "hold_cell_policy": "DEL_BUFF_INV_only_v1",
        "clock_period_ns": "3.000",
        "setup_uncertainty_ns": "0.200",
        "hold_uncertainty_ns": "0.050",
        "parasitic_tech": "n28_1p9m_6x1z1u_typ",
        "parasitic_corner_scope": "same_typical_rc_on_ss_ff_tt",
        "common_external_sram_bytes": "294912",
        "common_external_sram_integrated": "false",
        "propagated_clock": "true",
        "macro_instances": "0",
        "physical_sdc_sha256": "a" * 64,
        "port_inventory_sha256": port_sha,
        "setup_wns_ns": "0.003",
        "hold_wns_ns": str(hold),
        "routed_standard_cell_area_um2": str(area),
        "routed_leaf_cell_count": "270000",
        "routed_sequential_cell_count": "74460",
        "clock_like_cell_count": "900",
        "hold_like_cell_count": "2000",
    }
    (root / "machine_facts.txt").write_text("".join(f"{k}={v}\n" for k, v in facts.items()))
    (root / "RUN_COMPLETE.txt").write_text(MOD.PASS_TOKEN + "\n")
    for name in (
        "reference_libraries.rpt", "design_mismatch.rpt", "pre_placement_check.rpt",
        "pre_clock_check.rpt", "pre_route_check.rpt", "route_check.rpt", "qor.rpt",
        "timing_setup.rpt", "timing_hold.rpt", "clock_qor.rpt", "congestion.rpt",
        "wirelength.rpt",
    ):
        (reports / name).write_text("fixture\n")
    for name in ("routed.v", "routed.sdc", "routed.def", "routed.n28_typ.spef"):
        (output / name).write_text("fixture\n")
    return root


class M2110ParserTests(unittest.TestCase):
    def test_pair_accepts_only_matched_hold_clean_axes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ordinary = make_axis(root / "ordinary", "ordinary_lru4", 260000.0)
            tsbg = make_axis(root / "tsbg", "tsbg_b4", 260100.0)
            result = MOD.parse_pair(ordinary, tsbg)
            self.assertTrue(result["comparison"]["both_hold_met"])
            self.assertFalse(result["claim_boundary"]["macro_inclusive"])

    def test_negative_hold_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ordinary = make_axis(root / "ordinary", "ordinary_lru4", 260000.0, hold=-0.001)
            tsbg = make_axis(root / "tsbg", "tsbg_b4", 260100.0)
            with self.assertRaisesRegex(ValueError, "timing/area admission failed"):
                MOD.parse_pair(ordinary, tsbg)

    def test_unmatched_floorplan_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ordinary = make_axis(root / "ordinary", "ordinary_lru4", 260000.0)
            tsbg = make_axis(root / "tsbg", "tsbg_b4", 260100.0)
            path = tsbg / "machine_facts.txt"
            path.write_text(path.read_text().replace(
                "core_bbox_um=40,40,760,760", "core_bbox_um=40,40,750,750"))
            with self.assertRaises(ValueError):
                MOD.parse_pair(ordinary, tsbg)


if __name__ == "__main__":
    unittest.main()
