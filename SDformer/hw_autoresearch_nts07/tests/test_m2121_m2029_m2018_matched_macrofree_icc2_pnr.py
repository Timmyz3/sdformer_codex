import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / "system_simulator/scripts/parse_m2121_m2029_m2018_matched_macrofree_icc2_pnr.py"
SPEC = importlib.util.spec_from_file_location("m2121_parser", PARSER)
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
    report_payloads = {
        "actual_floorplan.txt": "die_boundary={{0 0} {800 0} {800 800} {0 800}}\ncore_bbox={{40 40} {760 760}}\n",
        "actual_routing_layers.rpt": "Min routing layer: M2\nMax routing layer: M8\n",
        "actual_cts_cells.txt": "CKBD1\nCKND1\n",
        "actual_hold_cells.txt": "BUFF1\nDEL1\nINV1\n",
        "actual_scenarios.rpt": "func_ss_setup setup=true\nfunc_ff_hold hold=true\nfunc_tt_power power=true\n",
    }
    for name, payload in report_payloads.items():
        (reports / name).write_text(payload)
    routing_bytes = b"".join((reports / name).read_bytes() for name in (
        "actual_routing_layers.rpt", "actual_cts_cells.txt", "actual_hold_cells.txt"))
    facts = {
        "status": MOD.PASS_TOKEN,
        "axis": axis,
        "top": "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend",
        "public_port_count": "4551",
        "input_master_count": "94",
        "tt_master_coverage": "94/94",
        "ss_master_coverage": "94/94",
        "ff_master_coverage": "94/94",
        "physical_master_coverage": "94/94",
        "unresolved_reference_count": "0",
        "accepted_mismatch_count": "0",
        "logical_physical_mismatch_count": "0",
        "routing_layer_gate_count": "9",
        "via_layer_gate_count": "8",
        "route_check_return": "1",
        "pre_placement_check_return": "1",
        "pre_clock_check_return": "1",
        "pre_route_check_return": "1",
        "die_bbox_um": "0,0,800,800",
        "core_bbox_um": "40,40,760,760",
        "die_boundary_actual": "{{0 0} {800 0} {800 800} {0 800}}",
        "core_bbox_actual": "{{40 40} {760 760}}",
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
        "setup_scenario_actual": "func_ss_setup",
        "hold_scenario_actual": "func_ff_hold",
        "power_scenario_actual": "func_tt_power",
        "common_external_sram_bytes": "294912",
        "common_external_sram_integrated": "false",
        "propagated_clock": "true",
        "macro_instances": "0",
        "physical_sdc_sha256": "a" * 64,
        "flow_tcl_sha256": "b" * 64,
        "floorplan_actual_sha256": MOD.sha256(reports / "actual_floorplan.txt"),
        "routing_policy_sha256": MOD.hashlib.sha256(routing_bytes).hexdigest(),
        "scenario_policy_sha256": MOD.sha256(reports / "actual_scenarios.rpt"),
        "route_open_net_count": "0",
        "route_drc_violation_count": "0",
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
        (reports / name).write_text(
            "Total number of open nets = 0\nTotal number of DRC violations = 0\n"
            if name == "route_check.rpt" else "fixture\n")
    pins = "\n".join(
        f"- p{i:04d} + NET p{i:04d} + DIRECTION INPUT + USE SIGNAL "
        f"+ LAYER M3 ( 0 0 ) ( 10 10 ) + FIXED ( {i} 0 ) N ;"
        for i in range(4551))
    routed_def = (
        "VERSION 5.8 ;\nUNITS DISTANCE MICRONS 1000 ;\n"
        "DIEAREA ( 0 0 ) ( 800000 800000 ) ;\n"
        f"PINS 4551 ;\n{pins}\nEND PINS\nEND DESIGN\n")
    (output / "routed.def").write_text(routed_def)
    for name in ("routed.v", "routed.sdc", "routed.spef"):
        (output / name).write_text("fixture\n")
    return root


class M2121ParserTests(unittest.TestCase):
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

    def test_route_count_mutations_fail_closed(self):
        for field, replacement in (("open nets = 0", "open nets = 999"),
                                   ("DRC violations = 0", "DRC violations = 777")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                ordinary = make_axis(root / "ordinary", "ordinary_lru4", 260000.0)
                tsbg = make_axis(root / "tsbg", "tsbg_b4", 260100.0)
                report = tsbg / "reports/route_check.rpt"
                report.write_text(report.read_text().replace(field, replacement))
                with self.assertRaisesRegex(ValueError, "nonzero or contradictory route counts"):
                    MOD.parse_pair(ordinary, tsbg)

    def test_spef_scenario_is_not_spef(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ordinary = make_axis(root / "ordinary", "ordinary_lru4", 260000.0)
            tsbg = make_axis(root / "tsbg", "tsbg_b4", 260100.0)
            spef = tsbg / "output/routed.spef"
            spef.unlink()
            (tsbg / "output/routed.spef_scenario").write_text("not parasitics\n")
            with self.assertRaisesRegex(ValueError, "require exactly routed.spef"):
                MOD.parse_pair(ordinary, tsbg)

    def test_actual_def_pin_inventory_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ordinary = make_axis(root / "ordinary", "ordinary_lru4", 260000.0)
            tsbg = make_axis(root / "tsbg", "tsbg_b4", 260100.0)
            path = tsbg / "output/routed.def"
            path.write_text(path.read_text().replace("FIXED ( 42 0 ) N", "FIXED ( 43 0 ) N", 1))
            with self.assertRaisesRegex(ValueError, "actual DEF"):
                MOD.parse_pair(ordinary, tsbg)


if __name__ == "__main__":
    unittest.main()
