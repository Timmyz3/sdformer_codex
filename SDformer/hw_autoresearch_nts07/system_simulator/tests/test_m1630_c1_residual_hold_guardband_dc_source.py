#!/usr/bin/env python3
"""Static author tests for source-only M1630 C1 residual-hold closure."""
from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
TCL = HW / "dc_handoff/scripts/run_dc_m1630_m993_c1_residual_hold_guardband_candidate.tcl"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1630_m993_c1_residual_hold_guardband_exact_sha_r1.sh"
CONTRACT = HW / "contracts/m1630_m993_c1_residual_hold_guardband_dc_source_contract_r1_20260901.json"
M993 = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
ORIGINAL = M993 / "original_quarantine"
M1006 = HW / "reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829"
M1614_NEGATIVE = HW / "dc_handoff/runs/m1614_m993_c1_macro_aware_hold_only_incremental_dc_r1_20260901.failed_or_incomplete.4065447.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "dc_handoff/runs/m1630_m993_c1_residual_hold_guardband_dc_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1630_m993_c1_residual_hold_guardband_dc_attempt_consumed"

EXPECTED = {
    TCL: "e1a138284a4e558c339d987fd4da8e248b0c2ea81858fe255e812c0a4ec592c1",
    RUNNER: "ceb4dcdf1a90b9285e10b06a6ff9e91faf8f840373dc6d2095c50004e7493b81",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.ddc":
        "d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc":
        "cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v":
        "9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.svf":
        "8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7",
    M1006 / "review.json":
        "d7b30ff3a82a099c080f3aa3dd32c13c1d2d5b5e278112eb9e3b1c24588809ea",
    M1614_NEGATIVE / "reports/setup_posthold_summary_machine.txt":
        "ff8f206815233c222c781a507d3ce504571148885ff7b113c5f94cb8824f639b",
    M1614_NEGATIVE / "reports/hold_posthold_summary_machine.txt":
        "fb12725e8bc76cce0c9f8198cb8b915dc91cf8c0fbb4e185b4fae2da2414da8c",
    M1614_NEGATIVE / "reports/area_posthold.rpt":
        "aa109ac641cbee88d6617e4b4f3008f6669a91d51e055bb151d7b3c324ec655e",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"):
        "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    Path("/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec"):
        "bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391",
    Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"):
        "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"):
        "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
    Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"):
        "cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf",
    Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db"):
        "8c163161060d8d4415837da4ad65bbd83c99eb64872df76f5e0adc0b18cedb5f",
}

FORBIDDEN = (
    "set_false_path", "set_multicycle_path", "set_min_delay",
    "set_max_delay", "set_disable_timing", "set_case_analysis",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command_text(text, comment="#"):
    return "\n".join(row.split(comment, 1)[0] for row in text.splitlines())


def verify_tree(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or name in expected:
            raise AssertionError("unsafe manifest row " + name)
        expected[name] = digest
    if outer.read_text(encoding="ascii").split() != [sha256(manifest),
                                                       "SHA256SUMS"]:
        raise AssertionError("outer seal drift " + str(root))
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        base_path = Path(base)
        dirs[:] = [name for name in dirs if not (base_path / name).is_symlink()]
        for name in files:
            path = base_path / name
            if path.name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            if not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink():
                raise AssertionError("nonregular tree member " + str(path))
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise AssertionError((set(expected) - actual, actual - set(expected)))
    for name, digest in expected.items():
        if sha256(root / name) != digest:
            raise AssertionError("tree member drift " + name)


def parse_kv(path):
    return dict(line.split("=", 1) for line in
                path.read_text(encoding="utf-8", errors="replace").splitlines()
                if "=" in line)


class M1630SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tcl = TCL.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.tcl_commands = command_text(cls.tcl)
        cls.runner_commands = command_text(cls.runner)

    def test_01_exact_source_input_evidence_tool_library_identities(self):
        for path, digest in EXPECTED.items():
            self.assertEqual(sha256(path), digest, str(path))
            if str(path).startswith("/opt/synopsys/syn/"):
                self.assertTrue(path.is_file(), str(path))
            else:
                self.assertTrue(stat.S_ISREG(path.lstat().st_mode), str(path))
                self.assertFalse(path.is_symlink(), str(path))

    def test_02_admitted_and_negative_evidence_trees_are_sealed(self):
        for root in (M993, ORIGINAL, M1006, M1614_NEGATIVE):
            verify_tree(root)

    def test_03_shell_is_syntax_clean_without_execution(self):
        completed = subprocess.run(
            ["/usr/bin/bash", "-n", str(RUNNER)], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=10, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_04_original_m993_is_only_ddc_input(self):
        self.assertEqual(len(re.findall(r"(?m)^\s*read_ddc\b",
                                        self.tcl_commands)), 1)
        self.assertIn('INPUT_DDC="${M993_ORIGINAL}/netlist/', self.runner)
        self.assertIn("input_generation=original_m993_m1006_admitted_ddc",
                      self.tcl)
        self.assertIn("failed_m1614_output_used=false", self.tcl)
        self.assertIsNone(re.search(r"(?m)^INPUT_DDC=.*m1614", self.runner))
        self.assertNotIn("m1614_hold_repaired.ddc", self.tcl)

    def test_05_current_design_uses_object_name(self):
        self.assertIn("set design_collection [current_design]", self.tcl)
        self.assertIn("set active_design [get_object_name $design_collection]",
                      self.tcl)
        self.assertIn("if {$active_design ne $design_name}", self.tcl)
        self.assertNotIn("if {[current_design] ne $design_name}", self.tcl)

    def test_06_exactly_one_hold_only_mapping(self):
        self.assertEqual(len(re.findall(
            r"(?m)^\s*set_fix_hold\s+\$core_clock\s*$",
            self.tcl_commands)), 1)
        self.assertEqual(len(re.findall(
            r"(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$",
            self.tcl_commands)), 1)
        self.assertEqual(len(re.findall(r"(?m)^\s*compile\b",
                                        self.tcl_commands)), 1)
        self.assertIsNone(re.search(r"(?m)^\s*compile_ultra\b",
                                    self.tcl_commands))

    def test_07_one_ps_guardband_then_restore(self):
        guard = self.tcl.index(
            "set_clock_uncertainty -hold $optimization_hold_guardband_ns $core_clock")
        compile_at = self.tcl.index("compile -incremental_mapping -only_hold_time")
        restore = self.tcl.index(
            "set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock")
        reports = self.tcl.index("report_qor > \"$output_dir/reports/qor_posthold.rpt\"")
        self.assertLess(guard, compile_at)
        self.assertLess(compile_at, restore)
        self.assertLess(restore, reports)
        self.assertEqual(self.tcl.count("set optimization_hold_guardband_ns 0.051"), 1)
        self.assertEqual(self.tcl.count("set reported_hold_uncertainty_ns 0.050"), 1)

    def test_08_reported_constraints_and_no_concealment(self):
        input_sdc = (ORIGINAL /
            "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc").read_text(encoding="utf-8")
        joined = self.tcl_commands + "\n" + command_text(input_sdc)
        for token in FORBIDDEN:
            self.assertIsNone(re.search(r"(?m)^\s*" + token + r"\b", joined),
                              token)
        self.assertEqual(len(re.findall(r"(?m)^\s*create_clock\b", input_sdc)), 1)
        self.assertRegex(input_sdc,
                         r"create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)")
        self.assertRegex(input_sdc,
                         r"set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b")
        self.assertRegex(input_sdc,
                         r"set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b")
        self.assertIn("set_wire_load_model -name ZeroWireload", self.tcl)

    def test_09_nine_exact_macros_and_min_libraries(self):
        for token in (
            "set expected_macro_count 9",
            "set_min_library $std_slow_db -min_version $std_fast_db",
            "set_min_library $macro_slow_db -min_version $macro_fast_db",
            "set_dont_touch $macro_cells_pre true",
            "if {$macro_count_pre != $expected_macro_count}",
            "if {$macro_count_post != $expected_macro_count}",
        ):
            self.assertEqual(self.tcl.count(token), 1, token)

    def test_10_clean_dc_invocation_and_no_home_repurpose(self):
        self.assertIn(
            '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"',
            self.runner)
        self.assertIsNone(re.search(r"(?m)^\s*(?:export\s+)?HOME=",
                                    self.runner_commands))
        self.assertNotIn("~/.synopsys", self.runner)

    def test_11_error_fatal_link_loop_fail_closed(self):
        for token in ("(Error|Fatal):", "LINK-[0-9]+",
                      "unresolved (reference|design|cell)",
                      "combinational[ _-]*loop", "timing[ _-]*loop",
                      "(TIM-209|OPT-150)"):
            self.assertIn(token, self.runner)
        for report in ("reports/link.rpt", "reports/check_design_prehold.rpt",
                       "reports/check_timing_prehold.rpt",
                       "reports/check_design_posthold.rpt",
                       "reports/check_timing_posthold.rpt"):
            self.assertIn(report, self.runner)

    def test_12_one_shot_future_review_release_and_no_current_attempt(self):
        self.assertIn("m1631_m1630_c1_residual_hold_guardband_dc_source_hammer",
                      self.runner)
        self.assertIn("m1632_m1631_m1630_c1_residual_hold_guardband_dc_launch_release",
                      self.runner)
        self.assertIn("M1630_EXPECTED_DC_RUNNER_SHA256", self.runner)
        self.assertIn("M1630_EXPECTED_DC_RELEASE_SHA256", self.runner)
        self.assertLess(self.runner.index('verify_dir_seal "${HAMMER_DIR}"'),
                        self.runner.index('mkdir -- "${ATTEMPT}"'))
        self.assertFalse(ATTEMPT.exists())
        self.assertFalse(RESULT.exists())
        self.assertNotIn("rm -rf", self.runner)

    def test_13_negative_m1614_motivation_is_exact(self):
        setup = parse_kv(M1614_NEGATIVE /
                         "reports/setup_posthold_summary_machine.txt")
        hold = parse_kv(M1614_NEGATIVE /
                        "reports/hold_posthold_summary_machine.txt")
        area = (M1614_NEGATIVE / "reports/area_posthold.rpt").read_text(
            encoding="utf-8", errors="replace")
        self.assertEqual(setup["wns_ns"], "0.001718520")
        self.assertEqual(setup["violating_paths"], "0")
        self.assertEqual(hold["wns_ns"], "-0.000353523")
        self.assertEqual(hold["tns_ns"], "-0.000401557")
        self.assertEqual(hold["violating_paths"], "3")
        self.assertRegex(area, r"Total cell area:\s+152834\.995973")

    def test_14_contract_binds_source_and_opens_no_execution(self):
        c = self.contract
        manifest = Path(str(CONTRACT) + ".sha256")
        outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
        self.assertEqual(manifest.read_text(encoding="ascii").split(),
                         [sha256(CONTRACT), CONTRACT.name])
        self.assertEqual(outer.read_text(encoding="ascii").split(),
                         [sha256(manifest), manifest.name])
        self.assertEqual(c["status"],
            "SOURCE_ONLY_M1630_C1_RESIDUAL_HOLD_GUARDBAND__NO_EDA_AUTHORIZED")
        self.assertEqual(c["identity"]["tcl_sha256"], EXPECTED[TCL])
        self.assertEqual(c["identity"]["runner_sha256"], EXPECTED[RUNNER])
        self.assertEqual(c["authorization"]["dc_runs_now"], 0)
        self.assertEqual(c["authorization"]["future_dc_runs_max"], 1)
        self.assertFalse(c["claim_boundary"]["hold_closed"])
        self.assertFalse(c["claim_boundary"]["paper_citable"])

    def test_15_only_dc_is_future_and_all_other_tools_closed(self):
        for token in ("pt_shell", "fm_shell", "vcs ", "simv ", "ptpx"):
            self.assertNotIn(token, self.runner_commands.lower())
        c = self.contract
        for field in ("vcs_runs", "pt_runs", "formality_runs",
                      "ptpx_runs", "gpu_runs", "remote_runs"):
            self.assertEqual(c["authorization"][field], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
