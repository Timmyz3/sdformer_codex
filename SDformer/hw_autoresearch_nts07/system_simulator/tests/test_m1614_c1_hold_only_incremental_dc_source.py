#!/usr/bin/env python3
"""Static author tests for the source-only M1614 C1 hold package."""

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
TCL = HW / "dc_handoff/scripts/run_dc_m1614_m993_c1_hold_only_incremental_candidate.tcl"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1614_m993_c1_hold_only_incremental_exact_sha_r1.sh"
CONTRACT = HW / "contracts/m1614_m993_c1_hold_only_incremental_dc_source_contract_r1_20260901.json"
M993 = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
ORIGINAL = M993 / "original_quarantine"
M1006 = HW / "reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829"
M1612 = HW / "reviews/m1612_m993_c1_hold_closure_first_principles_readonly_review_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    TCL: "82cc53ac5f07162143a9ca99170daff9d64f03da3843abe7e0b4d830d24c9659",
    RUNNER: "c21fed97d28ec06b898548c4b406eeab1e1880f9f59d813c61bc8619357119dc",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.ddc":
        "d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc":
        "cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v":
        "9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.svf":
        "8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7",
    M1612 / "review.json":
        "7baba71a21be61842be8c76bddfa40abf8d2c0b0736e06aa44a80d53556cef72",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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


def command_text(text, comment):
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
    outer_rows = outer.read_text(encoding="ascii").split()
    if outer_rows != [sha256(manifest), "SHA256SUMS"]:
        raise AssertionError("outer seal drift " + str(root))
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        base_path = Path(base)
        dirs[:] = [name for name in dirs if not (base_path / name).is_symlink()]
        for name in files:
            path = base_path / name
            rel = path.relative_to(root).as_posix()
            # A parent seal deliberately does not absorb a nested tree's two
            # seal metadata files.  That nested tree is verified separately
            # by this test, so exclude seal metadata at every depth here.
            if path.name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                actual.add(rel)
            else:
                raise AssertionError("nonregular tree member " + rel)
    if actual != set(expected):
        raise AssertionError((set(expected) - actual, actual - set(expected)))
    for name, digest in expected.items():
        if sha256(root / name) != digest:
            raise AssertionError("tree member drift " + name)


class M1614SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tcl = TCL.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.tcl_commands = command_text(cls.tcl, "#")
        cls.runner_commands = command_text(cls.runner, "#")

    def test_01_frozen_identities(self):
        for path, digest in EXPECTED.items():
            self.assertEqual(sha256(path), digest, str(path))
            self.assertTrue(stat.S_ISREG(path.lstat().st_mode))
            self.assertFalse(path.is_symlink())

    def test_02_predecessor_and_review_trees_sealed(self):
        for root in (M993, ORIGINAL, M1006, M1612):
            verify_tree(root)

    def test_03_shell_is_static_syntax_clean(self):
        completed = subprocess.run(
            ["/usr/bin/bash", "-n", str(RUNNER)], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=10, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_04_exactly_one_hold_only_optimizer(self):
        self.assertEqual(len(re.findall(
            r"(?m)^\s*set_fix_hold\s+\[get_clocks core_clk\]\s*$",
            self.tcl_commands)), 1)
        self.assertEqual(len(re.findall(
            r"(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$",
            self.tcl_commands)), 1)
        self.assertEqual(len(re.findall(r"(?m)^\s*compile\b",
                                        self.tcl_commands)), 1)
        self.assertIsNone(re.search(r"(?m)^\s*compile_ultra\b",
                                    self.tcl_commands))

    def test_05_no_timing_concealment_or_constraint_rewrite(self):
        input_sdc = (ORIGINAL /
            "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc").read_text(encoding="utf-8")
        joined = self.tcl_commands + "\n" + command_text(input_sdc, "#")
        for token in FORBIDDEN:
            self.assertIsNone(re.search(r"(?m)^\s*" + token + r"\b", joined),
                              token)
        self.assertNotIn("set_clock_period", self.tcl_commands)
        self.assertNotIn("set_clock_uncertainty", self.tcl_commands)
        self.assertEqual(len(re.findall(r"(?m)^\s*read_sdc\b",
                                        self.tcl_commands)), 1)
        self.assertRegex(input_sdc,
                         r"create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)")
        self.assertRegex(input_sdc,
                         r"set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b")
        self.assertRegex(input_sdc,
                         r"set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b")

    def test_06_macro_and_library_boundary(self):
        for token in (
            "set_min_library $std_slow_db -min_version $std_fast_db",
            "set_min_library $macro_slow_db -min_version $macro_fast_db",
            "set expected_macro_count 9",
            "set_dont_touch $macro_cells_pre true",
        ):
            self.assertEqual(self.tcl.count(token), 1, token)
        for phase in ("pre", "post"):
            self.assertEqual(self.tcl.count(
                "if {$macro_count_" + phase + " != $expected_macro_count}"), 1)
            self.assertEqual(self.tcl.count(
                "puts $audit_fp \"macro_count_" + phase
                + "=$macro_count_" + phase + "\""), 1)
        self.assertIn("macro_count_pre=9", self.runner)
        self.assertIn("macro_count_post=9", self.runner)

    def test_07_pre_and_post_reports_and_outputs_are_complete(self):
        for token in (
            "setup_prehold_summary_machine.txt",
            "hold_prehold_summary_machine.txt",
            "setup_posthold_summary_machine.txt",
            "hold_posthold_summary_machine.txt",
            "timing_setup_prehold_top100.rpt",
            "timing_hold_prehold_top100.rpt",
            "timing_setup_posthold_top100.rpt",
            "timing_hold_posthold_top100.rpt",
            "constraint_design_rules_posthold.rpt",
            "m1614_hold_repaired_mapped.v",
            "m1614_hold_repaired_mapped.sdc",
            "m1614_hold_repaired.ddc",
            "m1614_hold_repaired.svf",
        ):
            self.assertIn(token, self.tcl)
            self.assertIn(token, self.runner)

    def test_08_success_gate_is_fail_closed(self):
        for token in (
            "post_setup['status']=='MET'",
            "post_hold['status']=='MET'",
            "ceiling=154608.7116945",
            "macro_ok",
            "drc_count==0",
            "positive=timing_ok and area_ok and macro_ok and drc_count==0",
            "SEALED_NEGATIVE_M1614_C1_HOLD_OR_AREA_GATE_FAILED__NO_RETRY",
        ):
            self.assertIn(token, self.runner)

    def test_09_attempt_is_consumed_before_tool_and_never_retried(self):
        attempt = self.runner.index('mkdir -- "${ATTEMPT}"')
        tool = self.runner.index('"${DC_SHELL}" -f "${TCL}"')
        release = self.runner.index('verify_dir_seal "${HAMMER_DIR}"')
        self.assertLess(release, attempt)
        self.assertLess(attempt, tool)
        self.assertGreaterEqual(self.runner.count("retry=false"), 4)
        self.assertIn("result identity already consumed or colliding", self.runner)
        self.assertNotIn("rm -rf", self.runner)

    def test_10_no_other_tool_execution(self):
        for token in ("pt_shell", "fm_shell", "vcs ", "simv ", "ptpx"):
            self.assertNotIn(token, self.runner_commands.lower())
        self.assertIn("'formality':False,'pt':False", self.runner)

    def test_11_contract_matches_sources_and_no_current_authority(self):
        c = self.contract
        manifest = Path(str(CONTRACT) + ".sha256")
        outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
        self.assertEqual(manifest.read_text(encoding="ascii").split(),
                         [sha256(CONTRACT), CONTRACT.name])
        self.assertEqual(outer.read_text(encoding="ascii").split(),
                         [sha256(manifest), manifest.name])
        self.assertEqual(c["status"],
            "SOURCE_ONLY_M1614_C1_HOLD_PACKAGE__NO_EDA_AUTHORIZED")
        self.assertEqual(c["identity"]["tcl_sha256"], EXPECTED[TCL])
        self.assertEqual(c["identity"]["runner_sha256"], EXPECTED[RUNNER])
        self.assertEqual(c["authorization"]["dc_runs_now"], 0)
        self.assertEqual(c["authorization"]["future_dc_runs_max"], 1)
        self.assertEqual(c["authorization"]["all_other_eda_runs"], 0)
        self.assertFalse(c["claim_boundary"]["hold_closed"])

    def test_12_future_different_author_gate_is_explicit(self):
        self.assertIn("m1615_m1614_c1_hold_only_incremental_dc_source_hammer",
                      self.runner)
        self.assertIn("m1616_m1615_m1614_c1_hold_only_incremental_dc_launch_release",
                      self.runner)
        self.assertEqual(self.contract["future_release_chain"]["source_hammer_status"],
            "PASS_M1615_M1614_C1_HOLD_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_DC_ATTEMPT")
        self.assertFalse(self.contract["future_release_chain"]["present_at_source_authoring"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
