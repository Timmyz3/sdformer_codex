#!/usr/bin/env python3
"""Static and mutation tests for source-only M1695 C1 hold closure."""
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
TCL = HW / "dc_handoff/scripts/run_dc_m1695_m1665_c1_fastmin_hold_closure_candidate.tcl"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1695_m1665_c1_fastmin_hold_closure_exact_one_shot.sh"
CONTRACT = HW / "contracts/m1695_m1665_c1_fastmin_hold_closure_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1695_m1665_c1_fastmin_hold_closure_source_author_receipt_r1_20260901"
M1665 = HW / "dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901"
ORIGINAL = M1665 / "original_quarantine"
M1678 = HW / "dc_handoff/runs/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_r1_20260901.failed_or_incomplete.1991841.quarantine"
RESULT = HW / "dc_handoff/runs/m1695_m1665_c1_fastmin_hold_closure_dc_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1695_m1665_c1_fastmin_hold_closure_dc_attempt_consumed"

EXPECTED = {
    TCL: "cb05b053078c7ab9d084cddf5028802aeff52ef1a4aef6d1b026ba6da2f41ad8",
    RUNNER: "f470eee1f4f68be76d4d680522efca4157472582e9f442721ef836bd5957ca5d",
    M1665 / "SHA256SUMS": "a16b9fb100bf7f1b3c6e7453035a5bf89a8f2ffbbeeca1d373038f6e899dba72",
    M1665 / "SHA256SUMS.seal.sha256": "12d87acb439b0cc171d3f42cd4f169fa6a531946c9c3c120cc9babc9c36fbc08",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc":
        "2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc":
        "5ab21dbeb46baabf6e0bec2ea2a8f8542e114308e77ded25486fa022e4c3e198",
    M1678 / "SHA256SUMS": "9556e3bfab30af74326473f6cb9e492d41d3b782d0f23fabb6564626ce6fc675",
    M1678 / "SHA256SUMS.seal.sha256": "7b90352dd62288415f12903cbc4c2745cf2f2fa574080b37f63871015bc77602",
    M1678 / "rtl_to_m993/FORMALITY_INTERNAL_COMPLETE.txt":
        "9eee52aa958d835e9b682d99e5b52cfed515bacee74854fb8f0a4a8ddfab7eb9",
    M1678 / "m993_to_m1665/FORMALITY_INTERNAL_COMPLETE.txt":
        "b27aeb9e49081c6fbc238a082dfe7c364270e25ca11579e7ee73c717d0a12fd8",
    M1678 / "ptsta/reports/global_timing.rpt":
        "c323bdd22a6f9137ee02f85aba0ed9c7792cf1febd6d8c3b11fb2650d41f7557",
    M1678 / "ptsta/reports/timing_setup_slow.rpt":
        "c0dc0bce139cdf1f8be3058c43bc40ed5b67fa8c2c82292b7265f0f232f35495",
    M1678 / "ptsta/reports/timing_hold_fast.rpt":
        "eeacd609124059018fdc1bbdafd460342adcc524473d0769c4d43daa43aa3445",
    M1678 / "ptsta/reports/constraint_violators.rpt":
        "d974d269d592fe02ea04db0c062c8061bba1f8d6e67fd479bb929a1da97526eb",
}

FORBIDDEN = ("set_false_path", "set_multicycle_path", "set_min_delay",
             "set_max_delay", "set_disable_timing", "set_case_analysis")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def commands(text):
    return "\n".join(row.split("#", 1)[0] for row in text.splitlines())


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


def verify_manifest_members(root):
    """Verify every sealed member; permit excluded tool scratch in quarantine."""
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if outer.read_text(encoding="ascii").split() != [sha256(manifest),
                                                       "SHA256SUMS"]:
        raise AssertionError("outer seal drift " + str(root))
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise AssertionError("unsafe manifest row " + name)
        path = root / rel
        if not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink():
            raise AssertionError("nonregular sealed member " + str(path))
        if sha256(path) != digest:
            raise AssertionError("sealed member drift " + name)


def validate_tcl(text):
    cmd = commands(text)
    checks = [
        (len(re.findall(r"(?m)^\s*read_ddc\b", cmd)) == 1, "read_ddc"),
        (len(re.findall(r"(?m)^\s*set_fix_hold\s+\$core_clock\s*$", cmd)) == 1,
         "set_fix_hold"),
        (len(re.findall(r"(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$", cmd)) == 1,
         "hold compile"),
        (len(re.findall(r"(?m)^\s*compile\b", cmd)) == 1, "all compile"),
        (text.count("set optimization_hold_uncertainty_ns 0.081") == 1,
         "optimization uncertainty"),
        (text.count("set reported_hold_uncertainty_ns 0.050") == 1,
         "reported uncertainty"),
        (text.count("set_min_library $std_slow_db -min_version $std_fast_db") == 1,
         "std fast-min"),
        (text.count("set_min_library $macro_slow_db -min_version $macro_fast_db") == 1,
         "macro fast-min"),
    ]
    for ok, name in checks:
        if not ok:
            raise ValueError(name)
    guard = text.index("set_clock_uncertainty -hold $optimization_hold_uncertainty_ns $core_clock")
    fix = text.index("set_fix_hold $core_clock")
    compile_at = text.index("compile -incremental_mapping -only_hold_time")
    restore = text.index("set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock")
    report = text.index('report_qor > "$output_dir/reports/qor_posthold.rpt"')
    if not guard < fix < compile_at < restore < report:
        raise ValueError("operation order")


class M1695SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tcl = TCL.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_01_exact_sources_and_evidence(self):
        for path, digest in EXPECTED.items():
            self.assertEqual(sha256(path), digest, str(path))

    def test_02_frozen_trees_are_double_sealed(self):
        for root in (M1665, ORIGINAL):
            verify_tree(root)
        verify_manifest_members(M1678)

    def test_03_shell_syntax_only(self):
        run = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                             cwd=str(ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True,
                             timeout=10, check=False)
        self.assertEqual(run.returncode, 0, run.stdout)

    def test_04_m1665_ddc_is_only_design_input(self):
        self.assertEqual(len(re.findall(r"(?m)^\s*read_ddc\b",
                                        commands(self.tcl))), 1)
        self.assertIn('INPUT_DDC="${M1665_ORIGINAL}/netlist/', self.runner)
        self.assertIn("input_generation=frozen_m1665_ddc_only", self.tcl)
        self.assertNotIn("read_verilog", commands(self.tcl))
        self.assertNotRegex(self.runner, r"(?m)^INPUT_DDC=.*m1678")

    def test_05_fastmin_guard_and_restore_contract(self):
        validate_tcl(self.tcl)
        self.assertIn("0.050 + 0.030 correction + 0.001 guard", self.tcl)
        self.assertIn("macro_hold_check_delta_ns=0.029174", self.tcl)

    def test_06_mutation_guard_rejects_wrong_uncertainty(self):
        with self.assertRaises(ValueError):
            validate_tcl(self.tcl.replace(
                "set optimization_hold_uncertainty_ns 0.081",
                "set optimization_hold_uncertainty_ns 0.080", 1))
        with self.assertRaises(ValueError):
            validate_tcl(self.tcl.replace(
                "set reported_hold_uncertainty_ns 0.050",
                "set reported_hold_uncertainty_ns 0.081", 1))

    def test_07_mutation_guard_rejects_extra_compile_or_missing_macro_min(self):
        with self.assertRaises(ValueError):
            validate_tcl(self.tcl.replace(
                "compile -incremental_mapping -only_hold_time",
                "compile -incremental_mapping -only_hold_time\ncompile -incremental_mapping",
                1))
        with self.assertRaises(ValueError):
            validate_tcl(self.tcl.replace(
                "set_min_library $macro_slow_db -min_version $macro_fast_db",
                "# removed macro min library", 1))

    def test_08_no_timing_concealment(self):
        sdc = (ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc").read_text(encoding="utf-8")
        joined = commands(self.tcl) + "\n" + commands(sdc)
        for token in FORBIDDEN:
            self.assertIsNone(re.search(r"(?m)^\s*" + token + r"\b", joined), token)
        self.assertRegex(sdc, r"create_clock[^\n]*-period\s+3(?:\.0+)?(?:\s|$)")
        self.assertRegex(sdc, r"set_clock_uncertainty\s+-setup\s+0?\.2(?:0+)?\b")
        self.assertRegex(sdc, r"set_clock_uncertainty\s+-hold\s+0?\.05(?:0+)?\b")

    def test_09_nine_macros_and_ten_percent_area_gate(self):
        for token in ("set expected_macro_count 9",
                      "set area_baseline_um2 152898.625984",
                      "set area_ceiling_um2 168188.4885824",
                      "set_dont_touch $macro_cells_pre true"):
            self.assertEqual(self.tcl.count(token), 1, token)

    def test_10_m1678_failure_and_two_formalities_bound(self):
        rtl = (M1678 / "rtl_to_m993/FORMALITY_INTERNAL_COMPLETE.txt").read_text()
        gate = (M1678 / "m993_to_m1665/FORMALITY_INTERNAL_COMPLETE.txt").read_text()
        glob = (M1678 / "ptsta/reports/global_timing.rpt").read_text(errors="replace")
        setup = (M1678 / "ptsta/reports/timing_setup_slow.rpt").read_text(errors="replace")
        hold = (M1678 / "ptsta/reports/timing_hold_fast.rpt").read_text(errors="replace")
        self.assertIn("INTERNAL_COMPLETE=PASS", rtl)
        self.assertIn("INTERNAL_COMPLETE=PASS", gate)
        self.assertRegex(glob, r"TNS\s+-40\.24\s+-40\.24")
        self.assertRegex(glob, r"NUM\s+10610\s+10610")
        self.assertRegex(setup, r"slack \(MET\)\s+0\.002221")
        self.assertRegex(hold, r"slack \(VIOLATED\)\s+-0\.028168")
        self.assertRegex(hold, r"library hold time\s+0\.12685[89]")

    def test_11_one_dc_process_no_other_eda_and_24gib_gate(self):
        cmd = commands(self.runner)
        self.assertEqual(cmd.count('"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"'), 1)
        self.assertIn('"${headroom}" -ge 25165824', self.runner)
        self.assertIn("max_dc_runs=1", self.runner)
        self.assertIn("retry=false", self.runner)
        for invocation in ('"${FM_SHELL}"', '"${PT_SHELL}"', "vcs -full64",
                           "simv", "ptpx"):
            self.assertNotIn(invocation, cmd)

    def test_11b_shared_queue_lock_and_two_collision_rechecks(self):
        shared = 'SHARED_QUEUE="/tmp/date_dual_synopsys_same_uid_eda_queue.lock"'
        acquire = '"${FLOCK}" -x 9'
        post_lock = 'fail "same-UID DC collision after shared lock"'
        prelaunch = 'fail "same-UID DC collision immediately before launch"'
        launch = '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"'
        for token in (shared, acquire, post_lock, prelaunch):
            self.assertEqual(self.runner.count(token), 1, token)
        self.assertLess(self.runner.index(acquire), self.runner.index(post_lock))
        self.assertLess(self.runner.index(post_lock), self.runner.index(prelaunch))
        self.assertLess(self.runner.index(prelaunch), self.runner.index(launch))
        self.assertIn("ancestry=set(); pid=os.getpid()", self.runner)
        self.assertIn("exec 9>", self.runner)
        self.assertNotIn("flock -u", self.runner)

    def test_12_future_review_release_keep_source_inert(self):
        self.assertIn("m1696_m1695_c1_fastmin_hold_closure_source_hammer", self.runner)
        self.assertIn("m1697_m1696_m1695_c1_fastmin_hold_closure_launch_release", self.runner)
        self.assertIn("M1695_EXPECTED_DC_RUNNER_SHA256", self.runner)
        self.assertIn("M1695_EXPECTED_DC_RELEASE_SHA256", self.runner)
        self.assertFalse(RESULT.exists())
        self.assertFalse(ATTEMPT.exists())
        self.assertFalse((HW / "reviews/m1696_m1695_c1_fastmin_hold_closure_source_hammer_r1_20260901").exists())

    def test_13_contract_and_author_are_double_sealed(self):
        manifest = Path(str(CONTRACT) + ".sha256")
        outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
        self.assertEqual(manifest.read_text(encoding="ascii").split(),
                         [sha256(CONTRACT), CONTRACT.name])
        self.assertEqual(outer.read_text(encoding="ascii").split(),
                         [sha256(manifest), manifest.name])
        verify_tree(AUTHOR)

    def test_14_contract_binds_sources_and_opens_no_execution(self):
        c = self.contract
        self.assertEqual(c["status"],
            "SOURCE_ONLY_M1695_C1_FASTMIN_HOLD_CLOSURE__NO_EDA_AUTHORIZED")
        self.assertEqual(c["identity"]["tcl_sha256"], EXPECTED[TCL])
        self.assertEqual(c["identity"]["runner_sha256"], EXPECTED[RUNNER])
        self.assertEqual(c["identity"]["author_test_sha256"], sha256(Path(__file__)))
        self.assertEqual(c["authorization"]["dc_runs_now"], 0)
        self.assertEqual(c["authorization"]["future_dc_runs_max"], 1)
        self.assertFalse(c["claim_boundary"]["hold_closed"])
        self.assertFalse(c["claim_boundary"]["paper_citable"])

    def test_15_docs359_and_rtl_are_not_write_targets(self):
        self.assertEqual(self.runner.count('DOC359="${HW_ROOT}/docs/359_'), 1)
        self.assertEqual(self.runner.count('"${DOC359}"'), 1)
        self.assertNotRegex(self.runner, r'(?:>|>>|cp|mv)\s+[^\n]*\$\{DOC359\}')
        self.assertNotIn("rtl_m935", self.tcl)
        self.assertFalse(any("docs/359" in x for x in self.contract.get("outputs", [])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
