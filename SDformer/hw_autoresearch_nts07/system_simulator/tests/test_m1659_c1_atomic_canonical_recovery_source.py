#!/usr/bin/env python3
"""Payload-free author regression for the inert M1659 recovery source."""
from __future__ import print_function

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import unittest


TESTS = Path(__file__).resolve().parent
HW = TESTS.parent.parent
SOURCE = HW / "dc_handoff/scripts/promote_m1659_m1649_c1_quarantine_atomic_canonical_recovery_r1.sh"
CONTRACT = HW / "contracts/m1659_m1649_c1_atomic_canonical_recovery_source_contract_r1_20260901.json"
Q = HW / ("dc_handoff/runs/m1649_m1630_c1_resource_gate_successor_dc_"
          "r1_20260901.failed_or_incomplete.519344.quarantine")
M1655 = HW / "reviews/m1655_m1649_c1_quarantine_forensic_recovery_review_r1_20260901"
FUTURE_REVIEW = HW / "reviews/m1660_m1659_c1_canonical_recovery_source_independent_review_r1_20260901"
FUTURE_RELEASE = HW / "contracts/m1664_m1660_m1659_c1_canonical_recovery_release_r1_20260901.json"
NEW_NAMESPACES = (
    HW / "dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901",
    HW / "dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_launch_lock",
    HW / "dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_attempt_consumed",
    HW / "dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_work",
    HW / "dc_handoff/runs/m1665_m1659_c1_canonical_recovery_failed_or_incomplete.quarantine",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def kv_text(text):
    output = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in output:
                raise ValueError("duplicate key")
            output[key] = value
    return output


def log_gate(text):
    lines = text.splitlines()
    errors = [(index + 1, line) for index, line in enumerate(lines)
              if re.match(r"^(Error|Fatal):", line)]
    expected = [(32, "Error: Error during sourcing of /opt/synopsys/syn/"
                     "V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl")]
    if errors != expected:
        raise ValueError("error population")
    if 'no such variable\n    (read trace on "::env(HOME)")' not in text:
        raise ValueError("HOME signature")
    start = next(index for index, line in enumerate(lines)
                 if line.startswith("Current time:"))
    if start <= 31 or any(re.match(r"^(Error|Fatal):", line)
                          for line in lines[start + 1:]):
        raise ValueError("error phase")
    for marker in ("Writing verilog file '", "Writing ddc file '",
                   "set_svf -off", "Thank you..."):
        if marker not in text:
            raise ValueError("completion marker")
    return True


def timing_gate(text, kind, wns):
    expected = {"phase": "POST_RESTORE_REPORTED", "delay_type": kind,
                "status": "MET", "wns_ns": wns,
                "tns_ns": "0.000000000", "violating_paths": "0",
                "negative_path_ceiling": "200000"}
    if kv_text(text) != expected:
        raise ValueError("timing drift")
    return True


class M1659SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text(encoding="utf-8")
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_01_source_identity_contract_and_shell_syntax(self):
        self.assertEqual(sha(SOURCE),
            "cfd06bc58023869350668ab256311f97728e86db1f5d19d1933e2c9753960e73")
        self.assertEqual(self.contract["identity"]["source_sha256"], sha(SOURCE))
        completed = subprocess.run(["bash", "-n", str(SOURCE)],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_02_contract_and_m1655_double_seals(self):
        for path in (CONTRACT,):
            sidecar = Path(str(path) + ".sha256")
            outer = Path(str(path) + ".sha256.seal.sha256")
            self.assertEqual(sidecar.read_text(encoding="ascii"),
                             sha(path) + "  " + path.name + "\n")
            self.assertEqual(outer.read_text(encoding="ascii"),
                             sha(sidecar) + "  " + sidecar.name + "\n")
        self.assertEqual(sha(M1655 / "review.json"),
            "4d6f3e2cb238fbe77038cfc213d31ce061e17d49f43badcbc6b30ee8ffb825b2")
        self.assertEqual(sha(M1655 / "SHA256SUMS"),
            "349a78db9de8d138445889f1566ff1764a66ce3aa28d6599788979e20a8b2268")
        self.assertEqual(sha(M1655 / "SHA256SUMS.seal.sha256"),
            "5c3e1346ac3e4ecd9935190be6f8e4acf5fa9435941f2ed0a21c66512b9534f7")

    def test_03_exact_39_member_quarantine_topology_and_hashes(self):
        rows = {}
        for line in (Q / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertNotIn(name, rows); rows[name] = digest
        actual = set()
        for base, dirs, files in os.walk(str(Q), followlinks=False):
            for name in list(dirs) + list(files):
                path = Path(base) / name
                self.assertFalse(path.is_symlink())
                rel = path.relative_to(Q).as_posix()
                if path.is_file() and rel not in (
                        "SHA256SUMS", "SHA256SUMS.seal.sha256"):
                    self.assertTrue(stat.S_ISREG(path.lstat().st_mode))
                    actual.add(rel)
        self.assertEqual(len(rows), 39)
        self.assertEqual(set(rows), actual)
        for name, digest in rows.items():
            self.assertEqual(sha(Q / name), digest)
        self.assertEqual(sha(Q / "SHA256SUMS"),
            "e94ffc3680513cb2f374676037cc7c3b14b77a7bc47b9d35edb812f17a9ae843")
        self.assertEqual(sha(Q / "SHA256SUMS.seal.sha256"),
            "c221bb79e4950780c6db04ef54ed1ea809ac880ad054f9316f7bba702a49ff44")

    def test_04_dc_rc_tcl_completion_and_provenance(self):
        self.assertEqual((Q / "dc.rc").read_text(), "0\n")
        terminal = kv_text((Q / "TCL_INTERNAL_COMPLETE.txt").read_text())
        self.assertEqual(terminal["status"],
            "M1630_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED")
        self.assertEqual(terminal["input_generation"],
                         "original_m993_m1006_admitted_ddc")
        self.assertEqual(terminal["failed_m1614_output_used"], "false")
        self.assertEqual(terminal["set_fix_hold_count"], "1")
        self.assertEqual(terminal["hold_only_incremental_mapping_count"], "1")
        self.assertEqual(terminal["formality_required"], "true")
        self.assertEqual(terminal["independent_pt_required"], "true")

    def test_05_only_exact_preflow_home_gui_error_is_accepted(self):
        self.assertTrue(log_gate((Q / "dc.log").read_text(errors="replace")))

    def test_06_log_mutations_fail_closed(self):
        text = (Q / "dc.log").read_text(errors="replace")
        mutations = (text.replace("::env(HOME)", "::env(USER)", 1),
                     text + "\nFatal: injected\n",
                     text.replace("Thank you...", "", 1))
        for value in mutations:
            with self.assertRaises((ValueError, StopIteration)):
                log_gate(value)

    def test_07_setup_and_hold_exact_positive_gates(self):
        self.assertTrue(timing_gate(
            (Q / "reports/setup_posthold_summary_machine.txt").read_text(),
            "max", "0.002221110"))
        self.assertTrue(timing_gate(
            (Q / "reports/hold_posthold_summary_machine.txt").read_text(),
            "min", "0.000999451"))
        bad = (Q / "reports/hold_posthold_summary_machine.txt").read_text(
            ).replace("0.000999451", "-0.000000001")
        with self.assertRaises(ValueError):
            timing_gate(bad, "min", "0.000999451")

    def test_08_area_macro_and_drc_positive_gates(self):
        area = (Q / "reports/area_posthold.rpt").read_text(errors="replace")
        value = float(re.search(r"Total cell area:\s*([0-9.]+)", area).group(1))
        self.assertEqual(value, 152898.625984)
        self.assertLessEqual(value, 154608.7116945)
        self.assertAlmostEqual((value / 147246.392090 - 1) * 100,
                               3.8386230139650923)
        macro = kv_text((Q / "reports/macro_binding_audit.txt").read_text())
        self.assertEqual([macro[x] for x in (
            "macro_count_pre", "macro_count_post", "expected_macro_count")],
            ["9", "9", "9"])
        qor = (Q / "reports/qor_posthold.rpt").read_text(errors="replace")
        self.assertIsNotNone(re.search(
            r"Nets With Violations:\s+0(?:\.00)?\s*$", qor, re.MULTILINE))

    def test_09_netlist_ddc_sdc_svf_are_exact_and_nonempty(self):
        names = self.contract["artifact_identity"]
        mapping = {
          "ddc_sha256": "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc",
          "svf_sha256": "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.svf",
          "sdc_sha256": "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc",
          "mapped_verilog_sha256": "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.v"}
        for field, name in mapping.items():
            self.assertGreater((Q / name).stat().st_size, 0)
            self.assertEqual(sha(Q / name), names[field])

    def test_10_future_authority_precedes_all_mutations(self):
        positions = [self.text.index(token) for token in (
            'verify_dir_seal "${FUTURE_REVIEW_DIR}" 7',
            'verify_file_seal "${FUTURE_RELEASE}"',
            'M1659_EXPECTED_SOURCE_SHA256',
            'forensic_gate "${SOURCE}"',
            'mkdir -- "${LOCK}"',
            'mkdir -- "${ATTEMPT}"',
            'mkdir -- "${WORK}"',
            'cp -a --no-dereference',
            'forensic_gate "${WORK}/original_quarantine"',
            'mv -T -- "${WORK}" "${TARGET}"')]
        self.assertEqual(positions, sorted(positions))

    def test_11_atomic_one_shot_no_replace_and_no_retry(self):
        self.assertEqual(self.text.count('mkdir -- "${LOCK}"'), 1)
        self.assertEqual(self.text.count('mkdir -- "${ATTEMPT}"'), 1)
        self.assertEqual(self.text.count('cp -a --no-dereference'), 1)
        self.assertEqual(self.text.count('mv -T -- "${WORK}" "${TARGET}"'), 1)
        self.assertNotIn("rm -", self.text)
        self.assertIn("retry=false", self.text)
        self.assertIn('[[ ! -e "${TARGET}" ]]', self.text)

    def test_12_no_eda_command_and_claims_remain_pending(self):
        executable_lines = "\n".join(line for line in self.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#"))
        for pattern in (r"(?m)^\s*(?:\$\{[^}]+\}/)?dc_shell\b",
                        r"(?m)^\s*(?:\$\{[^}]+\}/)?fm_shell\b",
                        r"(?m)^\s*(?:\$\{[^}]+\}/)?pt_shell\b",
                        r"(?m)^\s*(?:\$\{[^}]+\}/)?vcs\b"):
            self.assertIsNone(re.search(pattern, executable_lines))
        boundary = self.contract["recovered_receipt_boundary"]
        self.assertTrue(boundary["dc_setup_hold_area_macro_drc_candidate"])
        for field in ("formality", "independent_prime_time", "power",
                      "energy", "cycle_speedup", "system_speedup",
                      "paper_ppa_ready", "paper_citable", "headline"):
            self.assertFalse(boundary[field])

    def test_13_future_and_runtime_namespaces_are_fresh(self):
        self.assertFalse(FUTURE_REVIEW.exists())
        self.assertFalse(FUTURE_RELEASE.exists())
        for path in NEW_NAMESPACES:
            self.assertFalse(path.exists(), str(path))

    def test_14_contract_requires_exact_copy_and_all_next_gates(self):
        copy = self.contract["copy_semantics"]
        self.assertFalse(copy["eda_rerun"])
        self.assertFalse(copy["source_quarantine_modified"])
        self.assertTrue(copy["source_attempt_preserved_as_provenance"])
        self.assertTrue(copy["original_failure_marker_preserved"])
        self.assertTrue(copy["forensic_gate_recomputed_before_and_after_copy"])
        self.assertTrue(copy["target_recursively_double_sealed"])
        self.assertEqual(self.contract["next_gates_after_future_recovery"], {
            "gate_to_gate_formality": True,
            "direct_or_transitive_rtl_formality": True,
            "independent_prime_time_max_min": True,
            "power": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
