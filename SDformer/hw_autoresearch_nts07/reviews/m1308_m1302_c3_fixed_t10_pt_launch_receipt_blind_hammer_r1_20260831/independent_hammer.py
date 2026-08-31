#!/usr/bin/python3.12
"""Receipt-blind static/mock hammer for the M1302 C3 PT launch wrapper."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


HW = Path(__file__).resolve().parents[3] / "hw_autoresearch_nts07"
WRAPPER = HW / "dc_handoff/scripts/run_m1302_m1288_fixed_t10_ptsta_exact_closed_one_shot.sh"
AUTHOR_TEST = HW / "tests/test_m1302_m1288_fixed_t10_ptsta_launch_source_static.py"
ADMISSION = HW / "contracts/m1289_m1288_c3_m917_fixed_t10_ptsta_launch_admission_r1_20260830.json"
CONTRACT = HW / "contracts/m1302_m1288_c3_fixed_t10_ptsta_launch_wrapper_source_contract_r1_20260830.json"
M1288_RUNNER = HW / "dc_handoff/scripts/run_m1288_m917_fixed_t10_ptsta_inert_exact_sha.sh"
M1288_CONTRACT = HW / "contracts/m1288_c3_m917_fixed_t10_ptsta_source_contract_r1_20260830.json"
M1299 = HW / "reviews/m1299_m1288_c3_m917_fixed_t10_ptsta_receipt_blind_hammer_r1_20260830"
M917 = HW / "dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829"
M928 = HW / "reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829"
M1285 = HW / "reviews/m1285_c3_m917_m928_pt_hold_saif_ptpx_readonly_audit_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

WRAPPER_SHA = "3f24a7d38df4e5c9df6b5316cc747b272fc4161d09b9a1580ea07f9998f18446"
AUTHOR_TEST_SHA = "db6d7de633e45629ffcc3308612b45669c0be9cada6f903578ffb60b06650e08"
ADMISSION_SHA = "1ea53ea55a8cc2bbc992aa932f73e7865561f7dde16e53f5d74efe3a7b146e3e"
CONTRACT_SHA = "21294ec80d8447a128c14201247d768b48cb3c8833d8752bd8e3da91479e6b92"
M1288_RUNNER_SHA = "a7fa2c5b031a446562d0bdb8f6f80112d7348fff6be92efdbf5b12830f6b928c"
M1288_CONTRACT_SHA = "91f130a09aa48b0f0f49aadb43c17d969abc026939199ee9acabccbb5a5a69a1"
M1299_OUTER_SHA = "16d5270894f13e98840124e06dea2fc075b93fc154911003589a86bfe15d71f0"
M917_OUTER_SHA = "e2f619c321218d78537528bb53d6de7b8817316008840198703103ff4c8c75b9"
M928_OUTER_SHA = "43e6cee08ed52c52d1e46d48afc8b6835fd735e74ce4320b671cd401cf9c17d3"
M1285_OUTER_SHA = "6c5fbaf805910022e9aecd25adb146b2b2ffaef92c2ee1ed3af885be45a54f7f"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(obj, keys):
    if type(obj) is not dict or set(obj) != set(keys):
        raise AssertionError("exact keyset mismatch")


def payload_seal(path: Path):
    digest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    if digest.read_text().split() != [sha(path), path.name]:
        raise AssertionError("payload digest seal")
    if outer.read_text().split() != [sha(digest), digest.name]:
        raise AssertionError("payload outer seal")


def dir_seal(path: Path):
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise AssertionError("directory outer seal")
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        member = path / name.lstrip("*")
        if not member.is_file() or member.is_symlink() or sha(member) != digest:
            raise AssertionError("directory member seal")


SOURCE = WRAPPER.read_text()


def heredoc_after(marker: str) -> str:
    if marker not in SOURCE:
        raise AssertionError("missing heredoc marker")
    return SOURCE.split(marker, 1)[1].split("\nPY\n", 1)[0]


ADMISSION_PARSER = heredoc_after(
    'python3 - "${m1302_admission}" "${M1302_EXPECTED_WRAPPER_SHA256}" <<\'PY\'\n')
ADJUDICATOR = heredoc_after(
    'python3 - "${m1302_m1288_canonical}" "${m1302_work}" <<\'PY\'\n')


def run_admission_parser(value, wrapper_sha=WRAPPER_SHA):
    with tempfile.TemporaryDirectory(prefix="m1308_admission.") as td:
        path = Path(td) / "admission.json"
        path.write_text(json.dumps(value))
        return subprocess.run([sys.executable, "-", str(path), wrapper_sha],
                              input=ADMISSION_PARSER, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def coverage_row(name, total=10, met=10, violated=0, untested=0):
    return "%s %d %d (100.00%%) %d (0.00%%) %d (0.00%%)\n" % (
        name, total, met, violated, untested)


def run_result_mock(setup=0.1, hold=0.0, unconstrained=0, untested=0,
                    constraint_violated=0, missing=None, m1288_status="fixture"):
    with tempfile.TemporaryDirectory(prefix="m1308_result.") as td:
        root = Path(td); src = root / "m1288"; out = root / "m1302"
        reports = src / "reports"; reports.mkdir(parents=True); out.mkdir()
        setup_state = "MET" if setup >= 0 else "VIOLATED"
        hold_state = "MET" if hold >= 0 else "VIOLATED"
        files = {
            "timing_setup_slow.rpt": "slack (%s) %.6f\n" % (setup_state, setup),
            "timing_hold_fast.rpt": "slack (%s) %.6f\n" % (hold_state, hold),
            "analysis_coverage.rpt": "".join(
                coverage_row(n, untested=untested)
                for n in ("setup", "hold", "out_setup", "out_hold")),
            "check_timing.rpt": (("Warning: There are %d endpoints which will be unconstrained.\n" % unconstrained)
                                  if unconstrained else "No unconstrained diagnostics.\n"),
            "constraint_violators.rpt": ("slack (VIOLATED) -0.1\n" * constraint_violated
                                          if constraint_violated else "No violators.\n"),
        }
        for name, payload in files.items():
            if name != missing:
                (reports / name).write_text(payload)
        (src / "m1288_m917_fixed_t10_prelayout_ptsta_receipt_r1.json").write_text(
            json.dumps({"status": m1288_status}))
        completed = subprocess.run([sys.executable, "-", str(src), str(out)],
                                   input=ADJUDICATOR, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            return completed, None, None
        receipt = json.loads((out / "m1302_adjudication_receipt_r1.json").read_text())
        return completed, receipt, (out / "GATE_EXIT_CODE.txt").read_text().strip()


def parse_license(text: str):
    match = re.search(
        r'Users of PrimeTime:.*?Total of\s+(\d+)\s+licenses? issued;\s+Total of\s+(\d+)\s+licenses? in use',
        text, re.S)
    if not match:
        raise ValueError("unparseable")
    return int(match.group(1)), int(match.group(2))


class Hammer(unittest.TestCase):
    def test_01_wrapper_admission_contract_and_docs_exact_bytes(self):
        exact_files = ((WRAPPER, WRAPPER_SHA), (AUTHOR_TEST, AUTHOR_TEST_SHA),
                       (ADMISSION, ADMISSION_SHA), (CONTRACT, CONTRACT_SHA),
                       (M1288_RUNNER, M1288_RUNNER_SHA),
                       (M1288_CONTRACT, M1288_CONTRACT_SHA),
                       (DOCS359, DOCS359_SHA))
        for path, wanted in exact_files:
            with self.subTest(path=path.name):
                self.assertEqual(sha(path), wanted)
                self.assertFalse(path.is_symlink())
        subprocess.run(["bash", "-n", str(WRAPPER)], check=True)

    def test_02_payload_and_dependency_double_seals(self):
        payload_seal(ADMISSION); payload_seal(CONTRACT); payload_seal(M1288_CONTRACT)
        for path, outer in ((M1299, M1299_OUTER_SHA), (M917, M917_OUTER_SHA),
                            (M928, M928_OUTER_SHA), (M1285, M1285_OUTER_SHA)):
            with self.subTest(path=path.name):
                self.assertEqual(sha(path / "SHA256SUMS.seal.sha256"), outer)
                dir_seal(path)

    def test_03_admission_and_contract_exact_closed_keysets(self):
        admission = json.loads(ADMISSION.read_text()); contract = json.loads(CONTRACT.read_text())
        exact(admission, ("schema", "date", "milestone", "status", "objective",
                          "identity", "exact_files", "tool", "preflight",
                          "authorization", "result_adjudication", "claim_boundary"))
        exact(contract, ("schema", "date", "status", "objective", "identity",
                         "frozen_inputs", "preflight", "result_gate",
                         "authorization", "claim_boundary"))
        self.assertEqual(run_admission_parser(admission).returncode, 0)
        self.assertEqual(contract["authorization"], {
            "launch_now": False, "max_attempts_now": 0, "run_pt_now": False,
            "run_dc_now": False, "run_vcs_now": False,
            "run_formality_now": False, "run_ptpx_now": False,
            "run_remote_now": False, "query_license_now": False,
            "independent_receipt_blind_hammer_required": True})

    def test_04_path_sha_keyset_and_claim_promotion_attacks_fail(self):
        base = json.loads(ADMISSION.read_text()); attacks = []
        bad = json.loads(json.dumps(base)); bad["extra"] = 1; attacks.append(bad)
        bad = json.loads(json.dumps(base)); bad["identity"]["wrapper_path"] = "/tmp/x"; attacks.append(bad)
        bad = json.loads(json.dumps(base)); bad["identity"]["wrapper_sha256"] = "0" * 64; attacks.append(bad)
        bad = json.loads(json.dumps(base)); bad["authorization"]["max_attempts"] = True; attacks.append(bad)
        bad = json.loads(json.dumps(base)); bad["authorization"]["run_ptpx"] = True; attacks.append(bad)
        bad = json.loads(json.dumps(base)); bad["claim_boundary"]["pt_executed"] = True; attacks.append(bad)
        bad = json.loads(json.dumps(base)); bad["claim_boundary"]["power"] = 0; attacks.append(bad)
        bad = json.loads(json.dumps(base)); bad["result_adjudication"]["hold_slack_min_ns"] = -0.1; attacks.append(bad)
        for index, value in enumerate(attacks):
            with self.subTest(index=index):
                self.assertNotEqual(run_admission_parser(value).returncode, 0)
        self.assertNotEqual(run_admission_parser(base, "0" * 64).returncode, 0)

    def test_05_all_exact_files_and_tool_library_pins_match(self):
        admission = json.loads(ADMISSION.read_text())
        for relative, wanted in admission["exact_files"].items():
            path = HW / relative
            with self.subTest(relative=relative):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(sha(path), wanted)
        self.assertEqual(admission["identity"]["m1299_outer_seal_sha256"], M1299_OUTER_SHA)
        self.assertEqual(admission["tool"]["pt_shell_sha256"],
                         "afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef")
        self.assertEqual(admission["tool"]["slow_db_sha256"],
                         "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af")
        self.assertEqual(admission["tool"]["fast_db_sha256"],
                         "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a")

    def test_06_same_uid_resource_license_repeat_freshness_attempt_order(self):
        order = [
            SOURCE.index('m1302_sealed_payload_ok "${m1302_admission}"'),
            SOURCE.index('[[ -z "$(m1302_collisions)" ]]'),
            SOURCE.index('m1302_mem_available="$(awk'),
            SOURCE.index('"${m1302_lmutil}" lmstat'),
            SOURCE.index('[[ -z "$(m1302_collisions)" ]]', SOURCE.index('"${m1302_lmutil}" lmstat')),
            SOURCE.index('mkdir "${m1302_attempt}"'),
            SOURCE.index('/usr/bin/bash "${m1302_m1288_runner}"'),
        ]
        self.assertEqual(order, sorted(order))
        self.assertIn('m1302_commit_headroom=$((m1302_commit_limit-m1302_committed))', SOURCE)
        self.assertIn('m1302_mem_available}" -ge 8388608', SOURCE)
        self.assertIn('m1302_commit_headroom}" -ge 8388608', SOURCE)
        self.assertIn('m1302_disk_available}" -ge 4194304', SOURCE)
        self.assertIn('m1302_issued}" -gt "${m1302_in_use}', SOURCE)

    def test_07_license_parser_requires_issued_strictly_greater_than_in_use(self):
        issued, used = parse_license(
            "Users of PrimeTime: Total of 4 licenses issued; Total of 3 licenses in use")
        self.assertGreater(issued, used)
        issued, used = parse_license(
            "Users of PrimeTime: Total of 4 licenses issued; Total of 4 licenses in use")
        self.assertFalse(issued > used)
        with self.assertRaises(ValueError):
            parse_license("PrimeTime license unknown")
        admission = json.loads(ADMISSION.read_text())
        self.assertEqual(admission["preflight"]["license_gate"], {
            "feature": "PrimeTime", "server": "27030@ic.ismd-nemo",
            "query_before_attempt": True, "issued_gt_in_use_required": True})

    def test_08_repeat_collision_and_atomic_attempt_reuse_mock(self):
        self.assertGreaterEqual(SOURCE.count('[[ -z "$(m1302_collisions)" ]]'), 2)
        with tempfile.TemporaryDirectory(prefix="m1308_attempt.") as td:
            attempt = Path(td) / "attempt"
            attempt.mkdir()
            with self.assertRaises(FileExistsError):
                attempt.mkdir()
        for name in ("m1288_canonical", "m1288_work", "m1288_attempt",
                     "m1302_canonical", "m1302_work", "m1302_attempt"):
            self.assertIn('! -e "${m1302_' + name.replace("m1302_", "") + '}"' if name.startswith("m1302_") else '! -e "${m1302_' + name + '}"', SOURCE)

    def test_09_no_eco_hold_fix_power_or_direct_pt_launch(self):
        lowered = SOURCE.lower()
        for forbidden in ("fix_eco_timing", "set_fix_hold", "report_power",
                          "update_power", "read_saif", "write_sdf"):
            self.assertNotIn(forbidden, lowered)
        self.assertEqual(SOURCE.count('"${m1302_pt}"'), 1)
        tcl = (HW / "dc_handoff/scripts/run_ptsta_m1288_m917_fixed_t10_slowmax_fastmin_inert.tcl").read_text().lower()
        for forbidden in ("fix_eco_timing", "set_fix_hold", "report_power",
                          "update_power", "read_saif"):
            self.assertNotIn(forbidden, tcl)

    def test_10_result_pass_mock_is_exact_and_nonpromoting(self):
        completed, receipt, rc = run_result_mock()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(rc, "0")
        self.assertIs(receipt["strict_timing_gate_pass"], True)
        self.assertEqual(receipt["status"],
                         "PASS_M1302_M1288_FIXED_T10_PRELAYOUT_PTSTA_STRICT_TIMING_GATE")
        self.assertEqual(receipt["claim_boundary"], {
            "fresh_result_hammer_required": True, "power": False,
            "energy": False, "speedup": False, "system": False,
            "paper_ppa_ready": False, "headline": False})
        self.assertIs(receipt["scope"]["mapped_identity_mutated"], False)

    def test_11_negative_hold_unconstrained_coverage_and_constraint_attacks_stop(self):
        cases = (
            {"hold": -0.01}, {"setup": -0.01}, {"unconstrained": 2},
            {"untested": 1}, {"constraint_violated": 1})
        for args in cases:
            with self.subTest(args=args):
                completed, receipt, rc = run_result_mock(**args)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(rc, "10")
                self.assertIs(receipt["strict_timing_gate_pass"], False)
                self.assertEqual(receipt["status"],
                                 "STOP_M1302_M1288_FIXED_T10_PRELAYOUT_PTSTA_STRICT_TIMING_GATE")

    def test_12_incomplete_or_malformed_result_cannot_pass(self):
        completed, receipt, rc = run_result_mock(missing="analysis_coverage.rpt")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIsNone(receipt); self.assertIsNone(rc)
        completed, receipt, rc = run_result_mock(m1288_status="FAKE_PASS")
        self.assertEqual(completed.returncode, 0)
        self.assertIs(receipt["strict_timing_gate_pass"], True)
        self.assertEqual(receipt["m1288_status"], "FAKE_PASS")
        self.assertTrue(receipt["claim_boundary"]["fresh_result_hammer_required"])

    def test_13_failure_quarantine_and_result_sealing_are_present(self):
        self.assertIn('m1302_seal_dir "${m1302_work}"', SOURCE)
        self.assertIn('"${m1302_canonical}.failed_or_incomplete.$$.quarantine"', SOURCE)
        self.assertIn('m1302_double_seal_ok "${m1302_m1288_canonical}"', SOURCE)
        self.assertIn('m1302_seal_dir "${m1302_attempt}"', SOURCE)
        self.assertIn('mv -T "${m1302_work}" "${m1302_canonical}"', SOURCE)

    def test_14_future_namespaces_are_fresh_and_no_action_was_run(self):
        paths = (
            HW / "dc_handoff/runs/m1288_m917_fixed_t10_prelayout_ptsta_r1_20260830",
            HW / "dc_handoff/runs/m1288_m917_fixed_t10_prelayout_ptsta_r1_20260830.work",
            HW / "dc_handoff/runs/.m1288_m917_fixed_t10_ptsta_attempt_consumed",
            HW / "dc_handoff/runs/m1302_m1288_fixed_t10_ptsta_adjudication_r1_20260830",
            HW / "dc_handoff/runs/m1302_m1288_fixed_t10_ptsta_adjudication_r1_20260830.work",
            HW / "dc_handoff/runs/.m1302_m1288_fixed_t10_ptsta_attempt_consumed",
        )
        for path in paths:
            self.assertFalse(path.exists(), str(path))
        self.assertEqual(sha(DOCS359), DOCS359_SHA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
