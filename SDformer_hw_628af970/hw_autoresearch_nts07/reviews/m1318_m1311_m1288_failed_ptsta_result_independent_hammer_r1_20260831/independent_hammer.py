#!/usr/bin/python3.12
"""Read-only hammer for the single consumed M1311/M1288 PT failure result."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import unittest


HW = Path(__file__).resolve().parents[3] / "hw_autoresearch_nts07"
RUNS = HW / "dc_handoff/runs"
M1288 = RUNS / "m1288_m917_fixed_t10_prelayout_ptsta_r1_20260830.failed_or_incomplete.3811273.quarantine"
M1311 = RUNS / "m1311_m1288_fixed_t10_ptsta_adjudication_r1_20260831.failed_or_incomplete.3810957.quarantine"
M1288_ATTEMPT = RUNS / ".m1288_m917_fixed_t10_ptsta_attempt_consumed"
M1311_ATTEMPT = RUNS / ".m1311_m1288_fixed_t10_ptsta_attempt_consumed"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    M1288 / "SHA256SUMS": "0597c6c5a2718469417e99b9444a1d521488d1833642adba167d7cb1fbb5701e",
    M1288 / "SHA256SUMS.seal.sha256": "59d5acca7487c92f051e016db82fc9dbe94d69d3bc7ff222b4bfd3aed9fbb563",
    M1311 / "SHA256SUMS": "36f20e293808b59fe189d734ef97089a8f87d212ab6e626922d89a1aec22e396",
    M1311 / "SHA256SUMS.seal.sha256": "b6eec054b286e1f27e5524f1c271a17cb206157508c88b5f8032a58d52402df2",
    M1288 / "reports/timing_setup_slow.rpt": "4c6bdf8ed28d09bbdbe3ec4b62ab43e2083aecbc2ea3bd68c5e07a322f68d60f",
    M1288 / "reports/timing_hold_fast.rpt": "fd76172bfc68a4fd9d183a2bd2584fc6b67e7520e6b304a69df0466956c7e93f",
    M1288 / "reports/global_timing.rpt": "d38531434d50a59f4ff396bc84c639e50bd758988a6ca2ac8404271489f2fe5e",
    M1288 / "reports/analysis_coverage.rpt": "2662fa4dbdcb8d0b6ec43ab10445cd70935e5497e259d06e62f7e9ca4d463747",
    M1288 / "reports/check_timing.rpt": "7267a999d832f1653b71dd213c0bff91ca848d37d2f11d293eba1c47c8adb024",
    M1288 / "reports/constraint_violators.rpt": "be3e85783b7d1cc86f5a6ffce1f2f39dda1d69e3092bd59730c4726a7b61e73c",
    M1288 / "reports/runtime_scope.rpt": "6b6a63d1992feac7f57fb148322cf322a7a169a404181644fd38bed765c17333",
    M1311 / "m1311_failure_receipt_r1.json": "8b162096a7e56018796ecfee34206d32060c32254d91eb01b0be7b71c27c68b2",
    M1311 / "RUN_COMPLETE.txt": "7a91bbc14b022482ab180a69cdc45fab570091d8ba34c872181505d97456dc7f",
    M1288_ATTEMPT / "attempt.txt": "a05d4542a8804a2fa216cca668de9bf7198f2eacd960f982fd4540e80f42b0b7",
    M1311_ATTEMPT / "attempt.txt": "cb87e372216f3d2675e5d471f9a9c991ed7a0ff6db2416ad7bed6a063da081a0",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path: Path) -> None:
    if not stat.S_ISREG(os.lstat(str(path)).st_mode):
        raise AssertionError("not regular: " + str(path))


def verify_dir(path: Path) -> None:
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise AssertionError("outer seal")
    seen = set()
    for row in manifest.read_text().splitlines():
        digest, rel = row.split(None, 1); rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        if rel in seen or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise AssertionError("manifest path")
        seen.add(rel)
        member = path / rel; regular(member)
        if sha(member) != digest:
            raise AssertionError("member drift: " + rel)


def slacks(path: Path):
    return [(state, float(value)) for state, value in
            re.findall(r"slack \((MET|VIOLATED)\)\s+(-?\d+(?:\.\d+)?)", path.read_text())]


def coverage(path: Path):
    text = path.read_text()
    rows = {}
    for name in ("setup", "hold", "out_setup", "out_hold"):
        match = re.search(r"^" + name + r"\s+(\d+)\s+(\d+) \([^\n]+?\)\s+(\d+) \([^\n]+?\)\s+(\d+) \(", text, re.M)
        if not match:
            raise AssertionError("missing coverage " + name)
        rows[name] = tuple(map(int, match.groups()))
    return rows


class Hammer(unittest.TestCase):
    def test_01_both_quarantine_directories_are_double_sealed(self):
        verify_dir(M1288); verify_dir(M1311)
        self.assertEqual(sha(M1288 / "SHA256SUMS"), PINS[M1288 / "SHA256SUMS"])
        self.assertEqual(sha(M1311 / "SHA256SUMS"), PINS[M1311 / "SHA256SUMS"])

    def test_02_pinned_evidence_is_exact_and_docs359_unchanged(self):
        for path, wanted in PINS.items():
            with self.subTest(path=path.name):
                regular(path); self.assertEqual(sha(path), wanted)

    def test_03_pt_and_monitor_completed_without_process_failure(self):
        self.assertEqual((M1288 / "pt.rc").read_text().strip(), "0")
        self.assertEqual((M1288 / "runtime_monitor.rc").read_text().strip(), "0")
        self.assertIn("PTSTA_INTERNAL_COMPLETE=PASS",
                      (M1288 / "PTSTA_INTERNAL_COMPLETE.txt").read_text())
        for name in ("timing_setup_slow.rpt", "timing_hold_fast.rpt", "global_timing.rpt",
                     "analysis_coverage.rpt", "check_timing.rpt", "constraint_violators.rpt"):
            regular(M1288 / "reports" / name)

    def test_04_worst_setup_and_hold_are_negative(self):
        setup = slacks(M1288 / "reports/timing_setup_slow.rpt")
        hold = slacks(M1288 / "reports/timing_hold_fast.rpt")
        self.assertEqual(len(setup), 100); self.assertEqual(len(hold), 100)
        self.assertEqual(min(v for _, v in setup), -0.001154)
        self.assertEqual(min(v for _, v in hold), -0.022628)
        self.assertTrue(any(state == "VIOLATED" for state, _ in setup))
        self.assertTrue(any(state == "VIOLATED" for state, _ in hold))

    def test_05_global_timing_violation_counts_and_tns(self):
        text = (M1288 / "reports/global_timing.rpt").read_text()
        self.assertRegex(text, r"Setup violations[\s\S]*?TNS\s+-0\.01[\s\S]*?NUM\s+16")
        self.assertRegex(text, r"Hold violations[\s\S]*?TNS\s+-101\.91[\s\S]*?NUM\s+10047")

    def test_06_coverage_rows_fail_strict_gate(self):
        self.assertEqual(coverage(M1288 / "reports/analysis_coverage.rpt"), {
            "setup": (10573, 10557, 16, 0),
            "hold": (10573, 526, 10047, 0),
            "out_setup": (607, 475, 0, 132),
            "out_hold": (607, 475, 0, 132),
        })

    def test_07_unconstrained_parser_count_is_zero_but_not_a_timing_pass(self):
        text = (M1288 / "reports/check_timing.rpt").read_text()
        counts = []
        for pattern in (r"There (?:is|are)\s+(\d+)\s+input ports?.{0,240}?will be unconstrained",
                        r"There (?:is|are)\s+(\d+)\s+endpoints?.{0,240}?unconstrained"):
            counts.extend(int(x) for x in re.findall(pattern, text, re.I | re.S))
        self.assertEqual(sum(counts), 0)
        self.assertIn("check_timing succeeded", text)
        self.assertGreater(16 + 10047 + 132 + 132, 0)

    def test_08_failure_receipt_no_retry_and_diagnostic_only(self):
        receipt = json.loads((M1311 / "m1311_failure_receipt_r1.json").read_text())
        self.assertEqual(receipt["status"], "STOP_M1311_LAUNCH_OR_ADJUDICATION_FAILED")
        self.assertFalse(receipt["claim_boundary"]["timing_gate_pass"])
        self.assertEqual((M1288_ATTEMPT / "attempt.txt").read_text().strip(),
                         "status=M1288_ONE_SHOT_ATTEMPT_CONSUMED")
        self.assertEqual((M1311_ATTEMPT / "attempt.txt").read_text().strip(),
                         "status=M1311_ONE_SHOT_ATTEMPT_CONSUMED")
        self.assertFalse((RUNS / "m1288_m917_fixed_t10_prelayout_ptsta_r1_20260830").exists())
        self.assertFalse((RUNS / "m1311_m1288_fixed_t10_ptsta_adjudication_r1_20260831").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
