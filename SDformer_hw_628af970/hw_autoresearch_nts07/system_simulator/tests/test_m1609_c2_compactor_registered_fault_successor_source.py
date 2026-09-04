#!/usr/bin/env python3
"""Compile-free source checks for the additive M1609 compactor successor."""

from __future__ import print_function

import hashlib
from pathlib import Path
import re
import stat
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PREDECESSOR = HW / "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
SUCCESSOR = HW / (
    "rtl_m1609/"
    "m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_"
    "registered_fault_successor.sv"
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
MODULE = "m214_fc2_raw4_to_descriptor4_terminal_hint_compactor"
PREDECESSOR_SHA256 = "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

INTRO = """// M1609 additive source successor of the frozen M214 compactor at
// rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv
// (frozen SHA-256 e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5).
// This file deliberately keeps the legacy module name and full port contract;
// a future filelist must select exactly one of the predecessor or this source.
// The only executable semantic delta is that public protocol_error exposes the
// compactor's synchronous sticky fault_q, not the current-cycle combinational
// illegal_request. illegal_request still gates raw/header acceptance and is
// still sampled into fault_q on a clock edge. Other C2/frontend/service error
// sources remain outside this local boundary and must not be masked upstream.
//
"""
NEW_ASSIGN = """    // M1609: only sampled, sticky compactor faults cross this public boundary.
    // A true illegal_request remains blocked by ready/legal gating and is
    // latched into fault_q by state_update at the accepting clock boundary.
    assign protocol_error = fault_q;
"""
OLD_ASSIGN = "    assign protocol_error = fault_q || illegal_request;\n"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class M1609SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = PREDECESSOR.read_text(encoding="utf-8")
        cls.new = SUCCESSOR.read_text(encoding="utf-8")

    def test_01_frozen_identities(self):
        self.assertEqual(sha256(PREDECESSOR), PREDECESSOR_SHA256)
        self.assertEqual(sha256(DOCS359), DOCS359_SHA256)
        for path in (PREDECESSOR, SUCCESSOR, DOCS359):
            mode = path.lstat().st_mode
            self.assertTrue(stat.S_ISREG(mode))
            self.assertFalse(path.is_symlink())

    def test_02_additive_identity_comment_is_exact(self):
        self.assertEqual(self.new.count(INTRO), 1)
        self.assertIn("future filelist must select exactly one", self.new)
        self.assertIn(PREDECESSOR_SHA256, self.new)

    def test_03_only_executable_delta_is_public_assignment(self):
        normalized = self.new.replace(INTRO, "")
        self.assertEqual(normalized.count(NEW_ASSIGN), 1)
        normalized = normalized.replace(NEW_ASSIGN, OLD_ASSIGN)
        self.assertEqual(normalized, self.old)

    def test_04_same_legacy_module_and_port_contract(self):
        pattern = r"\bmodule\s+" + re.escape(MODULE) + r"\b"
        self.assertEqual(len(re.findall(pattern, self.old)), 1)
        self.assertEqual(len(re.findall(pattern, self.new)), 1)
        self.assertEqual(len(re.findall(r"^module\b", self.new,
                                        flags=re.MULTILINE)), 1)
        self.assertEqual(len(re.findall(r"^endmodule\b", self.new,
                                        flags=re.MULTILINE)), 1)

    def test_05_public_output_is_registered_fault_only(self):
        self.assertEqual(self.new.count("assign protocol_error = fault_q;"), 1)
        self.assertNotIn("assign protocol_error = fault_q || illegal_request;",
                         self.new)
        self.assertEqual(self.new.count("logic fault_q, token_active_q"), 1)

    def test_06_illegal_request_and_fault_latch_are_preserved(self):
        for token in (
            "assign illegal_request = (header_valid",
            "|| (raw_valid && !raw_packet_legal);",
            "if (illegal_request) fault_q <= 1;",
            "if (rst_core) begin\n            fault_q <= 0;",
        ):
            self.assertEqual(self.old.count(token), 1)
            self.assertEqual(self.new.count(token), 1)

    def test_07_ready_and_acceptance_gates_are_preserved(self):
        for token in (
            "assign header_ready = !fault_q && !token_active_q && header_legal;",
            "assign header_accept = header_valid && header_ready;",
            "assign raw_ready = !fault_q && raw_packet_legal",
            "assign raw_accept = raw_valid && raw_ready;",
        ):
            self.assertEqual(self.old.count(token), 1)
            self.assertEqual(self.new.count(token), 1)

    def test_08_no_outer_c2_error_source_is_present_or_masked(self):
        for token in (
            "m216_fc2_raw4_to_source_cap_frontend",
            "m204_protocol_error", "svc_protocol_error",
            "numeric_overflow", "stale_response_seen",
        ):
            self.assertNotIn(token, self.new)

    def test_09_compile_preflight_hygiene(self):
        self.assertTrue(self.new.startswith("`timescale 1ns/1ps\n"
                                            "`default_nettype none\n"))
        self.assertTrue(self.new.endswith("`default_nettype wire\n"))
        self.assertNotIn("`include", self.new)
        self.assertNotIn("force ", self.new)
        self.assertNotIn("release ", self.new)
        self.assertNotIn("initial begin", self.new)
        self.assertNotIn("$display", self.new)
        self.assertNotIn("$finish", self.new)


if __name__ == "__main__":
    unittest.main(verbosity=2)
