#!/usr/bin/env python3
"""Compile-free audit for the M1613 exclusive M1609 VCS source handoff."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PREDECESSOR = HW / "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
SUCCESSOR = HW / (
    "rtl_m1609/"
    "m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_"
    "registered_fault_successor.sv"
)
OLD_TB = HW / "tb_m214/tb_m214_fc2_raw4_to_same_done_load_frontend.sv"
OLD_FILELIST = HW / "dc_handoff/filelists/date_m214_fc2_raw4_to_same_done_load_directed_vcs.f"
SETTLED_TB = HW / "dc_handoff/tb/tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault.sv"
SETTLED_FILELIST = HW / "dc_handoff/filelists/date_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault_source.f"
TB = HW / "dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1613_c2_m1609_registered_fault_directed_vcs.f"
CONTRACT = HW / "contracts/m1613_c2_m1609_registered_fault_directed_source_contract_r1_20260901.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1613_c2_m1609_registered_fault_directed_exact_sha_r1.sh"
M1611 = HW / "reviews/m1611_m1609_c2_registered_fault_successor_source_independent_review_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

OLD_REL = "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
NEW_REL = (
    "rtl_m1609/"
    "m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_"
    "registered_fault_successor.sv"
)
TB_REL = "dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv"
RESULT_REL = "results/m1613_c2_m1609_registered_fault_directed_vcs_r1_20260901"
ATTEMPT_REL = "results/.m1613_c2_m1609_registered_fault_directed_vcs_attempt_consumed"

PINS = {
    PREDECESSOR: "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    SUCCESSOR: "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    OLD_TB: "d8d64c0aef21213c3e7a139abba58f6897bb4c0738eec999bc5ba9204ba22b4e",
    OLD_FILELIST: "642d23156c35de14d5d7d60b87d8d331b03100eb9b4fae929f24443db54b01c2",
    SETTLED_TB: "3e8a9254fd9104aeeb4d3f05077a9f2b8ae33a9617d3236447108a5b666ba8e4",
    SETTLED_FILELIST: "b6e384a3b7de9541a66af0302722c9ae9ca12b50e5e57a1ac764bf1576a39a53",
    TB: "096f32095a81abbedf4c0bda59a2a146df764d18bdc2136c31e0c7a7319a57a4",
    FILELIST: "071e37d731988a12c3b0adc6380e179de55260c34325fce944daabba6c58671d",
    CONTRACT: "248c9065d81608a8fc2aacdd8539a3287462653e411ee545a8f320a98a8a5f8d",
    M1611 / "review.json": "6109dff51fb6b60463afbfa32f3756c6ceffae1b12dc085134a1c008cd2bf480",
    M1611 / "SHA256SUMS": "58f2e9701fab6450557d1bef44604997b4b501a18d799c4f9e91719a6494f0d5",
    M1611 / "SHA256SUMS.seal.sha256": "6e56d25c27c59fad37875d533e2dcc9e03abd0635d687d874fdbed41bbbf45fd",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise AssertionError("duplicate JSON key: " + key)
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(
                             AssertionError("nonfinite JSON: " + token)))


def audit_filelist(text):
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    assert rows == [NEW_REL, TB_REL], "filelist is not the exact successor-only pair"
    assert OLD_REL not in text, "frozen predecessor leaked into M1613 filelist"
    assert text.count(NEW_REL) == 1 and text.count(TB_REL) == 1


def audit_tb(text):
    required_once = [
        "module tb_m1613_c2_m1609_registered_fault_directed;",
        "m214_fc2_raw4_to_descriptor4_terminal_hint_compactor dut (.*);",
        "task automatic legal_terminal_linger_case;",
        "task automatic illegal_header_case;",
        "task automatic illegal_raw_case;",
        "if (dut.illegal_request !== 1)\n                $fatal(1, \"M1613 did not exercise post-terminal linger seam\");",
        "if (protocol_error !== 0 || dut.fault_q !== 0)\n                $fatal(1, \"M1613 legal terminal produced a false fault pulse\");",
        "if (dut.fault_q !== 1 || protocol_error !== 1)\n                $fatal(1, \"M1613 illegal header did not latch after edge\");",
        "if (dut.fault_q !== 1 || protocol_error !== 1)\n                $fatal(1, \"M1613 illegal raw did not latch after edge\");",
        "raw_lane_valid = 4'b1011;",
        "property p_registered_fault_sticky;\n        @(posedge clk_core) disable iff (rst_core)\n        protocol_error |=> protocol_error;\n    endproperty",
        "#100000;",
    ]
    for token in required_once:
        assert text.count(token) == 1, "TB token count mismatch: " + token
    assert text.count("#1ps;") == 12, "settled sampling count changed"
    assert text.count("raw_lane_valid = 4'b1111;") == 1
    assert text.count("legal_descriptor_accepts++;") == 1
    assert text.index("legal_descriptor_accepts++;") < text.index("@(posedge clk_core);", text.index("legal_descriptor_accepts++;"))
    assert "legal_terminal_no_false_pulse=%%0d".replace("%%", "%") in text
    assert "illegal_header_latched=%%0d".replace("%%", "%") in text
    assert "illegal_raw_latched=%%0d".replace("%%", "%") in text
    assert "sticky_checks=%%0d".replace("%%", "%") in text
    assert "performance=false" in text
    assert "force " not in text and "release " not in text


def audit_runner(text):
    required = [
        "M1613_EXPECTED_RUNNER_SHA256",
        "M1613_EXPECTED_RELEASE_SHA256",
        "m1617_m1613_c2_m1609_registered_fault_directed_source_hammer_r1_20260901",
        "m1618_m1617_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json",
        "PASS_M1617_M1613_C2_REGISTERED_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT",
        "AUTHORIZE_ONE_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT",
        "date_m1613_c2_m1609_registered_fault_directed_vcs.f",
        "tb_m1613_c2_m1609_registered_fault_directed",
        RESULT_REL,
        ATTEMPT_REL,
        "test_m1613_c2_m1609_registered_fault_directed_source.py",
        "VCS_COMPILE",
        "SIMV_RUN",
        "+ntb_random_seed=1613",
        "legal_terminal_no_false_pulse=1",
        "legal_descriptor_accepts=1",
        "illegal_header_latched=1",
        "illegal_raw_latched=1",
        "sticky_checks=3",
        "vcs_compiles=1",
        "simv_runs=1",
        "performance=false",
        "same-UID VCS collision",
    ]
    for token in required:
        assert token in text, "runner token absent: " + token
    assert ('result="${hw_root}/' + RESULT_REL + '"') in text
    assert ('attempt="${hw_root}/' + ATTEMPT_REL + '"') in text
    assert text.count(OLD_REL) == 1, "predecessor exclusion guard changed"
    assert "-f " + OLD_REL not in text, "runner directly selects predecessor"
    for forbidden in ("dc_shell", "pt_shell", "ptpx", "fm_shell", "ucli"):
        assert forbidden not in text.lower(), "runner contains forbidden tool: " + forbidden
    assert "pgrep -x simv" not in text
    assert "p.stat().st_uid != os.getuid()" in text
    assert "int(p.name) in ancestry" in text
    assert text.count("\n\"${vcs}\" -full64") == 1, "runner VCS compile budget is not one"
    assert text.count("\n\"${simv}\" +ntb_random_seed=1613") == 1, "runner simv budget is not one"


class M1613SourceTests(unittest.TestCase):
    def test_01_all_frozen_and_new_pre_runner_identities(self):
        for path, expected in PINS.items():
            self.assertTrue(path.is_file() and not path.is_symlink(), str(path))
            self.assertEqual(sha256(path), expected, str(path))

    def test_02_exclusive_filelist(self):
        audit_filelist(FILELIST.read_text(encoding="utf-8"))

    def test_03_directed_tb_contract(self):
        audit_tb(TB.read_text(encoding="utf-8"))

    def test_04_contract_is_fail_closed(self):
        value = strict_json(CONTRACT)
        self.assertTrue(value["claim_boundary"]["source_only"])
        for key in ("vcs", "rtl_behavior_proven", "cycle_performance",
                    "speedup", "area", "timing", "power", "energy",
                    "paper_result"):
            self.assertFalse(value["claim_boundary"][key], key)
        self.assertFalse(value["future_execution_budget"]["authorized_now"])
        self.assertEqual(value["future_execution_budget"]
                         ["after_independent_launch_admission"]["vcs_compiles"], 1)
        self.assertEqual(value["future_execution_budget"]
                         ["after_independent_launch_admission"]["simv_runs"], 1)
        self.assertEqual(value["rtl_selection"]["selection"], "successor_only")
        self.assertFalse(value["rtl_selection"]["predecessor_in_new_filelist"])
        chain = value["future_release_chain"]
        self.assertFalse(chain["present_at_source_authoring"])
        self.assertTrue(chain["caller_must_pin_runner_and_release_sha"])
        self.assertEqual(chain["source_hammer_status"],
            "PASS_M1617_M1613_C2_REGISTERED_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT")
        self.assertEqual(chain["release_status"],
            "AUTHORIZE_ONE_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT")

    def test_05_runner_source_contract(self):
        self.assertTrue(RUNNER.is_file() and not RUNNER.is_symlink())
        audit_runner(RUNNER.read_text(encoding="utf-8"))

    def test_06_source_only_namespaces_are_fresh(self):
        self.assertFalse((HW / RESULT_REL).exists())
        self.assertFalse((HW / ATTEMPT_REL).exists())

    def test_07_filelist_predecessor_injection_is_rejected(self):
        text = FILELIST.read_text(encoding="utf-8") + OLD_REL + "\n"
        with self.assertRaises(AssertionError):
            audit_filelist(text)

    def test_08_tb_sampling_and_fault_mutations_are_rejected(self):
        text = TB.read_text(encoding="utf-8")
        mutations = [
            text.replace("#1ps;", "", 1),
            text.replace("M1613 illegal raw did not latch after edge",
                         "M1613 removed raw latch check", 1),
            text.replace("raw_lane_valid = 4'b1011;",
                         "raw_lane_valid = 4'b1111;", 1),
            text.replace(
                "if (protocol_error !== 0 || dut.fault_q !== 0)\n"
                "                $fatal(1, \"M1613 legal terminal produced a false fault pulse\");",
                "if (protocol_error !== 0)\n"
                "                $fatal(1, \"M1613 legal terminal produced a false fault pulse\");",
                1),
        ]
        for mutant in mutations:
            with self.assertRaises(AssertionError):
                audit_tb(mutant)

    def test_09_runner_namespace_and_budget_mutations_are_rejected(self):
        text = RUNNER.read_text(encoding="utf-8")
        mutations = [
            text.replace(RESULT_REL, RESULT_REL + "_changed", 1),
            text.replace("+ntb_random_seed=1613", "+ntb_random_seed=7", 1),
            text.replace("sticky_checks=3", "sticky_checks=2", 1),
        ]
        for mutant in mutations:
            with self.assertRaises(AssertionError):
                audit_runner(mutant)


if __name__ == "__main__":
    unittest.main(verbosity=2)
