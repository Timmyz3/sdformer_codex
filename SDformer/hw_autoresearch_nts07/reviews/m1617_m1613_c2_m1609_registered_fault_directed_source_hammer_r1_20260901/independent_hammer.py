#!/usr/bin/env python3
"""Compile-free, different-author hammer for the exact M1613 source package."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SUCCESSOR = HW / ("rtl_m1609/"
    "m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_"
    "registered_fault_successor.sv")
PREDECESSOR = HW / (
    "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv")
FILELIST = HW / (
    "dc_handoff/filelists/date_m1613_c2_m1609_registered_fault_directed_vcs.f")
TB = HW / "dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv"
RUNNER = HW / (
    "dc_handoff/scripts/run_vcs_m1613_c2_m1609_registered_fault_directed_exact_sha_r1.sh")
AUTHOR_TEST = HW / (
    "system_simulator/tests/test_m1613_c2_m1609_registered_fault_directed_source.py")
CONTRACT = HW / (
    "contracts/m1613_c2_m1609_registered_fault_directed_source_contract_r1_20260901.json")
M1611 = HW / (
    "reviews/m1611_m1609_c2_registered_fault_successor_source_independent_review_r1_20260901")
AUTHOR_HANDOFF = HW / (
    "reviews/m1613_c2_m1609_registered_fault_directed_source_author_handoff_r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

OLD_REL = "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
NEW_REL = ("rtl_m1609/"
    "m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_"
    "registered_fault_successor.sv")
TB_REL = "dc_handoff/tb/tb_m1613_c2_m1609_registered_fault_directed.sv"
RESULT_REL = "results/m1613_c2_m1609_registered_fault_directed_vcs_r1_20260901"
ATTEMPT_REL = "results/.m1613_c2_m1609_registered_fault_directed_vcs_attempt_consumed"
HAMMER_REL = "reviews/m1617_m1613_c2_m1609_registered_fault_directed_source_hammer_r1_20260901"
RELEASE_REL = "contracts/m1618_m1617_m1613_c2_registered_fault_directed_vcs_launch_release_r1_20260901.json"

PINS = {
    SUCCESSOR: "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    PREDECESSOR: "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    FILELIST: "071e37d731988a12c3b0adc6380e179de55260c34325fce944daabba6c58671d",
    TB: "096f32095a81abbedf4c0bda59a2a146df764d18bdc2136c31e0c7a7319a57a4",
    RUNNER: "f2b3888879cb5a6af4396eb8b4971510453a47622299e17dd6702925587c0b29",
    AUTHOR_TEST: "0f8ef678ef40ee1413939e894bd86fc658821847b6254437e7ecca08ea59b4ea",
    CONTRACT: "248c9065d81608a8fc2aacdd8539a3287462653e411ee545a8f320a98a8a5f8d",
    M1611 / "review.json": "6109dff51fb6b60463afbfa32f3756c6ceffae1b12dc085134a1c008cd2bf480",
    M1611 / "SHA256SUMS": "58f2e9701fab6450557d1bef44604997b4b501a18d799c4f9e91719a6494f0d5",
    M1611 / "SHA256SUMS.seal.sha256": "6e56d25c27c59fad37875d533e2dcc9e03abd0635d687d874fdbed41bbbf45fd",
    AUTHOR_HANDOFF / "handoff.json": "5e018383142cd07f5b94437a26f48a52d7b0f8c8686b1cd4b68d9e4bd8b2982d",
    AUTHOR_HANDOFF / "SHA256SUMS": "6bae59e822b048eb42b83868c897852d6a06c5f09ef171bb32b03b4849b6a78c",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_text(text):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise AssertionError("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("nonfinite JSON: " + token)))


def verify_tree_seal(path):
    path = Path(path)
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    rows = [row for row in manifest.read_text(encoding="utf-8").splitlines()
            if row]
    assert rows
    for row in rows:
        expected, rel = row.split("  ", 1)
        assert rel not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")
        item = path / rel
        assert item.is_file() and not item.is_symlink()
        assert sha256(item) == expected
    expected_outer = "%s  SHA256SUMS\n" % sha256(manifest)
    assert outer.read_text(encoding="utf-8") == expected_outer


def exactly_once(text, token):
    assert text.count(token) == 1, "token count is not one: " + token


def before(text, left, right):
    assert left in text and right in text and text.index(left) < text.index(right), (
        "ordering failure: %s before %s" % (left, right))


def audit_filelist(text):
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    assert rows == [NEW_REL, TB_REL]
    assert OLD_REL not in text
    assert text.count(NEW_REL) == 1 and text.count(TB_REL) == 1


def audit_tb(text):
    required_once = [
        "module tb_m1613_c2_m1609_registered_fault_directed;",
        "m214_fc2_raw4_to_descriptor4_terminal_hint_compactor dut (.*);",
        "task automatic legal_terminal_linger_case;",
        "task automatic illegal_header_case;",
        "task automatic illegal_raw_case;",
        "if (protocol_error !== 0 || dut.fault_q !== 0)\n"
        "                $fatal(1, \"M1613 reset failed to clear registered fault\");",
        "if (raw_ready !== 1 || raw_accept !== 1\n"
        "                    || protocol_error !== 0)\n"
        "                $fatal(1, \"M1613 legal terminal packet was not accepted\");",
        "if (descriptor_accept !== 1)\n"
        "                $fatal(1, \"M1613 legal terminal descriptor did not bypass\");",
        "if (dut.illegal_request !== 1)\n"
        "                $fatal(1, \"M1613 did not exercise post-terminal linger seam\");",
        "if (protocol_error !== 0 || dut.fault_q !== 0)\n"
        "                $fatal(1, \"M1613 legal terminal produced a false fault pulse\");",
        "if (dut.illegal_request !== 1 || header_ready !== 0\n"
        "                    || header_accept !== 0 || protocol_error !== 0)",
        "if (dut.fault_q !== 1 || protocol_error !== 1)\n"
        "                $fatal(1, \"M1613 illegal header did not latch after edge\");",
        "if (dut.illegal_request !== 1 || raw_ready !== 0\n"
        "                    || raw_accept !== 0 || protocol_error !== 0)",
        "if (dut.fault_q !== 1 || protocol_error !== 1)\n"
        "                $fatal(1, \"M1613 illegal raw did not latch after edge\");",
        "raw_lane_valid = 4'b1011;",
        "property p_header_accept_requires_ready;",
        "property p_raw_accept_requires_ready;",
        "property p_registered_fault_sticky;\n"
        "        @(posedge clk_core) disable iff (rst_core)\n"
        "        protocol_error |=> protocol_error;\n"
        "    endproperty",
        "if (legal_terminal_no_false_pulse != 1\n"
        "                || legal_descriptor_accepts != 1\n"
        "                || illegal_header_latched != 1\n"
        "                || illegal_raw_latched != 1\n"
        "                || sticky_checks != 3)",
        "source_only=false performance=false",
        "#100000;",
    ]
    for token in required_once:
        exactly_once(text, token)
    assert text.count("#1ps;") == 12
    assert text.count("apply_reset();") == 3
    assert text.count("sticky_checks++;") == 2
    assert text.count("if (protocol_error !== 1) $fatal(1, \"M1613 fault not sticky\");") == 1
    assert text.count("if (protocol_error !== 1) $fatal(1, \"M1613 raw fault not sticky\");") == 1
    assert "force " not in text and "release " not in text
    assert "$time" not in text and "$realtime" not in text
    terminal_drive = "raw_valid = 1;\n            #1ps;"
    terminal_edge = "@(posedge clk_core);\n            #1ps;"
    assert terminal_drive in text
    assert text.index(terminal_drive) < text.index(
        terminal_edge, text.index(terminal_drive))
    before(text, "legal_descriptor_accepts++;", "M1613 did not exercise post-terminal linger seam")


def audit_contract(text):
    value = strict_json_text(text)
    assert value["status"].endswith("__NO_EXECUTION")
    assert value["rtl_selection"]["successor_sha256"] == PINS[SUCCESSOR]
    assert value["rtl_selection"]["selection"] == "successor_only"
    assert value["rtl_selection"]["predecessor_in_new_filelist"] is False
    source = value["directed_source"]
    assert source["filelist_sha256"] == PINS[FILELIST]
    assert source["testbench_sha256"] == PINS[TB]
    assert source["new_result_namespace"] == RESULT_REL
    assert source["new_attempt_namespace"] == ATTEMPT_REL
    mandatory = value["mandatory_cases"]
    for key in ("legal_terminal_packet_accepted",
                "legal_descriptor_same_cycle_bypass_accepted",
                "post_terminal_raw_valid_linger_exercised",
                "post_terminal_combinational_illegal_request_observed",
                "illegal_header_fault_latched_after_edge",
                "illegal_raw_fault_latched_after_edge",
                "registered_fault_sticky_until_reset",
                "reset_clears_registered_fault"):
        assert mandatory[key] is True
    assert mandatory["legal_terminal_public_false_error_pulses"] == 0
    assert mandatory["illegal_header_ready"] == 0
    assert mandatory["illegal_header_accept"] == 0
    assert mandatory["illegal_raw_ready"] == 0
    assert mandatory["illegal_raw_accept"] == 0
    budget = value["future_execution_budget"]
    assert budget["authorized_now"] is False
    assert budget["after_independent_launch_admission"] == {
        "vcs_compiles": 1, "simv_runs": 1, "seeds": [1613],
        "dc_runs": 0, "ptpx_runs": 0}
    assert budget["automatic_retry"] is False
    chain = value["future_release_chain"]
    assert chain["source_hammer_directory"] == HAMMER_REL
    assert chain["release_path"] == RELEASE_REL
    assert chain["present_at_source_authoring"] is False
    assert chain["release_must_bind_runner_contract_and_hammer_review_sha"] is True
    assert chain["caller_must_pin_runner_and_release_sha"] is True
    claim = value["claim_boundary"]
    assert claim["source_only"] is True and claim["author_static_test_only"] is True
    for key in ("vcs", "rtl_behavior_proven", "integration_outer_error_proven",
                "cycle_performance", "speedup", "area", "timing", "power",
                "energy", "paper_result"):
        assert claim[key] is False, key


def audit_runner(text):
    required_once = [
        'result="${hw_root}/' + RESULT_REL + '"',
        'attempt="${hw_root}/' + ATTEMPT_REL + '"',
        'hammer_dir="${hw_root}/' + HAMMER_REL + '"',
        'release="${hw_root}/' + RELEASE_REL + '"',
        'verify_dir_seal "${hammer_dir}"',
        'verify_file_seal "${release}"',
        'h["status"] == "PASS_M1617_M1613_C2_REGISTERED_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT"',
        'r["status"] == "AUTHORIZE_ONE_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_ATTEMPT"',
        'r["identity"]["runner_sha256"] == sha(runner)',
        'r["identity"]["source_contract_sha256"] == sha(contract)',
        'r["identity"]["hammer_review_sha256"] == sha(hammer)',
        'mkdir "${attempt}"',
        'printf \'M1613_ATTEMPT_CONSUMED runner_sha256=%s automatic_retry=false\\n\'',
        'same-UID VCS collision:',
        'p.stat().st_uid != os.getuid()',
        'int(p.name) in ancestry',
        'blocked = {"vcs", "vcs1", "vlogan", "simv"}',
        '"${vcs}" -full64 -sverilog -assert svaext -timescale=1ns/1ps',
        '"${simv}" +ntb_random_seed=1613 -no_save -cm assert',
        'vcs_compiles=1',
        'simv_runs=1',
        'source_only=false performance=false',
        '"performance":false,"dc":false,"power":false',
        'mv -T -n "${work}" "${result}"',
    ]
    for token in required_once:
        exactly_once(text, token)
    assert text.count("M1613_EXPECTED_RUNNER_SHA256") == 2
    assert text.count("M1613_EXPECTED_RELEASE_SHA256") == 2
    assert text.count('\n"${vcs}" -full64') == 1
    assert text.count('\n"${simv}" +ntb_random_seed=1613') == 1
    assert text.count('mkdir "${attempt}"') == 1
    assert text.count('mkdir "${work}"') == 1
    assert "pgrep -x simv" not in text
    assert "rm -rf" not in text and 'rm "${attempt}"' not in text
    for forbidden in ("dc_shell", "pt_shell", "ptpx", "fm_shell", "ucli"):
        assert forbidden not in text.lower()
    assert text.count(OLD_REL) == 1
    assert '-f "${filelist}" -top "${top}"' in text
    before(text, 'M1613_EXPECTED_RUNNER_SHA256', 'verify_dir_seal "${hammer_dir}"')
    before(text, 'verify_dir_seal "${hammer_dir}"', 'M1613_EXPECTED_RELEASE_SHA256')
    before(text, 'M1613_EXPECTED_RELEASE_SHA256', 'result/attempt/work namespace is not fresh')
    before(text, 'result/attempt/work namespace is not fresh', 'same-UID VCS collision:')
    before(text, 'same-UID VCS collision:', 'mkdir "${attempt}"')
    before(text, 'VCS environment mismatch', 'mkdir "${attempt}"')
    before(text, 'mkdir "${attempt}"', 'mkdir "${work}"')
    before(text, 'mkdir "${attempt}"', '\n"${vcs}" -full64')
    before(text, '\n"${vcs}" -full64', '\n"${simv}" +ntb_random_seed=1613')
    before(text, '\n"${simv}" +ntb_random_seed=1613', 'mv -T -n "${work}" "${result}"')


class M1617Hammer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filelist = FILELIST.read_text(encoding="utf-8")
        cls.tb = TB.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.contract = CONTRACT.read_text(encoding="utf-8")

    def test_01_exact_sha_and_regular_files(self):
        for path, expected in PINS.items():
            mode = path.lstat().st_mode
            self.assertTrue(stat.S_ISREG(mode), str(path))
            self.assertFalse(path.is_symlink(), str(path))
            self.assertEqual(sha256(path), expected, str(path))

    def test_02_prior_and_author_trees_are_sealed(self):
        verify_tree_seal(M1611)
        verify_tree_seal(AUTHOR_HANDOFF)

    def test_03_current_filelist_tb_runner_contract_pass(self):
        audit_filelist(self.filelist)
        audit_tb(self.tb)
        audit_runner(self.runner)
        audit_contract(self.contract)

    def test_04_old_m214_simultaneous_definition_rejected(self):
        with self.assertRaises(AssertionError):
            audit_filelist(self.filelist + OLD_REL + "\n")

    def test_05_settled_sampling_deletion_rejected(self):
        with self.assertRaises(AssertionError):
            audit_tb(self.tb.replace("#1ps;", "", 1))

    def test_06_legal_terminal_false_error_check_deletion_rejected(self):
        mutant = self.tb.replace(
            "if (protocol_error !== 0 || dut.fault_q !== 0)\n"
            "                $fatal(1, \"M1613 legal terminal produced a false fault pulse\");",
            "if (protocol_error !== 0)\n"
            "                $fatal(1, \"M1613 legal terminal produced a false fault pulse\");", 1)
        with self.assertRaises(AssertionError):
            audit_tb(mutant)

    def test_07_illegal_header_and_raw_latch_deletions_rejected(self):
        for message in ("M1613 illegal header did not latch after edge",
                        "M1613 illegal raw did not latch after edge"):
            with self.assertRaises(AssertionError):
                audit_tb(self.tb.replace(message, "removed latch check", 1))

    def test_08_sticky_and_reset_breakage_rejected(self):
        mutants = [
            self.tb.replace("protocol_error |=> protocol_error;",
                            "protocol_error |=> 1'b1;", 1),
            self.tb.replace("M1613 reset failed to clear registered fault",
                            "removed reset check", 1),
            self.tb.replace("sticky_checks != 3", "sticky_checks != 0", 1),
        ]
        for mutant in mutants:
            with self.assertRaises(AssertionError):
                audit_tb(mutant)

    def test_09_compile_and_sim_budget_expansion_rejected(self):
        vcs_call = '\n"${vcs}" -full64 -sverilog -assert svaext -timescale=1ns/1ps'
        sim_call = '\n"${simv}" +ntb_random_seed=1613 -no_save -cm assert'
        for mutant in (self.runner + vcs_call, self.runner + sim_call):
            with self.assertRaises(AssertionError):
                audit_runner(mutant)

    def test_10_hammer_or_release_bypass_rejected(self):
        for token in ('verify_dir_seal "${hammer_dir}"',
                      'verify_file_seal "${release}"',
                      'M1613_EXPECTED_RELEASE_SHA256'):
            with self.assertRaises(AssertionError):
                audit_runner(self.runner.replace(token, "BYPASSED", 1))

    def test_11_other_uid_false_block_and_same_uid_leak_rejected(self):
        mutants = [
            self.runner.replace("if p.stat().st_uid != os.getuid():",
                                "if False:", 1),
            self.runner.replace("or int(p.name) in ancestry", "", 1),
            self.runner.replace('blocked = {"vcs", "vcs1", "vlogan", "simv"}',
                                'blocked = {"vcs"}', 1),
        ]
        for mutant in mutants:
            with self.assertRaises(AssertionError):
                audit_runner(mutant)

    def test_12_namespace_and_attempt_order_mutations_rejected(self):
        namespace_mutant = self.runner.replace(RESULT_REL, RESULT_REL + "_weak", 1)
        attempt_line = 'mkdir "${attempt}"'
        order_mutant = self.runner.replace(attempt_line, "", 1) + "\n" + attempt_line + "\n"
        for mutant in (namespace_mutant, order_mutant):
            with self.assertRaises(AssertionError):
                audit_runner(mutant)

    def test_13_performance_claim_mutations_rejected(self):
        tb_mutant = self.tb.replace("performance=false", "performance=true", 1)
        contract_value = strict_json_text(self.contract)
        contract_value["claim_boundary"]["speedup"] = True
        runner_mutant = self.runner.replace('"performance":false',
                                             '"performance":true', 1)
        with self.assertRaises(AssertionError):
            audit_tb(tb_mutant)
        with self.assertRaises(AssertionError):
            audit_contract(json.dumps(contract_value))
        with self.assertRaises(AssertionError):
            audit_runner(runner_mutant)

    def test_14_future_namespaces_and_authority_are_absent(self):
        self.assertFalse((HW / RESULT_REL).exists())
        self.assertFalse((HW / ATTEMPT_REL).exists())
        self.assertFalse((HW / RELEASE_REL).exists())


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(M1617Hammer)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("M1617_STATIC_HAMMER checks=%d failures=%d errors=%d vcs=0 simv=0 eda=0" %
          (result.testsRun, len(result.failures), len(result.errors)))
    sys.exit(0 if result.wasSuccessful() else 1)
