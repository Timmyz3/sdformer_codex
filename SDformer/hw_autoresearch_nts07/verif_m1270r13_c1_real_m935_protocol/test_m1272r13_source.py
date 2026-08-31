#!/usr/bin/env python3
"""M1272 tests for exactly the four M1271 P1 repair classes."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m1272check", HERE / "check_m1272r13_source.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)
BASE = (HERE / "tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv").read_text()


def rejected(text):
    try:
        MOD.check_text(text)
        return False
    except Exception:
        return True


def once(old, new):
    assert BASE.count(old) == 1, old
    return BASE.replace(old, new, 1)


def main():
    MOD.check_text(BASE)
    attacks = {
        # M1271 P1-1: comments and near-neighbour runtime tokens.
        "comment_pass": once(
            '        $display("PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE',
            '        // $display("PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE'),
        "comment_phase_done": once(
            '        $display("PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE");',
            '        // $display("PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE");'),
        "phase_shadow": BASE.replace(
            "PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE",
            "PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE_SHADOW"),
        # M1271 P1-2: disappearance/dormancy of real workload.
        "comment_completion": once(
            "        real_m935_completion();",
            "        // real_m935_completion();"),
        "false_guard_beats": once(
            "            serve_real_m935_beat(1'b1, 0);\n            serve_real_m935_beat(1'b0, 1);",
            "            if (1'b0) begin\n                serve_real_m935_beat(1'b1, 0);\n                serve_real_m935_beat(1'b0, 1);\n            end"),
        "comment_nonfirst": once(
            "            serve_real_m935_beat(1'b0, 1);",
            "            // serve_real_m935_beat(1'b0, 1);"),
        # M1271 P1-3: bare and hierarchical request-object writes.
        "bare_blocking_assignment": once(
            "        real_m935_completion();",
            "        issue_request_first = 1'b0;\n        real_m935_completion();"),
        "bare_nonblocking_assignment": once(
            "        real_m935_completion();",
            "        issue_request_valid <= 1'b1;\n        real_m935_completion();"),
        "child_assignment": once(
            "        real_m935_completion();",
            "        dut.u_frozen_m935.issue_request_first = 1'b0;\n        real_m935_completion();"),
        "parent_force": once(
            "        real_m935_completion();",
            "        force dut.issue_request_first = 1'b0;\n        real_m935_completion();"),
        "inline_bare_force": once(
            "        real_m935_completion();",
            "        if (reset_n) force issue_request_valid = 1'b0;\n"
            "        real_m935_completion();"),
        "bare_release": once(
            "        real_m935_completion();",
            "        release issue_request_last;\n        real_m935_completion();"),
        # M1271 P1-4: executable oracle print/flush dominance.
        "comment_oracle_display": once(
            '            $display("ORACLE_M1270R13 site=%s',
            '            /* $display("ORACLE_M1270R13 site=%s'),
        "comment_oracle_flush": once(
            "            $fflush();\n            if (condition !== 1'b1)",
            "            // $fflush();\n            if (condition !== 1'b1)"),
        "runtime_guard_oracle_display": once(
            '            $display("ORACLE_M1270R13 site=%s',
            '            if (oracle_count == -1) $display("ORACLE_M1270R13 site=%s'),
        "false_guard_completion": once(
            "        real_m935_completion();",
            "        if (1'b0) real_m935_completion();"),
    }
    # Close the block-comment mutation after the display statement so the
    # remaining oracle fatal is syntactically visible to the checker.
    attacks["comment_oracle_display"] = attacks["comment_oracle_display"].replace(
        "                row_complete_count, task_done_count);\n            $fflush();",
        "                row_complete_count, task_done_count); */\n            $fflush();", 1)
    failed = [name for name, text in attacks.items() if not rejected(text)]
    if failed:
        raise AssertionError("M1272 accepted attacks: " + repr(failed))
    print("PASS M1272/R13 checker-only tests={} attacks={} p1_classes=4".format(
        1 + len(attacks), len(attacks)))


if __name__ == "__main__":
    main()
