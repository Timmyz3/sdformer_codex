#!/usr/bin/env python3
"""Mutation tests for the M1270/R13 source-only checker."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m1270check", HERE / "check_m1270r13_source.py")
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


def mutate_once(old, new):
    assert BASE.count(old) == 1, old
    return BASE.replace(old, new, 1)


def main():
    MOD.check_text(BASE)
    attacks = {
        "parent_force": BASE.replace(
            "real_m935_completion();",
            "force dut.issue_request_first = 1'b0;\n        real_m935_completion();",
            1),
        "child_release": BASE.replace(
            "real_m935_completion();",
            "release dut.u_frozen_m935.issue_request_first;\n        real_m935_completion();",
            1),
        "hierarchical_assignment": BASE.replace(
            "real_m935_completion();",
            "dut.u_frozen_m935.issue_request_first = 1'b0;\n        real_m935_completion();",
            1),
        "second_first": mutate_once(
            "serve_real_m935_beat(1'b0, 1);",
            "serve_real_m935_beat(1'b1, 1);"),
        "mask_one_source": mutate_once("16'h0003 : 16'h0000",
                                         "16'h0001 : 16'h0000"),
        "psum_every_beat": BASE.replace("p0 + expect_first", "p0 + 1"),
        "ii_weakened": mutate_once(
            "second_response_cycle - first_response_cycle >= 2",
            "second_response_cycle - first_response_cycle >= 1"),
        "join_deleted": BASE.replace("first_weight_only_join_hold",
                                       "first_weight_only_join_deleted"),
        "row_completion_deleted": mutate_once(
            "row_complete_count == row0 + 1",
            "row_complete_count >= row0"),
        "core_fault_deleted": mutate_once(
            "&& !dut.core_protocol_error\n                    && !dut.u_frozen_m935.fault_q\n                    && !weight_service_fault",
            "&& !dut.u_frozen_m935.fault_q\n                    && !weight_service_fault"),
        "sva_instance_renamed": mutate_once(
            "m1168r3_m1162_common_charge_protocol_assertions_r3 u_protocol_sva",
            "m1168r3_m1162_common_charge_protocol_assertions_r3 u_shadow"),
        "extra_fatal": BASE.replace("$finish;", "$fatal(1, \"shadow\");\n        $finish;", 1),
        "pass_escalation": mutate_once("system_speedup=false",
                                         "system_speedup=true"),
        "pass_shadow": BASE.replace(
            "PHASE_M1270R13_REAL_M935_INTEGRATED_ENTER",
            "PHASE_M1270R13_REAL_M935_INTEGRATED_ENTER_SHADOW"),
        "duplicate_initial": BASE.replace(
            "initial begin", "initial begin\n    end\n    initial begin", 1),
    }
    failed = [name for name, text in attacks.items() if not rejected(text)]
    if failed:
        raise AssertionError(f"attacks accepted: {failed}")
    print(f"PASS M1270/R13 source tests={1 + len(attacks)} attacks={len(attacks)}")


if __name__ == "__main__":
    main()
