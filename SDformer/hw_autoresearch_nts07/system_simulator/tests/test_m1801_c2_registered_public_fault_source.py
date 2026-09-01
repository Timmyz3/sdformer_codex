#!/usr/bin/env python3
"""Mutation regression for M1801 source checker.  CPython 3.6 compatible."""
from __future__ import print_function

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve()
HW = HERE.parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1801_c2_registered_public_fault_source.py"
SPEC = importlib.util.spec_from_file_location("m1801_checker", str(CHECKER))
M = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("checker loader missing")
SPEC.loader.exec_module(M)


def rejected(function, *args):
    try:
        function(*args)
    except Exception:
        return 1
    raise RuntimeError("mutation accepted")


def main():
    export = M.source_text(M.RTL_EXPORT)
    top = M.source_text(M.RTL_TOP)
    tb_export = M.source_text(M.TB_EXPORT)
    tb_full = M.source_text(M.TB_FULL)
    fl_export = [row.strip() for row in M.source_text(M.FL_EXPORT).splitlines()
                 if row.strip()]
    fl_full = [row.strip() for row in M.source_text(M.FL_FULL).splitlines()
               if row.strip()]
    value = M.main()
    if value["status"] != "PASS_M1801_SOURCE_ONLY_NO_EDA_NO_ATTEMPT":
        raise RuntimeError("live checker status")

    mutations = 0
    for mutant in (
        export.replace("core_req_valid\n        &&", "1'b1\n        &&", 1),
        export.replace("core_rsp_valid\n        &&", "1'b1\n        &&", 1),
        export.replace("always_ff @(posedge clk_core)", "always_comb", 1),
        export.replace("protocol_error <= 1'b1", "protocol_error <= 1'b0", 1),
        export.replace("assign core_fault_event = core_fault_sample_enable",
                       "assign core_fault_event = 1'b1", 1),
        export.replace("assign adapter_fault_event = adapter_fault_sample_enable",
                       "assign adapter_fault_event = 1'b1", 1),
        export + "\nassign protocol_error = core_fault_event_raw;\n",
        export + "\nforce protocol_error = 1'b0;\n",
        export.replace("!= adapter_req_accept", "!== adapter_req_accept", 1),
    ):
        mutations += rejected(M.audit_export, mutant)
    for mutant in (
        top.replace(".core_req_valid(core_mem_req_valid)",
                    ".core_req_valid(1'b1)", 1),
        top.replace("m1801_c2_registered_public_fault_export public_fault_export", "", 1),
        top + "\nassign protocol_error = consistency_fault_now;\n",
    ):
        mutations += rejected(M.audit_top, mutant)
    core_terms = [
        "header_valid", "raw_valid", "core_busy", "core_mem_req_valid",
        "core_mem_rsp_valid", "result_valid", "token_done_valid",
    ]
    adapter_terms = ["adapter_busy", "core_mem_req_valid", "(|mem_rsp_valid)"]
    core_equation = (
        "assign core_fault_sample_enable = header_valid || raw_valid || core_busy\n"
        "        || core_mem_req_valid || core_mem_rsp_valid || result_valid\n"
        "        || token_done_valid;")
    adapter_equation = (
        "assign adapter_fault_sample_enable = adapter_busy || core_mem_req_valid\n"
        "        || (|mem_rsp_valid);")
    mutations += rejected(M.audit_top,
                          top.replace(core_equation,
                                      "assign core_fault_sample_enable = 1'b0;", 1))
    mutations += rejected(M.audit_top,
                          top.replace(adapter_equation,
                                      "assign adapter_fault_sample_enable = 1'b0;", 1))
    for removed in core_terms:
        survivors = [term for term in core_terms if term != removed]
        mutant_equation = "assign core_fault_sample_enable = " \
            + " || ".join(survivors) + ";"
        mutations += rejected(M.audit_top,
                              top.replace(core_equation, mutant_equation, 1))
    for removed in adapter_terms:
        survivors = [term for term in adapter_terms if term != removed]
        mutant_equation = "assign adapter_fault_sample_enable = " \
            + " || ".join(survivors) + ";"
        mutations += rejected(M.audit_top,
                              top.replace(adapter_equation, mutant_equation, 1))
    for mutant_export, mutant_full in (
        (tb_export.replace("#1ps", ""), tb_full),
        (tb_export.replace("core_fault_event_raw = 1'b1", "", 1), tb_full),
        (tb_export, tb_full.replace("run_raw_attack_k8();", "", 1)),
        (tb_export.replace("adapter_fault_event_raw = 1'b1", "", 1), tb_full),
        (tb_export.replace("reset_boundary();", ""), tb_full),
        (tb_export.replace("repeat (2) @(posedge clk_core);\n        #1ps;\n        require_binary_zero(\"reset asserted\")",
                           "repeat (1) @(posedge clk_core);\n        #1ps;\n        require_binary_zero(\"reset asserted\")", 1), tb_full),
        (tb_export.replace("require_binary_zero(\"reset recovery\");", "", 1), tb_full),
        (tb_export, tb_full.replace("||protocol_attack_count!=5||numeric_mismatch_count!=0",
                                   "||protocol_attack_count!=4||numeric_mismatch_count!=0", 1)),
        (tb_export, tb_full.replace("protocol_attacks=5", "protocol_attacks=4", 1)),
        (tb_export, tb_full.replace("M1801 K8 illegal header accepted", "", 1)),
        (tb_export, tb_full.replace("M349 K8 illegal header escaped", "", 1)),
        (tb_export, tb_full.replace("M1801 K8 illegal raw packet accepted", "", 1)),
        (tb_export, tb_full.replace("M1801 K8 illegal raw packet escaped sticky fault", "", 1)),
        (tb_export, tb_full.replace("M1801 K8 spurious response accepted", "", 1)),
        (tb_export, tb_full.replace("M349 K8 spurious response escaped", "", 1)),
    ):
        mutations += rejected(M.audit_tb, mutant_export, mutant_full)
    mutations += rejected(M.audit_filelists, fl_export[1:], fl_full)
    mutations += rejected(M.audit_filelists, fl_export,
                          [row for row in fl_full if "rtl_m1609" not in row])
    mutations += rejected(M.audit_filelists, fl_export,
                          fl_full + [fl_full[-1]])
    if mutations != 42:
        raise RuntimeError("mutation count: %d" % mutations)
    print("PASS_M1801_SOURCE_MUTATIONS rejected=42 owner_terms=10 valid_gates=2 reset_recovery=true pass_attack_count=true cpython_compatible=true eda_runs=0 attempts_created=0")
    return {"status": "PASS_M1801_SOURCE_MUTATIONS", "rejected": mutations}


if __name__ == "__main__":
    main()
