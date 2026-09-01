#!/usr/bin/env python3
"""CPU-only fail-closed source audit for M1801.  CPython 3.6 compatible."""
from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
RTL_EXPORT = HW / "rtl_m1801/m1801_c2_registered_public_fault_export.sv"
RTL_TOP = HW / "rtl_m1801/m1801_m803_fc2_k8_registered_public_fault_8bank_raw4_acc24.sv"
TB_EXPORT = HW / "tb_m1801/tb_m1801_c2_registered_public_fault_export_directed.sv"
TB_FULL = HW / "tb_m1801/tb_m1801_c2_registered_public_fault_k8_vs_k1x8_raw4_acc24.sv"
FL_EXPORT = HW / "dc_handoff/filelists/iscas_m1801_c2_registered_public_fault_export_directed_vcs.f"
FL_FULL = HW / "dc_handoff/filelists/iscas_m1801_c2_registered_public_fault_k8_vs_k1x8_vcs.f"
CONTRACT = HW / "contracts/m1801_m1797_c2_registered_public_fault_evidence_successor_source_contract_r1_20260902.json"
PRE_TOP = HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv"
CORE = HW / "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv"
FRONTEND = HW / "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv"
DESCRIPTOR = HW / "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv"
SERVICE = HW / "rtl_m218/m218_fc2_tagged_slice_service_island.sv"
ADAPTER = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M1785_FAIL = HW / "results/m1785_c2_m1777_k8_mapped_primary_axis_first_fault_r1_20260902.failed_or_incomplete.quarantine"
M1786 = HW / "reviews/m1786_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source_hammer_r1_20260902"
M1797 = HW / "reviews/m1797_m1796_c2_registered_public_fault_source_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=pairs,
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root must be object")
    return value


def source_text(path):
    need(path.is_file() and not path.is_symlink(), "missing source: " + str(path))
    return path.read_text(encoding="utf-8")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", row) for row in text.splitlines())


def audit_export(text):
    body = strip_comments(text)
    required = (
        "module m1801_c2_registered_public_fault_export",
        "req_accept_mismatch = core_req_valid",
        "&& (core_req_accept != adapter_req_accept)",
        "rsp_accept_mismatch = core_rsp_valid",
        "&& (core_rsp_accept != adapter_rsp_accept)",
        "core_fault_event = core_fault_sample_enable",
        "adapter_fault_event = adapter_fault_sample_enable",
        "always_ff @(posedge clk_core)",
        "if (rst_core)",
        "protocol_error <= 1'b0",
        "protocol_error <= 1'b1",
    )
    for token in required:
        need(token in body, "export missing: " + token)
    need(body.count("protocol_error <=") == 2,
         "public fault must have exactly reset and sticky assignments")
    need("assign protocol_error" not in body,
         "public fault gained a combinational assignment")
    need("===" not in body and "!==" not in body,
         "case equality is forbidden in synthesizable repair")
    for token in ("force", "release", "initreg", "ignore_x", "ignorex",
                  "noassert", "assertoff"):
        need(token not in body.lower(), "forbidden repair token: " + token)


def audit_top(text):
    body = strip_comments(text)
    normalized = re.sub(r"\s+", " ", body).strip()
    required = (
        "module m1801_m803_fc2_k8_registered_public_fault_8bank_raw4_acc24",
        "m519_fc2_registered_release_standalone_raw4_acc24",
        "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter",
        "m1801_c2_registered_public_fault_export public_fault_export",
        ".core_req_valid(core_mem_req_valid)",
        ".core_req_accept(core_mem_req_accept)",
        ".adapter_req_accept(adapter_core_mem_req_accept)",
        ".core_rsp_valid(core_mem_rsp_valid)",
        ".core_rsp_accept(core_mem_rsp_accept)",
        ".adapter_rsp_accept(adapter_core_mem_rsp_accept)",
        ".protocol_error(protocol_error)",
    )
    for token in required:
        need(token in body, "top missing: " + token)
    core_owner_equation = (
        "assign core_fault_sample_enable = header_valid || raw_valid || "
        "core_busy || core_mem_req_valid || core_mem_rsp_valid || "
        "result_valid || token_done_valid;")
    adapter_owner_equation = (
        "assign adapter_fault_sample_enable = adapter_busy || "
        "core_mem_req_valid || (|mem_rsp_valid);")
    need(normalized.count(core_owner_equation) == 1,
         "exact core owner-enable equation")
    need(normalized.count(adapter_owner_equation) == 1,
         "exact adapter owner-enable equation")
    need(body.count(".core_req_valid(core_mem_req_valid)") == 2
         and body.count(".core_rsp_valid(core_mem_rsp_valid)") == 2,
         "fault export and adapter valid ownership wiring changed")
    need("consistency_fault_now" not in body
         and "assign protocol_error =" not in body,
         "frozen combinational export leaked into successor")
    need(body.count("m519_fc2_registered_release_standalone_raw4_acc24") == 1
         and body.count("m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter") == 1,
         "child topology changed")
    for data in ("result_accumulator", "mem_req_source_channel",
                 "mem_rsp_weight", "token_done_had_event"):
        need(data in body, "functional port missing: " + data)


def audit_tb(export_tb, full_tb):
    export_body = strip_comments(export_tb)
    full_body = strip_comments(full_tb)
    for token in (
            "legal_case0++", "invalid_payload_cases++",
            "core_fault_event_raw = 1'b1",
            "adapter_fault_event_raw = 1'b1",
            "core_req_accept = 1'b1",
            "adapter_req_accept = 1'b0",
            "core_rsp_accept = 1'b0",
            "adapter_rsp_accept = 1'b1",
            "@(posedge clk_core)", "@(negedge clk_core)",
            "#1ps", "$isunknown(protocol_error)",
            "ap_public_fault_binary_posedge",
            "ap_public_fault_binary_negedge",
            "ap_sticky_until_reset",
            "PASS M1801 registered public fault export"):
        need(token in export_tb, "directed TB missing: " + token)
    lowered = export_body.lower()
    need(not re.search(r"(?im)(?:^|[;{])\s*(?:force|release)\s+", lowered),
         "TB active force/release")
    for token in ("+vcs+initreg", "ignore_x=1", "noassert", "assertoff"):
        need(token not in lowered, "TB forbidden mechanism: " + token)
    need("m1801_m803_fc2_k8_registered_public_fault_8bank_raw4_acc24 candidate" in full_body,
         "full legal workload does not instantiate successor")
    for token in ("run_header_attack(8)", "run_response_attack(8)",
                  "run_raw_attack_k8()",
                  "M1801 K8 illegal header accepted",
                  "M1801 K8 illegal raw packet accepted",
                  "M1801 K8 spurious response accepted",
                  "run_clean_pair(1,0,51,53)", "numeric_mismatch_count!=0",
                  "tuple_mismatch_count!=0", "weight_mismatch_count!=0",
                  "PASS M1801"):
        need(token in full_tb, "full TB missing: " + token)
    need(export_tb.count("reset_boundary();") == 4,
         "unit reset recovery count")
    reset_fragment = (
        "rst_core = 1'b1;\n        repeat (2) @(posedge clk_core);\n"
        "        #1ps;\n        require_binary_zero(\"reset asserted\");\n"
        "        @(negedge clk_core);\n        rst_core = 1'b0;\n"
        "        #1ps;\n        require_binary_zero(\"reset recovery\");")
    need(export_tb.count(reset_fragment) == 1,
         "multi-cycle reset recovery gate")
    need("||protocol_attack_count!=5||numeric_mismatch_count!=0" in full_tb,
         "full TB five-attack hard gate")
    need(full_tb.count("protocol_attacks=5") == 1
         and "protocol_attacks=4" not in full_tb,
         "PASS token attack count inconsistent")
    attack_end = full_tb.index("run_response_attack(8);run_response_attack(1);")
    clean_start = full_tb.index("run_clean_pair(1,0,51,53)")
    need(attack_end < clean_start, "clean transaction must follow attack/reset campaign")
    for token in (
            "M1801 K8 illegal header accepted",
            "M349 K8 illegal header escaped",
            "M1801 K8 illegal raw packet accepted",
            "M1801 K8 illegal raw packet escaped sticky fault",
            "M1801 K8 spurious response accepted",
            "M349 K8 spurious response escaped"):
        need(full_tb.count(token) == 1,
             "real K8 accept-zero/sticky gate: " + token)


def audit_filelists(export_rows, full_rows):
    export_expected = [
        "+define+SVA_RUNTIME_ENABLED",
        "rtl_m1801/m1801_c2_registered_public_fault_export.sv",
        "tb_m1801/tb_m1801_c2_registered_public_fault_export_directed.sv",
    ]
    need(export_rows == export_expected, "export filelist population/order")
    required = {
        "+define+SVA_RUNTIME_ENABLED",
        "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv",
        "rtl_m1801/m1801_c2_registered_public_fault_export.sv",
        "rtl_m1801/m1801_m803_fc2_k8_registered_public_fault_8bank_raw4_acc24.sv",
        "tb_m1801/tb_m1801_c2_registered_public_fault_k8_vs_k1x8_raw4_acc24.sv",
    }
    need(required.issubset(set(full_rows)), "full filelist required population")
    need("rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv" not in full_rows,
         "frozen combinational M214 selected instead of M1609")
    need(len(full_rows) == len(set(full_rows)), "duplicate full filelist row")


def audit_root_cause():
    predecessor = source_text(PRE_TOP)
    need("if (core_mem_req_accept != adapter_core_mem_req_accept)" in predecessor
         and "if (core_mem_rsp_accept != adapter_core_mem_rsp_accept)" in predecessor,
         "predecessor unconditional accept comparison not found")
    need("assign protocol_error = core_protocol_error || adapter_protocol_error\n        || consistency_fault_q || consistency_fault_now;" in predecessor,
         "predecessor combinational public export not found")
    cone = {
        "core": source_text(CORE),
        "frontend": source_text(FRONTEND),
        "descriptor": source_text(DESCRIPTOR),
        "service": source_text(SERVICE),
        "adapter": source_text(ADAPTER),
    }
    need("assign protocol_error = adapter_fault_q" in cone["core"], "core cone")
    need("|| m202_protocol_error || m204_protocol_error" in cone["frontend"], "frontend cone")
    need("assign protocol_error = fault_q || illegal_request" in cone["descriptor"], "descriptor cone")
    need("assign protocol_error = fault_q || illegal_header || illegal_group" in cone["service"], "service cone")
    need("assign protocol_error = fault_q || illegal_request || illegal_response" in cone["adapter"], "adapter cone")
    log = source_text(M1785_FAIL / "sim.log")
    need("M1785_FIRST_UNKNOWN code=1 class=FAULT field=protocol_error time_ps=26000" in log,
         "M1785 first-public-X fact")
    need("registered_fault_taps=000000" in log
         and "endpoint_fault=00000000" in log,
         "M1785 registered-fault exclusion fact")
    review = strict_json(M1786 / "review.json")
    need(str(review.get("status", "")).startswith("PASS_M1786"),
         "M1786 diagnostic review authority")


def main():
    need(sha(DOCS359) == DOCS359_SHA, "docs/359 drift")
    for path in (RTL_EXPORT, RTL_TOP, TB_EXPORT, TB_FULL, FL_EXPORT, FL_FULL,
                 CONTRACT, PRE_TOP, CORE, FRONTEND, DESCRIPTOR, SERVICE,
                 ADAPTER):
        need(path.is_file() and not path.is_symlink(), "missing/nonregular: " + str(path))
    need(M1785_FAIL.is_dir() and M1786.is_dir() and M1797.is_dir(),
         "diagnostic/review evidence missing")
    audit_root_cause()
    m1797 = strict_json(M1797 / "review.json")
    need(str(m1797.get("status", "")).startswith("FAIL_CLOSED_M1797")
         and m1797["severity_counts"]["p1"] == 1,
         "M1797 fail-closed authority")
    audit_export(source_text(RTL_EXPORT))
    audit_top(source_text(RTL_TOP))
    audit_tb(source_text(TB_EXPORT), source_text(TB_FULL))
    audit_filelists([row.strip() for row in source_text(FL_EXPORT).splitlines() if row.strip()],
                    [row.strip() for row in source_text(FL_FULL).splitlines() if row.strip()])
    contract = strict_json(CONTRACT)
    need(contract["status"] == "SOURCE_ONLY_PENDING_DIFFERENT_AUTHOR_M1802_HAMMER",
         "contract status")
    evidence = contract["evidence_successor"]
    need(evidence["core_owner_enable_exact"] ==
         "header_valid||raw_valid||core_busy||core_mem_req_valid||core_mem_rsp_valid||result_valid||token_done_valid"
         and evidence["adapter_owner_enable_exact"] ==
         "adapter_busy||core_mem_req_valid||(|mem_rsp_valid)"
         and evidence["each_owner_term_mutation_rejected"] is True
         and evidence["complete_enable_constant_zero_mutations_rejected"] is True
         and evidence["request_and_response_valid_gate_mutations_rejected"] is True
         and evidence["reset_recovery_mutations_rejected"] is True
         and evidence["full_tb_protocol_attack_hard_gate"] == 5
         and evidence["full_tb_protocol_attack_pass_token"] == 5
         and evidence["mutations_rejected_per_python_runtime"] == 42,
         "contract M1797 evidence closure")
    need(contract["execution"]["eda_runs"] == 0
         and contract["execution"]["attempts_created"] == 0
         and contract["execution"]["authorized_now"] is False,
         "source-only execution boundary")
    need(contract["future_closure"]["must_rerun_vcs"] is True
         and contract["future_closure"]["must_resynthesize_k8"] is True
         and contract["future_closure"]["must_resynthesize_k1x8"] is True
         and contract["future_closure"]["old_ppa_energy_invalidated"] is True,
         "future fairness closure")
    attempt = HW / "results/.m1801_c2_registered_public_fault_vcs_attempt_consumed"
    result = HW / "results/m1801_c2_registered_public_fault_vcs_r1_20260902"
    need(not os.path.lexists(str(attempt)) and not os.path.lexists(str(result)),
         "M1801 execution namespace consumed")
    output = {
        "schema": "m1801_c2_registered_public_fault_source_check_r1_v1",
        "status": "PASS_M1801_SOURCE_ONLY_NO_EDA_NO_ATTEMPT",
        "docs359_sha256": DOCS359_SHA,
        "root_cause": "unqualified current-cycle fault cone on public protocol_error; unconditional accept mismatch permits invalid payload X",
        "public_fault_registered_only": True,
        "invalid_payload_valid0_isolated": True,
        "fault_sources_enumerated": ["core_current_or_sticky", "adapter_current_or_sticky", "request_accept_mismatch", "response_accept_mismatch"],
        "functional_paths_changed": False,
        "owner_enable_equations_exact": True,
        "full_tb_protocol_attacks_hard_and_printed": 5,
        "eda_runs": 0,
        "attempts_created": 0,
        "paper_citable": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return output


if __name__ == "__main__":
    main()
