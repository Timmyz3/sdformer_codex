#!/usr/bin/env python3
"""Independent, simulator-free QA for the M1578 diagnostic source."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import re
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source_contract_r1_20260901.json"
CONTRACT_INNER = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
FILELIST = HW / "dc_handoff/filelists/date_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.f"
RUNNER = HW / "dc_handoff/scripts/run_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.py"
TB = HW / "dc_handoff/tb/tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv"
AUTHOR_TEST = HW / "system_simulator/tests/test_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.py"
RTL = HW / "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv"
MAPPED = HW / ("dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829/"
               "k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v")
MEMORY = HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv"
AUTHOR = HW / "reviews/m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source_author_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    CONTRACT: "29fdfd080ade36ca373a8e716771ebc896c156ffadea0590c7c3b00c3c616d2d",
    CONTRACT_INNER: "24a17e1502c1bb430e81a86f9d638daaf3db9375a83f2150c797f8e2208753f6",
    FILELIST: "09166d29aedc0a03266f9726ec006ac96efdd396c5290edb423ae303ad2548f1",
    RUNNER: "4c2dcca813329b4f1aaac906b3e198720961a8de8276754986b1ca1c9bc405b6",
    TB: "1c4659304c63b84cb9be443dbec33c71c61a92db092fed55718c0453d7099308",
    AUTHOR_TEST: "ef25d2a005c584ba0c0e1440635f8ac72ab133b67ea5a82286833dcc6b7de9c0",
    RTL: "3328e52d8cf1eec6098ebb7b0525ac55cd8bd6b2fe5b5e504b337d1a678e3c4b",
    MAPPED: "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
    MEMORY: "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
    AUTHOR / "review.json": "f8a205ff18d959133371df83f739c9650ae77e5a3699f954b68950ee03e304ff",
    AUTHOR / "SHA256SUMS": "5a1833d4c1ff81f3e5eb6a68a2885ff8bc1680c22995d92973ee29b5302a7642",
    AUTHOR / "SHA256SUMS.seal.sha256": "e73c3d67d0ca971b2abb5aed0729d556fdc195080b618f79d03ee2d7f8d23a23",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_FILELIST = (
    "/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v",
    "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv",
    "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
    "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
    "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
    "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
    "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
    "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
    "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv",
    "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
    "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829/k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v",
    "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv",
    "dc_handoff/tb/tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv",
)
TOP = "tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault"
RTL_MODULE = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
MAPPED_MODULE = RTL_MODULE + "_ARCH_MODE1"


class QAError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise QAError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          QAError("nonfinite JSON: " + token)))


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def verify_contract(value):
    require(value["schema"] ==
            "m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source_contract_r1_v1",
            "contract schema")
    require(value["execution"] == {
        "vcs_compiles": 0, "simv_runs": 0, "ucli": False,
        "initreg": False, "saif": False, "ptpx": False,
        "attempt_consumed": False}, "pre-review execution must remain zero")
    future = value["future_execution"]
    require(future["authorized_now"] is False and
            future["different_author_hammer_required"] is True,
            "independent gate must precede authorization")
    require(future["budget"] == {
        "vcs_compiles": 1, "simv_runs": 1, "cases": ["k8_case0"],
        "saif": 0, "ptpx": 0}, "future budget drift")
    require(value["claim_boundary"] == {
        "diagnostic_only": True, "paper_citable": False,
        "rtl_pass": False, "mapped_pass": False,
        "timing_verified": False, "power": False, "ppa": False,
        "system_speedup": False, "headline": False},
        "claim boundary drift")


def instance_block(active, marker):
    start = active.find(marker)
    require(start >= 0, "instance marker missing: " + marker)
    end = active.find(");", start)
    require(end > start, "instance terminator missing: " + marker)
    return active[start:end + 2]


def verify_tb(text):
    active = strip_comments(text)
    require(active.count("module " + TOP + ";") == 1,
            "exact diagnostic top missing")
    require(active.count(") rtl_dut (") == 1 and
            active.count("mapped_dut (") == 1,
            "dual DUT population drift")
    require(RTL_MODULE + " #(" in active and ".ARCH_MODE(1)" in active,
            "RTL ARCH_MODE=1 binding drift")
    require(MAPPED_MODULE in active, "mapped ARCH_MODE1 binding drift")
    require(active.count("m1578_case0_memory_fabric rtl_memory") == 1 and
            active.count("m1578_case0_memory_fabric mapped_memory") == 1,
            "two memory fabrics required")
    rtl_memory = instance_block(active, "m1578_case0_memory_fabric rtl_memory")
    mapped_memory = instance_block(active, "m1578_case0_memory_fabric mapped_memory")
    require("mapped_" not in rtl_memory and "rtl_" not in mapped_memory,
            "memory endpoint cross-connection")
    for token in ("rtl_mem_req_valid", "rtl_mem_rsp_valid", "rtl_endpoint_fault"):
        require(token in rtl_memory, "RTL memory endpoint incomplete")
    for token in ("mapped_mem_req_valid", "mapped_mem_rsp_valid", "mapped_endpoint_fault"):
        require(token in mapped_memory, "mapped memory endpoint incomplete")

    require('value === 1\'b0' in active and 'value === 1\'b1' in active and
            'else tri = "X"' in active and 'else tri = "0"' not in active,
            "four-state scalar rendering drift")
    require('$isunknown(value)' in active and 'event8 = "X"' in active,
            "four-state vector rendering drift")
    require("!== {mapped_header_accept" in active and
            "rtl_protocol_error !== 1'b0" in active and
            "mapped_protocol_error !== 1'b0" in active and
            "control_unknown_now = $isunknown" in active,
            "X-aware compare/fault gate drift")
    require("first_difference_cycle = -1" in active and
            "first_fault_cycle = -1" in active and
            "first_difference_cycle = cycle_ordinal" in active and
            "first_fault_cycle = cycle_ordinal" in active,
            "first-cycle capture drift")
    require(active.index("trace_edge();") < active.index("if (difference_now"),
            "event trace must precede stop decision")
    trace_fields = ("header=%s/%s", "source=%s/%s", "endpoint=%s/%s",
                    "mem=%s/%s", "commit=%s/%s", "done=%s/%s",
                    "top_pns=%s%s%s/%s%s%s", "endpoint_fault=%b/%b",
                    "taps_csfamS=%b/%b")
    require(all(field in active for field in trace_fields),
            "event/fault trace field missing")
    require("M1578_FIRST_STOP" in active and
            "first_difference_cycle=%0d" in active and
            "first_fault_cycle=%0d" in active and
            "rtl_taps=%b mapped_taps=%b" in active,
            "first-stop record incomplete")
    require(active.index('print_stop("FAULT_OR_X")') <
            active.index('print_stop("FIRST_RTL_MAPPED_DIFFERENCE")') <
            active.index('print_stop("BOTH_CLEAN_TO_DONE")'),
            "stop priority drift")
    require("header_tag = 24'h979000" in active and
            "header_raw_beat_count = 6'd4" in active and
            "header_window_depth = 4'd2" in active and
            "header_output_blocks = 4'd1" in active,
            "M979 K8 case0 identity drift")
    require("rtl_internal_fault_taps" in active and
            "mapped_internal_fault_taps" in active and
            active.count("fault_q,") >= 8,
            "named internal taps drift")
    lower = active.lower()
    for token in ("ucli", "initreg", "saif", "ptpx"):
        require(token not in lower, "prohibited active token: " + token)
    require(re.search(r"\bforce\b", active) is None and
            re.search(r"\brelease\b", active) is None and
            "$stop" not in active and "assert property" not in active,
            "prohibited runtime control in TB")
    return active


def verify_filelist(rows, tb_text, mapped_text, rtl_text):
    require(tuple(rows) == EXPECTED_FILELIST, "ordered filelist drift")
    require(rows[-1].endswith("tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv"),
            "diagnostic top source must be final filelist entry")
    require(rows[-3].endswith("m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"),
            "mapped netlist filelist binding drift")
    require(mapped_text.count("module " + MAPPED_MODULE + " ") == 1,
            "mapped ARCH_MODE1 module definition drift")
    require(rtl_text.count("module " + RTL_MODULE + " #(") == 1,
            "RTL wrapper module definition drift")
    require(("module " + TOP + ";") in strip_comments(tb_text),
            "top name/source binding drift")


def rejected(function):
    try:
        function()
    except (QAError, KeyError, TypeError, ValueError):
        return True
    return False


def main(output):
    for path, expected in EXPECTED.items():
        metadata = path.lstat()
        require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
                "nonregular identity: " + str(path))
        require(sha256(path) == expected, "identity drift: " + str(path))
    require(CONTRACT_INNER.read_text(encoding="ascii").split() ==
            [EXPECTED[CONTRACT], CONTRACT.name], "contract inner seal drift")
    require(CONTRACT_OUTER.read_text(encoding="ascii").split() ==
            [EXPECTED[CONTRACT_INNER], CONTRACT_INNER.name],
            "contract outer seal drift")
    require((AUTHOR / "SHA256SUMS.seal.sha256").read_text(encoding="ascii").split() ==
            [EXPECTED[AUTHOR / "SHA256SUMS"], "SHA256SUMS"],
            "author outer seal drift")
    for line in (AUTHOR / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        require(sha256(AUTHOR / name.strip()) == digest,
                "author member seal drift")

    contract_text = CONTRACT.read_text(encoding="utf-8")
    contract = strict_loads(contract_text)
    verify_contract(contract)
    tb_text = TB.read_text(encoding="utf-8")
    active = verify_tb(tb_text)
    rows = [row for row in FILELIST.read_text(encoding="utf-8").splitlines()
            if row.strip()]
    mapped_text = MAPPED.read_text(encoding="utf-8", errors="strict")
    rtl_text = RTL.read_text(encoding="utf-8")
    verify_filelist(rows, tb_text, mapped_text, rtl_text)
    runner_text = RUNNER.read_text(encoding="utf-8")
    for token in ("subprocess", "os.system", "Popen(", "execv(",
                  "vcs -", "./simv", "--run"):
        require(token not in runner_text, "runner execution primitive: " + token)

    mapped_taps = (
        "g_k8_implementation_core_frontend_compactor_fault_q",
        "g_k8_implementation_core_frontend_paired_sink_fault_q",
        "g_k8_implementation_core_adapter_fault_q",
        "g_k8_implementation_core_g_k8_service_fault_q",
        "g_k8_implementation_memory_adapter_fault_q",
        "g_k8_implementation_memory_adapter_stale_q",
    )
    require(all(tap in mapped_text and ("mapped_dut." + tap) in active
                for tap in mapped_taps), "mapped retained-tap binding drift")

    mutations = []

    def mutation(name, function):
        mutations.append({"name": name, "rejected": rejected(function)})

    mutation("top_module_removed", lambda: verify_tb(
        tb_text.replace("module " + TOP + ";", "module removed_top;")))
    mutation("rtl_dut_removed", lambda: verify_tb(
        tb_text.replace(") rtl_dut (", ") removed_rtl_dut (")))
    mutation("mapped_dut_removed", lambda: verify_tb(
        tb_text.replace("mapped_dut (", "removed_netlist_instance (")))
    mutation("mapped_memory_aliased_to_rtl_name", lambda: verify_tb(
        tb_text.replace("m1578_case0_memory_fabric mapped_memory",
                        "m1578_case0_memory_fabric rtl_memory")))
    mutation("mapped_memory_crosswired_to_rtl_request", lambda: verify_tb(
        tb_text.replace(".mem_req_valid(mapped_mem_req_valid)",
                        ".mem_req_valid(rtl_mem_req_valid)")))
    mutation("scalar_x_folded_to_zero", lambda: verify_tb(
        tb_text.replace('else tri = "X"', 'else tri = "0"')))
    mutation("vector_x_marker_removed", lambda: verify_tb(
        tb_text.replace('event8 = "X"', 'event8 = "0"')))
    mutation("case_inequality_removed", lambda: verify_tb(
        tb_text.replace("!== {mapped_header_accept", "== {mapped_header_accept")))
    mutation("unknown_control_gate_removed", lambda: verify_tb(
        tb_text.replace("control_unknown_now = $isunknown",
                        "control_unknown_removed = $isunknown")))
    mutation("first_difference_cycle_removed", lambda: verify_tb(
        tb_text.replace("first_difference_cycle = cycle_ordinal",
                        "removed_difference_cycle = cycle_ordinal")))
    mutation("first_fault_cycle_removed", lambda: verify_tb(
        tb_text.replace("first_fault_cycle = cycle_ordinal",
                        "removed_fault_cycle = cycle_ordinal")))
    mutation("memory_event_field_removed", lambda: verify_tb(
        tb_text.replace("mem=%s/%s", "memory_removed=%s/%s")))
    mutation("pre_review_sim_count_promoted", lambda: verify_contract(
        dict(copy.deepcopy(contract), execution=dict(contract["execution"], simv_runs=1))))
    promoted = copy.deepcopy(contract)
    promoted["future_execution"]["authorized_now"] = True
    mutation("pre_review_authorized_now_promoted", lambda: verify_contract(promoted))
    expanded = copy.deepcopy(contract)
    expanded["future_execution"]["budget"]["vcs_compiles"] = 2
    mutation("future_compile_budget_expanded", lambda: verify_contract(expanded))
    duplicate = contract_text.rstrip()[:-1] + ',"schema":"duplicate"}'
    mutation("duplicate_contract_key", lambda: strict_loads(duplicate))
    failed_mutations = [row["name"] for row in mutations if not row["rejected"]]
    require(len(mutations) == 16 and not failed_mutations,
            "robustness mutation gate failed: " + repr(failed_mutations))

    result = {
        "schema": "m1580_m1578_c2_rtl_mapped_case0_source_independent_qa_runtime_r1_v1",
        "status": "PASS_M1580_SOURCE_QA__AUTHORIZE_ONE_FUTURE_COMPILE_AND_ONE_CASE0_SIM__NO_TOOL_RUN_IN_REVIEW",
        "runtime": {"implementation": sys.implementation.name,
                    "version": ".".join(str(value) for value in sys.version_info[:3])},
        "static_checks": {
            "ordered_filelist_entries": len(rows),
            "top": TOP,
            "top_source_is_last_filelist_entry": True,
            "rtl_module": RTL_MODULE,
            "rtl_arch_mode": 1,
            "mapped_module": MAPPED_MODULE,
            "mapped_definition_count": 1,
            "dut_instances": 2,
            "memory_fabric_instances": 2,
            "memory_signal_namespaces_independent": True,
            "mapped_retained_taps": len(mapped_taps),
            "top_fault_bits_per_dut": 3,
            "endpoint_fault_bits_per_dut": 8,
            "four_state_reporting": True,
            "first_difference_and_fault_cycles": True,
            "events": ["header", "source", "endpoint_request",
                       "memory_response", "commit", "done"],
            "classification": {
                "reset_or_uninitialized": "X plus control_unknown",
                "protocol": "protocol_error plus endpoint/taps",
                "stale": "stale_response_seen plus memory-adapter stale tap",
                "numeric": "numeric_overflow",
            },
        },
        "robustness_mutations": {"count": len(mutations), "passed": len(mutations),
                                 "rows": mutations},
        "pre_review_execution": contract["execution"],
        "authorization_recommendation": {
            "authorized_after_independent_review": True,
            "vcs_compiles": 1, "simv_runs": 1, "cases": ["k8_case0"],
            "required_top": TOP,
            "required_filelist_sha256": EXPECTED[FILELIST],
            "reuse_m1502_simv": False,
            "ucli": False, "initreg": False, "saif": False, "ptpx": False,
            "force_release": False,
        },
        "claim_boundary": {
            "source_qa_only": True, "vcs_run_by_review": False,
            "simv_run_by_review": False, "eda": False, "rtl_pass": False,
            "mapped_pass": False, "timing": False, "power": False,
            "ppa": False, "speedup": False, "paper_citable": False,
        },
    }
    output = Path(output)
    require(not output.exists(), "output exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"] + " mutations=16/16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    main(arguments.output)
