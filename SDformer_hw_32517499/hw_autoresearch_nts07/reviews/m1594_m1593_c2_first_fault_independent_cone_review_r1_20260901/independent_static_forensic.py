#!/usr/bin/env python3
"""Simulator-free independent forensic review of the consumed M1593 run."""

from __future__ import print_function

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m1593_c2_rtl_mapped_k8_case0_first_fault_r1_20260901"
ATTEMPT = HW / "results/.m1593_c2_rtl_mapped_k8_case0_first_fault_attempt_consumed"
COMPILE = RESULT / "compile.log"
SIM = RESULT / "sim.log"
SIMV = RESULT / "simv"
FILELIST = HW / "dc_handoff/filelists/date_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.f"
TB = HW / "dc_handoff/tb/tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv"
RTL = HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv"
CORE = HW / "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv"
ADAPTER = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SERVICE = HW / "rtl_m218/m218_fc2_tagged_slice_service_island.sv"
MAPPED = HW / ("dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829/"
               "k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v")
MEMORY = HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv"
CELL_LIB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
                "TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
                "tcbn28hpcplusbwp35p140.v")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

TOP = "tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault"
MAPPED_TOP = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_ARCH_MODE1"
EXPECTED = {
    COMPILE: "0a6069565a7e9217ec68da6a73b462f4b519fccffbcd223565f2ae93106c7dc7",
    SIM: "f93d0a110827a34f19429d4e7343272b3dfbefb9174e574fe7b9c59f6d93c566",
    SIMV: "096f3eb739fcbbdf446641c4cb544aa3b3f6ae93dc3fac3f33d52d3082cdc71f",
    FILELIST: "09166d29aedc0a03266f9726ec006ac96efdd396c5290edb423ae303ad2548f1",
    TB: "4a2ef4c40037274aadd936db8dbe38258aa39fa14a7e0322741f92acd958c435",
    RTL: "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
    MAPPED: "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
    MEMORY: "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


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


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def normalize_command(lines, index):
    row = lines[index]
    while row.rstrip().endswith("\\"):
        index += 1
        require(index < len(lines), "unterminated command continuation")
        row = row.rstrip()[:-1] + " " + lines[index].strip()
    return " ".join(row.split())


def parse_compile_log(text, filelist_rows):
    lines = text.splitlines()
    command_rows = [i for i, row in enumerate(lines)
                    if row.startswith("Command: vcs ")]
    require(len(command_rows) == 1, "compile must contain exactly one executed VCS command")
    command = normalize_command(lines, command_rows[0])
    require(("-top " + TOP) in command, "frozen top missing from executed command")
    require("-f dc_handoff/filelists/" + FILELIST.name in command,
            "frozen filelist missing from executed command")
    require("Top Level Modules:\n       " + TOP in text, "elaborated top mismatch")
    lower = command.lower()
    for token in ("ucli", "initreg", "saif", "ptpx"):
        require(token not in lower, "prohibited executed option: " + token)
    require(re.search(r"(?:^|\s)-ucli(?:\s|$)", lower) is None,
            "interactive UCLI command forbidden")
    parsed = re.findall(r"^Parsing design file '([^']+)'", text, flags=re.M)
    require(parsed == filelist_rows, "compile parse order differs from frozen filelist")
    require(len(re.findall(r"^Warning-\[", text, flags=re.M)) == 4,
            "unexpected compile warning population")
    require(len(re.findall(r"^Warning-\[TFIPC\]", text, flags=re.M)) == 3,
            "expected exactly three TFIPC diagnostics")
    require(re.search(r"^Error-\[", text, flags=re.M) is None,
            "compile log contains a compiler error")
    require("71 modules and 3 UDPs read." in text, "compile population drift")
    tfipc = re.findall(r'"(HA1D0BWP35P140[^\n]+)"', text)
    require(len(tfipc) == 3, "TFIPC instance extraction failed")
    require(all(".A (" in row and ".B (" in row and ".S (" in row
                and ".CO (" not in row for row in tfipc),
            "TFIPC is not limited to unused half-adder carry outputs")
    return {"command": command, "parsed_files": len(parsed),
            "warnings": 4, "tfipc_unused_co": 3}


TRACE_RE = re.compile(
    r"^M1578_TRACE cycle=(\d+) header=([^ ]+) source=([^ ]+) endpoint=([^ ]+) "
    r"mem=([^ ]+) commit=([^ ]+) done=([^ ]+) top_pns=([^ ]+) "
    r"endpoint_fault=([^ ]+) taps_csfamS=([^ ]+)$")


def parse_sim_log(text):
    lines = text.splitlines()
    command_rows = [i for i, row in enumerate(lines)
                    if row.startswith("Command: ") and "/simv " in row]
    require(len(command_rows) == 1, "simulation must contain exactly one executed simv command")
    command = normalize_command(lines, command_rows[0])
    lower = command.lower()
    for token in ("ucli", "initreg", "saif", "ptpx"):
        require(token not in lower, "prohibited simulation option: " + token)
    traces = []
    for row in lines:
        match = TRACE_RE.match(row)
        if match:
            traces.append(match.groups())
    require([int(row[0]) for row in traces] == [1, 2, 3, 4, 5, 6],
            "trace must stop on the first six sampled edges")
    require(all(row[7] == "000/000" and row[8] == "00000000/00000000"
                and row[9] == "000000/000000" for row in traces[:5]),
            "pre-fault trace is not clean")
    cycle6 = traces[5]
    require(cycle6[1:7] == ("0/0", "0/0", "0/0", "0/0", "0/0", "0/0"),
            "cycle-6 event interface is not idle/equal")
    require(cycle6[7] == "000/X00", "cycle-6 first X is not mapped protocol_error")
    require(cycle6[8] == "00000000/00000000", "endpoint fault differs at cycle 6")
    require(cycle6[9] == "000000/000000", "named registered taps differ at cycle 6")
    stop_rows = [row for row in lines if row.startswith("M1578_FIRST_STOP ")]
    require(len(stop_rows) == 1, "expected exactly one first-stop receipt")
    stop = stop_rows[0]
    for token in ("reason=                      FAULT_OR_X", "cycle=6",
                  "first_difference_cycle=6", "first_fault_cycle=6",
                  "rtl_top_pns=000", "mapped_top_pns=X00",
                  "rtl_endpoint_fault=00000000", "mapped_endpoint_fault=00000000",
                  "rtl_taps=000000", "mapped_taps=000000"):
        require(token in stop, "first-stop field drift: " + token)
    require("$finish at simulation time                28500" in text,
            "first-stop timestamp drift")
    require("PASS" not in text and "M1578_FIRST_STOP" in text,
            "failed diagnostic must not be labeled PASS")
    return {"command": command, "trace_edges": 6, "first_fault_cycle": 6,
            "rtl_top_pns": "000", "mapped_top_pns": "X00",
            "rtl_endpoint_fault": "00000000", "mapped_endpoint_fault": "00000000",
            "rtl_taps": "000000", "mapped_taps": "000000"}


def checker_analysis(text):
    active = strip_comments(text)
    start = active.index("always @(posedge clk_core) begin")
    end = active.index("task automatic initialize_inputs", start)
    block = active[start:end]
    require("cycle_ordinal = cycle_ordinal + 1;\n            trace_edge();" in block,
            "checker no longer samples immediately after edge entry")
    prefix = block[:block.index("trace_edge();")]
    require("#1ps" not in prefix and "#1step" not in prefix
            and "clocking" not in prefix,
            "post-edge settling unexpectedly present")
    require("always_comb begin" in active and "mapped_fault_now" in active,
            "four-state checker structure missing")
    require("mapped_protocol_error !== 1'b0" in active,
            "mapped X must remain fail closed")
    require("mapped_dut.g_k8_implementation_core_frontend_compactor_fault_q" in active,
            "named mapped registered taps missing")
    return {"sample_event": "posedge_active_region",
            "post_edge_settle": False,
            "x_fail_closed": True,
            "named_taps_are_registered_fault_state": True}


def rtl_cone_analysis():
    rtl = strip_comments(RTL.read_text(encoding="utf-8"))
    core = strip_comments(CORE.read_text(encoding="utf-8"))
    adapter = strip_comments(ADAPTER.read_text(encoding="utf-8"))
    service = strip_comments(SERVICE.read_text(encoding="utf-8"))
    require("assign protocol_error = core_protocol_error || adapter_protocol_error\n        || consistency_fault_q || consistency_fault_now;" in rtl,
            "top RTL protocol cone drift")
    require("if (rst_core) consistency_fault_q <= 0;" in rtl,
            "consistency fault register lacks explicit reset")
    require("core_mem_req_accept != adapter_core_mem_req_accept" in rtl and
            "core_mem_rsp_accept != adapter_core_mem_rsp_accept" in rtl,
            "combinational consistency comparisons missing")
    require("assign protocol_error = adapter_fault_q" in core and
            "header_valid && !integration_header_legal" in core,
            "core protocol cone is not registered-plus-combinational")
    require("assign protocol_error = fault_q || illegal_request || illegal_response;" in adapter,
            "memory adapter protocol cone drift")
    require("assign protocol_error = fault_q || illegal_header || illegal_group" in service,
            "service protocol cone drift")
    return {"top_protocol_is_combinational_or": True,
            "registered_consistency_fault_reset": True,
            "unregistered_consistency_compare_present": True,
            "nested_protocol_outputs_include_combinational_illegal_terms": True}


def module_body(text, module):
    match = re.search(r"\bmodule\s+" + re.escape(module) + r"\b.*?\bendmodule\b",
                      text, flags=re.S)
    require(match is not None, "module body missing: " + module)
    return match.group(0)


def library_ports(text):
    result = {}
    for match in re.finditer(r"\bmodule\s+(\w+)\s*\([^;]*?\);(.*?)\bendmodule\b",
                             text, flags=re.S):
        name, body = match.groups()
        inputs, outputs = set(), set()
        for direction, names in re.findall(r"\b(input|output)\b\s*(?:reg\s*)?(?:\[[^]]+\]\s*)?([^;]+);",
                                           body):
            parsed = {row.strip() for row in names.split(",")
                      if re.match(r"^\w+$", row.strip())}
            (inputs if direction == "input" else outputs).update(parsed)
        if outputs:
            result[name] = (inputs, outputs)
    return result


SIGNAL_RE = re.compile(r"(?:\\[^\s,()]+|[A-Za-z_$][\w$]*(?:\[[^]]+\])?)")


def expression_nets(expr):
    values = []
    for token in SIGNAL_RE.findall(expr):
        if token in ("x", "z") or token.startswith("1'b"):
            continue
        if re.match(r"^\d+'[bdhoBDHO]", token):
            continue
        values.append(token)
    return values


def mapped_cone_analysis():
    mapped_text = MAPPED.read_text(encoding="utf-8", errors="strict")
    active = strip_comments(module_body(mapped_text, MAPPED_TOP))
    lib = library_ports(CELL_LIB.read_text(encoding="utf-8", errors="strict"))
    require("ND3D1BWP35P140 U160335" in active and
            ".ZN(\n        protocol_error)" in active,
            "mapped protocol_error direct driver drift")

    input_bases = set()
    for names in re.findall(r"\binput\b\s*(?:\[[^]]+\]\s*)?([^;]+);", active):
        input_bases.update(row.strip() for row in names.split(","))
    drivers = {}
    instances = {}
    for match in re.finditer(r"^\s*(\w+)\s+(\w+)\s*\((.*?)\);", active,
                             flags=re.M | re.S):
        cell, inst, body = match.groups()
        if cell not in lib:
            continue
        pins = dict(re.findall(r"\.(\w+)\s*\((.*?)\)(?:\s*,|\s*$)", body,
                               flags=re.S))
        inputs, outputs = lib[cell]
        instances[inst] = (cell, pins, inputs, outputs)
        is_state = "CP" in inputs and ("Q" in outputs or "QN" in outputs)
        for pin in outputs:
            nets = expression_nets(pins.get(pin, ""))
            if len(nets) == 1:
                require(nets[0] not in drivers, "multiply driven mapped net: " + nets[0])
                drivers[nets[0]] = (inst, pin, is_state)

    comb_seen, state_seen, pi_seen, undriven = set(), set(), set(), set()

    def is_pi(net):
        return net.split("[")[0] in input_bases

    def walk(net):
        if is_pi(net):
            pi_seen.add(net)
            return
        item = drivers.get(net)
        if item is None:
            undriven.add(net)
            return
        inst, _pin, is_state = item
        if is_state:
            state_seen.add(inst)
            return
        if inst in comb_seen:
            return
        comb_seen.add(inst)
        _cell, pins, inputs, _outputs = instances[inst]
        for pin in inputs:
            for child in expression_nets(pins.get(pin, "")):
                walk(child)

    walk("protocol_error")
    require(not undriven, "undriven mapped protocol cone leaves: " + repr(sorted(undriven)[:8]))

    reset_memo = {}
    visiting = set()

    def reaches_reset(net):
        if net == "rst_core":
            return True
        if is_pi(net):
            return False
        if net in reset_memo:
            return reset_memo[net]
        if net in visiting:
            return False
        visiting.add(net)
        item = drivers.get(net)
        if item is None:
            value = False
        else:
            inst, _pin, is_state = item
            if is_state:
                value = False
            else:
                _cell, pins, inputs, _outputs = instances[inst]
                value = any(reaches_reset(child) for pin in inputs
                            for child in expression_nets(pins.get(pin, "")))
        visiting.remove(net)
        reset_memo[net] = value
        return value

    reset_dependent = 0
    for inst in state_seen:
        _cell, pins, _inputs, _outputs = instances[inst]
        if any(reaches_reset(net) for net in expression_nets(pins.get("D", ""))):
            reset_dependent += 1

    return {"direct_driver": "ND3D1BWP35P140/U160335",
            "combinational_instances_in_cone": len(comb_seen),
            "state_instances_in_cone": len(state_seen),
            "primary_input_bits_in_cone": len(pi_seen),
            "undriven_leaves": 0,
            "state_d_cones_with_structural_rst_dependency": reset_dependent,
            "state_d_cones_total": len(state_seen),
            "structural_reset_dependency_is_not_reset_dominance_proof": True}


def main(output):
    for path, expected in EXPECTED.items():
        metadata = path.lstat()
        require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
                "nonregular frozen input: " + str(path))
        require(sha256(path) == expected, "frozen input drift: " + str(path))
    require(ATTEMPT.is_dir() and not ATTEMPT.is_symlink(),
            "consumed-attempt marker missing or unsafe")
    require(not any(ATTEMPT.iterdir()), "consumed-attempt marker must be empty")

    filelist_rows = [row.strip() for row in FILELIST.read_text(encoding="utf-8").splitlines()
                     if row.strip()]
    require(len(filelist_rows) == 16 and filelist_rows[0] == str(CELL_LIB),
            "frozen filelist population drift")
    compile_result = parse_compile_log(COMPILE.read_text(encoding="utf-8", errors="strict"),
                                       filelist_rows)
    sim_result = parse_sim_log(SIM.read_text(encoding="utf-8", errors="strict"))
    checker = checker_analysis(TB.read_text(encoding="utf-8", errors="strict"))
    rtl_cone = rtl_cone_analysis()
    mapped_cone = mapped_cone_analysis()

    result = {
        "schema": "m1594_m1593_c2_first_fault_independent_static_forensic_r1_v1",
        "status": "PASS_INDEPENDENT_FORENSIC",
        "execution_by_m1594": {"vcs": 0, "simv": 0, "dc": 0, "ptpx": 0},
        "m1593_execution": {"vcs_compiles": 1, "simv_runs": 1,
                            "attempt_consumed": True},
        "frozen_sha256": {str(path.relative_to(ROOT)): expected
                          for path, expected in EXPECTED.items()},
        "compile": compile_result,
        "simulation": sim_result,
        "checker": checker,
        "rtl_protocol_cone": rtl_cone,
        "mapped_protocol_cone": mapped_cone,
        "ruling": {
            "proven_observation": "mapped combinational protocol_error is X at the immediate cycle-6 active-region sample",
            "proven_not_faulted": ["RTL protocol/numeric/stale", "both endpoint fault vectors",
                                   "all six registered fault/stale taps in both DUTs"],
            "primary_classification": "CHECKER_ACTIVE_REGION_SAMPLING_DEFECT_WITH_COMBINATIONAL_X_OBSERVATION",
            "stable_reset_or_invalid_state_isolation": "UNRESOLVED_SECONDARY_RISK",
            "minimum_repair": "add exactly one 1ps/one-timeprecision post-posedge settle before trace and all stop decisions",
            "resynthesis_required_for_minimum_repair": False,
            "if_x_persists_after_settle": "repair RTL validity/reset isolation and rerun DC before any mapped rerun",
            "initreg_force_masking_authorized": False,
            "m1593_claim": "FAILED_DO_NOT_CITE_DO_NOT_RETRY",
            "paper_citable": False,
        },
    }
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                       sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception as exc:  # fail closed with one diagnostic
        print("M1594_FAIL: " + str(exc), file=__import__("sys").stderr)
        raise
