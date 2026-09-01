#!/usr/bin/env python3
"""Fail-closed source/result checker for M1684; never launches EDA."""
from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
BASE = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
DESIGN = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
NET_REL = "netlist/" + DESIGN + "_mapped.v"
SDC_REL = "netlist/" + DESIGN + "_mapped.sdc"
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
TT_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")
SS_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
MEM = HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv"
CASE_TB = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
OLD_ASSERT = HW / "dc_handoff/tb/m1334_c2_production_activity_assertions.sv"
ASSERT = HW / "dc_handoff/tb/m1684_c2_m1609_production_binary_fault_assertions.sv"
TOP_TB = HW / "dc_handoff/tb/tb_m1684_c2_m1609_fresh_mapped_production_energy.sv"
UCLI = HW / "dc_handoff/scripts/m1684_c2_m1609_fresh_mapped_production_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1684_c2_m1609_fresh_mapped_production_energy_tt0p9v25c.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1684_m1661_c2_m1609_fresh_mapped_production_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1684_c2_m1609_fresh_mapped_production_energy_source.py"
CONTRACT = HW / "contracts/m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_contract_r1_20260901.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1609 = HW / "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
M1677 = HW / "reviews/m1677_m1661_m1652_c2_resource_gate_successor_three_axis_dc_result_hammer_r1_20260901"
M1627 = HW / "reviews/m1627_m1613_c2_registered_fault_directed_vcs_result_independent_hammer_r1_20260901"
M1568 = HW / "reviews/m1568_m1502_c2_mapped_first_fault_forensic_r1_20260901"
M1502_FAILURE = HW / "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine"
M1685 = HW / "reviews/m1685_m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_hammer_r1_20260901"
M1686 = HW / "contracts/m1686_m1685_m1684_m1661_c2_m1609_fresh_mapped_production_energy_launch_release_r1_20260901.json"

FILELISTS = {
    "k8": HW / "dc_handoff/filelists/date_m1684_c2_m1609_k8_fresh_mapped_production_energy.f",
    "k1x8": HW / "dc_handoff/filelists/date_m1684_c2_m1609_k1x8_fresh_mapped_production_energy.f",
}
AXES = {
    "k8": {"define": "M979_AXIS_K8", "net_sha": "6c62d99b444ba25f8eb3f1e491479b44f5613b0323e032af8150e81c84f393c4", "sdc_sha": "852c62c1ed8d4a6c69a8fdd17ac7c3b18f0cdee271fb4aaa25fba6a2f77535eb", "cycles": [51, 131, 486, 1231, 14]},
    "k1x8": {"define": "M979_AXIS_K1X8", "net_sha": "5316db453f0ca70524ea18091e0924f79d116afd46d5432906f3182d1ccfd704", "sdc_sha": "17414d50eda57b2ba6f1ff3f376c24d2be6c70e9b625f717202cc72ce53c49f2", "cycles": [53, 133, 499, 1246, 14]},
}
EVENTS = [20, 41, 90, 110, 0]
PACKETS = [1, 2, 4, 8, 1]
TOP = "tb_m1684_c2_m1609_fresh_mapped_production_energy"
SAIF_SCOPE = TOP + ".core.dut"
CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))

FIXED = {
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1609: "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    MEM: "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
    CASE_TB: "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    CELL: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    TT_DB: "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
    SS_DB: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    BASE / "SHA256SUMS": "22388b70b68f4b038a464446704bdc37fb9f51d536fc12b656b0e51045f5efac",
    BASE / "SHA256SUMS.seal.sha256": "f41253a98d74e7b5087c39f49ddbade856ac825f1286c0c73ccf18bdbc6cd4a2",
    M1677 / "review.json": "b05b551375c244746ce10990f2f4ac0757b6e82e3922fc7db8583bd5d1ffc2f5",
    M1677 / "SHA256SUMS": "760dcc9226414e205b8498cbf5a2b051e272f2c64313ac2942e90982f3c0b83d",
    M1677 / "SHA256SUMS.seal.sha256": "8966dd938975e183784c9017e9eaaf59c641b14ce832c30f9735a08f73463708",
    M1627 / "review.json": "ab4f2187667301a37fbd5f523687a8971282e642163d42886edcdc138edc43d4",
    M1627 / "SHA256SUMS": "670edd3dbf60d0d6122fd4ee769c623456f9774da9c0960c9ce2a3291276df51",
    M1627 / "SHA256SUMS.seal.sha256": "7443f9553a22cf9189320cb0f1b9850b839dea16f8bb0d92c94da6659113034c",
    M1568 / "review.json": "b88067a9ef94b24960d9d5ba86973b23c7b10a89386c9c624ffa82d8131081b2",
    M1568 / "SHA256SUMS": "279a60e1aaec03523da21f216ef9bbcc22eaba3daf75feb92a0d4976f2a17d71",
    M1568 / "SHA256SUMS.seal.sha256": "74a8848d6b082ce954d1182a2438ad7f2be6bce7fadda9c8b324feeee0e3bbc8",
    M1502_FAILURE / "failure.json": "2bad717f51fa99e2526b4ec8b7b305b4bbbf60b84728d6f799de59aa72bfe7d2",
    M1502_FAILURE / "SHA256SUMS": "a5f02446e2a687c535b16498d5f3cd5a69bd0c15b5eb8ff43d032103a397081e",
    M1502_FAILURE / "SHA256SUMS.seal.sha256": "82dfce2ce39c59fdfd61f9501d9806bef67d865c1896d247f2a70d381d237129",
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    need(Path(path).is_file() and not Path(path).is_symlink(), "JSON not regular")
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root")
    return value


def active_lines(path):
    lines = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def expected_filelist(axis):
    return [
        "+define+" + AXES[axis]["define"],
        "+define+SVA_RUNTIME_ENABLED",
        str(CELL), str(BASE / axis / NET_REL), str(MEM), str(CASE_TB),
        str(OLD_ASSERT), str(ASSERT), str(TOP_TB),
    ]


def validate_filelist(axis, path=None):
    path = FILELISTS[axis] if path is None else Path(path)
    need(active_lines(path) == expected_filelist(axis),
         axis + " exact filelist/order mismatch")
    need(sha(BASE / axis / NET_REL) == AXES[axis]["net_sha"],
         axis + " fresh mapped netlist drift")
    need(sha(BASE / axis / SDC_REL) == AXES[axis]["sdc_sha"],
         axis + " fresh mapped SDC drift")


def validate_predecessors():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift: " + str(path))
    m1677 = strict_json(M1677 / "review.json")
    need(m1677.get("status") == "PASS100_M1677_M1661_M1652_C2_RESOURCE_GATE_SUCCESSOR_THREE_AXIS_DC_RESULT_ADMITTED",
         "M1677 status drift")
    need(m1677.get("fresh_dc_evidence", {}).get("same_m1609_filelist_tcl_sdc_libraries_clock") is True,
         "M1677 does not bind fresh M1609 synthesis")
    m1627 = strict_json(M1627 / "review.json")
    behavior = m1627.get("directed_behavior", {})
    need(m1627.get("status") == "PASS_M1627_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_RESULT_HAMMER",
         "M1627 status drift")
    need(behavior == {"legal_terminal_no_false_pulse": 1,
                      "legal_descriptor_accepts": 1,
                      "illegal_header_latched": 1,
                      "illegal_raw_latched": 1,
                      "sticky_checks": 3}, "M1627 registered-fault semantics drift")
    m1568 = strict_json(M1568 / "review.json")
    need(m1568.get("sealed_boundary", {}).get("first_fault_ps") == 28500
         and m1568.get("sealed_boundary", {}).get("endpoint_accept_cover") == 0,
         "M1502 first-fault forensic premise drift")
    failure = strict_json(M1502_FAILURE / "failure.json")
    need(failure.get("phase") == "SIM_k8_0"
         and failure.get("counts") == {"ptpx_runs": 0, "saif_files": 0,
                                        "simv_runs": 1, "vcs_compiles": 1}
         and failure.get("automatic_retry") is False,
         "M1502 sealed failure drift")


def validate_sources():
    validate_predecessors()
    for axis in AXES:
        validate_filelist(axis)
    monitor = ASSERT.read_text()
    for token in ("ap_public_fault_binary", "ap_registered_public_fault_zero",
                  "@(negedge clk_core)", "$isunknown({protocol_error",
                  "accepted_sources != expected_sources(case_id)",
                  "registered_fault_public_zero=1"):
        need(token in monitor, "M1684 monitor omits " + token)
    wrapper = TOP_TB.read_text()
    for token in ("tb_m979_c2_three_axis_mapped_gate_case_saif core()",
                  "m1334_c2_production_activity_assertions production_checks",
                  "m1684_c2_m1609_production_binary_fault_assertions fault_checks",
                  "core.g_memory[bank].memory.endpoint_protocol_fault_q"):
        need(token in wrapper, "M1684 wrapper omits " + token)
    expected_ucli = [
        "power -gate_level all mda sv", "power " + SAIF_SCOPE, "run",
        "power -enable", "run", "power -disable",
        "power -report $::env(M1684_SAIF_FILE) 1e-9 " + SAIF_SCOPE, "quit"]
    need(active_lines(UCLI) == expected_ucli, "UCLI scope/order drift")
    pt_text = PT_TCL.read_text()
    for token in ("tt0p9v25c", "ZeroWireload", "read_saif -strip_path",
                  "annotated_nets != $total_nets",
                  "annotated_leaf_cells != $total_leaf_cells",
                  "check_power succeeded", "report_power -unit mW",
                  "PASS_M1684_C2_M1609_FRESH_MAPPED_PRODUCTION_PTPX"):
        need(token in pt_text, "PTPX Tcl omits " + token)
    runner = RUNNER.read_text()
    need("initreg" not in runner.lower(), "runner contains forbidden initreg")
    need(runner.count('for axis in ("k8", "k1x8"):') >= 3,
         "runner axis geometry drift")
    need(runner.count("for case_id in range(5):") >= 2,
         "runner case geometry drift")
    for token in ('state["vcs_compiles"] += 1', 'state["simv_runs"] += 1',
                  'state["saif_files"] += 1', 'state["ptpx_runs"] += 1',
                  '"vcs_compiles": 2', '"simv_runs": 10',
                  '"saif_files": 10', '"ptpx_runs": 10'):
        need(token in runner, "runner count/gate omits " + token)
    need(runner.index("all ten mapped production SAIF gates")
         < runner.index('state["phase"] = "PTPX_"'),
         "PTPX can precede ten-SAIF gate")
    for path in (ASSERT, TOP_TB, UCLI, PT_TCL, RUNNER) + tuple(FILELISTS.values()):
        need("initreg" not in path.read_text().lower(),
             "forbidden initreg in source: " + str(path))
    contract = strict_json(CONTRACT)
    need(contract.get("schema") == "m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_contract_r1_v1",
         "contract schema drift")
    need(contract.get("status") == "SOURCE_ONLY__M1685_REVIEW_AND_M1686_RELEASE_REQUIRED__NO_EDA",
         "contract status drift")
    need(contract.get("claim_boundary") == CLAIMS, "source claim promotion")
    files = contract.get("source_files", [])
    need(isinstance(files, list), "contract source files absent")
    mapping = dict((row.get("path"), row.get("sha256")) for row in files)
    need(len(mapping) == len(files), "duplicate contract source path")
    expected_paths = (RUNNER, CHECKER, TEST, ASSERT, TOP_TB, UCLI, PT_TCL,
                      FILELISTS["k8"], FILELISTS["k1x8"])
    need(set(mapping) == set(path.relative_to(HW).as_posix()
                             for path in expected_paths),
         "contract source path set drift")
    for path in expected_paths:
        need(mapping[path.relative_to(HW).as_posix()] == sha(path),
             "contract source SHA drift: " + str(path))
    for path in (M1685, M1686,
                 HW / "results/.m1684_c2_mapped_production_energy_attempt_consumed",
                 HW / "results/m1684_c2_mapped_production_energy_r1_20260901"):
        need(not os.path.lexists(path), "future/result namespace already exists")
    return {"schema": "m1684_c2_source_check_r1_v1",
            "status": "PASS_M1684_SOURCE_ONLY_NO_EDA",
            "axes": ["k8", "k1x8"], "cases_per_axis": 5,
            "accepted_sources_per_axis": sum(EVENTS),
            "m1502_root_cause_handled_by": "M1609_registered_sticky_fault_plus_fresh_M1661_netlists",
            "claim_boundary": CLAIMS}


def sexpr_tokens(text):
    return re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', text)


def parse_saif(text):
    tokens = sexpr_tokens(text)
    pos = [0]
    def parse_one():
        need(pos[0] < len(tokens) and tokens[pos[0]] == "(", "malformed SAIF")
        pos[0] += 1
        node = []
        while pos[0] < len(tokens) and tokens[pos[0]] != ")":
            if tokens[pos[0]] == "(":
                node.append(parse_one())
            else:
                node.append(tokens[pos[0]])
                pos[0] += 1
        need(pos[0] < len(tokens), "unterminated SAIF")
        pos[0] += 1
        return node
    root = parse_one()
    need(pos[0] == len(tokens) and root and root[0] == "SAIFILE", "SAIF root")
    return root


def forms(node, tag):
    return [item for item in node[1:]
            if isinstance(item, list) and item and item[0] == tag]


def all_forms(node, tag):
    found = []
    if isinstance(node, list):
        if node and node[0] == tag:
            found.append(node)
        for item in node:
            if isinstance(item, list):
                found.extend(all_forms(item, tag))
    return found


def direct_instance(node, name):
    hits = [item for item in forms(node, "INSTANCE")
            if len(item) >= 2 and item[1].lstrip("\\") == name]
    need(len(hits) == 1, "SAIF instance absent/duplicated: " + name)
    return hits[0]


def activity_under(node):
    activity = {}
    def walk(value):
        if not isinstance(value, list):
            return
        tc = forms(value, "TC")
        if value and isinstance(value[0], str) and len(tc) == 1:
            need(len(tc[0]) == 2, "malformed TC")
            name = value[0].lstrip("\\")
            activity[name] = activity.get(name, 0.0) + float(tc[0][1])
        for child in value[1:]:
            walk(child)
    walk(node)
    return activity


def cone(activity, prefixes):
    return sum(value for name, value in activity.items()
               if any(name == prefix or name.startswith(prefix + "[")
                      for prefix in prefixes))


def validate_runtime_log(path, axis, case_id):
    text = Path(path).read_text(errors="strict")
    forbidden = ("Assertion failed", "Fatal:", "$fatal", "Error-[",
                 "contains X/Z", "fault asserted", "coverage incomplete")
    need(not any(token in text for token in forbidden), "runtime fatal/assertion")
    pattern = (r"PASS M1684 M1609 binary-clean production case=" + str(case_id)
               + r" accepted_sources=([0-9]+) source_packets=([0-9]+)"
               + r" endpoint_accepts=([0-9]+) result_accepts=([1-9][0-9]*)"
               + r" done_accepts=1 fault_binary_clean=1 registered_fault_public_zero=1")
    hits = re.findall(pattern, text)
    need(len(hits) == 1, "M1684 runtime PASS absent/duplicated")
    need(int(hits[0][0]) == EVENTS[case_id]
         and int(hits[0][1]) == PACKETS[case_id], "runtime source denominator drift")
    endpoint = int(hits[0][2])
    need((case_id < 4 and endpoint > 0) or (case_id == 4 and endpoint == 0),
         "runtime endpoint activity drift")
    display = "K8" if axis == "k8" else "K1x8"
    old_pass = ("PASS M979 mapped replay axis=" + display
                + " case=" + str(case_id) + " events=" + str(EVENTS[case_id])
                + " cycles=" + str(AXES[axis]["cycles"][case_id]))
    need(text.count(old_pass) == 1, "M979 exact PASS absent/duplicated")
    need(text.count("PASS M1334 coverage case=" + str(case_id)) == 1,
         "M1334 coverage PASS absent/duplicated")
    return {"log_sha256": sha(path), "accepted_sources": EVENTS[case_id],
            "endpoint_accepts": endpoint}


def validate_saif(path, axis, case_id, cycles):
    need(axis in AXES and case_id in range(5), "axis/case")
    need(cycles == AXES[axis]["cycles"][case_id], "cycle anchor")
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "SAIF not regular")
    root = parse_saif(path.read_text(errors="strict"))
    duration = forms(root, "DURATION")
    need(len(duration) == 1 and len(duration[0]) == 2, "SAIF duration")
    duration_ns = float(duration[0][1])
    need(abs(duration_ns - cycles * 3.0) <= 1e-6, "SAIF duration/cycle mismatch")
    tx = all_forms(root, "TX")
    need(tx and all(len(item) == 2 and float(item[1]) == 0.0 for item in tx),
         "SAIF TX unknown activity")
    top = direct_instance(root, TOP)
    core = direct_instance(top, "core")
    dut = direct_instance(core, "dut")
    activity = activity_under(dut)
    need(activity, "DUT SAIF empty")
    for name, prefixes in {
            "clock": ("clk_core",),
            "source": ("raw_valid", "raw_accept", "raw_bitmap"),
            "endpoint": ("mem_req_valid", "mem_req_accept", "mem_rsp_valid", "mem_rsp_accept"),
            "commit": ("result_valid", "result_accept", "result_accumulator"),
            "done": ("token_done_valid", "token_done_accept")}.items():
        value = cone(activity, prefixes)
        if name == "endpoint" and case_id == 4:
            need(value == 0.0, "zero case endpoint toggled")
        else:
            need(value > 0.0, "zero production cone: " + name)
    for fault in ("protocol_error", "numeric_overflow", "stale_response_seen"):
        hits = [name for name in activity
                if name == fault or name.startswith(fault + "[")]
        need(hits and sum(activity[name] for name in hits) == 0.0,
             "fault absent/toggled in SAIF: " + fault)
    need(cone(activity, ("rst_core",)) == 0.0, "reset toggled in measurement")
    return {"status": "PASS_M1684_BINARY_CLEAN_DUT_ONLY_PRODUCTION_SAIF",
            "axis": axis, "case": case_id, "cycles": cycles,
            "accepted_sources": EVENTS[case_id], "duration_ns": duration_ns,
            "saif_sha256": sha(path), "tx_nonzero": 0}


POWER_FIELDS = ("Net Switching Power", "Cell Internal Power",
                "Cell Leakage Power", "Total Power")


def parse_power_report(path):
    text = Path(path).read_text(errors="strict")
    need("Report : Averaged Power" in text and "-unit mW" in text,
         "power report mode/unit")
    values = {}
    for field in POWER_FIELDS:
        hits = re.findall(re.escape(field) + r"\s*=\s*([0-9.eE+-]+)", text)
        need(len(hits) == 1, "power field absent/duplicated: " + field)
        value = float(hits[0])
        need(math.isfinite(value) and value >= 0.0, "invalid power field")
        values[field] = value
    need(values["Total Power"] > 0.0, "nonpositive total power")
    subtotal = (values["Net Switching Power"] + values["Cell Internal Power"]
                + values["Cell Leakage Power"])
    need(abs(subtotal - values["Total Power"])
         <= max(1e-6, values["Total Power"] * 1e-4), "power subtotal mismatch")
    return {"net_switching_mw": values["Net Switching Power"],
            "cell_internal_mw": values["Cell Internal Power"],
            "cell_leakage_mw": values["Cell Leakage Power"],
            "total_mw": values["Total Power"]}


def aggregate_metrics(entries):
    need(len(entries) == 10, "metrics require ten coordinates")
    coordinates = set((entry["axis"], entry["case"]) for entry in entries)
    need(coordinates == set((axis, case_id) for axis in AXES for case_id in range(5)),
         "metrics Cartesian product")
    axes = {}
    for axis in AXES:
        rows = sorted((entry for entry in entries if entry["axis"] == axis),
                      key=lambda item: item["case"])
        need([row["cycles"] for row in rows] == AXES[axis]["cycles"],
             "metrics cycle anchor")
        need([row["accepted_sources"] for row in rows] == EVENTS,
             "metrics accepted-source denominator")
        total_cycles = sum(row["cycles"] for row in rows)
        total_sources = sum(row["accepted_sources"] for row in rows)
        energy_pj = sum(row["total_mw"] * row["cycles"] * 3.0 for row in rows)
        duration_ns = total_cycles * 3.0
        axes[axis] = {
            "cycles": total_cycles, "accepted_sources": total_sources,
            "duration_ns": duration_ns, "energy_pj": energy_pj,
            "cycle_weighted_average_power_mw": energy_pj / duration_ns,
            "energy_pj_per_cycle": energy_pj / total_cycles,
            "energy_pj_per_accepted_source": energy_pj / total_sources,
            "throughput_gsource_per_second_per_watt":
                (total_sources / energy_pj) * 1000.0,
        }
    energy_ratio = axes["k1x8"]["energy_pj"] / axes["k8"]["energy_pj"]
    return {
        "axes": axes,
        "equal_bandwidth_cycle_speedup_k8_vs_k1x8":
            axes["k1x8"]["cycles"] / axes["k8"]["cycles"],
        "equal_bandwidth_energy_ratio_k1x8_over_k8": energy_ratio,
        "equal_bandwidth_k8_energy_saving_fraction": 1.0 - 1.0 / energy_ratio,
        "equal_bandwidth_throughput_per_watt_ratio_k8_over_k1x8": energy_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "saif", "power"), required=True)
    parser.add_argument("--axis", choices=sorted(AXES))
    parser.add_argument("--case", dest="case_id", type=int)
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--saif", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--power-report", type=Path)
    args = parser.parse_args()
    if args.mode == "source":
        output = validate_sources()
    elif args.mode == "saif":
        need(args.axis is not None and args.case_id is not None
             and args.cycles is not None and args.saif and args.log,
             "saif arguments")
        output = validate_saif(args.saif, args.axis, args.case_id, args.cycles)
        output["runtime"] = validate_runtime_log(args.log, args.axis, args.case_id)
    else:
        need(args.power_report is not None, "power report argument")
        output = parse_power_report(args.power_report)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
