#!/usr/bin/env python3
"""Fail-closed static/runtime checker for M1790; never launches EDA."""
from __future__ import print_function

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1454 = HW / "dc_handoff/runs/m1454_c3_fixed_t10_hold_repair_singlepass_exploratory_r1_20260831"
M1456 = HW / "dc_handoff/runs/m1456_m1454_c3_hold_repair_prelayout_ptsta_r1_20260831"
M1457 = HW / "dc_handoff/runs/m1457_m917_vs_m1454_c3_gate_equivalence_r1_20260831"
M1473 = HW / "contracts/m1473_c3_fixed_t10_hold_closed_corrected_candidate_receipt_r1_20260831.json"
M1479 = HW / "reviews/m1479_m1473_c3_fixed_t10_hold_closed_corrected_candidate_result_blind_hammer_r1_20260831"
M518 = HW / "reviews/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_r1_20260827"
NET = M1454 / "netlist/_sel1_hold_repaired_mapped.v"
SDC = M1454 / "netlist/_sel1_hold_repaired_mapped.sdc"
TB = HW / "dc_handoff/tb/tb_m1790_c3_m1454_fixed_t10_mapped_energy.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1790_c3_m1454_fixed_t10_mapped_energy.f"
UCLI = HW / "dc_handoff/scripts/m1790_c3_m1454_fixed_t10_mapped_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1790_c3_m1454_fixed_t10_mapped_energy.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1790_c3_m1454_fixed_t10_mapped_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1790_c3_m1454_fixed_t10_mapped_energy_source.py"
CONTRACT = HW / "contracts/m1790_c3_m1454_fixed_t10_mapped_energy_source_contract_r1_20260902.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
CELL_V = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
TT_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")
TOP = "tb_m1790_c3_m1454_fixed_t10_mapped_energy"
SAIF_SCOPE = TOP + ".dut"

CLAIMS = dict((key, False) for key in (
    "launch_authorized", "launch_executed", "mapped_vcs",
    "production_saif", "ptpx", "component_power", "component_energy",
    "energy_per_frame", "performance", "speedup", "system_speedup",
    "silicon", "signoff", "paper_ppa_ready", "headline"))

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    NET: "7c01af42322b8feed904df2862aac6e21cbe165b988f1b248f2e94d23f23a7a7",
    SDC: "bb3697e833cb987e4a85ab2a62b4f40946a8c3d6b7eaba08504570f5a862f23f",
    M1454 / "SHA256SUMS": "fecab5cfb7b6fec4666ff1f4be27ced07d6aedb09d9f17503240122777ce77ef",
    M1454 / "SHA256SUMS.seal.sha256": "902a88c9cbbf9363068717a04f52a9e5f95ab26751a86f48f4e3d197a99f4dec",
    M1456 / "SHA256SUMS": "35d0ae3802dd98e25b78b1927dd1e865bf11b51f8b84544ebcb01475e8eb4f6c",
    M1456 / "SHA256SUMS.seal.sha256": "7ff9bf6a6571da0c7be06bbac5bb29b740298996c0731ef33e0ce06388ec4235",
    M1457 / "SHA256SUMS": "83f988da5dc8a256d2ec926db2841f81f73fc6ba1cb7a1fe56e9850603e4e705",
    M1457 / "SHA256SUMS.seal.sha256": "95c72ddcfbf30a475b457211ca2640b41bb42740c16bd4dce5e694b30b1b7b4b",
    M1473: "93f91ae015828cabee7f0f0141a7ef991fd2feafe046d4a8142848c104d96e27",
    M1479 / "review.json": "a971988ce028ab0b961d321e2220066ef3f17b451dde5fc0de075c519b54cdac",
    M1479 / "SHA256SUMS": "845240992a13843551e045f3affc3454b9f9d38f85bd83321acd1b20f9cd0d0f",
    M1479 / "SHA256SUMS.seal.sha256": "5bccb01ec376d4a5a2e0f41133c2f585de4c6c9d09c9a7908786e9d218bf1b90",
    M518 / "SHA256SUMS": "76aa238d8ab7feb864a33c5320da2b37acaddb91f952e2635ce80c1de7f7e3c0",
    M518 / "SHA256SUMS.seal.sha256": "55c661095245364b4f76645f05f48e3a4901129c28d1918e22e6c582d8fd0dcb",
    CELL_V: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    TT_DB: "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
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
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_seal(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal content")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest")
        need((root / rel).is_file() and not (root / rel).is_symlink()
             and sha(root / rel) == fields[0], "manifest drift " + name)
        listed.add(name)


def active_lines(text):
    return [raw.split("#", 1)[0].strip() for raw in text.splitlines()
            if raw.split("#", 1)[0].strip()]


def strip_sv_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def validate_semantics(texts):
    tb = texts[TB]
    active_tb = strip_sv_comments(tb).lower()
    need("force " not in active_tb and "$root" not in active_tb
         and "dut." not in active_tb, "TB hierarchy/state bypass")
    for forbidden in ("+notimingcheck", "+no_notifier", "+nospecify",
                      "+initreg", "deposit(", "vpi_handle_by_name"):
        need(forbidden not in (tb + texts[RUNNER]).lower(),
             "forbidden gate bypass " + forbidden)
    for token in (
            "always #1.5 clk_core", "send_config();",
            "send_tile(100, 48'h1790_0000_0000)",
            "MEASURE_TILES = 8", "M1790_UCLI_SAIF",
            "sampled_result_tag !== expected_tag",
            "sampled_result_beat !== expected_beat",
            "sampled_result_data !== expected_data",
            "context_retire_cycles != expected_retire",
            "$isunknown({config_ready", "result_stall_cycles == 0",
            "raw_stall_cycles == 0", "protocol_error",
            "PASS_M1790_C3_M1454_FIXED_T10_MAPPED_DIRECTED_COMPONENT_ACTIVITY"):
        need(token in tb, "TB omits " + token)
    need(tb.count("$stop") == 2, "TB must have two UCLI stops")

    expected_filelist = [str(CELL_V), str(NET), str(TB)]
    need(active_lines(texts[FILELIST]) == expected_filelist,
         "filelist/order drift")
    need(active_lines(texts[UCLI]) == [
        "power -gate_level all mda sv", "power " + SAIF_SCOPE, "run",
        "power -enable", "run", "power -disable",
        "power -report $::env(M1790_SAIF_FILE) 1e-9 " + SAIF_SCOPE,
        "quit"], "UCLI scope/order drift")

    pt = texts[PT_TCL]
    for token in (
            "M1790_FAIL_BLACK_BOX_AFTER_LINK", "M1790_FAIL_NONZERO_MACRO_COUNT",
            "read_saif -strip_path", "M1790_FAIL_EXACT_NET_ANNOTATION_GATE",
            "M1790_FAIL_EXACT_LEAF_ANNOTATION_GATE", "check_power succeeded",
            "ptpx_whole_mapped_c3_logic.rpt", "power_corner=TT_0p9V_25C",
            "clock_network=ideal_no_cts", "wireload=ZeroWireload",
            "spef=false", "macro_count=0", "not_speedup=true",
            "not_system_or_frame_energy=true", "not_silicon_or_signoff=true"):
        need(token in pt, "PTPX Tcl omits " + token)

    runner = texts[RUNNER]
    for token in (
            "results/.m1790_c3_mapped_energy_attempt_consumed",
            "date_dual_synopsys_same_uid_eda_queue.lock", "collision_gate()",
            "automatic_retry\": False", "reuse_prior_simv_saif_ptpx\": False",
            "M1790_EXPECTED_M1791_MANIFEST_SHA256",
            "AUTHORIZE_ONE_FRESH_M1790_C3_MAPPED_ENERGY_CAMPAIGN",
            "+define+UNIT_DELAY", "CHECK.validate_runtime(sim_log)",
            "CHECK.validate_saif(saif", "CHECK.component_power(",
            "vcs_compiles\": 1", "simv_runs\": 1", "saif_files\": 1",
            "ptpx_runs\": 1", "publish_no_replace(STAGE, RESULT)"):
        need(token in runner, "runner omits " + token)
    need(runner.count("state[\"vcs_compiles\"] += 1") == 1
         and runner.count("state[\"simv_runs\"] += 1") == 1
         and runner.count("state[\"saif_files\"] += 1") == 1
         and runner.count("state[\"ptpx_runs\"] += 1") == 1,
         "runner execution budget drift")


def validate_sources():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift " + str(path))
    for root in (M1454, M1456, M1457, M1479, M518):
        verify_seal(root)
    corrected = strict_json(M1473)
    admitted = strict_json(M1479 / "review.json")
    m518 = strict_json(M518 / "m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_verdict_r1.json")
    need(corrected.get("dc", {}).get("area_um2") == 63756.125879
         and corrected.get("dc", {}).get("macro_count") == 0
         and corrected.get("prime_time", {}).get("setup_wns_ns") == 0.000299
         and corrected.get("prime_time", {}).get("hold_wns_ns") == 0.030474
         and corrected.get("formality", {}).get("passing_compare_points") == 11180,
         "M1473 numeric identity drift")
    need(admitted.get("status") ==
         "PASS_M1479_M1473_CORRECTED_C3_PRELAYOUT_COMPONENT_RESULT_ADMITTED_WITH_STRICT_BOUNDARIES"
         and admitted.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
         "M1479 admission drift")
    need(m518.get("status") ==
         "PASS_DIRECTED_FIXED_T10_VCS_BEHAVIOR__DC_PPA_SYSTEM_HEADLINE_NOT_ADMITTED"
         and m518.get("cycle_anchors", {}).get("issue_cycles_per_tile") == 17
         and m518.get("vcs", {}).get("numeric_mismatches") == 0,
         "M518 VCS admission drift")
    formal = (M1457 / "reports/formality_status.rpt").read_text()
    need("Verification SUCCEEDED" in formal and "11180 Passing compare points" in formal,
         "M1457 Formality drift")
    for token in ("slack (MET)                                                    0.000299",):
        need(token in (M1456 / "reports/timing_setup_slow.rpt").read_text(),
             "M1456 setup drift")
    need("slack (MET)                                                    0.030474" in
         (M1456 / "reports/timing_hold_fast.rpt").read_text(),
         "M1456 hold drift")

    source_paths = (TB, FILELIST, UCLI, PT_TCL, RUNNER, CHECKER, TEST)
    for path in source_paths:
        need(path.is_file() and not path.is_symlink(), "source absent " + str(path))
    texts = dict((path, path.read_text()) for path in source_paths)
    validate_semantics(texts)
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1790_c3_m1454_fixed_t10_mapped_energy_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1454_M1456_M1457_M1479_M518_BOUND__M1791_REVIEW_AND_M1792_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIMS, "source claim promotion")
    need(contract.get("execution_budget") == dict(
        vcs_compiles=1, simv_runs=1, saif_files=1, ptpx_runs=1,
        automatic_retry=False, reuse_prior_simv_saif_ptpx=False),
        "contract budget")
    workload = contract.get("workload", {})
    need(workload.get("warmup_tiles_outside_saif") == 1
         and workload.get("measured_dense_tiles") == 8
         and workload.get("checkpoint_capture") is False
         and workload.get("public_port_only") is True,
         "contract workload")
    mapping = dict((row.get("path"), row.get("sha256"))
                   for row in contract.get("source_files", []))
    need(len(mapping) == len(source_paths), "source inventory cardinality")
    for path in source_paths:
        need(mapping.get(str(path.relative_to(HW))) == sha(path),
             "source inventory drift " + str(path))
    return {"status": "PASS_M1790_SOURCE_STATIC", "source_files": len(source_paths),
            "checks": 1}


def validate_runtime(path):
    text = Path(path).read_text(errors="strict")
    need(text.count("PASS_M1790_C3_M1454_FIXED_T10_MAPPED_DIRECTED_COMPONENT_ACTIVITY") == 1,
         "runtime PASS count")
    need("Error-" not in text and "$fatal" not in text
         and "Assertion failed" not in text, "runtime failure signature")
    window = re.findall(r"M1790_SAIF_WINDOW_STOP cycles=([0-9]+)", text)
    need(len(window) == 1 and int(window[0]) > 0, "runtime window")
    check = re.findall(r"M1790_PUBLIC_RESULT_CHECK tiles=([0-9]+) beats=([0-9]+) mismatches=([0-9]+) xz=([0-9]+)", text)
    need(check == [("8", "40", "0", "0")], "runtime result checker")
    counters = re.findall(r"M1790_PUBLIC_COUNTER_DELTAS raw_beats=([0-9]+) tiles=([0-9]+) issues=([0-9]+) done=([0-9]+) pushes=([0-9]+) departures=([0-9]+)", text)
    need(counters == [("40", "8", "136", "8", "40", "40")],
         "runtime conservation")
    cover = re.findall(r"M1790_PUBLIC_COVERAGE result_stall_cycles=([0-9]+) raw_stall_cycles=([0-9]+) retire_cycles=([0-9]+)", text)
    need(len(cover) == 1 and int(cover[0][0]) > 0 and int(cover[0][1]) > 0
         and int(cover[0][2]) > 0, "runtime cover")
    return {"status": "PASS_M1790_PUBLIC_RUNTIME",
            "measurement_cycles": int(window[0]), "measured_tiles": 8,
            "result_beats": 40, "result_stall_cycles": int(cover[0][0]),
            "raw_stall_cycles": int(cover[0][1]),
            "retire_cycles": int(cover[0][2]), "numeric_mismatches": 0,
            "public_xz": 0}


def strip_saif_block_comments(text):
    output = []
    index = 0
    count = 0
    while index < len(text):
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            need(end >= 0, "unterminated SAIF comment")
            index = end + 2
            count += 1
        else:
            output.append(text[index]); index += 1
    return "".join(output), count


def parse_saif(text):
    cleaned, comment_count = strip_saif_block_comments(text)
    tokens = re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', cleaned)
    pos = [0]
    def parse_one():
        need(pos[0] < len(tokens) and tokens[pos[0]] == "(", "malformed SAIF")
        pos[0] += 1
        node = []
        while pos[0] < len(tokens) and tokens[pos[0]] != ")":
            if tokens[pos[0]] == "(": node.append(parse_one())
            else: node.append(tokens[pos[0]]); pos[0] += 1
        need(pos[0] < len(tokens), "unterminated SAIF")
        pos[0] += 1
        return node
    root = parse_one()
    need(pos[0] == len(tokens) and root and root[0] == "SAIFILE", "SAIF root")
    return root, comment_count


def forms(node, tag):
    return [item for item in node[1:]
            if isinstance(item, list) and item and item[0] == tag]


def all_forms(node, tag):
    found = []
    if isinstance(node, list):
        if node and node[0] == tag: found.append(node)
        for item in node:
            if isinstance(item, list): found.extend(all_forms(item, tag))
    return found


def direct_instance(node, name):
    hits = [item for item in forms(node, "INSTANCE")
            if len(item) >= 2 and item[1].lstrip("\\") == name]
    need(len(hits) == 1, "SAIF instance absent/duplicated " + name)
    return hits[0]


def validate_saif(path, cycles):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
         "SAIF input")
    root, comment_count = parse_saif(path.read_text(errors="strict"))
    duration = forms(root, "DURATION")
    need(len(duration) == 1 and len(duration[0]) == 2, "SAIF duration")
    duration_ns = float(duration[0][1])
    need(math.isfinite(duration_ns) and abs(duration_ns-cycles*3.0) <= 1e-6,
         "SAIF duration/cycle mismatch")
    top = direct_instance(root, TOP)
    dut = direct_instance(top, "dut")
    groups = dict((tag, all_forms(dut, tag)) for tag in ("T0", "T1", "TX", "TC", "IG"))
    count = len(groups["T0"])
    need(count > 0 and all(len(value) == count for value in groups.values()),
         "SAIF form count")
    need(all(len(item) == 2 and float(item[1]) == 0.0 for item in groups["TX"]),
         "SAIF contains TX")
    for t0, t1, tx in zip(groups["T0"], groups["T1"], groups["TX"]):
        need(all(len(item) == 2 for item in (t0, t1, tx))
             and abs(float(t0[1])+float(t1[1])+float(tx[1])-duration_ns) <= 1e-6,
             "SAIF activity conservation")
    need(any(float(item[1]) > 0 for item in groups["TC"] if len(item) == 2),
         "SAIF has no toggles")
    return {"status": "PASS_M1790_DUT_ONLY_SAIF", "cycles": cycles,
            "duration_ns": duration_ns, "activity_forms_per_tag": count,
            "tx_nonzero": 0, "saif_scope": SAIF_SCOPE,
            "block_comments_skipped": comment_count, "saif_sha256": sha(path)}


POWER_FIELDS = ("Net Switching Power", "Cell Internal Power",
                "Cell Leakage Power", "Total Power")


def parse_power(path):
    text = Path(path).read_text(errors="strict")
    need("Report : Averaged Power" in text and "-unit mW" in text,
         "power report mode/unit")
    values = {}
    for field in POWER_FIELDS:
        hits = re.findall(re.escape(field) + r"\s*=\s*([0-9.eE+-]+)", text)
        need(len(hits) == 1, "power field " + field)
        values[field] = float(hits[0])
        need(math.isfinite(values[field]) and values[field] >= 0.0,
             "invalid power field")
    return values


def component_power(path, cycles):
    need(type(cycles) is int and cycles > 0, "cycle domain")
    power = parse_power(path)
    subtotal = power["Net Switching Power"] + power["Cell Internal Power"] \
        + power["Cell Leakage Power"]
    tolerance = max(1e-8, 1e-6 * max(1.0, power["Total Power"]))
    need(abs(subtotal-power["Total Power"]) <= tolerance,
         "power conservation")
    duration_ns = cycles * 3.0
    return {"status": "PASS_M1790_COMPONENT_METRIC_PENDING_RESULT_HAMMER",
            "cycles": cycles, "duration_ns": duration_ns,
            "net_switching_power_mw": power["Net Switching Power"],
            "cell_internal_power_mw": power["Cell Internal Power"],
            "cell_leakage_power_mw": power["Cell Leakage Power"],
            "total_power_mw": power["Total Power"],
            "directed_window_energy_pj": power["Total Power"] * duration_ns,
            "component_total_conserved": True, "macro_count": 0,
            "claim_boundary": {"directed_component_workload": True,
                "prelayout_logic_only": True, "tt_0p9v_25c": True,
                "ideal_clock": True, "zero_wireload": True, "spef": False,
                "macro_count": 0, "component_power": True,
                "directed_window_energy": True, "energy_per_frame": False,
                "speedup": False, "system_speedup": False, "silicon": False,
                "signoff": False, "paper_ppa_ready": False, "headline": False}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()
    need(args.static, "only --static is allowed")
    print(json.dumps(validate_sources(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
