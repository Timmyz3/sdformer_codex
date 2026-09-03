#!/usr/bin/python3.12
"""Fail-closed source/runtime/result parser for the M2058 two-axis campaign."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m2058_m2056_m2018_tsbg_matched_mapped_energy_source_contract_r1_20260903.json"
TOP = "tb_m2056_m2018_tsbg_matched_mapped_energy"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m2056_m2018_tsbg_matched_mapped_energy.tcl"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2056_REVIEW = HW / "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903"

AXIS_ORDER = ("ordinary_lru4", "tsbg_b4")
AXES = {
    "ordinary_lru4": {
        "filelist": HW / "dc_handoff/filelists/iscas_m2056_m2018_tsbg_ordinary_mapped_energy.f",
        "ucli": HW / "dc_handoff/scripts/m2056_m2018_tsbg_ordinary_mapped_energy.ucli.tcl",
        "netlist": HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
        "sdc": HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.sdc",
        "design": "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE0",
        "scope": TOP + ".core.dut_base.g_mapped.mapped_implementation",
        "cycles": 20292,
        "scalar_weight_reads": 14304,
        "end_marker": "M2056_SAIF_WINDOW_END axis=ordinary_lru4 global_slot=42 measurement_cycles=20292",
    },
    "tsbg_b4": {
        "filelist": HW / "dc_handoff/filelists/iscas_m2056_m2018_tsbg_tsbg_mapped_energy.f",
        "ucli": HW / "dc_handoff/scripts/m2056_m2018_tsbg_tsbg_mapped_energy.ucli.tcl",
        "netlist": HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
        "sdc": HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.sdc",
        "design": "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE1",
        "scope": TOP + ".core.dut_tsbg.g_mapped.mapped_implementation",
        "cycles": 7569,
        "scalar_weight_reads": 4608,
        "end_marker": "M2056_SAIF_WINDOW_END axis=tsbg_b4 global_slot=42 measurement_cycles=7569",
    },
}

BEGIN_MARKER = ("M2056_SAIF_WINDOW_BEGIN global_slot=42 m2047_anchor_slot=0 "
                "sample=0 layer=28 is_fc2=0 token_start=0 "
                "source_groups=48 preload_cycles=383")
PASS_PREFIX = "PASS_M2051_EP34_TSBG_FULL40_CYCLE "
EXPECTED_PASS = {
    "workload_slot": "42", "sample_id": "0", "layer": "28",
    "is_fc2": "0", "token_start": "0", "source_groups": "48",
    "physical_groups": "48", "rows": "149", "issues": "1278",
    "products": "29472", "commits": "24", "base_cycles": "20292",
    "tsbg_cycles": "7569", "bundles_base": "1788",
    "bundles_tsbg": "576", "scalar_base": "14304",
    "scalar_tsbg": "4608", "stale": "1", "retired_replay": "1",
    "replay_accept": "0", "reset": "2", "recovery": "1",
    "real_weights": "false", "system_speedup": "false",
}

SOURCE_SHA256 = {
    "dc_handoff/filelists/iscas_m2056_m2018_tsbg_ordinary_mapped_energy.f": "fab3a7d602b12588ec66986d0b70791d71ca79bd086dd6e038d40a90d4054559",
    "dc_handoff/filelists/iscas_m2056_m2018_tsbg_tsbg_mapped_energy.f": "7ce2eab8600a99debd9cde8f885b215a073dee0c59d68751a2ffc29c41e01a9e",
    "rtl_m2018/m2056_m2018_matched_mapped_axis_adapter.sv": "5c84f5f8c61b7f48f3560b54a34b3a1df669421a16a15255fe206db9239a7fcd",
    "tb_m2018/tb_m2056_m2018_tsbg_matched_mapped_energy.sv": "25a21714b568d99bf60aeea5daf767ebdf03a8fcda6e194e224f40341118879e",
    "dc_handoff/scripts/m2056_m2018_tsbg_ordinary_mapped_energy.ucli.tcl": "33c962dd0e7fb6f52d4af86498cf037f0dd7da0747b7892da415155c9cc68558",
    "dc_handoff/scripts/m2056_m2018_tsbg_tsbg_mapped_energy.ucli.tcl": "670d645e267ecc59bda6ac95c421563bb9b7935768f2084e1aebe17bd1610e5e",
    "dc_handoff/scripts/run_ptpx_m2056_m2018_tsbg_matched_mapped_energy.tcl": "9747ed852ca649fed6f6fd82e369825ae27af08ac80858215e5b1a94f48db907",
    "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v": "f5847f355329a52511ab044ef458284a19ae424ac778418a4bc4778bb2d3a2b0",
    "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.sdc": "46b4bd73ace0cfb67f7794321f641ebfabfc0cabd542776ed586d65438970838",
    "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v": "739eb76dcb732ec0c66b75392c768cbe36027ecc5d458bd4b088f8488f67c9af",
    "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.sdc": "c7b894cee479badcca22977b29d6ba69a20ca85d9b20e402c9c46ad92ed16d70",
    "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv": "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv": "64805bdedb7c80d5c6141bc36e59ef61234507b40942e69ccbf4a30ac2383436",
    "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv": "e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2",
    "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh": "487ca0073526b973220abd77c91d12dbc2420901443541ec5a79e36a780e1bf0",
    "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920_stats.memh": "70810fdf3ac4ba2d281d750995810f08561addb50871550aa83343a2a04a6dca",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/review.json": "9d2c98fbe80c4eaebcce60109ec9d795cbb10d78127605be3c5b65e4ca0bd76f",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/input_manifest.sha256": "0416ad6fe5b2ff1cbc3f900bc80e330a2df356e6b1a22a1947f54c0c7e1d5bf4",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/SHA256SUMS": "bc308ec9dfd27afd6e00b231c54461c9628eec5678768c9613db33425d7ea9c2",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/SHA256SUMS.seal.sha256": "2f5a193dbd79728a1b3dd75e7889accaedfd1bd750610f69a731dcc9a66b2c86",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

POWER_FIELDS = ("Net Switching Power", "Cell Internal Power",
                "Cell Leakage Power", "Total Power")
CLAIM_BOUNDARY = {
    "workload": "single_pre_registered_ep34_G48_component_workload",
    "selection_uses_performance": False,
    "weights": "deterministic_directed_INT8_not_checkpoint_weights",
    "logic_scope": "mapped_standard_cells_only",
    "external_weight_sram_included": False,
    "power_corner": "TT_0p9V_25C",
    "power_mode": "averaged_prelayout",
    "clock_network": "ideal_no_cts",
    "wireload": "ZeroWireload",
    "macro_count": 0,
    "system_speedup": False,
    "paper_ppa_ready": False,
}


class Failure(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise Failure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, digest):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "missing/symlink " + str(path))
    need(sha(path) == digest, "identity drift " + str(path))


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            Failure("nonfinite JSON " + token)))
    need(type(result) is dict, "JSON root")
    return result


def verify_double_sealed_directory(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal content " + str(root))
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts,
             "unsafe manifest member")
        name = rel.as_posix()
        need(name not in mapping, "duplicate manifest member")
        exact(root / rel, fields[0])
        mapping[name] = fields[0]
    actual = set()
    for member in root.rglob("*"):
        need(not member.is_symlink(), "symlink in sealed result/review")
        if member.is_file() and member.name not in {
                "SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(member.relative_to(root).as_posix())
    need(actual == set(mapping), "non-exhaustive manifest")
    return mapping


def verify_contract_seal():
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract outer seal")


def validate_sources():
    for rel, digest in SOURCE_SHA256.items():
        exact(HW / rel, digest)
    verify_contract_seal()
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m2058_m2056_m2018_tsbg_matched_mapped_energy_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M2059_REVIEW_REQUIRED_BEFORE_ONE_M2058_EXECUTION__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIM_BOUNDARY, "claim boundary")
    need(contract.get("execution_budget") == {
        "license_preflight_lmstat": 1, "vcs_compiles": 2,
        "simv_runs": 2, "saif_files": 2, "ptpx_runs": 2,
        "p1_serial": True, "automatic_retry": False,
        "reuse_prior_simv_saif_ptpx": False}, "execution budget")
    source_rows = contract.get("m2056_frozen_sources", [])
    need(type(source_rows) is list and len(source_rows) == len(SOURCE_SHA256),
         "contract source inventory cardinality")
    need(all(type(row) is dict and set(row) == {"path", "sha256"}
             for row in source_rows), "contract source row schema")
    inventory = {row["path"]: row["sha256"] for row in source_rows}
    need(len(inventory) == len(source_rows), "contract duplicate source path")
    need(inventory == SOURCE_SHA256, "contract frozen-source inventory")
    for axis in AXIS_ORDER:
        cfg = AXES[axis]
        flines = cfg["filelist"].read_text().splitlines()
        need(sum("_mapped.v" in row for row in flines) == 1,
             "mapped netlist cardinality " + axis)
        need(str(cfg["netlist"]) in flines, "mapped netlist mismatch " + axis)
        ucli = cfg["ucli"].read_text()
        need(ucli.count("\nrun\n") + int(ucli.startswith("run\n")) == 3,
             "UCLI run count " + axis)
        need(ucli.count("power -enable") == 1
             and ucli.count("power -disable") == 1
             and ucli.count("power -report") == 1
             and ucli.count(cfg["scope"]) == 2, "UCLI scope/commands " + axis)
    tb_text = (HW / "tb_m2018/tb_m2056_m2018_tsbg_matched_mapped_energy.sv").read_text()
    need(tb_text.count("$stop;") == 2, "two-stop TB")
    need("FROZEN_WORKLOAD_SLOT = 42" in tb_text
         and "measurement_window_active" in tb_text,
         "slot/XZ window lock")
    review_map = verify_double_sealed_directory(M2056_REVIEW)
    need(review_map.get("review.json") == sha(M2056_REVIEW / "review.json"),
         "M2056 review not sealed")
    review = strict_json(M2056_REVIEW / "review.json")
    need(review.get("status", "").startswith("PASS_M2056_")
         and review.get("severity_counts", {}).get("p0") == 0
         and review.get("severity_counts", {}).get("p1") == 0,
         "M2056 source admission")
    return {"status": "PASS_M2058_STATIC_SOURCE_IDENTITY",
            "frozen_sources": len(SOURCE_SHA256), "axes": list(AXIS_ORDER)}


def parse_command_log(path, axis):
    text = Path(path).read_text(errors="strict")
    need(not re.search(r"(?im)^\s*(?:Error-\[|Fatal:)", text),
         "compile fatal/error " + axis)
    rows = re.findall(r"^M2058_COMMAND_JSON=(.+)$", text, flags=re.MULTILINE)
    need(len(rows) == 1, "compile command record " + axis)
    command = json.loads(rows[0])
    cfg = AXES[axis]
    need(command.count("-top") == 1 and command[command.index("-top") + 1] == TOP,
         "compile top " + axis)
    need(command.count("-f") == 1
         and command[command.index("-f") + 1] == str(cfg["filelist"]),
         "compile filelist " + axis)
    return {"axis": axis, "log_sha256": sha(path), "command": command}


def parse_runtime(path, axis):
    cfg = AXES[axis]
    text = Path(path).read_text(errors="strict")
    need(not any(token in text for token in
                 ("Fatal:", "$fatal", "Assertion failed", "contains X/Z",
                  "mapped load/reset X/Z", "mapped memory handshake X/Z",
                  "mapped bridge/commit/control X/Z", "mapped counter X/Z",
                  "mapped bank metadata X/Z", "mapped payload X/Z",
                  "mapped accumulator X/Z")), "runtime fatal/XZ " + axis)
    need(text.count(BEGIN_MARKER) == 1, "first stop marker " + axis)
    need(text.count(cfg["end_marker"]) == 1, "second stop marker " + axis)
    pass_lines = [row for row in text.splitlines() if row.startswith(PASS_PREFIX)]
    need(len(pass_lines) == 1, "final M2051 PASS count " + axis)
    fields = {}
    for token in pass_lines[0][len(PASS_PREFIX):].split():
        need(token.count("=") == 1, "PASS token syntax")
        key, value = token.split("=", 1)
        need(key not in fields, "duplicate PASS field " + key)
        fields[key] = value
    need(fields == EXPECTED_PASS, "M2051 PASS identity/ledger drift " + axis)
    return {"axis": axis, "log_sha256": sha(path), "stop_markers": 2,
            "final_m2051_passes": 1, "cycles": cfg["cycles"],
            "scalar_weight_reads": cfg["scalar_weight_reads"]}


def parse_saif(path, axis):
    cfg = AXES[axis]
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
         "SAIF regular/nonempty " + axis)
    header = ""
    tx_count = 0
    positive_tc = 0
    scope_tokens = {"mapped_implementation": False}
    with path.open("r", errors="strict") as handle:
        for index, line in enumerate(handle):
            if index < 256:
                header += line
            if "mapped_implementation" in line:
                scope_tokens["mapped_implementation"] = True
            for value in re.findall(r"\(TX\s+([0-9.eE+-]+)\)", line):
                tx_count += 1
                need(float(value) == 0.0, "nonzero SAIF TX " + axis)
            for value in re.findall(r"\(TC\s+([0-9.eE+-]+)\)", line):
                if float(value) > 0.0:
                    positive_tc += 1
    timescale = re.findall(
        r"\(TIMESCALE\s+([0-9.eE+-]+)\s+([a-zA-Z]+)\)", header)
    duration = re.findall(r"\(DURATION\s+([0-9.eE+-]+)\)", header)
    need(len(timescale) == 1 and len(duration) == 1, "SAIF header " + axis)
    unit_ns = {"s": 1.0e9, "ms": 1.0e6, "us": 1.0e3,
               "ns": 1.0, "ps": 1.0e-3, "fs": 1.0e-6}
    need(timescale[0][1] in unit_ns, "SAIF timescale unit " + axis)
    duration_ns = (float(duration[0]) * float(timescale[0][0])
                   * unit_ns[timescale[0][1]])
    need(abs(duration_ns - cfg["cycles"] * 3.0) <= 1.0e-6,
         "SAIF duration " + axis)
    need(tx_count > 0 and positive_tc > 0, "SAIF activity/TX census " + axis)
    need(all(scope_tokens.values()), "SAIF mapped scope " + axis)
    return {"axis": axis, "saif_sha256": sha(path),
            "duration_ns": duration_ns, "tx_entries": tx_count,
            "nonzero_tx_entries": 0, "positive_tc_entries": positive_tc}


def parse_power_report(path):
    text = Path(path).read_text(errors="strict")
    need("Report : Averaged Power" in text and "-unit mW" in text,
         "averaged mW report")
    values = {}
    for field in POWER_FIELDS:
        hits = re.findall(re.escape(field) + r"\s*=\s*([0-9.eE+-]+)", text)
        need(len(hits) == 1, "unique power field " + field)
        value = float(hits[0])
        need(math.isfinite(value) and value >= 0.0, "power value " + field)
        values[field] = value
    need(values["Total Power"] > 0.0, "positive total power")
    subtotal = sum(values[field] for field in POWER_FIELDS[:3])
    need(abs(subtotal - values["Total Power"])
         <= max(1.0e-6, values["Total Power"] * 1.0e-4), "power subtotal")
    return {"switching_mw": values[POWER_FIELDS[0]],
            "internal_mw": values[POWER_FIELDS[1]],
            "leakage_mw": values[POWER_FIELDS[2]],
            "total_mw": values[POWER_FIELDS[3]]}


def parse_ptpx(root, axis):
    cfg = AXES[axis]
    root = Path(root)
    log_text = (root / "ptpx.log").read_text(errors="strict")
    need(not re.search(r"(?im)^\s*(?:Error:|Fatal:)", log_text),
         "PTPX fatal/error " + axis)
    marker = (root / "PTPX_INTERNAL_COMPLETE.txt").read_text(errors="strict")
    need(marker.count(
        "PASS_M2056_M2018_TSBG_MATCHED_MAPPED_PTPX_PENDING_RESULT_HAMMER") == 1,
        "PTPX marker " + axis)
    need(marker.count("axis=" + axis) == 1
         and marker.count("measurement_cycles=" + str(cfg["cycles"])) == 1,
         "PTPX marker identity " + axis)
    annotation = (root / "reports/saif_annotation_summary.rpt").read_text(
        errors="strict")
    total_net = re.findall(r"Total number of nets = ([0-9]+)", annotation)
    net = re.findall(r"Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)",
                     annotation)
    total_leaf = re.findall(r"Total number of leaf cells = ([0-9]+)", annotation)
    leaf = re.findall(
        r"Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)",
        annotation)
    need(len(total_net) == len(net) == len(total_leaf) == len(leaf) == 1,
         "annotation parse " + axis)
    need(int(total_net[0]) > 0 and int(net[0][0]) == int(total_net[0])
         and float(net[0][1]) == 100.0 and int(total_leaf[0]) > 0
         and int(leaf[0][0]) == int(total_leaf[0])
         and float(leaf[0][1]) == 100.0, "annotation coverage " + axis)
    boundary = {}
    for row in (root / "reports/scope_and_boundary.rpt").read_text().splitlines():
        need(row.count("=") == 1, "boundary syntax")
        key, value = row.split("=", 1)
        need(key not in boundary, "boundary duplicate")
        boundary[key] = value
    expected_boundary = {
        "milestone": "M2056", "design": cfg["design"], "axis": axis,
        "analysis": "averaged_prelayout_standard_cell_power",
        "power_corner": "tt0p9v25c", "clock_period_ns": "3.0",
        "measurement_cycles": str(cfg["cycles"]),
        "measurement_duration_ns": str(float(cfg["cycles"] * 3)),
        "saif_timescale": "1 ns", "saif_duration_raw": str(float(cfg["cycles"] * 3)),
        "descriptor_preload_cycles_excluded": "383",
        "workload": "ep34_full40_global_slot42_sample0_layer28_fc1_token0_g48",
        "m2047_semantic_anchor_slot": "0", "saif_scope": cfg["scope"],
        "clock_network": "ideal_no_cts", "wireload": "ZeroWireload",
        "spef": "false", "macro_count": "0",
        "external_weight_sram_excluded": "true",
    }
    # Timescale/duration formatting can vary while remaining numerically exact.
    for key, value in expected_boundary.items():
        if key not in {"measurement_duration_ns", "saif_timescale",
                        "saif_duration_raw"}:
            need(boundary.get(key) == value, "boundary field " + key + " " + axis)
    need(abs(float(boundary.get("measurement_duration_ns", "nan"))
             - cfg["cycles"] * 3.0) <= 1.0e-6, "boundary duration " + axis)
    power = parse_power_report(root / "reports/power.rpt")
    duration_ns = cfg["cycles"] * 3.0
    return {"axis": axis, "ptpx_log_sha256": sha(root / "ptpx.log"),
            "annotation": {"nets": int(total_net[0]),
                           "net_percent": 100.0,
                           "leaf_cells": int(total_leaf[0]),
                           "leaf_percent": 100.0},
            "power": power,
            "execute_energy_pj": {
                "switching": power["switching_mw"] * duration_ns,
                "internal": power["internal_mw"] * duration_ns,
                "leakage": power["leakage_mw"] * duration_ns,
                "total": power["total_mw"] * duration_ns}}


def parse_candidate(candidate, compile_dir):
    candidate = Path(candidate)
    compile_dir = Path(compile_dir)
    rows = {}
    for axis in AXIS_ORDER:
        axis_root = candidate / axis
        rows[axis] = {
            "compile": parse_command_log(compile_dir / (axis + ".compile.log"), axis),
            "runtime": parse_runtime(axis_root / "mapped_sim.log", axis),
            "saif": parse_saif(axis_root / "mapped_execute.saif", axis),
            "ptpx": parse_ptpx(axis_root / "ptpx", axis),
        }
    ordinary = rows["ordinary_lru4"]
    tsbg = rows["tsbg_b4"]
    ordinary_energy = ordinary["ptpx"]["execute_energy_pj"]["total"]
    tsbg_energy = tsbg["ptpx"]["execute_energy_pj"]["total"]
    need(tsbg_energy > 0.0, "positive TSBG energy denominator")
    return {
        "schema": "m2058_m2056_tsbg_matched_mapped_energy_candidate_receipt_r1_v1",
        "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_budget_observed": {
            "license_preflight_lmstat": 1, "vcs_compiles": 2,
            "simv_runs": 2, "saif_files": 2, "ptpx_runs": 2,
            "p1_serial": True, "automatic_retry": False},
        "workload": {"global_slot": 42, "m2047_anchor_slot": 0,
                     "sample_id": 0, "layer_id": 28, "operator": "FC1",
                     "token_start": 0, "source_groups": 48,
                     "descriptor_preload_cycles_excluded": 383,
                     "real_activity_masks": True,
                     "real_checkpoint_weights": False},
        "axes": rows,
        "measured_logic_only_comparison": {
            "cycle_speedup_ordinary_over_tsbg": 20292 / 7569,
            "cycle_reduction_fraction": 1.0 - 7569 / 20292,
            "logic_execute_energy_ratio_ordinary_over_tsbg":
                ordinary_energy / tsbg_energy,
            "logic_execute_energy_reduction_fraction":
                1.0 - tsbg_energy / ordinary_energy},
        "external_weight_sram_symbolic_only": {
            "ordinary_scalar_128b_reads": 14304,
            "tsbg_scalar_128b_reads": 4608,
            "read_reduction_fraction": 1.0 - 4608 / 14304,
            "formula_with_Eread_128b_pJ": {
                "ordinary": "14304 * Eread_128b_pJ",
                "tsbg": "4608 * Eread_128b_pJ"},
            "formula_with_Eread_bit_pJ": {
                "ordinary": "14304 * 128 * Eread_bit_pJ",
                "tsbg": "4608 * 128 * Eread_bit_pJ"},
            "numeric_macro_energy_reported": False,
            "logic_plus_sram_total_numeric_reported": False}}


def validate_sealed_result(root):
    mapping = verify_double_sealed_directory(root)
    need("receipt.json" in mapping, "sealed receipt absent")
    receipt = strict_json(Path(root) / "receipt.json")
    need(receipt.get("status") ==
         "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
         "sealed receipt status")
    need(receipt.get("claim_boundary") == CLAIM_BOUNDARY,
         "sealed receipt boundary")
    return {"status": "PASS_M2058_SEALED_RESULT_STRUCTURE",
            "members": len(mapping), "receipt_sha256": sha(Path(root) / "receipt.json")}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--static", action="store_true")
    group.add_argument("--sealed-result")
    args = parser.parse_args()
    result = validate_sources() if args.static else validate_sealed_result(
        Path(args.sealed_result))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
