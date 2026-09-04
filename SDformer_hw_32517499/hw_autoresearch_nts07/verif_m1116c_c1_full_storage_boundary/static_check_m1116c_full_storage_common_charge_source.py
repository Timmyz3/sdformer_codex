#!/usr/bin/env python3
"""Source-only M1116C full-storage/common-charge boundary checker.

The checker performs no VCS, synthesis, PT, Formality, GPU, remote, or full
replay action.  It validates the additive source package and its claim limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent
WRAPPER = HW / "rtl_m1116c_c1_full_storage_boundary/m1116c_m935_c1_full_storage_common_charge_boundary.sv"
MAPPING = HW / "dc_handoff/manifests/m1116c_c1_full_storage_boundary_mapping_r1.tsv"
FILELIST = HW / "dc_handoff/filelists/date_m1116c_m935_c1_full_storage_common_charge_dc.f"
SDC = HW / "dc_handoff/constraints/date_m1116c_m935_c1_full_storage_common_charge_3ns.sdc"
TCL = HW / "dc_handoff/scripts/run_dc_m1116c_m935_c1_full_storage_common_charge_candidate.tcl"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
CONTRACT = HW / "contracts/m1116c_m1114_m1006_m963_m959_m935_full_storage_common_charge_source_contract_r1_20260830.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

M935_SHA = "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8"
PARENT_SHA = "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783"
DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
TOTAL_BYTES = 214_912
BUDGET_BYTES = 245_760


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + value)))


def active_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def parse_mapping(path: Path = MAPPING) -> dict[str, Any]:
    rows = []
    expected_start = 0
    for line in active_lines(path):
        fields = line.split("|")
        require(len(fields) == 14, "mapping field count")
        (name, start, end, byte_count, placement, macro_cell,
         physical_count, physical_capacity, common_equiv, port, latency,
         live_binding, axis_scope, area_in_dc) = fields
        row = {
            "class": name, "byte_start": int(start), "byte_end": int(end),
            "bytes": int(byte_count), "placement": placement,
            "physical_macro_cell": macro_cell,
            "physical_macro_count": int(physical_count),
            "physical_macro_capacity_bytes": int(physical_capacity),
            "common_macro_equivalents": int(common_equiv),
            "port_model": port, "latency_cycles": latency,
            "live_binding": live_binding, "axis_scope": axis_scope,
            "area_in_candidate_dc": area_in_dc,
        }
        require(row["byte_start"] == expected_start, "mapping gap/overlap")
        require(row["byte_end"] - row["byte_start"] + 1 == row["bytes"],
                "mapping range/count mismatch")
        require(row["live_binding"], "empty live/common binding")
        require(area_in_dc in ("true", "false"), "area boolean drift")
        expected_start = row["byte_end"] + 1
        rows.append(row)
    require(expected_start == TOTAL_BYTES, "mapping does not end at 214912")
    require(len(rows) == 4 and [row["class"] for row in rows] ==
            ["parent_scratch", "psum_store", "weight_store", "metadata_reserve"],
            "mapping class population/order drift")

    internal = [row for row in rows if row["placement"] == "foundry_macro_internal"]
    external = [row for row in rows
                if row["placement"] == "identical_external_common_charge"]
    require(len(internal) == 1 and len(external) == 3,
            "mapping placement population drift")
    for row in internal:
        require(row["physical_macro_cell"] == "TS1N28HPCPHVTB128X128M4S" and
                row["physical_macro_count"] == 9 and
                row["physical_macro_capacity_bytes"] == 2048 and
                row["bytes"] == row["physical_macro_count"] *
                row["physical_macro_capacity_bytes"] and
                row["area_in_candidate_dc"] == "true",
                "internal parent macro mapping drift")
    for row in external:
        require(row["physical_macro_cell"] == "NONE" and
                row["physical_macro_count"] == 0 and
                row["physical_macro_capacity_bytes"] == 0 and
                row["area_in_candidate_dc"] == "false",
                "external common charge materialized as macro")
        require(row["axis_scope"] ==
                "candidate,strongest_zero,same_coordinate_bit",
                "external three-axis equality drift")

    totals = {
        "represented_bytes": sum(row["bytes"] for row in rows),
        "internal_macro_bytes": sum(row["bytes"] for row in internal),
        "external_common_charge_bytes": sum(row["bytes"] for row in external),
        "physical_macro_count": sum(row["physical_macro_count"] for row in rows),
        "external_common_macro_equivalents_diagnostic":
            sum(row["common_macro_equivalents"] for row in external),
        "budget_bytes": BUDGET_BYTES,
        "margin_bytes": BUDGET_BYTES - TOTAL_BYTES,
    }
    require(totals == {
        "represented_bytes": 214_912,
        "internal_macro_bytes": 18_432,
        "external_common_charge_bytes": 196_480,
        "physical_macro_count": 9,
        "external_common_macro_equivalents_diagnostic": 84,
        "budget_bytes": 245_760,
        "margin_bytes": 30_848,
    }, "mapping totals drift")
    return {"rows": rows, "totals": totals}


def check_wrapper() -> dict[str, Any]:
    text = WRAPPER.read_text()
    frozen_text = M935.read_text()
    require(text.count("m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935") == 1,
            "wrapper must instantiate frozen M935 exactly once")
    require("TS1N28HPCPHVTB128X128M4S" not in text,
            "wrapper directly instantiates area macro")
    require("m528_dw1rw_parent_scratch_9x128_macro" not in text,
            "wrapper duplicates parent scratch")
    required = (
        "weight_read_request_valid", "weight_read_request_ready",
        "weight_read_response_valid", "weight_read_response_ready",
        "weight_product_residual_data", "psum_read_request_valid",
        "psum_read_request_ready", "psum_read_response_valid",
        "psum_read_response_ready", "psum_read_response_data",
        ".issue_data_valid(core_issue_data_valid)",
        ".issue_residual_data(weight_product_residual_data)",
        ".issue_psum_prior(psum_read_response_data)",
        ".psum_write_valid(psum_write_valid)",
        ".psum_write_ready(psum_write_ready)",
    )
    for token in required:
        require(token in text, "missing live storage binding: " + token)
    require("service_outstanding_q" in text and "service_first_q" in text and
            "boundary_fault_q" in text, "boundary control state missing")
    require(not re.search(r"(?m)^\s+logic\s+\[1151:0\]", text),
            "unexpected internal 1152-bit payload state")
    require(not re.search(r"(?m)^\s+logic\s+\[1823:0\]", text),
            "unexpected internal 1824-bit payload state")
    frozen_header = frozen_text[
        frozen_text.index("module m935_m912_three_stage_exact_parent_match_product_capture_island"):
        frozen_text.index(");", frozen_text.index(
            "module m935_m912_three_stage_exact_parent_match_product_capture_island")) + 2]
    frozen_ports = re.findall(
        r"\b(?:input|output)\s+logic(?:\s+\[[^\]]+\])?\s+([A-Za-z_][A-Za-z0-9_]*)",
        frozen_header)
    instance = text[text.index(
        "m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935"):]
    instance = instance[:instance.index(");") + 2]
    connected_ports = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", instance)
    require(len(connected_ports) == len(set(connected_ports)) and
            set(connected_ports) == set(frozen_ports),
            "frozen M935 port connection exact-set drift")
    return {"frozen_m935_instances": 1, "direct_macro_instances": 0,
            "frozen_m935_ports_connected_exactly_once": len(frozen_ports),
            "live_weight_service": True, "live_psum_read_write_service": True,
            "added_payload_fifo_bits": 0, "added_control_state_bits": 3}


def check_filelist() -> dict[str, Any]:
    members = active_lines(FILELIST)
    expected = [
        "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
        "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv",
        "rtl_m1116c_c1_full_storage_boundary/m1116c_m935_c1_full_storage_common_charge_boundary.sv",
    ]
    require(members == expected, "synthesis filelist exact-set/order drift")
    forbidden = ("tb_", "verif_", "assert", "attack", "sva", "unit_delay", ".v")
    require(all(not any(token in member.lower() for token in forbidden)
                for member in members), "non-production member in filelist")
    return {"members": members, "synthesis_only": True,
            "tb_sva_attack_members": 0, "behavioral_macro_verilog_members": 0}


def check_sdc() -> dict[str, Any]:
    commands = "\n".join(active_lines(SDC))
    require(re.search(r"create_clock\s+-name\s+core_clk\s+-period\s+3\.000", commands),
            "3ns clock missing")
    prohibited = ("set_false_path", "set_multicycle_path", "set_disable_timing",
                  "set_case_analysis", "set_max_delay", "set_min_delay")
    hits = {command: len(re.findall(r"(?m)^\s*" + re.escape(command) + r"\b", commands))
            for command in prohibited}
    require(not any(hits.values()), "timing exception present")
    require("[all_inputs]" in commands and "[all_outputs]" in commands and
            "reset_n" not in re.sub(r"remove_from_collection[^\n]+", "", commands),
            "I/O timing boundary drift")
    return {"clock_period_ns": 3.0, "timing_exception_counts": hits,
            "reset_false_pathed": False}


def check_tcl() -> dict[str, Any]:
    text = TCL.read_text()
    required = (
        "STORAGE_MAPPING_MANIFEST", "split $line \"|\"",
        "represented_bytes", "internal_macro_count",
        "external_common_charge_bytes", "mapping gap/overlap",
        "get_cells -hierarchical -filter \"ref_name == $macro_cell\"",
        "standard_cell_logic_area", "internal_parent_macro_area",
        "physical_dc_total_area", "external_common_charge_area_um2=UNMODELED_EXCLUDED",
        "full_214912B_total_area_um2=NOT_ADMITTED", "compile_ultra -no_autoungroup",
    )
    for token in required:
        require(token in text, "DC Tcl missing fail-closed token: " + token)
    require("set expected_macro_count 93" not in text and
            "set expected_macro_count 105" not in text,
            "dummy macro target in Tcl")
    require(text.count("compile_ultra -no_autoungroup") == 1 and
            "compile_ultra -incremental" not in text,
            "compile count drift")
    return {"mapping_derived": True, "literal_93_or_105_macro_target": False,
            "compile_ultra_count": 1, "incremental_compile_count": 0,
            "reports_logic_parent_macro_physical_total_separately": True,
            "external_common_charge_area_unmodeled": True}


def check_contract(contract_path: Path = CONTRACT) -> dict[str, Any]:
    value = strict_json(contract_path)
    require(value["status"] ==
            "PASS_M1116C_ADDITIVE_FULL_STORAGE_COMMON_CHARGE_SOURCE_ONLY__HAMMER_REQUIRED__NO_EDA",
            "contract state drift")
    identity = value["source_identity"]
    for key, path in (("wrapper", WRAPPER), ("mapping_manifest", MAPPING),
                      ("filelist", FILELIST), ("sdc", SDC), ("dc_tcl", TCL)):
        require(identity[key]["sha256"] == sha256(path), "contract identity drift: " + key)
    boundary = value["claim_boundary"]
    false_keys = (
        "vcs_executed", "dc_executed", "pt_executed", "formality_executed",
        "ptpx_executed", "full_214912B_physically_integrated",
        "external_common_charge_numeric_area", "setup_or_hold",
        "power_or_energy", "rtl_cycles", "speedup", "throughput_per_area",
        "system_speedup", "paper_ppa_ready", "paper_citable_performance")
    require(all(boundary[key] is False for key in false_keys),
            "contract claim boundary weakened")
    require(value["authorization"]["different_author_source_hammer_next"] is True and
            all(value["authorization"][key] is False for key in
                ("vcs_now", "dc_now", "pt_now", "formality_now", "ptpx_now")),
            "contract execution authorization drift")
    return {"status": value["status"], "false_claim_keys": list(false_keys)}


def self_test(contract_path: Path = CONTRACT) -> dict[str, Any]:
    require(sha256(M935) == M935_SHA, "frozen M935 modified")
    require(sha256(PARENT) == PARENT_SHA, "frozen parent wrapper modified")
    require(sha256(DOC359) == DOC359_SHA, "docs359 modified")
    mapping = parse_mapping()
    wrapper = check_wrapper()
    filelist = check_filelist()
    sdc = check_sdc()
    tcl = check_tcl()
    contract = check_contract(contract_path)
    result_glob = list((HW / "dc_handoff/runs").glob("m1116c_m935_c1_full_storage_dc_3p000ns_r1_20260830"))
    require(not result_glob, "M1116C DC result exists during source-only authoring")
    return {
        "schema": "m1116c_full_storage_common_charge_source_check_v1",
        "status": "PASS_M1116C_SOURCE_ONLY_SELF_TEST__HAMMER_REQUIRED__NO_EDA",
        "mapping": mapping["totals"], "wrapper": wrapper,
        "filelist": filelist, "sdc": sdc, "dc_tcl": tcl,
        "contract": contract,
        "frozen_m935_sha256": M935_SHA,
        "docs359_sha256": DOC359_SHA,
        "vcs_dc_pt_fm_ptpx_executed": False,
        "full_214912B_physically_integrated": False,
        "external_common_charge_numeric_area": False,
        "speedup_admitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    require(args.self_test, "source-only checker requires --self-test")
    print(json.dumps(self_test(args.contract), indent=2, sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
