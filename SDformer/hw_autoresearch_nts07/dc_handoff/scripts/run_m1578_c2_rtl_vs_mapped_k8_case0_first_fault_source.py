#!/usr/bin/env python3
"""Static source gate for the M1578 C2 RTL/mapped first-fault diagnostic.

This file deliberately has no execution path.  It validates the frozen inputs,
the additive dual-DUT source, its exact filelist, and the non-claiming contract.
It cannot compile or run a simulator and cannot consume a production attempt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any


SOURCE = Path(__file__).resolve()
HW = SOURCE.parents[2]
CONTRACT = HW / "contracts/m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source_contract_r1_20260901.json"
TB = HW / "dc_handoff/tb/tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.f"
TEST = HW / "system_simulator/tests/test_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
            "TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
            "tcbn28hpcplusbwp35p140.v")

M1568 = HW / "reviews/m1568_m1502_c2_mapped_first_fault_forensic_r1_20260901"
M1502_FAILURE = HW / ("results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831."
                      "failed_or_incomplete.quarantine/failure.json")

FROZEN_PINS = {
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    CELL: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    M1568 / "review.json": "b88067a9ef94b24960d9d5ba86973b23c7b10a89386c9c624ffa82d8131081b2",
    M1568 / "SHA256SUMS": "279a60e1aaec03523da21f216ef9bbcc22eaba3daf75feb92a0d4976f2a17d71",
    M1568 / "SHA256SUMS.seal.sha256": "74a8848d6b082ce954d1182a2438ad7f2be6bce7fadda9c8b324feeee0e3bbc8",
    M1502_FAILURE: "2bad717f51fa99e2526b4ec8b7b305b4bbbf60b84728d6f799de59aa72bfe7d2",
    HW / "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv": "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    HW / "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv": "8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0",
    HW / "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv": "529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267",
    HW / "rtl_m218/m218_fc2_tagged_slice_service_island.sv": "f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1",
    HW / "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv": "44f7df331af66ba62fadf5e336b9c0c00d00f809e215aa8e091e9de011c5627e",
    HW / "rtl_m519/m519_fc2_k1_registered_release_service_island.sv": "3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871",
    HW / "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv": "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
    HW / "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv": "6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815",
    HW / "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv": "11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff",
    HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv": "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
    HW / "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv": "3328e52d8cf1eec6098ebb7b0525ac55cd8bd6b2fe5b5e504b337d1a678e3c4b",
    HW / ("dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829/"
          "k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"): "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
    HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv": "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
}

EXPECTED_FILELIST = (
    str(CELL),
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


class Failure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_regular(path: Path, expected: str | None = None) -> None:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
            "nonregular identity: " + str(path))
    if expected is not None:
        require(sha256(path) == expected, "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + value)))


def strip_sv_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def verify_double_seal(path: Path) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path); verify_regular(side); verify_regular(outer)
    contract_hash = sha256(path)
    side_hash = sha256(side)
    require(side.read_text(encoding="utf-8").split() ==
            [contract_hash, path.name], "contract inner seal drift")
    require(outer.read_text(encoding="utf-8").split() ==
            [side_hash, side.name], "contract outer seal drift")


def verify_review_tree() -> dict[str, Any]:
    manifest = M1568 / "SHA256SUMS"
    outer = M1568 / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="utf-8").split() ==
            [sha256(manifest), manifest.name], "M1568 outer seal drift")
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        fields = row.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "M1568 manifest row")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(name not in listed and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "M1568 manifest member")
        listed[name] = fields[0]
    actual = {member.relative_to(M1568).as_posix() for member in M1568.rglob("*")
              if member.is_file() and member.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(listed), "M1568 manifest set drift")
    for name, expected in listed.items():
        verify_regular(M1568 / name, expected)
    return strict_json(M1568 / "review.json")


def verify_tb_text(text: str) -> dict[str, int]:
    active = strip_sv_comments(text)
    required = (
        "module m1578_case0_memory_fabric",
        "module tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault",
        "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24 #(",
        ".ARCH_MODE(1)",
        ") rtl_dut (",
        "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_ARCH_MODE1",
        "mapped_dut (",
        "m1578_case0_memory_fabric rtl_memory",
        "m1578_case0_memory_fabric mapped_memory",
        "header_tag = 24'h979000",
        "header_raw_beat_count = 6'd4",
        "header_window_depth = 4'd2",
        "header_output_blocks = 4'd1",
        "protocol_error",
        "numeric_overflow",
        "stale_response_seen",
        "rtl_endpoint_fault",
        "mapped_endpoint_fault",
        "rtl_internal_fault_taps",
        "mapped_internal_fault_taps",
        "M1578_TRACE",
        "header=%s/%s",
        "source=%s/%s",
        "endpoint=%s/%s",
        "mem=%s/%s",
        "commit=%s/%s",
        "done=%s/%s",
        "first_difference_cycle",
        "first_fault_cycle",
        "FIRST_RTL_MAPPED_DIFFERENCE",
        "FAULT_OR_X",
        "BOTH_CLEAN_TO_DONE",
        "value === 1'b0",
        "value === 1'b1",
        "else tri = \"X\"",
        "$isunknown(value)",
        "rtl_protocol_error !== 1'b0",
        "mapped_protocol_error !== 1'b0",
        "!== {mapped_header_accept",
    )
    missing = [token for token in required if token not in active]
    require(not missing, "TB semantic token missing: " + repr(missing))
    require(active.count(") rtl_dut (") == 1 and
            active.count("mapped_dut (") == 1,
            "exactly one RTL and one mapped DUT required")
    require(active.count("m1578_case0_memory_fabric rtl_memory") == 1 and
            active.count("m1578_case0_memory_fabric mapped_memory") == 1,
            "independent identical memories required")
    require("$stop" not in active and "assert property" not in active and
            "cover property" not in active and " bind " not in active,
            "runtime assertion/control mechanism prohibited")
    lower = active.lower()
    for token in ("ucli", "initreg", "saif", "ptpx"):
        require(token not in lower, "prohibited mechanism in active TB: " + token)
    require(re.search(r"\bforce\b", active) is None and
            re.search(r"\brelease\b", active) is None,
            "force/release prohibited")
    require("else tri = \"0\"" not in active,
            "X-to-zero coercion prohibited")
    return {"dut_instances": 2, "memory_instances": 2,
            "named_internal_taps_per_dut": 6, "top_fault_bits_per_dut": 3,
            "endpoint_fault_bits_per_dut": 8}


def validate_contract_obj(contract: dict[str, Any]) -> None:
    require(contract.get("schema") ==
            "m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source_contract_r1_v1",
            "contract schema")
    execution = contract.get("execution", {})
    require(execution == {"vcs_compiles": 0, "simv_runs": 0,
                          "ucli": False, "initreg": False, "saif": False,
                          "ptpx": False, "attempt_consumed": False},
            "source-only execution boundary")
    claims = contract.get("claim_boundary", {})
    require(claims == {"diagnostic_only": True, "paper_citable": False,
                       "rtl_pass": False, "mapped_pass": False,
                       "timing_verified": False, "power": False,
                       "ppa": False, "system_speedup": False,
                       "headline": False}, "claim promotion")
    require(contract.get("future_execution", {}).get("different_author_hammer_required")
            is True, "independent hammer required")
    require(contract.get("future_execution", {}).get("budget") ==
            {"vcs_compiles": 1, "simv_runs": 1, "cases": ["k8_case0"],
             "saif": 0, "ptpx": 0}, "future budget drift")
    require(contract.get("diagnostic", {}).get("stop_on_first_difference") is True and
            contract.get("diagnostic", {}).get("stop_on_first_fault_or_x") is True and
            contract.get("diagnostic", {}).get("four_state_reporting") == "0/1/X",
            "first-fault diagnostic drift")


def static_check() -> dict[str, Any]:
    for path, expected in FROZEN_PINS.items():
        verify_regular(path, expected)
    review = verify_review_tree()
    require(review["one_run_successor_gate"]["independent_source_hammer_before_run"]
            is True and review["one_run_successor_gate"]["initreg_for_root_diagnostic"]
            is False, "M1568 successor boundary drift")
    failure = strict_json(M1502_FAILURE)
    require(failure["phase"] == "SIM_k8_0" and failure["counts"] ==
            {"ptpx_runs": 0, "saif_files": 0, "simv_runs": 1,
             "vcs_compiles": 1}, "M1502 sealed failure drift")

    rows = tuple(row for row in FILELIST.read_text(encoding="utf-8").splitlines()
                 if row.strip())
    require(rows == EXPECTED_FILELIST, "exact ordered filelist drift")
    for row in rows:
        require(not row.startswith("+"), "defines/options prohibited in filelist")
        path = Path(row) if Path(row).is_absolute() else HW / row
        verify_regular(path)

    tb_summary = verify_tb_text(TB.read_text(encoding="utf-8"))
    verify_double_seal(CONTRACT)
    contract = strict_json(CONTRACT)
    validate_contract_obj(contract)
    expected_sources = contract.get("source_sha256", {})
    actual_sources = {path.relative_to(HW).as_posix(): sha256(path)
                      for path in (TB, FILELIST, SOURCE, TEST)}
    require(expected_sources == actual_sources, "new source identity drift")
    return {
        "schema": "m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source_check_v1",
        "status": "PASS_SOURCE_ONLY_READY_FOR_INDEPENDENT_HAMMER__NO_TOOL_RUN",
        "tb": tb_summary,
        "filelist_entries": len(rows),
        "frozen_pins": len(FROZEN_PINS),
        "execution": {"vcs_compiles": 0, "simv_runs": 0,
                      "saif": 0, "ptpx": 0, "attempt_consumed": False},
        "claim": False,
    }


def describe() -> dict[str, Any]:
    return {
        "schema": "m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source_description_v1",
        "purpose": "four-state first-fault localization only",
        "dut_pair": ["frozen RTL ARCH_MODE=1", "frozen mapped ARCH_MODE1"],
        "case": "exact M979 K8 case0",
        "memory": "two independent instances of the same reset-safe model",
        "events": ["header", "source", "endpoint", "memory", "commit", "done"],
        "faults": ["protocol_error", "numeric_overflow",
                   "stale_response_seen", "endpoint_fault[7:0]",
                   "six named internal taps"],
        "execution": {"vcs_compiles": 0, "simv_runs": 0,
                      "saif": 0, "ptpx": 0, "attempt_consumed": False},
    }


def main() -> int:
    require(len(sys.argv) == 2 and sys.argv[1] in {"--describe", "--static-check"},
            "only --describe or --static-check is supported")
    result = describe() if sys.argv[1] == "--describe" else static_check()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
