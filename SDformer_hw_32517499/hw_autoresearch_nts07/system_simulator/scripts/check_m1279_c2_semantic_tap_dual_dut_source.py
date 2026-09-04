#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Static/fail-closed checker for the M1279 source-only milestone."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any

HERE = Path(__file__).resolve()
HW = HERE.parent.parent.parent
DOCS = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONTRACT = HW / "contracts/m1279_c2_semantic_tap_dual_dut_source_only_contract_r1_20260830.json"

TAPS = (
    "tap_frontend_compactor_fault_q",
    "tap_frontend_paired_sink_fault_q",
    "tap_core_adapter_fault_q",
    "tap_service_fault_q",
    "tap_memory_adapter_fault_q",
    "tap_core_mem_req_accept",
    "tap_adapter_core_mem_req_accept",
    "tap_core_mem_rsp_accept",
    "tap_adapter_core_mem_rsp_accept",
    "tap_consistency_fault_now",
    "tap_consistency_fault_q",
    "tap_core_protocol_error",
    "tap_adapter_protocol_error",
)

CLONES = (
    ("rtl_m1279/m1279_fc2_raw4_to_descriptor4_terminal_hint_compactor_tapped.sv",
     "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"),
    ("rtl_m1279/m1279_fc2_descriptor4_source_cap_frontend_tapped.sv",
     "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv"),
    ("rtl_m1279/m1279_fc2_raw4_to_source_cap_frontend_tapped.sv",
     "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv"),
    ("rtl_m1279/m1279_fc2_k1_reset_hygiene_registered_release_service_island_tapped.sv",
     "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_service_island.sv"),
    ("rtl_m1279/m1279_fc2_reset_hygiene_registered_release_standalone_raw4_acc24_tapped.sv",
     "rtl_m1058/m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24.sv"),
    ("rtl_m1279/m1279_fc2_bundle_to_8bank_no_reuse_adapter_tapped.sv",
     "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv"),
    ("rtl_m1279/m1279_c2_k1_semantic_tap_wrapper.sv",
     "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24.sv"),
)

NAME_MAP = {
    "m1279_fc2_raw4_to_descriptor4_terminal_hint_compactor_tapped":
        "m214_fc2_raw4_to_descriptor4_terminal_hint_compactor",
    "m1279_fc2_descriptor4_source_cap_frontend_tapped":
        "m216_fc2_descriptor4_source_cap_frontend",
    "m1279_fc2_raw4_to_source_cap_frontend_tapped":
        "m216_fc2_raw4_to_source_cap_frontend",
    "m1279_fc2_k1_reset_hygiene_registered_release_service_island_tapped":
        "m1058_fc2_k1_reset_hygiene_registered_release_service_island",
    "m1279_fc2_reset_hygiene_registered_release_standalone_raw4_acc24_tapped":
        "m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24",
    "m1279_fc2_bundle_to_8bank_no_reuse_adapter_tapped":
        "m499_fc2_bundle_to_8bank_no_reuse_adapter",
    "m1279_c2_k1_semantic_tap_wrapper":
        "m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24",
}

UPSTREAM = {
    "reviews/m1151r6_m1146r6_c2_case0_x_failure_audit_r1_20260830/SHA256SUMS.seal.sha256":
        "72bf8c7500a45961aefada1cb3b720bfc0b357eb7e08257379015fb6c1288c5f",
    "reviews/m1155r6_m1154r6_c2_dual_dut_source_stop_hammer_r1_20260830/SHA256SUMS.seal.sha256":
        "f27a738dc55a06de9d9cb906c395b9ec94dcfd7b0fd0ba84527bec29700e039d",
    "reviews/m1274_c2_mapped_saif_ptpx_chain_readonly_audit_r1_20260830/review.md":
        "381f5e662a00fea65fb7b8f39beee4f3cb2036144c9e6aa759d375794af15de7",
}


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def functional_normal_form(text: str) -> str:
    text = strip_comments(text)
    for successor, frozen in NAME_MAP.items():
        text = text.replace(successor, frozen)
    lines = []
    for line in text.splitlines():
        if "tap_" in line:
            if line.strip().endswith("));"):
                lines.append(");")
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r",\s*\);", ");", text)
    return re.sub(r"\s+", "", text)


def check_clone_equivalence() -> list[dict[str, str]]:
    rows = []
    for successor_name, frozen_name in CLONES:
        successor, frozen = HW / successor_name, HW / frozen_name
        regular(successor); regular(frozen)
        successor_nf = functional_normal_form(successor.read_text(encoding="utf-8"))
        frozen_nf = functional_normal_form(frozen.read_text(encoding="utf-8"))
        require(successor_nf == frozen_nf,
                "functional clone drift: " + successor_name)
        rows.append({"successor": successor_name, "successor_sha256": sha256(successor),
                     "frozen": frozen_name, "frozen_sha256": sha256(frozen)})
    return rows


def check_tap_topology() -> dict[str, Any]:
    top_path = HW / "rtl_m1279/m1279_c2_k1_semantic_tap_wrapper.sv"
    top = strip_comments(top_path.read_text(encoding="utf-8"))
    declared = set(re.findall(r"output\s+logic\s+(tap_[A-Za-z0-9_]+)", top))
    require(declared == set(TAPS), "top tap declaration set drift")
    for tap in TAPS:
        require(top.count("output logic " + tap) == 1,
                "tap declaration multiplicity: " + tap)
        require("(* keep = \"true\" *) output logic " + tap in top,
                "tap keep attribute missing: " + tap)
    require("implementation." not in top and "anonymous" not in top.lower(),
            "hierarchical/anonymous tap binding prohibited")

    for path_name, _ in CLONES:
        text = strip_comments((HW / path_name).read_text(encoding="utf-8"))
        for line in text.splitlines():
            if "tap_" not in line:
                continue
            if line.lstrip().startswith("module "):
                continue
            compact = re.sub(r"\s+", "", line)
            allowed = ("outputlogictap_" in compact or
                       compact.startswith("assigntap_") or
                       re.fullmatch(r"\.tap_[A-Za-z0-9_]+\([^;]+\)[,;]?", compact)
                       is not None)
            require(allowed, "tap appears outside declaration/read-only fanout: " + line)
    return {"tap_count": len(TAPS), "tap_names": list(TAPS),
            "top_sha256": sha256(top_path), "hierarchical_binding": False,
            "functional_fanin_from_taps": False}


def check_endpoint_and_tb() -> dict[str, Any]:
    endpoint_path = HW / "dc_handoff/tb/m1279_valid_qualified_scalar_bank_endpoint.sv"
    tb_path = HW / "dc_handoff/tb/tb_m1279_c2_dual_dut_valid_qualified_endpoint.sv"
    filelist_path = HW / "dc_handoff/filelists/date_m1279_c2_dual_dut_source_only_vcs.f"
    for path in (endpoint_path, tb_path, filelist_path): regular(path)
    endpoint = endpoint_path.read_text(encoding="utf-8")
    tb = tb_path.read_text(encoding="utf-8")
    joined = endpoint + "\n" + tb
    prohibited = (r"\bforce\b", r"\brelease\b", r"\+?initreg", r"\bcasex\b",
                  r"\bcasez\b", r"set_case_analysis", r"=\s*1'b[xXzZ]")
    for pattern in prohibited:
        require(re.search(pattern, strip_comments(joined), flags=re.I) is None,
                "prohibited diagnostic mechanism: " + pattern)
    for token in ("mem_req_valid === 1'b1", "request_payload_known",
                  "mem_req_ready = 1'b0", "endpoint_protocol_fault_now = 1'b1",
                  ".mem_req_valid(qualified_request_valid)",
                  ".mem_req_accept(qualified_request_accept)"):
        require(token in endpoint, "qualified endpoint gate missing: " + token)
    require(tb.count("m1279_c2_k1_diagnostic_system #(") == 3,
            "dual DUT instance count")
    for token in (".VALID_QUALIFIED_ENDPOINT(1'b0)",
                  ".VALID_QUALIFIED_ENDPOINT(1'b1)",
                  "logic [31:0] sample_unknown_bitmap",
                  "for (integer tap = 0; tap < 13; tap++)",
                  "next_union = unknown_union_bitmap | sample_unknown_bitmap",
                  "if (window_cycle == 128)",
                  "if (qualified_unknown)",
                  "PASS_M1279_DUAL_DUT_ROOT_DIAGNOSTIC",
                  "$fatal", "$finish"):
        require(token in tb, "executable dual-DUT/atomic fail-close missing: " + token)
    members = [line.strip() for line in filelist_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    require(len(members) == len(set(members)) == 11, "filelist member count/duplicate")
    for name in members: regular(HW / name)
    require(members[-2:] == [
        "dc_handoff/tb/m1279_valid_qualified_scalar_bank_endpoint.sv",
        "dc_handoff/tb/tb_m1279_c2_dual_dut_valid_qualified_endpoint.sv"],
        "filelist diagnostic tail order")
    return {"endpoint_sha256": sha256(endpoint_path), "tb_sha256": sha256(tb_path),
            "filelist_sha256": sha256(filelist_path), "filelist_members": len(members),
            "dual_dut_instances": 2, "atomic_bitmap_bits": 32,
            "semantic_taps_per_dut": 13, "window_cycles": 128}


def check_contract() -> dict[str, Any]:
    regular(CONTRACT)
    data = strict_json(CONTRACT)
    require(data["schema"] == "m1279_c2_semantic_tap_dual_dut_source_only_contract_r1",
            "contract schema")
    require(data["status"] == "SOURCE_ONLY_NOT_RELEASED" and
            data["launch_now"] is False, "contract launch boundary")
    for key in ("vcs", "dc", "pt", "ptpx", "gpu", "remote", "saif", "power",
                "energy", "mapped_functionality", "paper_ppa_ready"):
        require(data["claim_boundary"][key] is False, "claim elevated: " + key)
    expected = {path: sha256(HW / path) for path, _ in CLONES}
    expected.update({
        "dc_handoff/tb/m1279_valid_qualified_scalar_bank_endpoint.sv":
            sha256(HW / "dc_handoff/tb/m1279_valid_qualified_scalar_bank_endpoint.sv"),
        "dc_handoff/tb/tb_m1279_c2_dual_dut_valid_qualified_endpoint.sv":
            sha256(HW / "dc_handoff/tb/tb_m1279_c2_dual_dut_valid_qualified_endpoint.sv"),
        "dc_handoff/filelists/date_m1279_c2_dual_dut_source_only_vcs.f":
            sha256(HW / "dc_handoff/filelists/date_m1279_c2_dual_dut_source_only_vcs.f"),
        "system_simulator/scripts/check_m1279_c2_semantic_tap_dual_dut_source.py":
            sha256(HERE),
        "system_simulator/tests/test_m1279_c2_semantic_tap_dual_dut_source.py":
            sha256(HW / "system_simulator/tests/test_m1279_c2_semantic_tap_dual_dut_source.py"),
    })
    actual = {row["path"]: row["sha256"] for row in data["sources"]}
    require(actual == expected, "contract source set/hash drift")
    require(data["upstream"] == UPSTREAM, "contract upstream identity drift")
    require(data["docs359_sha256"] == DOCS_SHA, "contract docs359 drift")
    return data


def run_checks(include_contract: bool = True) -> dict[str, Any]:
    regular(DOCS); require(sha256(DOCS) == DOCS_SHA, "docs359 drift")
    for name, digest in UPSTREAM.items():
        regular(HW / name); require(sha256(HW / name) == digest, "upstream drift: " + name)
    result = {
        "schema": "m1279_c2_semantic_tap_dual_dut_source_check_r1",
        "status": "PASS_M1279_SOURCE_ONLY__NO_EXECUTION_AUTHORIZED",
        "clone_equivalence": check_clone_equivalence(),
        "tap_topology": check_tap_topology(),
        "dual_dut": check_endpoint_and_tb(),
        "docs359_sha256": DOCS_SHA,
        "real_tool_calls": 0,
    }
    if include_contract:
        check_contract()
        result["contract_sha256"] = sha256(CONTRACT)
    return result


if __name__ == "__main__":
    print(json.dumps(run_checks(), sort_keys=True, allow_nan=False, indent=2))
