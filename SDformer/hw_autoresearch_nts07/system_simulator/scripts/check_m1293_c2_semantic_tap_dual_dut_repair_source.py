#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fail-closed source checker for additive M1293; never invokes HDL/EDA tools."""
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
CONTRACT = HW / "contracts/m1293_c2_semantic_tap_dual_dut_repair_source_contract_r1_20260830.json"
ENDPOINT = HW / "dc_handoff/tb/m1293_valid_qualified_scalar_bank_endpoint.sv"
TB = HW / "dc_handoff/tb/tb_m1293_c2_dual_dut_functional_reachability.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1293_c2_dual_dut_source_only_vcs.f"
TOP = HW / "rtl_m1279/m1279_c2_k1_semantic_tap_wrapper.sv"

UPSTREAM = {
    "contracts/m1279_c2_semantic_tap_dual_dut_source_only_contract_r1_20260830.json":
        "a93fa602788d6e6fe89f0260fe2f9ce8b3468212ca1c718d8c16c01beaa63bf4",
    "reviews/m1279_c2_semantic_tap_dual_dut_source_receipt_r1_20260830/SHA256SUMS.seal.sha256":
        "eed04b292c7fcb5aed30c7a2e7c191f65c3fdaa082a81af314f49e43c0e74d2f",
    "reviews/m1287_m1279_c2_semantic_tap_dual_dut_receipt_blind_hammer_r1_20260830/SHA256SUMS.seal.sha256":
        "3d77ef035e4f29369fca510b06fd4fd7856ebb9a5c9fc8872bbf024b2d7334b9",
}

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

DIRECT_TAP_RHS = {
    "tap_core_mem_req_accept": "core_mem_req_accept",
    "tap_adapter_core_mem_req_accept": "adapter_core_mem_req_accept",
    "tap_core_mem_rsp_accept": "core_mem_rsp_accept",
    "tap_adapter_core_mem_rsp_accept": "adapter_core_mem_rsp_accept",
    "tap_consistency_fault_now": "consistency_fault_now",
    "tap_consistency_fault_q": "consistency_fault_q",
    "tap_core_protocol_error": "core_protocol_error",
    "tap_adapter_protocol_error": "adapter_protocol_error",
}

HIERARCHICAL_TAP_PORT = {
    "tap_frontend_compactor_fault_q": "tap_frontend_compactor_fault_q",
    "tap_frontend_paired_sink_fault_q": "tap_frontend_paired_sink_fault_q",
    "tap_core_adapter_fault_q": "tap_core_adapter_fault_q",
    "tap_service_fault_q": "tap_service_fault_q",
    "tap_memory_adapter_fault_q": "tap_fault_q",
}

FROZEN_RTL = (
    "rtl_m1279/m1279_fc2_raw4_to_descriptor4_terminal_hint_compactor_tapped.sv",
    "rtl_m1279/m1279_fc2_descriptor4_source_cap_frontend_tapped.sv",
    "rtl_m1279/m1279_fc2_raw4_to_source_cap_frontend_tapped.sv",
    "rtl_m1279/m1279_fc2_k1_reset_hygiene_registered_release_service_island_tapped.sv",
    "rtl_m1279/m1279_fc2_reset_hygiene_registered_release_standalone_raw4_acc24_tapped.sv",
    "rtl_m1279/m1279_fc2_bundle_to_8bank_no_reuse_adapter_tapped.sv",
    "rtl_m1279/m1279_c2_k1_semantic_tap_wrapper.sv",
)

EXPECTED_FILELIST = (
    *FROZEN_RTL[:3],
    "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
    *FROZEN_RTL[3:],
    "tb_m349/m349_fc2_scalar_bank_memory_model.sv",
    "dc_handoff/tb/m1293_valid_qualified_scalar_bank_endpoint.sv",
    "dc_handoff/tb/tb_m1293_c2_dual_dut_functional_reachability.sv",
)

ENDPOINT_GUARD_TOKEN_SHA = "9b6e504a6d7d7bf4cae7dfa1cd005535a37b5a552ba47ecb2a979420df6e173d"
TB_COMPARE_TOKEN_SHA = "833c00145a86b41ea09ca9b405d71d6d284bb4b13dc2ef3563be275b3cffbd9e"
TB_ATOMIC_TOKEN_SHA = "e651e1cf23cb88abc6fa34172e2aa5442e253ef9e950d186849926ca94a1a539"

CLAIM_BOUNDARY = {
    "source_only": True,
    "k1_diagnostic_axis_only": True,
    "k8_present": False,
    "equal_bandwidth_k1x8_present": False,
    "vcs": False,
    "dc": False,
    "pt": False,
    "ptpx": False,
    "gpu": False,
    "remote": False,
    "saif": False,
    "single_k1_power_admitted": False,
    "fair_energy_comparison_admitted": False,
    "performance_admitted": False,
    "mapped_functionality": False,
    "system_speedup": False,
    "paper_ppa_ready": False,
    "paper_headline": False,
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
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            Failure("nonfinite JSON: " + token)))


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


TOKEN_RE = re.compile(
    r"\$[A-Za-z_][A-Za-z0-9_$]*|[A-Za-z_][A-Za-z0-9_$]*|"
    r"(?:\d+)?'[sS]?[bBdDhHoO][0-9a-fA-F_xXzZ?]+|\d+|"
    r"===|!==|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|\+=|-=|"
    r"[(){}\[\]:;,.#@?~!%^&*+\-/=<>|]"
)


def tokens(text: str) -> list[str]:
    clean = strip_comments(text)
    output = TOKEN_RE.findall(clean)
    require(output, "empty token stream")
    return output


def named_block_tokens(text: str, prefix: tuple[str, ...]) -> tuple[str, ...]:
    stream = tokens(text)
    start = -1
    for index in range(len(stream) - len(prefix) + 1):
        if tuple(stream[index:index + len(prefix)]) == prefix:
            require(start < 0, "block prefix is not unique: " + " ".join(prefix))
            start = index
    require(start >= 0, "named block missing: " + " ".join(prefix))
    require(prefix.count("begin") == 1, "prefix must contain one begin")
    begin_index = start + prefix.index("begin")
    depth = 0
    for index in range(begin_index, len(stream)):
        if stream[index] == "begin":
            depth += 1
        elif stream[index] == "end":
            depth -= 1
            if depth == 0:
                return tuple(stream[start:index + 1])
    raise Failure("unterminated named block")


def token_sha(block: tuple[str, ...]) -> str:
    return hashlib.sha256("\x1f".join(block).encode()).hexdigest()


def check_endpoint_text(text: str) -> dict[str, Any]:
    block = named_block_tokens(text,
        ("always_comb", "begin", ":", "valid_qualified_guard",))
    require(token_sha(block) == ENDPOINT_GUARD_TOKEN_SHA,
            "endpoint normalized guard block drift")
    payload_fields = ("mem_req_epoch", "mem_req_slot", "mem_req_generation",
        "mem_req_tag", "mem_req_output_block", "mem_req_slice",
        "mem_req_source_channel")
    for field in payload_fields:
        require(field in block, "endpoint payload-known field missing: " + field)
    required_subsequences = (
        ("mem_req_valid", "===", "1'b1"),
        ("if", "(", "request_payload_known", ")", "begin"),
        ("mem_req_accept", "===", "1'b1"),
        ("mem_req_accept", "!==", "1'b0"),
        ("mem_req_valid", "!==", "1'b0"),
    )
    joined = "\x1f".join(block)
    for sequence in required_subsequences:
        require("\x1f".join(sequence) in joined,
                "endpoint structural guard missing: " + " ".join(sequence))
    clean = strip_comments(text)
    require(clean.count(".mem_req_valid(qualified_request_valid)") == 1 and
            clean.count(".mem_req_accept(qualified_request_accept)") == 1,
            "endpoint inner qualified binding drift")
    return {"guard_token_sha256": token_sha(block),
            "payload_fields": len(payload_fields), "unconditional_gate": False}


def check_tap_exact_rhs_text(text: str) -> dict[str, Any]:
    clean = strip_comments(text)
    declared = re.findall(
        r'\(\*\s*keep\s*=\s*"true"\s*\*\)\s*output\s+logic\s+(tap_[A-Za-z0-9_]+)',
        clean)
    require(len(declared) == 13 and set(declared) == set(TAPS),
            "exact kept tap declaration drift")
    for tap in TAPS:
        expected_occurrences = 2
        if tap in HIERARCHICAL_TAP_PORT and HIERARCHICAL_TAP_PORT[tap] == tap:
            expected_occurrences = 3
        require(len(re.findall(r"\b" + re.escape(tap) + r"\b", clean)) ==
                expected_occurrences,
                "tap occurrence/fanout count drift: " + tap)
    for tap, rhs in DIRECT_TAP_RHS.items():
        match = re.search(r"assign\s+" + re.escape(tap) + r"\s*=\s*([^;]+);", clean)
        require(match is not None and re.sub(r"\s+", "", match.group(1)) == rhs,
                "tap exact RHS drift: " + tap)
        statement = match.group(0)
        for forbidden in ("$isunknown", "===", "!==", "?", "1'bx", "1'bz"):
            require(forbidden.lower() not in statement.lower(),
                    "tap X coercion forbidden: " + tap + " / " + forbidden)
    for tap, port in HIERARCHICAL_TAP_PORT.items():
        pattern = r"\." + re.escape(port) + r"\s*\(\s*" + re.escape(tap) + r"\s*\)"
        require(len(re.findall(pattern, clean)) == 1,
                "tap exact hierarchical RHS drift: " + tap)
    require("implementation." not in clean and "anonymous" not in clean.lower(),
            "hierarchical/anonymous tap binding prohibited")
    return {"tap_count": 13, "direct_exact_rhs": 8,
            "hierarchical_exact_rhs": 5, "x_coercion": False,
            "functional_fanin_from_taps": False}


def check_tb_text(text: str) -> dict[str, Any]:
    compare = named_block_tokens(text,
        ("always", "@", "(", "posedge", "clk_core", ")", "begin", ":",
         "transaction_class_compare"))
    atomic = named_block_tokens(text,
        ("always", "@", "(", "posedge", "clk_core", ")", "begin", ":",
         "atomic_window"))
    require(token_sha(compare) == TB_COMPARE_TOKEN_SHA,
            "TB transaction-class compare block drift")
    require(token_sha(atomic) == TB_ATOMIC_TOKEN_SHA,
            "TB atomic reachability/PASS block drift")
    clean = strip_comments(text)
    require(clean.count("m1293_c2_k1_diagnostic_system #(") == 3,
            "TB must contain one system declaration and exactly two instances")
    for token in (
        "request_count_original <= 0", "request_count_qualified <= 0",
        "result_count_original <= 0", "result_count_qualified <= 0",
        "done_count_original <= 0", "done_count_qualified <= 0",
        "first_result_cycle <= first_request_cycle",
        "first_done_cycle < first_request_cycle",
        "request_class_mismatch_count != 0",
        "result_class_mismatch_count != 0",
        "done_class_mismatch_count != 0",
        "req_accept_original !== req_accept_qualified",
        "result_accept_original !== result_accept_qualified",
        "done_accept_original !== done_accept_qualified",
    ):
        require(token in clean, "TB reachability/class compare missing: " + token)
    pass_token = "PASS_M1293_DUAL_DUT_FUNCTIONAL_REACHABILITY"
    atomic_text = "\x1f".join(atomic)
    require(clean.count(pass_token) == 2 and atomic_text.count(pass_token) == 2,
            "PASS token must occur only inside guarded atomic block")
    prohibited = (r"\bforce\b", r"\brelease\b", r"\+?initreg", r"\bcasex\b",
        r"\bcasez\b", r"set_case_analysis", r"=\s*1'b[xXzZ]")
    for pattern in prohibited:
        require(re.search(pattern, clean, flags=re.I) is None,
                "prohibited TB mechanism: " + pattern)
    return {"dual_dut_instances": 2, "transaction_classes": 3,
            "request_reachability_required": True,
            "result_reachability_required": True,
            "token_done_reachability_required": True,
            "endpoint_can_be_unreached_and_pass": False,
            "compare_token_sha256": token_sha(compare),
            "atomic_token_sha256": token_sha(atomic), "window_cycles": 256}


def check_filelist() -> dict[str, Any]:
    regular(FILELIST)
    members = tuple(line.strip() for line in FILELIST.read_text(encoding="utf-8").splitlines()
                    if line.strip())
    require(members == EXPECTED_FILELIST and len(set(members)) == len(members),
            "filelist exact order/set drift")
    for member in members:
        regular(HW / member)
    return {"members": len(members), "exact_order": True,
            "eda_compile_or_elaboration": False}


def exact_bool_map(value: Any, expected: dict[str, bool], label: str) -> None:
    require(type(value) is dict and set(value) == set(expected), label + " keyset drift")
    for key, wanted in expected.items():
        require(type(value[key]) is bool and value[key] is wanted,
                label + " exact bool drift: " + key)


def check_contract_data(data: Any, validate_source_hashes: bool = True) -> dict[str, Any]:
    top_keys = {"schema", "status", "date", "launch_now", "purpose", "sources",
        "upstream", "semantic_taps", "tap_exact_rhs", "endpoint_contract",
        "tb_contract", "claim_boundary", "next_gate", "docs359_sha256"}
    require(type(data) is dict and set(data) == top_keys, "contract top keyset drift")
    require(data["schema"] ==
        "m1293_c2_semantic_tap_dual_dut_repair_source_contract_r1" and
        data["status"] == "SOURCE_ONLY_REPAIR_NOT_RELEASED" and
        type(data["launch_now"]) is bool and data["launch_now"] is False,
        "contract identity/launch drift")
    require(data["date"] == "2026-08-30" and type(data["purpose"]) is str and
        type(data["next_gate"]) is str, "contract scalar drift")
    require(data["upstream"] == UPSTREAM, "contract upstream drift")
    require(type(data["semantic_taps"]) is list and
        tuple(data["semantic_taps"]) == TAPS, "contract tap set/order drift")
    tap_rhs_expected = {**DIRECT_TAP_RHS,
        **{tap: "port:" + port for tap, port in HIERARCHICAL_TAP_PORT.items()}}
    require(data["tap_exact_rhs"] == tap_rhs_expected, "contract exact tap RHS drift")
    exact_bool_map(data["endpoint_contract"], {
        "normalized_token_block_guard": True,
        "valid_must_equal_one": True,
        "payload_must_be_fully_known": True,
        "unknown_is_quarantined_and_faulted": True,
        "unconditional_gate_forbidden": True,
    }, "endpoint_contract")
    exact_bool_map(data["tb_contract"], {
        "dual_dut_same_stimulus": True,
        "request_bank_accept_reachability_required": True,
        "result_reachability_required": True,
        "token_done_reachability_required": True,
        "transaction_class_functional_compare_required": True,
        "endpoint_unreached_pass_forbidden": True,
        "unknown_or_fault_fail_closed": True,
    }, "tb_contract")
    exact_bool_map(data["claim_boundary"], CLAIM_BOUNDARY, "claim_boundary")
    require(data["docs359_sha256"] == DOCS_SHA, "contract docs359 drift")
    expected_sources = {
        **{path: sha256(HW / path) for path in FROZEN_RTL},
        "dc_handoff/tb/m1293_valid_qualified_scalar_bank_endpoint.sv": sha256(ENDPOINT),
        "dc_handoff/tb/tb_m1293_c2_dual_dut_functional_reachability.sv": sha256(TB),
        "dc_handoff/filelists/date_m1293_c2_dual_dut_source_only_vcs.f": sha256(FILELIST),
        "system_simulator/scripts/check_m1293_c2_semantic_tap_dual_dut_repair_source.py":
            sha256(HERE),
        "system_simulator/tests/test_m1293_c2_semantic_tap_dual_dut_repair_source.py":
            sha256(HW / "system_simulator/tests/test_m1293_c2_semantic_tap_dual_dut_repair_source.py"),
    }
    require(type(data["sources"]) is list, "contract sources type")
    actual = {}
    for row in data["sources"]:
        require(type(row) is dict and set(row) == {"path", "sha256"} and
            type(row["path"]) is str and type(row["sha256"]) is str,
            "contract source row schema drift")
        require(row["path"] not in actual, "duplicate contract source")
        actual[row["path"]] = row["sha256"]
    require(set(actual) == set(expected_sources), "contract source set drift")
    if validate_source_hashes:
        require(actual == expected_sources, "contract source hash drift")
    return data


def check_contract() -> dict[str, Any]:
    regular(CONTRACT)
    return check_contract_data(strict_json(CONTRACT))


def run_checks(include_contract: bool = True) -> dict[str, Any]:
    regular(DOCS); require(sha256(DOCS) == DOCS_SHA, "docs359 drift")
    for path, digest in UPSTREAM.items():
        regular(HW / path); require(sha256(HW / path) == digest,
            "upstream identity drift: " + path)
    for path in FROZEN_RTL:
        regular(HW / path)
    result = {
        "schema": "m1293_c2_semantic_tap_dual_dut_repair_source_check_r1",
        "status": "PASS_M1293_SOURCE_REPAIR__NO_EXECUTION_AUTHORIZED",
        "tap_exact_rhs": check_tap_exact_rhs_text(TOP.read_text(encoding="utf-8")),
        "endpoint": check_endpoint_text(ENDPOINT.read_text(encoding="utf-8")),
        "dual_dut_tb": check_tb_text(TB.read_text(encoding="utf-8")),
        "filelist": check_filelist(),
        "docs359_sha256": DOCS_SHA,
        "real_tool_calls": 0,
    }
    if include_contract:
        check_contract()
        result["contract_sha256"] = sha256(CONTRACT)
    return result


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, sort_keys=True, allow_nan=False))
