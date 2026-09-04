#!/usr/bin/env python3
"""M1287 source/static/synthetic-only independent hammer of M1279.

This script never invokes VCS or any EDA tool and never reads the M1279 author
receipt.  It deliberately attacks the source checker as well as independently
recomputing the frozen-clone, tap, endpoint, TB and file-list properties.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
DOCS = HW / "docs/359_DATE终局冻结_20260813.md"
CONTRACT = HW / "contracts/m1279_c2_semantic_tap_dual_dut_source_only_contract_r1_20260830.json"
CHECKER = HW / "system_simulator/scripts/check_m1279_c2_semantic_tap_dual_dut_source.py"
ENDPOINT = HW / "dc_handoff/tb/m1279_valid_qualified_scalar_bank_endpoint.sv"
TB = HW / "dc_handoff/tb/tb_m1279_c2_dual_dut_valid_qualified_endpoint.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1279_c2_dual_dut_source_only_vcs.f"

TAPS = (
    "tap_frontend_compactor_fault_q", "tap_frontend_paired_sink_fault_q",
    "tap_core_adapter_fault_q", "tap_service_fault_q",
    "tap_memory_adapter_fault_q", "tap_core_mem_req_accept",
    "tap_adapter_core_mem_req_accept", "tap_core_mem_rsp_accept",
    "tap_adapter_core_mem_rsp_accept", "tap_consistency_fault_now",
    "tap_consistency_fault_q", "tap_core_protocol_error",
    "tap_adapter_protocol_error",
)

CLONES = (
    ("rtl_m1279/m1279_fc2_raw4_to_descriptor4_terminal_hint_compactor_tapped.sv",
     "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv",
     "m1279_fc2_raw4_to_descriptor4_terminal_hint_compactor_tapped",
     "m214_fc2_raw4_to_descriptor4_terminal_hint_compactor"),
    ("rtl_m1279/m1279_fc2_descriptor4_source_cap_frontend_tapped.sv",
     "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
     "m1279_fc2_descriptor4_source_cap_frontend_tapped",
     "m216_fc2_descriptor4_source_cap_frontend"),
    ("rtl_m1279/m1279_fc2_raw4_to_source_cap_frontend_tapped.sv",
     "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
     "m1279_fc2_raw4_to_source_cap_frontend_tapped",
     "m216_fc2_raw4_to_source_cap_frontend"),
    ("rtl_m1279/m1279_fc2_k1_reset_hygiene_registered_release_service_island_tapped.sv",
     "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_service_island.sv",
     "m1279_fc2_k1_reset_hygiene_registered_release_service_island_tapped",
     "m1058_fc2_k1_reset_hygiene_registered_release_service_island"),
    ("rtl_m1279/m1279_fc2_reset_hygiene_registered_release_standalone_raw4_acc24_tapped.sv",
     "rtl_m1058/m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24.sv",
     "m1279_fc2_reset_hygiene_registered_release_standalone_raw4_acc24_tapped",
     "m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24"),
    ("rtl_m1279/m1279_fc2_bundle_to_8bank_no_reuse_adapter_tapped.sv",
     "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
     "m1279_fc2_bundle_to_8bank_no_reuse_adapter_tapped",
     "m499_fc2_bundle_to_8bank_no_reuse_adapter"),
    ("rtl_m1279/m1279_c2_k1_semantic_tap_wrapper.sv",
     "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24.sv",
     "m1279_c2_k1_semantic_tap_wrapper",
     "m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24"),
)

NAME_MAP = {successor: frozen for _, _, successor, frozen in CLONES}

EXPECTED_SHA = {
    "rtl_m1279/m1279_c2_k1_semantic_tap_wrapper.sv": "1ae6fc8107367817123e12f0b1ff70722de65d129792282b87c0532435334f43",
    "rtl_m1279/m1279_fc2_bundle_to_8bank_no_reuse_adapter_tapped.sv": "ed0e31f74bfa3f424ab1783a7aaffe15dab32d3e75bb7b996301837513e3a950",
    "rtl_m1279/m1279_fc2_descriptor4_source_cap_frontend_tapped.sv": "9f00a0d6de0d4c7eb01156aac749fc808274d7aa0949d0f3de562cb7ab200826",
    "rtl_m1279/m1279_fc2_k1_reset_hygiene_registered_release_service_island_tapped.sv": "43b9684ae9f38c864e67d54322008f8d8767fec6fb11aed2185fbd16788ee743",
    "rtl_m1279/m1279_fc2_raw4_to_descriptor4_terminal_hint_compactor_tapped.sv": "4cb345112b62b5b5ab5195c96f42521fade76e1d9845633b2e7e8eafb576818b",
    "rtl_m1279/m1279_fc2_raw4_to_source_cap_frontend_tapped.sv": "0f5dd8fc9247fd16160646d27d481c3d2dfe5e1001be0d4c95525675106c96f6",
    "rtl_m1279/m1279_fc2_reset_hygiene_registered_release_standalone_raw4_acc24_tapped.sv": "dc2abbc2e2ce8dc955672d267fd0a51ea160ad05eb0b1af29e0ddb61c3de03e7",
    "dc_handoff/tb/m1279_valid_qualified_scalar_bank_endpoint.sv": "defe7c86b3aeaf41a8d9b794848895bbaf6409aee28b7aba81b13569f4a983f9",
    "dc_handoff/tb/tb_m1279_c2_dual_dut_valid_qualified_endpoint.sv": "f5cc756d7d632a7cda90ba9fcec872295de15fb32e35905c1d9c465b8237d28c",
    "dc_handoff/filelists/date_m1279_c2_dual_dut_source_only_vcs.f": "3e24a97d62bf095c2282cd6669a89513135dc459abbda2aae8503d390629d926",
    "system_simulator/scripts/check_m1279_c2_semantic_tap_dual_dut_source.py": "84dc1780f3b0dae447c6591b1835aafceb08846c68c316315ec6689762c8d475",
    "system_simulator/tests/test_m1279_c2_semantic_tap_dual_dut_source.py": "c6f8fba34e230ed5fd1330880d8f729807e777543bbb812caa44d420248843e1",
    "contracts/m1279_c2_semantic_tap_dual_dut_source_only_contract_r1_20260830.json": "a93fa602788d6e6fe89f0260fe2f9ce8b3468212ca1c718d8c16c01beaa63bf4",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def normalized_without_readonly_taps(text: str, successor: str, frozen: str) -> str:
    """Remove only syntactically read-only tap declarations/fanout/ports."""
    text = strip_comments(text)
    for successor_name, frozen_name in NAME_MAP.items():
        text = text.replace(successor_name, frozen_name)
    kept = []
    for line in text.splitlines():
        compact = re.sub(r"\s+", "", line)
        tap_only = (
            re.fullmatch(r'(?:\(\*keep="true"\*\))?outputlogictap_[A-Za-z0-9_]+,?', compact)
            or re.fullmatch(r'assigntap_[A-Za-z0-9_]+=[^;]+;', compact)
            or re.fullmatch(r'\.tap_[A-Za-z0-9_]+\([^;]+\)[,;]?', compact)
        )
        if tap_only:
            if compact.endswith("));"):
                kept.append(");")
            continue
        kept.append(line)
    value = "\n".join(kept)
    value = re.sub(r",\s*\);", ");", value)
    return re.sub(r"\s+", "", value)


def verify_identity_and_clones() -> dict[str, Any]:
    for relative, digest in EXPECTED_SHA.items():
        path = HW / relative
        regular(path); require(sha(path) == digest, "identity drift: " + relative)
    rows = []
    for successor_name, frozen_name, successor_module, frozen_module in CLONES:
        successor = (HW / successor_name).read_text(encoding="utf-8")
        frozen = (HW / frozen_name).read_text(encoding="utf-8")
        require(normalized_without_readonly_taps(successor, successor_module, frozen_module)
                == normalized_without_readonly_taps(frozen, successor_module, frozen_module),
                "independent clone mismatch: " + successor_name)
        rows.append({"successor": successor_name, "frozen": frozen_name,
                     "successor_sha256": sha(HW / successor_name),
                     "frozen_sha256": sha(HW / frozen_name)})
    return {"exact_identities": len(EXPECTED_SHA), "clone_pairs": rows}


def verify_exact_taps() -> dict[str, Any]:
    top = strip_comments((HW / CLONES[-1][0]).read_text(encoding="utf-8"))
    declared = re.findall(r'\(\*\s*keep\s*=\s*"true"\s*\*\)\s*output\s+logic\s+(tap_[A-Za-z0-9_]+)', top)
    require(len(declared) == 13 and set(declared) == set(TAPS), "13 exact kept taps missing")
    direct = {
        "tap_core_mem_req_accept": "core_mem_req_accept",
        "tap_adapter_core_mem_req_accept": "adapter_core_mem_req_accept",
        "tap_core_mem_rsp_accept": "core_mem_rsp_accept",
        "tap_adapter_core_mem_rsp_accept": "adapter_core_mem_rsp_accept",
        "tap_consistency_fault_now": "consistency_fault_now",
        "tap_consistency_fault_q": "consistency_fault_q",
        "tap_core_protocol_error": "core_protocol_error",
        "tap_adapter_protocol_error": "adapter_protocol_error",
    }
    for tap, source in direct.items():
        require(re.search(r"assign\s+%s\s*=\s*%s\s*;" % (tap, source), top) is not None,
                "non-exact top tap: " + tap)
    for tap in TAPS:
        for line in top.splitlines():
            if ("assign " + tap) in line:
                require("$isunknown" not in line and "?" not in line and
                        "===" not in line and "!==" not in line,
                        "tap X coercion: " + tap)
    expected_connections = {
        "tap_frontend_compactor_fault_q": "tap_frontend_compactor_fault_q",
        "tap_frontend_paired_sink_fault_q": "tap_frontend_paired_sink_fault_q",
        "tap_core_adapter_fault_q": "tap_core_adapter_fault_q",
        "tap_service_fault_q": "tap_service_fault_q",
        "tap_memory_adapter_fault_q": "tap_fault_q",
    }
    for tap, port in expected_connections.items():
        require(top.count("." + port + "(" + tap + ")") == 1,
                "tap hierarchy connection drift: " + tap)
    return {"tap_count": 13, "keep_count": len(declared),
            "direct_exact_fanouts": len(direct),
            "hierarchical_exact_connections": len(expected_connections),
            "x_coercion": False, "functional_fanin_from_taps": False}


def verify_endpoint_semantics() -> dict[str, Any]:
    endpoint = strip_comments(ENDPOINT.read_text(encoding="utf-8"))
    payload_fields = ("mem_req_epoch", "mem_req_slot", "mem_req_generation",
                      "mem_req_tag", "mem_req_output_block", "mem_req_slice",
                      "mem_req_source_channel")
    payload_match = re.search(r"request_payload_known\s*=\s*!\$isunknown\(\{(.*?)\}\)\s*;",
                              endpoint, flags=re.S)
    require(payload_match is not None and all(field in payload_match.group(1)
            for field in payload_fields), "payload-known gate is incomplete")
    required = (
        "mem_req_ready = 1'b0;", "qualified_request_valid = 1'b0;",
        "qualified_request_accept = 1'b0;", "endpoint_protocol_fault_now = 1'b0;",
        "if (mem_req_valid === 1'b1)", "if (request_payload_known)",
        "else if (mem_req_accept !== 1'b0)",
        "else if (mem_req_valid !== 1'b0)",
        ".mem_req_valid(qualified_request_valid)",
        ".mem_req_accept(qualified_request_accept)",
    )
    for token in required:
        require(token in endpoint, "endpoint semantic token missing: " + token)

    # Independent four-state truth-table projection of the exact source branch.
    cases = []
    for valid in ("0", "1", "X", "Z"):
        for known in (False, True):
            for accept in ("0", "1", "X"):
                qvalid = qaccept = ready = fault = 0
                if valid == "1":
                    if known:
                        qvalid = ready = 1
                        if accept == "1": qaccept = 1
                        elif accept != "0": fault = 1
                    else: fault = 1
                elif valid != "0": fault = 1
                if valid != "1" or not known:
                    require(qvalid == 0 and qaccept == 0 and ready == 0,
                            "unknown request escaped endpoint model")
                if valid in ("X", "Z") or (valid == "1" and not known) or (
                        valid == "1" and known and accept == "X"):
                    require(fault == 1, "malformed request did not report fault")
                cases.append((valid, known, accept, qvalid, qaccept, ready, fault))
    return {"truth_table_cases": len(cases), "payload_fields": len(payload_fields),
            "unknown_valid_isolated": True, "unknown_payload_isolated": True,
            "unknown_accept_isolated": True, "malformed_reports_fault": True}


def verify_filelist_closure() -> dict[str, Any]:
    members = [line.strip() for line in FILELIST.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    require(len(members) == len(set(members)) == 11, "filelist count/duplicate")
    texts = []
    for member in members:
        path = HW / member; regular(path); texts.append(path.read_text(encoding="utf-8"))
    definitions = set()
    for text in texts:
        definitions.update(re.findall(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b",
                                      strip_comments(text), flags=re.M))
    expected = {
        "m1279_fc2_raw4_to_descriptor4_terminal_hint_compactor_tapped",
        "m1279_fc2_descriptor4_source_cap_frontend_tapped",
        "m1279_fc2_raw4_to_source_cap_frontend_tapped",
        "m218_fc2_tagged_slice_service_island",
        "m1279_fc2_k1_reset_hygiene_registered_release_service_island_tapped",
        "m1279_fc2_reset_hygiene_registered_release_standalone_raw4_acc24_tapped",
        "m1279_fc2_bundle_to_8bank_no_reuse_adapter_tapped",
        "m1279_c2_k1_semantic_tap_wrapper",
        "m349_fc2_scalar_bank_memory_model",
        "m1279_valid_qualified_scalar_bank_endpoint",
        "m1279_c2_k1_diagnostic_system",
        "tb_m1279_c2_dual_dut_valid_qualified_endpoint",
    }
    require(definitions == expected, "filelist module-definition closure drift")
    # Explicit direct-instantiation closure avoids pretending a lexical regex is elaboration.
    for module in expected - {"tb_m1279_c2_dual_dut_valid_qualified_endpoint"}:
        require(sum(strip_comments(text).count(module) for text in texts) >= 2,
                "defined module is not instantiated/referenced: " + module)
    return {"members": len(members), "module_definitions": len(definitions),
            "expected_definition_set_exact": True, "eda_elaboration": False}


def audit_tb_executability() -> dict[str, Any]:
    tb = strip_comments(TB.read_text(encoding="utf-8"))
    require(tb.count("m1279_c2_k1_diagnostic_system #(") == 3,
            "not exactly two DUT instances plus one declaration")
    require("always #1.5 clk_core = ~clk_core;" in tb and
            "if (window_cycle == 128)" in tb and "$finish;" in tb and
            "#1000 $fatal" in tb, "termination source missing")
    require("raw_seen_original" in tb and "raw_seen_qualified" in tb,
            "raw reachability missing")

    request_reachability = bool(re.search(
        r"if\s*\(\s*\|?req_(?:valid|accept)_(?:original|qualified)", tb))
    completion_reachability = bool(re.search(
        r"if\s*\(.*(?:result_valid|result_accept|done_valid|done_accept)_(?:original|qualified)",
        tb))
    functional_compare = bool(re.search(
        r"(?:result|done|req_slot)_[A-Za-z0-9_]*original\s*(?:===|==|!==|!=)\s*"
        r"(?:result|done|req_slot)_[A-Za-z0-9_]*qualified", tb))
    # A fixed-window $finish is reachable after raw acceptance even if no bank
    # request, result or token completion ever occurs.
    fixed_window_can_pass_without_endpoint = (
        "if (!raw_seen_original || !raw_seen_qualified)" in tb and
        not request_reachability and not completion_reachability)
    return {"two_duts": True, "clock_and_watchdog": True,
            "fixed_window_termination": True, "raw_reachability": True,
            "bank_request_reachability": request_reachability,
            "result_or_done_reachability": completion_reachability,
            "functional_dual_dut_compare": functional_compare,
            "fixed_window_can_pass_without_endpoint": fixed_window_can_pass_without_endpoint}


def load_checker():
    spec = importlib.util.spec_from_file_location("m1287_target_m1279_checker", CHECKER)
    require(spec is not None and spec.loader is not None, "cannot import checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def contract_promotion_attacks(checker) -> dict[str, list[str]]:
    base = checker.strict_json(CONTRACT)
    rejected, escaped = [], []
    attacks = {
        "existing_vcs": ("vcs", True),
        "existing_power": ("power", True),
        "existing_energy": ("energy", True),
        "added_k8": ("k8_present", True),
        "added_equal_bandwidth_k1x8": ("equal_bandwidth_k1x8_present", True),
        "added_single_k1_power": ("single_k1_power_admitted", True),
        "added_fair_energy": ("fair_energy_comparison_admitted", True),
        "added_performance": ("performance_admitted", True),
        "added_system_speedup": ("system_speedup", True),
        "added_paper_headline": ("paper_headline", True),
    }
    with tempfile.TemporaryDirectory(prefix="m1287_contract_") as temp:
        for name, (key, value) in attacks.items():
            attacked = copy.deepcopy(base); attacked["claim_boundary"][key] = value
            path = Path(temp) / (name + ".json")
            path.write_text(json.dumps(attacked, sort_keys=True), encoding="utf-8")
            try:
                with mock.patch.object(checker, "CONTRACT", path):
                    checker.check_contract()
            except Exception:
                rejected.append(name)
            else:
                escaped.append(name)
    return {"rejected": rejected, "escaped": escaped}


def endpoint_lexical_attacks(checker) -> dict[str, list[str]]:
    rejected, escaped = [], []
    original_endpoint = ENDPOINT.read_text(encoding="utf-8")
    original_tb = TB.read_text(encoding="utf-8")
    members = [line.strip() for line in FILELIST.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    attacks = {
        "bypass_valid_gate": (original_endpoint.replace(
            "if (mem_req_valid === 1'b1) begin",
            "if (1'b1) begin // mem_req_valid === 1'b1", 1), original_tb),
        "bypass_payload_known_gate": (original_endpoint.replace(
            "if (request_payload_known) begin",
            "if (1'b1) begin // request_payload_known", 1), original_tb),
        "force": (original_endpoint, original_tb + "\ninitial force rst_core = 1'b0;\n"),
        "initreg": (original_endpoint, original_tb + "\ninitial $display(\"+initreg\");\n"),
    }
    with tempfile.TemporaryDirectory(prefix="m1287_endpoint_") as temp:
        root = Path(temp)
        for member in members:
            destination = root / member; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(HW / member, destination)
        copied_filelist = root / FILELIST.relative_to(HW)
        copied_filelist.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FILELIST, copied_filelist)
        for name, (endpoint_text, tb_text) in attacks.items():
            (root / ENDPOINT.relative_to(HW)).write_text(endpoint_text, encoding="utf-8")
            (root / TB.relative_to(HW)).write_text(tb_text, encoding="utf-8")
            try:
                with mock.patch.object(checker, "HW", root):
                    checker.check_endpoint_and_tb()
            except Exception:
                rejected.append(name)
            else:
                escaped.append(name)
    return {"rejected": rejected, "escaped": escaped}


def tap_x_coercion_attack(checker) -> dict[str, Any]:
    successor_name, frozen_name, successor_module, frozen_module = CLONES[-1]
    successor = (HW / successor_name).read_text(encoding="utf-8")
    attacked = successor.replace(
        "assign tap_core_protocol_error = core_protocol_error;",
        "assign tap_core_protocol_error = $isunknown(core_protocol_error) ? 1'b0 : core_protocol_error;",
        1)
    frozen = (HW / frozen_name).read_text(encoding="utf-8")
    checker_equivalence_escaped = (
        checker.functional_normal_form(attacked) == checker.functional_normal_form(frozen))
    independent_exact_tap_rejects = re.search(
        r"assign\s+tap_core_protocol_error\s*=\s*core_protocol_error\s*;",
        strip_comments(attacked)) is None
    # The target topology checker accepts every line beginning "assign tap_".
    target_topology_shape_accepts = bool(re.fullmatch(
        r"assigntap_[A-Za-z0-9_]+=[^;]+;",
        re.sub(r"\s+", "", "assign tap_core_protocol_error = $isunknown(core_protocol_error) ? 1'b0 : core_protocol_error;")))
    return {"target_normalizer_escape": checker_equivalence_escaped,
            "target_topology_shape_accepts": target_topology_shape_accepts,
            "independent_exact_tap_rejects": independent_exact_tap_rejects}


def main() -> int:
    identity = verify_identity_and_clones()
    taps = verify_exact_taps()
    endpoint = verify_endpoint_semantics()
    filelist = verify_filelist_closure()
    tb = audit_tb_executability()
    checker = load_checker()
    baseline = checker.run_checks()
    require(baseline["status"] == "PASS_M1279_SOURCE_ONLY__NO_EXECUTION_AUTHORIZED",
            "target checker baseline did not pass")
    claims = contract_promotion_attacks(checker)
    lexical = endpoint_lexical_attacks(checker)
    tap_attack = tap_x_coercion_attack(checker)
    findings = []
    if (not tb["bank_request_reachability"] or not tb["result_or_done_reachability"]
            or not tb["functional_dual_dut_compare"]):
        findings.append("P1_01_TB_ENDPOINT_COMPLETION_COMPARE_NOT_REQUIRED")
    if claims["escaped"]:
        findings.append("P1_02_OPEN_WORLD_CLAIM_PROMOTION_ESCAPES")
    if any(name in lexical["escaped"] for name in
           ("bypass_valid_gate", "bypass_payload_known_gate")):
        findings.append("P1_03_ENDPOINT_GATE_LEXICAL_BYPASS_ESCAPES")
    if tap_attack["target_normalizer_escape"] and tap_attack["target_topology_shape_accepts"]:
        findings.append("P1_04_TAP_X_COERCION_ESCAPES_TARGET_NORMALIZER")
    require(len(findings) == 4, "unexpected finding population")
    output = {
        "schema": "m1287_m1279_c2_semantic_tap_source_hammer_mechanical_r1_v1",
        "status": "STOP_M1287_M1279_FOUR_P1__NO_FRESH_VCS_RELEASE",
        "identity_and_clones": identity, "semantic_taps": taps,
        "endpoint_semantics": endpoint, "filelist": filelist,
        "dual_dut_tb": tb, "contract_attacks": claims,
        "endpoint_checker_attacks": lexical, "tap_x_coercion_attack": tap_attack,
        "findings": findings, "issue_counts": {"P0": 0, "P1": 4, "P2": 0},
        "execution": {"vcs": False, "eda": False, "gpu": False,
                      "remote": False, "temporary_synthetic_only": True,
                      "author_receipt_consumed_by_script": False},
        "release": {"fresh_rtl_vcs_authorized": False,
                    "required_successor": "additive source repair plus new different-author hammer"},
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
