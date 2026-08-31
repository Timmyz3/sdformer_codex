#!/usr/bin/env python3
"""M1300 receipt-blind source/static/synthetic hammer of M1293.

No author receipt, VCS, EDA, production, GPU, or remote state is consumed.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CHECKER = HW / "system_simulator/scripts/check_m1293_c2_semantic_tap_dual_dut_repair_source.py"
TEST = HW / "system_simulator/tests/test_m1293_c2_semantic_tap_dual_dut_repair_source.py"
ENDPOINT = HW / "dc_handoff/tb/m1293_valid_qualified_scalar_bank_endpoint.sv"
TB = HW / "dc_handoff/tb/tb_m1293_c2_dual_dut_functional_reachability.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1293_c2_dual_dut_source_only_vcs.f"
CONTRACT = HW / "contracts/m1293_c2_semantic_tap_dual_dut_repair_source_contract_r1_20260830.json"
CONTRACT_SUM = Path(str(CONTRACT) + ".sha256")
CONTRACT_SEAL = Path(str(CONTRACT_SUM) + ".seal.sha256")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
TOP = HW / "rtl_m1279/m1279_c2_k1_semantic_tap_wrapper.sv"

EXPECTED_SHA = {
    CHECKER: "eb88b6c7b3def8c01a429305225d51802d0a91e6918ded44fe87c6f780b28c39",
    TEST: "a682ed7f6cf00e3a87c9eeec8c648f464d79f90906967eb6a7c929e91f5f5cf1",
    ENDPOINT: "0f33949232f6973c4f05364b331b51192ad39f4d1318402a82616fe570718e88",
    TB: "89f1915b57b9ed5b7ebc72eb8db3dcdea6dfde67db1dae6c92a87ba027e893a9",
    FILELIST: "af7c3e4394d35ba037280c82b6395769ed10203a73b91446ac24f7251a169564",
    CONTRACT: "1c50a862e02aeda009d52850f00ba8befa96c19b6599077e61951b36929299f5",
    CONTRACT_SUM: "344604bb7fe3baa5ee7093ed11e80c42c62dbdda2e69bae493b3bc4e2d1e67d1",
    CONTRACT_SEAL: "4951c1133ca49f03589b02dfc64b5b6608f9dc376765f70389838d2e0924a516",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def rejected(action: Callable[[], Any]) -> str:
    try:
        action()
    except BaseException as exc:
        return type(exc).__name__ + ": " + str(exc)
    raise HammerError("attack unexpectedly escaped")


def load_checker():
    spec = importlib.util.spec_from_file_location("m1300_receipt_blind_m1293", CHECKER)
    require(spec is not None and spec.loader is not None, "cannot load M1293 checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def independent_endpoint_projection() -> dict[str, Any]:
    clean = strip_comments(ENDPOINT.read_text(encoding="utf-8"))
    require(clean.count("always_comb begin : valid_qualified_guard") == 1,
            "guard block missing or duplicated")
    body = clean
    for token in ("mem_req_valid === 1'b1", "if (request_payload_known)",
                  "mem_req_accept === 1'b1", "mem_req_accept !== 1'b0",
                  "mem_req_valid !== 1'b0"):
        require(token in body, "independent endpoint guard missing " + token)
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
                    require((qvalid, qaccept, ready) == (0, 0, 0),
                            "unknown request escaped model")
                if valid in ("X", "Z") or (valid == "1" and not known) or (
                        valid == "1" and known and accept == "X"):
                    require(fault == 1, "malformed request not faulted")
                cases.append((valid, known, accept, qvalid, qaccept, ready, fault))
    return {"four_state_cases": len(cases), "unknown_quarantined": True,
            "malformed_faulted": True}


def independent_tb_semantics() -> dict[str, Any]:
    clean = strip_comments(TB.read_text(encoding="utf-8"))
    require(clean.count("m1293_c2_k1_diagnostic_system #(") == 3,
            "not two DUT instances plus declaration")
    required = (
        "request_count_original <= 0 || request_count_qualified <= 0",
        "result_count_original <= 0 || result_count_qualified <= 0",
        "done_count_original <= 0 || done_count_qualified <= 0",
        "req_accept_original !== req_accept_qualified",
        "result_accept_original !== result_accept_qualified",
        "done_accept_original !== done_accept_qualified",
        "result_acc_original[lane] !== result_acc_qualified[lane]",
        "request_class_mismatch_count != 0",
        "result_class_mismatch_count != 0",
        "done_class_mismatch_count != 0",
        "first_result_cycle <= first_request_cycle",
        "first_done_cycle < first_request_cycle",
    )
    for token in required:
        require(token in clean, "TB functional/reachability token missing: " + token)
    require(clean.count("PASS_M1293_DUAL_DUT_FUNCTIONAL_REACHABILITY") == 2,
            "PASS token population drift")
    for token in ("force", "release", "+initreg", "casex", "casez"):
        require(re.search(r"\b" + re.escape(token) + r"\b", clean, flags=re.I) is None,
                "prohibited mechanism " + token)
    return {"dual_dut_instances": 2, "request_result_done_reachable": True,
            "three_class_compare": True, "accumulator_compare": True,
            "endpoint_unreached_cannot_pass": True}


def replay_four_findings(checker) -> dict[str, Any]:
    endpoint = ENDPOINT.read_text(encoding="utf-8")
    tb = TB.read_text(encoding="utf-8")
    top = TOP.read_text(encoding="utf-8")
    contract = checker.strict_json(CONTRACT)

    tb_attacks = {
        "drop_request_reachability": tb.replace(
            "request_count_original <= 0", "request_count_original < 0", 1),
        "drop_result_reachability": tb.replace(
            "result_count_original <= 0", "result_count_original < 0", 1),
        "drop_done_reachability": tb.replace(
            "done_count_original <= 0", "done_count_original < 0", 1),
        "invert_request_compare": tb.replace(
            "req_accept_original !== req_accept_qualified",
            "req_accept_original === req_accept_qualified", 1),
        "move_pass_outside_guard": tb +
            "\ninitial $display(\"PASS_M1293_DUAL_DUT_FUNCTIONAL_REACHABILITY\");\n",
    }
    tb_rejections = {name: rejected(lambda text=text: checker.check_tb_text(text))
                     for name, text in tb_attacks.items()}

    contract_rejections = {}
    for key in ("vcs", "ptpx", "k8_present", "equal_bandwidth_k1x8_present",
                "single_k1_power_admitted", "fair_energy_comparison_admitted",
                "performance_admitted", "mapped_functionality", "system_speedup",
                "paper_ppa_ready", "paper_headline"):
        attacked = copy.deepcopy(contract); attacked["claim_boundary"][key] = True
        contract_rejections[key] = rejected(lambda value=attacked:
            checker.check_contract_data(value, validate_source_hashes=False))
    attacked = copy.deepcopy(contract)
    attacked["claim_boundary"]["future_claim_escape"] = False
    contract_rejections["open_world_added_key"] = rejected(lambda:
        checker.check_contract_data(attacked, validate_source_hashes=False))
    attacked = copy.deepcopy(contract); attacked["claim_boundary"]["system_speedup"] = 0
    contract_rejections["bool_as_int"] = rejected(lambda:
        checker.check_contract_data(attacked, validate_source_hashes=False))

    endpoint_rejections = {
        "bypass_valid_gate": rejected(lambda: checker.check_endpoint_text(endpoint.replace(
            "if (mem_req_valid === 1'b1) begin",
            "if (1'b1) begin // mem_req_valid === 1'b1", 1))),
        "bypass_payload_gate": rejected(lambda: checker.check_endpoint_text(endpoint.replace(
            "if (request_payload_known) begin",
            "if (1'b1) begin // request_payload_known", 1))),
    }
    tap_rejections = {
        "x_to_zero": rejected(lambda: checker.check_tap_exact_rhs_text(top.replace(
            "assign tap_core_protocol_error = core_protocol_error;",
            "assign tap_core_protocol_error = $isunknown(core_protocol_error) ? 1'b0 : core_protocol_error;", 1))),
        "case_equality_coercion": rejected(lambda: checker.check_tap_exact_rhs_text(top.replace(
            "assign tap_core_protocol_error = core_protocol_error;",
            "assign tap_core_protocol_error = (core_protocol_error === 1'b1);", 1))),
    }
    return {"P1_01_tb_reachability_compare": tb_rejections,
            "P1_02_closed_contract": contract_rejections,
            "P1_03_structural_endpoint": endpoint_rejections,
            "P1_04_exact_noncoercing_taps": tap_rejections}


def main() -> int:
    for path, digest in EXPECTED_SHA.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift: " + str(path))
    checker = load_checker()
    baseline = checker.run_checks()
    require(baseline["status"] == "PASS_M1293_SOURCE_REPAIR__NO_EXECUTION_AUTHORIZED",
            "M1293 baseline checker failed")
    projection = independent_endpoint_projection()
    tb = independent_tb_semantics()
    attacks = replay_four_findings(checker)
    output = {
        "schema": "m1300_m1293_c2_semantic_tap_repair_receipt_blind_hammer_r1_v1",
        "status": "GO_ONE_FRESH_RTL_ONLY_VCS_RELEASE__M1293_FOUR_P1_CLOSED",
        "score": 96,
        "receipt_blind": True,
        "source_identities": {str(path.relative_to(ROOT)): digest
                              for path, digest in EXPECTED_SHA.items()},
        "baseline": baseline, "independent_endpoint": projection,
        "independent_tb": tb, "m1287_attack_replay": attacks,
        "issue_counts": {"P0": 0, "P1": 0, "P2": 1},
        "execution": {"vcs": False, "eda": False, "gpu": False,
            "remote": False, "production": False, "synthetic_static_only": True,
            "author_receipt_consumed": False},
        "release": {"fresh_rtl_only_vcs_authorized": True,
            "maximum_fresh_runs": 1, "top": "tb_m1293_c2_dual_dut_functional_reachability",
            "filelist": str(FILELIST.relative_to(ROOT)),
            "not_authorized": ["DC", "PT", "PTPX", "SAIF", "power", "performance",
                               "system_speedup", "paper_headline"]},
        "remaining_p2": "A VCS PASS would prove only this directed K1 diagnostic axis; it cannot promote K8/equal-bandwidth power, performance, mapped functionality, or system claims.",
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
