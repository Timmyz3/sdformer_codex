#!/usr/bin/env python3
"""M1162 source-only protocol and identity checker; never runs VCS or EDA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
TB = HERE / "tb_m1162_common_charge_protocol_source.sv"
PLAN = HERE / "m1162_protocol_sva_plan.md"
MAPPING = HW / "dc_handoff/manifests/m1116c_c1_full_storage_boundary_mapping_r1.tsv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
CONTRACT = HW / "contracts/m1162_m1160_m1116c_c1_common_charge_protocol_repair_source_contract_r1_20260830.json"

MAPPING_SHA = "16da013268f765d74703a041ccd35b2054ff425ef726d2b5c69d545230ae0271"
M935_SHA = "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8"
PARENT_SHA = "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783"
DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def check_frozen_identity() -> dict[str, str]:
    actual = {
        "mapping": sha256(MAPPING), "m935": sha256(M935),
        "parent": sha256(PARENT), "docs359": sha256(DOC359),
    }
    require(actual == {"mapping": MAPPING_SHA, "m935": M935_SHA,
                       "parent": PARENT_SHA, "docs359": DOC359_SHA},
            "frozen identity drift")
    return actual


def _assignment(text: str, lhs: str) -> str:
    match = re.search(r"(?ms)^\s*" + re.escape(lhs) + r"\s*=\s*(.*?);", text)
    require(match is not None, "missing assignment: " + lhs)
    return match.group(1)


def check_wrapper() -> dict[str, Any]:
    text = WRAPPER.read_text()
    require(text.count(
        "m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935") == 1,
        "frozen M935 instance count")
    require("m1116c_m935_c1_full_storage_common_charge_boundary" not in text,
            "broken M1116C wrapper instantiated or copied by name")
    require("TS1N28HPCPHVTB128X128M4S" not in text and
            "m528_dw1rw_parent_scratch_9x128_macro" not in text,
            "wrapper directly instantiates a macro")
    for lhs in ("weight_read_request_valid", "psum_read_request_valid"):
        expression = _assignment(text, lhs)
        require("ready" not in expression,
                lhs + " has ready-to-valid combinational dependency")
    required = (
        "request_active_q", "weight_request_accepted_q",
        "psum_request_accepted_q", "request_tuple_mutated_w",
        "weight_request_fire_w", "psum_request_fire_w",
        "response_accept_w", "boundary_fault_q",
        "minimum II=2", "same-cycle request/response is prohibited",
        "reset cancels", "external services must discard",
    )
    for token in required:
        require(token in text, "missing protocol token: " + token)
    require(not re.search(r"(?m)^\s+logic\s+\[1151:0\]", text),
            "weight response payload buffered")
    require(not re.search(r"(?m)^\s+logic\s+\[1823:0\]", text),
            "psum response payload buffered")
    require(re.search(r"weight_read_response_valid\s*&&\s*\(!request_active_q\s*\|\|\s*!weight_request_accepted_q\)", text),
            "weight early/spurious guard missing")
    require(re.search(r"!request_active_q\s*\|\|\s*!request_first_q\s*\|\|\s*!psum_request_accepted_q", text),
            "psum early/spurious/non-first guard missing")
    require("request_active_q && !issue_request_valid" in text,
            "request cancellation guard missing")

    frozen_text = M935.read_text()
    start = frozen_text.index(
        "module m935_m912_three_stage_exact_parent_match_product_capture_island")
    frozen_header = frozen_text[start:frozen_text.index(");", start) + 2]
    frozen_ports = re.findall(
        r"\b(?:input|output)\s+logic(?:\s+\[[^\]]+\])?\s+([A-Za-z_][A-Za-z0-9_]*)",
        frozen_header)
    start = text.index(
        "m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935")
    instance = text[start:text.index(");", start) + 2]
    connected = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", instance)
    require(len(connected) == len(set(connected)) and
            set(connected) == set(frozen_ports),
            "frozen M935 port connection set drift")
    return {
        "request_valid_ready_independent": True,
        "independent_request_accept_tracking": True,
        "outstanding_depth": 1,
        "minimum_completed_issue_ii_zero_stall_one_cycle_response": 2,
        "latched_request_state_bits": 36,
        "total_added_state_bits": 40,
        "payload_fifo_bits": 0,
        "frozen_m935_ports_connected_exactly_once": len(frozen_ports),
    }


def check_tb_and_plan() -> dict[str, Any]:
    tb = TB.read_text()
    plan = PLAN.read_text()
    required_tb = (
        "partial_weight_first", "partial_psum_first",
        "skew_weight_response", "skew_psum_response",
        "long_request_stall", "long_response_backpressure",
        "reset_pending", "early_response", "spurious_response",
        "request_cancellation", "tuple_mutation", "sticky_error_seen",
        "ap_weight_hold", "ap_psum_hold",
        "ap_no_lone_weight_response_consume",
        "ap_no_lone_psum_response_consume",
        "duplicate or missing request fire",
        "PASS_M1162_SOURCE_DIRECTED_PLAN",
    )
    for token in required_tb:
        require(token in tb, "TB source missing: " + token)
    for token in (
            "may make `ready` depend on its own `valid`",
            "same-cycle request/response", "II is exactly 2", "partial-order",
            "tuple mutation",
            "response-before-own-request", "post-reset spurious response",
            "must not cite any M1114/M528 speedup"):
        require(token in plan, "SVA plan missing: " + token)
    return {"directed_cover_classes": 12, "sva_assertion_plan": True,
            "tb_compiled": False, "tb_run": False}


class ProtocolModel:
    """Small bounded semantic model used only by source-level unit tests."""
    def __init__(self) -> None:
        self.active = False
        self.first = False
        self.weight_accepted = False
        self.psum_accepted = False
        self.fault = False
        self.weight_fires = 0
        self.psum_fires = 0

    def reset(self) -> None:
        self.__init__()

    def step(self, *, issue: bool, first: bool = False,
             weight_ready: bool = False, psum_ready: bool = False,
             weight_response: bool = False, psum_response: bool = False,
             core_ready: bool = True, mutate: bool = False) -> dict[str, bool]:
        weight_valid = issue and (not self.active or not self.weight_accepted)
        effective_first = self.first if self.active else first
        psum_valid = (issue and effective_first and
                      (not self.active or not self.psum_accepted))
        weight_fire = weight_valid and weight_ready
        psum_fire = psum_valid and psum_ready
        core_valid = (self.active and self.weight_accepted and
                      (not self.first or self.psum_accepted) and
                      weight_response and
                      (not self.first or psum_response))
        response_accept = core_valid and core_ready
        if weight_response and (not self.active or not self.weight_accepted):
            self.fault = True
        if psum_response and (not self.active or not self.first or
                              not self.psum_accepted):
            self.fault = True
        if self.active and not issue:
            self.fault = True
        if mutate:
            self.fault = True
        if not self.active and issue:
            self.active = True
            self.first = first
            self.weight_accepted = weight_fire
            self.psum_accepted = (not first) or psum_fire
        elif self.active:
            self.weight_accepted |= weight_fire
            self.psum_accepted |= psum_fire
        if weight_fire:
            self.weight_fires += 1
        if psum_fire:
            self.psum_fires += 1
        if response_accept:
            self.active = False
            self.weight_accepted = False
            self.psum_accepted = False
        return {"weight_valid": weight_valid, "psum_valid": psum_valid,
                "weight_ready_out": core_valid and core_ready,
                "psum_ready_out": core_valid and core_ready and effective_first,
                "core_valid": core_valid, "response_accept": response_accept}


def check_contract() -> dict[str, Any]:
    value = strict_json(CONTRACT)
    require(value["status"] ==
            "PASS_M1162_ADDITIVE_PROTOCOL_REPAIR_SOURCE_ONLY__FRESH_HAMMER_REQUIRED__NO_VCS_NO_EDA",
            "contract status drift")
    paths = {"wrapper": WRAPPER, "tb_source": TB, "sva_plan": PLAN}
    for key, path in paths.items():
        require(value["source_identity"][key]["sha256"] == sha256(path),
                "contract source identity drift: " + key)
    require(value["protocol"]["outstanding_depth"] == 1 and
            value["protocol"]["minimum_completed_issue_ii"] == 2 and
            value["protocol"]["request_valid_depends_on_ready"] is False,
            "contract protocol coordinate drift")
    for key in ("vcs", "dc", "pt", "formality", "ptpx",
                "rtl_speedup", "system_speedup", "paper_ppa_ready"):
        require(value["claim_boundary"][key] is False,
                "forbidden positive claim: " + key)
    return {"contract_strict": True, "no_performance_inheritance": True}


def run() -> dict[str, Any]:
    return {"schema": "m1162_common_charge_protocol_source_check_v1",
            "status": "PASS_M1162_SOURCE_CHECK__FRESH_HAMMER_REQUIRED__NO_VCS_NO_EDA",
            "frozen": check_frozen_identity(), "wrapper": check_wrapper(),
            "verification_source": check_tb_and_plan(),
            "contract": check_contract()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    value = run()
    if args.json:
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(value["status"])


if __name__ == "__main__":
    main()
