#!/usr/bin/env python3
"""Read-only structural checks for the additive M912 draft.

This is not an RTL compiler and is not timing, VCS, or formal evidence.  It
only prevents accidental interface drift, r2 replacement, or wide-payload
buffering before the commercial-tool launch gate is authored.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv"
RTL = ROOT / "rtl_m912_c1_pipeline/m912_m528_metadata_pipelined_product_capture_island.sv"
SVA = ROOT / "verif_m912_c1_pipeline/m912_m528_metadata_pipeline_assertions.sv"
TB = ROOT / "verif_m912_c1_pipeline/tb_m912_metadata_pipeline_unit_delay_r1.sv"
M863_TB = ROOT / "tb_m528_dw1rw/tb_m528_dead_write_only_1rw_product_capture_r10.sv"

R2_SHA256 = "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1"
M863_TB_SHA256 = "783526023fbea0edece27ab33bc7c1233d9fa5e803baf369ee3fd89fec06ce9d"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL M912 static: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def port_contract(text: str) -> List[Tuple[str, str, str]]:
    header = text.split(");", 1)[0]
    ports = []  # type: List[Tuple[str, str, str]]
    pattern = re.compile(
        r"\b(input|output)\s+logic\s*(\[[^\]]+\])?\s*([A-Za-z_][A-Za-z0-9_]*)"
    )
    for match in pattern.finditer(strip_comments(header)):
        direction, width, name = match.groups()
        ports.append((direction, (width or "scalar").replace(" ", ""), name))
    return ports


def balanced(text: str) -> bool:
    clean = strip_comments(text)
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []  # type: List[str]
    for char in clean:
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def main() -> None:
    require(R2.is_file() and RTL.is_file() and SVA.is_file()
            and TB.is_file() and M863_TB.is_file(), "missing source")
    require(sha256(R2) == R2_SHA256, "frozen r2 changed")
    require(sha256(M863_TB) == M863_TB_SHA256, "frozen M863 TB changed")
    r2_text = R2.read_text()
    rtl_text = RTL.read_text()
    sva_text = SVA.read_text()
    tb_text = TB.read_text()

    require(port_contract(r2_text) == port_contract(rtl_text),
            "top-level port contract drift")
    require("module m912_m528_metadata_pipelined_product_capture_island" in rtl_text,
            "M912 top absent")
    require("m528_dead_write_only_1rw_product_capture_island_r2 u_" not in rtl_text,
            "r2 wrapper masquerading as a pipeline")
    require("active_ctx_valid_q" in rtl_text and "next_ctx_valid_q" in rtl_text,
            "active/next metadata slots absent")
    require("pf_token_valid_q" in rtl_text, "prefetch token absent")
    require("candidate_min" in rtl_text
            and "row_key_s6_w" in rtl_text
            and "pf_key_s6_w" in rtl_text,
            "bank-local balanced selector absent")

    # The only registered 1152-bit payloads are the two pre-existing response
    # slots.  No 1824-bit psum payload register may be added.
    rtl_body = rtl_text.split(");", 1)[1]
    wide_declarations = re.findall(
        r"logic\s+\[1151:0\]\s+([^;]+);", rtl_body)
    wide_q = []
    for declaration in wide_declarations:
        wide_q.extend(name.strip() for name in declaration.split(",")
                      if name.strip().endswith("_q"))
    require(wide_q == ["slot0_data_q", "slot1_data_q"],
            f"unexpected 1152-bit registered payload declaration: {wide_q}")
    require(not re.search(r"logic\s+\[1823:0\]\s+[^;]*_q", rtl_body),
            "new 1824-bit registered psum payload")
    require(not re.search(
        r"\[(1151|1823):0\][^;\n]*(active_ctx|next_ctx|pf_token)", rtl_text),
        "wide data embedded in a metadata slot")

    for required in (
        "issue_request_valid = exec_active_q && active_ctx_valid_q",
        "&& active_ctx_primed_q && !fault_q",
        "prefetch_accept_w = forward_accept_w || macro_read_accept_w",
        "scratch_enable_w = live_write_accept_w || macro_read_accept_w",
        "row_complete_valid = psum_write_valid",
        "debug_scratch_read_q <= macro_read_accept_w",
        "debug_scratch_write_q <= live_write_accept_w",
    ):
        require(required in rtl_text, f"required boundary missing: {required}")

    require(rtl_text.count("module ") == rtl_text.count("endmodule") == 1,
            "RTL module/endmodule mismatch")
    require(sva_text.count("module ") == sva_text.count("endmodule") == 1,
            "SVA module/endmodule mismatch")
    require(balanced(rtl_text), "RTL delimiter imbalance")
    require(balanced(sva_text), "SVA delimiter imbalance")

    for prop in (
        "ap_active_hold", "ap_priming_progress", "ap_primed_hold",
        "ap_next_hold", "ap_pf_hold",
        "ap_request_payload_stable", "ap_no_interrow_bubble",
        "ap_final_candidate_is_reserved",
        "ap_one_port", "ap_completion_atomic", "ap_final_accept_atomic",
        "ap_preaccept_fault_atomic", "ap_parent_conservation",
        "ap_debug_read_delay", "ap_debug_write_delay",
    ):
        require(prop in sva_text, f"SVA obligation absent: {prop}")

    require("module tb_m912_metadata_pipeline_unit_delay_r1" in tb_text,
            "M912 TB top absent")
    require("m912_m528_metadata_pipelined_product_capture_island dut (.*);"
            in tb_text, "M912 DUT binding absent")
    require("oracle_startup_countdown = 2" in tb_text
            and "oracle_startup_countdown == 0" in tb_text,
            "two-cycle latency-aware oracle absent")
    for observer in (
        "oracle_debug_scratch_read_q", "oracle_debug_scratch_write_q",
        "oracle_debug_forward_q", "oracle_debug_read_response_q",
        "oracle_debug_dual_enqueue_q", "oracle_debug_dead_elision_q",
        "oracle_debug_deadline_q", "oracle_debug_overflow_q",
        "oracle_debug_stalled_raw_q",
    ):
        require(observer in tb_text, f"delayed debug oracle absent: {observer}")
    for attack in (
        "attack_dirty_reserved", "attack_stale_epoch", "attack_overflow",
        "attack_wrong_parent_and_dead_live", "attack_read_before_write",
        "attack_parent_only_nonzero_atomic",
    ):
        require(f"task automatic {attack}" in tb_text
                and f"{attack}();" in tb_text,
                f"attack corpus absent: {attack}")
    pass_token = (
        "PASS_M912_C1_METADATA_PIPELINE_UNIT_DELAY_"
        "DIRECTED_RANDOM_AND_ATTACKS"
    )
    require(tb_text.count(pass_token) == 1, "M912 PASS token not unique")
    require("PASS_M533_M528_DW1RW_R8_DIRECTED_RANDOM_AND_ATTACKS"
            not in tb_text, "old PASS token leaked")
    require("dut.lookahead_parent_w" not in tb_text,
            "obsolete r2 lookahead hierarchy leaked")
    require("functional_vcs_only=true timing_verified=false" in tb_text
            and "speedup=false ppa=false energy=false" in tb_text,
            "fail-closed PASS labels absent")
    require(balanced(tb_text), "TB delimiter imbalance")

    print("PASS M912 metadata-pipeline static interface/width/assertion checks")
    print(f"r2_sha256={sha256(R2)}")
    print(f"rtl_sha256={sha256(RTL)}")
    print(f"sva_sha256={sha256(SVA)}")
    print(f"tb_sha256={sha256(TB)}")
    print(f"m863_tb_sha256={sha256(M863_TB)}")
    print("commercial_rtl_compile=false vcs=false dc=false timing=false")


if __name__ == "__main__":
    main()
