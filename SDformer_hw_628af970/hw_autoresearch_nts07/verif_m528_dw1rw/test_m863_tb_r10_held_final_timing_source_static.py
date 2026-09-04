#!/usr/bin/env python3
"""Exact source-only admission for the M863 TB-r10 held-final timing repair."""

import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


OLD_HANDSHAKE = """            @(negedge clk_core);
            if (!issue_data_ready || debug_overflow_block_event
                    || protocol_error)
                $fatal(1, \"later authoritative parent did not release held final\");
            @(posedge clk_core);
            release dut.slot0_valid_q;
            release dut.slot0_parent_id_q;
            release dut.slot0_consumer_id_q;
            release dut.slot0_data_q;
            release psum_write_ready;
            release row_complete_ready;
            @(negedge clk_core);
            if (protocol_error || count_row_completions != before_rows + 1)
                $fatal(1, \"legal parent completion failed after stale hold\");
"""

NEW_HANDSHAKE = """            // Observe the authoritative valid/ready handshake before its
            // active edge.  The prior TB sampled ready at the following
            // negedge, after the accepted final had already retired and
            // issue_request_valid had correctly fallen.
            #1ps;
            if (!issue_data_valid || !issue_data_ready
                    || !psum_write_valid || !row_complete_valid
                    || debug_overflow_block_event
                    || protocol_error)
                $fatal(1, \"later authoritative parent did not release held final\");
            // Keep every force stable through exactly one accepting edge.
            @(posedge clk_core);
            // Release only at the inactive edge, outside the sampling/NBA
            // race.  Do not resample ready after the completed handshake.
            @(negedge clk_core);
            release dut.slot0_valid_q;
            release dut.slot0_parent_id_q;
            release dut.slot0_consumer_id_q;
            release dut.slot0_data_q;
            release psum_write_ready;
            release row_complete_ready;
            #1ps;
            if (protocol_error || count_psum_commits != before_psum + 1
                    || count_row_completions != before_rows + 1)
                $fatal(1, \"legal parent completion failed after stale hold\");
            cov_held_final_stale_parent_recovery =
                cov_held_final_stale_parent_recovery + 1;
            $display(\"HELD_FINAL_RECOVERY_M533_M528_DW1RW_R10 preedge_handshake=1 accept_edges=1 psum_delta=1 row_delta=1 cover=%0d\",
                cov_held_final_stale_parent_recovery);
"""


def main():
    if len(sys.argv) != 8:
        raise RuntimeError("usage: TEST TB_R10 TB_R9 TOP_R2 SVA_R2 MACRO BINDING FOUNDRY_V")
    tb, old_tb, top, sva, macro, binding, foundry = [Path(arg) for arg in sys.argv[1:]]
    for path in (tb, old_tb, top, sva, macro, binding, foundry):
        require(path.is_file() and not path.is_symlink(), "non-regular input " + str(path))
    require(sha256(old_tb) == "863b6793f48a79a2dfa57cd4d543ff5c9cdfbb95e23180a894e788d14643988a",
            "TB r9 identity")
    require(sha256(top) == "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1", "top r2 drift")
    require(sha256(sva) == "b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b", "SVA r2 drift")
    require(sha256(macro) == "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783", "macro adapter drift")
    require(sha256(binding) == "db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983", "binding plan drift")
    require(sha256(foundry) == "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d", "foundry model drift")

    old = old_tb.read_text(encoding="utf-8")
    new = tb.read_text(encoding="utf-8")
    declaration = "    integer cov_held_final_stale_parent_recovery;\n"
    initialization = "        cov_held_final_stale_parent_recovery = 0;\n"
    call_old = "        test_held_final_stale_parent_then_legal();\n"
    call_new = call_old + """        if (cov_held_final_stale_parent_recovery != 1)
            $fatal(1, \"held-final recovery coverage count=%0d\",
                cov_held_final_stale_parent_recovery);
"""
    for needle, count in ((NEW_HANDSHAKE, 1), (declaration, 1),
                          (initialization, 1), (call_new, 1)):
        require(new.count(needle) == count, "new TB exact multiplicity failure")
    require(new.count(OLD_HANDSHAKE) == 0, "old post-handshake sample remains")
    require(old.count(OLD_HANDSHAKE) == 1 and old.count(NEW_HANDSHAKE) == 0,
            "old held-final handshake identity")
    require(declaration not in old and initialization not in old and call_new not in old,
            "old TB unexpectedly contains R21 additions")
    reconstructed = new.replace(NEW_HANDSHAKE, OLD_HANDSHAKE, 1)
    reconstructed = reconstructed.replace(declaration, "", 1)
    reconstructed = reconstructed.replace(initialization, "", 1)
    reconstructed = reconstructed.replace(call_new, call_old, 1)
    require(reconstructed == old, "TB r9->r10 delta exceeds held-final observation repair")

    task_start = new.index("task automatic test_held_final_stale_parent_then_legal;")
    force_pos = new.index("force dut.slot0_valid_q = 1'b1;", task_start)
    settle_pos = new.index("#1ps;", force_pos)
    precheck_pos = new.index("if (!issue_data_valid || !issue_data_ready", settle_pos)
    accept_pos = new.index("@(posedge clk_core);", precheck_pos)
    inactive_pos = new.index("@(negedge clk_core);", accept_pos)
    release_pos = new.index("release dut.slot0_valid_q;", inactive_pos)
    delta_check_pos = new.index("count_psum_commits != before_psum + 1", release_pos)
    token_pos = new.index("HELD_FINAL_RECOVERY_M533_M528_DW1RW_R10", delta_check_pos)
    require(force_pos < settle_pos < precheck_pos < accept_pos < inactive_pos
            < release_pos < delta_check_pos < token_pos, "held-final event order")
    post_accept = new[accept_pos:token_pos]
    require("if (!issue_data_ready" not in post_accept,
            "post-handshake ready resample remains")
    require(post_accept.count("@(posedge clk_core);") == 1,
            "held-final acceptance is not exactly one active edge")

    coverage = new.index("COVERAGE_M533_M528_DW1RW_R8")
    p2 = new.index("P2_STRENGTH_M533_M528_DW1RW_R3", coverage)
    held_call = new.index(call_new, p2)
    attacks = [new.index(call, held_call) for call in (
        "attack_dirty_reserved();", "attack_stale_epoch();", "attack_overflow();",
        "attack_wrong_parent_and_dead_live();", "attack_read_before_write();",
        "attack_parent_only_nonzero_atomic();")]
    final_pass = new.index("PASS_M533_M528_DW1RW_R8_DIRECTED_RANDOM_AND_ATTACKS", attacks[-1])
    require(coverage < p2 < held_call < min(attacks) <= max(attacks) < final_pass,
            "normal/P2/held/attack/final phase order")
    require("normal_covers=13" in new and "minima=1" in new,
            "13 normal cover gate weakened")
    require("minima_pairs=1 minima_responses=2" in new,
            "P2 minima weakened")
    require("attack_count != 6" in new, "six-attack gate weakened")

    print(json.dumps({
        "schema": "m863_m533_tb_r10_held_final_timing_source_static_v1",
        "status": "PASS_EXACT_R9_TO_R10_HELD_FINAL_EVENT_ORDER_REPAIR_SOURCE_ONLY",
        "tb_r10_sha256": sha256(tb),
        "tb_r9_sha256": sha256(old_tb),
        "preedge_valid_ready_check": True,
        "exactly_one_accepting_posedge": True,
        "release_at_inactive_edge": True,
        "post_handshake_ready_resample_absent": True,
        "exact_psum_and_row_delta_checks": [1, 1],
        "dedicated_recovery_cover_and_token": True,
        "normal_cover_count": 13,
        "p2_epoch": 14,
        "protocol_attack_count": 6,
        "rtl_sva_macro_binding_foundry_frozen": True,
        "vcs_or_simv_executed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
