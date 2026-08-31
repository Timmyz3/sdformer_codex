#!/usr/bin/env python3
"""Source-static admission checks for the additive M814/M533 TB r8 delta."""

import hashlib
import json
import re
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def task_body(text, name):
    match = re.search(r"task automatic " + re.escape(name) + r"\s*;(.*?)endtask", text, re.S)
    require(match is not None, "missing task " + name)
    return match.group(1)


def popcount(value):
    return bin(value).count("1")


def matcher_parent(masks, row):
    current = masks[row]
    best = None
    best_pop = 0
    if popcount(current) >= 2:
        for candidate, mask in enumerate(masks):
            candidate_pop = popcount(mask)
            eligible = ((mask & current) == mask and candidate_pop >= 1
                        and not (mask == current and candidate >= row))
            if eligible and candidate_pop > best_pop:
                best = candidate
                best_pop = candidate_pop
    return best


def main():
    if len(sys.argv) != 5:
        raise RuntimeError("usage: test TB_R8 TB_R7 TOP_R2 SVA_R2")
    tb, old_tb, top, sva = [Path(arg) for arg in sys.argv[1:]]
    for path in (tb, old_tb, top, sva):
        require(path.is_file() and not path.is_symlink(), "non-regular input " + str(path))
    require(sha256(old_tb) == "d194f91293cf7e533e099d8b36956fb00db16402340c8e6e678059cb9adb0fd2", "r7 identity")
    require(sha256(top) == "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1", "top r2 changed")
    require(sha256(sva) == "b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b", "SVA r2 changed")
    text = tb.read_text(encoding="utf-8")

    witness = task_body(text, "make_dual_enqueue_masks")
    assignments = re.findall(r"stimulus_masks\[(\d+)\]\s*=\s*16'h([0-9a-fA-F]{4})", witness)
    observed = {int(row): int(value, 16) for row, value in assignments}
    expected = {0: 0x0001, 1: 0x0003, 2: 0x000c,
                3: 0x0031, 4: 0x004c, 5: 0x0083}
    require(observed == expected, "six-row witness changed: " + repr(observed))
    masks = [observed.get(row, 0) for row in range(64)]
    parents = [matcher_parent(masks, row) for row in range(6)]
    require(parents == [None, 0, None, 0, 2, 1], "witness parent map " + repr(parents))
    require([popcount(masks[row]) for row in range(6)] == [1, 2, 2, 3, 3, 3], "witness population order")
    refcounts = [0] * 64
    for row in range(64):
        parent = matcher_parent(masks, row)
        if parent is not None:
            refcounts[parent] += 1
    require(refcounts[0] == 2 and refcounts[1] == 1 and refcounts[2] == 1,
            "P0/A/P1 liveness witness")
    require(popcount(masks[1] ^ masks[0]) == 1, "A must complete in one residual beat")
    require(popcount(masks[2]) == 2, "P1 must have two residual beats")

    require("if (dut.prep_active_q && dut.exec_active_q)\n                    cov_pingpong_overlap" in text,
            "true internal pingpong counter missing")
    require("if (prep_valid && prep_ready)\n                    cov_pingpong_overlap" not in text,
            "old pingpong handshake proxy remains")
    require("force psum_write_ready = 1'b0;" in text and
            "force row_complete_ready = 1'b0;" in text and
            "repeat (96) @(posedge clk_core);" in text,
            "legal sink-stall overlap stimulus missing")
    require(text.count("release psum_write_ready;") >= 3 and
            text.count("release row_complete_ready;") >= 3,
            "sink force/release closure weakened")
    require("build_reference(16'd4);" in text and "load_task(16'd4);" in text and
            "wait_done(16'd4);" in text, "dual witness not executed")

    fatal_pos = text.find('normal coverage minima missed')
    p2_call_pos = text.rfind("make_consecutive_distinct_read_masks();")
    p2_gate_pos = text.find("P2 foundry response strength missed")
    held_call_pos = text.rfind("test_held_final_stale_parent_then_legal();")
    require(0 <= fatal_pos < p2_call_pos < p2_gate_pos < held_call_pos,
            "normal/P2/held-final phase order")
    attacks = ["attack_dirty_reserved();", "attack_stale_epoch();",
               "attack_overflow();", "attack_wrong_parent_and_dead_live();",
               "attack_read_before_write();", "attack_parent_only_nonzero_atomic();"]
    require(all(text.find(call, held_call_pos) > held_call_pos for call in attacks),
            "six attacks not retained after held-final")
    normal_fields = ["dead_plus_read", "deadline_read_write", "same_address_forward",
                     "pending_plus_forward", "full_no_credit", "liveness_sequences",
                     "parent_modes", "stalled_raw_recovery",
                     "stalled_raw_forward_recovery", "stalled_raw_response_recovery",
                     "pingpong_overlap", "endpoint_rows", "all_slices"]
    require(all("cov_" + field + " < 1" in text for field in normal_fields),
            "one or more normal cover minima weakened")
    require("PASS_M533_M528_DW1RW_R8_DIRECTED_RANDOM_AND_ATTACKS" in text,
            "R8 unique PASS token missing")

    print(json.dumps({
        "schema": "m814_m533_tb_r8_source_static_v1",
        "status": "PASS_SOURCE_STATIC_ONLY",
        "tb_r8_sha256": sha256(tb),
        "tb_r7_sha256": sha256(old_tb),
        "top_r2_sha256": sha256(top),
        "sva_r2_sha256": sha256(sva),
        "witness_masks": {str(k): "0x%04x" % v for k, v in sorted(expected.items())},
        "witness_parent_map": parents,
        "true_pingpong_counter": True,
        "normal_gate_precedes_p2": True,
        "p2_precedes_held_final_and_six_attacks": True,
        "vcs_or_simv_executed": False
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
