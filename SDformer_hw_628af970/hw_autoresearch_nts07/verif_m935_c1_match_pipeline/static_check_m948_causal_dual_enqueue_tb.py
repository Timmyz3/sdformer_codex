#!/usr/bin/env python3
"""Fail-closed source-only checks for the unique M948 TB-only successor.

This script is intentionally not an HDL compiler and does not authorize VCS,
timing, PPA, speedup, energy, or paper claims.  It proves source identity,
expected additive boundaries, the exact software form of the parent rule, and
the presence of the planned TB/SVA gates before any commercial-tool attempt.
"""

import hashlib
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_RTL = ROOT / "rtl_m912_c1_pipeline/m912_m528_metadata_pipelined_product_capture_island.sv"
NEW_RTL = ROOT / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
OLD_SVA = ROOT / "verif_m912_c1_pipeline/m919_m912_metadata_pipeline_assertions_r2.sv"
NEW_EXEC_SVA = ROOT / "verif_m935_c1_match_pipeline/m935_m912_inherited_execution_assertions_r1.sv"
NEW_MATCH_SVA = ROOT / "verif_m935_c1_match_pipeline/m938_three_stage_exact_match_assertions_r2.sv"
NEW_TB = ROOT / "verif_m935_c1_match_pipeline/tb_m948_three_stage_match_pipeline_unit_delay_r3.sv"
BASE_TB = ROOT / "verif_m935_c1_match_pipeline/tb_m938_three_stage_match_pipeline_unit_delay_r2.sv"
FROZEN_359 = ROOT / "docs/359_DATE终局冻结_20260813.md"
M937_DIR = ROOT / "reviews/m937_m935_m912_three_stage_exact_parent_match_source_hammer_r1_20260829"
M937_REVIEW_MD = M937_DIR / "review.md"
M937_REVIEW_JSON = M937_DIR / "review.json"
M937_MANIFEST = M937_DIR / "SHA256SUMS"
M937_OUTER = M937_DIR / "SHA256SUMS.seal.sha256"
M947_DIR = ROOT / "reviews/m947_m943_m938_c1_vcs_failure_forensic_r1_20260829"
M947_REVIEW_MD = M947_DIR / "review.md"
M947_REVIEW_JSON = M947_DIR / "review.json"
M947_MANIFEST = M947_DIR / "SHA256SUMS"
M947_OUTER = M947_DIR / "SHA256SUMS.seal.sha256"

EXPECTED = {
    OLD_RTL: "eef2f8d3344620cfbf518bf4ac382a2f0be5b46084d56308a660e4c172c65e53",
    NEW_RTL: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    OLD_SVA: "7dfb91f6d11aa2be8f8c9472ba3784145f290215b67a826fd9f53e32c22b7837",
    NEW_EXEC_SVA: "ad89adc7e9aefd350a225e58e85540ec579bbbe1ce9730891633f311de4eb4f5",
    NEW_MATCH_SVA: "eb20ffb5f910d0e3b8eebf836194298d38c719f512b207f38d15e75fc2df9f07",
    BASE_TB: "6b5d58bd35176b5532c21526c6406eaaf7928693c90d1daea51a23998c260e9e",
    NEW_TB: "ab4b4d41ae1daedced757b9682f9b005776921eff4f2f1b9ae2dc40e654388e3",
    M937_REVIEW_MD: "a4c4ae5cd5b24820ff419caced795b0ae08d0cea1194698b444110ed0cba52e8",
    M937_REVIEW_JSON: "34d1f64ba97da8209f1c9fd0976c082e880ea336b276375c2d5d10b2f0c5be78",
    M937_MANIFEST: "182ece3a8b389ab5d6d495c86f1890a53d6c9aa45057b81058a5236543b10b39",
    M937_OUTER: "752b28ab04502efeb7436443960c2aa840e28c8a0a76c709986b932548691097",
    M947_REVIEW_MD: "ddd4436612015e632706e02e77305563aba2dc6569565cac3b02028eb100e5ae",
    M947_REVIEW_JSON: "1fe2b0458b492abebdc07a136980ac5c2e6a971771bd5515a5ff5491a0a94c71",
    M947_MANIFEST: "bb33245c2d7b6334ca9c4d97bb2f5f9d20f0c5830c4f94d197e889da7afc3e41",
    M947_OUTER: "dffdc6802e859624fe101654b23f634e446f7414a04716917af983237ee12d99",
    FROZEN_359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL_M948_STATIC: {message}")


def normalized_port_block(text, module):
    match = re.search(
        rf"module\s+{re.escape(module)}\s*\((.*?)\n\);", text, re.S
    )
    require(match is not None, f"module port block absent: {module}")
    return re.sub(r"\s+", " ", match.group(1)).strip()


def pc16(value):
    return bin(value & 0xffff).count("1")


def original_parent(masks, row):
    current = masks[row]
    current_pop = pc16(current)
    best = -1
    best_pop = 0
    if current_pop >= 2:
        for candidate, candidate_mask in enumerate(masks):
            candidate_pop = pc16(candidate_mask)
            if (
                (candidate_mask & current) == candidate_mask
                and candidate_pop >= 1
                and not (candidate_mask == current and candidate >= row)
                and candidate_pop > best_pop
            ):
                best = candidate
                best_pop = candidate_pop
    return best


def tree_winner(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if lhs[0] != rhs[0]:
        return lhs if lhs[0] > rhs[0] else rhs
    return lhs if lhs[1] <= rhs[1] else rhs


def m935_parent(masks, row):
    current = masks[row]
    leaves = []
    for candidate, candidate_mask in enumerate(masks):
        candidate_pop = pc16(candidate_mask)
        legal = (
            pc16(current) >= 2
            and candidate_pop >= 1
            and (candidate_mask & current) == candidate_mask
            and not (candidate_mask == current and candidate >= row)
        )
        leaves.append((candidate_pop, candidate, candidate_mask) if legal else None)
    local = []
    for group in range(8):
        level = leaves[group * 8:(group + 1) * 8]
        while len(level) > 1:
            level = [tree_winner(level[i], level[i + 1])
                     for i in range(0, len(level), 2)]
        local.append(level[0])
    while len(local) > 1:
        local = [tree_winner(local[i], local[i + 1])
                 for i in range(0, len(local), 2)]
    return -1 if local[0] is None else local[0][1]


def algorithm_check():
    directed = [0] * 64
    directed[:17] = [
        0x0001, 0x0003, 0x0005, 0x0007, 0x0003, 0x000F,
        0x003F, 0x00FF, 0x0FFF, 0xFFFF, 0x00F3, 0x0303,
        0x3333, 0x5555, 0xAAAA, 0x8000, 0x8001,
    ]
    directed[31] = 0x00FF
    directed[47] = 0x0F0F
    directed[63] = 0xFFFF
    corpora = [directed]
    rng = random.Random(0x935_528)
    for _ in range(4096):
        masks = [rng.randrange(1 << 16) if rng.randrange(9) else 0
                 for _ in range(64)]
        # Force exact ties and nested subsets into every random corpus.
        masks[0] = 1
        masks[1] = 3
        masks[4] = 3
        masks[31] = masks[7]
        masks[63] = 0xFFFF
        corpora.append(masks)
    rows = 0
    for masks in corpora:
        for row in range(64):
            expected = original_parent(masks, row)
            got = m935_parent(masks, row)
            require(got == expected,
                    f"tree mismatch row={row} got={got} expected={expected}")
            rows += 1
    require(original_parent(directed, 4) == 1,
            "same-pop lowest-id directed witness absent")
    return rows


def main():
    for path, expected in EXPECTED.items():
        require(path.is_file(), f"required frozen source absent: {path}")
        require(sha256(path) == expected,
                f"frozen SHA drift: {path} got={sha256(path)}")
    for path in (NEW_RTL, NEW_EXEC_SVA, NEW_MATCH_SVA, NEW_TB):
        require(path.is_file(), f"new candidate source absent: {path}")

    m937 = json.loads(M937_REVIEW_JSON.read_text())
    require(m937["review_status"] == "PASS_M937_SOURCE_HAMMER",
            "M937 review status drift")
    require(m937["verdict"] == "REPAIR_BEFORE_VCS_RELEASE",
            "M937 repair verdict drift")
    require(m937["issue_counts"] == {"P0": 0, "P1": 4, "P2": 3},
            "M937 issue-count contract drift")
    require(M937_OUTER.read_text().strip()
            == "182ece3a8b389ab5d6d495c86f1890a53d6c9aa45057b81058a5236543b10b39  SHA256SUMS",
            "M937 outer seal content drift")
    m947 = json.loads(M947_REVIEW_JSON.read_text())
    require(m947["review_status"]
            == "PASS_M947_M943_FAILURE_FORENSIC",
            "M947 review status drift")
    require(m947["design_verdict"]
            == "FAIL_M943_TB_COVERAGE_ONLY__DO_NOT_RERUN__ONE_TB_ONLY_SUCCESSOR",
            "M947 unique-successor verdict drift")
    require(m947["issue_counts"] == {"P0": 0, "P1": 1, "P2": 1},
            "M947 issue-count contract drift")
    require(m947["successor"]["count_max"] == 1
            and m947["successor"]["tb_only"] is True
            and m947["successor"]["rtl_sva_unchanged"] is True
            and m947["successor"]["preserve_minimum"] is True
            and m947["successor"]["preserve_cover"] is True,
            "M947 successor boundary drift")
    require(M947_OUTER.read_text().strip()
            == "bb33245c2d7b6334ca9c4d97bb2f5f9d20f0c5830c4f94d197e889da7afc3e41  SHA256SUMS",
            "M947 outer seal content drift")

    old = OLD_RTL.read_text()
    new = NEW_RTL.read_text()
    old_ports = normalized_port_block(
        old, "m912_m528_metadata_pipelined_product_capture_island")
    new_ports = normalized_port_block(
        new, "m935_m912_three_stage_exact_parent_match_product_capture_island")
    require(old_ports == new_ports, "external top-level port contract changed")

    # Everything from the frozen execution ownership block to EOF must remain
    # byte-identical, including 1RW arbitration, parent queue, issue, psum,
    # completion and atomic fault semantics.
    exec_anchor = "            if (!exec_active_q && ready_bank_valid_w) begin"
    require(old[old.index(exec_anchor):] == new[new.index(exec_anchor):],
            "execution-side sequential tail drifted from M912")
    comb_start = "    // Directory stage:"
    comb_end = "    integer reset_lane;"
    require(
        old[old.index(comb_start):old.index(comb_end)]
        == new[new.index(comb_start):new.index(comb_end)],
        "execution-side combinational/arbitration cone drifted from M912",
    )

    required_rtl = [
        "function automatic logic [27:0] parent_winner",
        "directory_q[free_bank_w][prep_row_id] <= {",
        "directory_q[prep_bank_q][prep_row_id] <= {",
        "match_f_valid_q <= match_active_q && !match_issue_done_q;",
        "match_g_valid_q <= match_f_valid_q;",
        "match_g_winner_q[group] <= match_octet_w[group];",
        "directory_q[match_g_bank_q][match_g_row_q]",
        "if (match_g_row_q == 6'd63) begin",
        "bank_state_q[match_g_bank_q] <= BANK_READY;",
    ]
    for token in required_rtl:
        require(token in new, f"required RTL token absent: {token}")
    require("popcount16(mask_q[match_f_bank_q][candidate])" not in new,
            "candidate popcount was recomputed instead of using prep cache")
    require(new.count("match_octet_w[group] = parent_winner(") == 1,
            "balanced eight-group local reduction not unique")
    require(new.count("match_r_winner_w = parent_winner(") == 1,
            "balanced 8-to-1 R reduction not unique")
    require(new.count("BANK_READY;") == old.count("BANK_READY;"),
            "unexpected additional BANK_READY assignment")

    # Added state: issue_done (1), F tuple (29), G tuple (29), and eight
    # compact 28-bit winners (224) = 283 bits including valid bits.  The old
    # match bank/row/active registers are reused; cached 64x5 popcounts occupy
    # pre-existing directory bits and add no storage.
    added_metadata_bits = 1 + (1 + 1 + 6 + 16 + 5) \
        + (1 + 1 + 6 + 16 + 5) + 8 * 28
    require(added_metadata_bits == 283, "internal metadata accounting bug")
    require(added_metadata_bits < 512, "metadata FF gate exceeded")
    require("logic [1151:0] match_" not in new
            and "logic [1823:0] match_" not in new,
            "wide payload was copied into match pipeline")

    old_sva = OLD_SVA.read_text()
    inherited_sva = NEW_EXEC_SVA.read_text().replace(
        "m935_m912_inherited_execution_assertions_r1",
        "m919_m912_metadata_pipeline_assertions_r2",
    ).replace(
        "bind m935_m912_three_stage_exact_parent_match_product_capture_island",
        "bind m912_m528_metadata_pipelined_product_capture_island",
    )
    require(old_sva == inherited_sva,
            "inherited M919 execution SVA changed beyond names/bind target")

    match_sva = NEW_MATCH_SVA.read_text()
    for token in (
        "ap_f_ii1", "ap_g_follows_f", "ap_r_directory_format",
        "ap_r_parent_legal", "ap_ready_after_r63_commit",
        "ap_overlap_is_bank_distinct", "cp_full_64_row_ii1",
        "cp_bank_distinct_overlap",
        "cp_same_pop_lowest_id_witness",
    ):
        require(token in match_sva, f"new match SVA gate absent: {token}")
    require("|| exec_active" not in match_sva,
            "row63 READY remains masked by unqualified exec_active")
    require("match_bank_state == BANK_READY" in match_sva
            and "match_g_bank == $past(match_g_bank)" in match_sva,
            "row63 READY is not qualified to the drained bank")
    require("match_g_valid && exec_active |-> match_g_bank != exec_bank"
            in match_sva,
            "same-bank match/execute exclusion absent")

    tb = NEW_TB.read_text()
    for token in (
        "oracle_match_countdown[slot] = 67;",
        "M938 matcher miter mismatch",
        "COVERAGE_M938_EXACT_MATCH_PIPELINE",
        "COVERAGE_M938_MATCH_RESET",
        "external_match_expected_epoch_q[2:0]",
        "external_bank_owned_q",
        "dut.match_f_bank_q",
        "dut.match_g_bank_q === dut.exec_bank_q",
        "make_random_masks(32'h9380_0002);",
        "reset_during_match_stage(0, 16'd900, 16'd910);",
        "reset_during_match_stage(1, 16'd901, 16'd911);",
        "reset_during_match_stage(2, 16'd902, 16'd912);",
        "no_stale_directory_write=1",
        "attack_dirty_reserved();",
        "attack_stale_epoch();",
        "attack_overflow();",
        "attack_wrong_parent_and_dead_live();",
        "attack_read_before_write();",
        "attack_parent_only_nonzero_atomic();",
        "PASS_M948_C1_THREE_STAGE_EXACT_MATCH_PIPELINE_UNIT_DELAY_CANDIDATE",
    ):
        require(token in tb, f"TB inheritance/oracle token absent: {token}")
    require("slot = dut.bank_epoch_q" not in tb,
            "DUT bank_epoch still selects its own reference slot")
    require("if (dut.exec_active_q)\n                cov_match_bank_overlap"
            not in tb,
            "overlap counter remains unqualified by distinct bank")

    base_tb = BASE_TB.read_text()
    masks_start = "    task automatic make_dual_enqueue_masks;"
    masks_end = "    task automatic make_random_masks"
    require(base_tb[base_tb.index(masks_start):base_tb.index(masks_end)]
            == tb[tb.index(masks_start):tb.index(masks_end)],
            "six-row dual-enqueue mask corpus changed")
    minima_start = "        if (cov_dead_plus_read < 1"
    minima_end = "        $display(\"COVERAGE_M938_EXACT_MATCH_PIPELINE"
    require(base_tb[base_tb.index(minima_start):base_tb.index(minima_end)]
            == tb[tb.index(minima_start):tb.index(minima_end)],
            "normal coverage minima or gate text changed")

    causal_start = "    task automatic run_causal_dual_enqueue_task"
    causal_end = tb.index("    endtask", tb.index(causal_start)) \
        + len("    endtask")
    causal = tb[tb.index(causal_start):causal_end]
    for token in (
        "force psum_write_ready = 1'b1;",
        "force row_complete_ready = 1'b1;",
        "dut.macro_read_accept_w",
        "dut.pf_token_consumer_q != 6'd3",
        "dut.pf_token_parent_q != 6'd0",
        "dut.read_pending_q",
        "dut.read_pending_consumer_q != 6'd3",
        "dut.read_pending_parent_q != 6'd0",
        "dut.forward_accept_w",
        "dut.pf_token_consumer_q != 6'd4",
        "dut.pf_token_parent_q != 6'd2",
        "debug_dual_enqueue_event",
        "debug_read_response_event",
        "debug_forward_event",
        "cov_pending_plus_forward < 1",
        "release psum_write_ready;",
        "release row_complete_ready;",
    ):
        require(token in causal, f"causal phase token absent: {token}")
    require(causal.count("force ") == 2 and causal.count("release ") == 2,
            "causal task force/release scope is not exactly two public sinks")
    require("force dut." not in causal and "release dut." not in causal,
            "causal task forces internal DUT state")
    require(tb.count("run_causal_dual_enqueue_task") == 2,
            "causal task is not a unique declaration plus epoch4 call")
    require("run_causal_dual_enqueue_task(16'd4);" in tb,
            "epoch4 does not consume the causal witness")
    require("COVERAGE_M948_CAUSAL_DUAL_ENQUEUE" in tb
            and "public_sink_window=1 internal_force=0" in tb,
            "causal witness receipt/claim boundary absent")

    checked_rows = algorithm_check()
    print(
        "PASS_M948_CAUSAL_DUAL_ENQUEUE_TB_SOURCE_STATIC "
        f"algorithm_rows={checked_rows} metadata_bits={added_metadata_bits} "
        "ports_unchanged=true execution_tail_byte_exact=true "
        "inherited_m919_sva_exact=true m937_bound=true m947_bound=true "
        "bank_distinct_overlap=true row63_ready_bank_qualified=true "
        "external_bank_epoch_oracle=true distinct_overlap_masks=true "
        "reset_F_G_R63_present=true dual_masks_unchanged=true "
        "coverage_minima_unchanged=true public_sink_force_only=true "
        "causal_macro_response_forward_debug=true unique_successor=true "
        "source_only=true vcs=false dc=false "
        "timing=false speedup=false ppa=false energy=false system=false "
        "headline=false"
    )


if __name__ == "__main__":
    main()
