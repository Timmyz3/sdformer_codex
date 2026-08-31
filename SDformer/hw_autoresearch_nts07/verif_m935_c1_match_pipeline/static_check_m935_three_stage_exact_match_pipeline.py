#!/usr/bin/env python3
"""Fail-closed source-only checks for the additive M935 candidate.

This script is intentionally not an HDL compiler and does not authorize VCS,
timing, PPA, speedup, energy, or paper claims.  It proves source identity,
expected additive boundaries, the exact software form of the parent rule, and
the presence of the planned TB/SVA gates before any commercial-tool attempt.
"""

import hashlib
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_RTL = ROOT / "rtl_m912_c1_pipeline/m912_m528_metadata_pipelined_product_capture_island.sv"
NEW_RTL = ROOT / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
OLD_SVA = ROOT / "verif_m912_c1_pipeline/m919_m912_metadata_pipeline_assertions_r2.sv"
NEW_EXEC_SVA = ROOT / "verif_m935_c1_match_pipeline/m935_m912_inherited_execution_assertions_r1.sv"
NEW_MATCH_SVA = ROOT / "verif_m935_c1_match_pipeline/m935_three_stage_exact_match_assertions_r1.sv"
NEW_TB = ROOT / "verif_m935_c1_match_pipeline/tb_m935_three_stage_match_pipeline_unit_delay_r1.sv"
FROZEN_359 = ROOT / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    OLD_RTL: "eef2f8d3344620cfbf518bf4ac382a2f0be5b46084d56308a660e4c172c65e53",
    OLD_SVA: "7dfb91f6d11aa2be8f8c9472ba3784145f290215b67a826fd9f53e32c22b7837",
    FROZEN_359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL_M935_STATIC: {message}")


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
        "cp_full_64_row_ii1", "cp_bank_overlap",
        "cp_same_pop_lowest_id_witness",
    ):
        require(token in match_sva, f"new match SVA gate absent: {token}")

    tb = NEW_TB.read_text()
    for token in (
        "oracle_match_countdown[slot] = 67;",
        "M935 matcher miter mismatch",
        "COVERAGE_M935_EXACT_MATCH_PIPELINE",
        "attack_dirty_reserved();",
        "attack_stale_epoch();",
        "attack_overflow();",
        "attack_wrong_parent_and_dead_live();",
        "attack_read_before_write();",
        "attack_parent_only_nonzero_atomic();",
        "PASS_M935_C1_THREE_STAGE_EXACT_MATCH_PIPELINE_UNIT_DELAY_CANDIDATE",
    ):
        require(token in tb, f"TB inheritance/oracle token absent: {token}")

    checked_rows = algorithm_check()
    print(
        "PASS_M935_THREE_STAGE_EXACT_MATCH_SOURCE_STATIC "
        f"algorithm_rows={checked_rows} metadata_bits={added_metadata_bits} "
        "ports_unchanged=true execution_tail_byte_exact=true "
        "inherited_m919_sva_exact=true source_only=true vcs=false dc=false "
        "timing=false speedup=false ppa=false energy=false system=false "
        "headline=false"
    )


if __name__ == "__main__":
    main()
