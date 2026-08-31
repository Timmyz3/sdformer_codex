#!/usr/bin/env python3
"""Exact source-only admission for the M849 TB-r9 P2 epoch repair."""

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


def main():
    if len(sys.argv) != 8:
        raise RuntimeError("usage: TEST TB_R9 TB_R8 TOP_R2 SVA_R2 MACRO BINDING FOUNDRY_V")
    tb, old_tb, top, sva, macro, binding, foundry = [Path(arg) for arg in sys.argv[1:]]
    for path in (tb, old_tb, top, sva, macro, binding, foundry):
        require(path.is_file() and not path.is_symlink(), "non-regular input " + str(path))
    require(sha256(old_tb) == "cd0cf9f91ffce8dcb35f5326ba77d2b99a0d86653c100af5593e7bc40001a9a4", "TB r8 identity")
    require(sha256(top) == "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1", "top r2 drift")
    require(sha256(sva) == "b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b", "SVA r2 drift")
    require(sha256(macro) == "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783", "macro adapter drift")
    require(sha256(binding) == "db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983", "binding plan drift")
    require(sha256(foundry) == "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d", "foundry model drift")

    old = old_tb.read_text(encoding="utf-8")
    new = tb.read_text(encoding="utf-8")
    replacements = (
        ("build_reference(16'd3);", "build_reference(16'd14);"),
        ("load_task(16'd3);", "load_task(16'd14);"),
        ("wait_done(16'd3);", "wait_done(16'd14);"),
    )
    reconstructed = new
    for before, after in replacements:
        require(old.count(before) == 1, "old literal multiplicity: " + before)
        require(new.count(after) == 1, "new literal multiplicity: " + after)
        require(new.count(before) == 0, "stale P2 epoch remains: " + before)
        reconstructed = reconstructed.replace(after, before, 1)
    require(reconstructed == old, "TB r8->r9 delta is not exactly the three P2 epoch literals")

    coverage = new.index("COVERAGE_M533_M528_DW1RW_R8")
    p2_build = new.index("build_reference(16'd14);", coverage)
    p2_load = new.index("load_task(16'd14);", p2_build)
    p2_wait = new.index("wait_done(16'd14);", p2_load)
    p2_token = new.index("P2_STRENGTH_M533_M528_DW1RW_R3", p2_wait)
    held = new.index("test_held_final_stale_parent_then_legal();", p2_token)
    attack_calls = [
        "attack_dirty_reserved();", "attack_stale_epoch();", "attack_overflow();",
        "attack_wrong_parent_and_dead_live();", "attack_read_before_write();",
        "attack_parent_only_nonzero_atomic();",
    ]
    attack_positions = [new.index(call, held) for call in attack_calls]
    final_pass = new.index("PASS_M533_M528_DW1RW_R8_DIRECTED_RANDOM_AND_ATTACKS", attack_positions[-1])
    require(coverage < p2_build < p2_load < p2_wait < p2_token < held
            < min(attack_positions) <= max(attack_positions) < final_pass,
            "normal/P2/held/attack/final phase order")
    require(new.count("reset_dut();", coverage, p2_build) == 0,
            "reset inserted between normal epoch frontier and P2")

    normal_epochs = [1, 2, 4, 10, 11, 12, 13]
    require(all(epoch < 14 for epoch in normal_epochs) and normal_epochs == sorted(normal_epochs),
            "normal epoch sequence expectation")
    for epoch in (1, 2, 4):
        require("build_reference(16'd{});".format(epoch) in new, "missing fixed normal epoch")
    require("build_reference(10 + test_index);" in new
            and "load_task(10 + test_index);" in new
            and "wait_done(10 + test_index);" in new,
            "10..13 normal epoch loop changed")
    require("normal_covers=13" in new and "minima=1" in new,
            "13-cover normal gate weakened")
    require("minima_pairs=1 minima_responses=2" in new,
            "P2 minima token weakened")
    require("attack_count != 6" in new and "attacks=%0d" in new,
            "six-attack/final token gate weakened")

    print(json.dumps({
        "schema": "m849_m533_tb_r9_epoch14_source_static_v1",
        "status": "PASS_EXACT_THREE_LITERAL_P2_EPOCH_REPAIR_SOURCE_ONLY",
        "tb_r9_sha256": sha256(tb),
        "tb_r8_sha256": sha256(old_tb),
        "semantic_diff": ["build_reference:3->14", "load_task:3->14", "wait_done:3->14"],
        "normal_epochs": normal_epochs,
        "p2_epoch": 14,
        "strict_monotonic": True,
        "reset_inserted_between_normal_and_p2": False,
        "normal_cover_count": 13,
        "p2_token_retained": True,
        "held_final_retained": True,
        "protocol_attack_count": 6,
        "final_pass_token_retained": True,
        "rtl_sva_macro_binding_foundry_frozen": True,
        "vcs_or_simv_executed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
