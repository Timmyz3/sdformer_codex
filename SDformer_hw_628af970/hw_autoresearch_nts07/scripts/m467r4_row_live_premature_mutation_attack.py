#!/usr/bin/env python3
import json

SLOTS = 8
poison = [1000 + 17 * slot for slot in range(SLOTS)]
delta = [-31 + slot for slot in range(SLOTS)]


def update(set_after_slot):
    live = False
    stored = poison[:]
    reads = []
    for slot in range(SLOTS):
        base = stored[slot] if live else 0
        reads.append({"slot": slot, "live_before": live, "base": base})
        stored[slot] = base + delta[slot]
        if slot == set_after_slot:
            live = True
    return stored, reads


def commit(clear_after_slot):
    live = True
    stored = [2000 + 19 * slot for slot in range(SLOTS)]
    committed = []
    for slot in range(SLOTS):
        committed.append(stored[slot] if live else 0)
        if slot == clear_after_slot:
            live = False
    return committed


early_update, early_reads = update(0)
correct_update, correct_reads = update(7)
early_commit = commit(0)
correct_commit = commit(7)

assert early_update[1] == poison[1] + delta[1]
assert correct_update == delta
assert early_commit[1] == 0
assert correct_commit == [2000 + 19 * slot for slot in range(SLOTS)]

print(json.dumps({
    "milestone": "M467R4_ROW_LIVE_PREMATURE_MUTATION_ATTACK",
    "status": "COUNTEREXAMPLE_REPRODUCED",
    "early_set_counterexample": {
        "mutation": "set row-live after slot0 write",
        "first_bad_slot": 1,
        "poison_base_consumed": early_reads[1]["base"],
        "observed": early_update[1],
        "expected_zero_based": delta[1]
    },
    "early_clear_counterexample": {
        "mutation": "clear row-live after slot0 commit",
        "first_bad_slot": 1,
        "observed": early_commit[1],
        "expected_live_commit": correct_commit[1]
    },
    "admitted_rule": {
        "set_after_slot": 7,
        "clear_after_slot": 7,
        "all_first_touches_zero_based": correct_update == delta,
        "all_live_commits_preserved": correct_commit == [2000 + 19 * slot for slot in range(SLOTS)]
    },
    "claims": {"rtl_simulation": False, "performance": False, "ppa": False,
               "system_or_full_network": False, "headline": False}
}, indent=2))
