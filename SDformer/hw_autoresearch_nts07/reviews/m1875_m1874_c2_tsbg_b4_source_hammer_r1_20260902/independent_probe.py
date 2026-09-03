#!/usr/bin/env python3
"""Independent, source-only M1875 hammer for M1874.

This probe imports only the M1874 static checker.  It never launches VCS,
simv, an EDA tool, a license command, an attempt, a result, or a release.
Exact M1871 attacks and new M1875 attacks are evaluated against the semantic
validators directly; contract SHA rejection is deliberately not counted.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CHECKER = HW / "system_simulator/scripts/check_m1874_c2_tsbg_b4_source.py"
SPEC = importlib.util.spec_from_file_location("m1874_checker_for_m1875", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one_replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("mutation anchor cardinality is not one: " + old[:96])
    return text.replace(old, new, 1)


def verdict(kind, text):
    try:
        if kind == "tb":
            CHECK.validate_tb_text(text)
        elif kind == "sva":
            CHECK.validate_sva_text(text)
        elif kind == "rtl":
            CHECK.validate_rtl_text(text)
        else:
            raise RuntimeError("unknown mutation kind " + kind)
    except CHECK.CheckFailure as error:
        return {"rejected": True, "reason": str(error)}
    return {"rejected": False,
            "reason": "semantic validator accepted weakened source"}


def normalize_to_m1870(text):
    return text.replace("M1874", "M1870").replace("m1874", "m1870")


def normalize_to_m1794(text):
    value = text.replace("M1874", "M1794").replace("m1874", "m1794")
    value = value.replace("B4", "B8").replace("b4", "b8")
    value = value.replace("LRU4", "LRU8").replace("lru4", "lru8")
    value = value.replace("four independent 96-value Acc24 contexts",
                          "eight independent 96-value Acc24 contexts")
    value = value.replace("parameter int BUNDLE = 4", "parameter int BUNDLE = 8")
    value = value.replace("parameter int CACHE_ROWS = 4", "parameter int CACHE_ROWS = 8")
    value = value.replace("BUNDLE == 4 && SOURCES_PER_GROUP == 16",
                          "BUNDLE == 8 && SOURCES_PER_GROUP == 16")
    value = value.replace("OUTPUT_SLICES == 6 && CACHE_ROWS == 4 && LANES == 16",
                          "OUTPUT_SLICES == 6 && CACHE_ROWS == 8 && LANES == 16")
    return value


def independent_lru(schedule, capacity=4):
    cache = []
    hits = misses = evictions = 0
    for group in schedule:
        if group in cache:
            cache.remove(group)
            hits += 1
        else:
            misses += 1
            if len(cache) == capacity:
                cache.pop(0)
                evictions += 1
        cache.append(group)
    return {"rows": len(schedule), "hits": hits, "misses": misses,
            "evictions": evictions,
            "aggregate_eight_bank_bundle_beats": misses * 2 * 6,
            "scalar_bank_beats": misses * 2 * 6 * 8}


def independent_arithmetic():
    accumulators = [[([0] * 16) for _ in range(6)] for _ in range(4)]
    issues = products = 0
    positive = negative = exact_neg128 = False
    for context in range(4):
        for group in range(12):
            source0 = (context + group) % 8
            source1 = 8 + ((context * 3 + group) % 8)
            value0 = -1 if (context + group) % 2 == 0 else 1
            sources = ((0, source0, value0), (1, source1 - 8, -value0))
            for half in range(2):
                for output_slice in range(6):
                    issues += 1
                    for candidate_half, bank, source_value in sources:
                        if candidate_half != half:
                            continue
                        products += 16
                        positive = positive or source_value == 1
                        negative = negative or source_value == -1
                        for lane in range(16):
                            weight = ((group * 17 + half * 11 + output_slice * 7
                                       + bank * 5 + lane * 3) % 255 - 127)
                            if (group, half, output_slice, bank, lane) == (0, 0, 0, 0, 0):
                                weight = -128
                            product = source_value * weight
                            exact_neg128 = exact_neg128 or (
                                weight == -128 and source_value == -1 and product == 128)
                            accumulators[context][output_slice][lane] += product
    flat = [value for context in accumulators for output in context for value in output]
    return {"issues": issues, "signed_products": products, "commits": 24,
            "positive": positive, "negative": negative,
            "negative_int8_min_to_positive_128": exact_neg128,
            "minimum_accumulator": min(flat),
            "maximum_accumulator": max(flat)}


def main():
    tb = CHECK.TB.read_text(encoding="utf-8")
    sva = CHECK.SVA.read_text(encoding="utf-8")
    rtl = CHECK.RTL.read_text(encoding="utf-8")

    exact_m1871 = []
    for name, kind, source in CHECK.semantic_mutation_cases(tb, sva):
        item = {"name": name, "kind": kind}
        item.update(verdict(kind, source))
        exact_m1871.append(item)

    novel = []

    def add(name, kind, source, obligation):
        item = {"name": name, "kind": kind, "obligation": obligation}
        item.update(verdict(kind, source))
        novel.append(item)

    add("baseline_arithmetic_scoreboard_neutralized", "tb", one_replace(
        tb,
        "if (base.commit_accumulator[lane] !==\n"
        "                            expected[base.commit_context][base.commit_slice][lane])",
        "if (1'b0 && (base.commit_accumulator[lane] !==\n"
        "                            expected[base.commit_context][base.commit_slice][lane]))"),
        "baseline Acc24 result must be checked against the independent oracle")
    add("candidate_arithmetic_scoreboard_neutralized", "tb", one_replace(
        tb,
        "if (tsbg.commit_accumulator[lane] !==\n"
        "                            expected[tsbg.commit_context][tsbg.commit_slice][lane])",
        "if (1'b0 && (tsbg.commit_accumulator[lane] !==\n"
        "                            expected[tsbg.commit_context][tsbg.commit_slice][lane]))"),
        "TSBG Acc24 result must be checked against the independent oracle")
    add("all_default_sva_disabled", "sva", one_replace(
        sva, "default disable iff (rst_core);",
        "default disable iff (rst_core || 1'b1);"),
        "default disable must be reset-only rather than permanently true")
    add("bank_response_stability_antecedent_disabled", "sva", one_replace(
        sva,
        "mem_rsp_valid[bank] && !mem_rsp_ready[bank]\n"
        "                    && !protocol_error |=>",
        "mem_rsp_valid[bank] && !mem_rsp_ready[bank]\n"
        "                    && !protocol_error && 1'b0 |=>"),
        "returned typed weights and identities must remain stable under response backpressure")
    add("bridge_header_stability_antecedent_disabled", "sva", one_replace(
        sva, "bridge_valid && !bridge_ready |=> bridge_valid",
        "bridge_valid && !bridge_ready && 1'b0 |=> bridge_valid"),
        "bridge identity must remain stable under backpressure")
    add("bridge_payload_stability_antecedent_disabled", "sva", one_replace(
        sva, "bridge_valid && !bridge_ready |=>\n"
        "            $stable(bridge_source_channel)",
        "bridge_valid && !bridge_ready && 1'b0 |=>\n"
        "            $stable(bridge_source_channel)"),
        "typed signed source/weight payload must remain stable under backpressure")
    add("commit_header_stability_antecedent_disabled", "sva", one_replace(
        sva, "commit_valid && !commit_ready |=> commit_valid",
        "commit_valid && !commit_ready && 1'b0 |=> commit_valid"),
        "Acc24 commit identity must remain stable under backpressure")
    add("commit_payload_stability_antecedent_disabled", "sva", one_replace(
        sva, "commit_valid && !commit_ready |=> $stable(commit_accumulator));",
        "commit_valid && !commit_ready && 1'b0 |=> $stable(commit_accumulator));"),
        "Acc24 commit payload must remain stable under backpressure")

    forged = (
        "\n    initial begin : forged_pass_bypass\n"
        "        $display(\"PASS_M1874_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED "
        "rows=48 issues=576 products=9216 commits=24 bundles_base=576 "
        "bundles_tsbg=144 scalar_base=4608 scalar_tsbg=1152 stale=1 "
        "retired_replay=1 replay_accept=0 reset=2 recovery=1\");\n"
        "        $finish;\n"
        "    end\n")
    add("forged_time_zero_pass_and_finish", "tb", tb.replace(
        "\nendmodule\n\n`default_nettype wire",
        forged + "\nendmodule\n\n`default_nettype wire", 1),
        "PASS must be unique and causally downstream of all workload, replay, reset, SVA, and ledger obligations")

    controls = []
    for name, kind, source in (
            ("b4_parameter_to_b8", "rtl", one_replace(
                rtl, "parameter int BUNDLE = 4", "parameter int BUNDLE = 8")),
            ("candidate_hit_ledger_36_to_35", "tb", one_replace(
                tb, "tsbg.cache_hit_count != 36", "tsbg.cache_hit_count != 35"))):
        item = {"name": name, "kind": kind}
        item.update(verdict(kind, source))
        controls.append(item)

    token_schedule = [group for _context in range(4) for group in range(12)]
    group_schedule = [group for group in range(12) for _context in range(4)]
    baseline = independent_lru(token_schedule)
    candidate = independent_lru(group_schedule)
    arithmetic = independent_arithmetic()
    resources = {
        "shared_lru4_weight_data_bytes": 4 * 2 * 6 * 8 * 16,
        "four_by_96_acc24_context_bytes": 4 * 96 * 3,
        "b4_active_bitmap_bytes_at_48_groups": 4 * 48 * 16 // 8,
        "b4_sign_bitmap_bytes_at_48_groups": 4 * 48 * 16 // 8,
        "context_tag_bytes": 4 * 3,
    }
    resources["explicit_datapath_state_bytes_excluding_m803_control"] = sum(
        resources.values())

    output = {
        "schema": "m1875_m1874_c2_tsbg_b4_independent_probe_r1_v1",
        "status": "FAIL_CLOSED" if any(not item["rejected"] for item in novel)
                  else "PASS",
        "exact_m1871_attacks": {
            "count": len(exact_m1871),
            "rejected": sum(item["rejected"] for item in exact_m1871),
            "escaped": sum(not item["rejected"] for item in exact_m1871),
            "items": exact_m1871,
        },
        "novel_m1875_attacks": {
            "count": len(novel),
            "rejected": sum(item["rejected"] for item in novel),
            "escaped": sum(not item["rejected"] for item in novel),
            "items": novel,
        },
        "controls": {
            "count": len(controls),
            "rejected": sum(item["rejected"] for item in controls),
            "items": controls,
        },
        "independent_identity": {
            "m1874_to_m1870_namespace_normalization_byte_exact":
                normalize_to_m1870(rtl) == CHECK.M1870_RTL.read_text(encoding="utf-8"),
            "m1874_to_m1794_b4_lru4_normalization_byte_exact":
                normalize_to_m1794(rtl) == CHECK.M1794_RTL.read_text(encoding="utf-8"),
            "docs359_sha256": sha(CHECK.DOC359),
        },
        "independent_ledgers": {
            "baseline_lru4": baseline,
            "candidate_lru4": candidate,
            "work_and_typed_signed": arithmetic,
            "resource_source_model": resources,
            "production_acc24_absolute_bound": 48 * 16 * 128,
            "signed_acc24_limit": 1 << 23,
        },
        "execution_boundary": {
            "vcs": 0, "simv": 0, "eda": 0, "license_queries": 0,
            "attempts": 0, "results": 0, "releases": 0,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 1 if output["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
