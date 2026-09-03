#!/usr/bin/env python3
"""Different-author, source-only M1881 fail-closed hammer for M1880.

The probe imports only the M1880 static checker.  It does not launch VCS,
simv, EDA, a license query, an attempt, a result, a campaign, or a release.
Mutations are sent directly to the semantic validators so rejection is not
credited merely to the sealed top-level source inventory.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CHECKER = HW / "system_simulator/scripts/check_m1880_c2_tsbg_b4_source.py"
SPEC = importlib.util.spec_from_file_location("m1880_checker_for_m1881", str(CHECKER))
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
        raise RuntimeError("mutation anchor cardinality is not one: " + old[:100])
    return text.replace(old, new, 1)


def verdict(kind, source):
    try:
        if kind == "tb":
            CHECK.validate_tb_text(source)
        elif kind == "sva":
            CHECK.validate_sva_text(source)
        elif kind == "rtl":
            CHECK.validate_rtl_text(source)
        else:
            raise RuntimeError("unknown mutation kind " + kind)
    except CHECK.CheckFailure as error:
        return {"rejected": True, "reason": str(error)}
    return {"rejected": False, "reason": "semantic validator accepted weakened source"}


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


def normalize_to_m1874(text):
    return text.replace("M1880", "M1874").replace("m1880", "m1874")


def normalize_to_m1870(text):
    return text.replace("M1880", "M1870").replace("m1880", "m1870")


def normalize_to_m1794(text):
    value = text.replace("M1880", "M1794").replace("m1880", "m1794")
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


def main():
    tb = CHECK.TB.read_text(encoding="utf-8")
    sva = CHECK.SVA.read_text(encoding="utf-8")
    rtl = CHECK.RTL.read_text(encoding="utf-8")

    exact_m1871 = []
    for name, kind, source in CHECK.semantic_mutation_cases(tb, sva):
        item = {"name": name, "kind": kind}
        item.update(verdict(kind, source))
        exact_m1871.append(item)

    exact_m1875 = []
    for name, kind, source in CHECK.m1875_mutation_cases(tb, sva):
        item = {"name": name, "kind": kind}
        item.update(verdict(kind, source))
        exact_m1875.append(item)

    novel = []

    def add(name, kind, source, obligation):
        item = {"name": name, "kind": kind, "obligation": obligation}
        item.update(verdict(kind, source))
        novel.append(item)

    add("baseline_scoreboard_self_comparison", "tb", one_replace(
        tb,
        "expected[base.commit_context][base.commit_slice][lane])",
        "base.commit_accumulator[lane])"),
        "baseline Acc24 must compare against the independent expected array")
    add("candidate_scoreboard_self_comparison", "tb", one_replace(
        tb,
        "expected[tsbg.commit_context][tsbg.commit_slice][lane])",
        "tsbg.commit_accumulator[lane])"),
        "TSBG Acc24 must compare against the independent expected array")
    add("baseline_duplicate_commit_guard_neutralized", "tb", one_replace(
        tb,
        "if (observed_base[base.commit_context][base.commit_slice])",
        "if (observed_base[base.commit_context][base.commit_slice] && 1'b0)"),
        "baseline scoreboard must reject duplicate commits")
    add("candidate_duplicate_commit_guard_neutralized", "tb", one_replace(
        tb,
        "if (observed_tsbg[tsbg.commit_context][tsbg.commit_slice])",
        "if (observed_tsbg[tsbg.commit_context][tsbg.commit_slice] && 1'b0)"),
        "TSBG scoreboard must reject duplicate commits")
    add("finish_changed_to_conditional_endpoint", "tb", one_replace(
        tb, "$finish;", "if (1'b1) $finish;"),
        "the sole finish must remain an unconditional endpoint after final ledgers")
    add("pass_moved_before_protocol_ledger", "tb", one_replace(
        tb,
        "if (stale_attack_count != 1 || retired_identity_replay_count != 1",
        "$display(\"PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED_EARLY\");\n"
        "        if (stale_attack_count != 1 || retired_identity_replay_count != 1"),
        "no alternative PASS-like endpoint may precede the final protocol ledger")
    add("default_disable_extended_to_fault", "sva", one_replace(
        sva, "default disable iff (rst_core);",
        "default disable iff (rst_core || protocol_error);"),
        "the global SVA disable must be reset-only")
    add("request_stability_consequent_drops_valid", "sva", one_replace(
        sva,
        "mem_req_valid[bank]\n"
        "                    && $stable({mem_req_epoch[bank]",
        "1'b1\n"
        "                    && $stable({mem_req_epoch[bank]"),
        "a stalled bank request must retain valid as well as its identity")
    add("request_stability_payload_drops_source_channel", "sva", one_replace(
        sva,
        "mem_req_output_block[bank], mem_req_slice[bank],\n"
        "                        mem_req_source_channel[bank]}));",
        "mem_req_output_block[bank], mem_req_slice[bank]}));"),
        "bank request source channel must remain stable under backpressure")
    add("response_stability_drops_weight_payload", "sva", one_replace(
        sva,
        "                    && $stable(mem_rsp_weight[bank]));",
        "                    && 1'b1);"),
        "bank response weights must remain stable under backpressure")
    add("bridge_header_stability_drops_bank_valid", "sva", one_replace(
        sva,
        "bridge_slice, bridge_bank_valid}));",
        "bridge_slice}));"),
        "bridge bank-valid header must remain stable under backpressure")
    add("bridge_payload_stability_drops_effective_weight", "sva", one_replace(
        sva,
        "            && $stable(bridge_effective_weight));",
        "            && 1'b1);"),
        "bridge effective weights must remain stable under backpressure")
    add("commit_header_stability_drops_terminal", "sva", one_replace(
        sva,
        "commit_context, commit_tag, commit_slice,\n"
        "                        commit_terminal}));",
        "commit_context, commit_tag, commit_slice}));"),
        "commit terminal identity must remain stable under backpressure")
    add("commit_payload_checks_only_lane_zero", "sva", one_replace(
        sva,
        "$stable(commit_accumulator));",
        "$stable(commit_accumulator[0]));"),
        "all Acc24 lanes must remain stable under commit backpressure")
    add("post_reset_response_ledger_removed", "tb", one_replace(
        tb,
        "                || base.scalar_bank_response_count != 96\n"
        "                || tsbg.scalar_bank_response_count != 96",
        "                || 1'b0\n"
        "                || 1'b0"),
        "post-reset legal service must account every bank response")
    add("post_reset_terminal_ledger_removed", "tb", one_replace(
        tb,
        "                || terminal_base != 4 || terminal_tsbg != 4)",
        "                || 1'b0 || 1'b0)"),
        "post-reset legal service must terminate all four contexts")
    add("second_reset_duration_zero", "tb", one_replace(
        tb,
        "// Three reset clocks satisfy the >=1-cycle recovery cover.  Do not\n"
        "        // stop at flag clearing: run the minimum complete legal B4 service.\n"
        "        rst_core = 1;\n"
        "        repeat (3) @(posedge clk_core);",
        "// Three reset clocks satisfy the >=1-cycle recovery cover.  Do not\n"
        "        // stop at flag clearing: run the minimum complete legal B4 service.\n"
        "        rst_core = 1;\n"
        "        repeat (0) @(posedge clk_core);"),
        "the second attack must be followed by a real reset before legal recovery")
    add("final_attack_recovery_ledger_neutralized", "tb", one_replace(
        tb,
        "if (stale_attack_count != 1 || retired_identity_replay_count != 1\n"
        "                || replay_accept_count != 0 || reset_recovery_count != 2\n"
        "                || post_reset_legal_service_count != 1)",
        "if (1'b0 && (stale_attack_count != 1 || retired_identity_replay_count != 1\n"
        "                || replay_accept_count != 0 || reset_recovery_count != 2\n"
        "                || post_reset_legal_service_count != 1))"),
        "the final attack/reset/recovery ledger must be live before PASS")

    controls = []
    for name, kind, source in (
            ("bundle_parameter_4_to_8", "rtl", one_replace(
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

    CHECK.verify_sealed_directory(
        HW / "reviews/m1880_m1875_m1874_c2_tsbg_b4_source_author_receipt_r1_20260902")
    CHECK.verify_sealed_directory(CHECK.M1871)
    CHECK.verify_sealed_directory(CHECK.M1875)
    baseline_validation = CHECK.validate_sources()
    identities = {
        "m1880_to_m1874_rtl_namespace_normalization_byte_exact":
            normalize_to_m1874(rtl) == CHECK.M1874_RTL.read_text(encoding="utf-8"),
        "m1880_to_m1874_sva_namespace_normalization_byte_exact":
            normalize_to_m1874(sva) == CHECK.M1874_SVA.read_text(encoding="utf-8"),
        "m1880_to_m1874_tb_namespace_normalization_byte_exact":
            normalize_to_m1874(tb) == CHECK.M1874_TB.read_text(encoding="utf-8"),
        "m1880_to_m1870_rtl_namespace_normalization_byte_exact":
            normalize_to_m1870(rtl) == CHECK.M1870_RTL.read_text(encoding="utf-8"),
        "m1880_to_m1794_b4_lru4_normalization_byte_exact":
            normalize_to_m1794(rtl) == CHECK.M1794_RTL.read_text(encoding="utf-8"),
        "docs359_sha256": sha(CHECK.DOC359),
    }

    all_rejected = all(item["rejected"] for item in
                       exact_m1871 + exact_m1875 + novel + controls)
    all_identity = all(value is True for key, value in identities.items()
                       if key != "docs359_sha256")
    status = "PASS" if (all_rejected and all_identity
                          and identities["docs359_sha256"] ==
                          "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
                          and baseline_validation["status"] ==
                          "PASS_M1880_C2_TSBG_B4_SOURCE_STATIC_NO_EDA") else "FAIL_CLOSED"
    output = {
        "schema": "m1881_m1880_c2_tsbg_b4_independent_probe_r1_v1",
        "status": status,
        "exact_m1871_attacks": {
            "count": len(exact_m1871),
            "rejected": sum(item["rejected"] for item in exact_m1871),
            "escaped": sum(not item["rejected"] for item in exact_m1871),
            "items": exact_m1871,
        },
        "exact_m1875_attacks": {
            "count": len(exact_m1875),
            "rejected": sum(item["rejected"] for item in exact_m1875),
            "escaped": sum(not item["rejected"] for item in exact_m1875),
            "items": exact_m1875,
        },
        "novel_m1881_attacks": {
            "count": len(novel),
            "rejected": sum(item["rejected"] for item in novel),
            "escaped": sum(not item["rejected"] for item in novel),
            "items": novel,
        },
        "controls": {"count": len(controls),
                     "rejected": sum(item["rejected"] for item in controls),
                     "items": controls},
        "source_identity": identities,
        "independent_ledgers": {
            "baseline_lru4": baseline,
            "candidate_lru4": candidate,
            "work_and_typed_signed": arithmetic,
            "resource_source_model": resources,
            "production_acc24_absolute_bound": 48 * 16 * 128,
            "signed_acc24_limit": 1 << 23,
        },
        "source_authority": {
            "official_status": baseline_validation["status"],
            "future_authority": baseline_validation["future_authority"],
            "author_execution": baseline_validation["author_execution"],
        },
        "execution_boundary": {
            "vcs": 0, "simv": 0, "eda": 0, "license_queries": 0,
            "attempts": 0, "results": 0, "campaign_sources": 0,
            "releases": 0,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
