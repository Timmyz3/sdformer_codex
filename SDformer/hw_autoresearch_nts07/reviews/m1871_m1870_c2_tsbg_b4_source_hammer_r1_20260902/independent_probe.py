#!/usr/bin/env python3
"""Independent semantic-mutation probe for the source-only M1870 review.

This script imports the producer checker but never invokes a simulator, an EDA
tool, a license command, or any launch/release path.  Each mutation preserves
the producer's searched vocabulary while weakening one stated TB/SVA
obligation.  A sound semantic checker must reject the mutation itself; an
exact-file SHA mismatch is deliberately not counted as semantic rejection.
"""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CHECKER = HW / "system_simulator/scripts/check_m1870_c2_tsbg_b4_source.py"
SPEC = importlib.util.spec_from_file_location("m1870_checker_for_m1871", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def one_replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("mutation anchor cardinality is not one: " + old[:80])
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
            raise RuntimeError("unknown mutation kind")
    except CHECK.CheckFailure as error:
        return {"rejected": True, "reason": str(error)}
    return {"rejected": False, "reason": "semantic validator accepted weakened source"}


def main():
    tb = CHECK.TB.read_text(encoding="utf-8")
    sva = CHECK.SVA.read_text(encoding="utf-8")
    rtl = CHECK.RTL.read_text(encoding="utf-8")

    mutations = []

    def add(name, kind, source):
        item = {"name": name, "kind": kind}
        item.update(verdict(kind, source))
        mutations.append(item)

    add("replay_slot_not_saved_identity", "tb", one_replace(
        tb, "tsbg.replay_slot[3] = saved_rsp_slot;",
        "tsbg.replay_slot[3] = 3'd0;"))
    add("replay_generation_not_saved_identity", "tb", one_replace(
        tb, "tsbg.replay_generation[3] = saved_rsp_generation;",
        "tsbg.replay_generation[3] = 32'd0;"))
    add("replay_tag_not_saved_identity", "tb", one_replace(
        tb, "tsbg.replay_tag[3] = saved_rsp_tag;",
        "tsbg.replay_tag[3] = 24'd0;"))
    add("replay_payload_not_saved_payload", "tb", one_replace(
        tb, "tsbg.replay_weight[3][lane] = saved_rsp_weight[lane];",
        "tsbg.replay_weight[3][lane] = '0;"))
    add("retired_replay_fault_gate_neutralized", "tb", one_replace(
        tb,
        "if (!tsbg.protocol_error || !tsbg.stale_response_seen\n"
        "                || replay_accept_count != 0)",
        "if ((!tsbg.protocol_error || !tsbg.stale_response_seen\n"
        "                || replay_accept_count != 0) && 1'b0)"))
    add("first_reset_duration_zero", "tb", one_replace(
        tb,
        "rst_core = 1;\n"
        "        repeat (3) @(posedge clk_core);\n"
        "        rst_core = 0;\n"
        "        repeat (2) @(posedge clk_core);\n"
        "        if (tsbg.protocol_error || tsbg.stale_response_seen)",
        "rst_core = 1;\n"
        "        repeat (0) @(posedge clk_core);\n"
        "        rst_core = 0;\n"
        "        repeat (2) @(posedge clk_core);\n"
        "        if (tsbg.protocol_error || tsbg.stale_response_seen)"))
    add("first_reset_clear_gate_neutralized", "tb", one_replace(
        tb,
        "if (tsbg.protocol_error || tsbg.stale_response_seen)\n"
        "            $fatal(1, \"M1870 first reset did not clear replay fault\");",
        "if ((tsbg.protocol_error || tsbg.stale_response_seen) && 1'b0)\n"
        "            $fatal(1, \"M1870 first reset did not clear replay fault\");"))
    add("post_reset_complete_service_ledger_neutralized", "tb", one_replace(
        tb,
        "if (base.row_access_count != 4 || tsbg.row_access_count != 4\n"
        "                || base.issue_count != 48 || tsbg.issue_count != 48\n"
        "                || base.product_count != 768 || tsbg.product_count != 768\n"
        "                || base.commit_count != 24 || tsbg.commit_count != 24\n"
        "                || base.weight_bundle_beat_count != 12\n"
        "                || tsbg.weight_bundle_beat_count != 12\n"
        "                || base.scalar_bank_response_count != 96\n"
        "                || tsbg.scalar_bank_response_count != 96\n"
        "                || terminal_base != 4 || terminal_tsbg != 4)",
        "if (1'b0 && (base.row_access_count != 4 || tsbg.row_access_count != 4\n"
        "                || base.issue_count != 48 || tsbg.issue_count != 48\n"
        "                || base.product_count != 768 || tsbg.product_count != 768\n"
        "                || base.commit_count != 24 || tsbg.commit_count != 24\n"
        "                || base.weight_bundle_beat_count != 12\n"
        "                || tsbg.weight_bundle_beat_count != 12\n"
        "                || base.scalar_bank_response_count != 96\n"
        "                || tsbg.scalar_bank_response_count != 96\n"
        "                || terminal_base != 4 || terminal_tsbg != 4))"))
    add("post_reset_cache_bridge_gate_neutralized", "tb", one_replace(
        tb,
        "if (base.cache_miss_count != 1 || tsbg.cache_miss_count != 1\n"
        "                || base.cache_hit_count != 3 || tsbg.cache_hit_count != 3\n"
        "                || saw_exact_neg128 == 0)",
        "if (1'b0 && (base.cache_miss_count != 1 || tsbg.cache_miss_count != 1\n"
        "                || base.cache_hit_count != 3 || tsbg.cache_hit_count != 3\n"
        "                || saw_exact_neg128 == 0))"))
    add("local_cycle_gate_neutralized", "tb", one_replace(
        tb,
        "if (full_tsbg_done_cycle <= 0 || full_base_done_cycle <= 0\n"
        "                || full_base_done_cycle * 1.0 / full_tsbg_done_cycle < 1.15)",
        "if (1'b0 && (full_tsbg_done_cycle <= 0 || full_base_done_cycle <= 0\n"
        "                || full_base_done_cycle * 1.0 / full_tsbg_done_cycle < 1.15))"))
    add("load_context_range_assertion_tautology", "sva", one_replace(
        sva, "load_accept |-> load_context < TOKEN_CONTEXTS);",
        "load_accept |-> load_context < TOKEN_CONTEXTS || 1'b1);"))
    add("bridge_context_range_assertion_tautology", "sva", one_replace(
        sva, "bridge_valid |-> bridge_context < TOKEN_CONTEXTS);",
        "bridge_valid |-> bridge_context < TOKEN_CONTEXTS || 1'b1);"))
    add("commit_context_range_assertion_tautology", "sva", one_replace(
        sva, "commit_valid |-> commit_context < TOKEN_CONTEXTS);",
        "commit_valid |-> commit_context < TOKEN_CONTEXTS || 1'b1);"))
    add("bank_request_stability_antecedent_disabled", "sva", one_replace(
        sva,
        "mem_req_valid[bank] && !mem_req_ready[bank] |=>",
        "mem_req_valid[bank] && !mem_req_ready[bank] && 1'b0 |=>"))
    add("reset_recovery_cover_made_impossible", "sva", one_replace(
        sva,
        "commit_accept && commit_terminal && !protocol_error));",
        "commit_accept && commit_terminal && !protocol_error && 1'b0));"))

    # Positive controls demonstrate that the imported validator does reject
    # changes it explicitly models.
    add("control_b4_parameter_to_b8", "rtl", one_replace(
        rtl, "parameter int BUNDLE = 4", "parameter int BUNDLE = 8"))
    add("control_candidate_hit_ledger_36_to_35", "tb", one_replace(
        tb, "tsbg.cache_hit_count != 36", "tsbg.cache_hit_count != 35"))

    semantic = [item for item in mutations if not item["name"].startswith("control_")]
    controls = [item for item in mutations if item["name"].startswith("control_")]
    output = {
        "schema": "m1871_m1870_c2_tsbg_b4_independent_semantic_probe_r1_v1",
        "status": "FAIL_CLOSED" if any(not item["rejected"] for item in semantic) else "PASS",
        "semantic_attacks": len(semantic),
        "semantic_rejected": sum(item["rejected"] for item in semantic),
        "semantic_escaped": sum(not item["rejected"] for item in semantic),
        "controls": len(controls),
        "controls_rejected": sum(item["rejected"] for item in controls),
        "mutations": mutations,
        "execution_boundary": {
            "vcs": 0, "simv": 0, "eda": 0, "license_queries": 0,
            "attempts": 0, "results": 0, "releases": 0,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 1 if output["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
