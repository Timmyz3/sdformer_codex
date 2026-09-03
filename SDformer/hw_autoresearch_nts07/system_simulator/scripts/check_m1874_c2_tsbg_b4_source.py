#!/usr/bin/env python3
"""Fail-closed, source-only checker for the additive M1874 B4 TSBG island."""
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
RTL = HW / "rtl_m1874/m1874_c2_tsbg_b4_real_channel_signed_frontend.sv"
SVA = HW / "verif_m1874/m1874_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv"
TB = HW / "tb_m1874/tb_m1874_c2_tsbg_b4_real_channel_signed_frontend.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1874_c2_tsbg_b4_real_channel_signed_frontend_directed_vcs.f"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1874_c2_tsbg_b4_source.py"
CONTRACT = HW / "contracts/m1874_m1871_m1870_c2_tsbg_b4_source_contract_r1_20260902.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M1794_RTL = HW / "rtl_m1794/m1794_c2_tsbg_b8_real_channel_signed_frontend.sv"
M1794_CONTRACT = HW / "contracts/m1794_m1788_c2_tsbg_reviewer_repair_source_contract_r1_20260902.json"
M1794_AUTHOR = HW / "reviews/m1794_m1788_c2_tsbg_reviewer_repair_source_author_receipt_r1_20260902"
M1795 = HW / "reviews/m1795_m1794_c2_tsbg_reviewer_repair_source_hammer_r1_20260902"
QUICKKILL = HW / "reviews/tsbg_ep34_same_io_b2_b4_b8_quickkill_self_review_r1_20260902"
M1866 = HW / "reviews/m1866_tsbg_ep34_same_io_b2_b4_b8_quickkill_independent_hammer_r1_20260902"
M1870_RTL = HW / "rtl_m1870/m1870_c2_tsbg_b4_real_channel_signed_frontend.sv"
M1870_CONTRACT = HW / "contracts/m1870_m1795_m1794_c2_tsbg_b4_source_contract_r1_20260902.json"
M1870_AUTHOR = HW / "reviews/m1870_m1866_m1794_c2_tsbg_b4_source_author_receipt_r1_20260902"
M1871 = HW / "reviews/m1871_m1870_c2_tsbg_b4_source_hammer_r1_20260902"

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    M1794_RTL: "283ef29727095255c8502a6f12f66170a41147f16358e09862a46d1d30dc4365",
    M1794_CONTRACT: "263529c4bfbdae896a69320a7ddf306c2f3b1f05739c09ee8b4fea5ca12dad18",
    M1794_AUTHOR / "SHA256SUMS": "6ff306d16c224e82a2cb2dc2a041dd7bf81a27dae14049093a404163bd1c9b01",
    M1794_AUTHOR / "SHA256SUMS.seal.sha256": "e5f2c46e66d47679341e5f10ffe94e4804887bebcf1e7e46ec45831961d37f5a",
    M1795 / "review.json": "4a8ba47c085920e047e0db4ac1a75fefce0eb99f515efe94acda8a2b7f639a0e",
    M1795 / "SHA256SUMS": "50a027e0c6ac0732e305821835de6029835150fd70766ebc963f3973c8902aab",
    M1795 / "SHA256SUMS.seal.sha256": "bed2a3dac746ddb84643c413098e72974e832ce3d9200ee58958a3dfb26b4c53",
    QUICKKILL / "review.json": "b945e85605728bf6fae03833f5b198764dd51f4d72bb5358eaf060e191a1e8b4",
    QUICKKILL / "SHA256SUMS": "af3222ad2df41f995126341c436de80b795281fef0d166cd2b887465161b9697",
    QUICKKILL / "SHA256SUMS.seal.sha256": "44fa62fe2d2b506bec17df1d116562747ed78524f963c109b311d392b9cae390",
    M1866 / "review.json": "6560b3660d247440691d31dea7cccd0ca0294cd203c7f2d957a183116eb81830",
    M1866 / "SHA256SUMS": "12e466e667cf133a4a4953199817180d24054b4aa39ec1ef4a277e602c18b897",
    M1866 / "SHA256SUMS.seal.sha256": "da826a3797d7586508f9f95dfa06430a47a59c9f3e328320453e83777e587fb7",
    M1870_RTL: "4966c59ccd605b59b5d09e82d09b6f23670fcad99a957c679e4a057bd92e3ec9",
    M1870_CONTRACT: "6347ec40f29387a098753e7151ad1f82a6c2159d44c54600d4134beb76396fbf",
    M1870_AUTHOR / "SHA256SUMS": "7f8ec2a6873acd40d153972ef69cdba4841ca9353f04379bba10b5f020cb3034",
    M1870_AUTHOR / "SHA256SUMS.seal.sha256": "ef2c59cc553d9cfbbc34feaa9a958cc423f940c7f09310cbd183eddfa30f38ae",
    M1871 / "review.json": "fb7d0e0d322111bcfaabf74bae0d640c50fe00ea9d7327ae0e3ac883065ad5a8",
    M1871 / "SHA256SUMS": "fbbf43b4614ca9fb90494d9087b13bbf3ca751b34c8c8d6b35c5fd655be4577a",
    M1871 / "SHA256SUMS.seal.sha256": "decd92229b18483577abd867f4ad4028b4d231f7da47642e3e5db3f488e4e8c4",
}

CLAIMS = {
    "source_only": True,
    "cpu_quickkill_context_only": True,
    "same_area": False,
    "rtl_executed": False,
    "vcs": False,
    "dc": False,
    "ptpx": False,
    "energy": False,
    "paper_admitted": False,
    "component_speedup_admitted": False,
    "system_speedup": False,
    "headline": False,
}


class CheckFailure(RuntimeError):
    pass


def need(value, message):
    if not value:
        raise CheckFailure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            CheckFailure("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_sealed_directory(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "sealed directory absent")
    need(outer.read_text(encoding="ascii").split() ==
         [sha(manifest), "SHA256SUMS"], "outer seal drift " + str(root))
    listed = set()
    for row in manifest.read_text(encoding="ascii").splitlines():
        fields = row.split(None, 1)
        need(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
             "manifest syntax")
        name = fields[1].strip().lstrip("*")
        rel = Path(name)
        need(name not in listed and not rel.is_absolute() and ".." not in rel.parts,
             "unsafe/duplicate manifest member")
        need(sha(root / rel) == fields[0], "manifest member drift " + name)
        listed.add(name)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def compact_sv(text):
    """Canonicalize SV structure without trusting comments or whitespace."""
    return re.sub(r"\s+", "", strip_comments(text))


def need_code_once(text, snippet, label):
    count = compact_sv(text).count(compact_sv(snippet))
    need(count == 1, label + " structural cardinality " + str(count))


def one_replace(text, old, new):
    need(text.count(old) == 1, "mutation anchor cardinality " + old[:80])
    return text.replace(old, new, 1)


def normalize_candidate_rtl_to_m1794(text):
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


def normalize_candidate_rtl_to_m1870(text):
    return text.replace("M1874", "M1870").replace("m1874", "m1870")


def validate_rtl_text(text):
    active = strip_comments(text)
    for token in (
            "module m1874_c2_tsbg_b4_real_channel_signed_frontend",
            "parameter int BUNDLE = 4", "parameter int CACHE_ROWS = 4",
            "BUNDLE == 4 && SOURCES_PER_GROUP == 16",
            "OUTPUT_SLICES == 6 && CACHE_ROWS == 4 && LANES == 16",
            "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter",
            "mem_req_valid", "mem_rsp_valid", "mem_req_source_channel [0:7]",
            "active_q [0:BUNDLE-1]", "sign_q [0:BUNDLE-1]",
            "acc_q [0:BUNDLE-1][0:OUTPUT_SLICES-1][0:LANES-1]",
            "cache_weight_q", "[0:CACHE_ROWS-1][0:1][0:OUTPUT_SLICES-1][0:7][0:LANES-1]",
            "if (SCHEDULE_MODE == 0)", "group_index = raw / BUNDLE",
            "context_index = raw % BUNDLE", "bridge_source_value",
            "bridge_effective_weight[bank][lane] = -widened_weight",
            "logic signed [23:0] acc_q", "PRODUCTION_SOURCE_GROUPS = 48",
            "PRODUCTION_ACC24_ABS_BOUND == 98304",
            "ELABORATED_ACC24_ABS_BOUND <= PRODUCTION_ACC24_ABS_BOUND",
            "debug_row_access_count", "debug_cache_hit_count",
            "debug_cache_miss_count", "debug_cache_eviction_count",
            "debug_issue_count", "debug_signed_product_count", "debug_commit_count"):
        need(token in text, "RTL omits " + token)
    need(active.count(
        "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter") == 1,
        "M803 adapter cardinality")
    for token in ("approx", "epsilon", "drop_source", "reuse_product",
                  "reuse_effective_weight", "product_cache"):
        need(token not in active, "lossy/product reuse token " + token)
    need(normalize_candidate_rtl_to_m1794(text) == M1794_RTL.read_text(),
         "compute/commit RTL differs from M1794 beyond B4/LRU4 specialization")
    need(normalize_candidate_rtl_to_m1870(text) == M1870_RTL.read_text(),
         "M1874 RTL differs from immutable M1870 beyond additive namespace")


def validate_tb_text(text):
    required = (
        "localparam int BUNDLE=4, GROUPS=12, SLICES=6, LANES=16",
        "EXPECTED_ROWS=BUNDLE*GROUPS", "EXPECTED_ISSUES=BUNDLE*GROUPS*2*SLICES",
        "EXPECTED_PRODUCTS=EXPECTED_ISSUES*LANES", "EXPECTED_COMMITS=BUNDLE*SLICES",
        "base.cache_miss_count != 48", "tsbg.cache_miss_count != 12",
        "base.cache_hit_count != 0", "tsbg.cache_hit_count != 36",
        "base.cache_eviction_count != 44", "tsbg.cache_eviction_count != 8",
        "terminal_base != 4", "terminal_tsbg != 4",
        "base.row_access_count != 4", "tsbg.row_access_count != 4",
        "base.issue_count != 48", "tsbg.issue_count != 48",
        "base.product_count != 768", "tsbg.product_count != 768",
        "base.commit_count != 24", "tsbg.commit_count != 24",
        "base.cache_hit_count != 3", "tsbg.cache_hit_count != 3",
        "base.bridge_ready = (tb_cycle % 11 != 3)",
        "base.commit_ready = (tb_cycle % 13 != 5)",
        "saved_rsp_epoch <= tsbg.mem_rsp_epoch[3]",
        "tsbg.replay_epoch[3] = saved_rsp_epoch",
        "retired legal identity replay was accepted",
        "replay_accept_count != 0", "repeat (3) @(posedge clk_core)",
        "load_minimal_legal_workload()", "post_reset_legal_service_count",
        "saw_exact_neg128 == 0", "directed local cycle gate below 1.15x",
        "PASS_M1874_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED")
    for token in required:
        need(token in text, "TB omits " + token)
    need(text.count(".SCHEDULE_MODE(mode), .SOURCE_GROUPS(GROUPS)") == 1,
         "DUT macro specialization")
    need(text.count("`CONNECT_M1874(dut_base, base, 0)") == 1 and
         text.count("`CONNECT_M1874(dut_tsbg, tsbg, 1)") == 1,
         "baseline/candidate instantiation")
    need_code_once(text, """
        saved_rsp_epoch <= tsbg.mem_rsp_epoch[3];
        saved_rsp_slot <= tsbg.mem_rsp_slot[3];
        saved_rsp_generation <= tsbg.mem_rsp_generation[3];
        saved_rsp_tag <= tsbg.mem_rsp_tag[3];
        for (int lane = 0; lane < LANES; lane++)
            saved_rsp_weight[lane] <= tsbg.mem_rsp_weight[3][lane];
    """, "accepted-response capture provenance")
    need_code_once(text, """
        tsbg.replay_epoch[3] = saved_rsp_epoch;
        tsbg.replay_slot[3] = saved_rsp_slot;
        tsbg.replay_generation[3] = saved_rsp_generation;
        tsbg.replay_tag[3] = saved_rsp_tag;
        for (int lane = 0; lane < LANES; lane++)
            tsbg.replay_weight[3][lane] = saved_rsp_weight[lane];
    """, "retired replay saved identity and payload provenance")
    need_code_once(text, """
        if (!tsbg.protocol_error || !tsbg.stale_response_seen
                || replay_accept_count != 0)
            $fatal(1, "M1874 retired legal identity replay did not fail closed");
    """, "retired replay zero-accept sticky-fault gate")
    need_code_once(text, """
        rst_core = 1;
        repeat (3) @(posedge clk_core);
        rst_core = 0;
        repeat (2) @(posedge clk_core);
        if (tsbg.protocol_error || tsbg.stale_response_seen)
            $fatal(1, "M1874 first reset did not clear replay fault");
    """, "first reset duration and clear gate")
    need_code_once(text, """
        rst_core = 1;
        repeat (3) @(posedge clk_core);
        rst_core = 0;
        repeat (2) @(posedge clk_core);
        if (base.protocol_error || tsbg.protocol_error
                || base.stale_response_seen || tsbg.stale_response_seen)
            $fatal(1, "M1874 second reset did not clear protocol state");
    """, "second reset duration and clear gate")
    need_code_once(text, """
        if (base.row_access_count != 4 || tsbg.row_access_count != 4
                || base.issue_count != 48 || tsbg.issue_count != 48
                || base.product_count != 768 || tsbg.product_count != 768
                || base.commit_count != 24 || tsbg.commit_count != 24
                || base.weight_bundle_beat_count != 12
                || tsbg.weight_bundle_beat_count != 12
                || base.scalar_bank_response_count != 96
                || tsbg.scalar_bank_response_count != 96
                || terminal_base != 4 || terminal_tsbg != 4)
            $fatal(1, "M1874 post-reset legal-service ledger mismatch");
    """, "complete post-reset request response compute commit terminal ledger")
    need_code_once(text, """
        if (base.cache_miss_count != 1 || tsbg.cache_miss_count != 1
                || base.cache_hit_count != 3 || tsbg.cache_hit_count != 3
                || saw_exact_neg128 == 0)
            $fatal(1, "M1874 post-reset bridge/cache coverage mismatch");
    """, "post-reset cache and signed bridge gate")
    need_code_once(text, """
        if (full_tsbg_done_cycle <= 0 || full_base_done_cycle <= 0
                || full_base_done_cycle * 1.0 / full_tsbg_done_cycle < 1.15)
            $fatal(1, "M1874 directed local cycle gate below 1.15x");
    """, "directed local cycle ratio gate")


def validate_sva_text(text):
    for token in (
            "parameter int TOKEN_CONTEXTS = 4", "ap_load_context_is_b4",
            "load_accept |-> load_context < TOKEN_CONTEXTS",
            "ap_bridge_context_is_b4", "bridge_valid |-> bridge_context < TOKEN_CONTEXTS",
            "ap_commit_context_is_b4", "commit_valid |-> commit_context < TOKEN_CONTEXTS",
            "cp_independent_bank_backpressure", "cp_bank_response_reorder",
            "cp_bridge_positive", "cp_bridge_negative", "cp_bridge_stall",
            "cp_commit_stall", "cp_cache_eviction", "cp_stale_attack",
            "ap_fault_is_sticky", "ap_no_legal_overflow",
            "cp_reset_recovery_minimum_one_cycle", "rst_core[*1:8]",
            "commit_accept && commit_terminal"):
        need(token in text, "SVA omits " + token)
    need_code_once(text, """
        ap_load_context_is_b4: assert property (
            load_accept |-> load_context < TOKEN_CONTEXTS);
    """, "non-tautological load-context assertion")
    need_code_once(text, """
        ap_bridge_context_is_b4: assert property (
            bridge_valid |-> bridge_context < TOKEN_CONTEXTS);
    """, "non-tautological bridge-context assertion")
    need_code_once(text, """
        ap_commit_context_is_b4: assert property (
            commit_valid |-> commit_context < TOKEN_CONTEXTS);
    """, "non-tautological commit-context assertion")
    need_code_once(text, """
        ap_bank_request_stable: assert property (
            mem_req_valid[bank] && !mem_req_ready[bank] |=>
                mem_req_valid[bank]
                && $stable({mem_req_epoch[bank], mem_req_slot[bank],
                    mem_req_generation[bank], mem_req_tag[bank],
                    mem_req_output_block[bank], mem_req_slice[bank],
                    mem_req_source_channel[bank]}));
    """, "effective bank-request stability assertion")
    need_code_once(text, """
        cp_reset_recovery_minimum_one_cycle: cover property (disable iff (1'b0)
            protocol_error ##[1:8] rst_core[*1:8] ##1 !rst_core
            ##[1:300000] (commit_accept && commit_terminal && !protocol_error));
    """, "satisfiable reset recovery cover")


def semantic_mutation_cases(tb_text=None, sva_text=None):
    """Return the exact 15 M1871 attacks without invoking EDA or SHA checks."""
    tb = TB.read_text(encoding="utf-8") if tb_text is None else tb_text
    sva = SVA.read_text(encoding="utf-8") if sva_text is None else sva_text
    return [
        ("replay_slot_not_saved_identity", "tb", one_replace(
            tb, "tsbg.replay_slot[3] = saved_rsp_slot;",
            "tsbg.replay_slot[3] = 3'd0;")),
        ("replay_generation_not_saved_identity", "tb", one_replace(
            tb, "tsbg.replay_generation[3] = saved_rsp_generation;",
            "tsbg.replay_generation[3] = 32'd0;")),
        ("replay_tag_not_saved_identity", "tb", one_replace(
            tb, "tsbg.replay_tag[3] = saved_rsp_tag;",
            "tsbg.replay_tag[3] = 24'd0;")),
        ("replay_payload_not_saved_payload", "tb", one_replace(
            tb, "tsbg.replay_weight[3][lane] = saved_rsp_weight[lane];",
            "tsbg.replay_weight[3][lane] = '0;")),
        ("retired_replay_fault_gate_neutralized", "tb", one_replace(
            tb,
            "if (!tsbg.protocol_error || !tsbg.stale_response_seen\n"
            "                || replay_accept_count != 0)",
            "if ((!tsbg.protocol_error || !tsbg.stale_response_seen\n"
            "                || replay_accept_count != 0) && 1'b0)")),
        ("first_reset_duration_zero", "tb", one_replace(
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
            "        if (tsbg.protocol_error || tsbg.stale_response_seen)")),
        ("first_reset_clear_gate_neutralized", "tb", one_replace(
            tb,
            "if (tsbg.protocol_error || tsbg.stale_response_seen)\n"
            "            $fatal(1, \"M1874 first reset did not clear replay fault\");",
            "if ((tsbg.protocol_error || tsbg.stale_response_seen) && 1'b0)\n"
            "            $fatal(1, \"M1874 first reset did not clear replay fault\");")),
        ("post_reset_complete_service_ledger_neutralized", "tb", one_replace(
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
            "                || terminal_base != 4 || terminal_tsbg != 4))")),
        ("post_reset_cache_bridge_gate_neutralized", "tb", one_replace(
            tb,
            "if (base.cache_miss_count != 1 || tsbg.cache_miss_count != 1\n"
            "                || base.cache_hit_count != 3 || tsbg.cache_hit_count != 3\n"
            "                || saw_exact_neg128 == 0)",
            "if (1'b0 && (base.cache_miss_count != 1 || tsbg.cache_miss_count != 1\n"
            "                || base.cache_hit_count != 3 || tsbg.cache_hit_count != 3\n"
            "                || saw_exact_neg128 == 0))")),
        ("local_cycle_gate_neutralized", "tb", one_replace(
            tb,
            "if (full_tsbg_done_cycle <= 0 || full_base_done_cycle <= 0\n"
            "                || full_base_done_cycle * 1.0 / full_tsbg_done_cycle < 1.15)",
            "if (1'b0 && (full_tsbg_done_cycle <= 0 || full_base_done_cycle <= 0\n"
            "                || full_base_done_cycle * 1.0 / full_tsbg_done_cycle < 1.15))")),
        ("load_context_range_assertion_tautology", "sva", one_replace(
            sva, "load_accept |-> load_context < TOKEN_CONTEXTS);",
            "load_accept |-> load_context < TOKEN_CONTEXTS || 1'b1);")),
        ("bridge_context_range_assertion_tautology", "sva", one_replace(
            sva, "bridge_valid |-> bridge_context < TOKEN_CONTEXTS);",
            "bridge_valid |-> bridge_context < TOKEN_CONTEXTS || 1'b1);")),
        ("commit_context_range_assertion_tautology", "sva", one_replace(
            sva, "commit_valid |-> commit_context < TOKEN_CONTEXTS);",
            "commit_valid |-> commit_context < TOKEN_CONTEXTS || 1'b1);")),
        ("bank_request_stability_antecedent_disabled", "sva", one_replace(
            sva, "mem_req_valid[bank] && !mem_req_ready[bank] |=>",
            "mem_req_valid[bank] && !mem_req_ready[bank] && 1'b0 |=>")),
        ("reset_recovery_cover_made_impossible", "sva", one_replace(
            sva, "commit_accept && commit_terminal && !protocol_error));",
            "commit_accept && commit_terminal && !protocol_error && 1'b0));")),
    ]


def lru_ledger(mode, capacity=4, contexts=4, groups=12):
    if mode == "token":
        schedule = [group for _context in range(contexts) for group in range(groups)]
    elif mode == "group":
        schedule = [group for group in range(groups) for _context in range(contexts)]
    else:
        raise CheckFailure("unknown schedule")
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
    return {
        "rows": len(schedule), "hits": hits, "misses": misses,
        "evictions": evictions,
        "aggregate_eight_bank_bundle_beats": misses * 2 * 6,
        "scalar_bank_beats": misses * 2 * 6 * 8,
    }


def source_code(context, group):
    active = [0] * 16
    sign = [0] * 16
    source0 = (context + group) % 8
    source1 = 8 + ((context * 3 + group) % 8)
    value0 = -1 if (context + group) % 2 == 0 else 1
    active[source0] = active[source1] = 1
    sign[source0] = int(value0 < 0)
    sign[source1] = int((-value0) < 0)
    return active, sign


def directed_weight(group, half, output_slice, bank, lane):
    value = (group * 17 + half * 11 + output_slice * 7
             + bank * 5 + lane * 3) % 255 - 127
    if (group, half, output_slice, bank, lane) == (0, 0, 0, 0, 0):
        return -128
    return value


def arithmetic_ledger(contexts=4, groups=12):
    accumulators = [[([0] * 16) for _ in range(6)] for _ in range(contexts)]
    issues = products = 0
    exact_neg128 = positive = negative = False
    for context in range(contexts):
        for group in range(groups):
            active, sign = source_code(context, group)
            for half in range(2):
                for output_slice in range(6):
                    issues += 1
                    for bank in range(8):
                        source = bank + half * 8
                        if not active[source]:
                            continue
                        products += 16
                        source_value = -1 if sign[source] else 1
                        positive |= source_value == 1
                        negative |= source_value == -1
                        for lane in range(16):
                            weight = directed_weight(group, half, output_slice, bank, lane)
                            effective = source_value * weight
                            exact_neg128 |= weight == -128 and source_value == -1 and effective == 128
                            accumulators[context][output_slice][lane] += effective
    flat = [value for context in accumulators for output in context for value in output]
    return {"issues": issues, "products": products, "commits": contexts * 6,
            "positive": positive, "negative": negative,
            "exact_neg128_to_positive128": exact_neg128,
            "minimum_accumulator": min(flat), "maximum_accumulator": max(flat),
            "accumulators": accumulators}


def resource_account(groups=48):
    values = {
        "shared_lru4_weight_data_bytes": 4 * 2 * 6 * 8 * 16,
        "four_by_96_acc24_context_bytes": 4 * 96 * 3,
        "b4_active_bitmap_bytes": 4 * groups * 16 // 8,
        "b4_sign_bitmap_bytes": 4 * groups * 16 // 8,
        "context_tag_bytes": 4 * 3,
    }
    values["explicit_datapath_state_bytes_excluding_m803_control"] = sum(values.values())
    return values


def validate_contract():
    value = strict_json(CONTRACT)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text(encoding="ascii").split() == [sha(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text(encoding="ascii").split() == [sha(sidecar), sidecar.name],
         "contract outer")
    need(value.get("schema") == "m1874_m1871_m1870_c2_tsbg_b4_source_contract_r1_v1",
         "contract schema")
    need(value.get("status") ==
         "SOURCE_ONLY_M1874_B4_TSBG_SEMANTIC_REPAIR__M1875_REVIEW_M1876_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    expected = dict((str(path.relative_to(ROOT)), sha(path)) for path in
                    (M803, RTL, SVA, TB, FILELIST, CHECKER, TEST))
    need(value.get("source_sha256") == expected, "contract source inventory")
    need(value.get("claim_boundary") == CLAIMS, "contract claim boundary")
    need(value.get("authorization") == {
        "run_vcs": False, "run_simv": False, "run_dc": False,
        "run_ptpx": False, "query_license": False,
        "create_attempt": False, "create_result": False,
        "create_release": False, "automatic_retry": False},
        "contract authorization")
    need(value.get("future_authority") == {
        "different_author_source_review": "M1875",
        "exact_launch_release": "M1876",
        "review_required_before_release": True,
        "postrun_different_author_result_review_required": True,
        "future_campaign_source_requires_separate_milestone": True,
        "m1876_not_direct_execution_authority_without_campaign_source": True},
        "contract future authority")
    return value


def validate_sources():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift " + str(path))
    verify_sealed_directory(M1794_AUTHOR)
    verify_sealed_directory(M1795)
    verify_sealed_directory(QUICKKILL)
    verify_sealed_directory(M1866)
    verify_sealed_directory(M1870_AUTHOR)
    verify_sealed_directory(M1871)
    m1795 = strict_json(M1795 / "review.json")
    need(m1795.get("status") ==
         "FAIL_CLOSED_M1795_M1794_C2_TSBG_SOURCE_HAMMER__P1_2__NO_VCS_NO_EDA",
         "M1795 disposition")
    need(m1795.get("authorization", {}).get("run_vcs") is False,
         "M1794 unexpectedly launchable")
    quickkill = strict_json(QUICKKILL / "review.json")
    need(quickkill.get("status") ==
         "PASS_CPU_PREMODEL_MILESTONE__GO_RTL_GATE_REQUIRED__NO_PAPER_ADMISSION",
         "quickkill context")
    need(quickkill.get("claim_boundary", {}).get("same_area") is False and
         quickkill.get("claim_boundary", {}).get("paper_result") is False,
         "quickkill claim promotion")
    m1866 = strict_json(M1866 / "review.json")
    need(m1866.get("status") ==
         "PASS_INDEPENDENT_REPLAY__B4_SOURCE_ONLY_NEXT__NO_RTL_EXECUTION_OR_PAPER_ADMISSION",
         "M1866 disposition")
    need(m1866.get("evidence_quality") == {
        "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "score_over_100": 99, "status": "PASS"}, "M1866 evidence quality")
    need(m1866.get("rtl_source_ruling", {}).get("single_selected_bundle") == 4
         and m1866.get("authorization", {}).get(
             "b4_new_fail_closed_source_contract_may_be_authored") is True
         and m1866.get("authorization", {}).get("b4_rtl_execution") is False,
         "M1866 B4 source-only authority")
    m1871 = strict_json(M1871 / "review.json")
    need(m1871.get("status") ==
         "FAIL_CLOSED_M1871_M1870_C2_TSBG_B4_SOURCE_HAMMER__P1_1__NO_M1872_NO_VCS_NO_EDA",
         "M1871 disposition")
    need(m1871.get("severity_counts") == {"p0": 0, "p1": 1, "p2": 0}
         and m1871.get("authorization", {}).get("create_m1872_release") is False,
         "M1871 fail-closed source successor requirement")

    rows = [row.split("#", 1)[0].strip() for row in
            FILELIST.read_text(encoding="utf-8").splitlines()
            if row.split("#", 1)[0].strip()]
    need(rows == [str(path.relative_to(ROOT)) for path in (M803, RTL, SVA, TB)],
         "filelist order/set")
    validate_rtl_text(RTL.read_text())
    validate_tb_text(TB.read_text())
    validate_sva_text(SVA.read_text())
    mutations = semantic_mutation_cases()
    need(len(mutations) == 15 and len(set(item[0] for item in mutations)) == 15,
         "M1871 semantic mutation inventory")
    for name, kind, mutated in mutations:
        try:
            if kind == "tb":
                validate_tb_text(mutated)
            elif kind == "sva":
                validate_sva_text(mutated)
            else:
                raise CheckFailure("unknown semantic mutation kind " + kind)
        except CheckFailure:
            continue
        raise CheckFailure("semantic mutation escaped " + name)
    contract = validate_contract()
    baseline = lru_ledger("token")
    candidate = lru_ledger("group")
    need(baseline == {"rows": 48, "hits": 0, "misses": 48, "evictions": 44,
                      "aggregate_eight_bank_bundle_beats": 576,
                      "scalar_bank_beats": 4608}, "baseline LRU4 ledger")
    need(candidate == {"rows": 48, "hits": 36, "misses": 12, "evictions": 8,
                       "aggregate_eight_bank_bundle_beats": 144,
                       "scalar_bank_beats": 1152}, "candidate LRU4 ledger")
    arithmetic = arithmetic_ledger()
    need(arithmetic["issues"] == 576 and arithmetic["products"] == 9216
         and arithmetic["commits"] == 24, "work conservation")
    need(arithmetic["positive"] and arithmetic["negative"]
         and arithmetic["exact_neg128_to_positive128"], "typed corner coverage")
    need(max(abs(arithmetic["minimum_accumulator"]),
             abs(arithmetic["maximum_accumulator"])) <= 98304 < (1 << 23),
         "Acc24 bound")
    return {
        "status": "PASS_M1874_C2_TSBG_B4_SOURCE_STATIC_NO_EDA",
        "baseline_lru4": baseline,
        "candidate_lru4": candidate,
        "work": {key: arithmetic[key] for key in
                 ("issues", "products", "commits", "positive", "negative",
                  "exact_neg128_to_positive128", "minimum_accumulator",
                  "maximum_accumulator")},
        "resources": resource_account(),
        "future_authority": contract["future_authority"],
        "claim_boundary": CLAIMS,
        "author_execution": {"vcs_runs": 0, "simv_runs": 0,
                             "dc_runs": 0, "ptpx_runs": 0,
                             "license_queries": 0, "attempts": 0,
                             "results": 0, "releases": 0},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    need(args.self_check, "M1874 source checker requires --self-check")
    print(json.dumps(validate_sources(), indent=2, sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL_M1874_C2_TSBG_B4_SOURCE_STATIC: " + str(error), file=sys.stderr)
        raise
