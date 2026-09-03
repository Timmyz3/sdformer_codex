#!/usr/bin/env python3
"""Static, source-only checks for M2018; this file never launches EDA."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
RTL = ROOT / "hw_autoresearch_nts07/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
M1995 = ROOT / "hw_autoresearch_nts07/rtl_m1995/m1995_m1880_c2_tsbg_b4_dc_keyword_legal_frontend.sv"
M1880 = ROOT / "hw_autoresearch_nts07/rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m2018_c2_tsbg_b4_divfree_fair_scheduler_source_contract_r1_20260902.json"
CONTRACT_MANIFEST = Path(str(CONTRACT) + ".sha256")
CONTRACT_SEAL = Path(str(CONTRACT_MANIFEST) + ".seal.sha256")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def module_header(text):
    begin = text.index("#(", text.index("module "))
    end = text.index(");", begin) + 2
    return text[begin:end]


def synthesizable_without_comments_or_strings(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = "\n".join(row for row in text.splitlines()
                     if not row.lstrip().startswith("`timescale"))
    return text


def mapped_order(mode, source_groups):
    rows = [None] * 192
    for context in range(4):
        for group in range(48):
            if mode == 0:
                index = context * 48 + group
            else:
                index = group * 4 + context
            if group < source_groups:
                rows[index] = (context, group)
    return rows


def hierarchical_select(live):
    lane_pick = [None] * 12
    block_live = [False] * 12
    for block in range(12):
        for lane in range(16):
            if live[block * 16 + lane] and lane_pick[block] is None:
                lane_pick[block] = lane
                block_live[block] = True
    for block in range(12):
        if block_live[block]:
            return block * 16 + lane_pick[block]
    return None


class M2018DivfreeFairSchedulerSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rtl = RTL.read_text()
        cls.old = M1995.read_text()
        cls.contract = json.loads(CONTRACT.read_text())

    def test_01_additive_identity_and_exact_public_header(self):
        self.assertIn(
            "module m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #(",
            self.rtl)
        self.assertEqual(module_header(self.rtl), module_header(self.old))
        self.assertEqual(sha(M1995),
                         "2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd")
        self.assertEqual(sha(M1880),
                         "8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05")
        self.assertEqual(sha(DOCS359),
                         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")

    def test_02_state_and_external_transaction_semantics_retained(self):
        for token in (
                "ST_LOAD, ST_FIND, ST_FETCH_REQ, ST_FETCH_RSP,",
                "ST_BRIDGE, ST_COMMIT, ST_DONE, ST_FAULT",
                "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter",
                "assign core_req_valid = state_q == ST_FETCH_REQ && !fault_q;",
                "assign core_rsp_ready = state_q == ST_FETCH_RSP && !fault_q;",
                "bridge_accept = bridge_valid && bridge_ready;",
                "commit_accept = commit_valid && commit_ready;",
                "bundle_done_valid = state_q == ST_DONE && !fault_q;",
                "debug_scalar_bank_request_count = adapter_bank_req_count;",
                "debug_scalar_bank_response_count = adapter_bank_rsp_count;",
                "debug_issue_count = issue_count_q;",
                "debug_commit_count = commit_count_q;"):
            self.assertIn(token, self.rtl)

    def test_03_no_runtime_division_remainder_or_old_active_cube(self):
        active = synthesizable_without_comments_or_strings(self.rtl)
        self.assertNotRegex(active, r"(?<!/)/(?!/)")
        self.assertNotIn("%", active)
        self.assertNotIn("active_q", active)
        scheduler = active.split(
            "always_comb begin : hierarchical_row_priority", 1)[1].split(
                "always_comb begin : cache_lookup", 1)[0]
        self.assertNotIn("active_row_q", scheduler)
        self.assertNotIn("sign_row_q", scheduler)
        self.assertNotIn("current_context_q][current_group_q", active)

    def test_04_common_192_bit_12_by_16_priority_structure(self):
        for token in (
                "localparam int ORDERED_ROWS = 192;",
                "localparam int PRIORITY_BLOCKS = 12;",
                "localparam int PRIORITY_LANES = 16;",
                "logic [ORDERED_ROWS-1:0] ordered_row_live;",
                "logic [PRIORITY_BLOCKS-1:0] priority_block_live;",
                "logic [PRIORITY_LANES-1:0] priority_lane_onehot",
                "logic [PRIORITY_BLOCKS-1:0] priority_block_onehot;",
                "always_comb begin : hierarchical_row_priority",
                "priority_lane_onehot[block][lane] =",
                "priority_block_onehot[block] =",
                "find_onehot[block * PRIORITY_LANES + lane] ="):
            self.assertIn(token, self.rtl)

    def test_05_two_modes_are_generate_time_static_mappings(self):
        for token in (
                "if (SCHEDULE_MODE == 0) begin : g_token_major_order",
                "map_ctx * PRODUCTION_SOURCE_GROUPS + map_group",
                "end else begin : g_group_major_order",
                "map_group * BUNDLE + map_ctx",
                "assign ordered_row_live[ORDER_INDEX] =",
                "row_live_q[map_ctx][map_group];"):
            self.assertIn(token, self.rtl)

        for groups in (1, 12, 48):
            for mode in (0, 1):
                order = mapped_order(mode, groups)
                live = [row is not None for row in order]
                observed = []
                while True:
                    selected = hierarchical_select(live)
                    if selected is None:
                        break
                    observed.append(order[selected])
                    live[selected] = False
                if mode == 0:
                    expected = [(ctx, group) for ctx in range(4)
                                for group in range(groups)]
                else:
                    expected = [(ctx, group) for group in range(groups)
                                for ctx in range(4)]
                self.assertEqual(observed, expected)

    def test_06_row_live_reset_load_select_and_done_lifecycle(self):
        for token in (
                "logic row_live_q [0:BUNDLE-1][0:PRODUCTION_SOURCE_GROUPS-1];",
                "row_live_q[load_context][load_group] <= |load_source_active;",
                "row_live_q[ctx][group] <= 1'b0;",
                "current_active_row_q <= find_active_row;",
                "current_sign_row_q <= find_sign_row;",
                "if (!find_valid) begin",
                "state_q <= ST_COMMIT;"):
            self.assertIn(token, self.rtl)
        self.assertGreaterEqual(
            self.rtl.count("row_live_q[ctx][group] <= 0;"), 2)

    def test_07_selected_payload_only_reaches_existing_bridge(self):
        for token in (
                "logic [SOURCES_PER_GROUP-1:0] current_active_row_q;",
                "logic [SOURCES_PER_GROUP-1:0] current_sign_row_q;",
                "bridge_bank_valid[bank] = current_active_row_q[bank + 8];",
                "bridge_bank_valid[bank] = current_active_row_q[bank];",
                "bridge_source_value[bank] = current_sign_row_q[bank + 8]",
                "bridge_source_value[bank] = current_sign_row_q[bank]",
                "bridge_effective_weight[bank][lane] = -widened_weight",
                "acc_q[current_context_q][current_slice_q][lane]",
                "signed_product_count_q <= signed_product_count_q"):
            self.assertIn(token, self.rtl)

    def test_08_double_sealed_source_contract_and_exact_hashes(self):
        manifest_row = CONTRACT_MANIFEST.read_text().strip().split()
        self.assertEqual(manifest_row,
                         [sha(CONTRACT), CONTRACT.name])
        seal_row = CONTRACT_SEAL.read_text().strip().split()
        self.assertEqual(seal_row,
                         [sha(CONTRACT_MANIFEST), CONTRACT_MANIFEST.name])
        sources = self.contract["source_sha256"]
        self.assertEqual(sources[str(RTL.relative_to(ROOT))], sha(RTL))
        self.assertEqual(sources[str(Path(__file__).resolve().relative_to(ROOT))],
                         sha(Path(__file__).resolve()))

    def test_09_source_only_claim_and_review_before_vcs(self):
        self.assertEqual(self.contract["status"],
                         "SOURCE_ONLY_M2018_DIVFREE_FAIR_SCHEDULER__NO_EDA")
        self.assertEqual(self.contract["author_execution"], {
            "vcs_runs": 0, "simv_runs": 0, "dc_runs": 0,
            "pt_runs": 0, "fm_runs": 0, "gpu_runs": 0,
            "license_queries": 0, "attempts": 0, "results": 0,
            "releases": 0})
        authority = self.contract["future_authority"]
        self.assertEqual(authority["different_author_source_review"], "M2019")
        self.assertTrue(authority["m2019_pass_required_before_vcs_source"])
        self.assertTrue(authority["source_review_does_not_authorize_execution"])
        for key in ("vcs", "dc", "pt", "fm", "gpu", "paper_admitted",
                    "component_speedup_admitted", "system_speedup",
                    "headline"):
            self.assertFalse(self.contract["claim_boundary"][key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
