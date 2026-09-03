#!/usr/bin/env python3
"""Independent source-only M2019 hammer; never invokes EDA or a license query."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RTL = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
TEST = HW / "system_simulator/tests/test_m2018_c2_tsbg_b4_divfree_fair_scheduler_source.py"
CONTRACT = HW / "contracts/m2018_c2_tsbg_b4_divfree_fair_scheduler_source_contract_r1_20260902.json"
M1995 = HW / "rtl_m1995/m1995_m1880_c2_tsbg_b4_dc_keyword_legal_frontend.sv"
M1880 = HW / "rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv"
ADAPTER = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M1999_REVIEW = HW / "reviews/m1999_m1998_m1995_c2_tsbg_keyword_legal_vcs_result_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RTL: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    TEST: "c42eb57245a50d8b5f688bd8db1a18fb826ad13bc6b218b3e37631a1bbf1d26c",
    CONTRACT: "714e9ed902409d855e1c70b01d2bd93a99bfda54d5b805747dd399a298daacb1",
    M1995: "2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd",
    M1880: "8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05",
    ADAPTER: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_seal(directory):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip(" *")
        target = directory / rel
        assert target.is_file() and not target.is_symlink()
        assert sha(target) == digest
    digest, rel = outer.read_text().split(maxsplit=1)
    assert rel.strip().lstrip("*") == "SHA256SUMS"
    assert sha(manifest) == digest


def verify_file_seal(path):
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(manifest) + ".seal.sha256")
    digest, name = manifest.read_text().split(maxsplit=1)
    assert name.strip().lstrip("*") == path.name
    assert sha(path) == digest
    seal_digest, seal_name = outer.read_text().split(maxsplit=1)
    assert seal_name.strip().lstrip("*") == manifest.name
    assert sha(manifest) == seal_digest


def module_header(text):
    begin = text.index("#(", text.index("module "))
    end = text.index(");", begin) + 2
    return text[begin:end]


def strip_comments_and_strings(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("`timescale"))


def ordered_index(mode, context, group):
    if mode == 0:
        return context * 48 + group
    return group * 4 + context


def old_schedule(mode, groups, live_rows):
    if mode == 0:
        return [(ctx, group) for ctx in range(4) for group in range(groups)
                if (ctx, group) in live_rows]
    return [(ctx, group) for group in range(groups) for ctx in range(4)
            if (ctx, group) in live_rows]


def priority_pick(bits):
    lane_onehot = [[False] * 16 for _ in range(12)]
    block_live = [False] * 12
    for block in range(12):
        seen_lane = False
        for lane in range(16):
            value = bool(bits[block * 16 + lane])
            block_live[block] = block_live[block] or value
            lane_onehot[block][lane] = value and not seen_lane
            seen_lane = seen_lane or value
    block_onehot = [False] * 12
    seen_block = False
    for block in range(12):
        block_onehot[block] = block_live[block] and not seen_block
        seen_block = seen_block or block_live[block]
    onehot = [False] * 192
    for block in range(12):
        for lane in range(16):
            index = block * 16 + lane
            onehot[index] = block_onehot[block] and lane_onehot[block][lane]
    assert sum(onehot) <= 1
    return onehot.index(True) if any(onehot) else None


def new_clear_schedule(mode, groups, live_rows):
    reverse = {}
    bits = [False] * 192
    for context in range(4):
        for group in range(groups):
            index = ordered_index(mode, context, group)
            assert 0 <= index < 192
            assert index not in reverse
            reverse[index] = (context, group)
            bits[index] = (context, group) in live_rows
    observed = []
    while True:
        selected = priority_pick(bits)
        if selected is None:
            break
        assert selected in reverse
        observed.append(reverse[selected])
        bits[selected] = False
    return observed


def model_check():
    rng = random.Random(0x2018)
    trials = 0
    for groups in (1, 12, 48):
        all_rows = {(ctx, group) for ctx in range(4)
                    for group in range(groups)}
        patterns = [set(), all_rows]
        for _ in range(128):
            patterns.append({row for row in all_rows if rng.randrange(4) == 0})
        for mode in (0, 1):
            for live_rows in patterns:
                expected = old_schedule(mode, groups, live_rows)
                observed = new_clear_schedule(mode, groups, live_rows)
                assert observed == expected
                assert len(observed) == len(set(observed))
                trials += 1

            # Padding positions must never become live for reduced geometries.
            mapped = {ordered_index(mode, ctx, group)
                      for ctx in range(4) for group in range(groups)}
            assert len(mapped) == 4 * groups
            assert all(0 <= item < 192 for item in mapped)
            if groups == 12 and mode == 0:
                assert ordered_index(mode, 1, 0) == 48
                assert all(index not in mapped for index in range(12, 48))
    return trials


def source_check():
    for path, digest in EXPECTED.items():
        assert path.is_file() and not path.is_symlink(), path
        assert sha(path) == digest, path
    verify_file_seal(CONTRACT)
    verify_seal(M1999_REVIEW)

    rtl = RTL.read_text()
    old = M1995.read_text()
    contract = json.loads(CONTRACT.read_text())
    assert module_header(rtl) == module_header(old)
    assert contract["source_sha256"][str(RTL.relative_to(ROOT))] == sha(RTL)
    assert contract["source_sha256"][str(TEST.relative_to(ROOT))] == sha(TEST)
    assert contract["status"] == "SOURCE_ONLY_M2018_DIVFREE_FAIR_SCHEDULER__NO_EDA"
    assert all(value == 0 for value in contract["author_execution"].values())
    assert all(value is False for value in contract["authorization"].values())

    active = strip_comments_and_strings(rtl)
    assert not re.search(r"(?<!/)/(?![/*])", active)
    assert "%" not in active
    assert "active_q" not in active and "sign_q" not in active
    assert "scan_linear_q" not in active and "find_linear" not in active
    assert "[current_context_q][current_group_q]" not in active

    # The row-live store has one procedural owner.  Its five assignment sites
    # are reset, load, the two elaboration-constant clear mappings, and done.
    ff_begin = active.index("always_ff @(posedge clk_core)")
    assert "row_live_q" not in active[:ff_begin].replace(
        "logic row_live_q [0:BUNDLE-1][0:PRODUCTION_SOURCE_GROUPS-1];", "") \
        or "assign ordered_row_live" in active[:ff_begin]
    row_live_writes = re.findall(r"row_live_q\s*\[[^\]]+\]\s*\[[^\]]+\]\s*<=", active)
    assert len(row_live_writes) == 5
    assert active.count("row_live_q[load_context][load_group] <= |load_source_active;") == 1
    assert active.count("row_live_q[ctx][group] <= 1'b0;") == 2
    assert active.count("row_live_q[ctx][group] <= 0;") == 2

    scheduler = active.split(
        "always_comb begin : hierarchical_row_priority", 1)[1].split(
            "always_comb begin : cache_lookup", 1)[0]
    assert "priority_lane_onehot[block][lane] =" in scheduler
    assert "priority_block_onehot[block] =" in scheduler
    assert "find_onehot[block * PRIORITY_LANES + lane] =" in scheduler
    assert "ordered_active_row[block * PRIORITY_LANES + lane]" in scheduler
    assert "ordered_sign_row[block * PRIORITY_LANES + lane]" in scheduler

    # Sized casts are IEEE SystemVerilog sized-value casts.  Their operands are
    # generate constants bounded by the exact destination widths.
    assert rtl.count("3'(map_ctx)") == 4
    assert rtl.count("6'(map_group)") == 4
    assert "map_ctx < BUNDLE" in rtl and "map_group < PRODUCTION_SOURCE_GROUPS" in rtl

    required_semantics = (
        "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter",
        "assign core_req_valid = state_q == ST_FETCH_REQ && !fault_q;",
        "assign core_rsp_ready = state_q == ST_FETCH_RSP && !fault_q;",
        "bridge_accept = bridge_valid && bridge_ready;",
        "commit_accept = commit_valid && commit_ready;",
        "bundle_done_valid = state_q == ST_DONE && !fault_q;",
        "bridge_effective_weight[bank][lane] = -widened_weight;",
        "bridge_next_acc[lane][24]",
        "!= bridge_next_acc[lane][23]",
        "debug_scalar_bank_request_count = adapter_bank_req_count;",
        "debug_scalar_bank_response_count = adapter_bank_rsp_count;",
        "debug_signed_product_count = signed_product_count_q;",
    )
    for token in required_semantics:
        assert token in rtl, token

    # Selected row is snapshotted on the same edge that consumes row_live.
    find_branch = active.split("if (state_q == ST_FIND) begin", 1)[1].split(
        "if (state_q == ST_FETCH_REQ", 1)[0]
    assert find_branch.count("current_active_row_q <= find_active_row;") == 1
    assert find_branch.count("current_sign_row_q <= find_sign_row;") == 1
    assert "row_live_q[ctx][group] <= 1'b0;" in find_branch
    bridge = active.split("always_comb begin : signed_bridge", 1)[1].split(
        "always_comb begin : commit_view", 1)[0]
    assert "current_active_row_q" in bridge and "current_sign_row_q" in bridge
    assert "row_live_q" not in bridge


def main():
    source_check()
    trials = model_check()
    print("PASS_M2019_INDEPENDENT_SOURCE_HAMMER trials={} modes=2 groups=3 p0=0 p1=0 p2=0".format(trials))


if __name__ == "__main__":
    main()
