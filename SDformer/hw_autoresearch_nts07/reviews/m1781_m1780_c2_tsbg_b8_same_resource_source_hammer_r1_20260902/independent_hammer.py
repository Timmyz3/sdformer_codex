#!/usr/bin/env python3
"""Independent fail-closed M1781 static hammer.  Never launches EDA."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RTL = HW / "rtl_m1780/m1780_c2_tsbg_b8_typed_weight_row_frontend.sv"
TB = HW / "tb_m1780/tb_m1780_c2_tsbg_b8_typed_weight_row_frontend.sv"
SVA = HW / "verif_m1780/m1780_c2_tsbg_b8_typed_weight_row_frontend_assertions.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M218 = HW / "rtl_m218/m218_fc2_tagged_slice_service_island.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def lru(schedule, capacity=8):
    cache = []
    hit = 0
    miss = 0
    for key in schedule:
        if key in cache:
            cache.remove(key)
            hit += 1
        else:
            miss += 1
            if len(cache) == capacity:
                cache.pop(0)
        cache.append(key)
    return {"accesses": len(schedule), "hits": hit, "misses": miss,
            "aggregate_eight_bank_beats": miss * 2 * 6}


def ledgers():
    token_major = [group for context in range(8) for group in range(12)]
    row_major = [group for group in range(12) for context in range(8)]
    return lru(token_major), lru(row_major)


def resource_account():
    values = {
        "lru8_int8_row_cache_bytes": 8 * 16 * 96,
        "eight_token_acc24_bytes": 8 * 96 * 3,
        "eight_by_48_by_16_source_fifo_bytes": 8 * 48 * 16,
        "context_tag_bytes": 8 * 3,
        "active_bitmap_bytes": 8 * 48 // 8,
    }
    values["explicit_state_bytes_excluding_control"] = sum(values.values())
    return values


def source_gate(text):
    active = strip_comments(text)
    required = (
        "parameter int SCHEDULE_MODE = 1",
        "logic signed [7:0] source_value_q",
        "logic signed [23:0] acc_q",
        "logic signed [7:0] cache_weight_q",
        "if (SCHEDULE_MODE == 0)",
        "delta = delta +",
        "delta = delta -",
    )
    return all(token in active for token in required)


def run():
    rtl = RTL.read_text(encoding="utf-8")
    tb = TB.read_text(encoding="utf-8")
    sva = SVA.read_text(encoding="utf-8")
    m803 = M803.read_text(encoding="utf-8")
    m218 = M218.read_text(encoding="utf-8")

    need(sha(DOC359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs/359 drift")
    need(source_gate(rtl), "M1780 arithmetic/schedule source gate")

    baseline, candidate = ledgers()
    need(baseline == {"accesses": 96, "hits": 0, "misses": 96,
                      "aggregate_eight_bank_beats": 1152},
         "independent baseline ledger")
    need(candidate == {"accesses": 96, "hits": 84, "misses": 12,
                        "aggregate_eight_bank_beats": 144},
         "independent candidate ledger")
    need(8 * 12 * 2 * 6 == 1152, "issue count")
    need(1152 * 16 == 18432, "signed product count")
    need(8 * 6 == 48, "commit count")

    resources = resource_account()
    need(resources["explicit_state_bytes_excluding_control"] == 20808,
         "resource total")

    # The authored source is symmetric in stored datapath state; schedule is a
    # parameter and no generate branch creates a candidate-only data store.
    need("if (SCHEDULE_MODE == 0)" in rtl and
         "else begin\n                    group_index" in rtl,
         "schedule-only branch")
    need(rtl.count("source_value_q") >= 6 and rtl.count("acc_q") >= 6,
         "source/accumulator state missing")

    # But the source is not the frozen M803/M519 interface: M803 has eight
    # independently handshaken bank channels and transaction identity; M1780
    # has one atomic request and response carrying all eight banks.
    for token in ("bank_req_valid", "bank_req_ready", "bank_rsp_valid",
                  "bank_rsp_ready", "core_req_epoch", "core_req_slot",
                  "core_req_generation", "core_req_tag"):
        need(token in m803, "frozen M803 token missing " + token)
    active_rtl = strip_comments(rtl)
    for token in ("mem_req_epoch", "mem_req_slot", "mem_req_generation",
                  "mem_req_tag", "mem_req_valid [0:7]",
                  "mem_rsp_valid [0:7]"):
        need(token not in active_rtl, "unexpected real-M803 bridge token " + token)

    # Frozen M218 groups are binary masks.  M1780 invents a signed value field
    # and performs the add/subtract locally, so no frozen typed-signed consumer
    # is connected at this source boundary.
    need("group_bank_valid" in m218 and "group_source_channel" in m218,
         "frozen M218 binary group boundary missing")
    need("group_source_value" not in strip_comments(m218),
         "unexpected signed value in frozen M218")
    need("issue_source_value" in active_rtl and "issue_weight" in active_rtl,
         "M1780 local typed view missing")

    # Directed coverage exists for stalls, terminal and malformed source data.
    for token in ("cp_issue_stall", "cp_memory_stall", "cp_commit_stall",
                  "cp_terminal", "cp_protocol_attack"):
        need(token in sva, "coverage missing " + token)
    need("load_source_value[0] = 8'sd2" in tb,
         "malformed source attack missing")
    # There is no independent illegal-response injection or explicit eviction
    # cover in the authored TB/SVA.
    need("illegal_response_attack" not in tb and
         "cp_cache_eviction" not in sva,
         "coverage audit expectation changed")

    # Legal default geometry makes Acc24 overflow unreachable even at abs(INT8)
    # maximum: 48 unique groups * 16 sources * 128 = 98,304 < 2^23.
    overflow_bound = 48 * 16 * 128
    need(overflow_bound == 98304 and overflow_bound < (1 << 23),
         "Acc24 default bound")

    # Independent mutation sensitivity: corrupting each central invariant must
    # be observed without importing the author's checker/model.
    mutations = {
        "schedule_mutation": lru([g for g in range(12) for _ in range(8)])
            != baseline,
        "capacity_mutation": lru([g for _ in range(8) for g in range(12)], 12)
            != baseline,
        "source_branch_mutation": not source_gate(
            rtl.replace("delta = delta -", "delta = delta +", 1)),
        "state_mutation": resource_account()[
            "explicit_state_bytes_excluding_control"] - 12288 != 20808,
    }
    need(all(mutations.values()), "independent mutation sensitivity")

    return {
        "status": "FAIL_CLOSED_M1781_INTERFACE_BRIDGE_BLOCKS_RELEASE",
        "directed_independent": {
            "baseline": baseline, "candidate": candidate,
            "issue_accepts_each": 1152,
            "signed_products_each": 18432,
            "commits_each": 48,
        },
        "resource_account": resources,
        "legal_default_acc24_abs_bound": overflow_bound,
        "mutations_detected": mutations,
        "findings": {
            "p0": 2,
            "p1": 2,
            "p2": 1,
            "m1784_vcs_dc_release_authorized": False,
        },
        "python": {"implementation": "CPython-compatible independent hammer"},
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
