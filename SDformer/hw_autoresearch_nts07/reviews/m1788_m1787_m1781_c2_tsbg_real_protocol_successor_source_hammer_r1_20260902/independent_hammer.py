#!/usr/bin/env python3
"""Independent fail-closed M1788 source hammer; never launches EDA."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RTL = HW / "rtl_m1787/m1787_c2_tsbg_b8_real_channel_signed_frontend.sv"
TB = HW / "tb_m1787/tb_m1787_c2_tsbg_b8_real_channel_signed_frontend.sv"
SVA = HW / "verif_m1787/m1787_c2_tsbg_b8_real_channel_signed_frontend_assertions.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1787_c2_tsbg_b8_real_channel_signed_frontend_directed_vcs.f"
CONTRACT = HW / "contracts/m1787_m1781_c2_tsbg_real_protocol_successor_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1787_m1781_c2_tsbg_real_protocol_successor_source_author_receipt_r1_20260902"
M1781 = HW / "reviews/m1781_m1780_c2_tsbg_b8_same_resource_source_hammer_r1_20260902"
M1780 = HW / "rtl_m1780/m1780_c2_tsbg_b8_typed_weight_row_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
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


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_manifest(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text(encoding="ascii").split() ==
         [sha(manifest), "SHA256SUMS"], "outer seal drift: " + str(root))
    members = []
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        need(len(fields) == 2 and len(fields[0]) == 64, "manifest syntax")
        name = fields[1].strip().lstrip("*")
        rel = Path(name)
        need(name not in members and not rel.is_absolute() and
             ".." not in rel.parts, "unsafe manifest member")
        need(sha(root / rel) == fields[0], "manifest member drift: " + name)
        members.append(name)
    return members


def verify_sidecar(path):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    need(sidecar.read_text(encoding="ascii").split() ==
         [sha(path), path.name], "sidecar drift")
    need(outer.read_text(encoding="ascii").split() ==
         [sha(sidecar), sidecar.name], "sidecar outer drift")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def lru(mode, contexts=8, groups=12, capacity=8):
    if mode == "token":
        schedule = [group for _context in range(contexts)
                    for group in range(groups)]
    else:
        schedule = [group for group in range(groups)
                    for _context in range(contexts)]
    cache = []
    hits = 0
    misses = 0
    evictions = 0
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
    return {"row_accesses": len(schedule), "hits": hits,
            "misses": misses, "evictions": evictions,
            "bundle_beats": misses * 2 * 6,
            "scalar_bank_beats": misses * 2 * 6 * 8}


def directed_weight(group, half, output_slice, bank, lane):
    value = (group * 17 + half * 11 + output_slice * 7 + bank * 5 +
             lane * 3) % 255 - 127
    if (group, half, output_slice, bank, lane) == (0, 0, 0, 0, 0):
        return -128
    return value


def arithmetic():
    acc = [[([0] * 16) for _slice in range(6)] for _context in range(8)]
    issues = 0
    products = 0
    exact_corner = False
    for context in range(8):
        for group in range(12):
            source0 = (context + group) % 8
            source1 = 8 + ((context * 3 + group) % 8)
            value0 = -1 if (context + group) % 2 == 0 else 1
            value1 = -value0
            for half in range(2):
                source = source0 if half == 0 else source1
                bank = source % 8
                value = value0 if half == 0 else value1
                for output_slice in range(6):
                    issues += 1
                    products += 16
                    for lane in range(16):
                        weight = directed_weight(group, half, output_slice,
                                                 bank, lane)
                        result = value * weight
                        if value == -1 and weight == -128 and result == 128:
                            exact_corner = True
                        acc[context][output_slice][lane] += result
    flat = [value for context in acc for output_slice in context
            for value in output_slice]
    return {"issues": issues, "signed_products": products,
            "commits": 48, "min_accumulator": min(flat),
            "max_accumulator": max(flat),
            "exact_negative_int8_min": exact_corner}


def probes(rtl, tb, sva, contract):
    active_rtl = strip_comments(rtl)
    claims = contract.get("claim_boundary", {})
    return {
        "frozen_adapter_composed_once": active_rtl.count(
            "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter") == 1,
        "independent_bank_protocol_exposed": all(token in active_rtl for token in (
            "mem_req_valid", "mem_req_ready", "mem_rsp_valid", "mem_rsp_ready",
            "mem_req_epoch [0:7]", "mem_req_slot [0:7]",
            "mem_req_generation [0:7]", "mem_req_tag [0:7]",
            "mem_req_output_block [0:7]", "mem_req_slice [0:7]",
            "mem_req_source_channel [0:7]")),
        "typed_active_sign_bridge": all(token in active_rtl for token in (
            "load_source_active", "load_source_sign", "bridge_source_value",
            "bridge_bank_valid[bank] = active_q")),
        "exact_nine_bit_negation": all(token in active_rtl for token in (
            "logic signed [8:0] widened_weight",
            "bridge_effective_weight[bank][lane] = -widened_weight")),
        "eight_by_96_acc24":
            "acc_q [0:BUNDLE-1][0:OUTPUT_SLICES-1][0:LANES-1]" in active_rtl,
        "schedule_is_parameterized": active_rtl.count("if (SCHEDULE_MODE == 0)") == 1,
        "eviction_is_observable": "cp_cache_eviction" in sva and
            "debug_cache_eviction_count" in active_rtl,
        "terminal_is_checked": "terminal_base != 8 || terminal_tsbg != 8" in tb,
        "static_bound_is_sealed": "STATIC_ACC24_ABS_BOUND = 48 * 16 * 128" in sva and
            "ap_no_legal_overflow" in sva,
        "all_claims_false": len(claims) == 10 and not any(claims.values()),
    }


def mutation_sensitivity(rtl, tb, sva, contract):
    base = probes(rtl, tb, sva, contract)
    need(all(base.values()), "positive source probe failed")
    mutations = [
        ("adapter", rtl.replace(
            "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter",
            "m803_atomic_proxy adapter", 1), tb, sva, contract,
         "frozen_adapter_composed_once"),
        ("bank_protocol", rtl.replace("mem_req_epoch [0:7]",
                                      "mem_req_epoch_atomic", 1), tb, sva,
         contract, "independent_bank_protocol_exposed"),
        ("typed_active", rtl.replace("load_source_active", "load_source_mask"),
         tb, sva, contract, "typed_active_sign_bridge"),
        ("typed_sign", rtl.replace("load_source_sign", "load_source_pol"),
         tb, sva, contract, "typed_active_sign_bridge"),
        ("nine_bit", rtl.replace("logic signed [8:0] widened_weight",
                                 "logic signed [7:0] widened_weight", 1), tb,
         sva, contract, "exact_nine_bit_negation"),
        ("acc_context", rtl.replace("acc_q [0:BUNDLE-1]",
                                    "acc_q [0:0]", 1), tb, sva, contract,
         "eight_by_96_acc24"),
        ("schedule", rtl.replace("if (SCHEDULE_MODE == 0)", "if (1'b0)", 1),
         tb, sva, contract, "schedule_is_parameterized"),
        ("eviction", rtl, tb, sva.replace("cp_cache_eviction",
                                          "cp_cache_removed", 1), contract,
         "eviction_is_observable"),
        ("terminal", rtl, tb.replace(
            "terminal_base != 8 || terminal_tsbg != 8",
            "terminal_base != 7 || terminal_tsbg != 7", 1), sva, contract,
         "terminal_is_checked"),
        ("claim", rtl, tb, sva, dict(contract,
            claim_boundary=dict(contract["claim_boundary"], vcs=True)),
         "all_claims_false"),
    ]
    detected = {}
    for name, mutant_rtl, mutant_tb, mutant_sva, mutant_contract, key in mutations:
        detected[name] = not probes(mutant_rtl, mutant_tb, mutant_sva,
                                    mutant_contract)[key]
    need(all(detected.values()), "mutation escaped")
    return detected


def run():
    need(sha(DOC359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs/359 drift")
    need(sha(M1780) ==
         "63599d57323fafce8003947df68fc890c2877e52dc8ee0e0806106440787f04c",
         "M1780 drift")
    need(sha(M803) ==
         "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
         "M803 drift")
    verify_manifest(AUTHOR)
    verify_manifest(M1781)
    verify_sidecar(CONTRACT)

    rtl = RTL.read_text(encoding="utf-8")
    tb = TB.read_text(encoding="utf-8")
    sva = SVA.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    need(all(probes(rtl, tb, sva, contract).values()), "positive probes")

    filelist = [line.split("#", 1)[0].strip() for line in
                FILELIST.read_text(encoding="utf-8").splitlines()
                if line.split("#", 1)[0].strip()]
    need(filelist == [
        "hw_autoresearch_nts07/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "hw_autoresearch_nts07/rtl_m1787/m1787_c2_tsbg_b8_real_channel_signed_frontend.sv",
        "hw_autoresearch_nts07/verif_m1787/m1787_c2_tsbg_b8_real_channel_signed_frontend_assertions.sv",
        "hw_autoresearch_nts07/tb_m1787/tb_m1787_c2_tsbg_b8_real_channel_signed_frontend.sv"],
         "filelist drift")

    baseline = lru("token")
    candidate = lru("row")
    need(baseline == {"row_accesses": 96, "hits": 0, "misses": 96,
                      "evictions": 88, "bundle_beats": 1152,
                      "scalar_bank_beats": 9216}, "baseline ledger")
    need(candidate == {"row_accesses": 96, "hits": 84, "misses": 12,
                        "evictions": 4, "bundle_beats": 144,
                        "scalar_bank_beats": 1152}, "candidate ledger")
    numbers = arithmetic()
    need(numbers["issues"] == 1152 and numbers["signed_products"] == 18432
         and numbers["commits"] == 48 and numbers["exact_negative_int8_min"],
         "arithmetic ledger")
    need(-98304 <= numbers["min_accumulator"] and
         numbers["max_accumulator"] <= 98304, "accumulator bound")

    # Launch-blocking contradiction: reduced GROUPS=12 is passed into each DUT,
    # while PARAMETERS_LEGAL requires SOURCE_GROUPS*16*128 == 98,304, hence 48.
    tb_groups = int(re.search(r"localparam int BUNDLE=8, GROUPS=(\d+)", tb).group(1))
    need(".SOURCE_GROUPS(GROUPS)" in tb, "TB group override missing")
    need("STATIC_ACC24_ABS_BOUND == 98304" in rtl,
         "RTL legal-point equality changed")
    directed_static_bound = tb_groups * 16 * 128
    time_zero_parameter_fatal = directed_static_bound != 98304
    need(tb_groups == 12 and directed_static_bound == 24576 and
         time_zero_parameter_fatal, "expected P0 contradiction absent")

    # Both response attacks drive the same fabricated dead identity.  No legal
    # retired response identity is saved/replayed, so duplicate coverage is
    # nominal rather than an executable duplicate of an accepted transaction.
    need(tb.count("inject_stale[3] = 1") == 2, "attack count drift")
    duplicate_replays_retired_legal_identity = any(token in tb for token in (
        "inject_duplicate", "replay_last_response", "last_rsp_epoch",
        "duplicate_rsp_epoch"))
    need(not duplicate_replays_retired_legal_identity,
         "duplicate coverage expectation changed")

    # Reset clears sticky status, but the TB does not run a legal transaction
    # after reset.  The SVA recovery cover expects one reset cycle while the TB
    # deliberately holds reset for three, so that cover cannot be the receipt.
    post_reset = tb.rsplit("rst_core = 1;", 1)[1]
    post_reset_restarts_work = "load_workload" in post_reset
    reset_cover_matches_tb = "rst_core[*3]" in sva
    need(not post_reset_restarts_work and not reset_cover_matches_tb,
         "reset coverage expectation changed")

    mutations = mutation_sensitivity(rtl, tb, sva, contract)
    return {
        "status": "FAIL_CLOSED_M1788_TIME_ZERO_PARAMETER_FATAL_BLOCKS_VCS_RELEASE",
        "score": 82,
        "severity": {"p0": 1, "p1": 2, "p2": 1},
        "m1781_p0_architecture_repairs": {
            "real_m803_independent_bank_protocol_present": True,
            "typed_signed_bridge_and_eight_token_acc24_present": True},
        "independent_ledgers": {"baseline": baseline,
                                "candidate": candidate,
                                "arithmetic": numbers},
        "launch_blocker": {
            "tb_source_groups": tb_groups,
            "tb_static_bound": directed_static_bound,
            "rtl_required_static_bound": 98304,
            "time_zero_parameter_fatal": time_zero_parameter_fatal},
        "coverage_gaps": {
            "duplicate_replays_retired_legal_identity":
                duplicate_replays_retired_legal_identity,
            "post_reset_legal_workload": post_reset_restarts_work,
            "reset_sva_cover_matches_three_cycle_tb_reset":
                reset_cover_matches_tb},
        "mutations_detected": mutations,
        "authorization": {"vcs": False, "simv": False, "dc": False,
                          "ptpx": False, "license_query": False,
                          "attempt": False, "result": False,
                          "release": False},
        "claim_boundary": {
            "same_resource_result": False, "component_speedup": False,
            "system_speedup": False, "paper_admitted": False,
            "five_point_one_two_x_hardware": False,
            "eight_x_hardware": False},
        "eda_or_license_actions": 0,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
