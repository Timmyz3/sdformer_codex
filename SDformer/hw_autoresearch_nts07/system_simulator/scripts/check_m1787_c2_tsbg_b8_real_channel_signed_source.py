#!/usr/bin/env python3
"""Fail-closed source/reference checker for M1787; never launches EDA."""
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
RTL = HW / "rtl_m1787/m1787_c2_tsbg_b8_real_channel_signed_frontend.sv"
SVA = HW / "verif_m1787/m1787_c2_tsbg_b8_real_channel_signed_frontend_assertions.sv"
TB = HW / "tb_m1787/tb_m1787_c2_tsbg_b8_real_channel_signed_frontend.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1787_c2_tsbg_b8_real_channel_signed_frontend_directed_vcs.f"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1787_c2_tsbg_b8_real_channel_signed_source.py"
CONTRACT = HW / "contracts/m1787_m1781_c2_tsbg_real_protocol_successor_source_contract_r1_20260902.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1780_RTL = HW / "rtl_m1780/m1780_c2_tsbg_b8_typed_weight_row_frontend.sv"
M1780_CONTRACT = HW / "contracts/m1780_m1763_c2_tsbg_b8_same_resource_source_contract_r1_20260902.json"
M1781 = HW / "reviews/m1781_m1780_c2_tsbg_b8_same_resource_source_hammer_r1_20260902"
M803_ADAPTER = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M803_K8 = HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv"

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1780_RTL: "63599d57323fafce8003947df68fc890c2877e52dc8ee0e0806106440787f04c",
    M1780_CONTRACT: "7749c59a619cfcdbf7a08ca7731b0ddffaf0f7e26e4f77da1ab4536c261b49d1",
    M1781 / "review.json": "8c950423d63c3b6c7fb944ba73c1c89bdbdfd6cee1afac9ac7df6a3859d61808",
    M1781 / "SHA256SUMS": "8775d4786b41422d815c91dad98fa3526f7ee4241b31eead38414967ba2b38e3",
    M1781 / "SHA256SUMS.seal.sha256": "bf225224e9fba7cb18be25c857d531850f7178cb7d72381a6bbf5cca36010dac",
    M803_ADAPTER: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    M803_K8: "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
}

CLAIMS = dict((key, False) for key in (
    "vcs", "dc", "ptpx", "area", "energy", "same_resource_result",
    "paper_admitted", "component_speedup", "system_speedup", "headline"))


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
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root must be object")
    return value


def verify_sealed_directory(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text(encoding="ascii").split() ==
         [sha(manifest), "SHA256SUMS"], "outer seal drift")
    members = []
    for row in manifest.read_text(encoding="ascii").splitlines():
        fields = row.split(None, 1)
        need(len(fields) == 2 and len(fields[0]) == 64, "manifest syntax")
        name = fields[1].strip().lstrip("*")
        rel = Path(name)
        need(name not in members and not rel.is_absolute()
             and ".." not in rel.parts, "unsafe manifest member")
        need(sha(root / rel) == fields[0], "manifest member drift " + name)
        members.append(name)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def lru_ledger(mode, capacity=8, contexts=8, groups=12):
    if mode == "token":
        schedule = [group for _context in range(contexts)
                    for group in range(groups)]
    elif mode == "row":
        schedule = [group for group in range(groups)
                    for _context in range(contexts)]
    else:
        raise RuntimeError("unknown schedule")
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
    return {
        "row_accesses": len(schedule),
        "hits": hits,
        "misses": misses,
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
    value1 = -value0
    active[source0] = active[source1] = 1
    sign[source0] = int(value0 < 0)
    sign[source1] = int(value1 < 0)
    return active, sign


def directed_weight(group, half, output_slice, bank, lane):
    value = (group * 17 + half * 11 + output_slice * 7
             + bank * 5 + lane * 3) % 255 - 127
    if (group, half, output_slice, bank, lane) == (0, 0, 0, 0, 0):
        return -128
    return value


def typed_effective(active, sign, weight):
    if not active:
        return 0
    return -weight if sign else weight


def arithmetic_ledger(groups=12):
    accumulators = [[([0] * 16) for _ in range(6)] for _ in range(8)]
    issues = 0
    products = 0
    exact_neg128 = False
    for context in range(8):
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
                        for lane in range(16):
                            weight = directed_weight(group, half,
                                output_slice, bank, lane)
                            effective = typed_effective(
                                active[source], sign[source], weight)
                            if weight == -128 and sign[source] and effective == 128:
                                exact_neg128 = True
                            accumulators[context][output_slice][lane] += effective
    flattened = [value for context in accumulators
                 for output_slice in context for value in output_slice]
    return {
        "issues": issues,
        "signed_products": products,
        "commits": 8 * 6,
        "exact_neg128_to_positive128": exact_neg128,
        "min_accumulator": min(flattened),
        "max_accumulator": max(flattened),
        "accumulators": accumulators,
    }


def resource_account(groups=48):
    result = {
        "shared_lru8_weight_data_bytes": 8 * 16 * 96,
        "eight_by_96_acc24_context_bytes": 8 * 96 * 3,
        "b8_active_bitmap_bytes": 8 * groups * 16 // 8,
        "b8_sign_bitmap_bytes": 8 * groups * 16 // 8,
        "context_tag_bytes": 8 * 3,
    }
    result["explicit_datapath_state_bytes_excluding_m803_control"] = sum(
        result.values())
    # Active+sign replaces M1780's signed INT8 FIFO.  Both modes use this same
    # representation.  Adapter/control bits remain for physical DC pricing.
    return result


def protocol_model():
    live = {bank: (0x1787, 1, 1, 0x870001) for bank in range(8)}
    arrival = [7, 5, 3, 1, 6, 4, 2, 0]
    accepted = []
    for bank in arrival:
        identity = live.get(bank)
        need(identity == (0x1787, 1, 1, 0x870001),
             "legal response identity")
        accepted.append(bank)
        del live[bank]
    stale = (0xdead, 7, 0xbad01787, 0xbad178)
    legal_stale = bool(live) and stale in live.values()
    return {
        "independent_arrival_order": accepted,
        "reordered": accepted != sorted(accepted),
        "all_eight_retired": not live,
        "stale_rejected": not legal_stale,
        "sticky_fault_required": True,
    }


def validate_contract():
    contract = strict_json(CONTRACT)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text(encoding="ascii").split() ==
         [sha(CONTRACT), CONTRACT.name], "contract sidecar drift")
    need(outer.read_text(encoding="ascii").split() ==
         [sha(sidecar), sidecar.name], "contract outer drift")
    need(contract.get("schema") ==
         "m1787_m1781_c2_tsbg_real_protocol_successor_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1780_FAILED_BY_M1781__DIFFERENT_AUTHOR_M1788_REQUIRED__NO_EDA",
         "contract status")
    expected = dict((str(path.relative_to(ROOT)), sha(path)) for path in
                    (M803_ADAPTER, RTL, SVA, TB, FILELIST, CHECKER, TEST))
    need(contract.get("source_sha256") == expected, "source map drift")
    need(contract.get("authorization") == {
        "run_vcs": False, "run_simv": False, "run_dc": False,
        "run_ptpx": False, "query_license": False,
        "create_attempt": False, "create_result": False,
        "release": False}, "authorization drift")
    need(contract.get("claim_boundary") == CLAIMS, "claim boundary drift")
    return contract


def validate_sources():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift " + str(path))
    verify_sealed_directory(M1781)
    review = strict_json(M1781 / "review.json")
    need(review.get("status") ==
         "FAIL_CLOSED__M1780_ARITHMETIC_AND_LEDGER_PASS__REAL_C2_INTERFACE_BRIDGE_ABSENT__NO_M1784_VCS_DC_RELEASE",
         "M1781 disposition drift")
    need(review.get("severity") == {"p0": 2, "p1": 2, "p2": 1},
         "M1781 severity drift")
    need(review.get("claim_boundary", {}).get("release_authorized") is False,
         "M1780 unexpectedly released")

    active_filelist = [row.split("#", 1)[0].strip()
                       for row in FILELIST.read_text(encoding="utf-8").splitlines()
                       if row.split("#", 1)[0].strip()]
    expected_filelist = [str(path.relative_to(ROOT)) for path in
                         (M803_ADAPTER, RTL, SVA, TB)]
    need(active_filelist == expected_filelist, "filelist order drift")

    rtl = RTL.read_text(encoding="utf-8")
    active_rtl = strip_comments(rtl)
    for token in (
            "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter",
            "output logic [7:0]                   mem_req_valid",
            "input  logic [7:0]                   mem_req_ready",
            "mem_req_epoch [0:7]", "mem_req_slot [0:7]",
            "mem_req_generation [0:7]", "mem_req_tag [0:7]",
            "mem_req_output_block [0:7]", "mem_req_slice [0:7]",
            "mem_req_source_channel [0:7]", "mem_rsp_valid",
            "load_source_active", "load_source_sign", "sign_q",
            "bridge_source_value", "bridge_effective_weight",
            "bridge_effective_weight[bank][lane] = -widened_weight",
            "logic signed [23:0] acc_q",
            "if (SCHEDULE_MODE == 0)",
            "STATIC_ACC24_ABS_BOUND == 98304",
            "debug_cache_eviction_count", "adapter_stale_response_seen"):
        need(token in rtl, "RTL omits " + token)
    for forbidden in ("mem_rsp_weight [0:7][0:LANES-1],\n+    output logic                         mem_rsp_accept",
                      "approx", "epsilon", "drop_source", "reuse_product"):
        need(forbidden not in active_rtl, "forbidden proxy/lossy token " + forbidden)
    need(active_rtl.count("m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter") == 1,
         "M803 adapter composition count")

    sva = SVA.read_text(encoding="utf-8")
    for token in (
            "ap_bank_request_stable", "ap_bank_response_stable",
            "ap_zero_does_not_issue", "ap_nonzero_is_typed_unit",
            "ap_bridge_payload_stable", "ap_fault_is_sticky",
            "ap_no_legal_overflow", "STATIC_ACC24_ABS_BOUND = 48 * 16 * 128",
            "cp_independent_bank_backpressure", "cp_bank_response_reorder",
            "cp_cache_eviction", "cp_stale_attack", "cp_reset_recovery"):
        need(token in sva, "SVA omits " + token)

    tb = TB.read_text(encoding="utf-8")
    for token in (
            "`CONNECT_M1787(dut_base, base, 0)",
            "`CONNECT_M1787(dut_tsbg, tsbg, 1)",
            "delay_q <= 8 - BANK_ID", "inject_stale",
            "stale_attack_count", "duplicate_attack_count",
            "reset_recovery_count", "saw_exact_neg128",
            "base.weight_bundle_beat_count != EXPECTED_BASE_BUNDLES",
            "tsbg.weight_bundle_beat_count != EXPECTED_TSBG_BUNDLES",
            "PASS_M1787_C2_TSBG_B8_REAL_M803_TYPED_SIGNED_DIRECTED"):
        need(token in tb, "TB omits " + token)
    need("force " not in strip_comments(tb).lower()
         and "release " not in strip_comments(tb).lower(),
         "TB uses hierarchical force/release")

    baseline = lru_ledger("token")
    candidate = lru_ledger("row")
    need(baseline == {
        "row_accesses": 96, "hits": 0, "misses": 96, "evictions": 88,
        "aggregate_eight_bank_bundle_beats": 1152,
        "scalar_bank_beats": 9216}, "baseline ledger drift")
    need(candidate == {
        "row_accesses": 96, "hits": 84, "misses": 12, "evictions": 4,
        "aggregate_eight_bank_bundle_beats": 144,
        "scalar_bank_beats": 1152}, "candidate ledger drift")
    arithmetic = arithmetic_ledger()
    need(arithmetic["issues"] == 1152
         and arithmetic["signed_products"] == 18432
         and arithmetic["commits"] == 48
         and arithmetic["exact_neg128_to_positive128"],
         "typed signed arithmetic drift")
    need(-98304 <= arithmetic["min_accumulator"] <= arithmetic["max_accumulator"] <= 98304,
         "directed arithmetic outside static bound")
    protocol = protocol_model()
    need(protocol["reordered"] and protocol["all_eight_retired"]
         and protocol["stale_rejected"], "protocol model drift")
    resources = resource_account()
    need(resources["shared_lru8_weight_data_bytes"] == 12288
         and resources["eight_by_96_acc24_context_bytes"] == 2304
         and resources["explicit_datapath_state_bytes_excluding_m803_control"] == 16152,
         "resource accounting drift")
    validate_contract()
    return {
        "status": "PASS_M1787_REAL_M803_TYPED_SIGNED_SUCCESSOR_SOURCE_ONLY_NO_EDA",
        "predecessor_disposition": {
            "m1780": "FAILED_DO_NOT_RELEASE",
            "m1781_p0_count": 2,
            "m1781_review_sha256": sha(M1781 / "review.json")},
        "directed_expectation_only": {
            "baseline": baseline, "candidate": candidate,
            "work_each": dict((key, arithmetic[key]) for key in
                              ("issues", "signed_products", "commits"))},
        "typed_signed_bridge": {
            "binary_active_sign0_is_plus_one": True,
            "negative_one_uses_exact_nine_bit_twos_complement": True,
            "negative_int8_min_maps_to_positive_128": True,
            "zero_does_not_issue": True},
        "protocol": protocol,
        "resource_account": resources,
        "static_acc24_abs_bound": 98304,
        "claim_boundary": dict(CLAIMS),
        "author_execution": {"vcs_runs": 0, "simv_runs": 0,
                             "dc_runs": 0, "ptpx_runs": 0,
                             "license_queries": 0, "attempts": 0,
                             "results": 0}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true", required=True)
    args = parser.parse_args(argv)
    del args
    print(json.dumps(validate_sources(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
