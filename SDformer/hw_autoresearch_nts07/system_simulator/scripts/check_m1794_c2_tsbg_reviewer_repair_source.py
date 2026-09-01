#!/usr/bin/env python3
"""Fail-closed source/reference checker for M1794; never launches EDA."""
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
RTL = HW / "rtl_m1794/m1794_c2_tsbg_b8_real_channel_signed_frontend.sv"
SVA = HW / "verif_m1794/m1794_c2_tsbg_b8_real_channel_signed_frontend_assertions.sv"
TB = HW / "tb_m1794/tb_m1794_c2_tsbg_b8_real_channel_signed_frontend.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1794_c2_tsbg_b8_real_channel_signed_frontend_directed_vcs.f"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1794_c2_tsbg_reviewer_repair_source.py"
CONTRACT = HW / "contracts/m1794_m1788_c2_tsbg_reviewer_repair_source_contract_r1_20260902.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1787_RTL = HW / "rtl_m1787/m1787_c2_tsbg_b8_real_channel_signed_frontend.sv"
M1787_CONTRACT = HW / "contracts/m1787_m1781_c2_tsbg_real_protocol_successor_source_contract_r1_20260902.json"
M1788 = HW / "reviews/m1788_m1787_m1781_c2_tsbg_real_protocol_successor_source_hammer_r1_20260902"
M1780_RTL = HW / "rtl_m1780/m1780_c2_tsbg_b8_typed_weight_row_frontend.sv"
M1780_CONTRACT = HW / "contracts/m1780_m1763_c2_tsbg_b8_same_resource_source_contract_r1_20260902.json"
M1781 = HW / "reviews/m1781_m1780_c2_tsbg_b8_same_resource_source_hammer_r1_20260902"
M803_ADAPTER = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M803_K8 = HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv"

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1787_RTL: "f7119779cd5e9adab98cb6252f6a946fd903f68ca341163baa6643033150be94",
    M1787_CONTRACT: "065b9b9d085f3683f3e97214d9559b800662a98dfe71d910b6de45c03532f491",
    M1788 / "review.json": "9b8a3ddfe6c35a0d2c10aa91c27f8c2ae0d7dffd48fd59a2d6e4283a54bee421",
    M1788 / "SHA256SUMS": "4d033ca9b02b0a76c391ae0a9152338bf46caa336332656f991094a50700624a",
    M1788 / "SHA256SUMS.seal.sha256": "462d17d94ca00fb109e237cccd9e92ce28a571c123c1bdee6513290e7a2e9460",
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


def elaborated_parameter_ledger(rtl_text, tb_text):
    """Evaluate the two TB elaborations against the actual RTL predicate."""
    rtl = strip_comments(rtl_text)
    tb = strip_comments(tb_text)
    production_match = re.search(
        r"PRODUCTION_SOURCE_GROUPS\s*=\s*(\d+)", rtl)
    minimum_match = re.search(r"SOURCE_GROUPS\s*>=\s*(\d+)", rtl)
    tb_groups_match = re.search(
        r"localparam\s+int\s+BUNDLE\s*=\s*8\s*,\s*GROUPS\s*=\s*(\d+)", tb)
    need(production_match and minimum_match and tb_groups_match,
         "cannot parse elaborated group tuple")
    production_groups = int(production_match.group(1))
    minimum_groups = int(minimum_match.group(1))
    directed_groups = int(tb_groups_match.group(1))
    need(rtl.count("SOURCE_GROUPS <= PRODUCTION_SOURCE_GROUPS") == 1,
         "maximum group predicate drift")
    production_bound = production_groups * 16 * 128
    directed_bound = directed_groups * 16 * 128
    predicate_tokens = all(token in rtl for token in (
        "PRODUCTION_ACC24_ABS_BOUND == 98304",
        "ELABORATED_ACC24_ABS_BOUND <= PRODUCTION_ACC24_ABS_BOUND",
        "PRODUCTION_ACC24_ABS_BOUND < (1 << 23)"))
    legal = (predicate_tokens and minimum_groups <= directed_groups
             and directed_groups <= production_groups
             and production_bound == 98304
             and directed_bound <= production_bound
             and production_bound < (1 << 23))
    need(tb.count(".SOURCE_GROUPS(GROUPS)") == 2,
         "DUT/SVA macro override count drift")
    tuples = [
        {"instance": "dut_base", "schedule_mode": 0,
         "source_groups": directed_groups, "legal": legal},
        {"instance": "dut_tsbg", "schedule_mode": 1,
         "source_groups": directed_groups, "legal": legal},
    ]
    return {
        "production_source_groups": production_groups,
        "production_acc24_abs_bound": production_bound,
        "directed_source_groups": directed_groups,
        "directed_acc24_abs_bound": directed_bound,
        "minimum_legal_source_groups": minimum_groups,
        "maximum_legal_source_groups": production_groups,
        "tuples": tuples,
        "all_dut_tuples_legal": all(item["legal"] for item in tuples),
        "time_zero_parameter_fatal": not legal,
    }


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
    legal_identity = (0x1794, 1, 1, 0x940001)
    live = dict((bank, legal_identity) for bank in range(8))
    arrival = [7, 5, 3, 1, 6, 4, 2, 0]
    accepted = []
    saved_bank = 3
    saved_identity = None
    for bank in arrival:
        identity = live.get(bank)
        need(identity == legal_identity, "legal response identity")
        accepted.append(bank)
        if bank == saved_bank:
            saved_identity = identity
        del live[bank]
    replay_accepted = saved_bank in live and live[saved_bank] == saved_identity
    bogus = (0xdead, 7, 0xbad01794, 0xbad194)
    bogus_accepted = bool(live) and bogus in live.values()
    return {
        "independent_arrival_order": accepted,
        "reordered": accepted != sorted(accepted),
        "all_eight_retired": not live,
        "saved_accepted_bank": saved_bank,
        "retired_legal_identity_replay_rejected": not replay_accepted,
        "bogus_stale_rejected": not bogus_accepted,
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
         "m1794_m1788_c2_tsbg_reviewer_repair_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1787_FAILED_BY_M1788__DIFFERENT_AUTHOR_M1795_REQUIRED__NO_EDA",
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
    verify_sealed_directory(M1788)
    review = strict_json(M1788 / "review.json")
    need(review.get("status") ==
         "FAIL_CLOSED__M1787_CLOSES_M1781_ARCHITECTURE_P0S__DIRECTED_PARAMETER_FATAL_BLOCKS_VCS_RELEASE",
         "M1788 disposition drift")
    need(review.get("severity") == {"p0": 1, "p1": 2, "p2": 1},
         "M1788 severity drift")
    need(review.get("authorization", {}).get("release") is False,
         "M1787 unexpectedly released")

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
            "PRODUCTION_SOURCE_GROUPS = 48",
            "PRODUCTION_ACC24_ABS_BOUND == 98304",
            "ELABORATED_ACC24_ABS_BOUND <= PRODUCTION_ACC24_ABS_BOUND",
            "SOURCE_GROUPS >= 1 && SOURCE_GROUPS <= PRODUCTION_SOURCE_GROUPS",
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
            "ap_no_legal_overflow", "PRODUCTION_ACC24_ABS_BOUND = 48 * 16 * 128",
            "ELABORATED_ACC24_ABS_BOUND = SOURCE_GROUPS * 16 * 128",
            "cp_independent_bank_backpressure", "cp_bank_response_reorder",
            "cp_cache_eviction", "cp_stale_attack",
            "cp_reset_recovery_minimum_one_cycle", "rst_core[*1:8]",
            "commit_accept && commit_terminal"):
        need(token in sva, "SVA omits " + token)

    tb = TB.read_text(encoding="utf-8")
    for token in (
            "`CONNECT_M1794(dut_base, base, 0)",
            "`CONNECT_M1794(dut_tsbg, tsbg, 1)",
            "delay_q <= 8 - BANK_ID", "inject_stale",
            "inject_replay", "saved_rsp_epoch <= tsbg.mem_rsp_epoch[3]",
            "tsbg.replay_epoch[3] = saved_rsp_epoch",
            "retired_identity_replay_count", "replay_accept_count",
            "load_minimal_legal_workload", "post_reset_legal_service_count",
            "repeat (3) @(posedge clk_core)", "reset_recovery_count",
            "saw_exact_neg128", "DIRECTED_ACC24_ABS_BOUND=GROUPS*16*128",
            "base.weight_bundle_beat_count != EXPECTED_BASE_BUNDLES",
            "tsbg.weight_bundle_beat_count != EXPECTED_TSBG_BUNDLES",
            "PASS_M1794_C2_TSBG_B8_REAL_M803_TYPED_SIGNED_DIRECTED"):
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
         and protocol["retired_legal_identity_replay_rejected"]
         and protocol["bogus_stale_rejected"], "protocol model drift")
    parameters = elaborated_parameter_ledger(rtl, tb)
    need(parameters == {
        "production_source_groups": 48,
        "production_acc24_abs_bound": 98304,
        "directed_source_groups": 12,
        "directed_acc24_abs_bound": 24576,
        "minimum_legal_source_groups": 1,
        "maximum_legal_source_groups": 48,
        "tuples": [
            {"instance": "dut_base", "schedule_mode": 0,
             "source_groups": 12, "legal": True},
            {"instance": "dut_tsbg", "schedule_mode": 1,
             "source_groups": 12, "legal": True}],
        "all_dut_tuples_legal": True,
        "time_zero_parameter_fatal": False},
        "elaborated tuple/predicate drift")
    parameter_mutations = {
        "minimum_raised_to_48_detects_time_zero_fatal":
            elaborated_parameter_ledger(
                rtl.replace("SOURCE_GROUPS >= 1", "SOURCE_GROUPS >= 48", 1),
                tb)["time_zero_parameter_fatal"],
        "directed_groups_49_detects_time_zero_fatal":
            elaborated_parameter_ledger(
                rtl, tb.replace("BUNDLE=8, GROUPS=12",
                                "BUNDLE=8, GROUPS=49", 1)
            )["time_zero_parameter_fatal"],
        "production_groups_12_detects_bad_proof":
            elaborated_parameter_ledger(
                rtl.replace("PRODUCTION_SOURCE_GROUPS = 48",
                            "PRODUCTION_SOURCE_GROUPS = 12", 1), tb
            )["time_zero_parameter_fatal"],
    }
    need(all(parameter_mutations.values()),
         "parameter mutation escaped time-zero-fatal checker")
    resources = resource_account()
    need(resources["shared_lru8_weight_data_bytes"] == 12288
         and resources["eight_by_96_acc24_context_bytes"] == 2304
         and resources["explicit_datapath_state_bytes_excluding_m803_control"] == 16152,
         "resource accounting drift")
    validate_contract()
    return {
        "status": "PASS_M1794_M1788_REVIEWER_REPAIR_SOURCE_ONLY_NO_EDA",
        "predecessor_disposition": {
            "m1787": "FAILED_DO_NOT_RELEASE",
            "m1788_findings": {"p0": 1, "p1": 2, "p2": 1},
            "m1788_review_sha256": sha(M1788 / "review.json")},
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
        "elaborated_parameter_ledger": parameters,
        "parameter_mutations_detected": parameter_mutations,
        "resource_account": resources,
        "production_acc24_abs_bound": 98304,
        "directed_acc24_abs_bound": 24576,
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
