#!/usr/bin/env python3
"""Fail-closed source checker/reference model for M1780; never launches EDA."""
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
RTL = HW / "rtl_m1780/m1780_c2_tsbg_b8_typed_weight_row_frontend.sv"
SVA = HW / "verif_m1780/m1780_c2_tsbg_b8_typed_weight_row_frontend_assertions.sv"
TB = HW / "tb_m1780/tb_m1780_c2_tsbg_b8_typed_weight_row_frontend.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1780_c2_tsbg_b8_typed_weight_row_frontend_directed_vcs.f"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1780_c2_tsbg_b8_same_resource_source.py"
CONTRACT = HW / "contracts/m1780_m1763_c2_tsbg_b8_same_resource_source_contract_r1_20260902.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1763 = HW / "results/m1763_m1707_ep34_tsbg_layer_private_s2_witness_r1_20260902"
M1775 = HW / "reviews/m1775_m1763_m1707_ep34_tsbg_layer_private_s2_witness_result_hammer_r1_20260902"
M803_RTL = HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv"
M803_ADAPTER = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M803_CONTRACT = HW / "contracts/m803_c2_r16_channel_split_source_only_contract_r1_20260828.json"

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1763 / "decision.json": "722aa302c983b63eae4e40816cffd123d0da34b09df56b872d52502d18cee961",
    M1763 / "SHA256SUMS": "e70b0f837bdaa24b0345193e5c0048f2ce14cb803b9106871c91aac2b8d48332",
    M1763 / "SHA256SUMS.seal.sha256": "9e08414fcb185fe8cd1251ca758d8319dab84fd7dd8367b55f29f2b02da8ff2e",
    M1775 / "review.json": "668394ce23a303dfb32177c1ddebd45807473380ce5e06cdaed30c8aa3564147",
    M1775 / "SHA256SUMS": "22a117badea64ed276ef14e3b74434b2ce5c9eb40891e39bf9a5fa1462d57246",
    M1775 / "SHA256SUMS.seal.sha256": "e25065f89f26f6880c5fcafe83cf017919db9bd9769929cdb5734d5a1760d80a",
    M803_RTL: "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
    M803_ADAPTER: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    M803_CONTRACT: "31290e029fdfefe15fe1eea9bb70537a9ce97f5137b3809761e0d28c30461067",
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
    listed = []
    for row in manifest.read_text(encoding="ascii").splitlines():
        fields = row.split(None, 1)
        need(len(fields) == 2 and len(fields[0]) == 64, "manifest syntax")
        name = fields[1].strip().lstrip("*")
        rel = Path(name)
        need(name not in listed and not rel.is_absolute()
             and ".." not in rel.parts, "unsafe manifest member")
        need(sha(root / rel) == fields[0], "manifest member drift " + name)
        listed.append(name)


def strip_sv_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def lru_miss_ledger(active, mode, capacity=8):
    """Independent ordinary-LRU model for token- versus group-major order."""
    contexts = len(active)
    groups = len(active[0])
    need(contexts == 8 and groups >= capacity and capacity > 0,
         "reference geometry")
    accesses = []
    if mode == "token":
        for context in range(contexts):
            for group in range(groups):
                if active[context][group]:
                    accesses.append(group)
    elif mode == "tsbg":
        for group in range(groups):
            for context in range(contexts):
                if active[context][group]:
                    accesses.append(group)
    else:
        raise RuntimeError("unknown schedule")
    cache = []
    misses = 0
    hits = 0
    for key in accesses:
        if key in cache:
            cache.remove(key)
            hits += 1
        else:
            misses += 1
            if len(cache) == capacity:
                cache.pop(0)
        cache.append(key)
    return {"accesses": len(accesses), "hits": hits, "misses": misses,
            "weight_beats": misses * 2 * 6}


def directed_codes(context, group):
    values = [0] * 16
    source0 = (context + group) % 8
    source1 = 8 + ((context * 3 + group) % 8)
    value0 = 1 if (context + group) % 2 == 0 else -1
    values[source0] = value0
    values[source1] = -value0
    return values


def directed_weight(group, half, output_slice, bank, lane):
    return (group * 17 + half * 11 + output_slice * 7
            + bank * 5 + lane * 3) % 63 - 31


def directed_accumulators(groups=12):
    """Independent signed-product reference; product sharing is forbidden."""
    result = [[([0] * 16) for _ in range(6)] for _ in range(8)]
    products = 0
    issues = 0
    for context in range(8):
        for group in range(groups):
            values = directed_codes(context, group)
            for half in range(2):
                half_values = values[half * 8:(half + 1) * 8]
                if not any(half_values):
                    continue
                for output_slice in range(6):
                    issues += 1
                    for bank, value in enumerate(half_values):
                        if not value:
                            continue
                        products += 16
                        for lane in range(16):
                            result[context][output_slice][lane] += (
                                value * directed_weight(
                                    group, half, output_slice, bank, lane))
    return {"accumulators": result, "issues": issues,
            "signed_products": products, "commits": 8 * 6}


def resource_account(groups=48):
    result = {
        "bundle": 8,
        "ordinary_lru_rows": 8,
        "banks": 8,
        "int8_weight_row_bytes": 16 * 96,
        "shared_row_cache_bytes": 8 * 16 * 96,
        "acc24_context_bytes": 8 * 96 * 3,
        "source_fifo_bytes": 8 * groups * 16,
        "context_tag_bytes": 8 * 3,
        "active_bitmap_bytes": 8 * groups // 8,
        "m1763_incremental_state_lower_bound_bytes": 2128,
    }
    result["explicit_state_bytes_excluding_control"] = sum(
        result[key] for key in ("shared_row_cache_bytes", "acc24_context_bytes",
                                "source_fifo_bytes", "context_tag_bytes",
                                "active_bitmap_bytes"))
    return result


def validate_contract():
    contract = strict_json(CONTRACT)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text(encoding="ascii").split() ==
         [sha(CONTRACT), CONTRACT.name], "contract sidecar drift")
    need(outer.read_text(encoding="ascii").split() ==
         [sha(sidecar), sidecar.name], "contract outer drift")
    need(contract.get("schema") ==
         "m1780_m1763_c2_tsbg_b8_same_resource_source_contract_r1_v1"
         and contract.get("status") ==
         "SOURCE_ONLY__DIFFERENT_AUTHOR_M1781_REQUIRED__NO_VCS_NO_EDA_NO_PAPER_RESULT",
         "contract schema/status")
    expected = {
        str(path.relative_to(ROOT)): sha(path)
        for path in (RTL, SVA, TB, FILELIST, CHECKER, TEST)}
    need(contract.get("source_sha256") == expected, "contract source map drift")
    need(contract.get("authorization") == {
        "run_vcs": False, "run_simv": False, "run_dc": False,
        "run_ptpx": False, "query_license": False,
        "create_attempt": False, "create_result": False,
        "release": False}, "source-only authorization drift")
    need(contract.get("claim_boundary") == CLAIMS, "claim boundary drift")
    return contract


def validate_sources():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift " + str(path))
    verify_sealed_directory(M1763)
    verify_sealed_directory(M1775)
    decision = strict_json(M1763 / "decision.json")
    rows = decision.get("tsbg", {}).get("rows", [])
    all_b8 = [row for row in rows if row.get("bundle") == 8
              and row.get("scope_type") == "all"]
    need(len(all_b8) == 1, "M1763 all/B8 row")
    b8 = all_b8[0]
    need(b8.get("ordinary_lru_capacity_rows") == 8
         and b8.get("candidate_incremental_state_bytes_lower_bound") == 2128
         and b8.get("compute_work_changed") is False
         and b8.get("same_resource_claim") is False
         and b8.get("full_area_energy_pricing_complete") is False,
         "M1763 boundary drift")
    review = strict_json(M1775 / "review.json")
    need(review.get("status") ==
         "PASS_M1775_M1763_DIAGNOSTIC_RESULT_HAMMER__TSBG_SCREENING_ONLY__S2_NO_ADMIT",
         "M1775 disposition drift")

    expected_filelist = [str(path.relative_to(ROOT))
                         for path in (RTL, SVA, TB)]
    active_filelist = [row.split("#", 1)[0].strip()
                       for row in FILELIST.read_text(encoding="utf-8").splitlines()
                       if row.split("#", 1)[0].strip()]
    need(active_filelist == expected_filelist, "filelist order drift")
    rtl = RTL.read_text(encoding="utf-8")
    active_rtl = strip_sv_comments(rtl)
    for token in (
            "parameter int SCHEDULE_MODE = 1", "SOURCE_GROUPS = 48",
            "CACHE_ROWS = 8", "logic signed [23:0] acc_q",
            "logic signed [7:0] source_value_q",
            "logic signed [7:0] cache_weight_q",
            "M1763_B8_INCREMENTAL_STATE_LOWER_BOUND_BYTES = 2128",
            "if (SCHEDULE_MODE == 0)", "issue_source_value",
            "issue_weight", "delta = delta +", "delta = delta -",
            "cache_hit_count_q", "cache_miss_count_q",
            "signed_product_count_q", "commit_count_q"):
        need(token in rtl, "RTL omits " + token)
    for forbidden in ("issue_source_value[bank] *", "reuse_product",
                      "approx", "epsilon", "drop_source"):
        need(forbidden not in active_rtl, "forbidden product/lossy path " + forbidden)
    need(rtl.count("ST_MEM_REQ") >= 3 and rtl.count("ST_MEM_RSP") >= 3,
         "eight-bank fetch protocol absent")
    sva = SVA.read_text(encoding="utf-8")
    for token in ("ap_mem_req_stable", "ap_issue_payload_stable",
                  "ap_commit_payload_stable", "ap_fault_closes_load",
                  "cp_negative_source", "cp_positive_source",
                  "cp_protocol_attack"):
        need(token in sva, "SVA omits " + token)
    tb = TB.read_text(encoding="utf-8")
    for token in ("`CONNECT_DUT(base, 0)", "`CONNECT_DUT(tsbg, 1)",
                  "cache_miss_count_base != 96",
                  "cache_miss_count_tsbg != 12",
                  "weight_beat_count_base != 1152",
                  "weight_beat_count_tsbg != 144",
                  "directed same-resource local gate below 1.15x",
                  "load_source_value[0] = 8'sd2",
                  "PASS_M1780_C2_TSBG_B8_TYPED_WEIGHT_ROW_FRONTEND_DIRECTED"):
        need(token in tb, "TB omits " + token)
    need("force " not in strip_sv_comments(tb).lower()
         and "release " not in strip_sv_comments(tb).lower(),
         "TB drives hierarchy")

    token_active = [[True] * 12 for _ in range(8)]
    base = lru_miss_ledger(token_active, "token")
    tsbg = lru_miss_ledger(token_active, "tsbg")
    need(base == {"accesses": 96, "hits": 0, "misses": 96,
                  "weight_beats": 1152}, "reference baseline drift")
    need(tsbg == {"accesses": 96, "hits": 84, "misses": 12,
                  "weight_beats": 144}, "reference TSBG drift")
    arithmetic = directed_accumulators()
    need(arithmetic["issues"] == 1152
         and arithmetic["signed_products"] == 18432
         and arithmetic["commits"] == 48,
         "reference work ledger drift")
    resources = resource_account()
    need(resources["shared_row_cache_bytes"] == 12288
         and resources["acc24_context_bytes"] == 2304
         and resources["source_fifo_bytes"] == 6144
         and resources["m1763_incremental_state_lower_bound_bytes"] == 2128,
         "resource accounting drift")
    contract = validate_contract()
    return {
        "status": "PASS_M1780_TSBG_B8_SAME_RESOURCE_SOURCE_ONLY_NO_EDA",
        "predecessor_b8_screening": {
            "roofline_ratio": b8["roofline_cycle_speedup"],
            "weight_fetch_reduction": b8["weight_fetch_reduction"],
            "same_resource_claim": False},
        "directed_reference": {"baseline": base, "candidate": tsbg,
                               "work": dict((key, arithmetic[key]) for key in
                                            ("issues", "signed_products", "commits"))},
        "resource_account": resources,
        "contract_sha256": sha(CONTRACT),
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
    value = validate_sources()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
