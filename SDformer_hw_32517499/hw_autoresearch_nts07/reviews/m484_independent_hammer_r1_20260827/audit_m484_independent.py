#!/usr/bin/env python3
"""Independent, read-only M484 evidence audit.

This checker intentionally does not import or execute the M484 producer.  It
recomputes invariants from sealed CSV/JSON artifacts and frozen inputs using
only the Python standard library.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/m484_row_coherent_signed_bundle_stationary_dse_r1_20260827"
CONTRACT = ROOT / "contracts/m484_row_coherent_signed_bundle_stationary_dse_contract_r1_20260827.json"
M51 = ROOT / "system_handoff/incoming/m51_capture_bundle_r2_20260823/manifest.json"
M22 = ROOT / "results/m22_ordered_system_transactions_s10_r2_final_20260822/m22_ordered_transactions.csv"
DOC359 = ROOT / "docs/359_DATE终局冻结_20260813.md"
M218_RTL = ROOT / "rtl_m218/m218_fc2_tagged_slice_service_island.sv"
M218_PREMODEL = ROOT / "results/m218_h67_fc2_tagged_slice_service_premodel_r1_20260825/m218_h67_fc2_tagged_slice_service_premodel_r1.json"
M219_DC = ROOT / "results/m219_dc_independent_hammer_review_r1_20260825/m219_dc_independent_hammer_review_r1.json"

EXPECTED = {
    "contract": "09e181a985cfd10abea131185a2f8cc20cf47111ca047bd07c3bac41aea29d30",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m51": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "m22": "dbd6630b3bec3726762270ae6c6c24b6328da7c65d6f2c6a5878be3940b4ef59",
    "m218_rtl": "f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1",
    "m218_premodel": "f4e1c72a6d6030fd83543d262fd5262a55ac09f0ba95b00b9be8f6023135a9ea",
    "m219_dc": "61c1521dc676a6267c3ab7709d8e13e9634e6f1cd6a3201e8956d48c368e5b4c",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def i(row, key):
    return int(row[key])


def f(row, key):
    return float(row[key])


def close(actual, expected, name, tol=1e-12):
    require(math.isclose(actual, expected, rel_tol=tol, abs_tol=tol),
            f"{name}: {actual} != {expected}")


def main():
    contract = json.loads(CONTRACT.read_text())
    result_path = OUT / "m484_row_coherent_signed_bundle_stationary_dse.json"
    result = json.loads(result_path.read_text())
    manifest = json.loads((OUT / "m484_manifest.json").read_text())

    # Frozen identity and the dual SHA seal.
    input_hashes = {
        "contract": sha(CONTRACT), "docs359": sha(DOC359), "m51": sha(M51),
        "m22": sha(M22), "m218_rtl": sha(M218_RTL),
        "m218_premodel": sha(M218_PREMODEL), "m219_dc": sha(M219_DC),
    }
    require(input_hashes == EXPECTED, f"frozen input hash mismatch: {input_hashes}")
    require(contract["inputs"]["docs359"]["sha256"] == EXPECTED["docs359"], "contract docs359 hash")
    require(manifest["contract_sha256"] == EXPECTED["contract"], "manifest contract hash")
    require(manifest["input_sha256"]["docs359"] == EXPECTED["docs359"], "manifest docs359 hash")

    sum_lines = [x.split() for x in (OUT / "SHA256SUMS").read_text().splitlines() if x.strip()]
    sealed = {}
    for digest, name in sum_lines:
        require(sha(OUT / name) == digest, f"producer SHA mismatch: {name}")
        sealed[name] = digest
    seal_digest, seal_name = (OUT / "SHA256SUMS.seal.sha256").read_text().split()
    require(seal_name == "SHA256SUMS", "bad dual-seal target")
    require(sha(OUT / "SHA256SUMS") == seal_digest, "bad dual seal")
    require(manifest["files"] == {k: v for k, v in sealed.items() if k not in {"m484_manifest.json"}},
            "manifest file map mismatch")

    ledger = rows(OUT / "m484_per_record_exact_row_ledger.csv")
    reconciliation = rows(OUT / "m484_dual_line_reconciliation.csv")
    dse = rows(OUT / "m484_row_coherent_signed_bundle_stationary_dse.csv")
    require(len(ledger) == 160 and len({(x["sample_id"], x["name"]) for x in ledger}) == 160,
            "ledger grain/uniqueness")
    require(len(reconciliation) == 160, "reconciliation row count")
    require(all(i(x, "selected_source_delta") == 0 and i(x, "current_source_delta") == 0
                for x in reconciliation), "dual-line reconciliation mismatch")
    require(len(dse) == 99, "DSE row count")

    # Manifest population: exactly 70 Conv + 100 FC1 selected, with ten missing
    # preds payloads; local ledger is the remaining 60 Conv + 100 FC1.
    m51 = json.loads(M51.read_text())
    selected_manifest = [x for x in m51["records"]
                         if x["operator"] == "Conv2d" or x["name"].endswith(".mlp.fc1")]
    missing = [x for x in selected_manifest
               if not (M51.parent / x["relative_path"]).is_file()]
    require(len(selected_manifest) == 170, "selected manifest population")
    require(len(missing) == 10, "missing payload count")
    require({x["name"] for x in missing} == {"sttmultires_unet.preds.3.conv.0"}, "missing module identity")
    ledger_keys = {(i(x, "sample_id"), x["name"]) for x in ledger}
    present_keys = {(int(x["sample_id"]), x["name"]) for x in selected_manifest
                    if (M51.parent / x["relative_path"]).is_file()}
    require(ledger_keys == present_keys, "ledger does not equal present manifest population")
    conv = [x for x in ledger if x["operator"] == "Conv2d"]
    fc1 = [x for x in ledger if x["operator"] == "Linear" and x["name"].endswith(".mlp.fc1")]
    paired = [x for x in conv if x["conv_atlif_pair"] == "True"]
    require((len(conv), len(fc1), len(paired)) == (60, 100, 40), "60/100/40 population")
    require({(x["sample_id"], x["name"]) for x in paired}.issubset(
            {(x["sample_id"], x["name"]) for x in conv}), "Conv->ATLIF subset")
    require(len({x["name"] for x in conv}) == 6 and len({x["name"] for x in fc1}) == 10,
            "unique operator populations")

    def category_rows(category):
        return conv if category == "Conv" else paired if category == "Conv->ATLIF" else fc1

    aggregate_fields = ["rows", "nonempty_rows", "selected_sources", "positive_sources",
                        "negative_sources", "motion_selected_rows"]
    n8 = {}
    for category in ("Conv", "Conv->ATLIF", "FC1"):
        offline = [x for x in dse if x["schedule"] == "offline_destination_major_oracle"
                   and x["category"] == category and i(x, "bundle_width") == 8]
        require(len(offline) == 4 and {i(x, "finite_slots") for x in offline} == {1, 2, 4, 8},
                f"{category} N8 slots")
        ref = offline[0]
        ignore = {"finite_slots"}
        require(all(all(x[k] == ref[k] for k in ref if k not in ignore) for x in offline[1:]),
                f"{category} offline slots changed arithmetic")
        source = category_rows(category)
        for field in aggregate_fields:
            require(i(ref, field) == sum(i(x, field) for x in source), f"{category} aggregate {field}")
        require(i(ref, "records") == len(source), f"{category} record count")
        require(i(ref, "operators") == len({x["name"] for x in source}), f"{category} operators")
        require(i(ref, "samples") == 10, f"{category} samples")

        N = 8
        bundles = i(ref, "bundles")
        full = i(ref, "full_bundles")
        remainder = i(ref, "remainder_bundles")
        selected = i(ref, "selected_sources")
        nonempty = i(ref, "nonempty_rows")
        total_rows = i(ref, "rows")
        require(bundles == full + remainder, f"{category} bundle partition")
        require(i(ref, "padding_slots") == N * bundles - selected, f"{category} padding")
        close(f(ref, "bundle_occupancy"), selected / (N * bundles), f"{category} occupancy")
        close(f(ref, "mean_sources_per_bundle"), selected / bundles, f"{category} bundle mean")

        k1 = selected + 2 * nonempty + (total_rows if category == "Conv->ATLIF" else 0)
        k8 = bundles + 2 * nonempty + (total_rows if category == "Conv->ATLIF" else 0)
        require(i(ref, "k1_resident_cycles") == k1, f"{category} K1 cycles")
        require(i(ref, "k8_resident_cycles") == k8, f"{category} K8 cycles")
        require(i(ref, "m484_signed_bundle_cycles") == k8, f"{category} M484 cycles")
        require(i(ref, "baseline_cycles") == k8 and i(ref, "candidate_cycles") == k8,
                f"{category} strong baseline cycles")
        close(f(ref, "k1_to_m484_resource_scaling_speedup"), k1 / k8, f"{category} K1 scaling ratio")
        close(f(ref, "same_resource_cycle_speedup"), 1.0, f"{category} same-resource ratio")

        base_traffic = i(ref, "baseline_state_psum_plus_weight_bits")
        cand_traffic = i(ref, "candidate_state_psum_plus_weight_metadata_padding_bits")
        require(base_traffic == i(ref, "baseline_state_psum_rw_bits")
                + i(ref, "weight_bits_both_modes") + i(ref, "k8_resident_row_header_bits")
                + i(ref, "candidate_event_metadata_bits"),
                f"{category} baseline traffic composition")
        require(i(ref, "candidate_metadata_bits") == i(ref, "candidate_header_bits")
                + i(ref, "candidate_event_metadata_bits"), f"{category} metadata composition")
        require(cand_traffic == i(ref, "candidate_state_psum_rw_bits")
                + i(ref, "weight_bits_both_modes") + i(ref, "candidate_metadata_bits")
                + i(ref, "candidate_padding_bits"), f"{category} candidate traffic composition")
        close(f(ref, "traffic_reduction_fraction"), 1.0 - cand_traffic / base_traffic,
              f"{category} traffic ratio")
        require(i(ref, "baseline_state_psum_rw_bits") == i(ref, "candidate_state_psum_rw_bits"),
                f"{category} same resident-state boundary")

        # For fixed-width packs, every full bundle contributes wait=7 and every
        # remainder wait is <=6.  A full-bundle fraction above 50/95/99 percent
        # is sufficient to force the reported nearest-rank quantile to seven.
        full_fraction = full / bundles
        for q, field in ((0.50, "pack_wait_accepted_events_p50"),
                         (0.95, "pack_wait_accepted_events_p95"),
                         (0.99, "pack_wait_accepted_events_p99")):
            require(full_fraction >= 1.0 - q and i(ref, field) == 7,
                    f"{category} {field} self-consistency")
        require(i(ref, "finite_slot_stall_cycles") == 0
                and "offline destination-major" in ref["finite_slot_stall_reason"],
                f"{category} offline-only zero-stall boundary")
        require(ref["screen_gate"] == "False" and ref["rtl_authorized"] == "False"
                and ref["paper_performance_admitted"] == "False", f"{category} fail closed")
        n8[category] = {
            "records": len(source), "operators": len({x["name"] for x in source}),
            "selected_sources": selected, "bundles": bundles,
            "occupancy": f(ref, "bundle_occupancy"),
            "wait_p50_p95_p99": [i(ref, "pack_wait_accepted_events_p50"),
                                  i(ref, "pack_wait_accepted_events_p95"),
                                  i(ref, "pack_wait_accepted_events_p99")],
            "k1_resident_cycles": k1, "k8_resident_cycles": k8,
            "m484_cycles": k8,
            "k1_to_k8_resource_scaling": f(ref, "k1_to_m484_resource_scaling_speedup"),
            "m484_vs_strong_k8_cycles": f(ref, "same_resource_cycle_speedup"),
            "m484_vs_strong_k8_traffic_reduction": f(ref, "traffic_reduction_fraction"),
            "screen_gate": False,
        }

    # Online boundary: FC1 is exact and numerically identical to the offline N8
    # point. Conv receives only the explicit one-source safe lower bound.
    numerical = [x for x in dse[0].keys() if x not in {"schedule", "finite_slot_stall_reason"}]
    online_summary = {}
    for category in ("Conv", "Conv->ATLIF", "FC1"):
        online = [x for x in dse if x["category"] == category and x["schedule"].startswith("online_original")]
        require(len(online) == 1, f"{category} online row")
        on = online[0]
        off = next(x for x in dse if x["category"] == category
                   and x["schedule"] == "offline_destination_major_oracle"
                   and i(x, "bundle_width") == 8 and i(x, "finite_slots") == 1)
        if category == "FC1":
            require(on["schedule"] == "online_original_C_order_exact", "FC1 online exact label")
            require(all(on[k] == off[k] for k in numerical), "FC1 online/offline N8 mismatch")
        else:
            require(on["schedule"] == "online_original_NCHW_safe_lower_bound", "Conv safe label")
            require(i(on, "bundles") == i(on, "selected_sources"), "Conv one source per packet")
            close(f(on, "bundle_occupancy"), 0.125, "Conv safe occupancy")
            require([i(on, x) for x in ("pack_wait_accepted_events_p50",
                                         "pack_wait_accepted_events_p95",
                                         "pack_wait_accepted_events_p99")] == [0, 0, 0], "Conv safe waits")
        require(f(on, "same_resource_cycle_speedup") == 1.0 and on["screen_gate"] == "False",
                f"{category} online must be NO-GO")
        online_summary[category] = {
            "schedule": on["schedule"], "cycle_speedup": f(on, "same_resource_cycle_speedup"),
            "traffic_reduction": f(on, "traffic_reduction_fraction"), "screen_gate": False,
        }

    # Independent M22 Acc32 transaction audit.  Each selected operator call has
    # ten temporal writes whose bytes sum to rows*out_channels*4.  Each of the
    # 40 declared adjacent Conv->ATLIF pairs is followed by one state read and
    # one state write at call_index+1, each at the same Acc32 tensor size.
    op_tx = defaultdict(list)
    call_phase = defaultdict(list)
    with M22.open(newline="") as stream:
        for tx in csv.DictReader(stream):
            if tx["identity"] != "h67_ep35" or tx["variant"] != "motion_selector_shared_state":
                continue
            sid, call = int(tx["sample_id"]), int(tx["call_index"])
            call_phase[(sid, call, tx["phase"])].append(tx)
            if tx["phase"] == "operator_acc_write" and (sid, tx["name"]) in ledger_keys:
                op_tx[(sid, tx["name"])].append(tx)
    require(len(op_tx) == 160, "M22 selected operator checks")
    operator_checks = 0
    atlif_checks = 0
    for rec in ledger:
        key = (i(rec, "sample_id"), rec["name"])
        txs = op_tx[key]
        expected_bytes = i(rec, "rows") * i(rec, "out_channels") * 4
        require(len(txs) == 10 and sum(int(x["byte_count"]) for x in txs) == expected_bytes,
                f"M22 operator Acc32 width {key}")
        require({int(x["call_index"]) for x in txs}.__len__() == 1, f"M22 operator call {key}")
        operator_checks += 1
        if rec["operator"] == "Conv2d" and rec["conv_atlif_pair"] == "True":
            next_call = int(txs[0]["call_index"]) + 1
            for phase in ("atlif_state_read", "atlif_state_write"):
                state = call_phase[(key[0], next_call, phase)]
                require(len(state) == 1 and int(state[0]["byte_count"]) == expected_bytes,
                        f"M22 adjacent ATLIF Acc32 width {key} {phase}")
                atlif_checks += 1
    require((operator_checks, atlif_checks) == (160, 80), "M22 width audit counts")

    # M218/M219 are a precedent for resident accumulation and K-width area
    # sensitivity only. Their Acc24 FC2 scope is not M484 Acc32 FC1 PPA.
    rtl_text = M218_RTL.read_text()
    require("logic signed [23:0] ctx_q [0:7][0:SLICES-1][0:SLICE_LANES-1]" in rtl_text,
            "M218 resident context declaration")
    require("ctx_q[rsp_skid_block_q][rsp_skid_slice_q][lane]" in rtl_text,
            "M218 context reuse/update")
    m218 = json.loads(M218_PREMODEL.read_text())
    m219 = json.loads(M219_DC.read_text())
    area = m219["matched_m218_m219_comparison"]
    perf = m219["conditional_performance_area_sensitivity"]
    require(area["common_acc24_context_bits"] == 18432, "M218/M219 common context")
    close(area["m218_k8_cell_area_um2"], 88851.042296, "M218 K8 area")
    close(area["m219_k1_cell_area_um2"], 76857.858437, "M219 K1 area")
    close(area["k8_area_overhead_vs_k1_percent"], 15.60436902991611, "K8 area overhead")
    close(perf["service_cycle_speedup"], 4.952121572835196, "M218 premodel speedup")
    require(perf["rtl_measured"] is False and perf["macro_aware"] is False,
            "M218/M219 scope boundary")

    require(result["status"] == "NO_GO_VS_STRONG_K8_RESIDENT_BASELINE_NOT_ADMISSION", "result status")
    require(result["gate"]["all_categories_pass"] is False
            and result["gate"]["all_online_original_order_pass"] is False
            and result["gate"]["performance_admitted"] is False
            and result["gate"]["rtl_authorized"] is False, "result fail-closed gate")
    require(contract["rtl_authorized"] is False and contract["system_speedup_admitted"] is False,
            "contract fail closed")

    return {
        "schema": "m484_independent_hammer_audit_v1",
        "status": "PASS_AUDIT__NO_GO_PERFORMANCE_OR_RTL",
        "score_0_to_100": 68,
        "decision": {
            "sealed_screen_arithmetic_and_population": "GO",
            "fc1_online_C_order_measurement": "GO_AS_ORDERING_FACT_ONLY",
            "offline_conv_destination_major": "RECORD_ONLY_NOT_ONLINE",
            "m484_vs_same_resource_k8_resident_advantage": "NO_GO",
            "new_rtl": "NO_GO",
            "performance_system_or_paper_claim": "NO_GO",
        },
        "producer_seal": {"sha256sums_sha256": sha(OUT / "SHA256SUMS"),
                          "dual_seal_verified": True, "files_verified": len(sealed)},
        "frozen_inputs": input_hashes,
        "population": {
            "selected_manifest": 170, "local_unique": 160, "conv": 60,
            "fc1": 100, "conv_to_atlif_overlap_subset": 40,
            "missing_payloads": 10, "missing_module": "sttmultires_unet.preds.3.conv.0",
            "dual_line_mismatches": 0,
        },
        "n8_independent_recompute": n8,
        "online_original_order": online_summary,
        "m22_acc32_width_audit": {"operator_checks": operator_checks,
                                  "atlif_read_write_checks": atlif_checks, "mismatches": 0},
        "strong_baseline_attack": {
            "finding": "With the same N=8 signed lanes, reduction tree and row-resident Acc32 context, K8-resident and M484 have identical cycle counts in all three categories. M484 adds header/padding traffic.",
            "k1_boundary": "K1-to-K8 ratios are resource-scaling references, not M484 mechanism speedups.",
            "ports_boundary": "Logical row activation/commit ports are matched; no SRAM macro, latency, arbitration, energy or routed-port result exists.",
            "area_boundary": {
                "precedent_only": "M218/M219 FC2 Acc24 service-island, not M484 FC1/Conv Acc32 PPA",
                "m218_k8_logic_only_um2": area["m218_k8_cell_area_um2"],
                "m219_k1_logic_only_um2": area["m219_k1_cell_area_um2"],
                "k8_over_k1_area_percent": area["k8_area_overhead_vs_k1_percent"],
                "macros": 0,
            },
        },
        "quantile_scope": "Pack wait is accepted events in an offline one-live-row oracle, not wall-clock latency.",
        "limitations": [
            "Ten windows from one Zurich sequence only.",
            "Conv online reorder construction/capacity/backpressure is absent; the online row is a one-source safe lower bound.",
            "No M484 RTL, VCS, synthesis, SRAM macro, PTPX, end-to-end cycles, FPS or energy.",
            "The screen finds row occupancy, but no incremental advantage over a fair K8-resident implementation.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False, sort_keys=True))
