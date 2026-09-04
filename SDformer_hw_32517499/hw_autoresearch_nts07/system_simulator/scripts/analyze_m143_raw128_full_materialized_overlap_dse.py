#!/usr/bin/env python3
"""Heldout cycle DSE matching the M142 raw-row/full-bank boundary.

Unlike M141r3, PWP cannot start until the complete owning window bank has
materialized.  Unlike M140, the producer accepts every raw 128-bit row (eight
16-bit source masks), including an all-zero row, and serializes its canonical
block-major K1..K4 descriptors.  One row consumes max(1, descriptor_count)
producer cycles.  This remains a module-cycle model, not physical or system
speedup evidence.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
AUDIT_SCRIPT = (HW / "results/m141r3_independent_hammer_review_r1_20260824"
                "/audit_m141r3_independent.py")
AUDIT_RECOMPUTE = (HW / "results/m141r3_independent_hammer_review_r1_20260824"
                   "/independent_recompute_and_attack.json")
AUDIT_MANIFEST = (HW / "results/m141r3_independent_hammer_review_r1_20260824"
                  "/immutable_manifest.sha256")
M141_OVERLAY = (HW / "contracts"
                "/m141r3_independent_review_correction_overlay_r1_20260824.json")
M142_RTL = (HW / "rtl_m142"
            "/m142_sparse_mask_k4_three_bank_overlap_controller.sv")

EXPECTED = {
    "audit_script": "19c3f2b07e506e716d1ca6ee3bf60d46d0a30986247b8899064e4981d19b9ff1",
    "audit_recompute": "0be45dbedac89957e110ad06c4608ef041451425aa3a1f37f1b352d34540983b",
    "audit_manifest": "f8354faa7f49a35a578ea66fa82b1e40ac52c83ceb53062158067451abfd7270",
    "m141_overlay": "309ac23757ed743a7731b018a4a94aec0802af6ba81289f514897102042ce3d3",
    "m142_rtl": "da80d61a4fe95bfd97ea50af388b48d924dcc0466836aa72f3809552d6c1915d",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, path)
    require(spec is not None and spec.loader is not None,
            "cannot import " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M143 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())

    frozen = {
        "audit_script": AUDIT_SCRIPT,
        "audit_recompute": AUDIT_RECOMPUTE,
        "audit_manifest": AUDIT_MANIFEST,
        "m141_overlay": M141_OVERLAY,
        "m142_rtl": M142_RTL,
    }
    observed = {label: sha256(path) for label, path in frozen.items()}
    require(observed == EXPECTED, "M143 frozen input identity drift")

    audit = load_module("m143_frozen_m141_audit", AUDIT_SCRIPT)
    for label, path in {
        "m141_script": audit.M141_SCRIPT,
        "m141_result": audit.M141_RESULT,
        "m141_contract": audit.M141_CONTRACT,
        "m141_r1_correction": audit.M141_R1_CORRECTION,
        "m132_script": audit.M132_SCRIPT,
        "m132_result": audit.M132_RESULT,
        "m109_result": audit.M109_RESULT,
        "docs359": audit.DOCS359,
    }.items():
        require(sha256(path) == audit.EXPECTED[label],
                "M141 review transitive identity drift: " + label)

    m132 = audit.load_module("m143_frozen_m132", audit.M132_SCRIPT)
    m105 = audit.load_module("m143_frozen_m105", m132.M105_SCRIPT)
    manifest = audit.strict_json(m132.M40_MANIFEST)
    m72 = audit.strict_json(m132.M72_RESULT)
    m41 = audit.strict_json(m132.M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "heldout record extent drift")

    m109 = audit.strict_json(audit.M109_RESULT)
    w384 = next(row for row in m109["frontier"]
                if int(row["window_rows"]) == 384)
    fixed_baseline = (
        int(w384["dual_timeline_recurrence"]
            ["fair_fixed8_baseline_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_commit_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_pipeline_flush_cycles"]))

    popcount = np.fromiter(
        (bin(value).count("1") for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    review_reference = {
        banks: audit.IndependentOverlap(
            banks, wait_full_descriptor=True, safe_zero_release=True)
        for banks in (2, 3, 4)
    }
    raw128 = {
        banks: audit.IndependentOverlap(
            banks, wait_full_descriptor=True, safe_zero_release=True)
        for banks in (2, 3, 4)
    }
    totals = Counter()
    starts = np.arange(0, 3000, 384, dtype=np.intp)
    ends = np.minimum(starts + 384, 3000)

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, _, pwp512_rows, _ = m132.build_record_rows(
            m105, masks, record["operator_index"], centers, widths,
            popcount)
        source_counts = popcount[event_masks]
        folded_rows = ((source_counts.astype(np.uint16) + 3) // 4).sum(
            axis=2, dtype=np.uint16)
        raw128_row_cycles = np.maximum(folded_rows, 1)
        folded_prefix = np.concatenate((
            np.zeros((432, 1), dtype=np.uint32),
            np.cumsum(folded_rows, axis=1, dtype=np.uint32)), axis=1)
        raw128_prefix = np.concatenate((
            np.zeros((432, 1), dtype=np.uint32),
            np.cumsum(raw128_row_cycles, axis=1, dtype=np.uint32)), axis=1)
        zero_prefix = np.concatenate((
            np.zeros((432, 1), dtype=np.uint32),
            np.cumsum(folded_rows == 0, axis=1, dtype=np.uint32)), axis=1)
        event_prefix = np.concatenate((
            np.zeros((432, 1), dtype=np.uint32),
            np.cumsum(source_counts.sum(axis=2, dtype=np.uint16),
                      axis=1, dtype=np.uint32)), axis=1)
        pwp_prefix = np.concatenate((
            np.zeros((432, 1), dtype=np.uint32),
            np.cumsum(pwp512_rows, axis=1, dtype=np.uint32)), axis=1)
        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)

        for window, (start, end) in enumerate(zip(starts, ends)):
            folded = folded_prefix[:, end] - folded_prefix[:, start]
            row_cycles = raw128_prefix[:, end] - raw128_prefix[:, start]
            zero_rows = zero_prefix[:, end] - zero_prefix[:, start]
            events = event_prefix[:, end] - event_prefix[:, start]
            pwp = pwp_prefix[:, end] - pwp_prefix[:, start]
            for partition in range(432):
                folded_count = int(folded[partition])
                row_cycle_count = int(row_cycles[partition])
                group_count = int(groups[partition, window])
                pwp_count = int(pwp[partition])
                correction_cycles = folded_count + int(folded_count != 0)
                for schedule in review_reference.values():
                    schedule.add(record_index, window, partition,
                                 folded_count, group_count, pwp_count,
                                 correction_cycles)
                for schedule in raw128.values():
                    schedule.add(record_index, window, partition,
                                 row_cycle_count, group_count, pwp_count,
                                 correction_cycles)
                totals["raw_128bit_rows"] += int(end - start)
                totals["raw_zero_rows"] += int(zero_rows[partition])
                totals["raw128_producer_cycles"] += row_cycle_count
                totals["packed_k4_descriptors"] += folded_count
                totals["source_events"] += int(events[partition])
                totals["pwp512_tokens"] += pwp_count
        print("[M143 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    reference_results = {
        banks: review_reference[banks].result(fixed_baseline)
        for banks in (2, 3, 4)
    }
    expected_reference = {2: 188168131, 3: 144690917, 4: 133991596}
    require({banks: result["candidate_cycles"]
             for banks, result in reference_results.items()}
            == expected_reference,
            "independent full-materialization reference replay drift")
    results = {
        banks: raw128[banks].result(fixed_baseline)
        for banks in (2, 3, 4)
    }
    require(all(result["pwp_service_tokens"] == 119447791
                and result["correction_service_tokens"] == 124730596
                for result in results.values()),
            "M143 consumer-work conservation drift")
    require(totals["raw_128bit_rows"] == 25920000,
            "raw 128-bit row extent drift")
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M143 analyzer changed during execution")

    frozen_m132 = audit.strict_json(audit.M132_RESULT)
    compact256 = int(frozen_m132["cycle_models"]
                     ["compact_k4_pwp256"]["candidate_cycles"])
    dualrow512 = int(frozen_m132["cycle_models"]
                    ["compact_k4_dualrow_pwp512"]["candidate_cycles"])
    fixed8 = int(results[4]["fair_fixed8_baseline_cycles"])
    comparisons = {
        "b3_ratio_vs_compact256": compact256 / results[3]["candidate_cycles"],
        "b3_ratio_vs_dualrow512": dualrow512 / results[3]["candidate_cycles"],
        "b4_ratio_vs_compact256": compact256 / results[4]["candidate_cycles"],
        "b4_ratio_vs_dualrow512": dualrow512 / results[4]["candidate_cycles"],
        "b4_ratio_vs_b3": (results[3]["candidate_cycles"]
                            / results[4]["candidate_cycles"]),
        "b4_same_clock_service_island_ratio_vs_fixed8":
            fixed8 / results[4]["candidate_cycles"],
    }
    payload = {
        "schema": "m143_raw128_full_materialized_overlap_dse_v1",
        "status": "PASS_EXACT_HELDOUT_RAW128_FULL_MATERIALIZATION_DSE",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED,
            "heldout_samples": list(range(5, 10)),
            "heldout_records": len(heldout),
        },
        "exact_work": dict(totals),
        "review_full_materialization_reference_cycles": {
            str(banks): reference_results[banks]["candidate_cycles"]
            for banks in (2, 3, 4)
        },
        "raw128_cycle_models": {
            "b{}".format(banks): results[banks] for banks in (2, 3, 4)
        },
        "comparisons": comparisons,
        "architecture_contract": {
            "input_per_row": "eight raw 16-bit signed source masks",
            "all_zero_rows_accepted": True,
            "row_producer_cycles": "sum(max(1, total canonical K4 descriptors in the 128-bit row))",
            "descriptor_order": "row then block[0..7] then strictly increasing source",
            "pwp_start": "only after complete owning bank materialization",
            "bank_release": "only after matching correction completion",
            "bank_depths_dse": [2, 3, 4],
            "unbounded_fifo": False,
            "consumer_arithmetic_changed": False,
        },
        "selection": {
            "bank_depth_frozen": False,
            "reason": "Controller RTL can support B3/B4, but external descriptor/result capacity and macro energy are not yet priced.",
        },
        "model_boundary": {
            "exact_heldout_raw_row_extent": True,
            "exact_heldout_width_placement": True,
            "m142_rtl_source_present": True,
            "m142_vcs_sealed": False,
            "m142_dc_sealed": False,
            "descriptor_result_sram_macro": False,
            "matched_frequency": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
        "paper_safe_statement": (
            "Raw-128 row ingress and complete-bank materialization are included; "
            "reported ratios are frozen-heldout same-clock module-cycle results, "
            "not physical, full-network, or system speedup."),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m143_raw128_full_materialized_overlap_dse.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "PASS M143 raw_rows={} zero_rows={} producer_cycles={} "
        "b2={} b3={} b4={} b3_vs_compact256={:.9f}x "
        "b4_vs_compact256={:.9f}x b4_vs_dualrow512={:.9f}x "
        "raw128_rtl_source=true sram_macro=false physical_speedup=false "
        "system_speedup=false headline=false".format(
            totals["raw_128bit_rows"], totals["raw_zero_rows"],
            totals["raw128_producer_cycles"],
            results[2]["candidate_cycles"], results[3]["candidate_cycles"],
            results[4]["candidate_cycles"],
            comparisons["b3_ratio_vs_compact256"],
            comparisons["b4_ratio_vs_compact256"],
            comparisons["b4_ratio_vs_dualrow512"]), flush=True)


if __name__ == "__main__":
    main()
