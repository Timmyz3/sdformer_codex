#!/usr/bin/env python3
"""Heldout DSE for destination-tagged cross-block K4 plus PWP1024.

M143 packs K4 source events independently inside each of eight destination
blocks.  M147 preserves row order and every source event, but lets one
descriptor carry up to four (destination block, source) tuples from the same
raw row.  This removes per-destination tail padding.  A separate DSE axis
doubles the PWP fetch payload from 512 to 1024 bits: signed 8/9/10-bit 96-lane
vectors use one beat and signed 11-bit vectors use two beats.

The recurrence is the independently reviewed four-bank, full-materialization
M143 recurrence.  Four conflict-resolved destination updates per cycle and a
1024-bit PWP source are assumptions, not implemented engine or SRAM evidence.
The output is heldout same-clock module-cycle opportunity only.
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
PATHS = {
    "m143_script": HW / "system_simulator/scripts/analyze_m143_raw128_full_materialized_overlap_dse.py",
    "m143_result": HW / "results/m143r2_raw128_full_materialized_overlap_dse_r1_20260824/m143_raw128_full_materialized_overlap_dse.json",
    "m143_contract": HW / "contracts/m143r2_raw128_full_materialized_overlap_dse_contract_r1_20260824.json",
    "m141_audit": HW / "results/m141r3_independent_hammer_review_r1_20260824/audit_m141r3_independent.py",
    "m141_recompute": HW / "results/m141r3_independent_hammer_review_r1_20260824/independent_recompute_and_attack.json",
    "m141_manifest": HW / "results/m141r3_independent_hammer_review_r1_20260824/immutable_manifest.sha256",
    "m132_script": HW / "system_simulator/scripts/analyze_m132_dualrow512_pwp_compact_k4_schedule.py",
    "m40_manifest": HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json",
    "m72_result": HW / "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/m72_phi_kmeans_k16q16_valid825_internal_screen.json",
    "m41_result": HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/m41_h67_ep35_bottleneck_int8_bridge.json",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
    "m146_vcs_receipt": HW / "dc_handoff/runs/m146_four_bank_age_queue_scheduler_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m146_dc_receipt": HW / "dc_handoff/runs/m146_four_bank_age_queue_scheduler_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m146_dc_parse_overlay": HW / "contracts/m146_r1_dc_receipt_parse_correction_overlay_r1_20260824.json",
}
EXPECTED = {
    "m143_script": "b8a702da04aa551d6bf4fe0e8b80d7fc976704a362a6d0d58fd8877e2d4b10b7",
    "m143_result": "8b5821d747e653ac9053a4cfe94fe9eb40c78ce0eaaca4c9af4fdf8073b5bd19",
    "m143_contract": "288f03c77556c3e9ea26bfeb18e457423e8f8d8c3dfac9bef070769436051413",
    "m141_audit": "19c3f2b07e506e716d1ca6ee3bf60d46d0a30986247b8899064e4981d19b9ff1",
    "m141_recompute": "0be45dbedac89957e110ad06c4608ef041451425aa3a1f37f1b352d34540983b",
    "m141_manifest": "f8354faa7f49a35a578ea66fa82b1e40ac52c83ceb53062158067451abfd7270",
    "m132_script": "f140b6b72559f04cdac374eaf696c3f6650b20d3b00bd580419b88494d89c952",
    "m40_manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m72_result": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m146_vcs_receipt": "1af5335adff4335773a12b7931c5c9844c8ec9d853be9db33f22fae71bbc7858",
    "m146_dc_receipt": "8622e60e4bc910a7b4f69d903cf7fd3b93a49ebddb141555fff6d5d67d6c80c6",
    "m146_dc_parse_overlay": "b7342f44ec462383ffc807f2ec5c667ff884c30842252f38603216951a6178fe",
}

ROWS = 3000
PARTITIONS = 432
OUTPUT_BLOCKS = 8
WINDOW_ROWS = 384
BANKS = 4
FOLD = 4


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


def pwp1024_rows(m105, masks, operator, centers, widths, popcount,
                 pwp512_rows):
    """Derive exact 1024-bit beats and independently re-evaluate eligibility."""
    eligible_uses = np.zeros((PARTITIONS, ROWS), dtype=np.uint8)
    width11_uses = 0
    for partition in range(PARTITIONS):
        values = masks[partition]
        center_values = centers[operator, partition]
        order = np.argsort(center_values, kind="stable")
        ordered_centers = center_values[order]
        distances = popcount[np.bitwise_xor(
            values[:, None], ordered_centers[None, :])]
        ordered_choice = distances.argmin(axis=1)
        best_index = order[ordered_choice]
        best_distance = distances[np.arange(ROWS), ordered_choice]
        population = popcount[values]
        beneficial = (1 + best_distance) < population
        selected_widths = widths[operator, partition, best_index]
        eligible = beneficial[:, None] & (selected_widths <= m105.CAP)
        eligible_uses[partition] = eligible.sum(axis=1, dtype=np.uint8)
        width11_uses += int((eligible & (selected_widths == 11)).sum())
    # PWP512 is 2 beats per eligible width-8/9/10 vector and 3 for width-11.
    # PWP1024 subtracts exactly one beat from every eligible vector.
    result = pwp512_rows - eligible_uses.astype(np.uint16)
    return result, int(eligible_uses.sum()), width11_uses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M147 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M147 frozen input identity drift")

    m143 = load_module("m147_frozen_m143", PATHS["m143_script"])
    audit = load_module("m147_frozen_audit", PATHS["m141_audit"])
    m132 = load_module("m147_frozen_m132", PATHS["m132_script"])
    m105 = load_module("m147_frozen_m105", m132.M105_SCRIPT)
    manifest = audit.strict_json(PATHS["m40_manifest"])
    m72 = audit.strict_json(PATHS["m72_result"])
    m41 = audit.strict_json(PATHS["m41_result"])
    m143_result = audit.strict_json(PATHS["m143_result"])
    m143_contract = audit.strict_json(PATHS["m143_contract"])
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "heldout record extent drift")

    popcount = np.fromiter(
        (bin(value).count("1") for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    starts = np.arange(0, ROWS, WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + WINDOW_ROWS, ROWS)
    configs = {
        (packing, port): audit.IndependentOverlap(
            BANKS, wait_full_descriptor=True, safe_zero_release=True)
        for packing in ("block_k4", "mosaic_k4")
        for port in (512, 1024)
    }
    totals = Counter()

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, _, pwp512, _ = m132.build_record_rows(
            m105, masks, record["operator_index"], centers, widths,
            popcount)
        pwp1024, eligible_uses, width11_uses = pwp1024_rows(
            m105, masks, record["operator_index"], centers, widths,
            popcount, pwp512)
        totals["eligible_pwp_vectors"] += eligible_uses
        totals["eligible_width11_vectors"] += width11_uses
        counts = popcount[event_masks]
        block_k4 = ((counts.astype(np.uint16) + FOLD - 1) // FOLD).sum(
            axis=2, dtype=np.uint16)
        mosaic_k4 = ((counts.sum(axis=2, dtype=np.uint16) + FOLD - 1)
                     // FOLD)
        require(np.all(mosaic_k4 <= block_k4),
                "mosaic packing increased a row")
        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)
        prefixes = {}
        for packing, descriptors in (("block_k4", block_k4),
                                     ("mosaic_k4", mosaic_k4)):
            row_cycles = np.maximum(descriptors, 1)
            prefixes[(packing, "descriptor")] = np.concatenate((
                np.zeros((PARTITIONS, 1), dtype=np.uint32),
                np.cumsum(descriptors, axis=1, dtype=np.uint32)), axis=1)
            prefixes[(packing, "row")] = np.concatenate((
                np.zeros((PARTITIONS, 1), dtype=np.uint32),
                np.cumsum(row_cycles, axis=1, dtype=np.uint32)), axis=1)
            totals[packing + "_descriptors"] += int(descriptors.sum())
            totals[packing + "_producer_cycles"] += int(row_cycles.sum())
        for port, pwp in ((512, pwp512), (1024, pwp1024)):
            prefixes[(port, "pwp")] = np.concatenate((
                np.zeros((PARTITIONS, 1), dtype=np.uint32),
                np.cumsum(pwp, axis=1, dtype=np.uint32)), axis=1)
            totals["pwp{}_tokens".format(port)] += int(pwp.sum())
        totals["source_events"] += int(counts.sum())
        totals["raw_rows"] += PARTITIONS * ROWS

        for window, (start, end) in enumerate(zip(starts, ends)):
            for packing in ("block_k4", "mosaic_k4"):
                descriptors = (prefixes[(packing, "descriptor")][:, end]
                               - prefixes[(packing, "descriptor")][:, start])
                producer = (prefixes[(packing, "row")][:, end]
                            - prefixes[(packing, "row")][:, start])
                for port in (512, 1024):
                    pwp = (prefixes[(port, "pwp")][:, end]
                           - prefixes[(port, "pwp")][:, start])
                    schedule = configs[(packing, port)]
                    for partition in range(PARTITIONS):
                        descriptor_count = int(descriptors[partition])
                        schedule.add(
                            record_index, window, partition,
                            int(producer[partition]),
                            int(groups[partition, window]),
                            int(pwp[partition]),
                            descriptor_count + int(descriptor_count != 0))
        print("[M147 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    expected_totals = {
        "raw_rows": 25920000,
        "source_events": 188148490,
        "block_k4_descriptors": 99847888,
        "block_k4_producer_cycles": 113925993,
        "mosaic_k4_descriptors": 47037211,
        "mosaic_k4_producer_cycles": 61115316,
        "pwp512_tokens": 119447791,
        "pwp1024_tokens": 60478417,
        "eligible_pwp_vectors": 58969374,
        "eligible_width11_vectors": 1509043,
    }
    require(dict(totals) == expected_totals, "M147 exact totals drift")

    m109 = audit.strict_json(audit.M109_RESULT)
    w384 = next(row for row in m109["frontier"]
                if int(row["window_rows"]) == WINDOW_ROWS)
    fixed_service = (
        int(w384["dual_timeline_recurrence"]["fair_fixed8_baseline_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_commit_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_pipeline_flush_cycles"]))
    results = {}
    for (packing, port), schedule in configs.items():
        results["{}_pwp{}".format(packing, port)] = schedule.result(
            fixed_service)
    expected_cycles = {
        "block_k4_pwp512": 135461009,
        "block_k4_pwp1024": 126581635,
        "mosaic_k4_pwp512": 122267417,
        "mosaic_k4_pwp1024": 75029590,
    }
    require({key: int(value["candidate_cycles"])
             for key, value in results.items()} == expected_cycles,
            "M147 cycle recurrence drift")
    base = results["block_k4_pwp512"]
    require(base["candidate_cycles"] == m143_result["raw128_cycle_models"]
            ["b4"]["candidate_cycles"], "M143 B4 replay mismatch")
    require(base["pwp_service_tokens"] == 119447791
            and base["correction_service_tokens"] == 124730596,
            "M143 consumer-work replay mismatch")

    compact = int(m143_contract["cycle_results"]
                  ["m132_compact256_serial_cycles"])
    dualrow = int(m143_contract["cycle_results"]
                  ["m132_dualrow512_serial_cycles"])
    candidate = results["mosaic_k4_pwp1024"]["candidate_cycles"]
    comparisons = {
        "candidate_ratio_vs_m143r2_b4":
            base["candidate_cycles"] / candidate,
        "candidate_ratio_vs_m132_compact256": compact / candidate,
        "candidate_ratio_vs_m132_dualrow512": dualrow / candidate,
        "candidate_ratio_vs_fair_fixed8":
            results["mosaic_k4_pwp1024"]
            ["same_clock_service_island_ratio"],
        "descriptor_reduction_fraction_vs_block_k4":
            1.0 - (totals["mosaic_k4_descriptors"]
                   / totals["block_k4_descriptors"]),
        "pwp_token_reduction_fraction_vs_pwp512":
            1.0 - (totals["pwp1024_tokens"]
                   / totals["pwp512_tokens"]),
    }

    payload = {
        "schema": "m147_destination_tagged_mosaic_k4_pwp1024_dse_v1",
        "status": "PASS_HELDOUT_SAME_CLOCK_OPPORTUNITY",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": observed,
        },
        "extent": {
            "lineage": "H67/Motion ep35 heldout sample IDs 5..9",
            "records": 20,
            "partitions_per_record": PARTITIONS,
            "rows_per_partition": ROWS,
            "windows_per_record": len(starts),
            "banks": BANKS,
        },
        "exact_work": dict(totals),
        "cycle_models": results,
        "comparisons": comparisons,
        "architecture_contract": {
            "row_order_preserved": True,
            "source_event_multiset_preserved": True,
            "stable_tuple_order": "destination block then increasing source",
            "tuples_per_descriptor": 4,
            "destination_tag_bits_per_tuple": 3,
            "destination_updates_per_cycle_assumed": 4,
            "same_destination_combining_or_independent_banking_required": True,
            "pwp_payload_bits_per_cycle_assumed": 1024,
            "pwp_beats_by_signed_width": {"8": 1, "9": 1,
                                           "10": 1, "11": 2},
        },
        "model_boundary": {
            "m143r2_b4_replayed_exactly": True,
            "m146_age_queue_vcs_and_dc_present": True,
            "m146_drop_in_equivalence": False,
            "mosaic_packer_rtl": False,
            "four_destination_update_engine_rtl": False,
            "pwp1024_sram_macro": False,
            "macro_bandwidth_and_energy": False,
            "matched_frequency": False,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "paper_safe_statement": (
            "On the frozen H67 heldout service-island recurrence, an ideal "
            "destination-tagged cross-block K4 packer plus a 1024-bit PWP "
            "source reduces the B4 opportunity model to {} cycles, or "
            "{:.6f}x versus compact256 and {:.6f}x versus dualrow512. "
            "Conflict-resolved update RTL, SRAM bandwidth/energy, and "
            "matched physical timing remain unimplemented."
        ).format(candidate,
                 comparisons["candidate_ratio_vs_m132_compact256"],
                 comparisons["candidate_ratio_vs_m132_dualrow512"]),
    }
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M147 analyzer changed during execution")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m147_destination_tagged_mosaic_k4_pwp1024_dse.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "PASS M147 cycles={} ratio_vs_m143={:.9f}x "
        "ratio_vs_compact256={:.9f}x ratio_vs_dualrow512={:.9f}x "
        "mosaic_rtl=false pwp1024_macro=false physical_speedup=false "
        "system_speedup=false headline=false".format(
            candidate,
            comparisons["candidate_ratio_vs_m143r2_b4"],
            comparisons["candidate_ratio_vs_m132_compact256"],
            comparisons["candidate_ratio_vs_m132_dualrow512"]),
        flush=True)


if __name__ == "__main__":
    main()
