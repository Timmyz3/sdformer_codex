#!/usr/bin/env python3
"""Heldout DSE for source-stationary, up-to-four-destination broadcast.

M147 permits arbitrary (destination, source) tuples in one descriptor.  That
minimizes descriptor count, but a full descriptor may require four unrelated
96-lane contribution vectors (up to 3072 payload bits even at INT8).  M150
transposes the local 8x16 event matrix by source: one descriptor owns exactly
one source and up to four distinct destinations.  A single resident/PWP source
vector can therefore feed all four updates by multicast.

This script evaluates only the frozen heldout cycle opportunity.  Four-bank
destination writeback, a four-destination broadcaster, resident source-vector
storage, and SRAM bandwidth/energy are not implemented here.
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
M147_SCRIPT = (HW / "system_simulator/scripts/"
               "analyze_m147_destination_tagged_mosaic_k4_pwp1024_dse.py")
EXTRA_PATHS = {
    "m147_script": M147_SCRIPT,
    "m147_result": HW / "results/m147_destination_tagged_mosaic_k4_pwp1024_dse_r1_20260824/m147_destination_tagged_mosaic_k4_pwp1024_dse.json",
    "m147_review_manifest": HW / "results/m147_independent_hammer_review_r1_20260824/immutable_manifest.sha256",
    "m147_correction_overlay": HW / "contracts/m147_independent_review_correction_overlay_r1_20260824.json",
    "m104_r3_vcs_receipt": HW / "dc_handoff/runs/m104_r3_accepted_last_event_grace_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m149_vcs_receipt": HW / "dc_handoff/runs/m149_destination_conflict_resolved_k4_combiner_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m149_dc_receipt": HW / "dc_handoff/runs/m149_destination_conflict_resolved_k4_combiner_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXTRA_EXPECTED = {
    "m147_script": "9e54e13dec3bfa765eb3c49b3b11ce0577e04543ddb2a2561e82cc0e028a551c",
    "m147_result": "825a2f8d8aadd729bf6b547b8e9144bac09392358d1646ecc4b94392f1628220",
    "m147_review_manifest": "ea789dcbd26be622ef68ca900fa367812711bc46daef3fd46b6da713f373f3ce",
    "m147_correction_overlay": "8a6ba2e9dce906378708a9ecc1cbc71b86153d7d011e1cb9cf8ff5718fa4c9af",
    "m104_r3_vcs_receipt": "385c38d3982c1d41591eee2a67e27e1c6502557050f480a21f34bc71cf4603a2",
    "m149_vcs_receipt": "ac77e562b015a7abae339a064181c78b1493876aa11bb0fdcd652eff3d6a2b27",
    "m149_dc_receipt": "cf552d8389ab98d8d523e5753e8c0e30988d9605d7e26ad933a97783766c643d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
    require(not args.output.exists(), "refusing M150 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    observed_extra = {label: sha256(path)
                      for label, path in EXTRA_PATHS.items()}
    require(observed_extra == EXTRA_EXPECTED,
            "M150 frozen evidence identity drift")

    m147 = load_module("m150_frozen_m147", M147_SCRIPT)
    observed_m147_inputs = {
        label: sha256(path) for label, path in m147.PATHS.items()}
    require(observed_m147_inputs == m147.EXPECTED,
            "M150 inherited M147 input drift")
    audit = m147.load_module("m150_frozen_audit",
                             m147.PATHS["m141_audit"])
    m132 = m147.load_module("m150_frozen_m132",
                            m147.PATHS["m132_script"])
    m105 = m147.load_module("m150_frozen_m105", m132.M105_SCRIPT)

    manifest = audit.strict_json(m147.PATHS["m40_manifest"])
    m72 = audit.strict_json(m147.PATHS["m72_result"])
    m41 = audit.strict_json(m147.PATHS["m41_result"])
    m143_contract = audit.strict_json(m147.PATHS["m143_contract"])
    m147_result = audit.strict_json(EXTRA_PATHS["m147_result"])
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "M150 heldout extent drift")

    popcount = np.fromiter(
        (bin(value).count("1") for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    starts = np.arange(0, m147.ROWS, m147.WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + m147.WINDOW_ROWS, m147.ROWS)
    configs = {
        port: audit.IndependentOverlap(
            m147.BANKS, wait_full_descriptor=True, safe_zero_release=True)
        for port in (512, 1024)
    }
    totals = Counter()

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, _, pwp512, _ = m132.build_record_rows(
            m105, masks, record["operator_index"], centers, widths,
            popcount)
        pwp1024, eligible_uses, width11_uses = m147.pwp1024_rows(
            m105, masks, record["operator_index"], centers, widths,
            popcount, pwp512)
        totals["eligible_pwp_vectors"] += eligible_uses
        totals["eligible_width11_vectors"] += width11_uses

        counts = popcount[event_masks]
        block_k4 = ((counts.astype(np.uint16) + m147.FOLD - 1)
                    // m147.FOLD).sum(axis=2, dtype=np.uint16)
        mosaic_k4 = ((counts.sum(axis=2, dtype=np.uint16)
                      + m147.FOLD - 1) // m147.FOLD)
        source_k4 = np.zeros((m147.PARTITIONS, m147.ROWS),
                             dtype=np.uint16)
        active_source_rows = np.zeros_like(source_k4)
        source_full4_descriptors = 0
        for source in range(16):
            destination_population = np.sum(
                (event_masks >> source) & 1, axis=2, dtype=np.uint16)
            source_k4 += ((destination_population + m147.FOLD - 1)
                          // m147.FOLD)
            active_source_rows += (destination_population != 0)
            source_full4_descriptors += int(
                (destination_population // m147.FOLD).sum())
        require(np.all(mosaic_k4 <= source_k4),
                "source-stationary packing beat unrestricted mosaic")
        totals["raw_rows"] += m147.PARTITIONS * m147.ROWS
        totals["source_events"] += int(counts.sum())
        totals["block_k4_descriptors"] += int(block_k4.sum())
        totals["mosaic_k4_descriptors"] += int(mosaic_k4.sum())
        totals["source_k4_descriptors"] += int(source_k4.sum())
        totals["source_active_keys"] += int(active_source_rows.sum())
        totals["source_full4_descriptors"] += source_full4_descriptors
        source_producer = np.maximum(source_k4, 1)
        totals["source_k4_producer_cycles"] += int(source_producer.sum())
        totals["pwp512_tokens"] += int(pwp512.sum())
        totals["pwp1024_tokens"] += int(pwp1024.sum())

        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)
        descriptor_prefix = np.concatenate((
            np.zeros((m147.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(source_k4, axis=1, dtype=np.uint32)), axis=1)
        producer_prefix = np.concatenate((
            np.zeros((m147.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(source_producer, axis=1, dtype=np.uint32)), axis=1)
        pwp_prefixes = {}
        for port, pwp in ((512, pwp512), (1024, pwp1024)):
            pwp_prefixes[port] = np.concatenate((
                np.zeros((m147.PARTITIONS, 1), dtype=np.uint32),
                np.cumsum(pwp, axis=1, dtype=np.uint32)), axis=1)

        for window, (start, end) in enumerate(zip(starts, ends)):
            descriptors = (descriptor_prefix[:, end]
                           - descriptor_prefix[:, start])
            producer = producer_prefix[:, end] - producer_prefix[:, start]
            for port in (512, 1024):
                pwp = (pwp_prefixes[port][:, end]
                       - pwp_prefixes[port][:, start])
                schedule = configs[port]
                for partition in range(m147.PARTITIONS):
                    descriptor_count = int(descriptors[partition])
                    schedule.add(
                        record_index, window, partition,
                        int(producer[partition]),
                        int(groups[partition, window]),
                        int(pwp[partition]),
                        descriptor_count + int(descriptor_count != 0))
        print("[M150 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    m109 = audit.strict_json(audit.M109_RESULT)
    w384 = next(row for row in m109["frontier"]
                if int(row["window_rows"]) == m147.WINDOW_ROWS)
    fixed_service = (
        int(w384["dual_timeline_recurrence"]
            ["fair_fixed8_baseline_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_commit_cycles"])
        - int(w384["dual_timeline_recurrence"]
              ["accumulator_pipeline_flush_cycles"]))
    results = {"source_k4_pwp{}".format(port): schedule.result(fixed_service)
               for port, schedule in configs.items()}
    m143_cycles = int(m147_result["cycle_models"]
                      ["block_k4_pwp512"]["candidate_cycles"])
    unrestricted_cycles = int(m147_result["cycle_models"]
                              ["mosaic_k4_pwp1024"]["candidate_cycles"])
    compact_cycles = int(m143_contract["cycle_results"]
                         ["m132_compact256_serial_cycles"])
    dualrow_cycles = int(m143_contract["cycle_results"]
                         ["m132_dualrow512_serial_cycles"])
    candidate = int(results["source_k4_pwp1024"]["candidate_cycles"])
    comparisons = {
        "ratio_vs_m143r2_b4": m143_cycles / candidate,
        "ratio_vs_m132_compact256": compact_cycles / candidate,
        "ratio_vs_m132_dualrow512": dualrow_cycles / candidate,
        "ratio_vs_unrestricted_m147_mosaic_pwp1024":
            unrestricted_cycles / candidate,
        "descriptor_reduction_fraction_vs_block_k4":
            1.0 - totals["source_k4_descriptors"]
            / totals["block_k4_descriptors"],
        "descriptor_overhead_fraction_vs_unrestricted_mosaic":
            totals["source_k4_descriptors"]
            / totals["mosaic_k4_descriptors"] - 1.0,
        "source_key_reuse_destinations_per_active_key":
            totals["source_events"] / totals["source_active_keys"],
    }

    payload = {
        "schema": "m150_source_stationary_destination_k4_pwp_dse_v1",
        "status": "PASS_HELDOUT_SOURCE_STATIONARY_OPPORTUNITY_ONLY",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_extra_inputs_sha256": observed_extra,
            "inherited_m147_inputs_sha256": observed_m147_inputs,
        },
        "extent": {
            "lineage": "H67/Motion ep35 heldout sample IDs 5..9",
            "records": 20,
            "partitions_per_record": m147.PARTITIONS,
            "rows_per_partition": m147.ROWS,
            "windows_per_record": len(starts),
            "destination_blocks": m147.OUTPUT_BLOCKS,
            "sources_per_block": 16,
            "maximum_destinations_per_descriptor": 4,
        },
        "exact_work": dict(totals),
        "cycle_models": results,
        "comparisons": comparisons,
        "architecture_contract": {
            "descriptor_key": "one source within one raw row",
            "destination_ids_per_descriptor": "one_to_four_distinct",
            "source_vectors_required_per_descriptor": 1,
            "maximum_destination_updates_per_cycle_assumed": 4,
            "source_vector_payload_bits_int8": 768,
            "fits_one_pwp1024_beat_at_width8_9_10": True,
            "signed_width11_requires_two_pwp1024_beats": True,
            "source_weight_hold_and_destination_multicast": True,
            "destination_order_preserved_within_source": True,
            "global_tuple_order_transposed": "raw_row_then_source_then_destination",
        },
        "model_boundary": {
            "m104_one_destination_broadcaster_vcs_present": True,
            "four_destination_broadcaster_rtl": False,
            "resident_source_vector_store": False,
            "four_destination_accumulator_write_ports": False,
            "ordered_trace_numeric_replay": False,
            "signed_negate_trace_replay": False,
            "pwp_sram_macro": False,
            "macro_bandwidth_and_energy": False,
            "matched_frequency": False,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M150 analyzer changed during execution")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m150_source_stationary_destination_k4_pwp_dse.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "PASS M150 source_k4_desc={} cycles={} ratio_vs_m143={:.9f}x "
        "ratio_vs_compact={:.9f}x ratio_vs_dualrow={:.9f}x "
        "four_destination_broadcaster_rtl=false pwp_macro=false "
        "physical_speedup=false system_speedup=false headline=false".format(
            totals["source_k4_descriptors"], candidate,
            comparisons["ratio_vs_m143r2_b4"],
            comparisons["ratio_vs_m132_compact256"],
            comparisons["ratio_vs_m132_dualrow512"]),
        flush=True)


if __name__ == "__main__":
    main()
