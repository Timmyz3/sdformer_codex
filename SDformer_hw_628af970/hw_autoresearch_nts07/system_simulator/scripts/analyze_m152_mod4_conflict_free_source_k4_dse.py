#!/usr/bin/env python3
"""Heldout DSE for conflict-free four-bank source-stationary multicast.

M150 assumes that one source-resident descriptor can update as many as four
destinations in one cycle.  This audit makes the accumulator-bank constraint
explicit.  Destination ``d`` maps to bank ``d % 4``; destinations ``d`` and
``d + 4`` therefore cannot share a descriptor.  For each (partition, raw row,
source) key, the exact minimum number of conflict-free descriptors is the
maximum population of its four modulo banks (zero, one, or two).

The resulting descriptor count is inserted into the same independently
reviewed overlap recurrence inherited by M150.  This remains a cycle-model
admission step: accumulator SRAM macros, four read-modify-write lanes and the
ordered signed numeric replay are still downstream cuts.
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
M150_SCRIPT = (HW / "system_simulator/scripts/"
               "analyze_m150_source_stationary_destination_k4_pwp_dse.py")
PATHS = {
    "m150_script": M150_SCRIPT,
    "m150_result": HW / "results/m150_source_stationary_destination_k4_pwp_dse_r1_20260824/m150_source_stationary_destination_k4_pwp_dse.json",
    "m150_contract": HW / "contracts/m150_source_stationary_destination_k4_pwp_dse_contract_r1_20260824.json",
    "m151_vcs_receipt": HW / "dc_handoff/runs/m151_dual_buffer_source_resident_k4_multicast_vcs_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m151_dc_receipt": HW / "dc_handoff/runs/m151_dual_buffer_source_resident_k4_multicast_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt",
    "m151_vcs_contract": HW / "contracts/m151_dual_buffer_source_resident_k4_multicast_vcs_contract_r1_20260824.json",
    "m151_dc_contract": HW / "contracts/m151_dual_buffer_source_resident_k4_multicast_logic_only_dc_contract_r1_20260824.json",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "m150_script": "3904e5b01e1c3ec8c119f89e8cd4dedfba36b8e2478ba82ce7e074c2eefd1e07",
    "m150_result": "b9d7097c40fffa3ca075a38b701815f4cb49432ec7216adf98cebbf1bd60a027",
    "m150_contract": "a85fd76796a07326dccbb484428d01127121719f44d78e6608c7cb891bb52237",
    "m151_vcs_receipt": "a8d51c29d7db6f819241287e667c0eda5b77184791f96e3d75c0e5f1c4b79b75",
    "m151_dc_receipt": "e35d5b7877dc6a91305c563859af0ec5942bd542b7d4a959ff7417cd9d744049",
    "m151_vcs_contract": "3e5f40ca35930a759d0ca82031b27aceb9f10a9fb20e18a13f640686b25da6a6",
    "m151_dc_contract": "06c01f87a9df80fc7a0fc8efd4608788da701d6d751e992284c2018febf67d0e",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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
    require(not args.output.exists(), "refusing M152 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M152 frozen input identity drift")

    m150 = load_module("m152_frozen_m150", M150_SCRIPT)
    m147 = load_module("m152_frozen_m147", m150.M147_SCRIPT)
    observed_m147 = {label: sha256(path)
                     for label, path in m147.PATHS.items()}
    require(observed_m147 == m147.EXPECTED,
            "M152 inherited M147 input identity drift")
    audit = m147.load_module("m152_frozen_audit", m147.PATHS["m141_audit"])
    m132 = m147.load_module("m152_frozen_m132", m147.PATHS["m132_script"])
    m105 = m147.load_module("m152_frozen_m105", m132.M105_SCRIPT)

    manifest = audit.strict_json(m147.PATHS["m40_manifest"])
    m72 = audit.strict_json(m147.PATHS["m72_result"])
    m41 = audit.strict_json(m147.PATHS["m41_result"])
    m150_result = audit.strict_json(PATHS["m150_result"])
    m147_result = audit.strict_json(m150.EXTRA_PATHS["m147_result"])
    m143_contract = audit.strict_json(m147.PATHS["m143_contract"])
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "M152 heldout extent drift")

    popcount = np.fromiter(
        (bin(value).count("1") for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    starts = np.arange(0, m147.ROWS, m147.WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + m147.WINDOW_ROWS, m147.ROWS)
    schedule = audit.IndependentOverlap(
        m147.BANKS, wait_full_descriptor=True, safe_zero_release=True)
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
        banked_descriptors = np.zeros((m147.PARTITIONS, m147.ROWS),
                                      dtype=np.uint16)
        ideal_descriptors = np.zeros_like(banked_descriptors)
        active_keys = np.zeros_like(banked_descriptors)
        for source in range(16):
            destination_active = ((event_masks >> source) & 1).astype(
                np.uint8, copy=False)
            destination_population = destination_active.sum(
                axis=2, dtype=np.uint16)
            ideal = ((destination_population + m147.FOLD - 1)
                     // m147.FOLD)
            bank_population = np.stack(
                (destination_active[:, :, 0]
                 + destination_active[:, :, 4],
                 destination_active[:, :, 1]
                 + destination_active[:, :, 5],
                 destination_active[:, :, 2]
                 + destination_active[:, :, 6],
                 destination_active[:, :, 3]
                 + destination_active[:, :, 7]), axis=2)
            banked = bank_population.max(axis=2).astype(np.uint16)
            require(np.all(banked >= ideal),
                    "bank-constrained descriptors beat ideal K4")
            require(np.all(banked <= 2), "modulo-four bank bound drift")
            banked_descriptors += banked
            ideal_descriptors += ideal
            active_keys += (destination_population != 0)
            conflict = banked > ideal
            totals["conflict_keys"] += int(conflict.sum())
            totals["extra_conflict_descriptors"] += int(
                (banked - ideal).sum())
            totals["one_cycle_source_keys"] += int((banked == 1).sum())
            totals["two_cycle_source_keys"] += int((banked == 2).sum())

        producer_cycles = np.maximum(banked_descriptors, 1)
        totals["raw_rows"] += m147.PARTITIONS * m147.ROWS
        totals["source_events"] += int(counts.sum())
        totals["source_active_keys"] += int(active_keys.sum())
        totals["ideal_source_k4_descriptors"] += int(
            ideal_descriptors.sum())
        totals["bank_conflict_free_descriptors"] += int(
            banked_descriptors.sum())
        totals["bank_conflict_free_producer_cycles"] += int(
            producer_cycles.sum())
        totals["pwp1024_tokens"] += int(pwp1024.sum())

        union = np.bitwise_or.reduceat(event_masks, starts, axis=1)
        groups = popcount[union].sum(axis=2, dtype=np.uint16)
        descriptor_prefix = np.concatenate((
            np.zeros((m147.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(banked_descriptors, axis=1, dtype=np.uint32)), axis=1)
        producer_prefix = np.concatenate((
            np.zeros((m147.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(producer_cycles, axis=1, dtype=np.uint32)), axis=1)
        pwp_prefix = np.concatenate((
            np.zeros((m147.PARTITIONS, 1), dtype=np.uint32),
            np.cumsum(pwp1024, axis=1, dtype=np.uint32)), axis=1)

        for window, (start, end) in enumerate(zip(starts, ends)):
            descriptors = descriptor_prefix[:, end] - descriptor_prefix[:, start]
            producer = producer_prefix[:, end] - producer_prefix[:, start]
            pwp = pwp_prefix[:, end] - pwp_prefix[:, start]
            for partition in range(m147.PARTITIONS):
                descriptor_count = int(descriptors[partition])
                schedule.add(
                    record_index, window, partition,
                    int(producer[partition]), int(groups[partition, window]),
                    int(pwp[partition]),
                    descriptor_count + int(descriptor_count != 0))
        print("[M152 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"]), flush=True)

    require(totals["ideal_source_k4_descriptors"]
            == m150_result["exact_work"]["source_k4_descriptors"],
            "independent ideal descriptor recount disagrees with M150")
    require(totals["source_events"]
            == m150_result["exact_work"]["source_events"],
            "source-event conservation disagrees with M150")

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
    result = schedule.result(fixed_service)
    candidate = int(result["candidate_cycles"])
    m150_cycles = int(m150_result["cycle_models"]
                      ["source_k4_pwp1024"]["candidate_cycles"])
    m143_cycles = int(m147_result["cycle_models"]
                      ["block_k4_pwp512"]["candidate_cycles"])
    compact_cycles = int(m143_contract["cycle_results"]
                         ["m132_compact256_serial_cycles"])
    dualrow_cycles = int(m143_contract["cycle_results"]
                         ["m132_dualrow512_serial_cycles"])
    comparisons = {
        "ratio_vs_m143r2_b4": m143_cycles / candidate,
        "ratio_vs_m132_compact256": compact_cycles / candidate,
        "ratio_vs_m132_dualrow512": dualrow_cycles / candidate,
        "ratio_vs_ideal_m150_source_k4": m150_cycles / candidate,
        "cycle_overhead_fraction_vs_ideal_m150": candidate / m150_cycles - 1.0,
        "descriptor_overhead_fraction_vs_ideal_m150":
            totals["bank_conflict_free_descriptors"]
            / totals["ideal_source_k4_descriptors"] - 1.0,
        "conflict_key_fraction_of_active_source_keys":
            totals["conflict_keys"] / totals["source_active_keys"],
    }
    payload = {
        "schema": "m152_mod4_conflict_free_source_k4_dse_v1",
        "status": "PASS_HELDOUT_BANK_CONFLICT_CYCLE_MODEL_ONLY",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_direct_inputs_sha256": observed,
            "inherited_m147_inputs_sha256": observed_m147,
        },
        "extent": {
            "lineage": "H67/Motion ep35 heldout sample IDs 5..9",
            "records": 20,
            "partitions_per_record": m147.PARTITIONS,
            "rows_per_partition": m147.ROWS,
            "windows_per_record": len(starts),
            "destination_blocks": 8,
            "accumulator_banks": 4,
        },
        "exact_work": dict(totals),
        "cycle_model": result,
        "comparisons": comparisons,
        "banking_contract": {
            "bank_function": "destination_id_modulo_4",
            "destinations_per_bank": 2,
            "maximum_one_cycle_updates": 4,
            "descriptor_round_zero": "at_most_one_lowest_destination_per_bank",
            "descriptor_round_one": "remaining_destination_per_bank",
            "minimum_cycles_per_source_key": "maximum_population_of_four_banks",
            "source_vector_read_once_per_descriptor": True,
            "source_vector_reused_across_two_descriptors": True,
            "bank_conflicts_serialized_without_dropping_updates": True,
        },
        "model_boundary": {
            "m151_resident_multicast_vcs_and_logic_only_dc": True,
            "four_bank_accumulator_update_rtl": False,
            "accumulator_sram_macro": False,
            "same_address_forwarding": False,
            "ordered_signed_numeric_trace_replay": False,
            "pwp1024_sram_macro": False,
            "macro_bandwidth_and_energy": False,
            "matched_frequency": False,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M152 analyzer changed during execution")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "m152_mod4_conflict_free_source_k4_dse.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "PASS M152 banked_desc={} extra_desc={} conflict_keys={} cycles={} "
        "ratio_vs_m143={:.9f}x ratio_vs_m150={:.9f}x "
        "accumulator_rtl=false macro=false physical_speedup=false "
        "system_speedup=false headline=false".format(
            totals["bank_conflict_free_descriptors"],
            totals["extra_conflict_descriptors"], totals["conflict_keys"],
            candidate, comparisons["ratio_vs_m143r2_b4"],
            comparisons["ratio_vs_ideal_m150_source_k4"]), flush=True)


if __name__ == "__main__":
    main()
