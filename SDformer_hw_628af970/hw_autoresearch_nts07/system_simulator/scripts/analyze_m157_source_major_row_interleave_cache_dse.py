#!/usr/bin/env python3
"""Exact heldout DSE for source-major weight reuse and RMW-safe row interleave.

For one (window, partition), descriptors are transposed to
``source -> destination half -> active row``.  Four destination banks cache
one low or high 96-lane INT8 vector each.  Rows are interleaved so consecutive
updates do not target the same accumulator bank/address.  This audit counts
the exact descriptor, cache-load and one-cycle RMW hazard populations; it does
not admit the cache, accumulator macro, integrated RTL or physical speedup.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M152_SCRIPT = HW / (
    "system_simulator/scripts/analyze_m152_mod4_conflict_free_source_k4_dse.py")
PATHS = {
    "m152_script": M152_SCRIPT,
    "m152_result": HW / (
        "results/m152_mod4_conflict_free_source_k4_dse_r1_20260824/"
        "m152_mod4_conflict_free_source_k4_dse.json"),
    "m152_contract": HW / (
        "contracts/m152_mod4_conflict_free_source_k4_dse_contract_r1_20260824.json"),
    "correction_overlay": HW / (
        "contracts/m150_m151_m152_cross_destination_vector_identity_correction_overlay_r1_20260824.json"),
    "m154_review_manifest": HW / (
        "results/m154_independent_hammer_review_r1_20260824/manifest.sha256"),
    "m154_vcs_receipt": HW / (
        "dc_handoff/runs/m154_four_bank_destination_vector_supplier_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"),
    "m154_dc_receipt": HW / (
        "dc_handoff/runs/m154_four_bank_destination_vector_supplier_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"),
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "m152_script": "5449a34eecea789166a24e75589d0d35d20ec5c5b29225fba6e80c27283c4b9f",
    "m152_result": "4699238e1d0a75bcabf7fa0fbd8d46a5f66ce53de16dba0843d85c24b549702d",
    "m152_contract": "ad1e29ebac603daee2c881e3eee5ff100eb054e935305472e0fac79fd3db9625",
    "correction_overlay": "9f913b451a80d9938a199c1ce648f45b3bd064641f2c47213c9092bb252a6c99",
    "m154_review_manifest": "b4dc68e2aa5ec7f172dbdfe2cf3d5eab96eef59d84587ba9061e328d090baae9",
    "m154_vcs_receipt": "49f737657408b1cd55a9a1ad2106081229f0c16b544304ea560e5898648f6c59",
    "m154_dc_receipt": "4e3e349c7adc7cc7e30470d5e9cd4da4cd714c09f1906751bd2655746e6242a9",
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


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def bank_mask(destination_mask, half):
    selected = (destination_mask >> (4 * half)) & 0x0f
    return int(selected)


def update_boundary(previous, phase, rows, bank_masks, optimize):
    """Return (new_previous, bubble, natural_hazard) for one nonempty phase."""
    require(len(rows) == len(bank_masks) and len(rows) > 0,
            "empty/misaligned phase")
    first = 0
    natural_hazard = 0
    if previous is not None:
        prev_phase, prev_row, prev_banks = previous
        natural_hazard = int(
            prev_phase == phase and prev_row == int(rows[0])
            and bool(prev_banks & int(bank_masks[0])))
    if optimize and natural_hazard and len(rows) > 1:
        first = next(index for index, row in enumerate(rows)
                     if int(row) != previous[1])

    first_row = int(rows[first])
    first_banks = int(bank_masks[first])
    bubble = 0
    if previous is not None:
        prev_phase, prev_row, prev_banks = previous
        bubble = int(prev_phase == phase and prev_row == first_row
                     and bool(prev_banks & first_banks))

    # Any permutation of distinct rows is internally RMW-safe.  Keep the
    # natural last row unless it was selected first; then choose another row.
    last = len(rows) - 1
    if len(rows) > 1 and last == first:
        last = len(rows) - 2
    return (phase, int(rows[last]), int(bank_masks[last])), bubble, natural_hazard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "refusing to overwrite M157 output")
    script_start = sha256(Path(__file__).resolve())
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M157 frozen input identity drift")

    m152 = load_module("m157_frozen_m152", M152_SCRIPT)
    m150 = m152.load_module("m157_frozen_m150", m152.M150_SCRIPT)
    m147 = m152.load_module("m157_frozen_m147", m150.M147_SCRIPT)
    observed_m147 = {label: sha256(path)
                     for label, path in m147.PATHS.items()}
    require(observed_m147 == m147.EXPECTED,
            "M157 inherited M147 input identity drift")
    audit = m147.load_module("m157_frozen_audit",
                             m147.PATHS["m141_audit"])
    m132 = m147.load_module("m157_frozen_m132", m147.PATHS["m132_script"])
    m105 = m147.load_module("m157_frozen_m105", m132.M105_SCRIPT)

    manifest = audit.strict_json(m147.PATHS["m40_manifest"])
    m72 = audit.strict_json(m147.PATHS["m72_result"])
    m41 = audit.strict_json(m147.PATHS["m41_result"])
    m152_result = strict_json(PATHS["m152_result"])
    heldout = sorted(
        (row for row in manifest["records"]
         if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]))
    require(len(heldout) == 20, "M157 heldout extent drift")

    popcount = np.fromiter((bin(value).count("1")
                            for value in range(1 << 16)),
                           dtype=np.uint8, count=1 << 16)
    popcount8 = np.fromiter((bin(value).count("1")
                             for value in range(1 << 8)),
                            dtype=np.uint8, count=1 << 8)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)
    starts = np.arange(0, m147.ROWS, m147.WINDOW_ROWS, dtype=np.intp)
    ends = np.minimum(starts + m147.WINDOW_ROWS, m147.ROWS)
    totals = Counter()
    per_record = []

    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, _, _, _ = m132.build_record_rows(
            m105, masks, record["operator_index"], centers, widths,
            popcount)
        record_totals = Counter()

        for partition in range(m147.PARTITIONS):
            for window, (start, end) in enumerate(zip(starts, ends)):
                previous_natural = None
                previous_optimized = None
                context_descriptors = 0
                context_phases = 0
                context_groups = 0
                context_active_sources = 0
                for source in range(16):
                    active = ((event_masks[partition, start:end, :] >> source)
                              & 1).astype(np.uint8, copy=False)
                    destination_masks = np.zeros(end - start, dtype=np.uint8)
                    for destination in range(8):
                        destination_masks |= (
                            active[:, destination] << destination)
                    union = int(np.bitwise_or.reduce(
                        destination_masks, initial=np.uint8(0)))
                    if union:
                        context_active_sources += 1
                        context_groups += int(popcount8[union])
                    record_totals["source_active_keys"] += int(
                        np.count_nonzero(destination_masks))
                    record_totals["source_events"] += int(
                        popcount8[destination_masks].sum())

                    for phase in (0, 1):
                        phase_masks = ((destination_masks >> (4 * phase))
                                       & np.uint8(0x0f))
                        indices = np.flatnonzero(phase_masks)
                        if len(indices) == 0:
                            continue
                        rows = indices.astype(np.int16, copy=False)
                        banks = phase_masks[indices]
                        context_phases += 1
                        context_descriptors += len(rows)
                        record_totals["single_descriptor_phases"] += int(
                            len(rows) == 1)
                        if record_totals["minimum_nonempty_phase_descriptors"] == 0:
                            record_totals["minimum_nonempty_phase_descriptors"] = len(rows)
                        else:
                            record_totals["minimum_nonempty_phase_descriptors"] = min(
                                record_totals["minimum_nonempty_phase_descriptors"],
                                len(rows))
                        record_totals["maximum_phase_descriptors"] = max(
                            record_totals["maximum_phase_descriptors"],
                            len(rows))
                        previous_natural, natural_bubble, natural_hazard = (
                            update_boundary(previous_natural, phase, rows,
                                            banks, optimize=False))
                        previous_optimized, optimized_bubble, _ = (
                            update_boundary(previous_optimized, phase, rows,
                                            banks, optimize=True))
                        record_totals["natural_rmw_hazard_bubbles"] += (
                            natural_bubble)
                        record_totals["natural_phase_boundary_hazards"] += (
                            natural_hazard)
                        record_totals["optimized_rmw_hazard_bubbles"] += (
                            optimized_bubble)
                        if optimized_bubble:
                            record_totals[
                                "unavoidable_single_row_phase_hazards"] += 1

                record_totals["partition_window_contexts"] += 1
                record_totals["nonempty_partition_window_contexts"] += int(
                    context_phases != 0)
                record_totals["active_source_contexts"] += (
                    context_active_sources)
                record_totals["cache_vector_group_loads"] += context_groups
                record_totals["cache_phase_loads"] += context_phases
                record_totals["source_major_descriptors"] += (
                    context_descriptors)
                record_totals["maximum_context_descriptors"] = max(
                    record_totals["maximum_context_descriptors"],
                    context_descriptors)
                record_totals["maximum_context_phase_loads"] = max(
                    record_totals["maximum_context_phase_loads"],
                    context_phases)

        row = {
            "sample_id": int(record["sample_id"]),
            "operator_index": int(record["operator_index"]),
            **dict(record_totals),
        }
        per_record.append(row)
        for key, value in record_totals.items():
            if key.startswith("maximum_"):
                totals[key] = max(totals[key], value)
            elif key.startswith("minimum_"):
                if totals[key] == 0:
                    totals[key] = value
                elif value != 0:
                    totals[key] = min(totals[key], value)
            else:
                totals[key] += value
        print("[M157 RECORD] {}/20 sample={} op={} desc={} bubbles={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"],
            record_totals["source_major_descriptors"],
            record_totals["optimized_rmw_hazard_bubbles"]), flush=True)

    expected_work = m152_result["exact_work"]
    require(totals["source_major_descriptors"]
            == expected_work["bank_conflict_free_descriptors"],
            "M157 descriptor conservation disagrees with M152")
    require(totals["source_events"] == expected_work["source_events"],
            "M157 tuple conservation disagrees with M152")
    require(totals["source_active_keys"]
            == expected_work["source_active_keys"],
            "M157 source-key conservation disagrees with M152")
    require(totals["partition_window_contexts"] == 20 * 432 * 8,
            "M157 context population drift")

    candidate = int(m152_result["cycle_model"]["candidate_cycles"])
    m143_cycles = int(round(
        candidate * m152_result["comparisons"]["ratio_vs_m143r2_b4"]))
    optimized_cycles = candidate + totals["optimized_rmw_hazard_bubbles"]
    pingpong_startup_cycles = optimized_cycles + totals[
        "nonempty_partition_window_contexts"]
    pessimistic_single_cache_cycles = optimized_cycles + totals[
        "cache_phase_loads"]
    payload = {
        "schema": "m157_source_major_row_interleave_cache_dse_v1",
        "status": "PASS_HELDOUT_SOURCE_MAJOR_RMW_HAZARD_DSE_ONLY",
        "identity": {
            "analyzer_start_end_sha256": script_start,
            "direct_inputs_sha256": observed,
            "inherited_m147_inputs_sha256": observed_m147,
        },
        "extent": {
            "lineage": "H67/Motion ep35 heldout sample IDs 5..9",
            "records": 20,
            "partitions_per_record": 432,
            "rows_per_partition": 3000,
            "window_rows": 384,
            "windows_per_record": 8,
            "sources_per_partition": 16,
            "destination_blocks": 8,
            "destination_banks": 4,
        },
        "architecture": {
            "descriptor_order":
                "window_then_partition_then_source_then_destination_half_then_active_row",
            "cache_payload_bits_single_phase": 4 * 96 * 8,
            "cache_payload_bits_ping_pong": 2 * 4 * 96 * 8,
            "destination_half_zero": "destination IDs 0..3",
            "destination_half_one": "destination IDs 4..7",
            "accumulator_bank": "destination_id_modulo_4",
            "accumulator_address": "destination_bit2_concat_window_local_row",
            "accumulator_macro_assumed_read_latency_cycles": 1,
            "rmw_constraint":
                "same bank and same accumulator address cannot be issued in adjacent cycles without forwarding or one bubble",
        },
        "exact_work": dict(totals),
        "per_record": per_record,
        "comparisons": {
            "uncached_destination_vector_group_reads":
                totals["source_events"],
            "source_major_cached_vector_group_loads":
                totals["cache_vector_group_loads"],
            "vector_group_read_reduction":
                totals["source_events"] / totals["cache_vector_group_loads"],
            "m152_candidate_cycles": candidate,
            "optimized_interleave_cycles_before_cache_load_charge":
                optimized_cycles,
            "ratio_vs_m143_before_cache_load_charge":
                m143_cycles / optimized_cycles,
            "pessimistic_pingpong_one_startup_per_nonempty_context_cycles":
                pingpong_startup_cycles,
            "ratio_vs_m143_pessimistic_pingpong_startup":
                m143_cycles / pingpong_startup_cycles,
            "pessimistic_nonoverlapped_one_cycle_per_phase_cycles":
                pessimistic_single_cache_cycles,
            "ratio_vs_m143_pessimistic_nonoverlapped_phase_load":
                m143_cycles / pessimistic_single_cache_cycles,
            "phase_load_cycles_already_covered_by_m152_weight_service": False,
        },
        "admission": {
            "descriptor_tuple_conservation": True,
            "source_major_cache_load_census": True,
            "one_cycle_rmw_hazard_census": True,
            "integer_reorder_exactness": False,
            "source_cache_rtl": False,
            "integrated_m154_accumulator_rtl": False,
            "accumulator_forwarding_removed": False,
            "accumulator_sram_macro": False,
            "cache_sram_or_register_cost": False,
            "pingpong_phase_load_overlap_rtl": False,
            "m152_cycle_ratio": False,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "paper_safe_statement":
            "On the frozen 20-record H67 extent, a source-major destination-half cache and active-row interleave exactly conserve M152 descriptors and source events while quantifying cache group loads and adjacent accumulator RMW hazards. The result is a scheduling/cache DSE, not integrated RTL, macro PPA or admitted speedup.",
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "M157 analyzer changed during execution")
    output.mkdir(parents=True, exist_ok=False)
    (output / "m157_source_major_row_interleave_cache_dse.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(
        "PASS M157 descriptors={} events={} cache_groups={} cache_phases={} "
        "read_reduction={:.9f}x natural_hazards={} optimized_bubbles={} "
        "ratio_vs_m143_preload={:.9f}x ratio_vs_m143_pingpong={:.9f}x "
        "ratio_vs_m143_nooverlap={:.9f}x "
        "rtl=false macro=false physical_speedup=false system_speedup=false "
        "headline=false".format(
            totals["source_major_descriptors"], totals["source_events"],
            totals["cache_vector_group_loads"], totals["cache_phase_loads"],
            payload["comparisons"]["vector_group_read_reduction"],
            totals["natural_phase_boundary_hazards"],
            totals["optimized_rmw_hazard_bubbles"],
            payload["comparisons"]["ratio_vs_m143_before_cache_load_charge"],
            payload["comparisons"][
                "ratio_vs_m143_pessimistic_pingpong_startup"],
            payload["comparisons"][
                "ratio_vs_m143_pessimistic_nonoverlapped_phase_load"]),
        flush=True)


if __name__ == "__main__":
    main()
