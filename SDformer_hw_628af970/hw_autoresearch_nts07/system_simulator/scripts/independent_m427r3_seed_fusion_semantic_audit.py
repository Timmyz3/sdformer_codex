#!/usr/bin/env python3
"""Independently audit M426 seed-fusion accumulator semantics.

This audit does not import or execute an M401/M426 analyzer.  It checks the
frozen accumulator/schedule sources by exact SHA and fixed semantic fragments,
then decodes all 51.84M M410R2 rows directly to determine whether a positive-
residual PWP seed generally enters an empty or already-valid accumulator row.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROWS = 3000
PARTITIONS = 432
OPERATORS = 4
SAMPLES = 10
PHASES = SAMPLES * OPERATORS * PARTITIONS
ROW_BYTES = 9


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def check_fragments(path, required, forbidden=()):
    source = Path(path).read_text(encoding="utf-8")
    for fragment in required:
        require(fragment in source,
                "required semantic fragment missing from {}: {}".format(
                    path, fragment))
    for fragment in forbidden:
        require(fragment not in source,
                "forbidden semantic fragment present in {}: {}".format(
                    path, fragment))
    return {"required_fragments": list(required),
            "forbidden_fragments": list(forbidden)}


def decode_phase(block, nibble_lut):
    require(len(block) == ROWS * ROW_BYTES, "M410R2 row transport truncated")
    raw = np.frombuffer(block, dtype=np.uint8).reshape(ROWS, ROW_BYTES)
    require(bool(np.all(raw[:, 8] == 10)), "M410R2 line layout drift")
    digits = nibble_lut[raw[:, :8]]
    require(not bool(np.any(digits == 255)), "M410R2 non-hex digit")
    words = np.zeros(ROWS, dtype=np.uint32)
    for column in range(8):
        words = (words << np.uint32(4)) | digits[:, column].astype(np.uint32)
    require(not bool(np.any(words >> np.uint32(29))),
            "M410R2 reserved row bits nonzero")
    return words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M427R3 overwrite")

    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m427r3_m426_seed_fusion_semantic_addendum_contract_v1" and
            contract.get("status") ==
            "FROZEN_BEFORE_M427R3_SEMANTIC_AUDIT",
            "M427R3 contract drift")
    hw_root = args.contract.resolve().parents[1]
    paths = {}
    observed_sha = {}
    for name, identity in contract["inputs"].items():
        path = hw_root / identity["path"]
        require(path.is_file(), "missing input: " + name)
        actual = sha256(path)
        require(actual == identity["sha256"], "input SHA drift: " + name)
        paths[name] = path
        observed_sha[name] = actual

    m108 = strict_json(paths["m108_schedule_contract"])
    require(m108["ordering"]["loop_nest"] ==
            "sample, operator, W64 raster window, partition" and
            m108["ordering"]["commit"] ==
            "after partition 431, one pipeline flush plus one 96-lane vector per valid row/output block",
            "M108 cross-partition accumulation schedule drift")

    source_audit = {
        "m118_accumulator": check_fragments(
            paths["m118_accumulator_rtl"], (
                "lane_base_ext[lane] = update_pipe_base_valid_q",
                "lane_sum_ext[lane] = lane_base_ext[lane] + lane_delta_ext[lane];",
                "update_pipe_base_valid_q\n                        <= row_valid_q[update_block][update_row];",
                "if (window_start_accept) begin",
                "row_valid_q[bank] <= '0;",
            )),
        "m120_integrated_island": check_fragments(
            paths["m120_integrated_rtl"], (
                ".update_delta(mapper_update_delta)",
                "m118_w384_signed19_lane_sliced_accumulator_adapter accumulator",
            )),
        "m405_selected_slice": check_fragments(
            paths["m405_selected_slice_rtl"], (
                "Non-contribution integration shell",
                "output logic [1151:0]           contribution_data",
            )),
        "m401_model": check_fragments(
            paths["m401_analyzer"], (
                "4 * phase[\"correction_ops_per_block\"] +",
                "8 * phase[\"pwp_rows\"]",
                "time += model[\"commit_cycles_per_sample\"]",
            ), (
                "local_partition_accumulator",
                "partition_merge_cycles",
                "three_operand_accumulator",
            )),
        "m426_claim": check_fragments(
            paths["m426_contract"], (
                "one existing 96-lane signed adder; seed fusion adds an operand-source mux, not another correction adder",
            )),
    }

    m401_contract = strict_json(paths["m401_contract"])
    require(m401_contract["cycle_model"]["commit_cycles_per_sample"] == 96000
            and m401_contract["claim_boundary"]["same_shared96_compute_port"]
            is True,
            "M401 shared accumulator/commit contract drift")
    m412_contract = strict_json(paths["m412_contract"])
    require(any("compute backend remain outside M412" in item
                for item in m412_contract["red_lines"]),
            "M412 compute-backend exclusion drift")
    m426_result = strict_json(paths["m426_result"])
    require(m426_result["hardware_contract"]["fused_datapath"] ==
            "one 96-lane signed accumulator adder; mux selects reconstructed PWP instead of prior accumulator for the first correction; no second correction is issued",
            "M426 fused datapath claim drift")

    row_path = paths["m410_rows"]
    require(row_path.stat().st_size == PHASES * ROWS * ROW_BYTES,
            "M410R2 row byte extent drift")
    nibble_lut = np.full(256, 255, dtype=np.uint8)
    for index, character in enumerate(b"0123456789abcdef"):
        nibble_lut[character] = index

    counts = {
        "raw_rows": 0,
        "active_rows": 0,
        "pwp_rows": 0,
        "positive_residual_pwp_rows": 0,
        "exact_pwp_rows": 0,
        "pwp_rows_with_prior_active_partition": 0,
        "positive_residual_pwp_rows_with_prior_active_partition": 0,
        "positive_residual_pwp_rows_without_prior_active_partition": 0,
        "exact_pwp_rows_with_prior_active_partition": 0,
    }
    examples = []
    partitions_with_prior_positive = set()
    sample_operator_with_prior_positive = set()
    with row_path.open("rb") as handle:
        for sample in range(SAMPLES):
            for operator in range(OPERATORS):
                prior_active = np.zeros(ROWS, dtype=np.bool_)
                for partition in range(PARTITIONS):
                    words = decode_phase(handle.read(ROWS * ROW_BYTES),
                                         nibble_lut)
                    original = words & np.uint32(0xffff)
                    active = original != 0
                    use_pwp = ((words >> np.uint32(26)) &
                               np.uint32(1)) != 0
                    distance = ((words >> np.uint32(21)) &
                                np.uint32(0x1f))
                    positive = np.logical_and(use_pwp, distance != 0)
                    exact = np.logical_and(use_pwp, distance == 0)
                    prior_pwp = np.logical_and(use_pwp, prior_active)
                    prior_positive = np.logical_and(positive, prior_active)
                    prior_exact = np.logical_and(exact, prior_active)
                    no_prior_positive = np.logical_and(positive,
                                                       np.logical_not(
                                                           prior_active))

                    counts["raw_rows"] += ROWS
                    counts["active_rows"] += int(np.count_nonzero(active))
                    counts["pwp_rows"] += int(np.count_nonzero(use_pwp))
                    counts["positive_residual_pwp_rows"] += int(
                        np.count_nonzero(positive))
                    counts["exact_pwp_rows"] += int(np.count_nonzero(exact))
                    counts["pwp_rows_with_prior_active_partition"] += int(
                        np.count_nonzero(prior_pwp))
                    counts[
                        "positive_residual_pwp_rows_with_prior_active_partition"
                    ] += int(np.count_nonzero(prior_positive))
                    counts[
                        "positive_residual_pwp_rows_without_prior_active_partition"
                    ] += int(np.count_nonzero(no_prior_positive))
                    counts[
                        "exact_pwp_rows_with_prior_active_partition"
                    ] += int(np.count_nonzero(prior_exact))
                    if bool(np.any(prior_positive)):
                        partitions_with_prior_positive.add(partition)
                        sample_operator_with_prior_positive.add(
                            (sample, operator))
                        if len(examples) < 16:
                            for row in np.flatnonzero(prior_positive):
                                examples.append({
                                    "sample": sample,
                                    "operator": operator,
                                    "partition": partition,
                                    "source_row": int(row),
                                    "original_u16": int(original[row]),
                                    "distance": int(distance[row]),
                                    "base_semantics": "old_psum_valid",
                                })
                                if len(examples) == 16:
                                    break
                    prior_active |= active
        require(handle.read(1) == b"", "M410R2 trailing transport data")

    frozen = contract["expected_population"]
    for key in ("raw_rows", "active_rows", "pwp_rows",
                "positive_residual_pwp_rows", "exact_pwp_rows"):
        require(counts[key] == frozen[key],
                "frozen population drift: " + key)
    prior_positive = counts[
        "positive_residual_pwp_rows_with_prior_active_partition"]
    no_prior_positive = counts[
        "positive_residual_pwp_rows_without_prior_active_partition"]
    require(prior_positive > 0 and
            prior_positive + no_prior_positive ==
            counts["positive_residual_pwp_rows"],
            "prior-active partition classification failure")

    variants = m426_result["variants"]
    dual_cycles = variants["dualbank_parallel_low8_high4"]["cycles"]
    fused_cycles = variants[
        "dualbank_seed_first_correction_fused"]["cycles"]
    claimed_savings = m426_result["cycle_savings"][
        "seed_first_correction_fusion_vs_coread"]
    require(claimed_savings ==
            8 * counts["positive_residual_pwp_rows"] and
            fused_cycles + claimed_savings == dual_cycles,
            "M426 fusion savings conservation drift")
    required_existing_adder_repair_cycles = 8 * prior_positive
    corrected_existing_adder_cycles = (
        fused_cycles + required_existing_adder_repair_cycles)
    surviving_empty_base_fusion_cycles = 8 * no_prior_positive
    require(corrected_existing_adder_cycles ==
            dual_cycles - surviving_empty_base_fusion_cycles,
            "corrected two-input-adder conservation failure")

    result = {
        "schema": "m427r3_m426_seed_fusion_semantic_audit_v1",
        "status": "P0_CONFIRMED_M426_SEED_FUSION_NOT_EXECUTABLE_AS_SPECIFIED",
        "date": "2026-08-26",
        "inputs_sha256": observed_sha,
        "schedule_semantics": {
            "loop_nest": m108["ordering"]["loop_nest"],
            "commit": m108["ordering"]["commit"],
            "accumulator_clear_event": "window_start_accept only",
            "accepted_update_equation": "new_psum = old_psum + update_delta",
            "m426_claimed_positive_first_cycle": "new_psum = PWP + correction",
            "required_positive_first_cycle":
                "new_psum = old_psum + PWP + correction",
            "already_charged_partition_local_accumulator_merge": False,
            "already_charged_three_operand_or_compressor_path": False,
        },
        "source_fragment_audit": source_audit,
        "full_runtime_partition_audit": {
            **counts,
            "partitions_with_prior_positive_count": len(
                partitions_with_prior_positive),
            "minimum_partition_with_prior_positive": min(
                partitions_with_prior_positive),
            "maximum_partition_with_prior_positive": max(
                partitions_with_prior_positive),
            "sample_operator_pairs_with_prior_positive": len(
                sample_operator_with_prior_positive),
            "examples": examples,
        },
        "cycle_consequence": {
            "dual_cycles": dual_cycles,
            "claimed_fused_cycles": fused_cycles,
            "claimed_fusion_saved_cycles": claimed_savings,
            "existing_two_input_adder_required_repair_cycles":
                required_existing_adder_repair_cycles,
            "existing_two_input_adder_corrected_cycles_with_empty_base_only_fusion":
                corrected_existing_adder_cycles,
            "empty_base_only_surviving_fusion_saved_cycles":
                surviving_empty_base_fusion_cycles,
            "fully_serial_partition_local_merge_cycles": claimed_savings,
            "fully_serial_partition_local_merge_total_cycles": dual_cycles,
            "note": "A new 96-lane pre-adder/compressor could restore the issue count, but it is a new uncharged datapath requiring a new contract, RTL value miter, DC/STA/Formality, and macro/interconnect PPA.",
        },
        "decision": {
            "seed_fusion": "REVOKED_NON_EXECUTABLE_OPPORTUNITY",
            "seed_fusion_rtl_as_specified": "NO_GO",
            "dual_coread_semantics": "SURVIVES",
            "dual_coread_admission":
                "STANDALONE_RTL_ONLY_THROUGHPUT_AREA_PARETO",
        },
        "claim_boundary": {
            "m426_cycle_arithmetic_reproduces": True,
            "m426_seed_fusion_datapath_semantically_executable": False,
            "dual_coread_exact_reconstruction": True,
            "dual_coread_resource_normalized": False,
            "rtl_measured_speedup": False,
            "physical_sram_or_interconnect_ppa": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m427r3_seed_fusion_semantic_audit_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("[M427R3] decoded_rows={}".format(counts["raw_rows"]))
    print("[M427R3] prior_active_positive={} no_prior_positive={}".format(
        prior_positive, no_prior_positive))
    print("[M427R3] claimed_fused={} corrected_two_input={} dual={}".format(
        fused_cycles, corrected_existing_adder_cycles, dual_cycles))
    print("[M427R3] P0 CONFIRMED; seed fusion NO-GO as specified")


if __name__ == "__main__":
    main()
