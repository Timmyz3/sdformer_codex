#!/usr/bin/env python3
"""Exact three-mode cycle replay for the frozen H67 four-Conv population.

This closes the M404 weak/strong baseline namespace with one executable
resource model.  It independently decodes the frozen M410R2 row transport,
cross-checks every phase against M401, and emits 17,280 ordered timestamp and
component records for each of dense16, exact zero-elided, and M401 combined.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


VAR_DENSE = "dense16_same_resource"
VAR_ZERO = "zero_elided_bit_sparse_exact_reproduction"
VAR_M401 = "m401_combined_exact_reproduction"
VARIANTS = (VAR_DENSE, VAR_ZERO, VAR_M401)


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
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def integer_row(row, key):
    value = row[key]
    require(value != "" and value.lstrip("-").isdigit(),
            "non-integer M401 CSV field: " + key)
    return int(value)


def read_m401_phases(path, expected):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample", "operator", "partition", "active_rows",
            "eligible_rows", "pwp_rows", "fallback_rows",
            "used_pwp_patterns", "used_center_runs", "narrow_tile0",
            "narrow_tile1", "reference_matcher", "early_matcher",
            "early_saved",
        }
        require(reader.fieldnames is not None and
                required.issubset(set(reader.fieldnames)),
                "M401 phase CSV schema drift")
        rows = list(reader)
    require(len(rows) == expected, "M401 phase CSV extent drift")
    for index, row in enumerate(rows):
        sample = index // (4 * 432)
        within = index % (4 * 432)
        operator = within // 432
        partition = within % 432
        require((integer_row(row, "sample"),
                 integer_row(row, "operator"),
                 integer_row(row, "partition")) ==
                (sample, operator, partition),
                "M401 phase ordering drift at {}".format(index))
    return rows


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decode_phase(block, rows_per_phase, nibble_lut):
    require(len(block) == rows_per_phase * 9,
            "M410R2 row transport truncated")
    raw = np.frombuffer(block, dtype=np.uint8).reshape(rows_per_phase, 9)
    require(bool(np.all(raw[:, 8] == 10)),
            "M410R2 row newline/layout drift")
    nibble = nibble_lut[raw[:, :8]]
    require(not bool(np.any(nibble == 255)),
            "M410R2 row contains non-hex payload")
    words = np.zeros(rows_per_phase, dtype=np.uint32)
    for column in range(8):
        words = words * np.uint32(16) + nibble[:, column].astype(np.uint32)
    return words


def baseline_phase(variant, time, phase, model, first, last):
    rows = model["rows_per_phase"]
    scan = rows + model["popcount_filter_pipeline_cycles"]
    weight_data = model["weight_phase_bytes"] // model["dram_bytes_per_cycle"]
    weight_command = model["dma_command_setup_cycles"]
    preprocess = max(scan, weight_data + weight_command)
    compute = (rows * model["dense_source_terms_per_row"] *
               model["output_blocks"] if variant == VAR_DENSE else
               phase["bit_sparse_vector_ops_per_block"] *
               model["output_blocks"])
    record_start = time
    initial = preprocess if first else 0
    time += initial
    compute_start = time
    compute_end = compute_start + compute
    next_preprocess = preprocess if not last else 0
    next_preprocess_start = compute_start
    next_preprocess_end = next_preprocess_start + next_preprocess
    overlap = min(compute, next_preprocess)
    exposed = max(0, next_preprocess - compute)
    body_end = max(compute_end, next_preprocess_end)
    tail_start = body_end
    tail = model["tail_cycles_per_phase"]
    time = tail_start + tail
    commit = model["commit_cycles_per_sample"] if last else 0
    time += commit
    additive = initial + compute + exposed + tail + commit
    require(time - record_start == additive,
            "baseline phase component conservation failure")
    result = {
        "record_start": record_start,
        "initial_preprocess_cycles": initial,
        "source_scan_cycles_requested": scan,
        "base_weight_bytes_requested": model["weight_phase_bytes"],
        "pwp_physical_bytes_requested": 0,
        "weight_dma_data_cycles_requested": weight_data,
        "weight_dma_command_cycles_requested": weight_command,
        "preprocess_service_cycles_requested": preprocess,
        "compute_start": compute_start,
        "compute_cycles": compute,
        "compute_end": compute_end,
        "next_preprocess_start": next_preprocess_start,
        "next_preprocess_cycles": next_preprocess,
        "next_preprocess_end": next_preprocess_end,
        "next_preprocess_overlap_cycles": overlap,
        "next_preprocess_exposed_cycles": exposed,
        "candidate_config_data_cycles": 0,
        "candidate_config_command_cycles": 0,
        "candidate_matcher_cycles": 0,
        "candidate_bitmap_seal_cycles": 0,
        "candidate_tile0_dma_data_cycles": 0,
        "candidate_tile0_dma_command_cycles": 0,
        "candidate_tile1_dma_requested_cycles": 0,
        "candidate_tile1_dma_overlap_cycles": 0,
        "candidate_tile1_dma_exposed_cycles": 0,
        "candidate_replay0_cycles": 0,
        "candidate_replay1_cycles": 0,
        "candidate_active_compute_work_cycles": 0,
        "candidate_descriptor_startup_cycles": 0,
        "tail_start": tail_start,
        "tail_cycles": tail,
        "commit_after_phase_cycles": commit,
        "record_end": time,
        "additive_cycles": additive,
    }
    return time, result


def selected_phase(time, phase, model, last):
    record_start = time
    config_data = ((model["elastic_config_bytes"] +
                    model["dram_bytes_per_cycle"] - 1) //
                   model["dram_bytes_per_cycle"])
    config_command = model["dma_command_setup_cycles"]
    matcher = phase["early_matcher"]
    seal = 1
    config_start = time
    time += config_data + config_command + matcher + seal
    config_end = time
    active = phase["active_rows"]
    tile_data = tile_command = tile_dma = 0
    replay0 = replay1 = work0 = work1 = 0
    tile0_dma_start = tile0_dma_end = time
    tile0_replay_start = tile0_replay_end = time
    tile1_dma_start = tile1_dma_end = time
    tile1_replay_start = tile1_replay_end = time
    tile1_overlap = tile1_exposed = 0
    if active:
        tile_bytes = (model["weight_bytes_per_tile"] +
                      phase["used_pwp_patterns"] *
                      model["elastic_center_stride_bytes"])
        require(model["elastic_config_bytes"] + tile_bytes <=
                model["tile_slot_bytes"], "M418 selected tile slot overflow")
        require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                "M418 selected tile DMA alignment drift")
        tile_data = tile_bytes // model["dram_bytes_per_cycle"]
        tile_command = ((1 + phase["used_center_runs"]) *
                        model["dma_command_setup_cycles"])
        tile_dma = tile_data + tile_command
        replay_work = (4 * phase["correction_ops_per_block"] +
                       8 * phase["pwp_rows"])
        work0 = replay_work - phase["narrow_tile0"]
        work1 = replay_work - phase["narrow_tile1"]
        require(work0 >= active and work1 >= active,
                "M418 selected descriptor service underflow")
        replay0 = work0 + model["descriptor_sram_latency_cycles"]
        replay1 = work1 + model["descriptor_sram_latency_cycles"]
        tile0_dma_start = time
        tile0_dma_end = tile0_dma_start + tile_dma
        tile0_replay_start = tile0_dma_end
        tile0_replay_end = tile0_replay_start + replay0
        tile1_dma_start = tile0_replay_start
        tile1_dma_end = tile1_dma_start + tile_dma
        tile1_replay_start = max(tile0_replay_end, tile1_dma_end)
        tile1_overlap = min(replay0, tile_dma)
        tile1_exposed = max(0, tile_dma - replay0)
        require(tile1_dma_start + tile1_overlap + tile1_exposed ==
                tile1_dma_end,
                "M418 selected tile1 DMA overlap conservation failure")
        tile1_replay_end = tile1_replay_start + replay1
        time = tile1_replay_end
    tail_start = time
    tail = model["tail_cycles_per_phase"]
    time += tail
    commit = model["commit_cycles_per_sample"] if last else 0
    time += commit
    additive = (config_data + config_command + matcher + seal +
                tile_dma + replay0 + tile1_exposed + replay1 + tail + commit)
    require(time - record_start == additive,
            "M418 selected phase component conservation failure")
    result = {
        "record_start": record_start,
        "initial_preprocess_cycles": 0,
        "source_scan_cycles_requested": matcher,
        "base_weight_bytes_requested": (
            model["weight_bytes_per_tile"] * 2 if active else 0),
        "pwp_physical_bytes_requested": (
            phase["used_pwp_patterns"] *
            model["elastic_center_stride_bytes"] * 2 if active else 0),
        "weight_dma_data_cycles_requested": tile_data * 2,
        "weight_dma_command_cycles_requested": tile_command * 2,
        "preprocess_service_cycles_requested": 0,
        "compute_start": tile0_replay_start,
        "compute_cycles": work0 + work1,
        "compute_end": tile1_replay_end,
        "next_preprocess_start": 0,
        "next_preprocess_cycles": 0,
        "next_preprocess_end": 0,
        "next_preprocess_overlap_cycles": 0,
        "next_preprocess_exposed_cycles": 0,
        "candidate_config_data_cycles": config_data,
        "candidate_config_command_cycles": config_command,
        "candidate_matcher_cycles": matcher,
        "candidate_bitmap_seal_cycles": seal,
        "candidate_tile0_dma_data_cycles": tile_data,
        "candidate_tile0_dma_command_cycles": tile_command,
        "candidate_tile1_dma_requested_cycles": tile_dma,
        "candidate_tile1_dma_overlap_cycles": tile1_overlap,
        "candidate_tile1_dma_exposed_cycles": tile1_exposed,
        "candidate_replay0_cycles": replay0,
        "candidate_replay1_cycles": replay1,
        "candidate_active_compute_work_cycles": work0 + work1,
        "candidate_descriptor_startup_cycles": (
            2 * model["descriptor_sram_latency_cycles"] if active else 0),
        "tail_start": tail_start,
        "tail_cycles": tail,
        "commit_after_phase_cycles": commit,
        "record_end": time,
        "additive_cycles": additive,
        "candidate_config_start": config_start,
        "candidate_config_end": config_end,
        "candidate_tile0_dma_start": tile0_dma_start,
        "candidate_tile0_dma_end": tile0_dma_end,
        "candidate_tile0_replay_start": tile0_replay_start,
        "candidate_tile0_replay_end": tile0_replay_end,
        "candidate_tile1_dma_start": tile1_dma_start,
        "candidate_tile1_dma_end": tile1_dma_end,
        "candidate_tile1_replay_start": tile1_replay_start,
        "candidate_tile1_replay_end": tile1_replay_end,
    }
    return time, result


def sum_field(rows, key):
    return sum(int(row[key]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M418 overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m418_h67_three_mode_exact_cycle_replay_contract_v1" and
            contract.get("status") == "FROZEN_BEFORE_M418_EXECUTION",
            "M418 contract identity drift")
    hw_root = args.contract.resolve().parents[1]
    identities = {}
    paths = {}
    for name, identity in contract["inputs"].items():
        path = hw_root / identity["path"]
        require(path.is_file(), "missing M418 input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"],
                "M418 exact-SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": identity["path"], "sha256": observed}

    prereview = strict_json(paths["m404_prereview"])
    m397 = strict_json(paths["m397_result"])
    m401 = strict_json(paths["m401_result"])
    m410 = strict_json(paths["m410r2_manifest"])
    gates = contract["execution_gates"]
    model = contract["cycle_model"]
    require(prereview["formal_simulator_work_required"]
            ["expected_phase_records"] == gates["expected_phase_records"],
            "M404 formal replay requirement drift")
    require(m397["m394_q32_reproduction"]["baseline_cycles"] ==
            gates["zero_cycles"] and
            m401["robust_variants"]["combined"]["candidate_cycles"] ==
            gates["selected_cycles"],
            "M397/M401 frozen cycle target drift")
    require(m410["population"]["phases"] == gates["phases_per_variant"] and
            m410["population"]["source_rows"] ==
            gates["total_partition_rows"],
            "M410R2 transport population drift")
    require(model["compute_port"] == "SHARED96" and
            model["descriptor_sram"] == "II1/L8/D8" and
            model["tile_slots"] == 2 and
            model["tile_slot_bytes"] == 32768 and
            model["dma_command_setup_cycles"] == 32 and
            model["dram_bytes_per_cycle"] == 32 and
            model["tail_cycles_per_phase"] == 2 and
            model["commit_cycles_per_sample"] == 96000,
            "M418 same-resource contract drift")

    phases = read_m401_phases(paths["m401_phase_csv"],
                              gates["phases_per_variant"])
    popcount = np.asarray([bin(index).count("1") for index in range(1 << 16)],
                          dtype=np.uint8)
    nibble_lut = np.full(256, 255, dtype=np.uint8)
    for index, value in enumerate(b"0123456789abcdef"):
        nibble_lut[value] = index
    phase_bytes = model["rows_per_phase"] * 9
    aggregate_population = {
        "source_rows": 0, "zero_rows": 0, "pop1_rows": 0,
        "active_rows": 0, "eligible_rows": 0, "pwp_rows": 0,
        "fallback_rows": 0, "bit_sparse_vector_ops_per_block": 0,
        "correction_ops_per_block": 0, "pass1_tasks": 0,
    }
    decoded = []
    with paths["m410r2_rows"].open("rb") as handle:
        for phase_index, frozen in enumerate(phases):
            words = decode_phase(handle.read(phase_bytes),
                                 model["rows_per_phase"], nibble_lut)
            original = np.bitwise_and(words, np.uint32(0xffff)).astype(
                np.uint16)
            population = popcount[original]
            distance = np.bitwise_and(np.right_shift(words, 21), 0x1f)
            use_pwp = np.bitwise_and(np.right_shift(words, 26), 1)
            pass1 = np.bitwise_and(np.right_shift(words, 27), 1)
            early = np.bitwise_and(np.right_shift(words, 28), 1)
            require(int(np.count_nonzero(np.right_shift(words, 29))) == 0,
                    "M410R2 reserved row bits nonzero")
            expected_pwp = (1 + distance < population).astype(np.uint32)
            require(bool(np.array_equal(use_pwp, expected_pwp)),
                    "M410R2 PWP predicate drift")
            eligible = population >= 2
            require(int(np.count_nonzero(pass1 + early > 1)) == 0 and
                    bool(np.array_equal((pass1 + early).astype(bool),
                                        eligible)),
                    "M410R2 pass partition drift")
            active_rows = int(np.count_nonzero(original))
            eligible_rows = int(np.count_nonzero(eligible))
            pwp_rows = int(np.count_nonzero(use_pwp))
            fallback_rows = active_rows - pwp_rows
            zero_rows = model["rows_per_phase"] - active_rows
            pop1_rows = int(np.count_nonzero(population == 1))
            bit_sparse = int(np.sum(population, dtype=np.uint64))
            correction = int(np.sum(np.where(use_pwp.astype(bool), distance,
                                             population), dtype=np.uint64))
            pass1_tasks = int(np.count_nonzero(pass1))
            early_matcher = model["rows_per_phase"] + pass1_tasks + 2
            reference_matcher = model["rows_per_phase"] + eligible_rows + 2
            observed = {
                "active_rows": active_rows,
                "eligible_rows": eligible_rows,
                "pwp_rows": pwp_rows,
                "fallback_rows": fallback_rows,
                "reference_matcher": reference_matcher,
                "early_matcher": early_matcher,
            }
            for key, value in observed.items():
                require(value == integer_row(frozen, key),
                        "M401 per-phase reproduction drift {} at {}".format(
                            key, phase_index))
            phase = {
                key: integer_row(frozen, key) for key in (
                    "sample", "operator", "partition", "active_rows",
                    "eligible_rows", "pwp_rows", "fallback_rows",
                    "used_pwp_patterns", "used_center_runs", "narrow_tile0",
                    "narrow_tile1", "reference_matcher", "early_matcher",
                    "early_saved")
            }
            phase.update({
                "bit_sparse_vector_ops_per_block": bit_sparse,
                "correction_ops_per_block": correction,
                "zero_rows": zero_rows,
                "pop1_rows": pop1_rows,
                "pass1_tasks": pass1_tasks,
            })
            decoded.append(phase)
            for key in aggregate_population:
                value = (model["rows_per_phase"] if key == "source_rows" else
                         phase[key])
                aggregate_population[key] += value
            if (phase_index + 1) % 1728 == 0:
                print("[M418 DECODE] sample={}/10 phases={}".format(
                    (phase_index + 1) // 1728, phase_index + 1), flush=True)
        require(handle.read(1) == b"", "M410R2 row transport has trailing data")

    for key in ("source_rows", "zero_rows", "pop1_rows", "pwp_rows",
                "pass1_tasks"):
        require(aggregate_population[key] == m410["population"][key],
                "M410R2 aggregate reproduction drift: " + key)
    for key in ("active_rows", "pwp_rows", "fallback_rows",
                "bit_sparse_vector_ops_per_block", "correction_ops_per_block"):
        require(aggregate_population[key] == m401["runtime_population"][key],
                "M401 runtime aggregate reproduction drift: " + key)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    common_fields = [
        "variant", "sample", "operator", "partition", "phase_global_index",
        "record_start", "initial_preprocess_cycles",
        "source_scan_cycles_requested", "base_weight_bytes_requested",
        "pwp_physical_bytes_requested", "weight_dma_data_cycles_requested",
        "weight_dma_command_cycles_requested",
        "preprocess_service_cycles_requested", "compute_start",
        "compute_cycles", "compute_end", "next_preprocess_start",
        "next_preprocess_cycles", "next_preprocess_end",
        "next_preprocess_overlap_cycles",
        "next_preprocess_exposed_cycles", "candidate_config_data_cycles",
        "candidate_config_command_cycles", "candidate_matcher_cycles",
        "candidate_bitmap_seal_cycles", "candidate_tile0_dma_data_cycles",
        "candidate_tile0_dma_command_cycles",
        "candidate_tile1_dma_requested_cycles",
        "candidate_tile1_dma_overlap_cycles",
        "candidate_tile1_dma_exposed_cycles", "candidate_replay0_cycles",
        "candidate_replay1_cycles", "candidate_active_compute_work_cycles",
        "candidate_descriptor_startup_cycles", "tail_start", "tail_cycles",
        "commit_after_phase_cycles", "record_end", "additive_cycles",
        "candidate_config_start", "candidate_config_end",
        "candidate_tile0_dma_start", "candidate_tile0_dma_end",
        "candidate_tile0_replay_start", "candidate_tile0_replay_end",
        "candidate_tile1_dma_start", "candidate_tile1_dma_end",
        "candidate_tile1_replay_start", "candidate_tile1_replay_end",
    ]
    output_names = {
        VAR_DENSE: "dense16_per_phase_timestamps_components.csv",
        VAR_ZERO: "zero_elided_per_phase_timestamps_components.csv",
        VAR_M401: "m401_combined_per_phase_timestamps_components.csv",
    }
    variant_rows = {}
    ledgers = []
    total_records = 0
    for variant in VARIANTS:
        time = 0
        rows = []
        for index, phase in enumerate(decoded):
            first = phase["partition"] == 0 and phase["operator"] == 0
            last = phase["partition"] == 431 and phase["operator"] == 3
            if variant == VAR_M401:
                time, components = selected_phase(time, phase, model, last)
            else:
                time, components = baseline_phase(
                    variant, time, phase, model, first, last)
                require(all(components[key] == 0 for key in (
                    "candidate_config_data_cycles",
                    "candidate_config_command_cycles",
                    "candidate_matcher_cycles",
                    "candidate_bitmap_seal_cycles",
                    "candidate_tile0_dma_data_cycles",
                    "candidate_tile0_dma_command_cycles",
                    "candidate_tile1_dma_requested_cycles",
                    "candidate_tile1_dma_overlap_cycles",
                    "candidate_tile1_dma_exposed_cycles",
                    "candidate_replay0_cycles", "candidate_replay1_cycles",
                    "candidate_descriptor_startup_cycles")),
                    "dense/zero charged candidate-only metadata")
            row = {
                "variant": variant,
                "sample": phase["sample"],
                "operator": phase["operator"],
                "partition": phase["partition"],
                "phase_global_index": index,
            }
            for field in common_fields[5:]:
                row[field] = components.get(field, 0)
            rows.append(row)
        require(len(rows) == gates["phases_per_variant"],
                "M418 variant phase count drift")
        require(time == gates[{VAR_DENSE: "dense_cycles",
                               VAR_ZERO: "zero_cycles",
                               VAR_M401: "selected_cycles"}[variant]],
                "M418 exact cycle gate failed: " + variant)
        require(sum_field(rows, "additive_cycles") == time and
                rows[0]["record_start"] == 0 and
                rows[-1]["record_end"] == time,
                "M418 aggregate timestamp conservation failure: " + variant)
        for previous, current in zip(rows, rows[1:]):
            require(previous["record_end"] == current["record_start"],
                    "M418 timestamp discontinuity: " + variant)
        write_csv(args.output_dir / output_names[variant], rows, common_fields)
        total_records += len(rows)
        variant_rows[variant] = rows
        ledgers.append({
            "variant": variant,
            "phase_records": len(rows),
            "cycles": time,
            "source_scan_cycles_requested": sum_field(
                rows, "source_scan_cycles_requested"),
            "base_weight_bytes_requested": sum_field(
                rows, "base_weight_bytes_requested"),
            "pwp_physical_bytes_requested": sum_field(
                rows, "pwp_physical_bytes_requested"),
            "weight_dma_data_cycles_requested": sum_field(
                rows, "weight_dma_data_cycles_requested"),
            "weight_dma_command_cycles_requested": sum_field(
                rows, "weight_dma_command_cycles_requested"),
            "preprocess_service_cycles_requested": sum_field(
                rows, "preprocess_service_cycles_requested"),
            "initial_preprocess_cycles": sum_field(
                rows, "initial_preprocess_cycles"),
            "compute_cycles": sum_field(rows, "compute_cycles"),
            "next_preprocess_overlap_cycles": sum_field(
                rows, "next_preprocess_overlap_cycles"),
            "next_preprocess_exposed_cycles": sum_field(
                rows, "next_preprocess_exposed_cycles"),
            "candidate_config_data_cycles": sum_field(
                rows, "candidate_config_data_cycles"),
            "candidate_config_command_cycles": sum_field(
                rows, "candidate_config_command_cycles"),
            "candidate_matcher_cycles": sum_field(
                rows, "candidate_matcher_cycles"),
            "candidate_bitmap_seal_cycles": sum_field(
                rows, "candidate_bitmap_seal_cycles"),
            "candidate_tile0_dma_data_cycles": sum_field(
                rows, "candidate_tile0_dma_data_cycles"),
            "candidate_tile0_dma_command_cycles": sum_field(
                rows, "candidate_tile0_dma_command_cycles"),
            "candidate_tile1_dma_requested_cycles": sum_field(
                rows, "candidate_tile1_dma_requested_cycles"),
            "candidate_tile1_dma_overlap_cycles": sum_field(
                rows, "candidate_tile1_dma_overlap_cycles"),
            "candidate_tile1_dma_exposed_cycles": sum_field(
                rows, "candidate_tile1_dma_exposed_cycles"),
            "candidate_replay0_cycles": sum_field(
                rows, "candidate_replay0_cycles"),
            "candidate_replay1_cycles": sum_field(
                rows, "candidate_replay1_cycles"),
            "candidate_descriptor_startup_cycles": sum_field(
                rows, "candidate_descriptor_startup_cycles"),
            "tail_cycles": sum_field(rows, "tail_cycles"),
            "commit_cycles": sum_field(rows, "commit_after_phase_cycles"),
            "additive_cycle_sum": sum_field(rows, "additive_cycles"),
        })
    ledger_by_variant = {row["variant"]: row for row in ledgers}
    dense_ledger = ledger_by_variant[VAR_DENSE]
    zero_ledger = ledger_by_variant[VAR_ZERO]
    selected_ledger = ledger_by_variant[VAR_M401]
    require(all(row["base_weight_bytes_requested"] ==
                gates["same_base_weight_bytes"] for row in ledgers),
            "M418 fixed 12,288-byte/phase weight ledger drift")
    require(dense_ledger["initial_preprocess_cycles"] ==
            gates["same_initial_preprocess_cycles"] and
            zero_ledger["initial_preprocess_cycles"] ==
            gates["same_initial_preprocess_cycles"] and
            dense_ledger["next_preprocess_exposed_cycles"] ==
            gates["dense_exposed_next_phase_preprocess_cycles"] and
            zero_ledger["next_preprocess_exposed_cycles"] ==
            gates["zero_exposed_next_phase_preprocess_cycles"],
            "M418 baseline preprocess component gate drift")
    require(all(row["tail_cycles"] == gates["same_tail_cycles"] and
                row["commit_cycles"] == gates["same_commit_cycles"]
                for row in ledgers),
            "M418 tail/commit component gate drift")
    require(selected_ledger["candidate_tile1_dma_exposed_cycles"] ==
            gates["selected_tile1_dma_exposed_cycles"],
            "M418 selected DMA-overlap component gate drift")
    candidate_only = (
        "candidate_config_data_cycles", "candidate_config_command_cycles",
        "candidate_matcher_cycles", "candidate_bitmap_seal_cycles",
        "candidate_tile0_dma_data_cycles",
        "candidate_tile0_dma_command_cycles",
        "candidate_tile1_dma_requested_cycles",
        "candidate_tile1_dma_overlap_cycles",
        "candidate_tile1_dma_exposed_cycles", "candidate_replay0_cycles",
        "candidate_replay1_cycles", "candidate_descriptor_startup_cycles")
    require(sum(dense_ledger[key] for key in candidate_only) ==
            gates["dense_candidate_only_metadata_cycles"] and
            sum(zero_ledger[key] for key in candidate_only) ==
            gates["zero_candidate_only_metadata_cycles"],
            "M418 dense/zero candidate-only metadata gate drift")
    require(total_records == gates["expected_phase_records"],
            "M418 total phase record count drift")
    ledger_fields = list(ledgers[0].keys())
    write_csv(args.output_dir / "three_mode_component_conservation.csv",
              ledgers, ledger_fields)

    dense_cycles = gates["dense_cycles"]
    zero_cycles = gates["zero_cycles"]
    selected_cycles = gates["selected_cycles"]
    comparison = [
        {"variant": VAR_DENSE, "baseline_strength": "weak",
         "cycles": dense_cycles, "speedup_vs_dense16_weak": 1.0,
         "speedup_vs_zero_elided_strong": zero_cycles / dense_cycles,
         "claim_scope": contract["scope"]["scope_label_required"]},
        {"variant": VAR_ZERO, "baseline_strength": "strong",
         "cycles": zero_cycles,
         "speedup_vs_dense16_weak": dense_cycles / zero_cycles,
         "speedup_vs_zero_elided_strong": 1.0,
         "claim_scope": contract["scope"]["scope_label_required"]},
        {"variant": VAR_M401, "baseline_strength": "candidate",
         "cycles": selected_cycles,
         "speedup_vs_dense16_weak": dense_cycles / selected_cycles,
         "speedup_vs_zero_elided_strong": zero_cycles / selected_cycles,
         "claim_scope": contract["scope"]["scope_label_required"]},
    ]
    write_csv(args.output_dir / "three_mode_cycle_comparison.csv", comparison,
              ["variant", "baseline_strength", "cycles",
               "speedup_vs_dense16_weak",
               "speedup_vs_zero_elided_strong", "claim_scope"])

    result = {
        "schema": "m418_h67_three_mode_exact_cycle_replay_v1",
        "status": "PASS_M418_FORMAL_THREE_MODE_EXECUTABLE_REPLAY",
        "identity": identities,
        "scope": contract["scope"],
        "resource_contract": model,
        "runtime_population_reproduction": aggregate_population,
        "variants": {row["variant"]: row for row in comparison},
        "component_conservation": {row["variant"]: row for row in ledgers},
        "execution_gates": {
            "exact_input_sha_mismatches": 0,
            "m401_per_phase_population_mismatches": 0,
            "pwp_predicate_or_pass_partition_mismatches": 0,
            "dense_candidate_metadata_cycles": 0,
            "zero_candidate_metadata_cycles": 0,
            "phase_records": total_records,
            "expected_phase_records": gates["expected_phase_records"],
            "dense_cycle_mismatch": 0,
            "zero_cycle_mismatch": 0,
            "selected_cycle_mismatch": 0,
            "component_or_timestamp_conservation_mismatches": 0,
            "docs359_modified": False,
        },
        "admission": {
            "executable_three_mode_cycle_replay": True,
            "dense16_weak_baseline_cycles": True,
            "zero_elided_strong_baseline_cycles": True,
            "m401_combined_candidate_cycles": True,
            "standalone_four_bottleneck_conv_trace_cycles": True,
            "independent_hammer_complete": False,
            "full_network": False,
            "system_speedup": False,
            "rtl_measured_speedup": False,
            "power": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "output_files": {
            "dense16_per_phase": output_names[VAR_DENSE],
            "zero_elided_per_phase": output_names[VAR_ZERO],
            "m401_combined_per_phase": output_names[VAR_M401],
            "component_conservation":
                "three_mode_component_conservation.csv",
            "cycle_comparison": "three_mode_cycle_comparison.csv",
        },
    }
    output = args.output_dir / "m418_h67_three_mode_exact_cycle_replay_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    readme = """# M418 exact three-mode H67 replay\n\n""" \
        "PASS: 51,840 ordered phase records under one frozen resource " \
        "contract.  Dense16 is the weak baseline; exact zero-elided is the " \
        "strong baseline.  Scope is **four H67 bottleneck Conv3x3 operators " \
        "only**.  These are trace-cycle simulator results, not full-network, " \
        "system, RTL-measured, power, or paper-ready PPA claims.\n"
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print("M418_PASS dense={} zero={} selected={} weak={:.9f}x "
          "strong={:.9f}x records={}".format(
              dense_cycles, zero_cycles, selected_cycles,
              dense_cycles / selected_cycles,
              zero_cycles / selected_cycles, total_records), flush=True)


if __name__ == "__main__":
    main()
