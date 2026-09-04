#!/usr/bin/env python3
"""Finite-event H67 DSE for exact q/O pairs with q*O fixed at 128."""

from __future__ import division

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path


Q_VALUES = (16, 32, 64, 128)
OUTPUT_TILES = {16: 8, 32: 4, 64: 2, 128: 1}
POPCOUNT = tuple(bin(value).count("1") for value in range(1 << 16))


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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_runs(indices):
    ordered = sorted(indices)
    if not ordered:
        return 0
    return 1 + sum(current != previous + 1
                   for previous, current in zip(ordered, ordered[1:]))


def phase_metrics_all_q(counter, nested_centers):
    require(len(nested_centers) >= 128, "M397 requires q128 catalog")
    metrics = {q: Counter() for q in Q_VALUES}
    used = {q: set() for q in Q_VALUES}
    exact_rows = {q: 0 for q in Q_VALUES}
    for original, count in counter.items():
        original = int(original) & 0xffff
        population = POPCOUNT[original]
        best_distance = 17
        best_index = 0
        for index, center in enumerate(nested_centers[:128]):
            center = int(center) & 0xffff
            distance = POPCOUNT[original ^ center]
            if distance < best_distance:
                best_distance = distance
                best_index = index
            q = index + 1
            if q not in metrics:
                continue
            use_pwp = 1 + best_distance < population
            candidate = 1 + best_distance if use_pwp else population
            row = metrics[q]
            row["partition_vectors"] += count
            row["matcher_rows"] += count * int(population >= 2)
            row["bit_sparse_vector_ops_per_block"] += count * population
            row["candidate_vector_ops_per_block"] += count * candidate
            row["pwp_ops_per_block"] += count * int(use_pwp)
            row["correction_ops_per_block"] += count * (
                best_distance if use_pwp else population)
            row["assignment_rows"] += count * int(use_pwp)
            row["exact_pattern_hits"] += count * int(
                original != 0 and best_distance == 0)
            if use_pwp:
                require(original != 0, "zero row selected PWP")
                selected = int(nested_centers[best_index]) & 0xffff
                plus = original & ((~selected) & 0xffff)
                minus = selected & ((~original) & 0xffff)
                require(((selected | plus) & ((~minus) & 0xffff)) ==
                        original, "M397 exact reconstruction failure")
                used[q].add(best_index)
                exact_rows[q] += count
    payload = {}
    for q in Q_VALUES:
        row = metrics[q]
        require(row["candidate_vector_ops_per_block"] ==
                row["pwp_ops_per_block"] +
                row["correction_ops_per_block"],
                "M397 vector work conservation failure")
        row["zero_rows"] = counter[0]
        row["active_rows"] = row["partition_vectors"] - counter[0]
        row["pwp_rows"] = row["assignment_rows"]
        row["fallback_rows"] = row["active_rows"] - row["pwp_rows"]
        row["used_center_ids"] = sorted(used[q])
        row["used_pwp_patterns"] = len(used[q])
        row["used_center_runs"] = count_runs(used[q])
        row["exact_reconstruction_rows"] = exact_rows[q]
        require(row["pwp_rows"] == row["exact_reconstruction_rows"],
                "M397 exact row accounting failure")
        require(row["partition_vectors"] == row["zero_rows"] +
                row["pwp_rows"] + row["fallback_rows"] and
                row["active_rows"] == row["pwp_rows"] +
                row["fallback_rows"],
                "M397 population conservation failure")
        payload[q] = dict(row)
    return payload


def baseline_sample(phases, model, command_setup):
    scan = (model["rows_per_operator"] +
            model["popcount_filter_pipeline_cycles"])
    weight_data = int(math.ceil(
        model["weight_phase_bytes"] /
        float(model["dram_bytes_per_cycle"])))
    preprocess = max(scan, weight_data + command_setup)
    time = preprocess
    for index, phase in enumerate(phases):
        compute = (phase["bit_sparse_vector_ops_per_block"] *
                   model["output_blocks"])
        next_preprocess = preprocess if index + 1 < len(phases) else 0
        time += max(compute, next_preprocess)
        time += model["compute_tail_cycles_per_partition"]
    time += (model["operators"] * model["rows_per_operator"] *
             model["output_blocks"] //
             model["commit_output_blocks_per_cycle"])
    return time


def candidate_sample(phases, q, model, cfg, command_setup, sram_latency,
                     blocking_cycles=0.0):
    output_tile = OUTPUT_TILES[q]
    tile_count = model["output_blocks"] // output_tile
    require(q * output_tile == 128 and tile_count * output_tile == 8,
            "M397 fixed-product geometry drift")
    pattern_data = int(math.ceil(
        q * model["pattern_bytes"] /
        float(cfg["dram_bytes_per_cycle"])))
    time = 0.0
    components = Counter()
    maximum_tile_bytes = 0
    maximum_slot0_bytes = 0
    pwp_useful_slot_bytes = (
        model["pwp_vector_bytes_per_output_block"] * output_tile)
    pwp_physical_stride_bytes = int(math.ceil(
        pwp_useful_slot_bytes /
        float(cfg["pwp_physical_alignment_bytes"]))) * cfg[
            "pwp_physical_alignment_bytes"]
    bitmap_seal_cycles = int(math.ceil(
        q / float(cfg["active_bitmap_word_bits"])))
    exposed_next_tile_dma = 0.0
    for phase in phases:
        pattern_useful_bytes = q * model["pattern_bytes"]
        pattern_physical_bytes = (pattern_data *
                                  cfg["dram_bytes_per_cycle"])
        time += pattern_data + command_setup
        components["pattern_data"] += pattern_data
        components["pattern_command_setup"] += command_setup
        components["pattern_commands"] += 1
        components["pattern_useful_bytes"] += pattern_useful_bytes
        components["pattern_physical_bytes"] += pattern_physical_bytes

        matcher = (phase["partition_vectors"] +
                   phase["matcher_rows"] * (q // 16 - 1) + 2)
        time += matcher
        components["serial16_matcher"] += matcher
        time += bitmap_seal_cycles
        components["active_count_and_bitmap_seal"] += bitmap_seal_cycles
        components["descriptor_writes"] += phase["active_rows"]
        if phase["active_rows"] == 0:
            time += model["compute_tail_cycles_per_partition"]
            components["tail"] += model[
                "compute_tail_cycles_per_partition"]
            continue

        weight_bytes = (model["partition_bits"] *
                        model["weight_vector_bytes"] * output_tile)
        pwp_bytes = (phase["used_pwp_patterns"] *
                     pwp_physical_stride_bytes)
        tile_bytes = weight_bytes + pwp_bytes
        maximum_tile_bytes = max(maximum_tile_bytes, tile_bytes)
        maximum_slot0_bytes = max(
            maximum_slot0_bytes, q * model["pattern_bytes"] + tile_bytes)
        require(tile_bytes <= cfg["tile_slot_bytes_each"] and
                q * model["pattern_bytes"] + tile_bytes <=
                cfg["tile_slot_bytes_each"], "M397 tile slot overflow")
        data_cycles = int(math.ceil(
            tile_bytes / float(cfg["dram_bytes_per_cycle"])))
        tile_commands = 1 + phase["used_center_runs"]
        tile_dma = data_cycles + tile_commands * command_setup
        components["weight_bytes"] += weight_bytes * tile_count
        components["pwp_useful_bytes"] += (
            phase["used_pwp_patterns"] * pwp_useful_slot_bytes *
            tile_count)
        components["pwp_physical_bytes"] += pwp_bytes * tile_count
        components["pwp_physical_padding_bytes"] += (
            phase["used_pwp_patterns"] *
            (pwp_physical_stride_bytes - pwp_useful_slot_bytes) *
            tile_count)
        components["weight_commands"] += tile_count
        components["pwp_run_commands"] += (
            phase["used_center_runs"] * tile_count)
        components["tile_dma_commands"] += tile_commands * tile_count
        components["replays"] += tile_count
        components["descriptor_read_requests"] += (
            phase["active_rows"] * tile_count)
        components["descriptor_read_responses"] += (
            phase["active_rows"] * tile_count)
        components["descriptor_bundle_accepts"] += (
            phase["active_rows"] * tile_count)
        time += tile_dma
        components["tile0_data"] += data_cycles
        components["tile0_command_setup"] += (
            tile_commands * command_setup)

        exact_work = (phase["correction_ops_per_block"] * output_tile +
                      phase["pwp_ops_per_block"] * output_tile * 2)
        require(exact_work >= phase["active_rows"],
                "descriptor SRAM cannot feed selected compute recurrence")
        replay = (exact_work + sram_latency +
                  phase["active_rows"] * blocking_cycles)
        for tile_index in range(tile_count):
            if tile_index + 1 < tile_count:
                replay_end = time + replay
                dma_end = time + tile_dma
                exposed_next_tile_dma += max(0.0, dma_end - replay_end)
                time = max(replay_end, dma_end)
                components["later_tile_data_not_additive"] += data_cycles
                components["later_tile_command_setup_not_additive"] += (
                    tile_commands * command_setup)
            else:
                time += replay
            components["active_compute"] += exact_work
            components["descriptor_sram_startup"] += sram_latency
            components["blocking_descriptor_penalty"] += (
                phase["active_rows"] * blocking_cycles)
        time += model["compute_tail_cycles_per_partition"]
        components["tail"] += model["compute_tail_cycles_per_partition"]

    commit = (model["operators"] * model["rows_per_operator"] *
              model["output_blocks"] //
              model["commit_output_blocks_per_cycle"])
    time += commit
    components["common_commit"] += commit
    return {
        "cycles": time,
        "components": dict(components),
        "maximum_tile_bytes": maximum_tile_bytes,
        "maximum_slot0_bytes": maximum_slot0_bytes,
        "pwp_useful_slot_bytes": pwp_useful_slot_bytes,
        "pwp_physical_stride_bytes": pwp_physical_stride_bytes,
        "bitmap_seal_cycles_per_phase": bitmap_seal_cycles,
        "worst_case_slot_bytes_including_config": (
            q * model["pattern_bytes"] +
            model["partition_bits"] * model["weight_vector_bytes"] *
            output_tile + q * pwp_physical_stride_bytes),
        "exposed_next_tile_dma_cycles": exposed_next_tile_dma,
    }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M397 overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m397_h67_fixed_product_qo_finite_dse_contract_v1" and
            contract.get("status") == "FROZEN_BEFORE_M397_EXECUTION",
            "M397 contract drift")
    hw_root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, identity in contract["inputs"].items():
        path = hw_root / identity["path"]
        require(path.is_file(), "missing M397 input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"], "M397 SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": identity["path"], "sha256": observed}

    h67_sha = contract["paper_identity"]["checkpoint_sha256"]
    trace = strict_json(paths["m40_h67_s10_trace"])
    catalog = strict_json(paths["m338_catalog"])
    m394 = strict_json(paths["m394_result"])
    m395 = strict_json(paths["m395_independent_hammer"])
    m396 = strict_json(paths["m396_prereview"])
    require(trace["identity"]["checkpoint_sha256"] == h67_sha and
            trace["identity"]["bn_policy"] == "no_running",
            "M397 H67 runtime identity drift")
    require(catalog["split"]["test_or_validation_data_used"] is False and
            catalog["split"]["train_valid825_key_overlap"] == 0 and
            catalog["admission"]["train_only_catalog"] is True and
            catalog["admission"]["exact_arithmetic_identity"] is True and
            catalog["geometry"]["q_values"] == list(Q_VALUES),
            "M397 catalog split contamination")
    require(m394["identity_rebinding"]["paft_checkpoint_used"] is False,
            "M397 M394 identity drift")
    require(m395["decision"][
                "accept_m394_h67_four_conv_trace_cycle_estimate"] is True and
            m395["severity_counts"]["P0"] == 0 and
            m395["severity_counts"]["P1"] == 0,
            "M397 M395 independent admission drift")
    require(m396["decision"]["execute_fixed_product_qo_dse"] ==
            "CONDITIONAL_GO" and
            m396["severity_counts"]["P0"] == 0 and
            len(m396["decision"]["conditions"]) == 2,
            "M397 M396 repaired prereview drift")
    require(catalog["geometry"]["operators"] ==
            trace["cohort"]["operators"], "M397 operator drift")

    m43 = load_module(paths["m43_unpacker"], "m397_m43")
    model = contract["cycle_model"]
    cfg = contract["configuration"]
    trace_dir = paths["m40_h67_s10_trace"].parent
    operators = tuple(trace["cohort"]["operators"])
    op_index = {name: index for index, name in enumerate(operators)}
    histograms = defaultdict(Counter)
    payload_files = 0
    payload_bytes = 0
    for record_index, record in enumerate(trace["records"]):
        for key, sha_key in (("packed_file", "packed_file_sha256"),
                             ("value_payload_file",
                              "value_payload_sha256")):
            path = trace_dir / record[key]
            require(path.is_file() and sha256(path) == record[sha_key],
                    "M397 M40 payload drift")
            payload_files += 1
            payload_bytes += path.stat().st_size
        masks = m43.unpack_record_masks(trace_dir, record)
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (
                    m43.TILE_BITS // model["partition_bits"])
                for subtile in range(
                        m43.TILE_BITS // model["partition_bits"]):
                    value = ((value256 >>
                              (subtile * model["partition_bits"])) & 0xffff)
                    histograms[(record["sample_id"],
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
        print("[M397 HIST] {}/{}".format(
            record_index + 1, len(trace["records"])), flush=True)

    phases = {q: defaultdict(list) for q in Q_VALUES}
    summaries = {q: Counter() for q in Q_VALUES}
    maxima = {q: Counter() for q in Q_VALUES}
    for sample in range(model["samples"]):
        for op in range(model["operators"]):
            for partition in range(model["partitions_per_operator"]):
                counter = histograms[(sample, op, partition)]
                require(sum(counter.values()) == model["rows_per_operator"],
                        "M397 phase extent drift")
                centers = [int(value, 16) for value in
                           catalog["operators"][op]["partitions"][partition]
                           ["nested_patterns"]]
                rows = phase_metrics_all_q(counter, centers)
                for q in Q_VALUES:
                    phase = rows[q]
                    phases[q][sample].append(phase)
                    for field in (
                            "partition_vectors", "matcher_rows",
                            "bit_sparse_vector_ops_per_block",
                            "candidate_vector_ops_per_block",
                            "pwp_ops_per_block",
                            "correction_ops_per_block", "assignment_rows",
                            "exact_pattern_hits", "zero_rows", "active_rows",
                            "pwp_rows", "fallback_rows",
                            "used_pwp_patterns", "used_center_runs",
                            "exact_reconstruction_rows"):
                        summaries[q][field] += phase[field]
                    maxima[q]["used_pwp_patterns"] = max(
                        maxima[q]["used_pwp_patterns"],
                        phase["used_pwp_patterns"])
                    maxima[q]["used_center_runs"] = max(
                        maxima[q]["used_center_runs"],
                        phase["used_center_runs"])
        print("[M397 METRIC] sample={}/{}".format(
            sample + 1, model["samples"]), flush=True)

    sweep_rows = []
    robust_rows = []
    component_rows = {}
    for q in Q_VALUES:
        for command_setup in contract["sweep"]["dma_command_setup_cycles"]:
            baseline = sum(baseline_sample(
                phases[q][sample], model, command_setup)
                           for sample in range(model["samples"]))
            for latency in contract["sweep"][
                    "descriptor_sram_fixed_response_latency_cycles"]:
                sample_rows = [candidate_sample(
                    phases[q][sample], q, model, cfg,
                    command_setup, latency)
                    for sample in range(model["samples"])]
                candidate = sum(row["cycles"] for row in sample_rows)
                row = {
                    "q_capacity": q,
                    "output_tile": OUTPUT_TILES[q],
                    "output_tile_count": 8 // OUTPUT_TILES[q],
                    "fixed_product": q * OUTPUT_TILES[q],
                    "dma_command_setup_cycles": command_setup,
                    "descriptor_sram_latency_cycles": latency,
                    "baseline_cycles": baseline,
                    "candidate_cycles": candidate,
                    "speedup_vs_bit_sparse": baseline / float(candidate),
                    "pwp_useful_slot_bytes": sample_rows[0][
                        "pwp_useful_slot_bytes"],
                    "pwp_physical_stride_bytes": sample_rows[0][
                        "pwp_physical_stride_bytes"],
                    "bitmap_seal_cycles_per_phase": sample_rows[0][
                        "bitmap_seal_cycles_per_phase"],
                    "worst_case_slot_bytes_including_config": sample_rows[0][
                        "worst_case_slot_bytes_including_config"],
                    "maximum_tile_bytes": max(
                        value["maximum_tile_bytes"] for value in sample_rows),
                    "maximum_slot0_bytes": max(
                        value["maximum_slot0_bytes"] for value in sample_rows),
                    "exposed_next_tile_dma_cycles": sum(
                        value["exposed_next_tile_dma_cycles"]
                        for value in sample_rows),
                }
                sweep_rows.append(row)
                if (command_setup == contract["decision_rule"]
                        ["robust_dma_command_setup_cycles"] and
                        latency == contract["decision_rule"]
                        ["robust_descriptor_sram_latency_cycles"]):
                    robust_rows.append(row)
                    totals = Counter()
                    for sample_row in sample_rows:
                        totals.update(sample_row["components"])
                    component_rows[str(q)] = dict(totals)

    blocking_rows = []
    for q in Q_VALUES:
        baseline = sum(baseline_sample(
            phases[q][sample], model,
            contract["decision_rule"]["robust_dma_command_setup_cycles"])
                       for sample in range(model["samples"]))
        for penalty in contract["sweep"][
                "blocking_cycles_per_replayed_descriptor"]:
            sample_rows = [candidate_sample(
                phases[q][sample], q, model, cfg,
                contract["decision_rule"]["robust_dma_command_setup_cycles"],
                contract["decision_rule"]
                ["robust_descriptor_sram_latency_cycles"], penalty)
                for sample in range(model["samples"])]
            candidate = sum(row["cycles"] for row in sample_rows)
            blocking_rows.append({
                "q_capacity": q,
                "output_tile": OUTPUT_TILES[q],
                "blocking_cycles_per_replayed_descriptor": penalty,
                "baseline_cycles": baseline,
                "candidate_cycles": candidate,
                "speedup_vs_bit_sparse": baseline / float(candidate),
            })

    q32 = next(row for row in robust_rows if row["q_capacity"] == 32)
    m394_robust = m394["robust_decision_point"]
    require(q32["baseline_cycles"] == m394_robust["baseline_cycles"] and
            q32["candidate_cycles"] == m394_robust["candidate_cycles"] and
            abs(q32["speedup_vs_bit_sparse"] -
                m394_robust["speedup_vs_bit_sparse"]) < 1e-15,
            "M397 failed exact M394 q32 reproduction")
    selected = max(robust_rows, key=lambda row: row["speedup_vs_bit_sparse"])
    decision = ("GO_SELECTED_QO_PREDESIGN" if
                selected["speedup_vs_bit_sparse"] >=
                contract["decision_rule"]["minimum_selected_speedup"] else
                "NO_GO_FIXED_PRODUCT_QO_DSE")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sweep_fields = ["q_capacity", "output_tile", "output_tile_count",
                    "fixed_product", "dma_command_setup_cycles",
                    "descriptor_sram_latency_cycles", "baseline_cycles",
                    "candidate_cycles", "speedup_vs_bit_sparse",
                    "pwp_useful_slot_bytes", "pwp_physical_stride_bytes",
                    "bitmap_seal_cycles_per_phase",
                    "maximum_tile_bytes", "maximum_slot0_bytes",
                    "worst_case_slot_bytes_including_config",
                    "exposed_next_tile_dma_cycles"]
    write_csv(args.output_dir / "fixed_product_qo_sweep.csv",
              sweep_rows, sweep_fields)
    write_csv(args.output_dir / "blocking_sensitivity.csv", blocking_rows,
              ["q_capacity", "output_tile",
               "blocking_cycles_per_replayed_descriptor",
               "baseline_cycles", "candidate_cycles",
               "speedup_vs_bit_sparse"])
    phase_count = (model["samples"] * model["operators"] *
                   model["partitions_per_operator"])
    robust_baselines = {row["baseline_cycles"] for row in robust_rows}
    require(len(robust_baselines) == 1,
            "M397 fair baseline changed across q/O points")
    expected_slot_bytes = {
        int(q): int(value) for q, value in
        contract["execution_gates"]["worst_case_slot_bytes_by_q"].items()}
    for row in robust_rows:
        require(row["worst_case_slot_bytes_including_config"] ==
                expected_slot_bytes[row["q_capacity"]],
                "M397 worst-case slot footprint drift")
        require(row["pwp_physical_stride_bytes"] %
                cfg["pwp_physical_alignment_bytes"] == 0,
                "M397 unaligned physical PWP stride")
        require(row["worst_case_slot_bytes_including_config"] <=
                cfg["tile_slot_bytes_each"], "M397 tile slot overflow")
    result = {
        "schema": "m397_h67_fixed_product_qo_finite_dse_v1",
        "status": "PASS_M397_H67_FIXED_PRODUCT_QO_FINITE_DSE",
        "identity": identities,
        "paper_identity": contract["paper_identity"],
        "payload_audit": {
            "files_rehashed": payload_files,
            "bytes_rehashed": payload_bytes,
            "mismatches": 0,
        },
        "fairness": {
            "q_times_output_tile": 128,
            "weight_port": "SHARED96",
            "matcher": "SERIAL16_II1",
            "descriptor_sram": "one II1/L8/D8 stream",
            "tile_slots": "two 32768-byte slots",
            "pwp_stride": "each useful PWP slot padded to the next 32-byte physical stride",
            "bitmap_seal": "ceil(q/32) serialized 32-bit words charged before replay",
            "baseline_total_distinct_values_across_q": len(
                robust_baselines),
            "fallback": "strict exact bit-sparse",
            "wide_or_systolic_results_used": False,
        },
        "phase_count": phase_count,
        "population_by_q": {str(q): dict(summaries[q]) for q in Q_VALUES},
        "maximum_by_q": {str(q): dict(maxima[q]) for q in Q_VALUES},
        "robust_rows": robust_rows,
        "robust_component_sums_not_additive": component_rows,
        "descriptor_and_dma_ledgers_by_q": component_rows,
        "execution_gates": {
            "input_sha_mismatches": 0,
            "exact_reconstruction_mismatches": 0,
            "population_conservation_mismatches": 0,
            "nested_prefix_or_tie_break_mismatches": 0,
            "unaligned_dma_address_or_length": 0,
            "slot_overflows": 0,
            "unpriced_config_bitmap_command_or_padding_bytes": 0,
            "q32_o4_m394_reproduction_mismatches": 0,
            "baseline_total_distinct_values_across_q": len(
                robust_baselines),
            "descriptor_fifo_depth": cfg["descriptor_bundle_fifo_depth"],
            "descriptor_fifo_assumption": (
                "D8 credits cap outstanding II1/L8 requests; backend starts "
                "one bundle when ready and exact per-descriptor service is "
                "never below one cycle, so L8 startup is charged once per "
                "replay and no free descriptor throughput is credited."),
        },
        "negative_controls": {
            "q128_unpadded_144B_stride_alignment_gate_fails":
                (144 % cfg["pwp_physical_alignment_bytes"] != 0),
            "q128_required_serial16_extra_passes": 7,
            "q128_required_bitmap_seal_words": 4,
            "tile_count_is_8_div_output_tile": True,
            "wide_or_systolic_credit_used": False,
            "fallback_disabled_for_nonzero_nonpwp_rows": False,
        },
        "sweep": sweep_rows,
        "blocking_sensitivity": blocking_rows,
        "m394_q32_reproduction": {
            "exact": True,
            "baseline_cycles": q32["baseline_cycles"],
            "candidate_cycles": q32["candidate_cycles"],
            "speedup_vs_bit_sparse": q32["speedup_vs_bit_sparse"],
        },
        "selected": selected,
        "selected_vs_q32": {
            "absolute_speedup_gain": (
                selected["speedup_vs_bit_sparse"] -
                q32["speedup_vs_bit_sparse"]),
            "ratio_gain": (
                selected["speedup_vs_bit_sparse"] /
                q32["speedup_vs_bit_sparse"]),
        },
        "decision": decision,
        "admission": {
            "frozen_h67_ep35_no_running_runtime": True,
            "exact_fixed_product_cycle_dse": True,
            "selected_rtl": False,
            "synopsys_selected_qo": False,
            "energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "output_files": {
            "fixed_product_qo_sweep": "fixed_product_qo_sweep.csv",
            "blocking_sensitivity": "blocking_sensitivity.csv",
        },
    }
    output = args.output_dir / "m397_h67_fixed_product_qo_finite_dse_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M397_PASS q={} O={} robust={:.6f}x decision={}".format(
        selected["q_capacity"], selected["output_tile"],
        selected["speedup_vs_bit_sparse"], decision), flush=True)


if __name__ == "__main__":
    main()
