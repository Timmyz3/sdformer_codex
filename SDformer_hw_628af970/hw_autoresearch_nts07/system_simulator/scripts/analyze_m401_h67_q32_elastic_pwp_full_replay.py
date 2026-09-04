#!/usr/bin/env python3
"""Full H67 replay for exact q32/O4 elastic-width PWP plus early-hit."""

from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

import numpy as np


PREFIXES = (16, 32, 48, 64, 80, 96, 112, 128)
POPCOUNT = tuple(bin(value).count("1") for value in range(1 << 16))
VARIANTS = ("m397_anchor", "elastic_only", "early_only", "combined")


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


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pack_high4(raw12):
    high = ((raw12 >> 8) & 0xf).astype(np.uint8)
    return (high[0::2] | (high[1::2] << 4)).astype(np.uint8)


def build_static_codec(catalog, weight_paths, static_csv_path):
    flags = []
    static_rows = []
    digest = hashlib.sha256()
    total_blocks = 0
    total_lanes = 0
    narrow_blocks = 0
    signed12_violations = 0
    wide_mismatches = 0
    narrow_mismatches = 0
    padding_nonzero = 0
    global_minimum = 1 << 30
    global_maximum = -(1 << 30)
    for operator in range(4):
        weights = np.fromfile(weight_paths[operator], dtype=np.int8)
        require(weights.size == 6912 * 768,
                "M401 weight extent drift")
        weights = weights.reshape(6912, 768).astype(np.int16)
        operator_flags = []
        for partition in range(432):
            center_words = [int(value, 16) for value in
                            catalog["operators"][operator]["partitions"]
                            [partition]["nested_patterns"][:32]]
            bits = np.asarray([
                [(center >> bit) & 1 for bit in range(16)]
                for center in center_words], dtype=np.int16)
            values = bits @ weights[partition * 16:(partition + 1) * 16]
            require(values.shape == (32, 768), "M401 PWP shape drift")
            block_flags = np.zeros((32, 8), dtype=np.bool_)
            for center_id in range(32):
                for output_block in range(8):
                    vector = values[center_id,
                                    output_block * 96:(output_block + 1) * 96]
                    minimum = int(vector.min())
                    maximum = int(vector.max())
                    global_minimum = min(global_minimum, minimum)
                    global_maximum = max(global_maximum, maximum)
                    violation = int(np.count_nonzero(
                        (vector < -2048) | (vector > 2047)))
                    signed12_violations += violation
                    raw12 = (vector.astype(np.int32) & 0xfff)
                    low8 = (raw12 & 0xff).astype(np.uint8)
                    high4 = pack_high4(raw12)
                    padding = np.zeros(16, dtype=np.uint8)
                    unpacked_high = np.empty(96, dtype=np.int32)
                    unpacked_high[0::2] = high4 & 0xf
                    unpacked_high[1::2] = high4 >> 4
                    wide_raw = (unpacked_high << 8) | low8.astype(np.int32)
                    wide = np.where(wide_raw >= 2048,
                                    wide_raw - 4096, wide_raw)
                    wide_mismatch = int(np.count_nonzero(
                        wide != vector.astype(np.int32)))
                    wide_mismatches += wide_mismatch
                    narrow = minimum >= -128 and maximum <= 127
                    narrow_recon = np.where(low8.astype(np.int32) >= 128,
                                            low8.astype(np.int32) - 256,
                                            low8.astype(np.int32))
                    narrow_mismatch = (int(np.count_nonzero(
                        narrow_recon != vector.astype(np.int32)))
                        if narrow else 0)
                    narrow_mismatches += narrow_mismatch
                    padding_nonzero += int(np.count_nonzero(padding))
                    block_flags[center_id, output_block] = narrow
                    narrow_blocks += int(narrow)
                    total_blocks += 1
                    total_lanes += 96
                    header = struct.pack(
                        "<HHBBH", operator, partition, center_id,
                        output_block, center_words[center_id])
                    block_digest = hashlib.sha256(
                        header + low8.tobytes() + high4.tobytes() +
                        padding.tobytes() + bytes([int(narrow)])).hexdigest()
                    digest.update(bytes.fromhex(block_digest))
                    static_rows.append({
                        "operator": operator,
                        "partition": partition,
                        "center_id": center_id,
                        "output_block": output_block,
                        "minimum": minimum,
                        "maximum": maximum,
                        "narrow": int(narrow),
                        "codec_sha256": block_digest,
                    })
            operator_flags.append(block_flags)
        flags.append(operator_flags)
    write_csv(static_csv_path, static_rows,
              ["operator", "partition", "center_id", "output_block",
               "minimum", "maximum", "narrow", "codec_sha256"])
    return flags, {
        "blocks": total_blocks,
        "lanes": total_lanes,
        "narrow_blocks": narrow_blocks,
        "narrow_fraction": narrow_blocks / float(total_blocks),
        "global_minimum": global_minimum,
        "global_maximum": global_maximum,
        "maximum_absolute": max(abs(global_minimum), abs(global_maximum)),
        "signed12_violations": signed12_violations,
        "wide_reconstruction_mismatches": wide_mismatches,
        "narrow_reconstruction_mismatches": narrow_mismatches,
        "nonzero_padding_bytes": padding_nonzero,
        "codec_global_sha256": digest.hexdigest(),
    }


def analyze_phase(counter, centers, narrow_flags):
    row = Counter()
    used = set()
    first_hit = Counter()
    cumulative_all = Counter()
    cumulative_eligible = Counter()
    for original, count in counter.items():
        original = int(original) & 0xffff
        population = POPCOUNT[original]
        row["source_rows"] += count
        row["zero_rows"] += count * int(original == 0)
        row["pop1_rows"] += count * int(population == 1)
        eligible = population >= 2
        row["eligible_rows"] += count * int(eligible)
        distances = [POPCOUNT[original ^ (int(center) & 0xffff)]
                     for center in centers[:128]]
        running = 17
        prefix_best = {}
        for index, distance in enumerate(distances):
            if distance < running:
                running = distance
            if index + 1 in PREFIXES:
                prefix_best[index + 1] = running
        if original != 0:
            for prefix in PREFIXES:
                cumulative_all[prefix] += count * int(
                    prefix_best[prefix] == 0)
        if eligible:
            first = None
            for prefix in PREFIXES:
                hit = prefix_best[prefix] == 0
                cumulative_eligible[prefix] += count * int(hit)
                if first is None and hit:
                    first = prefix
            first_hit[first if first is not None else 0] += count
            row["q32_early_extra_prefix_tasks"] += count * int(
                prefix_best[16] != 0)

        if original == 0:
            continue
        best_distance = min(distances[:32])
        best_index = distances[:32].index(best_distance)
        use_pwp = 1 + best_distance < population
        row["active_rows"] += count
        row["bit_sparse_vector_ops_per_block"] += count * population
        row["candidate_vector_ops_per_block"] += count * (
            1 + best_distance if use_pwp else population)
        if use_pwp:
            selected = int(centers[best_index]) & 0xffff
            plus = original & ((~selected) & 0xffff)
            minus = selected & ((~original) & 0xffff)
            require(((selected | plus) & ((~minus) & 0xffff)) == original,
                    "M401 exact residual reconstruction failure")
            row["pwp_rows"] += count
            row["correction_ops_per_block"] += count * best_distance
            row["exact_reconstruction_rows"] += count
            used.add(best_index)
            row["narrow_block_descriptors_tile0"] += (
                count * int(np.count_nonzero(
                    narrow_flags[best_index, 0:4])))
            row["narrow_block_descriptors_tile1"] += (
                count * int(np.count_nonzero(
                    narrow_flags[best_index, 4:8])))
        else:
            row["fallback_rows"] += count
            row["correction_ops_per_block"] += count * population
    row["used_pwp_patterns"] = len(used)
    row["used_center_runs"] = count_runs(used)
    row["q32_reference_matcher_cycles"] = (
        row["source_rows"] + row["eligible_rows"] + 2)
    row["q32_early_matcher_cycles"] = (
        row["source_rows"] + row["q32_early_extra_prefix_tasks"] + 2)
    row["q32_early_saved_cycles"] = (
        row["eligible_rows"] - row["q32_early_extra_prefix_tasks"])
    require(row["source_rows"] == row["zero_rows"] + row["active_rows"] and
            row["active_rows"] == row["pwp_rows"] + row["fallback_rows"] and
            row["pwp_rows"] == row["exact_reconstruction_rows"],
            "M401 population conservation failure")
    require(row["narrow_block_descriptors_tile0"] +
            row["narrow_block_descriptors_tile1"] <=
            row["pwp_rows"] * 8,
            "M401 narrow descriptor overcount")
    return dict(row), first_hit, cumulative_all, cumulative_eligible


def baseline_sample(phases, command_setup, model):
    preprocess = max(
        model["rows_per_phase"] + model["popcount_filter_pipeline_cycles"],
        model["weight_phase_bytes"] // model["dram_bytes_per_cycle"] +
        command_setup)
    time = preprocess
    for index, phase in enumerate(phases):
        compute = phase["bit_sparse_vector_ops_per_block"] * 8
        next_preprocess = preprocess if index + 1 < len(phases) else 0
        time += max(compute, next_preprocess)
        time += model["tail_cycles"]
    time += model["commit_cycles_per_sample"]
    return time


def candidate_sample(phases, variant, command_setup, latency, model,
                     blocking=0.0, capture_phase_timestamps=False):
    elastic = variant in ("elastic_only", "combined")
    early = variant in ("early_only", "combined")
    config_bytes = (model["elastic_config_bytes"] if elastic else
                    model["anchor_config_bytes"])
    stride = (model["elastic_center_stride_bytes"] if elastic else
              model["anchor_center_stride_bytes"])
    time = 0.0
    components = Counter()
    timestamps = []
    maximum_slot = 0
    for phase_index, phase in enumerate(phases):
        config_data = int(math.ceil(
            config_bytes / float(model["dram_bytes_per_cycle"])))
        config = config_data + command_setup
        matcher = (phase["q32_early_matcher_cycles"] if early else
                   phase["q32_reference_matcher_cycles"])
        seal = 1
        start = time
        time += config + matcher + seal
        components["config_data"] += config_data
        components["config_command"] += command_setup
        components["matcher"] += matcher
        components["bitmap_seal"] += seal
        if phase["active_rows"] == 0:
            time += model["tail_cycles"]
            components["tail"] += model["tail_cycles"]
            continue
        tile_bytes = (model["weight_bytes_per_tile"] +
                      phase["used_pwp_patterns"] * stride)
        maximum_slot = max(maximum_slot, config_bytes + tile_bytes)
        require(config_bytes + tile_bytes <= model["tile_slot_bytes"],
                "M401 slot overflow")
        tile_data = tile_bytes // model["dram_bytes_per_cycle"]
        require(tile_data * model["dram_bytes_per_cycle"] == tile_bytes,
                "M401 unaligned tile DMA")
        tile_commands = 1 + phase["used_center_runs"]
        tile_dma = tile_data + tile_commands * command_setup
        current_replay_work = (
            4 * phase["correction_ops_per_block"] +
            8 * phase["pwp_rows"])
        work0 = current_replay_work
        work1 = current_replay_work
        if elastic:
            work0 -= phase["narrow_block_descriptors_tile0"]
            work1 -= phase["narrow_block_descriptors_tile1"]
        require(work0 >= phase["active_rows"] and
                work1 >= phase["active_rows"],
                "M401 descriptor service underflow")
        replay0 = work0 + latency + phase["active_rows"] * blocking
        replay1 = work1 + latency + phase["active_rows"] * blocking
        time += tile_dma
        tile0_replay_start = time
        tile0_replay_end = tile0_replay_start + replay0
        tile1_dma_end = tile0_replay_start + tile_dma
        tile1_replay_start = max(tile0_replay_end, tile1_dma_end)
        exposed_tile1_dma = max(0.0, tile1_dma_end - tile0_replay_end)
        time = tile1_replay_start + replay1 + model["tail_cycles"]
        components["tile0_dma_data"] += tile_data
        components["tile0_dma_commands"] += tile_commands * command_setup
        components["tile1_dma_exposed"] += exposed_tile1_dma
        components["replay0"] += replay0
        components["replay1"] += replay1
        components["active_compute"] += work0 + work1
        components["descriptor_sram_startup"] += 2 * latency
        components["blocking_descriptor_penalty"] += (
            2 * phase["active_rows"] * blocking)
        components["tail"] += model["tail_cycles"]
        components["pwp_physical_bytes"] += (
            phase["used_pwp_patterns"] * stride * 2)
        components["weight_bytes"] += model["weight_bytes_per_tile"] * 2
        components["tile_dma_commands"] += tile_commands * 2
        components["descriptor_reads_responses_bundles"] += (
            phase["active_rows"] * 2)
        if capture_phase_timestamps:
            timestamps.append({
                "phase_index": phase_index,
                "phase_start": start,
                "tile0_replay_start": tile0_replay_start,
                "tile0_replay_end": tile0_replay_end,
                "tile1_dma_end": tile1_dma_end,
                "exposed_tile1_dma": exposed_tile1_dma,
                "tile1_replay_start": tile1_replay_start,
                "phase_end": time,
            })
    time += model["commit_cycles_per_sample"]
    components["commit"] += model["commit_cycles_per_sample"]
    return {
        "cycles": time,
        "components": dict(components),
        "maximum_slot_bytes": maximum_slot,
        "timestamps": timestamps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M401 overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m401_h67_q32_elastic_pwp_full_replay_contract_v1" and
            contract.get("status") == "FROZEN_BEFORE_M401_EXECUTION",
            "M401 contract drift")
    hw_root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, identity in contract["inputs"].items():
        path = hw_root / identity["path"]
        require(path.is_file(), "missing M401 input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"],
                "M401 SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": identity["path"], "sha256": observed}

    trace = strict_json(paths["m40_trace"])
    catalog = strict_json(paths["m338_catalog"])
    bridge = strict_json(paths["m41_bridge"])
    m397 = strict_json(paths["m397_result"])
    m398 = strict_json(paths["m398_hammer"])
    m399 = strict_json(paths["m399_prereview"])
    m400 = strict_json(paths["m400_prereview"])
    m400a = strict_json(paths["m400a_arithmetic_hammer"])
    paper = contract["paper_identity"]
    require(trace["identity"]["checkpoint_sha256"] ==
            paper["checkpoint_sha256"] and
            trace["identity"]["bn_policy"] == "no_running",
            "M401 H67 identity drift")
    require(bridge["identity"]["checkpoint_sha256"] ==
            paper["checkpoint_sha256"], "M401 M41 identity drift")
    require(m398["severity_counts"]["P0"] == 0 and
            m398["severity_counts"]["P1"] == 0 and
            m399["decision"]["execute_full_phase_prefix_histogram"] ==
            "GO" and
            m400["decision"]["execute_full_phase_center_tile_replay"] ==
            "GO" and
            m400a["severity_counts"]["P0"] == 0 and
            m400a["severity_counts"]["P1"] == 0,
            "M401 prereview/admission drift")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    weight_paths = [paths["weight_o{}".format(index)]
                    for index in range(4)]
    flags, codec = build_static_codec(
        catalog, weight_paths, args.output_dir / "static_codec_audit.csv")
    gates = contract["execution_gates"]
    require(codec["blocks"] == 442368 and codec["lanes"] == 42467328 and
            codec["signed12_violations"] == 0 and
            codec["wide_reconstruction_mismatches"] == 0 and
            codec["narrow_reconstruction_mismatches"] == 0 and
            codec["nonzero_padding_bytes"] == 0 and
            codec["narrow_blocks"] == gates["expected_static_narrow_blocks"] and
            codec["maximum_absolute"] == gates["expected_pwp_maximum_absolute"],
            "M401 static codec gate failure")

    m43 = load_module(paths["m43_unpacker"], "m401_m43")
    trace_dir = paths["m40_trace"].parent
    operators = tuple(trace["cohort"]["operators"])
    operator_index = {name: index for index, name in enumerate(operators)}
    histograms = defaultdict(Counter)
    payload_files = 0
    payload_bytes = 0
    for record_index, record in enumerate(trace["records"]):
        for key, sha_key in (("packed_file", "packed_file_sha256"),
                             ("value_payload_file",
                              "value_payload_sha256")):
            path = trace_dir / record[key]
            require(path.is_file() and sha256(path) == record[sha_key],
                    "M401 M40 payload drift")
            payload_files += 1
            payload_bytes += path.stat().st_size
        masks = m43.unpack_record_masks(trace_dir, record)
        for source_row in range(m43.ROWS):
            base = source_row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * 16
                for subtile in range(16):
                    value = (value256 >> (subtile * 16)) & 0xffff
                    histograms[(record["sample_id"],
                                operator_index[record["operator"]],
                                partition_base + subtile)][value] += 1
        print("[M401 HIST] {}/{}".format(
            record_index + 1, len(trace["records"])), flush=True)

    phases = defaultdict(list)
    aggregate = Counter()
    first_hit_aggregate = Counter()
    cumulative_all_aggregate = Counter()
    cumulative_eligible_aggregate = Counter()
    phase_rows = []
    for sample in range(10):
        for operator in range(4):
            for partition in range(432):
                counter = histograms[(sample, operator, partition)]
                require(sum(counter.values()) == 3000,
                        "M401 phase extent drift")
                centers = [int(value, 16) for value in
                           catalog["operators"][operator]["partitions"]
                           [partition]["nested_patterns"]]
                phase, first_hit, cumulative_all, cumulative_eligible = (
                    analyze_phase(counter, centers,
                                  flags[operator][partition]))
                phases[sample].append(phase)
                aggregate.update(phase)
                first_hit_aggregate.update(first_hit)
                cumulative_all_aggregate.update(cumulative_all)
                cumulative_eligible_aggregate.update(cumulative_eligible)
                record = {
                    "sample": sample,
                    "operator": operator,
                    "partition": partition,
                    "active_rows": phase["active_rows"],
                    "eligible_rows": phase["eligible_rows"],
                    "pwp_rows": phase["pwp_rows"],
                    "fallback_rows": phase["fallback_rows"],
                    "used_pwp_patterns": phase["used_pwp_patterns"],
                    "used_center_runs": phase["used_center_runs"],
                    "narrow_tile0": phase[
                        "narrow_block_descriptors_tile0"],
                    "narrow_tile1": phase[
                        "narrow_block_descriptors_tile1"],
                    "reference_matcher": phase[
                        "q32_reference_matcher_cycles"],
                    "early_matcher": phase["q32_early_matcher_cycles"],
                    "early_saved": phase["q32_early_saved_cycles"],
                }
                for prefix in PREFIXES:
                    record["all_exact_q{}".format(prefix)] = cumulative_all[
                        prefix]
                    record["eligible_exact_q{}".format(prefix)] = (
                        cumulative_eligible[prefix])
                    record["first_hit_q{}".format(prefix)] = first_hit[
                        prefix]
                record["first_hit_none"] = first_hit[0]
                phase_rows.append(record)
        print("[M401 PHASE] sample={}/10".format(sample + 1), flush=True)

    phase_fields = ["sample", "operator", "partition", "active_rows",
                    "eligible_rows", "pwp_rows", "fallback_rows",
                    "used_pwp_patterns", "used_center_runs",
                    "narrow_tile0", "narrow_tile1", "reference_matcher",
                    "early_matcher", "early_saved"]
    for prefix in PREFIXES:
        phase_fields.extend(["all_exact_q{}".format(prefix),
                             "eligible_exact_q{}".format(prefix),
                             "first_hit_q{}".format(prefix)])
    phase_fields.append("first_hit_none")
    write_csv(args.output_dir / "per_phase_runtime_replay.csv",
              phase_rows, phase_fields)

    model = contract["cycle_model"]
    sweep_rows = []
    robust = {}
    robust_components = {}
    timestamp_rows = []
    for command_setup in contract["sweep"]["dma_command_setup_cycles"]:
        baseline = sum(baseline_sample(phases[sample], command_setup, model)
                       for sample in range(10))
        for latency in contract["sweep"]["descriptor_sram_latency_cycles"]:
            for variant in VARIANTS:
                sample_results = [candidate_sample(
                    phases[sample], variant, command_setup, latency, model,
                    capture_phase_timestamps=(
                        variant == "combined" and
                        command_setup == contract["decision_rule"]
                        ["robust_dma_command_setup_cycles"] and
                        latency == contract["decision_rule"]
                        ["robust_descriptor_sram_latency_cycles"]))
                    for sample in range(10)]
                candidate = sum(value["cycles"] for value in sample_results)
                sweep_row = {
                    "variant": variant,
                    "dma_command_setup_cycles": command_setup,
                    "descriptor_sram_latency_cycles": latency,
                    "baseline_cycles": baseline,
                    "candidate_cycles": candidate,
                    "speedup_vs_bit_sparse": baseline / float(candidate),
                    "maximum_slot_bytes": max(
                        value["maximum_slot_bytes"]
                        for value in sample_results),
                }
                sweep_rows.append(sweep_row)
                if (command_setup == contract["decision_rule"]
                        ["robust_dma_command_setup_cycles"] and
                        latency == contract["decision_rule"]
                        ["robust_descriptor_sram_latency_cycles"]):
                    robust[variant] = sweep_row
                    totals = Counter()
                    for sample_result in sample_results:
                        totals.update(sample_result["components"])
                    robust_components[variant] = dict(totals)
                    if variant == "combined":
                        for sample, sample_result in enumerate(sample_results):
                            for timestamp in sample_result["timestamps"]:
                                row = dict(timestamp)
                                row["sample"] = sample
                                timestamp_rows.append(row)

    write_csv(args.output_dir / "variant_cycle_sweep.csv", sweep_rows,
              ["variant", "dma_command_setup_cycles",
               "descriptor_sram_latency_cycles", "baseline_cycles",
               "candidate_cycles", "speedup_vs_bit_sparse",
               "maximum_slot_bytes"])
    write_csv(args.output_dir / "combined_phase_timestamps.csv",
              timestamp_rows,
              ["sample", "phase_index", "phase_start",
               "tile0_replay_start", "tile0_replay_end", "tile1_dma_end",
               "exposed_tile1_dma", "tile1_replay_start", "phase_end"])

    blocking_rows = []
    for penalty in contract["sweep"][
            "blocking_cycles_per_replayed_descriptor"]:
        baseline = sum(baseline_sample(
            phases[sample], contract["decision_rule"]
            ["robust_dma_command_setup_cycles"], model) for sample in range(10))
        candidate = sum(candidate_sample(
            phases[sample], "combined",
            contract["decision_rule"]["robust_dma_command_setup_cycles"],
            contract["decision_rule"]
            ["robust_descriptor_sram_latency_cycles"], model,
            blocking=penalty)["cycles"] for sample in range(10))
        blocking_rows.append({
            "blocking_cycles_per_replayed_descriptor": penalty,
            "baseline_cycles": baseline,
            "candidate_cycles": candidate,
            "speedup_vs_bit_sparse": baseline / float(candidate),
        })
    write_csv(args.output_dir / "combined_blocking_sensitivity.csv",
              blocking_rows,
              ["blocking_cycles_per_replayed_descriptor", "baseline_cycles",
               "candidate_cycles", "speedup_vs_bit_sparse"])

    anchor = robust["m397_anchor"]
    require(anchor["baseline_cycles"] == 742148386 and
            anchor["candidate_cycles"] == 669012336 and
            abs(anchor["speedup_vs_bit_sparse"] -
                m397["m394_q32_reproduction"]["speedup_vs_bit_sparse"]) < 1e-15,
            "M401 failed exact M397 anchor reproduction")
    p397 = m397["population_by_q"]["32"]
    for field in ("active_rows", "pwp_rows", "fallback_rows",
                  "correction_ops_per_block", "used_pwp_patterns",
                  "used_center_runs", "exact_reconstruction_rows"):
        require(aggregate[field] == p397[field],
                "M401 M397 population drift: " + field)
    for prefix in (16, 32, 64, 128):
        require(cumulative_all_aggregate[prefix] ==
                m397["population_by_q"][str(prefix)]["exact_pattern_hits"],
                "M401 M397 exact-hit drift")
    runtime_narrow = (aggregate["narrow_block_descriptors_tile0"] +
                      aggregate["narrow_block_descriptors_tile1"])
    require(runtime_narrow == gates["expected_runtime_narrow_descriptors"] and
            aggregate["used_pwp_patterns"] ==
            gates["expected_used_center_occurrences"],
            "M401 read-only expectation mismatch")
    combined = robust["combined"]
    decision = ("GO_ELASTIC_PWP_SELECTED_RTL" if
                combined["candidate_cycles"] <=
                contract["decision_rule"]["candidate_cycle_ceiling"] and
                combined["speedup_vs_bit_sparse"] >= 1.15 else
                "NO_GO_ELASTIC_PWP_SELECTED_RTL")

    result = {
        "schema": "m401_h67_q32_elastic_pwp_full_replay_v1",
        "status": "PASS_M401_H67_Q32_EXACT_ELASTIC_PWP_FULL_REPLAY",
        "identity": identities,
        "paper_identity": paper,
        "payload_audit": {"files_rehashed": payload_files,
                          "bytes_rehashed": payload_bytes,
                          "mismatches": 0},
        "static_codec": codec,
        "runtime_population": dict(aggregate),
        "prefix_first_hit_eligible": {
            ("q{}".format(prefix) if prefix else "no_exact"):
            first_hit_aggregate[prefix]
            for prefix in PREFIXES + (0,)},
        "prefix_cumulative_all_nonzero": {
            "q{}".format(prefix): cumulative_all_aggregate[prefix]
            for prefix in PREFIXES},
        "prefix_cumulative_eligible": {
            "q{}".format(prefix): cumulative_eligible_aggregate[prefix]
            for prefix in PREFIXES},
        "runtime_elastic": {
            "pwp_block_descriptors": aggregate["pwp_rows"] * 8,
            "narrow_block_descriptors": runtime_narrow,
            "wide_block_descriptors": (
                aggregate["pwp_rows"] * 8 - runtime_narrow),
            "narrow_fraction": runtime_narrow /
            float(aggregate["pwp_rows"] * 8),
            "used_center_occurrences": aggregate["used_pwp_patterns"],
            "anchor_center_stride_bytes": 576,
            "elastic_center_stride_bytes": 640,
            "config_bytes": 96,
            "worst_case_slot_bytes": 26720,
        },
        "robust_variants": robust,
        "robust_component_ledgers": robust_components,
        "sweep": sweep_rows,
        "blocking_sensitivity": blocking_rows,
        "m397_anchor_reproduction": {"exact": True,
                                     "baseline_cycles": 742148386,
                                     "candidate_cycles": 669012336},
        "decision": decision,
        "decision_rule": contract["decision_rule"],
        "execution_gates": {
            "input_or_payload_sha_mismatch": 0,
            "signed12_overflow": codec["signed12_violations"],
            "wide_or_narrow_lane_reconstruction_mismatch": (
                codec["wide_reconstruction_mismatches"] +
                codec["narrow_reconstruction_mismatches"]),
            "nonzero_high_sidecar_padding_byte": codec[
                "nonzero_padding_bytes"],
            "m397_q32_reproduction_mismatch": 0,
            "config_stride_or_dma_undercharge": 0,
            "all_17280_phases_complete": len(phase_rows) == 17280,
        },
        "admission": {
            "exact_arithmetic": True,
            "frozen_h67_trace_cycle_replay": True,
            "standalone_four_bottleneck_conv_module_cycles": True,
            "selected_rtl": False,
            "rtl_measured_speedup": False,
            "synopsys": False,
            "energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "output_files": {
            "static_codec_audit": "static_codec_audit.csv",
            "per_phase_runtime_replay": "per_phase_runtime_replay.csv",
            "variant_cycle_sweep": "variant_cycle_sweep.csv",
            "combined_phase_timestamps": "combined_phase_timestamps.csv",
            "combined_blocking_sensitivity":
                "combined_blocking_sensitivity.csv",
        },
    }
    output = args.output_dir / "m401_h67_q32_elastic_pwp_full_replay_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M401_PASS combined={:.9f}x cycles={} decision={}".format(
        combined["speedup_vs_bit_sparse"], combined["candidate_cycles"],
        decision), flush=True)


if __name__ == "__main__":
    main()
