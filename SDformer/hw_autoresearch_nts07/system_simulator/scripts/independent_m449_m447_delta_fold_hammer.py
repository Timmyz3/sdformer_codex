#!/usr/bin/env python3
"""Independent full-population audit of the M447 correction-fold DSE.

This auditor intentionally imports no M40/M401/M423/M430/M447 analyzer and
does not use an upstream phase row or aggregate to form a result.  It decodes
the frozen M40 packed positive planes, reconstructs all 51.84M 16-bit Conv3x3
source words, applies the frozen M430 q32 catalog, and independently replays
the K-fold schedules.  Upstream M430/M447 artifacts are read only after the
independent derivation, for mismatch accounting.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PARTITIONS = 432
ROWS_PER_PHASE = 3000
OUTPUT_BLOCKS = 8
SAMPLES = 10
OPERATORS = 4
FOLDS = (1, 2, 4)
ZERO_DIAGNOSTIC_FOLDS = (1, 2, 3, 4, 5)
POPCOUNT = np.asarray([bin(value).count("1") for value in range(1 << 16)],
                      dtype=np.uint8)


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def count_runs(indices):
    ordered = sorted(indices)
    if not ordered:
        return 0
    return 1 + sum(right != left + 1
                   for left, right in zip(ordered, ordered[1:]))


def centers(catalog, operator, partition):
    words = catalog["operators"][operator]["partitions"][partition][
        "nested_patterns"]
    require(len(words) >= 32, "catalog has fewer than q32 patterns")
    return np.asarray([int(word, 16) for word in words[:32]],
                      dtype=np.uint16)


def unpack_conv_words(trace_dir, record):
    """Independently form 3000 rows x 432 little-endian 16-bit words."""
    require(record["shape"] == [10, 1, 768, 15, 20],
            "M40 source shape drift")
    packed_path = trace_dir / record["packed_file"]
    value_path = trace_dir / record["value_payload_file"]
    require(sha256(packed_path) == record["packed_file_sha256"],
            "M40 packed payload SHA mismatch")
    require(sha256(value_path) == record["value_payload_sha256"],
            "M40 value payload SHA mismatch")
    raw = np.frombuffer(packed_path.read_bytes(), dtype=np.uint8)
    plane_bytes = int(record["positive_plane_bytes"])
    require(raw.size == 3 * plane_bytes, "M40 packed plane extent drift")
    require(not np.any(raw[plane_bytes:2 * plane_bytes]),
            "frozen source trace has a negative support bit")

    source = np.unpackbits(raw[:plane_bytes], bitorder="little")
    source = source[:10 * 768 * 15 * 20].reshape(10, 768, 15, 20)
    padded = np.pad(source, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")
    taps = np.stack(
        [padded[:, :, kernel_y:kernel_y + 15,
                kernel_x:kernel_x + 20]
         for kernel_y in range(3) for kernel_x in range(3)], axis=2)
    feature_rows = np.ascontiguousarray(
        taps.transpose(0, 3, 4, 1, 2).reshape(ROWS_PER_PHASE, 768 * 9))
    packed_features = np.packbits(feature_rows, axis=1, bitorder="little")
    words = np.ascontiguousarray(packed_features).view("<u2")
    require(words.shape == (ROWS_PER_PHASE, PARTITIONS),
            "independent convolution word extent drift")
    return words


def analyze_phase(words, catalog_centers):
    values, counts = np.unique(words, return_counts=True)
    values = values.astype(np.uint16)
    counts = counts.astype(np.int64)
    pops = POPCOUNT[values].astype(np.int16)
    distances = POPCOUNT[np.bitwise_xor(
        catalog_centers[:, None], values[None, :])].astype(np.int16)
    best_id = distances.argmin(axis=0)
    best_distance = distances[best_id, np.arange(values.size)]
    nonzero = values != 0
    eligible = pops >= 2
    use_pwp = nonzero & (1 + best_distance < pops)
    fallback = nonzero & (~use_pwp)
    q16_exact = distances[:16].min(axis=0) == 0
    correction = np.where(use_pwp, best_distance, pops).astype(np.int64)

    selected = catalog_centers[best_id]
    plus = values & np.bitwise_not(selected)
    minus = selected & np.bitwise_not(values)
    reconstructed = (selected | plus) & np.bitwise_not(minus)
    symbolic_mismatches = int(np.count_nonzero(
        reconstructed[use_pwp] != values[use_pwp]))
    residual_count_mismatches = int(np.count_nonzero(
        (POPCOUNT[plus] + POPCOUNT[minus])[use_pwp] !=
        best_distance[use_pwp]))
    disjoint_mismatches = int(np.count_nonzero((plus & minus)[use_pwp]))
    require(symbolic_mismatches == residual_count_mismatches ==
            disjoint_mismatches == 0,
            "independent exact bit-residual proof failed")

    used_ids = set(int(index) for index in np.unique(
        best_id[use_pwp & (counts > 0)]))
    result = {
        "source_rows": int(counts.sum()),
        "zero_rows": int(counts[~nonzero].sum()),
        "active_rows": int(counts[nonzero].sum()),
        "eligible_rows": int(counts[eligible].sum()),
        "pwp_rows": int(counts[use_pwp].sum()),
        "positive_residual_pwp_rows": int(
            counts[use_pwp & (best_distance > 0)].sum()),
        "exact_pwp_rows": int(counts[use_pwp & (best_distance == 0)].sum()),
        "fallback_rows": int(counts[fallback].sum()),
        "correction_source_terms": int(np.dot(counts, correction)),
        "bit_sparse_source_terms": int(np.dot(
            counts, pops.astype(np.int64))),
        "q32_early_extra_prefix_tasks": int(
            counts[eligible & (~q16_exact)].sum()),
        "used_pwp_patterns": len(used_ids),
        "used_center_runs": count_runs(used_ids),
        "symbolic_reconstruction_mismatches": symbolic_mismatches,
        "residual_count_mismatches": residual_count_mismatches,
        "residual_disjoint_mismatches": disjoint_mismatches,
    }
    result["early_matcher"] = (result["source_rows"] +
                               result["q32_early_extra_prefix_tasks"] + 2)

    distance_histogram = Counter()
    for value_index in range(values.size):
        if use_pwp[value_index]:
            path = "pwp"
        elif fallback[value_index]:
            path = "fallback"
        else:
            continue
        distance_histogram[(path, int(correction[value_index]))] += int(
            counts[value_index])

    for fold in FOLDS:
        folded = (correction + fold - 1) // fold
        separate = np.where(use_pwp, 1 + folded, folded)
        fused = np.where(use_pwp, np.maximum(1, folded), folded)
        result[f"k{fold}_separate_issues_per_block"] = int(np.dot(
            counts, separate.astype(np.int64)))
        result[f"k{fold}_fused_issues_per_block"] = int(np.dot(
            counts, fused.astype(np.int64)))
        result[f"k{fold}_folded_correction_descriptors"] = int(np.dot(
            counts, folded.astype(np.int64)))
        expected_saving = result["positive_residual_pwp_rows"]
        actual_saving = (result[f"k{fold}_separate_issues_per_block"] -
                         result[f"k{fold}_fused_issues_per_block"])
        require(actual_saving == expected_saving,
                "fused issue saving identity failed")

    for fold in ZERO_DIAGNOSTIC_FOLDS:
        zero_folded = np.where(nonzero, (pops + fold - 1) // fold, 0)
        result[f"zero_k{fold}_issues_per_block"] = int(np.dot(
            counts, zero_folded.astype(np.int64)))

    require(result["source_rows"] == ROWS_PER_PHASE and
            result["source_rows"] ==
            result["zero_rows"] + result["active_rows"] and
            result["active_rows"] ==
            result["pwp_rows"] + result["fallback_rows"] and
            result["pwp_rows"] ==
            result["positive_residual_pwp_rows"] +
            result["exact_pwp_rows"],
            "phase population conservation failure")
    return result, distance_histogram


def catalog_sample(phases, fold, fused, model, command_setup, latency,
                   ideal_correction_elimination=False):
    now = 0
    components = Counter()
    maximum_slot = 0
    for phase in phases:
        config_data = math.ceil(
            model["elastic_config_bytes"] / model["dram_bytes_per_cycle"])
        now += config_data + command_setup + phase["early_matcher"] + 1
        components["config_data"] += config_data
        components["config_command"] += command_setup
        components["matcher"] += phase["early_matcher"]
        components["bitmap_seal"] += 1
        if phase["active_rows"] == 0:
            now += model["tail_cycles"]
            components["tail"] += model["tail_cycles"]
            continue

        tile_bytes = (model["weight_bytes_per_tile"] +
                      phase["used_pwp_patterns"] *
                      model["elastic_center_stride_bytes"])
        maximum_slot = max(maximum_slot,
                           model["elastic_config_bytes"] + tile_bytes)
        require(model["elastic_config_bytes"] + tile_bytes <=
                model["tile_slot_bytes"], "catalog tile slot overflow")
        require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                "catalog tile DMA alignment drift")
        tile_data = tile_bytes // model["dram_bytes_per_cycle"]
        tile_command = (1 + phase["used_center_runs"]) * command_setup
        tile_dma = tile_data + tile_command

        if ideal_correction_elimination:
            per_block = phase["pwp_rows"]
        else:
            mode = "fused" if fused else "separate"
            per_block = phase[f"k{fold}_{mode}_issues_per_block"]
        work = model["output_blocks_per_tile"] * per_block
        replay = work + latency
        now += tile_dma
        tile0_end = now + replay
        tile1_dma_end = now + tile_dma
        tile1_start = max(tile0_end, tile1_dma_end)
        components["tile1_dma_exposed"] += max(0,
                                                    tile1_dma_end - tile0_end)
        now = tile1_start + replay + model["tail_cycles"]
        components["tile0_dma_data"] += tile_data
        components["tile0_dma_commands"] += tile_command
        components["replay0"] += replay
        components["replay1"] += replay
        components["active_compute"] += 2 * work
        components["descriptor_sram_startup"] += 2 * latency
        components["tail"] += model["tail_cycles"]
    now += model["commit_cycles_per_sample"]
    components["commit"] += model["commit_cycles_per_sample"]
    return int(now), components, maximum_slot


def zero_fold_sample(phases, fold, model, command_setup):
    """Same M401 zero-elided recurrence, with a charged K-way source fold."""
    preprocess = max(
        model["rows_per_phase"] + model["popcount_filter_pipeline_cycles"],
        model["weight_phase_bytes"] // model["dram_bytes_per_cycle"] +
        command_setup)
    now = preprocess
    preprocess_exposed = 0
    compute_cycles = 0
    for phase_index, phase in enumerate(phases):
        compute = phase[f"zero_k{fold}_issues_per_block"] * OUTPUT_BLOCKS
        next_preprocess = preprocess if phase_index + 1 < len(phases) else 0
        exposed = max(compute, next_preprocess)
        now += exposed + model["tail_cycles"]
        compute_cycles += compute
        preprocess_exposed += max(0, next_preprocess - compute)
    now += model["commit_cycles_per_sample"]
    return int(now), {"compute_cycles": compute_cycles,
                      "preprocess_exposed_cycles": preprocess_exposed,
                      "initial_preprocess_cycles": preprocess,
                      "tail_cycles": len(phases) * model["tail_cycles"],
                      "commit_cycles": model["commit_cycles_per_sample"]}


def signed_bits(minimum, maximum):
    for bits in range(2, 65):
        if minimum >= -(1 << (bits - 1)) and maximum <= (1 << (bits - 1)) - 1:
            return bits
    raise RuntimeError("signed range exceeds 64 bits")


def static_numeric_bounds(catalog, weight_paths):
    pwp_minimum = 1 << 30
    pwp_maximum = -(1 << 30)
    accumulator_abs_maximum = 0
    signed12_violations = 0
    centers_evaluated = 0
    lanes_evaluated = 0
    weight_minimum = 127
    weight_maximum = -128
    for operator, weight_path in enumerate(weight_paths):
        weights = np.fromfile(weight_path, dtype=np.int8)
        require(weights.size == 6912 * 768,
                "frozen INT8 weight extent drift")
        weights = weights.reshape(6912, 768).astype(np.int16)
        weight_minimum = min(weight_minimum, int(weights.min()))
        weight_maximum = max(weight_maximum, int(weights.max()))
        per_lane_abs = np.abs(weights.astype(np.int32)).sum(axis=0)
        accumulator_abs_maximum = max(accumulator_abs_maximum,
                                      int(per_lane_abs.max()))
        for partition in range(PARTITIONS):
            catalog_centers = centers(catalog, operator, partition)
            bits = np.asarray(
                [[(int(center) >> bit) & 1 for bit in range(16)]
                 for center in catalog_centers], dtype=np.int16)
            products = bits @ weights[partition * 16:(partition + 1) * 16]
            pwp_minimum = min(pwp_minimum, int(products.min()))
            pwp_maximum = max(pwp_maximum, int(products.max()))
            signed12_violations += int(np.count_nonzero(
                (products < -2048) | (products > 2047)))
            centers_evaluated += products.shape[0]
            lanes_evaluated += products.size
    correction_term_abs_bound = max(abs(weight_minimum), abs(weight_maximum),
                                    128)
    k4_correction_minimum = -4 * correction_term_abs_bound
    k4_correction_maximum = 4 * correction_term_abs_bound
    fused_minimum = pwp_minimum + k4_correction_minimum
    fused_maximum = pwp_maximum + k4_correction_maximum
    return {
        "weight_minimum": weight_minimum,
        "weight_maximum": weight_maximum,
        "pwp_minimum": pwp_minimum,
        "pwp_maximum": pwp_maximum,
        "pwp_maximum_absolute": max(abs(pwp_minimum), abs(pwp_maximum)),
        "pwp_required_signed_bits": signed_bits(pwp_minimum, pwp_maximum),
        "signed12_pwp_violations": signed12_violations,
        "k4_correction_conservative_minimum": k4_correction_minimum,
        "k4_correction_conservative_maximum": k4_correction_maximum,
        "k4_fused_conservative_minimum": fused_minimum,
        "k4_fused_conservative_maximum": fused_maximum,
        "k4_fused_required_signed_bits": signed_bits(fused_minimum,
                                                       fused_maximum),
        "signed13_is_sufficient":
            fused_minimum >= -4096 and fused_maximum <= 4095,
        "signed12_is_also_sufficient_under_conservative_bound":
            fused_minimum >= -2048 and fused_maximum <= 2047,
        "downstream_accumulator_abs_bound_from_weights":
            accumulator_abs_maximum,
        "downstream_accumulator_required_signed_bits":
            signed_bits(-accumulator_abs_maximum,
                        accumulator_abs_maximum),
        "catalog_centers_evaluated": centers_evaluated,
        "pwp_lanes_evaluated": lanes_evaluated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing output overwrite")
    source_path = Path(__file__).resolve()
    source_sha = sha256(source_path)
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m449_m447_independent_hammer_contract_v1" and
            contract["status"] == "FROZEN_BEFORE_INDEPENDENT_RECOMPUTATION",
            "M449 contract status drift")
    hw_root = args.contract.resolve().parents[1]
    require(contract["inputs"]["auditor"]["sha256"] == source_sha and
            (hw_root / contract["inputs"]["auditor"]["path"]).resolve() ==
            source_path, "M449 auditor identity drift")
    paths = {}
    for name, specification in contract["inputs"].items():
        path = hw_root / specification["path"]
        require(path.is_file() and sha256(path) == specification["sha256"],
                "M449 frozen input identity drift: " + name)
        paths[name] = path
    docs_before = sha256(paths["docs359"])

    trace = strict_json(paths["m40_manifest"])
    catalog = strict_json(paths["m430_catalog"])
    m430_contract = strict_json(paths["m430_contract"])
    require(trace["cohort"]["records"] == 40 and
            trace["cohort"]["samples"] == SAMPLES and
            len(trace["cohort"]["operators"]) == OPERATORS,
            "M40 cohort drift")
    require(catalog["status"] ==
            "PASS_M430_TRAIN_ONLY_DUALAWARE_Q32_FROZEN_BEFORE_HELDOUT" and
            catalog["split"]["runtime_or_validation_data_used"] is False,
            "M430 catalog freeze drift")
    operators = tuple(trace["cohort"]["operators"])
    require(tuple(catalog["geometry"]["operators"]) == operators,
            "catalog/runtime operator order drift")
    operator_index = {name: index for index, name in enumerate(operators)}

    phases_by_sample = [[] for _ in range(SAMPLES)]
    aggregate = Counter()
    distance_histogram = Counter()
    phase_rows = []
    symbolic_mismatches = 0
    payload_files = 0
    payload_bytes = 0
    records = sorted(trace["records"], key=lambda record: (
        int(record["sample_id"]), operator_index[record["operator"]]))
    trace_dir = paths["m40_manifest"].parent
    for record_index, record in enumerate(records):
        sample = int(record["sample_id"])
        operator = operator_index[record["operator"]]
        words = unpack_conv_words(trace_dir, record)
        payload_files += 2
        payload_bytes += ((trace_dir / record["packed_file"]).stat().st_size +
                          (trace_dir /
                           record["value_payload_file"]).stat().st_size)
        for partition in range(PARTITIONS):
            phase, histogram = analyze_phase(
                words[:, partition], centers(catalog, operator, partition))
            phases_by_sample[sample].append(phase)
            aggregate.update(phase)
            distance_histogram.update(histogram)
            symbolic_mismatches += (
                phase["symbolic_reconstruction_mismatches"] +
                phase["residual_count_mismatches"] +
                phase["residual_disjoint_mismatches"])
            phase_rows.append({
                "sample": sample,
                "operator": operator,
                "partition": partition,
                "source_rows": phase["source_rows"],
                "active_rows": phase["active_rows"],
                "pwp_rows": phase["pwp_rows"],
                "positive_residual_pwp_rows":
                    phase["positive_residual_pwp_rows"],
                "exact_pwp_rows": phase["exact_pwp_rows"],
                "fallback_rows": phase["fallback_rows"],
                "correction_source_terms": phase["correction_source_terms"],
                "bit_sparse_source_terms": phase["bit_sparse_source_terms"],
                "used_pwp_patterns": phase["used_pwp_patterns"],
                "used_center_runs": phase["used_center_runs"],
                "early_matcher": phase["early_matcher"],
                **{f"k{fold}_separate_issues_per_block":
                   phase[f"k{fold}_separate_issues_per_block"]
                   for fold in FOLDS},
                **{f"k{fold}_fused_issues_per_block":
                   phase[f"k{fold}_fused_issues_per_block"]
                   for fold in FOLDS},
                **{f"zero_k{fold}_issues_per_block":
                   phase[f"zero_k{fold}_issues_per_block"]
                   for fold in ZERO_DIAGNOSTIC_FOLDS},
            })
        print(f"[M449] independently decoded record {record_index + 1}/40",
              flush=True)

    require(len(phase_rows) == SAMPLES * OPERATORS * PARTITIONS and
            aggregate["source_rows"] == 51_840_000 and
            symbolic_mismatches == 0,
            "M449 full-population reconstruction failed")
    for sample, phases in enumerate(phases_by_sample):
        require(len(phases) == OPERATORS * PARTITIONS,
                f"sample {sample} phase extent drift")

    model = m430_contract["cycle_model"]
    command_setup = m430_contract["decision_rule"][
        "dma_command_setup_cycles"]
    latency = m430_contract["decision_rule"][
        "descriptor_sram_latency_cycles"]
    point_rows = []
    point_components = {}
    for fold in FOLDS:
        for fused in (False, True):
            cycle_total = 0
            components = Counter()
            max_slot = 0
            for phases in phases_by_sample:
                cycles_value, component, slot = catalog_sample(
                    phases, fold, fused, model, command_setup, latency)
                cycle_total += cycles_value
                components.update(component)
                max_slot = max(max_slot, slot)
            mode = "fused_delta_composer" if fused else "separate_fold"
            name = f"k{fold}_{mode}"
            point_rows.append({
                "name": name,
                "fold_k": fold,
                "fused": fused,
                "cycles": cycle_total,
                "correction_input_bytes_per_cycle": 96 * fold,
                "pwp_input_bytes_per_cycle": 160,
                "simultaneous_fused_input_bytes_per_cycle":
                    160 + 96 * fold if fused else None,
                "issues_per_block": aggregate[
                    f"k{fold}_{'fused' if fused else 'separate'}_issues_per_block"],
                "max_slot_bytes": max_slot,
            })
            point_components[name] = dict(components)

    zero_rows = []
    zero_cycles = {}
    zero_components = {}
    for fold in ZERO_DIAGNOSTIC_FOLDS:
        cycles_total = 0
        component = Counter()
        for phases in phases_by_sample:
            cycles_value, sample_component = zero_fold_sample(
                phases, fold, model, command_setup)
            cycles_total += cycles_value
            component.update(sample_component)
        zero_cycles[fold] = cycles_total
        zero_components[fold] = dict(component)
        zero_rows.append({
            "fold_k": fold,
            "cycles": cycles_total,
            "input_bytes_per_cycle": 96 * fold,
            "issues_per_block": aggregate[f"zero_k{fold}_issues_per_block"],
        })

    ideal_cycles = sum(catalog_sample(
        phases, 4, True, model, command_setup, latency,
        ideal_correction_elimination=True)[0] for phases in phases_by_sample)

    k1_separate = next(row for row in point_rows
                       if row["name"] == "k1_separate_fold")
    require(k1_separate["cycles"] == zero_cycles[1] -
            (zero_cycles[1] - k1_separate["cycles"]),
            "internal integer recurrence failure")
    for row in point_rows:
        row["speedup_vs_zero_k1"] = zero_cycles[1] / row["cycles"]
        row["speedup_vs_m430_k1_separate"] = (
            k1_separate["cycles"] / row["cycles"])
        equal_k_cycles = zero_cycles[row["fold_k"]]
        row["equal_k_zero_cycles"] = equal_k_cycles
        row["speedup_vs_equal_k_zero"] = equal_k_cycles / row["cycles"]
        if row["fused"]:
            byte_equivalent_k = row[
                "simultaneous_fused_input_bytes_per_cycle"] // 96
            row["byte_floor_equivalent_zero_k"] = byte_equivalent_k
            row["byte_floor_equivalent_zero_cycles"] = zero_cycles[
                byte_equivalent_k]
            row["speedup_vs_byte_floor_equivalent_zero"] = (
                zero_cycles[byte_equivalent_k] / row["cycles"])
        else:
            row["byte_floor_equivalent_zero_k"] = None
            row["byte_floor_equivalent_zero_cycles"] = None
            row["speedup_vs_byte_floor_equivalent_zero"] = None

    numeric_bounds = static_numeric_bounds(
        catalog, [paths[f"weight_o{operator}"]
                  for operator in range(OPERATORS)])
    require(numeric_bounds["signed13_is_sufficient"] and
            numeric_bounds["downstream_accumulator_required_signed_bits"] <= 19,
            "numeric width proof failed")

    # Upstream artifacts are parsed only after independent derivation.
    m430_result = strict_json(paths["m430_result"])
    m447_result = strict_json(paths["m447_result"])
    m430_phase = read_csv(paths["m430_phase_csv"])
    m447_phase = read_csv(paths["m447_phase_csv"])
    require(len(m430_phase) == len(m447_phase) == len(phase_rows),
            "upstream phase CSV extent drift")
    m430_phase_index = {(int(row["sample"]), int(row["operator"]),
                         int(row["partition"])): row for row in m430_phase}
    m447_phase_index = {(int(row["sample"]), int(row["operator"]),
                         int(row["partition"])): row for row in m447_phase}
    phase_mismatches = 0
    early_mismatches = 0
    for row in phase_rows:
        key = (row["sample"], row["operator"], row["partition"])
        up430 = m430_phase_index[key]
        up447 = m447_phase_index[key]
        early_mismatches += int(row["early_matcher"] !=
                                int(up430["early_matcher"]))
        comparisons = (
            ("pwp_rows", "pwp_rows"),
            ("positive_residual_pwp_rows", "positive_residual_pwp_rows"),
            ("correction_source_terms", "correction_source_terms"),
            ("k1_separate_issues_per_block", "k1_separate_issues_per_block"),
            ("k1_fused_issues_per_block", "k1_fused_issues_per_block"),
            ("k2_separate_issues_per_block", "k2_separate_issues_per_block"),
            ("k2_fused_issues_per_block", "k2_fused_issues_per_block"),
            ("k4_separate_issues_per_block", "k4_separate_issues_per_block"),
            ("k4_fused_issues_per_block", "k4_fused_issues_per_block"),
        )
        phase_mismatches += sum(
            row[left] != int(up447[right]) for left, right in comparisons)

    upstream_points = {row["name"]: row for row in m447_result["points"]}
    point_cycle_mismatches = sum(
        row["cycles"] != upstream_points[row["name"]]["cycles"]
        for row in point_rows)
    upstream_hist = {(row["path"], int(row["correction_distance"])):
                     int(row["rows"])
                     for row in read_csv(paths["m447_histogram_csv"])}
    histogram_mismatches = sum(
        distance_histogram[(path, distance)] !=
        upstream_hist.get((path, distance), 0)
        for path in ("pwp", "fallback") for distance in range(17))
    m430_crosscheck_mismatches = sum((
        k1_separate["cycles"] !=
        m430_result["comparisons"]["m430_catalog_dual_cycles"],
        zero_cycles[1] != m430_result["comparisons"]["strong_zero_cycles"],
        aggregate["source_rows"] !=
        m430_result["runtime_population"]["source_rows"],
        aggregate["pwp_rows"] !=
        m430_result["runtime_population"]["pwp_rows"],
        aggregate["fallback_rows"] !=
        m430_result["runtime_population"]["fallback_rows"],
        aggregate["correction_source_terms"] !=
        m430_result["runtime_population"]["correction_ops_per_block"],
    ))
    require(phase_mismatches == early_mismatches ==
            point_cycle_mismatches == histogram_mismatches ==
            m430_crosscheck_mismatches == 0,
            "M449 independent/upstream crosscheck mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "m449_phase_recomputation.csv", phase_rows,
              list(phase_rows[0].keys()))
    histogram_rows = [
        {"path": path, "correction_distance": distance,
         "rows": distance_histogram[(path, distance)]}
        for path in ("pwp", "fallback") for distance in range(17)]
    write_csv(args.output_dir / "m449_correction_distance_histogram.csv",
              histogram_rows, ["path", "correction_distance", "rows"])
    write_csv(args.output_dir / "m449_equal_k_zero_diagnostics.csv",
              zero_rows, list(zero_rows[0].keys()))
    write_csv(args.output_dir / "m449_six_point_recomputation.csv",
              point_rows, list(point_rows[0].keys()))

    result = {
        "schema": "m449_m447_independent_recomputation_v1",
        "status": "PASS_INDEPENDENT_FULL_POPULATION_WITH_RESOURCE_WARNING",
        "identity": {
            "contract": {"path": str(args.contract.resolve().relative_to(hw_root)),
                         "sha256": sha256(args.contract)},
            "auditor": {"path": str(source_path.relative_to(hw_root)),
                        "sha256": source_sha},
            "docs359_before": docs_before,
            "docs359_after": sha256(paths["docs359"]),
        },
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "independence": {
            "imported_upstream_analyzers": False,
            "used_upstream_derived_rows_to_form_result": False,
            "upstream_read_only_after_independent_derivation": True,
            "payload_files_rehashed": payload_files,
            "payload_bytes_rehashed": payload_bytes,
            "rows_recomputed": aggregate["source_rows"],
            "phases_recomputed": len(phase_rows),
        },
        "population": {
            key: int(aggregate[key]) for key in (
                "source_rows", "zero_rows", "active_rows", "pwp_rows",
                "positive_residual_pwp_rows", "exact_pwp_rows",
                "fallback_rows", "correction_source_terms",
                "bit_sparse_source_terms")
        },
        "issue_formula": {
            "pwp_separate": "1 + ceil(d/K)",
            "pwp_fused": "max(1, ceil(d/K))",
            "fallback_both": "ceil(popcount/K)",
            "per_positive_residual_pwp_saving_per_block": 1,
            "fused_delta_sequence":
                "first chunk=PWP+up_to_K_signed_corrections; later chunks=up_to_K_signed_corrections",
            "downstream_update":
                "new_psum=old_psum+sum(all emitted delta chunks)",
        },
        "six_points": point_rows,
        "point_components": point_components,
        "equal_k_zero_diagnostic": zero_rows,
        "equal_k_zero_components": zero_components,
        "ideal_correction_elimination": {
            "cycles": ideal_cycles,
            "speedup_vs_zero_k1": zero_cycles[1] / ideal_cycles,
            "exact_architectural_point": False,
            "warning": "Drops all fallback and residual correction work; an unattainable arithmetic lower-bound cycle diagnostic only.",
        },
        "numeric_width": numeric_bounds,
        "semantic_legality": {
            "full_population_symbolic_reconstruction_mismatches":
                symbolic_mismatches,
            "persistent_old_psum_is_preserved_by_stated_recurrence": True,
            "m426_overwrite_semantics_reused": False,
            "requires_downstream_accumulate_each_delta_chunk": True,
            "no_rtl_exists_for_this_composer": True,
        },
        "crosschecks": {
            "m430_crosscheck_mismatches": m430_crosscheck_mismatches,
            "m430_early_matcher_phase_mismatches": early_mismatches,
            "m447_phase_field_mismatches": phase_mismatches,
            "m447_point_cycle_mismatches": point_cycle_mismatches,
            "m447_distance_histogram_mismatches": histogram_mismatches,
        },
        "resource_model_gaps": {
            "k_distinct_correction_vectors_available_each_cycle": "assumed",
            "correction_sram_banking_replication_conflict_model": "absent",
            "simultaneous_pwp_and_correction_delivery": "assumed",
            "pwp_plus_k_correction_reduction_tree_area_delay_power": "absent",
            "composer_pipeline_latency_and_frequency_derating": "absent",
            "routing_and_physical_interconnect": "absent",
            "matched_sram_macro": "absent",
            "accumulator_one_delta_per_cycle": "assumed existing backend",
            "total_correction_source_bytes_reduced_by_folding": False,
        },
    }
    require(result["identity"]["docs359_before"] ==
            result["identity"]["docs359_after"] ==
            contract["inputs"]["docs359"]["sha256"],
            "docs359 changed during M449 audit")
    require(source_sha == sha256(source_path),
            "M449 auditor changed during execution")
    result_path = args.output_dir / "m449_independent_recomputation.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS_M449 rows={} phases={} k1={} zero1={} ideal={} docs359={}".format(
        aggregate["source_rows"], len(phase_rows), k1_separate["cycles"],
        zero_cycles[1], ideal_cycles, result["identity"]["docs359_after"]),
        flush=True)


if __name__ == "__main__":
    main()
