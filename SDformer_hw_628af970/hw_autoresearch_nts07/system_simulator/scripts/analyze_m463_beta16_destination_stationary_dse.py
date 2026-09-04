#!/usr/bin/env python3
"""M463 CPU-only beta={0,16} destination-stationary schedule DSE.

The frozen M430 catalog and row decisions are replayed without tuning.  The
cycle model is exact with respect to the stated recurrence, but beta=16 is a
lossy raw-Conv opportunity point and never an admitted hardware speedup.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np


POPCOUNT = np.asarray([bin(value).count("1") for value in range(1 << 16)],
                      dtype=np.uint8)
BETAS = (0, 16)
SELECTOR_SETUP = (0, 1, 2, 4)
PIPELINE_FILL_DRAIN = (0, 2, 4, 8)


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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_outer_seal(path):
    expected, name = Path(path).read_text(encoding="utf-8").strip().split(
        "  ", 1)
    require(name == "SHA256SUMS", "outer seal target drift")
    manifest = Path(path).parent / name
    require(manifest.is_file() and sha256(manifest) == expected,
            "outer seal digest mismatch: " + str(path))
    for line in manifest.read_text(encoding="utf-8").splitlines():
        inner, filename = line.split("  ", 1)
        require(sha256(manifest.parent / filename) == inner,
                "inner seal mismatch: " + filename)


def count_runs(indices):
    ordered = sorted(indices)
    return (0 if not ordered else
            1 + sum(current != previous + 1
                    for previous, current in zip(ordered, ordered[1:])))


def read_reference_phases(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample", "operator", "partition", "active_rows",
            "eligible_rows", "pwp_rows", "fallback_rows",
            "correction_ops_per_block", "used_pwp_patterns",
            "used_center_runs", "early_matcher",
        }
        require(reader.fieldnames is not None and
                required.issubset(set(reader.fieldnames)),
                "M430 phase CSV schema drift")
        for row in reader:
            parsed = {key: int(value) for key, value in row.items()}
            key = (parsed["sample"], parsed["operator"],
                   parsed["partition"])
            require(key not in rows, "duplicate M430 phase row")
            rows[key] = parsed
    require(len(rows) == 17280, "M430 phase CSV extent drift")
    return rows


def masks_to_matrix(masks, rows, tiles, partitions):
    require(len(masks) == rows * tiles, "M40 unpacked mask extent drift")
    result = np.empty((rows, partitions), dtype=np.uint16)
    for row in range(rows):
        base = row * tiles
        column = 0
        for tile in range(tiles):
            value = int(masks[base + tile])
            for subtile in range(16):
                if column < partitions:
                    result[row, column] = np.uint16(
                        (value >> (16 * subtile)) & 0xffff)
                    column += 1
        require(column == partitions, "partition matrix width drift")
    return result


def bit_matrix(values):
    values = np.asarray(values, dtype=np.uint16).reshape(-1, 1)
    shifts = np.arange(16, dtype=np.uint16).reshape(1, -1)
    return np.bitwise_and(np.right_shift(values, shifts), 1).astype(np.int16)


def select_rows(unique, centers):
    unique = np.asarray(unique, dtype=np.uint16)
    centers = np.asarray(centers, dtype=np.uint16)
    distances = POPCOUNT[np.bitwise_xor(unique[:, None], centers[None, :])]
    best_index = np.argmin(distances, axis=1).astype(np.int16)
    best_distance = distances[np.arange(unique.size), best_index].astype(
        np.int16)
    population = POPCOUNT[unique].astype(np.int16)
    use_pwp = np.logical_and(unique != 0, 1 + best_distance < population)
    selected = centers[best_index]
    correction = np.where(use_pwp, np.bitwise_xor(unique, selected), unique)
    correction = correction.astype(np.uint16)
    eligible = population >= 2
    q16_exact = np.min(distances[:, :16], axis=1) == 0
    return {
        "population": population,
        "use_pwp": use_pwp,
        "selected": selected,
        "best_index": best_index,
        "best_distance": best_distance,
        "correction": correction,
        "eligible": eligible,
        "q16_exact": q16_exact,
    }


def destination_cost(correction, keep_masks):
    correction = np.asarray(correction, dtype=np.uint16)
    keep_masks = np.asarray(keep_masks, dtype=np.uint16)
    require(keep_masks.shape == (8, 96), "keep-mask geometry drift")
    costs = np.empty((correction.size, 8), dtype=np.uint8)
    for block in range(8):
        costs[:, block] = np.max(
            POPCOUNT[np.bitwise_and(correction[:, None],
                                    keep_masks[block][None, :])], axis=1)
    return costs


def phase_metrics(unique, counts, centers, keep_by_beta, weight_partition):
    selected = select_rows(unique, centers)
    population = selected["population"]
    active = unique != 0
    use_pwp = selected["use_pwp"]
    correction = selected["correction"]
    counts64 = np.asarray(counts, dtype=np.int64)
    used = set(int(value) for value in selected["best_index"][use_pwp])
    result = {
        "source_rows": int(np.sum(counts64)),
        "active_rows": int(np.sum(counts64[active])),
        "eligible_rows": int(np.sum(counts64[selected["eligible"]])),
        "pwp_rows": int(np.sum(counts64[use_pwp])),
        "fallback_rows": int(np.sum(counts64[np.logical_and(active,
                                                              ~use_pwp)])),
        "correction_ops_per_block": int(np.sum(
            counts64 * POPCOUNT[correction].astype(np.int64))),
        "nonzero_correction_rows": int(np.sum(
            counts64[np.logical_and(active, correction != 0)])),
        "used_pwp_patterns": len(used),
        "used_center_runs": count_runs(used),
        "early_matcher": int(np.sum(counts64) + np.sum(
            counts64[np.logical_and(selected["eligible"],
                                    ~selected["q16_exact"])]) + 2),
    }
    support = bit_matrix(unique)
    center_bits = bit_matrix(selected["selected"])
    correction_coeff = np.where(use_pwp[:, None],
                                support - center_bits, support)
    for beta in BETAS:
        kept = np.where(np.abs(weight_partition.astype(np.int16)) > beta,
                        weight_partition, 0).astype(np.int16)
        direct = np.matmul(support, kept.reshape(16, 768).astype(np.int16))
        pwp = np.matmul(center_bits, kept.reshape(16, 768).astype(np.int16))
        pwp[~use_pwp, :] = 0
        reconstructed = pwp + np.matmul(
            correction_coeff, kept.reshape(16, 768).astype(np.int16))
        require(bool(np.array_equal(direct, reconstructed)),
                "PWP/direct pruned-dot miter mismatch beta={}".format(beta))
        costs = destination_cost(correction, keep_by_beta[beta])
        weighted = np.sum(costs.astype(np.int64) * counts64[:, None], axis=0)
        result["beta{}_correction_work_by_block".format(beta)] = [
            int(value) for value in weighted]
    all_keep = np.full((8, 96), 0xffff, dtype=np.uint16)
    dense_cost = destination_cost(correction, all_keep)
    require(bool(np.all(dense_cost == POPCOUNT[correction][:, None])),
            "dense-keep destination cost drift")
    dense_weighted = np.sum(
        dense_cost.astype(np.int64) * counts64[:, None], axis=0)
    result["dense_keep_correction_work_by_block"] = [
        int(value) for value in dense_weighted]
    return result


def replay(phases_by_sample, model, setup, fill_drain, setup_scope,
           correction_key):
    total_cycles = 0
    aggregate = Counter()
    rows = []
    maximum_slot = 0
    for sample in range(10):
        time = 0
        for phase_index, phase in enumerate(phases_by_sample[sample]):
            start = time
            config_data = int(math.ceil(
                model["elastic_config_bytes"] /
                float(model["dram_bytes_per_cycle"])))
            config_command = model["dma_command_setup_cycles"]
            matcher = phase["early_matcher"]
            seal = 1
            time += config_data + config_command + matcher + seal
            aggregate["config_data"] += config_data
            aggregate["config_command"] += config_command
            aggregate["matcher"] += matcher
            aggregate["bitmap_seal"] += seal
            active = phase["active_rows"]
            tile0_work = tile1_work = 0
            tile_dma = tile1_exposed = 0
            replay0 = replay1 = 0
            setup0 = setup1 = 0
            if active:
                tile_bytes = (model["weight_bytes_per_tile"] +
                              phase["used_pwp_patterns"] *
                              model["elastic_center_stride_bytes"])
                maximum_slot = max(maximum_slot,
                                   model["elastic_config_bytes"] + tile_bytes)
                require(model["elastic_config_bytes"] + tile_bytes <=
                        model["tile_slot_bytes"], "M463 tile slot overflow")
                require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                        "M463 tile DMA alignment drift")
                tile_data = tile_bytes // model["dram_bytes_per_cycle"]
                tile_commands = 1 + phase["used_center_runs"]
                tile_dma = (tile_data + tile_commands *
                            model["dma_command_setup_cycles"])
                correction = phase[correction_key]
                require(len(correction) == 8, "correction block extent drift")
                tile0_work = 4 * phase["pwp_rows"] + sum(correction[:4])
                tile1_work = 4 * phase["pwp_rows"] + sum(correction[4:])
                nonzero = phase["nonzero_correction_rows"]
                if setup_scope == "block_local":
                    setup0 = 4 * nonzero * setup
                    setup1 = 4 * nonzero * setup
                elif setup_scope == "shared_row_optimistic":
                    setup0 = nonzero * setup
                    setup1 = 0
                else:
                    require(setup_scope == "none" and setup == 0,
                            "invalid setup scope")
                replay0 = (tile0_work + setup0 + fill_drain +
                           model["descriptor_sram_latency_cycles"])
                replay1 = (tile1_work + setup1 + fill_drain +
                           model["descriptor_sram_latency_cycles"])
                time += tile_dma
                tile0_end = time + replay0
                tile1_dma_end = time + tile_dma
                tile1_start = max(tile0_end, tile1_dma_end)
                tile1_exposed = max(0, tile1_dma_end - tile0_end)
                time = tile1_start + replay1
                aggregate["tile0_dma_data"] += tile_data
                aggregate["tile0_dma_commands"] += (
                    tile_commands * model["dma_command_setup_cycles"])
                aggregate["tile1_dma_exposed"] += tile1_exposed
                aggregate["replay0"] += replay0
                aggregate["replay1"] += replay1
                aggregate["active_compute"] += tile0_work + tile1_work
                aggregate["selector_setup"] += setup0 + setup1
                aggregate["pipeline_fill_drain"] += 2 * fill_drain
                aggregate["descriptor_sram_startup"] += (
                    2 * model["descriptor_sram_latency_cycles"])
            tail = model["tail_cycles"]
            time += tail
            aggregate["tail"] += tail
            rows.append({
                "sample": sample,
                "phase_index": phase_index,
                "operator": phase["operator"],
                "partition": phase["partition"],
                "phase_start": start,
                "active_rows": active,
                "pwp_rows": phase["pwp_rows"],
                "nonzero_correction_rows": phase["nonzero_correction_rows"],
                "tile0_compute": tile0_work,
                "tile1_compute": tile1_work,
                "selector_setup_tile0": setup0,
                "selector_setup_tile1": setup1,
                "fill_drain_per_active_tile": fill_drain if active else 0,
                "tile_dma": tile_dma,
                "tile1_dma_exposed": tile1_exposed,
                "replay0": replay0,
                "replay1": replay1,
                "tail": tail,
                "phase_end": time,
            })
        time += model["commit_cycles_per_sample"]
        aggregate["commit"] += model["commit_cycles_per_sample"]
        total_cycles += time
    return {"cycles": int(total_cycles), "components": dict(aggregate),
            "phase_rows": rows, "maximum_slot_bytes": maximum_slot}


def histogram_quantile(histogram, offset, numerator, denominator):
    total = int(np.sum(histogram, dtype=np.int64))
    require(total > 0, "empty error histogram")
    target = (numerator * total + denominator - 1) // denominator
    index = int(np.searchsorted(np.cumsum(histogram, dtype=np.int64),
                                target, side="left"))
    return index - offset


def summarize_error(histogram, offset, scales, layer_bound_q):
    indices = np.nonzero(histogram)[0]
    values = indices.astype(np.int64) - offset
    counts = histogram[indices].astype(np.int64)
    elements = int(np.sum(counts, dtype=np.int64))
    signed_sum = int(np.sum(values * counts, dtype=np.int64))
    abs_sum = int(np.sum(np.abs(values) * counts, dtype=np.int64))
    square_sum = int(np.sum(values * values * counts, dtype=np.int64))
    return {
        "accumulators": elements,
        "nonzero_error_accumulators": int(elements - histogram[offset]),
        "nonzero_error_fraction": float((elements - histogram[offset]) /
                                         float(elements)),
        "minimum_q": int(values[0]),
        "maximum_q": int(values[-1]),
        "mean_signed_q": signed_sum / float(elements),
        "mean_absolute_q": abs_sum / float(elements),
        "rmse_q": math.sqrt(square_sum / float(elements)),
        "p50_signed_q": histogram_quantile(histogram, offset, 50, 100),
        "p95_signed_q": histogram_quantile(histogram, offset, 95, 100),
        "p99_signed_q": histogram_quantile(histogram, offset, 99, 100),
        "maximum_sumabs_bound_q": int(np.max(layer_bound_q)),
        "minimum_sumabs_bound_q": int(np.min(layer_bound_q)),
        "mean_sumabs_bound_q": float(np.mean(layer_bound_q)),
        "maximum_sumabs_bound_scaled": float(np.max(
            layer_bound_q.astype(np.float64) * scales.astype(np.float64))),
        "bound_violations": 0,
    }


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_double_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(output_dir / name), name)
        for name in sorted(names)), encoding="utf-8")
    outer = output_dir / "SHA256SUMS.seal.sha256"
    outer.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                     encoding="utf-8")
    return manifest, outer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M463 overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m463_beta16_destination_stationary_dse_contract_v1" and
            contract.get("status") ==
            "FROZEN_BEFORE_UNIQUE_CPU_ONLY_M40_REPLAY",
            "M463 contract identity drift")
    root = args.contract.resolve().parents[1]
    script_start = sha256(Path(__file__).resolve())
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M463 input identity drift: " + name)
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": spec["sha256"]}
    require(paths["analyzer"].resolve() == Path(__file__).resolve() and
            identities["analyzer"]["sha256"] == script_start,
            "M463 analyzer self identity drift")
    for name in contract["outer_seals_to_verify"]:
        verify_outer_seal(paths[name])

    trace = strict_json(paths["m40_trace"])
    m41 = strict_json(paths["m41_result"])
    m418 = strict_json(paths["m418_result"])
    m430_contract = strict_json(paths["m430_contract"])
    m430 = strict_json(paths["m430_result"])
    catalog = strict_json(paths["m430_catalog"])
    require(trace["identity"]["checkpoint_sha256"] ==
            m430["paper_identity"]["checkpoint_sha256"] ==
            m41["identity"]["checkpoint_sha256"] and
            trace["identity"]["bn_policy"] == "no_running",
            "M463 checkpoint/BN identity drift")
    require(m418["component_conservation"]
            ["zero_elided_bit_sparse_exact_reproduction"]["cycles"] ==
            m430["comparisons"]["strong_zero_cycles"] and
            m430["status"] ==
            "PASS_M430B_ONE_COMPLETED_M40_HELDOUT_DUAL_REPLAY" and
            catalog["status"] ==
            "PASS_M430_TRAIN_ONLY_DUALAWARE_Q32_FROZEN_BEFORE_HELDOUT",
            "M463 upstream admission drift")
    require(tuple(contract["fixed_axis"]["beta_q"]) == BETAS and
            tuple(contract["fixed_axis"]["selector_setup_cycles"]) ==
            SELECTOR_SETUP and
            tuple(contract["fixed_axis"]["pipeline_fill_drain_cycles"]) ==
            PIPELINE_FILL_DRAIN,
            "M463 fixed DSE axis drift")

    weight_paths = [paths["weight_o{}".format(index)] for index in range(4)]
    weights = []
    scales = []
    keep_masks = {beta: [] for beta in BETAS}
    storage_rows = []
    cover_rows = []
    pwp_allzero = {beta: 0 for beta in BETAS}
    bound_by_operator = []
    for operator, weight_path in enumerate(weight_paths):
        raw = np.fromfile(str(weight_path), dtype=np.int8)
        require(raw.size == 768 * 768 * 3 * 3,
                "M41 weight extent drift")
        weight = raw.reshape(432, 16, 8, 96)
        weights.append(weight)
        scale_path = paths["scale_o{}".format(operator)]
        scale = np.fromfile(str(scale_path), dtype="<f4")
        require(scale.size == 768 and bool(np.all(np.isfinite(scale))) and
                bool(np.all(scale > 0)), "M41 scale payload drift")
        scales.append(scale)
        for beta in BETAS:
            kept = np.abs(weight.astype(np.int16)) > beta
            masks = np.zeros((432, 8, 96), dtype=np.uint16)
            for source in range(16):
                masks |= (kept[:, source, :, :].astype(np.uint16) << source)
            keep_masks[beta].append(masks)
            retained = int(np.count_nonzero(kept))
            dense_bytes = int(weight.size)
            bitmap_bytes = 432 * 16 * 8 * 12
            encoded = bitmap_bytes + retained
            maximum_sources = np.max(np.sum(kept, axis=1), axis=2)
            storage_rows.append({
                "operator": operator, "beta_q": beta,
                "dense_int8_bytes": dense_bytes,
                "bitmap_bytes": bitmap_bytes,
                "retained_int8_bytes": retained,
                "encoded_bytes": encoded,
            })
            for capacity in (3, 4, 8, 12, 15, 16):
                cover_rows.append({
                    "operator": operator, "beta_q": beta,
                    "source_capacity": capacity,
                    "blocks_covered": int(np.count_nonzero(
                        maximum_sources <= capacity)),
                    "blocks_total": 432 * 8,
                })
            pruned = np.where(kept, weight, 0).astype(np.int16)
            centers_by_partition = catalog["operators"][operator]["partitions"]
            for partition in range(432):
                centers = np.asarray([
                    int(value, 16) for value in
                    centers_by_partition[partition]["nested_patterns"][:32]
                ], dtype=np.uint16)
                center_bits = bit_matrix(centers)
                pwp = np.matmul(center_bits,
                                pruned[partition].reshape(16, 768)).reshape(
                                    32, 8, 96)
                pwp_allzero[beta] += int(np.count_nonzero(
                    np.all(pwp == 0, axis=2)))
        dropped16 = np.where(np.abs(weight.astype(np.int16)) <= 16,
                             weight, 0).astype(np.int16)
        bound = np.sum(np.abs(dropped16.astype(np.int32)), axis=(0, 1))
        bound_by_operator.append(bound.reshape(768))

    storage_totals = {}
    for beta in BETAS:
        rows = [row for row in storage_rows if row["beta_q"] == beta]
        storage_totals[str(beta)] = {
            key: sum(row[key] for row in rows) for key in (
                "dense_int8_bytes", "bitmap_bytes", "retained_int8_bytes",
                "encoded_bytes")
        }
    cover_totals = {}
    for beta in BETAS:
        cover_totals[str(beta)] = {}
        for capacity in (3, 4, 8, 12, 15, 16):
            cover_totals[str(beta)][str(capacity)] = sum(
                row["blocks_covered"] for row in cover_rows
                if row["beta_q"] == beta and
                row["source_capacity"] == capacity)
    require(storage_totals["16"]["encoded_bytes"] == 14283256 and
            cover_totals["16"]["3"] == 119 and
            cover_totals["16"]["4"] == 2694 and
            cover_totals["16"]["16"] == 13824 and
            cover_totals["16"]["15"] == 13824 - 891 and
            pwp_allzero[16] == 0,
            "M463 static beta16 census gate drift")

    marker = root / contract["one_shot"]["marker_path"]
    require(not marker.exists(), "M463 one-shot marker already exists")
    marker.write_text(
        "M463 unique CPU-only M40 replay consumed before first payload read.\n"
        "Analyzer SHA256: {}\nCatalog SHA256: {}\n"
        "No rerun or tuning is authorized by this marker.\n".format(
            script_start, identities["m430_catalog"]["sha256"]),
        encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    reference = read_reference_phases(paths["m430_phase_csv"])
    m43 = load_module(paths["m43_unpacker"], "m463_m43")
    operators = tuple(trace["cohort"]["operators"])
    require(tuple(catalog["geometry"]["operators"]) == operators and
            len(operators) == 4, "M463 operator identity drift")
    operator_index = {name: index for index, name in enumerate(operators)}
    trace_dir = paths["m40_trace"].parent
    phases_by_sample = defaultdict(list)
    phase_rows = []
    aggregates = Counter()
    reference_mismatches = 0
    pwp_miter_mismatches = 0
    payload_files = payload_bytes = 0

    max_bound = max(int(np.max(bound)) for bound in bound_by_operator)
    error_offset = max_bound
    error_histograms = [np.zeros(2 * max_bound + 1, dtype=np.int64)
                        for _ in range(4)]
    bits = np.arange(16, dtype=np.uint16).reshape(1, 1, 16)
    for record_index, record in enumerate(trace["records"]):
        operator = operator_index[record["operator"]]
        sample = int(record["sample_id"])
        for key, sha_key in (("packed_file", "packed_file_sha256"),
                             ("value_payload_file", "value_payload_sha256")):
            path = trace_dir / record[key]
            require(path.is_file() and sha256(path) == record[sha_key],
                    "M463 payload identity drift")
            payload_files += 1
            payload_bytes += path.stat().st_size
        unpacked = m43.unpack_record_masks(trace_dir, record)
        matrix = masks_to_matrix(unpacked, m43.ROWS, m43.TILES, 432)
        del unpacked
        support = np.bitwise_and(
            np.right_shift(matrix[:, :, None], bits), 1).reshape(
                m43.ROWS, 6912).astype(np.float32)
        dropped = np.where(np.abs(weights[operator].astype(np.int16)) <= 16,
                           weights[operator], 0).reshape(6912, 768)
        error = -np.matmul(support, dropped.astype(np.float32))
        require(bool(np.all(error == np.rint(error))),
                "M463 integer error GEMM lost exactness")
        error_i32 = error.astype(np.int32)
        bound = bound_by_operator[operator]
        require(bool(np.all(np.abs(error_i32) <= bound[None, :])),
                "M463 sumabs error bound violation")
        histogram = np.bincount(
            error_i32.reshape(-1).astype(np.int64) + error_offset,
            minlength=2 * max_bound + 1)
        error_histograms[operator] += histogram.astype(np.int64)
        del support, dropped, error, error_i32, histogram

        for partition in range(432):
            unique, counts = np.unique(matrix[:, partition],
                                       return_counts=True)
            centers = np.asarray([
                int(value, 16) for value in
                catalog["operators"][operator]["partitions"][partition]
                ["nested_patterns"][:32]], dtype=np.uint16)
            keep = {beta: keep_masks[beta][operator][partition]
                    for beta in BETAS}
            phase = phase_metrics(unique, counts, centers, keep,
                                  weights[operator][partition])
            phase.update({"sample": sample, "operator": operator,
                          "partition": partition})
            frozen = reference[(sample, operator, partition)]
            for key in ("active_rows", "eligible_rows", "pwp_rows",
                        "fallback_rows", "correction_ops_per_block",
                        "used_pwp_patterns", "used_center_runs",
                        "early_matcher"):
                reference_mismatches += int(phase[key] != frozen[key])
            phases_by_sample[sample].append(phase)
            aggregates.update({key: phase[key] for key in (
                "source_rows", "active_rows", "pwp_rows", "fallback_rows",
                "correction_ops_per_block", "nonzero_correction_rows")})
            phase_rows.append({
                "sample": sample, "operator": operator,
                "partition": partition,
                "active_rows": phase["active_rows"],
                "pwp_rows": phase["pwp_rows"],
                "fallback_rows": phase["fallback_rows"],
                "nonzero_correction_rows":
                    phase["nonzero_correction_rows"],
                "dense_keep_correction_work": sum(
                    phase["dense_keep_correction_work_by_block"]),
                "beta0_correction_work": sum(
                    phase["beta0_correction_work_by_block"]),
                "beta16_correction_work": sum(
                    phase["beta16_correction_work_by_block"]),
                "early_matcher": phase["early_matcher"],
            })
        print("[M463 CPU] record={}/40 sample={} operator={}".format(
            record_index + 1, sample, operator), flush=True)

    require(len(phase_rows) == 17280 and reference_mismatches == 0 and
            aggregates["source_rows"] == 51840000 and
            aggregates["pwp_rows"] ==
            m430["runtime_population"]["pwp_rows"] and
            aggregates["correction_ops_per_block"] ==
            m430["runtime_population"]["correction_ops_per_block"],
            "M463 full-population M430 crosscheck failed")
    for sample in range(10):
        phases_by_sample[sample].sort(
            key=lambda row: (row["operator"], row["partition"]))
        require(len(phases_by_sample[sample]) == 1728,
                "M463 sample phase extent drift")

    model = dict(m430_contract["cycle_model"])
    model["dma_command_setup_cycles"] = m430_contract[
        "decision_rule"]["dma_command_setup_cycles"]
    model["descriptor_sram_latency_cycles"] = m430_contract[
        "decision_rule"]["descriptor_sram_latency_cycles"]
    dense = replay(phases_by_sample, model, 0, 0, "none",
                   "dense_keep_correction_work_by_block")
    beta0 = replay(phases_by_sample, model, 0, 0, "none",
                   "beta0_correction_work_by_block")
    beta16_free = replay(phases_by_sample, model, 0, 0, "none",
                         "beta16_correction_work_by_block")
    require(dense["cycles"] == 517041352 and
            dense["cycles"] ==
            m430["comparisons"]["m430_catalog_dual_cycles"] and
            beta16_free["cycles"] == 437037880,
            "M463 recurrence anchor drift")
    expected_components = m430["component_ledger"]
    for key in ("config_data", "config_command", "matcher", "bitmap_seal",
                "tile0_dma_data", "tile0_dma_commands",
                "tile1_dma_exposed", "replay0", "replay1",
                "active_compute", "descriptor_sram_startup", "tail",
                "commit"):
        require(dense["components"].get(key, 0) ==
                expected_components.get(key, 0),
                "M463 dense recurrence component drift: " + key)

    sensitivity = []
    for scope in ("block_local", "shared_row_optimistic"):
        for setup in SELECTOR_SETUP:
            for fill in PIPELINE_FILL_DRAIN:
                point = replay(phases_by_sample, model, setup, fill, scope,
                               "beta16_correction_work_by_block")
                sensitivity.append({
                    "setup_scope": scope,
                    "selector_setup_cycles": setup,
                    "fill_drain_cycles_per_active_tile": fill,
                    "cycles": point["cycles"],
                    "speedup_vs_m430": dense["cycles"] /
                    float(point["cycles"]),
                    "speedup_vs_strong_zero":
                    m430["comparisons"]["strong_zero_cycles"] /
                    float(point["cycles"]),
                    "passes_1p10_m430_integer_gate":
                    point["cycles"] <=
                    contract["decision_rule"]["maximum_cycles_for_1p10"],
                    "conditional_go_eligible": (
                        scope == "block_local" and setup >= 1 and
                        point["cycles"] <=
                        contract["decision_rule"]
                        ["maximum_cycles_for_1p10"]),
                    "selector_area_timing_priced": False,
                })
    block_realistic = [row for row in sensitivity
                       if row["setup_scope"] == "block_local" and
                       row["selector_setup_cycles"] >= 1]
    best_block_realistic = min(block_realistic,
                               key=lambda row: row["cycles"])
    conditional = best_block_realistic["passes_1p10_m430_integer_gate"]

    error_summaries = []
    global_hist = np.zeros_like(error_histograms[0])
    for operator in range(4):
        summary = summarize_error(error_histograms[operator], error_offset,
                                  scales[operator],
                                  bound_by_operator[operator])
        summary["operator"] = operator
        summary["operator_name"] = operators[operator]
        error_summaries.append(summary)
        global_hist += error_histograms[operator]
    global_bound = np.concatenate(bound_by_operator)
    global_scale = np.concatenate(scales)
    global_error = summarize_error(global_hist, error_offset, global_scale,
                                   global_bound)

    phase_csv = "m463_per_phase_destination_stationary_work.csv"
    sensitivity_csv = "m463_selector_fill_sensitivity.csv"
    storage_csv = "m463_static_weight_encoding.csv"
    cover_csv = "m463_static_block_source_cover.csv"
    error_csv = "m463_beta16_raw_conv_error_summary.csv"
    timestamp_csv = "m463_beta16_free_per_phase_timestamps.csv"
    write_csv(args.output_dir / phase_csv, phase_rows,
              list(phase_rows[0].keys()))
    write_csv(args.output_dir / sensitivity_csv, sensitivity,
              list(sensitivity[0].keys()))
    write_csv(args.output_dir / storage_csv, storage_rows,
              list(storage_rows[0].keys()))
    write_csv(args.output_dir / cover_csv, cover_rows,
              list(cover_rows[0].keys()))
    write_csv(args.output_dir / error_csv, error_summaries,
              list(error_summaries[0].keys()))
    write_csv(args.output_dir / timestamp_csv, beta16_free["phase_rows"],
              list(beta16_free["phase_rows"][0].keys()))

    result = {
        "schema": "m463_beta16_destination_stationary_dse_v1",
        "status": "PASS_M463_CPU_ONLY_EXACT_SCHEDULE_DSE",
        "identity": {"analyzer": {
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": script_start}, **identities},
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "population": {
            "phases": len(phase_rows), "source_rows":
                aggregates["source_rows"],
            "active_rows": aggregates["active_rows"],
            "pwp_rows": aggregates["pwp_rows"],
            "fallback_rows": aggregates["fallback_rows"],
            "correction_source_terms_per_block":
                aggregates["correction_ops_per_block"],
            "nonzero_correction_or_fallback_rows":
                aggregates["nonzero_correction_rows"],
        },
        "static_weight_encoding": {
            "definition": "per 96-vector: 96-bit bitmap plus packed retained signed INT8 values",
            "keep_rule": "beta0:q!=0; beta16:abs(q)>16",
            "totals": storage_totals,
            "storage_proxy_only_not_energy": True,
            "beta16_blocks_total": 13824,
            "beta16_blocks_source_capacity_le_3": cover_totals["16"]["3"],
            "beta16_blocks_source_capacity_le_4": cover_totals["16"]["4"],
            "beta16_blocks_requiring_all_16_sources": 891,
            "beta16_pruned_pwp_allzero_vectors": pwp_allzero[16],
        },
        "cycle_points": {
            "dense_keep_control": {
                "cycles": dense["cycles"],
                "exactly_reproduces_m430": True,
                "components": dense["components"],
            },
            "beta0_exact_destination_stationary": {
                "cycles": beta0["cycles"],
                "speedup_vs_m430": dense["cycles"] / float(beta0["cycles"]),
                "accuracy_loss": False,
                "selector_area_timing_priced": False,
            },
            "beta16_free_selector_optimistic": {
                "cycles": beta16_free["cycles"],
                "speedup_vs_m430":
                    dense["cycles"] / float(beta16_free["cycles"]),
                "speedup_vs_strong_zero":
                    m430["comparisons"]["strong_zero_cycles"] /
                    float(beta16_free["cycles"]),
                "components": beta16_free["components"],
                "hardware_admitted": False,
                "reason": "96 independent selectors/address muxes have zero setup, area, timing, and energy charge",
            },
        },
        "selector_sensitivity": {
            "block_local_is_resource_fair_decision_scope": True,
            "shared_row_scope":
                "optimistic_unrealizable_without_8_block_context_or_overlap_proof",
            "hard_gate_cycles":
                contract["decision_rule"]["maximum_cycles_for_1p10"],
            "best_block_local_with_nonzero_setup": best_block_realistic,
            "points": sensitivity,
        },
        "numeric_error": {
            "semantics": "y_keep-y_orig=-sum(active signed-support * q_drop); frozen M40 negative planes are zero, so every active support code is +1",
            "pwp_correction_miter":
                "for every unique row/phase and beta, pruned PWP plus signed residual equals direct W_keep dot",
            "pwp_miter_mismatches": pwp_miter_mismatches,
            "integer_accumulator_global": global_error,
            "per_operator": error_summaries,
            "sumabs_bound_definition":
                "per output channel sum_i(abs(q_i)) for abs(q_i)<=16",
            "raw_conv_integer_only": True,
            "post_bn_or_end_to_end_accuracy": False,
        },
        "execution_gates": {
            "payload_files_rehashed": payload_files,
            "payload_bytes_rehashed": payload_bytes,
            "payload_sha_mismatches": 0,
            "m430_phase_field_mismatches": reference_mismatches,
            "dense_keep_cycle_mismatch": 0,
            "dense_keep_component_mismatches": 0,
            "beta16_free_cycle_mismatch": 0,
            "sumabs_bound_violations": 0,
            "pwp_direct_pruned_dot_mismatches": pwp_miter_mismatches,
            "unique_m40_payload_replays_this_milestone": 1,
        },
        "decision": {
            "decision": ("CONDITIONAL_GO_BUILD_MICROARCH_AND_DC_SCREEN"
                         if conditional else
                         "NO_GO_SELECTOR_RESOURCE_GATE"),
            "reason": ("A block-local nonzero-selector-setup point crosses the frozen integer 1.10x gate; RTL/VCS/DC is still required."
                       if conditional else
                       "Every resource-fair block-local nonzero-selector-setup point misses the frozen integer 1.10x gate."),
            "free_selector_or_shared_row_can_authorize_hardware": False,
            "rtl": False, "synopsys": False,
            "resource_normalized_speedup": False,
            "system_speedup": False, "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    result_name = "m463_beta16_destination_stationary_dse_r1.json"
    readme_name = "README.md"
    (args.output_dir / result_name).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / readme_name).write_text(
        "# M463 beta16 destination-stationary CPU DSE\n\n"
        "The dense-keep control exactly reconstructs M430.  Beta16 free-selector "
        "is an optimistic upper bound; only block-local nonzero-setup rows drive "
        "the decision.  All cycles remain four-Conv simulator results, not RTL, "
        "system speedup, power, energy, or paper-ready PPA.\n",
        encoding="utf-8")
    require(sha256(Path(__file__).resolve()) == script_start,
            "M463 analyzer changed during one-shot replay")
    names = [result_name, readme_name, phase_csv, sensitivity_csv,
             storage_csv, cover_csv, error_csv, timestamp_csv]
    manifest, outer = write_double_seal(args.output_dir, names)
    print("PASS_M463_CPU_ONLY_EXACT_SCHEDULE_DSE dense={} beta0={} beta16_free={} best_block={} decision={} manifest={} outer={}".format(
        dense["cycles"], beta0["cycles"], beta16_free["cycles"],
        best_block_realistic["cycles"], result["decision"]["decision"],
        sha256(manifest), sha256(outer)), flush=True)


if __name__ == "__main__":
    main()
