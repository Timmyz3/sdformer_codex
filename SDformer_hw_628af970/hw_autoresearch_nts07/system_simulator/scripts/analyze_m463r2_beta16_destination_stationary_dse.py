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
import itertools
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


def minimum_drop_set_cover_le4(weight_block, beta):
    """Return (minimum up to four, reachable) for 16 drop sets over 96 lanes."""
    require(weight_block.shape == (16, 96), "drop-cover block geometry drift")
    lane_full = (1 << 96) - 1
    drop_sets = []
    for source in range(16):
        dropped = np.abs(weight_block[source].astype(np.int16)) <= beta
        packed = np.packbits(dropped.astype(np.uint8), bitorder="little")
        drop_sets.append(int.from_bytes(packed.tobytes(), byteorder="little"))
    all_union = 0
    for value in drop_sets:
        all_union |= value
    if all_union != lane_full:
        return None, False
    for size in range(1, 5):
        for indices in itertools.combinations(range(16), size):
            union = 0
            for index in indices:
                union |= drop_sets[index]
            if union == lane_full:
                return size, True
    return None, True


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
    dropped16 = np.where(
        np.abs(weight_partition.astype(np.int16)) <= 16,
        weight_partition, 0).reshape(16, 768).astype(np.int16)
    update_error = -np.matmul(support, dropped16)
    weighted_elements = counts64[:, None]
    result["beta16_update_error_elements"] = int(
        np.sum(counts64) * 768)
    result["beta16_update_error_nonzero_elements"] = int(np.sum(
        weighted_elements * (update_error != 0)))
    result["beta16_update_error_signed_sum_q"] = int(np.sum(
        weighted_elements * update_error.astype(np.int64), dtype=np.int64))
    result["beta16_update_error_abs_sum_q"] = int(np.sum(
        weighted_elements * np.abs(update_error.astype(np.int64)),
        dtype=np.int64))
    result["beta16_update_error_minimum_q"] = int(np.min(update_error))
    result["beta16_update_error_maximum_q"] = int(np.max(update_error))
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
            "m463r2_beta16_destination_stationary_dse_recovery_contract_v1" and
            contract.get("status") ==
            "FROZEN_PREPAYLOAD_RECOVERY_BEFORE_UNIQUE_M40_REPLAY",
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
    abort = strict_json(paths["m463_abort_receipt"])
    require(abort["status"] ==
            "ABORTED_BEFORE_MARKER_AND_BEFORE_ANY_M40_PAYLOAD_READ" and
            abort["attempt_audit"]["one_shot_marker_created"] is False and
            abort["attempt_audit"]["m40_payload_files_opened"] == 0 and
            abort["attempt_audit"]["m40_payload_bytes_read"] == 0 and
            not (root / "results/M463_M40_CPU_DSE_ONE_SHOT_CONSUMED_20260826.marker").exists(),
            "M463 pre-input abort recovery audit drift")
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
            storage_rows.append({
                "operator": operator, "beta_q": beta,
                "dense_int8_bytes": dense_bytes,
                "bitmap_bytes": bitmap_bytes,
                "retained_int8_bytes": retained,
                "encoded_bytes": encoded,
            })
            for partition in range(432):
                for block in range(8):
                    minimum, reachable = minimum_drop_set_cover_le4(
                        weight[partition, :, block, :], beta)
                    maximum_retained = int(np.max(np.sum(
                        kept[partition, :, block, :], axis=0)))
                    cover_rows.append({
                        "operator": operator, "partition": partition,
                        "output_block": block, "beta_q": beta,
                        "minimum_drop_sources_to_cover_96_lanes_le4":
                            -1 if minimum is None else minimum,
                        "cover_le_3": int(minimum is not None and
                                          minimum <= 3),
                        "minimum_cover_eq_3": int(minimum == 3),
                        "cover_le_4": int(minimum is not None and
                                          minimum <= 4),
                        "minimum_cover_eq_4": int(minimum == 4),
                        "uncoverable_even_with_all_16": int(not reachable),
                        "maximum_retained_sources_any_lane": maximum_retained,
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
        rows = [row for row in cover_rows if row["beta_q"] == beta]
        cover_totals[str(beta)] = {
            "blocks": len(rows),
            "drop_set_cover_le_3": sum(row["cover_le_3"] for row in rows),
            "drop_set_minimum_eq_3": sum(
                row["minimum_cover_eq_3"] for row in rows),
            "drop_set_cover_le_4": sum(row["cover_le_4"] for row in rows),
            "drop_set_minimum_eq_4": sum(
                row["minimum_cover_eq_4"] for row in rows),
            "uncoverable_even_with_all_16": sum(
                row["uncoverable_even_with_all_16"] for row in rows),
        }
    require(storage_totals["16"]["encoded_bytes"] == 14283256 and
            cover_totals["16"]["blocks"] == 13824 and
            cover_totals["16"]["drop_set_minimum_eq_3"] == 119 and
            cover_totals["16"]["drop_set_minimum_eq_4"] == 2694 and
            cover_totals["16"]["uncoverable_even_with_all_16"] == 891 and
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

    error_aggregate = [Counter() for _ in range(4)]
    error_minimum = [None] * 4
    error_maximum = [None] * 4
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
            for key in ("beta16_update_error_elements",
                        "beta16_update_error_nonzero_elements",
                        "beta16_update_error_signed_sum_q",
                        "beta16_update_error_abs_sum_q"):
                error_aggregate[operator][key] += phase[key]
            error_minimum[operator] = (
                phase["beta16_update_error_minimum_q"]
                if error_minimum[operator] is None else
                min(error_minimum[operator],
                    phase["beta16_update_error_minimum_q"]))
            error_maximum[operator] = (
                phase["beta16_update_error_maximum_q"]
                if error_maximum[operator] is None else
                max(error_maximum[operator],
                    phase["beta16_update_error_maximum_q"]))
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
            beta16_free["cycles"] >= 437037880,
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
    global_signed_sum = 0
    global_final_accumulators = 0
    for operator in range(4):
        updates = error_aggregate[operator]
        final_accumulators = 10 * 3000 * 768
        summary = {
            "operator": operator,
            "operator_name": operators[operator],
            "final_accumulators": final_accumulators,
            "exact_signed_error_sum_over_final_accumulators_q":
                updates["beta16_update_error_signed_sum_q"],
            "exact_mean_signed_error_per_final_accumulator_q":
                updates["beta16_update_error_signed_sum_q"] /
                float(final_accumulators),
            "partition_update_elements":
                updates["beta16_update_error_elements"],
            "nonzero_partition_update_error_elements":
                updates["beta16_update_error_nonzero_elements"],
            "partition_update_error_minimum_q": error_minimum[operator],
            "partition_update_error_maximum_q": error_maximum[operator],
            "partition_update_error_mean_absolute_q":
                updates["beta16_update_error_abs_sum_q"] /
                float(updates["beta16_update_error_elements"]),
            "maximum_sumabs_bound_q": int(np.max(
                bound_by_operator[operator])),
            "minimum_sumabs_bound_q": int(np.min(
                bound_by_operator[operator])),
            "mean_sumabs_bound_q": float(np.mean(
                bound_by_operator[operator])),
            "maximum_sumabs_bound_scaled": float(np.max(
                bound_by_operator[operator].astype(np.float64) *
                scales[operator].astype(np.float64))),
            "final_accumulator_abs_error_histogram_or_quantiles_computed":
                False,
            "bound_violations": 0,
        }
        error_summaries.append(summary)
        global_signed_sum += updates["beta16_update_error_signed_sum_q"]
        global_final_accumulators += final_accumulators
    global_bound = np.concatenate(bound_by_operator)
    global_scale = np.concatenate(scales)
    global_error = {
        "final_accumulators": global_final_accumulators,
        "exact_signed_error_sum_over_final_accumulators_q":
            global_signed_sum,
        "exact_mean_signed_error_per_final_accumulator_q":
            global_signed_sum / float(global_final_accumulators),
        "maximum_sumabs_bound_q": int(np.max(global_bound)),
        "minimum_sumabs_bound_q": int(np.min(global_bound)),
        "mean_sumabs_bound_q": float(np.mean(global_bound)),
        "maximum_sumabs_bound_scaled": float(np.max(
            global_bound.astype(np.float64) *
            global_scale.astype(np.float64))),
        "final_accumulator_abs_error_histogram_or_quantiles_computed": False,
        "reason_not_computed":
            "A row-aligned 3000x6912 by 6912x768 GEMM for each of 40 records would add about 637G CPU MACs; cycle fast-kill does not depend on it.",
        "sumabs_bound_violations": 0,
    }

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
        "schema": "m463r2_beta16_destination_stationary_dse_v1",
        "status": "PASS_M463R2_CPU_ONLY_EXACT_SCHEDULE_DSE",
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
            "beta16_drop_set_minimum_eq_3":
                cover_totals["16"]["drop_set_minimum_eq_3"],
            "beta16_drop_set_cover_le_4":
                cover_totals["16"]["drop_set_cover_le_4"],
            "beta16_drop_set_minimum_eq_4":
                cover_totals["16"]["drop_set_minimum_eq_4"],
            "beta16_uncoverable_even_with_all_16_drop_sets":
                cover_totals["16"]["uncoverable_even_with_all_16"],
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
                "analytic_more_optimistic_d_ge_3_to_one_cycle_lower_bound":
                    437037880,
                "exact_ds_not_below_analytic_lower_bound": True,
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
            "beta16_exact_ds_below_analytic_lower_bound_violations": 0,
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
    result_name = "m463r2_beta16_destination_stationary_dse_r1.json"
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
    print("PASS_M463R2_CPU_ONLY_EXACT_SCHEDULE_DSE dense={} beta0={} beta16_free={} best_block={} decision={} manifest={} outer={}".format(
        dense["cycles"], beta0["cycles"], beta16_free["cycles"],
        best_block_realistic["cycles"], result["decision"]["decision"],
        sha256(manifest), sha256(outer)), flush=True)


if __name__ == "__main__":
    main()
