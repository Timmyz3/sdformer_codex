#!/usr/bin/env python3
"""Exact-identity row-tile SRAM/cycle DSE for the frozen M430 workload.

This analyzer deliberately consumes the contract-visible M410R2 derivative,
not the one-shot M40 payload.  It reclassifies every original16 row with the
already frozen M430 q32 catalog, first reproduces M430 at row_tile=3000 and
32 B/cycle, and only then sweeps row-tile capacity and bandwidth.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def count_runs(values):
    if values.size == 0:
        return 0
    ordered = np.unique(values)
    return int(1 + np.count_nonzero(ordered[1:] != ordered[:-1] + 1))


def data_cycles(byte_count, bandwidth):
    if bandwidth == "infinite":
        return 0
    return int(math.ceil(byte_count / float(bandwidth)))


def candidate_task_cycles(stat, model, bandwidth):
    config = data_cycles(model["elastic_config_bytes"], bandwidth)
    config += model["dma_command_setup_cycles"]
    matcher = stat["rows"] + stat["early_extra"] + 2
    cycles = config + matcher + 1
    if stat["active"] == 0:
        return cycles + model["tail_cycles"]
    tile_bytes = (model["weight_bytes_per_tile"] +
                  stat["used_patterns"] *
                  model["elastic_center_stride_bytes"])
    tile_dma = data_cycles(tile_bytes, bandwidth)
    tile_dma += ((1 + stat["used_runs"]) *
                 model["dma_command_setup_cycles"])
    work = model["output_blocks_per_tile"] * (
        stat["correction"] + stat["pwp"])
    replay = work + model["descriptor_sram_latency_cycles"]
    # Tile0 DMA is paid first.  Tile1 DMA starts with tile0 replay and is
    # exposed only when it is longer than replay0.
    return (cycles + tile_dma + max(replay, tile_dma) + replay +
            model["tail_cycles"])


def baseline_preprocess(rows, model, bandwidth):
    return max(rows + model["popcount_filter_pipeline_cycles"],
               data_cycles(model["weight_phase_bytes"], bandwidth) +
               model["dma_command_setup_cycles"])


def write_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("\n".join(
        "{}  {}".format(sha256(output_dir / name), name)
        for name in sorted(names)) + "\n", encoding="utf-8")
    seal = output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")
    return manifest, seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M468 overwrite")

    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m468r2_h67_peer_budget_rowtile_recovery_contract_v2" and
            contract.get("status") ==
            "FROZEN_RECOVERY_AFTER_PY36_PRE_INPUT_ABORT",
            "M468 contract identity drift")
    root = args.contract.resolve().parents[1]
    inputs = {}
    identity = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M468 input SHA drift: " + name)
        inputs[name] = path
        identity[name] = dict(spec)
    require(inputs["analyzer"].resolve() == Path(__file__).resolve() and
            identity["analyzer"]["sha256"] == source_start,
            "M468 analyzer self SHA drift")

    m430 = strict_json(inputs["m430_result"])
    catalog = strict_json(inputs["m430_catalog"])
    derivative = strict_json(inputs["m410r2_manifest"])
    require(m430["status"] ==
            "PASS_M430B_ONE_COMPLETED_M40_HELDOUT_DUAL_REPLAY" and
            m430["paper_identity"] == contract["paper_identity"],
            "M468 M430 identity/status drift")
    require(derivative["status"] ==
            "PASS_M410R2_CONTRACT_VISIBLE_FULL_RUNTIME_STIMULUS_EXPORT" and
            derivative["output"]["rows"]["sha256"] ==
            identity["m410r2_rows"]["sha256"],
            "M468 M410R2 derivative identity drift")

    phase_reference = []
    with inputs["m430_phase_csv"].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phase_reference.append({key: int(value) for key, value in row.items()})
    model = contract["cycle_model"]
    phases = model["samples"] * model["operators"] * model["partitions"]
    require(len(phase_reference) == phases, "M468 phase CSV extent drift")

    tile_sizes = tuple(contract["dse"]["row_tile_sizes"])
    bandwidths = tuple(contract["dse"]["dram_bytes_per_cycle"])
    points = {}
    baseline_tasks = {}
    for tile_size in tile_sizes:
        points[tile_size] = {}
        baseline_tasks[tile_size] = {
            sample: [] for sample in range(model["samples"])}
        for bandwidth in bandwidths:
            points[tile_size][str(bandwidth)] = {
                "candidate_cycles": 0,
                "config_bytes": 0,
                "weight_bytes": 0,
                "pwp_bytes": 0,
                "source_stream_bytes": 0,
                "psum_reads": 0,
                "psum_writes": 0,
                "used_pattern_loads": 0,
                "nonempty_tasks": 0,
                "tasks": 0,
                "maximum_used_patterns": 0,
            }

    popcount = np.fromiter((bin(value).count("1")
                            for value in range(1 << 16)),
                           dtype=np.uint8, count=1 << 16)
    aggregate = {
        "source_rows": 0, "active_rows": 0, "pwp_rows": 0,
        "fallback_rows": 0, "correction_ops_per_block": 0,
        "bit_sparse_vector_ops_per_block": 0,
    }
    phase_mismatches = 0
    row_path = inputs["m410r2_rows"]
    with row_path.open("r", encoding="ascii") as rows_handle:
        for phase_index in range(phases):
            sample, rem = divmod(
                phase_index, model["operators"] * model["partitions"])
            operator, partition = divmod(rem, model["partitions"])
            originals = np.fromiter(
                (int(rows_handle.readline(), 16) & 0xffff
                 for _ in range(model["rows_per_phase"])),
                dtype=np.uint16, count=model["rows_per_phase"])
            require(originals.size == model["rows_per_phase"],
                    "M468 premature row EOF")
            centers = np.asarray([
                int(value, 16) for value in
                catalog["operators"][operator]["partitions"][partition]
                ["nested_patterns"]], dtype=np.uint16)
            require(centers.size == 32, "M468 requires q32 catalog")

            unique, inverse = np.unique(originals, return_inverse=True)
            population_u = popcount[unique]
            distances = popcount[np.bitwise_xor(
                unique[:, None], centers[None, :])]
            best_index_u = np.argmin(distances, axis=1).astype(np.int16)
            best_distance_u = distances[
                np.arange(unique.size), best_index_u].astype(np.int16)
            active_u = unique != 0
            pwp_u = active_u & ((1 + best_distance_u) < population_u)
            correction_u = np.where(
                active_u, np.where(pwp_u, best_distance_u, population_u), 0)
            q16_exact_u = np.any(
                unique[:, None] == centers[None, :16], axis=1)
            early_extra_u = ((population_u >= 2) & ~q16_exact_u)

            active = active_u[inverse]
            pwp = pwp_u[inverse]
            correction = correction_u[inverse]
            population_rows = population_u[inverse]
            best_index = best_index_u[inverse]
            early_extra = early_extra_u[inverse]
            full_used = np.unique(best_index[pwp])
            ref = phase_reference[phase_index]
            actual = {
                "sample": sample,
                "operator": operator,
                "partition": partition,
                "active_rows": int(active.sum()),
                "pwp_rows": int(pwp.sum()),
                "fallback_rows": int(active.sum() - pwp.sum()),
                "correction_ops_per_block": int(correction.sum()),
                "used_pwp_patterns": int(full_used.size),
                "used_center_runs": count_runs(full_used),
                "early_matcher": int(model["rows_per_phase"] +
                                     early_extra.sum() + 2),
            }
            for key in actual:
                if actual[key] != ref[key]:
                    phase_mismatches += 1

            aggregate["source_rows"] += model["rows_per_phase"]
            aggregate["active_rows"] += actual["active_rows"]
            aggregate["pwp_rows"] += actual["pwp_rows"]
            aggregate["fallback_rows"] += actual["fallback_rows"]
            aggregate["correction_ops_per_block"] += actual[
                "correction_ops_per_block"]
            aggregate["bit_sparse_vector_ops_per_block"] += int(
                population_rows.sum())

            for tile_size in tile_sizes:
                for start in range(0, model["rows_per_phase"], tile_size):
                    stop = min(start + tile_size, model["rows_per_phase"])
                    tile_pwp = pwp[start:stop]
                    used = np.unique(best_index[start:stop][tile_pwp])
                    stat = {
                        "rows": stop - start,
                        "active": int(active[start:stop].sum()),
                        "pwp": int(tile_pwp.sum()),
                        "correction": int(correction[start:stop].sum()),
                        "early_extra": int(early_extra[start:stop].sum()),
                        "used_patterns": int(used.size),
                        "used_runs": count_runs(used),
                        "bit_sparse": int(population_rows[start:stop].sum()),
                    }
                    baseline_tasks[tile_size][sample].append(stat)
                    for bandwidth in bandwidths:
                        point = points[tile_size][str(bandwidth)]
                        point["candidate_cycles"] += candidate_task_cycles(
                            stat, model, bandwidth)
                        point["tasks"] += 1
                        point["config_bytes"] += model[
                            "elastic_config_bytes"]
                        point["source_stream_bytes"] += stat["rows"] * 2
                        point["psum_reads"] += stat["active"] * model[
                            "output_blocks"]
                        point["psum_writes"] += stat["active"] * model[
                            "output_blocks"]
                        point["maximum_used_patterns"] = max(
                            point["maximum_used_patterns"],
                            stat["used_patterns"])
                        if stat["active"]:
                            point["nonempty_tasks"] += 1
                            point["weight_bytes"] += (
                                model["weight_bytes_per_tile"] * 2)
                            point["pwp_bytes"] += (
                                stat["used_patterns"] *
                                model["elastic_center_stride_bytes"] * 2)
                            point["used_pattern_loads"] += stat[
                                "used_patterns"]
            if (phase_index + 1) % 432 == 0:
                print("[M468] phases={}/{}".format(
                    phase_index + 1, phases), flush=True)
        require(rows_handle.readline() == "", "M468 unexpected trailing rows")

    expected_aggregate = m430["runtime_population"]
    require(phase_mismatches == 0, "M468 row derivative != M430 phase ledger")
    for key, value in aggregate.items():
        require(value == expected_aggregate[key],
                "M468 aggregate mismatch: " + key)

    result_rows = []
    for tile_size in tile_sizes:
        psum_bytes = (tile_size * model["output_blocks"] *
                      model["lanes"] * model["accumulator_bits"] // 8)
        descriptor_bytes = tile_size * model["descriptor_bits"] // 8
        source_buffer_bytes = tile_size * 2
        # Two payload slots are required to overlap the two output tiles.
        maximum_patterns = max(
            points[tile_size][str(bandwidths[0])]["maximum_used_patterns"], 0)
        payload_slot_bytes = 2 * (
            model["weight_bytes_per_tile"] + maximum_patterns *
            model["elastic_center_stride_bytes"])
        total_sram_bytes = (psum_bytes + descriptor_bytes +
                            source_buffer_bytes + payload_slot_bytes +
                            model["elastic_config_bytes"])
        for bandwidth in bandwidths:
            point = points[tile_size][str(bandwidth)]
            baseline_cycles = 0
            for sample in range(model["samples"]):
                tasks = baseline_tasks[tile_size][sample]
                require(tasks, "M468 empty baseline task list")
                preprocess = [baseline_preprocess(
                    task["rows"], model, bandwidth) for task in tasks]
                baseline_cycles += preprocess[0]
                for index, task in enumerate(tasks):
                    next_preprocess = (preprocess[index + 1]
                                       if index + 1 < len(tasks) else 0)
                    baseline_cycles += max(
                        task["bit_sparse"] * model["output_blocks"],
                        next_preprocess)
                    baseline_cycles += model["tail_cycles"]
                baseline_cycles += model["commit_cycles_per_sample"]
            candidate_cycles = point["candidate_cycles"]
            # Commit population is unchanged by row tiling.
            candidate_cycles += (model["samples"] *
                                 model["commit_cycles_per_sample"])
            point["candidate_cycles"] = candidate_cycles
            point["strong_zero_cycles"] = baseline_cycles
            point["speedup_vs_same_tile_strong_zero"] = (
                baseline_cycles / float(candidate_cycles))
            point["psum_read_bytes"] = (point["psum_reads"] *
                                        model["psum_vector_bytes"])
            point["psum_write_bytes"] = (point["psum_writes"] *
                                         model["psum_vector_bytes"])
            point["dram_bytes"] = (point["config_bytes"] +
                                   point["weight_bytes"] +
                                   point["pwp_bytes"])
            point["row_tile"] = tile_size
            point["bandwidth_bytes_per_cycle"] = bandwidth
            point["psum_bytes"] = psum_bytes
            point["descriptor_bytes"] = descriptor_bytes
            point["source_buffer_bytes"] = source_buffer_bytes
            point["payload_slot_bytes"] = payload_slot_bytes
            point["total_sram_bytes"] = total_sram_bytes
            point["fits_128k"] = total_sram_bytes <= 128 * 1024
            point["fits_240k"] = total_sram_bytes <= 240 * 1024
            point["fits_512k"] = total_sram_bytes <= 512 * 1024
            result_rows.append(dict(point))

    anchors = [row for row in result_rows
               if row["row_tile"] == model["rows_per_phase"] and
               row["bandwidth_bytes_per_cycle"] == 32]
    require(len(anchors) == 1 and
            anchors[0]["candidate_cycles"] ==
            m430["comparisons"]["m430_catalog_dual_cycles"] and
            anchors[0]["strong_zero_cycles"] ==
            m430["comparisons"]["strong_zero_cycles"],
            "M468 full-resident anchor mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = args.output_dir / "m468_rowtile_capacity_cycle_sweep.csv"
    fields = list(result_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)
    result = {
        "schema": "m468_h67_peer_budget_rowtile_result_v1",
        "status": "PASS_M468_EXACT_DERIVATIVE_ROWTILE_DSE",
        "identity": identity,
        "paper_identity": contract["paper_identity"],
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "aggregate_reproduction": {
            "phase_field_mismatches": phase_mismatches,
            "aggregate": aggregate,
            "full_resident_m430_cycle_mismatches": 0,
            "m40_payload_reads": 0,
        },
        "points": result_rows,
        "claim_boundary": contract["claim_boundary"],
        "decision_rule": contract["decision_rule"],
    }
    result_path = args.output_dir / "m468_h67_peer_budget_rowtile_result_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(source_start == sha256(Path(__file__).resolve()),
            "M468 analyzer changed during run")
    manifest, seal = write_seal(
        args.output_dir, [csv_path.name, result_path.name])
    print("M468_PASS points={} anchor={} seal={}".format(
        len(result_rows), anchors[0]["candidate_cycles"], sha256(seal)),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
