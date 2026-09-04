#!/usr/bin/env python3
"""M471 exact cost-aware G15 DIRECT/PARENT same-K CPU DSE.

Consumes only sealed M468R3/M469/M470 results.  The candidate compute route is
exact; missing G15-specific per-task center masks are conservatively replaced
by the M430 superset masks and are never described as exact traffic.
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


def data_cycles(values, bandwidth):
    values = np.asarray(values, dtype=np.int64)
    if bandwidth == "infinite":
        return np.zeros(values.shape, dtype=np.int64)
    return (values + int(bandwidth) - 1) // int(bandwidth)


def popcount_u32(values):
    byte_pop = np.fromiter((bin(value).count("1") for value in range(256)),
                           dtype=np.uint8, count=256)
    values = np.asarray(values, dtype=np.uint32)
    return (byte_pop[values & 255] + byte_pop[(values >> 8) & 255] +
            byte_pop[(values >> 16) & 255] +
            byte_pop[(values >> 24) & 255]).astype(np.int64)


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
    require(not args.output_dir.exists(), "refusing M471 overwrite")

    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m471_g15_exact_cost_aware_direct_parent_execution_v1" and
            contract.get("status") == "FROZEN_EXACT_SHA_BEFORE_EXECUTION",
            "M471 execution contract drift")
    root = args.contract.resolve().parents[1]
    inputs = {}
    identity = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M471 input SHA drift: " + name)
        inputs[name] = path
        identity[name] = dict(spec)
    require(inputs["analyzer"].resolve() == source_path and
            identity["analyzer"]["sha256"] == source_start,
            "M471 analyzer self-SHA drift")

    preflight = strict_json(inputs["preflight"])
    require(preflight["status"] ==
            "FROZEN_PREFLIGHT_BEFORE_ANALYZER_EXECUTION",
            "M471 preflight status drift")
    for name, spec in preflight["inputs"].items():
        require(name in inputs and inputs[name] == root / spec["path"] and
                identity[name]["sha256"] == spec["sha256"],
                "M471 execution/preflight identity mismatch: " + name)

    r6 = strict_json(inputs["m468r3_m469_r6_result"])
    require(r6["status"] ==
            "PASS_EXACT_DERIVATIVE_FAIR_EMPTY_LAZY_PWP_CPU_DSE" and
            r6["identity_reproduction"]["m40_payload_reads"] == 0 and
            r6["identity_reproduction"]["phase_field_mismatches"] == 0,
            "M471 R6 upstream status/identity drift")
    require(r6["performance_admitted"] is False,
            "M471 refuses an unexpectedly admitted R6")

    ks = tuple(preflight["resource_axes"]["source_lanes_k"])
    tiles = tuple(preflight["resource_axes"]["row_tile_sizes"])
    banks_axis = tuple(preflight["resource_axes"]["resident_block_banks"])
    bandwidths = tuple(preflight["resource_axes"]["dram_bytes_per_cycle"])
    parent_modes = tuple(preflight["resource_axes"]["parent_supply"])
    model = preflight["cycle_model"]
    budget = preflight["traffic_and_capacity"]["budget_bytes"]

    phase_totals = {k: {
        "rows": 0, "empty_rows": 0,
        "direct_rows": 0, "parent_rows": 0,
        "direct_cycle_sum": 0, "parent_cycle_sum": 0,
        "selected_cycle_sum": 0,
    } for k in ks}
    phase_rows = 0
    m430_parent_rows = 0
    with inputs["m469_phase_sidecar"].open(
            "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phase_rows += 1
            source_rows = int(row["source_rows"])
            m430_parent_rows += int(row["m430_use_pwp_rows"])
            for k in ks:
                stat = phase_totals[k]
                stat["rows"] += source_rows
                stat["empty_rows"] += int(row["original_pop0"])
                stat["direct_rows"] += int(row[
                    "k{}_direct_rows".format(k)])
                stat["parent_rows"] += int(row[
                    "k{}_parent_rows".format(k)])
                stat["direct_cycle_sum"] += int(row[
                    "k{}_direct_cycle_sum".format(k)])
                stat["parent_cycle_sum"] += int(row[
                    "k{}_parent_cycle_sum".format(k)])
                stat["selected_cycle_sum"] += int(row[
                    "k{}_selected_cycle_sum".format(k)])
    require(phase_rows == 17280, "M471 phase-sidecar extent drift")

    rowtile_totals = {}
    with inputs["m469_rowtile_sidecar"].open(
            "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["row_tile"]), int(row["source_lanes_k"]))
            require(key not in rowtile_totals, "duplicate M471 rowtile/K")
            rowtile_totals[key] = {
                field: int(row[field]) for field in (
                    "rows", "empty_rows", "direct_rows", "parent_rows",
                    "direct_cycle_sum", "parent_cycle_sum",
                    "selected_cycle_sum")}
    require(len(rowtile_totals) == len(tiles) * len(ks),
            "M471 rowtile-sidecar extent drift")
    for tile in tiles:
        for k in ks:
            require(rowtile_totals[(tile, k)] == phase_totals[k],
                    "M471 phase/rowtile compact mismatch")
            require(phase_totals[k]["selected_cycle_sum"] <=
                    phase_totals[k]["direct_cycle_sum"] and
                    phase_totals[k]["parent_rows"] <= m430_parent_rows,
                    "M471 route/subset invariant failed")
    require(phase_totals[1]["parent_rows"] == m430_parent_rows,
            "M471 K1 parent route must reproduce M430 route population")

    task = np.load(str(inputs["m470_task_sidecar"]), allow_pickle=False)
    required_fields = {
        "sample", "operator", "partition", "row_tile_size", "row_start",
        "row_stop", "active_rows", "pwp_rows",
        "correction_ops_per_block", "strong_zero_pop_ops_per_block",
        "early_matcher_cycles", "used_center_mask_u32",
        "used_center_runs", "used_center_population_sum",
        "generator_center_ceil_sum_k1", "generator_center_ceil_sum_k2",
        "generator_center_ceil_sum_k4", "generator_center_ceil_sum_k8",
    }
    require(set(task.files) == required_fields and
            len(task["sample"]) == 4250880,
            "M471 task-sidecar schema/extent drift")

    # Capacity is inherited byte-for-byte from R6.  The first bandwidth copy
    # is enough because capacity is bandwidth-independent; assert all copies.
    capacity = {}
    for point in r6["points"]:
        mode = point["mode"]
        if mode == "strong_zero":
            key = (point["row_tile"], point["resident_block_banks"],
                   "direct_only", 0)
        elif mode == "stored_pwp":
            key = (point["row_tile"], point["resident_block_banks"],
                   mode, 0)
        else:
            key = (point["row_tile"], point["resident_block_banks"],
                   mode, point["generator_source_lanes_k"])
        value = {
            "logical_total_bytes": point["capacity"]["logical_total_bytes"],
            "macro_rounded_total_bytes": point["capacity"][
                "macro_rounded_total_bytes"],
            "logical_items": point["capacity"]["logical_items"],
            "macro_rounded_items": point["capacity"]["macro_rounded_items"],
            "fits_240k_logical": point["fits_240k_logical"],
            "fits_240k_macro_rounded": point[
                "fits_240k_macro_rounded"],
            "fits_both_240k_gates": point["fits_both_240k_gates"],
        }
        if key in capacity:
            require(capacity[key] == value,
                    "M471 R6 capacity changed across bandwidth")
        else:
            capacity[key] = value
        require(value["logical_total_bytes"] <=
                value["macro_rounded_total_bytes"],
                "M471 rounded capacity smaller than logical")

    points = []
    direct_index = {}
    candidate_index = {}
    task_population = {}
    for tile in tiles:
        select = task["row_tile_size"] == tile
        rows = (task["row_stop"][select].astype(np.int64) -
                task["row_start"][select].astype(np.int64))
        active = task["active_rows"][select] != 0
        nonempty = int(active.sum())
        task_count = int(rows.size)
        used = popcount_u32(task["used_center_mask_u32"][select])
        runs = task["used_center_runs"][select].astype(np.int64)
        matcher = task["early_matcher_cycles"][select].astype(np.int64)
        center_population_sum = task[
            "used_center_population_sum"][select].astype(np.int64)
        active_rows = int(task["active_rows"][select].astype(np.int64).sum())
        task_population[tile] = {
            "tasks": task_count,
            "nonempty_tasks": nonempty,
            "empty_tasks": task_count - nonempty,
            "active_rows": active_rows,
            "m430_upper_used_center_instances": int(used.sum()),
            "m430_upper_center_runs": int(runs.sum()),
        }
        for banks in banks_axis:
            passes = 1 if banks == 8 else 2
            direct_scan_cycles = int(((rows + 5) * passes).sum())
            candidate_route_cycles = int(np.where(
                active, matcher * passes, (rows + 5) * passes).sum())
            tail_cycles = task_count * passes * model[
                "tail_cycles_per_pass_task"]
            psum_accesses = active_rows * 8
            source_sram_bytes = int(rows.sum()) * 2 * passes
            for bandwidth in bandwidths:
                bw_key = str(bandwidth)
                weight_half_cycles = int(data_cycles(
                    np.asarray([model["weight_bytes_per_half"]]),
                    bandwidth)[0])
                config_cycles = int(data_cycles(
                    np.asarray([model["config_bytes"]]), bandwidth)[0])

                direct_weight_data_cycles = nonempty * 2 * weight_half_cycles
                direct_dma_commands = nonempty * 2
                direct_dma_command_cycles = (direct_dma_commands *
                                             model[
                                                 "dma_command_setup_cycles"])
                for k in ks:
                    route = phase_totals[k]
                    direct_compute_cycles = route["direct_cycle_sum"] * 8
                    direct_cycles = (direct_compute_cycles +
                                     direct_scan_cycles +
                                     direct_weight_data_cycles +
                                     direct_dma_command_cycles + tail_cycles +
                                     model["commit_cycles"])
                    cap = capacity[(tile, banks, "direct_only", 0)]
                    direct = {
                        "mode": "direct_only",
                        "row_tile": tile,
                        "resident_block_banks": banks,
                        "bandwidth_bytes_per_cycle": bandwidth,
                        "source_lanes_k": k,
                        "cycles": direct_cycles,
                        "exact_compute_cycles": direct_compute_cycles,
                        "source_scan_cycles": direct_scan_cycles,
                        "matcher_cycles": 0,
                        "payload_data_cycles": direct_weight_data_cycles,
                        "dma_command_cycles": direct_dma_command_cycles,
                        "tail_cycles": tail_cycles,
                        "commit_cycles": model["commit_cycles"],
                        "source_sram_bytes": source_sram_bytes,
                        "source_dram_bytes": 0,
                        "weight_dram_bytes": nonempty * 2 * 6144,
                        "pwp_dram_bytes": 0,
                        "config_dram_bytes": 0,
                        "dram_bytes": nonempty * 2 * 6144,
                        "dma_commands": direct_dma_commands,
                        "psum_sram_reads": psum_accesses,
                        "psum_sram_writes": psum_accesses,
                        "psum_sram_read_bytes": psum_accesses * 228,
                        "psum_sram_write_bytes": psum_accesses * 228,
                        "psum_dram_spill_bytes": 0,
                        "logical_sram_bytes": cap["logical_total_bytes"],
                        "macro_rounded_sram_bytes": cap[
                            "macro_rounded_total_bytes"],
                        "fits_240k_logical": cap["fits_240k_logical"],
                        "fits_240k_macro_rounded": cap[
                            "fits_240k_macro_rounded"],
                        "fits_both_240k_gates": cap[
                            "fits_both_240k_gates"],
                        "same_resource_across_k": False,
                        "performance_admitted": False,
                    }
                    points.append(direct)
                    direct_index[(tile, banks, bw_key, k)] = direct

                    for mode in parent_modes:
                        config_data_cycles = (nonempty * passes *
                                              config_cycles)
                        config_commands = nonempty * passes
                        selected_compute_cycles = (
                            route["selected_cycle_sum"] * 8)
                        descriptor_cycles = (nonempty * 2 *
                                             model[
                                                 "descriptor_latency_cycles_per_half"])
                        if mode == "stored_pwp":
                            half_bytes = (model[
                                "weight_bytes_per_half"] + used * model[
                                    "stored_pwp_stride_bytes_per_center_half"])
                            payload_data_cycles = int((
                                data_cycles(half_bytes, bandwidth) * 2
                            )[active].sum())
                            payload_commands = int((
                                (1 + runs[active]) * 2).sum())
                            pwp_bytes = int(used[active].sum()) * 2 * 640
                            generator_cycles = 0
                            generator_commands = 0
                            generator_weight_read_bytes = 0
                            generator_signed_adds_upper = 0
                            generator_cache_write_bytes = 0
                            cap = capacity[(tile, banks, mode, 0)]
                        else:
                            payload_data_cycles = (
                                nonempty * 2 * weight_half_cycles)
                            payload_commands = nonempty * 2
                            pwp_bytes = 0
                            ceil_sum = task[
                                "generator_center_ceil_sum_k{}".format(k)
                            ][select].astype(np.int64)
                            generator_cycles = int((2 * (
                                used + 4 * ceil_sum +
                                np.where(used != 0, 2, 0))).sum())
                            generator_commands = int(used.sum()) * 2
                            generator_weight_read_bytes = int(
                                center_population_sum.sum()) * 4 * 96 * 2
                            generator_signed_adds_upper = int(
                                center_population_sum.sum()) * 4 * 96 * 2
                            generator_cache_write_bytes = int(
                                used.sum()) * 576 * 2
                            cap = capacity[(tile, banks, mode, k)]

                        dma_commands = config_commands + payload_commands
                        dma_command_cycles = (dma_commands * model[
                            "dma_command_setup_cycles"])
                        candidate_cycles = (
                            selected_compute_cycles + candidate_route_cycles +
                            config_data_cycles + payload_data_cycles +
                            dma_command_cycles + descriptor_cycles +
                            generator_cycles + tail_cycles +
                            model["commit_cycles"])
                        weight_bytes = nonempty * 2 * 6144
                        config_bytes = nonempty * passes * 96
                        candidate = {
                            "mode": mode,
                            "row_tile": tile,
                            "resident_block_banks": banks,
                            "bandwidth_bytes_per_cycle": bandwidth,
                            "source_lanes_k": k,
                            "cycles": candidate_cycles,
                            "exact_compute_cycles": selected_compute_cycles,
                            "direct_exact_compute_cycles":
                                direct_compute_cycles,
                            "exact_compute_only_speedup":
                                direct_compute_cycles /
                                float(selected_compute_cycles),
                            "source_scan_cycles": 0,
                            "matcher_cycles_conservative_upper":
                                candidate_route_cycles,
                            "config_data_cycles": config_data_cycles,
                            "payload_data_cycles_conservative_upper":
                                payload_data_cycles,
                            "dma_command_cycles_conservative_upper":
                                dma_command_cycles,
                            "descriptor_cycles": descriptor_cycles,
                            "generator_cycles_conservative_upper":
                                generator_cycles,
                            "tail_cycles": tail_cycles,
                            "commit_cycles": model["commit_cycles"],
                            "source_sram_bytes": source_sram_bytes,
                            "source_dram_bytes": 0,
                            "weight_dram_bytes": weight_bytes,
                            "pwp_dram_bytes_conservative_upper": pwp_bytes,
                            "config_dram_bytes": config_bytes,
                            "dram_bytes_conservative_upper":
                                weight_bytes + pwp_bytes + config_bytes,
                            "dma_commands_conservative_upper": dma_commands,
                            "generator_commands_conservative_upper":
                                generator_commands,
                            "generator_weight_read_bytes_conservative_upper":
                                generator_weight_read_bytes,
                            "generator_signed_adds_conservative_upper":
                                generator_signed_adds_upper,
                            "generator_cache_write_bytes_conservative_upper":
                                generator_cache_write_bytes,
                            "pwp_sram_read_bytes":
                                route["parent_rows"] * 8 * 144,
                            "psum_sram_reads": psum_accesses,
                            "psum_sram_writes": psum_accesses,
                            "psum_sram_read_bytes": psum_accesses * 228,
                            "psum_sram_write_bytes": psum_accesses * 228,
                            "psum_dram_spill_bytes": 0,
                            "g15_parent_rows": route["parent_rows"],
                            "m430_parent_rows_upper": m430_parent_rows,
                            "g15_specific_center_mask_available": False,
                            "pwp_traffic_exact": False,
                            "pwp_traffic_conservative_upper_bound": True,
                            "logical_sram_bytes": cap[
                                "logical_total_bytes"],
                            "macro_rounded_sram_bytes": cap[
                                "macro_rounded_total_bytes"],
                            "fits_240k_logical": cap["fits_240k_logical"],
                            "fits_240k_macro_rounded": cap[
                                "fits_240k_macro_rounded"],
                            "fits_both_240k_gates": cap[
                                "fits_both_240k_gates"],
                            "source_peak_bytes_per_cycle": 96 * k,
                            "source_banks_or_ports": k,
                            "physical_product_slots": 96 * k,
                            "signed_preadder_proxy": 96 * (k - 1),
                            "same_resource_across_k": False,
                            "same_k_source_resource_as_direct": True,
                            "incremental_matcher_pwp_resources_charged": True,
                            "performance_admitted": False,
                        }
                        points.append(candidate)
                        candidate_index[(tile, banks, bw_key, k, mode)] = \
                            candidate

    comparisons = []
    nominations = []
    for banks in banks_axis:
        for bandwidth in bandwidths:
            bw_key = str(bandwidth)
            for k in ks:
                eligible_direct = [
                    direct_index[(tile, banks, bw_key, k)] for tile in tiles
                    if direct_index[(tile, banks, bw_key, k)]
                    ["fits_both_240k_gates"]]
                require(eligible_direct, "M471 no best-budget direct baseline")
                best_direct = min(eligible_direct,
                                  key=lambda point: point["cycles"])
                for mode in parent_modes:
                    for tile in tiles:
                        candidate = candidate_index[(tile, banks, bw_key,
                                                     k, mode)]
                        same_tile = direct_index[(tile, banks, bw_key, k)]
                        vs_best = best_direct["cycles"] / float(
                            candidate["cycles"])
                        vs_same = same_tile["cycles"] / float(
                            candidate["cycles"])
                        nominated = (candidate["fits_both_240k_gates"] and
                                     vs_best >= 1.15)
                        comparison = {
                            "resident_block_banks": banks,
                            "bandwidth_bytes_per_cycle": bandwidth,
                            "source_lanes_k": k,
                            "candidate_mode": mode,
                            "candidate_row_tile": tile,
                            "candidate_cycles": candidate["cycles"],
                            "candidate_fits_both_240k_gates": candidate[
                                "fits_both_240k_gates"],
                            "same_tile_direct_cycles": same_tile["cycles"],
                            "speedup_vs_same_tile_direct": vs_same,
                            "best_budget_direct_row_tile": best_direct[
                                "row_tile"],
                            "best_budget_direct_cycles": best_direct["cycles"],
                            "speedup_vs_best_budget_same_k_direct": vs_best,
                            "material_1p15": nominated,
                            "performance_admitted": False,
                        }
                        comparisons.append(comparison)
                        if nominated:
                            nominations.append(comparison)

    if nominations:
        decision = "NOMINATE_FOR_INDEPENDENT_HAMMER_NOT_ADMITTED"
    else:
        decision = "KILL_G15_AS_HEADLINE_RETAIN_ONLY_NONHEADLINE_SIDECAR"
    result = {
        "schema": "m471_g15_exact_cost_aware_direct_parent_result_v1",
        "status": "PASS_M471_EXACT_COMPUTE_CONSERVATIVE_TRAFFIC_DSE",
        "identity": identity,
        "scope": preflight["scope"],
        "sidecar_identity_checks": {
            "phase_rows": phase_rows,
            "task_rows": len(task["sample"]),
            "phase_rowtile_mismatches": 0,
            "k1_parent_vs_m430_mismatch": 0,
            "m410r2_derivative_reads": 0,
            "m40_payload_reads": 0,
        },
        "exact_route_aggregates": phase_totals,
        "m430_parent_rows_upper": m430_parent_rows,
        "task_population_by_row_tile": task_population,
        "points": points,
        "comparisons": comparisons,
        "decision": decision,
        "nominations": nominations,
        "claim_boundary": preflight["claim_boundary"],
        "performance_admitted": False,
        "independent_hammer_pending": True,
        "rtl_nominated": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    point_csv = args.output_dir / "m471_g15_cycle_traffic_capacity_points.csv"
    fields = sorted(set(key for point in points for key in point.keys()))
    with point_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)
    compare_csv = args.output_dir / "m471_g15_same_k_comparisons.csv"
    with compare_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)
    result_path = args.output_dir / \
        "m471_g15_exact_cost_aware_direct_parent_result_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(source_start == sha256(source_path),
            "M471 analyzer changed during execution")
    manifest, seal = write_seal(
        args.output_dir, [point_csv.name, compare_csv.name, result_path.name])
    (args.output_dir / "RUN_COMPLETE.txt").write_text(
        "PASS_M471_EXACT_COMPUTE_CONSERVATIVE_TRAFFIC_DSE\n",
        encoding="utf-8")
    print("M471_PASS decision={} nominations={} points={} seal={}".format(
        decision, len(nominations), len(points), sha256(seal)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
