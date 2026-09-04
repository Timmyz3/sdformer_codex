#!/usr/bin/env python3
"""Replay nested q16..q128 exact PAFT catalogs with explicit selective-PWP traffic."""

from __future__ import division

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path


Q_VALUES = (16, 32, 64, 128)
PORTS = (
    {"name": "WIDE144_PWP_96_WEIGHT", "weight_cycles": 1, "pwp_cycles": 1},
    {"name": "SHARED96", "weight_cycles": 1, "pwp_cycles": 2},
)
CACHE_BYTES = (64 << 10, 128 << 10, 256 << 10, 512 << 10)
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
        return json.load(handle, object_pairs_hook=pairs, parse_constant=reject)


def load_module(path):
    spec = importlib.util.spec_from_file_location("m339_frozen_m43", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M43 support unpacker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def popcount(value):
    return POPCOUNT[int(value) & 0xffff]


def phase_metrics_all_q(counter, nested_centers):
    """Evaluate all nested prefixes with one distance scan per runtime value."""
    require(len(nested_centers) >= Q_VALUES[-1],
            "M339 requires full q128 capacity in every partition")
    results = {q: Counter() for q in Q_VALUES}
    used = {q: set() for q in Q_VALUES}
    for value, count in counter.items():
        population = popcount(value)
        best_distance = 17
        best_index = 0
        for index, center in enumerate(nested_centers[:Q_VALUES[-1]]):
            distance = popcount(value ^ center)
            if distance < best_distance:
                best_distance = distance
                best_index = index
            q = index + 1
            if q not in results:
                continue
            pwp_selected = 1 + best_distance < population
            candidate = 1 + best_distance if pwp_selected else population
            result = results[q]
            result["partition_vectors"] += count
            result["matcher_rows"] += count * int(population >= 2)
            result["bit_sparse_vector_ops_per_block"] += count * population
            result["candidate_vector_ops_per_block"] += count * candidate
            result["pwp_ops_per_block"] += count * int(pwp_selected)
            result["correction_ops_per_block"] += count * (
                best_distance if pwp_selected else population)
            result["assignment_rows"] += count * int(pwp_selected)
            result["exact_pattern_hits"] += count * int(
                value != 0 and best_distance == 0)
            if pwp_selected:
                used[q].add(best_index)
    payload = {}
    for q in Q_VALUES:
        result = results[q]
        require(result["candidate_vector_ops_per_block"] ==
                result["pwp_ops_per_block"] +
                result["correction_ops_per_block"],
                "phase work conservation failure")
        result["used_pwp_patterns"] = len(used[q])
        payload[q] = dict(result)
    return payload


def percentile(values, fraction):
    ordered = sorted(values)
    require(ordered, "empty percentile population")
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def load_cycles(phase, q, selective, model):
    patterns = phase["used_pwp_patterns"] if selective else q
    payload = (model["weight_phase_bytes"] +
               patterns * model["pwp_pattern_all_blocks_bytes"])
    return int(math.ceil(payload / float(model["dram_bytes_per_cycle"])))


def matcher_cycles(phase, q, architecture):
    raw = phase["partition_vectors"]
    if architecture == "SYSTOLIC_Q_II1":
        return raw + q
    require(architecture == "SERIAL16_II1", "unknown matcher architecture")
    return raw + phase["matcher_rows"] * (int(math.ceil(q / 16.0)) - 1) + 2


def replay_kfirst(phases, q, port, architecture, selective, model,
                  common_commit_cycles):
    """Two-context K-first recurrence with explicit preprocess-before-DMA order.

    The next phase is scanned/matched first; only then is its discovered PWP
    working set transferred. That serial dependency is overlapped with current
    compute, avoiding the old free max(matcher,DMA) assumption.
    """
    def preprocess(phase):
        matcher = matcher_cycles(phase, q, architecture)
        packer = (int(math.ceil(phase["assignment_rows"] /
                                float(model["packer_lanes"]))) +
                  model["packer_pipeline_cycles"])
        dma = load_cycles(phase, q, selective, model)
        return matcher + max(packer, dma), matcher, packer, dma

    first, first_matcher, first_packer, first_dma = preprocess(phases[0])
    cycles = first
    components = Counter({"initial_preprocess": first,
                          "matcher_cycles_sum": first_matcher,
                          "packer_cycles_sum": first_packer,
                          "dma_cycles_sum": first_dma})
    bindings = Counter()
    for index, phase in enumerate(phases):
        compute = (
            phase["correction_ops_per_block"] * model["output_blocks"] *
            port["weight_cycles"] +
            phase["pwp_ops_per_block"] * model["output_blocks"] *
            port["pwp_cycles"])
        if index + 1 < len(phases):
            next_stage, matcher, packer, dma = preprocess(phases[index + 1])
            components["matcher_cycles_sum"] += matcher
            components["packer_cycles_sum"] += packer
            components["dma_cycles_sum"] += dma
        else:
            next_stage = 0
        body = max(compute, next_stage)
        bindings["compute" if compute >= next_stage else "preprocess"] += 1
        cycles += body + model["compute_tail_cycles_per_partition"]
        components["compute_cycles_sum"] += compute
    cycles += common_commit_cycles
    return {"cycles": cycles, "binding_phases": dict(bindings),
            "component_sums_not_additive": dict(components)}


def replay_bit_sparse(phases, port, model, common_commit_cycles):
    def preprocess(phase):
        scan = phase["partition_vectors"] + model["popcount_filter_pipeline_cycles"]
        dma = int(math.ceil(model["weight_phase_bytes"] /
                            float(model["dram_bytes_per_cycle"])))
        return max(scan, dma)

    cycles = preprocess(phases[0])
    for index, phase in enumerate(phases):
        compute = (phase["bit_sparse_vector_ops_per_block"] *
                   model["output_blocks"] * port["weight_cycles"])
        next_stage = preprocess(phases[index + 1]) if index + 1 < len(phases) else 0
        cycles += max(compute, next_stage) + model["compute_tail_cycles_per_partition"]
    return cycles + common_commit_cycles


def replay_legacy_m251(phases, port, model):
    baseline_load = int(math.ceil(model["weight_phase_bytes"] /
                                  float(model["dram_bytes_per_cycle"])))
    candidate_load = int(math.ceil(
        (model["weight_phase_bytes"] + 16 *
         model["pwp_pattern_all_blocks_bytes"]) /
        float(model["dram_bytes_per_cycle"])))
    sparse = baseline_load
    candidate = candidate_load
    for index, phase in enumerate(phases):
        sparse_compute = (phase["bit_sparse_vector_ops_per_block"] *
                          model["output_blocks"] * port["weight_cycles"])
        candidate_compute = (
            phase["correction_ops_per_block"] * model["output_blocks"] *
            port["weight_cycles"] +
            phase["pwp_ops_per_block"] * model["output_blocks"] *
            port["pwp_cycles"])
        matcher = phase["matcher_rows"] + 16
        packer = int(math.ceil(phase["assignment_rows"] / 8.0)) + 4
        next_sparse = baseline_load if index + 1 < len(phases) else 0
        next_candidate = candidate_load if index + 1 < len(phases) else 0
        sparse += max(sparse_compute, next_sparse) + 2
        candidate += max(candidate_compute, matcher, packer, next_candidate) + 2
    return sparse, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M339 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m339_q128_selective_pwp_kfirst_cycle_contract_v1",
            "M339 contract schema drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256(path)
        require(observed == spec["sha256"], "SHA drift for " + name)
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}

    catalog = strict_json(paths["m338_catalog"])
    trace = strict_json(paths["m248_runtime_trace"])
    m251 = strict_json(paths["m251_result"])
    m251r2 = strict_json(paths["m251r2_range_correction"])
    require(catalog["status"] ==
            "PASS_M338_TRAIN_ONLY_NESTED_Q16_Q32_Q64_Q128_EXACT_WORK_NO_CYCLES" and
            catalog["split"]["test_or_validation_data_used"] is False and
            catalog["split"]["train_valid825_key_overlap"] == 0,
            "M338 admission drift")
    require(trace["status"] ==
            "PASS_PAFT_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE" and
            trace["cohort"]["samples"] == 10 and trace["cohort"]["records"] == 40,
            "M248 runtime trace drift")
    require(m251["status"] ==
            "PASS_PAFT_RUNNING_BN_TRAIN_CATALOG_ISOLATED_CONV_CYCLE_MODEL" and
            m251r2["status"] == "PASS_M251_FIXED12_RANGE_CORRECTED_CYCLES_UNCHANGED",
            "M251 parent drift")
    operators = tuple(trace["cohort"]["operators"])
    require(catalog["geometry"]["operators"] == list(operators),
            "catalog/runtime operator order drift")
    model = contract["cycle_model"]
    m43 = load_module(paths["m43_support_unpacker"])
    require(m43.ROWS == model["rows_per_operator"] and
            m43.TILES * (m43.TILE_BITS // 16) == model["partitions_per_operator"],
            "M43/cycle geometry drift")

    trace_dir = paths["m248_runtime_trace"].parent
    op_index = {name: index for index, name in enumerate(operators)}
    histograms = defaultdict(Counter)
    for record_index, record in enumerate(trace["records"]):
        packed = trace_dir / record["packed_file"]
        values = trace_dir / record["value_payload_file"]
        require(sha256(packed) == record["packed_file_sha256"] and
                sha256(values) == record["value_payload_sha256"],
                "M248 payload SHA drift")
        masks = m43.unpack_record_masks(trace_dir, record)
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // 16)
                for subtile in range(m43.TILE_BITS // 16):
                    value = (value256 >> (subtile * 16)) & 0xffff
                    histograms[(record["sample_id"],
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
        print("[M339 HIST] {}/40".format(record_index + 1), flush=True)

    q_phases = {q: defaultdict(list) for q in Q_VALUES}
    aggregates = {q: Counter() for q in Q_VALUES}
    working_sets = {q: [] for q in Q_VALUES}
    for sample in range(model["samples"]):
        for op in range(model["operators"]):
            for partition in range(model["partitions_per_operator"]):
                nested = catalog["operators"][op]["partitions"][partition][
                    "nested_patterns"]
                nested_centers = [int(value, 16) for value in nested]
                phase_by_q = phase_metrics_all_q(
                    histograms[(sample, op, partition)], nested_centers)
                for q in Q_VALUES:
                    phase = phase_by_q[q]
                    require(phase["partition_vectors"] == model["rows_per_operator"],
                            "runtime phase population drift")
                    q_phases[q][sample].append(phase)
                    aggregates[q].update(phase)
                    working_sets[q].append(phase["used_pwp_patterns"])
        print("[M339 METRIC] sample={}/10".format(sample + 1), flush=True)

    common_commit = (model["operators"] * model["rows_per_operator"] *
                     model["output_blocks"] /
                     model["commit_output_blocks_per_cycle"])
    require(int(common_commit) == common_commit, "nonintegral common commit")
    common_commit = int(common_commit)
    cycle_rows = []
    legacy_rows = []
    for port in PORTS:
        legacy_sparse = legacy_candidate = 0
        for sample in range(model["samples"]):
            sparse, candidate = replay_legacy_m251(
                q_phases[16][sample], port, model)
            legacy_sparse += sparse
            legacy_candidate += candidate
        parent_row = next(row for row in m251["same_resource_cycle_simulations"]
                          if row["port"] == port["name"])
        require(legacy_sparse == parent_row["bit_sparse_cycles"] and
                legacy_candidate == parent_row["candidate_cycles"],
                "q16 legacy M251 cycle reproduction failure")
        legacy_rows.append({"port": port["name"],
                            "bit_sparse_cycles": legacy_sparse,
                            "candidate_cycles": legacy_candidate,
                            "reproduces_m251": True})
        for q in Q_VALUES:
            for architecture in ("SYSTOLIC_Q_II1", "SERIAL16_II1"):
                bit_total = candidate_total = 0
                for sample in range(model["samples"]):
                    phases = q_phases[q][sample]
                    bit_total += replay_bit_sparse(phases, port, model, common_commit)
                    candidate_total += replay_kfirst(
                        phases, q, port, architecture, True, model,
                        common_commit)["cycles"]
                maximum_ws = max(working_sets[q])
                row = {
                    "q_capacity": q,
                    "port": port["name"],
                    "matcher_architecture": architecture,
                    "traffic": "SELECTIVE_RUNTIME_USED_PWP",
                    "bit_sparse_cycles": bit_total,
                    "candidate_cycles": candidate_total,
                    "speedup_vs_bit_sparse": bit_total / float(candidate_total),
                    "maximum_used_patterns_per_phase": maximum_ws,
                    "single_64kb_current_phase_fit":
                        maximum_ws * model["pwp_pattern_all_blocks_bytes"] <= 64 << 10,
                    "cache_double_buffer_fit": [{
                        "total_cache_bytes": cache,
                        "per_context_bytes": cache // 2,
                        "all_phases_fit": maximum_ws *
                            model["pwp_pattern_all_blocks_bytes"] <= cache // 2,
                    } for cache in CACHE_BYTES],
                    "cycle_admitted": False,
                    "reason":
                        "K-first recurrence prices preprocess-before-DMA and all raw rows, but finite queues, chunking for non-fitting phases, bank conflicts and RTL cycle match remain open.",
                }
                cycle_rows.append(row)

    work_rows = []
    phases_total = model["samples"] * model["operators"] * model[
        "partitions_per_operator"]
    for q in Q_VALUES:
        total = aggregates[q]
        ws = working_sets[q]
        selective_bytes = sum(ws) * model["pwp_pattern_all_blocks_bytes"]
        full_bytes = phases_total * q * model["pwp_pattern_all_blocks_bytes"]
        work_rows.append({
            "q_capacity": q,
            "bit_sparse_vector_ops_per_block":
                total["bit_sparse_vector_ops_per_block"],
            "candidate_vector_ops_per_block":
                total["candidate_vector_ops_per_block"],
            "exact_vector_op_speedup":
                total["bit_sparse_vector_ops_per_block"] /
                float(total["candidate_vector_ops_per_block"]),
            "pwp_ops_per_block": total["pwp_ops_per_block"],
            "correction_ops_per_block": total["correction_ops_per_block"],
            "used_patterns_per_phase": {
                "minimum": min(ws), "p50": percentile(ws, 0.50),
                "p90": percentile(ws, 0.90), "p99": percentile(ws, 0.99),
                "maximum": max(ws), "mean": sum(ws) / float(len(ws)),
            },
            "full_table_pwp_bytes": full_bytes,
            "selective_pwp_bytes": selective_bytes,
            "selective_traffic_reduction": full_bytes / float(selective_bytes),
        })

    payload = {
        "schema": "m339_q128_selective_pwp_kfirst_cycle_v1",
        "status": "PASS_M339_EXACT_WORK_AND_PINNED_KFIRST_CYCLE_UPPER_UNADMITTED",
        "identity": identities,
        "cycle_model": model,
        "exact_runtime_working_set": work_rows,
        "legacy_q16_replay": legacy_rows,
        "kfirst_cycle_dse": cycle_rows,
        "admission": {
            "exact_runtime_vector_work": True,
            "selective_pwp_traffic_count": True,
            "m251_q16_legacy_reproduction": True,
            "executable_finite_queue_cycle": False,
            "rtl_cycle_match": False,
            "area_matched": False,
            "energy": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "claim_boundary":
            "Exact four-Conv PAFT-ep4 S10 vector work, selective PWP traffic and a pinned K-first recurrence only. The recurrence consumes every raw row and serializes next-phase match before its DMA, but finite queues, cache chunking, bank conflicts, area normalization, RTL cycle match, energy, system and headline remain unadmitted.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m339_q128_selective_pwp_kfirst_cycle_r1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M339_PASS " + " ".join(
        "q{}={:.6f}x/ws{}".format(row["q_capacity"],
                                   row["exact_vector_op_speedup"],
                                   row["used_patterns_per_phase"]["maximum"])
        for row in work_rows), flush=True)


if __name__ == "__main__":
    main()
