#!/usr/bin/env python3
"""Price Phi-style output-block tiling for q128 exact PWP reuse in 64 KiB."""

from __future__ import division

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path


Q_TO_OUTPUT_TILE = {16: 8, 32: 4, 64: 2, 128: 1}
PORTS = (
    {"name": "WIDE144_PWP_96_WEIGHT", "weight_cycles": 1, "pwp_cycles": 1},
    {"name": "SHARED96", "weight_cycles": 1, "pwp_cycles": 2},
)


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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matcher_cycles(phase, q, architecture):
    if architecture == "SYSTOLIC_Q_II1":
        return phase["partition_vectors"] + q
    require(architecture == "SERIAL16_II1", "unknown matcher")
    return (phase["partition_vectors"] +
            phase["matcher_rows"] * (int(math.ceil(q / 16.0)) - 1) + 2)


def candidate_tile_bytes(phase, q, output_tile, model):
    return (phase["used_pwp_patterns"] *
            model["pwp_vector_bytes_per_output_block"] * output_tile +
            model["partition_bits"] * model["weight_vector_bytes"] * output_tile +
            q * model["pattern_bytes"])


def candidate_tile_load_cycles(phase, q, output_tile, model):
    return int(math.ceil(candidate_tile_bytes(phase, q, output_tile, model) /
                         float(model["dram_bytes_per_cycle"])))


def candidate_body_cycles(phase, q, output_tile, port, model):
    tiles = model["output_blocks"] // output_tile
    compute = (
        phase["correction_ops_per_block"] * output_tile *
        port["weight_cycles"] +
        phase["pwp_ops_per_block"] * output_tile * port["pwp_cycles"])
    load = candidate_tile_load_cycles(phase, q, output_tile, model)
    # The first output tile is loaded by the prior/initial stage. Every following
    # tile load shares the alternate cache context with current-tile compute.
    return ((tiles - 1) * max(compute, load) + compute +
            model["compute_tail_cycles_per_partition"])


def replay_candidate(phases, q, output_tile, port, architecture, model,
                     common_commit, allow_next_first_tile_overlap):
    def matcher_packer(phase):
        matcher = matcher_cycles(phase, q, architecture)
        packer = int(math.ceil(phase["assignment_rows"] /
                               float(model["packer_lanes"]))) + model[
                                   "packer_pipeline_cycles"]
        # Pattern data must be resident before the first match of a partition.
        pattern_load = int(math.ceil(q * model["pattern_bytes"] /
                                     float(model["dram_bytes_per_cycle"])))
        return pattern_load + matcher + packer

    first_pre = matcher_packer(phases[0])
    first_load = candidate_tile_load_cycles(
        phases[0], q, output_tile, model)
    cycles = first_pre + first_load
    bindings = Counter()
    for index, phase in enumerate(phases):
        body = candidate_body_cycles(phase, q, output_tile, port, model)
        if index + 1 < len(phases):
            next_pre = matcher_packer(phases[index + 1])
            next_load = candidate_tile_load_cycles(
                phases[index + 1], q, output_tile, model)
            if allow_next_first_tile_overlap:
                other = next_pre + next_load
                cycles += max(body, other)
            else:
                other = next_pre
                cycles += max(body, other) + next_load
            bindings["body" if body >= other else "next_preprocess"] += 1
        else:
            cycles += body
    cycles += common_commit
    return {"cycles": cycles, "binding_phases": dict(bindings)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M344 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m344_output_block_tiled_q128_kfirst_contract_v1",
            "M344 contract schema drift")
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

    m339 = load_module(paths["m339_analyzer"], "m344_frozen_m339")
    m43 = load_module(paths["m43_support_unpacker"], "m344_frozen_m43")
    catalog = strict_json(paths["m338_catalog"])
    trace = strict_json(paths["m248_runtime_trace"])
    parent = strict_json(paths["m339_result"])
    require(parent["status"] ==
            "PASS_M339_EXACT_WORK_AND_PINNED_KFIRST_CYCLE_UPPER_UNADMITTED" and
            parent["admission"]["exact_runtime_vector_work"] is True and
            parent["admission"]["executable_finite_queue_cycle"] is False,
            "M339 parent admission drift")
    require(catalog["status"] ==
            "PASS_M338_TRAIN_ONLY_NESTED_Q16_Q32_Q64_Q128_EXACT_WORK_NO_CYCLES" and
            trace["status"] ==
            "PASS_PAFT_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE",
            "M338/M248 status drift")
    model = contract["cycle_model"]
    require(m43.ROWS == model["rows_per_operator"] and
            m43.TILES * (m43.TILE_BITS // model["partition_bits"]) ==
                model["partitions_per_operator"],
            "M43 geometry drift")

    operators = tuple(trace["cohort"]["operators"])
    op_index = {name: index for index, name in enumerate(operators)}
    trace_dir = paths["m248_runtime_trace"].parent
    histograms = defaultdict(Counter)
    for record_index, record in enumerate(trace["records"]):
        require(sha256(trace_dir / record["packed_file"]) ==
                record["packed_file_sha256"] and
                sha256(trace_dir / record["value_payload_file"]) ==
                record["value_payload_sha256"], "M248 payload drift")
        masks = m43.unpack_record_masks(trace_dir, record)
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // model["partition_bits"])
                for subtile in range(m43.TILE_BITS // model["partition_bits"]):
                    value = ((value256 >> (subtile * model["partition_bits"])) &
                             0xffff)
                    histograms[(record["sample_id"],
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
        print("[M344 HIST] {}/40".format(record_index + 1), flush=True)

    phases = {q: defaultdict(list) for q in Q_TO_OUTPUT_TILE}
    aggregate = {q: Counter() for q in Q_TO_OUTPUT_TILE}
    for sample in range(model["samples"]):
        for op in range(model["operators"]):
            for partition in range(model["partitions_per_operator"]):
                nested = catalog["operators"][op]["partitions"][partition][
                    "nested_patterns"]
                phase_by_q = m339.phase_metrics_all_q(
                    histograms[(sample, op, partition)],
                    [int(value, 16) for value in nested])
                for q in Q_TO_OUTPUT_TILE:
                    phase = phase_by_q[q]
                    phases[q][sample].append(phase)
                    aggregate[q].update(phase)
        print("[M344 METRIC] sample={}/10".format(sample + 1), flush=True)

    common_commit = (model["operators"] * model["rows_per_operator"] *
                     model["output_blocks"] //
                     model["commit_output_blocks_per_cycle"])
    parent_work = {row["q_capacity"]: row
                   for row in parent["exact_runtime_working_set"]}
    for q in Q_TO_OUTPUT_TILE:
        require(aggregate[q]["candidate_vector_ops_per_block"] ==
                parent_work[q]["candidate_vector_ops_per_block"] and
                aggregate[q]["bit_sparse_vector_ops_per_block"] ==
                parent_work[q]["bit_sparse_vector_ops_per_block"],
                "M339 exact work reproduction failure")

    rows = []
    for q, output_tile in sorted(Q_TO_OUTPUT_TILE.items()):
        maximum_context = max(
            candidate_tile_bytes(phase, q, output_tile, model)
            for sample in range(model["samples"]) for phase in phases[q][sample])
        require(2 * maximum_context <= model["pwp_weight_pattern_cache_bytes"],
                "frozen q/output-tile pair does not fit 64KiB double contexts")
        for port in PORTS:
            bit_cycles = sum(
                m339.replay_bit_sparse(phases[q][sample], port, model,
                                       common_commit)
                for sample in range(model["samples"]))
            for architecture in ("SYSTOLIC_Q_II1", "SERIAL16_II1"):
                strict = sum(replay_candidate(
                    phases[q][sample], q, output_tile, port, architecture,
                    model, common_commit, False)["cycles"]
                    for sample in range(model["samples"]))
                overlap = sum(replay_candidate(
                    phases[q][sample], q, output_tile, port, architecture,
                    model, common_commit, True)["cycles"]
                    for sample in range(model["samples"]))
                require(overlap <= strict, "overlap/strict ordering failure")
                rows.append({
                    "q_capacity": q,
                    "output_block_tile": output_tile,
                    "output_tiles_per_partition":
                        model["output_blocks"] // output_tile,
                    "port": port["name"],
                    "matcher_architecture": architecture,
                    "maximum_context_bytes_including_weight_pattern_pwp":
                        maximum_context,
                    "double_context_bytes": 2 * maximum_context,
                    "fits_64kb": True,
                    "descriptor_sram_bytes_two_contexts":
                        2 * model["rows_per_operator"] *
                        model["descriptor_bytes_per_row"],
                    "bit_sparse_cycles": bit_cycles,
                    "strict_first_tile_serial_cycles": strict,
                    "strict_speedup_vs_bit_sparse": bit_cycles / float(strict),
                    "last_tile_first_tile_overlap_cycles": overlap,
                    "overlap_speedup_vs_bit_sparse": bit_cycles / float(overlap),
                    "cycle_admitted": False,
                })

    payload = {
        "schema": "m344_output_block_tiled_q128_kfirst_v1",
        "status": "PASS_M344_FIXED64KB_OUTPUT_BLOCK_TILING_CYCLE_BOUNDS_UNADMITTED",
        "identity": identities,
        "mechanism": {
            "fixed_total_cache_bytes": model["pwp_weight_pattern_cache_bytes"],
            "equal_contexts": True,
            "q_to_output_block_tile": {str(q): tile
                                        for q, tile in Q_TO_OUTPUT_TILE.items()},
            "invariant": "q_capacity times output_block_tile is 128",
            "matcher_once_per_partition": True,
            "assignment_descriptor_reused_across_output_tiles": True,
            "exact_arithmetic": True,
        },
        "cycle_bounds": rows,
        "admission": {
            "m339_exact_work_reproduced": True,
            "fixed64kb_capacity_fit": True,
            "cycle_bound": True,
            "finite_queue_executable_cycle": False,
            "bank_conflict_trace": False,
            "rtl_cycle_match": False,
            "area_matched": False,
            "energy": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "claim_boundary":
            "Fixed-64KiB output-block-tiling capacity proof and two K-first module cycle bounds only. Descriptor SRAM is priced separately. Finite queues, bank conflicts, RTL cycle match, area normalization, energy, system and headline remain unadmitted.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m344_output_block_tiled_q128_kfirst_r1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    selected = [row for row in rows if row["q_capacity"] == 128 and
                row["port"] == "WIDE144_PWP_96_WEIGHT" and
                row["matcher_architecture"] == "SYSTOLIC_Q_II1"][0]
    print("M344_PASS q128_wide_strict={:.6f}x overlap={:.6f}x context={}B".format(
        selected["strict_speedup_vs_bit_sparse"],
        selected["overlap_speedup_vs_bit_sparse"],
        selected["maximum_context_bytes_including_weight_pattern_pwp"]),
        flush=True)


if __name__ == "__main__":
    main()
