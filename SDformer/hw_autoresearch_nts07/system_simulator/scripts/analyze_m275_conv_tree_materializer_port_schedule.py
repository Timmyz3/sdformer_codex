#!/usr/bin/env python3
"""Replay an explicit two-bank PWP-tree preparation schedule on M251 phases."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path


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
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + token)),
        )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen support module: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_flips(centers):
    values = [0] + list(centers)
    visited = {0}
    total = 0
    while len(visited) != len(values):
        distance, parent, child = min(
            (bin(values[parent] ^ values[child]).count("1"), parent, child)
            for parent in visited
            for child in range(len(values)) if child not in visited
        )
        visited.add(child)
        total += distance
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m275_conv_tree_materializer_port_schedule_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "frozen input SHA drift {}: {}".format(name, observed))
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}

    m251_contract = strict_json(paths["m251_contract"])
    catalog = strict_json(paths["m77_train_catalog"])
    trace = strict_json(paths["m248_trace"])
    m251_result = strict_json(paths["m251_result"])
    m267_result = strict_json(paths["m267_result"])
    require(m251_result["status"] ==
            "PASS_PAFT_RUNNING_BN_TRAIN_CATALOG_ISOLATED_CONV_CYCLE_MODEL" and
            m267_result["status"] ==
            "PASS_EXACT_PWP_DERIVED_PAYLOAD_ELISION_CYCLES_UNCHANGED",
            "parent admission drift")
    geometry = m251_contract["geometry"]
    model = m251_contract["same_resource_cycle_model"]
    require(model["ports"][0]["name"] == "WIDE144_PWP_96_WEIGHT" and
            model["ports"][0]["pwp_vector_service_cycles"] == 1 and
            geometry["output_blocks"] == 8 and
            contract["preparation_recurrence"]["serial_load_cycles"] == 387,
            "wide-port/preparation contract drift")

    m251 = load_module(paths["m251_analyzer"], "m275_frozen_m251")
    m43 = load_module(paths["m43_support_unpacker"], "m275_frozen_m43")
    operator_names = trace["cohort"]["operators"]
    op_index = {name: index for index, name in enumerate(operator_names)}
    require(len(operator_names) == geometry["operators"] == 4 and
            len(trace["records"]) == 40 and
            m43.ROWS == geometry["rows_per_operator"],
            "trace geometry drift")

    histograms = defaultdict(Counter)
    trace_dir = paths["m248_trace"].parent
    reconstructed_records = []
    for record in trace["records"]:
        packed_path = trace_dir / record["packed_file"]
        require(packed_path.is_file() and
                sha256(packed_path) == record["packed_file_sha256"] and
                int(record["negative_count"]) == 0,
                "M248 payload/nonnegative identity drift")
        masks = m43.unpack_record_masks(trace_dir, record)
        reconstructed = 0
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // 16)
                for subtile in range(m43.TILE_BITS // 16):
                    value = (value256 >> (subtile * 16)) & 0xffff
                    histograms[(int(record["sample_id"]),
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
                    reconstructed += m251.popcount(value)
        reconstructed_records.append({
            "sample_id": int(record["sample_id"]),
            "operator_index": op_index[record["operator"]],
            "expanded_source_events": reconstructed,
        })

    flips_by_phase = []
    flip_histogram = Counter()
    for operator in catalog["operators"]:
        require(len(operator["partitions"]) ==
                geometry["partitions_per_operator"],
                "catalog partition extent drift")
        for partition, row in enumerate(operator["partitions"]):
            require(int(row["partition"]) == partition and
                    len(row["patterns"]) == 16,
                    "catalog pattern order drift")
            centers = [int(item["value_hex"], 16)
                       for item in row["patterns"]]
            require(len(set(centers)) == 16 and all(centers),
                    "catalog pattern domain drift")
            flips = tree_flips(centers)
            flips_by_phase.append(flips)
            flip_histogram[flips] += 1
    require(len(flips_by_phase) == 1728 and
            {str(key): value for key, value in sorted(flip_histogram.items())} ==
            m267_result["minimum_hamming_trees"][
                "partition_flip_count_histogram"],
            "M267 flip histogram drift")

    first_prep = 387 + 8 * flips_by_phase[0]
    sample_rows = []
    total = Counter()
    minimum_overlap_slack = None
    worst_transition = None
    for sample in range(geometry["samples"]):
        phases = []
        for op in range(geometry["operators"]):
            for partition, row in enumerate(
                    catalog["operators"][op]["partitions"]):
                centers = [int(item["value_hex"], 16)
                           for item in row["patterns"]]
                phases.append(m251.phase_metrics(
                    histograms[(sample, op, partition)], centers))
        require(len(phases) == 1728, "sample phase population drift")
        old_cycles = 960
        new_cycles = first_prep
        sample_min_slack = None
        exposed = 0
        compute_sum = 0
        for index, phase in enumerate(phases):
            compute = ((phase["correction_ops_per_block"] +
                        phase["pwp_ops_per_block"]) * 8)
            matcher = phase["matcher_rows"] + 16
            packer = (int(math.ceil(phase["assignment_rows"] / 8.0)) + 4)
            current_service = max(compute, matcher, packer)
            old_next = 960 if index + 1 < len(phases) else 0
            new_next = (387 + 8 * flips_by_phase[index + 1]
                        if index + 1 < len(phases) else 0)
            old_cycles += max(current_service, old_next) + 2
            new_cycles += max(current_service, new_next) + 2
            compute_sum += compute
            if index + 1 < len(phases):
                slack = current_service - new_next
                sample_min_slack = (slack if sample_min_slack is None
                                    else min(sample_min_slack, slack))
                if minimum_overlap_slack is None or slack < minimum_overlap_slack:
                    minimum_overlap_slack = slack
                    worst_transition = {
                        "sample_id": sample,
                        "current_phase": index,
                        "next_phase": index + 1,
                        "current_service_cycles": current_service,
                        "next_preparation_cycles": new_next,
                        "slack_cycles": slack,
                        "next_tree_flips": flips_by_phase[index + 1],
                    }
                if slack < 0:
                    exposed += -slack
        parent = m251_result["same_resource_cycle_simulations"][0][
            "per_sample"][sample]
        require(old_cycles == int(parent["candidate_cycles"]),
                "M251 per-sample cycle replay drift")
        sample_rows.append({
            "sample_id": sample,
            "stored_pwp_cycles": old_cycles,
            "tree_materialized_cycles": new_cycles,
            "cycle_reduction": old_cycles - new_cycles,
            "minimum_transition_slack_cycles": sample_min_slack,
            "exposed_transition_cycles": exposed,
            "candidate_compute_cycles_sum": compute_sum,
        })
        total["old_cycles"] += old_cycles
        total["new_cycles"] += new_cycles
        total["exposed"] += exposed

    parent_wide = m251_result["same_resource_cycle_simulations"][0]
    require(total["old_cycles"] == int(parent_wide["candidate_cycles"]) and
            total["exposed"] == 0 and minimum_overlap_slack is not None and
            minimum_overlap_slack >= 0,
            "port-feasible overlap gate failed")
    bit_cycles = int(parent_wide["bit_sparse_cycles"])
    dense_cycles = int(parent_wide["dense_cycles"])
    output = {
        "schema": "m275_conv_tree_materializer_port_schedule_v1",
        "status": "PASS_EXACT_TWO_BANK_ZERO_EXPOSED_MATERIALIZATION_SCHEDULE",
        "identity": identities,
        "scope": {
            "samples": 10,
            "operators": 4,
            "phases": 17280,
            "partition_transitions": 17270,
            "cold_sample_starts": 10,
            "records": 40,
        },
        "physical_port_contract": contract["physical_port_contract"],
        "onchip_capacity": {
            "two_weight_banks_bytes": 24576,
            "two_pwp_banks_bytes": 36864,
            "total_weight_plus_pwp_bytes": 61440,
            "pwp_capacity_eliminated": False,
        },
        "preparation": {
            "serial_load_cycles": 387,
            "first_partition_tree_flips": flips_by_phase[0],
            "cold_start_cycles_per_sample": first_prep,
            "maximum_next_preparation_cycles":
                387 + 8 * max(flips_by_phase),
            "minimum_overlap_slack_cycles": minimum_overlap_slack,
            "worst_transition": worst_transition,
            "exposed_transition_cycles": total["exposed"],
            "operator_boundaries_included": True,
            "sample_boundaries_cold": True,
            "final_drain_has_no_next_preparation": True,
            "bank_switch_covered_by_existing_two_cycle_tail": True,
        },
        "cycles": {
            "stored_fixed_pwp_wide": total["old_cycles"],
            "tree_materialized_wide": total["new_cycles"],
            "cycle_reduction": total["old_cycles"] - total["new_cycles"],
            "bit_sparse": bit_cycles,
            "dense": dense_cycles,
            "speedup_vs_bit_sparse": bit_cycles / float(total["new_cycles"]),
            "speedup_vs_dense": dense_cycles / float(total["new_cycles"]),
            "speedup_vs_stored_fixed_pwp":
                total["old_cycles"] / float(total["new_cycles"]),
        },
        "sample_rows": sample_rows,
        "m267_storage_and_dram": m267_result["storage_and_dram"],
        "admission": {
            "exact_phase_replay": True,
            "explicit_bank_lifecycle": True,
            "zero_exposed_transition_cycles": True,
            "modeled_offchip_pwp_payload_elimination": True,
            "onchip_pwp_capacity_elimination": False,
            "generator_energy": False,
            "sram_macro": False,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "complete_conv": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "record_expansion_audit": reconstructed_records,
    }
    require(sha256(Path(__file__).resolve()) == source_start,
            "M275 analyzer changed during execution")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = (
        args.output_dir / "m275_conv_tree_materializer_port_schedule_r1.json"
    )
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("M275_PASS old={} new={} exposed={} min_slack={} sparse={:.9f}".format(
        total["old_cycles"], total["new_cycles"], total["exposed"],
        minimum_overlap_slack, output["cycles"]["speedup_vs_bit_sparse"]),
        flush=True)


if __name__ == "__main__":
    main()
