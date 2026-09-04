#!/usr/bin/env python3
"""Replay M275r2 with explicit conservative single-port materializer events."""

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


def popcount(value):
    method = getattr(int(value), "bit_count", None)
    return method() if method is not None else bin(int(value)).count("1")


def prim_tree(centers):
    values = [0] + list(centers)
    visited = {0}
    edges = []
    while len(visited) != len(values):
        distance, parent, child = min(
            (popcount(values[parent] ^ values[child]), parent, child)
            for parent in visited
            for child in range(len(values)) if child not in visited
        )
        edges.append((parent, child, distance))
        visited.add(child)
    return edges


def phase_service(phase, output_blocks):
    compute = ((phase["correction_ops_per_block"] +
                phase["pwp_ops_per_block"]) * output_blocks)
    matcher = phase["matcher_rows"] + 16
    packer = int(math.ceil(phase["assignment_rows"] / 8.0)) + 4
    return max(compute, matcher, packer), compute, matcher, packer


def schedule_tree(edges, output_blocks, serial_fill_cycles):
    """Build one catalog-phase schedule and reject every port/valid hazard."""
    events = Counter()
    weight_port = {}
    pwp_port = {}
    metadata_port = {}
    dram_port = {}

    # One shared 32-byte/cycle link: 384 weight cycles then three metadata cycles.
    for cycle in range(384):
        require(cycle not in weight_port and cycle not in dram_port,
                "weight/DRAM fill conflict")
        weight_port[cycle] = "weight_fill_32B"
        dram_port[cycle] = "weight_fill_32B"
        events["weight_fill_32B_cycles"] += 1
    for cycle in range(384, serial_fill_cycles):
        require(cycle not in metadata_port and cycle not in dram_port,
                "metadata/DRAM fill conflict")
        metadata_port[cycle] = "metadata_fill_32B"
        dram_port[cycle] = "metadata_fill_32B"
        events["metadata_fill_32B_cycles"] += 1

    weight_valid_cycle = 384
    metadata_valid_cycle = serial_fill_cycles
    cycle = serial_fill_cycles
    nonresident_parent_read_events = 0
    flips = sum(distance for _, _, distance in edges)
    edge_distance_histogram = Counter(distance for _, _, distance in edges)
    entry_use_before_write = 0

    for output_block in range(output_blocks):
        resident = 0
        written = {0}
        for parent, child, distance in edges:
            require(parent in written and child not in written,
                    "Prim traversal violates parent-before-child")
            needs_parent_read = parent != 0 and parent != resident
            if needs_parent_read:
                nonresident_parent_read_events += 1
                if parent not in written:
                    entry_use_before_write += 1
                require(cycle not in pwp_port,
                        "single PWP port parent-read conflict")
                pwp_port[cycle] = "parent_pwp_read_144B"
                require(cycle not in metadata_port,
                        "metadata descriptor conflict")
                metadata_port[cycle] = "tree_descriptor_read_4B"
                events["parent_pwp_read_144B_cycles"] += 1
                events["tree_descriptor_read_events"] += 1
                require(cycle >= metadata_valid_cycle,
                        "descriptor read before metadata valid")
                cycle += 1
            else:
                require(cycle not in metadata_port,
                        "metadata descriptor conflict")
                metadata_port[cycle] = "tree_descriptor_read_4B"
                events["tree_descriptor_read_events"] += 1
                require(cycle >= metadata_valid_cycle,
                        "descriptor read before metadata valid")

            require(distance >= 1, "zero-distance catalog edge")
            for update in range(distance):
                require(cycle >= weight_valid_cycle,
                        "generator weight read before weight valid")
                require(cycle not in weight_port,
                        "next weight fill/read or read/read conflict")
                weight_port[cycle] = "generator_weight_read_96B"
                events["generator_weight_read_96B_cycles"] += 1
                if update + 1 == distance:
                    require(cycle not in pwp_port,
                            "single PWP port child-write conflict")
                    pwp_port[cycle] = "child_pwp_write_144B"
                    events["child_pwp_write_144B_cycles"] += 1
                cycle += 1
            written.add(child)
            resident = child

    require(entry_use_before_write == 0, "parent PWP use-before-write")
    require(events["weight_fill_32B_cycles"] == 384 and
            events["metadata_fill_32B_cycles"] == 3 and
            events["generator_weight_read_96B_cycles"] ==
                output_blocks * flips and
            events["parent_pwp_read_144B_cycles"] ==
                nonresident_parent_read_events and
            events["child_pwp_write_144B_cycles"] ==
                output_blocks * len(edges) and
            events["tree_descriptor_read_events"] ==
                output_blocks * len(edges),
            "event conservation")
    require(nonresident_parent_read_events % output_blocks == 0,
            "per-output-block parent-read count drift")
    nonresident_edges = nonresident_parent_read_events // output_blocks
    expected = serial_fill_cycles + output_blocks * (flips + nonresident_edges)
    require(cycle == expected, "preparation recurrence drift")
    return {
        "tree_edges": len(edges),
        "tree_flips": flips,
        "nonresident_parent_edges": nonresident_edges,
        "edge_distance_histogram": {
            str(key): value for key, value in sorted(
                edge_distance_histogram.items())
        },
        "preparation_cycles": cycle,
        "weight_valid_offset": weight_valid_cycle,
        "metadata_valid_offset": metadata_valid_cycle,
        "pwp_valid_offset": cycle,
        "generator_idle_offset": cycle,
        "same_bank_port_conflicts": 0,
        "entry_use_before_write": entry_use_before_write,
        "events": dict(events),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    analyzer_path = Path(__file__).resolve()
    analyzer_start = sha256(analyzer_path)
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m275r2_conv_tree_single_port_materializer_contract_v1",
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

    require(contract["capacity_contract"]["minimum_complete_modeled_bytes"] ==
            61776 and
            contract["memory_and_port_contract"]["generator_accumulators"] == 1 and
            contract["preparation_recurrence"]["serial_fill_cycles"] == 387,
            "conservative resource contract drift")
    m251_contract = strict_json(paths["m251_contract"])
    m251_result = strict_json(paths["m251_result"])
    m251r2_contract = strict_json(paths["m251r2_contract"])
    m251r2_result = strict_json(paths["m251r2_result"])
    m267_result = strict_json(paths["m267_result"])
    m279 = strict_json(paths["m279_independent_recompute"])
    catalog = strict_json(paths["m77_train_catalog"])
    trace = strict_json(paths["m248_trace"])
    require(m251r2_contract["correct_numeric_domain"]["pwp_sum_minimum"] ==
            -2048 and
            m251r2_result["corrected_full_signed_int8_pwp_range"]["sum_range"] ==
            [-2048, 2032] and
            m251r2_result["corrected_full_signed_int8_pwp_range"]["signed12_safe"]
            is True and
            m251r2_result["unchanged_performance"]
                ["m251_r1_result_remains_cycle_source"] is True,
            "M251r2 numeric correction not bound")

    geometry = m251_contract["geometry"]
    output_blocks = int(geometry["output_blocks"])
    require((geometry["samples"], geometry["operators"],
             geometry["partitions_per_operator"], output_blocks) ==
            (10, 4, 432, 8), "geometry drift")
    m251 = load_module(paths["m251_analyzer"], "m275r2_frozen_m251")
    m43 = load_module(paths["m43_support_unpacker"], "m275r2_frozen_m43")

    operator_names = trace["cohort"]["operators"]
    op_index = {name: index for index, name in enumerate(operator_names)}
    require(len(trace["records"]) == 40 and len(operator_names) == 4 and
            [row["operator"] for row in catalog["operators"]] == operator_names,
            "trace/catalog extent drift")
    histograms = defaultdict(Counter)
    trace_dir = paths["m248_trace"].parent
    record_audit = []
    for record in trace["records"]:
        packed_path = trace_dir / record["packed_file"]
        require(packed_path.is_file() and
                sha256(packed_path) == record["packed_file_sha256"] and
                int(record["negative_count"]) == 0,
                "payload SHA or source sign drift")
        masks = m43.unpack_record_masks(trace_dir, record)
        reconstructed = 0
        sample = int(record["sample_id"])
        op = op_index[record["operator"]]
        for row_index in range(m43.ROWS):
            base = row_index * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // 16)
                for subtile in range(m43.TILE_BITS // 16):
                    value = (value256 >> (subtile * 16)) & 0xffff
                    histograms[(sample, op, partition_base + subtile)][value] += 1
                    reconstructed += popcount(value)
        record_audit.append({
            "sample_id": sample,
            "operator_index": op,
            "packed_file_sha256": record["packed_file_sha256"],
            "expanded_source_events": reconstructed,
        })

    centers_by_phase = []
    trees = []
    flip_histogram = Counter()
    edge_distance_histogram = Counter()
    phase_schedule_catalog = []
    tree_events_per_catalog = Counter()
    for operator_index, operator in enumerate(catalog["operators"]):
        require(len(operator["partitions"]) == 432,
                "catalog partition extent")
        for partition, row in enumerate(operator["partitions"]):
            require(int(row["partition"]) == partition and
                    len(row["patterns"]) == 16,
                    "catalog partition order")
            centers = [int(item["value_hex"], 16)
                       for item in row["patterns"]]
            require(len(set(centers)) == 16 and all(centers),
                    "catalog pattern domain")
            edges = prim_tree(centers)
            schedule = schedule_tree(edges, output_blocks, 387)
            phase_index = len(trees)
            phase_schedule_catalog.append({
                "phase_index": phase_index,
                "operator_index": operator_index,
                "partition": partition,
                **schedule,
            })
            centers_by_phase.append(centers)
            trees.append(schedule)
            flip_histogram[schedule["tree_flips"]] += 1
            edge_distance_histogram.update(
                {int(key): value for key, value in
                 schedule["edge_distance_histogram"].items()})
            tree_events_per_catalog.update(schedule["events"])

    require(len(trees) == 1728 and
            {str(key): value for key, value in sorted(flip_histogram.items())} ==
                m267_result["minimum_hamming_trees"]
                    ["partition_flip_count_histogram"] and
            {str(key): value for key, value in
             sorted(edge_distance_histogram.items())} ==
                m267_result["minimum_hamming_trees"]
                    ["edge_distance_histogram"],
            "M267 tree reconstruction drift")

    phases_by_sample = {}
    work = Counter()
    for sample in range(10):
        phases = []
        for phase_index, centers in enumerate(centers_by_phase):
            op, partition = divmod(phase_index, 432)
            phase = m251.phase_metrics(
                histograms[(sample, op, partition)], centers)
            phases.append(phase)
            work.update(phase)
        require(len(phases) == 1728, "sample phase extent")
        phases_by_sample[sample] = phases
    exact_work = m251_result["exact_natural_work"]
    require(work["partition_vectors"] == 51840000 and
            work["bit_sparse_vector_ops_per_block"] ==
                exact_work["bit_sparse_vector_ops_per_block"] and
            work["candidate_vector_ops_per_block"] ==
                exact_work["candidate_vector_ops_per_block"] and
            work["correction_ops_per_block"] ==
                exact_work["correction_ops_per_block"] and
            work["pwp_ops_per_block"] == exact_work["pwp_ops_per_block"],
            "M251 work reconstruction drift")

    first_prep = trees[0]["preparation_cycles"]
    sample_rows = []
    total = Counter()
    global_min_slack = None
    worst_transition = None
    for sample in range(10):
        old_cycles = 960
        new_cycles = first_prep
        sample_min_slack = None
        exposed = 0
        compute_sum = 0
        for index, phase in enumerate(phases_by_sample[sample]):
            service, compute, matcher, packer = phase_service(
                phase, output_blocks)
            old_next = 960 if index + 1 < 1728 else 0
            new_next = (trees[index + 1]["preparation_cycles"]
                        if index + 1 < 1728 else 0)
            old_cycles += max(service, old_next) + 2
            new_cycles += max(service, new_next) + 2
            compute_sum += compute
            if index + 1 < 1728:
                slack = service - new_next
                sample_min_slack = (slack if sample_min_slack is None else
                                    min(sample_min_slack, slack))
                exposed += max(0, -slack)
                if global_min_slack is None or slack < global_min_slack:
                    global_min_slack = slack
                    worst_transition = {
                        "sample_id": sample,
                        "current_phase": index,
                        "next_phase": index + 1,
                        "current_service_cycles": service,
                        "current_compute_cycles": compute,
                        "current_matcher_cycles": matcher,
                        "current_packer_cycles": packer,
                        "next_preparation_cycles": new_next,
                        "next_tree_flips": trees[index + 1]["tree_flips"],
                        "next_nonresident_parent_edges":
                            trees[index + 1]["nonresident_parent_edges"],
                        "slack_cycles": slack,
                    }
        parent_sample = m251_result["same_resource_cycle_simulations"][0][
            "per_sample"][sample]
        require(old_cycles == int(parent_sample["candidate_cycles"]),
                "stored-PWP per-sample replay drift")
        sample_rows.append({
            "sample_id": sample,
            "stored_fixed_pwp_cycles": old_cycles,
            "tree_single_port_cycles": new_cycles,
            "cycle_reduction": old_cycles - new_cycles,
            "minimum_transition_slack_cycles": sample_min_slack,
            "exposed_transition_cycles": exposed,
            "candidate_compute_cycles_sum": compute_sum,
        })
        total["stored"] += old_cycles
        total["tree"] += new_cycles
        total["exposed"] += exposed

    wide = m251_result["same_resource_cycle_simulations"][0]
    m279_sync = m279["port_challenge"][
        "single_pwp_port_one_accumulator_sync_read_stress"]
    require(total["stored"] == int(wide["candidate_cycles"]) == 352335120 and
            total["tree"] == 352332590 and total["exposed"] == 0 and
            first_prep == 707 and max(row["preparation_cycles"] for row in trees)
                == 827 and global_min_slack == 333,
            "M275r2 conservative admission values drift")
    require(total["tree"] == int(m279_sync["tree_cycles"]) and
            first_prep == int(m279_sync["cold_start"]) and
            global_min_slack == int(m279_sync["minimum_slack"]) and
            sum(row["nonresident_parent_edges"] for row in trees) == 5041,
            "M279 repair target mismatch")

    # Current banks serve the active Conv. Next-bank materializer events repeat
    # once per sample because the contract forbids cross-sample cache reuse.
    event_ledger = {
        "current_weight_child_read": {
            "event": "correction-source weight read",
            "bytes_per_event": 96,
            "events": work["correction_ops_per_block"] * output_blocks,
            "cycles": work["correction_ops_per_block"] * output_blocks,
            "port": "current_weight_bank_96B_read",
        },
        "current_pwp_child_read": {
            "event": "selected child PWP read",
            "bytes_per_event": 144,
            "events": work["pwp_ops_per_block"] * output_blocks,
            "cycles": work["pwp_ops_per_block"] * output_blocks,
            "port": "current_pwp_bank_144B_read",
        },
        "next_weight_fill": {
            "bytes_per_cycle": 32,
            "events": 17280 * 384,
            "bytes": 17280 * 12288,
            "cycles": 17280 * 384,
            "port": "next_weight_bank_time_multiplexed_fill",
        },
        "next_metadata_fill": {
            "bytes_per_cycle": 32,
            "events": 17280 * 3,
            "bytes": 17280 * 96,
            "cycles": 17280 * 3,
            "port": "shared_DRAM_link_to_next_metadata_bank",
        },
        "next_generator_weight_read": {
            "bytes_per_event": 96,
            "events": tree_events_per_catalog[
                "generator_weight_read_96B_cycles"] * 10,
            "cycles": tree_events_per_catalog[
                "generator_weight_read_96B_cycles"] * 10,
            "port": "next_weight_bank_96B_generator_read",
        },
        "next_parent_pwp_read": {
            "bytes_per_event": 144,
            "events": tree_events_per_catalog[
                "parent_pwp_read_144B_cycles"] * 10,
            "cycles": tree_events_per_catalog[
                "parent_pwp_read_144B_cycles"] * 10,
            "port": "next_pwp_bank_single_144B_read_or_write",
        },
        "next_child_pwp_write": {
            "bytes_per_event": 144,
            "events": tree_events_per_catalog[
                "child_pwp_write_144B_cycles"] * 10,
            "cycles": tree_events_per_catalog[
                "child_pwp_write_144B_cycles"] * 10,
            "port": "next_pwp_bank_single_144B_read_or_write",
            "cycle_overlap": "final generator weight-update cycle only",
        },
        "next_tree_descriptor_read": {
            "bytes_per_event": 4,
            "events": tree_events_per_catalog[
                "tree_descriptor_read_events"] * 10,
            "port": "next_metadata_bank_independent_descriptor_read",
            "cycle_overlap": "parent read when present, otherwise first update",
        },
    }

    transition_count = 10 * (1728 - 1)
    output = {
        "schema": "m275r2_conv_tree_single_port_materializer_v1",
        "status": "PASS_CONSERVATIVE_SINGLE_PORT_SINGLE_ACCUMULATOR_ZERO_EXPOSED",
        "identity": {
            **identities,
            "m275r2_contract": {
                "path": str(args.contract.resolve().relative_to(root)),
                "sha256": sha256(args.contract),
            },
            "m275r2_analyzer": {
                "path": str(analyzer_path.relative_to(root)),
                "sha256": analyzer_start,
            },
        },
        "scope": {
            "samples": 10,
            "records": 40,
            "operators": 4,
            "phases": 17280,
            "in_sample_transitions": transition_count,
            "operator_boundaries": 30,
            "cold_sample_starts": 10,
            "final_drains": 10,
            "partition_vectors": work["partition_vectors"],
        },
        "m251r2_numeric_binding": {
            "exact_pwp_sum_range": [-2048, 2032],
            "signed12_range": [-2048, 2047],
            "signed12_safe": True,
            "m251_r1_cycles_remain_source": True,
        },
        "capacity": {
            "two_weight_banks_bytes": 24576,
            "two_pwp_banks_bytes": 36864,
            "two_metadata_banks_bytes": 192,
            "one_signed12_96lane_accumulator_bytes": 144,
            "minimum_complete_modeled_bytes": 61776,
            "is_lower_bound_not_macro_area": True,
            "pwp_capacity_eliminated": False,
        },
        "ports": contract["memory_and_port_contract"],
        "event_ledger": event_ledger,
        "tree_catalog": {
            "partitions": len(trees),
            "edges": sum(row["tree_edges"] for row in trees),
            "total_flips": sum(row["tree_flips"] for row in trees),
            "nonresident_parent_edges_frozen_order":
                sum(row["nonresident_parent_edges"] for row in trees),
            "flip_histogram": {
                str(key): value for key, value in sorted(flip_histogram.items())
            },
            "edge_distance_histogram": {
                str(key): value for key, value in
                sorted(edge_distance_histogram.items())
            },
            "minimum_preparation_cycles":
                min(row["preparation_cycles"] for row in trees),
            "maximum_preparation_cycles":
                max(row["preparation_cycles"] for row in trees),
            "first_preparation_cycles": first_prep,
            "phase_port_schedule": phase_schedule_catalog,
        },
        "bank_lifecycle_ledger": {
            "prepare_start_invalidations": 17280,
            "weight_valid_assertions_after_fill": 17280,
            "metadata_valid_assertions_after_fill": 17280,
            "pwp_valid_assertions_after_final_child_write": 17280,
            "cold_bank_activations": 10,
            "in_sample_role_switches": transition_count,
            "role_switches_inside_charged_two_cycle_tail": transition_count,
            "final_drains_without_role_switch": 10,
            "early_current_bank_reads": 0,
            "early_role_switches": 0,
            "parent_entry_use_before_write": 0,
            "same_bank_single_port_conflicts": 0,
            "generator_busy_at_role_switch": 0,
        },
        "overlap_replay": {
            "cold_start_cycles_per_sample": first_prep,
            "maximum_preparation_cycles":
                max(row["preparation_cycles"] for row in trees),
            "minimum_transition_slack_cycles": global_min_slack,
            "worst_transition": worst_transition,
            "exposed_transition_cycles": total["exposed"],
            "all_operator_boundaries_included": True,
            "sample_boundaries_are_cold": True,
            "final_drain_has_no_next_preparation": True,
        },
        "cycles": {
            "stored_fixed_pwp_complete_modeled_conv": total["stored"],
            "tree_single_port_complete_modeled_conv": total["tree"],
            "cycle_reduction_vs_stored_fixed_pwp": total["stored"] - total["tree"],
            "bit_sparse_complete_modeled_conv": int(wide["bit_sparse_cycles"]),
            "dense_complete_modeled_conv": int(wide["dense_cycles"]),
            "materializer_only_speedup_vs_stored_fixed_pwp":
                total["stored"] / float(total["tree"]),
            "complete_modeled_pwp_conv_speedup_vs_bit_sparse":
                int(wide["bit_sparse_cycles"]) / float(total["tree"]),
            "complete_modeled_pwp_conv_speedup_vs_dense":
                int(wide["dense_cycles"]) / float(total["tree"]),
            "tree_materializer_is_source_of_complete_conv_ratios": False,
        },
        "sample_rows": sample_rows,
        "m267_storage_and_dram": m267_result["storage_and_dram"],
        "record_expansion_audit": record_audit,
        "admission": {
            "exact_phase_replay": True,
            "explicit_current_child_reads": True,
            "explicit_next_parent_reads_and_child_writes": True,
            "single_pwp_port": True,
            "single_generator_accumulator": True,
            "explicit_weight_and_metadata_events": True,
            "explicit_bank_valid_and_role_switch_ledger": True,
            "zero_exposed_transition_cycles": True,
            "minimum_complete_modeled_capacity_bytes": 61776,
            "m251r2_corrected_range_bound": True,
            "rtl": False,
            "vcs": False,
            "sram_macro": False,
            "dc": False,
            "energy": False,
            "complete_conv_rtl": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    require(sha256(analyzer_path) == analyzer_start,
            "analyzer changed during execution")
    require(not args.output_dir.exists(), "refusing output overwrite")
    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / (
        "m275r2_conv_tree_single_port_materializer_r1.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(
        "M275R2_PASS stored={} tree={} reduction={} cold={} max_prep={} "
        "slack={} exposed={} nonresident={} capacity={}".format(
            total["stored"], total["tree"], total["stored"] - total["tree"],
            first_prep, max(row["preparation_cycles"] for row in trees),
            global_min_slack, total["exposed"],
            sum(row["nonresident_parent_edges"] for row in trees), 61776),
        flush=True,
    )


if __name__ == "__main__":
    main()
