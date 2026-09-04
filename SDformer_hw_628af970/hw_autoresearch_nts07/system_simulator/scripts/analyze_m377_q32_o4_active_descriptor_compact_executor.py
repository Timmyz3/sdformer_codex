#!/usr/bin/env python3
"""Replay exact active-only q32/O4 descriptors with finite resources."""

from __future__ import division

import argparse
from collections import Counter, defaultdict
import csv
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


def active_sample_schedule(m373, sample, phases, model, cfg, events):
    q = cfg["q_capacity"]
    output_tile = cfg["output_block_tile"]
    pattern_bytes = q * model["pattern_bytes"]
    time = 0
    component = Counter()
    maximum_slot0 = 0
    maximum_active_rows = 0
    active_rows_total = 0
    empty_partitions = 0
    for phase_index, phase in enumerate(phases):
        op = phase_index // model["partitions_per_operator"]
        partition = phase_index % model["partitions_per_operator"]
        bank = phase_index & 1
        descriptor_base = (cfg["descriptor0_base_address"] if bank == 0
                           else cfg["descriptor1_base_address"])
        pattern_start = time
        pattern_cycles = int(math.ceil(
            pattern_bytes / float(cfg["dram_bytes_per_cycle"])))
        time += pattern_cycles
        m373.add_event(events, sample, op, partition, "PATTERN_DMA", "DMA0",
                       pattern_start, time, 0,
                       cfg["pattern_base_address"], pattern_bytes, bank)
        component["pattern_dma"] += pattern_cycles

        matcher_start = time
        matcher_cycles = phase["partition_vectors"] + phase["matcher_rows"] + 2
        time += matcher_cycles
        active_rows = phase["partition_vectors"] - phase["zero_rows"]
        require(active_rows >= 0, "negative active population")
        require(active_rows == phase["active_rows"],
                "active descriptor population drift")
        descriptor_bytes = active_rows * model["descriptor_bytes_per_row"]
        require(descriptor_bytes <=
                cfg["descriptor_rows_per_bank"] *
                model["descriptor_bytes_per_row"],
                "compact descriptor bank overflow")
        m373.add_event(
            events, sample, op, partition,
            "SERIAL16_MATCH_AND_ACTIVE_DESCRIPTOR_COMPACT_WRITE",
            "MATCHER_DESC_W", matcher_start, time, bank, descriptor_base,
            descriptor_bytes, bank)
        component["matcher_active_descriptor_write"] += matcher_cycles
        maximum_active_rows = max(maximum_active_rows, active_rows)
        active_rows_total += active_rows

        if active_rows == 0:
            empty_partitions += 1
            tail_start = time
            time += model["compute_tail_cycles_per_partition"]
            m373.add_event(events, sample, op, partition,
                           "EMPTY_ACTIVE_STREAM_TAIL", "COMMIT_PIPE",
                           tail_start, time, -1, 0, 0, bank)
            component["tail"] += model[
                "compute_tail_cycles_per_partition"]
            continue

        tile_payload = (
            phase["used_pwp_patterns"] *
            model["pwp_vector_bytes_per_output_block"] * output_tile +
            model["partition_bits"] * model["weight_vector_bytes"] *
            output_tile)
        require(tile_payload % cfg["dram_bytes_per_cycle"] == 0,
                "unaligned q32/O4 tile payload")
        maximum_slot0 = max(maximum_slot0, pattern_bytes + tile_payload)
        require(pattern_bytes + tile_payload <=
                cfg["tile_slot_bytes_each"] and
                tile_payload <= cfg["tile_slot_bytes_each"],
                "tile slot overflow")
        tile_dma = tile_payload // cfg["dram_bytes_per_cycle"]
        dma0_start = time
        time += tile_dma
        m373.add_event(events, sample, op, partition, "TILE0_DMA", "DMA0",
                       dma0_start, time, 0,
                       cfg["slot0_base_address"] + pattern_bytes,
                       tile_payload, bank)
        component["initial_tile_dma"] += tile_dma

        exact_work = (phase["correction_ops_per_block"] * output_tile +
                      phase["pwp_ops_per_block"] * output_tile * 2)
        require(exact_work >= active_rows * output_tile,
                "active bundle minimum service failure")
        descriptor_service = 1 + exact_work
        tile0_start = time
        tile0_end = tile0_start + descriptor_service
        dma1_start = tile0_start
        dma1_end = dma1_start + tile_dma
        m373.add_event(
            events, sample, op, partition,
            "TILE0_ACTIVE_DESCRIPTOR_REPLAY_COMPUTE", "DESC_R_COMPUTE",
            tile0_start, tile0_end, bank, descriptor_base,
            descriptor_bytes, bank)
        m373.add_event(events, sample, op, partition, "TILE1_DMA", "DMA0",
                       dma1_start, dma1_end, 1, cfg["slot1_base_address"],
                       tile_payload, bank)
        time = max(tile0_end, dma1_end)
        tile1_start = time
        time += descriptor_service
        m373.add_event(
            events, sample, op, partition,
            "TILE1_ACTIVE_DESCRIPTOR_REPLAY_COMPUTE", "DESC_R_COMPUTE",
            tile1_start, time, bank, descriptor_base, descriptor_bytes, bank)
        tail_start = time
        time += model["compute_tail_cycles_per_partition"]
        m373.add_event(events, sample, op, partition, "TAIL", "COMMIT_PIPE",
                       tail_start, time, -1, 0, 0, bank)
        component["active_descriptor_compute"] += descriptor_service * 2
        component["later_tile_dma_not_additive"] += tile_dma
        component["tail"] += model["compute_tail_cycles_per_partition"]

    commit = (model["operators"] * model["rows_per_operator"] *
              model["output_blocks"] //
              model["commit_output_blocks_per_cycle"])
    commit_start = time
    time += commit
    m373.add_event(events, sample, -1, -1, "COMMON_COMMIT",
                   "COMMIT_PIPE", commit_start, time, -1, 0, 0, sample)
    component["common_commit"] += commit
    return {
        "cycles": time,
        "components": dict(component),
        "maximum_slot0_bytes": maximum_slot0,
        "maximum_active_descriptor_rows": maximum_active_rows,
        "active_descriptor_rows": active_rows_total,
        "empty_partitions": empty_partitions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M377 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m377_q32_o4_active_descriptor_compact_executor_contract_v1",
            "M377 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M377_EXECUTION",
            "M377 contract not frozen")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, identity in contract["inputs"].items():
        path = root / identity["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"], "SHA drift for " + name)
        paths[name] = path
        identities[name] = {"path": identity["path"], "sha256": observed}

    parent_contract = strict_json(paths["m373_contract"])
    parent_result = strict_json(paths["m373_result"])
    require(parent_result["status"] ==
            "PASS_M373_TIMESTAMPED_FINITE_EVENT_EXECUTION" and
            parent_result["decision"] ==
            "NO_GO_Q32_EXACT_PWP_PERFORMANCE_CONTRIBUTION",
            "M373 parent decision drift")
    cfg = parent_contract["configuration"]
    m373 = load_module(paths["m373_analyzer"], "m377_m373")
    m358_path = root / parent_contract["inputs"]["m358_contract"]["path"]
    require(sha256(m358_path) ==
            parent_contract["inputs"]["m358_contract"]["sha256"],
            "M358 contract drift through M373")
    m358_contract = strict_json(m358_path)
    model = m358_contract["cycle_model"]
    m358_root = m358_path.resolve().parents[1]
    m358_paths = {}
    for name, identity in m358_contract["inputs"].items():
        path = m358_root / identity["path"]
        require(path.is_file() and sha256(path) == identity["sha256"],
                "transitive M358 input drift: " + name)
        m358_paths[name] = path
    m339 = load_module(m358_paths["m339_analyzer"], "m377_m339")
    m43 = load_module(m358_paths["m43_support_unpacker"], "m377_m43")
    catalog = strict_json(m358_paths["m338_catalog"])
    trace = strict_json(m358_paths["m248_runtime_trace"])
    trace_dir = m358_paths["m248_runtime_trace"].parent
    operators = tuple(trace["cohort"]["operators"])
    op_index = {name: index for index, name in enumerate(operators)}
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
                partition_base = tile * (m43.TILE_BITS //
                                         model["partition_bits"])
                for subtile in range(m43.TILE_BITS //
                                     model["partition_bits"]):
                    value = ((value256 >>
                              (subtile * model["partition_bits"])) & 0xffff)
                    histograms[(record["sample_id"],
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
        print("[M377 HIST] {}/40".format(record_index + 1), flush=True)

    phases = defaultdict(list)
    total_rows = 0
    total_zero = 0
    for sample in range(model["samples"]):
        for op in range(model["operators"]):
            for partition in range(model["partitions_per_operator"]):
                counter = histograms[(sample, op, partition)]
                nested = catalog["operators"][op]["partitions"][partition][
                    "nested_patterns"]
                phase = m339.phase_metrics_all_q(
                    counter, [int(value, 16) for value in nested])[32]
                phase["zero_rows"] = counter[0]
                phase["active_rows"] = sum(
                    count for value, count in counter.items() if value != 0)
                require(phase["zero_rows"] + phase["active_rows"] ==
                        phase["partition_vectors"],
                        "active/zero partition failure")
                total_rows += phase["partition_vectors"]
                total_zero += phase["zero_rows"]
                phases[sample].append(phase)
        print("[M377 METRIC] sample={}/10".format(sample + 1), flush=True)

    candidate_events = []
    baseline_events = []
    sample_rows = []
    candidate_total = 0
    baseline_total = 0
    for sample in range(model["samples"]):
        candidate = active_sample_schedule(
            m373, sample, phases[sample], model, cfg, candidate_events)
        baseline = m373.baseline_sample_schedule(
            sample, phases[sample], model, baseline_events)
        candidate_total += candidate["cycles"]
        baseline_total += baseline
        sample_rows.append(dict({"sample": sample}, **candidate))
    require(baseline_total ==
            parent_result["cycles"]["bit_sparse_reproduced_cycles"],
            "M373 bit-sparse baseline mismatch")
    candidate_dma = m373.verify_resource_events(candidate_events, True)
    baseline_dma = m373.verify_resource_events(baseline_events, False)
    active_rows = total_rows - total_zero
    require(sum(row["active_descriptor_rows"] for row in sample_rows) ==
            active_rows, "compact descriptor conservation failure")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = ["sample", "operator_index", "partition", "event", "resource",
              "start_cycle", "end_cycle", "duration_cycles", "slot",
              "address", "bytes", "context"]
    for name, rows in (("candidate_events.csv", candidate_events),
                       ("baseline_events.csv", baseline_events)):
        with (args.output_dir / name).open("w", newline="",
                                           encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    speedup = baseline_total / float(candidate_total)
    decision = ("GO_VCS_ACTIVE_DESCRIPTOR_SCHEDULER_RTL" if speedup >= 1.05
                else "NO_GO_ACTIVE_DESCRIPTOR_COMPACTION")
    result = {
        "schema": "m377_q32_o4_active_descriptor_compact_executor_v1",
        "status": "PASS_M377_EXACT_ACTIVE_DESCRIPTOR_FINITE_EXECUTION",
        "identity": identities,
        "configuration": cfg,
        "population": {
            "source_rows": total_rows,
            "zero_rows_elided_exactly": total_zero,
            "active_descriptor_rows": active_rows,
            "active_fraction": active_rows / float(total_rows),
            "empty_partitions": sum(row["empty_partitions"]
                                    for row in sample_rows),
            "maximum_active_rows_in_one_partition": max(
                row["maximum_active_descriptor_rows"]
                for row in sample_rows),
        },
        "cycles": {
            "bit_sparse_reproduced_cycles": baseline_total,
            "m373_full_row_candidate_cycles":
                parent_result["cycles"]["m373_candidate_cycles"],
            "m377_active_compact_candidate_cycles": candidate_total,
            "m377_speedup_vs_bit_sparse": speedup,
            "difference_vs_m373_cycles":
                parent_result["cycles"]["m373_candidate_cycles"] -
                candidate_total,
            "tile_startup_cycles_added": 2 * (
                model["samples"] * model["operators"] *
                model["partitions_per_operator"] -
                sum(row["empty_partitions"] for row in sample_rows)),
        },
        "finite_resource_audit": {
            "candidate_events": len(candidate_events),
            "baseline_events": len(baseline_events),
            "candidate_dma_events": candidate_dma,
            "baseline_dma_events": baseline_dma,
            "single_dma_overlap_violations": 0,
            "all_active_descriptors_conserved": True,
            "zero_descriptors_written_or_replayed": 0,
            "descriptor_row_id_preserved": True,
            "descriptor_bundle_fifo_depth":
                cfg["descriptor_bundle_fifo_depth"],
            "sample_rows": sample_rows,
        },
        "decision": decision,
        "admission": {
            "timestamped_finite_event_module_cycles": True,
            "exact_active_descriptor_compaction": True,
            "frozen_bit_sparse_baseline_reproduced": True,
            "rtl_cycle_match": False,
            "synopsys_area": False,
            "energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "output_files": {
            "candidate_events": "candidate_events.csv",
            "baseline_events": "baseline_events.csv",
        },
    }
    output = args.output_dir / (
        "m377_q32_o4_active_descriptor_compact_executor_r1.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M377_PASS baseline={} candidate={} speedup={:.6f}x decision={}".
          format(baseline_total, candidate_total, speedup, decision),
          flush=True)


if __name__ == "__main__":
    main()
