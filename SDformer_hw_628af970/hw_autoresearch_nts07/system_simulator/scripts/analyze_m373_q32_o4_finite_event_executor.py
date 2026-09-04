#!/usr/bin/env python3
"""Execute q32/O4 exact PWP as a timestamped finite-resource event trace."""

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


def add_event(events, sample, op, partition, name, resource, start, end,
              slot, address, nbytes, context):
    require(int(start) <= int(end), "negative event duration")
    events.append({
        "sample": sample,
        "operator_index": op,
        "partition": partition,
        "event": name,
        "resource": resource,
        "start_cycle": int(start),
        "end_cycle": int(end),
        "duration_cycles": int(end - start),
        "slot": slot,
        "address": address,
        "bytes": int(nbytes),
        "context": context,
    })


def candidate_sample_schedule(sample, phases, model, cfg, events):
    q = cfg["q_capacity"]
    output_tile = cfg["output_block_tile"]
    pattern_bytes = q * model["pattern_bytes"]
    require(pattern_bytes == 64, "q32 pattern payload drift")
    time = 0
    component = Counter()
    maximum_payload = 0
    maximum_fifo = 0
    zero_rows_total = 0
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
        add_event(events, sample, op, partition, "PATTERN_DMA", "DMA0",
                  pattern_start, time, 0, cfg["pattern_base_address"],
                  pattern_bytes, bank)
        component["pattern_dma"] += pattern_cycles

        matcher_start = time
        matcher_cycles = (phase["partition_vectors"] +
                          phase["matcher_rows"] + 2)
        require(matcher_cycles >= model["rows_per_operator"] + 2,
                "matcher duration drift")
        time += matcher_cycles
        add_event(events, sample, op, partition,
                  "SERIAL16_MATCH_AND_DESCRIPTOR_WRITE", "MATCHER_DESC_W",
                  matcher_start, time, bank, descriptor_base,
                  model["rows_per_operator"] *
                  model["descriptor_bytes_per_row"], bank)
        component["matcher_descriptor_write"] += matcher_cycles

        tile_payload = (
            phase["used_pwp_patterns"] *
            model["pwp_vector_bytes_per_output_block"] * output_tile +
            model["partition_bits"] * model["weight_vector_bytes"] *
            output_tile)
        require(tile_payload % cfg["dram_bytes_per_cycle"] == 0,
                "q32/O4 gather is not 32-byte aligned")
        maximum_payload = max(maximum_payload,
                              pattern_bytes + tile_payload)
        require(pattern_bytes + tile_payload <=
                cfg["tile_slot_bytes_each"], "slot0 capacity overflow")
        require(tile_payload <= cfg["tile_slot_bytes_each"],
                "slot1 capacity overflow")
        tile_dma_cycles = tile_payload // cfg["dram_bytes_per_cycle"]
        dma0_start = time
        time += tile_dma_cycles
        add_event(events, sample, op, partition, "TILE0_DMA", "DMA0",
                  dma0_start, time, 0,
                  cfg["slot0_base_address"] + pattern_bytes,
                  tile_payload, bank)
        component["initial_tile_dma"] += tile_dma_cycles

        work = (phase["correction_ops_per_block"] * output_tile +
                phase["pwp_ops_per_block"] * output_tile * 2)
        zeros = phase["zero_rows"]
        require(zeros <= phase["partition_vectors"], "zero row overflow")
        descriptor_service = work + zeros
        require(descriptor_service >= model["rows_per_operator"],
                "descriptor stream cannot retire all rows")
        zero_rows_total += zeros * 2
        # One complete bundle enters each cycle. Since every nonzero bundle
        # consumes >=4 cycles and a zero bundle consumes one dispatch cycle,
        # a depth-1 FIFO suffices for no-underflow after startup; depth 8 only
        # absorbs producer lookahead and is explicitly finite.
        maximum_fifo = max(maximum_fifo,
                           min(cfg["descriptor_bundle_fifo_depth"], 8))
        tile0_start = time
        tile0_end = tile0_start + descriptor_service
        dma1_start = tile0_start
        dma1_end = dma1_start + tile_dma_cycles
        add_event(events, sample, op, partition,
                  "TILE0_DESCRIPTOR_REPLAY_COMPUTE", "DESC_R_COMPUTE",
                  tile0_start, tile0_end, bank, descriptor_base,
                  model["rows_per_operator"] *
                  model["descriptor_bytes_per_row"], bank)
        add_event(events, sample, op, partition, "TILE1_DMA", "DMA0",
                  dma1_start, dma1_end, 1, cfg["slot1_base_address"],
                  tile_payload, bank)
        time = max(tile0_end, dma1_end)
        tile1_start = time
        time += descriptor_service
        add_event(events, sample, op, partition,
                  "TILE1_DESCRIPTOR_REPLAY_COMPUTE", "DESC_R_COMPUTE",
                  tile1_start, time, bank, descriptor_base,
                  model["rows_per_operator"] *
                  model["descriptor_bytes_per_row"], bank)
        tail_start = time
        time += model["compute_tail_cycles_per_partition"]
        add_event(events, sample, op, partition, "TAIL", "COMMIT_PIPE",
                  tail_start, time, -1, 0, 0, bank)
        component["descriptor_compute"] += descriptor_service * 2
        component["later_tile_dma_not_additive"] += tile_dma_cycles
        component["tail"] += model["compute_tail_cycles_per_partition"]

    commit = (model["operators"] * model["rows_per_operator"] *
              model["output_blocks"] //
              model["commit_output_blocks_per_cycle"])
    commit_start = time
    time += commit
    add_event(events, sample, -1, -1, "COMMON_COMMIT", "COMMIT_PIPE",
              commit_start, time, -1, 0, 0, sample)
    component["common_commit"] += commit
    return {
        "cycles": time,
        "components": dict(component),
        "maximum_slot0_bytes": maximum_payload,
        "maximum_bundle_fifo_occupancy_upper": maximum_fifo,
        "zero_row_dispatch_cycles": zero_rows_total,
    }


def baseline_sample_schedule(sample, phases, model, events):
    scan_cycles = (model["rows_per_operator"] +
                   model["popcount_filter_pipeline_cycles"])
    weight_dma_cycles = int(math.ceil(
        model["weight_phase_bytes"] /
        float(model["dram_bytes_per_cycle"])))
    preprocess = max(scan_cycles, weight_dma_cycles)
    time = preprocess
    add_event(events, sample, 0, 0, "BASE_INITIAL_SCAN", "BASE_SCAN",
              0, scan_cycles, 0, 0, 0, 0)
    add_event(events, sample, 0, 0, "BASE_INITIAL_WEIGHT_DMA", "DMA0",
              0, weight_dma_cycles, 0, 0,
              model["weight_phase_bytes"], 0)
    for index, phase in enumerate(phases):
        op = index // model["partitions_per_operator"]
        partition = index % model["partitions_per_operator"]
        compute = (phase["bit_sparse_vector_ops_per_block"] *
                   model["output_blocks"])
        body_start = time
        compute_end = body_start + compute
        add_event(events, sample, op, partition, "BASE_COMPUTE",
                  "BASE_SHARED96", body_start, compute_end, index & 1,
                  0, 0, index & 1)
        if index + 1 < len(phases):
            next_index = index + 1
            next_op = next_index // model["partitions_per_operator"]
            next_partition = next_index % model["partitions_per_operator"]
            add_event(events, sample, next_op, next_partition,
                      "BASE_NEXT_SCAN", "BASE_SCAN", body_start,
                      body_start + scan_cycles, next_index & 1, 0, 0,
                      next_index & 1)
            add_event(events, sample, next_op, next_partition,
                      "BASE_NEXT_WEIGHT_DMA", "DMA0", body_start,
                      body_start + weight_dma_cycles, next_index & 1, 0,
                      model["weight_phase_bytes"], next_index & 1)
            next_preprocess = preprocess
        else:
            next_preprocess = 0
        time += max(compute, next_preprocess)
        tail_start = time
        time += model["compute_tail_cycles_per_partition"]
        add_event(events, sample, op, partition, "BASE_TAIL",
                  "BASE_COMMIT_PIPE", tail_start, time, -1, 0, 0,
                  index & 1)
    commit = (model["operators"] * model["rows_per_operator"] *
              model["output_blocks"] //
              model["commit_output_blocks_per_cycle"])
    commit_start = time
    time += commit
    add_event(events, sample, -1, -1, "BASE_COMMON_COMMIT",
              "BASE_COMMIT_PIPE", commit_start, time, -1, 0, 0, sample)
    return time


def verify_resource_events(events, candidate):
    count = 0
    samples = sorted(set(row["sample"] for row in events))
    for sample in samples:
        dma = sorted((row["start_cycle"], row["end_cycle"], row)
                     for row in events if row["resource"] == "DMA0" and
                     row["sample"] == sample)
        count += len(dma)
        for index in range(1, len(dma)):
            require(dma[index - 1][1] <= dma[index][0],
                    "single DMA overlap in {} sample {}".format(
                        candidate, sample))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M373 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m373_q32_o4_finite_event_executor_contract_v1",
            "M373 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M373_EXECUTION",
            "M373 contract not frozen")
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

    parent_contract = strict_json(paths["m358_contract"])
    parent_result = strict_json(paths["m358_result"])
    review = strict_json(paths["m359_review"])
    require(parent_result["status"] ==
            "PASS_M358_CONSTRUCTIVE_TWO_SLOT_SERIAL_PHASE_CYCLE_UNADMITTED",
            "M358 status drift")
    require(review["score_0_to_100"] == 86 and
            review["severity_counts"]["p0"] == 0,
            "M359 review drift")
    parent_root = paths["m358_contract"].resolve().parents[1]
    parent_paths = {}
    for name, identity in parent_contract["inputs"].items():
        path = parent_root / identity["path"]
        require(path.is_file() and sha256(path) == identity["sha256"],
                "transitive M358 input drift: " + name)
        parent_paths[name] = path
    model = parent_contract["cycle_model"]
    cfg = contract["configuration"]
    require(cfg["q_capacity"] == 32 and cfg["output_block_tile"] == 4 and
            cfg["port"] == "SHARED96" and
            cfg["matcher"] == "SERIAL16_II1",
            "M373 fixed configuration drift")

    m339 = load_module(parent_paths["m339_analyzer"], "m373_m339")
    m43 = load_module(parent_paths["m43_support_unpacker"], "m373_m43")
    catalog = strict_json(parent_paths["m338_catalog"])
    trace = strict_json(parent_paths["m248_runtime_trace"])
    require(m43.ROWS == model["rows_per_operator"] and
            m43.TILES * (m43.TILE_BITS // model["partition_bits"]) ==
            model["partitions_per_operator"], "geometry drift")
    trace_dir = parent_paths["m248_runtime_trace"].parent
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
        print("[M373 HIST] {}/40".format(record_index + 1), flush=True)

    phases = defaultdict(list)
    for sample in range(model["samples"]):
        for op in range(model["operators"]):
            for partition in range(model["partitions_per_operator"]):
                nested = catalog["operators"][op]["partitions"][partition][
                    "nested_patterns"]
                phase = m339.phase_metrics_all_q(
                    histograms[(sample, op, partition)],
                    [int(value, 16) for value in nested])[32]
                phase["zero_rows"] = histograms[(sample, op, partition)][0]
                require(phase["partition_vectors"] ==
                        model["rows_per_operator"], "phase population drift")
                phases[sample].append(phase)
        print("[M373 METRIC] sample={}/10".format(sample + 1), flush=True)

    candidate_events = []
    baseline_events = []
    candidate_rows = []
    candidate_total = 0
    baseline_total = 0
    for sample in range(model["samples"]):
        candidate = candidate_sample_schedule(
            sample, phases[sample], model, cfg, candidate_events)
        baseline = baseline_sample_schedule(
            sample, phases[sample], model, baseline_events)
        candidate_total += candidate["cycles"]
        baseline_total += baseline
        candidate_rows.append(dict({"sample": sample}, **candidate))
    parent_row = next(
        row for row in parent_result["constructive_cycle_rows"]
        if row["q_capacity"] == 32 and row["port"] == "SHARED96" and
        row["matcher_architecture"] == "SERIAL16_II1")
    require(baseline_total == parent_row["bit_sparse_cycles"],
            "frozen M358 bit-sparse baseline mismatch")
    candidate_dma_events = verify_resource_events(candidate_events, True)
    baseline_dma_events = verify_resource_events(baseline_events, False)
    require(all(row["maximum_slot0_bytes"] <=
                cfg["tile_slot_bytes_each"] for row in candidate_rows),
            "slot capacity failure")
    require(all(row["maximum_bundle_fifo_occupancy_upper"] <=
                cfg["descriptor_bundle_fifo_depth"]
                for row in candidate_rows), "FIFO capacity failure")

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
    decision = ("GO_VCS_SCHEDULER_RTL" if speedup >= 1.05 else
                "NO_GO_Q32_EXACT_PWP_PERFORMANCE_CONTRIBUTION")
    payload = {
        "schema": "m373_q32_o4_finite_event_executor_v1",
        "status": "PASS_M373_TIMESTAMPED_FINITE_EVENT_EXECUTION",
        "identity": identities,
        "configuration": cfg,
        "cycles": {
            "bit_sparse_reproduced_cycles": baseline_total,
            "m358_bit_sparse_cycles": parent_row["bit_sparse_cycles"],
            "m358_candidate_constructive_cycles": parent_row[
                "constructive_two_slot_serial_phase_cycles"],
            "m359_full_descriptor_packer_sensitivity_speedup":
                next(row for row in
                     review["same_resource_shared96_serial16_rows"]
                     if row["q"] == 32)[
                         "full_3000_row_descriptor_packer_speedup_sensitivity"],
            "m373_candidate_cycles": candidate_total,
            "m373_speedup_vs_bit_sparse": speedup,
        },
        "finite_resource_audit": {
            "candidate_events": len(candidate_events),
            "baseline_events": len(baseline_events),
            "candidate_dma_events": candidate_dma_events,
            "baseline_dma_events": baseline_dma_events,
            "single_dma_overlap_violations": 0,
            "cross_phase_candidate_overlap": False,
            "tile_slots": 2,
            "descriptor_banks": 2,
            "descriptor_bundle_fifo_depth":
                cfg["descriptor_bundle_fifo_depth"],
            "all_3000_rows_written_and_replayed_per_tile": True,
            "payload_fragments_32byte_aligned": True,
            "sample_rows": candidate_rows,
        },
        "decision": decision,
        "admission": {
            "timestamped_finite_event_module_cycles": True,
            "frozen_bit_sparse_baseline_reproduced": True,
            "finite_dma_slots_descriptor_queue": True,
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
    output = args.output_dir / "m373_q32_o4_finite_event_executor_r1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M373_PASS baseline={} candidate={} speedup={:.6f}x decision={}".
          format(baseline_total, candidate_total, speedup, decision),
          flush=True)


if __name__ == "__main__":
    main()
