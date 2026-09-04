#!/usr/bin/env python3
"""Stress M377 with exact PWP burst runs and streaming-SRAM latency."""

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


def count_runs(indices):
    ordered = sorted(indices)
    if not ordered:
        return 0
    runs = 1
    for previous, current in zip(ordered, ordered[1:]):
        require(current > previous, "used center IDs are not unique")
        runs += int(current != previous + 1)
    return runs


def popcount16(value):
    return bin(int(value) & 0xffff).count("1")


def phase_metrics(counter, centers, m339):
    frozen = m339.phase_metrics_all_q(counter, centers)[32]
    q32_centers = centers[:32]
    used = set()
    pwp_rows = 0
    fallback_rows = 0
    pop1_fallback_rows = 0
    exact_reconstruction_rows = 0
    for original, count in counter.items():
        population = popcount16(original)
        best_distance = 17
        best_index = 0
        for index, center in enumerate(q32_centers):
            distance = popcount16(int(original) ^ int(center))
            if distance < best_distance:
                best_distance = distance
                best_index = index
        use_pwp = 1 + best_distance < population
        if use_pwp:
            require(original != 0, "zero row illegally selected PWP")
            used.add(best_index)
            pwp_rows += count
            plus = int(original) & ((~int(q32_centers[best_index])) & 0xffff)
            minus = int(q32_centers[best_index]) & ((~int(original)) & 0xffff)
            require((((int(q32_centers[best_index]) | plus) &
                      ((~minus) & 0xffff)) == int(original)),
                    "PWP exact reconstruction failure")
            exact_reconstruction_rows += count
        elif original != 0:
            fallback_rows += count
            if population == 1:
                pop1_fallback_rows += count
    require(len(used) == frozen["used_pwp_patterns"],
            "used-center population drift")
    require(pwp_rows == frozen["assignment_rows"],
            "PWP row population drift")
    require(pwp_rows + fallback_rows ==
            frozen["partition_vectors"] - counter[0],
            "active descriptor conservation failure")
    return dict(frozen, **{
        "zero_rows": counter[0],
        "active_rows": frozen["partition_vectors"] - counter[0],
        "used_center_ids": sorted(used),
        "used_center_bitmap": sum(1 << index for index in used),
        "used_center_runs": count_runs(used),
        "pwp_rows": pwp_rows,
        "fallback_rows": fallback_rows,
        "pop1_fallback_rows": pop1_fallback_rows,
        "exact_reconstruction_rows": exact_reconstruction_rows,
    })


def candidate_sample(phases, model, cfg, command_setup, sram_latency,
                     blocking_cycles_per_replayed_descriptor=0.0):
    require(command_setup >= 0, "negative command setup")
    require(sram_latency >= 1, "streaming SRAM latency must be positive")
    require(blocking_cycles_per_replayed_descriptor >= 0.0,
            "negative blocking penalty")
    q = cfg["q_capacity"]
    output_tile = cfg["output_block_tile"]
    pattern_data = int(math.ceil(
        q * model["pattern_bytes"] /
        float(cfg["dram_bytes_per_cycle"])))
    time = 0.0
    components = Counter()
    command_counts = Counter()
    exposed_tile1_setup = 0.0
    for phase in phases:
        # The 32 q16 center masks form one contiguous pattern burst.
        time += pattern_data + command_setup
        components["pattern_data"] += pattern_data
        components["pattern_command_setup"] += command_setup
        command_counts["pattern"] += 1

        matcher = phase["partition_vectors"] + phase["matcher_rows"] + 2
        time += matcher
        components["matcher"] += matcher

        # Seal active_count and the 32-bit used-center bitmap before replay.
        time += 1
        components["active_count_and_bitmap_seal"] += 1
        if phase["active_rows"] == 0:
            time += model["compute_tail_cycles_per_partition"]
            components["tail"] += model["compute_tail_cycles_per_partition"]
            continue

        weight_bytes = (model["partition_bits"] *
                        model["weight_vector_bytes"] * output_tile)
        pwp_bytes = (phase["used_pwp_patterns"] *
                     model["pwp_vector_bytes_per_output_block"] *
                     output_tile)
        data_cycles = int(math.ceil(
            (weight_bytes + pwp_bytes) /
            float(cfg["dram_bytes_per_cycle"])))
        # One contiguous weight burst plus one scatter/gather burst for every
        # maximal consecutive run in the direct-address center-ID cache.
        tile_commands = 1 + phase["used_center_runs"]
        tile_dma = data_cycles + tile_commands * command_setup
        time += tile_dma
        components["tile0_data"] += data_cycles
        components["tile0_command_setup"] += (
            tile_commands * command_setup)
        command_counts["weight"] += 1
        command_counts["pwp_runs"] += phase["used_center_runs"]

        exact_work = (phase["correction_ops_per_block"] * output_tile +
                      phase["pwp_ops_per_block"] * output_tile * 2)
        require(exact_work >= phase["active_rows"] * output_tile,
                "active-bundle minimum service failure")
        # Fixed response latency is paid once at replay startup. A separate
        # sensitivity axis captures any illegal non-overlapped per-descriptor
        # penalty that violates the intended II=1 SRAM contract.
        replay = (exact_work + sram_latency +
                  phase["active_rows"] *
                  blocking_cycles_per_replayed_descriptor)
        tile0_end = time + replay
        tile1_dma_end = time + tile_dma
        if tile1_dma_end > tile0_end:
            exposed_tile1_setup += tile1_dma_end - tile0_end
        time = max(tile0_end, tile1_dma_end)
        time += replay
        components["active_compute"] += 2 * exact_work
        components["descriptor_sram_startup"] += 2 * sram_latency
        components["blocking_descriptor_penalty"] += (
            2 * phase["active_rows"] *
            blocking_cycles_per_replayed_descriptor)
        components["tile1_data_not_additive"] += data_cycles
        components["tile1_command_setup_not_additive"] += (
            tile_commands * command_setup)
        time += model["compute_tail_cycles_per_partition"]
        components["tail"] += model["compute_tail_cycles_per_partition"]

    common_commit = (model["operators"] * model["rows_per_operator"] *
                     model["output_blocks"] //
                     model["commit_output_blocks_per_cycle"])
    time += common_commit
    components["common_commit"] += common_commit
    return {
        "cycles": time,
        "components": dict(components),
        "command_counts": dict(command_counts),
        "exposed_tile1_dma_cycles": exposed_tile1_setup,
    }


def baseline_sample(phases, model, command_setup):
    scan = (model["rows_per_operator"] +
            model["popcount_filter_pipeline_cycles"])
    weight_data = int(math.ceil(
        model["weight_phase_bytes"] /
        float(model["dram_bytes_per_cycle"])))
    weight_dma = weight_data + command_setup
    preprocess = max(scan, weight_dma)
    time = preprocess
    for index, phase in enumerate(phases):
        compute = (phase["bit_sparse_vector_ops_per_block"] *
                   model["output_blocks"])
        next_preprocess = preprocess if index + 1 < len(phases) else 0
        time += max(compute, next_preprocess)
        time += model["compute_tail_cycles_per_partition"]
    time += (model["operators"] * model["rows_per_operator"] *
             model["output_blocks"] //
             model["commit_output_blocks_per_cycle"])
    return time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M381 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m381_q32_o4_burst_streaming_sensitivity_contract_v1",
            "M381 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M381_EXECUTION",
            "M381 contract not frozen")
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

    m377 = strict_json(paths["m377_result"])
    m373_contract = strict_json(paths["m373_contract"])
    m358_path = root / m373_contract["inputs"]["m358_contract"]["path"]
    require(sha256(m358_path) ==
            m373_contract["inputs"]["m358_contract"]["sha256"],
            "M358 contract drift through M373")
    m358 = strict_json(m358_path)
    model = m358["cycle_model"]
    cfg = m373_contract["configuration"]
    m358_root = m358_path.resolve().parents[1]
    transitive = {}
    for name, identity in m358["inputs"].items():
        path = m358_root / identity["path"]
        require(path.is_file() and sha256(path) == identity["sha256"],
                "M358 transitive input drift: " + name)
        transitive[name] = path
    m339 = load_module(transitive["m339_analyzer"], "m381_m339")
    m43 = load_module(transitive["m43_support_unpacker"], "m381_m43")
    catalog = strict_json(transitive["m338_catalog"])
    trace = strict_json(transitive["m248_runtime_trace"])
    trace_dir = transitive["m248_runtime_trace"].parent
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
                partition_base = tile * (
                    m43.TILE_BITS // model["partition_bits"])
                for subtile in range(
                        m43.TILE_BITS // model["partition_bits"]):
                    value = ((value256 >>
                              (subtile * model["partition_bits"])) & 0xffff)
                    histograms[(record["sample_id"],
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
        print("[M381 HIST] {}/{}".format(record_index + 1,
                                         len(trace["records"])), flush=True)

    phases = defaultdict(list)
    phase_summary = Counter()
    used_counts = []
    run_counts = []
    for sample in range(model["samples"]):
        for op in range(model["operators"]):
            for partition in range(model["partitions_per_operator"]):
                counter = histograms[(sample, op, partition)]
                require(sum(counter.values()) == model["rows_per_operator"],
                        "phase row extent drift")
                centers = [
                    int(value, 16) for value in
                    catalog["operators"][op]["partitions"][partition]
                    ["nested_patterns"]
                ]
                require(len(centers) >= 128,
                        "nested catalog does not cover frozen q128 prefixes")
                phase = phase_metrics(counter, centers, m339)
                phases[sample].append(phase)
                used_counts.append(phase["used_pwp_patterns"])
                run_counts.append(phase["used_center_runs"])
                phase_summary["source_rows"] += phase["partition_vectors"]
                phase_summary["active_rows"] += phase["active_rows"]
                phase_summary["zero_rows"] += phase["zero_rows"]
                phase_summary["pwp_rows"] += phase["pwp_rows"]
                phase_summary["fallback_rows"] += phase["fallback_rows"]
                phase_summary["pop1_fallback_rows"] += phase[
                    "pop1_fallback_rows"]
                phase_summary["used_centers"] += phase[
                    "used_pwp_patterns"]
                phase_summary["used_center_runs"] += phase[
                    "used_center_runs"]
        print("[M381 METRIC] sample={}/{}".format(
            sample + 1, model["samples"]), flush=True)

    require(phase_summary["source_rows"] ==
            m377["population"]["source_rows"], "source population drift")
    require(phase_summary["active_rows"] ==
            m377["population"]["active_descriptor_rows"],
            "active population drift")
    require(phase_summary["zero_rows"] ==
            m377["population"]["zero_rows_elided_exactly"],
            "zero population drift")

    sweeps = []
    baseline_reference = m377["cycles"]["bit_sparse_reproduced_cycles"]
    candidate_reference = m377["cycles"][
        "m377_active_compact_candidate_cycles"]
    for command_setup in contract["sweep"]["dma_command_setup_cycles"]:
        baseline = sum(baseline_sample(
            phases[sample], model, command_setup)
                       for sample in range(model["samples"]))
        for sram_latency in contract["sweep"][
                "descriptor_sram_fixed_response_latency_cycles"]:
            samples = [candidate_sample(
                phases[sample], model, cfg, command_setup, sram_latency)
                       for sample in range(model["samples"])]
            candidate = sum(row["cycles"] for row in samples)
            sweeps.append({
                "scenario": "cmd{}_sramL{}_II1".format(
                    command_setup, sram_latency),
                "dma_command_setup_cycles": command_setup,
                "descriptor_sram_fixed_response_latency_cycles":
                    sram_latency,
                "blocking_cycles_per_replayed_descriptor": 0.0,
                "baseline_cycles": baseline,
                "candidate_cycles": candidate,
                "speedup_vs_bit_sparse": baseline / float(candidate),
                "exposed_tile1_dma_cycles": sum(
                    row["exposed_tile1_dma_cycles"] for row in samples),
            })

    blocking_rows = []
    blocking_point = contract["sweep"]["blocking_stress_point"]
    command_setup = blocking_point["dma_command_setup_cycles"]
    sram_latency = blocking_point[
        "descriptor_sram_fixed_response_latency_cycles"]
    baseline = sum(baseline_sample(phases[sample], model, command_setup)
                   for sample in range(model["samples"]))
    for penalty in contract["sweep"][
            "blocking_cycles_per_replayed_descriptor"]:
        samples = [candidate_sample(
            phases[sample], model, cfg, command_setup, sram_latency, penalty)
                   for sample in range(model["samples"])]
        candidate = sum(row["cycles"] for row in samples)
        blocking_rows.append({
            "scenario": "cmd{}_sramL{}_block{}".format(
                command_setup, sram_latency, penalty),
            "dma_command_setup_cycles": command_setup,
            "descriptor_sram_fixed_response_latency_cycles": sram_latency,
            "blocking_cycles_per_replayed_descriptor": penalty,
            "baseline_cycles": baseline,
            "candidate_cycles": candidate,
            "speedup_vs_bit_sparse": baseline / float(candidate),
            "exposed_tile1_dma_cycles": sum(
                row["exposed_tile1_dma_cycles"] for row in samples),
        })

    m377_replay = sum(candidate_sample(
        phases[sample], model, cfg, 0, 1)["cycles"]
                      for sample in range(model["samples"]))
    # M377 did not yet charge the one-cycle active-count/bitmap seal.
    phase_count = (model["samples"] * model["operators"] *
                   model["partitions_per_operator"])
    require(m377_replay - phase_count == candidate_reference,
            "M377 recurrence reproduction failure")
    require(sum(baseline_sample(phases[sample], model, 0)
                for sample in range(model["samples"])) ==
            baseline_reference, "baseline recurrence reproduction failure")

    robust = next(row for row in sweeps if
                  row["dma_command_setup_cycles"] ==
                  contract["decision_rule"]["robust_dma_command_setup_cycles"]
                  and row["descriptor_sram_fixed_response_latency_cycles"] ==
                  contract["decision_rule"][
                      "robust_descriptor_sram_latency_cycles"])
    decision = ("GO_VCS_STREAMING_ACTIVE_DESCRIPTOR_CONTROLLER" if
                robust["speedup_vs_bit_sparse"] >=
                contract["decision_rule"]["minimum_module_speedup"] else
                "NO_GO_BURST_STREAMING_OVERHEAD")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = ["scenario", "dma_command_setup_cycles",
              "descriptor_sram_fixed_response_latency_cycles",
              "blocking_cycles_per_replayed_descriptor", "baseline_cycles",
              "candidate_cycles", "speedup_vs_bit_sparse",
              "exposed_tile1_dma_cycles"]
    for name, rows in (("burst_streaming_sweep.csv", sweeps),
                       ("blocking_descriptor_sensitivity.csv",
                        blocking_rows)):
        with (args.output_dir / name).open("w", encoding="utf-8",
                                          newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    result = {
        "schema": "m381_q32_o4_burst_streaming_sensitivity_v1",
        "status": "PASS_M381_EXACT_BURST_STREAMING_SENSITIVITY",
        "identity": identities,
        "population": dict(phase_summary),
        "pwp_scatter_gather": {
            "phase_count": phase_count,
            "maximum_used_centers": max(used_counts),
            "maximum_used_center_runs": max(run_counts),
            "mean_used_centers": (sum(used_counts) /
                                  float(len(used_counts))),
            "mean_used_center_runs": (sum(run_counts) /
                                      float(len(run_counts))),
            "direct_address_center_slots": 32,
            "used_center_bitmap_bits": 32,
            "center_id_remap_required": False,
        },
        "reproduction": {
            "m377_candidate_without_new_count_seal":
                m377_replay - phase_count,
            "m377_frozen_candidate": candidate_reference,
            "bit_sparse_reproduced_cycles": baseline_reference,
            "m377_recurrence_match": True,
            "baseline_recurrence_match": True,
        },
        "sweep": sweeps,
        "blocking_sensitivity": blocking_rows,
        "robust_decision_point": robust,
        "decision": decision,
        "admission": {
            "timestamped_finite_event_parent": True,
            "exact_used_center_run_measurement": True,
            "finite_dma_command_setup_sweep": True,
            "finite_streaming_sram_latency_sweep": True,
            "rtl_cycle_match": False,
            "synopsys_area": False,
            "energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "output_files": {
            "burst_streaming_sweep": "burst_streaming_sweep.csv",
            "blocking_descriptor_sensitivity":
                "blocking_descriptor_sensitivity.csv",
        },
    }
    output = args.output_dir / (
        "m381_q32_o4_burst_streaming_sensitivity_r1.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M381_PASS used={:.3f} runs={:.3f} robust={:.6f}x decision={}".
          format(result["pwp_scatter_gather"]["mean_used_centers"],
                 result["pwp_scatter_gather"]["mean_used_center_runs"],
                 robust["speedup_vs_bit_sparse"], decision), flush=True)


if __name__ == "__main__":
    main()
