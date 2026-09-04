#!/usr/bin/env python3
"""Rebind the exact q32/O4 burst-streaming model to frozen H67 ep35 S10.

This audit deliberately reuses only the arithmetic/scheduling functions from
M381.  It rebuilds every phase histogram from the H67 ep35/no-running M40
payload instead of the rejected PAFT-ep4 runtime population.
"""

from __future__ import division

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
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


def write_csv(path, rows):
    fields = ["scenario", "dma_command_setup_cycles",
              "descriptor_sram_fixed_response_latency_cycles",
              "blocking_cycles_per_replayed_descriptor", "baseline_cycles",
              "candidate_cycles", "speedup_vs_bit_sparse",
              "exposed_tile1_dma_cycles"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_transitive(hw_root, identity):
    path = hw_root / identity["path"]
    require(path.is_file(), "missing transitive input: " + str(path))
    require(sha256(path) == identity["sha256"],
            "transitive SHA drift: " + str(path))
    return path


def popcount16(value):
    return bin(int(value) & 0xffff).count("1")


def count_runs(indices):
    ordered = sorted(indices)
    if not ordered:
        return 0
    return 1 + sum(current != previous + 1
                   for previous, current in zip(ordered, ordered[1:]))


def phase_metrics_q32(counter, nested_centers):
    """Compute only the deployed q32 prefix, preserving M339 semantics."""
    require(len(nested_centers) >= 32, "M394 q32 catalog too short")
    centers = nested_centers[:32]
    result = Counter()
    used = set()
    exact_reconstruction_rows = 0
    pop1_fallback_rows = 0
    for original, count in counter.items():
        original = int(original) & 0xffff
        population = popcount16(original)
        best_distance = 17
        best_index = 0
        for index, center in enumerate(centers):
            distance = popcount16(original ^ int(center))
            if distance < best_distance:
                best_distance = distance
                best_index = index
        use_pwp = 1 + best_distance < population
        candidate = 1 + best_distance if use_pwp else population
        result["partition_vectors"] += count
        result["matcher_rows"] += count * int(population >= 2)
        result["bit_sparse_vector_ops_per_block"] += count * population
        result["candidate_vector_ops_per_block"] += count * candidate
        result["pwp_ops_per_block"] += count * int(use_pwp)
        result["correction_ops_per_block"] += count * (
            best_distance if use_pwp else population)
        result["assignment_rows"] += count * int(use_pwp)
        result["exact_pattern_hits"] += count * int(
            original != 0 and best_distance == 0)
        if use_pwp:
            require(original != 0, "zero row illegally selected PWP")
            center = int(centers[best_index]) & 0xffff
            plus = original & ((~center) & 0xffff)
            minus = center & ((~original) & 0xffff)
            require(((center | plus) & ((~minus) & 0xffff)) == original,
                    "M394 PWP exact reconstruction failure")
            used.add(best_index)
            exact_reconstruction_rows += count
        elif original != 0 and population == 1:
            pop1_fallback_rows += count
    require(result["candidate_vector_ops_per_block"] ==
            result["pwp_ops_per_block"] +
            result["correction_ops_per_block"],
            "M394 phase work conservation failure")
    result["used_pwp_patterns"] = len(used)
    result["zero_rows"] = counter[0]
    result["active_rows"] = result["partition_vectors"] - counter[0]
    result["used_center_ids"] = sorted(used)
    result["used_center_bitmap"] = sum(1 << index for index in used)
    result["used_center_runs"] = count_runs(used)
    result["pwp_rows"] = result["assignment_rows"]
    result["fallback_rows"] = (
        result["active_rows"] - result["assignment_rows"])
    result["pop1_fallback_rows"] = pop1_fallback_rows
    result["exact_reconstruction_rows"] = exact_reconstruction_rows
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M394 output overwrite")

    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m394_h67_ep35_q32_o4_burst_streaming_contract_v1",
            "M394 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M394_EXECUTION",
            "M394 contract not frozen")
    hw_root = args.contract.resolve().parents[1]
    identities = {}
    paths = {}
    for name, identity in contract["inputs"].items():
        path = hw_root / identity["path"]
        require(path.is_file(), "missing M394 input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"],
                "M394 SHA drift for " + name)
        paths[name] = path
        identities[name] = {"path": identity["path"],
                            "sha256": observed}

    h67_sha = contract["paper_identity"]["checkpoint_sha256"]
    catalog = strict_json(paths["m338_catalog"])
    trace = strict_json(paths["m40_h67_s10_trace"])
    bridge = strict_json(paths["m41_h67_int8_bridge"])
    parent = strict_json(paths["m381_paft_result"])
    require(catalog["status"] ==
            "PASS_M338_TRAIN_ONLY_NESTED_Q16_Q32_Q64_Q128_EXACT_WORK_NO_CYCLES",
            "M338 catalog status drift")
    require(catalog["split"]["test_or_validation_data_used"] is False and
            catalog["split"]["train_valid825_key_overlap"] == 0,
            "M338 split contamination")
    require(trace["status"] ==
            "PASS_EXACT_H67_EP35_S10_FOUR_BOTTLENECK_SUPPORT_SIGN_BITMAPS_AND_RECONSTRUCTABLE_FLOAT32_VALUES",
            "M40 H67 runtime status drift")
    require(trace["identity"]["checkpoint_sha256"] == h67_sha and
            trace["identity"]["bn_policy"] == "no_running",
            "M40 H67 paper identity drift")
    require(bridge["identity"]["checkpoint_sha256"] == h67_sha and
            bridge["identity"]["m40_source_manifest_sha256"] ==
            identities["m40_h67_s10_trace"]["sha256"],
            "M41 H67 bridge identity drift")
    require(parent["admission"]["system_speedup"] is False and
            parent["admission"]["date_headline"] is False,
            "M381 PAFT parent claim boundary drift")

    # The nested catalog is fitted on disjoint H67 training samples, not on a
    # PAFT checkpoint.  Verify both transitive sources and their checkpoint.
    m73_path = resolve_transitive(
        hw_root, catalog["identity"]["m73_train_trace_manifest"])
    m77_path = resolve_transitive(
        hw_root, catalog["identity"]["m77_q16_catalog"])
    m73 = strict_json(m73_path)
    m77 = strict_json(m77_path)
    require(m73["identity"]["checkpoint_sha256"] == h67_sha and
            m73["identity"]["bn_policy"] == "no_running" and
            m73["admission"]["paft_catalog"] is False,
            "M73 is not the frozen H67 train-only source")
    require(m77["identity"]["checkpoint_sha256"] == h67_sha and
            m77["admission"]["paft_checkpoint"] is False,
            "M77 catalog checkpoint drift")

    m381 = load_module(paths["m381_analyzer"], "m394_m381")
    m339 = load_module(paths["m339_analyzer"], "m394_m339")
    m43 = load_module(paths["m43_unpacker"], "m394_m43")
    model = contract["cycle_model"]
    cfg = contract["configuration"]
    require(model["samples"] == trace["cohort"]["samples"] == 10 and
            model["operators"] == len(trace["cohort"]["operators"]) == 4,
            "M394 population geometry drift")
    require(catalog["geometry"]["operators"] ==
            trace["cohort"]["operators"], "catalog/runtime operator drift")

    trace_dir = paths["m40_h67_s10_trace"].parent
    operators = tuple(trace["cohort"]["operators"])
    op_index = {name: index for index, name in enumerate(operators)}
    histograms = defaultdict(Counter)
    payload_files = 0
    payload_bytes = 0
    for record_index, record in enumerate(trace["records"]):
        for key, sha_key in (("packed_file", "packed_file_sha256"),
                             ("value_payload_file",
                              "value_payload_sha256")):
            payload = trace_dir / record[key]
            require(payload.is_file() and
                    sha256(payload) == record[sha_key],
                    "M40 payload drift: " + str(payload))
            payload_files += 1
            payload_bytes += payload.stat().st_size
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
        print("[M394 H67 HIST] {}/{}".format(
            record_index + 1, len(trace["records"])), flush=True)

    phases = defaultdict(list)
    summary = Counter()
    used_counts = []
    run_counts = []
    for sample in range(model["samples"]):
        for op in range(model["operators"]):
            for partition in range(model["partitions_per_operator"]):
                counter = histograms[(sample, op, partition)]
                require(sum(counter.values()) == model["rows_per_operator"],
                        "M394 phase row extent drift")
                centers = [int(value, 16) for value in
                           catalog["operators"][op]["partitions"][partition]
                           ["nested_patterns"]]
                require(len(centers) >= 128, "M394 q128 prefix missing")
                phase = phase_metrics_q32(counter, centers)
                if op == 0 and partition == 0:
                    frozen = m381.phase_metrics(counter, centers, m339)
                    require(phase == frozen,
                            "M394 q32-only/frozen phase metric mismatch")
                phases[sample].append(phase)
                used_counts.append(phase["used_pwp_patterns"])
                run_counts.append(phase["used_center_runs"])
                for source, target in (
                        ("partition_vectors", "source_rows"),
                        ("active_rows", "active_rows"),
                        ("zero_rows", "zero_rows"),
                        ("pwp_rows", "pwp_rows"),
                        ("fallback_rows", "fallback_rows"),
                        ("pop1_fallback_rows", "pop1_fallback_rows"),
                        ("used_pwp_patterns", "used_centers"),
                        ("used_center_runs", "used_center_runs"),
                        ("exact_reconstruction_rows",
                         "exact_reconstruction_rows")):
                    summary[target] += phase[source]
        print("[M394 H67 METRIC] sample={}/{}".format(
            sample + 1, model["samples"]), flush=True)
    require(summary["source_rows"] ==
            model["samples"] * model["operators"] *
            model["partitions_per_operator"] * model["rows_per_operator"],
            "M394 source population conservation failure")
    require(summary["active_rows"] ==
            summary["pwp_rows"] + summary["fallback_rows"],
            "M394 active population conservation failure")
    require(summary["pwp_rows"] == summary["exact_reconstruction_rows"],
            "M394 exact reconstruction population failure")

    sweeps = []
    for command_setup in contract["sweep"]["dma_command_setup_cycles"]:
        baseline = sum(m381.baseline_sample(
            phases[sample], model, command_setup)
                       for sample in range(model["samples"]))
        for latency in contract["sweep"][
                "descriptor_sram_fixed_response_latency_cycles"]:
            rows = [m381.candidate_sample(
                phases[sample], model, cfg, command_setup, latency)
                    for sample in range(model["samples"])]
            candidate = sum(row["cycles"] for row in rows)
            sweeps.append({
                "scenario": "cmd{}_sramL{}_II1".format(
                    command_setup, latency),
                "dma_command_setup_cycles": command_setup,
                "descriptor_sram_fixed_response_latency_cycles": latency,
                "blocking_cycles_per_replayed_descriptor": 0.0,
                "baseline_cycles": baseline,
                "candidate_cycles": candidate,
                "speedup_vs_bit_sparse": baseline / float(candidate),
                "exposed_tile1_dma_cycles": sum(
                    row["exposed_tile1_dma_cycles"] for row in rows),
            })

    point = contract["sweep"]["blocking_stress_point"]
    baseline = sum(m381.baseline_sample(
        phases[sample], model, point["dma_command_setup_cycles"])
                   for sample in range(model["samples"]))
    blocking = []
    for penalty in contract["sweep"][
            "blocking_cycles_per_replayed_descriptor"]:
        rows = [m381.candidate_sample(
            phases[sample], model, cfg,
            point["dma_command_setup_cycles"],
            point["descriptor_sram_fixed_response_latency_cycles"], penalty)
                for sample in range(model["samples"])]
        candidate = sum(row["cycles"] for row in rows)
        blocking.append({
            "scenario": "cmd{}_sramL{}_block{}".format(
                point["dma_command_setup_cycles"],
                point["descriptor_sram_fixed_response_latency_cycles"],
                penalty),
            "dma_command_setup_cycles": point["dma_command_setup_cycles"],
            "descriptor_sram_fixed_response_latency_cycles":
                point["descriptor_sram_fixed_response_latency_cycles"],
            "blocking_cycles_per_replayed_descriptor": penalty,
            "baseline_cycles": baseline,
            "candidate_cycles": candidate,
            "speedup_vs_bit_sparse": baseline / float(candidate),
            "exposed_tile1_dma_cycles": sum(
                row["exposed_tile1_dma_cycles"] for row in rows),
        })

    rule = contract["decision_rule"]
    robust = next(row for row in sweeps if
                  row["dma_command_setup_cycles"] ==
                  rule["robust_dma_command_setup_cycles"] and
                  row["descriptor_sram_fixed_response_latency_cycles"] ==
                  rule["robust_descriptor_sram_latency_cycles"])
    decision = ("GO_H67_CONTROLLER_REALTRACE_MITER" if
                robust["speedup_vs_bit_sparse"] >=
                rule["minimum_module_speedup"] else
                "NO_GO_H67_Q32_O4_PERFORMANCE")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "burst_streaming_sweep.csv", sweeps)
    write_csv(args.output_dir / "blocking_descriptor_sensitivity.csv",
              blocking)
    result = {
        "schema": "m394_h67_ep35_q32_o4_burst_streaming_v1",
        "status": "PASS_M394_H67_EP35_EXACT_Q32_O4_BURST_STREAMING",
        "identity": identities,
        "identity_rebinding": {
            "checkpoint_sha256": h67_sha,
            "bn_policy": "no_running",
            "runtime_trace": "M40 H67 ep35 S10",
            "catalog_fit": "M73 disjoint DSEC train-only H67 ep35",
            "paft_checkpoint_used": False,
            "paft_m381_result_role": "exploratory comparison only",
            "payload_files_rehashed": payload_files,
            "payload_bytes_rehashed": payload_bytes,
        },
        "population": dict(summary),
        "pwp_scatter_gather": {
            "phase_count": len(used_counts),
            "maximum_used_centers": max(used_counts),
            "maximum_used_center_runs": max(run_counts),
            "mean_used_centers": sum(used_counts) / float(len(used_counts)),
            "mean_used_center_runs": sum(run_counts) / float(len(run_counts)),
            "direct_address_center_slots": 32,
            "used_center_bitmap_bits": 32,
            "center_id_remap_required": False,
        },
        "sweep": sweeps,
        "blocking_sensitivity": blocking,
        "robust_decision_point": robust,
        "comparison_only": {
            "m381_paft_robust_speedup":
                parent["robust_decision_point"]["speedup_vs_bit_sparse"],
            "h67_minus_paft_speedup_ratio":
                robust["speedup_vs_bit_sparse"] /
                parent["robust_decision_point"]["speedup_vs_bit_sparse"],
            "no_cross_checkpoint_population_mixing": True,
        },
        "decision": decision,
        "admission": {
            "frozen_h67_ep35_no_running_runtime": True,
            "disjoint_train_only_catalog": True,
            "exact_zero_elision_and_center_residual_reconstruction": True,
            "finite_dma_command_and_sram_latency_sweep": True,
            "standalone_four_conv_module_cycles": True,
            "rtl_realtrace_cycle_match": False,
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
        "m394_h67_ep35_q32_o4_burst_streaming_r1.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M394_PASS active={} used={:.3f} runs={:.3f} robust={:.6f}x decision={}".
          format(summary["active_rows"],
                 result["pwp_scatter_gather"]["mean_used_centers"],
                 result["pwp_scatter_gather"]["mean_used_center_runs"],
                 robust["speedup_vs_bit_sparse"], decision), flush=True)


if __name__ == "__main__":
    main()
