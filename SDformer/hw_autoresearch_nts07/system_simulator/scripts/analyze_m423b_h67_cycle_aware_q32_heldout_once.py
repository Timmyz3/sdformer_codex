#!/usr/bin/env python3
"""One-shot held-out M40 replay of the sealed M423a train-only q32 catalog."""

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


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_phase_q32(counter, centers, narrow_flags, m401):
    """M401-equivalent q32 analysis without unused q48..q128 diagnostics."""
    require(len(centers) == 32, "M423b requires exactly q32 centers")
    row = Counter()
    used = set()
    first_hit = Counter()
    cumulative_all = Counter()
    cumulative_eligible = Counter()
    for original, count in counter.items():
        original = int(original) & 0xffff
        population = m401.POPCOUNT[original]
        row["source_rows"] += count
        row["zero_rows"] += count * int(original == 0)
        row["pop1_rows"] += count * int(population == 1)
        eligible = population >= 2
        row["eligible_rows"] += count * int(eligible)
        distance = [m401.POPCOUNT[original ^ (int(center) & 0xffff)]
                    for center in centers]
        q16_exact = min(distance[:16]) == 0
        q32_exact = min(distance) == 0
        if original != 0:
            cumulative_all[16] += count * int(q16_exact)
            cumulative_all[32] += count * int(q32_exact)
        if eligible:
            cumulative_eligible[16] += count * int(q16_exact)
            cumulative_eligible[32] += count * int(q32_exact)
            first = 16 if q16_exact else (32 if q32_exact else 0)
            first_hit[first] += count
            row["q32_early_extra_prefix_tasks"] += count * int(not q16_exact)

        if original == 0:
            continue
        best_distance = min(distance)
        best_index = distance.index(best_distance)
        use_pwp = 1 + best_distance < population
        row["active_rows"] += count
        row["bit_sparse_vector_ops_per_block"] += count * population
        row["candidate_vector_ops_per_block"] += count * (
            1 + best_distance if use_pwp else population)
        if use_pwp:
            selected = int(centers[best_index]) & 0xffff
            plus = original & ((~selected) & 0xffff)
            minus = selected & ((~original) & 0xffff)
            require(((selected | plus) & ((~minus) & 0xffff)) == original,
                    "M423b exact residual reconstruction failure")
            row["pwp_rows"] += count
            row["correction_ops_per_block"] += count * best_distance
            row["exact_reconstruction_rows"] += count
            used.add(best_index)
            row["narrow_block_descriptors_tile0"] += (
                count * int(narrow_flags[best_index, 0:4].sum()))
            row["narrow_block_descriptors_tile1"] += (
                count * int(narrow_flags[best_index, 4:8].sum()))
        else:
            row["fallback_rows"] += count
            row["correction_ops_per_block"] += count * population
    row["used_pwp_patterns"] = len(used)
    row["used_center_runs"] = m401.count_runs(used)
    row["q32_reference_matcher_cycles"] = (
        row["source_rows"] + row["eligible_rows"] + 2)
    row["q32_early_matcher_cycles"] = (
        row["source_rows"] + row["q32_early_extra_prefix_tasks"] + 2)
    row["q32_early_saved_cycles"] = (
        row["eligible_rows"] - row["q32_early_extra_prefix_tasks"])
    require(row["source_rows"] == row["zero_rows"] + row["active_rows"] and
            row["active_rows"] == row["pwp_rows"] + row["fallback_rows"] and
            row["pwp_rows"] == row["exact_reconstruction_rows"],
            "M423b population conservation failure")
    require(row["narrow_block_descriptors_tile0"] +
            row["narrow_block_descriptors_tile1"] <= row["pwp_rows"] * 8,
            "M423b narrow descriptor overcount")
    return dict(row), first_hit, cumulative_all, cumulative_eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(),
            "refusing a second/overwrite M423b held-out execution")
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m423b_recovery_h67_cycle_aware_q32_heldout_contract_v1" and
            contract.get("status") ==
            "FROZEN_RECOVERY_AFTER_PRE_EVALUATION_ABORT",
            "M423b recovery contract drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M423b input SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": spec["path"],
                            "sha256": spec["sha256"]}
    require(paths["analyzer"].resolve() == Path(__file__).resolve() and
            identities["analyzer"]["sha256"] == source_start,
            "M423b analyzer self-identity drift")

    catalog = strict_json(paths["m423a_catalog"])
    train_audit = strict_json(paths["m423a_audit"])
    trace = strict_json(paths["m40_trace"])
    m401_result = strict_json(paths["m401_result"])
    require(catalog["status"] ==
            "PASS_M423A_TRAIN_ONLY_Q32_CATALOG_FROZEN_BEFORE_HELDOUT" and
            catalog["split"]["runtime_or_validation_data_used"] is False and
            catalog["admission"]["heldout_runtime_evaluated"] is False and
            train_audit["heldout_gate"]["m40_executions_so_far"] == 0 and
            train_audit["heldout_gate"]["tuning_after_m40"] is False,
            "M423b train-only freeze drift")
    require(contract["attempt_audit"]["prior_payload_read_attempts"] == 1 and
            contract["attempt_audit"]["prior_completed_phase_rows"] == 0 and
            contract["attempt_audit"]["prior_completed_heldout_evaluations"] == 0 and
            contract["attempt_audit"]["catalog_changed_after_abort"] is False,
            "M423b recovery-attempt audit drift")
    paper = contract["paper_identity"]
    require(trace["identity"]["checkpoint_sha256"] ==
            paper["checkpoint_sha256"] and
            trace["identity"]["bn_policy"] == paper["bn_policy"] and
            m401_result["paper_identity"] == paper,
            "M423b held-out paper identity drift")
    require(m401_result["robust_variants"]["combined"]["candidate_cycles"] ==
            contract["comparison"]["m401_candidate_cycles"] and
            m401_result["robust_variants"]["combined"]["baseline_cycles"] ==
            contract["comparison"]["strong_baseline_cycles"],
            "M423b M401 comparison identity drift")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    m401 = load_module(paths["m401_analyzer"], "m423b_m401")
    m43 = load_module(paths["m43_unpacker"], "m423b_m43")
    weights = [paths["weight_o{}".format(index)] for index in range(4)]
    flags, codec = m401.build_static_codec(
        catalog, weights, args.output_dir / "static_codec_audit.csv")
    require(codec["blocks"] == 442368 and codec["lanes"] == 42467328 and
            codec["signed12_violations"] == 0 and
            codec["wide_reconstruction_mismatches"] == 0 and
            codec["narrow_reconstruction_mismatches"] == 0 and
            codec["nonzero_padding_bytes"] == 0,
            "M423b static codec gate failure")

    trace_dir = paths["m40_trace"].parent
    operators = tuple(trace["cohort"]["operators"])
    require(tuple(catalog["geometry"]["operators"]) == operators and
            len(operators) == 4,
            "M423b catalog/runtime operator order drift")
    operator_index = {name: index for index, name in enumerate(operators)}
    histograms = defaultdict(Counter)
    payload_files = 0
    payload_bytes = 0
    for record_index, record in enumerate(trace["records"]):
        for key, sha_key in (("packed_file", "packed_file_sha256"),
                             ("value_payload_file",
                              "value_payload_sha256")):
            path = trace_dir / record[key]
            require(path.is_file() and sha256(path) == record[sha_key],
                    "M423b M40 payload drift")
            payload_files += 1
            payload_bytes += path.stat().st_size
        masks = m43.unpack_record_masks(trace_dir, record)
        for source_row in range(m43.ROWS):
            base = source_row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                for subtile in range(16):
                    value = (value256 >> (subtile * 16)) & 0xffff
                    histograms[(int(record["sample_id"]),
                                operator_index[record["operator"]],
                                tile * 16 + subtile)][value] += 1
        print("[M423B HIST] {}/{}".format(
            record_index + 1, len(trace["records"])), flush=True)

    phases = defaultdict(list)
    aggregate = Counter()
    first_hit = Counter()
    cumulative_all = Counter()
    cumulative_eligible = Counter()
    phase_rows = []
    q16_prefix_mismatches = 0
    old_catalog = strict_json(paths["m338_catalog"])
    for sample in range(10):
        for operator in range(4):
            for partition in range(432):
                counter = histograms[(sample, operator, partition)]
                require(sum(counter.values()) == 3000,
                        "M423b phase extent drift")
                centers = [int(value, 16) for value in
                           catalog["operators"][operator]["partitions"]
                           [partition]["nested_patterns"]]
                old_centers = [int(value, 16) for value in
                               old_catalog["operators"][operator]
                               ["partitions"][partition]["nested_patterns"]]
                q16_prefix_mismatches += int(centers[:16] != old_centers[:16])
                phase, phase_first, phase_all, phase_eligible = (
                    analyze_phase_q32(counter, centers,
                                      flags[operator][partition], m401))
                phases[sample].append(phase)
                aggregate.update(phase)
                first_hit.update(phase_first)
                cumulative_all.update(phase_all)
                cumulative_eligible.update(phase_eligible)
                phase_rows.append({
                    "sample": sample,
                    "operator": operator,
                    "partition": partition,
                    "active_rows": phase["active_rows"],
                    "eligible_rows": phase["eligible_rows"],
                    "pwp_rows": phase["pwp_rows"],
                    "fallback_rows": phase["fallback_rows"],
                    "correction_ops_per_block":
                        phase["correction_ops_per_block"],
                    "used_pwp_patterns": phase["used_pwp_patterns"],
                    "used_center_runs": phase["used_center_runs"],
                    "narrow_tile0":
                        phase["narrow_block_descriptors_tile0"],
                    "narrow_tile1":
                        phase["narrow_block_descriptors_tile1"],
                    "early_matcher": phase["q32_early_matcher_cycles"],
                    "early_saved": phase["q32_early_saved_cycles"],
                })
        print("[M423B PHASE] sample={}/10".format(sample + 1), flush=True)
    require(q16_prefix_mismatches == 0,
            "M423b q16 prefix mismatch")
    require(len(phase_rows) == 17280 and
            aggregate["source_rows"] == 51840000 and
            aggregate["source_rows"] ==
            aggregate["zero_rows"] + aggregate["active_rows"] and
            aggregate["active_rows"] ==
            aggregate["pwp_rows"] + aggregate["fallback_rows"] and
            aggregate["pwp_rows"] == aggregate["exact_reconstruction_rows"],
            "M423b population/exact reconstruction drift")
    require(aggregate["q32_early_matcher_cycles"] ==
            contract["execution_gates"]["expected_q16_early_matcher_cycles"],
            "M423b q16 early-hit ledger drift")

    model = contract["cycle_model"]
    baseline = sum(m401.baseline_sample(
        phases[sample], contract["decision_rule"]
        ["robust_dma_command_setup_cycles"], model) for sample in range(10))
    sample_results = [m401.candidate_sample(
        phases[sample], "combined",
        contract["decision_rule"]["robust_dma_command_setup_cycles"],
        contract["decision_rule"]["robust_descriptor_sram_latency_cycles"],
        model, capture_phase_timestamps=True) for sample in range(10)]
    candidate = int(sum(result["cycles"] for result in sample_results))
    components = Counter()
    timestamps = []
    for sample, sample_result in enumerate(sample_results):
        components.update(sample_result["components"])
        for timestamp in sample_result["timestamps"]:
            row = dict(timestamp)
            row["sample"] = sample
            timestamps.append(row)
    maximum_slot = max(result["maximum_slot_bytes"]
                       for result in sample_results)
    require(baseline == contract["comparison"]["strong_baseline_cycles"] and
            maximum_slot <= model["tile_slot_bytes"] and
            components["tile1_dma_exposed"] == 0,
            "M423b cycle/resource gate failure")
    old_cycles = contract["comparison"]["m401_candidate_cycles"]
    decision = ("GO_M423_Q32_CATALOG" if candidate < old_cycles else
                "NO_GO_RETAIN_M401_Q32_CATALOG")

    write_csv(args.output_dir / "per_phase_heldout_replay.csv", phase_rows,
              ["sample", "operator", "partition", "active_rows",
               "eligible_rows", "pwp_rows", "fallback_rows",
               "correction_ops_per_block", "used_pwp_patterns",
               "used_center_runs", "narrow_tile0", "narrow_tile1",
               "early_matcher", "early_saved"])
    write_csv(args.output_dir / "combined_phase_timestamps.csv", timestamps,
              ["sample", "phase_index", "phase_start",
               "tile0_replay_start", "tile0_replay_end", "tile1_dma_end",
               "exposed_tile1_dma", "tile1_replay_start", "phase_end"])
    require(source_start == sha256(Path(__file__).resolve()),
            "M423b analyzer changed during one-shot execution")
    result = {
        "schema": "m423b_recovery_h67_cycle_aware_q32_heldout_v1",
        "status": "PASS_M423B_RECOVERY_ONE_COMPLETED_HELDOUT_EVALUATION",
        "identity": identities,
        "paper_identity": paper,
        "split_audit": {
            "catalog_population": "M73 disjoint DSEC train-only S32/18 sequences",
            "heldout_population": "M40 frozen S10 zurich_city_09_a",
            "catalog_sealed_before_heldout": True,
            "payload_read_attempts_including_aborted_r1": 2,
            "completed_heldout_evaluation_count": 1,
            "prior_attempt_completed_phase_rows": 0,
            "post_heldout_tuning": False,
            "valid825_used": False,
        },
        "payload_audit": {"files_rehashed": payload_files,
                          "bytes_rehashed": payload_bytes,
                          "mismatches": 0},
        "static_codec": codec,
        "runtime_population": dict(aggregate),
        "prefix_first_hit_eligible": {
            ("q{}".format(prefix) if prefix else "no_exact"):
            first_hit[prefix] for prefix in (16, 32, 0)},
        "prefix_cumulative_all_nonzero": {
            "q{}".format(prefix): cumulative_all[prefix]
            for prefix in (16, 32)},
        "prefix_cumulative_eligible": {
            "q{}".format(prefix): cumulative_eligible[prefix]
            for prefix in (16, 32)},
        "robust_point": {
            "strong_zero_elided_baseline_cycles": baseline,
            "m401_original_candidate_cycles": old_cycles,
            "m423_candidate_cycles": candidate,
            "m423_speedup_vs_strong_zero_elided": baseline / float(candidate),
            "m423_speedup_vs_m401_candidate": old_cycles / float(candidate),
            "cycle_reduction_vs_m401": old_cycles - candidate,
            "cycle_reduction_fraction_vs_m401":
                (old_cycles - candidate) / float(old_cycles),
            "dma_command_setup_cycles": contract["decision_rule"]
            ["robust_dma_command_setup_cycles"],
            "descriptor_sram_latency_cycles": contract["decision_rule"]
            ["robust_descriptor_sram_latency_cycles"],
            "maximum_slot_bytes": maximum_slot,
        },
        "component_ledger": dict(components),
        "decision": decision,
        "execution_gates": {
            "input_or_payload_sha_mismatches": 0,
            "q16_prefix_mismatches": q16_prefix_mismatches,
            "signed12_or_codec_mismatches": 0,
            "population_or_exact_reconstruction_mismatches": 0,
            "q16_early_matcher_cycle_mismatch": 0,
            "tile_slot_overflows": 0,
            "all_17280_phases_complete": True,
            "payload_read_attempts": 2,
            "completed_heldout_evaluations": 1,
            "post_heldout_retuning": False,
        },
        "admission": {
            "exact_arithmetic": True,
            "checkpoint_or_accuracy_changed": False,
            "accuracy_loss": False,
            "frozen_h67_trace_cycle_replay": True,
            "standalone_four_bottleneck_conv_module_cycles": True,
            "catalog_selected_for_next_exact_stimulus": decision.startswith("GO"),
            "rtl_measured_speedup": False,
            "synopsys": False,
            "energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "output_files": {
            "static_codec_audit": "static_codec_audit.csv",
            "per_phase_heldout_replay": "per_phase_heldout_replay.csv",
            "combined_phase_timestamps": "combined_phase_timestamps.csv"
        }
    }
    output = args.output_dir / "m423b_recovery_h67_cycle_aware_q32_heldout_r2.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M423B_PASS candidate={} old={} improvement={:.6%} strong={:.9f}x decision={}".format(
        candidate, old_cycles, (old_cycles - candidate) / float(old_cycles),
        baseline / float(candidate), decision), flush=True)


if __name__ == "__main__":
    main()
