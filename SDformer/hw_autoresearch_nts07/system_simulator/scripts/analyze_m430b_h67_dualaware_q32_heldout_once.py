#!/usr/bin/env python3
"""One completed M40 replay for the pre-heldout-sealed M430 q32 catalog."""

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
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dual_sample(phases, command_setup, latency, model,
                capture_phase_timestamps=False):
    time = 0
    components = Counter()
    timestamps = []
    maximum_slot = 0
    for phase_index, phase in enumerate(phases):
        config_data = int(math.ceil(
            model["elastic_config_bytes"] /
            float(model["dram_bytes_per_cycle"])))
        config = config_data + command_setup
        matcher = phase["q32_early_matcher_cycles"]
        seal = 1
        start = time
        time += config + matcher + seal
        components["config_data"] += config_data
        components["config_command"] += command_setup
        components["matcher"] += matcher
        components["bitmap_seal"] += seal
        if phase["active_rows"] == 0:
            time += model["tail_cycles"]
            components["tail"] += model["tail_cycles"]
            continue
        tile_bytes = (model["weight_bytes_per_tile"] +
                      phase["used_pwp_patterns"] *
                      model["elastic_center_stride_bytes"])
        maximum_slot = max(maximum_slot,
                           model["elastic_config_bytes"] + tile_bytes)
        require(model["elastic_config_bytes"] + tile_bytes <=
                model["tile_slot_bytes"], "M430 heldout slot overflow")
        require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                "M430 heldout tile DMA alignment drift")
        tile_data = tile_bytes // model["dram_bytes_per_cycle"]
        tile_commands = 1 + phase["used_center_runs"]
        tile_dma = tile_data + tile_commands * command_setup
        # Legal co-read has one PWP issue for every output block.  It does not
        # fuse a correction and cannot replace persistent old_psum.
        work = (model["output_blocks_per_tile"] *
                (phase["correction_ops_per_block"] + phase["pwp_rows"]))
        require(work >= phase["active_rows"],
                "M430 heldout descriptor service underflow")
        replay0 = work + latency
        replay1 = work + latency
        time += tile_dma
        tile0_start = time
        tile0_end = tile0_start + replay0
        tile1_dma_end = tile0_start + tile_dma
        tile1_start = max(tile0_end, tile1_dma_end)
        tile1_exposed = max(0, tile1_dma_end - tile0_end)
        time = tile1_start + replay1 + model["tail_cycles"]
        components["tile0_dma_data"] += tile_data
        components["tile0_dma_commands"] += tile_commands * command_setup
        components["tile1_dma_exposed"] += tile1_exposed
        components["replay0"] += replay0
        components["replay1"] += replay1
        components["active_compute"] += 2 * work
        components["descriptor_sram_startup"] += 2 * latency
        components["tail"] += model["tail_cycles"]
        components["pwp_dram_physical_bytes"] += (
            phase["used_pwp_patterns"] *
            model["elastic_center_stride_bytes"] * 2)
        components["weight_dram_bytes"] += model["weight_bytes_per_tile"] * 2
        components["tile_dma_commands"] += tile_commands * 2
        components["descriptor_reads_responses_bundles"] += (
            phase["active_rows"] * 2)
        components["pwp_output_block_issues"] += (
            phase["pwp_rows"] * model["output_blocks"])
        components["pwp_logical_onchip_read_bytes"] += (
            phase["pwp_rows"] * model["output_blocks"] *
            model["dual_pwp_logical_bytes_per_issue"])
        components["pwp_padded_signal_bytes"] += (
            phase["pwp_rows"] * model["output_blocks"] *
            model["dual_pwp_padded_signal_bytes_per_issue"])
        components["correction_output_block_issues"] += (
            phase["correction_ops_per_block"] * model["output_blocks"])
        components["correction_onchip_read_bytes"] += (
            phase["correction_ops_per_block"] * model["output_blocks"] *
            model["correction_bytes_per_issue"])
        if capture_phase_timestamps:
            timestamps.append({
                "phase_index": phase_index,
                "phase_start": start,
                "tile0_replay_start": tile0_start,
                "tile0_replay_end": tile0_end,
                "tile1_dma_end": tile1_dma_end,
                "exposed_tile1_dma": tile1_exposed,
                "tile1_replay_start": tile1_start,
                "phase_end": time,
            })
    time += model["commit_cycles_per_sample"]
    components["commit"] += model["commit_cycles_per_sample"]
    return {"cycles": int(time), "components": dict(components),
            "maximum_slot_bytes": maximum_slot, "timestamps": timestamps}


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    require(not args.output_dir.exists(), "refusing M430 heldout overwrite")
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m430b_h67_dualaware_q32_heldout_once_contract_v1" and
            contract.get("status") ==
            "FROZEN_AFTER_TRAIN_DOUBLE_SEAL_BEFORE_M40_ONCE",
            "M430 heldout contract drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M430 heldout input SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": spec["sha256"]}
    require(paths["analyzer"].resolve() == Path(__file__).resolve() and
            identities["analyzer"]["sha256"] == source_start,
            "M430 heldout analyzer self-identity drift")

    catalog = strict_json(paths["m430_catalog"])
    train_audit = strict_json(paths["m430_train_audit"])
    trace = strict_json(paths["m40_trace"])
    m338 = strict_json(paths["m338_catalog"])
    m423_result = strict_json(paths["m423_result"])
    m423_review = strict_json(paths["m428_m423_review"])
    semantic = strict_json(paths["m427r3_semantic_review"])
    require(catalog["status"] ==
            "PASS_M430_TRAIN_ONLY_DUALAWARE_Q32_FROZEN_BEFORE_HELDOUT" and
            catalog["split"]["runtime_or_validation_data_used"] is False and
            catalog["admission"]["heldout_runtime_evaluated"] is False and
            train_audit["heldout_gate"]["m40_payload_reads_so_far"] == 0 and
            train_audit["heldout_gate"]
            ["m40_completed_evaluations_so_far"] == 0 and
            train_audit["heldout_gate"]["post_m40_tuning_allowed"] is False,
            "M430 pre-heldout train seal drift")
    require(m423_result["robust_point"]["m423_candidate_cycles"] ==
            contract["comparisons"]["m423_serial_cycles"] and
            m423_review["severity_counts"]["P0"] == 0 and
            semantic["verdict"]["dual_coread_530606660_cycles"] ==
            "SEMANTICALLY_SURVIVES" and
            semantic["dual_coread_independent_judgment"]
            ["logical_pwp_read_bytes_per_cycle"] == 144,
            "M430 comparison admission drift")
    paper = contract["paper_identity"]
    require(trace["identity"]["checkpoint_sha256"] ==
            paper["checkpoint_sha256"] and
            trace["identity"]["bn_policy"] == paper["bn_policy"],
            "M430 paper identity drift")

    # Validate both pre-heldout seal layers before creating the one-shot marker.
    train_dir = paths["m430_catalog"].parent
    train_manifest = paths["m430_double_manifest"]
    for line in train_manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(sha256(train_dir / name) == expected,
                "M430 train inner seal mismatch: " + name)
    expected_manifest, manifest_name = paths["m430_double_seal"].read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(manifest_name == "SHA256SUMS" and
            sha256(train_manifest) == expected_manifest,
            "M430 train outer seal mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    m401 = load_module(paths["m401_analyzer"], "m430_m401")
    m423 = load_module(paths["m423b_helper"], "m430_m423b")
    m43 = load_module(paths["m43_unpacker"], "m430_m43")
    weight_paths = [paths["weight_o{}".format(index)] for index in range(4)]
    flags, codec = m401.build_static_codec(
        catalog, weight_paths, args.output_dir / "static_codec_audit.csv")
    require(codec["blocks"] == 442368 and codec["lanes"] == 42467328 and
            codec["signed12_violations"] == 0 and
            codec["wide_reconstruction_mismatches"] == 0 and
            codec["narrow_reconstruction_mismatches"] == 0 and
            codec["nonzero_padding_bytes"] == 0,
            "M430 heldout static codec gate failure")

    marker = root / contract["one_shot"]["marker_path"]
    require(not marker.exists(), "M430 M40 one-shot marker already exists")
    marker.write_text(
        "M430B M40 one-shot consumed before first payload read.\n"
        "Catalog SHA256: {}\nAnalyzer SHA256: {}\n"
        "A failure must use an explicit recovery contract and cannot retune.\n".format(
            identities["m430_catalog"]["sha256"], source_start),
        encoding="utf-8")

    trace_dir = paths["m40_trace"].parent
    operators = tuple(trace["cohort"]["operators"])
    require(tuple(catalog["geometry"]["operators"]) == operators and
            len(operators) == 4, "M430 operator order drift")
    operator_index = {name: index for index, name in enumerate(operators)}
    histograms = defaultdict(Counter)
    payload_files = 0
    payload_bytes = 0
    for record_index, record in enumerate(trace["records"]):
        for key, sha_key in (("packed_file", "packed_file_sha256"),
                             ("value_payload_file", "value_payload_sha256")):
            path = trace_dir / record[key]
            require(path.is_file() and sha256(path) == record[sha_key],
                    "M430 heldout payload drift")
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
        print("[M430 HELDOUT HIST] {}/{}".format(
            record_index + 1, len(trace["records"])), flush=True)

    phases = defaultdict(list)
    aggregate = Counter()
    first_hit = Counter()
    cumulative_all = Counter()
    cumulative_eligible = Counter()
    phase_rows = []
    q16_prefix_mismatches = 0
    tail_pool_mismatches = 0
    for sample in range(10):
        for operator in range(4):
            for partition in range(432):
                counter = histograms[(sample, operator, partition)]
                require(sum(counter.values()) == 3000,
                        "M430 heldout phase extent drift")
                centers = [int(value, 16) for value in
                           catalog["operators"][operator]["partitions"]
                           [partition]["nested_patterns"]]
                old_centers = [int(value, 16) for value in
                               m338["operators"][operator]["partitions"]
                               [partition]["nested_patterns"]]
                q16_prefix_mismatches += int(centers[:16] != old_centers[:16])
                pool = set(old_centers[16:128])
                tail_pool_mismatches += sum(
                    center not in pool for center in centers[16:])
                phase, phase_first, phase_all, phase_eligible = (
                    m423.analyze_phase_q32(
                        counter, centers, flags[operator][partition], m401))
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
                    "exact_pwp_rows": phase_first[16] + phase_first[32],
                    "fallback_rows": phase["fallback_rows"],
                    "correction_ops_per_block":
                        phase["correction_ops_per_block"],
                    "used_pwp_patterns": phase["used_pwp_patterns"],
                    "used_center_runs": phase["used_center_runs"],
                    "early_matcher": phase["q32_early_matcher_cycles"],
                })
        print("[M430 HELDOUT PHASE] sample={}/10".format(sample + 1),
              flush=True)
    require(q16_prefix_mismatches == 0 and tail_pool_mismatches == 0,
            "M430 heldout q16/pool identity drift")
    require(len(phase_rows) == 17280 and
            aggregate["source_rows"] == 51840000 and
            aggregate["source_rows"] ==
            aggregate["zero_rows"] + aggregate["active_rows"] and
            aggregate["active_rows"] ==
            aggregate["pwp_rows"] + aggregate["fallback_rows"] and
            aggregate["pwp_rows"] == aggregate["exact_reconstruction_rows"] and
            aggregate["q32_early_matcher_cycles"] ==
            contract["execution_gates"]["expected_q16_early_matcher_cycles"],
            "M430 heldout population/exactness drift")

    model = contract["cycle_model"]
    baseline = sum(m401.baseline_sample(
        phases[sample], contract["decision_rule"]["dma_command_setup_cycles"],
        model) for sample in range(10))
    sample_results = [dual_sample(
        phases[sample], contract["decision_rule"]["dma_command_setup_cycles"],
        contract["decision_rule"]["descriptor_sram_latency_cycles"], model,
        capture_phase_timestamps=True) for sample in range(10)]
    candidate = sum(result["cycles"] for result in sample_results)
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
    require(baseline == contract["comparisons"]["strong_zero_cycles"] and
            maximum_slot <= model["tile_slot_bytes"] and
            components["tile1_dma_exposed"] == 0,
            "M430 heldout cycle/resource gate failure")
    target = contract["comparisons"]["m423_catalog_dual_diagnostic_cycles"]
    decision = ("GO_M430_DUALAWARE_CATALOG" if candidate < target else
                "NO_GO_RETAIN_M423_CATALOG_FOR_DUAL")

    write_csv(args.output_dir / "per_phase_heldout_dual_replay.csv", phase_rows,
              ["sample", "operator", "partition", "active_rows",
               "eligible_rows", "pwp_rows", "exact_pwp_rows",
               "fallback_rows", "correction_ops_per_block",
               "used_pwp_patterns", "used_center_runs", "early_matcher"])
    write_csv(args.output_dir / "dual_phase_timestamps.csv", timestamps,
              ["sample", "phase_index", "phase_start",
               "tile0_replay_start", "tile0_replay_end", "tile1_dma_end",
               "exposed_tile1_dma", "tile1_replay_start", "phase_end"])
    require(source_start == sha256(Path(__file__).resolve()),
            "M430 analyzer changed during one-shot replay")
    comparisons = {
        "strong_zero_cycles": baseline,
        "m401_serial_cycles": contract["comparisons"]["m401_serial_cycles"],
        "m338_catalog_dual_cycles":
            contract["comparisons"]["m338_catalog_dual_cycles"],
        "m423_catalog_serial_cycles":
            contract["comparisons"]["m423_serial_cycles"],
        "m423_catalog_dual_diagnostic_cycles": target,
        "m430_catalog_dual_cycles": candidate,
        "m430_speedup_vs_strong_zero": baseline / float(candidate),
        "m430_speedup_vs_m401_serial":
            contract["comparisons"]["m401_serial_cycles"] / float(candidate),
        "m430_speedup_vs_m338_dual":
            contract["comparisons"]["m338_catalog_dual_cycles"] /
            float(candidate),
        "m430_speedup_vs_m423_dual": target / float(candidate),
        "m430_cycles_saved_vs_m423_dual": target - candidate,
        "m430_fraction_saved_vs_m423_dual":
            (target - candidate) / float(target),
    }
    traffic = {
        "dram_model_bytes_per_cycle": model["dram_bytes_per_cycle"],
        "pwp_dram_physical_bytes":
            components["pwp_dram_physical_bytes"],
        "weight_dram_bytes": components["weight_dram_bytes"],
        "pwp_output_block_issues": components["pwp_output_block_issues"],
        "pwp_logical_onchip_read_bytes":
            components["pwp_logical_onchip_read_bytes"],
        "pwp_padded_signal_bytes": components["pwp_padded_signal_bytes"],
        "correction_output_block_issues":
            components["correction_output_block_issues"],
        "correction_onchip_read_bytes":
            components["correction_onchip_read_bytes"],
        "peak_dual_pwp_logical_bytes_per_cycle":
            model["dual_pwp_logical_bytes_per_issue"],
        "peak_dual_pwp_padded_signal_bytes_per_cycle":
            model["dual_pwp_padded_signal_bytes_per_issue"],
        "strong_zero_reference_source_bytes_per_cycle":
            model["correction_bytes_per_issue"],
        "resource_interpretation": "Dual co-read is a new 144 logical B/cycle (160 padded signal B/cycle) PWP source point versus the 96 B/cycle SHARED96 correction/strong-zero source. It is not a free same-bandwidth upgrade; report throughput-area/port and eventual power Pareto.",
    }
    result = {
        "schema": "m430b_h67_dualaware_q32_heldout_once_v1",
        "status": "PASS_M430B_ONE_COMPLETED_M40_HELDOUT_DUAL_REPLAY",
        "decision": decision,
        "identity": identities,
        "paper_identity": paper,
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "split_audit": {
            "catalog_population": "M73 disjoint DSEC train-only S32/18 sequences",
            "heldout_population": "M40 frozen S10 zurich_city_09_a",
            "catalog_double_sealed_before_first_m40_read": True,
            "m40_payload_read_attempts": 1,
            "completed_full_17280_phase_evaluations": 1,
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
        "comparisons": comparisons,
        "component_ledger": dict(components),
        "traffic_and_port_ledger": traffic,
        "execution_gates": {
            "input_or_payload_sha_mismatches": 0,
            "train_double_seal_mismatches": 0,
            "q16_prefix_mismatches": q16_prefix_mismatches,
            "tail_outside_m338_pool_mismatches": tail_pool_mismatches,
            "signed12_or_codec_mismatches": 0,
            "population_or_exact_reconstruction_mismatches": 0,
            "q16_early_matcher_cycle_mismatch": 0,
            "tile_slot_overflows": 0,
            "all_17280_phases_complete": True,
            "m40_payload_read_attempts": 1,
            "completed_heldout_evaluations": 1,
            "post_heldout_retuning": False,
            "persistent_old_psum_preserved": True,
            "seed_first_correction_fusion_used": False,
        },
        "admission": {
            "exact_arithmetic": True,
            "checkpoint_or_accuracy_changed": False,
            "accuracy_loss": False,
            "frozen_h67_trace_cycle_replay": True,
            "standalone_four_bottleneck_conv_cycles": True,
            "catalog_selected_for_dual_rtl": decision.startswith("GO"),
            "resource_normalized_speedup": False,
            "rtl_measured_speedup": False,
            "synopsys": False,
            "physical_sram_or_interconnect": False,
            "energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "output_files": {
            "static_codec_audit": "static_codec_audit.csv",
            "per_phase_heldout_dual_replay":
                "per_phase_heldout_dual_replay.csv",
            "dual_phase_timestamps": "dual_phase_timestamps.csv",
            "one_shot_marker": contract["one_shot"]["marker_path"],
        },
    }
    result_path = args.output_dir / "m430b_h67_dualaware_q32_heldout_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    manifest_path, seal_path = write_seal(args.output_dir, [
        "static_codec_audit.csv", "per_phase_heldout_dual_replay.csv",
        "dual_phase_timestamps.csv", result_path.name])
    print("M430B_PASS cycles={} strong={:.9f}x vs_m423_dual={:.9f}x saved={} decision={} seal={}".format(
        candidate, comparisons["m430_speedup_vs_strong_zero"],
        comparisons["m430_speedup_vs_m423_dual"],
        comparisons["m430_cycles_saved_vs_m423_dual"], decision,
        sha256(seal_path)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
