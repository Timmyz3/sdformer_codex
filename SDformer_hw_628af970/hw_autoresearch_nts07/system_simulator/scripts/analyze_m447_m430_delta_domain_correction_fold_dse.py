#!/usr/bin/env python3
"""Exact post-M430 K1/K2/K4 correction fold and delta-composer DSE.

This is a fixed architecture ablation over the already frozen M430 catalog.
It never changes a center, checkpoint, or row decision.  The fused variant
forms update_delta = PWP + signed_correction_fold before the persistent
accumulator performs new_psum = old_psum + update_delta.  It is therefore a
new charged pre-adder architecture, not the revoked M426 seed mux.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path


FOLDS = (1, 2, 4)
POPCOUNT = tuple(bin(value).count("1") for value in range(1 << 16))
EXPECTED = {
    "m430_contract": "261cb8fc3fec3d08570f55423da71188b3b8c17b5537f695309075d16f72c912",
    "m430_result": "6cf413e93d8159d9516ad048eaa26c741e49c2c9a3b330fb1d6dd20ba64dab2a",
    "m430_phase_csv": "0717e2c4ffd33cf95184df5acc2cb04751edbe42789f8b9d63ed5fbc6a20d006",
    "m430_seal": "462501b849f42f1a0690d2fe8dbe3dc226e83ae05dea86f7cb0396d60e9faf7e",
    "m430_catalog": "3ff522ff2296a021b005ca5733d846cc169560c125c8713c814b22a14d372f78",
    "m435_review": "be3e9106774d55f642d01285259b0e75886223240808367ee800439c78964c6d",
    "m435_seal": "616785a93468fc2f626d422a67f9d8bce0cc1450e4d9ace75ae1c69d9a7fcb34",
    "m40_trace": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m43_unpacker": "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3",
    "m427r3_review": "b62955b7130a4b18245a514ae68ad777ff2a4543b59c18c2ee91bc063805ba5d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


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


def read_reference_phases(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: int(value) for key, value in row.items()})
    require(len(rows) == 17280, "M447 reference phase extent drift")
    return rows


def count_runs(indices):
    ordered = sorted(indices)
    return (0 if not ordered else
            1 + sum(current != previous + 1
                    for previous, current in zip(ordered, ordered[1:])))


def analyze_phase(counter, centers):
    result = Counter()
    used = set()
    distance_histogram = Counter()
    for original, population_count in counter.items():
        original = int(original) & 0xffff
        pop = POPCOUNT[original]
        result["source_rows"] += population_count
        if original == 0:
            result["zero_rows"] += population_count
            continue
        result["active_rows"] += population_count
        distances = [POPCOUNT[original ^ center] for center in centers[:32]]
        best_distance = min(distances)
        best_index = distances.index(best_distance)
        use_pwp = 1 + best_distance < pop
        correction = best_distance if use_pwp else pop
        result["pwp_rows"] += population_count * int(use_pwp)
        result["fallback_rows"] += population_count * int(not use_pwp)
        result["positive_residual_pwp_rows"] += population_count * int(
            use_pwp and correction != 0)
        result["exact_pwp_rows"] += population_count * int(
            use_pwp and correction == 0)
        result["correction_source_terms"] += population_count * correction
        if use_pwp:
            used.add(best_index)
        distance_histogram[("pwp" if use_pwp else "fallback",
                            correction)] += population_count
        for fold in FOLDS:
            folded = (correction + fold - 1) // fold
            separate = int(use_pwp) + folded
            fused = (1 + ((max(0, correction - fold) + fold - 1) // fold)
                     if use_pwp else folded)
            # For PWP rows, one delta-composer issue always materializes the
            # base, even when the residual is empty.
            if use_pwp:
                fused = max(1, fused)
            result[f"k{fold}_separate_issues_per_block"] += (
                population_count * separate)
            result[f"k{fold}_fused_issues_per_block"] += (
                population_count * fused)
            result[f"k{fold}_folded_correction_descriptors_per_block"] += (
                population_count * folded)
            result[f"k{fold}_fused_pwp_correction_rows"] += (
                population_count * int(use_pwp and correction != 0))
    result["used_pwp_patterns"] = len(used)
    result["used_center_runs"] = count_runs(used)
    require(result["source_rows"] ==
            result["zero_rows"] + result["active_rows"] and
            result["active_rows"] ==
            result["pwp_rows"] + result["fallback_rows"] and
            result["pwp_rows"] ==
            result["positive_residual_pwp_rows"] + result["exact_pwp_rows"],
            "M447 phase population conservation failure")
    return dict(result), distance_histogram


def replay_sample(phases, fold, fused, model, command_setup, latency):
    time = 0
    components = Counter()
    maximum_slot = 0
    for phase in phases:
        config_data = math.ceil(
            model["elastic_config_bytes"] / model["dram_bytes_per_cycle"])
        time += config_data + command_setup + phase["early_matcher"] + 1
        components["config_data"] += config_data
        components["config_command"] += command_setup
        components["matcher"] += phase["early_matcher"]
        components["bitmap_seal"] += 1
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
                model["tile_slot_bytes"], "M447 tile slot overflow")
        require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                "M447 tile DMA alignment drift")
        tile_data = tile_bytes // model["dram_bytes_per_cycle"]
        tile_commands = 1 + phase["used_center_runs"]
        tile_dma = tile_data + tile_commands * command_setup
        mode = "fused" if fused else "separate"
        per_block = phase[f"k{fold}_{mode}_issues_per_block"]
        work = model["output_blocks_per_tile"] * per_block
        replay0 = work + latency
        replay1 = work + latency
        time += tile_dma
        tile0_end = time + replay0
        tile1_dma_end = time + tile_dma
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
        components["issued_contributions"] += 2 * work
    time += model["commit_cycles_per_sample"]
    components["commit"] += model["commit_cycles_per_sample"]
    return {"cycles": int(time), "components": dict(components),
            "maximum_slot_bytes": maximum_slot}


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M447 output overwrite")
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m447_m430_delta_domain_correction_fold_dse_contract_v1" and
            contract["status"] ==
            "FROZEN_FIXED_K_AXIS_BEFORE_M447_SECONDARY_TRACE_REPLAY",
            "M447 contract status drift")
    root = args.contract.resolve().parents[1]
    script_start = sha256(Path(__file__).resolve())
    paths = {
        "m430_contract": root / "contracts/m430b_h67_dualaware_q32_heldout_once_contract_r1_20260826.json",
        "m430_result": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/m430b_h67_dualaware_q32_heldout_r1.json",
        "m430_phase_csv": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/per_phase_heldout_dual_replay.csv",
        "m430_seal": root / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/SHA256SUMS.seal.sha256",
        "m430_catalog": root / "results/m430a_trainonly_dualaware_q32_catalog_r1_20260826/m430_trainonly_dualaware_q32_catalog_r1.json",
        "m435_review": root / "results/m435r3_m430_independent_hammer_r1_20260826/m435_m430_independent_hammer_review.json",
        "m435_seal": root / "results/m435r3_m430_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256",
        "m40_trace": root / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json",
        "m43_unpacker": root / "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py",
        "m427r3_review": root / "results/m427r3_m426_seed_fusion_semantic_addendum_r1_20260826/m427r3_m426_seed_fusion_semantic_addendum_review_r1.json",
        "docs359": root / "docs/359_DATE终局冻结_20260813.md",
    }
    identities = {}
    for name, path in paths.items():
        actual = sha256(path)
        require(actual == EXPECTED[name], "M447 frozen input SHA drift: " + name)
        identities[name] = {"path": str(path.relative_to(root)),
                            "sha256": actual}
    require(contract["inputs"]["analyzer"]["sha256"] == script_start,
            "M447 analyzer self identity drift")
    require(tuple(contract["fixed_architecture_axis"]["fold_k"]) == FOLDS,
            "M447 fold axis drift")

    m430_contract = strict_json(paths["m430_contract"])
    m430_result = strict_json(paths["m430_result"])
    catalog = strict_json(paths["m430_catalog"])
    review = strict_json(paths["m435_review"])
    semantic = strict_json(paths["m427r3_review"])
    trace = strict_json(paths["m40_trace"])
    require(m430_result["status"] ==
            "PASS_M430B_ONE_COMPLETED_M40_HELDOUT_DUAL_REPLAY" and
            review["severity_counts"]["P0"] == 0 and
            semantic["verdict"]["m426_seed_fusion_1p695794x"] ==
            "REVOKED_DO_NOT_CITE",
            "M447 upstream admission drift")
    reference_phases = read_reference_phases(paths["m430_phase_csv"])
    m43 = load_module(paths["m43_unpacker"], "m447_m43")
    operators = tuple(trace["cohort"]["operators"])
    operator_index = {name: index for index, name in enumerate(operators)}
    trace_dir = paths["m40_trace"].parent
    histograms = defaultdict(Counter)
    payload_files = payload_bytes = 0
    for record_index, record in enumerate(trace["records"]):
        for key, sha_key in (("packed_file", "packed_file_sha256"),
                             ("value_payload_file", "value_payload_sha256")):
            path = trace_dir / record[key]
            require(path.is_file() and sha256(path) == record[sha_key],
                    "M447 payload identity drift")
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
        print(f"[M447 HIST] {record_index+1}/{len(trace['records'])}",
              flush=True)

    phases = defaultdict(list)
    aggregate = Counter()
    distance_histogram = Counter()
    phase_audit_rows = []
    reference_mismatches = 0
    phase_index = 0
    for sample in range(10):
        for operator in range(4):
            for partition in range(432):
                counter = histograms[(sample, operator, partition)]
                require(sum(counter.values()) == 3000,
                        "M447 phase row extent drift")
                centers = [int(value, 16) for value in
                           catalog["operators"][operator]["partitions"]
                           [partition]["nested_patterns"][:32]]
                phase, hist = analyze_phase(counter, centers)
                reference = reference_phases[phase_index]
                reference_mismatches += sum((
                    reference["sample"] != sample,
                    reference["operator"] != operator,
                    reference["partition"] != partition,
                    reference["active_rows"] != phase["active_rows"],
                    reference["pwp_rows"] != phase["pwp_rows"],
                    reference["fallback_rows"] != phase["fallback_rows"],
                    reference["correction_ops_per_block"] !=
                        phase["correction_source_terms"],
                    reference["used_pwp_patterns"] !=
                        phase["used_pwp_patterns"],
                    reference["used_center_runs"] !=
                        phase["used_center_runs"],
                ))
                phase["early_matcher"] = reference["early_matcher"]
                phases[sample].append(phase)
                aggregate.update(phase)
                distance_histogram.update(hist)
                phase_audit_rows.append({
                    "sample": sample, "operator": operator,
                    "partition": partition,
                    "pwp_rows": phase["pwp_rows"],
                    "positive_residual_pwp_rows":
                        phase["positive_residual_pwp_rows"],
                    "correction_source_terms":
                        phase["correction_source_terms"],
                    **{f"k{fold}_separate_issues_per_block":
                       phase[f"k{fold}_separate_issues_per_block"]
                       for fold in FOLDS},
                    **{f"k{fold}_fused_issues_per_block":
                       phase[f"k{fold}_fused_issues_per_block"]
                       for fold in FOLDS},
                })
                phase_index += 1
        print(f"[M447 PHASE] sample={sample+1}/10", flush=True)
    require(reference_mismatches == 0 and phase_index == 17280 and
            aggregate["source_rows"] == 51840000 and
            aggregate["correction_source_terms"] ==
            m430_result["runtime_population"]["correction_ops_per_block"] and
            aggregate["pwp_rows"] ==
            m430_result["runtime_population"]["pwp_rows"],
            "M447 M430 exact crosscheck failed")

    model = m430_contract["cycle_model"]
    command_setup = m430_contract["decision_rule"]["dma_command_setup_cycles"]
    latency = m430_contract["decision_rule"]["descriptor_sram_latency_cycles"]
    points = []
    component_ledgers = {}
    for fold in FOLDS:
        for fused in (False, True):
            samples = [replay_sample(phases[sample], fold, fused, model,
                                     command_setup, latency)
                       for sample in range(10)]
            cycles = sum(row["cycles"] for row in samples)
            components = Counter()
            for row in samples:
                components.update(row["components"])
            mode = "fused_delta_composer" if fused else "separate_fold"
            name = f"k{fold}_{mode}"
            component_ledgers[name] = dict(components)
            peak_pwp_plus_correction = (
                model["dual_pwp_padded_signal_bytes_per_issue"] +
                fold * model["correction_bytes_per_issue"])
            point = {
                "name": name, "fold_k": fold,
                "delta_domain_pwp_correction_fusion": fused,
                "cycles": cycles,
                "speedup_vs_strong_zero":
                    m430_result["comparisons"]["strong_zero_cycles"] / cycles,
                "speedup_vs_m430":
                    m430_result["comparisons"]["m430_catalog_dual_cycles"] / cycles,
                "cycle_reduction_vs_m430":
                    m430_result["comparisons"]["m430_catalog_dual_cycles"] - cycles,
                "peak_correction_input_bytes_per_cycle":
                    fold * model["correction_bytes_per_issue"],
                "peak_fused_input_bytes_per_cycle":
                    peak_pwp_plus_correction if fused else None,
                "correction_source_bytes_total":
                    aggregate["correction_source_terms"] * 8 *
                    model["correction_bytes_per_issue"],
                "folded_correction_output_block_descriptors":
                    aggregate[f"k{fold}_folded_correction_descriptors_per_block"] * 8,
                "positive_residual_pwp_rows_per_block":
                    aggregate["positive_residual_pwp_rows"],
                "rtl": False, "dc": False, "formality": False,
                "pt": False, "sram_or_interconnect": False,
                "resource_normalized": False,
                "rtl_measured_speedup": False,
                "system_speedup": False,
                "date_headline": False,
            }
            points.append(point)

    k1_separate = next(row for row in points
                       if row["name"] == "k1_separate_fold")
    require(k1_separate["cycles"] ==
            m430_result["comparisons"]["m430_catalog_dual_cycles"],
            "M447 K1 separate recurrence mismatch")
    strong = m430_result["comparisons"]["strong_zero_cycles"]
    ideal = replay_sample(
        [{**phase, "k4_fused_issues_per_block": phase["pwp_rows"]}
         for phase in phases[0]], 4, True, model, command_setup, latency)
    # Full ideal correction removal is recomputed sample-wise below; this is
    # an explicit arithmetic ceiling and never an architectural point.
    ideal_cycles = 0
    for sample in range(10):
        ideal_phases = []
        for phase in phases[sample]:
            copied = dict(phase)
            copied["k4_fused_issues_per_block"] = phase["pwp_rows"]
            ideal_phases.append(copied)
        ideal_cycles += replay_sample(ideal_phases, 4, True, model,
                                      command_setup, latency)["cycles"]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(phase_audit_rows[0].keys())
    write_csv(args.output_dir / "m447_phase_fold_audit.csv",
              phase_audit_rows, fields)
    distance_rows = [
        {"path": path, "correction_distance": distance,
         "rows": distance_histogram[(path, distance)]}
        for path in ("pwp", "fallback") for distance in range(17)
    ]
    write_csv(args.output_dir / "m447_correction_distance_histogram.csv",
              distance_rows, ["path", "correction_distance", "rows"])
    result = {
        "schema": "m447_m430_delta_domain_correction_fold_dse_v1",
        "status": "PASS_M447_EXACT_FIXED_K_OPPORTUNITY_DSE",
        "identity": {"analyzer": {"path": str(Path(__file__).resolve().relative_to(root)),
                                    "sha256": script_start}, **identities},
        "paper_identity": m430_result["paper_identity"],
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "secondary_replay_contract": {
            "catalog_or_checkpoint_changed": False,
            "post_m40_catalog_retuning": False,
            "fixed_fold_axis_before_replay": list(FOLDS),
            "purpose": "hardware resource/cycle ablation after M430 admission",
            "m40_payload_read_attempts_this_milestone": 1,
            "completed_17280_phase_replays_this_milestone": 1,
        },
        "population": {
            "source_rows": aggregate["source_rows"],
            "active_rows": aggregate["active_rows"],
            "pwp_rows_per_block": aggregate["pwp_rows"],
            "positive_residual_pwp_rows_per_block":
                aggregate["positive_residual_pwp_rows"],
            "exact_pwp_rows_per_block": aggregate["exact_pwp_rows"],
            "fallback_rows_per_block": aggregate["fallback_rows"],
            "correction_source_terms_per_block":
                aggregate["correction_source_terms"],
            "correction_output_block_source_terms":
                aggregate["correction_source_terms"] * 8,
        },
        "points": points,
        "component_ledgers": component_ledgers,
        "ideal_correction_elimination_ceiling": {
            "cycles": ideal_cycles,
            "speedup_vs_strong_zero": strong / ideal_cycles,
            "architectural_point": False,
        },
        "semantic_contract": {
            "revoked_m426": "new_psum=PWP+correction (drops old_psum)",
            "m447_new_architecture":
                "update_delta=PWP+signed_correction_fold; new_psum=old_psum+update_delta",
            "persistent_old_psum_preserved": True,
            "integer_addition_exact_without_saturation_or_rounding": True,
            "maximum_static_pwp_absolute":
                m430_result["static_codec"]["maximum_absolute"],
            "k4_correction_absolute_bound": 512,
            "k4_fused_delta_safe_signed_bits": 13,
        },
        "traffic_and_resource_boundary": {
            "baseline_strong_zero_correction_bytes_per_cycle": 96,
            "m433_pwp_physical_bytes_per_cycle": 160,
            "total_correction_source_bytes_unchanged": True,
            "peak_input_width_scales_with_k": True,
            "pre_adder_and_fold_arithmetic_new_and_uncharged": True,
            "accumulator_backend_new_and_uncharged": False,
            "sram_ports_interconnect_area_frequency_power_unmeasured": True,
        },
        "execution_gates": {
            "payload_files_rehashed": payload_files,
            "payload_bytes_rehashed": payload_bytes,
            "payload_sha_mismatches": 0,
            "m430_phase_crosscheck_mismatches": reference_mismatches,
            "all_phases": phase_index == 17280,
            "k1_separate_exactly_reproduces_m430": True,
        },
        "decision": {
            "opportunity_dse": "GO",
            "next": "Implement only Pareto-relevant K points as new delta-domain composer RTL, then VCS/DC/Formality/PT and physical port pricing.",
            "cycle_speedup_admitted": False,
            "resource_normalized_speedup": False,
            "paper_or_system_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    result_path = args.output_dir / "m447_m430_delta_domain_correction_fold_dse_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(sha256(Path(__file__).resolve()) == script_start,
            "M447 analyzer changed during replay")
    manifest = args.output_dir / "SHA256SUMS"
    names = ["m447_phase_fold_audit.csv",
             "m447_correction_distance_histogram.csv", result_path.name]
    manifest.write_text("".join(
        f"{sha256(args.output_dir / name)}  {name}\n" for name in names),
        encoding="utf-8")
    seal = args.output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text(f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")
    best = min(points, key=lambda row: row["cycles"])
    print("PASS_M447_EXACT_FIXED_K_OPPORTUNITY_DSE "
          f"m430={k1_separate['cycles']} best={best['name']} "
          f"cycles={best['cycles']} strong_speedup={best['speedup_vs_strong_zero']:.9f}x "
          "resource_normalized=false rtl=false system_speedup=false headline=false",
          flush=True)


if __name__ == "__main__":
    main()
