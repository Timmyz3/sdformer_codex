#!/usr/bin/env python3
"""Build an exact q32 catalog from M73 train-only data using issue cost.

The admitted M77 q16 prefix is immutable.  Only IDs 16..31 are selected.
No M40/validation/runtime input is accepted by this program.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


K = 16
PARTITIONS = 432
SAMPLES = 32
TAIL = 16
POPCOUNT = np.asarray([bin(value).count("1")
                       for value in range(1 << K)], dtype=np.uint8)
OPTION_ORDER = (
    "m338_q32",
    "m338_anchored_lloyd_issue_best",
    "issue_gain_q32",
    "issue_gain_anchored_lloyd_issue_best",
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
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def load_module(path):
    spec = importlib.util.spec_from_file_location("m423_m43", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M43 unpacker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_runs(indices):
    ordered = sorted(indices)
    if not ordered:
        return 0
    return 1 + sum(current != previous + 1
                   for previous, current in zip(ordered, ordered[1:]))


def collect_histograms(m43, manifest, manifest_path, operators):
    histograms = defaultdict(Counter)
    operator_index = {name: index for index, name in enumerate(operators)}
    seen = Counter()
    payload_files = 0
    payload_bytes = 0
    for record_index, record in enumerate(manifest["records"]):
        for key, size_key, sha_key in (
                ("packed_file", "packed_file_bytes", "packed_file_sha256"),
                ("value_payload_file", "value_payload_compressed_bytes",
                 "value_payload_sha256")):
            path = manifest_path.parent / record[key]
            require(path.is_file() and path.stat().st_size == record[size_key]
                    and sha256(path) == record[sha_key],
                    "M423 train payload identity drift")
            payload_files += 1
            payload_bytes += path.stat().st_size
        sample = int(record["sample_id"])
        operator = record["operator"]
        require(0 <= sample < SAMPLES and operator in operator_index,
                "M423 sample/operator extent drift")
        seen[(sample, operator)] += 1
        masks = m43.unpack_record_masks(manifest_path.parent, record)
        for source_row in range(m43.ROWS):
            base = source_row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // K)
                for subtile in range(m43.TILE_BITS // K):
                    value = (value256 >> (subtile * K)) & 0xffff
                    histograms[(sample, operator_index[operator],
                                partition_base + subtile)][value] += 1
        print("[M423 HIST] {}/{} sample={} op={}".format(
            record_index + 1, len(manifest["records"]), sample,
            operator_index[operator]), flush=True)
    require(len(seen) == SAMPLES * len(operators) and
            all(value == 1 for value in seen.values()),
            "M423 train sample/operator uniqueness failure")
    require(len(histograms) == SAMPLES * len(operators) * PARTITIONS,
            "M423 train histogram extent failure")
    return histograms, payload_files, payload_bytes


def distances(values, centers):
    center_array = np.asarray(centers, dtype=np.uint16)
    return POPCOUNT[np.bitwise_xor(center_array[:, None],
                                   values[None, :])]


def issue_units(values, centers):
    pops = POPCOUNT[values]
    best = distances(values, centers).min(axis=0)
    # This is the exact pre-elastic work per output tile divided by four:
    # fallback=popcount, PWP=two issue slots plus signed corrections.
    return np.where(1 + best < pops,
                    2 + best.astype(np.int16),
                    pops.astype(np.int16))


def issue_objective(values, counts, centers):
    return int(np.dot(counts, issue_units(values, centers).astype(np.int64)))


def issue_gain_initial(values, counts, q16, pool):
    base = issue_units(values, q16).astype(np.int16)
    q16_best = distances(values, q16).min(axis=0).astype(np.int16)
    candidate_distance = distances(values, pool).astype(np.int16)
    pops = POPCOUNT[values].astype(np.int16)
    gain_rows = []
    for index, center in enumerate(pool):
        best = np.minimum(q16_best, candidate_distance[index])
        candidate = np.where(1 + best < pops, 2 + best, pops)
        gain = int(np.dot(counts, (base - candidate).astype(np.int64)))
        exact_count = int(counts[values == center].sum())
        gain_rows.append((gain, exact_count, -int(center), int(center)))
    gain_rows.sort(reverse=True)
    selected = [row[-1] for row in gain_rows[:TAIL]]
    require(len(selected) == TAIL and len(set(selected)) == TAIL,
            "M423 issue-gain initialization extent failure")
    return selected


def anchored_lloyd_issue_best(values, counts, q16, tail, iterations):
    centers = list(q16) + list(tail)
    require(len(centers) == 32 and len(set(centers)) == 32,
            "M423 anchored Lloyd initial extent failure")
    best = list(centers)
    best_issue = issue_objective(values, counts, best)
    best_iteration = 0
    completed = 0
    for iteration in range(iterations):
        matrix = distances(values, centers)
        assignment = matrix.argmin(axis=0)
        updated = list(q16)
        used = set(q16)
        for center_id in range(16, 32):
            member = assignment == center_id
            proposed = centers[center_id]
            if np.any(member):
                member_values = values[member]
                member_counts = counts[member]
                total = int(member_counts.sum())
                majority = 0
                for bit in range(K):
                    ones = int(member_counts[
                        (member_values & (1 << bit)) != 0].sum())
                    if 2 * ones > total:
                        majority |= 1 << bit
                if majority != 0 and int(POPCOUNT[majority]) >= 2:
                    proposed = majority
            if proposed == 0 or int(POPCOUNT[proposed]) < 2 or proposed in used:
                proposed = centers[center_id]
            if proposed in used:
                # Deterministic fail-safe; the original tail is unique.
                proposed = next(value for value in tail if value not in used)
            updated.append(int(proposed))
            used.add(int(proposed))
        require(updated[:16] == list(q16) and len(set(updated)) == 32,
                "M423 anchored Lloyd prefix/uniqueness failure")
        completed = iteration + 1
        issue = issue_objective(values, counts, updated)
        if (issue, tuple(updated[16:])) < (best_issue, tuple(best[16:])):
            best = list(updated)
            best_issue = issue
            best_iteration = completed
        if updated == centers:
            break
        centers = updated
    return best, best_issue, best_iteration, completed


def narrow_flags(weight_slice, centers):
    bits = np.asarray([[(center >> bit) & 1 for bit in range(K)]
                       for center in centers], dtype=np.int16)
    products = bits @ weight_slice
    require(products.shape == (32, 768),
            "M423 PWP product shape drift")
    minimum = int(products.min())
    maximum = int(products.max())
    require(minimum >= -2048 and maximum <= 2047,
            "M423 signed12 PWP overflow")
    blocks = products.reshape(32, 8, 96)
    narrow = ((blocks.min(axis=2) >= -128) &
              (blocks.max(axis=2) <= 127))
    return narrow, minimum, maximum


def evaluate_partition(sample_counters, centers, weight_slice, model):
    union = sorted(set().union(*(counter.keys()
                                for counter in sample_counters)))
    values = np.asarray(union, dtype=np.uint16)
    counts = np.zeros((SAMPLES, len(union)), dtype=np.int64)
    lookup = {value: index for index, value in enumerate(union)}
    for sample, counter in enumerate(sample_counters):
        for value, count in counter.items():
            counts[sample, lookup[value]] = count
    require(np.all(counts.sum(axis=1) == model["rows_per_phase"]),
            "M423 phase row extent drift")
    pops = POPCOUNT[values].astype(np.int16)
    matrix = distances(values, centers)
    best_id = matrix.argmin(axis=0)
    best_distance = matrix[best_id, np.arange(len(values))].astype(np.int16)
    pwp = (values != 0) & (1 + best_distance < pops)
    correction = np.where(pwp, best_distance, pops).astype(np.int64)
    q16_exact = distances(values, centers[:16]).min(axis=0) == 0
    eligible = pops >= 2
    narrow, pwp_minimum, pwp_maximum = narrow_flags(weight_slice, centers)
    narrow0_per_value = narrow[best_id, 0:4].sum(axis=1).astype(np.int64)
    narrow1_per_value = narrow[best_id, 4:8].sum(axis=1).astype(np.int64)
    total = Counter()
    cycles = 0
    for sample in range(SAMPLES):
        row_counts = counts[sample]
        active = int(row_counts[values != 0].sum())
        zero = int(row_counts[values == 0].sum())
        eligible_rows = int(row_counts[eligible].sum())
        extra = int(row_counts[eligible & (~q16_exact)].sum())
        matcher = model["rows_per_phase"] + extra + 2
        config = (int(math.ceil(model["elastic_config_bytes"] /
                                float(model["dram_bytes_per_cycle"]))) +
                  model["dma_command_setup_cycles"])
        phase_cycles = config + matcher + 1
        pwp_rows = int(row_counts[pwp].sum())
        corr = int(np.dot(row_counts, correction))
        bit_sparse = int(np.dot(row_counts, pops.astype(np.int64)))
        used = set(int(index) for index in np.unique(best_id[
            pwp & (row_counts > 0)]))
        runs = count_runs(used)
        narrow0 = int(np.dot(row_counts[pwp],
                             narrow0_per_value[pwp]))
        narrow1 = int(np.dot(row_counts[pwp],
                             narrow1_per_value[pwp]))
        if active == 0:
            phase_cycles += model["tail_cycles"]
        else:
            tile_bytes = (model["weight_bytes_per_tile"] +
                          len(used) * model["elastic_center_stride_bytes"])
            require(model["elastic_config_bytes"] + tile_bytes <=
                    model["tile_slot_bytes"], "M423 tile slot overflow")
            require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                    "M423 unaligned tile DMA")
            tile_dma = (tile_bytes // model["dram_bytes_per_cycle"] +
                        (1 + runs) * model["dma_command_setup_cycles"])
            work = 4 * corr + 8 * pwp_rows
            replay0 = work - narrow0 + model["descriptor_sram_latency_cycles"]
            replay1 = work - narrow1 + model["descriptor_sram_latency_cycles"]
            require(replay0 >= active and replay1 >= active,
                    "M423 descriptor service underflow")
            phase_cycles += (tile_dma + max(replay0, tile_dma) + replay1 +
                             model["tail_cycles"])
        cycles += phase_cycles
        total.update({
            "source_rows": model["rows_per_phase"],
            "zero_rows": zero,
            "active_rows": active,
            "eligible_rows": eligible_rows,
            "q32_early_extra_prefix_tasks": extra,
            "q32_early_matcher_cycles": matcher,
            "bit_sparse_vector_ops_per_block": bit_sparse,
            "pwp_rows": pwp_rows,
            "fallback_rows": active - pwp_rows,
            "correction_ops_per_block": corr,
            "candidate_vector_ops_per_block": corr + pwp_rows,
            "used_pwp_patterns": len(used),
            "used_center_runs": runs,
            "narrow_block_descriptors_tile0": narrow0,
            "narrow_block_descriptors_tile1": narrow1,
        })
    return cycles, total, pwp_minimum, pwp_maximum


def baseline_sample(bit_sparse_phases, model):
    preprocess = max(
        model["rows_per_phase"] + model["popcount_filter_pipeline_cycles"],
        model["weight_phase_bytes"] // model["dram_bytes_per_cycle"] +
        model["dma_command_setup_cycles"])
    time = preprocess
    for index, bit_sparse in enumerate(bit_sparse_phases):
        compute = bit_sparse * 8
        next_preprocess = preprocess if index + 1 < len(bit_sparse_phases) else 0
        time += max(compute, next_preprocess) + model["tail_cycles"]
    return time + model["commit_cycles_per_sample"]


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
    require(not args.output_dir.exists(), "refusing M423a output overwrite")
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m423a_trainonly_cycle_aware_q32_catalog_contract_v1" and
            contract.get("status") == "FROZEN_BEFORE_M423A_EXECUTION",
            "M423a contract drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M423a input SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": spec["path"],
                            "sha256": spec["sha256"]}
    require(paths["builder"].resolve() == Path(__file__).resolve() and
            identities["builder"]["sha256"] == source_start,
            "M423a builder self-identity drift")

    manifest = strict_json(paths["m73_train_trace_manifest"])
    parent = strict_json(paths["m77_q16_catalog"])
    admission = strict_json(paths["m77_q16_admission"])
    m338 = strict_json(paths["m338_q128_catalog"])
    require(manifest["status"] ==
            "PASS_M73_DSEC_TRAIN_ONLY_S32_ALL18_SEQUENCES_EXACT_H67_EP35_FOUR_BOTTLENECK_TRACE" and
            manifest["split_audit"]["role"] ==
            "DSEC_TRAIN_ONLY_PAFT_CALIBRATION" and
            manifest["split_audit"]["full_train_valid825_key_overlap"] == 0 and
            manifest["split_audit"]["selected_valid825_key_overlap"] == 0 and
            manifest["split_audit"]["selected_samples"] == SAMPLES and
            manifest["split_audit"]["selected_sequences"] == 18,
            "M423a train/held-out isolation drift")
    require(admission["train_only_admitted"] is True and
            admission["catalog_sha256"] ==
            identities["m77_q16_catalog"]["sha256"] and
            admission["train_valid825_key_overlap"] == 0,
            "M423a parent q16 admission drift")
    require(m338["split"]["test_or_validation_data_used"] is False and
            m338["split"]["train_valid825_key_overlap"] == 0,
            "M423a M338 train-only identity drift")
    operators = tuple(manifest["cohort"]["operators"])
    require(len(operators) == 4 and
            [row["operator"] for row in parent["operators"]] == list(operators) and
            [row["operator"] for row in m338["operators"]] == list(operators),
            "M423a operator order drift")

    model = dict(contract["cycle_model"])
    method = contract["frozen_method"]
    require(method["q16_prefix"] == "M77_BIT_IDENTICAL" and
            method["tail_entries"] == TAIL and
            method["candidate_pool"] == "M338_Q128_IDS_16_TO_127" and
            method["anchored_lloyd_max_iterations"] == 12 and
            method["random_seed"] is None,
            "M423a frozen method drift")
    m43 = load_module(paths["m43_unpacker"])
    histograms, payload_files, payload_bytes = collect_histograms(
        m43, manifest, paths["m73_train_trace_manifest"], operators)
    weights = [np.fromfile(paths["weight_o{}".format(op)], dtype=np.int8)
               .reshape(6912, 768).astype(np.int16) for op in range(4)]

    option_cycles = Counter()
    option_totals = {name: Counter() for name in OPTION_ORDER}
    selected_totals = Counter()
    selected_cycles = 0
    selected_counts = Counter()
    rows = []
    operator_payloads = []
    baseline_by_sample = [[] for _ in range(SAMPLES)]
    global_minimum = 1 << 30
    global_maximum = -(1 << 30)
    for op, operator in enumerate(operators):
        partition_payloads = []
        for partition in range(PARTITIONS):
            sample_counters = [histograms[(sample, op, partition)]
                               for sample in range(SAMPLES)]
            aggregate_counter = Counter()
            for counter in sample_counters:
                aggregate_counter.update(counter)
            union = sorted(aggregate_counter)
            values = np.asarray(union, dtype=np.uint16)
            counts = np.asarray([aggregate_counter[value] for value in union],
                                dtype=np.int64)
            q16 = [int(item["value_hex"], 16) for item in
                   parent["operators"][op]["partitions"][partition]["patterns"]]
            nested = [int(value, 16) for value in
                      m338["operators"][op]["partitions"][partition]
                      ["nested_patterns"]]
            require(len(q16) == 16 and nested[:16] == q16 and
                    len(nested) >= 128 and len(set(nested[:128])) == 128,
                    "M423a q16/q128 prefix drift")
            pool = nested[16:128]
            issue_tail = issue_gain_initial(values, counts, q16, pool)
            m338_lloyd, m338_issue, m338_best_iter, m338_done = (
                anchored_lloyd_issue_best(
                    values, counts, q16, nested[16:32],
                    method["anchored_lloyd_max_iterations"]))
            issue_lloyd, issue_best, issue_best_iter, issue_done = (
                anchored_lloyd_issue_best(
                    values, counts, q16, issue_tail,
                    method["anchored_lloyd_max_iterations"]))
            candidates = {
                "m338_q32": q16 + nested[16:32],
                "m338_anchored_lloyd_issue_best": m338_lloyd,
                "issue_gain_q32": q16 + issue_tail,
                "issue_gain_anchored_lloyd_issue_best": issue_lloyd,
            }
            evaluations = {}
            weight_slice = weights[op][partition * 16:(partition + 1) * 16]
            for option in OPTION_ORDER:
                centers = candidates[option]
                require(centers[:16] == q16 and len(centers) == 32 and
                        len(set(centers)) == 32,
                        "M423a candidate prefix/extent drift")
                cycles, metrics, minimum, maximum = evaluate_partition(
                    sample_counters, centers, weight_slice, model)
                evaluations[option] = (cycles, metrics, minimum, maximum)
                option_cycles[option] += cycles
                option_totals[option].update(metrics)
            chosen = min(OPTION_ORDER,
                         key=lambda name: (evaluations[name][0],
                                           OPTION_ORDER.index(name)))
            chosen_cycles, chosen_metrics, minimum, maximum = evaluations[chosen]
            selected_cycles += chosen_cycles
            selected_totals.update(chosen_metrics)
            selected_counts[chosen] += 1
            global_minimum = min(global_minimum, minimum)
            global_maximum = max(global_maximum, maximum)
            for sample in range(SAMPLES):
                baseline_by_sample[sample].append(int(sum(
                    count * int(POPCOUNT[value])
                    for value, count in sample_counters[sample].items())))
            partition_payloads.append({
                "partition": partition,
                "nested_patterns": ["{:04x}".format(value)
                                    for value in candidates[chosen]],
                "selected_train_option": chosen,
                "train_phase_cycles_by_option": {
                    name: evaluations[name][0] for name in OPTION_ORDER},
                "issue_objective_by_option": {
                    "m338_q32": issue_objective(
                        values, counts, candidates["m338_q32"]),
                    "m338_anchored_lloyd_issue_best": m338_issue,
                    "issue_gain_q32": issue_objective(
                        values, counts, candidates["issue_gain_q32"]),
                    "issue_gain_anchored_lloyd_issue_best": issue_best,
                },
                "lloyd_audit": {
                    "m338_best_iteration": m338_best_iter,
                    "m338_iterations_completed": m338_done,
                    "issue_gain_best_iteration": issue_best_iter,
                    "issue_gain_iterations_completed": issue_done,
                },
            })
            row = {"operator": op, "partition": partition,
                   "selected_option": chosen,
                   "selected_train_phase_cycles": chosen_cycles}
            for name in OPTION_ORDER:
                row[name + "_cycles"] = evaluations[name][0]
            rows.append(row)
            if (partition + 1) % 108 == 0:
                print("[M423 OPT] operator={}/4 partition={}/432".format(
                    op + 1, partition + 1), flush=True)
        operator_payloads.append({"operator": operator,
                                  "partitions": partition_payloads})

    commit = SAMPLES * model["commit_cycles_per_sample"]
    selected_cycles += commit
    for option in OPTION_ORDER:
        option_cycles[option] += commit
    strong_baseline = sum(baseline_sample(baseline_by_sample[sample], model)
                          for sample in range(SAMPLES))
    require(all(len(row) == 4 * PARTITIONS for row in baseline_by_sample),
            "M423a baseline phase extent failure")
    require(selected_totals["source_rows"] ==
            SAMPLES * 4 * PARTITIONS * model["rows_per_phase"] and
            selected_totals["source_rows"] ==
            selected_totals["zero_rows"] + selected_totals["active_rows"] and
            selected_totals["active_rows"] ==
            selected_totals["pwp_rows"] + selected_totals["fallback_rows"],
            "M423a selected population conservation failure")
    require(source_start == sha256(Path(__file__).resolve()),
            "M423a builder changed during execution")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    catalog = {
        "schema": "m423_trainonly_cycle_aware_q32_catalog_v1",
        "status": "PASS_M423A_TRAIN_ONLY_Q32_CATALOG_FROZEN_BEFORE_HELDOUT",
        "identity": identities,
        "split": {
            "role": "DSEC_TRAIN_ONLY_HARDWARE_CYCLE_AWARE_CATALOG",
            "selected_train_samples": SAMPLES,
            "selected_train_sequences": 18,
            "train_valid825_key_overlap": 0,
            "runtime_or_validation_data_used": False,
        },
        "algorithm": {
            "q16_prefix": "bit-identical admitted M77 q16, including order",
            "tail": "16 entries selected only from deterministic M338 q128 train pool, with anchored binary-Lloyd refinements",
            "objective": "minimum exact M401 combined cmd32/L8 train phase cycles among four frozen candidates per partition",
            "surrogate": "fallback popcount versus PWP two issue slots plus signed Hamming corrections",
            "candidate_order": list(OPTION_ORDER),
            "anchored_lloyd_max_iterations": 12,
            "random_seed": None,
            "runtime_arithmetic": "exact W*x=PWP[p]+signed W*(x-p), with exact bit-sparse fallback",
            "accuracy_loss": False,
        },
        "geometry": {
            "partition_bits": 16,
            "partitions_per_operator": PARTITIONS,
            "operators": list(operators),
            "q_capacity": 32,
            "output_blocks": 8,
            "shared_lanes": 96,
            "pwp_stride_bytes": 640,
            "tile_slot_bytes": model["tile_slot_bytes"],
        },
        "operators": operator_payloads,
        "admission": {
            "train_only_catalog": True,
            "q16_parent_bit_identical": True,
            "exact_arithmetic_identity": True,
            "checkpoint_or_accuracy_changed": False,
            "heldout_runtime_evaluated": False,
            "cycle_speedup": False,
            "energy": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "claim_boundary": "Frozen train-only exact q32 catalog. The four H67 bottleneck Conv held-out cycle result remains false until one subsequent M40 execution; no accuracy, energy, system, physical-PPA or headline claim.",
    }
    catalog_path = args.output_dir / "m423_trainonly_cycle_aware_q32_catalog_r1.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    audit = {
        "schema": "m423a_trainonly_cycle_aware_q32_catalog_audit_v1",
        "status": "PASS_M423A_TRAIN_ONLY_CATALOG_FROZEN_HELDOUT_NOT_RUN",
        "identity": identities,
        "payload_audit": {"files_rehashed": payload_files,
                          "bytes_rehashed": payload_bytes,
                          "mismatches": 0},
        "frozen_method": method,
        "cycle_model": model,
        "train_only_observation": {
            "strong_zero_elided_baseline_cycles": strong_baseline,
            "candidate_cycles_by_fixed_global_option": dict(option_cycles),
            "hybrid_selected_cycles": selected_cycles,
            "hybrid_speedup_vs_train_strong_baseline":
                strong_baseline / float(selected_cycles),
            "hybrid_improvement_vs_m338_q32_cycles":
                (option_cycles["m338_q32"] - selected_cycles),
            "hybrid_improvement_vs_m338_q32_fraction":
                ((option_cycles["m338_q32"] - selected_cycles) /
                 float(option_cycles["m338_q32"])),
            "selected_partition_counts": dict(selected_counts),
            "selected_population": dict(selected_totals),
            "selected_static_pwp_minimum": global_minimum,
            "selected_static_pwp_maximum": global_maximum,
            "selected_static_signed12_overflow": 0,
        },
        "exactness": {
            "q16_prefix_mismatches": 0,
            "population_conservation_mismatches": 0,
            "arithmetic_identity": "W*x=PWP[p]+W*(x-p), fallback=W*x",
            "checkpoint_changed": False,
            "accuracy_loss": False,
        },
        "heldout_gate": {
            "m40_executions_so_far": 0,
            "catalog_must_be_sealed_before_m40": True,
            "tuning_after_m40": False,
        },
        "admission": catalog["admission"],
    }
    audit_path = args.output_dir / "m423a_trainonly_cycle_aware_q32_catalog_audit_r1.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    write_csv(args.output_dir / "per_partition_train_option_cycles.csv", rows,
              ["operator", "partition", "selected_option",
               "selected_train_phase_cycles"] +
              [name + "_cycles" for name in OPTION_ORDER])
    print("M423A_PASS train_m338={} train_selected={} improvement={:.6%} heldout=0".format(
        option_cycles["m338_q32"], selected_cycles,
        (option_cycles["m338_q32"] - selected_cycles) /
        float(option_cycles["m338_q32"])), flush=True)


if __name__ == "__main__":
    main()
