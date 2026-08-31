#!/usr/bin/env python3
"""Build and pre-heldout-seal an exact dual-co-read-aware H67 q32 catalog.

The first 16 entries are the admitted M77 q16 catalog, bit-for-bit and in
order.  The remaining 16 entries are selected only from M338 IDs 16..127 and
only from the disjoint M73 training population.  This program has no M40 or
validation input.  Its exact issue objective is fallback ``popcount(x)``
versus legal dual-bank PWP ``1 + Hamming(x,p)``; PWP is selected only when the
latter is strictly smaller.
"""

import argparse
from collections import Counter
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
OPTIONS = ("m338_q32", "dual_single_gain_q32", "dual_greedy_q32")
POPCOUNT = np.asarray([bin(value).count("1")
                       for value in range(1 << K)], dtype=np.uint8)


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


def distances(values, centers):
    centers = np.asarray(centers, dtype=np.uint16)
    return POPCOUNT[np.bitwise_xor(centers[:, None], values[None, :])]


def dual_units_from_distance(values, best_distance):
    pops = POPCOUNT[values].astype(np.int16)
    distance = best_distance.astype(np.int16)
    return np.where(1 + distance < pops, 1 + distance, pops)


def dual_objective(values, counts, centers):
    best = distances(values, centers).min(axis=0)
    return int(np.dot(counts,
                      dual_units_from_distance(values, best).astype(np.int64)))


def single_gain_tail(values, counts, q16, pool):
    base_distance = distances(values, q16).min(axis=0)
    base_units = dual_units_from_distance(values, base_distance)
    pool_distance = distances(values, pool)
    scored = []
    for pool_id, center in enumerate(pool):
        units = dual_units_from_distance(
            values, np.minimum(base_distance, pool_distance[pool_id]))
        gain = int(np.dot(counts,
                          (base_units - units).astype(np.int64)))
        exact = int(counts[values == center].sum())
        scored.append((-gain, -exact, pool_id, int(center)))
    scored.sort()
    tail = [row[-1] for row in scored[:TAIL]]
    require(len(tail) == TAIL and len(set(tail)) == TAIL,
            "single-gain tail extent failure")
    return tail


def greedy_tail(values, counts, q16, pool):
    pool_distance = distances(values, pool)
    best_distance = distances(values, q16).min(axis=0)
    selected = []
    selected_ids = set()
    for _ in range(TAIL):
        best_key = None
        best_pool_id = None
        best_next_distance = None
        for pool_id, center in enumerate(pool):
            if pool_id in selected_ids:
                continue
            candidate_distance = np.minimum(best_distance,
                                            pool_distance[pool_id])
            objective = int(np.dot(
                counts, dual_units_from_distance(
                    values, candidate_distance).astype(np.int64)))
            # Stable order is the M338 pool order, then numeric value.
            key = (objective, pool_id, int(center))
            if best_key is None or key < best_key:
                best_key = key
                best_pool_id = pool_id
                best_next_distance = candidate_distance
        require(best_pool_id is not None, "greedy selection exhausted pool")
        selected_ids.add(best_pool_id)
        selected.append(int(pool[best_pool_id]))
        best_distance = best_next_distance
    require(len(selected) == TAIL and len(set(selected)) == TAIL,
            "greedy tail extent failure")
    return selected


def static_pwp_range(weight_slice, centers):
    bits = np.asarray([[(center >> bit) & 1 for bit in range(K)]
                       for center in centers], dtype=np.int16)
    products = bits @ weight_slice
    require(products.shape == (32, 768), "PWP product shape drift")
    return int(products.min()), int(products.max())


def count_runs(indices):
    ordered = sorted(indices)
    if not ordered:
        return 0
    return 1 + sum(current != previous + 1
                   for previous, current in zip(ordered, ordered[1:]))


def evaluate_partition(sample_counters, centers, model):
    union = sorted(set().union(*(counter.keys()
                                for counter in sample_counters)))
    values = np.asarray(union, dtype=np.uint16)
    counts = np.zeros((SAMPLES, len(values)), dtype=np.int64)
    lookup = {value: index for index, value in enumerate(union)}
    for sample, counter in enumerate(sample_counters):
        for value, count in counter.items():
            counts[sample, lookup[value]] = count
    require(bool(np.all(counts.sum(axis=1) == model["rows_per_phase"])),
            "train phase row extent drift")
    pops = POPCOUNT[values].astype(np.int16)
    matrix = distances(values, centers)
    best_id = matrix.argmin(axis=0)
    best_distance = matrix[best_id, np.arange(len(values))].astype(np.int16)
    pwp = (values != 0) & (1 + best_distance < pops)
    correction = np.where(pwp, best_distance, pops).astype(np.int64)
    q16_exact = distances(values, centers[:16]).min(axis=0) == 0
    eligible = pops >= 2
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
        exact_rows = int(row_counts[pwp & (best_distance == 0)].sum())
        corr = int(np.dot(row_counts, correction))
        bit_sparse = int(np.dot(row_counts, pops.astype(np.int64)))
        used = set(int(index) for index in np.unique(
            best_id[pwp & (row_counts > 0)]))
        runs = count_runs(used)
        if active == 0:
            phase_cycles += model["tail_cycles"]
        else:
            tile_bytes = (model["weight_bytes_per_tile"] +
                          len(used) * model["elastic_center_stride_bytes"])
            require(model["elastic_config_bytes"] + tile_bytes <=
                    model["tile_slot_bytes"], "train tile slot overflow")
            require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                    "train tile DMA alignment drift")
            tile_dma = (tile_bytes // model["dram_bytes_per_cycle"] +
                        (1 + runs) * model["dma_command_setup_cycles"])
            # Four output blocks/tile.  A PWP consumes one simultaneous
            # low8/high4 issue and every residual/fallback source one issue.
            work = 4 * corr + 4 * pwp_rows
            replay0 = work + model["descriptor_sram_latency_cycles"]
            replay1 = work + model["descriptor_sram_latency_cycles"]
            require(work >= active, "dual descriptor service underflow")
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
            "bit_sparse_ops_per_block": bit_sparse,
            "pwp_rows": pwp_rows,
            "exact_pwp_rows": exact_rows,
            "positive_residual_pwp_rows": pwp_rows - exact_rows,
            "fallback_rows": active - pwp_rows,
            "correction_ops_per_block": corr,
            "dual_issue_ops_per_block": corr + pwp_rows,
            "used_pwp_patterns": len(used),
            "used_center_runs": runs,
        })
    return cycles, total


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


def write_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    lines = ["{}  {}".format(sha256(output_dir / name), name)
             for name in sorted(names)]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    seal = output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")
    return manifest, seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M430 train overwrite")
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m430_trainonly_dualaware_q32_catalog_contract_v1" and
            contract.get("status") == "FROZEN_BEFORE_M430_TRAIN_EXECUTION",
            "M430 train contract drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M430 train input SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": spec["sha256"]}
    require(paths["builder"].resolve() == Path(__file__).resolve() and
            identities["builder"]["sha256"] == source_start,
            "M430 builder self-identity drift")

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
            "M430 train/heldout isolation drift")
    require(admission["train_only_admitted"] is True and
            admission["catalog_sha256"] ==
            identities["m77_q16_catalog"]["sha256"] and
            admission["train_valid825_key_overlap"] == 0 and
            m338["split"]["test_or_validation_data_used"] is False and
            m338["split"]["train_valid825_key_overlap"] == 0,
            "M430 train parent identity drift")
    operators = tuple(manifest["cohort"]["operators"])
    require(len(operators) == 4 and
            [row["operator"] for row in parent["operators"]] == list(operators) and
            [row["operator"] for row in m338["operators"]] == list(operators),
            "M430 operator order drift")

    helper = load_module(paths["m423_train_helper"], "m430_m423_helper")
    m43 = helper.load_module(paths["m43_unpacker"])
    histograms, payload_files, payload_bytes = helper.collect_histograms(
        m43, manifest, paths["m73_train_trace_manifest"], operators)
    weights = [np.fromfile(paths["weight_o{}".format(op)], dtype=np.int8)
               .reshape(6912, 768).astype(np.int16) for op in range(4)]
    model = contract["cycle_model"]

    option_cycles = Counter()
    option_totals = {name: Counter() for name in OPTIONS}
    selected_totals = Counter()
    selected_cycles = 0
    selected_counts = Counter()
    baseline_by_sample = [[] for _ in range(SAMPLES)]
    operator_payloads = []
    csv_rows = []
    global_minimum = 1 << 30
    global_maximum = -(1 << 30)
    for op, operator in enumerate(operators):
        partitions = []
        for partition in range(PARTITIONS):
            sample_counters = [histograms[(sample, op, partition)]
                               for sample in range(SAMPLES)]
            aggregate = Counter()
            for counter in sample_counters:
                aggregate.update(counter)
            values = np.asarray(sorted(aggregate), dtype=np.uint16)
            counts = np.asarray([aggregate[int(value)] for value in values],
                                dtype=np.int64)
            q16 = [int(item["value_hex"], 16) for item in
                   parent["operators"][op]["partitions"][partition]["patterns"]]
            nested = [int(value, 16) for value in
                      m338["operators"][op]["partitions"][partition]
                      ["nested_patterns"]]
            require(len(q16) == 16 and len(nested) >= 128 and
                    nested[:16] == q16 and len(set(nested[:128])) == 128,
                    "M430 q16/q128 prefix drift")
            pool = nested[16:128]
            candidates = {
                "m338_q32": q16 + pool[:16],
                "dual_single_gain_q32": q16 +
                    single_gain_tail(values, counts, q16, pool),
                "dual_greedy_q32": q16 +
                    greedy_tail(values, counts, q16, pool),
            }
            evaluations = {}
            objectives = {}
            for option in OPTIONS:
                centers = candidates[option]
                require(centers[:16] == q16 and len(centers) == 32 and
                        len(set(centers)) == 32 and
                        all(center in pool for center in centers[16:]),
                        "M430 center extent/pool drift")
                objectives[option] = dual_objective(values, counts, centers)
                evaluations[option] = evaluate_partition(
                    sample_counters, centers, model)
                option_cycles[option] += evaluations[option][0]
                option_totals[option].update(evaluations[option][1])
            chosen = min(OPTIONS,
                         key=lambda name: (evaluations[name][0],
                                           objectives[name],
                                           OPTIONS.index(name)))
            chosen_cycles, chosen_metrics = evaluations[chosen]
            selected_cycles += chosen_cycles
            selected_totals.update(chosen_metrics)
            selected_counts[chosen] += 1
            minimum, maximum = static_pwp_range(
                weights[op][partition * 16:(partition + 1) * 16],
                candidates[chosen])
            require(minimum >= -2048 and maximum <= 2047,
                    "M430 selected signed12 overflow")
            global_minimum = min(global_minimum, minimum)
            global_maximum = max(global_maximum, maximum)
            for sample in range(SAMPLES):
                baseline_by_sample[sample].append(sum(
                    count * int(POPCOUNT[value])
                    for value, count in sample_counters[sample].items()))
            partitions.append({
                "partition": partition,
                "nested_patterns": ["{:04x}".format(value)
                                    for value in candidates[chosen]],
                "selected_train_option": chosen,
                "train_phase_cycles_by_option": {
                    name: evaluations[name][0] for name in OPTIONS},
                "dual_issue_objective_by_option": objectives,
            })
            row = {"operator": op, "partition": partition,
                   "selected_option": chosen,
                   "selected_train_phase_cycles": chosen_cycles}
            for name in OPTIONS:
                row[name + "_cycles"] = evaluations[name][0]
                row[name + "_objective"] = objectives[name]
            csv_rows.append(row)
            if (partition + 1) % 108 == 0:
                print("[M430 TRAIN] operator={}/4 partition={}/432".format(
                    op + 1, partition + 1), flush=True)
        operator_payloads.append({"operator": operator,
                                  "partitions": partitions})

    commit = SAMPLES * model["commit_cycles_per_sample"]
    selected_cycles += commit
    for option in OPTIONS:
        option_cycles[option] += commit
    strong_baseline = sum(baseline_sample(baseline_by_sample[sample], model)
                          for sample in range(SAMPLES))
    require(selected_totals["source_rows"] ==
            SAMPLES * 4 * PARTITIONS * model["rows_per_phase"] and
            selected_totals["source_rows"] ==
            selected_totals["zero_rows"] + selected_totals["active_rows"] and
            selected_totals["active_rows"] ==
            selected_totals["pwp_rows"] + selected_totals["fallback_rows"],
            "M430 train population conservation failure")
    require(source_start == sha256(Path(__file__).resolve()),
            "M430 builder changed during execution")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    catalog = {
        "schema": "m430_trainonly_dualaware_q32_catalog_v1",
        "status": "PASS_M430_TRAIN_ONLY_DUALAWARE_Q32_FROZEN_BEFORE_HELDOUT",
        "identity": identities,
        "split": {
            "role": "DSEC_TRAIN_ONLY_HARDWARE_DUAL_COREAD_CATALOG",
            "selected_train_samples": SAMPLES,
            "selected_train_sequences": 18,
            "train_valid825_key_overlap": 0,
            "runtime_or_validation_data_used": False,
        },
        "algorithm": {
            "q16_prefix": "M77 bit-identical including order",
            "tail": "16 unique entries selected only from each partition's M338 q128 IDs 16..127",
            "objective": "minimum exact legal dual-co-read cmd32/L8 train phase cycles among deterministic M338, single-gain and sequential-greedy candidates",
            "per_row_cost": "fallback=popcount(x); PWP=1+Hamming(x,p); select PWP iff 1+d<popcount",
            "candidate_order": list(OPTIONS),
            "random_seed": None,
            "runtime_arithmetic": "W*x=PWP[p]+signed W*(x-p), then add the delta to persistent old_psum; no seed/correction fusion",
            "accuracy_loss": False,
        },
        "geometry": {
            "partition_bits": 16,
            "partitions_per_operator": PARTITIONS,
            "operators": list(operators),
            "q_capacity": 32,
            "output_blocks": 8,
            "shared_lanes": 96,
            "pwp_stride_bytes": model["elastic_center_stride_bytes"],
            "tile_slot_bytes": model["tile_slot_bytes"],
        },
        "operators": operator_payloads,
        "admission": {
            "train_only_catalog": True,
            "q16_parent_bit_identical": True,
            "tail_from_m338_ids16_to127_only": True,
            "exact_arithmetic_identity": True,
            "persistent_old_psum_preserved": True,
            "checkpoint_or_accuracy_changed": False,
            "heldout_runtime_evaluated": False,
            "cycle_speedup": False,
            "selected_rtl": False,
            "synopsys": False,
            "energy": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "claim_boundary": "Frozen train-only exact q32 catalog for the legal dual-bank co-read adapter. Held-out cycles remain false until the single subsequent M40 replay. Four H67 bottleneck Conv3x3 only; not system, RTL, PPA, power or DATE headline.",
    }
    catalog_path = args.output_dir / "m430_trainonly_dualaware_q32_catalog_r1.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    audit = {
        "schema": "m430_trainonly_dualaware_q32_catalog_audit_v1",
        "status": "PASS_M430_TRAIN_CATALOG_DOUBLE_SEALED_HELDOUT_NOT_RUN",
        "identity": identities,
        "payload_audit": {"files_rehashed": payload_files,
                          "bytes_rehashed": payload_bytes,
                          "mismatches": 0},
        "cycle_model": model,
        "train_only_observation": {
            "strong_zero_elided_baseline_cycles": strong_baseline,
            "candidate_cycles_by_fixed_global_option": dict(option_cycles),
            "hybrid_selected_cycles": selected_cycles,
            "hybrid_speedup_vs_train_strong_baseline":
                strong_baseline / float(selected_cycles),
            "selected_partition_counts": dict(selected_counts),
            "selected_population": dict(selected_totals),
            "selected_static_pwp_minimum": global_minimum,
            "selected_static_pwp_maximum": global_maximum,
            "selected_static_signed12_overflow": 0,
        },
        "exactness": {
            "q16_prefix_mismatches": 0,
            "tail_outside_m338_pool_mismatches": 0,
            "population_conservation_mismatches": 0,
            "arithmetic_identity": "old_psum += PWP[p] + W*(x-p); fallback old_psum += W*x",
            "checkpoint_changed": False,
            "accuracy_loss": False,
        },
        "heldout_gate": {
            "m40_payload_reads_so_far": 0,
            "m40_completed_evaluations_so_far": 0,
            "catalog_sealed_before_m40": True,
            "post_m40_tuning_allowed": False,
        },
        "admission": catalog["admission"],
    }
    audit_path = args.output_dir / "m430_trainonly_dualaware_q32_catalog_audit_r1.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    csv_path = args.output_dir / "per_partition_train_dualaware_options.csv"
    fields = ["operator", "partition", "selected_option",
              "selected_train_phase_cycles"]
    for name in OPTIONS:
        fields.extend([name + "_cycles", name + "_objective"])
    import csv
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest_path, seal_path = write_seal(args.output_dir, [
        catalog_path.name, audit_path.name, csv_path.name])
    print("M430_TRAIN_PASS m338={} selected={} gain={:.6%} seal={} heldout=0".format(
        option_cycles["m338_q32"], selected_cycles,
        (option_cycles["m338_q32"] - selected_cycles) /
        float(option_cycles["m338_q32"]), sha256(seal_path)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
