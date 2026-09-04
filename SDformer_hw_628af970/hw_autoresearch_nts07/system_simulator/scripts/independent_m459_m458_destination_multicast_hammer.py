#!/usr/bin/env python3
"""Independent full-population hammer of M458 exact destination multicast.

This implementation imports neither the M458 analyzer nor any upstream analyzer.
It forms all destination groups directly from the frozen M40 packed support and
the sealed M430 q32 catalog.  M430/M458 derived ledgers are opened only after the
independent 17,280-phase derivation is complete.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PARTITIONS = 432
ROWS_PER_PHASE = 3000
OUTPUT_BLOCKS = 8
SAMPLES = 10
OPERATORS = 4
MULTICAST_WIDTHS = (1, 2, 4, 8)
POPCOUNT = np.asarray([bin(value).count("1") for value in range(1 << 16)],
                      dtype=np.uint8)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def center_words(catalog, operator, partition):
    raw = catalog["operators"][operator]["partitions"][partition]["nested_patterns"]
    require(len(raw) >= 32, "sealed q32 catalog has fewer than 32 centers")
    return np.asarray([int(word, 16) for word in raw[:32]], dtype=np.uint16)


def count_runs(ids):
    ordered = sorted(ids)
    if not ordered:
        return 0
    return 1 + sum(right != left + 1
                   for left, right in zip(ordered, ordered[1:]))


def group_ledger(counts, width):
    population = [int(value) for value in counts if int(value) > 0]
    issues = sum((value + width - 1) // width for value in population)
    tails = sum((value % width) != 0 for value in population)
    tail_occupancy = sum(value % width for value in population if value % width)
    capacity = issues * width
    contributions = sum(population)
    return {
        "nonempty_groups": len(population),
        "contributions": contributions,
        "issues": issues,
        "full_issues": sum(value // width for value in population),
        "tail_issues": tails,
        "tail_occupancy": tail_occupancy,
        "capacity_slots": capacity,
        "wasted_slots": capacity - contributions,
    }


def merge_ledgers(target, source):
    for key, value in source.items():
        target[key] += int(value)


def finish_ledger(counter):
    value = {key: int(counter[key]) for key in (
        "nonempty_groups", "contributions", "issues", "full_issues",
        "tail_issues", "tail_occupancy", "capacity_slots", "wasted_slots")}
    value["utilization"] = (value["contributions"] / value["capacity_slots"]
                            if value["capacity_slots"] else 1.0)
    require(value["capacity_slots"] ==
            value["contributions"] + value["wasted_slots"],
            "group capacity conservation failed")
    return value


def form_words(record, trace_dir, read_payload):
    require(record["shape"] == [10, 1, 768, 15, 20], "M40 shape drift")
    packed_path = trace_dir / record["packed_file"]
    value_path = trace_dir / record["value_payload_file"]
    packed = read_payload(packed_path, record["packed_file_sha256"], "packed",
                          int(record["sample_id"]), record["operator"])
    read_payload(value_path, record["value_payload_sha256"], "value_identity",
                 int(record["sample_id"]), record["operator"])
    raw = np.frombuffer(packed, dtype=np.uint8)
    plane_bytes = int(record["positive_plane_bytes"])
    require(raw.size == 3 * plane_bytes, "M40 packed plane extent drift")
    require(not np.any(raw[plane_bytes:2 * plane_bytes]),
            "negative support exists in frozen M40 payload")
    positive = np.unpackbits(raw[:plane_bytes], bitorder="little")
    positive = positive[:10 * 768 * 15 * 20].reshape(10, 768, 15, 20)
    padded = np.pad(positive, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")
    taps = np.stack([
        padded[:, :, ky:ky + 15, kx:kx + 20]
        for ky in range(3) for kx in range(3)
    ], axis=2)
    rows = np.ascontiguousarray(
        taps.transpose(0, 3, 4, 1, 2).reshape(ROWS_PER_PHASE, 768 * 9))
    words = np.ascontiguousarray(
        np.packbits(rows, axis=1, bitorder="little")).view("<u2")
    require(words.shape == (ROWS_PER_PHASE, PARTITIONS),
            "independent word matrix extent drift")
    return words


def analyze_phase(words, centers):
    values, multiplicity = np.unique(words, return_counts=True)
    values = values.astype(np.uint16)
    multiplicity = multiplicity.astype(np.int64)
    pops = POPCOUNT[values].astype(np.int16)
    distances = POPCOUNT[np.bitwise_xor(
        centers[:, None], values[None, :])].astype(np.int16)
    best_center = distances.argmin(axis=0)
    best_distance = distances[best_center, np.arange(values.size)]
    nonzero = values != 0
    eligible = pops >= 2
    pwp = nonzero & (1 + best_distance < pops)
    fallback = nonzero & (~pwp)
    exact = pwp & (best_distance == 0)
    q16_exact = distances[:16].min(axis=0) == 0
    selected = centers[best_center]
    plus = values & np.bitwise_not(selected)
    minus = selected & np.bitwise_not(values)
    reconstructed = (selected | plus) & np.bitwise_not(minus)
    reconstruction_mismatches = int(np.count_nonzero(reconstructed[pwp] != values[pwp]))
    residual_mismatches = int(np.count_nonzero(
        (POPCOUNT[plus] + POPCOUNT[minus])[pwp] != best_distance[pwp]))
    sign_overlap_mismatches = int(np.count_nonzero((plus & minus)[pwp]))
    require(reconstruction_mismatches == residual_mismatches == sign_overlap_mismatches == 0,
            "exact PWP residual reconstruction failed")

    zero_groups = []
    pwp_groups = np.bincount(best_center[pwp], weights=multiplicity[pwp],
                             minlength=32).astype(np.int64).tolist()
    correction_plus_groups = []
    correction_minus_groups = []
    for bit in range(16):
        mask = np.uint16(1 << bit)
        zero_groups.append(int(multiplicity[(values & mask) != 0].sum()))
        pwp_plus = pwp & ((plus & mask) != 0)
        fallback_plus = fallback & ((values & mask) != 0)
        pwp_minus = pwp & ((minus & mask) != 0)
        correction_plus_groups.append(int(multiplicity[pwp_plus | fallback_plus].sum()))
        correction_minus_groups.append(int(multiplicity[pwp_minus].sum()))

    active_rows = int(multiplicity[nonzero].sum())
    pwp_rows = int(multiplicity[pwp].sum())
    fallback_rows = int(multiplicity[fallback].sum())
    bit_sparse_ops = int(sum(zero_groups))
    correction_ops = int(sum(correction_plus_groups) + sum(correction_minus_groups))
    require(int(multiplicity.sum()) == ROWS_PER_PHASE and
            active_rows == pwp_rows + fallback_rows and
            bit_sparse_ops == int(np.dot(multiplicity, pops.astype(np.int64))) and
            correction_ops == int(np.dot(
                multiplicity, np.where(pwp, best_distance, pops).astype(np.int64))),
            "phase destination contribution conservation failed")

    used_ids = set(int(index) for index in np.unique(best_center[pwp]))
    row = {
        "active_rows": active_rows,
        "eligible_rows": int(multiplicity[eligible].sum()),
        "pwp_rows": pwp_rows,
        "exact_pwp_rows": int(multiplicity[exact].sum()),
        "fallback_rows": fallback_rows,
        "correction_ops_per_block": correction_ops,
        "bit_sparse_ops_per_block": bit_sparse_ops,
        "used_pwp_patterns": len(used_ids),
        "used_center_runs": count_runs(used_ids),
        "early_matcher": ROWS_PER_PHASE + int(multiplicity[eligible & (~q16_exact)].sum()) + 2,
        "reconstruction_mismatches": reconstruction_mismatches,
        "residual_count_mismatches": residual_mismatches,
        "plus_minus_overlap_mismatches": sign_overlap_mismatches,
    }
    ledgers = {}
    for width in MULTICAST_WIDTHS:
        zero = group_ledger(zero_groups, width)
        pwp_value = group_ledger(pwp_groups, width)
        plus_value = group_ledger(correction_plus_groups, width)
        minus_value = group_ledger(correction_minus_groups, width)
        correction = Counter()
        merge_ledgers(correction, plus_value)
        merge_ledgers(correction, minus_value)
        correction = dict(correction)
        row["zero_issues_b{}".format(width)] = zero["issues"]
        row["pwp_issues_b{}".format(width)] = pwp_value["issues"]
        row["correction_issues_b{}".format(width)] = correction["issues"]
        ledgers[width] = {
            "zero": zero,
            "pwp": pwp_value,
            "correction_plus": plus_value,
            "correction_minus": minus_value,
            "correction": correction,
        }
    return row, ledgers


def catalog_sample(phases, width, model):
    now = 0
    component = Counter()
    for phase in phases:
        config_data = int(math.ceil(model["elastic_config_bytes"] /
                                    model["dram_bytes_per_cycle"]))
        now += config_data + model["dma_command_setup_cycles"] + phase["early_matcher"] + 1
        component["config_data"] += config_data
        component["config_command"] += model["dma_command_setup_cycles"]
        component["matcher"] += phase["early_matcher"]
        component["bitmap_seal"] += 1
        if phase["active_rows"] == 0:
            now += model["tail_cycles"]
            component["tail"] += model["tail_cycles"]
            continue
        tile_bytes = (model["weight_bytes_per_tile"] +
                      phase["used_pwp_patterns"] * model["elastic_center_stride_bytes"])
        require(model["elastic_config_bytes"] + tile_bytes <= model["tile_slot_bytes"],
                "catalog tile slot overflow")
        require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                "catalog tile DMA alignment drift")
        tile_data = tile_bytes // model["dram_bytes_per_cycle"]
        tile_command = ((1 + phase["used_center_runs"]) *
                        model["dma_command_setup_cycles"])
        tile_dma = tile_data + tile_command
        per_block = (phase["pwp_issues_b{}".format(width)] +
                     phase["correction_issues_b{}".format(width)])
        tile_work = model["output_blocks_per_tile"] * per_block
        replay = tile_work + model["descriptor_sram_latency_cycles"]
        now += tile_dma
        tile0_end = now + replay
        tile1_dma_end = now + tile_dma
        tile1_start = max(tile0_end, tile1_dma_end)
        component["tile1_dma_exposed"] += max(0, tile1_dma_end - tile0_end)
        now = tile1_start + replay + model["tail_cycles"]
        component["tile0_dma_data"] += tile_data
        component["tile0_dma_commands"] += tile_command
        component["active_compute"] += 2 * tile_work
        component["descriptor_sram_startup"] += 2 * model["descriptor_sram_latency_cycles"]
        component["tail"] += model["tail_cycles"]
    now += model["commit_cycles_per_sample"]
    component["commit"] += model["commit_cycles_per_sample"]
    return int(now), dict(component)


def zero_sample(phases, width, model):
    preprocess = max(
        model["rows_per_phase"] + model["popcount_filter_pipeline_cycles"],
        model["weight_phase_bytes"] // model["dram_bytes_per_cycle"] +
        model["dma_command_setup_cycles"])
    now = preprocess
    component = Counter(initial_preprocess=preprocess)
    for index, phase in enumerate(phases):
        compute = phase["zero_issues_b{}".format(width)] * OUTPUT_BLOCKS
        next_preprocess = preprocess if index + 1 < len(phases) else 0
        now += max(compute, next_preprocess) + model["tail_cycles"]
        component["active_compute"] += compute
        component["preprocess_exposed"] += max(0, next_preprocess - compute)
        component["tail"] += model["tail_cycles"]
    now += model["commit_cycles_per_sample"]
    component["commit"] += model["commit_cycles_per_sample"]
    return int(now), dict(component)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing output overwrite")
    script = Path(__file__).resolve()
    contract = strict_json(args.contract)
    require(contract["schema"] == "m459_m458_independent_hammer_contract_v1" and
            contract["status"] == "FROZEN_BEFORE_INDEPENDENT_RAW_M40_SECOND_PASS",
            "M459 contract not frozen")
    hw_root = args.contract.resolve().parents[1]
    paths = {}
    for label, item in contract["inputs"].items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = hw_root / path
        require(path.is_file() and sha256(path) == item["sha256"],
                "M459 input identity mismatch: " + label)
        paths[label] = path
    require(paths["auditor"].resolve() == script and
            sha256(script) == contract["inputs"]["auditor"]["sha256"],
            "M459 auditor identity drift")
    docs_before = sha256(paths["docs359"])
    catalog_before = sha256(paths["m430_catalog"])
    m458_before = {path.name: sha256(path) for path in
                   paths["m458_result"].parent.iterdir() if path.is_file()}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    marker = args.output_dir / "M459_INDEPENDENT_RAW_M40_SECOND_PASS_AUTHORIZED.marker"
    marker.write_text(
        "status=FROZEN_CONTRACT_AUTHORIZES_ONE_READ_ONLY_INDEPENDENT_SECOND_PASS\n"
        "contract_sha256={}\n"
        "catalog_retuning=false\n"
        "m458_or_upstream_mutation=false\n".format(sha256(args.contract)),
        encoding="utf-8")

    read_counts = Counter()
    read_rows = []

    def read_payload(path, expected_sha, role, sample, operator):
        resolved = Path(path).resolve()
        require(marker.is_file(), "M459 marker missing before payload read")
        require(read_counts[str(resolved)] == 0,
                "M459 payload would be read more than once: " + str(resolved))
        data = resolved.read_bytes()
        read_counts[str(resolved)] += 1
        observed = sha256_bytes(data)
        require(observed == expected_sha, "M459 payload SHA mismatch")
        read_rows.append({
            "read_ordinal": len(read_rows) + 1,
            "sample": sample,
            "operator": operator,
            "role": role,
            "path": str(resolved),
            "bytes": len(data),
            "expected_sha256": expected_sha,
            "actual_sha256": observed,
            "read_count": read_counts[str(resolved)],
        })
        return data

    trace = strict_json(paths["m40_manifest"])
    catalog = strict_json(paths["m430_catalog"])
    require(trace["cohort"]["records"] == 40 and
            trace["cohort"]["samples"] == SAMPLES and
            len(trace["cohort"]["operators"]) == OPERATORS,
            "M40 cohort drift")
    require(catalog["status"] ==
            "PASS_M430_TRAIN_ONLY_DUALAWARE_Q32_FROZEN_BEFORE_HELDOUT" and
            catalog["split"]["runtime_or_validation_data_used"] is False,
            "M430 catalog is not frozen train-only")
    operators = tuple(trace["cohort"]["operators"])
    require(tuple(catalog["geometry"]["operators"]) == operators,
            "operator order drift")
    op_index = {name: index for index, name in enumerate(operators)}
    records = sorted(trace["records"], key=lambda item: (
        int(item["sample_id"]), op_index[item["operator"]]))

    phases_by_sample = [[] for _ in range(SAMPLES)]
    phase_rows = []
    group_totals = {
        width: {path: Counter() for path in (
            "zero", "pwp", "correction_plus", "correction_minus", "correction")}
        for width in MULTICAST_WIDTHS
    }
    aggregate = Counter()
    for record_number, record in enumerate(records, 1):
        sample = int(record["sample_id"])
        operator = op_index[record["operator"]]
        words = form_words(record, paths["m40_manifest"].parent, read_payload)
        for partition in range(PARTITIONS):
            phase, ledgers = analyze_phase(
                words[:, partition], center_words(catalog, operator, partition))
            phase["sample"] = sample
            phase["operator"] = operator
            phase["partition"] = partition
            phase["group_boundary_key"] = "{}:{}:{}".format(sample, operator, partition)
            phases_by_sample[sample].append(phase)
            phase_rows.append(phase)
            aggregate.update({key: phase[key] for key in (
                "active_rows", "eligible_rows", "pwp_rows", "exact_pwp_rows",
                "fallback_rows", "correction_ops_per_block",
                "bit_sparse_ops_per_block", "used_pwp_patterns",
                "used_center_runs", "early_matcher")})
            for width in MULTICAST_WIDTHS:
                for path in group_totals[width]:
                    merge_ledgers(group_totals[width][path], ledgers[width][path])
        print("[M459] independently decoded record {}/40".format(record_number),
              flush=True)

    require(len(phase_rows) == 17280 and
            all(len(phases) == OPERATORS * PARTITIONS for phases in phases_by_sample),
            "M459 phase extent drift")
    require(len(read_rows) == 80 and len(read_counts) == 80 and
            all(count == 1 for count in read_counts.values()),
            "M459 payload exact-once contract failed")

    # From this point on, all independent phase/group/cycle values exist.  Only now
    # are sealed M430 and M458 derived ledgers opened for mismatch accounting.
    model = contract["cycle_model"]
    group_result = {}
    cycle_rows = []
    zero_components = {}
    catalog_components = {}
    for width in MULTICAST_WIDTHS:
        group_result[str(width)] = {
            path: finish_ledger(group_totals[width][path])
            for path in group_totals[width]
        }
        zero_cycles = 0
        catalog_cycles = 0
        zero_component = Counter()
        catalog_component = Counter()
        for phases in phases_by_sample:
            value, component = zero_sample(phases, width, model)
            zero_cycles += value
            zero_component.update(component)
            value, component = catalog_sample(phases, width, model)
            catalog_cycles += value
            catalog_component.update(component)
        zero_components[str(width)] = dict(zero_component)
        catalog_components[str(width)] = dict(catalog_component)
        zero_issues = group_result[str(width)]["zero"]["issues"] * OUTPUT_BLOCKS
        pwp_issues = group_result[str(width)]["pwp"]["issues"] * OUTPUT_BLOCKS
        correction_issues = group_result[str(width)]["correction"]["issues"] * OUTPUT_BLOCKS
        cycle_rows.append({
            "B": width,
            "strong_zero_cycles": zero_cycles,
            "m430_catalog_cycles": catalog_cycles,
            "equal_B_catalog_speedup_vs_strong_zero": zero_cycles / catalog_cycles,
            "strong_zero_source_output_block_issues": zero_issues,
            "catalog_pwp_output_block_issues": pwp_issues,
            "catalog_correction_output_block_issues": correction_issues,
            "zero_utilization": group_result[str(width)]["zero"]["utilization"],
            "pwp_utilization": group_result[str(width)]["pwp"]["utilization"],
            "correction_utilization": group_result[str(width)]["correction"]["utilization"],
            "zero_wasted_slots": group_result[str(width)]["zero"]["wasted_slots"] * OUTPUT_BLOCKS,
            "catalog_wasted_slots": (
                group_result[str(width)]["pwp"]["wasted_slots"] +
                group_result[str(width)]["correction"]["wasted_slots"]) * OUTPUT_BLOCKS,
        })
    b1 = cycle_rows[0]
    for row in cycle_rows:
        row["strong_zero_speedup_vs_B1"] = b1["strong_zero_cycles"] / row["strong_zero_cycles"]
        row["catalog_speedup_vs_B1"] = b1["m430_catalog_cycles"] / row["m430_catalog_cycles"]
        row["equal_B_catalog_advantage_ratio_vs_B1"] = (
            row["equal_B_catalog_speedup_vs_strong_zero"] /
            b1["equal_B_catalog_speedup_vs_strong_zero"])
        row["catalog_throughput_per_accumulator_bank_proxy_vs_B1"] = (
            row["catalog_speedup_vs_B1"] / row["B"])

    require(b1["strong_zero_cycles"] == 742148386 and
            b1["m430_catalog_cycles"] == 517041352 and
            b1["catalog_pwp_output_block_issues"] == 127277168 and
            b1["catalog_correction_output_block_issues"] == 304443912,
            "M459 B1 reconciliation failed")
    for width in MULTICAST_WIDTHS:
        zero = group_result[str(width)]["zero"]
        pwp = group_result[str(width)]["pwp"]
        correction = group_result[str(width)]["correction"]
        plus = group_result[str(width)]["correction_plus"]
        minus = group_result[str(width)]["correction_minus"]
        require(zero["contributions"] == aggregate["bit_sparse_ops_per_block"] and
                pwp["contributions"] == aggregate["pwp_rows"] and
                correction["contributions"] == aggregate["correction_ops_per_block"] and
                correction["contributions"] == plus["contributions"] + minus["contributions"],
                "per-destination contribution conservation failed")

    m430_rows = read_csv(paths["m430_phase_csv"])
    m430_index = {(int(row["sample"]), int(row["operator"]), int(row["partition"])): row
                  for row in m430_rows}
    sealed_fields = contract["reconciliation"]["sealed_phase_fields"]
    sealed_phase_mismatches = 0
    for phase in phase_rows:
        upstream = m430_index[(phase["sample"], phase["operator"], phase["partition"])]
        sealed_phase_mismatches += sum(
            phase[field] != int(upstream[field]) for field in sealed_fields)

    m458_phase_rows = read_csv(paths["m458_per_phase"])
    m458_phase_index = {(int(row["sample"]), int(row["operator"]), int(row["partition"])): row
                        for row in m458_phase_rows}
    m458_phase_mismatches = 0
    issue_fields = []
    for width in MULTICAST_WIDTHS:
        issue_fields.extend(("zero_issues_b{}".format(width),
                             "pwp_issues_b{}".format(width),
                             "correction_issues_b{}".format(width)))
    for phase in phase_rows:
        upstream = m458_phase_index[(phase["sample"], phase["operator"], phase["partition"])]
        m458_phase_mismatches += sum(
            phase[field] != int(upstream[field]) for field in issue_fields)

    m458_b_rows = {int(row["B"]): row for row in read_csv(paths["m458_per_B"])}
    m458_b_mismatches = 0
    integer_fields = (
        "strong_zero_cycles", "m430_catalog_cycles",
        "strong_zero_source_output_block_issues",
        "catalog_pwp_output_block_issues", "catalog_correction_output_block_issues",
        "zero_wasted_slots", "catalog_wasted_slots")
    float_fields = (
        "equal_B_catalog_speedup_vs_strong_zero", "zero_utilization",
        "pwp_utilization", "correction_utilization", "strong_zero_speedup_vs_B1",
        "catalog_speedup_vs_B1", "equal_B_catalog_advantage_ratio_vs_B1",
        "catalog_throughput_per_accumulator_bank_proxy_vs_B1")
    for row in cycle_rows:
        upstream = m458_b_rows[row["B"]]
        m458_b_mismatches += sum(row[field] != int(upstream[field]) for field in integer_fields)
        m458_b_mismatches += sum(
            not math.isclose(row[field], float(upstream[field]), rel_tol=0.0, abs_tol=1e-12)
            for field in float_fields)

    m430_result = strict_json(paths["m430_result"])
    m430_result_mismatches = sum((
        b1["strong_zero_cycles"] != m430_result["comparisons"]["strong_zero_cycles"],
        b1["m430_catalog_cycles"] != m430_result["comparisons"]["m430_catalog_dual_cycles"],
        aggregate["pwp_rows"] * OUTPUT_BLOCKS != 127277168,
        aggregate["correction_ops_per_block"] * OUTPUT_BLOCKS != 304443912,
    ))
    require(sealed_phase_mismatches == m458_phase_mismatches ==
            m458_b_mismatches == m430_result_mismatches == 0,
            "independent/upstream ledger mismatch")

    m458_audit_rows = read_csv(paths["m458_payload_audit"])
    m458_exact_once = (len(m458_audit_rows) == 80 and
                       len(set(row["path"] for row in m458_audit_rows)) == 80 and
                       all(int(row["read_count"]) == 1 for row in m458_audit_rows))
    require(m458_exact_once and paths["m458_marker"].is_file(),
            "M458 single-pass marker/audit invalid")

    # M458 and upstream seals are verified only after the independent result exists.
    seal_checks = {}
    for label, directory, manifest, seal in (
            ("m430_train", paths["m430_train_manifest"].parent, "SHA256SUMS", "SHA256SUMS.seal.sha256"),
            ("m430_heldout", paths["m430_manifest"].parent, "SHA256SUMS", "SHA256SUMS.seal.sha256"),
            ("m435", paths["m435_manifest"].parent, "SHA256SUMS", "SHA256SUMS.seal.sha256"),
            ("m458", paths["m458_result"].parent, "SHA256SUMS", "SHA256SUMS.seal.sha256")):
        manifest_lines = (directory / manifest).read_text().splitlines()
        targets = []
        mismatches = 0
        for line in manifest_lines:
            digest, target = line.split(maxsplit=1)
            target = target.lstrip("*")
            targets.append(target)
            path = directory / target
            mismatches += int(not path.is_file() or sha256(path) != digest)
        outer = (directory / seal).read_text().strip().split(maxsplit=1)
        outer_ok = (len(outer) == 2 and outer[0] == sha256(directory / manifest))
        require(mismatches == 0 and outer_ok, label + " seal check failed")
        seal_checks[label] = {"manifest_entries": len(manifest_lines),
                              "mismatches": mismatches, "outer_seal_ok": outer_ok}

    equal_b_ratios = [row["equal_B_catalog_speedup_vs_strong_zero"] for row in cycle_rows]
    ratio_strictly_degrades = all(right < left for left, right in
                                  zip(equal_b_ratios, equal_b_ratios[1:]))
    require(ratio_strictly_degrades, "equal-B catalog advantage is not strictly degrading")
    all_gt1_below_b1 = all(
        row["equal_B_catalog_advantage_ratio_vs_B1"] < 1.0 for row in cycle_rows[1:])
    require(all_gt1_below_b1, "a B>1 point improves the equal-B catalog advantage")

    write_csv(args.output_dir / "m459_payload_second_pass_audit.csv", read_rows,
              list(read_rows[0].keys()))
    phase_fields = ["sample", "operator", "partition", "group_boundary_key",
                    "active_rows", "eligible_rows", "pwp_rows", "exact_pwp_rows",
                    "fallback_rows", "correction_ops_per_block", "bit_sparse_ops_per_block",
                    "used_pwp_patterns", "used_center_runs", "early_matcher"]
    for width in MULTICAST_WIDTHS:
        phase_fields.extend(("zero_issues_b{}".format(width),
                             "pwp_issues_b{}".format(width),
                             "correction_issues_b{}".format(width)))
    write_csv(args.output_dir / "m459_per_phase_recomputation.csv", phase_rows, phase_fields)
    write_csv(args.output_dir / "m459_per_B_recomputation.csv", cycle_rows,
              list(cycle_rows[0].keys()))

    catalog_after = sha256(paths["m430_catalog"])
    m458_after = {path.name: sha256(path) for path in
                  paths["m458_result"].parent.iterdir() if path.is_file()}
    docs_after = sha256(paths["docs359"])
    require(catalog_before == catalog_after and m458_before == m458_after and
            docs_before == docs_after == contract["inputs"]["docs359"]["sha256"],
            "M459 changed frozen evidence")
    result = {
        "schema": "m459_m458_independent_recomputation_v1",
        "status": "PASS_M459_INDEPENDENT_CONFIRM_M458_B_GT1_RTL_NO_GO",
        "identity": {
            "contract_sha256": sha256(args.contract),
            "auditor_sha256": sha256(script),
            "docs359_before_after": docs_after,
            "m430_catalog_before_after": catalog_after,
            "m458_tree_unchanged": True,
        },
        "independence": {
            "imported_m458_analyzer": False,
            "used_m458_derived_rows_to_form_result": False,
            "m430_m458_ledgers_opened_only_after_independent_derivation": True,
            "phases_recomputed": len(phase_rows),
            "destinations_recomputed": len(phase_rows) * ROWS_PER_PHASE,
            "raw_payload_second_pass_explicitly_authorized_by_frozen_contract": True,
            "marker_created_before_first_payload_read": True,
            "payload_files_read": len(read_rows),
            "payload_files_read_exactly_once": all(count == 1 for count in read_counts.values()),
        },
        "population": {key: int(aggregate[key]) for key in (
            "active_rows", "eligible_rows", "pwp_rows", "exact_pwp_rows",
            "fallback_rows", "correction_ops_per_block", "bit_sparse_ops_per_block")},
        "architecture_checks": {
            "group_boundary": "sample/operator/partition/output-block",
            "cross_boundary_groups": 0,
            "output_blocks_kept_separate": True,
            "correction_plus_minus_groups_separate": True,
            "persistent_old_psum_per_destination_output_block": True,
            "destination_output_block_contexts_checked": len(phase_rows) * ROWS_PER_PHASE * OUTPUT_BLOCKS,
            "exact_reconstruction_mismatches": 0,
            "destination_contribution_mismatches": 0,
            "tail_or_capacity_conservation_mismatches": 0,
        },
        "group_ledgers_per_block": group_result,
        "per_B": cycle_rows,
        "components": {"zero": zero_components, "catalog": catalog_components},
        "crosschecks": {
            "m430_sealed_phase_field_mismatches": sealed_phase_mismatches,
            "m430_result_mismatches": m430_result_mismatches,
            "m458_phase_issue_mismatches": m458_phase_mismatches,
            "m458_per_B_mismatches": m458_b_mismatches,
            "m458_payload_marker_present": paths["m458_marker"].is_file(),
            "m458_payload_exact_once_audit": m458_exact_once,
            "seal_checks": seal_checks,
        },
        "fairness": {
            "same_B_accumulator_banks_and_update_ports_granted_to_strong_zero": True,
            "equal_B_ratios_strictly_degrade_with_B": ratio_strictly_degrades,
            "every_B_gt1_equal_B_advantage_below_B1": all_gt1_below_b1,
            "throughput_per_B_proxy_is_actual_throughput_per_area": False,
            "proxy_interpretation": "A deliberately pessimistic diagnostic because B banks are not the entire area; not sufficient alone for NO-GO.",
            "no_go_independent_of_proxy": "Equal-B catalog advantage strictly degrades even if the additional B resources are treated as area-free.",
            "matched_area_power_evidence": False,
        },
        "decision": {
            "exact_trace_arithmetic": "GO",
            "M458_negative_screen": "GO",
            "B_gt1_destination_multicast_RTL": "NO_GO",
            "reason": "Same-B strong zero improves faster; equal-B catalog advantage strictly degrades for B=2/4/8, while matched area/power evidence is absent.",
            "resource_normalized_or_system_or_headline_claim": "NO_GO",
        },
    }
    (args.output_dir / "m459_independent_recomputation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS_M459 phases={} zero={} catalog={} ratios={} docs359={}".format(
        len(phase_rows), b1["strong_zero_cycles"], b1["m430_catalog_cycles"],
        ",".join("{:.9f}".format(value) for value in equal_b_ratios), docs_after),
        flush=True)


if __name__ == "__main__":
    main()
