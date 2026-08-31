#!/usr/bin/env python3
"""M458 exact destination-multicast, matched-resource CPU screen.

The experiment keeps the sealed M430 q32 catalog and its exact arithmetic
unchanged.  For B in {1,2,4,8}, one source/PWP read may be broadcast to at
most B independent destination accumulators.  Strong-zero and catalog paths
receive the same B-way destination hardware.  Grouping is legal only inside
one sample/operator/partition phase and one output block; no source read is
shared across output blocks.

This is a trace-cycle and traffic screen, not RTL, Synopsys, physical SRAM,
interconnect, energy, resource-normalized, full-network, or headline evidence.
"""

from __future__ import print_function

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


TIMESTEPS = 10
CHANNELS = 768
HEIGHT = 15
WIDTH = 20
FEATURES = CHANNELS * 3 * 3
TILE_BITS = 256
TILES = (FEATURES + TILE_BITS - 1) // TILE_BITS
ROWS = TIMESTEPS * HEIGHT * WIDTH
PARTITIONS = TILES * 16
OUTPUT_BLOCKS = 8
OUTPUT_BLOCKS_PER_TILE = 4
OUTPUT_TILES = 2
MASK16 = 0xffff


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


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


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(output_dir / name), name)
        for name in sorted(names)), encoding="utf-8")
    seal = output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")
    return manifest, seal


def verify_double_seal(directory, manifest, seal, label):
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(sha256(directory / name) == expected,
                "{} inner seal mismatch: {}".format(label, name))
        checked += 1
    expected_manifest, name = seal.read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(name == manifest.name and sha256(manifest) == expected_manifest,
            label + " outer seal mismatch")
    return checked


class PayloadReader(object):
    """Read every raw M40 payload exactly once from the filesystem."""

    def __init__(self):
        self.seen = set()
        self.audit = []

    def read_once(self, path, expected, role, sample, operator):
        resolved = Path(path).resolve()
        require(resolved not in self.seen,
                "M458 repeated payload read: " + str(resolved))
        self.seen.add(resolved)
        raw = resolved.read_bytes()
        actual = sha256_bytes(raw)
        require(actual == expected,
                "M458 payload SHA drift: " + str(resolved))
        self.audit.append({
            "read_ordinal": len(self.audit) + 1,
            "sample": sample,
            "operator": operator,
            "role": role,
            "path": resolved.name,
            "bytes": len(raw),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "read_count": 1,
        })
        return raw


def population(value):
    method = getattr(value, "bit_count", None)
    return method() if method is not None else bin(value).count("1")


def iter_set_bits(value):
    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def unpack_record_masks_from_bytes(record, raw):
    """Self-contained M43-equivalent positive support expansion."""
    require(record["shape"] == [TIMESTEPS, 1, CHANNELS, HEIGHT, WIDTH],
            "M458 record shape drift")
    plane_bytes = record["positive_plane_bytes"]
    require(len(raw) == 3 * plane_bytes, "M458 packed extent drift")
    positive = raw[:plane_bytes]
    negative = raw[plane_bytes:2 * plane_bytes]
    require(not any(negative),
            "M458 requires frozen nonnegative two-code M40 trace")

    masks = [0] * (ROWS * TILES)
    total_bits = TIMESTEPS * CHANNELS * HEIGHT * WIDTH
    for byte_index, original_byte in enumerate(positive):
        byte = original_byte
        if byte == 0:
            continue
        bit_base = byte_index * 8
        while byte:
            low = byte & -byte
            bit = low.bit_length() - 1
            flat = bit_base + bit
            require(flat < total_bits, "M458 nonzero packed tail bit")
            tc, spatial = divmod(flat, HEIGHT * WIDTH)
            timestep, channel = divmod(tc, CHANNELS)
            input_y, input_x = divmod(spatial, WIDTH)
            feature_base = channel * 9
            for kernel_y in range(3):
                output_y = input_y - kernel_y + 1
                if output_y < 0 or output_y >= HEIGHT:
                    continue
                for kernel_x in range(3):
                    output_x = input_x - kernel_x + 1
                    if output_x < 0 or output_x >= WIDTH:
                        continue
                    feature = feature_base + kernel_y * 3 + kernel_x
                    tile, tile_bit = divmod(feature, TILE_BITS)
                    row = (timestep * HEIGHT + output_y) * WIDTH + output_x
                    masks[row * TILES + tile] |= 1 << tile_bit
            byte ^= low
    return masks


def count_runs(indices):
    ordered = sorted(indices)
    if not ordered:
        return 0
    return 1 + sum(1 for left, right in zip(ordered, ordered[1:])
                   if right != left + 1)


def split_group_ledger(groups, b_value):
    """Ceiling-tail ledger for one group family.

    The member lists are immutable destination/contribution identities.  The
    split changes only issue packing, so sum(chunk sizes)==len(members) proves
    that B-way multicast neither drops nor duplicates a contribution.
    """
    ledger = Counter()
    remainder_histogram = Counter()
    for members in groups:
        if not members:
            continue
        count = len(members)
        quotient, remainder = divmod(count, b_value)
        issues = quotient + int(remainder != 0)
        require(issues == int(math.ceil(count / float(b_value))),
                "M458 non-ceiling group split")
        require(quotient * b_value + remainder == count,
                "M458 group-tail destination loss")
        require(remainder == 0 or 1 <= remainder <= b_value,
                "M458 illegal tail size")
        ledger["groups"] += 1
        ledger["groups_smaller_than_b"] += int(count < b_value)
        ledger["full_chunks"] += quotient
        ledger["partial_chunks"] += int(remainder != 0)
        ledger["issues"] += issues
        ledger["useful_destination_updates"] += count
        ledger["capacity_destination_slots"] += issues * b_value
        ledger["wasted_destination_slots"] += issues * b_value - count
        remainder_histogram[remainder] += 1
    require(ledger["capacity_destination_slots"] ==
            ledger["useful_destination_updates"] +
            ledger["wasted_destination_slots"],
            "M458 multicast capacity conservation mismatch")
    for key in ("groups", "groups_smaller_than_b", "full_chunks",
                "partial_chunks", "issues", "useful_destination_updates",
                "capacity_destination_slots", "wasted_destination_slots"):
        ledger[key] += 0
    return ledger, remainder_histogram


def analyze_phase(values, centers, sealed, b_values):
    """Build exact destination memberships and all B issue ledgers."""
    require(len(values) == ROWS and len(centers) == 32,
            "M458 phase geometry drift")
    zero_groups = [[] for _ in range(16)]
    pwp_groups = [[] for _ in range(32)]
    # Sign +1 and -1 are distinct correction source groups.  Fallback terms
    # enter only the +1 groups and are never merged with -1.
    correction_groups = dict(((bit, sign), [])
                             for bit in range(16) for sign in (1, -1))
    zero_updates = [0] * ROWS
    catalog_updates = [0] * ROWS
    expected_zero_updates = [0] * ROWS
    expected_catalog_updates = [0] * ROWS
    phase = Counter()
    used_centers = set()
    exact_q16_rows = 0
    exact_q32_rows = 0
    reconstruction_mismatches = 0

    for destination, original in enumerate(values):
        original &= MASK16
        pop = population(original)
        phase["source_rows"] += 1
        phase["zero_rows"] += int(original == 0)
        phase["active_rows"] += int(original != 0)
        phase["eligible_rows"] += int(pop >= 2)
        phase["bit_sparse_vector_ops_per_block"] += pop
        expected_zero_updates[destination] = pop
        for bit in iter_set_bits(original):
            zero_groups[bit].append(destination)
            zero_updates[destination] += 1

        distances = [population(original ^ center) for center in centers]
        if pop >= 2:
            q16_exact = min(distances[:16]) == 0
            q32_exact = min(distances) == 0
            exact_q16_rows += int(q16_exact)
            exact_q32_rows += int(q32_exact and not q16_exact)
            phase["q32_early_extra_prefix_tasks"] += int(not q16_exact)
        if original == 0:
            continue
        best_distance = min(distances)
        best_index = distances.index(best_distance)
        use_pwp = 1 + best_distance < pop
        if use_pwp:
            selected = centers[best_index]
            plus = original & ((~selected) & MASK16)
            minus = selected & ((~original) & MASK16)
            reconstructed = ((selected | plus) & ((~minus) & MASK16))
            reconstruction_mismatches += int(reconstructed != original)
            phase["pwp_rows"] += 1
            phase["correction_ops_per_block"] += best_distance
            expected_catalog_updates[destination] = 1 + best_distance
            pwp_groups[best_index].append(destination)
            catalog_updates[destination] += 1
            used_centers.add(best_index)
            for bit in iter_set_bits(plus):
                correction_groups[(bit, 1)].append(destination)
                catalog_updates[destination] += 1
            for bit in iter_set_bits(minus):
                correction_groups[(bit, -1)].append(destination)
                catalog_updates[destination] += 1
        else:
            phase["fallback_rows"] += 1
            phase["correction_ops_per_block"] += pop
            expected_catalog_updates[destination] = pop
            for bit in iter_set_bits(original):
                correction_groups[(bit, 1)].append(destination)
                catalog_updates[destination] += 1

    zero_destination_mismatches = sum(
        actual != expected for actual, expected in
        zip(zero_updates, expected_zero_updates))
    catalog_destination_mismatches = sum(
        actual != expected for actual, expected in
        zip(catalog_updates, expected_catalog_updates))
    require(reconstruction_mismatches == 0 and
            zero_destination_mismatches == 0 and
            catalog_destination_mismatches == 0,
            "M458 per-destination contribution/update conservation failure")

    phase["used_pwp_patterns"] = len(used_centers)
    phase["used_center_runs"] = count_runs(used_centers)
    phase["q32_early_matcher_cycles"] = (
        ROWS + phase["q32_early_extra_prefix_tasks"] + 2)
    exact_pwp_rows = exact_q16_rows + exact_q32_rows
    expected_sealed = {
        "active_rows": phase["active_rows"],
        "eligible_rows": phase["eligible_rows"],
        "pwp_rows": phase["pwp_rows"],
        "exact_pwp_rows": exact_pwp_rows,
        "fallback_rows": phase["fallback_rows"],
        "correction_ops_per_block": phase["correction_ops_per_block"],
        "used_pwp_patterns": phase["used_pwp_patterns"],
        "used_center_runs": phase["used_center_runs"],
        "early_matcher": phase["q32_early_matcher_cycles"],
    }
    sealed_mismatches = 0
    for key, actual in expected_sealed.items():
        sealed_mismatches += int(int(sealed[key]) != actual)
    require(sealed_mismatches == 0,
            "M458 sealed M430 phase field mismatch")

    by_b = {}
    correction_group_list = [correction_groups[(bit, sign)]
                             for bit in range(16) for sign in (1, -1)]
    for b_value in b_values:
        zero, zero_remainders = split_group_ledger(zero_groups, b_value)
        pwp, pwp_remainders = split_group_ledger(pwp_groups, b_value)
        correction, correction_remainders = split_group_ledger(
            correction_group_list, b_value)
        require(zero["useful_destination_updates"] ==
                phase["bit_sparse_vector_ops_per_block"],
                "M458 zero useful-update mismatch")
        require(pwp["useful_destination_updates"] == phase["pwp_rows"] and
                correction["useful_destination_updates"] ==
                phase["correction_ops_per_block"],
                "M458 catalog useful-update mismatch")
        by_b[b_value] = {
            "zero": dict(zero),
            "pwp": dict(pwp),
            "correction": dict(correction),
            "zero_remainders": dict(zero_remainders),
            "pwp_remainders": dict(pwp_remainders),
            "correction_remainders": dict(correction_remainders),
        }
    # B=1 must reproduce the old per-destination, per-source service counts.
    require(by_b[1]["zero"]["issues"] ==
            phase["bit_sparse_vector_ops_per_block"] and
            by_b[1]["pwp"]["issues"] == phase["pwp_rows"] and
            by_b[1]["correction"]["issues"] ==
            phase["correction_ops_per_block"],
            "M458 B1 issue reconciliation failure")
    return dict(phase), by_b, {
        "destinations_checked": ROWS,
        "zero_destination_update_count_mismatches":
            zero_destination_mismatches,
        "catalog_destination_update_count_mismatches":
            catalog_destination_mismatches,
        "exact_reconstruction_mismatches": reconstruction_mismatches,
        "sealed_phase_field_mismatches": sealed_mismatches,
        "persistent_old_psum_independent_per_destination": True,
    }


def baseline_sample(phases, model, b_value):
    preprocess = max(
        model["rows_per_phase"] + model["popcount_filter_pipeline_cycles"],
        model["weight_phase_bytes"] // model["dram_bytes_per_cycle"] +
        model["dma_command_setup_cycles"])
    time = preprocess
    components = Counter()
    components["initial_preprocess"] += preprocess
    for index, phase in enumerate(phases):
        issues = phase["by_b"][b_value]["zero"]["issues"]
        compute = issues * model["output_blocks"]
        next_preprocess = preprocess if index + 1 < len(phases) else 0
        elapsed = max(compute, next_preprocess)
        time += elapsed + model["tail_cycles"]
        components["active_compute"] += compute
        components["preprocess_hidden_or_compute"] += elapsed
        components["tail"] += model["tail_cycles"]
        components["source_output_block_issues"] += (
            issues * model["output_blocks"])
    time += model["commit_cycles_per_sample"]
    components["commit"] += model["commit_cycles_per_sample"]
    return int(time), components


def catalog_sample(phases, model, b_value):
    time = 0
    components = Counter()
    maximum_slot = 0
    for phase in phases:
        config_data = int(math.ceil(
            model["elastic_config_bytes"] /
            float(model["dram_bytes_per_cycle"])))
        time += config_data + model["dma_command_setup_cycles"]
        time += phase["q32_early_matcher_cycles"] + 1
        components["config_data"] += config_data
        components["config_command"] += model["dma_command_setup_cycles"]
        components["matcher"] += phase["q32_early_matcher_cycles"]
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
                model["tile_slot_bytes"], "M458 tile slot overflow")
        require(tile_bytes % model["dram_bytes_per_cycle"] == 0,
                "M458 unaligned tile DMA")
        tile_data = tile_bytes // model["dram_bytes_per_cycle"]
        tile_commands = 1 + phase["used_center_runs"]
        tile_dma = (tile_data + tile_commands *
                    model["dma_command_setup_cycles"])
        pwp_issues = phase["by_b"][b_value]["pwp"]["issues"]
        correction_issues = phase["by_b"][b_value]["correction"]["issues"]
        work = model["output_blocks_per_tile"] * (
            pwp_issues + correction_issues)
        replay = work + model["descriptor_sram_latency_cycles"]
        time += tile_dma
        tile0_end = time + replay
        tile1_dma_end = time + tile_dma
        tile1_start = max(tile0_end, tile1_dma_end)
        exposed = max(0, tile1_dma_end - tile0_end)
        time = tile1_start + replay + model["tail_cycles"]

        components["tile0_dma_data"] += tile_data
        components["tile0_dma_commands"] += (
            tile_commands * model["dma_command_setup_cycles"])
        components["tile1_dma_exposed"] += exposed
        components["replay0"] += replay
        components["replay1"] += replay
        components["active_compute"] += 2 * work
        components["descriptor_sram_startup"] += (
            2 * model["descriptor_sram_latency_cycles"])
        components["tail"] += model["tail_cycles"]
        components["pwp_dram_physical_bytes"] += (
            phase["used_pwp_patterns"] *
            model["elastic_center_stride_bytes"] * 2)
        components["weight_dram_bytes"] += (
            model["weight_bytes_per_tile"] * 2)
        components["pwp_output_block_issues"] += (
            pwp_issues * model["output_blocks"])
        components["correction_output_block_issues"] += (
            correction_issues * model["output_blocks"])
        components["pwp_destination_updates"] += (
            phase["pwp_rows"] * model["output_blocks"])
        components["correction_destination_updates"] += (
            phase["correction_ops_per_block"] * model["output_blocks"])
        components["pwp_descriptor_chunks_before_output_block_expansion"] += (
            pwp_issues)
        components[
            "correction_descriptor_chunks_before_output_block_expansion"] += (
                correction_issues)
    time += model["commit_cycles_per_sample"]
    components["commit"] += model["commit_cycles_per_sample"]
    return int(time), components, maximum_slot


def aggregate_group_ledgers(phases, b_value, kind):
    total = Counter()
    remainders = Counter()
    for phase in phases:
        total.update(phase["by_b"][b_value][kind])
        remainders.update(phase["by_b"][b_value][kind + "_remainders"])
    return total, remainders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M458 output overwrite")

    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract_start = sha256(args.contract)
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m458_exact_destination_multicast_matched_resource_contract_v1" and
            contract.get("status") ==
            "FROZEN_BEFORE_SINGLE_PASS_M40_TRACE",
            "M458 contract drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M458 input SHA drift: " + name)
        paths[name] = path
        identities[name] = dict(spec)
    require(paths["analyzer"].resolve() == source_path and
            identities["analyzer"]["sha256"] == source_start,
            "M458 analyzer self-identity drift")

    seal_checks = {
        "m430_train_files": verify_double_seal(
            paths["m430_catalog"].parent, paths["m430_train_manifest"],
            paths["m430_train_seal"], "M430 train"),
        "m430_heldout_files": verify_double_seal(
            paths["m430_result"].parent, paths["m430_manifest"],
            paths["m430_seal"], "M430 heldout"),
        "m435_review_files": verify_double_seal(
            paths["m435_review"].parent, paths["m435_manifest"],
            paths["m435_seal"], "M435 review"),
    }
    catalog = strict_json(paths["m430_catalog"])
    m430 = strict_json(paths["m430_result"])
    m435 = strict_json(paths["m435_review"])
    trace = strict_json(paths["m40_trace"])
    require(catalog["status"] ==
            "PASS_M430_TRAIN_ONLY_DUALAWARE_Q32_FROZEN_BEFORE_HELDOUT" and
            m430["status"] ==
            "PASS_M430B_ONE_COMPLETED_M40_HELDOUT_DUAL_REPLAY" and
            m435["status"] == "PASS_WITH_RESOURCE_QUALIFICATION" and
            m435["severity_counts"]["P0"] == 0,
            "M458 upstream admission drift")
    require(m430["comparisons"]["strong_zero_cycles"] ==
            contract["reconciliation"]["strong_zero_b1_cycles"] and
            m430["comparisons"]["m430_catalog_dual_cycles"] ==
            contract["reconciliation"]["m430_catalog_b1_cycles"],
            "M458 upstream cycle drift")
    require(trace["identity"]["checkpoint_sha256"] ==
            contract["paper_identity"]["checkpoint_sha256"] and
            trace["identity"]["bn_policy"] ==
            contract["paper_identity"]["bn_policy"],
            "M458 checkpoint/BN identity drift")

    sealed_phases = {}
    with paths["m430_phase_csv"].open(
            "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["sample"]), int(row["operator"]),
                   int(row["partition"]))
            require(key not in sealed_phases, "duplicate sealed M430 phase")
            sealed_phases[key] = row
    require(len(sealed_phases) == 17280, "M458 sealed phase extent drift")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    marker = args.output_dir / "M458_M40_SINGLE_PASS_CONSUMED.marker"
    marker.write_text(
        "M458 single-pass M40 payload consumption marker.\n"
        "Created before first raw payload read.\n"
        "Analyzer SHA256: {}\nContract SHA256: {}\n"
        "M430 catalog remains immutable; failure cannot authorize tuning.\n".format(
            source_start, contract_start), encoding="utf-8")

    b_values = tuple(int(value) for value in contract["architecture"]["B"])
    require(b_values == (1, 2, 4, 8), "M458 B-axis drift")
    operators = tuple(trace["cohort"]["operators"])
    require(tuple(catalog["geometry"]["operators"]) == operators and
            len(operators) == 4, "M458 operator order drift")
    operator_index = dict((name, index) for index, name in enumerate(operators))
    require(len(trace["records"]) == 40, "M458 record extent drift")

    reader = PayloadReader()
    all_phases = []
    phases_by_sample = dict((sample, []) for sample in range(10))
    phase_rows = []
    conservation = Counter()
    seen_records = set()
    for record_index, record in enumerate(trace["records"]):
        sample = int(record["sample_id"])
        operator = operator_index[record["operator"]]
        require((sample, operator) not in seen_records,
                "M458 duplicate sample/operator record")
        seen_records.add((sample, operator))
        packed = reader.read_once(
            paths["m40_trace"].parent / record["packed_file"],
            record["packed_file_sha256"], "packed", sample, operator)
        # Value payload is an identity input only.  Read and hash it exactly
        # once, but do not use values to tune or alter the sealed catalog.
        reader.read_once(
            paths["m40_trace"].parent / record["value_payload_file"],
            record["value_payload_sha256"], "value_zlib_identity_only",
            sample, operator)
        masks = unpack_record_masks_from_bytes(record, packed)
        for partition in range(PARTITIONS):
            tile, subtile = divmod(partition, 16)
            shift = subtile * 16
            values = [
                (masks[destination * TILES + tile] >> shift) & MASK16
                for destination in range(ROWS)]
            centers = [int(value, 16) for value in
                       catalog["operators"][operator]["partitions"]
                       [partition]["nested_patterns"]]
            sealed = sealed_phases[(sample, operator, partition)]
            phase, by_b, audit = analyze_phase(
                values, centers, sealed, b_values)
            phase["sample"] = sample
            phase["operator"] = operator
            phase["partition"] = partition
            phase["by_b"] = by_b
            all_phases.append(phase)
            phases_by_sample[sample].append(phase)
            conservation.update(audit)
            phase_row = {
                "sample": sample,
                "operator": operator,
                "partition": partition,
                "active_rows": phase["active_rows"],
                "eligible_rows": phase["eligible_rows"],
                "pwp_rows": phase["pwp_rows"],
                "fallback_rows": phase["fallback_rows"],
                "correction_ops_per_block":
                    phase["correction_ops_per_block"],
                "bit_sparse_ops_per_block":
                    phase["bit_sparse_vector_ops_per_block"],
                "used_pwp_patterns": phase["used_pwp_patterns"],
                "used_center_runs": phase["used_center_runs"],
                "early_matcher": phase["q32_early_matcher_cycles"],
            }
            for b_value in b_values:
                phase_row["zero_issues_b{}".format(b_value)] = (
                    by_b[b_value]["zero"]["issues"])
                phase_row["pwp_issues_b{}".format(b_value)] = (
                    by_b[b_value]["pwp"]["issues"])
                phase_row["correction_issues_b{}".format(b_value)] = (
                    by_b[b_value]["correction"]["issues"])
            phase_rows.append(phase_row)
        print("[M458 TRACE] record={}/40 sample={} operator={}".format(
            record_index + 1, sample, operator), flush=True)

    require(len(seen_records) == 40 and len(all_phases) == 17280 and
            all(len(phases_by_sample[sample]) == 1728
                for sample in range(10)),
            "M458 completed phase extent mismatch")
    for sample in range(10):
        phases_by_sample[sample].sort(
            key=lambda phase: (phase["operator"], phase["partition"]))
    require(len(reader.audit) == 80 and len(reader.seen) == 80,
            "M458 payload single-read count mismatch")
    require(conservation["destinations_checked"] == 51840000 and
            conservation["zero_destination_update_count_mismatches"] == 0 and
            conservation["catalog_destination_update_count_mismatches"] == 0 and
            conservation["exact_reconstruction_mismatches"] == 0 and
            conservation["sealed_phase_field_mismatches"] == 0,
            "M458 global destination conservation mismatch")

    model = dict(contract["cycle_model"])
    per_b = []
    component_ledgers = {}
    group_ledgers = {}
    for b_value in b_values:
        zero_cycles = 0
        catalog_cycles = 0
        zero_components = Counter()
        catalog_components = Counter()
        maximum_slot = 0
        for sample in range(10):
            value, components = baseline_sample(
                phases_by_sample[sample], model, b_value)
            zero_cycles += value
            zero_components.update(components)
            value, components, slot = catalog_sample(
                phases_by_sample[sample], model, b_value)
            catalog_cycles += value
            catalog_components.update(components)
            maximum_slot = max(maximum_slot, slot)

        kinds = {}
        for kind in ("zero", "pwp", "correction"):
            ledger, remainders = aggregate_group_ledgers(
                all_phases, b_value, kind)
            ledger["utilization"] = (
                ledger["useful_destination_updates"] /
                float(ledger["capacity_destination_slots"])
                if ledger["capacity_destination_slots"] else 1.0)
            ledger["remainder_histogram"] = dict(
                (str(key), value) for key, value in sorted(remainders.items()))
            kinds[kind] = dict(ledger)
        group_ledgers[str(b_value)] = kinds
        component_ledgers[str(b_value)] = {
            "strong_zero": dict(zero_components),
            "m430_catalog": dict(catalog_components),
            "maximum_tile_slot_bytes": maximum_slot,
        }
        per_b.append({
            "B": b_value,
            "strong_zero_cycles": zero_cycles,
            "m430_catalog_cycles": catalog_cycles,
            "equal_B_catalog_speedup_vs_strong_zero":
                zero_cycles / float(catalog_cycles),
            "strong_zero_source_output_block_issues":
                zero_components["source_output_block_issues"],
            "catalog_pwp_output_block_issues":
                catalog_components["pwp_output_block_issues"],
            "catalog_correction_output_block_issues":
                catalog_components["correction_output_block_issues"],
            "zero_source_onchip_read_bytes":
                zero_components["source_output_block_issues"] *
                model["correction_bytes_per_issue"],
            "catalog_pwp_logical_onchip_read_bytes":
                catalog_components["pwp_output_block_issues"] *
                model["dual_pwp_logical_bytes_per_issue"],
            "catalog_pwp_padded_signal_bytes":
                catalog_components["pwp_output_block_issues"] *
                model["dual_pwp_padded_signal_bytes_per_issue"],
            "catalog_correction_onchip_read_bytes":
                catalog_components["correction_output_block_issues"] *
                model["correction_bytes_per_issue"],
            "pwp_dram_physical_bytes":
                catalog_components["pwp_dram_physical_bytes"],
            "weight_dram_bytes": catalog_components["weight_dram_bytes"],
            "zero_utilization": kinds["zero"]["utilization"],
            "pwp_utilization": kinds["pwp"]["utilization"],
            "correction_utilization": kinds["correction"]["utilization"],
            "zero_wasted_slots": kinds["zero"]["wasted_destination_slots"],
            "catalog_wasted_slots": (
                kinds["pwp"]["wasted_destination_slots"] +
                kinds["correction"]["wasted_destination_slots"]),
        })

    require(per_b[0]["strong_zero_cycles"] ==
            contract["reconciliation"]["strong_zero_b1_cycles"] and
            per_b[0]["m430_catalog_cycles"] ==
            contract["reconciliation"]["m430_catalog_b1_cycles"],
            "M458 B1 full cycle reconciliation mismatch")
    require(component_ledgers["1"]["m430_catalog"][
                "pwp_output_block_issues"] ==
            m430["traffic_and_port_ledger"]["pwp_output_block_issues"] and
            component_ledgers["1"]["m430_catalog"][
                "correction_output_block_issues"] ==
            m430["traffic_and_port_ledger"][
                "correction_output_block_issues"],
            "M458 B1 traffic reconciliation mismatch")

    base_zero = per_b[0]["strong_zero_cycles"]
    base_catalog = per_b[0]["m430_catalog_cycles"]
    base_advantage = per_b[0]["equal_B_catalog_speedup_vs_strong_zero"]
    for row in per_b:
        row["strong_zero_speedup_vs_B1"] = (
            base_zero / float(row["strong_zero_cycles"]))
        row["catalog_speedup_vs_B1"] = (
            base_catalog / float(row["m430_catalog_cycles"]))
        row["equal_B_catalog_advantage_ratio_vs_B1"] = (
            row["equal_B_catalog_speedup_vs_strong_zero"] / base_advantage)
        # Diagnostic lower bound using only the explicit B-fold accumulator
        # bank multiplier.  Fixed logic is omitted, so this is not real area.
        row["catalog_throughput_per_accumulator_bank_proxy_vs_B1"] = (
            row["catalog_speedup_vs_B1"] / row["B"])

    candidates = [row for row in per_b if row["B"] > 1]
    best_advantage = max(candidates,
                         key=lambda row: row[
                             "equal_B_catalog_advantage_ratio_vs_B1"])
    best_proxy = max(candidates,
                     key=lambda row: row[
                         "catalog_throughput_per_accumulator_bank_proxy_vs_B1"])
    improves = best_advantage[
        "equal_B_catalog_advantage_ratio_vs_B1"] > 1.0
    material = best_advantage[
        "equal_B_catalog_advantage_ratio_vs_B1"] >= contract[
            "decision_rule"][
                "minimum_equal_B_advantage_ratio_vs_B1_for_rtl"]
    proxy_noninferior = best_proxy[
        "catalog_throughput_per_accumulator_bank_proxy_vs_B1"] >= 1.0
    resource_evidence = contract["decision_rule"][
        "matched_area_power_evidence_available_before_run"]
    if not improves:
        decision = "NO_GO_B_GT1_EQUAL_B_CATALOG_ADVANTAGE_NOT_IMPROVED"
    elif not material:
        decision = "NO_GO_B_GT1_EQUAL_B_ADVANTAGE_GAIN_IMMATERIAL"
    elif not proxy_noninferior or not resource_evidence:
        decision = "NO_GO_B_GT1_NO_DEFENSIBLE_THROUGHPUT_AREA"
    else:
        decision = "GO_B_GT1_RTL"

    per_b_fields = list(per_b[0].keys())
    write_csv(args.output_dir / "m458_per_B.csv", per_b, per_b_fields)
    phase_fields = list(phase_rows[0].keys())
    write_csv(args.output_dir / "m458_per_phase.csv", phase_rows, phase_fields)
    write_csv(args.output_dir / "m458_payload_read_audit.csv", reader.audit,
              ["read_ordinal", "sample", "operator", "role", "path",
               "bytes", "expected_sha256", "actual_sha256", "read_count"])

    immutable_end = {
        "analyzer": sha256(source_path),
        "contract": sha256(args.contract),
        "m40_trace": sha256(paths["m40_trace"]),
        "m430_catalog": sha256(paths["m430_catalog"]),
        "m430_result": sha256(paths["m430_result"]),
        "docs359": sha256(paths["docs359"]),
    }
    require(immutable_end["analyzer"] == source_start and
            immutable_end["contract"] == contract_start and
            all(immutable_end[name] == identities[name]["sha256"]
                for name in ("m40_trace", "m430_catalog", "m430_result",
                             "docs359")),
            "M458 immutable input changed during run")

    result = {
        "schema": "m458_exact_destination_multicast_matched_resource_v1",
        "status": "PASS_M458_SINGLE_PASS_EXACT_MULTICAST_SCREEN",
        "decision": decision,
        "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
        "paper_identity": contract["paper_identity"],
        "input_identity": identities,
        "immutable_end_sha256": immutable_end,
        "upstream_seal_checks": seal_checks,
        "architecture": contract["architecture"],
        "cycle_model": model,
        "per_B": per_b,
        "group_ledgers": group_ledgers,
        "component_ledgers": component_ledgers,
        "destination_conservation": {
            "phases_checked": 17280,
            "destinations_checked": conservation["destinations_checked"],
            "destination_output_block_contexts_checked":
                conservation["destinations_checked"] * OUTPUT_BLOCKS,
            "zero_destination_update_count_mismatches":
                conservation["zero_destination_update_count_mismatches"],
            "catalog_destination_update_count_mismatches":
                conservation["catalog_destination_update_count_mismatches"],
            "exact_reconstruction_mismatches":
                conservation["exact_reconstruction_mismatches"],
            "sealed_phase_field_mismatches":
                conservation["sealed_phase_field_mismatches"],
            "persistent_old_psum_rule":
                "For every destination and every output block, each PWP or signed correction/fallback contribution performs exactly one independent old_psum[d,ob] += delta update; destinations are never algebraically fused.",
            "all_B_tail_rule":
                "Every nonempty group uses ceil(count/B) chunks; full*B+tail=count and useful updates are invariant for B=1,2,4,8.",
        },
        "payload_audit": {
            "marker_created_before_first_payload_read": True,
            "files_read_exactly_once": len(reader.audit),
            "unique_files": len(reader.seen),
            "bytes_read": sum(row["bytes"] for row in reader.audit),
            "packed_files": sum(row["role"] == "packed" for row in reader.audit),
            "value_identity_files": sum(
                row["role"] == "value_zlib_identity_only"
                for row in reader.audit),
            "catalog_tuning_after_trace": False,
        },
        "reconciliation": {
            "B1_all_17280_sealed_phase_fields_match": True,
            "B1_strong_zero_cycles": per_b[0]["strong_zero_cycles"],
            "B1_m430_catalog_cycles": per_b[0]["m430_catalog_cycles"],
            "B1_pwp_output_block_issues":
                component_ledgers["1"]["m430_catalog"][
                    "pwp_output_block_issues"],
            "B1_correction_output_block_issues":
                component_ledgers["1"]["m430_catalog"][
                    "correction_output_block_issues"],
            "mismatches": 0,
        },
        "decision_evidence": {
            "B1_equal_B_catalog_speedup": base_advantage,
            "best_B_gt1_advantage_row": best_advantage,
            "best_B_gt1_bank_proxy_row": best_proxy,
            "strict_equal_B_catalog_advantage_improves": improves,
            "material_advantage_threshold_passes": material,
            "bank_multiplier_proxy_noninferior": proxy_noninferior,
            "matched_area_power_evidence_available": resource_evidence,
            "rule": contract["decision_rule"],
        },
        "resource_disclosure": {
            "source_sram_read_ports":
                "one source/PWP read per service issue; no B-fold source SRAM port expansion",
            "new_resources_for_B":
                "B independent destination accumulator banks/update ports per active 96-lane output block, destination gather/list storage, and B-way broadcast/crossbar",
            "output_block_rule":
                "Eight output blocks remain separate services (four per tile); no read is fused across output blocks.",
            "area_power_status":
                "accumulator/gather/crossbar area and power absent; throughput/bank is diagnostic only",
        },
        "admission": {
            "exact_arithmetic": True,
            "checkpoint_or_accuracy_changed": False,
            "accuracy_loss": False,
            "frozen_trace_cycle_screen": True,
            "standalone_four_bottleneck_conv_cycles": True,
            "catalog_tuned_on_m40": False,
            "rtl": False,
            "vcs": False,
            "synopsys": False,
            "physical_sram_or_interconnect": False,
            "resource_normalized_speedup": False,
            "power_or_energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "output_files": {
            "per_B": "m458_per_B.csv",
            "per_phase": "m458_per_phase.csv",
            "payload_read_audit": "m458_payload_read_audit.csv",
            "marker": marker.name,
        },
    }
    result_path = args.output_dir / (
        "m458_exact_destination_multicast_matched_resource_r1.json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    complete = args.output_dir / "RUN_COMPLETE.txt"
    complete.write_text(
        "{}\nB1 zero={} catalog={} best_B_gt1={}\n".format(
            decision, base_zero, base_catalog, best_advantage["B"]),
        encoding="utf-8")
    manifest, seal = write_seal(args.output_dir, [
        marker.name, "m458_per_B.csv", "m458_per_phase.csv",
        "m458_payload_read_audit.csv", result_path.name, complete.name])
    print("M458_PASS decision={} B1_zero={} B1_catalog={} best_B={} "
          "best_advantage_ratio={:.9f}x seal={}".format(
              decision, base_zero, base_catalog, best_advantage["B"],
              best_advantage[
                  "equal_B_catalog_advantage_ratio_vs_B1"],
              sha256(seal)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
