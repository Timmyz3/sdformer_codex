#!/usr/bin/env python3
"""Independent, read-only hammer for the single frozen M453b M40 replay.

This program never imports or executes the M453b analyzer.  It independently
checks all seals, streams every published ledger row, reconstructs every RLE
source row, recomputes the cycle model, and directly reconstructs masks from
sealed M40 payloads for fixed boundary and deterministic random phases.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


REVIEW_CONTRACT_SHA256 = (
    "cbae40c60bce601788ebf0d0bdc5168c437af4618695e10362a2b25428be6702")
EXPECTED_SUBJECT = {
    "final_contract":
        "3292e54ef0bf64b96b421de0d5b374a1552573bf29868e5f218fdaeac1bd2c4f",
    "analyzer":
        "84c3a2c79ad5926ba72a4727aac64b6ca29b7530a3120ada350d68d8ce12f6ca",
    "result":
        "df0f37c0e0321a74ff9018f3aad8ab94abeb7babf3285152a9301e561cc22bbd",
    "inner_manifest":
        "113b6d7b336691eab90566f3f7b93c48eff9edaa0282c9ddfc5c9e6e2026b70a",
    "outer_seal":
        "4d8096b0fa3e68b3403a1b1377d77b5bcd2a41b3ca89424951401fe0a1cc2079",
    "micro_manifest":
        "462ac5886d7453856f36e64116426c288b79a964f2f243854fd6c08043e603fb",
    "micro_outer_seal":
        "3e81ee1ae67f7c86c483a2673d8e184818115b203104fd2c22d9df768444d8c2",
    "docs359":
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
DISTANCE_FIELDS = tuple("distance{}_rows".format(i) for i in range(17))
PHASE_FIELDS = (
    "source_rows", "zero_rows", "active_rows", "pwp_rows",
    "fallback_rows", "exact_pwp_rows", "positive_residual_pwp_rows",
    "pwp_correction_ops", "fallback_source_ops",
    "correction_ops_per_block", "separate_issues_per_block",
    "fused_k1_issues_per_block", "parent_selected_rows",
    "child_selected_rows", "used_pwp_patterns",
    "used_parent_pwp_patterns", "used_child_pwp_patterns",
    "used_center_runs", "triangle_child_comparisons_potential",
    "triangle_child_comparisons_gated",
    "triangle_child_comparisons_executed",
    "triangle_selection_mismatches", "q32_parent_matcher_cycles",
    "child_matcher_pipeline_latency_cycles", "hierarchical_matcher_cycles",
    "actual_pwp_bytes_per_tile", "actual_slot_bytes",
    "current_slot_overflow") + DISTANCE_FIELDS
POP16 = tuple(bin(value).count("1") for value in range(1 << 16))


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


def verify_manifest(manifest_path, seal_path, label):
    manifest_path = Path(manifest_path)
    seal_path = Path(seal_path)
    entries = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        require(line and "  " in line, label + " malformed manifest")
        expected, name = line.split("  ", 1)
        target = Path(name)
        if not target.is_absolute():
            target = manifest_path.parent / target
        require(target.is_file(), label + " missing entry: " + name)
        require(sha256(target) == expected,
                label + " inner mismatch: " + name)
        entries += 1
    require(entries > 0, label + " empty manifest")
    seal_line = seal_path.read_text(encoding="utf-8").strip()
    require("  " in seal_line, label + " malformed outer seal")
    expected, name = seal_line.split("  ", 1)
    require(Path(name).name == manifest_path.name,
            label + " outer seal target drift")
    require(expected == sha256(manifest_path),
            label + " outer manifest hash mismatch")
    return {
        "entries": entries,
        "manifest_sha256": sha256(manifest_path),
        "outer_seal_file_sha256": sha256(seal_path),
    }


def phase_key(parts):
    return (int(parts[0]), int(parts[1]), int(parts[3]))


def expected_phase_keys():
    return [(sample, operator, partition)
            for sample in range(10)
            for operator in range(4)
            for partition in range(432)]


def stream_phases(path, operators):
    phases = []
    aggregate = Counter()
    expected_header = ["sample", "operator", "operator_name", "partition"] + \
        list(PHASE_FIELDS)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        require(header == expected_header, "phase CSV header drift")
        for line_number, line in enumerate(handle, 2):
            parts = line.rstrip("\r\n").split(",")
            require(len(parts) == len(header),
                    "phase CSV width drift at {}".format(line_number))
            key = phase_key(parts)
            require(parts[2] == operators[key[1]],
                    "phase operator identity drift")
            values = {field: int(parts[4 + index])
                      for index, field in enumerate(PHASE_FIELDS)}
            require(values["source_rows"] == 3000,
                    "phase source extent drift")
            require(values["source_rows"] ==
                    values["zero_rows"] + values["active_rows"],
                    "phase zero/active conservation drift")
            require(values["active_rows"] ==
                    values["pwp_rows"] + values["fallback_rows"],
                    "phase pwp/fallback conservation drift")
            require(values["pwp_rows"] == values["exact_pwp_rows"] +
                    values["positive_residual_pwp_rows"],
                    "phase PWP split drift")
            require(values["correction_ops_per_block"] ==
                    values["pwp_correction_ops"] +
                    values["fallback_source_ops"],
                    "phase correction conservation drift")
            require(values["separate_issues_per_block"] ==
                    values["pwp_rows"] +
                    values["correction_ops_per_block"],
                    "phase separate issue conservation drift")
            require(values["fused_k1_issues_per_block"] ==
                    values["separate_issues_per_block"] -
                    values["positive_residual_pwp_rows"],
                    "phase fused diagnostic conservation drift")
            require(values["parent_selected_rows"] == values["active_rows"],
                    "phase parent selection population drift")
            require(sum(values[field] for field in DISTANCE_FIELDS) ==
                    values["active_rows"],
                    "phase distance histogram drift")
            require(values["triangle_child_comparisons_potential"] ==
                    3 * values["active_rows"] and
                    values["triangle_child_comparisons_potential"] ==
                    values["triangle_child_comparisons_gated"] +
                    values["triangle_child_comparisons_executed"] and
                    values["triangle_selection_mismatches"] == 0,
                    "phase triangle conservation drift")
            require(values["child_matcher_pipeline_latency_cycles"] == 2 and
                    values["hierarchical_matcher_cycles"] ==
                    values["q32_parent_matcher_cycles"] + 2,
                    "phase matcher cycle drift")
            require(values["actual_pwp_bytes_per_tile"] ==
                    640 * values["used_pwp_patterns"] and
                    values["actual_slot_bytes"] ==
                    288 + 6144 + values["actual_pwp_bytes_per_tile"] and
                    values["current_slot_overflow"] ==
                    int(values["actual_slot_bytes"] > 32768),
                    "phase slot equation drift")
            phases.append((key, values))
            aggregate.update(values)
    require([key for key, _ in phases] == expected_phase_keys(),
            "phase key/order/extent drift")
    return phases, aggregate


def catalog_geometry(catalog):
    geometry = {}
    for op, operator in enumerate(catalog["operators"]):
        require(len(operator["partitions"]) == 432,
                "catalog partition extent drift")
        for partition, item in enumerate(operator["partitions"]):
            parents = tuple(int(value, 16)
                            for value in item["parent_patterns"])
            children = tuple(tuple(int(value, 16) for value in group)
                             for group in item["children_by_parent"])
            flat = tuple(int(value, 16) for value in item["flat_patterns"])
            require(len(parents) == 32 and len(children) == 32 and
                    all(len(group) == 3 for group in children) and
                    len(flat) == 128 and len(set(flat)) == 128 and
                    flat[:32] == parents and
                    flat[32:] == tuple(value for group in children
                                      for value in group),
                    "catalog geometry/uniqueness drift")
            geometry[(op, partition)] = (parents, children, flat)
    return geometry


def compare_phase_summary(label, summary, phase):
    fields = (
        "source_rows", "zero_rows", "active_rows", "pwp_rows",
        "fallback_rows", "exact_pwp_rows", "positive_residual_pwp_rows",
        "pwp_correction_ops", "fallback_source_ops",
        "correction_ops_per_block", "separate_issues_per_block",
        "fused_k1_issues_per_block", "parent_selected_rows",
        "child_selected_rows", "used_pwp_patterns",
        "used_parent_pwp_patterns", "used_child_pwp_patterns",
        "used_center_runs", "triangle_child_comparisons_potential",
        "triangle_child_comparisons_gated",
        "triangle_child_comparisons_executed",
        "triangle_selection_mismatches", "q32_parent_matcher_cycles") + \
        DISTANCE_FIELDS
    for field in fields:
        require(int(summary.get(field, 0)) == int(phase[field]),
                "{} phase mismatch for {}: {} != {}".format(
                    label, field, summary.get(field, 0), phase[field]))


def stream_centers(path, phase_map, geometry, operators):
    expected_prefix = [
        "sample", "operator", "operator_name", "partition",
        "global_center_id", "parent_id", "child_slot", "center_hex",
        "parent_hex", "parent_child_hamming", "selected_rows", "pwp_rows",
        "exact_pwp_rows", "positive_residual_pwp_rows", "fallback_rows",
        "pwp_correction_ops", "fallback_source_ops",
        "separate_issues_per_block", "fused_k1_issues_per_block",
        "selected_hamming_flip_terms", "pwp_residual_flip_terms"]
    expected_header = expected_prefix + list(DISTANCE_FIELDS)
    total = Counter()
    rows = 0
    current_key = None
    current = Counter()
    last_gid = -1
    used = set()

    def finish():
        if current_key is None:
            return
        current["source_rows"] = 3000
        current["zero_rows"] = 3000 - current["selected_rows"]
        current["active_rows"] = current["selected_rows"]
        current["correction_ops_per_block"] = (
            current["pwp_correction_ops"] + current["fallback_source_ops"])
        current["parent_selected_rows"] = current["selected_rows"]
        current["used_pwp_patterns"] = len(used)
        current["used_parent_pwp_patterns"] = sum(gid < 32 for gid in used)
        current["used_child_pwp_patterns"] = sum(gid >= 32 for gid in used)
        ordered = sorted(used)
        current["used_center_runs"] = (0 if not ordered else
            1 + sum(b != a + 1 for a, b in zip(ordered, ordered[1:])))
        phase = phase_map[current_key]
        # Triangle fields are not present in the center ledger; preserve the
        # phase values only outside this center-conservation comparison.
        for field in ("triangle_child_comparisons_potential",
                      "triangle_child_comparisons_gated",
                      "triangle_child_comparisons_executed",
                      "triangle_selection_mismatches",
                      "q32_parent_matcher_cycles"):
            current[field] = phase[field]
        compare_phase_summary("center", current, phase)

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        require(header == expected_header, "center CSV header drift")
        for line_number, line in enumerate(handle, 2):
            parts = line.rstrip("\r\n").split(",")
            require(len(parts) == len(header),
                    "center CSV width drift at {}".format(line_number))
            key = phase_key(parts)
            if key != current_key:
                finish()
                current_key = key
                current = Counter()
                last_gid = -1
                used = set()
            require(key in phase_map and parts[2] == operators[key[1]],
                    "center phase/operator identity drift")
            gid = int(parts[4])
            parent = int(parts[5])
            child = int(parts[6])
            center = int(parts[7], 16)
            parent_value = int(parts[8], 16)
            parent_child_hamming = int(parts[9])
            require(0 <= gid < 128 and gid > last_gid,
                    "center ID order/uniqueness drift")
            last_gid = gid
            expected_parent = gid if gid < 32 else (gid - 32) // 3
            expected_child = -1 if gid < 32 else (gid - 32) % 3
            parents, children, flat = geometry[(key[1], key[2])]
            require(parent == expected_parent and child == expected_child and
                    center == flat[gid] and parent_value == parents[parent] and
                    parent_child_hamming ==
                    (0 if child < 0 else
                     POP16[parents[parent] ^ children[parent][child]]),
                    "center catalog identity/hamming drift")
            selected = int(parts[10])
            pwp = int(parts[11])
            exact = int(parts[12])
            positive = int(parts[13])
            fallback = int(parts[14])
            correction = int(parts[15])
            fallback_ops = int(parts[16])
            separate = int(parts[17])
            fused = int(parts[18])
            flip_terms = int(parts[19])
            residual_terms = int(parts[20])
            distances = [int(value) for value in parts[21:38]]
            require(selected > 0 and selected == sum(distances) and
                    selected == pwp + fallback and pwp == exact + positive and
                    exact <= distances[0] and
                    separate == pwp + correction + fallback_ops and
                    fused == separate - positive and
                    flip_terms == sum(i * value
                                      for i, value in enumerate(distances)) and
                    residual_terms == correction,
                    "center issue/distance conservation drift")
            current["selected_rows"] += selected
            current["pwp_rows"] += pwp
            current["exact_pwp_rows"] += exact
            current["positive_residual_pwp_rows"] += positive
            current["fallback_rows"] += fallback
            current["pwp_correction_ops"] += correction
            current["fallback_source_ops"] += fallback_ops
            current["separate_issues_per_block"] += separate
            current["fused_k1_issues_per_block"] += fused
            current["child_selected_rows"] += selected * int(child >= 0)
            for index, field in enumerate(DISTANCE_FIELDS):
                current[field] += distances[index]
            if pwp:
                used.add(gid)
            total.update({
                "selected_rows": selected,
                "pwp_rows": pwp,
                "exact_pwp_rows": exact,
                "positive_residual_pwp_rows": positive,
                "fallback_rows": fallback,
                "pwp_correction_ops": correction,
                "fallback_source_ops": fallback_ops,
                "separate_issues_per_block": separate,
                "fused_k1_issues_per_block": fused,
            })
            for index, field in enumerate(DISTANCE_FIELDS):
                total[field] += distances[index]
            rows += 1
    finish()
    require(rows == 1914090, "center row extent drift")
    return rows, total


def selected_phase_set():
    selected = {
        (0, 0, 0), (0, 0, 431), (0, 3, 0),
        (9, 0, 431), (9, 3, 0), (9, 3, 431),
    }
    rng = random.Random(453002)
    while len(selected) < 18:
        selected.add((rng.randrange(10), rng.randrange(4), rng.randrange(432)))
    return tuple(sorted(selected))


def stream_ordered(path, phase_map, geometry, operators, selected_phases):
    expected_header = [
        "sample", "operator", "operator_name", "partition", "run_index",
        "source_row_start", "source_row_count", "source_row_end_exclusive",
        "original_mask_hex", "selected_global_id", "parent_id", "child_slot",
        "selected_center_hex", "selected_distance", "path",
        "pwp_correction_ops_per_row", "fallback_source_ops_per_row",
        "separate_issues_per_block_per_row",
        "fused_issues_per_block_per_row"]
    total = Counter()
    rows = 0
    reconstructed_rows = 0
    current_key = None
    current = Counter()
    expected_run = 0
    expected_start = 0
    used = set()
    selected_masks = defaultdict(list)
    selected_descriptors = defaultdict(list)
    observed_keys = []
    parent_prefix_sets = {
        key: frozenset(values[0][:16]) for key, values in geometry.items()
    }

    def finish():
        if current_key is None:
            return
        require(expected_start == 3000,
                "ordered phase source extent drift: {}".format(current_key))
        current["source_rows"] = expected_start
        current["parent_selected_rows"] = current["active_rows"]
        current["correction_ops_per_block"] = (
            current["pwp_correction_ops"] + current["fallback_source_ops"])
        current["used_pwp_patterns"] = len(used)
        current["used_parent_pwp_patterns"] = sum(gid < 32 for gid in used)
        current["used_child_pwp_patterns"] = sum(gid >= 32 for gid in used)
        ordered = sorted(used)
        current["used_center_runs"] = (0 if not ordered else
            1 + sum(b != a + 1 for a, b in zip(ordered, ordered[1:])))
        current["triangle_child_comparisons_potential"] = (
            current["triangle_child_comparisons_gated"] +
            current["triangle_child_comparisons_executed"])
        current["q32_parent_matcher_cycles"] = (
            3000 + current["q32_early_extra_prefix_tasks"] + 2)
        compare_phase_summary("ordered", current, phase_map[current_key])

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        require(header == expected_header, "ordered CSV header drift")
        for line_number, line in enumerate(handle, 2):
            parts = line.rstrip("\r\n").split(",")
            require(len(parts) == 19,
                    "ordered CSV width drift at {}".format(line_number))
            key = phase_key(parts)
            if key != current_key:
                finish()
                observed_keys.append(key)
                current_key = key
                current = Counter()
                expected_run = 0
                expected_start = 0
                used = set()
            require(key in phase_map and parts[2] == operators[key[1]],
                    "ordered phase/operator identity drift")
            run_index = int(parts[4])
            start = int(parts[5])
            count = int(parts[6])
            end = int(parts[7])
            require(run_index == expected_run and start == expected_start and
                    count > 0 and end == start + count and end <= 3000,
                    "ordered run index/extent drift")
            expected_run += 1
            expected_start = end
            mask = int(parts[8], 16)
            require(0 <= mask < (1 << 16), "ordered mask width drift")
            gid = int(parts[9])
            parent = int(parts[10])
            child = int(parts[11])
            center = int(parts[12], 16)
            distance = int(parts[13])
            path_name = parts[14]
            pwp_correction = int(parts[15])
            fallback_ops = int(parts[16])
            separate = int(parts[17])
            fused = int(parts[18])
            pop = POP16[mask]
            if mask == 0:
                require(gid == -1 and parent == -1 and child == -1 and
                        center == 0 and distance == 0 and path_name == "zero" and
                        pwp_correction == 0 and fallback_ops == 0 and
                        separate == 0 and fused == 0,
                        "ordered zero descriptor drift")
                current["zero_rows"] += count
            else:
                require(0 <= gid < 128, "ordered selected ID range drift")
                expected_parent = gid if gid < 32 else (gid - 32) // 3
                expected_child = -1 if gid < 32 else (gid - 32) % 3
                parents, children, flat = geometry[(key[1], key[2])]
                require(parent == expected_parent and child == expected_child and
                        center == flat[gid] and distance == POP16[mask ^ center],
                        "ordered ID/center/distance drift")
                use_pwp = 1 + distance < pop
                expected_path = ("exact_pwp" if use_pwp and distance == 0 else
                                 "positive_residual_pwp" if use_pwp else
                                 "fallback")
                expected_correction = distance if use_pwp else 0
                expected_fallback = 0 if use_pwp else pop
                expected_separate = ((1 + distance) if use_pwp else pop)
                expected_fused = (max(1, distance) if use_pwp else pop)
                require(path_name == expected_path and
                        pwp_correction == expected_correction and
                        fallback_ops == expected_fallback and
                        separate == expected_separate and fused == expected_fused,
                        "ordered path/issue formula drift")
                current["active_rows"] += count
                current["pwp_rows"] += count * int(use_pwp)
                current["fallback_rows"] += count * int(not use_pwp)
                current["exact_pwp_rows"] += count * int(
                    use_pwp and distance == 0)
                current["positive_residual_pwp_rows"] += count * int(
                    use_pwp and distance > 0)
                current["pwp_correction_ops"] += count * pwp_correction
                current["fallback_source_ops"] += count * fallback_ops
                current["separate_issues_per_block"] += count * separate
                current["fused_k1_issues_per_block"] += count * fused
                current["child_selected_rows"] += count * int(child >= 0)
                current["distance{}_rows".format(distance)] += count
                if use_pwp:
                    used.add(gid)
                current["q32_early_extra_prefix_tasks"] += count * int(
                    pop >= 2 and mask not in parent_prefix_sets[(key[1], key[2])])
                parent_distance = POP16[mask ^ parents[parent]]
                best = parent_distance
                local_id = 0
                for slot, child_value in enumerate(children[parent]):
                    lower = abs(parent_distance -
                                POP16[parents[parent] ^ child_value])
                    if lower >= best:
                        current["triangle_child_comparisons_gated"] += count
                    else:
                        current["triangle_child_comparisons_executed"] += count
                        candidate = POP16[mask ^ child_value]
                        if candidate < best:
                            best = candidate
                            local_id = slot + 1
                current["triangle_selection_mismatches"] += count * int(
                    best != distance or local_id != (child + 1 if child >= 0 else 0))
            if key in selected_phases:
                selected_masks[key].extend([mask] * count)
                descriptor = (gid, parent, child, center, distance, path_name,
                              pwp_correction, fallback_ops, separate, fused)
                selected_descriptors[key].extend([descriptor] * count)
            total["source_rows"] += count
            total["zero_rows"] += count * int(mask == 0)
            total["active_rows"] += count * int(mask != 0)
            if mask != 0:
                total["pwp_rows"] += count * int(path_name != "fallback")
                total["fallback_rows"] += count * int(path_name == "fallback")
                total["exact_pwp_rows"] += count * int(path_name == "exact_pwp")
                total["positive_residual_pwp_rows"] += count * int(
                    path_name == "positive_residual_pwp")
                total["pwp_correction_ops"] += count * pwp_correction
                total["fallback_source_ops"] += count * fallback_ops
                total["separate_issues_per_block"] += count * separate
                total["fused_k1_issues_per_block"] += count * fused
                total["child_selected_rows"] += count * int(child >= 0)
                total["distance{}_rows".format(distance)] += count
            reconstructed_rows += count
            rows += 1
    finish()
    require(observed_keys == expected_phase_keys(),
            "ordered phase key/order/extent drift")
    require(rows == 29114711 and reconstructed_rows == 51840000,
            "ordered RLE/source population extent drift")
    return rows, reconstructed_rows, total, selected_masks, selected_descriptors


def direct_m40_masks(trace, trace_dir, selected):
    """Reconstruct fixed phases directly from sealed little-endian planes."""
    by_record = defaultdict(set)
    for sample, operator, partition in selected:
        by_record[(sample, operator)].add(partition)
    records = {(int(row["sample_id"]), int(row["operator_index"])): row
               for row in trace["records"]}
    result = {}
    for record_key, partitions in by_record.items():
        record = records[record_key]
        raw = (trace_dir / record["packed_file"]).read_bytes()
        plane_bytes = int(record["positive_plane_bytes"])
        require(len(raw) == 3 * plane_bytes and
                not any(raw[plane_bytes:2 * plane_bytes]),
                "direct M40 packed/negative plane drift")
        positive = raw[:plane_bytes]
        masks = {partition: [0] * 3000 for partition in partitions}
        for byte_index, byte0 in enumerate(positive):
            byte = byte0
            if byte == 0:
                continue
            base = byte_index * 8
            while byte:
                low = byte & -byte
                bit = low.bit_length() - 1
                flat = base + bit
                require(flat < 10 * 768 * 15 * 20,
                        "direct M40 nonzero tail bit")
                tc, spatial = divmod(flat, 15 * 20)
                timestep, channel = divmod(tc, 768)
                input_y, input_x = divmod(spatial, 20)
                feature_base = channel * 9
                for kernel_y in range(3):
                    output_y = input_y - kernel_y + 1
                    if not 0 <= output_y < 15:
                        continue
                    for kernel_x in range(3):
                        output_x = input_x - kernel_x + 1
                        if not 0 <= output_x < 20:
                            continue
                        feature = feature_base + 3 * kernel_y + kernel_x
                        partition, partition_bit = divmod(feature, 16)
                        if partition in masks:
                            row = (timestep * 15 + output_y) * 20 + output_x
                            masks[partition][row] |= 1 << partition_bit
                byte ^= low
        for partition, values in masks.items():
            result[(record_key[0], record_key[1], partition)] = values
    return result


def direct_selection_audit(selected_masks, selected_descriptors,
                           direct_masks, geometry):
    descriptor_mismatches = 0
    mask_mismatches = 0
    checked_rows = 0
    for key in sorted(direct_masks):
        direct = direct_masks[key]
        published = selected_masks[key]
        descriptors = selected_descriptors[key]
        require(len(direct) == len(published) == len(descriptors) == 3000,
                "direct selected phase extent drift")
        parents, children, flat = geometry[(key[1], key[2])]
        for mask, ledger_mask, descriptor in zip(
                direct, published, descriptors):
            mask_mismatches += int(mask != ledger_mask)
            if mask == 0:
                expected = (-1, -1, -1, 0, 0, "zero", 0, 0, 0, 0)
            else:
                parent_distances = [POP16[mask ^ center]
                                    for center in parents]
                parent = parent_distances.index(min(parent_distances))
                local = (parents[parent],) + children[parent]
                distances = [POP16[mask ^ center] for center in local]
                local_id = distances.index(min(distances))
                center = local[local_id]
                distance = distances[local_id]
                gid = (parent if local_id == 0 else
                       32 + parent * 3 + local_id - 1)
                require(flat[gid] == center,
                        "direct catalog flat/local mapping drift")
                pop = POP16[mask]
                use = 1 + distance < pop
                path = ("exact_pwp" if use and distance == 0 else
                        "positive_residual_pwp" if use else "fallback")
                correction = distance if use else 0
                fallback = 0 if use else pop
                separate = 1 + distance if use else pop
                fused = max(1, distance) if use else pop
                expected = (gid, parent, -1 if local_id == 0 else local_id - 1,
                            center, distance, path, correction, fallback,
                            separate, fused)
            descriptor_mismatches += int(expected != descriptor)
            checked_rows += 1
    require(mask_mismatches == 0 and descriptor_mismatches == 0,
            "direct M40 mask/selection mismatch")
    return {
        "fixed_boundary_and_random_phases": len(direct_masks),
        "direct_source_rows": checked_rows,
        "mask_mismatches": mask_mismatches,
        "full_parent_plus_local_selection_mismatches": descriptor_mismatches,
    }


def replay(phases, model, issue_field):
    total_cycles = 0
    components = Counter()
    maximum_slot = 0
    overflow = 0
    for sample in range(10):
        time = 0
        for key, phase in phases:
            if key[0] != sample:
                continue
            config_data = (model["hierarchical_config_bytes"] +
                           model["dram_bytes_per_cycle"] - 1) // \
                model["dram_bytes_per_cycle"]
            time += (config_data + model["dma_command_setup_cycles"] +
                     phase["hierarchical_matcher_cycles"] +
                     model["bitmap_seal_cycles"])
            components["config_data"] += config_data
            components["config_command"] += model["dma_command_setup_cycles"]
            components["q32_parent_matcher"] += phase[
                "q32_parent_matcher_cycles"]
            components["child_matcher_pipeline_latency"] += phase[
                "child_matcher_pipeline_latency_cycles"]
            components["bitmap_seal"] += model["bitmap_seal_cycles"]
            maximum_slot = max(maximum_slot, phase["actual_slot_bytes"])
            overflow += phase["current_slot_overflow"]
            if phase["active_rows"] == 0:
                time += model["tail_cycles"]
                components["tail"] += model["tail_cycles"]
                continue
            tile_bytes = (model["weight_bytes_per_tile"] +
                          phase["actual_pwp_bytes_per_tile"])
            require(model["hierarchical_config_bytes"] + tile_bytes <=
                    model["expanded_q128_tile_slot_bytes"] and
                    tile_bytes % model["dram_bytes_per_cycle"] == 0,
                    "review replay slot/alignment drift")
            tile_data = tile_bytes // model["dram_bytes_per_cycle"]
            tile_commands = 1 + phase["used_center_runs"]
            tile_dma = tile_data + tile_commands * model["dma_command_setup_cycles"]
            work = model["output_blocks_per_tile"] * phase[issue_field]
            replay0 = work + model["descriptor_sram_latency_cycles"]
            replay1 = work + model["descriptor_sram_latency_cycles"]
            time += tile_dma
            tile0_end = time + replay0
            tile1_dma_end = time + tile_dma
            tile1_start = max(tile0_end, tile1_dma_end)
            components["tile1_dma_exposed"] += max(
                0, tile1_dma_end - tile0_end)
            time = tile1_start + replay1 + model["tail_cycles"]
            components["tile0_dma_data"] += tile_data
            components["tile0_dma_commands"] += (
                tile_commands * model["dma_command_setup_cycles"])
            components["replay0"] += replay0
            components["replay1"] += replay1
            components["active_compute"] += 2 * work
            components["descriptor_sram_startup"] += (
                2 * model["descriptor_sram_latency_cycles"])
            components["tail"] += model["tail_cycles"]
            components["actual_pwp_dram_bytes"] += (
                phase["actual_pwp_bytes_per_tile"] * 2)
            components["weight_dram_bytes"] += model["weight_bytes_per_tile"] * 2
            components["used_pwp_pattern_slots_across_tiles"] += (
                phase["used_pwp_patterns"] * 2)
        time += model["commit_cycles_per_sample"]
        components["commit"] += model["commit_cycles_per_sample"]
        total_cycles += time
    return {
        "cycles": total_cycles,
        "components": dict(components),
        "maximum_actual_slot_bytes": maximum_slot,
        "current_slot_overflow_phases": overflow,
    }


def main():
    review_dir = Path(__file__).resolve().parent
    hw = review_dir.parents[1]
    review_contract_path = review_dir / "m453b_h2_independent_review_contract_r1.json"
    require(sha256(review_contract_path) == REVIEW_CONTRACT_SHA256,
            "review contract SHA drift")
    review_contract = strict_json(review_contract_path)
    for name, expected in EXPECTED_SUBJECT.items():
        spec = review_contract["frozen_subject"][name]
        path = hw / spec["path"]
        require(spec["sha256"] == expected and sha256(path) == expected,
                "frozen subject identity drift: " + name)

    final_contract_path = hw / review_contract["frozen_subject"][
        "final_contract"]["path"]
    contract = strict_json(final_contract_path)
    result_dir = hw / "results/m453b_h67_hierarchical_q32x3_secondary_replay_final_r1_20260826"
    result_path = result_dir / "m453b_h67_hierarchical_q32x3_secondary_replay_r1.json"
    result = strict_json(result_path)

    subject_seal = verify_manifest(
        result_dir / "SHA256SUMS", result_dir / "SHA256SUMS.seal.sha256",
        "M453b subject")
    micro_dir = hw / "results/m453b_final_freeze_micro_r1_20260826"
    micro_seal = verify_manifest(
        micro_dir / "SHA256SUMS", micro_dir / "SHA256SUMS.seal.sha256",
        "M453b final-freeze micro")

    for name, spec in contract["inputs"].items():
        require(sha256(hw / spec["path"]) == spec["sha256"],
                "final contract input drift: " + name)
    chains = {}
    for label, manifest_name, seal_name in (
            ("m453a", "m453a_manifest", "m453a_seal"),
            ("m453a_h1", "m453a_h1_manifest", "m453a_h1_seal"),
            ("m430", "m430_manifest", "m430_seal"),
            ("m451", "m451_manifest", "m451_seal"),
            ("m455", "m455_manifest", "m455_seal"),
            ("m457", "m457_manifest", "m457_seal")):
        chains[label] = verify_manifest(
            hw / contract["inputs"][manifest_name]["path"],
            hw / contract["inputs"][seal_name]["path"], label)

    trace_path = hw / contract["inputs"]["m40_trace"]["path"]
    trace = strict_json(trace_path)
    trace_dir = trace_path.parent
    payload_files = 0
    payload_bytes = 0
    seen_records = set()
    for record in trace["records"]:
        key = (int(record["sample_id"]), int(record["operator_index"]))
        require(key not in seen_records, "M40 record duplicate")
        seen_records.add(key)
        for file_key, hash_key in (("packed_file", "packed_file_sha256"),
                                   ("value_payload_file",
                                    "value_payload_sha256")):
            payload = trace_dir / record[file_key]
            require(payload.is_file() and sha256(payload) == record[hash_key],
                    "M40 payload identity mismatch")
            payload_files += 1
            payload_bytes += payload.stat().st_size
    require(seen_records == set((sample, op) for sample in range(10)
                                for op in range(4)),
            "M40 record population drift")
    require(payload_files == 80 and payload_bytes == 42346309,
            "M40 payload file/byte ledger drift")

    catalog = strict_json(hw / contract["inputs"]["m453a_catalog"]["path"])
    operators = [item["operator"] for item in catalog["operators"]]
    require(operators == trace["cohort"]["operators"],
            "catalog/M40 operator order drift")
    geometry = catalog_geometry(catalog)
    phases, phase_total = stream_phases(
        result_dir / "m453b_per_phase_secondary_replay.csv", operators)
    phase_map = dict(phases)
    center_rows, center_total = stream_centers(
        result_dir / "m453b_per_phase_selected_center_materialization_ledger.csv",
        phase_map, geometry, operators)
    selected = selected_phase_set()
    ordered_rows, source_rows, ordered_total, selected_masks, \
        selected_descriptors = stream_ordered(
            result_dir / "m453b_ordered_selected_id_descriptor_runs.csv",
            phase_map, geometry, operators, set(selected))
    direct_masks = direct_m40_masks(trace, trace_dir, selected)
    direct = direct_selection_audit(
        selected_masks, selected_descriptors, direct_masks, geometry)

    population = result["population"]
    aggregate_mapping = {
        "source_rows": "source_rows",
        "zero_rows": "zero_rows",
        "active_rows": "active_rows",
        "pwp_rows": "pwp_rows",
        "fallback_rows": "fallback_rows",
        "exact_pwp_rows": "exact_pwp_rows",
        "positive_residual_pwp_rows": "positive_residual_pwp_rows",
        "pwp_correction_ops": "pwp_correction_ops",
        "fallback_source_ops": "fallback_source_ops",
        "correction_ops_per_block": "correction_ops_per_block",
        "separate_issues_per_block": "separate_issues_per_block",
        "fused_k1_issues_per_block": "fused_k1_issues_per_block",
        "child_selected_rows": "child_selected_rows",
    }
    for result_field, aggregate_field in aggregate_mapping.items():
        require(phase_total[aggregate_field] == population[result_field],
                "phase/result population drift: " + result_field)
    for field in DISTANCE_FIELDS:
        require(phase_total[field] == result["selected_distance_histogram"][field],
                "phase/result distance drift: " + field)
    require(center_total["selected_rows"] == population["active_rows"] and
            center_total["separate_issues_per_block"] ==
            population["separate_issues_per_block"] and
            ordered_total["source_rows"] == population["source_rows"] and
            ordered_total["separate_issues_per_block"] ==
            population["separate_issues_per_block"],
            "independent aggregate ledger/result drift")

    model = contract["cycle_model"]
    separate = replay(phases, model, "separate_issues_per_block")
    fused = replay(phases, model, "fused_k1_issues_per_block")
    require(separate == result["component_ledgers"]["separate"],
            "independent separate cycle/component replay drift")
    require(fused == result["component_ledgers"]["m451_fused_opportunity"],
            "independent fused cycle/component replay drift")
    strong = result["cycles"]["strong_zero"]
    m430 = result["cycles"]["m430_q32_separate"]
    ratios = {
        "tree_separate_speedup_vs_m430": m430 / float(separate["cycles"]),
        "tree_separate_speedup_vs_strong_zero":
            strong / float(separate["cycles"]),
        "tree_fused_speedup_vs_m430": m430 / float(fused["cycles"]),
        "tree_fused_speedup_vs_strong_zero": strong / float(fused["cycles"]),
    }
    for name, value in ratios.items():
        require(abs(value - result["comparisons"][name]) < 1e-15,
                "ratio recomputation drift: " + name)
    require(ratios["tree_separate_speedup_vs_m430"] < 1.10 and
            ratios["tree_fused_speedup_vs_strong_zero"] < 2.0,
            "frozen decision threshold unexpectedly passed")
    triangle = result["triangle_bound_child_comparator_gating_diagnostic"]
    triangle_fraction = (phase_total["triangle_child_comparisons_gated"] /
                         float(phase_total[
                             "triangle_child_comparisons_potential"]))
    require(abs(triangle_fraction - triangle["gated_fraction"]) < 1e-15 and
            triangle["selection_mismatches"] == 0 and
            triangle["changes_main_matcher_cycles"] is False and
            triangle["energy_or_clock_gating_opportunity_only"] is True,
            "triangle diagnostic boundary drift")
    require(sum(row["current_slot_overflow"] for _, row in phases) == 17270 and
            max(row["actual_slot_bytes"] for _, row in phases) == 88352,
            "slot overflow/maximum drift")

    # Red-team the authoritative claim controls.  PENDING text is explicitly
    # non-authoritative and is recorded as a P1 wording/template conflict.
    require(result["status"] ==
            "PASS_M453B_TREE_SEPARATE_BELOW_GATE_NO_GO" and
            result["decision"]["next"] ==
            "NO_GO_TREE_SEPARATE_BELOW_1P10" and
            "do not proceed" in result["decision"]["reason"] and
            result["decision"]["cycle_speedup_admitted"] is False and
            result["decision"]["resource_normalized_speedup"] is False and
            result["decision"]["system_speedup"] is False and
            result["decision"]["date_headline"] is False,
            "authoritative no-go decision boundary drift")
    require(result["decision"]["matcher_rtl"] ==
            "NO_GO_PENDING_MATCHED_RESOURCE_SCREEN",
            "expected matcher_rtl template conflict missing/drifted")
    require(result["resource_boundary"]["m451_resource_killed_by_m457"] is True and
            result["claim_boundary"]["m451_fused_resource_killed_diagnostic"] is True and
            result["claim_boundary"]["m451_fused_executable"] is False and
            result["claim_boundary"]["selected_rtl"] is False and
            result["claim_boundary"]["system_speedup"] is False and
            result["claim_boundary"]["stored_q128_pwp_resource_normalized"] is False and
            result["pwp_capacity_dma_and_cache"][
                "expanded_storage_area_power_charged_in_cycle_result"] is False,
            "red-team resource/claim boundary drift")
    require(result["secondary_fixed_hardware_ablation"][
                "post_m40_catalog_or_cycle_model_tuning"] is False and
            contract["execution_gates"]["post_m40_tuning"] is False and
            contract["decision_rule"]["minimum_separate_speedup_vs_m430"] == 1.10 and
            contract["decision_rule"]["minimum_fused_speedup_vs_strong_zero"] == 2.0,
            "red-team tuning/threshold boundary drift")
    require(sha256(hw / contract["inputs"]["docs359"]["path"]) ==
            EXPECTED_SUBJECT["docs359"], "docs359 final hash drift")

    report = {
        "status": "PASS_INDEPENDENT_AUDIT_NO_GO_TREE_LINE",
        "unique_conclusion": "NO_GO_TREE_LINE",
        "score": 94,
        "findings": {
            "P0": [],
            "P1": [{
                "id": "M453B-H2-P1-001",
                "finding": "decision.matcher_rtl remains NO_GO_PENDING_MATCHED_RESOURCE_SCREEN while authoritative status/next/reason forbid proceeding to matcher RTL or M461.",
                "impact": "Template wording could be misread as authorization, but it cannot override the frozen below-1.10 gate and does not change NO_GO_TREE_LINE.",
                "repair": "Do not mutate the sealed subject. In the review/paper ledger, normalize matcher_rtl to NO_GO_TREE_LINE and cite status plus decision.next/reason as authoritative. Documentation/receipt-only repair; no replay, RTL, or M461.",
            }],
            "P2": [],
        },
        "identity": {
            "review_contract_sha256": REVIEW_CONTRACT_SHA256,
            "subject_contract_sha256": EXPECTED_SUBJECT["final_contract"],
            "analyzer_sha256": EXPECTED_SUBJECT["analyzer"],
            "result_sha256": EXPECTED_SUBJECT["result"],
            "subject_manifest_sha256": EXPECTED_SUBJECT["inner_manifest"],
            "subject_outer_seal_file_sha256": EXPECTED_SUBJECT["outer_seal"],
            "docs359_sha256": EXPECTED_SUBJECT["docs359"],
        },
        "seals": {
            "subject": subject_seal,
            "micro": micro_seal,
            "upstream": chains,
        },
        "independent_ledger_reconstruction": {
            "phase_rows": len(phases),
            "center_rows": center_rows,
            "ordered_rle_rows": ordered_rows,
            "ordered_source_rows": source_rows,
            "m40_payload_files_hashed_for_review": payload_files,
            "m40_payload_bytes_hashed_for_review": payload_bytes,
            "m40_subject_or_analyzer_executions": 0,
            "m40_catalog_or_cycle_model_tuning": False,
            "sealed_m40_access_purpose": "independent review only",
            "direct_mask_selection": direct,
        },
        "population": {key: phase_total[key] for key in (
            "source_rows", "zero_rows", "active_rows", "pwp_rows",
            "fallback_rows", "exact_pwp_rows",
            "positive_residual_pwp_rows", "pwp_correction_ops",
            "fallback_source_ops", "correction_ops_per_block",
            "separate_issues_per_block", "fused_k1_issues_per_block",
            "child_selected_rows")},
        "cycles": {
            "strong_zero": strong,
            "m430_q32_separate": m430,
            "tree_separate": separate,
            "tree_m451_fused_resource_killed_diagnostic": fused,
            "ratios": ratios,
            "separate_gate_1p10_pass": False,
            "fused_diagnostic_gate_2p0_pass": False,
        },
        "capacity_and_triangle": {
            "current_slot_overflow_phases": 17270,
            "maximum_actual_slot_bytes": 88352,
            "triangle_gated_fraction": triangle_fraction,
            "triangle_selection_mismatches": 0,
            "triangle_cycles_admitted": False,
        },
        "claim_boundary": {
            "scope": "four frozen H67 ep35 bottleneck Conv3x3 operators only",
            "secondary_non_pristine_replay": True,
            "system_speedup": False,
            "selected_rtl": False,
            "resource_normalized_speedup": False,
            "date_headline": False,
            "m451_fused_executable": False,
            "m461_authorized": False,
            "matcher_rtl_authorized": False,
        },
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
