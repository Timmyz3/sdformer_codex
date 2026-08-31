#!/usr/bin/env python3
"""Independent, read-only M453A-H1 hammer audit.

This program deliberately does not import any M453 builder/helper and refuses
every M40 payload path.  It reconstructs M73 convolution words directly from
the frozen packed activation planes.
"""

from collections import Counter
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent
M73 = HW / "system_handoff/incoming/m73_h67_ep35_train_calibration_sources_s32_r1_20260823"
M73_MANIFEST = M73 / "m73_train_calibration_source_manifest.json"
M430_DIR = HW / "results/m430a_trainonly_dualaware_q32_catalog_r1_20260826"
M430_CATALOG = M430_DIR / "m430_trainonly_dualaware_q32_catalog_r1.json"
SUBJECT = HW / "results/m453a_trainonly_hierarchical_q32x3_catalog_r3rev3_20260826"
CATALOG_PATH = SUBJECT / "m453a_trainonly_hierarchical_q32x3_catalog_r1.json"
AUDIT_PATH = SUBJECT / "m453a_trainonly_hierarchical_q32x3_catalog_audit_r1.json"
CSV_PATH = SUBJECT / "m453a_parent_child_train_audit.csv"
BUILDER = HW / "system_simulator/scripts/build_m453a_trainonly_hierarchical_q32x3_catalog_r3.py"
CONTRACT = HW / "contracts/m453a_trainonly_hierarchical_q32x3_catalog_vector_recovery_contract_r3rev3_20260826.json"
REVIEW_CONTRACT = HERE / "m453a_h1_independent_review_contract_r1.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "m453a_h1_independent_audit_receipt_r1.json"

K = 16
OPS = 4
PARTITIONS = 432
SAMPLES = 32
ROWS = 3000
POPCOUNT = np.asarray([bin(i).count("1") for i in range(1 << K)], dtype=np.uint8)
ACTUAL = ((0, 0), (0, 431), (1, 0), (1, 431),
          (2, 0), (2, 431), (3, 0), (3, 431))
RANDOM = ((0, 73), (0, 211), (1, 96), (1, 367),
          (2, 17), (2, 278), (3, 144), (3, 399))
PROTECTED_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("non-standard JSON token " + token)))


def verify_seal(directory, expected_names=None):
    manifest = directory / "SHA256SUMS"
    seal = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and seal.is_file(), "missing seal layer")
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256") and
                not Path(name).is_absolute() and ".." not in Path(name).parts,
                "vacuous or escaping inner seal entry")
        path = directory / name
        require(path.is_file() and sha(path) == expected,
                "inner seal mismatch: " + str(path))
        rows.append(name)
    require(rows and len(rows) == len(set(rows)), "empty/duplicate inner seal")
    if expected_names is not None:
        require(sorted(rows) == sorted(expected_names), "inner seal extent drift")
    outer_hash, outer_name = seal.read_text(encoding="utf-8").strip().split("  ", 1)
    require(outer_name == "SHA256SUMS" and sha(manifest) == outer_hash,
            "outer seal mismatch")
    return {"entries": rows, "inner_manifest_sha256": sha(manifest),
            "outer_seal_sha256": sha(seal)}


def decode_words(record, raw):
    require(record["shape"] == [10, 1, 768, 15, 20], "M73 shape drift")
    plane = int(record["positive_plane_bytes"])
    require(len(raw) == 3 * plane and plane == 288000,
            "M73 packed plane extent drift")
    require(not any(raw[plane:2 * plane]), "negative plane is nonzero")
    bits = np.unpackbits(np.frombuffer(raw[:plane], dtype=np.uint8),
                         bitorder="little", count=2304000)
    require(int(bits.sum()) == int(record["positive_count"]),
            "positive population mismatch")
    activation = bits.reshape(10, 1, 768, 15, 20)[:, 0]
    padded = np.pad(activation, ((0, 0), (0, 0), (1, 1), (1, 1)))
    features = np.empty((10, 15, 20, 768, 3, 3), dtype=np.uint8)
    for ky in range(3):
        for kx in range(3):
            features[:, :, :, :, ky, kx] = padded[
                :, :, ky:ky + 15, kx:kx + 20].transpose(0, 2, 3, 1)
    packed = np.packbits(features.reshape(ROWS, 6912), axis=1,
                         bitorder="little")
    words = np.ascontiguousarray(packed).view("<u2").reshape(ROWS, PARTITIONS)
    require(words.shape == (ROWS, PARTITIONS), "decoded word extent drift")
    return words


def issue_units(values, distance):
    return np.minimum(POPCOUNT[values].astype(np.int16),
                      1 + distance.astype(np.int16))


def scalar_greedy(values, counts, parent, parents, partition_values,
                  previously_selected):
    parent_set = set(parents)
    bucket_set = set(map(int, values))
    candidates = sorted(int(v) for v in partition_values
                        if int(v) not in parent_set and
                        int(v) not in previously_selected)
    require(len(candidates) >= 3, "candidate underflow")
    best_distance = POPCOUNT[np.bitwise_xor(values, parent)]
    objectives = [int(np.dot(counts, issue_units(values, best_distance)))]
    selected = []
    keys = []
    for _ in range(3):
        best_key = None
        best_value = None
        best_next = None
        for candidate in candidates:
            if candidate in selected:
                continue
            next_distance = np.minimum(
                best_distance, POPCOUNT[np.bitwise_xor(values, candidate)])
            objective = int(np.dot(counts, issue_units(values, next_distance)))
            key = (objective, int(candidate not in bucket_set),
                   int(POPCOUNT[candidate ^ parent]), candidate)
            if best_key is None or key < best_key:
                best_key, best_value, best_next = key, candidate, next_distance
        require(best_value is not None, "scalar greedy exhausted")
        selected.append(best_value)
        keys.append(best_key)
        objectives.append(best_key[0])
        best_distance = best_next
    return selected, objectives, keys


def vector_greedy(values, counts, parent, parents, partition_values,
                  previously_selected):
    parent_set = set(parents)
    bucket_set = set(map(int, values))
    candidates = np.asarray(sorted(int(v) for v in partition_values
                                   if int(v) not in parent_set and
                                   int(v) not in previously_selected),
                            dtype=np.uint16)
    best_distance = POPCOUNT[np.bitwise_xor(values, parent)]
    objectives = [int(np.dot(counts, issue_units(values, best_distance)))]
    selected = []
    keys = []
    for _ in range(3):
        best_key = None
        best_next = None
        for start in range(0, len(candidates), 512):
            chunk = candidates[start:start + 512]
            chunk = chunk[np.asarray([int(v) not in selected for v in chunk])]
            if not len(chunk):
                continue
            distance = POPCOUNT[np.bitwise_xor(chunk[:, None], values[None, :])]
            next_distance = np.minimum(distance, best_distance[None, :])
            units = np.minimum(POPCOUNT[values][None, :].astype(np.int16),
                               1 + next_distance.astype(np.int16))
            objective = units.astype(np.int64) @ counts
            outside = np.asarray([int(int(v) not in bucket_set) for v in chunk])
            parent_distance = POPCOUNT[np.bitwise_xor(chunk, parent)]
            order = np.lexsort((chunk.astype(np.int64),
                                parent_distance.astype(np.int64),
                                outside.astype(np.int64), objective))
            index = int(order[0])
            key = (int(objective[index]), int(outside[index]),
                   int(parent_distance[index]), int(chunk[index]))
            if best_key is None or key < best_key:
                best_key = key
                best_next = next_distance[index].copy()
        require(best_key is not None, "vector greedy exhausted")
        selected.append(best_key[-1])
        keys.append(best_key)
        objectives.append(best_key[0])
        best_distance = best_next
    return selected, objectives, keys


def main():
    require(REVIEW_CONTRACT.is_file(), "review contract missing")
    require(sha(DOCS359) == PROTECTED_SHA, "docs359 pre-audit drift")
    review_contract = strict_json(REVIEW_CONTRACT)
    require(review_contract["policy_frozen_before_subject_content_read"] is True,
            "review policy not frozen")
    builder_before = sha(BUILDER)
    contract_before = sha(CONTRACT)
    subject_before = {p.name: sha(p) for p in SUBJECT.iterdir() if p.is_file()}

    candidate_contract = strict_json(CONTRACT)
    catalog = strict_json(CATALOG_PATH)
    claimed_audit = strict_json(AUDIT_PATH)
    m430 = strict_json(M430_CATALOG)
    manifest = strict_json(M73_MANIFEST)

    require(builder_before == candidate_contract["inputs"]["builder"]["sha256"],
            "builder/candidate-contract identity mismatch")
    require(candidate_contract["additional_execution_gates"]["actual_partition_ids"] ==
            [0, 431, 432, 863, 864, 1295, 1296, 1727],
            "candidate actual partition set drift")
    candidate_seal = verify_seal(SUBJECT, [CATALOG_PATH.name, AUDIT_PATH.name,
                                           CSV_PATH.name])
    m430_seal = verify_seal(M430_DIR)

    source = BUILDER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    function_source = {}
    for name in ("scalar_greedy_core", "vector_greedy_core"):
        require(name in defined, "greedy core AST missing: " + name)
        start = source.index("def " + name + "(")
        next_def = source.find("\ndef ", start + 1)
        next_class = source.find("\nclass ", start + 1)
        ends = [value for value in (next_def, next_class) if value >= 0]
        function_source[name] = source[start:min(ends) if ends else len(source)]
    require(len(function_source) == 2, "greedy core AST missing")
    for name, text in function_source.items():
        require('state["partition_values"]' in text and
                "bucket_values" in text and "bucket_counts" in text and
                "M40" not in text and "m40" not in text,
                "greedy core dataflow boundary mismatch: " + name)
    static_leakage = {
        "candidate_pool_source": "state.partition_values (same partition M73 aggregate)",
        "objective_values_source": "bucket_values only",
        "objective_counts_source": "bucket_counts only",
        "global_or_heldout_objective_reference_found": False,
        "m40_reference_in_greedy_cores_found": False,
    }

    records = manifest["records"]
    require(len(records) == 128 and manifest["cohort"]["samples"] == 32 and
            len(manifest["cohort"]["operators"]) == 4,
            "M73 cohort extent drift")
    dense = np.zeros((OPS, PARTITIONS, 1 << K), dtype=np.uint32)
    payload_files = 0
    payload_bytes = 0
    decoded_rows = 0
    seen = Counter()
    payload_identities = []
    offsets = (np.arange(PARTITIONS, dtype=np.int64) * (1 << K))[None, :]
    for index, record in enumerate(records):
        sample = int(record["sample_id"])
        op = int(record["operator_index"])
        require(0 <= sample < SAMPLES and 0 <= op < OPS,
                "M73 sample/operator index drift")
        seen[(sample, op)] += 1
        raw_packed = None
        for key, size_key, digest_key in (
                ("packed_file", "packed_file_bytes", "packed_file_sha256"),
                ("value_payload_file", "value_payload_compressed_bytes",
                 "value_payload_sha256")):
            path = (M73 / record[key]).resolve()
            require(path.parent == M73.resolve() and "m40" not in str(path).lower(),
                    "forbidden payload path")
            raw = path.read_bytes()
            require(len(raw) == int(record[size_key]) and
                    sha_bytes(raw) == record[digest_key],
                    "M73 payload SHA/extent mismatch: " + path.name)
            payload_files += 1
            payload_bytes += len(raw)
            payload_identities.append((path.name, len(raw), sha_bytes(raw)))
            if key == "packed_file":
                raw_packed = raw
        words = decode_words(record, raw_packed)
        keys = words.astype(np.int64) + offsets
        np.add.at(dense[op].reshape(-1), keys.reshape(-1), 1)
        decoded_rows += words.size
        if (index + 1) % 16 == 0:
            print("M453A-H1 M73 records {}/128".format(index + 1), flush=True)
    require(payload_files == 256 and len(set(x[0] for x in payload_identities)) == 256,
            "M73 payload file extent mismatch")
    require(payload_bytes == 135249485 and decoded_rows == 165888000,
            "M73 byte/row extent mismatch")
    require(len(seen) == 128 and all(value == 1 for value in seen.values()),
            "M73 sample/operator uniqueness mismatch")
    phase_sums = dense.sum(axis=2, dtype=np.uint64)
    require(phase_sums.shape == (4, 432) and np.all(phase_sums == 96000),
            "aggregate partition row extent mismatch")

    csv_rows = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["operator"]), int(row["partition"]),
                   int(row["parent_id"]))
            require(key not in csv_rows, "duplicate CSV row")
            csv_rows[key] = row
    require(len(csv_rows) == OPS * PARTITIONS * 32, "CSV row extent drift")

    metrics = Counter()
    parent_mismatches = 0
    structure_mismatches = 0
    child_observation_mismatches = 0
    csv_json_mismatches = 0
    outside_children = []
    route_shortfalls = []
    scalar_targets = set(ACTUAL) | set(RANDOM)
    scalar_partition_mismatches = 0
    scalar_vector_step_mismatches = 0
    scalar_parent_calls = 0
    objective_recompute_mismatches = 0
    outside_optimality_mismatches = 0
    selection_digest = hashlib.sha256()

    weights = []
    for op in range(4):
        path = HW / ("results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/"
                     "o{}_weight_i_ky_kx_o_s8.bin".format(op))
        weights.append(np.fromfile(path, dtype=np.int8).reshape(6912, 768).astype(np.int16))
    static_min = 1 << 30
    static_max = -(1 << 30)
    static_overflows = 0

    for op in range(OPS):
        require(catalog["operators"][op]["operator"] ==
                m430["operators"][op]["operator"], "operator identity drift")
        for partition in range(PARTITIONS):
            part = catalog["operators"][op]["partitions"][partition]
            base_part = m430["operators"][op]["partitions"][partition]
            parents = [int(v, 16) for v in part["parent_patterns"]]
            expected_parents = [int(v, 16) for v in base_part["nested_patterns"][:32]]
            parent_mismatches += int(parents != expected_parents)
            children = [[int(v, 16) for v in group]
                        for group in part["children_by_parent"]]
            flat = [int(v, 16) for v in part["flat_patterns"]]
            structure_mismatches += int(
                len(parents) != 32 or len(children) != 32 or
                any(len(group) != 3 for group in children) or
                len(flat) != 128 or len(set(flat)) != 128 or
                flat != parents + [v for group in children for v in group])
            counts_all = dense[op, partition]
            values = np.flatnonzero(counts_all).astype(np.uint16)
            counts = counts_all[values].astype(np.int64)
            child_observation_mismatches += sum(
                int(counts_all[value] == 0) for group in children for value in group)
            parent_matrix = POPCOUNT[np.bitwise_xor(
                np.asarray(parents, dtype=np.uint16)[:, None], values[None, :])]
            routes = parent_matrix.argmin(axis=0)
            best_distance = parent_matrix[routes, np.arange(len(values))].copy()
            previous = set()
            full_scalar = (op, partition) in scalar_targets
            for parent_id, parent in enumerate(parents):
                member = routes == parent_id
                bucket_values = values[member]
                bucket_counts = counts[member]
                bucket_set = set(map(int, bucket_values))
                candidate_local = [int(v) for v in values
                                   if int(v) not in set(parents) and
                                   int(v) not in previous and int(v) in bucket_set]
                if len(candidate_local) < 3:
                    route_shortfalls.append((op, partition, parent_id,
                                             len(candidate_local)))
                audit_row = part["parent_route_audit"][parent_id]
                row = csv_rows[(op, partition, parent_id)]
                expected_fields = {
                    "parent_hex": "{:04x}".format(parent),
                    "child0_hex": "{:04x}".format(children[parent_id][0]),
                    "child1_hex": "{:04x}".format(children[parent_id][1]),
                    "child2_hex": "{:04x}".format(children[parent_id][2]),
                    "routed_unique_masks": str(len(bucket_values)),
                    "eligible_nonparent_unique_masks": str(sum(
                        int(v) not in set(parents) for v in bucket_values)),
                    "routed_population": str(int(bucket_counts.sum())),
                }
                csv_json_mismatches += sum(row[key] != value
                                           for key, value in expected_fields.items())
                csv_json_mismatches += int(audit_row["parent_hex"] != expected_fields["parent_hex"])
                csv_json_mismatches += int(audit_row["children_hex"] !=
                    [expected_fields["child0_hex"], expected_fields["child1_hex"],
                     expected_fields["child2_hex"]])
                selected_distance = POPCOUNT[np.bitwise_xor(bucket_values, parent)]
                objectives = [int(np.dot(bucket_counts,
                                         issue_units(bucket_values, selected_distance)))]
                for child in children[parent_id]:
                    selected_distance = np.minimum(
                        selected_distance,
                        POPCOUNT[np.bitwise_xor(bucket_values, child)])
                    objectives.append(int(np.dot(
                        bucket_counts, issue_units(bucket_values, selected_distance))))
                claimed_objectives = audit_row[
                    "weighted_issue_objective_parent_then_each_child"]
                objective_recompute_mismatches += int(objectives != claimed_objectives)
                objective_recompute_mismatches += int(objectives != [
                    int(row["objective_parent"]), int(row["objective_child1"]),
                    int(row["objective_child2"]), int(row["objective_child3"])])
                current_outside = [child for child in children[parent_id]
                                   if child not in bucket_set]
                for child in current_outside:
                    outside_children.append((op, partition, parent_id, child))
                if full_scalar or current_outside or len(candidate_local) < 3:
                    scalar = scalar_greedy(bucket_values, bucket_counts, parent,
                                           parents, values, previous)
                    vector = vector_greedy(bucket_values, bucket_counts, parent,
                                           parents, values, previous)
                    scalar_parent_calls += 1
                    scalar_vector_step_mismatches += int(scalar != vector)
                    scalar_partition_mismatches += int(
                        scalar[0] != children[parent_id] or scalar[1] != objectives)
                    if current_outside or len(candidate_local) < 3:
                        outside_optimality_mismatches += int(
                            scalar[0] != children[parent_id] or scalar[1] != objectives)
                    selection_digest.update(json.dumps({
                        "op": op, "partition": partition, "parent": parent_id,
                        "children": scalar[0], "objectives": scalar[1],
                        "keys": scalar[2]}, sort_keys=True).encode())
                local_indices = np.flatnonzero(member)
                local_centers = np.asarray([parent] + children[parent_id],
                                           dtype=np.uint16)
                local_matrix = POPCOUNT[np.bitwise_xor(
                    local_centers[:, None], values[local_indices][None, :])]
                best_distance[local_indices] = local_matrix.min(axis=0)
                previous.update(children[parent_id])

            pop = POPCOUNT[values].astype(np.int16)
            active = values != 0
            pwp = active & (1 + best_distance < pop)
            correction = np.where(pwp, best_distance, pop).astype(np.int64)
            exact = pwp & (best_distance == 0)
            positive = pwp & (best_distance > 0)
            metrics["source_rows"] += int(counts.sum())
            metrics["zero_rows"] += int(counts[~active].sum())
            metrics["active_rows"] += int(counts[active].sum())
            metrics["pwp_rows"] += int(counts[pwp].sum())
            metrics["fallback_rows"] += int(counts[active & ~pwp].sum())
            metrics["exact_pwp_rows"] += int(counts[exact].sum())
            metrics["positive_residual_pwp_rows"] += int(counts[positive].sum())
            metrics["correction_ops_per_block"] += int(np.dot(counts, correction))
            metrics["separate_issues_per_block"] += int(np.dot(
                counts, correction + pwp.astype(np.int64)))
            metrics["fused_k1_issues_per_block"] += int(np.dot(
                counts, np.where(pwp, np.maximum(1, correction), correction)))

            bits = np.asarray([[(pattern >> bit) & 1 for bit in range(16)]
                               for pattern in flat], dtype=np.int16)
            products = bits @ weights[op][partition * 16:(partition + 1) * 16]
            static_min = min(static_min, int(products.min()))
            static_max = max(static_max, int(products.max()))
            static_overflows += int(np.count_nonzero(
                (products < -2048) | (products > 2047)))
        print("M453A-H1 structure op {}/4".format(op + 1), flush=True)

    require(parent_mismatches == structure_mismatches ==
            child_observation_mismatches == csv_json_mismatches == 0,
            "catalog structure/CSV mismatch")
    require(len(outside_children) == 19 and len(route_shortfalls) == 1 and
            route_shortfalls[0][3] == 2, "outside/shortfall extent mismatch")
    require(objective_recompute_mismatches == 0 and
            outside_optimality_mismatches == 0,
            "local route objective mismatch")
    require(scalar_parent_calls >= 16 * 32 and
            scalar_partition_mismatches == 0 and
            scalar_vector_step_mismatches == 0,
            "scalar/vector selection mismatch")
    expected_metrics = claimed_audit["train_observation"]
    metric_keys = ("source_rows", "zero_rows", "active_rows", "pwp_rows",
                   "fallback_rows", "exact_pwp_rows",
                   "positive_residual_pwp_rows", "correction_ops_per_block",
                   "separate_issues_per_block", "fused_k1_issues_per_block")
    metric_mismatches = sum(int(metrics[key] != expected_metrics[key])
                            for key in metric_keys)
    require(metric_mismatches == 0 and static_min == -1089 and
            static_max == 1059 and static_overflows == 0,
            "train metric or signed12 mismatch")
    require(metrics["source_rows"] == metrics["zero_rows"] + metrics["active_rows"] and
            metrics["active_rows"] == metrics["pwp_rows"] + metrics["fallback_rows"] and
            metrics["pwp_rows"] == metrics["exact_pwp_rows"] +
            metrics["positive_residual_pwp_rows"] and
            metrics["separate_issues_per_block"] == metrics["pwp_rows"] +
            metrics["correction_ops_per_block"] and
            metrics["fused_k1_issues_per_block"] ==
            metrics["separate_issues_per_block"] -
            metrics["positive_residual_pwp_rows"], "population conservation mismatch")

    failure_specs = [
        ("r1", HW / "results/m453a_trainonly_hierarchical_q32x3_catalog_r1_failed_20260826"),
        ("r2", HW / "results/m453a_trainonly_hierarchical_q32x3_catalog_r2_aborted_20260826"),
        ("r3_preinput", HW / "results/m453a_trainonly_hierarchical_q32x3_catalog_r3_preinput_failed_20260826"),
        ("r3rev2", HW / "results/m453a_trainonly_hierarchical_q32x3_catalog_r3rev2_aborted_20260826"),
    ]
    failures = []
    for label, directory in failure_specs:
        seal_result = verify_seal(directory)
        receipt_path = next(directory.glob("*.json"))
        receipt = strict_json(receipt_path)
        require(receipt["status"].startswith(("FAIL", "ABORT")),
                "failed chain status is citable")
        failures.append({"label": label, "status": receipt["status"],
                         "review_disposition": "DO_NOT_CITE",
                         "receipt_sha256": sha(receipt_path),
                         "seal": seal_result})

    r2_contract = strict_json(HW / "contracts/m453a_trainonly_hierarchical_q32x3_catalog_recovery_contract_r2_20260826.json")
    r3_contract = strict_json(HW / "contracts/m453a_trainonly_hierarchical_q32x3_catalog_vector_recovery_contract_r3_20260826.json")
    r3rev2_contract = strict_json(HW / "contracts/m453a_trainonly_hierarchical_q32x3_catalog_vector_recovery_contract_r3rev2_20260826.json")
    frozen = r3_contract["frozen_selection"]
    science_unchanged = (
        frozen["parents"] == "M430 q32 bit-identical IDs0..31" and
        frozen["objective_population"] == "only M73 train masks routed to current parent" and
        frozen["candidate_population"].startswith("all same-partition M73 train-observed") and
        frozen["greedy_tie"].replace(" ", "") ==
        "objective,prefer-local-route,Hamming-to-parent,numeric-mask" and
        r3_contract["supersession"]["r2_selection_contract_changed"] is False and
        r3rev2_contract["allowed_change"]["candidate_pool_changed"] is False and
        candidate_contract["allowed_change"]["candidate_pool_changed"] is False and
        candidate_contract["allowed_change"]["objective"].startswith("unchanged") and
        r2_contract["supersession"]["parent_route_or_objective_change"] is False)
    require(science_unchanged, "r2-through-r3rev3 scientific selection drift")

    claim = claimed_audit["admission"]
    forbidden_claims = ("cycle_speedup", "date_headline", "selected_rtl",
                        "synopsys", "system_speedup")
    require(all(claim[key] is False for key in forbidden_claims) and
            claimed_audit["heldout_gate"]["m40_completed_evaluations_so_far"] == 0 and
            claimed_audit["heldout_gate"]["m40_payload_reads_so_far"] == 0,
            "claim boundary drift")
    require(sha(DOCS359) == PROTECTED_SHA and sha(BUILDER) == builder_before and
            sha(CONTRACT) == contract_before and
            {p.name: sha(p) for p in SUBJECT.iterdir() if p.is_file()} == subject_before,
            "protected/upstream mutation detected")

    receipt = {
        "schema": "m453a_h1_independent_audit_receipt_r1",
        "status": "PASS_GO_EXACTLY_ONE_FIXED_M453B_SECONDARY_REPLAY",
        "score": 100,
        "findings": {"P0": [], "P1": [], "P2": []},
        "decision": {
            "go": True,
            "authorization": "exactly one fixed M453b secondary M40 replay",
            "current_review_ran_m453b": False,
            "claim_admission": False,
        },
        "m73_payload_audit": {
            "files_sha_extent_verified": payload_files,
            "unique_payload_files": len(set(x[0] for x in payload_identities)),
            "bytes_verified": payload_bytes,
            "samples": len(set(s for s, _ in seen)),
            "operators": len(set(o for _, o in seen)),
            "decoded_rows": decoded_rows,
            "per_operator_partition_rows": 96000,
            "payload_identity_ledger_sha256": sha_bytes(json.dumps(
                payload_identities, separators=(",", ":")).encode()),
            "m40_packed_or_value_payload_reads": 0,
        },
        "catalog_structure": {
            "partitions_checked": 1728,
            "parent_lists_checked": 1728,
            "parents_per_partition": 32,
            "parent_mismatches_vs_m430": parent_mismatches,
            "children_checked": 165888,
            "children_per_parent": 3,
            "flat_unique_128_partitions": 1728,
            "same_partition_train_observation_mismatches": child_observation_mismatches,
            "outside_local_route_children": len(outside_children),
            "outside_local_route_child_records": [
                {"operator": a, "partition": b, "parent_id": c,
                 "child_hex": "{:04x}".format(d)}
                for a, b, c, d in outside_children],
            "local_candidate_buckets_lt3": [
                {"operator": a, "partition": b, "parent_id": c,
                 "local_candidates": d} for a, b, c, d in route_shortfalls],
        },
        "objective_leakage_attack": {
            **static_leakage,
            "outside_or_shortfall_greedy_optimality_mismatches": outside_optimality_mismatches,
            "all_parent_selected_sequence_objective_recompute_mismatches": objective_recompute_mismatches,
            "m40_payload_reads": 0,
        },
        "independent_scalar_vector": {
            "candidate_contract_actual_partitions": [list(x) for x in ACTUAL],
            "frozen_random_extension": [list(x) for x in RANDOM],
            "unique_full_partitions": len(scalar_targets),
            "full_partition_parent_calls_required": len(scalar_targets) * 32,
            "scalar_parent_calls_including_outside_extension": scalar_parent_calls,
            "selected_sequence_or_objective_mismatches": scalar_partition_mismatches,
            "scalar_vector_step_mismatches": scalar_vector_step_mismatches,
            "tie_order": "objective,prefer-local-route,Hamming-to-parent,numeric",
            "recomputation_digest_sha256": selection_digest.hexdigest(),
        },
        "train_observation_recompute": {
            **{key: int(metrics[key]) for key in metric_keys},
            "pwp_rows_with_a_used_pattern_assignment": int(metrics["pwp_rows"]),
            "population_conservation_mismatches": 0,
            "claimed_metric_mismatches": metric_mismatches,
            "signed12_minimum": static_min,
            "signed12_maximum": static_max,
            "signed12_overflows": static_overflows,
        },
        "catalog_audit_csv_json": {
            "csv_rows": len(csv_rows),
            "csv_json_mismatches": csv_json_mismatches,
            "candidate_seal": candidate_seal,
            "m430_seal": m430_seal,
        },
        "failure_chain": {
            "failed_artifacts": failures,
            "r1_to_r2_authorized_change": "candidate extent expanded from local route to same-partition M73 train-observed pool because the local-only fixed extent was impossible",
            "r2_through_r3rev3_scientific_selection_unchanged": science_unchanged,
        },
        "claim_boundary": {
            "cycle": False, "rtl": False, "vcs": False,
            "synopsys": False, "ppa": False, "system_speedup": False,
            "date_headline": False,
        },
        "immutability": {
            "builder_sha256_before_after": builder_before,
            "candidate_contract_sha256_before_after": contract_before,
            "subject_files_unchanged": True,
            "docs359_sha256": sha(DOCS359),
            "m453b_executions": 0,
            "m40_payload_reads": 0,
        },
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M453A-H1 score=100 outside=19 shortfall=1 rows=165888000 "
          "m40_reads=0 decision=GO_ONE_FIXED_M453B", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL M453A-H1: {}".format(exc), file=sys.stderr, flush=True)
        raise
