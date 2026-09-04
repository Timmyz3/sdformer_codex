#!/usr/bin/env python3
"""Extend the admitted M77 q16 catalog to nested q32/q64/q128 on train data only."""

from __future__ import division

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path


Q_VALUES = (16, 32, 64, 128)
K = 16
MASK16 = (1 << K) - 1
POPCOUNT = tuple(bin(value).count("1") for value in range(1 << K))


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
        return json.load(handle, object_pairs_hook=pairs, parse_constant=reject)


def load_module(path):
    spec = importlib.util.spec_from_file_location("m338_frozen_m43", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M43 support unpacker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_histograms(m43, manifest, manifest_path, operators, partitions):
    histograms = defaultdict(Counter)
    operator_index = {name: index for index, name in enumerate(operators)}
    seen = Counter()
    for record_index, record in enumerate(manifest["records"]):
        packed_path = manifest_path.parent / record["packed_file"]
        value_path = manifest_path.parent / record["value_payload_file"]
        require(packed_path.is_file() and
                packed_path.stat().st_size == record["packed_file_bytes"] and
                sha256(packed_path) == record["packed_file_sha256"],
                "M338 packed payload drift")
        require(value_path.is_file() and
                value_path.stat().st_size == record["value_payload_compressed_bytes"] and
                sha256(value_path) == record["value_payload_sha256"],
                "M338 value payload drift")
        operator = record["operator"]
        sample = int(record["sample_id"])
        require(operator in operator_index, "unexpected train operator")
        seen[(sample, operator)] += 1
        masks = m43.unpack_record_masks(manifest_path.parent, record)
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // K)
                for subtile in range(m43.TILE_BITS // K):
                    value = (value256 >> (subtile * K)) & MASK16
                    histograms[(operator_index[operator],
                                partition_base + subtile)][value] += 1
        print("[M338 HIST] {}/{} sample={} op={}".format(
            record_index + 1, len(manifest["records"]), sample,
            operator_index[operator]), flush=True)
    require(len(seen) == len(manifest["records"]) and
            all(count == 1 for count in seen.values()),
            "train sample/operator uniqueness failure")
    require(len(histograms) == len(operators) * partitions,
            "train histogram extent failure")
    return histograms


def extend_nested(counter, parent_centers):
    """One-pass q16-anchored count-weighted farthest-frequency expansion.

    The order is deterministic and nested. It intentionally does not claim to be
    a q128 Lloyd optimum: the frozen admitted q16 centers are the anchor, and all
    extra entries are selected only from train-observed nonzero, non-onehot rows.
    """
    centers = list(parent_centers)
    require(len(centers) == 16 and len(set(centers)) == 16,
            "parent q16 center extent/uniqueness drift")
    # M77 filters one-hot observations before Lloyd, but a weighted-majority
    # update can still produce a one-hot center. Preserve those entries exactly
    # for the admitted q16 prefix; only the newly appended capacity is required
    # to avoid zero/one-hot entries that cannot beat exact bit-sparse fallback.
    require(all(value != 0 for value in centers),
            "parent q16 contains an invalid zero center")
    eligible = [value for value in counter
                if value != 0 and POPCOUNT[value] >= 2 and value not in centers]
    scored = []
    for value in eligible:
        distance = min(POPCOUNT[value ^ center] for center in parent_centers)
        scored.append((counter[value] * distance, distance, counter[value],
                       -value, value))
    scored.sort(reverse=True)
    centers.extend(item[-1] for item in scored[:Q_VALUES[-1] - len(centers)])
    require(len(set(centers)) == len(centers), "extended center duplication")
    return centers, len(eligible) + len(parent_centers)


def evaluate(counter, centers):
    result = Counter()
    center_set = frozenset(centers)
    for value, count in counter.items():
        population = POPCOUNT[value]
        distance = min(POPCOUNT[value ^ center] for center in centers)
        candidate = min(population, 1 + distance)
        result["partition_vectors"] += count
        result["bit_sparse_vector_ops_per_block"] += count * population
        result["candidate_vector_ops_per_block"] += count * candidate
        result["exact_pattern_hits"] += count * int(value in center_set and value != 0)
        result["pwp_selected_rows"] += count * int(1 + distance < population)
        result["correction_vector_ops_per_block"] += count * (
            distance if 1 + distance < population else population)
    require(result["candidate_vector_ops_per_block"] >=
            result["correction_vector_ops_per_block"],
            "candidate/correction conservation failure")
    result["pwp_vector_ops_per_block"] = (
        result["candidate_vector_ops_per_block"] -
        result["correction_vector_ops_per_block"])
    require(result["pwp_vector_ops_per_block"] == result["pwp_selected_rows"],
            "one-PWP-per-selected-row conservation failure")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M338 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m338_trainonly_nested_q128_catalog_contract_v1",
            "M338 contract schema drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "SHA drift for {}: {}".format(name, observed))
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}

    manifest_path = paths["m73_train_trace_manifest"]
    manifest = strict_json(manifest_path)
    parent = strict_json(paths["m77_q16_catalog"])
    admission = strict_json(paths["m77_q16_admission"])
    require(manifest["schema"] ==
            "m73_h67_ep35_train_calibration_packed_source_trace_v1" and
            manifest["status"] ==
            "PASS_M73_DSEC_TRAIN_ONLY_S32_ALL18_SEQUENCES_EXACT_H67_EP35_FOUR_BOTTLENECK_TRACE",
            "M73 train trace status drift")
    require(manifest["split_audit"]["role"] ==
            "DSEC_TRAIN_ONLY_PAFT_CALIBRATION" and
            manifest["split_audit"]["full_train_valid825_key_overlap"] == 0 and
            manifest["split_audit"]["selected_valid825_key_overlap"] == 0,
            "M73 train/validation isolation drift")
    require(parent["status"] ==
            "PASS_M77_TRAIN_ONLY_KMEANS_PAFT_CATALOG_ACCURACY_CYCLES_UNADMITTED" and
            parent["identity"]["train_trace_manifest_sha256"] ==
                identities["m73_train_trace_manifest"]["sha256"] and
            admission["train_only_admitted"] is True and
            admission["catalog_sha256"] == identities["m77_q16_catalog"]["sha256"] and
            admission["train_valid825_key_overlap"] == 0,
            "M77 parent admission drift")
    operators = tuple(manifest["cohort"]["operators"])
    require([row["operator"] for row in parent["operators"]] == list(operators),
            "M73/M77 operator order drift")
    partitions = int(parent["format"]["partitions_per_operator"])
    require(parent["format"]["partition_bits"] == K and partitions == 432,
            "M77 geometry drift")
    m43 = load_module(paths["m43_support_unpacker"])
    histograms = collect_histograms(m43, manifest, manifest_path,
                                    operators, partitions)

    q_totals = {q: Counter() for q in Q_VALUES}
    op_totals = [{q: Counter() for q in Q_VALUES} for _ in operators]
    operator_payloads = []
    short_partitions = Counter()
    for op_index, operator in enumerate(operators):
        rows = []
        for partition in range(partitions):
            parent_row = parent["operators"][op_index]["partitions"][partition]
            require(parent_row["partition"] == partition and
                    len(parent_row["patterns"]) == 16,
                    "M77 parent partition order/extent drift")
            q16 = [int(item["value_hex"], 16)
                   for item in parent_row["patterns"]]
            counter = histograms[(op_index, partition)]
            centers, eligible = extend_nested(counter, q16)
            observations = {}
            for q in Q_VALUES:
                used = min(q, len(centers))
                short_partitions[q] += int(used < q)
                observation = evaluate(counter, centers[:used])
                observation["active_patterns"] = used
                observations[str(q)] = dict(observation)
                q_totals[q].update(observation)
                op_totals[op_index][q].update(observation)
            rows.append({
                "partition": partition,
                "train_eligible_patterns": eligible,
                "nested_patterns": ["{:04x}".format(value) for value in centers],
                "observations": observations,
            })
        operator_payloads.append({"operator": operator, "partitions": rows})
        print("[M338 EXTEND] operator={}/{}".format(
            op_index + 1, len(operators)), flush=True)

    q_rows = []
    for q in Q_VALUES:
        total = q_totals[q]
        q_rows.append({
            "q_capacity": q,
            "bit_sparse_vector_ops_per_block":
                total["bit_sparse_vector_ops_per_block"],
            "candidate_vector_ops_per_block":
                total["candidate_vector_ops_per_block"],
            "exact_signed_vector_op_speedup":
                total["bit_sparse_vector_ops_per_block"] /
                float(total["candidate_vector_ops_per_block"]),
            "pwp_vector_ops_per_block": total["pwp_vector_ops_per_block"],
            "correction_vector_ops_per_block":
                total["correction_vector_ops_per_block"],
            "exact_pattern_hits": total["exact_pattern_hits"],
            "short_partitions": short_partitions[q],
            "pattern_table_capacity_bytes": len(operators) * partitions * q * 2,
            "all_signed12_pwp_capacity_bytes":
                len(operators) * partitions * q * 8 * 144,
            "operators": [{
                "operator": operators[op_index],
                "exact_signed_vector_op_speedup":
                    op_totals[op_index][q]["bit_sparse_vector_ops_per_block"] /
                    float(op_totals[op_index][q]["candidate_vector_ops_per_block"]),
            } for op_index in range(len(operators))],
        })
    require(q_rows[0]["q_capacity"] == 16 and
            all(operator_payloads[op]["partitions"][part]["nested_patterns"][:16] ==
                [item["value_hex"] for item in
                 parent["operators"][op]["partitions"][part]["patterns"]]
                for op in range(len(operators)) for part in range(partitions)),
            "q16 parent-prefix preservation failure")
    require(all(q_rows[index]["candidate_vector_ops_per_block"] <=
                q_rows[index - 1]["candidate_vector_ops_per_block"]
                for index in range(1, len(q_rows))),
            "nested q work monotonicity failure")

    payload = {
        "schema": "m338_trainonly_nested_q128_catalog_v1",
        "status": "PASS_M338_TRAIN_ONLY_NESTED_Q16_Q32_Q64_Q128_EXACT_WORK_NO_CYCLES",
        "identity": identities,
        "split": {
            "role": "DSEC_TRAIN_ONLY_PAFT_CALIBRATION",
            "test_or_validation_data_used": False,
            "selected_train_samples": manifest["split_audit"]["selected_samples"],
            "selected_train_sequences": manifest["split_audit"]["selected_sequences"],
            "train_valid825_key_overlap": 0,
        },
        "algorithm": {
            "q16_anchor": "bit-identical admitted M77 weighted-Hamming Lloyd centers",
            "q32_q64_q128_extension":
                "single-pass q16-anchored count-times-Hamming-distance ranking over train-observed popcount>=2 patterns",
            "nested_prefixes": True,
            "runtime_arithmetic":
                "exact W*x = PWP[p] plus signed W*(x-p), with exact bit-sparse fallback",
            "accuracy_loss": False,
        },
        "geometry": {
            "partition_bits": K,
            "partitions_per_operator": partitions,
            "operators": list(operators),
            "q_values": list(Q_VALUES),
            "output_blocks": 8,
            "signed12_pwp_vector_bytes": 144,
        },
        "calibration_observations": q_rows,
        "operators": operator_payloads,
        "admission": {
            "train_only_catalog": True,
            "exact_arithmetic_identity": True,
            "independent_runtime_trace": False,
            "cycle_speedup": False,
            "energy": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "claim_boundary":
            "Train-only nested codebook and exact vector-operation observation only. q>16 is a deterministic expansion, not a Lloyd optimum. No runtime trace, cache/DMA cycle, energy, system, physical-PPA or headline claim is admitted.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m338_trainonly_nested_q128_catalog_r1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M338_PASS " + " ".join(
        "q{}={:.6f}x".format(row["q_capacity"],
                              row["exact_signed_vector_op_speedup"])
        for row in q_rows), flush=True)


if __name__ == "__main__":
    main()
