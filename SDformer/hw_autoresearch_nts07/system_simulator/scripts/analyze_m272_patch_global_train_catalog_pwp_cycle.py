#!/usr/bin/env python3
"""Replay a global train-only K16 PWP catalog on exact-binary patch Conv."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


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
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(
                             RuntimeError("non-finite JSON: " + token)))


def decode_record(record, payload_root):
    path = payload_root / record["relative_path"]
    require(path.is_file() and sha256(path) == record["file_sha256"],
            "M51 payload identity drift: " + str(path))
    shape = tuple(int(value) for value in record["input_shape"])
    elements = int(np.prod(shape))
    packed = np.fromfile(str(path), dtype=np.uint8)
    require(packed.size == int(record["packed_bytes"]),
            "M51 packed byte extent drift")
    bits = np.unpackbits(packed, bitorder="little")[:elements]
    require(int(bits.sum(dtype=np.uint64)) == int(record["active_elements"]),
            "M51 active-bit population drift")
    return bits.reshape(shape).astype(np.uint8, copy=False)


def popcount_lut():
    values = np.arange(65536, dtype="<u2")
    octets = values.view(np.uint8).reshape(65536, 2)
    return np.unpackbits(octets, axis=1).sum(axis=1).astype(np.uint8)


def global_catalog(m77):
    counts = Counter()
    for operator in m77["operators"]:
        for partition in operator["partitions"]:
            for pattern in partition["patterns"]:
                counts[int(pattern["value_hex"], 16)] += int(
                    pattern["calibration_count"])
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    centers = [value for value, _ in ranked[:16]]
    require(len(centers) == len(set(centers)) == 16 and
            all(0 < value < 65536 for value in centers),
            "global center population/domain drift")
    return centers, counts


def minimum_tree_flips(centers):
    values = [0] + list(centers)
    visited = {0}
    edges = []
    while len(visited) != len(values):
        distance, parent, child = min(
            (bin(values[parent] ^ values[child]).count("1"), parent, child)
            for parent in visited
            for child in range(len(values)) if child not in visited)
        visited.add(child)
        edges.append({"parent": parent, "child": child,
                      "xor_mask": values[parent] ^ values[child],
                      "distance": distance})
    return sum(edge["distance"] for edge in edges), edges


def normalize_shapes(bits, output_shape):
    if bits.ndim == 4:
        bits = bits[:, np.newaxis, :, :, :]
    require(bits.ndim == 5, "expected T(B)CHW patch input")
    if len(output_shape) == 4:
        output_shape = [output_shape[0], 1] + list(output_shape[1:])
    require(len(output_shape) == 5, "expected T(B)CHW patch output")
    t_count, batch, channels, height, width = bits.shape
    out_t, out_b, out_c, out_h, out_w = (
        int(value) for value in output_shape)
    require((t_count, batch, out_c) == (out_t, out_b, 96),
            "patch output geometry drift")
    require(height % out_h == 0 and width % out_w == 0 and
            height // out_h == width // out_w and
            height // out_h in (1, 2) and channels % 16 == 0,
            "patch stride/channel geometry drift")
    return bits, height // out_h, out_h, out_w


def replay_record(bits, output_shape, pop_lut, candidate_lut,
                  pwp_lut, tree_flips):
    bits, stride, out_h, out_w = normalize_shapes(bits, output_shape)
    t_count, batch, channels, height, width = bits.shape
    padded = np.pad(bits, ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")
    phases = []
    source_contributions = 0
    for kernel_y in range(3):
        for kernel_x in range(3):
            sampled = padded[:, :, :,
                kernel_y:kernel_y + stride * out_h:stride,
                kernel_x:kernel_x + stride * out_w:stride]
            require(sampled.shape ==
                    (t_count, batch, channels, out_h, out_w),
                    "receptive-field sample geometry drift")
            source_contributions += int(sampled.sum(dtype=np.uint64))
            channel_last = np.moveaxis(sampled, 2, -1)
            packed = np.packbits(channel_last, axis=-1, bitorder="little")
            require(packed.shape[-1] == channels // 8,
                    "packed source group extent drift")
            masks = (packed[..., 0::2].astype(np.uint16) |
                     (packed[..., 1::2].astype(np.uint16) << 8))
            for group in range(channels // 16):
                histogram = np.bincount(masks[..., group].reshape(-1),
                                        minlength=65536).astype(np.int64)
                population = int(histogram.sum())
                bit_ops = int(np.dot(histogram, pop_lut))
                candidate_ops = int(np.dot(histogram, candidate_lut))
                pwp_rows = int(histogram[pwp_lut].sum())
                correction_ops = candidate_ops - pwp_rows
                require(population == t_count * batch * out_h * out_w and
                        0 <= candidate_ops <= bit_ops and
                        pwp_rows + correction_ops == candidate_ops,
                        "partition work conservation drift")
                phases.append({
                    "rows": population,
                    "bit_sparse_ops": bit_ops,
                    "candidate_ops": candidate_ops,
                    "pwp_rows": pwp_rows,
                    "correction_ops": correction_ops,
                    "matcher_rows": int(histogram[pop_lut >= 2].sum()),
                })

    require(len(phases) == 9 * channels // 16,
            "patch phase population drift")
    input_vectors = int(t_count * batch * height * width)
    output_tokens = int(t_count * batch * out_h * out_w)
    bit_cycles = 48
    wide_cycles = 51 + tree_flips
    shared_cycles = 51 + tree_flips
    bindings = {"wide": Counter(), "shared": Counter()}
    for index, phase in enumerate(phases):
        next_bit_load = 48 if index + 1 < len(phases) else 0
        next_tree = (51 + tree_flips) if index + 1 < len(phases) else 0
        bit_cycles += max(phase["bit_sparse_ops"], next_bit_load) + 2
        matcher = phase["matcher_rows"] + 16
        packer = int(math.ceil(phase["pwp_rows"] / 8.0)) + 4
        wide_candidates = ((phase["candidate_ops"], "compute"),
                           (matcher, "matcher"), (packer, "packer"),
                           (next_tree, "materialize"))
        wide_service, wide_binding = max(wide_candidates)
        wide_cycles += wide_service + 2
        bindings["wide"][wide_binding] += 1
        shared_compute = phase["correction_ops"] + 2 * phase["pwp_rows"]
        shared_candidates = ((shared_compute, "compute"),
                             (matcher, "matcher"), (packer, "packer"),
                             (next_tree, "materialize"))
        shared_service, shared_binding = max(shared_candidates)
        shared_cycles += shared_service + 2
        bindings["shared"][shared_binding] += 1
    bit_total = input_vectors + bit_cycles + output_tokens
    wide_total = input_vectors + wide_cycles + output_tokens
    shared_total = input_vectors + shared_cycles + output_tokens
    totals = Counter()
    for phase in phases:
        totals.update(phase)
    return {
        "stride": stride,
        "input_channels": channels,
        "partitions": len(phases),
        "input_scan_cycles": input_vectors,
        "output_commit_cycles": output_tokens,
        "valid_receptive_field_source_contributions": source_contributions,
        "bit_sparse_vector_ops": totals["bit_sparse_ops"],
        "candidate_vector_ops": totals["candidate_ops"],
        "pwp_vector_ops": totals["pwp_rows"],
        "correction_vector_ops": totals["correction_ops"],
        "bit_sparse_cycles": bit_total,
        "wide_pwp_cycles": wide_total,
        "shared96_pwp_cycles": shared_total,
        "wide_speedup_vs_bit_sparse": bit_total / float(wide_total),
        "shared96_speedup_vs_bit_sparse": bit_total / float(shared_total),
        "binding_phases": {
            name: {key: value for key, value in sorted(counts.items())}
            for name, counts in bindings.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m272_patch_global_train_catalog_pwp_cycle_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "frozen input SHA drift {}: {}".format(name, observed))
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}

    manifest = strict_json(paths["m51_manifest"])
    m77 = strict_json(paths["m77_train_catalog"])
    admission = strict_json(paths["m77_admission"])
    m222 = strict_json(paths["m222_patch_reference"])
    require(manifest["status"] ==
            "PASS_EXACT_BINARY_INPUT_TRACE_NO_OUTPUT_OR_PERFORMANCE_CLAIM" and
            m77["split"]["test_or_validation_data_used"] is False and
            m77["split"]["train_valid825_key_overlap"] == 0 and
            admission["catalog_sha256"] == identities["m77_train_catalog"]["sha256"] and
            admission["train_only_admitted"] is True,
            "trace/catalog admission drift")
    centers, center_counts = global_catalog(m77)
    tree_flips, tree_edges = minimum_tree_flips(centers)
    pop_lut = popcount_lut().astype(np.int64)
    values = np.arange(65536, dtype=np.uint16)
    minimum_distance = np.full(65536, 17, dtype=np.int64)
    for center in centers:
        minimum_distance = np.minimum(
            minimum_distance, pop_lut[np.bitwise_xor(values, center)])
    candidate_lut = np.minimum(pop_lut, 1 + minimum_distance)
    pwp_lut = (1 + minimum_distance) < pop_lut

    selected = set(contract["source_geometry"]["selected_m51_module_indices"])
    records = [record for record in manifest["records"]
               if int(record["module_index"]) in selected]
    require(len(records) == 60 and
            sorted(set(int(record["module_index"]) for record in records)) ==
            sorted(selected), "selected patch record population drift")
    m222_rows = {(int(row["sample_id"]), int(row["module_index"])): row
                 for row in m222["per_record"]}
    per_record = []
    aggregate = Counter()
    per_sample = defaultdict(Counter)
    per_module = defaultdict(Counter)
    payload_root = paths["m51_manifest"].parent
    for ordinal, record in enumerate(sorted(
            records, key=lambda row: (int(row["sample_id"]),
                                      int(row["module_index"])))):
        bits = decode_record(record, payload_root)
        replay = replay_record(bits, record["output_shape"], pop_lut,
                               candidate_lut, pwp_lut, tree_flips)
        key = (int(record["sample_id"]), int(record["module_index"]))
        require(key in m222_rows and
                replay["valid_receptive_field_source_contributions"] ==
                int(m222_rows[key]["valid_receptive_field_source_contributions"]),
                "M222 receptive-field source conservation drift")
        replay.update({
            "ordinal": ordinal,
            "sample_id": key[0],
            "module_index": key[1],
            "name": record["name"],
            "payload_sha256": record["file_sha256"],
        })
        per_record.append(replay)
        for field in ("valid_receptive_field_source_contributions",
                      "bit_sparse_vector_ops", "candidate_vector_ops",
                      "pwp_vector_ops", "correction_vector_ops",
                      "bit_sparse_cycles", "wide_pwp_cycles",
                      "shared96_pwp_cycles"):
            aggregate[field] += replay[field]
            per_sample[key[0]][field] += replay[field]
            per_module[key[1]][field] += replay[field]
        print("[M272] {}/60 sample={} module={} wide={:.6f} shared={:.6f}".format(
            ordinal + 1, key[0], key[1],
            replay["wide_speedup_vs_bit_sparse"],
            replay["shared96_speedup_vs_bit_sparse"]), flush=True)

    require(aggregate["pwp_vector_ops"] + aggregate["correction_vector_ops"] ==
            aggregate["candidate_vector_ops"],
            "aggregate PWP work conservation drift")
    def summary(rows):
        result = []
        for index in sorted(rows):
            row = rows[index]
            result.append({
                "index": index,
                "bit_sparse_cycles": row["bit_sparse_cycles"],
                "wide_pwp_cycles": row["wide_pwp_cycles"],
                "shared96_pwp_cycles": row["shared96_pwp_cycles"],
                "wide_speedup_vs_bit_sparse":
                    row["bit_sparse_cycles"] / float(row["wide_pwp_cycles"]),
                "shared96_speedup_vs_bit_sparse":
                    row["bit_sparse_cycles"] /
                    float(row["shared96_pwp_cycles"]),
            })
        return result

    sample_rows = summary(per_sample)
    module_rows = summary(per_module)
    wide_speedup = (aggregate["bit_sparse_cycles"] /
                    float(aggregate["wide_pwp_cycles"]))
    shared_speedup = (aggregate["bit_sparse_cycles"] /
                      float(aggregate["shared96_pwp_cycles"]))
    all_faster = (all(row["wide_speedup_vs_bit_sparse"] > 1.0
                      for row in per_record) and
                  all(row["wide_speedup_vs_bit_sparse"] > 1.0
                      for row in sample_rows) and
                  all(row["wide_speedup_vs_bit_sparse"] > 1.0
                      for row in module_rows))
    output = {
        "schema": "m272_patch_global_train_catalog_pwp_cycle_v1",
        "status": ("PASS_GLOBAL_TRAIN_CATALOG_PATCH_MODULE_CYCLE_OPPORTUNITY"
                   if all_faster else
                   "PASS_REPLAY_GLOBAL_CATALOG_PATCH_PERFORMANCE_NO_GO"),
        "identity": identities,
        "global_train_only_catalog": {
            "centers_hex": ["{:04x}".format(value) for value in centers],
            "calibration_counts": [center_counts[value] for value in centers],
            "selection_used_patch_trace": False,
            "minimum_hamming_tree_flips": tree_flips,
            "minimum_hamming_tree_edges": tree_edges,
        },
        "scope": {
            "samples": 10,
            "modules": 6,
            "records": 60,
            "source_partition": contract["source_geometry"]["source_partition"],
            "output_lanes": 96,
        },
        "aggregate_exact_work": {
            "valid_receptive_field_source_contributions":
                aggregate["valid_receptive_field_source_contributions"],
            "bit_sparse_vector_ops": aggregate["bit_sparse_vector_ops"],
            "candidate_vector_ops": aggregate["candidate_vector_ops"],
            "pwp_vector_ops": aggregate["pwp_vector_ops"],
            "correction_vector_ops": aggregate["correction_vector_ops"],
            "natural_vector_op_speedup":
                aggregate["bit_sparse_vector_ops"] /
                float(aggregate["candidate_vector_ops"]),
        },
        "same_resource_module_cycles": {
            "bit_sparse": aggregate["bit_sparse_cycles"],
            "wide_pwp": aggregate["wide_pwp_cycles"],
            "shared96_pwp": aggregate["shared96_pwp_cycles"],
            "wide_speedup_vs_bit_sparse": wide_speedup,
            "shared96_speedup_vs_bit_sparse": shared_speedup,
            "all_60_records_all_10_samples_all_6_modules_wide_faster": all_faster,
        },
        "per_sample": sample_rows,
        "per_module": module_rows,
        "per_record": per_record,
        "admission": {
            "global_catalog_train_only": True,
            "trace_exact_binary": True,
            "m222_source_conservation": True,
            "isolated_patch_module_cycles": all_faster,
            "patch_int8_pwp_numeric": False,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "energy": False,
            "complete_patch_embed": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    require(sha256(Path(__file__).resolve()) == source_start,
            "M272 analyzer changed during execution")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = (
        args.output_dir /
        "m272_patch_global_train_catalog_pwp_cycle_r1.json"
    )
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("M272_PASS wide={:.6f} shared={:.6f} natural={:.6f} all={}".format(
        wide_speedup, shared_speedup,
        output["aggregate_exact_work"]["natural_vector_op_speedup"],
        all_faster), flush=True)


if __name__ == "__main__":
    main()
