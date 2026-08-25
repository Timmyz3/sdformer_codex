#!/usr/bin/env python3
"""Exact full-network local/motion parent opportunity audit for M51 payloads.

This deliberately stops at source-bit work.  It does not model cycles, output
reuse, parent-result availability, SRAM ports, metadata traffic, or PPA.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


PARENT_NAMES = ("zero", "left", "up", "previous_timestep")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def unpack_little(path, elements):
    packed = np.fromfile(str(path), dtype=np.uint8)
    require(int(packed.size) == (int(elements) + 7) // 8,
            "packed byte count mismatch: {}".format(path))
    # Avoid relying on numpy's newer bitorder keyword on older server images.
    lut = ((np.arange(256, dtype=np.uint16)[:, None] >>
            np.arange(8, dtype=np.uint16)[None, :]) & 1).astype(np.uint8)
    bits = lut[packed].reshape(-1)[:int(elements)]
    if int(elements) % 8:
        used_mask = (1 << (int(elements) % 8)) - 1
        require((int(packed[-1]) & ~used_mask) == 0,
                "nonzero high tail padding: {}".format(path))
    return bits


def as_tbhwc(bits, operator, shape):
    shape = [int(value) for value in shape]
    require(product(shape) == int(bits.size), "shape/element mismatch")
    if operator == "Linear":
        require(len(shape) == 5, "unsupported Linear rank")
        array = bits.reshape(shape)
    elif operator == "Conv2d" and len(shape) == 5:
        array = bits.reshape(shape).transpose(0, 1, 3, 4, 2)
    elif operator == "Conv2d" and len(shape) == 4:
        array = bits.reshape(shape).transpose(0, 2, 3, 1)[:, None, :, :, :]
    else:
        raise ValueError("unsupported operator/rank: {} {}".format(
            operator, shape))
    require(array.ndim == 5 and array.shape[0] == 10,
            "expected exact T=10 T,B,H,W,C view")
    return array


def update_choice(best, choice, candidate, candidate_id, target_slice):
    target_best = best[target_slice]
    target_choice = choice[target_slice]
    take = candidate < target_best
    target_best[take] = candidate[take]
    target_choice[take] = candidate_id


def choice_counts(choice):
    counts = np.bincount(choice.reshape(-1), minlength=4)
    return dict((PARENT_NAMES[index], int(counts[index])) for index in range(4))


def timestep_sums(cost):
    return [int(value) for value in cost.reshape(cost.shape[0], -1).sum(
        axis=1, dtype=np.int64)]


def analyze_vectors(array):
    require(array.dtype == np.uint8 and array.ndim == 5,
            "internal vector view mismatch")
    zero = array.sum(axis=-1, dtype=np.int32)
    local = zero.copy()
    local_choice = np.zeros(zero.shape, dtype=np.uint8)

    left = np.not_equal(array[:, :, :, 1:, :],
                        array[:, :, :, :-1, :]).sum(axis=-1, dtype=np.int32)
    update_choice(local, local_choice, left, 1,
                  (slice(None), slice(None), slice(None), slice(1, None)))
    left_total = int(left.sum(dtype=np.int64))
    del left

    up = np.not_equal(array[:, :, 1:, :, :],
                      array[:, :, :-1, :, :]).sum(axis=-1, dtype=np.int32)
    update_choice(local, local_choice, up, 2,
                  (slice(None), slice(None), slice(1, None), slice(None)))
    up_total = int(up.sum(dtype=np.int64))
    del up

    previous = np.not_equal(array[1:, :, :, :, :],
                            array[:-1, :, :, :, :]).sum(axis=-1,
                                                         dtype=np.int32)
    previous_total = int(previous.sum(dtype=np.int64))

    motion = zero.copy()
    motion_choice = np.zeros(zero.shape, dtype=np.uint8)
    update_choice(motion, motion_choice, previous, 3,
                  (slice(1, None), slice(None), slice(None), slice(None)))

    dual = local.copy()
    dual_choice = local_choice.copy()
    update_choice(dual, dual_choice, previous, 3,
                  (slice(1, None), slice(None), slice(None), slice(None)))
    del previous

    vector_count = int(zero.size)
    row = {
        "candidate_xor_bits_on_valid_coordinates": {
            "left": left_total,
            "previous_timestep": previous_total,
            "up": up_total,
        },
        "channel_width": int(array.shape[-1]),
        "choice_counts": {
            "dual": choice_counts(dual_choice),
            "local": choice_counts(local_choice),
            "motion": choice_counts(motion_choice),
        },
        "choice_metadata_bits_if_naive_2b_per_vector": 2 * vector_count,
        "source_bits": {
            "dual": int(dual.sum(dtype=np.int64)),
            "local": int(local.sum(dtype=np.int64)),
            "motion": int(motion.sum(dtype=np.int64)),
            "zero": int(zero.sum(dtype=np.int64)),
        },
        "source_bits_by_timestep": {
            "dual": timestep_sums(dual),
            "local": timestep_sums(local),
            "motion": timestep_sums(motion),
            "zero": timestep_sums(zero),
        },
        "vector_count": vector_count,
        "vector_shape_tbhwc": [int(value) for value in array.shape],
    }
    return row


def empty_totals():
    return {
        "choice_counts": dict((name, 0) for name in PARENT_NAMES),
        "dual_source_bits": 0,
        "hook_calls": 0,
        "input_elements": 0,
        "local_source_bits": 0,
        "motion_source_bits": 0,
        "vector_count": 0,
        "zero_source_bits": 0,
    }


def add_totals(total, analyzed, record):
    total["hook_calls"] += 1
    total["input_elements"] += int(record["input_elements"])
    total["vector_count"] += int(analyzed["vector_count"])
    for key in ("zero", "local", "motion", "dual"):
        total[key + "_source_bits"] += int(analyzed["source_bits"][key])
    for name, count in analyzed["choice_counts"]["dual"].items():
        total["choice_counts"][name] += int(count)


def ratio(numerator, denominator):
    require(int(denominator) > 0, "zero ratio denominator")
    return float(numerator) / float(denominator)


def finish_totals(total):
    zero = total["zero_source_bits"]
    local = total["local_source_bits"]
    motion = total["motion_source_bits"]
    dual = total["dual_source_bits"]
    total["opportunity_ratios_not_speedup"] = {
        "zero_over_dual_source_work": ratio(zero, dual),
        "zero_over_local_source_work": ratio(zero, local),
        "zero_over_motion_source_work": ratio(zero, motion),
    }
    total["marginal_dual_reduction_vs_local_percent"] = (
        100.0 * float(local - dual) / float(local))
    total["dual_reduction_vs_zero_percent"] = (
        100.0 * float(zero - dual) / float(zero))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--payload-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    contract = strict_json(arguments.contract)
    manifest = strict_json(arguments.manifest)
    payload_root = Path(arguments.payload_root).resolve()
    output = Path(arguments.output)
    require(contract["schema"] ==
            "m55_h67_full_network_dual_parent_opportunity_contract_v1",
            "contract schema mismatch")
    require(sha256_path(arguments.manifest) ==
            contract["identity"]["manifest_sha256"],
            "manifest SHA mismatch")
    require(manifest["status"] ==
            "PASS_EXACT_BINARY_INPUT_TRACE_NO_OUTPUT_OR_PERFORMANCE_CLAIM",
            "manifest status mismatch")
    require(manifest["population"] == contract["expected"],
            "manifest population mismatch")
    require(manifest["identity"]["target_plan_sha256"] ==
            contract["identity"]["target_plan_sha256"],
            "target plan identity mismatch")
    require(payload_root.is_dir(), "missing payload root")
    require(len(manifest["records"]) == contract["expected"]["hook_calls"],
            "record population mismatch")
    require(not output.exists(), "refusing existing output")

    aggregate = empty_totals()
    per_module = {}
    per_sample = {}
    rows = []
    for ordinal, record in enumerate(manifest["records"]):
        require(int(record["sample_id"]) in range(10) and
                int(record["module_index"]) in range(31),
                "record identity out of range")
        path = payload_root / record["relative_path"]
        require(path.is_file() and path.stat().st_size ==
                int(record["packed_bytes"]), "missing/size payload mismatch")
        require(sha256_path(path) == record["file_sha256"],
                "payload SHA mismatch: {}".format(path))
        bits = unpack_little(path, record["input_elements"])
        require(int(bits.sum(dtype=np.int64)) ==
                int(record["active_elements"]),
                "payload popcount mismatch")
        vectors = as_tbhwc(bits, record["operator"], record["input_shape"])
        analyzed = analyze_vectors(vectors)
        require(analyzed["source_bits"]["zero"] ==
                int(record["active_elements"]), "zero source mismatch")
        row = {
            "active_elements": int(record["active_elements"]),
            "analysis": analyzed,
            "file_sha256": record["file_sha256"],
            "input_elements": int(record["input_elements"]),
            "module_index": int(record["module_index"]),
            "name": record["name"],
            "operator": record["operator"],
            "ordinal": ordinal,
            "relative_path": record["relative_path"],
            "sample_id": int(record["sample_id"]),
        }
        rows.append(row)
        module_key = str(record["module_index"])
        sample_key = str(record["sample_id"])
        if module_key not in per_module:
            per_module[module_key] = empty_totals()
            per_module[module_key]["module_index"] = int(record["module_index"])
            per_module[module_key]["name"] = record["name"]
            per_module[module_key]["operator"] = record["operator"]
        if sample_key not in per_sample:
            per_sample[sample_key] = empty_totals()
            per_sample[sample_key]["sample_id"] = int(record["sample_id"])
            per_sample[sample_key]["sample_key"] = record["sample_key"]
            per_sample[sample_key]["sequence_key"] = record["sequence_key"]
        add_totals(aggregate, analyzed, record)
        add_totals(per_module[module_key], analyzed, record)
        add_totals(per_sample[sample_key], analyzed, record)

    require(aggregate["input_elements"] ==
            contract["expected"]["input_elements"] and
            aggregate["zero_source_bits"] ==
            contract["expected"]["active_elements"],
            "aggregate population mismatch")
    for key in sorted(per_module, key=int):
        finish_totals(per_module[key])
    for key in sorted(per_sample, key=int):
        finish_totals(per_sample[key])
    finish_totals(aggregate)
    result = {
        "aggregate": aggregate,
        "claim_boundary": contract["claim_boundary"],
        "contract_sha256": sha256_path(arguments.contract),
        "identity": {
            "manifest_sha256": sha256_path(arguments.manifest),
            "payload_root": str(payload_root),
        },
        "parent_policy": contract["parent_policy"],
        "per_module": [per_module[key] for key in sorted(per_module, key=int)],
        "per_record": rows,
        "per_sample": [per_sample[key] for key in sorted(per_sample, key=int)],
        "schema": "m55_h67_full_network_dual_parent_opportunity_result_v1",
        "status": "PASS_EXACT_SOURCE_BIT_WORK_NO_CYCLE_SPEEDUP_ENERGY_OR_PPA_CLAIM",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.{}".format(os.getpid()))
    require(not temporary.exists(), "refusing existing temporary output")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.link(str(temporary), str(output))
    temporary.unlink()
    print(json.dumps({
        "aggregate": aggregate,
        "output": str(output),
        "output_sha256": sha256_path(output),
        "status": result["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
