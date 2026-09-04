#!/usr/bin/env python3
"""Independent raw-bitpack reconstruction for the M55/M56 hammer review.

This implementation does not import either producer analyzer.  It recomputes
all payload SHA/size/popcount/layout identities, parent choices, signed masks,
and the module-30 row-bounded lane-fold sweep directly from the 310 files.
"""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np


PARENTS = ("zero", "left", "up", "previous_timestep")
MODES_M55 = ("zero", "local", "motion", "dual")
MODES_M56 = ("zero", "local", "dual")
WIDTHS = (1, 2, 4, 8, 16, 24, 32, 40, 48)
ISSUE_WIDTH = 8
PHYSICAL_LANES = 96
OUTPUT_CHANNELS = 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON: {}".format(raw))

    def pairs(raw_pairs):
        result = {}
        for key, value in raw_pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


LITTLE_LUT = ((np.arange(256, dtype=np.uint16)[:, None] >>
               np.arange(8, dtype=np.uint16)[None, :]) & 1).astype(np.uint8)


def unpack_record(path, record):
    payload = path.read_bytes()
    elements = int(record["input_elements"])
    require(len(payload) == int(record["packed_bytes"]) == (elements + 7) // 8,
            "payload size mismatch: {}".format(record["relative_path"]))
    observed_sha = sha256_bytes(payload)
    require(observed_sha == record["file_sha256"],
            "payload SHA mismatch: {}".format(record["relative_path"]))
    packed = np.frombuffer(payload, dtype=np.uint8)
    if elements % 8:
        used_mask = (1 << (elements % 8)) - 1
        require((int(packed[-1]) & ~used_mask) == 0,
                "nonzero high tail padding: {}".format(record["relative_path"]))
    flat = LITTLE_LUT[packed].reshape(-1)[:elements]
    popcount = int(flat.sum(dtype=np.int64))
    require(popcount == int(record["active_elements"]),
            "payload popcount mismatch: {}".format(record["relative_path"]))
    shape = [int(value) for value in record["input_shape"]]
    require(product(shape) == elements, "shape mismatch")
    operator = record["operator"]
    if operator == "Linear":
        require(len(shape) == 5, "Linear rank/layout mismatch")
        array = flat.reshape(shape)
    elif operator == "Conv2d" and len(shape) == 5:
        array = flat.reshape(shape).transpose(0, 1, 3, 4, 2)
    elif operator == "Conv2d" and len(shape) == 4:
        array = flat.reshape(shape).transpose(0, 2, 3, 1)[:, None, :, :, :]
    else:
        raise ValueError("unsupported layout {} {}".format(operator, shape))
    require(array.ndim == 5 and int(array.shape[0]) == 10,
            "normalized T,B,H,W,C layout mismatch")
    return payload, array, popcount


def fresh_mode(zero_cost):
    return {"cost": zero_cost.copy(),
            "choice": np.zeros(zero_cost.shape, dtype=np.uint8),
            "positive": zero_cost.copy(),
            "negative": np.zeros(zero_cost.shape, dtype=np.int32)}


def apply_cost(mode, candidate_cost, candidate_positive, candidate_negative,
               candidate_id, target_slice):
    target_cost = mode["cost"][target_slice]
    take = candidate_cost < target_cost
    target_cost[take] = candidate_cost[take]
    target_choice = mode["choice"][target_slice]
    target_choice[take] = candidate_id
    target_positive = mode["positive"][target_slice]
    target_positive[take] = candidate_positive[take]
    target_negative = mode["negative"][target_slice]
    target_negative[take] = candidate_negative[take]


def signed_cost(target_cost, parent_cost, xor_cost):
    # For binary vectors: xor=positive+negative and
    # target_pop-parent_pop=positive-negative.
    positive = (xor_cost + target_cost - parent_cost) // 2
    negative = (xor_cost - target_cost + parent_cost) // 2
    require(np.all(positive >= 0) and np.all(negative >= 0) and
            np.array_equal(positive + negative, xor_cost),
            "signed residual decomposition mismatch")
    return positive, negative


def analyze_parent_costs(array):
    zero_cost = array.sum(axis=-1, dtype=np.int32)
    local = fresh_mode(zero_cost)
    motion = fresh_mode(zero_cost)

    left_xor = np.not_equal(array[:, :, :, 1:, :],
                            array[:, :, :, :-1, :]).sum(axis=-1,
                                                         dtype=np.int32)
    left_pos, left_neg = signed_cost(zero_cost[:, :, :, 1:],
                                     zero_cost[:, :, :, :-1], left_xor)
    apply_cost(local, left_xor, left_pos, left_neg, 1,
               (slice(None), slice(None), slice(None), slice(1, None)))
    left_total = int(left_xor.sum(dtype=np.int64))
    del left_xor, left_pos, left_neg

    up_xor = np.not_equal(array[:, :, 1:, :, :],
                          array[:, :, :-1, :, :]).sum(axis=-1,
                                                       dtype=np.int32)
    up_pos, up_neg = signed_cost(zero_cost[:, :, 1:, :],
                                 zero_cost[:, :, :-1, :], up_xor)
    apply_cost(local, up_xor, up_pos, up_neg, 2,
               (slice(None), slice(None), slice(1, None), slice(None)))
    up_total = int(up_xor.sum(dtype=np.int64))
    del up_xor, up_pos, up_neg

    dual = {key: value.copy() for key, value in local.items()}
    previous_xor = np.not_equal(array[1:, :, :, :, :],
                                array[:-1, :, :, :, :]).sum(axis=-1,
                                                              dtype=np.int32)
    previous_pos, previous_neg = signed_cost(zero_cost[1:], zero_cost[:-1],
                                              previous_xor)
    previous_slice = (slice(1, None), slice(None), slice(None), slice(None))
    apply_cost(motion, previous_xor, previous_pos, previous_neg, 3,
               previous_slice)
    apply_cost(dual, previous_xor, previous_pos, previous_neg, 3,
               previous_slice)
    previous_total = int(previous_xor.sum(dtype=np.int64))
    del previous_xor, previous_pos, previous_neg

    zero = fresh_mode(zero_cost)
    modes = {"zero": zero, "local": local, "motion": motion, "dual": dual}
    result = {
        "candidate_xor_bits_on_valid_coordinates": {
            "left": left_total, "up": up_total,
            "previous_timestep": previous_total},
        "choice_counts": {},
        "signed_source_bits": {},
        "source_bits": {},
        "source_bits_by_timestep": {},
        "vector_count": int(zero_cost.size),
        "vector_shape_tbhwc": [int(value) for value in array.shape],
    }
    for name, mode in modes.items():
        result["source_bits"][name] = int(mode["cost"].sum(dtype=np.int64))
        result["source_bits_by_timestep"][name] = [
            int(value) for value in mode["cost"].reshape(
                mode["cost"].shape[0], -1).sum(axis=1, dtype=np.int64)]
        positive = int(mode["positive"].sum(dtype=np.int64))
        negative = int(mode["negative"].sum(dtype=np.int64))
        require(positive + negative == result["source_bits"][name],
                "signed/source total mismatch")
        result["signed_source_bits"][name] = {
            "positive_0_to_1": positive,
            "negative_1_to_0": negative,
        }
        counts = np.bincount(mode["choice"].reshape(-1), minlength=4)
        result["choice_counts"][name] = dict(
            (PARENTS[index], int(counts[index])) for index in range(4))
    return result


def choose_signed_residuals(array):
    target = array.astype(np.int8)
    zero_cost = array.sum(axis=-1, dtype=np.int32)
    local_cost = zero_cost.copy()
    local = target.copy()

    left_parent = array[:, :, :, :-1, :]
    left_target = array[:, :, :, 1:, :]
    left_bits = left_target.astype(np.int8) - left_parent.astype(np.int8)
    left_cost = np.count_nonzero(left_bits, axis=-1).astype(np.int32)
    take = left_cost < local_cost[:, :, :, 1:]
    local_cost[:, :, :, 1:][take] = left_cost[take]
    local_slice = local[:, :, :, 1:, :]
    local_slice[:] = np.where(take[..., None], left_bits, local_slice)
    del left_parent, left_target, left_bits, left_cost, take

    up_parent = array[:, :, :-1, :, :]
    up_target = array[:, :, 1:, :, :]
    up_bits = up_target.astype(np.int8) - up_parent.astype(np.int8)
    up_cost = np.count_nonzero(up_bits, axis=-1).astype(np.int32)
    take = up_cost < local_cost[:, :, 1:, :]
    local_cost[:, :, 1:, :][take] = up_cost[take]
    local_slice = local[:, :, 1:, :, :]
    local_slice[:] = np.where(take[..., None], up_bits, local_slice)
    del up_parent, up_target, up_bits, up_cost, take

    dual_cost = local_cost.copy()
    dual = local.copy()
    previous_parent = array[:-1, :, :, :, :]
    previous_target = array[1:, :, :, :, :]
    previous_bits = previous_target.astype(np.int8) - previous_parent.astype(np.int8)
    previous_cost = np.count_nonzero(previous_bits, axis=-1).astype(np.int32)
    take = previous_cost < dual_cost[1:]
    dual_cost[1:][take] = previous_cost[take]
    dual_slice = dual[1:]
    dual_slice[:] = np.where(take[..., None], previous_bits, dual_slice)
    del previous_parent, previous_target, previous_bits, previous_cost, take
    return {"zero": target, "local": local, "dual": dual}


def empty_width(width):
    def mode():
        return {
            "event_cycle_histogram": dict((str(index), 0) for index in range(13)),
            "event_cycles": 0, "event_plus_one_commit_cycle_per_group": 0,
            "groups": 0, "union_source_indices": 0,
            "product_updates": 0, "positive_product_updates": 0,
            "negative_product_updates": 0,
            "physical_product_slots": 0,
            "allocated_lane_product_slots": 0, "zero_event_groups": 0,
        }
    return {"pixels_per_group": width,
            "modes": dict((name, mode()) for name in MODES_M56)}


def add_lane_fold(total, residual, width):
    t_size, b_size, height, image_width, channels = residual.shape
    require((t_size, b_size, height, image_width, channels) ==
            (10, 1, 240, 320, 96), "module30 normalized geometry mismatch")
    mode = total
    positive_sources = int((residual > 0).sum(dtype=np.int64))
    negative_sources = int((residual < 0).sum(dtype=np.int64))
    mode["positive_product_updates"] += positive_sources * OUTPUT_CHANNELS
    mode["negative_product_updates"] += negative_sources * OUTPUT_CHANNELS
    mode["product_updates"] += (positive_sources + negative_sources) * OUTPUT_CHANNELS
    for start in range(0, image_width, width):
        stop = min(image_width, start + width)
        pixels = stop - start
        chunk = residual[:, :, :, start:stop, :]
        union = np.any(chunk != 0, axis=3)
        union_count = union.sum(axis=-1, dtype=np.int32)
        cycles = (union_count + ISSUE_WIDTH - 1) // ISSUE_WIDTH
        groups = int(cycles.size)
        event_cycles = int(cycles.sum(dtype=np.int64))
        mode["groups"] += groups
        mode["event_cycles"] += event_cycles
        mode["event_plus_one_commit_cycle_per_group"] += event_cycles + groups
        mode["union_source_indices"] += int(union_count.sum(dtype=np.int64))
        mode["physical_product_slots"] += (
            event_cycles * ISSUE_WIDTH * PHYSICAL_LANES)
        mode["allocated_lane_product_slots"] += (
            event_cycles * ISSUE_WIDTH * pixels * OUTPUT_CHANNELS)
        mode["zero_event_groups"] += int((cycles == 0).sum(dtype=np.int64))
        histogram = np.bincount(cycles.reshape(-1), minlength=13)
        require(len(histogram) <= 13, "event histogram exceeds ceil(96/8)")
        for index in range(13):
            mode["event_cycle_histogram"][str(index)] += int(histogram[index])


def finish_width(row):
    groups = row["modes"]["zero"]["groups"]
    require(all(mode["groups"] == groups for mode in row["modes"].values()),
            "mode group mismatch")
    for mode in row["modes"].values():
        require(sum(mode["event_cycle_histogram"].values()) == mode["groups"] and
                sum(int(key) * value
                    for key, value in mode["event_cycle_histogram"].items()) ==
                mode["event_cycles"] and
                mode["positive_product_updates"] +
                mode["negative_product_updates"] == mode["product_updates"],
                "event/signed conservation mismatch")
    dense_event = groups * 12
    row["fixed_dense"] = {
        "groups": groups,
        "event_cycles": dense_event,
        "event_plus_one_commit_cycle_per_group": dense_event + groups,
    }
    dual = row["modes"]["dual"]
    row["ratios_not_system_speedup"] = {
        "dense_over_dual_event_cycles":
            float(dense_event) / float(dual["event_cycles"]),
        "dense_over_dual_event_plus_commit":
            float(dense_event + groups) /
            float(dual["event_plus_one_commit_cycle_per_group"]),
    }
    return row


def fresh_totals():
    return {
        "choice_counts": dict((name, 0) for name in PARENTS),
        "hook_calls": 0, "input_elements": 0, "vector_count": 0,
        "source_bits": dict((mode, 0) for mode in MODES_M55),
        "signed_source_bits": dict((mode, {
            "positive_0_to_1": 0, "negative_1_to_0": 0})
            for mode in MODES_M55),
    }


def add_totals(total, analysis, record):
    total["hook_calls"] += 1
    total["input_elements"] += int(record["input_elements"])
    total["vector_count"] += int(analysis["vector_count"])
    for mode in MODES_M55:
        total["source_bits"][mode] += int(analysis["source_bits"][mode])
        for direction in ("positive_0_to_1", "negative_1_to_0"):
            total["signed_source_bits"][mode][direction] += int(
                analysis["signed_source_bits"][mode][direction])
    for parent in PARENTS:
        total["choice_counts"][parent] += int(
            analysis["choice_counts"]["dual"][parent])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--m55-result", type=Path, required=True)
    parser.add_argument("--m56-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing output overwrite")
    manifest = strict_json(args.manifest)
    m55 = strict_json(args.m55_result)
    m56 = strict_json(args.m56_result)
    require(len(manifest["records"]) == 310, "manifest record population")
    require(sha256_path(args.manifest) ==
            "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
            "manifest SHA drift")
    require(sha256_path(args.m55_result) ==
            "9639903ea82e90b1a8403ff0bee66b01ec732ee6baa11d275ec2725e0a4d531b",
            "M55 result SHA drift")
    require(sha256_path(args.m56_result) ==
            "1aca6c0d6215f91035434cca45a04dd1d21100f1e5bbd2138851c575188b808a",
            "M56 result SHA drift")

    aggregate = fresh_totals()
    modules = dict((index, fresh_totals()) for index in range(31))
    samples = dict((index, fresh_totals()) for index in range(10))
    widths = dict((width, empty_width(width)) for width in WIDTHS)
    per_record = []
    collection = hashlib.sha256()
    identities = set()
    for ordinal, record in enumerate(manifest["records"]):
        sample = int(record["sample_id"])
        module = int(record["module_index"])
        require((sample, module) not in identities and sample in range(10) and
                module in range(31), "record Cartesian identity")
        identities.add((sample, module))
        path = args.payload_root / record["relative_path"]
        require(path.is_file(), "missing payload {}".format(path))
        payload, array, popcount = unpack_record(path, record)
        analysis = analyze_parent_costs(array)
        require(analysis["source_bits"]["zero"] == popcount,
                "zero/popcount mismatch")
        add_totals(aggregate, analysis, record)
        add_totals(modules[module], analysis, record)
        add_totals(samples[sample], analysis, record)
        collection.update(record["relative_path"].encode("utf-8") + b"\0")
        collection.update(bytes.fromhex(record["file_sha256"]))
        collection.update(int(record["packed_bytes"]).to_bytes(8, "big"))
        row = {
            "active_elements": popcount,
            "file_sha256": record["file_sha256"],
            "input_elements": int(record["input_elements"]),
            "module_index": module,
            "normalized_shape_tbhwc": analysis["vector_shape_tbhwc"],
            "operator": record["operator"],
            "ordinal": ordinal,
            "packed_bytes": int(record["packed_bytes"]),
            "relative_path": record["relative_path"],
            "sample_id": sample,
            "source_bits": analysis["source_bits"],
            "signed_source_bits": analysis["signed_source_bits"],
            "choice_counts": analysis["choice_counts"],
            "source_bits_by_timestep": analysis["source_bits_by_timestep"],
        }
        per_record.append(row)
        if module == 30:
            require(list(array.shape) == [10, 1, 240, 320, 96],
                    "prediction-head layout mismatch")
            residuals = choose_signed_residuals(array)
            for width in WIDTHS:
                for mode in MODES_M56:
                    add_lane_fold(widths[width]["modes"][mode],
                                  residuals[mode], width)
            del residuals
        del payload, array
        if (ordinal + 1) % 10 == 0 or ordinal == 309:
            print("review_reconstruction_progress={}/310".format(ordinal + 1),
                  flush=True)

    require(identities == set((sample, module) for sample in range(10)
                              for module in range(31)),
            "record Cartesian population incomplete")
    final_widths = [finish_width(widths[width]) for width in WIDTHS]
    module30 = modules[30]
    marginal_all = (aggregate["source_bits"]["local"] -
                    aggregate["source_bits"]["dual"])
    marginal_head = (module30["source_bits"]["local"] -
                     module30["source_bits"]["dual"])
    contribution = 100.0 * float(marginal_head) / float(marginal_all)
    p1 = final_widths[0]
    p48 = final_widths[-1]
    result = {
        "schema": "m55_m56_raw_payload_independent_reconstruction_v1",
        "status": "PASS_ALL310_RAW_SHA_POPCOUNT_LAYOUT_PARENT_SIGNED_AND_HEAD_DSE",
        "identity": {
            "manifest_sha256": sha256_path(args.manifest),
            "m55_result_sha256": sha256_path(args.m55_result),
            "m56_result_sha256": sha256_path(args.m56_result),
            "reviewer_defined_payload_collection_sha256": collection.hexdigest(),
        },
        "population": {
            "records": len(per_record), "samples": 10, "modules": 31,
            "input_elements": sum(row["input_elements"] for row in per_record),
            "packed_bytes": sum(row["packed_bytes"] for row in per_record),
            "active_elements": sum(row["active_elements"] for row in per_record),
        },
        "aggregate": aggregate,
        "per_module": [modules[index] for index in range(31)],
        "per_sample": [samples[index] for index in range(10)],
        "per_record": per_record,
        "module30_marginal_dual_vs_local_contribution": {
            "aggregate_saved_source_bits": marginal_all,
            "module30_saved_source_bits": marginal_head,
            "percent": contribution,
        },
        "module30_lane_fold_widths": final_widths,
        "headline_checks_not_system_speedup": {
            "p48_dual_event_cycles": p48["modes"]["dual"]["event_cycles"],
            "p48_dual_event_plus_commit": p48["modes"]["dual"][
                "event_plus_one_commit_cycle_per_group"],
            "p48_dense_over_dual_plus_commit": p48[
                "ratios_not_system_speedup"]["dense_over_dual_event_plus_commit"],
            "p1_dual_over_p48_dual_event_cycles":
                float(p1["modes"]["dual"]["event_cycles"]) /
                float(p48["modes"]["dual"]["event_cycles"]),
        },
        "claim_boundary": {
            "source_work_or_issue_opportunity_only": True,
            "system_speedup": False,
            "parent_state_or_memory_cycles": False,
            "rtl_or_ppa": False,
            "int8_or_numerical_equivalence": False,
        },
    }
    require(result["population"] == {
        "records": 310, "samples": 10, "modules": 31,
        "input_elements": 10506240000, "packed_bytes": 1313280000,
        "active_elements": 712894209}, "frozen population mismatch")
    require(abs(contribution - 84.74818112765011) < 1e-12,
            "module30 contribution mismatch")
    require(p48["modes"]["dual"]["event_cycles"] == 539614 and
            p48["modes"]["dual"]["event_plus_one_commit_cycle_per_group"] ==
            707614, "P48 headline mismatch")
    require(not args.output.exists(), "refusing output overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp.{}".format(os.getpid()))
    require(not temporary.exists(), "temporary exists")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.link(str(temporary), str(args.output))
    temporary.unlink()
    print("PASS M55+M56 independent raw reconstruction output_sha256={}".format(
        sha256_path(args.output)))


if __name__ == "__main__":
    main()
