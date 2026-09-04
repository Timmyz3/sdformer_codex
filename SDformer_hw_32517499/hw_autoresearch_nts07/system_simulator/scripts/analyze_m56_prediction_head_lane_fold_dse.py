#!/usr/bin/env python3
"""Exact P8/L96 lane-fold source-union DSE for the H67 prediction head."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import analyze_m55_h67_full_network_dual_parent_opportunity as m55


PARENTS = ("zero", "left", "up", "previous_timestep")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def apply_candidate(best, choice, residual, candidate_cost, candidate_bits,
                    candidate_id, target_slice):
    target_best = best[target_slice]
    target_choice = choice[target_slice]
    target_residual = residual[target_slice]
    take = candidate_cost < target_best
    target_best[take] = candidate_cost[take]
    target_choice[take] = candidate_id
    target_residual[:] = np.where(take[..., None], candidate_bits,
                                  target_residual)


def make_residuals(array):
    zero = array
    zero_cost = zero.sum(axis=-1, dtype=np.int32)
    local_cost = zero_cost.copy()
    local_choice = np.zeros(zero_cost.shape, dtype=np.uint8)
    local = zero.copy()

    left_bits = np.not_equal(array[:, :, :, 1:, :],
                             array[:, :, :, :-1, :]).astype(np.uint8)
    left_cost = left_bits.sum(axis=-1, dtype=np.int32)
    apply_candidate(local_cost, local_choice, local, left_cost, left_bits, 1,
                    (slice(None), slice(None), slice(None), slice(1, None)))
    del left_bits, left_cost

    up_bits = np.not_equal(array[:, :, 1:, :, :],
                           array[:, :, :-1, :, :]).astype(np.uint8)
    up_cost = up_bits.sum(axis=-1, dtype=np.int32)
    apply_candidate(local_cost, local_choice, local, up_cost, up_bits, 2,
                    (slice(None), slice(None), slice(1, None), slice(None)))
    del up_bits, up_cost

    dual_cost = local_cost.copy()
    dual_choice = local_choice.copy()
    dual = local.copy()
    previous_bits = np.not_equal(array[1:, :, :, :, :],
                                 array[:-1, :, :, :, :]).astype(np.uint8)
    previous_cost = previous_bits.sum(axis=-1, dtype=np.int32)
    apply_candidate(dual_cost, dual_choice, dual, previous_cost,
                    previous_bits, 3,
                    (slice(1, None), slice(None), slice(None), slice(None)))
    del previous_bits, previous_cost
    return {"zero": zero, "local": local, "dual": dual}, {
        "local": local_choice, "dual": dual_choice}


def count_choices(choice):
    counts = np.bincount(choice.reshape(-1), minlength=4)
    return dict((PARENTS[index], int(counts[index])) for index in range(4))


def mode_width_metrics(residual, pixels, issue_width, physical_lanes,
                       output_channels):
    timesteps, batches, height, width, channels = residual.shape
    require(channels == 96 and pixels <= physical_lanes // output_channels,
            "lane-fold geometry mismatch")
    event_cycles = 0
    groups = 0
    union_source_indices = 0
    physical_product_slots = 0
    allocated_lane_product_slots = 0
    zero_event_groups = 0
    event_cycle_histogram = dict((str(index), 0) for index in range(13))
    for start in range(0, width, pixels):
        stop = min(width, start + pixels)
        group_pixels = stop - start
        chunk = residual[:, :, :, start:stop, :]
        union = np.any(chunk != 0, axis=3)
        union_count = union.sum(axis=-1, dtype=np.int32)
        cycles = (union_count + issue_width - 1) // issue_width
        count = int(cycles.size)
        cycle_sum = int(cycles.sum(dtype=np.int64))
        groups += count
        event_cycles += cycle_sum
        union_source_indices += int(union_count.sum(dtype=np.int64))
        physical_product_slots += cycle_sum * issue_width * physical_lanes
        allocated_lane_product_slots += (
            cycle_sum * issue_width * group_pixels * output_channels)
        zero_event_groups += int((cycles == 0).sum(dtype=np.int64))
        histogram = np.bincount(cycles.reshape(-1), minlength=13)
        for index in range(13):
            event_cycle_histogram[str(index)] += int(histogram[index])
    product_updates = int(residual.sum(dtype=np.int64)) * output_channels
    require(sum(event_cycle_histogram.values()) == groups and
            event_cycles == sum(int(key) * value
                                for key, value in event_cycle_histogram.items()),
            "histogram mismatch")
    return {
        "allocated_lane_product_slots": allocated_lane_product_slots,
        "allocated_lane_utilization": (float(product_updates) /
                                        float(allocated_lane_product_slots)
                                        if allocated_lane_product_slots else 1.0),
        "event_cycle_histogram": event_cycle_histogram,
        "event_cycles": event_cycles,
        "event_plus_one_commit_cycle_per_group": event_cycles + groups,
        "groups": groups,
        "physical_lane_utilization": (float(product_updates) /
                                      float(physical_product_slots)
                                      if physical_product_slots else 1.0),
        "physical_product_slots": physical_product_slots,
        "product_updates": product_updates,
        "union_source_indices": union_source_indices,
        "zero_event_groups": zero_event_groups,
    }


def empty_width(pixels):
    return {
        "modes": dict((mode, {
            "allocated_lane_product_slots": 0,
            "event_cycle_histogram": dict((str(index), 0)
                                           for index in range(13)),
            "event_cycles": 0,
            "event_plus_one_commit_cycle_per_group": 0,
            "groups": 0,
            "physical_product_slots": 0,
            "product_updates": 0,
            "union_source_indices": 0,
            "zero_event_groups": 0,
        }) for mode in ("zero", "local", "dual")),
        "pixels_per_group": pixels,
    }


def add_width(total, observed):
    for mode in ("zero", "local", "dual"):
        for key in ("allocated_lane_product_slots", "event_cycles",
                    "event_plus_one_commit_cycle_per_group", "groups",
                    "physical_product_slots", "product_updates",
                    "union_source_indices", "zero_event_groups"):
            total["modes"][mode][key] += int(observed[mode][key])
        for key, value in observed[mode]["event_cycle_histogram"].items():
            total["modes"][mode]["event_cycle_histogram"][key] += int(value)


def finish_width(total, channels, issue_width):
    modes = total["modes"]
    for mode in modes:
        row = modes[mode]
        row["allocated_lane_utilization"] = (
            float(row["product_updates"]) /
            float(row["allocated_lane_product_slots"])
            if row["allocated_lane_product_slots"] else 1.0)
        row["physical_lane_utilization"] = (
            float(row["product_updates"]) /
            float(row["physical_product_slots"])
            if row["physical_product_slots"] else 1.0)
    groups = modes["zero"]["groups"]
    require(modes["local"]["groups"] == groups and
            modes["dual"]["groups"] == groups, "mode group mismatch")
    dense_event = groups * ((channels + issue_width - 1) // issue_width)
    dense_commit = dense_event + groups
    total["fixed_dense"] = {
        "event_cycles": dense_event,
        "event_plus_one_commit_cycle_per_group": dense_commit,
        "groups": groups,
    }
    total["head_kernel_ratios_not_system_speedup"] = {
        "dense_over_dual_event_cycles": float(dense_event) /
        float(modes["dual"]["event_cycles"]),
        "dense_over_dual_event_plus_commit": float(dense_commit) /
        float(modes["dual"]["event_plus_one_commit_cycle_per_group"]),
        "local_over_dual_event_cycles":
        float(modes["local"]["event_cycles"]) /
        float(modes["dual"]["event_cycles"]),
        "zero_over_dual_event_cycles":
        float(modes["zero"]["event_cycles"]) /
        float(modes["dual"]["event_cycles"]),
    }
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--m55-result", required=True)
    parser.add_argument("--payload-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    contract = strict_json(arguments.contract)
    manifest = strict_json(arguments.manifest)
    m55_result = strict_json(arguments.m55_result)
    payload_root = Path(arguments.payload_root).resolve()
    output = Path(arguments.output)
    require(contract["schema"] ==
            "m56_prediction_head_lane_fold_dse_contract_v1",
            "contract schema mismatch")
    require(sha256_path(arguments.manifest) ==
            contract["identity"]["manifest_sha256"] and
            sha256_path(arguments.m55_result) ==
            contract["identity"]["m55_result_sha256"] and
            sha256_path(Path(m55.__file__).resolve()) ==
            contract["identity"]["m55_analyzer_sha256"],
            "upstream identity mismatch")
    require(not output.exists(), "refusing existing output")
    target_records = [row for row in manifest["records"]
                      if row["module_index"] ==
                      contract["identity"]["module_index"]]
    require(len(target_records) == contract["population"]["records"] and
            sum(row["input_elements"] for row in target_records) ==
            contract["population"]["input_elements"],
            "target population mismatch")
    module_name = contract["identity"]["module_name"]
    identity = manifest["module_identities"][module_name]
    require(identity["weight"]["shape"] ==
            contract["identity"]["weight_shape"] and
            identity["weight"]["content_sha256"] ==
            contract["identity"]["weight_content_sha256"],
            "prediction-head weight identity mismatch")

    geometry = contract["geometry"]
    widths = dict((pixels, empty_width(pixels))
                  for pixels in geometry["swept_pixels_per_group"])
    per_record = []
    total_choices = dict((mode, dict((parent, 0) for parent in PARENTS))
                         for mode in ("local", "dual"))
    source_bits = dict((mode, 0) for mode in ("zero", "local", "dual"))
    m55_module30 = m55_result["per_module"][30]
    for record in target_records:
        path = payload_root / record["relative_path"]
        require(path.is_file() and path.stat().st_size == record["packed_bytes"]
                and sha256_path(path) == record["file_sha256"],
                "payload identity mismatch")
        bits = m55.unpack_little(path, record["input_elements"])
        array = m55.as_tbhwc(bits, record["operator"], record["input_shape"])
        require(list(array.shape) == [10, 1, 240, 320, 96],
                "head layout mismatch")
        residuals, choices = make_residuals(array)
        record_widths = {}
        for pixels in geometry["swept_pixels_per_group"]:
            observed = {}
            for mode in ("zero", "local", "dual"):
                observed[mode] = mode_width_metrics(
                    residuals[mode], pixels,
                    geometry["source_channels_issued_per_cycle"],
                    geometry["physical_output_lanes"],
                    geometry["output_channels"])
            add_width(widths[pixels], observed)
            record_widths[str(pixels)] = observed
        record_choices = dict((mode, count_choices(choices[mode]))
                              for mode in choices)
        record_sources = dict((mode, int(residuals[mode].sum(dtype=np.int64)))
                              for mode in residuals)
        for mode in source_bits:
            source_bits[mode] += record_sources[mode]
        for mode in total_choices:
            for parent in PARENTS:
                total_choices[mode][parent] += record_choices[mode][parent]
        per_record.append({
            "choice_counts": record_choices,
            "file_sha256": record["file_sha256"],
            "relative_path": record["relative_path"],
            "sample_id": record["sample_id"],
            "source_bits": record_sources,
            "widths": record_widths,
        })

    require(source_bits["zero"] == m55_module30["zero_source_bits"] and
            source_bits["local"] == m55_module30["local_source_bits"] and
            source_bits["dual"] == m55_module30["dual_source_bits"] and
            total_choices["dual"] == m55_module30["choice_counts"],
            "M55 module-30 reconciliation mismatch")
    final_widths = [finish_width(widths[pixels], 96, 8)
                    for pixels in geometry["swept_pixels_per_group"]]
    best_event = min(final_widths,
                     key=lambda row: row["modes"]["dual"]["event_cycles"])
    best_commit = min(
        final_widths,
        key=lambda row: row["modes"]["dual"][
            "event_plus_one_commit_cycle_per_group"])
    result = {
        "claim_boundary": contract["claim_boundary"],
        "contract_sha256": sha256_path(arguments.contract),
        "identity": contract["identity"],
        "parent_choice_counts": total_choices,
        "per_record": per_record,
        "schema": "m56_prediction_head_lane_fold_dse_result_v1",
        "selected_by_minimum_dual_event_cycles":
            best_event["pixels_per_group"],
        "selected_by_minimum_dual_event_plus_commit":
            best_commit["pixels_per_group"],
        "source_bits": source_bits,
        "status": "PASS_EXACT_HEAD_SOURCE_ISSUE_DSE_NO_SYSTEM_RTL_PPA_ENERGY_CLAIM",
        "widths": final_widths,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.link(str(temporary), str(output))
    temporary.unlink()
    print(json.dumps({
        "best_event": best_event,
        "best_event_plus_commit": best_commit,
        "output_sha256": sha256_path(output),
        "source_bits": source_bits,
        "status": result["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
