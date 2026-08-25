#!/usr/bin/env python3
"""Bounded signed tile DSE repairing the M56 head opportunity P1s."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
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


def ceil_bytes(bits):
    return (int(bits) + 7) // 8


def apply_candidate(best, choice, signed, candidate_signed, candidate_id,
                    target_slice, validity):
    target_best = best[target_slice]
    target_choice = choice[target_slice]
    target_signed = signed[target_slice]
    candidate_cost = np.abs(candidate_signed).sum(axis=-1, dtype=np.int32)
    take = np.logical_and(candidate_cost < target_best, validity)
    target_best[take] = candidate_cost[take]
    target_choice[take] = candidate_id
    target_signed[:] = np.where(take[..., None], candidate_signed,
                                target_signed)


def bounded_signed_residual(array, tile_h, tile_w):
    current = array.astype(np.int8, copy=False)
    zero_signed = current.copy()
    best = np.abs(zero_signed).sum(axis=-1, dtype=np.int32)
    choice = np.zeros(best.shape, dtype=np.uint8)
    signed = zero_signed.copy()

    left_signed = (current[:, :, :, 1:, :] -
                   current[:, :, :, :-1, :]).astype(np.int8, copy=False)
    left_positions = np.arange(1, array.shape[3], dtype=np.int32)
    left_valid = (left_positions % int(tile_w) != 0)[None, None, None, :]
    apply_candidate(best, choice, signed, left_signed, 1,
                    (slice(None), slice(None), slice(None), slice(1, None)),
                    left_valid)
    del left_signed

    up_signed = (current[:, :, 1:, :, :] -
                 current[:, :, :-1, :, :]).astype(np.int8, copy=False)
    up_positions = np.arange(1, array.shape[2], dtype=np.int32)
    up_valid = (up_positions % int(tile_h) != 0)[None, None, :, None]
    apply_candidate(best, choice, signed, up_signed, 2,
                    (slice(None), slice(None), slice(1, None), slice(None)),
                    up_valid)
    del up_signed

    previous_signed = (current[1:, :, :, :, :] -
                       current[:-1, :, :, :, :]).astype(np.int8, copy=False)
    apply_candidate(best, choice, signed, previous_signed, 3,
                    (slice(1, None), slice(None), slice(None), slice(None)),
                    True)
    del previous_signed
    require(np.all(np.logical_or(signed == -1,
                                 np.logical_or(signed == 0, signed == 1))),
            "signed residual outside -1/0/+1")
    absolute = np.abs(signed).astype(np.uint8)
    require(np.array_equal(absolute.sum(axis=-1, dtype=np.int32), best),
            "signed/absolute cost mismatch")
    return signed, absolute, choice


def choice_counts(choice):
    counts = np.bincount(choice.reshape(-1), minlength=4)
    return dict((PARENTS[index], int(counts[index])) for index in range(4))


def issue_metrics(absolute, tile_w):
    _, _, _, width, _ = absolute.shape
    cycles = groups = union_sources = product_updates = 0
    physical_slots = zero_groups = 0
    histogram = dict((str(index), 0) for index in range(13))
    for start in range(0, width, int(tile_w)):
        stop = min(width, start + int(tile_w))
        union = np.any(absolute[:, :, :, start:stop, :] != 0, axis=3)
        union_count = union.sum(axis=-1, dtype=np.int32)
        group_cycles = (union_count + 7) // 8
        groups += int(group_cycles.size)
        cycles += int(group_cycles.sum(dtype=np.int64))
        union_sources += int(union_count.sum(dtype=np.int64))
        zero_groups += int((group_cycles == 0).sum(dtype=np.int64))
        observed = np.bincount(group_cycles.reshape(-1), minlength=13)
        for index in range(13):
            histogram[str(index)] += int(observed[index])
    product_updates = int(absolute.sum(dtype=np.int64)) * 2
    physical_slots = cycles * 8 * 96
    return {
        "event_cycle_histogram": histogram,
        "event_cycles": cycles,
        "event_plus_one_commit_cycle_per_group": cycles + groups,
        "groups": groups,
        "physical_lane_utilization": (float(product_updates) /
                                      float(physical_slots)
                                      if physical_slots else 1.0),
        "physical_product_slots": physical_slots,
        "product_updates": product_updates,
        "union_source_indices": union_sources,
        "zero_event_groups": zero_groups,
    }


def capacity(tile_h, tile_w, contract):
    c = contract["capacity"]
    input_pair = ceil_bytes(2 * tile_h * tile_w * 96)
    accumulator_pair = ceil_bytes(
        2 * tile_h * tile_w * 2 *
        c["accumulator_bits_conditional_pending_head_int8_bridge"])
    metadata = ceil_bytes(tile_h * tile_w * 2)
    signed_masks = ceil_bytes(2 * tile_w * 96)
    dynamic = (input_pair + accumulator_pair + metadata + signed_masks +
               c["weight_cache_bytes_float32_identity"] +
               c["bias_cache_bytes_float32_identity"])
    combined = c["fixed_nonframe_bytes_from_m53_k4_ctx16"] + dynamic
    return {
        "components_bytes": {
            "activation_tile_pair": input_pair,
            "bias_cache_float32_identity":
                c["bias_cache_bytes_float32_identity"],
            "choice_metadata_2b_per_pixel": metadata,
            "conditional_19b_output_accumulator_tile_pair": accumulator_pair,
            "float32_weight_cache": c["weight_cache_bytes_float32_identity"],
            "positive_and_negative_group_masks": signed_masks,
        },
        "combined_capacity_bytes": combined,
        "dynamic_head_tile_bytes": dynamic,
        "headroom_bytes": c["maximum_combined_capacity_bytes"] - combined,
        "maximum_combined_capacity_bytes":
            c["maximum_combined_capacity_bytes"],
        "passes": combined <= c["maximum_combined_capacity_bytes"],
        "qualification": (
            "19-bit head accumulator remains conditional pending INT8 numeric bridge"),
    }


def empty_config(tile_h, tile_w, cap):
    return {
        "capacity": cap,
        "choice_counts": dict((name, 0) for name in PARENTS),
        "event_cycle_histogram": dict((str(index), 0) for index in range(13)),
        "event_cycles": 0,
        "event_plus_one_commit_cycle_per_group": 0,
        "groups": 0,
        "negative_residual_events": 0,
        "physical_product_slots": 0,
        "positive_residual_events": 0,
        "product_updates": 0,
        "source_bits": 0,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "union_source_indices": 0,
        "zero_event_groups": 0,
    }


def add_config(total, signed, absolute, choice, metrics):
    for name, value in choice_counts(choice).items():
        total["choice_counts"][name] += value
    total["positive_residual_events"] += int((signed > 0).sum(dtype=np.int64))
    total["negative_residual_events"] += int((signed < 0).sum(dtype=np.int64))
    total["source_bits"] += int(absolute.sum(dtype=np.int64))
    for key in ("event_cycles", "event_plus_one_commit_cycle_per_group",
                "groups", "physical_product_slots", "product_updates",
                "union_source_indices", "zero_event_groups"):
        total[key] += int(metrics[key])
    for key, value in metrics["event_cycle_histogram"].items():
        total["event_cycle_histogram"][key] += int(value)


def finish_config(total):
    require(total["positive_residual_events"] +
            total["negative_residual_events"] == total["source_bits"] and
            total["product_updates"] == 2 * total["source_bits"] and
            sum(total["choice_counts"].values()) == 7680000,
            "aggregate signed/choice population mismatch")
    dense_event = total["groups"] * 12
    dense_commit = total["groups"] * 13
    total["fixed_dense"] = {
        "event_cycles": dense_event,
        "event_plus_one_commit_cycle_per_group": dense_commit,
    }
    total["ratios_not_system_speedup"] = {
        "dense_over_bounded_signed_event_cycles":
            float(dense_event) / float(total["event_cycles"]),
        "dense_over_bounded_signed_event_plus_commit":
            float(dense_commit) /
            float(total["event_plus_one_commit_cycle_per_group"]),
    }
    total["physical_lane_utilization"] = (
        float(total["product_updates"]) /
        float(total["physical_product_slots"]))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--m55-result", required=True)
    parser.add_argument("--m56-result", required=True)
    parser.add_argument("--payload-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    contract = strict_json(arguments.contract)
    manifest = strict_json(arguments.manifest)
    require(contract["schema"] ==
            "m60_prediction_head_bounded_signed_tile_dse_contract_v1",
            "contract schema mismatch")
    require(sha256_path(arguments.manifest) ==
            contract["identity"]["manifest_sha256"] and
            sha256_path(arguments.m55_result) ==
            contract["identity"]["m55_result_sha256"] and
            sha256_path(arguments.m56_result) ==
            contract["identity"]["m56_result_sha256"] and
            sha256_path(Path(m55.__file__).resolve()) ==
            contract["identity"]["m55_analyzer_sha256"],
            "upstream identity mismatch")
    output = Path(arguments.output)
    require(not output.exists(), "refusing existing output")
    payload_root = Path(arguments.payload_root).resolve()
    target = [row for row in manifest["records"] if row["module_index"] == 30]
    require(len(target) == 10, "head record population mismatch")
    module = manifest["module_identities"][contract["identity"]["module_name"]]
    require(module["weight"]["shape"] == contract["identity"]["weight_shape"] and
            module["weight"]["content_sha256"] ==
            contract["identity"]["weight_content_sha256"] and
            module["bias"]["shape"] == contract["identity"]["bias_shape"] and
            module["bias"]["content_sha256"] ==
            contract["identity"]["bias_content_sha256"],
            "head weight/bias identity mismatch")

    configurations = {}
    for tile_h, tile_w in contract["geometry"]["tile_candidates_h_w"]:
        key = "H{}_W{}".format(tile_h, tile_w)
        cap = capacity(tile_h, tile_w, contract)
        configurations[key] = empty_config(tile_h, tile_w, cap)
    per_record = []
    for record in target:
        path = payload_root / record["relative_path"]
        require(path.is_file() and path.stat().st_size == record["packed_bytes"]
                and sha256_path(path) == record["file_sha256"],
                "head payload identity mismatch")
        bits = m55.unpack_little(path, record["input_elements"])
        array = m55.as_tbhwc(bits, record["operator"], record["input_shape"])
        record_configs = {}
        for tile_h, tile_w in contract["geometry"]["tile_candidates_h_w"]:
            key = "H{}_W{}".format(tile_h, tile_w)
            signed, absolute, choice = bounded_signed_residual(
                array, tile_h, tile_w)
            metrics = issue_metrics(absolute, tile_w)
            add_config(configurations[key], signed, absolute, choice, metrics)
            record_configs[key] = {
                "choice_counts": choice_counts(choice),
                "negative_residual_events": int((signed < 0).sum(dtype=np.int64)),
                "positive_residual_events": int((signed > 0).sum(dtype=np.int64)),
                "source_bits": int(absolute.sum(dtype=np.int64)),
                "issue": metrics,
            }
        per_record.append({
            "configs": record_configs,
            "file_sha256": record["file_sha256"],
            "relative_path": record["relative_path"],
            "sample_id": record["sample_id"],
        })
    finished = [finish_config(configurations[key]) for key in sorted(configurations)]
    feasible = [row for row in finished if row["capacity"]["passes"]]
    require(feasible, "no capacity-feasible tile")
    selected = min(feasible,
                   key=lambda row: row["event_plus_one_commit_cycle_per_group"])
    result = {
        "claim_boundary": contract["claim_boundary"],
        "configurations": finished,
        "contract_sha256": sha256_path(arguments.contract),
        "identity": contract["identity"],
        "per_record": per_record,
        "schema": "m60_prediction_head_bounded_signed_tile_dse_result_v1",
        "selected_capacity_feasible_tile": {
            "tile_h": selected["tile_h"], "tile_w": selected["tile_w"]},
        "status": "PASS_BOUNDED_SIGNED_HEAD_DSE_INT8_RTL_PPA_SYSTEM_OPEN",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.link(str(temporary), str(output))
    temporary.unlink()
    print(json.dumps({
        "output_sha256": sha256_path(output),
        "selected": selected,
        "status": result["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
