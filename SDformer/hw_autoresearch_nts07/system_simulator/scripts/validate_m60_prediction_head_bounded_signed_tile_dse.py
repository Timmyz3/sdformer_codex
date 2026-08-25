#!/usr/bin/env python3
"""Fail-closed validator for the corrected M60 bounded signed head DSE."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SHA = {
    "analyzer": "4159e0d695854b4eb4a0b1fa00ce8bb663eda1fa39fbb2d39454fe9784eba868",
    "contract": "fdcf83f8b1ddc1011cad4999d85fbca0c58480b98105150c6f8e4616e86b9448",
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "m55_result": "9639903ea82e90b1a8403ff0bee66b01ec732ee6baa11d275ec2725e0a4d531b",
    "m56_result": "1aca6c0d6215f91035434cca45a04dd1d21100f1e5bbd2138851c575188b808a",
    "result": "71fd1eac81daa63b4cf13eb6191c9b31d7b5c754b498175d09a60ba536ed091d",
}
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


def integer(value, label):
    require(isinstance(value, int) and not isinstance(value, bool),
            "non-integer {}".format(label))
    return value


def close(actual, expected, label):
    require(isinstance(actual, float) and not isinstance(actual, bool),
            "non-float {}".format(label))
    require(abs(actual - expected) <= 1e-12 * max(1.0, abs(expected)),
            "float mismatch {}".format(label))


def ceil_bytes(bits):
    return (integer(bits, "bits") + 7) // 8


def empty_total():
    return {
        "choice_counts": dict((name, 0) for name in PARENTS),
        "event_cycle_histogram": dict((str(index), 0)
                                       for index in range(13)),
        "event_cycles": 0,
        "event_plus_one_commit_cycle_per_group": 0,
        "groups": 0,
        "negative_residual_events": 0,
        "physical_product_slots": 0,
        "positive_residual_events": 0,
        "product_updates": 0,
        "source_bits": 0,
        "union_source_indices": 0,
        "zero_event_groups": 0,
    }


def validate_issue(issue, source_bits, groups, label):
    histogram = issue["event_cycle_histogram"]
    require(set(histogram) == set(str(index) for index in range(13)),
            "histogram keys {}".format(label))
    for key, value in histogram.items():
        require(integer(value, label + ".hist." + key) >= 0,
                "negative histogram")
    event_cycles = integer(issue["event_cycles"], label + ".events")
    physical_slots = integer(issue["physical_product_slots"],
                             label + ".slots")
    product_updates = integer(issue["product_updates"], label + ".products")
    union_sources = integer(issue["union_source_indices"], label + ".union")
    zero_groups = integer(issue["zero_event_groups"], label + ".zero_groups")
    require(integer(issue["groups"], label + ".groups") == groups and
            sum(histogram.values()) == groups and
            sum(int(key) * value for key, value in histogram.items()) ==
            event_cycles and histogram["0"] == zero_groups,
            "histogram equation {}".format(label))
    require(integer(issue["event_plus_one_commit_cycle_per_group"],
                    label + ".commit") == event_cycles + groups and
            physical_slots == event_cycles * 8 * 96 and
            product_updates == 2 * source_bits,
            "cycle/product equation {}".format(label))
    nonzero_groups = groups - zero_groups
    require(union_sources <= event_cycles * 8 and
            union_sources >= event_cycles * 8 - nonzero_groups * 7,
            "union ceil envelope {}".format(label))
    close(issue["physical_lane_utilization"],
          (float(product_updates) / float(physical_slots)
           if physical_slots else 1.0), label + ".util")


def add_record(total, row):
    for name in PARENTS:
        total["choice_counts"][name] += row["choice_counts"][name]
    total["positive_residual_events"] += row["positive_residual_events"]
    total["negative_residual_events"] += row["negative_residual_events"]
    total["source_bits"] += row["source_bits"]
    issue = row["issue"]
    for key in ("event_cycles", "event_plus_one_commit_cycle_per_group",
                "groups", "physical_product_slots", "product_updates",
                "union_source_indices", "zero_event_groups"):
        total[key] += issue[key]
    for key in total["event_cycle_histogram"]:
        total["event_cycle_histogram"][key] += \
            issue["event_cycle_histogram"][key]


def capacity(contract, tile_h, tile_w):
    cap = contract["capacity"]
    components = {
        "activation_tile_pair": ceil_bytes(2 * tile_h * tile_w * 96),
        "bias_cache_float32_identity":
            cap["bias_cache_bytes_float32_identity"],
        "choice_metadata_2b_per_pixel": ceil_bytes(tile_h * tile_w * 2),
        "conditional_19b_output_accumulator_tile_pair": ceil_bytes(
            2 * tile_h * tile_w * 2 *
            cap["accumulator_bits_conditional_pending_head_int8_bridge"]),
        "float32_weight_cache": cap["weight_cache_bytes_float32_identity"],
        "positive_and_negative_group_masks":
            ceil_bytes(2 * tile_w * 96),
    }
    dynamic = sum(components.values())
    combined = cap["fixed_nonframe_bytes_from_m53_k4_ctx16"] + dynamic
    maximum = cap["maximum_combined_capacity_bytes"]
    return {
        "components_bytes": components,
        "combined_capacity_bytes": combined,
        "dynamic_head_tile_bytes": dynamic,
        "headroom_bytes": maximum - combined,
        "maximum_combined_capacity_bytes": maximum,
        "passes": combined <= maximum,
        "qualification":
            "19-bit head accumulator remains conditional pending INT8 numeric bridge",
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", default=str(root /
        "system_simulator/scripts/analyze_m60_prediction_head_bounded_signed_tile_dse.py"))
    parser.add_argument("--contract", default=str(root /
        "contracts/m60_prediction_head_bounded_signed_tile_dse_contract_r1_20260823.json"))
    parser.add_argument("--manifest", default=str(root /
        "results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/manifest.json"))
    parser.add_argument("--m55-result", default=str(root /
        "results/m55_h67_full_network_dual_parent_opportunity_r1_20260823/"
        "m55_h67_full_network_dual_parent_opportunity_result_r1.json"))
    parser.add_argument("--m56-result", default=str(root /
        "results/m56_prediction_head_lane_fold_dse_r1_20260823/"
        "m56_prediction_head_lane_fold_dse_result_r1.json"))
    parser.add_argument("--result", default=str(root /
        "results/m60_prediction_head_bounded_signed_tile_dse_r1_20260823/"
        "m60_prediction_head_bounded_signed_tile_dse_result_r2.json"))
    args = parser.parse_args()
    paths = {"analyzer": args.analyzer, "contract": args.contract,
             "manifest": args.manifest, "m55_result": args.m55_result,
             "m56_result": args.m56_result, "result": args.result}
    for name, path in paths.items():
        require(Path(path).is_file() and sha256_path(path) == EXPECTED_SHA[name],
                "{} identity mismatch".format(name))
    contract = strict_json(args.contract)
    manifest = strict_json(args.manifest)
    m55 = strict_json(args.m55_result)
    result = strict_json(args.result)
    require(contract["schema"] ==
            "m60_prediction_head_bounded_signed_tile_dse_contract_v1" and
            result["schema"] ==
            "m60_prediction_head_bounded_signed_tile_dse_result_v1" and
            result["status"] ==
            "PASS_BOUNDED_SIGNED_HEAD_DSE_INT8_RTL_PPA_SYSTEM_OPEN" and
            result["contract_sha256"] == EXPECTED_SHA["contract"] and
            result["claim_boundary"] == contract["claim_boundary"],
            "schema/status/claim mismatch")
    forbidden = " ".join(contract["claim_boundary"]["forbidden"]).lower()
    require("system speedup" in forbidden and "rtl" in forbidden and
            "numerically qualified" in forbidden and "dram" in forbidden,
            "claim boundary weakened")
    module = manifest["module_identities"][contract["identity"]["module_name"]]
    require(module["weight"]["content_sha256"] ==
            contract["identity"]["weight_content_sha256"] and
            module["bias"]["content_sha256"] ==
            contract["identity"]["bias_content_sha256"] and
            module["weight"]["content_bytes"] == 768 and
            module["bias"]["content_bytes"] == 8,
            "weight/bias identity mismatch")
    records = [row for row in manifest["records"] if row["module_index"] == 30]
    m55_by_path = dict((row["relative_path"], row)
                       for row in m55["per_record"]
                       if row["module_index"] == 30)
    require(len(records) == len(result["per_record"]) == 10,
            "record population")
    candidates = [tuple(row) for row in
                  contract["geometry"]["tile_candidates_h_w"]]
    aggregate = dict((item, empty_total()) for item in candidates)
    for index, (record, source) in enumerate(zip(result["per_record"], records)):
        require(record["sample_id"] == source["sample_id"] == index and
                record["relative_path"] == source["relative_path"] and
                record["file_sha256"] == source["file_sha256"],
                "record identity {}".format(index))
        require(set(record["configs"]) ==
                set("H{}_W{}".format(*item) for item in candidates),
                "record configs {}".format(index))
        for tile_h, tile_w in candidates:
            key = "H{}_W{}".format(tile_h, tile_w)
            row = record["configs"][key]
            choices = row["choice_counts"]
            require(set(choices) == set(PARENTS) and
                    all(integer(value, key + ".choice") >= 0
                        for value in choices.values()) and
                    sum(choices.values()) == 768000,
                    "choice conservation {}".format(key))
            source_bits = integer(row["source_bits"], key + ".source")
            require(row["positive_residual_events"] +
                    row["negative_residual_events"] == source_bits and
                    source_bits >= m55_by_path[source["relative_path"]][
                        "analysis"]["source_bits"]["dual"],
                    "signed/source conservation {}".format(key))
            groups = 10 * 240 * ((320 + tile_w - 1) // tile_w)
            validate_issue(row["issue"], source_bits, groups,
                           "record{}.{}".format(index, key))
            add_record(aggregate[(tile_h, tile_w)], row)
    rows = dict(((row["tile_h"], row["tile_w"]), row)
                for row in result["configurations"])
    require(set(rows) == set(candidates), "aggregate candidates")
    for item in candidates:
        row = rows[item]
        expected = aggregate[item]
        for key in expected:
            require(row[key] == expected[key],
                    "aggregate mismatch {}.{}".format(item, key))
        require(row["capacity"] == capacity(contract, *item),
                "capacity mismatch {}".format(item))
        require(row["positive_residual_events"] +
                row["negative_residual_events"] == row["source_bits"] and
                sum(row["choice_counts"].values()) == 7680000,
                "aggregate conservation {}".format(item))
        dense_event = row["groups"] * 12
        dense_commit = row["groups"] * 13
        require(row["fixed_dense"] == {
            "event_cycles": dense_event,
            "event_plus_one_commit_cycle_per_group": dense_commit},
            "dense equation {}".format(item))
        close(row["ratios_not_system_speedup"][
                  "dense_over_bounded_signed_event_cycles"],
              float(dense_event) / float(row["event_cycles"]),
              "event ratio {}".format(item))
        close(row["ratios_not_system_speedup"][
                  "dense_over_bounded_signed_event_plus_commit"],
              float(dense_commit) /
              float(row["event_plus_one_commit_cycle_per_group"]),
              "commit ratio {}".format(item))
    feasible = [row for row in rows.values() if row["capacity"]["passes"]]
    selected = min(feasible,
                   key=lambda row: row["event_plus_one_commit_cycle_per_group"])
    require(result["selected_capacity_feasible_tile"] == {
                "tile_h": selected["tile_h"], "tile_w": selected["tile_w"]} ==
            {"tile_h": 16, "tile_w": 48}, "selected tile mismatch")
    require(selected["source_bits"] == 52069263 and
            selected["positive_residual_events"] == 46676443 and
            selected["negative_residual_events"] == 5392820 and
            selected["event_cycles"] == 539623 and
            selected["event_plus_one_commit_cycle_per_group"] == 707623 and
            selected["capacity"]["combined_capacity_bytes"] == 67736 and
            selected["capacity"]["headroom_bytes"] == 125992,
            "selected frozen anchor mismatch")
    print(json.dumps({
        "selected": result["selected_capacity_feasible_tile"],
        "signed": {"positive": selected["positive_residual_events"],
                   "negative": selected["negative_residual_events"]},
        "event_cycles": selected["event_cycles"],
        "event_plus_commit":
            selected["event_plus_one_commit_cycle_per_group"],
        "combined_capacity_bytes":
            selected["capacity"]["combined_capacity_bytes"],
        "status": "PASS_M60_BOUNDED_SIGNED_HEAD_OPPORTUNITY_ONLY",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
