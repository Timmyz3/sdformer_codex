#!/usr/bin/env python3
"""Fail-closed arithmetic validator for the canonical M56 r1 DSE."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SHA = {
    "analyzer": "713705f3550646fe63efaca60493dad4bad4f07cf4d447ba7b9dfd405b68b67e",
    "contract": "cba82292504bfaa1015f54e254a27719749122c9783ad16d6f0ff6a6cc961263",
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "m55_result": "9639903ea82e90b1a8403ff0bee66b01ec732ee6baa11d275ec2725e0a4d531b",
    "result": "1aca6c0d6215f91035434cca45a04dd1d21100f1e5bbd2138851c575188b808a",
}
MODES = ("zero", "local", "dual")
WIDTHS = (1, 2, 4, 8, 16, 24, 32, 40, 48)


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


def close(actual, expected, label):
    require(abs(float(actual) - float(expected)) <=
            1e-12 * max(1.0, abs(float(expected))),
            "float mismatch: {}".format(label))


def empty_mode():
    return {
        "allocated_lane_product_slots": 0,
        "event_cycle_histogram": dict((str(index), 0) for index in range(13)),
        "event_cycles": 0,
        "event_plus_one_commit_cycle_per_group": 0,
        "groups": 0,
        "physical_product_slots": 0,
        "product_updates": 0,
        "union_source_indices": 0,
        "zero_event_groups": 0,
    }


def add_mode(total, row):
    for key in ("allocated_lane_product_slots", "event_cycles",
                "event_plus_one_commit_cycle_per_group", "groups",
                "physical_product_slots", "product_updates",
                "union_source_indices", "zero_event_groups"):
        total[key] += int(row[key])
    for key, value in row["event_cycle_histogram"].items():
        total["event_cycle_histogram"][key] += int(value)


def validate_mode(row, label):
    histogram = row["event_cycle_histogram"]
    require(set(histogram) == set(str(index) for index in range(13)) and
            all(isinstance(value, int) and value >= 0
                for value in histogram.values()),
            "histogram format mismatch: {}".format(label))
    require(sum(histogram.values()) == row["groups"] and
            sum(int(key) * value for key, value in histogram.items()) ==
            row["event_cycles"] and
            histogram["0"] == row["zero_event_groups"],
            "histogram total mismatch: {}".format(label))
    require(row["event_plus_one_commit_cycle_per_group"] ==
            row["event_cycles"] + row["groups"] and
            row["physical_product_slots"] ==
            row["event_cycles"] * 8 * 96 and
            0 <= row["allocated_lane_product_slots"] <=
            row["physical_product_slots"] and
            0 <= row["product_updates"] <=
            row["allocated_lane_product_slots"],
            "cycle/slot equation mismatch: {}".format(label))
    close(row["physical_lane_utilization"],
          (float(row["product_updates"]) /
           float(row["physical_product_slots"])
           if row["physical_product_slots"] else 1.0),
          label + ".physical_util")
    close(row["allocated_lane_utilization"],
          (float(row["product_updates"]) /
           float(row["allocated_lane_product_slots"])
           if row["allocated_lane_product_slots"] else 1.0),
          label + ".allocated_util")
    nonzero_groups = row["groups"] - row["zero_event_groups"]
    require(row["union_source_indices"] <= row["event_cycles"] * 8 and
            row["union_source_indices"] >=
            row["event_cycles"] * 8 - nonzero_groups * 7,
            "union/ceil envelope mismatch: {}".format(label))


def compare_mode(actual, expected, label):
    for key in expected:
        require(actual[key] == expected[key],
                "aggregate mismatch: {}.{}".format(label, key))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", default=str(
        root / "system_simulator/scripts/analyze_m56_prediction_head_lane_fold_dse.py"))
    parser.add_argument("--contract", default=str(
        root / "contracts/m56_prediction_head_lane_fold_dse_contract_r1_20260823.json"))
    parser.add_argument("--manifest", default=str(
        root / "results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/manifest.json"))
    parser.add_argument("--m55-result", default=str(
        root / "results/m55_h67_full_network_dual_parent_opportunity_r1_20260823/"
        "m55_h67_full_network_dual_parent_opportunity_result_r1.json"))
    parser.add_argument("--result", default=str(
        root / "results/m56_prediction_head_lane_fold_dse_r1_20260823/"
        "m56_prediction_head_lane_fold_dse_result_r1.json"))
    arguments = parser.parse_args()
    paths = {"analyzer": arguments.analyzer, "contract": arguments.contract,
             "manifest": arguments.manifest, "m55_result": arguments.m55_result,
             "result": arguments.result}
    for name, path in paths.items():
        require(Path(path).is_file() and sha256_path(path) == EXPECTED_SHA[name],
                "{} identity mismatch".format(name))
    contract = strict_json(arguments.contract)
    manifest = strict_json(arguments.manifest)
    m55_result = strict_json(arguments.m55_result)
    result = strict_json(arguments.result)
    require(result["schema"] == "m56_prediction_head_lane_fold_dse_result_v1" and
            result["status"] ==
            "PASS_EXACT_HEAD_SOURCE_ISSUE_DSE_NO_SYSTEM_RTL_PPA_ENERGY_CLAIM" and
            result["contract_sha256"] == EXPECTED_SHA["contract"],
            "result status/contract mismatch")
    forbidden = " ".join(contract["claim_boundary"]["forbidden"]).lower()
    require("system speedup" in forbidden and "rtl" in forbidden and
            "energy" in forbidden and "numerical-equivalence" in forbidden,
            "claim boundary weakened")
    module_rows = [row for row in manifest["records"]
                   if row["module_index"] == 30]
    require(len(module_rows) == 10 and len(result["per_record"]) == 10,
            "record population mismatch")
    require(result["source_bits"] == {
        "zero": m55_result["per_module"][30]["zero_source_bits"],
        "local": m55_result["per_module"][30]["local_source_bits"],
        "dual": m55_result["per_module"][30]["dual_source_bits"],
    } and result["parent_choice_counts"]["dual"] ==
            m55_result["per_module"][30]["choice_counts"],
            "M55 reconciliation mismatch")

    totals = dict((width, dict((mode, empty_mode()) for mode in MODES))
                  for width in WIDTHS)
    for index, (row, source) in enumerate(zip(result["per_record"], module_rows)):
        require(row["sample_id"] == index and
                row["relative_path"] == source["relative_path"] and
                row["file_sha256"] == source["file_sha256"],
                "record identity mismatch")
        require(row["source_bits"]["zero"] == source["active_elements"] and
                row["source_bits"]["dual"] <= row["source_bits"]["local"] <=
                row["source_bits"]["zero"], "record source monotonicity mismatch")
        require(set(row["widths"]) == set(str(value) for value in WIDTHS),
                "record width population mismatch")
        for width in WIDTHS:
            observed = row["widths"][str(width)]
            expected_groups = 10 * 240 * ((320 + width - 1) // width)
            for mode in MODES:
                validate_mode(observed[mode],
                              "record{}.width{}.{}".format(index, width, mode))
                require(observed[mode]["groups"] == expected_groups and
                        observed[mode]["product_updates"] ==
                        2 * row["source_bits"][mode],
                        "record group/product mismatch")
                add_mode(totals[width][mode], observed[mode])

    require([row["pixels_per_group"] for row in result["widths"]] ==
            list(WIDTHS), "aggregate width order mismatch")
    for width, actual in zip(WIDTHS, result["widths"]):
        groups = totals[width]["zero"]["groups"]
        require(actual["pixels_per_group"] == width and
                actual["fixed_dense"] == {
                    "event_cycles": groups * 12,
                    "event_plus_one_commit_cycle_per_group": groups * 13,
                    "groups": groups,
                }, "fixed dense equation mismatch")
        for mode in MODES:
            expected = totals[width][mode]
            compare_mode({key: actual["modes"][mode][key] for key in expected},
                         expected, "width{}.{}".format(width, mode))
            validate_mode(actual["modes"][mode],
                          "aggregate.width{}.{}".format(width, mode))
        ratios = actual["head_kernel_ratios_not_system_speedup"]
        close(ratios["dense_over_dual_event_cycles"],
              float(actual["fixed_dense"]["event_cycles"]) /
              float(actual["modes"]["dual"]["event_cycles"]), "dense/dual")
        close(ratios["dense_over_dual_event_plus_commit"],
              float(actual["fixed_dense"][
                  "event_plus_one_commit_cycle_per_group"]) /
              float(actual["modes"]["dual"][
                  "event_plus_one_commit_cycle_per_group"]), "dense/dual+commit")
        close(ratios["local_over_dual_event_cycles"],
              float(actual["modes"]["local"]["event_cycles"]) /
              float(actual["modes"]["dual"]["event_cycles"]), "local/dual")
        close(ratios["zero_over_dual_event_cycles"],
              float(actual["modes"]["zero"]["event_cycles"]) /
              float(actual["modes"]["dual"]["event_cycles"]), "zero/dual")
    require(result["selected_by_minimum_dual_event_cycles"] == 48 and
            result["selected_by_minimum_dual_event_plus_commit"] == 48,
            "selected width mismatch")
    width1 = result["widths"][0]
    width48 = result["widths"][-1]
    print(json.dumps({
        "p1_dual_over_p48_dual_event_cycle_ratio_not_system":
            float(width1["modes"]["dual"]["event_cycles"]) /
            float(width48["modes"]["dual"]["event_cycles"]),
        "p1_dual_over_p48_dual_event_plus_commit_ratio_not_system":
            float(width1["modes"]["dual"][
                "event_plus_one_commit_cycle_per_group"]) /
            float(width48["modes"]["dual"][
                "event_plus_one_commit_cycle_per_group"]),
        "p48_dense_over_dual_event_plus_commit_not_system":
            width48["head_kernel_ratios_not_system_speedup"][
                "dense_over_dual_event_plus_commit"],
        "status": "PASS_M56_EXACT_HEAD_DSE_NO_SYSTEM_RTL_PPA_ENERGY_CLAIM",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
