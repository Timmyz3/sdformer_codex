#!/usr/bin/env python3
"""Fail-closed structural and arithmetic validator for the canonical M55 r1."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "analyzer": "9532e09845956abde97138fc763d704e963c408291bea72675181b67047620c3",
    "contract": "31df83ef6adf6b1e567deeaa6cce1af8e3b4e6f7f35a092e47133a59f00a5bda",
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "payload_receipt": "d37e26a9e3206229746eb21209603376a4c07c3aa69f7500d0b960f64c580c32",
    "result": "9639903ea82e90b1a8403ff0bee66b01ec732ee6baa11d275ec2725e0a4d531b",
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


def empty_totals():
    return {
        "choice_counts": dict((name, 0) for name in PARENTS),
        "dual_source_bits": 0,
        "hook_calls": 0,
        "input_elements": 0,
        "local_source_bits": 0,
        "motion_source_bits": 0,
        "vector_count": 0,
        "zero_source_bits": 0,
    }


def add(total, row):
    analysis = row["analysis"]
    total["hook_calls"] += 1
    total["input_elements"] += int(row["input_elements"])
    total["vector_count"] += int(analysis["vector_count"])
    for name in ("zero", "local", "motion", "dual"):
        total[name + "_source_bits"] += int(analysis["source_bits"][name])
    for name in PARENTS:
        total["choice_counts"][name] += int(
            analysis["choice_counts"]["dual"][name])


def finish(total):
    zero = total["zero_source_bits"]
    local = total["local_source_bits"]
    motion = total["motion_source_bits"]
    dual = total["dual_source_bits"]
    require(min(zero, local, motion, dual) > 0, "zero source total")
    total["opportunity_ratios_not_speedup"] = {
        "zero_over_dual_source_work": float(zero) / float(dual),
        "zero_over_local_source_work": float(zero) / float(local),
        "zero_over_motion_source_work": float(zero) / float(motion),
    }
    total["marginal_dual_reduction_vs_local_percent"] = (
        100.0 * float(local - dual) / float(local))
    total["dual_reduction_vs_zero_percent"] = (
        100.0 * float(zero - dual) / float(zero))
    return total


def equal_value(actual, expected, label):
    if isinstance(expected, float):
        require(isinstance(actual, (int, float)) and
                abs(float(actual) - expected) <= 1e-12 *
                max(1.0, abs(expected)), "float mismatch: {}".format(label))
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                "dict keys mismatch: {}".format(label))
        for key in expected:
            equal_value(actual[key], expected[key], label + "." + str(key))
    else:
        require(actual == expected, "value mismatch: {}".format(label))


def validate_record(row, manifest_record, ordinal):
    require(row["ordinal"] == ordinal and
            row["sample_id"] == manifest_record["sample_id"] and
            row["module_index"] == manifest_record["module_index"] and
            row["name"] == manifest_record["name"] and
            row["operator"] == manifest_record["operator"] and
            row["relative_path"] == manifest_record["relative_path"] and
            row["file_sha256"] == manifest_record["file_sha256"] and
            row["input_elements"] == manifest_record["input_elements"] and
            row["active_elements"] == manifest_record["active_elements"],
            "record/manifest mismatch at {}".format(ordinal))
    analysis = row["analysis"]
    vector_count = analysis["vector_count"]
    require(vector_count > 0 and
            analysis["choice_metadata_bits_if_naive_2b_per_vector"] ==
            2 * vector_count, "metadata/vector mismatch")
    require(len(analysis["vector_shape_tbhwc"]) == 5 and
            analysis["vector_shape_tbhwc"][0] == 10 and
            vector_count ==
            analysis["vector_shape_tbhwc"][0] *
            analysis["vector_shape_tbhwc"][1] *
            analysis["vector_shape_tbhwc"][2] *
            analysis["vector_shape_tbhwc"][3],
            "vector shape/count mismatch")
    sources = analysis["source_bits"]
    require(sources["zero"] == row["active_elements"] and
            0 <= sources["dual"] <= sources["local"] <= sources["zero"] and
            0 <= sources["dual"] <= sources["motion"] <= sources["zero"],
            "source monotonicity mismatch")
    for mode in ("zero", "local", "motion", "dual"):
        by_t = analysis["source_bits_by_timestep"][mode]
        require(len(by_t) == 10 and all(isinstance(value, int) and value >= 0
                                       for value in by_t) and
                sum(by_t) == sources[mode],
                "timestep source mismatch")
    for mode in ("local", "motion", "dual"):
        counts = analysis["choice_counts"][mode]
        require(set(counts) == set(PARENTS) and
                all(isinstance(value, int) and value >= 0
                    for value in counts.values()) and
                sum(counts.values()) == vector_count,
                "choice population mismatch")
    require(analysis["choice_counts"]["local"]["previous_timestep"] == 0 and
            analysis["choice_counts"]["motion"]["left"] == 0 and
            analysis["choice_counts"]["motion"]["up"] == 0,
            "mode parent policy mismatch")
    candidates = analysis["candidate_xor_bits_on_valid_coordinates"]
    require(set(candidates) == {"left", "up", "previous_timestep"} and
            all(isinstance(value, int) and value >= 0
                for value in candidates.values()),
            "candidate XOR mismatch")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", default=str(
        root / "system_simulator/scripts/"
        "analyze_m55_h67_full_network_dual_parent_opportunity.py"))
    parser.add_argument("--contract", default=str(
        root / "contracts/"
        "m55_h67_full_network_dual_parent_opportunity_contract_r1_20260823.json"))
    parser.add_argument("--manifest", default=str(
        root / "results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/"
        "manifest.json"))
    parser.add_argument("--payload-receipt", default=str(
        root / "results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/"
        "m51_h67_ep35_binary_input_trace_gpu_payload_validation_receipt_r1.json"))
    parser.add_argument("--result", default=str(
        root / "results/m55_h67_full_network_dual_parent_opportunity_r1_20260823/"
        "m55_h67_full_network_dual_parent_opportunity_result_r1.json"))
    arguments = parser.parse_args()
    paths = {
        "analyzer": arguments.analyzer,
        "contract": arguments.contract,
        "manifest": arguments.manifest,
        "payload_receipt": arguments.payload_receipt,
        "result": arguments.result,
    }
    for name, path in paths.items():
        require(Path(path).is_file(), "missing {}".format(name))
        require(sha256_path(path) == EXPECTED[name],
                "{} SHA mismatch".format(name))

    contract = strict_json(arguments.contract)
    manifest = strict_json(arguments.manifest)
    receipt = strict_json(arguments.payload_receipt)
    result = strict_json(arguments.result)
    require(receipt["status"] ==
            "PASS_REAL_GPU_ALL310_PAYLOAD_SHA_SIZE_POPCOUNT_PLAN_IDENTITY" and
            all(receipt["checks"].values()), "upstream payload receipt mismatch")
    require(result["schema"] ==
            "m55_h67_full_network_dual_parent_opportunity_result_v1" and
            result["status"] ==
            "PASS_EXACT_SOURCE_BIT_WORK_NO_CYCLE_SPEEDUP_ENERGY_OR_PPA_CLAIM" and
            result["contract_sha256"] == EXPECTED["contract"] and
            result["identity"]["manifest_sha256"] == EXPECTED["manifest"],
            "result identity/status mismatch")
    forbidden = " ".join(contract["claim_boundary"]["forbidden"]).lower()
    require("speedup" in forbidden and "energy" in forbidden and
            "ppa" in forbidden, "claim boundary weakened")
    rows = result["per_record"]
    require(len(rows) == 310 and len(manifest["records"]) == 310,
            "record count mismatch")

    aggregate = empty_totals()
    modules = dict((index, empty_totals()) for index in range(31))
    samples = dict((index, empty_totals()) for index in range(10))
    identities = set()
    for ordinal, (row, manifest_row) in enumerate(zip(rows,
                                                       manifest["records"])):
        validate_record(row, manifest_row, ordinal)
        identity = (row["sample_id"], row["module_index"])
        require(identity not in identities, "duplicate record identity")
        identities.add(identity)
        add(aggregate, row)
        add(modules[row["module_index"]], row)
        add(samples[row["sample_id"]], row)
    require(identities == set((sample, module) for sample in range(10)
                              for module in range(31)),
            "sample/module Cartesian population mismatch")
    equal_value(result["aggregate"], finish(aggregate), "aggregate")

    require(len(result["per_module"]) == 31 and
            len(result["per_sample"]) == 10, "summary population mismatch")
    for index, actual in enumerate(result["per_module"]):
        expected = finish(modules[index])
        manifest_row = manifest["records"][index]
        expected.update({"module_index": index,
                         "name": manifest_row["name"],
                         "operator": manifest_row["operator"]})
        equal_value(actual, expected, "module{}".format(index))
    for index, actual in enumerate(result["per_sample"]):
        expected = finish(samples[index])
        manifest_row = manifest["records"][index * 31]
        expected.update({"sample_id": index,
                         "sample_key": manifest_row["sample_key"],
                         "sequence_key": manifest_row["sequence_key"]})
        equal_value(actual, expected, "sample{}".format(index))
    require(result["aggregate"]["zero_source_bits"] == 712894209 and
            result["aggregate"]["input_elements"] == 10506240000 and
            result["aggregate"]["hook_calls"] == 310,
            "frozen aggregate population mismatch")
    print("PASS M55 exact source-bit opportunity; no cycle/speedup/energy/PPA claim")


if __name__ == "__main__":
    main()
