#!/usr/bin/env python3
"""Fail-closed G7/G8 eligibility audit on frozen H67 traces.

This milestone deliberately does not modify the model or assign cycles.  It
checks whether the proposed amplitude and whole-FFN-token lossy axes have an
opportunity large enough to justify a modified-forward experiment.
"""

from __future__ import division

import argparse
from collections import Counter
import csv
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

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def classify_operator(name, operator):
    if ".mlp.fc1" in name:
        return "ffn_fc1"
    if ".mlp.fc2" in name:
        return "ffn_fc2"
    if "patch_embed" in name:
        return "patch_embed_and_residual_encoding"
    if operator == "Conv2d":
        return "other_conv"
    if "attn." in name:
        return "attention_linear_projection"
    if operator == "Linear":
        return "other_linear"
    return "other"


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return float(numerator) / float(denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract_path = args.contract.resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m286_nonattention_lossy_axis_eligibility_contract_v1",
            "M286 contract schema drift")
    hw = contract_path.parents[1]
    repo = hw.parent
    observed = {}
    observed["contract"] = {
        "path": str(contract_path.relative_to(hw)),
        "sha256": sha256(contract_path),
    }
    observed["analyzer"] = {
        "path": str(source_path.relative_to(repo)),
        "sha256": source_start,
    }
    paths = {}
    for label, spec in contract["identity"].items():
        path = (hw / spec["path"]).resolve()
        require(path.is_file(), "missing M286 input: " + str(path))
        digest = sha256(path)
        require(digest == spec["sha256"],
                "M286 input SHA drift for {}: {}".format(label, digest))
        paths[label] = path
        observed[label] = {"path": spec["path"], "sha256": digest}

    profile = strict_json(paths["full_network_profile"])
    atlif = profile["atlif_summary"]
    require(atlif["num_modules"] == 105 and
            atlif["official_atlif_modules"] == 105,
            "official ATLIF module population drift")
    require(atlif["ternary_activity_mean"] == 0.0 and
            atlif["binary_activity_mean"] == atlif["activity_mean"],
            "H67 ATLIF output-mode drift")
    source_text = paths["atlif_source"].read_text(encoding="utf-8")
    require("return active * thre, thre_updates" in source_text,
            "official binary ATLIF output equation drift")

    theta_rows = []
    for theta in contract["policy"]["g7_theta_grid"]:
        theta = float(theta)
        # Official binary ATLIF nonzero output magnitude is exactly its scalar
        # threshold.  A theta below the minimum threshold cannot remove a
        # single nonzero event from any official module.
        removes_any_official_event = theta > float(atlif["threshold_min"])
        theta_rows.append({
            "theta": theta,
            "below_or_equal_minimum_official_nonzero_amplitude":
                theta <= float(atlif["threshold_min"]),
            "additional_official_atlif_event_sparsity":
                None if removes_any_official_event else 0.0,
            "removes_any_official_event": removes_any_official_event,
        })
    require(all(row["additional_official_atlif_event_sparsity"] == 0.0
                for row in theta_rows),
            "precommitted G7 theta grid unexpectedly crosses ATLIF threshold")

    with paths["operator_runtime"].open(newline="", encoding="utf-8") as handle:
        operator_rows = list(csv.DictReader(handle))
    require(len(operator_rows) == 79, "operator runtime population drift")
    classes = {}
    total_dense = 0.0
    total_proxy = 0.0
    for row in operator_rows:
        dense = float(row["dense_macs"])
        proxy = float(row["activity_weighted_macs_proxy"])
        total_dense += dense
        total_proxy += proxy
        label = classify_operator(row["name"], row["operator"])
        entry = classes.setdefault(label, {"operators": 0, "dense_macs": 0.0,
                                           "activity_weighted_macs_proxy": 0.0})
        entry["operators"] += 1
        entry["dense_macs"] += dense
        entry["activity_weighted_macs_proxy"] += proxy
    for entry in classes.values():
        entry["dense_share"] = entry["dense_macs"] / total_dense
        entry["activity_weighted_proxy_share"] = (
            entry["activity_weighted_macs_proxy"] / total_proxy)
        entry["input_activity_proxy"] = (
            entry["activity_weighted_macs_proxy"] / entry["dense_macs"])

    manifest = strict_json(paths["m51_manifest"])
    require(manifest["packing"]["layout"] == "C_ORDER_FLAT" and
            manifest["packing"]["bit_order"] == "LITTLE_WITHIN_BYTE",
            "M51 packing drift")
    lookup = np.asarray([bin(value).count("1") for value in range(256)],
                        dtype=np.uint8)
    thresholds = [int(value) for value in
                  contract["policy"]["g8_event_count_grid"]]
    aggregate = {
        "records": 0, "per_timestep_tokens": 0, "source_events": 0,
        "whole_temporal_tokens": 0,
        "per_timestep_skipped_tokens": Counter(),
        "per_timestep_removed_events": Counter(),
        "whole_temporal_skipped_tokens": Counter(),
        "whole_temporal_removed_events": Counter(),
    }
    module_names = set()
    for record in manifest["records"]:
        if ".mlp.fc1" not in record["name"]:
            continue
        payload = paths["m51_manifest"].parent / record["relative_path"]
        if not payload.is_file():
            continue
        require(sha256(payload) == record["file_sha256"],
                "M51 FC1 payload SHA drift: " + record["relative_path"])
        shape = [int(value) for value in record["input_shape"]]
        channels = shape[-1]
        require(channels % 8 == 0 and
                product(shape) == int(record["input_elements"]),
                "M51 FC1 payload geometry drift")
        packed = np.fromfile(str(payload), dtype=np.uint8)
        require(packed.size * 8 == int(record["input_elements"]),
                "M51 FC1 packed extent drift")
        per_timestep = lookup[packed.reshape(-1, channels // 8)].sum(
            axis=1, dtype=np.uint16)
        temporal = per_timestep.reshape(shape[:-1]).sum(
            axis=0, dtype=np.uint16).reshape(-1)
        require(int(per_timestep.sum()) == int(record["active_elements"]),
                "M51 FC1 active count drift")
        aggregate["records"] += 1
        aggregate["per_timestep_tokens"] += int(per_timestep.size)
        aggregate["source_events"] += int(per_timestep.sum())
        aggregate["whole_temporal_tokens"] += int(temporal.size)
        module_names.add(record["name"])
        for threshold in thresholds:
            mask = per_timestep <= threshold
            temporal_mask = temporal <= threshold
            aggregate["per_timestep_skipped_tokens"][threshold] += int(mask.sum())
            aggregate["per_timestep_removed_events"][threshold] += int(
                per_timestep[mask].sum())
            aggregate["whole_temporal_skipped_tokens"][threshold] += int(
                temporal_mask.sum())
            aggregate["whole_temporal_removed_events"][threshold] += int(
                temporal[temporal_mask].sum())
    require(aggregate["records"] == 100 and len(module_names) == 10,
            "selected binary FC1 payload population drift")
    require(aggregate["per_timestep_tokens"] == 5520000 and
            aggregate["whole_temporal_tokens"] == 552000 and
            aggregate["source_events"] == 112213979,
            "FC1 opportunity population drift")

    g8_rows = []
    for threshold in thresholds:
        g8_rows.append({
            "maximum_source_events": threshold,
            "per_timestep_token_fraction": fraction(
                aggregate["per_timestep_skipped_tokens"][threshold],
                aggregate["per_timestep_tokens"]),
            "per_timestep_source_event_fraction_removed": fraction(
                aggregate["per_timestep_removed_events"][threshold],
                aggregate["source_events"]),
            "whole_temporal_token_fraction": fraction(
                aggregate["whole_temporal_skipped_tokens"][threshold],
                aggregate["whole_temporal_tokens"]),
            "whole_temporal_source_event_fraction_removed": fraction(
                aggregate["whole_temporal_removed_events"][threshold],
                aggregate["source_events"]),
        })

    m160 = strict_json(paths["m160_ffn_boundary"])
    require(m160["zero_input_semantics"]["full_branch_nonzero_fraction"] == 1.0,
            "M160 zero-input FFN response drift")
    require(m160["zero_input_semantics"]["sn2_active_values_on_full_zero_mlp_input"]
            == 927, "M160 zero-input trigger population drift")
    m156 = strict_json(paths["m156_weight_census"])
    require(m156["ffn"]["exact_summary"]["float_exact_zero_groups"] == 0 and
            m156["ffn"]["exact_summary"]["int8_exact_zero_groups"] == 0,
            "M156 exact FFN group-zero census drift")
    sensitivity = [row for row in
                   m156["ffn"]["low_energy_prune_sensitivities"]
                   if row["group_size"] == 16]

    result = {
        "schema": "m286_nonattention_lossy_axis_eligibility_v1",
        "status": "PASS_G7_G8_MAIN_AXIS_NO_GO_AND_G11_G12_PIVOT",
        "identity": observed,
        "operator_work_scope": {
            "rows": len(operator_rows),
            "total_dense_macs": total_dense,
            "total_activity_weighted_macs_proxy": total_proxy,
            "classes": classes,
            "warning": "operator proxy shares are not the frozen 620868243-cycle system envelope",
        },
        "g7_official_atlif_amplitude_gate": {
            "official_modules": int(atlif["official_atlif_modules"]),
            "threshold_min": float(atlif["threshold_min"]),
            "threshold_mean": float(atlif["threshold_mean"]),
            "threshold_max": float(atlif["threshold_max"]),
            "binary_activity_mean": float(atlif["binary_activity_mean"]),
            "theta_rows": theta_rows,
            "decision": "NO_GO_MAIN_AXIS_GRID_HAS_ZERO_INCREMENTAL_SPARSITY",
            "reason": "each official binary ATLIF event is exactly scalar-threshold magnitude; every precommitted theta is below the smallest nonzero magnitude",
        },
        "g8_whole_temporal_ffn_token_gate": {
            "scope": "10 binary FC1 modules, 100 local payloads, stage3 nonbinary modules excluded",
            "records": aggregate["records"],
            "per_timestep_tokens": aggregate["per_timestep_tokens"],
            "whole_temporal_tokens": aggregate["whole_temporal_tokens"],
            "source_events": aggregate["source_events"],
            "rows": g8_rows,
            "zero_input_constant_path_required": True,
            "m160_zero_input_sn2_active_values": 927,
            "decision": "NO_GO_MAIN_AXIS_SMALL_COUNT_WHOLE_T_OPPORTUNITY_NEAR_ZERO",
            "reason": "per-timestep empty work is already represented by bit sparsity; a hardware-feasible whole-temporal FFN bypass has negligible low-count population",
        },
        "weight_group_pruning_context": {
            "exact_float_zero_groups": 0,
            "exact_int8_zero_groups": 0,
            "group16_sensitivity": sensitivity,
            "decision": "NO_GO_BLIND_GROUP_PRUNING_AS_PRIMARY_THREE_WEEK_AXIS",
        },
        "next_selected_axes": [
            {
                "axis": "G11 source-by-destination-block bounded contribution gating",
                "priority": 0,
                "hardware_contract": "static per-layer keep metadata skips a complete 96-lane update; beta=0 retains the exact engine; each skipped block contributes to an explicit accumulator error budget",
                "next_measurement": "checkpoint-bound FC1/FC2/Conv weighted source-block DSE, then modified-forward only if full-envelope sensitivity reaches 1.15x",
            },
            {
                "axis": "G12 ATLIF remaining-budget early stop",
                "priority": 1,
                "hardware_contract": "prove no future firing under an exact remaining-positive-input bound before considering a lossy predictor",
                "next_measurement": "capture per-neuron per-timestep state and remaining positive contribution on frozen ep35",
            },
        ],
        "admission": {
            "frozen_trace_eligibility_audit": True,
            "g7_main_axis": False,
            "g8_main_axis": False,
            "accuracy": False,
            "hardware_cycles": False,
            "rtl": False,
            "system_speedup": False,
            "headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    output = args.output_dir.resolve()
    require(not output.exists(), "refusing to overwrite M286 output")
    output.mkdir(parents=True)
    target = output / "m286_nonattention_lossy_axis_eligibility_r1.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    require(sha256(source_path) == source_start, "M286 analyzer self drift")
    print("PASS M286 g7={} g8={} fc1_records={} output={}".format(
        result["g7_official_atlif_amplitude_gate"]["decision"],
        result["g8_whole_temporal_ffn_token_gate"]["decision"],
        aggregate["records"], target))


if __name__ == "__main__":
    main()
