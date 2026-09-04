#!/usr/bin/env python3
"""Exact paired PAFT/control valid825 comparison with hardware-safe BN scope."""

from __future__ import division

import argparse
import copy
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_training_config(value):
    result = copy.deepcopy(value)
    result.pop("experiment", None)
    result.pop("note", None)
    result.pop("pattern_paft", None)
    require("runtime" in result and "paired_arm" in result["runtime"],
            "missing runtime paired_arm")
    result["runtime"].pop("paired_arm")
    return result


def metric_mean(rows, metric):
    return statistics.mean(float(row[metric]) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = load_json(args.contract)
    require(contract.get("schema") ==
            "m247_paft_vs_control_paired_valid825_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    identities = {}
    resolved = {}
    loaded = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: {}".format(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "SHA drift for {}: {}".format(name, observed))
        identities[name] = {"path": spec["path"], "sha256": observed}
        resolved[name] = path
        if path.suffix == ".json":
            loaded[name] = load_json(path)

    with resolved["paft_config"].open("r", encoding="utf-8") as handle:
        paft_config = yaml.safe_load(handle)
    with resolved["control_config"].open("r", encoding="utf-8") as handle:
        control_config = yaml.safe_load(handle)
    require(normalized_training_config(paft_config) ==
            normalized_training_config(control_config),
            "paired training config drift outside declared PAFT fields")
    paft_block = paft_config["pattern_paft"]
    control_block = control_config["pattern_paft"]
    expected_ops = contract["paired_protocol"]["paft_expected_operators"]
    require(paft_block["enabled"] is True and
            paft_block["expected_operator_names"] == expected_ops,
            "PAFT target set drift")
    require(control_block["enabled"] is False,
            "control PAFT must be disabled")
    require(control_block["paired_catalog_sha256"] ==
            paft_block["catalog_sha256"],
            "paired catalog identity drift")

    summary = loaded["paft_summary"]
    require(summary["status"] ==
            "PASS_PAIRED_VALID825__PAFT_NOT_HARDWARE_ACCURACY_PROMOTED",
            "M162 status drift")
    completion = resolved["control_completion"].read_text(encoding="utf-8")
    require("PASS_M239R2_CONTROL_EP4_BN_POLICY_AB_VALID825" in completion and
            "no_running_valid825=true" in completion and
            "running_valid825=true" in completion,
            "control completion receipt drift")

    protocol = contract["paired_protocol"]
    comparisons = {}
    for policy in ("no_running", "running"):
        paft_profile = loaded["paft_{}_profile".format(policy)]
        control_profile = loaded["control_{}_profile".format(policy)]
        paft_rows = read_csv(resolved["paft_{}_frames".format(policy)])
        control_rows = read_csv(resolved["control_{}_frames".format(policy)])

        require(len(paft_rows) == len(control_rows) ==
                protocol["expected_frames"],
                "frame count drift for {}".format(policy))
        identity_fields = ("file", "sequence", "valid_pixels")
        require([[row[field] for field in identity_fields] for row in paft_rows] ==
                [[row[field] for field in identity_fields] for row in control_rows],
                "paired cohort/order/valid-pixel drift for {}".format(policy))
        require(len(set(row["sequence"] for row in paft_rows)) ==
                protocol["expected_sequences"],
                "sequence count drift for {}".format(policy))

        for profile in (paft_profile, control_profile):
            require(profile["eval_protocol"]["bn_policy"] == policy and
                    profile["eval_protocol"]["eval_batch_size"] ==
                    protocol["expected_eval_batch_size"] and
                    profile["checkpoint_load_audit"]["missing_count"] == 0 and
                    profile["checkpoint_load_audit"]["unexpected_count"] == 0 and
                    profile["module_counts"]["ATLIFTernaryPSN"] ==
                    protocol["expected_atlif_modules"] and
                    profile["module_counts"]["ShiftmaxAttention"] ==
                    protocol["expected_attention_modules"],
                    "evaluation protocol/load drift for {}".format(policy))

        metric_results = {}
        for metric in protocol["metrics"]:
            paft_mean = metric_mean(paft_rows, metric)
            control_mean = metric_mean(control_rows, metric)
            paft_audit_mean = float(
                paft_profile["metric_aggregation_audit"]["frame_equal_mean"][metric])
            control_audit_mean = float(
                control_profile["metric_aggregation_audit"]["frame_equal_mean"][metric])
            require(abs(paft_mean - paft_audit_mean) < 1e-8 and
                    abs(control_mean - control_audit_mean) < 1e-8,
                    "profile/per-frame metric mismatch for {} {}".format(policy,
                                                                           metric))
            deltas = [float(a[metric]) - float(b[metric])
                      for a, b in zip(paft_rows, control_rows)]
            by_sequence = defaultdict(list)
            for row, delta in zip(paft_rows, deltas):
                by_sequence[row["sequence"]].append(delta)
            sequence_means = {key: statistics.mean(values)
                              for key, values in sorted(by_sequence.items())}
            metric_results[metric] = {
                "paft_frame_equal_mean": paft_mean,
                "control_frame_equal_mean": control_mean,
                "paft_top_level_profile_minus_per_frame":
                    float(paft_profile["metrics"][metric]) - paft_mean,
                "control_top_level_profile_minus_per_frame":
                    float(control_profile["metrics"][metric]) - control_mean,
                "paft_minus_control": statistics.mean(deltas),
                "relative_change_percent": 100.0 * (paft_mean / control_mean - 1.0),
                "lower_is_better_frame_wins": sum(value < 0 for value in deltas),
                "frame_losses": sum(value > 0 for value in deltas),
                "frame_ties": sum(value == 0 for value in deltas),
                "sequence_balanced_delta": statistics.mean(sequence_means.values()),
                "lower_is_better_sequence_wins":
                    sum(value < 0 for value in sequence_means.values()),
                "sequence_count": len(sequence_means),
                "per_sequence_delta": sequence_means
            }

        paft_spikes = int(paft_profile["total_spikes"])
        control_spikes = int(control_profile["total_spikes"])
        comparisons[policy] = {
            "metrics": metric_results,
            "paft_total_spikes": paft_spikes,
            "control_total_spikes": control_spikes,
            "paft_spike_change_percent":
                100.0 * (paft_spikes / control_spikes - 1.0),
            "paft_global_firing_rate": float(paft_profile["global_firing_rate"]),
            "control_global_firing_rate":
                float(control_profile["global_firing_rate"])
        }

    running_aee = comparisons["running"]["metrics"]["AEE"]
    require(running_aee["paft_minus_control"] < 0,
            "PAFT running-BN AEE direction is not positive")
    require(summary["running"]["hardware_static_bn_fold_eligible"] is True and
            summary["no_running"]["hardware_static_bn_fold_eligible"] is False,
            "M162 BN hardware eligibility drift")

    result = {
        "schema": "m247_paft_vs_control_paired_valid825_v1",
        "status": "PASS_SINGLE_SEED_SMALL_POSITIVE_RUNNING_BN_DIRECTION",
        "identity": identities,
        "paired_training_audit": {
            "non_paft_config_fields_exactly_equal": True,
            "paft_enabled": True,
            "control_paft_enabled": False,
            "paft_target_operator_count": len(expected_ops),
            "paft_target_operators": expected_ops,
            "paft_checkpoint_sha256":
                loaded["paft_running_profile"]["artifact_identity"]["checkpoint_sha256"],
            "control_checkpoint_sha256":
                loaded["control_running_profile"]["artifact_identity"]["checkpoint_sha256"]
        },
        "paired_validation": comparisons,
        "hardware_decision": {
            "primary_policy": "running",
            "running_bn_static_fold_eligible": True,
            "paft_running_aee_improvement_percent":
                -running_aee["relative_change_percent"],
            "paft_running_total_spike_reduction_percent":
                -comparisons["running"]["paft_spike_change_percent"],
            "direction": "KEEP_PAFT_FOR_REAL_TRACE_REPLAY",
            "strength": "SMALL_SINGLE_SEED_GAIN_NOT_A_HEADLINE",
            "algorithm_feedback": "increase hardware-weighted PAFT effect only if a repeated-seed or stronger regularization run preserves running-BN accuracy; do not trade accuracy for synthetic sparsity",
            "next_gate": "INT8-export the PAFT four-bottleneck weights and replay the exact M248 source trace through the M152/M157/M158/M241 cycle model"
        },
        "limitations": {
            "same_evaluator_runtime_sha_bound_for_both_arms": False,
            "reason": "M162 binds evaluator SHA, but the M239 control completion receipt did not record a run-time evaluator SHA",
            "multi_seed_or_confidence_interval": False,
            "no_running_policy_is_hardware_accuracy": False,
            "conv_cycle_speedup_from_this_result": False,
            "system_speedup": False
        },
        "admission": contract["claim_boundary"],
        "claim_boundary": "Exact-SHA paired single-seed valid825 output comparison on an identical ordered cohort. Running BN is the hardware-foldable primary policy. The small PAFT advantage is a direction for trace replay, not a statistically replicated accuracy headline, Conv cycle speedup, system speedup or paper PPA."
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m247_paft_vs_control_paired_valid825_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M247_PASS running_AEE_improvement={:.6f}% spike_reduction={:.6f}%".format(
        result["hardware_decision"]["paft_running_aee_improvement_percent"],
        result["hardware_decision"]["paft_running_total_spike_reduction_percent"]))


if __name__ == "__main__":
    main()
