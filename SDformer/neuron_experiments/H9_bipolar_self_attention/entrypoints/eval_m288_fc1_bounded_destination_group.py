#!/usr/bin/env python3
"""Run the production DSEC evaluator with M288 FC1 group pruning."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
EVALUATOR = REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
MODULE = (REPO / "neuron_experiments/H9_bipolar_self_attention/overlay/"
          "models/STSwinNet_SNN/bounded_destination_group_pruning.py")
DOCS359 = REPO / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--path-results", required=True, type=Path)
    parser.add_argument("--destination-group-size", required=True, type=int)
    parser.add_argument("--maximum-absolute-int8-weight", required=True, type=int)
    parser.add_argument("--max-samples", default=0, type=int)
    parser.add_argument("--bn-policy", default="running",
                        choices=("running", "no_running"))
    parser.add_argument("--dump-per-frame", default="")
    parser.add_argument("--baseline-profile", default="", type=Path)
    args = parser.parse_args()

    wrapper_path = Path(__file__).resolve()
    source_start = sha256(wrapper_path)
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m288_fc1_bounded_destination_group_modified_forward_contract_v1",
            "M288 contract schema drift")
    require(source_start == contract["runtime_identity"]["wrapper_sha256"],
            "M288 wrapper SHA drift")
    inputs = contract["inputs"]
    resolved = {
        "m287_result": REPO / inputs["m287_result"]["path"],
        "m287_seal": REPO / inputs["m287_seal"]["path"],
        "config": args.config.resolve(),
        "checkpoint": args.checkpoint.resolve(),
        "module": MODULE,
        "evaluator": EVALUATOR,
        "docs359": DOCS359,
    }
    for name, path in resolved.items():
        require(path.is_file(), "M288 missing input " + str(path))
        expected = (inputs[name]["sha256"] if name in inputs
                    else contract["runtime_identity"][name + "_sha256"])
        require(sha256(path) == expected, "M288 input SHA drift: " + name)
    require(sha256(DOCS359) == DOCS359_SHA256,
            "M288 protected docs359 drift")
    group_size = int(args.destination_group_size)
    beta = int(args.maximum_absolute_int8_weight)
    require(group_size in contract["policy"]["destination_group_sizes"] and
            beta in contract["policy"]["maximum_absolute_int8_weights"],
            "M288 point outside contract")
    require(args.bn_policy == "running",
            "M288 promotion requires foldable running BN")
    require(not args.path_results.exists(),
            "M288 refuses to overwrite result directory")

    dse = strict_json(resolved["m287_result"])
    require(dse["status"] ==
            "PASS_CHECKPOINT_TRACE_BOUND_FC1_OPTIMISTIC_COMPACTION_DSE",
            "M288 M287 status drift")
    matches = [row for row in dse["aggregate_dse"][str(group_size)]
               if int(row["maximum_absolute_int8_weight"]) == beta]
    require(len(matches) == 1, "M288 missing unique M287 point")
    dse_point = matches[0]

    evaluator_dir = str(EVALUATOR.parent)
    if evaluator_dir not in sys.path:
        sys.path.insert(0, evaluator_dir)
    saved_argv = list(sys.argv)
    sys.argv = [str(EVALUATOR), "--config", str(args.config)]
    try:
        evaluator = load_module(EVALUATOR, "m288_frozen_dsec_evaluator")
    finally:
        sys.argv = saved_argv
    pruning = load_module(MODULE, "m288_bounded_group_module")
    original_install = evaluator._install_h9_modules
    installed_model = {"value": None}

    def install_with_pruning(model, config):
        original_install(model, config)
        installed = pruning.install_bounded_destination_group_pruning(model, {
            "destination_group_size": group_size,
            "maximum_absolute_int8_weight": beta,
            "allowed_group_sizes":
                contract["policy"]["destination_group_sizes"],
            "allowed_betas":
                contract["policy"]["maximum_absolute_int8_weights"],
        })
        require(installed == contract["policy"]["operators"],
                "M288 installed operator drift")
        installed_model["value"] = model
        print("[M288] installed group={} beta={} modules={}".format(
            group_size, beta, len(installed)), flush=True)

    evaluator._install_h9_modules = install_with_pruning
    eval_args = argparse.Namespace(
        config=str(args.config.resolve()),
        checkpoint=str(args.checkpoint.resolve()),
        path_results=str(args.path_results.resolve()),
        mode="valid",
        max_samples=int(args.max_samples),
        bn_policy=args.bn_policy,
        dump_per_frame=str(args.dump_per_frame),
        dump_selected_frames_dir="",
        dump_frame_list="",
        runid="",
        save_path="results/checkpoint_epoch{}.pth",
    )
    evaluator.valid_test(eval_args, evaluator.YAMLParser(eval_args.config))
    require(installed_model["value"] is not None,
            "M288 installer did not observe the model")
    runtime = pruning.bounded_destination_group_summary(
        installed_model["value"])
    require(abs(runtime["removed_source_group_pair_fraction"] -
                float(dse_point["static_source_group_fraction_removed"]))
            <= 1.0e-15,
            "M288 runtime/M287 static mask fraction mismatch")
    profile = args.path_results / "spike_profile.json"
    require(profile.is_file(), "M288 evaluator produced no spike profile")
    profile_value = strict_json(profile)
    samples = int(profile_value.get("samples", -1))
    expected_samples = int(args.max_samples) if args.max_samples else 825
    require(samples == expected_samples,
            "M288 evaluated sample population drift")
    paired = None
    if str(args.baseline_profile):
        baseline_path = args.baseline_profile.resolve()
        require(baseline_path.is_file(), "M288 missing baseline profile")
        baseline = strict_json(baseline_path)
        require(int(baseline.get("samples", -1)) == samples,
                "M288 paired baseline sample population drift")
        baseline_aee = float(baseline["metrics"]["AEE"])
        candidate_aee = float(profile_value["metrics"]["AEE"])
        delta = candidate_aee - baseline_aee
        paired = {
            "path": str(baseline_path),
            "sha256": sha256(baseline_path),
            "aee": baseline_aee,
            "candidate_aee": candidate_aee,
            "candidate_minus_baseline_aee": delta,
            "absolute_aee_increase_gate":
                contract["promotion_gates"]["absolute_aee_increase_maximum"],
            "accuracy_gate_pass": delta <= float(
                contract["promotion_gates"]["absolute_aee_increase_maximum"]),
        }
    summary = {
        "schema": "m288_fc1_bounded_destination_group_modified_forward_receipt_v1",
        "status": "PASS_MODIFIED_FORWARD_ACCURACY_GATE" if
            paired and paired["accuracy_gate_pass"] else
            "PASS_MODIFIED_FORWARD_REQUIRES_OR_FAILS_PAIRED_GATE",
        "identity": {
            "contract_sha256": sha256(args.contract),
            "wrapper_sha256": source_start,
            "module_sha256": sha256(MODULE),
            "evaluator_sha256": sha256(EVALUATOR),
            "config_sha256": sha256(args.config),
            "checkpoint_sha256": sha256(args.checkpoint),
            "m287_result_sha256": sha256(resolved["m287_result"]),
            "profile_sha256": sha256(profile),
            "docs359_sha256": sha256(DOCS359),
        },
        "scope": {
            "samples": samples,
            "bn_policy": args.bn_policy,
            "destination_group_size": group_size,
            "maximum_absolute_int8_weight": beta,
        },
        "runtime": runtime,
        "m287_optimistic_point": dse_point,
        "metrics": profile_value.get("metrics", {}),
        "paired_baseline": paired,
        "admission": {
            "modified_network_forward": True,
            "s10_screen": samples == 10,
            "valid825": samples == 825,
            "accuracy_gate_pass": bool(paired and
                                       paired["accuracy_gate_pass"]),
            "executable_cycles": False,
            "rtl": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    receipt = args.path_results / "m288_fc1_bounded_group_receipt_r1.json"
    receipt.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    require(sha256(wrapper_path) == source_start,
            "M288 wrapper changed during execution")
    print("M288_PASS group={} beta={} samples={} removed_pairs={:.9f}".format(
        group_size, beta, samples,
        runtime["removed_source_group_pair_fraction"]), flush=True)


if __name__ == "__main__":
    main()
