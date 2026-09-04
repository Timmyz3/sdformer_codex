#!/usr/bin/env python3
"""Run frozen DSEC evaluation with shared FC1+patch-Conv group pruning."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
EVALUATOR = REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
MODULE = (REPO / "neuron_experiments/H9_bipolar_self_attention/overlay/"
          "models/STSwinNet_SNN/shared_fc1_patch_group_pruning.py")
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

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


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
    parser.add_argument("--maximum-absolute-int8-weight", required=True,
                        type=int)
    parser.add_argument("--max-samples", default=0, type=int)
    parser.add_argument("--bn-policy", default="no_running",
                        choices=("running", "no_running"))
    parser.add_argument("--dump-per-frame", default="")
    parser.add_argument("--baseline-profile", default="", type=Path)
    parser.add_argument("--baseline-per-frame", default="", type=Path)
    args = parser.parse_args()

    wrapper_path = Path(__file__).resolve()
    source_start = sha256(wrapper_path)
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m301_shared_fc1_patch_group_modified_forward_contract_v1",
            "M301 contract schema drift")
    require(source_start == contract["runtime_identity"]["wrapper_sha256"],
            "M301 wrapper SHA drift")
    resolved = {
        "m300_result": REPO / contract["inputs"]["m300_result"]["path"],
        "m300_review": REPO / contract["inputs"]["m300_review"]["path"],
        "config": args.config.resolve(),
        "checkpoint": args.checkpoint.resolve(),
        "module": MODULE,
        "evaluator": EVALUATOR,
        "docs359": DOCS359,
    }
    for name, path in resolved.items():
        require(path.is_file(), "M301 missing input: " + str(path))
        expected = (contract["inputs"][name]["sha256"]
                    if name in contract["inputs"] else
                    contract["runtime_identity"][name + "_sha256"])
        require(sha256(path) == expected, "M301 input SHA drift: " + name)
    require(sha256(DOCS359) == DOCS359_SHA256,
            "M301 protected docs359 drift")
    require(args.bn_policy == "no_running",
            "M301 frozen evaluation requires no_running dynamic BN")
    require(not args.path_results.exists(),
            "M301 refuses to overwrite result directory")
    require(bool(str(args.dump_per_frame)),
            "M301 requires an ordered per-frame CSV receipt")

    group_size = int(args.destination_group_size)
    beta = int(args.maximum_absolute_int8_weight)
    require(group_size == int(contract["policy"]["destination_group_size"]) and
            beta in tuple(int(value) for value in
                          contract["policy"]["maximum_absolute_int8_weights"]),
            "M301 point outside frozen paired screen")
    dse = strict_json(resolved["m300_result"])
    require(dse["status"] ==
            "PASS_COMBINED_SENSITIVITY_ELIGIBLE_FOR_PAIRED_S10_ONLY",
            "M301 M300 status drift")
    matches = [row for row in dse["combined_grid"][str(group_size)]
               if int(row["maximum_absolute_int8_weight"]) == beta]
    require(len(matches) == 1, "M301 missing unique M300 point")
    dse_point = matches[0]

    evaluator_dir = str(EVALUATOR.parent)
    if evaluator_dir not in sys.path:
        sys.path.insert(0, evaluator_dir)
    saved_argv = list(sys.argv)
    sys.argv = [str(EVALUATOR), "--config", str(args.config)]
    os.environ["SDFORMER_USE_MLFLOW"] = "0"
    try:
        evaluator = load_module(EVALUATOR, "m301_frozen_dsec_evaluator")
    finally:
        sys.argv = saved_argv
    pruning = load_module(MODULE, "m301_shared_fc1_patch_group_module")
    dataset_length_cap = int(args.max_samples)
    if dataset_length_cap > 0:
        original_dataset = evaluator.DSECDatasetLite

        class M301LengthCappedDSECDataset(original_dataset):
            def __len__(self):
                return min(super().__len__(), dataset_length_cap)

        evaluator.DSECDatasetLite = M301LengthCappedDSECDataset
    installed_model = {"value": None}

    # The frozen evaluator installs H9 modules before restoring the checkpoint.
    # M301 must run after that restore; otherwise load_state_dict silently
    # overwrites the pruned weights.  Wrap the audited checkpoint loader rather
    # than the H9 installer so the mask identity is derived from frozen weights.
    from models.STSwinNet_SNN import h9_load_audit
    original_checkpoint_loader = h9_load_audit.load_checkpoint_with_h9_audit

    def load_checkpoint_then_prune(*loader_args, **loader_kwargs):
        model = original_checkpoint_loader(*loader_args, **loader_kwargs)
        installed = pruning.install_shared_fc1_patch_group_pruning(model, {
            "destination_group_size": group_size,
            "maximum_absolute_int8_weight": beta,
            "allowed_group_sizes": [group_size],
            "allowed_betas":
                contract["policy"]["maximum_absolute_int8_weights"],
        })
        require(installed == contract["policy"]["operators"],
                "M301 installed operator drift")
        installed_model["value"] = model
        print("[M301] post-checkpoint installed group={} beta={} modules={}".format(
            group_size, beta, len(installed)), flush=True)
        return model

    h9_load_audit.load_checkpoint_with_h9_audit = load_checkpoint_then_prune
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
        path_mlflow="",
        save_path="results/checkpoint_epoch{}.pth",
    )
    evaluator.valid_test(eval_args, evaluator.YAMLParser(eval_args.config))
    require(installed_model["value"] is not None,
            "M301 installer did not observe the model")
    runtime = pruning.shared_fc1_patch_group_summary(
        installed_model["value"])
    require(abs(runtime["summaries"]["fc1"]
                        ["removed_source_group_pair_fraction"] -
                float(dse_point[
                    "fc1_static_source_group_pair_fraction_removed"])) <=
            1.0e-15,
            "M301 runtime/M300 FC1 mask fraction mismatch")
    require(abs(runtime["summaries"]["patch_conv"]
                        ["removed_source_group_pair_fraction"] -
                float(dse_point[
                    "patch_conv_static_source_group_pair_fraction_removed"])) <=
            1.0e-15,
            "M301 runtime/M300 patch mask fraction mismatch")
    profile = args.path_results / "spike_profile.json"
    require(profile.is_file(), "M301 evaluator produced no spike profile")
    profile_value = strict_json(profile)
    samples = int(profile_value.get("samples", -1))
    expected_samples = int(args.max_samples) if args.max_samples else 825
    require(samples == expected_samples,
            "M301 evaluated sample population drift")
    dump_path = Path(args.dump_per_frame).resolve()
    require(dump_path.is_file(), "M301 missing ordered per-frame CSV")
    with dump_path.open("r", encoding="utf-8", newline="") as handle:
        dump_rows = list(csv.DictReader(handle))
    require(len(dump_rows) == samples and
            all(row.get("file") and row.get("sequence") for row in dump_rows),
            "M301 per-frame population/identity drift")
    ordered_population = [
        {"file": row["file"], "sequence": row["sequence"]}
        for row in dump_rows
    ]

    paired = None
    if args.baseline_profile and str(args.baseline_profile) not in ("", "."):
        baseline_path = args.baseline_profile.resolve()
        require(baseline_path.is_file(), "M301 missing baseline profile")
        baseline = strict_json(baseline_path)
        require(int(baseline.get("samples", -1)) == samples,
                "M301 paired baseline sample population drift")
        baseline_aee = float(baseline["metrics"]["AEE"])
        candidate_aee = float(profile_value["metrics"]["AEE"])
        delta = candidate_aee - baseline_aee
        baseline_dump_path = args.baseline_per_frame.resolve()
        require(baseline_dump_path.is_file(),
                "M301 missing paired baseline per-frame CSV")
        with baseline_dump_path.open("r", encoding="utf-8",
                                     newline="") as handle:
            baseline_dump_rows = list(csv.DictReader(handle))
        baseline_order = [
            {"file": row["file"], "sequence": row["sequence"]}
            for row in baseline_dump_rows
        ]
        require(baseline_order == ordered_population,
                "M301 paired sample/order mismatch")
        paired = {
            "path": str(baseline_path),
            "sha256": sha256(baseline_path),
            "aee": baseline_aee,
            "candidate_aee": candidate_aee,
            "candidate_minus_baseline_aee": delta,
            "baseline_per_frame_path": str(baseline_dump_path),
            "baseline_per_frame_sha256": sha256(baseline_dump_path),
            "candidate_per_frame_sha256": sha256(dump_path),
            "ordered_population_identical": True,
            "absolute_aee_increase_gate":
                contract["promotion_gates"]
                        ["absolute_aee_increase_maximum"],
            "accuracy_gate_pass": delta <= float(
                contract["promotion_gates"]
                        ["absolute_aee_increase_maximum"]),
        }
    summary = {
        "schema": "m301_shared_fc1_patch_group_modified_forward_receipt_v1",
        "status": ("PASS_MODIFIED_FORWARD_ACCURACY_GATE" if
                   paired and paired["accuracy_gate_pass"] else
                   "PASS_MODIFIED_FORWARD_REQUIRES_OR_FAILS_PAIRED_GATE"),
        "identity": {
            "contract_sha256": sha256(args.contract),
            "wrapper_sha256": source_start,
            "module_sha256": sha256(MODULE),
            "evaluator_sha256": sha256(EVALUATOR),
            "config_sha256": sha256(args.config),
            "checkpoint_sha256": sha256(args.checkpoint),
            "m300_result_sha256": sha256(resolved["m300_result"]),
            "m300_review_sha256": sha256(resolved["m300_review"]),
            "profile_sha256": sha256(profile),
            "per_frame_sha256": sha256(dump_path),
            "docs359_sha256": sha256(DOCS359),
        },
        "scope": {
            "samples": samples,
            "bn_policy": args.bn_policy,
            "destination_group_size": group_size,
            "maximum_absolute_int8_weight": beta,
            "operators": len(runtime["operator_names"]),
            "dataset_length_cap": dataset_length_cap,
        },
        "runtime": runtime,
        "m300_optimistic_point": dse_point,
        "metrics": profile_value.get("metrics", {}),
        "ordered_population": ordered_population,
        "paired_baseline": paired,
        "admission": {
            "modified_network_forward": True,
            "s10_screen": samples == 10,
            "valid825": samples == 825,
            "accuracy_gate_pass": bool(
                paired and paired["accuracy_gate_pass"]),
            "executable_cycles": False,
            "rtl": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    receipt = args.path_results / \
        "m301_shared_fc1_patch_group_receipt_r1.json"
    receipt.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    require(sha256(wrapper_path) == source_start,
            "M301 wrapper changed during execution")
    print("M301_PASS group={} beta={} samples={} delta_aee={}".format(
        group_size, beta, samples,
        None if paired is None else
        paired["candidate_minus_baseline_aee"]), flush=True)


if __name__ == "__main__":
    main()
