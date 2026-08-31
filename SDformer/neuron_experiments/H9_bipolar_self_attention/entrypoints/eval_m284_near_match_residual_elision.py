#!/usr/bin/env python3
"""Run the production DSEC evaluator with M284 near-match Conv elision."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import csv


REPO = Path(__file__).resolve().parents[3]
EVALUATOR = REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
MODULE = (REPO / "neuron_experiments/H9_bipolar_self_attention/overlay/"
          "models/STSwinNet_SNN/near_match_residual_elision.py")
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
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
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
    parser.add_argument("--distance-threshold", required=True, type=int)
    parser.add_argument("--max-samples", default=0, type=int)
    parser.add_argument("--bn-policy", default="running",
                        choices=("running", "no_running"))
    parser.add_argument("--dump-per-frame", default="")
    parser.add_argument("--baseline-profile", default="", type=Path)
    parser.add_argument("--baseline-per-frame", default="", type=Path)
    args = parser.parse_args()

    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m284_near_match_modified_forward_contract_v1",
            "M284 contract schema drift")
    require(source_start == contract["runtime_identity"]["wrapper_sha256"],
            "M284 wrapper SHA drift")
    inputs = contract["inputs"]
    resolved = {
        "catalog": REPO / inputs["catalog"]["path"],
        "m280_result": REPO / inputs["m280_result"]["path"],
        "m280_seal": REPO / inputs["m280_seal"]["path"],
        "m283_review": REPO / inputs["m283_review"]["path"],
        "config": args.config.resolve(),
        "checkpoint": args.checkpoint.resolve(),
        "module": MODULE,
        "evaluator": EVALUATOR,
        "docs359": DOCS359,
    }
    for name, path in resolved.items():
        require(path.is_file(), "M284 missing input " + str(path))
        expected = (inputs[name]["sha256"] if name in inputs
                    else contract["runtime_identity"][name + "_sha256"])
        require(sha256(path) == expected,
                "M284 input SHA drift: " + name)
    require(sha256(DOCS359) == DOCS359_SHA256,
            "M284 protected docs359 drift")
    threshold = int(args.distance_threshold)
    require(threshold in contract["policy"]["distance_thresholds"],
            "M284 threshold not admitted by contract")
    require(args.bn_policy == "running",
            "M284 promotion requires foldable running BN")
    require(not args.path_results.exists(),
            "M284 refuses to overwrite result directory")
    require(bool(str(args.dump_per_frame)),
            "M284 requires an ordered per-frame CSV receipt")

    evaluator_dir = str(EVALUATOR.parent)
    if evaluator_dir not in sys.path:
        sys.path.insert(0, evaluator_dir)
    saved_argv = list(sys.argv)
    sys.argv = [str(EVALUATOR), "--config", str(args.config)]
    os.environ["SDFORMER_USE_MLFLOW"] = "0"
    try:
        evaluator = load_module(EVALUATOR, "m284_frozen_dsec_evaluator")
    finally:
        sys.argv = saved_argv
    near_match = load_module(MODULE, "m284_near_match_module")
    dataset_length_cap = int(args.max_samples)
    if dataset_length_cap > 0:
        original_dataset = evaluator.DSECDatasetLite

        class M284LengthCappedDSECDataset(original_dataset):
            def __len__(self):
                return min(super().__len__(), dataset_length_cap)

        evaluator.DSECDatasetLite = M284LengthCappedDSECDataset
    installed_model = {"value": None}

    # Install after the audited checkpoint restore.  The module only replaces
    # forward methods, but the explicit ordering removes any ambiguity about
    # which checkpoint weights and operator instances the modified forward uses.
    from models.STSwinNet_SNN import h9_load_audit
    original_checkpoint_loader = h9_load_audit.load_checkpoint_with_h9_audit

    def load_checkpoint_then_install(*loader_args, **loader_kwargs):
        model = original_checkpoint_loader(*loader_args, **loader_kwargs)
        installed = near_match.install_near_match_residual_elision(model, {
            "distance_threshold": threshold,
            "catalog_path": str(resolved["catalog"]),
            "catalog_sha256": inputs["catalog"]["sha256"],
            "source_alpha_delta_u24":
                contract["policy"]["source_alpha_delta_u24"],
            "partition_chunk":
                contract["policy"]["modified_forward_partition_chunk"],
        })
        require(installed == contract["policy"]["operators"],
                "M284 installed operator drift")
        installed_model["value"] = model
        print("[M284] post-checkpoint installed near-match threshold={} modules={}".format(
            threshold, len(installed)), flush=True)
        return model

    h9_load_audit.load_checkpoint_with_h9_audit = load_checkpoint_then_install
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
            "M284 installer did not observe the model")
    runtime = near_match.near_match_residual_elision_summary(
        installed_model["value"])
    profile = args.path_results / "spike_profile.json"
    require(profile.is_file(), "M284 evaluator produced no spike profile")
    profile_value = strict_json(profile)
    samples = int(profile_value.get("samples", -1))
    expected_samples = (int(args.max_samples) if args.max_samples else 825)
    require(samples == expected_samples,
            "M284 evaluated sample population drift")
    dump_path = Path(args.dump_per_frame).resolve()
    require(dump_path.is_file(), "M284 missing ordered per-frame CSV")
    with dump_path.open("r", encoding="utf-8", newline="") as handle:
        dump_rows = list(csv.DictReader(handle))
    require(len(dump_rows) == samples and
            all(row.get("file") and row.get("sequence") for row in dump_rows),
            "M284 per-frame population/identity drift")
    ordered_population = [
        {"file": row["file"], "sequence": row["sequence"]}
        for row in dump_rows
    ]
    baseline = None
    if args.baseline_profile and str(args.baseline_profile) not in ("", "."):
        baseline_path = args.baseline_profile.resolve()
        require(baseline_path.is_file(), "M284 missing baseline profile")
        baseline_value = strict_json(baseline_path)
        require(int(baseline_value.get("samples", -1)) == samples,
                "M284 baseline sample population drift")
        baseline_aee = float(baseline_value["metrics"]["AEE"])
        candidate_aee = float(profile_value["metrics"]["AEE"])
        delta_aee = candidate_aee - baseline_aee
        baseline_dump_path = args.baseline_per_frame.resolve()
        require(baseline_dump_path.is_file(),
                "M284 missing paired baseline per-frame CSV")
        with baseline_dump_path.open("r", encoding="utf-8",
                                     newline="") as handle:
            baseline_dump_rows = list(csv.DictReader(handle))
        baseline_population = [
            {"file": row["file"], "sequence": row["sequence"]}
            for row in baseline_dump_rows
        ]
        require(baseline_population == ordered_population,
                "M284 paired ordered population mismatch")
        baseline = {
            "path": str(baseline_path),
            "sha256": sha256(baseline_path),
            "aee": baseline_aee,
            "candidate_aee": candidate_aee,
            "candidate_minus_baseline_aee": delta_aee,
            "baseline_per_frame_path": str(baseline_dump_path),
            "baseline_per_frame_sha256": sha256(baseline_dump_path),
            "candidate_per_frame_sha256": sha256(dump_path),
            "ordered_population_identical": True,
            "absolute_aee_increase_gate":
                contract["promotion_gates"]["absolute_aee_increase_maximum"],
            "accuracy_gate_pass": delta_aee <= float(
                contract["promotion_gates"]["absolute_aee_increase_maximum"]),
        }
    summary = {
        "schema": "m284_near_match_modified_forward_receipt_v1",
        "status": ("PASS_MODIFIED_FORWARD_ACCURACY_GATE" if
                   baseline and baseline["accuracy_gate_pass"] else
                   "NO_GO_MODIFIED_FORWARD_ACCURACY_GATE" if baseline else
                   "PASS_EXACT_BASELINE_REQUIRES_PAIRED_CANDIDATE"),
        "identity": {
            "contract_path": str(args.contract.resolve()),
            "contract_sha256": sha256(args.contract),
            "wrapper_sha256": source_start,
            "module_sha256": sha256(MODULE),
            "evaluator_sha256": sha256(EVALUATOR),
            "config_sha256": sha256(args.config),
            "checkpoint_sha256": sha256(args.checkpoint),
            "catalog_sha256": sha256(resolved["catalog"]),
            "profile_sha256": sha256(profile),
            "per_frame_sha256": sha256(dump_path),
            "docs359_sha256": sha256(DOCS359),
        },
        "scope": {
            "samples": samples,
            "bn_policy": args.bn_policy,
            "distance_threshold": threshold,
        },
        "runtime": runtime,
        "ordered_population": ordered_population,
        "metrics": profile_value.get("metrics", {}),
        "paired_baseline": baseline,
        "admission": {
            "modified_network_forward": True,
            "s10_screen": samples == 10,
            "valid825": samples == 825,
            "accuracy_promoted": bool(
                samples == 825 and baseline and
                baseline["accuracy_gate_pass"]),
            "system_speedup": False,
            "ppa": False,
            "headline": False,
        },
    }
    receipt = args.path_results / "m284_near_match_receipt_r1.json"
    receipt.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    require(sha256(Path(__file__).resolve()) == source_start,
            "M284 wrapper changed during execution")
    print("M284_DONE status={} threshold={} samples={} snapped={} fraction={:.9f}".format(
        summary["status"], threshold, samples,
        runtime["aggregate_snapped_partition_vectors"],
        runtime["aggregate_snapped_fraction"]), flush=True)


if __name__ == "__main__":
    main()
