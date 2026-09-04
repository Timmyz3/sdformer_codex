#!/usr/bin/env python3
"""Capture the no-PAFT ep4 control bottleneck inputs under running BN."""

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M248_TRACER = HW / (
    "system_simulator/scripts/trace_m248_paft_ep4_running_bn_bottleneck_sources.py")
BASE_TRACER = HW / "system_simulator/scripts/trace_m40_bottleneck_packed_sources.py"
PROFILE_SCRIPT = (ROOT /
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "profile_nts11_hardware_p0.py")
EXPECTED = {
    "m248_tracer": "d5c18ec3dd358b0ef10e66f7682bd87442be1cf5ddc9de53ea761a42a5451bf6",
    "base_tracer": "b02ac10fb95e68fa2871b74330d6f39d7d3d8cbfa6440990d43ec832e943bf19",
    "profile": "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    "checkpoint": "6964264bde719f36c8ef7537d50ab90a7237973109f610c662d95d071bb0ab4b",
    "config": "12e36e7b88699ebc7ccd138c9e7c16991d6abe029773fb11b037a3463116f739",
    "sample_workload": "bb45f8b5406e34835f05e1993692d8cba241c748471037d75fcfa1ec2478cffa",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    inputs = {
        "m248_tracer": M248_TRACER,
        "base_tracer": BASE_TRACER,
        "profile": PROFILE_SCRIPT,
        "checkpoint": args.checkpoint.resolve(),
        "config": args.config.resolve(),
        "sample_workload": args.sample_workload.resolve(),
        "docs359": HW / "docs/359_DATE终局冻结_20260813.md"
    }
    observed = {name: sha256(path) for name, path in inputs.items()}
    require(observed == EXPECTED,
            "M252 strict input SHA drift: {}".format(observed))

    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing to overwrite M252 output")
    output_dir.mkdir(parents=True)
    m248 = load_module("m252_frozen_m248", M248_TRACER)
    base = m248.load_base()
    sample_keys = base.read_frozen_sample_keys(inputs["sample_workload"])
    profile = base.load_profile_module()
    config, device = profile.load_config(inputs["config"])
    require(device.type == "cuda" and torch.cuda.is_available(),
            "M252 requires an available CUDA device")

    source_bn_policy = config.get("test", {}).get("bn_policy", "running")
    config.setdefault("test", {})["bn_policy"] = "running"
    dataset = profile.DSECDatasetLite(
        config, file_list="valid", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1))
    observed_keys = tuple(
        "|".join(str(item) for item in row)
        if isinstance(row, (list, tuple)) else str(row)
        for row in dataset.files[:len(sample_keys)])
    require(observed_keys == sample_keys,
            "dataset first-ten identity/order drift")
    dataset_receipts = base.dataset_file_receipts(
        config["data"]["path"], sample_keys)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        pin_memory=False, num_workers=0)
    transform = None
    if config["loader"].get("crop") is not None:
        transform = profile.Compose([
            profile.CenterCrop(tuple(config["loader"]["crop"]))])

    model = profile.build_model(config, inputs["checkpoint"], device)
    load_audit = profile.validate_h9_load_audit(model, config)
    require(load_audit.get("missing_count") == 0 and
            load_audit.get("unexpected_count") == 0,
            "control checkpoint load mismatch")
    require(not hasattr(model, "_m71_pattern_paft_state"),
            "PAFT observation hook unexpectedly present in control capture")
    bn_changed = profile.configure_batch_norm_evaluation(model, "running")

    writer = base.PackedBottleneckWriter(output_dir, sample_keys)
    writer.attach(model)
    processed = 0
    try:
        with torch.no_grad():
            for chunk, mask, label in itertools.islice(loader, len(sample_keys)):
                profile.functional.reset_net(model)
                writer.begin_sample(processed, observed_keys[processed])
                x, transformed_label, transformed_mask = profile.preprocess_chunk(
                    config, chunk, label, mask, transform, device)
                del transformed_label, transformed_mask
                model(x)
                processed += 1
                print("[M252 control running-BN trace] {}/{}".format(
                    processed, len(sample_keys)), flush=True)
    except BaseException:
        writer.detach()
        raise
    writer.finalize()
    require(processed == len(sample_keys), "M252 cohort incomplete")
    rows = sorted(writer.rows,
                  key=lambda row: (row["sample_id"], row["operator_index"]))

    manifest = {
        "schema": "m252_control_ep4_running_bn_bottleneck_source_trace_v1",
        "status": "PASS_CONTROL_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE",
        "identity": {
            "checkpoint_path": str(inputs["checkpoint"]),
            "checkpoint_sha256": observed["checkpoint"],
            "config_path": str(inputs["config"]),
            "config_sha256": observed["config"],
            "sample_workload_path": str(inputs["sample_workload"]),
            "sample_workload_sha256": observed["sample_workload"],
            "m252_tracer_sha256": script_start,
            "frozen_m248_tracer_sha256": observed["m248_tracer"],
            "frozen_m40_base_tracer_sha256": observed["base_tracer"],
            "parent_profile_script_sha256": observed["profile"],
            "checkpoint_load_audit": load_audit,
            "source_config_bn_policy": source_bn_policy,
            "capture_bn_policy": "running",
            "bn_policy_override_is_evaluation_only": True,
            "bn_modules_changed": bn_changed,
            "training_only_pattern_paft_hook_installed_during_capture": False,
            "paired_arm": "NO_PAFT_CONTROL",
            "cuda_device_name": torch.cuda.get_device_name(device),
            "dataset_root": str(Path(config["data"]["path"]).resolve()),
            "dataset_input_files": dataset_receipts,
            "docs359_sha256_unchanged": observed["docs359"]
        },
        "cohort": {
            "samples": len(sample_keys),
            "sample_keys": list(sample_keys),
            "operators": list(base.TARGETS),
            "records": len(rows),
            "shape": list(base.EXPECTED_SHAPE)
        },
        "records": rows,
        "coverage": {
            "source_support_and_sign_bitmap_exact": True,
            "source_float_values_reconstructable": True,
            "source_weight_content_hash_present": True,
            "source_and_destination_product_events_materialized": False,
            "weight_quantization_present": False,
            "cycle_schedule_present": False,
            "paft_vs_control_comparison_present": False
        },
        "admission": {
            "control_checkpoint_running_bn_source_trace": True,
            "conv_cycle_speedup": False,
            "paft_hardware_gain": False,
            "power_energy": False,
            "system_speedup": False,
            "headline": False
        },
        "claim_boundary": (
            "Exact no-PAFT control ep4 running-BN four-bottleneck input support, "
            "sign and value trace on the frozen ten-sample cohort. Cycle replay, "
            "PAFT hardware gain, power, system speedup and headline claims remain "
            "unadmitted.")
    }
    manifest_path = output_dir / (
        "m252_control_ep4_running_bn_bottleneck_source_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    require(sha256(script_path) == script_start,
            "M252 tracer changed during run")
    print(manifest_path)


if __name__ == "__main__":
    main()
