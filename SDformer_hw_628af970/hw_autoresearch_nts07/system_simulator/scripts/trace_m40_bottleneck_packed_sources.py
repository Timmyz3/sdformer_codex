#!/usr/bin/env python3
"""Capture exact support/sign bitmaps for the four M39 Conv3x3 calls.

This is a new M40 trace identity.  It intentionally does not reuse the M36
profile-script identity: M36 captured two patch-embed inputs and did not retain
the four bottleneck tensors.  The output is a compact, deterministic pair of
positive/negative bitmaps and a content hash/value audit for every target call.
It is not a product-event or cycle trace; the manifest says so explicitly.
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import zlib

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
PROFILE_SCRIPT = (
    ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "profile_nts11_hardware_p0.py"
)
TARGETS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
EXPECTED_SHAPE = (10, 1, 768, 15, 20)
EXPECTED_OUTPUT_SHAPE = (10, 1, 768, 15, 20)
EXPECTED_CHECKPOINT_SHA256 = (
    "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158")
EXPECTED_CONFIG_SHA256 = (
    "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49")
EXPECTED_SAMPLE_WORKLOAD_SHA256 = (
    "bb45f8b5406e34835f05e1993692d8cba241c748471037d75fcfa1ec2478cffa")
EXPECTED_SAMPLE_KEYS = (
    "zurich_city_09_a_0001.npy",
    "zurich_city_09_a_0011.npy",
    "zurich_city_09_a_0021.npy",
    "zurich_city_09_a_0031.npy",
    "zurich_city_09_a_0041.npy",
    "zurich_city_09_a_0051.npy",
    "zurich_city_09_a_0061.npy",
    "zurich_city_09_a_0071.npy",
    "zurich_city_09_a_0081.npy",
    "zurich_city_09_a_0091.npy",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_profile_module():
    profile_dir = str(PROFILE_SCRIPT.parent)
    if profile_dir not in sys.path:
        sys.path.insert(0, profile_dir)
    spec = importlib.util.spec_from_file_location("m40_profile_entry", str(PROFILE_SCRIPT))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M40 parent profile entry")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packed_bits(values):
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    return np.packbits(flat, bitorder="little").tobytes()


class PackedBottleneckWriter:
    def __init__(self, output_dir, sample_keys):
        self.output_dir = Path(output_dir)
        self.sample_keys = tuple(sample_keys)
        self.current_sample = None
        self.current_key = None
        self.rows = []
        self.handles = []
        self.call_counts = {name: 0 for name in TARGETS}
        self.module_geometry = {}

    def begin_sample(self, sample_id, sample_key):
        require(type(sample_id) is int and 0 <= sample_id < len(self.sample_keys),
                "sample id outside frozen cohort")
        require(sample_key == self.sample_keys[sample_id],
                "sample identity/order drift")
        self.current_sample = sample_id
        self.current_key = sample_key

    def _hook(self, name):
        def capture(_module, inputs, output):
            require(self.current_sample is not None, "target fired outside sample")
            require(len(inputs) >= 1 and torch.is_tensor(inputs[0]),
                    "target input is not a tensor")
            value = inputs[0].detach()
            shape = tuple(int(item) for item in value.shape)
            require(shape == EXPECTED_SHAPE,
                    "{} input shape {} != {}".format(name, shape, EXPECTED_SHAPE))
            require(torch.is_tensor(output), "target output is not a tensor")
            output_shape = tuple(int(item) for item in output.shape)
            require(output_shape == EXPECTED_OUTPUT_SHAPE,
                    "{} output shape {} != {}".format(
                        name, output_shape, EXPECTED_OUTPUT_SHAPE))
            require(bool(torch.isfinite(value).all().item()),
                    "target input contains NaN/Infinity")
            host_value = value.to(device="cpu").contiguous().numpy()
            host = np.sign(host_value).astype(np.int8, copy=False)
            positive = packed_bits(host_value > 0)
            negative = packed_bits(host_value < 0)
            per_timestep = host_value.reshape(EXPECTED_SHAPE[0], -1)
            previous_value = np.zeros_like(per_timestep)
            previous_value[1:] = per_timestep[:-1]
            numeric_changed = per_timestep != previous_value
            changed = packed_bits(numeric_changed)
            require(len(positive) == len(negative) == len(changed) ==
                    math.ceil(host.size / 8),
                    "packed bitmap extent mismatch")
            call_index = self.call_counts[name]
            require(call_index == self.current_sample,
                    "target must fire exactly once per sample in cohort order")
            self.call_counts[name] += 1
            target_index = TARGETS.index(name)
            filename = "s{:02d}_o{}_ternary_sign2_le.bin".format(
                self.current_sample, target_index)
            path = self.output_dir / filename
            require(not path.exists(), "refusing to overwrite packed capture")
            path.write_bytes(positive + negative + changed)
            support = per_timestep != 0
            local_counts = np.count_nonzero(support, axis=1)
            previous_support = np.zeros_like(support)
            previous_support[1:] = support[:-1]
            support_delta = support.astype(np.int8) - previous_support.astype(np.int8)
            sign_state = np.sign(per_timestep).astype(np.int8)
            previous_sign = np.zeros_like(sign_state)
            previous_sign[1:] = sign_state[:-1]
            sign_delta = sign_state - previous_sign
            motion_counts = np.count_nonzero(numeric_changed, axis=1)
            direction_counts = {
                str(delta): [int(np.count_nonzero(sign_delta[timestep] == delta))
                             for timestep in range(EXPECTED_SHAPE[0])]
                for delta in (-2, -1, 0, 1, 2)
            }
            rounded = np.rint(host_value)
            integer_mask = host_value == rounded
            ternary_mask = integer_mask & (np.abs(rounded) <= 1)
            require(sys.byteorder == "little", "M40 float payload requires little endian")
            raw_bytes = host_value.astype("<f4", copy=False).tobytes(order="C")
            require(str(host_value.dtype) == "float32",
                    "M40 exact value payload currently requires float32")
            bit_patterns, pattern_counts = np.unique(
                host_value.view(np.uint32), return_counts=True)
            probabilities = pattern_counts.astype(np.float64) / float(host_value.size)
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))
            value_filename = "s{:02d}_o{}_values_f32le.zlib".format(
                self.current_sample, target_index)
            value_path = self.output_dir / value_filename
            require(not value_path.exists(), "refusing to overwrite exact value payload")
            value_path.write_bytes(zlib.compress(raw_bytes, 9))
            self.rows.append({
                "sample_id": self.current_sample,
                "sample_key": self.current_key,
                "operator": name,
                "operator_index": target_index,
                "shape": list(EXPECTED_SHAPE),
                "output_shape": list(output_shape),
                "module_geometry": self.module_geometry[name],
                "elements": int(host.size),
                "input_dtype": str(host_value.dtype),
                "input_content_bytes": len(raw_bytes),
                "input_content_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "value_payload_file": value_filename,
                "value_payload_codec": "ZLIB_LEVEL9_RAW_C_ORDER_FLOAT32_NATIVE_LE",
                "value_payload_compressed_bytes": value_path.stat().st_size,
                "value_payload_sha256": sha256(value_path),
                "value_bit_pattern_population": {
                    "unique_float32_bit_patterns": int(bit_patterns.size),
                    "shannon_entropy_bits_per_value": entropy,
                    "full_codebook_in_manifest": bool(bit_patterns.size <= 256),
                    "codebook": ([{
                        "float32_bits_hex": "{:08x}".format(int(pattern)),
                        "count": int(count),
                    } for pattern, count in zip(bit_patterns, pattern_counts)]
                        if bit_patterns.size <= 256 else None),
                },
                "positive_count": int(np.count_nonzero(host_value > 0)),
                "negative_count": int(np.count_nonzero(host_value < 0)),
                "nonzero_count": int(np.count_nonzero(host_value)),
                "value_audit": {
                    "integer_valued_count": int(np.count_nonzero(integer_mask)),
                    "noninteger_count": int(host_value.size - np.count_nonzero(integer_mask)),
                    "ternary_valued_count": int(np.count_nonzero(ternary_mask)),
                    "all_values_integer": bool(np.all(integer_mask)),
                    "all_values_ternary": bool(np.all(ternary_mask)),
                    "minimum": float(np.min(host_value)),
                    "maximum": float(np.max(host_value)),
                },
                "local_nonzero_count_by_timestep": [int(x) for x in local_counts],
                "motion_numeric_transition_count_by_timestep": [int(x) for x in motion_counts],
                "motion_sign_delta_population_by_timestep": direction_counts,
                "motion_support_delta_population_by_timestep": {
                    str(delta): [int(np.count_nonzero(
                        support_delta[timestep] == delta))
                        for timestep in range(EXPECTED_SHAPE[0])]
                    for delta in (-1, 0, 1)
                },
                "motion_t0_previous_state": "RESET_ZERO_PER_SAMPLE",
                "packed_file": filename,
                "packed_file_bytes": path.stat().st_size,
                "positive_plane_bytes": len(positive),
                "negative_plane_offset_bytes": len(positive),
                "numeric_change_plane_offset_bytes": len(positive) + len(negative),
                "numeric_change_plane_bytes": len(changed),
                "packing": (
                    "C_ORDER_FLAT_NP_PACKBITS_LITTLE_POSITIVE_THEN_NEGATIVE_"
                    "THEN_EXACT_FLOAT_VALUE_CHANGED_VS_PREVIOUS_T_WITH_T0_ZERO"),
                "packed_file_sha256": sha256(path),
            })
        return capture

    def attach(self, model):
        modules = dict(model.named_modules())
        require(all(name in modules for name in TARGETS),
                "one or more M40 target modules are absent")
        for name in TARGETS:
            module = modules[name]
            require(isinstance(module, torch.nn.Conv2d),
                    "M40 target is not Conv2d")
            require(tuple(int(x) for x in module.kernel_size) == (3, 3),
                    "M40 target kernel drift")
            require(int(module.in_channels) == 768 and int(module.out_channels) == 768,
                    "M40 target channel geometry drift")
            weight = module.weight.detach().to(device="cpu").contiguous()
            weight_bytes = weight.numpy().tobytes(order="C")
            bias = None if module.bias is None else module.bias.detach().to(
                device="cpu").contiguous()
            self.module_geometry[name] = {
                "kernel_size": [int(x) for x in module.kernel_size],
                "stride": [int(x) for x in module.stride],
                "padding": [int(x) for x in module.padding],
                "dilation": [int(x) for x in module.dilation],
                "groups": int(module.groups),
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
                "weight_dtype": str(weight.dtype),
                "weight_shape": [int(x) for x in weight.shape],
                "weight_content_sha256": hashlib.sha256(weight_bytes).hexdigest(),
                "weight_content_bytes": len(weight_bytes),
                "bias_present": bias is not None,
                "bias_dtype": None if bias is None else str(bias.dtype),
                "bias_shape": None if bias is None else [int(x) for x in bias.shape],
                "bias_content_sha256": None if bias is None else hashlib.sha256(
                    bias.numpy().tobytes(order="C")).hexdigest(),
            }
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def finalize(self):
        self.detach()
        require(all(value == len(self.sample_keys)
                    for value in self.call_counts.values()),
                "target population incomplete")
        require(len(self.rows) == len(TARGETS) * len(self.sample_keys),
                "capture row population incomplete")


def read_frozen_sample_keys(path):
    require(sha256(path) == EXPECTED_SAMPLE_WORKLOAD_SHA256,
            "frozen sample_workload identity drift")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = tuple(row["sample_key"] for row in rows)
    require(keys == EXPECTED_SAMPLE_KEYS, "frozen M36 cohort identity drift")
    return keys


def dataset_file_receipts(data_root, sample_keys):
    receipts = []
    for sample_key in sample_keys:
        sequence = "_".join(Path(sample_key).stem.split("_")[:-1])
        for role, relative in (
                ("event", Path("event_tensors/10bins/left") / sequence / sample_key),
                ("ground_truth", Path("gt_tensors") / sample_key),
                ("mask", Path("mask_tensors") / sample_key)):
            path = Path(data_root) / relative
            require(path.is_file(), "M40 dataset input missing: {}".format(path))
            receipts.append({
                "role": role,
                "sample_key": sample_key,
                "relative_path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing to overwrite M40 packed trace directory")
    output_dir.mkdir(parents=True)
    sample_keys = read_frozen_sample_keys(args.sample_workload.resolve())
    require(sha256(args.checkpoint.resolve()) == EXPECTED_CHECKPOINT_SHA256,
            "M40 checkpoint identity drift")
    require(sha256(args.config.resolve()) == EXPECTED_CONFIG_SHA256,
            "M40 config identity drift")
    profile = load_profile_module()
    config, device = profile.load_config(args.config.resolve())
    require(device.type == "cuda" and torch.cuda.is_available(),
            "M40 real trace requires an available CUDA device")
    dataset = profile.DSECDatasetLite(
        config, file_list="valid", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1))
    require(len(dataset.files) >= len(sample_keys), "validation dataset too short")
    observed_keys = tuple(
        "|".join(str(item) for item in row)
        if isinstance(row, (list, tuple)) else str(row)
        for row in dataset.files[:len(sample_keys)]
    )
    require(observed_keys == sample_keys, "dataset first-ten identity/order drift")
    dataset_receipts = dataset_file_receipts(config["data"]["path"], sample_keys)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        pin_memory=False, num_workers=0)
    transform = None
    if config["loader"].get("crop") is not None:
        transform = profile.Compose([profile.CenterCrop(tuple(config["loader"]["crop"]))])
    model = profile.build_model(config, args.checkpoint.resolve(), device)
    load_audit = profile.validate_h9_load_audit(model, config)
    bn_policy = config.get("test", {}).get("bn_policy", "running")
    bn_changed = profile.configure_batch_norm_evaluation(model, bn_policy)
    writer = PackedBottleneckWriter(output_dir, sample_keys)
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
                print("[M40 packed trace] {}/{}".format(
                    processed, len(sample_keys)), flush=True)
    except BaseException:
        writer.detach()
        raise
    writer.finalize()
    require(processed == len(sample_keys), "M40 cohort did not complete")

    rows = sorted(writer.rows, key=lambda row: (row["sample_id"], row["operator_index"]))
    manifest = {
        "schema": "m40_bottleneck_packed_source_trace_v1",
        "status": "PASS_EXACT_H67_EP35_S10_FOUR_BOTTLENECK_SUPPORT_SIGN_BITMAPS_AND_RECONSTRUCTABLE_FLOAT32_VALUES",
        "identity": {
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint.resolve()),
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256(args.config.resolve()),
            "sample_workload_path": str(args.sample_workload.resolve()),
            "sample_workload_sha256": sha256(args.sample_workload.resolve()),
            "m40_tracer_sha256": sha256(Path(__file__).resolve()),
            "parent_profile_script_sha256": sha256(PROFILE_SCRIPT),
            "checkpoint_load_audit": load_audit,
            "bn_policy": bn_policy,
            "bn_modules_changed": bn_changed,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "dataset_root": str(Path(config["data"]["path"]).resolve()),
            "dataset_input_files": dataset_receipts,
        },
        "cohort": {
            "samples": len(sample_keys),
            "sample_keys": list(sample_keys),
            "operators": list(TARGETS),
            "records": len(rows),
            "shape": list(EXPECTED_SHAPE),
        },
        "records": rows,
        "coverage": {
            "source_support_and_sign_bitmap_exact": True,
            "source_float_value_content_hash_exact": True,
            "source_float_values_reconstructable": True,
            "source_float_values_zlib_payload_retained": True,
            "weight_content_hash_present": True,
            "weight_payload_retained": False,
            "local_activity_exact": True,
            "within_sample_exact_float_value_changed_bitmap_with_t0_reset_zero": True,
            "within_sample_sign_delta_direction_exact_with_t0_reset_zero": True,
            "within_sample_numeric_delta_magnitude_exact": False,
            "source_and_destination_product_events_materialized": False,
            "weight_quantization_present": False,
            "accumulator_and_threshold_product_present": False,
            "physical_addresses_present": False,
            "cycle_schedule_present": False,
        },
        "claim_boundary": (
            "Exact nonzero support and sign for the four H67 ep35 bottleneck "
            "Conv3x3 float inputs over the frozen ten-sample cohort, plus exact "
            "reconstructable float32 magnitude payloads and integer/ternary/codebook audit. "
            "Product expansion, "
            "weight quantization, accumulator values, physical addresses, cycle "
            "scheduling, Local/Motion system speedup, PPA, energy, and headline "
            "claims remain unadmitted."
        ),
    }
    manifest_path = output_dir / "m40_bottleneck_packed_source_manifest.json"
    require(not manifest_path.exists(), "refusing to overwrite M40 manifest")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
