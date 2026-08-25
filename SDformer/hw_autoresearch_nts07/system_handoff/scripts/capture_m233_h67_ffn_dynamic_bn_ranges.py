#!/usr/bin/env python3
"""Capture checkpoint-bound H67 FFN current-batch BN numeric ranges on CUDA.

The output is a hardware-design receipt, not an accuracy or speedup result.  It
records exact per-sample/per-channel BN input moments, affine coefficients and
input/output ranges for the 12 FFN BN1 and 12 FFN BN2 modules.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints"
sys.path.insert(0, str(ENTRYPOINT))
import profile_nts11_hardware_p0 as profile  # noqa: E402


EXPECTED = {
    "profile": "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    "config": "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
    "checkpoint": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
}
EXPECTED_BN_MODULES_CHANGED = 78
EXPECTED_FFN_BN_MODULES = 24
EXPECTED_SAMPLES = 10


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_identity(dataset, sample_id):
    row = dataset.files[sample_id]
    names = list(row) if isinstance(row, (list, tuple)) else [str(row)]
    key = "|".join(str(item) for item in names)
    sequence = "|".join(
        "_".join(Path(str(item)).stem.split("_")[:-1]) for item in names)
    return key, sequence


def safe_key(name):
    return name.replace(".", "__")


def vector_summary(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    require(array.size > 0 and np.all(np.isfinite(array)),
            "non-finite summary vector")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p0001": float(np.quantile(array, 0.0001)),
        "p001": float(np.quantile(array, 0.001)),
        "p01": float(np.quantile(array, 0.01)),
        "p50": float(np.quantile(array, 0.5)),
        "p99": float(np.quantile(array, 0.99)),
        "p999": float(np.quantile(array, 0.999)),
        "p9999": float(np.quantile(array, 0.9999)),
        "max": float(array.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=EXPECTED_SAMPLES)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    require(args.samples == EXPECTED_SAMPLES and args.num_workers == 0,
            "M233 requires exact ten samples and num-workers=0")
    require(not args.output_dir.resolve().exists(),
            "refusing to overwrite M233 output")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    require(config_path.is_file() and checkpoint_path.is_file(),
            "missing config/checkpoint")
    observed = {
        "profile": sha256(Path(profile.__file__).resolve()),
        "config": sha256(config_path),
        "checkpoint": sha256(checkpoint_path),
    }
    require(observed == EXPECTED, "M233 frozen source identity drift")
    require(torch.cuda.is_available(), "M233 requires CUDA")

    config, device = profile.load_config(config_path)
    require(torch.device(device).type == "cuda", "M233 requires CUDA device")
    require(config.get("test", {}).get("bn_policy") == "no_running",
            "M233 requires frozen no-running BN policy")
    dataset = profile.DSECDatasetLite(
        config, file_list="valid", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        pin_memory=False, num_workers=args.num_workers)
    transform_valid = None
    if config["loader"].get("crop") is not None:
        transform_valid = profile.Compose([profile.CenterCrop((
            config["loader"]["crop"][0], config["loader"]["crop"][1]))])

    model = profile.build_model(config, checkpoint_path, device)
    load_audit = profile.validate_h9_load_audit(model, config)
    require(load_audit is not None and
            int(load_audit.get("missing_count", 0)) == 0 and
            int(load_audit.get("unexpected_count", 0)) == 0,
            "M233 checkpoint load is not exact")
    changed = profile.configure_batch_norm_evaluation(model, "no_running")
    require(changed == EXPECTED_BN_MODULES_CHANGED,
            "M233 changed BN module count drift")

    named = dict(model.named_modules())
    target_names = sorted(name for name in named if
        (name.endswith(".mlp.bn1.norm_layer") or
         name.endswith(".mlp.bn2.norm_layer")))
    require(len(target_names) == EXPECTED_FFN_BN_MODULES,
            "M233 FFN BN target count drift")
    require(sum(name.endswith(".mlp.bn1.norm_layer")
                for name in target_names) == 12 and
            sum(name.endswith(".mlp.bn2.norm_layer")
                for name in target_names) == 12,
            "M233 BN1/BN2 target split drift")

    captured = {name: {} for name in target_names}
    static = {}
    records = []
    current_sample = {"index": -1}

    for name in target_names:
        module = named[name]
        require(hasattr(module, "weight") and hasattr(module, "bias") and
                module.weight is not None and module.bias is not None,
                "M233 BN affine payload missing: " + name)
        gamma = module.weight.detach().float().cpu().numpy().copy()
        beta = module.bias.detach().float().cpu().numpy().copy()
        require(gamma.ndim == 1 and beta.shape == gamma.shape,
                "M233 BN affine shape mismatch: " + name)
        static[name] = {
            "channels": int(gamma.size),
            "eps": float(module.eps),
            "gamma": gamma,
            "beta": beta,
        }
        for metric in ("mean", "variance", "invstd", "alpha", "offset",
                       "input_min", "input_max", "output_min", "output_max"):
            captured[name][metric] = []

    handles = []

    def make_hook(name):
        module = named[name]

        def hook(_module, inputs, output):
            require(current_sample["index"] >= 0,
                    "M233 hook outside sample")
            require(isinstance(inputs, tuple) and len(inputs) == 1 and
                    torch.is_tensor(inputs[0]) and torch.is_tensor(output),
                    "M233 BN hook payload mismatch")
            value = inputs[0].detach().float()
            normalized = output.detach().float()
            require(value.ndim == 5 and normalized.shape == value.shape,
                    "M233 expects T,N,C,H,W BN tensors")
            require(int(value.shape[2]) == static[name]["channels"],
                    "M233 BN channel mismatch")
            dims = (0, 1, 3, 4)
            mean = torch.mean(value, dim=dims)
            variance = torch.var(value, dim=dims, unbiased=False)
            invstd = torch.rsqrt(variance + float(module.eps))
            gamma = module.weight.detach().float()
            beta = module.bias.detach().float()
            alpha = gamma * invstd
            offset = beta - alpha * mean
            input_min = torch.amin(value, dim=dims)
            input_max = torch.amax(value, dim=dims)
            output_min = torch.amin(normalized, dim=dims)
            output_max = torch.amax(normalized, dim=dims)
            maximum_error = 0.0
            view_shape = (1, -1, 1, 1)
            for time_index in range(int(value.shape[0])):
                reconstructed = (value[time_index] * alpha.view(view_shape)
                                 + offset.view(view_shape))
                maximum_error = max(maximum_error, float(torch.max(torch.abs(
                    normalized[time_index] - reconstructed)).item()))
            tensors = {
                "mean": mean, "variance": variance, "invstd": invstd,
                "alpha": alpha, "offset": offset,
                "input_min": input_min, "input_max": input_max,
                "output_min": output_min, "output_max": output_max,
            }
            for metric, tensor in tensors.items():
                array = tensor.cpu().numpy().astype(np.float32, copy=False)
                require(np.all(np.isfinite(array)),
                        "M233 non-finite " + name + " " + metric)
                captured[name][metric].append(array.copy())
            records.append({
                "sample_id": current_sample["index"],
                "module": name,
                "kind": "bn1" if ".bn1." in name else "bn2",
                "shape": [int(item) for item in value.shape],
                "population_per_channel": int(value.numel() // value.shape[2]),
                "channels": int(value.shape[2]),
                "eps": float(module.eps),
                "maximum_affine_reconstruction_error": maximum_error,
            })
        return hook

    for name in target_names:
        handles.append(named[name].register_forward_hook(make_hook(name)))

    sample_rows = []
    try:
        processed = 0
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for chunk, mask, label in loader:
                if processed >= EXPECTED_SAMPLES:
                    break
                profile.functional.reset_net(model)
                current_sample["index"] = processed
                sample_key, sequence = sample_identity(dataset, processed)
                before = len(records)
                x, _label, _mask = profile.preprocess_chunk(
                    config, chunk, label, mask, transform_valid, device)
                model(x)
                torch.cuda.synchronize(device)
                require(len(records) - before == EXPECTED_FFN_BN_MODULES,
                        "M233 per-sample hook count drift")
                sample_rows.append({
                    "sample_id": processed,
                    "sample_key": sample_key,
                    "sequence": sequence,
                })
                processed += 1
                print("[M233] captured sample {}/10".format(processed),
                      flush=True)
        require(processed == EXPECTED_SAMPLES,
                "M233 dataset ended before ten samples")
    finally:
        current_sample["index"] = -1
        for handle in handles:
            handle.remove()

    require(len(records) == EXPECTED_SAMPLES * EXPECTED_FFN_BN_MODULES,
            "M233 total hook count drift")
    arrays = {}
    summaries = {}
    all_metrics = {metric: [] for metric in captured[target_names[0]]}
    for name in target_names:
        key = safe_key(name)
        arrays[key + "__gamma"] = static[name]["gamma"]
        arrays[key + "__beta"] = static[name]["beta"]
        summaries[name] = {
            "channels": static[name]["channels"],
            "eps": static[name]["eps"],
            "gamma": vector_summary(static[name]["gamma"]),
            "beta": vector_summary(static[name]["beta"]),
        }
        for metric, values in captured[name].items():
            array = np.stack(values, axis=0)
            require(array.shape == (EXPECTED_SAMPLES, static[name]["channels"]),
                    "M233 stacked shape mismatch")
            arrays[key + "__" + metric] = array
            summaries[name][metric] = vector_summary(array)
            all_metrics[metric].append(array.reshape(-1))

    args.output_dir.mkdir(parents=True)
    npz_path = args.output_dir / "m233_h67_ffn_dynamic_bn_ranges_s10.npz"
    np.savez_compressed(npz_path, **arrays)
    record_path = args.output_dir / "per_sample_module_records.csv"
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            encoded = dict(row)
            encoded["shape"] = json.dumps(encoded["shape"], separators=(",", ":"))
            writer.writerow(encoded)
    samples_path = args.output_dir / "samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    global_summary = {
        metric: vector_summary(np.concatenate(values))
        for metric, values in all_metrics.items()
    }
    max_error = max(row["maximum_affine_reconstruction_error"]
                    for row in records)
    payload = {
        "schema": "m233_h67_ffn_dynamic_bn_range_capture_v1",
        "status": "PASS_CHECKPOINT_BOUND_S10_DYNAMIC_BN_RANGE_CAPTURE",
        "scope": "H67 ep35 frozen first-ten valid samples, 12 FFN BN1 plus 12 FFN BN2, no-running/current-batch",
        "identity": {
            "source_sha256": sha256(Path(__file__).resolve()),
            "profile_config_checkpoint_sha256": observed,
            "checkpoint_load_audit": load_audit,
            "bn_modules_changed": changed,
            "samples": sample_rows,
        },
        "capture": {
            "sample_count": EXPECTED_SAMPLES,
            "target_module_count": EXPECTED_FFN_BN_MODULES,
            "records": len(records),
            "reduction_dimensions": "T,N,H,W for T,N,C,H,W input",
            "variance": "biased/current-batch normalization variance",
            "maximum_float32_affine_reconstruction_error": max_error,
            "cuda_max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)),
            "npz": npz_path.name,
        },
        "global_numeric_summary": global_summary,
        "per_module_summary": summaries,
        "admission": {
            "checkpoint_bound_float32_ranges": True,
            "current_batch_bn_affine_identity": True,
            "fixed_point_format": False,
            "rsqrt_approximation": False,
            "integer_hardware_order_miter": False,
            "valid825_accuracy": False,
            "cycle_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    summary_path = args.output_dir / "m233_h67_ffn_dynamic_bn_range_summary_r1.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    manifest_path = args.output_dir / "manifest.sha256"
    evidence = [npz_path, record_path, samples_path, summary_path]
    manifest_path.write_text("".join(
        "{}  {}\n".format(sha256(path), path.name) for path in evidence),
        encoding="utf-8")
    print("PASS M233 {} {}".format(summary_path, sha256(manifest_path)),
          flush=True)


if __name__ == "__main__":
    os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")
    raise SystemExit(main())
