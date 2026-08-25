#!/usr/bin/env python3
"""Recalibrate PAFT BatchNorm running buffers without changing parameters.

This is a hardware-feedback probe: `no_running` valid825 is accurate but needs
sample-dependent BN reductions, while the foldable `running` policy is worse.
The script resets all BatchNorm running buffers, performs deterministic forward
passes over unaugmented training samples, and writes a new checkpoint only if
every non-BN-buffer tensor remains bit-identical to the PAFT source checkpoint.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def prepare_voxel_input(chunk: torch.Tensor, config: dict) -> torch.Tensor:
    if config["model"]["encoding"] != "voxel":
        raise RuntimeError("M193 supports the frozen voxel encoding only")
    if config["loader"].get("polarity", False):
        negative = torch.nn.functional.relu(-chunk)
        positive = torch.nn.functional.relu(chunk)
        chunk = torch.cat(
            (positive.unsqueeze(2), negative.unsqueeze(2)), dim=2
        )
    if config["model"].get("norm_input") == "minmax":
        nonzero = chunk != 0
        if bool(nonzero.any()):
            minimum = torch.min(chunk[nonzero])
            maximum = torch.max(chunk[nonzero])
            if not bool(minimum == maximum):
                chunk[nonzero] = (chunk[nonzero] - minimum) / (
                    maximum - minimum
                )
    elif config["model"].get("norm_input") == "std":
        nonzero = chunk != 0
        if bool(nonzero.any()):
            values = chunk[nonzero]
            standard = values.std()
            if bool(standard != 0):
                chunk[nonzero] = (values - values.mean()) / standard
    return chunk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = args.repo.resolve()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    require(args.samples > 0, "samples must be positive")
    require(not output.exists(), "refusing to overwrite M193 output")
    require(config_path.is_file(), "config missing")
    require(checkpoint_path.is_file(), "checkpoint missing")
    script_path = Path(__file__).resolve()
    script_start = file_sha256(script_path)
    checkpoint_start = file_sha256(checkpoint_path)
    config_sha = file_sha256(config_path)

    experiment = repo / "neuron_experiments/H9_bipolar_self_attention"
    upstream = repo / "third_party/SDformerFlow"
    overlay = experiment / "overlay"
    sys.path.insert(0, str(overlay))
    sys.path.insert(1, str(upstream))
    os.environ["SDFORMER_USE_MLFLOW"] = "0"
    os.environ["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    os.environ.setdefault("SDFORMER_SNN_BACKEND", "cupy")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import eval_DSEC_flow_SNN as evaluator  # pylint: disable=import-error
    from models.STSwinNet_SNN.h9_load_audit import (  # pylint: disable=import-error
        load_checkpoint_with_h9_audit,
    )

    parser_config = evaluator.YAMLParser(str(config_path))
    config = evaluator.YAMLParser.combine_entries(parser_config.config)
    if not os.path.isabs(config["data"]["path"]):
        candidates = [
            upstream / config["data"]["path"],
            repo / config["data"]["path"],
        ]
        resolved = next((path.resolve() for path in candidates if path.is_dir()), None)
        require(resolved is not None, "DSEC data path could not be resolved")
        config["data"]["path"] = str(resolved)
    config["loader"]["batch_size"] = 1
    config["loader"]["augment_prob"] = [0.0 for _ in
                                         config["loader"].get("augment", [])]
    config["swin_transformer"]["input_size"] = list(
        config["loader"]["resolution"]
    )

    evaluator._install_h9_overlay(str(config_path))
    dataset = evaluator.DSECDatasetLite(
        config, file_list="train", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1),
    )
    require(len(dataset) >= args.samples, "calibration population too small")

    model_class = getattr(evaluator, config["model"]["name"])
    if config["swin_transformer"]["use_arc"][0]:
        model = model_class(
            config["model"].copy(), config["swin_transformer"].copy()
        )
    else:
        model = model_class(config["model"].copy())
    device = torch.device(args.device)
    model.to(device)
    model.init_weights()
    evaluator._install_h9_modules(model, config)
    model = load_checkpoint_with_h9_audit(
        str(checkpoint_path), model, device, config=config,
        remap=config["loader"].get("remap"), test=True,
    )
    evaluator.functional.reset_net(model)
    evaluator.functional.set_step_mode(model, config["data"]["step_mode"])

    neuron_type = config["model"]["spiking_neuron"]["neuron_type"]
    neuron_map = {
        "if": evaluator.neuron.IFNode,
        "lif": evaluator.neuron.LIFNode,
        "plif": evaluator.neuron.ParametricLIFNode,
        "glif": evaluator.GatedLIFNode,
        "psn": evaluator.PSN,
        "SLTTlif": evaluator.SLTTLIFNode,
    }
    require(neuron_type in neuron_map, "unsupported neuron type")
    evaluator.configure_snn_backend(
        model, device, config, neuron_map[neuron_type]
    )

    module_counts = {}
    for module in model.modules():
        class_name = module.__class__.__name__
        module_counts[class_name] = module_counts.get(class_name, 0) + 1
    attention_modules = sum(
        1 for module in model.modules() if hasattr(module, "_h9_shiftmax_cfg")
    )
    require(module_counts.get("ATLIFTernaryPSN", 0) == 105,
            "ATLIF module population drift")
    require(attention_modules == 12,
            "Shiftmax attention population drift")

    source_payload = torch.load(checkpoint_path, map_location="cpu")
    require(set(source_payload) == {"model_state_dict"},
            "checkpoint wrapper drift")
    source_state = source_payload["model_state_dict"]
    model.eval()
    bn_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, _BatchNorm):
            continue
        require(module.track_running_stats, name + " does not track running stats")
        module.reset_running_stats()
        module.momentum = None
        module.train()
        bn_modules.append((name, module))
    require(len(bn_modules) == 78, "BatchNorm population drift")

    if args.samples == 1:
        calibration_indices = [0]
    else:
        calibration_indices = [
            (sample * (len(dataset) - 1)) // (args.samples - 1)
            for sample in range(args.samples)
        ]
    require(len(set(calibration_indices)) == args.samples,
            "calibration indices are not unique")
    calibration_files = []
    for index in calibration_indices:
        row = dataset.files[index]
        calibration_files.append([str(value) for value in row])
    calibration_identity = json.dumps(
        calibration_files, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    calibration_files_sha = hashlib.sha256(calibration_identity).hexdigest()
    loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset, calibration_indices),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    observed_samples = 0
    with torch.no_grad():
        for chunk, _mask, _label in loader:
            if observed_samples >= args.samples:
                break
            evaluator.functional.reset_net(model)
            chunk = chunk.to(device=device, dtype=torch.float32,
                             non_blocking=False)
            chunk = prepare_voxel_input(chunk, config)
            model(chunk)
            observed_samples += int(chunk.shape[0])
            print(
                "[M193] calibrated {}/{}".format(
                    observed_samples, args.samples
                ),
                flush=True,
            )
    require(observed_samples == args.samples, "calibration sample drift")

    calibrated_state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    require(set(calibrated_state) == set(source_state),
            "calibrated state-key drift")
    allowed_suffixes = (
        ".running_mean", ".running_var", ".num_batches_tracked"
    )
    changed = []
    forbidden = []
    for key in sorted(source_state):
        equal = torch.equal(source_state[key].detach().cpu(), calibrated_state[key])
        if equal:
            continue
        if key.endswith(allowed_suffixes):
            changed.append(key)
        else:
            forbidden.append(key)
    require(not forbidden, "non-BN-buffer tensors changed: " + str(forbidden[:8]))
    changed_by_suffix = {
        suffix: sum(1 for key in changed if key.endswith(suffix))
        for suffix in allowed_suffixes
    }
    tracked = [
        int(module.num_batches_tracked.detach().cpu())
        for _name, module in bn_modules
    ]
    require(min(tracked) == args.samples and max(tracked) == args.samples,
            "BN batch-counter drift")
    require(all(torch.isfinite(module.running_mean).all() and
                torch.isfinite(module.running_var).all()
                for _name, module in bn_modules),
            "non-finite BN running statistics")

    sn2_thresholds = []
    for name, module in model.named_modules():
        if name.endswith(".mlp.sn2.spiking_neuron"):
            sn2_thresholds.append(float(module.thresh.detach().cpu()))
    require(len(sn2_thresholds) == 12, "FFN sn2 population drift")
    require(all(value == 1.0 for value in sn2_thresholds),
            "unit sn2 threshold identity drift")

    output.mkdir(parents=True)
    checkpoint_out = output / "checkpoint_paft_ep4_bncalib{}.pth".format(
        args.samples
    )
    temporary = output / (checkpoint_out.name + ".partial")
    torch.save({"model_state_dict": calibrated_state}, temporary)
    os.replace(temporary, checkpoint_out)
    receipt = {
        "schema": "m193_paft_bn_recalibration_probe_v1",
        "status": "PASS_BN_RUNNING_BUFFER_ONLY_RECALIBRATION_VALID825_OPEN",
        "identity": {
            "script_sha256": script_start,
            "source_checkpoint_sha256": checkpoint_start,
            "config_sha256": config_sha,
            "calibrated_checkpoint_sha256": file_sha256(checkpoint_out),
            "ATLIFTernaryPSN_modules": module_counts["ATLIFTernaryPSN"],
            "Shiftmax_attention_modules": attention_modules,
            "checkpoint_load_audit": getattr(model, "_h9_load_audit", None),
        },
        "calibration": {
            "dataset": "DSEC train only, deterministic evenly spaced indices",
            "dataset_population": len(dataset),
            "sequence_file": str(Path(dataset.sequence_file).resolve()),
            "sequence_file_sha256": file_sha256(Path(dataset.sequence_file)),
            "samples": observed_samples,
            "first_index": calibration_indices[0],
            "last_index": calibration_indices[-1],
            "calibration_files_sha256": calibration_files_sha,
            "calibration_files": calibration_files,
            "batch_size": 1,
            "bn_modules": len(bn_modules),
            "bn_momentum": "cumulative_average",
            "reset_running_stats": True,
            "changed_bn_running_buffers": len(changed),
            "changed_by_suffix": changed_by_suffix,
            "changed_non_bn_tensors": len(forbidden),
            "num_batches_tracked_min": min(tracked),
            "num_batches_tracked_max": max(tracked),
            "sn2_thresholds_exactly_one": True,
        },
        "claim_boundary": {
            "weights_changed": False,
            "valid825_running_policy": False,
            "accuracy_promotion": False,
            "cycle_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    receipt_path = output / "m193_bn_recalibration_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    require(file_sha256(script_path) == script_start,
            "script changed during calibration")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
