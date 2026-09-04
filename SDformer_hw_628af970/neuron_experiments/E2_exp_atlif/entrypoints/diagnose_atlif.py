"""Batch-level diagnostics for ATLIF training mechanics.

Runs a few train batches and reports whether threshold updates, activity
regularization, threshold values, and firing rates move in the expected
direction. Kept in the experiment directory so the baseline tree stays intact.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _install_paths() -> Path:
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = experiment_root / "overlay"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
    return baseline_root


def _stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} min={min(values):.6g} "
        f"mean={sum(values) / len(values):.6g} max={max(values):.6g}"
    )


def _delta_stats(before: list[float], after: list[float]) -> str:
    return _stats([right - left for left, right in zip(before, after)])


def _atlif_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "thresh") and hasattr(module, "update_value"):
            yield name, module


def _threshold_values(model) -> list[float]:
    return [float(module.thresh.detach().reshape(-1)[0]) for _, module in _atlif_modules(model)]


def _optimizer_threshold_stats(model, optimizer) -> tuple[int, int, int]:
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    threshold_params = [module.thresh for _, module in _atlif_modules(model)]
    requires_grad = sum(1 for param in threshold_params if param.requires_grad)
    in_optimizer = sum(1 for param in threshold_params if id(param) in optimizer_param_ids)
    return len(threshold_params), requires_grad, in_optimizer


def _update_values(model) -> list[float]:
    values = []
    for _, module in _atlif_modules(model):
        update_value = module.update_value
        if torch.is_tensor(update_value):
            values.append(float(update_value.detach().reshape(-1)[0].cpu()))
        else:
            values.append(float(update_value))
    return values


def _firing_rates(model) -> list[float]:
    values = []
    for _, module in _atlif_modules(model):
        if hasattr(module, "r"):
            values.append(float(module.r))
    return values


def _activity_values(model) -> list[float]:
    values = []
    for _, module in _atlif_modules(model):
        act_value = getattr(module, "act_value", None)
        if torch.is_tensor(act_value):
            values.append(float(act_value.detach().cpu()))
        else:
            values.append(float(act_value))
    return values


def _prepare_batch(chunk, label, mask, config, transform, device):
    if transform is not None:
        chunk, label, mask = transform((chunk, label, mask.float()))

    if config["model"]["encoding"] == "voxel":
        if config["loader"]["polarity"]:
            neg = torch.nn.functional.relu(-chunk)
            pos = torch.nn.functional.relu(chunk)
            chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)), dim=2)
    elif config["model"]["encoding"] == "cnt":
        if config["swin_transformer"]["use_arc"][1] == "PatchEmbed3D":
            chunk = torch.transpose(chunk, 1, 2)
        elif config["loader"]["polarity"]:
            chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))
    else:
        raise ValueError(f"Unsupported encoding: {config['model']['encoding']}")

    if config["model"]["norm_input"] == "minmax":
        non_zero = chunk != 0
        if non_zero.any():
            min_value = torch.min(chunk[non_zero])
            max_value = torch.max(chunk[non_zero])
            if min_value != max_value:
                chunk[non_zero] = (chunk[non_zero] - min_value) / (max_value - min_value)
    elif config["model"]["norm_input"] == "std":
        non_zero = chunk != 0
        if non_zero.any():
            stddev = chunk[non_zero].std()
            if stddev > 0:
                chunk[non_zero] = (chunk[non_zero] - chunk[non_zero].mean()) / stddev

    if config["data"]["spike_th"] is not None:
        chunk[chunk > config["data"]["spike_th"]] = 1
        chunk[chunk < config["data"]["spike_th"]] = 0
    return chunk.to(device), label.to(device), mask.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    baseline_root = _install_paths()
    os.chdir(baseline_root)

    from configs.parser import YAMLParser
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from DSEC_dataloader.data_augmentation import CenterCrop, Compose
    from loss.flow_supervised import flow_loss_supervised
    from models.STSwinNet_SNN.Spiking_STSwinNet import (
        MS_SpikingformerFlowNet,
        MS_SpikingformerFlowNet_en4,
        SpikingformerFlowNet,
    )
    from models.STSwinNet_SNN.experimental_neurons.factory import resolve_backend_neuron_type
    from models.STSwinNet_SNN.experimental_neurons.training import regularize_activity, sanitize_threshold_grads, threshold_update
    from spikingjelly.activation_based import functional
    from torch.utils.data import DataLoader
    from utils.runtime_backend import configure_snn_backend

    yaml_parser = YAMLParser(str(config_path))
    config = YAMLParser.combine_entries(yaml_parser.config)
    device = yaml_parser.device
    config.setdefault("runtime", {})
    config["runtime"]["snn_backend"] = "torch"
    config["loader"]["batch_size"] = min(int(config["loader"]["batch_size"]), 4)
    config["loader"]["n_workers"] = 0

    if config["loader"]["crop"] is not None:
        transform = Compose([CenterCrop((config["loader"]["crop"][0], config["loader"]["crop"][1]))])
        config["swin_transformer"]["input_size"] = [config["loader"]["crop"][0], config["loader"]["crop"][1]]
    else:
        transform = None
        config["swin_transformer"]["input_size"] = [config["loader"]["resolution"][0], config["loader"]["resolution"][1]]

    dataset = DSECDatasetLite(config, file_list="train", stereo=False)
    loader = DataLoader(dataset, batch_size=config["loader"]["batch_size"], shuffle=False, drop_last=False, pin_memory=True)

    model_cls = {
        "SpikingformerFlowNet": SpikingformerFlowNet,
        "MS_SpikingformerFlowNet": MS_SpikingformerFlowNet,
        "MS_SpikingformerFlowNet_en4": MS_SpikingformerFlowNet_en4,
    }[config["model"]["name"]]
    if config["swin_transformer"]["use_arc"][0]:
        model = model_cls(config["model"].copy(), config["swin_transformer"].copy())
    else:
        model = model_cls(config["model"].copy())
    model.to(device)
    model.init_weights()
    functional.reset_net(model)
    functional.set_step_mode(model, config["data"]["step_mode"])
    configure_snn_backend(model, device, config, resolve_backend_neuron_type(config["model"]["spiking_neuron"]["neuron_type"]))
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["wd"]),
    )
    loss_function = flow_loss_supervised(config, device)
    scaler = torch.cuda.amp.GradScaler(enabled=(config["optimizer"].get("use_amp", False) and not args.disable_amp))

    print(f"config={config_path}")
    print(f"neuron_type={config['model']['spiking_neuron']['neuron_type']}")
    print(f"v_th={config['model']['spiking_neuron']['v_th']} tau={config['model']['spiking_neuron']['tau']}")
    print(f"threshold_eta={config['model']['spiking_neuron'].get('threshold_eta')}")
    print(f"activity_eta={config.get('experimental_neuron', {}).get('activity_eta')}")
    print(f"thresholds_initial {_stats(_threshold_values(model))}")
    threshold_count, threshold_requires_grad, threshold_in_optimizer = _optimizer_threshold_stats(model, optimizer)
    print(
        "optimizer_threshold_params="
        f"count={threshold_count} requires_grad={threshold_requires_grad} in_optimizer={threshold_in_optimizer}"
    )

    for batch_idx, (chunk, mask, label) in enumerate(loader):
        if batch_idx >= args.batches:
            break
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)
        chunk = chunk.to(device=device, dtype=torch.float32, non_blocking=True)
        label = label.to(device=device, dtype=torch.float32, non_blocking=True)
        mask = torch.unsqueeze(mask.bool().to(device=device, non_blocking=True), dim=1)
        chunk, label, mask = _prepare_batch(chunk, label, mask, config, transform, device)

        before = _threshold_values(model)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            pred_list = model(chunk)
            pred = pred_list["flow"]
            flow_loss = loss_function(pred, label, mask, gamma=config["loss"]["gamma"])
            activity_loss = regularize_activity(model, config)
            total_loss = flow_loss + activity_loss
        update_before_step = _update_values(model)
        firing = _firing_rates(model)
        act_values = _activity_values(model)

        scaler.scale(total_loss).backward()
        grad_values = []
        for _, module in _atlif_modules(model):
            if module.thresh.grad is not None:
                grad_values.append(float(module.thresh.grad.detach().reshape(-1)[0].cpu()))
        sanitize_stats = sanitize_threshold_grads(model, config)
        sanitized_grad_values = []
        for _, module in _atlif_modules(model):
            if module.thresh.grad is not None:
                sanitized_grad_values.append(float(module.thresh.grad.detach().reshape(-1)[0].cpu()))
        if config["loss"]["clip_grad"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
        scaler.step(optimizer)
        threshold_update(model, optimizer.param_groups[0]["lr"], config)
        scaler.update()
        after = _threshold_values(model)

        print(f"batch={batch_idx}")
        print(f"  flow_loss={float(flow_loss.detach().cpu()):.6g} activity_loss={float(activity_loss.detach().cpu()):.6g} total={float(total_loss.detach().cpu()):.6g}")
        print(f"  firing {_stats(firing)}")
        print(f"  act_value {_stats(act_values)} sum={sum(act_values):.6g}")
        print(f"  update_value {_stats(update_before_step)}")
        print(f"  thresh_grad {_stats(grad_values)}")
        print(f"  sanitize_threshold_grads {sanitize_stats}")
        print(f"  thresh_grad_sanitized {_stats(sanitized_grad_values)}")
        print(f"  thresholds_before {_stats(before)}")
        print(f"  thresholds_after  {_stats(after)}")
        print(f"  thresholds_delta  {_delta_stats(before, after)}")


if __name__ == "__main__":
    main()
