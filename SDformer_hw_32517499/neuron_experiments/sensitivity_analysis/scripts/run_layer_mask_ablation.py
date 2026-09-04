"""Run exact-layer output masking ablations for SDFormerFlow.

The script loads a checkpoint, masks selected module outputs with forward hooks,
and reports AEE/AAE on a small validation subset. It is intended to estimate
accuracy sensitivity before adding trainable sparse gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.profile_sops import (  # noqa: E402
    MetricAccumulator,
    build_model_and_data,
    install_project_paths,
    prepare_batch,
)


def map_output(value: Any, scale: float):
    if torch.is_tensor(value):
        return value * scale
    if isinstance(value, tuple):
        return tuple(map_output(item, scale) for item in value)
    if isinstance(value, list):
        return [map_output(item, scale) for item in value]
    if isinstance(value, dict):
        return {key: map_output(item, scale) for key, item in value.items()}
    return value


def attach_masks(model: torch.nn.Module, layers: set[str], scale: float):
    handles = []
    found = set()
    for name, module in model.named_modules():
        if name in layers:
            found.add(name)

            def hook(_module, _inputs, output, mask_scale=scale):
                return map_output(output, mask_scale)

            handles.append(module.register_forward_hook(hook))
    missing = sorted(layers - found)
    if missing:
        raise ValueError("Missing target layers: " + ", ".join(missing))
    return handles


def read_target_set(path: Path, set_name: str) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row["layer"] for row in reader if row["set"] == set_name]


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    install_project_paths(REPO_ROOT, None)
    baseline_cwd = REPO_ROOT / "third_party" / "SDformerFlow"
    old_cwd = Path.cwd()
    try:
        os.chdir(baseline_cwd)
        config, model, loader, transform, device = build_model_and_data(args, REPO_ROOT)
        from loss.flow_supervised import AAE, AEE
        from spikingjelly.activation_based import functional

        target_layers = set(read_target_set(args.target_sets, args.set_name)) if args.set_name != "none" else set()
        handles = attach_masks(model, target_layers, args.mask_scale) if target_layers else []
        metric_acc = MetricAccumulator(args.metrics)
        num_seen = 0
        try:
            with torch.no_grad():
                for chunk, mask, label in loader:
                    functional.reset_net(model)
                    chunk = chunk.to(device=device, dtype=torch.float32, non_blocking=True)
                    label = label.to(device=device, dtype=torch.float32, non_blocking=True)
                    mask = torch.unsqueeze(mask.to(device=device, non_blocking=True), dim=1)
                    chunk, label, mask = prepare_batch(chunk, label, mask, config, transform)
                    pred_list = model(chunk.to(device))
                    pred = pred_list["flow"][-1]
                    if config.get("metrics", {}).get("mask_events", False):
                        event_mask = torch.sum(torch.sum(chunk, dim=1), dim=1, keepdim=True).bool()
                        mask = mask * event_mask
                    for metric in args.metrics:
                        if metric == "AEE":
                            metric_acc.update_aee(AEE(pred, label, mask, config["metrics"]["flow_scaling"])())
                        elif metric == "AAE":
                            metric_acc.update_scalar("AAE", AAE(pred, label, mask, config["metrics"]["flow_scaling"])()[0])
                    num_seen += chunk.shape[0]
                    if args.num_samples is not None and num_seen >= args.num_samples:
                        break
        finally:
            for handle in handles:
                handle.remove()
    finally:
        os.chdir(old_cwd)

    return {
        "set_name": args.set_name,
        "mask_scale": args.mask_scale,
        "num_target_layers": len(target_layers),
        "target_layers": sorted(target_layers),
        "samples": num_seen,
        "metrics": metric_acc.summary(),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--target-sets", required=True, type=Path)
    parser.add_argument("--set-name", required=True)
    parser.add_argument("--mask-scale", default=0.0, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--num-samples", default=8, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--remap", action="store_true")
    parser.add_argument("--snn-backend", default="torch")
    parser.add_argument("--metrics", nargs="+", default=["AEE", "AAE"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.target_sets = args.target_sets.resolve()
    args.output = args.output.resolve()
    result = run_eval(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
