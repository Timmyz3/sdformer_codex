"""Layer-wise profiler for MACs, sparsity, and timing proxies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from src.datasets import build_dataset
from src.models.registry import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.logging import write_csv, write_json
from src.utils.seed import set_seed


def flatten_record(prefix: str, data, rows: List[Dict[str, float]]) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flatten_record(child_prefix, value, rows)
    else:
        rows.append({"layer": prefix, "mac_proxy": float(data)})


def mask_ratio(mask: torch.Tensor | None, default: float = 1.0) -> float:
    if mask is None:
        return float(default)
    return float(mask.float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"]["device"] != "cpu" else "cpu")

    dataset = build_dataset(cfg, "eval")
    sample = dataset[0]
    model = build_model(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=str(device))
    if hasattr(model, "configure_backend"):
        model.configure_backend()

    event_voxel = sample["event_voxel"].unsqueeze(0).to(device)
    pre = model._preprocess_input(event_voxel)
    flops_record = {}
    if hasattr(model.model, "record_flops"):
        flops_record = model.model.record_flops()

    rows: List[Dict[str, float]] = []
    flatten_record("", flops_record, rows)
    token_keep_ratio = mask_ratio(pre.get("token_mask"), cfg["model"]["sparsity"].get("token_keep_ratio", 1.0))
    active_timestep_ratio = mask_ratio(pre.get("timestep_mask"), 1.0)
    active_window_ratio = mask_ratio(pre.get("window_mask"), 1.0)
    head_keep_ratio = mask_ratio(pre.get("head_mask"), 1.0)
    spike_density = float((pre["chunk"] != 0).float().mean().item())
    for row in rows:
        row["weight_bytes_proxy"] = row["mac_proxy"] * cfg["model"]["quant"]["weight_bits"] / 8.0
        row["activation_bytes_proxy"] = row["mac_proxy"] * cfg["model"]["quant"]["activation_bits"] / 8.0
        row["token_keep_ratio"] = token_keep_ratio
        row["active_timestep_ratio"] = active_timestep_ratio
        row["active_window_ratio"] = active_window_ratio
        row["head_keep_ratio"] = head_keep_ratio
        row["spike_density"] = spike_density
        row["latency_cycle_proxy"] = row["mac_proxy"] / 64.0

    out_dir = Path("experiments/results/tables")
    write_csv(out_dir / f"{cfg['model']['name']}_profile.csv", rows)
    write_json(
        out_dir / f"{cfg['model']['name']}_profile.json",
        {
            "variant": cfg["model"]["name"],
            "dataset": cfg["dataset"]["name"],
            "active_timestep_ratio": active_timestep_ratio,
            "active_window_ratio": active_window_ratio,
            "head_keep_ratio": head_keep_ratio,
            "spike_density": spike_density,
            "token_keep_ratio": token_keep_ratio,
        },
    )


if __name__ == "__main__":
    main()
