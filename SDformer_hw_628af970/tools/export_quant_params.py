"""Export quantization metadata aligned with the RTL interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml

from src.utils.config import load_config


def tensor_scale(tensor: torch.Tensor, bits: int) -> float:
    max_abs = float(tensor.abs().max().item()) if tensor.numel() else 1.0
    levels = max(1, 2 ** (bits - 1) - 1)
    return max_abs / levels if max_abs > 0 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="experiments/results/hw_export/quant_params.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    spec_path = Path(cfg["hw_export"]["quant_spec"])
    with spec_path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    export = {
        "variant": cfg["model"]["name"],
        "quant_spec": spec,
        "layers": [],
        "tile": spec["tiling"],
    }

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        for name, tensor in state.items():
            if not torch.is_tensor(tensor):
                continue
            export["layers"].append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "weight_scale": tensor_scale(tensor.float(), spec["quantization"]["weight_bits"]),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(export, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

