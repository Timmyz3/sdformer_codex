"""Generate paper-inspired sparse pruning and voxel adapter configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "autoresearch_sparsity" / "configs" / "baseline_upstream.yml"
OUT = ROOT / "autoresearch_sparsity" / "configs"


def set_common(cfg: dict, name: str) -> dict:
    cfg = deepcopy(cfg)
    cfg["experiment"] = name
    cfg.setdefault("loader", {})
    cfg["loader"]["batch_size"] = 8
    cfg["loader"]["n_workers"] = 8
    cfg.setdefault("optimizer", {})
    cfg["optimizer"]["lr"] = 1.0e-5
    cfg["optimizer"]["wd"] = 0.001
    cfg["optimizer"]["use_amp"] = True
    cfg.setdefault("runtime", {})
    cfg["runtime"]["allow_tf32"] = True
    cfg["runtime"]["cudnn_benchmark"] = True
    cfg.setdefault("sparsity", {})
    cfg["sparsity"]["enabled"] = True
    return cfg


def write(name: str, cfg: dict) -> None:
    path = OUT / f"{name}.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(path)


def main() -> None:
    base = yaml.safe_load(BASE.read_text())

    cfg = set_common(base, "s41a_sparsespikformer_token85")
    cfg["sparsity"].update({
        "sparsespikformer_token_pruning": {
            "enabled": True,
            "keep_ratio": 0.85,
            "min_keep_ratio": 0.25,
            "window_size": [8, 8],
            "stochastic": True,
        }
    })
    write("s41a_sparsespikformer_token85", cfg)

    cfg = set_common(base, "s41b_qpsnn_svs90")
    cfg["sparsity"].update({
        "qpsnn_svs_pruning": {
            "enabled": True,
            "keep_ratio": 0.90,
            "remove_dc": False,
        }
    })
    write("s41b_qpsnn_svs90", cfg)

    cfg = set_common(base, "v41a_edcflow_temporal_diff")
    cfg["sparsity"].update({
        "voxel_adapter": {
            "enabled": True,
            "method": "edcflow_temporal_diff",
            "mode": "residual",
            "alpha": 0.20,
            "rescale": True,
            "clamp": True,
            "clamp_min": 0.0,
            "clamp_max": 1.0,
        }
    })
    write("v41a_edcflow_temporal_diff", cfg)

    cfg = set_common(base, "v41b_eventpillars_lite")
    cfg["sparsity"].update({
        "voxel_adapter": {
            "enabled": True,
            "method": "eventpillars_lite",
            "density_alpha": 0.12,
            "range_alpha": 0.08,
            "rescale": True,
            "preserve_zero": True,
            "clamp": True,
            "clamp_min": 0.0,
            "clamp_max": 1.0,
        }
    })
    write("v41b_eventpillars_lite", cfg)

    cfg = set_common(base, "s41c_ssf_qpsnn_edc")
    cfg["sparsity"].update({
        "sparsespikformer_token_pruning": {
            "enabled": True,
            "keep_ratio": 0.88,
            "min_keep_ratio": 0.30,
            "window_size": [8, 8],
            "stochastic": True,
        },
        "qpsnn_svs_pruning": {
            "enabled": True,
            "keep_ratio": 0.95,
            "remove_dc": False,
        },
        "voxel_adapter": {
            "enabled": True,
            "method": "edcflow_temporal_diff",
            "mode": "residual",
            "alpha": 0.15,
            "rescale": True,
            "clamp": True,
            "clamp_min": 0.0,
            "clamp_max": 1.0,
        },
    })
    write("s41c_ssf_qpsnn_edc", cfg)


if __name__ == "__main__":
    main()
