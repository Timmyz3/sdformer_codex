"""Helpers for composing upstream SDformerFlow configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_upstream_config(cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    upstream_key = "base_train_config" if mode == "train" else "base_eval_config"
    upstream_path = Path(cfg["upstream"][upstream_key])
    upstream_cfg = load_yaml(upstream_path)
    upstream_cfg = deepcopy(upstream_cfg)

    dataset = cfg["dataset"]
    model = cfg["model"]
    runtime = cfg["runtime"]
    optimizer = cfg["optimizer"]
    loss = cfg["loss"]
    metrics = cfg["metrics"]

    upstream_cfg.setdefault("data", {})
    upstream_cfg.setdefault("loader", {})
    upstream_cfg.setdefault("model", {})
    upstream_cfg.setdefault("swin_transformer", {})
    upstream_cfg.setdefault("spiking_neuron", {})
    upstream_cfg.setdefault("optimizer", {})
    upstream_cfg.setdefault("loss", {})
    upstream_cfg.setdefault("metrics", {})
    upstream_cfg.setdefault("test", {})

    upstream_cfg["data"]["path"] = dataset["root"]
    upstream_cfg["data"]["preprocessed"] = dataset.get("preprocessed", True)
    upstream_cfg["data"]["num_frames"] = model["num_bins"]
    upstream_cfg["data"]["num_chunks"] = dataset.get("num_chunks", 1)
    upstream_cfg["data"]["step_mode"] = "m"
    upstream_cfg["data"]["spike_th"] = cfg["model"].get("spike_threshold")

    upstream_cfg["loader"]["resolution"] = dataset["resolution"]
    upstream_cfg["loader"]["crop"] = dataset.get("crop")
    upstream_cfg["loader"]["n_workers"] = runtime["num_workers"]
    upstream_cfg["loader"]["batch_size"] = runtime.get("batch_size", 1)
    upstream_cfg["loader"]["polarity"] = dataset.get("polarity", True)

    upstream_cfg["model"]["name"] = model["upstream_model_name"]
    upstream_cfg["model"]["encoding"] = model["encoding"]
    upstream_cfg["model"]["norm_input"] = model["norm_input"]
    upstream_cfg["model"]["num_bins"] = model["num_bins"]
    upstream_cfg["model"]["mask_output"] = model["mask_output"]

    upstream_cfg["swin_transformer"]["window_size"] = model["attention"]["window_size"]
    upstream_cfg["swin_transformer"]["pretrained_window_size"] = model["attention"]["pretrained_window_size"]
    if dataset.get("crop") is not None:
        upstream_cfg["swin_transformer"]["input_size"] = dataset["crop"]
    else:
        upstream_cfg["swin_transformer"]["input_size"] = dataset["resolution"]

    upstream_cfg["spiking_neuron"]["num_steps"] = model["temporal"]["max_steps"]
    upstream_cfg["spiking_neuron"]["v_th"] = model["neuron"]["v_th"]
    upstream_cfg["spiking_neuron"]["v_reset"] = model["neuron"]["v_reset"]
    upstream_cfg["spiking_neuron"]["neuron_type"] = model["neuron"]["type"]
    upstream_cfg["spiking_neuron"]["tau"] = model["neuron"]["tau"]
    upstream_cfg["spiking_neuron"]["detach_reset"] = model["neuron"]["detach_reset"]
    upstream_cfg["spiking_neuron"]["surrogate_fun"] = model["neuron"]["surrogate_fun"]
    upstream_cfg["spiking_neuron"]["spike_norm"] = "BN" if model["norm"]["type"] == "RMSNorm" else model["norm"]["type"]

    upstream_cfg["optimizer"]["lr"] = optimizer["lr"]
    upstream_cfg["optimizer"]["wd"] = optimizer["weight_decay"]
    upstream_cfg["optimizer"]["scheduler"] = optimizer["scheduler"]
    upstream_cfg["optimizer"]["milestones"] = optimizer["milestones"]
    upstream_cfg["optimizer"]["use_amp"] = optimizer["use_amp"]

    upstream_cfg["loss"]["lambda_mod"] = loss["lambda_mod"]
    upstream_cfg["loss"]["lambda_ang"] = loss["lambda_ang"]
    upstream_cfg["loss"]["gamma"] = loss["gamma"]
    upstream_cfg["loss"]["clip_grad"] = loss["clip_grad"]

    upstream_cfg["metrics"]["name"] = metrics["names"]
    upstream_cfg["metrics"]["flow_scaling"] = metrics["flow_scaling"]
    upstream_cfg["metrics"]["mask_events"] = metrics["mask_events"]

    if cfg["dataset"]["name"] == "mvsec":
        upstream_cfg["data"]["path"] = "data/Datasets/MVSEC"
        upstream_cfg["data"]["test_sequence"] = cfg["dataset"].get("test_sequence", "indoor_flying3")
        upstream_cfg["data"]["event_interval"] = cfg["dataset"].get("event_interval", "dt1")
        upstream_cfg["loader"]["resolution"] = cfg["dataset"].get("resolution", [260, 346])
        upstream_cfg["loader"]["crop"] = cfg["dataset"].get("crop", [256, 256])

    return upstream_cfg
