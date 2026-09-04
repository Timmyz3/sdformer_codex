"""Baseline adapter around the upstream SDformerFlow implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.models.modules.normalization.rmsnorm import RMSNorm
from src.models.sdformer.layers import build_upstream_config
from src.models.sdformer.spiking_neurons import resolve_upstream_neuron_type


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _upstream_root(cfg: Dict[str, Any]) -> Path:
    return (_repo_root() / cfg["upstream"]["repo_root"]).resolve()


def ensure_upstream_path(cfg: Dict[str, Any]) -> None:
    upstream_root = str(_upstream_root(cfg))
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)


def replace_layer_norms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, RMSNorm(child.normalized_shape))
        else:
            replace_layer_norms(child)


class SDFormerFlowAdapter(nn.Module):
    """
    Wraps the upstream SDformerFlow model and exposes a stable project interface.

    Input:
        batch["event_voxel"]: `[B, T, H, W]` or `[B, T, 2, H, W]`
    Output:
        dict(flow_pred=`[B, 2, H, W]`, aux={flow_list, token_mask, timestep_mask, stats})
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        ensure_upstream_path(cfg)
        from src.models.registry import build_module

        from models.STSwinNet_SNN.Spiking_STSwinNet import (
            MS_SpikingformerFlowNet,
            MS_SpikingformerFlowNet_en4,
            SpikingformerFlowNet,
        )

        model_map = {
            "SpikingformerFlowNet": SpikingformerFlowNet,
            "MS_SpikingformerFlowNet": MS_SpikingformerFlowNet,
            "MS_SpikingformerFlowNet_en4": MS_SpikingformerFlowNet_en4,
        }

        upstream_cfg = build_upstream_config(cfg, mode="train")
        model_name = upstream_cfg["model"]["name"]
        model_cls = model_map[model_name]
        self.runtime_cfg = upstream_cfg
        self.model = model_cls(upstream_cfg["model"].copy(), upstream_cfg["swin_transformer"].copy())
        self.model.init_weights()

        if cfg["model"]["norm"]["type"] == "RMSNorm":
            replace_layer_norms(self.model)

        self.spike_encoder = build_module(
            "spike_encoding",
            self.cfg["model"]["spike_encoder"]["type"],
            cfg=self.cfg["model"]["spike_encoder"],
        )
        self.preprocess_specs = self._resolve_preprocess_specs()
        self.preprocess_modules = nn.ModuleList()
        for spec in self.preprocess_specs:
            module_kwargs = {key: value for key, value in spec.items() if key not in {"kind", "name"}}
            self.preprocess_modules.append(build_module(spec["kind"], spec["name"], **module_kwargs))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def configure_backend(self) -> None:
        from spikingjelly.activation_based import functional

        functional.reset_net(self.model)
        functional.set_step_mode(self.model, self.runtime_cfg["data"]["step_mode"])
        if self.device.type == "cpu":
            return
        neuron_cls = resolve_upstream_neuron_type(self.runtime_cfg["spiking_neuron"]["neuron_type"])
        functional.set_backend(self.model, "cupy", neuron_cls)

    def _normalize_nonzero(self, chunk: torch.Tensor) -> torch.Tensor:
        norm_mode = self.cfg["model"]["norm_input"]
        non_zero = chunk != 0
        if not torch.any(non_zero):
            return chunk
        if norm_mode == "minmax":
            min_val = torch.min(chunk[non_zero])
            max_val = torch.max(chunk[non_zero])
            if min_val != max_val:
                chunk = chunk.clone()
                chunk[non_zero] = (chunk[non_zero] - min_val) / (max_val - min_val)
        elif norm_mode == "std":
            mean_val = chunk[non_zero].mean()
            std_val = chunk[non_zero].std()
            if std_val > 0:
                chunk = chunk.clone()
                chunk[non_zero] = (chunk[non_zero] - mean_val) / std_val
        return chunk

    def _resolve_preprocess_specs(self) -> List[Dict[str, Any]]:
        explicit_specs = self.cfg["model"].get("plug_in_modules")
        if explicit_specs:
            return [dict(spec) for spec in explicit_specs]

        specs: List[Dict[str, Any]] = []
        token_mixer_cfg = self.cfg["model"].get("token_mixer", {})
        token_mixer_type = token_mixer_cfg.get("type", "identity")
        if token_mixer_type != "identity":
            spec = {key: value for key, value in token_mixer_cfg.items() if key != "type"}
            spec.update({"kind": "token_mixer", "name": token_mixer_type})
            specs.append(spec)

        temporal_cfg = self.cfg["model"].get("temporal", {})
        if temporal_cfg.get("adaptive_t"):
            specs.append(
                {
                    "kind": "sparse_ops",
                    "name": "timestep_budget",
                    "threshold": float(temporal_cfg.get("early_exit_threshold", 0.0)),
                }
            )

        sparsity_cfg = self.cfg["model"].get("sparsity", {})
        if sparsity_cfg.get("window_enabled"):
            specs.append(
                {
                    "kind": "sparse_ops",
                    "name": "window_topk",
                    "keep_ratio": float(sparsity_cfg.get("window_keep_ratio", 1.0)),
                    "window_size": tuple(sparsity_cfg.get("window_size", [8, 8])),
                }
            )
        if sparsity_cfg.get("enabled"):
            specs.append(
                {
                    "kind": "sparse_ops",
                    "name": "structured_token",
                    "keep_ratio": float(sparsity_cfg.get("token_keep_ratio", 1.0)),
                }
            )
        if sparsity_cfg.get("head_enabled"):
            specs.append(
                {
                    "kind": "sparse_ops",
                    "name": "head_group",
                    "keep_ratio": float(sparsity_cfg.get("head_keep_ratio", 1.0)),
                    "num_groups": int(sparsity_cfg.get("head_groups", 1)),
                }
            )
        return specs

    def _merge_mask(self, current: torch.Tensor | None, update: torch.Tensor) -> torch.Tensor:
        if current is None:
            return update
        if current.shape != update.shape:
            return update
        if current.dtype == torch.bool and update.dtype == torch.bool:
            return current & update
        return update

    def _preprocess_input(self, event_voxel: torch.Tensor) -> Dict[str, Any]:
        if event_voxel.dim() == 4:
            pos = torch.relu(event_voxel)
            neg = torch.relu(-event_voxel)
            chunk = torch.stack((pos, neg), dim=2)
        elif event_voxel.dim() == 5:
            chunk = event_voxel
        else:
            raise ValueError(f"unexpected event tensor shape: {tuple(event_voxel.shape)}")

        chunk = self.spike_encoder(chunk)
        aux_masks: Dict[str, torch.Tensor | None] = {
            "timestep_mask": torch.ones(chunk.shape[:2], dtype=torch.bool, device=chunk.device),
            "token_mask": None,
            "window_mask": None,
            "head_mask": None,
        }
        plugin_stats: Dict[str, torch.Tensor] = {}

        for spec, module in zip(self.preprocess_specs, self.preprocess_modules):
            result = module(chunk)
            if isinstance(result, dict):
                chunk = result.get("tensor", chunk)
                for key, value in result.items():
                    if key == "tensor":
                        continue
                    alias = "token_mask" if key == "mask" else key
                    if alias.endswith("_mask"):
                        aux_masks[alias] = self._merge_mask(aux_masks.get(alias), value)
                    else:
                        plugin_stats[f"{spec['name']}.{alias}"] = value
            else:
                chunk = result

        chunk = self._normalize_nonzero(chunk)
        return {
            "chunk": chunk,
            "plugin_masks": aux_masks,
            "plugin_stats": plugin_stats,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        from spikingjelly.activation_based import functional

        functional.reset_net(self.model)
        pre = self._preprocess_input(batch["event_voxel"])
        output = self.model(pre["chunk"])
        flow_list = output["flow"]
        return {
            "flow_pred": flow_list[-1],
            "aux": {
                "flow_list": flow_list,
                "attn": output.get("attn"),
                "plugin_stats": pre["plugin_stats"],
                **pre["plugin_masks"],
            },
        }
