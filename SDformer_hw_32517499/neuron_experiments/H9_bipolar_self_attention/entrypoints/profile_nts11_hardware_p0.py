"""Profile NTS11 hardware-facing H60/ATLIF/skip statistics."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torchvision  # noqa: F401 - keep SDFormerFlow import order
import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
BASELINE_ROOT = REPO_ROOT / "third_party" / "SDformerFlow"
OVERLAY_ROOT = EXP_ROOT / "overlay"

sys.path.insert(0, str(OVERLAY_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")

from configs.parser import YAMLParser  # noqa: E402
from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite  # noqa: E402
from DSEC_dataloader.data_augmentation import CenterCrop, Compose  # noqa: E402
from models.STSwinNet_SNN.Spiking_STSwinNet import (  # noqa: E402
    MS_SpikingformerFlowNet,
    MS_SpikingformerFlowNet_en4,
    SpikingformerFlowNet,
)
from spikingjelly.activation_based import functional, neuron  # noqa: E402
from utils.runtime_backend import configure_snn_backend  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_batch_norm_evaluation(model: torch.nn.Module, policy: str) -> int:
    """Apply the same BatchNorm evaluation policy as the standard evaluator."""
    policy = str(policy or "running").lower()
    if policy not in {"running", "no_running"}:
        raise ValueError(f"unsupported BN evaluation policy: {policy}")
    if policy == "running":
        print("[profile protocol] batch_norm=running", flush=True)
        return 0

    from torch.nn.modules.batchnorm import _BatchNorm

    changed = 0
    for module in model.modules():
        if not isinstance(module, _BatchNorm):
            continue
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None
        changed += 1
    print(f"[profile protocol] batch_norm=no_running modules={changed}", flush=True)
    return changed


def h9_module_counts(model: torch.nn.Module) -> dict[str, int]:
    modules = list(model.modules())
    return {
        "ATLIFTernaryPSN": sum(
            module.__class__.__name__ == "ATLIFTernaryPSN" for module in modules
        ),
        "ShiftmaxAttention": sum(
            hasattr(module, "_h9_shiftmax_cfg") for module in modules
        ),
    }


def validate_h9_load_audit(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, Any] | None:
    audit = getattr(model, "_h9_load_audit", None)
    h9_enabled = bool(
        config.get("atlif_ternary_psn", {}).get("enabled")
        or config.get("bsa_attention", {}).get("enabled")
        or config.get("simple_ternary_psn", {}).get("enabled")
    )
    if not h9_enabled:
        return audit
    if audit is None:
        raise RuntimeError("H9 hardware profile requires a checkpoint load audit")
    if int(audit.get("missing_count", 0)) or int(audit.get("unexpected_count", 0)):
        raise RuntimeError(
            "H9 hardware profile refuses an incomplete checkpoint load: "
            f"missing={audit.get('missing_count')} unexpected={audit.get('unexpected_count')}"
        )
    return audit


def load_config(path: Path) -> tuple[dict[str, Any], torch.device]:
    parser = YAMLParser(str(path))
    config = YAMLParser.combine_entries(parser.config)
    # Match eval_DSEC_flow_SNN.py: historical configs express the dataset path
    # relative to the baseline repository, not to the caller's working directory.
    if not os.path.isabs(config["data"]["path"]):
        baseline_path = os.path.normpath(os.path.join(str(BASELINE_ROOT), config["data"]["path"]))
        repo_path = os.path.normpath(os.path.join(str(REPO_ROOT), config["data"]["path"]))
        if os.path.exists(baseline_path):
            config["data"]["path"] = baseline_path
        elif os.path.exists(repo_path):
            config["data"]["path"] = repo_path
    config["loader"]["batch_size"] = 1
    config["loader"]["shuffle"] = False
    config["loader"]["pin_memory"] = False
    config["loader"]["num_workers"] = 0
    if config["loader"].get("crop") is not None:
        config["swin_transformer"]["input_size"] = [
            int(config["loader"]["crop"][0]),
            int(config["loader"]["crop"][1]),
        ]
    else:
        config["swin_transformer"]["input_size"] = [
            int(config["loader"]["resolution"][0]),
            int(config["loader"]["resolution"][1]),
        ]
    return config, parser.device


def threshold_training_semantics(config: dict[str, Any]) -> dict[str, Any]:
    """Describe ATLIF threshold behavior without conflating two freeze paths."""
    atlif_cfg = config.get("atlif_ternary_psn") or {}
    threshold_modes = {str(atlif_cfg.get("threshold_mode", "direct"))}
    for group in atlif_cfg.get("target_groups") or []:
        threshold_modes.add(
            str(group.get("threshold_mode", atlif_cfg.get("threshold_mode", "direct")))
        )
    freeze_after = atlif_cfg.get("threshold_freeze_after_step")
    gradient_freeze = bool(atlif_cfg.get("freeze_threshold_grad_after_step", False))
    optimizer_cfg = config.get("optimizer") or {}
    param_groups = optimizer_cfg.get("param_groups") or {}
    threshold_lr = (
        param_groups.get("threshold_lr", optimizer_cfg.get("lr"))
        if param_groups.get("enabled", False)
        else optimizer_cfg.get("lr")
    )
    official_only = threshold_modes == {"official_atlif"}
    has_configured_clamp = (
        atlif_cfg.get("min_threshold") is not None
        or atlif_cfg.get("max_threshold") is not None
    )
    return {
        "threshold_modes": sorted(threshold_modes),
        "homeostatic_freeze_after_step": freeze_after,
        "homeostatic_update_frozen_after_boundary": freeze_after is not None,
        "optimizer_gradient_freeze_enabled": gradient_freeze,
        "optimizer_threshold_lr": (
            None if threshold_lr is None else float(threshold_lr)
        ),
        "configured_min_threshold": atlif_cfg.get("min_threshold"),
        "configured_max_threshold": atlif_cfg.get("max_threshold"),
        "official_atlif_runtime_clamp_applied": (
            False if official_only and has_configured_clamp else None
        ),
        "inference_threshold_source": "checkpoint_static_parameter",
        "statement": (
            "threshold_freeze_after_step stops only the separate homeostatic "
            "threshold_update path; optimizer threshold gradients remain active "
            "unless freeze_threshold_grad_after_step is true. official_atlif "
            "does not apply the configured min/max runtime clamp. Inference "
            "uses the threshold parameter stored in the checkpoint."
        ),
    }


def install_overlay_modules(model: torch.nn.Module, config: dict[str, Any]) -> None:
    import models
    import models.STSwinNet_SNN as stsnn

    overlay_models = str(OVERLAY_ROOT / "models")
    overlay_stsnn = str(OVERLAY_ROOT / "models" / "STSwinNet_SNN")
    if overlay_models not in list(models.__path__):
        models.__path__.append(overlay_models)
    if overlay_stsnn not in list(stsnn.__path__):
        stsnn.__path__.append(overlay_stsnn)
    if config.get("atlif_ternary_psn", {}).get("enabled"):
        from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn

        installed = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
        print(f"[profile] installed ATLIF modules: {len(installed)}", flush=True)
    if config.get("bsa_attention", {}).get("enabled"):
        from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, register_shiftmax_pickle_compat

        register_shiftmax_pickle_compat()
        installed = install_shiftmax_attention(model, config.get("bsa_attention"))
        print(f"[profile] installed H60/Shiftmax modules: {len(installed)}", flush=True)


def build_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = eval(config["model"]["name"])(config["model"].copy(), config["swin_transformer"].copy())
    model.to(device)
    model.init_weights()
    install_overlay_modules(model, config)
    from models.STSwinNet_SNN.h9_load_audit import load_checkpoint_with_h9_audit

    remap = config["loader"].get("remap")
    model = load_checkpoint_with_h9_audit(
        str(checkpoint),
        model,
        device,
        config=config,
        remap=remap,
        test=True,
    )
    functional.reset_net(model)
    functional.set_step_mode(model, config["data"]["step_mode"])

    neuron_type = config["model"]["spiking_neuron"]["neuron_type"]
    if neuron_type == "if":
        neurontype = neuron.IFNode
    elif neuron_type == "lif":
        neurontype = neuron.LIFNode
    elif neuron_type == "plif":
        neurontype = neuron.ParametricLIFNode
    elif neuron_type == "psn":
        from models.STSwinNet_SNN.Spiking_submodules import PSN

        neurontype = PSN
    else:
        neurontype = None
    if neurontype is not None:
        configure_snn_backend(model, device, config, neurontype)
    model.eval()
    return model


def iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def tensor_record(tensor: torch.Tensor, *, include_value_stats: bool = False) -> dict[str, Any]:
    shape = list(tensor.shape)
    elements = int(tensor.numel())
    detached = tensor.detach()
    active = int(detached.ne(0).sum().item())
    record = {
        "shape": shape,
        "elements": elements,
        "active": active,
        "density": active / elements if elements else 0.0,
        "bytes_fp16": elements * 2,
        "bytes_fp32": elements * 4,
        "bytes_binary_packed": (elements + 7) // 8,
        "bytes_ternary_packed": (elements * 2 + 7) // 8,
    }
    if not include_value_stats:
        return record

    finite = torch.isfinite(detached) if detached.is_floating_point() else torch.ones_like(detached, dtype=torch.bool)
    finite_count = int(finite.sum().item())
    values = detached[finite]
    if finite_count:
        if values.is_floating_point():
            near_integer = torch.isclose(values, values.round(), atol=1e-6, rtol=0.0)
        else:
            near_integer = torch.ones_like(values, dtype=torch.bool)
        binary01 = values.eq(0) | values.eq(1)
        ternary = binary01 | values.eq(-1)
        record.update({
            "finite_count": finite_count,
            "finite_ratio": finite_count / elements if elements else 0.0,
            "value_min": float(values.min().item()),
            "value_max": float(values.max().item()),
            "value_absmax": float(values.abs().max().item()),
            "value_mean_abs": float(values.abs().float().mean().item()),
            "near_integer_ratio": float(near_integer.float().mean().item()),
            "binary01_ratio": float(binary01.float().mean().item()),
            "ternary_ratio": float(ternary.float().mean().item()),
        })
    else:
        record.update({
            "finite_count": 0,
            "finite_ratio": 0.0,
            "value_min": None,
            "value_max": None,
            "value_absmax": None,
            "value_mean_abs": None,
            "near_integer_ratio": None,
            "binary01_ratio": None,
            "ternary_ratio": None,
        })
    return record


class HardwareProfiler:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        ordered_trace: bool = False,
        dual_line_trace: bool = False,
        bit_trace_writer: Any | None = None,
        dual_line_tile_writer: Any | None = None,
        full_spatial_c4_writer: Any | None = None,
        dual_line_cohort_writer: Any | None = None,
        shift_residual_writer: Any | None = None,
    ):
        self.model = model
        self.ordered_trace = bool(ordered_trace)
        self.dual_line_trace = bool(dual_line_trace)
        self.bit_trace_writer = bit_trace_writer
        self.dual_line_tile_writer = dual_line_tile_writer
        self.full_spatial_c4_writer = full_spatial_c4_writer
        self.dual_line_cohort_writer = dual_line_cohort_writer
        self.shift_residual_writer = shift_residual_writer
        self.handles: list[Any] = []
        self.h60_records: list[dict[str, Any]] = []
        self.activation_records: list[dict[str, Any]] = []
        self.sample_records: list[dict[str, Any]] = []
        self.current_sample = -1
        self.current_sample_key = ""
        self.current_sequence_key = ""
        self.same_sequence_as_previous = False
        self._previous_stage_samples: dict[str, torch.Tensor] = {}
        self.operator_records: dict[str, dict[str, Any]] = {}
        self.execution_records: list[dict[str, Any]] = []
        self.dual_line_records: list[dict[str, Any]] = []
        self._sample_call_index = 0
        self.atlif_records: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "calls": 0,
            "elements": 0,
            "active": 0,
            "pos": 0,
            "neg": 0,
        })

    def attach(self) -> None:
        unet = self.model.sttmultires_unet
        swin = unet.encoders.swin3d
        self.handles.append(swin.patch_embed.register_forward_hook(self._activation_hook("patch_embed", "patch")))
        for stage_idx, layer in enumerate(swin.layers):
            self.handles.append(layer.register_forward_hook(self._stage_hook(stage_idx)))
            if getattr(layer, "downsample", None) is not None:
                self.handles.append(layer.downsample.register_forward_hook(self._activation_hook(f"S{stage_idx}.downsample", "downsample")))
            for block_idx, block in enumerate(layer.swin_blocks):
                self.handles.append(block.register_forward_hook(self._activation_hook(f"S{stage_idx}.B{block_idx}", "swin_block")))
                attn = getattr(block, "attn", None)
                if attn is not None and hasattr(attn, "_h9_shiftmax_cfg"):
                    height, width = (int(value) for value in block.input_resolution)
                    _, window_height, window_width = (
                        int(value) for value in block.window_size
                    )
                    attn._h9_windows_per_sample = (
                        (height + window_height - 1) // window_height
                    ) * ((width + window_width - 1) // window_width)
                    attn._h9_profile_collector = self._h60_collector(f"S{stage_idx}.B{block_idx}.attn")
                    attn._h9_profile_ordered_trace = self.ordered_trace
                    if self.bit_trace_writer is not None:
                        attn._h9_bit_trace_collector = self._h60_bit_trace_collector(
                            f"S{stage_idx}.B{block_idx}.attn"
                        )
        for idx, module in enumerate(unet.resblocks):
            self.handles.append(module.register_forward_hook(self._activation_hook(f"resblock{idx}", "resblock")))
        for idx, module in enumerate(unet.decoders):
            self.handles.append(module.register_forward_hook(self._activation_hook(f"decoder{idx}", "decoder")))
        for idx, module in enumerate(unet.preds):
            self.handles.append(module.register_forward_hook(self._activation_hook(f"pred{idx}", "prediction")))
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "ATLIFTernaryPSN":
                self.handles.append(module.register_forward_hook(self._atlif_hook(name)))
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)):
                self.handles.append(module.register_forward_hook(self._operator_hook(name)))

    def close(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "_h9_profile_collector"):
                delattr(module, "_h9_profile_collector")
            if hasattr(module, "_h9_profile_ordered_trace"):
                delattr(module, "_h9_profile_ordered_trace")
            if hasattr(module, "_h9_windows_per_sample"):
                delattr(module, "_h9_windows_per_sample")
            if hasattr(module, "_h9_bit_trace_collector"):
                delattr(module, "_h9_bit_trace_collector")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _h60_collector(self, name: str):
        def collect(module: torch.nn.Module, stats: dict[str, Any]) -> None:
            stats = dict(stats)
            stats["name"] = name
            stats["sample_id"] = self.current_sample
            self.h60_records.append(stats)
            self._record_execution(
                kind="attention",
                name=name,
                windows=int(getattr(module, "_h9_windows_per_sample", 0)),
                stage=stats.get("stage", ""),
                pair_total=stats.get("pair_total", 0),
                token_total=stats.get("token_total", 0),
            )

        return collect

    def _h60_bit_trace_collector(self, name: str):
        def collect(
            module: torch.nn.Module,
            *,
            q_orig: torch.Tensor,
            k_orig: torch.Tensor,
            gate: torch.Tensor,
        ) -> None:
            self.bit_trace_writer.capture(
                name=name,
                sample_id=self.current_sample,
                sample_key=self.current_sample_key,
                module=module,
                q_orig=q_orig,
                k_orig=k_orig,
                gate=gate,
            )

        return collect

    def _activation_hook(self, name: str, kind: str):
        def hook(_module: torch.nn.Module, _inp: Any, out: Any) -> None:
            tensors = list(iter_tensors(out))
            if not tensors:
                return
            rec = tensor_record(tensors[0])
            rec.update({"name": name, "kind": kind})
            self.activation_records.append(rec)

        return hook

    def begin_sample(self, sample_id: int, *, sample_key: str = "", sequence_key: str = "") -> None:
        previous_sequence = self.current_sequence_key
        self.current_sample = int(sample_id)
        self.current_sample_key = str(sample_key)
        self.current_sequence_key = str(sequence_key)
        self._sample_call_index = 0
        self.same_sequence_as_previous = bool(
            self.current_sample > 0
            and self.current_sequence_key
            and self.current_sequence_key == previous_sequence
        )
        if not self.same_sequence_as_previous:
            self._previous_stage_samples.clear()

    def record_sample(
        self,
        *,
        chunk: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor,
        prediction: torch.Tensor,
        flow_scaling: float,
    ) -> None:
        with torch.no_grad():
            active = chunk.detach().ne(0)
            spatial_reduce = tuple(range(1, max(1, active.ndim - 2)))
            active_pixels = active.any(dim=spatial_reduce) if spatial_reduce else active
            valid = mask.detach().bool()
            if valid.ndim == 4:
                valid = valid[:, 0]
            truth = label.detach().float()
            pred = prediction.detach().float() * float(flow_scaling)
            magnitude = truth.square().sum(dim=1).sqrt()
            error = (pred - truth).square().sum(dim=1).sqrt()
            valid_magnitude = magnitude[valid]
            valid_error = error[valid]

            def quantile(data: torch.Tensor, q: float) -> float:
                return float(torch.quantile(data, q).item()) if data.numel() else 0.0

            dx = truth[:, :, :, 1:] - truth[:, :, :, :-1]
            dy = truth[:, :, 1:, :] - truth[:, :, :-1, :]
            dx_valid = valid[:, :, 1:] & valid[:, :, :-1]
            dy_valid = valid[:, 1:, :] & valid[:, :-1, :]
            dx_mag = dx.square().sum(dim=1).sqrt()[dx_valid]
            dy_mag = dy.square().sum(dim=1).sqrt()[dy_valid]
            gradient_sum = float(dx_mag.sum().item() + dy_mag.sum().item())
            gradient_count = int(dx_mag.numel() + dy_mag.numel())
            u = truth[:, 0][valid]
            v = truth[:, 1][valid]
            rec: dict[str, Any] = {
                "sample_id": self.current_sample,
                "sample_key": self.current_sample_key,
                "sequence_key": self.current_sequence_key,
                "same_sequence_as_previous": self.same_sequence_as_previous,
                "input_elements": int(active.numel()),
                "input_events": int(active.sum().item()),
                "input_event_density": float(active.float().mean().item()),
                "input_active_pixels": int(active_pixels.sum().item()),
                "input_active_pixel_ratio": float(active_pixels.float().mean().item()),
                "valid_flow_pixels": int(valid.sum().item()),
                "label_flow_mag_mean": float(valid_magnitude.mean().item()) if valid_magnitude.numel() else 0.0,
                "label_flow_mag_p50": quantile(valid_magnitude, 0.50),
                "label_flow_mag_p90": quantile(valid_magnitude, 0.90),
                "label_flow_mag_max": float(valid_magnitude.max().item()) if valid_magnitude.numel() else 0.0,
                "label_flow_nearzero_ratio": float((valid_magnitude < 0.5).float().mean().item()) if valid_magnitude.numel() else 0.0,
                "label_flow_u_pos_ratio": float((u > 0).float().mean().item()) if u.numel() else 0.0,
                "label_flow_v_pos_ratio": float((v > 0).float().mean().item()) if v.numel() else 0.0,
                "label_flow_gradient_mean": gradient_sum / gradient_count if gradient_count else 0.0,
                "sample_aee": float(valid_error.mean().item()) if valid_error.numel() else 0.0,
            }
            records = [row for row in self.h60_records if int(row.get("sample_id", -1)) == self.current_sample]
            pair_total = sum(int(row.get("pair_total", 0)) for row in records)
            pair_empty = sum(int(row.get("pair_empty", 0)) for row in records)
            token_total = sum(int(row.get("token_total", 0)) for row in records)
            token_kzero = sum(int(row.get("token_kzero", 0)) for row in records)
            events = 0
            union_lanes = 0
            for row in records:
                events += sum(index * int(count) for index, count in enumerate(row.get("four_vector_event_histogram", [])))
                union_lanes += sum(index * int(count) for index, count in enumerate(row.get("four_vector_union_histogram", [])))
            rec.update({
                "pair_total": pair_total,
                "pair_empty_ratio": pair_empty / pair_total if pair_total else 0.0,
                "mean_events_per_pair": events / pair_total if pair_total else 0.0,
                "mean_union_lanes_per_pair": union_lanes / pair_total if pair_total else 0.0,
                "token_kzero_ratio": token_kzero / token_total if token_total else 0.0,
            })
            for stage in range(4):
                stage_rows = [row for row in records if int(row.get("stage", -1)) == stage]
                stage_pairs = sum(int(row.get("pair_total", 0)) for row in stage_rows)
                stage_empty = sum(int(row.get("pair_empty", 0)) for row in stage_rows)
                stage_events = sum(
                    index * int(count)
                    for row in stage_rows
                    for index, count in enumerate(row.get("four_vector_event_histogram", []))
                )
                rec[f"s{stage}_pair_empty_ratio"] = stage_empty / stage_pairs if stage_pairs else 0.0
                rec[f"s{stage}_mean_events_per_pair"] = stage_events / stage_pairs if stage_pairs else 0.0
            self.sample_records.append(rec)

    def _stage_hook(self, stage_idx: int):
        def hook(_module: torch.nn.Module, _inp: Any, out: Any) -> None:
            if not isinstance(out, tuple) or len(out) != 2:
                return
            x_out, skip = out
            # Stage boundaries cover every long skip while avoiding expensive value
            # reductions on all 12 block outputs during profile100.
            skip_rec = tensor_record(skip, include_value_stats=True)
            skip_kind = "stage_skip_predownsample" if stage_idx < 3 else "stage_skip_final"
            skip_rec.update({"name": f"S{stage_idx}.skip", "kind": skip_kind})
            self._record_cross_sample_stage_delta(f"S{stage_idx}.skip", skip, skip_rec)
            self.activation_records.append(skip_rec)
            out_rec = tensor_record(x_out, include_value_stats=True)
            out_rec.update({"name": f"S{stage_idx}.x_out", "kind": "stage_x_out"})
            self._record_cross_sample_stage_delta(f"S{stage_idx}.x_out", x_out, out_rec)
            self.activation_records.append(out_rec)

        return hook

    def _record_cross_sample_stage_delta(
        self,
        name: str,
        tensor: torch.Tensor,
        rec: dict[str, Any],
    ) -> None:
        flattened = tensor.detach().float().reshape(-1)
        max_values = 1 << 20
        stride = max(1, (flattened.numel() + max_values - 1) // max_values)
        sampled = flattened[::stride][:max_values].cpu()
        rec.update({
            "sample_id": self.current_sample,
            "sample_key": self.current_sample_key,
            "sequence_key": self.current_sequence_key,
            "cross_sample_values": int(sampled.numel()),
            "cross_sample_comparable": False,
        })
        previous = self._previous_stage_samples.get(name)
        if self.same_sequence_as_previous and previous is not None and previous.shape == sampled.shape:
            delta = sampled - previous
            current_class = torch.sign(sampled)
            previous_class = torch.sign(previous)
            denominator = max(float(sampled.abs().mean().item()), 1.0 / 256.0)
            rec.update({
                "cross_sample_comparable": True,
                "cross_sample_exact_equal_ratio": float(sampled.eq(previous).float().mean().item()),
                "cross_sample_active_xor_ratio": float(sampled.ne(0).ne(previous.ne(0)).float().mean().item()),
                "cross_sample_sign_class_change_ratio": float(current_class.ne(previous_class).float().mean().item()),
                "cross_sample_mean_abs_delta": float(delta.abs().mean().item()),
                "cross_sample_normalized_mean_abs_delta": float(delta.abs().mean().item()) / denominator,
            })
        self._previous_stage_samples[name] = sampled

    def _atlif_hook(self, name: str):
        def hook(module: torch.nn.Module, inp: Any, out: Any) -> None:
            tensors = list(iter_tensors(out))
            if not tensors:
                return
            t = tensors[0].detach()
            rec = self.atlif_records[name]
            if rec["calls"] == 0:
                rec["deployment_dead_result"] = name.endswith(".attn.attn_sn.spiking_neuron")
                inputs = list(iter_tensors(inp))
                if inputs:
                    input_stats = tensor_record(inputs[0], include_value_stats=True)
                    rec.update({f"input_first_{key}": value for key, value in input_stats.items()})
                    self._record_atlif_quant_sample(module, inputs[0], t, rec)
            rec["calls"] += 1
            rec["elements"] += int(t.numel())
            rec["active"] += int(t.ne(0).sum().item())
            rec["pos"] += int(t.gt(0).sum().item())
            rec["neg"] += int(t.lt(0).sum().item())
            rec["output_mode"] = str(getattr(module, "output_mode", "unknown"))
            rec["threshold_mode"] = str(getattr(module, "threshold_mode", "unknown"))
            inputs = list(iter_tensors(inp))
            temporal = int(inputs[0].shape[0]) if inputs else 0
            self._record_execution(
                kind="atlif",
                name=name,
                input_elements=int(inputs[0].numel()) if inputs else 0,
                output_elements=int(t.numel()),
                temporal_steps=temporal,
                dense_macs=int(t.numel()) * temporal,
                input_shape=list(inputs[0].shape) if inputs else [],
                output_shape=list(t.shape),
            )

        return hook

    def _operator_hook(self, name: str):
        def hook(module: torch.nn.Module, inp: Any, out: Any) -> None:
            inputs = list(iter_tensors(inp))
            outputs = list(iter_tensors(out))
            if not inputs or not outputs:
                return
            input_tensor = inputs[0].detach()
            output_tensor = outputs[0].detach()
            if isinstance(module, torch.nn.Linear):
                macs_per_output = int(module.in_features)
            elif isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                kernel_elements = math.prod(int(value) for value in module.kernel_size)
                macs_per_output = int(module.in_channels // module.groups) * kernel_elements
            else:
                return
            dense_macs = int(output_tensor.numel()) * macs_per_output
            input_elements = int(input_tensor.numel())
            input_active = int(input_tensor.ne(0).sum().item())
            row = self.operator_records.setdefault(name, {
                "name": name,
                "operator": module.__class__.__name__,
                "scope": self._operator_scope(name),
                "calls": 0,
                "input_elements": 0,
                "input_active": 0,
                "output_elements": 0,
                "dense_macs": 0,
                "activity_weighted_macs_proxy": 0.0,
                "weight_elements": int(module.weight.numel()),
                "input_shape_first": list(input_tensor.shape),
                "output_shape_first": list(output_tensor.shape),
            })
            if row["calls"] == 0:
                flattened = input_tensor.float().reshape(-1)
                max_values = 1 << 20
                stride = max(1, (flattened.numel() + max_values - 1) // max_values)
                sample = flattened[::stride][:max_values]
                stats = tensor_record(sample, include_value_stats=True)
                row.update({f"input_sample_{key}": value for key, value in stats.items()})
            row["calls"] += 1
            row["input_elements"] += input_elements
            row["input_active"] += input_active
            row["output_elements"] += int(output_tensor.numel())
            row["dense_macs"] += dense_macs
            row["activity_weighted_macs_proxy"] += (
                dense_macs * input_active / input_elements if input_elements else 0.0
            )
            if self.dual_line_trace and isinstance(
                module, (torch.nn.Linear, torch.nn.Conv2d)
            ):
                from h67_dual_line_trace import profile_operator_temporal_work

                operator_call_index = int(row["calls"]) - 1
                temporal_work = profile_operator_temporal_work(
                    module, input_tensor, temporal_steps=10
                )
                for work in temporal_work:
                    self.dual_line_records.append({
                        "sample_id": self.current_sample,
                        "sample_key": self.current_sample_key,
                        "sequence_key": self.current_sequence_key,
                        "name": name,
                        "operator": module.__class__.__name__,
                        "scope": self._operator_scope(name),
                        "operator_call_index": operator_call_index,
                        **work,
                    })
                if self.dual_line_tile_writer is not None:
                    self.dual_line_tile_writer.record_operator(
                        module,
                        input_tensor,
                        name=name,
                        sample_id=self.current_sample,
                        sample_key=self.current_sample_key,
                        sequence_key=self.current_sequence_key,
                        operator_call_index=operator_call_index,
                        temporal_steps=10,
                    )
                if self.dual_line_cohort_writer is not None:
                    self.dual_line_cohort_writer.record_operator(
                        module,
                        input_tensor,
                        reference_work=temporal_work,
                        name=name,
                        sample_id=self.current_sample,
                        sample_key=self.current_sample_key,
                        sequence_key=self.current_sequence_key,
                        operator_call_index=operator_call_index,
                        temporal_steps=10,
                    )
                if self.shift_residual_writer is not None:
                    self.shift_residual_writer.record_operator(
                        module,
                        input_tensor,
                        name=name,
                        sample_id=self.current_sample,
                        sample_key=self.current_sample_key,
                        sequence_key=self.current_sequence_key,
                        operator_call_index=operator_call_index,
                        temporal_steps=10,
                    )
                if self.full_spatial_c4_writer is not None:
                    self.full_spatial_c4_writer.record_operator(
                        module,
                        input_tensor,
                        name=name,
                        sample_id=self.current_sample,
                        sample_key=self.current_sample_key,
                        sequence_key=self.current_sequence_key,
                        operator_call_index=operator_call_index,
                        temporal_steps=10,
                    )
            self._record_execution(
                kind="operator",
                name=name,
                operator=module.__class__.__name__,
                scope=self._operator_scope(name),
                input_elements=input_elements,
                input_active=input_active,
                output_elements=int(output_tensor.numel()),
                dense_macs=dense_macs,
                input_shape=list(input_tensor.shape),
                output_shape=list(output_tensor.shape),
            )

        return hook

    def _record_execution(self, *, kind: str, name: str, **payload: Any) -> None:
        if not self.ordered_trace:
            return
        self.execution_records.append({
            "sample_id": self.current_sample,
            "sample_key": self.current_sample_key,
            "sequence_key": self.current_sequence_key,
            "call_index": self._sample_call_index,
            "kind": kind,
            "name": name,
            **payload,
        })
        self._sample_call_index += 1

    @staticmethod
    def _operator_scope(name: str) -> str:
        if ".encoders." in name:
            return "encoder"
        if ".decoders." in name:
            return "decoder"
        if ".preds." in name:
            return "prediction"
        if ".resblocks." in name:
            return "bottleneck"
        return "other"

    @staticmethod
    def _record_atlif_quant_sample(
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        rec: dict[str, Any],
    ) -> None:
        if not all(hasattr(module, field) for field in ("weight", "bias", "thresh")):
            return
        temporal = int(input_tensor.shape[0])
        flattened = input_tensor.detach().float().reshape(temporal, -1)
        output = output_tensor.detach().reshape(temporal, -1).ne(0)
        max_vectors = 4096
        stride = max(1, (flattened.shape[1] + max_vectors - 1) // max_vectors)
        flattened = flattened[:, ::stride][:, :max_vectors]
        output = output[:, ::stride][:, :max_vectors]
        weight = module.weight.detach().float()
        bias = module.bias.detach().float()
        threshold = module.thresh.detach().float()
        h_seq = torch.addmm(bias, weight, flattened)
        reference = h_seq.ge(threshold)
        rec.update({
            "temporal_steps": temporal,
            "parameter_entries": int(weight.numel() + bias.numel() + threshold.numel()),
            "quant_sample_events": int(reference.numel()),
            "recomputed_reference_mismatch": int(reference.ne(output).sum().item()),
            "margin_abs_le_1_128": int((h_seq.sub(threshold).abs() <= 1.0 / 128).sum().item()),
            "margin_abs_le_1_64": int((h_seq.sub(threshold).abs() <= 1.0 / 64).sum().item()),
            "margin_abs_le_1_32": int((h_seq.sub(threshold).abs() <= 1.0 / 32).sum().item()),
            "margin_abs_le_1_16": int((h_seq.sub(threshold).abs() <= 1.0 / 16).sum().item()),
        })

        def quantize(value: torch.Tensor, bits: int, fractional: int) -> torch.Tensor:
            scale = float(1 << fractional)
            lower = -(1 << (bits - 1))
            upper = (1 << (bits - 1)) - 1
            return torch.round(value * scale).clamp(lower, upper) / scale

        for bits, fractional in ((4, 2), (6, 4), (8, 6)):
            quant_h = torch.addmm(
                quantize(bias, bits, fractional),
                quantize(weight, bits, fractional),
                flattened,
            )
            quant_reference = quant_h.ge(quantize(threshold, bits, fractional))
            rec[f"parameter_q{bits}_event_mismatch"] = int(quant_reference.ne(reference).sum().item())

    @staticmethod
    def _aggregate_numeric(records: list[dict[str, Any]], keys: list[str], group_key: str) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for rec in records:
            group = str(rec[group_key])
            row = grouped.setdefault(group, {"group": group, "calls": 0})
            row["calls"] += 1
            for key in keys:
                if key in rec:
                    row[key] = row.get(key, 0.0) + float(rec[key])
        for row in grouped.values():
            calls = max(int(row["calls"]), 1)
            for key in keys:
                if key in row:
                    row[key] /= calls
        return sorted(grouped.values(), key=lambda row: row["group"])

    def summary(self) -> dict[str, Any]:
        def aggregate_histogram(key: str) -> list[int]:
            size = max((len(row.get(key, [])) for row in self.h60_records), default=0)
            total = [0] * size
            for row in self.h60_records:
                values = row.get(key, [])
                for index, value in enumerate(values):
                    total[index] += int(value)
            return total

        h60_keys = [
            "tx_mean", "tx_std", "sc_mean", "sc_std", "fused_mean", "fused_std",
            "gate_entropy_mean", "top1_mass_mean", "top4_mass_mean", "effective_tokens_mean",
            "q_active_density", "k_active_density", "q_token_active_density", "k_token_active_density",
            "zaf_kzero_token_ratio", "zaf_active_entries_mean", "zaf_fold_classes_mean",
            "q_temporal_toggle_density", "k_temporal_toggle_density", "qk_temporal_update_density",
            "score_clip_ratio",
            "ttb1_empty_ratio", "ttb2_empty_ratio", "ttb4_empty_ratio",
            "ttb1_low_density_ratio", "ttb2_low_density_ratio", "ttb4_low_density_ratio",
        ]
        act_by_kind: dict[str, dict[str, Any]] = {}
        for rec in self.activation_records:
            row = act_by_kind.setdefault(rec["kind"], {"kind": rec["kind"], "calls": 0, "elements": 0, "active": 0, "bytes_fp16": 0, "bytes_ternary_packed": 0})
            row["calls"] += 1
            row["elements"] += rec["elements"]
            row["active"] += rec["active"]
            row["bytes_fp16"] += rec["bytes_fp16"]
            row["bytes_ternary_packed"] += rec["bytes_ternary_packed"]
        for row in act_by_kind.values():
            row["density"] = row["active"] / row["elements"] if row["elements"] else 0.0
        cross_sample_fields = (
            "cross_sample_exact_equal_ratio",
            "cross_sample_active_xor_ratio",
            "cross_sample_sign_class_change_ratio",
            "cross_sample_mean_abs_delta",
            "cross_sample_normalized_mean_abs_delta",
        )
        cross_sample_by_stage: dict[str, dict[str, Any]] = {}
        for rec in self.activation_records:
            if not rec.get("cross_sample_comparable", False):
                continue
            name = str(rec["name"])
            values = int(rec.get("cross_sample_values", 0))
            row = cross_sample_by_stage.setdefault(
                name,
                {"name": name, "comparable_pairs": 0, "sampled_values": 0},
            )
            row["comparable_pairs"] += 1
            row["sampled_values"] += values
            for field in cross_sample_fields:
                row[field] = row.get(field, 0.0) + float(rec[field]) * values
        for row in cross_sample_by_stage.values():
            values = max(int(row["sampled_values"]), 1)
            for field in cross_sample_fields:
                row[field] /= values
        atlif_rows = []
        for name, rec in sorted(self.atlif_records.items()):
            row = dict(rec)
            row["name"] = name
            row["activity"] = row["active"] / row["elements"] if row["elements"] else 0.0
            row["pos_rate"] = row["pos"] / row["elements"] if row["elements"] else 0.0
            row["neg_rate"] = row["neg"] / row["elements"] if row["elements"] else 0.0
            atlif_rows.append(row)
        operator_rows = []
        operator_by_scope: dict[str, dict[str, Any]] = {}
        for name, rec in sorted(self.operator_records.items()):
            row = dict(rec)
            row["input_activity"] = (
                row["input_active"] / row["input_elements"] if row["input_elements"] else 0.0
            )
            operator_rows.append(row)
            scope = str(row["scope"])
            scope_row = operator_by_scope.setdefault(scope, {
                "scope": scope,
                "modules": 0,
                "calls": 0,
                "dense_macs": 0,
                "activity_weighted_macs_proxy": 0.0,
                "input_elements": 0,
                "input_active": 0,
            })
            scope_row["modules"] += 1
            for field in ("calls", "dense_macs", "input_elements", "input_active"):
                scope_row[field] += int(row[field])
            scope_row["activity_weighted_macs_proxy"] += float(row["activity_weighted_macs_proxy"])
        for row in operator_by_scope.values():
            row["input_activity"] = (
                row["input_active"] / row["input_elements"] if row["input_elements"] else 0.0
            )
        temporal_lane_elements = sum(int(row.get("temporal_lane_elements", 0)) for row in self.h60_records)
        q_toggle_elements = sum(int(row.get("q_temporal_toggle_elements", 0)) for row in self.h60_records)
        k_toggle_elements = sum(int(row.get("k_temporal_toggle_elements", 0)) for row in self.h60_records)
        update_elements = sum(int(row.get("qk_temporal_update_elements", 0)) for row in self.h60_records)
        score_quant_total = sum(int(row.get("score_quant_total", 0)) for row in self.h60_records)
        score_clip_low = sum(int(row.get("score_clip_low", 0)) for row in self.h60_records)
        score_clip_high = sum(int(row.get("score_clip_high", 0)) for row in self.h60_records)
        locality_keys = [
            "delta_token_heads", "delta_zero_update_token_heads", "delta_changed_token_heads",
            "delta_changed_token_runs", "delta_update_count_0", "delta_update_count_1",
            "delta_update_count_2", "delta_update_count_3_4", "delta_update_count_5_8",
            "delta_update_count_9_16", "delta_update_count_17_plus",
            "delta_bundle4_total", "delta_bundle4_empty",
            "delta_bundle8_total", "delta_bundle8_empty",
        ]
        for threshold in (2, 4, 8, 12, 16):
            locality_keys.extend([
                f"delta_active_le{threshold}",
                f"delta_active_lane_sum_le{threshold}",
            ])
        locality = {
            key: sum(int(row.get(key, 0)) for row in self.h60_records)
            for key in locality_keys
        }
        token_heads = locality["delta_token_heads"]
        changed_tokens = locality["delta_changed_token_heads"]
        changed_runs = locality["delta_changed_token_runs"]
        delta_ttx = {
            "temporal_lane_elements": temporal_lane_elements,
            "q_temporal_toggle_elements": q_toggle_elements,
            "k_temporal_toggle_elements": k_toggle_elements,
            "qk_temporal_update_elements": update_elements,
            "q_temporal_toggle_density": q_toggle_elements / temporal_lane_elements if temporal_lane_elements else 0.0,
            "k_temporal_toggle_density": k_toggle_elements / temporal_lane_elements if temporal_lane_elements else 0.0,
            "qk_temporal_update_density": update_elements / temporal_lane_elements if temporal_lane_elements else 0.0,
            "t1_ideal_lane_skip_ratio": 1.0 - update_elements / temporal_lane_elements if temporal_lane_elements else 0.0,
            "full_t2_ideal_compare_reduction": 0.5 * (1.0 - update_elements / temporal_lane_elements) if temporal_lane_elements else 0.0,
            **locality,
            "delta_update_histogram": aggregate_histogram("delta_update_histogram"),
            "delta_zero_update_token_ratio": locality["delta_zero_update_token_heads"] / token_heads if token_heads else 0.0,
            "delta_mean_changed_run_length": changed_tokens / changed_runs if changed_runs else 0.0,
            "delta_bundle4_empty_ratio": locality["delta_bundle4_empty"] / locality["delta_bundle4_total"] if locality["delta_bundle4_total"] else 0.0,
            "delta_bundle8_empty_ratio": locality["delta_bundle8_empty"] / locality["delta_bundle8_total"] if locality["delta_bundle8_total"] else 0.0,
        }

        ttb_rows = []
        for bundle in (1, 2, 4, 8):
            prefix = f"ttb_tok{bundle}"
            total = sum(int(row.get(f"{prefix}_total", 0)) for row in self.h60_records)
            active_lanes = sum(int(row.get(f"{prefix}_active_lanes", 0)) for row in self.h60_records)
            capacity_lanes = sum(int(row.get(f"{prefix}_capacity_lanes", 0)) for row in self.h60_records)
            item = {
                "token_bundle": bundle,
                "time_slices": 2,
                "bundles": total,
                "activity_density": active_lanes / capacity_lanes if capacity_lanes else 0.0,
            }
            for name in ("empty", "kzero", "motion_zero"):
                count = sum(int(row.get(f"{prefix}_{name}", 0)) for row in self.h60_records)
                item[f"{name}_count"] = count
                item[f"{name}_ratio"] = count / total if total else 0.0
            item["active_histogram"] = aggregate_histogram(f"{prefix}_active_histogram")
            for threshold in (2, 4, 8, 12, 16, 32):
                count = sum(
                    int(row.get(f"{prefix}_active_le{threshold}", 0))
                    for row in self.h60_records
                )
                item[f"active_1_{threshold}_count"] = count
                item[f"active_1_{threshold}_ratio"] = count / total if total else 0.0
                item[f"active_lane_sum_1_{threshold}"] = sum(
                    int(row.get(f"{prefix}_active_lane_sum_le{threshold}", 0))
                    for row in self.h60_records
                )
            ttb_rows.append(item)

        pair_count_keys = (
            "pair_total", "pair_empty", "pair_motion_zero", "pair_update_zero",
            "pair_score_equal_ttx", "pair_score_equal_h67", "pair_kzero_both", "pair_kzero_one",
            "pair_both_active", "pair_kzero_same_class_ttx", "pair_kzero_same_class_h67",
            "pair_kzero_dual_class_ttx", "pair_kzero_dual_class_h67",
            "k_temporal_baseline_reads", "k_temporal_union_reads",
            "k_temporal_intersection_reuse",
            "projection_baseline_active_lanes", "projection_class_channel_terms_ttx",
            "projection_class_channel_terms_h67", "projection_gate_class_channel_terms_deploy",
            "projection_h67_factor_class_segments",
            "projection_h67_factor_class_lane_segments",
            "projection_gate_class_channel_max_fanout_deploy", "projection_gate_q17_out_of_range",
            "projection_gate_ppdi_delivery_exact",
            "row_active_projection_gate_classes_sum_deploy",
            "token_total", "token_kzero", "row_total", "row_all_occupied_classes_sum_ttx",
            "row_all_occupied_classes_sum_h67", "row_kzero_fold_classes_sum_ttx",
            "row_kzero_fold_classes_sum_h67",
            "row_active_projection_classes_sum_ttx", "row_active_projection_classes_sum_h67",
            "spatial_row_total", "spatial_union_tokens", "spatial_persistent_tokens",
            "spatial_changed_tokens", "spatial_horizontal_adjacent_active",
            "spatial_horizontal_adjacent_total", "spatial_vertical_adjacent_active",
            "spatial_vertical_adjacent_total", "spatial_diag_down_adjacent_active",
            "spatial_diag_down_adjacent_total", "spatial_diag_up_adjacent_active",
            "spatial_diag_up_adjacent_total",
        )
        pair_count_keys += tuple(
            f"pair_score_equal_h67_qf{bits}" for bits in (5, 6, 7, 8)
        )
        pair_count_keys += tuple(
            f"spatial_bank{banks}_{mapping}_cycles_sum"
            for banks in (4, 8)
            for mapping in ("rowmajor", "diagonal", "xor")
        )
        pair_count_keys += tuple(
            f"projection_gate_multicast_delivery_m{width}"
            for width in (1, 2, 4, 8, 16)
        )
        pair_count_keys += tuple(
            f"projection_gate_group_terms_g{group_windows}"
            for group_windows in (1, 2, 4, 8, 16)
        )
        pair_count_keys += tuple(
            f"projection_gate_group_{metric}_g{group_windows}"
            for metric in (
                "active_lanes",
                "active_classes",
                "window_count",
                "ppdi_delivery",
            )
            for group_windows in (1, 2, 4, 8, 16)
        )
        pair_count_keys += tuple(
            f"projection_gate_group_delivery_g{group_windows}_m{width}"
            for group_windows in (1, 2, 4, 8, 16)
            for width in (1, 2, 4, 8, 16)
        )
        temporal_pairs = {
            key: sum(int(row.get(key, 0)) for row in self.h60_records)
            for key in pair_count_keys
        }
        for group_windows in (1, 2, 4, 8, 16):
            key = f"projection_gate_group_max_fanout_g{group_windows}"
            temporal_pairs[key] = max(
                (int(row.get(key, 0)) for row in self.h60_records), default=0
            )
        for key in (
            "q_count_histogram", "k_count_histogram", "overlap_histogram",
            "same_zero_histogram", "motion_histogram", "update_histogram",
            "k_temporal_intersection_histogram", "k_temporal_union_histogram",
            "four_vector_event_histogram", "four_vector_union_histogram",
            "ttx_score_q7_histogram", "h67_score_q7_histogram",
            "row_all_occupied_classes_ttx_histogram", "row_all_occupied_classes_h67_histogram",
            "row_kzero_fold_classes_ttx_histogram", "row_kzero_fold_classes_h67_histogram",
            "row_active_projection_classes_ttx_histogram",
            "row_active_projection_classes_h67_histogram",
            "projection_h67_factor_class_segments_histogram",
            "projection_h67_factor_class_lane_segments_histogram",
            "row_score_span_ttx_histogram", "row_score_span_h67_histogram",
            "projection_gate_class_channel_term_histogram",
            "projection_active_lane_gate_q17_histogram",
            "spatial_union_count_histogram",
        ):
            temporal_pairs[key] = aggregate_histogram(key)
        for banks in (4, 8):
            for mapping in ("rowmajor", "diagonal", "xor"):
                key = f"spatial_bank{banks}_{mapping}_cycles_histogram"
                temporal_pairs[key] = aggregate_histogram(key)
        pair_total = temporal_pairs["pair_total"]
        token_total = temporal_pairs["token_total"]
        row_total = temporal_pairs["row_total"]
        temporal_pairs.update({
            "pair_empty_ratio": temporal_pairs["pair_empty"] / pair_total if pair_total else 0.0,
            "pair_motion_zero_ratio": temporal_pairs["pair_motion_zero"] / pair_total if pair_total else 0.0,
            "pair_update_zero_ratio": temporal_pairs["pair_update_zero"] / pair_total if pair_total else 0.0,
            "pair_score_equal_ttx_ratio": temporal_pairs["pair_score_equal_ttx"] / pair_total if pair_total else 0.0,
            "pair_score_equal_h67_ratio": temporal_pairs["pair_score_equal_h67"] / pair_total if pair_total else 0.0,
            **{
                f"pair_score_equal_h67_qf{bits}_ratio": (
                    temporal_pairs[f"pair_score_equal_h67_qf{bits}"] / pair_total
                    if pair_total
                    else 0.0
                )
                for bits in (5, 6, 7, 8)
            },
            "pair_kzero_both_ratio": temporal_pairs["pair_kzero_both"] / pair_total if pair_total else 0.0,
            "pair_kzero_one_ratio": temporal_pairs["pair_kzero_one"] / pair_total if pair_total else 0.0,
            "pair_both_active_ratio": temporal_pairs["pair_both_active"] / pair_total if pair_total else 0.0,
            "pair_kzero_same_class_ttx_ratio": temporal_pairs["pair_kzero_same_class_ttx"] / pair_total if pair_total else 0.0,
            "pair_kzero_same_class_h67_ratio": temporal_pairs["pair_kzero_same_class_h67"] / pair_total if pair_total else 0.0,
            "pair_kzero_dual_class_ttx_ratio": temporal_pairs["pair_kzero_dual_class_ttx"] / pair_total if pair_total else 0.0,
            "pair_kzero_dual_class_h67_ratio": temporal_pairs["pair_kzero_dual_class_h67"] / pair_total if pair_total else 0.0,
            "token_kzero_ratio": temporal_pairs["token_kzero"] / token_total if token_total else 0.0,
            "k_temporal_union_read_ratio": (
                temporal_pairs["k_temporal_union_reads"] /
                temporal_pairs["k_temporal_baseline_reads"]
                if temporal_pairs["k_temporal_baseline_reads"] else 0.0
            ),
            "k_temporal_exact_reuse_ratio": (
                temporal_pairs["k_temporal_intersection_reuse"] /
                temporal_pairs["k_temporal_baseline_reads"]
                if temporal_pairs["k_temporal_baseline_reads"] else 0.0
            ),
            "projection_class_channel_ratio_ttx": (
                temporal_pairs["projection_class_channel_terms_ttx"] /
                temporal_pairs["projection_baseline_active_lanes"]
                if temporal_pairs["projection_baseline_active_lanes"] else 0.0
            ),
            "projection_class_channel_ratio_h67": (
                temporal_pairs["projection_class_channel_terms_h67"] /
                temporal_pairs["projection_baseline_active_lanes"]
                if temporal_pairs["projection_baseline_active_lanes"] else 0.0
            ),
            "projection_gate_class_channel_ratio_deploy": (
                temporal_pairs["projection_gate_class_channel_terms_deploy"] /
                temporal_pairs["projection_baseline_active_lanes"]
                if temporal_pairs["projection_baseline_active_lanes"] else 0.0
            ),
            "row_all_occupied_classes_mean_ttx": (
                temporal_pairs["row_all_occupied_classes_sum_ttx"] / row_total
                if row_total else 0.0
            ),
            "row_all_occupied_classes_mean_h67": (
                temporal_pairs["row_all_occupied_classes_sum_h67"] / row_total
                if row_total else 0.0
            ),
            "row_kzero_fold_classes_mean_ttx": (
                temporal_pairs["row_kzero_fold_classes_sum_ttx"] / row_total
                if row_total else 0.0
            ),
            "row_kzero_fold_classes_mean_h67": (
                temporal_pairs["row_kzero_fold_classes_sum_h67"] / row_total
                if row_total else 0.0
            ),
            "row_active_projection_classes_mean_ttx": (
                temporal_pairs["row_active_projection_classes_sum_ttx"] / row_total
                if row_total else 0.0
            ),
            "row_active_projection_classes_mean_h67": (
                temporal_pairs["row_active_projection_classes_sum_h67"] / row_total
                if row_total else 0.0
            ),
            "row_active_projection_gate_classes_mean_deploy": (
                temporal_pairs["row_active_projection_gate_classes_sum_deploy"] / row_total
                if row_total else 0.0
            ),
        })
        if temporal_pairs["k_temporal_baseline_reads"] != (
            temporal_pairs["k_temporal_union_reads"] +
            temporal_pairs["k_temporal_intersection_reuse"]
        ):
            raise RuntimeError("K时间复用守恒失败：baseline必须等于union加intersection")
        for model in ("ttx", "h67"):
            if temporal_pairs[f"projection_class_channel_terms_{model}"] > temporal_pairs[
                "projection_baseline_active_lanes"
            ]:
                raise RuntimeError(f"{model.upper()}类通道投影项超过活动K lane基线")
        if temporal_pairs["projection_gate_class_channel_terms_deploy"] > temporal_pairs[
            "projection_baseline_active_lanes"
        ]:
            raise RuntimeError("最终gate类通道投影项超过活动K lane基线")
        if temporal_pairs["projection_gate_q17_out_of_range"]:
            raise RuntimeError("最终gate码超出Q1.7部署范围0..256")
        if temporal_pairs["projection_gate_group_terms_g1"] != temporal_pairs[
            "projection_gate_class_channel_terms_deploy"
        ]:
            raise RuntimeError("单窗口gate-group项必须等于逐row gate类通道项")
        previous_group_terms = temporal_pairs["projection_gate_group_terms_g1"]
        for group_windows in (2, 4, 8, 16):
            current_group_terms = temporal_pairs[f"projection_gate_group_terms_g{group_windows}"]
            if current_group_terms > previous_group_terms:
                raise RuntimeError("扩大窗口组不能增加唯一gate类通道项")
            previous_group_terms = current_group_terms
        for group_windows in (1, 2, 4, 8, 16):
            if temporal_pairs[f"projection_gate_group_active_lanes_g{group_windows}"] != temporal_pairs[
                "projection_baseline_active_lanes"
            ]:
                raise RuntimeError("任意窗口组的活动lane总数必须守恒")
            if temporal_pairs[f"projection_gate_group_window_count_g{group_windows}"] != row_total:
                raise RuntimeError("任意窗口组的有效窗口数必须等于逐row总数")
            ppdi = temporal_pairs[
                f"projection_gate_group_ppdi_delivery_g{group_windows}"
            ]
            m1 = temporal_pairs[
                f"projection_gate_group_delivery_g{group_windows}_m1"
            ]
            m2 = temporal_pairs[
                f"projection_gate_group_delivery_g{group_windows}_m2"
            ]
            if not m2 <= ppdi <= m1:
                raise RuntimeError(
                    "PPDI命令数必须位于无约束M2下界与标量M1上界之间"
                )
        if not (
            temporal_pairs["projection_gate_multicast_delivery_m2"]
            <= temporal_pairs["projection_gate_ppdi_delivery_exact"]
            <= temporal_pairs["projection_gate_multicast_delivery_m1"]
        ):
            raise RuntimeError("逐row PPDI命令数越过M2/M1边界")
        spatial_rows = temporal_pairs["spatial_row_total"]
        spatial_union = temporal_pairs["spatial_union_tokens"]
        temporal_pairs.update({
            "spatial_persistence_ratio": (
                temporal_pairs["spatial_persistent_tokens"] / spatial_union
                if spatial_union else 0.0
            ),
            "spatial_change_ratio": (
                temporal_pairs["spatial_changed_tokens"] / spatial_union
                if spatial_union else 0.0
            ),
        })
        for direction in ("horizontal", "vertical", "diag_down", "diag_up"):
            active_key = f"spatial_{direction}_adjacent_active"
            total_key = f"spatial_{direction}_adjacent_total"
            temporal_pairs[f"spatial_{direction}_adjacent_ratio"] = (
                temporal_pairs[active_key] / temporal_pairs[total_key]
                if temporal_pairs[total_key] else 0.0
            )
        for banks in (4, 8):
            for mapping in ("rowmajor", "diagonal", "xor"):
                key = f"spatial_bank{banks}_{mapping}_cycles_sum"
                temporal_pairs[f"spatial_bank{banks}_{mapping}_cycles_mean"] = (
                    temporal_pairs[key] / spatial_rows if spatial_rows else 0.0
                )
        correlation_fields = (
            "pair_empty_ratio", "mean_events_per_pair", "mean_union_lanes_per_pair",
            "token_kzero_ratio", "s0_pair_empty_ratio", "s1_pair_empty_ratio",
            "s2_pair_empty_ratio", "s3_pair_empty_ratio",
        )
        feature_fields = (
            "input_event_density", "input_active_pixel_ratio", "label_flow_mag_mean",
            "label_flow_mag_p90", "label_flow_gradient_mean", "sample_aee",
        )

        def pearson(x_key: str, y_key: str) -> float | None:
            if len(self.sample_records) < 2:
                return None
            x = torch.tensor([float(row[x_key]) for row in self.sample_records], dtype=torch.float64)
            y = torch.tensor([float(row[y_key]) for row in self.sample_records], dtype=torch.float64)
            x = x - x.mean()
            y = y - y.mean()
            denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
            if denominator == 0:
                return None
            return float((x * y).sum().div(denominator).item())

        sample_correlations = {
            f"{feature}__vs__{workload}": pearson(feature, workload)
            for feature in feature_fields
            for workload in correlation_fields
        }
        return {
            "profile_features": [
                "projection_gate_ppdi_delivery_exact",
                "projection_gate_group_ppdi_delivery_g1_g2_g4_g8_g16",
                "ordered_ppdi_delivery_trace",
            ],
            "h60_records": self.h60_records,
            "h60_by_block": self._aggregate_numeric(self.h60_records, h60_keys, "name"),
            "h60_by_stage": self._aggregate_numeric(self.h60_records, h60_keys, "stage"),
            "activation_records": self.activation_records,
            "dual_line_records": self.dual_line_records,
            "activation_by_kind": sorted(act_by_kind.values(), key=lambda row: row["kind"]),
            "cross_sample_by_stage": sorted(cross_sample_by_stage.values(), key=lambda row: row["name"]),
            "atlif_rows": atlif_rows,
            "operator_rows": operator_rows,
            "execution_records": self.execution_records,
            "operator_by_scope": sorted(operator_by_scope.values(), key=lambda row: row["scope"]),
            "delta_ttx": delta_ttx,
            "score_quantization": {
                "total": score_quant_total,
                "clip_low": score_clip_low,
                "clip_high": score_clip_high,
                "clip_ratio": (
                    (score_clip_low + score_clip_high) / score_quant_total
                    if score_quant_total else 0.0
                ),
            },
            "token_time_bundles": ttb_rows,
            "binary_temporal_pairs": temporal_pairs,
            "sample_records": self.sample_records,
            "sample_correlations": sample_correlations,
        }


def preprocess_chunk(config: dict[str, Any], chunk: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, transform_valid: Any, device: torch.device):
    chunk = chunk.to(device=device, dtype=torch.float32)
    label = label.to(device=device, dtype=torch.float32)
    mask = torch.unsqueeze(mask.to(device=device), dim=1)
    if transform_valid is not None:
        chunk, label, mask = transform_valid((chunk, label, mask.float()))
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
        raise AttributeError("Unsupported event encoding")
    if config["model"]["norm_input"] == "minmax" and torch.any(chunk != 0):
        mn = torch.min(chunk[chunk != 0])
        mx = torch.max(chunk[chunk != 0])
        if mn != mx:
            chunk[chunk != 0] = (chunk[chunk != 0] - mn) / (mx - mn)
    elif config["model"]["norm_input"] == "std" and torch.any(chunk != 0):
        mean = chunk[chunk != 0].mean()
        std = chunk[chunk != 0].std()
        if std > 0:
            chunk[chunk != 0] = (chunk[chunk != 0] - mean) / std
    if config["data"]["spike_th"] is not None:
        chunk[chunk > config["data"]["spike_th"]] = 1
        chunk[chunk < config["data"]["spike_th"]] = 0
    return chunk, label, mask


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    h60_stage = summary["h60_by_stage"]
    activ = summary["activation_by_kind"]
    atlif = summary["atlif_rows"]
    delta_ttx = summary["delta_ttx"]
    ttb_rows = summary.get("token_time_bundles", [])
    pair = summary.get("binary_temporal_pairs", {})
    correlations = summary.get("sample_correlations", {})
    lines = [
        "# NTS11 硬件 P0 Profiling 报告",
        "",
        f"- 实验：`{result['experiment']}`",
        f"- checkpoint：`{result['checkpoint']}`",
        f"- samples：{result['samples']}",
        f"- 评估协议：`{result['eval_protocol']}`",
        f"- 模块数量：`{result['module_counts']}`",
        f"- 权重加载：`{result['checkpoint_load_audit']}`",
        f"- ATLIF 阈值训练/部署语义：`{result['threshold_training_semantics']}`",
        f"- H60 调用记录：{len(summary['h60_records'])}",
        f"- ATLIF 记录模块：{len(atlif)}",
        "",
        "## H60 分 stage 统计",
        "",
        "| stage | calls | gate_entropy | effective_tokens | q_active | k_active | K-zero token | active entries/row | fold classes/row | TTB2 empty |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in h60_stage:
        lines.append(
            f"| {row['group']} | {int(row['calls'])} | {row.get('gate_entropy_mean', 0):.4f} | "
            f"{row.get('effective_tokens_mean', 0):.2f} | {row.get('q_active_density', 0):.5f} | "
            f"{row.get('k_active_density', 0):.5f} | {row.get('zaf_kzero_token_ratio', 0):.4f} | "
            f"{row.get('zaf_active_entries_mean', 0):.2f} | {row.get('zaf_fold_classes_mean', 0):.2f} | "
            f"{row.get('ttb2_empty_ratio', 0):.4f} |"
        )
    lines += [
        "",
        "## Exact Delta-TTX temporal toggle",
        "",
        "| metric | element-weighted result |",
        "|---|---:|",
        f"| temporal lanes | {delta_ttx['temporal_lane_elements']} |",
        f"| Q toggle density | {delta_ttx['q_temporal_toggle_density']:.6%} |",
        f"| K toggle density | {delta_ttx['k_temporal_toggle_density']:.6%} |",
        f"| Q-or-K update density | {delta_ttx['qk_temporal_update_density']:.6%} |",
        f"| t1 ideal lane skip | {delta_ttx['t1_ideal_lane_skip_ratio']:.6%} |",
        f"| full T=2 ideal TX compare reduction | {delta_ttx['full_t2_ideal_compare_reduction']:.6%} |",
        f"| zero-update token/head | {delta_ttx['delta_zero_update_token_ratio']:.6%} |",
        f"| mean changed-token run length | {delta_ttx['delta_mean_changed_run_length']:.4f} |",
        f"| empty 4-token update bundle | {delta_ttx['delta_bundle4_empty_ratio']:.6%} |",
        f"| empty 8-token update bundle | {delta_ttx['delta_bundle8_empty_ratio']:.6%} |",
        "",
        "### Update lanes per token/head",
        "",
        "| updated lanes | token/head count |",
        "|---|---:|",
        f"| 0 | {delta_ttx['delta_update_count_0']} |",
        f"| 1 | {delta_ttx['delta_update_count_1']} |",
        f"| 2 | {delta_ttx['delta_update_count_2']} |",
        f"| 3--4 | {delta_ttx['delta_update_count_3_4']} |",
        f"| 5--8 | {delta_ttx['delta_update_count_5_8']} |",
        f"| 9--16 | {delta_ttx['delta_update_count_9_16']} |",
        f"| 17+ | {delta_ttx['delta_update_count_17_plus']} |",
        "",
        "## True Token-Time Bundle density (T=2)",
        "",
        "| spatial tokens/bundle | bundles | Q-or-K density | empty | K-zero | no K-motion | active 1--8 | active 1--12 | active 1--16 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ttb_rows:
        lines.append(
            f"| {row['token_bundle']} | {row['bundles']} | {row['activity_density']:.6%} | "
            f"{row['empty_ratio']:.6%} | {row['kzero_ratio']:.6%} | "
            f"{row['motion_zero_ratio']:.6%} | {row['active_1_8_ratio']:.6%} | "
            f"{row['active_1_12_ratio']:.6%} | "
            f"{row['active_1_16_ratio']:.6%} |"
        )
    lines += [
        "",
        "## TTX/H67 二值时间对充分统计",
        "",
        "| metric | result |",
        "|---|---:|",
        f"| temporal pairs | {pair.get('pair_total', 0)} |",
        f"| all-four-vector empty | {pair.get('pair_empty_ratio', 0.0):.6%} |",
        f"| K motion zero | {pair.get('pair_motion_zero_ratio', 0.0):.6%} |",
        f"| Q/K temporal update zero | {pair.get('pair_update_zero_ratio', 0.0):.6%} |",
        f"| TTX paired scores equal | {pair.get('pair_score_equal_ttx_ratio', 0.0):.6%} |",
        f"| H67 paired scores equal | {pair.get('pair_score_equal_h67_ratio', 0.0):.6%} |",
        f"| both K slices zero | {pair.get('pair_kzero_both_ratio', 0.0):.6%} |",
        f"| exactly one K slice zero | {pair.get('pair_kzero_one_ratio', 0.0):.6%} |",
        f"| both K slices active | {pair.get('pair_both_active_ratio', 0.0):.6%} |",
        f"| both K zero and same TTX class | {pair.get('pair_kzero_same_class_ttx_ratio', 0.0):.6%} |",
        f"| both K zero and same H67 class | {pair.get('pair_kzero_same_class_h67_ratio', 0.0):.6%} |",
        f"| per-token K zero | {pair.get('token_kzero_ratio', 0.0):.6%} |",
        f"| TTX all score classes/row | {pair.get('row_all_occupied_classes_mean_ttx', 0.0):.4f} |",
        f"| H67 all score classes/row | {pair.get('row_all_occupied_classes_mean_h67', 0.0):.4f} |",
        f"| TTX K-zero fold classes/row | {pair.get('row_kzero_fold_classes_mean_ttx', 0.0):.4f} |",
        f"| H67 K-zero fold classes/row | {pair.get('row_kzero_fold_classes_mean_h67', 0.0):.4f} |",
        "",
        "完整 Q/K cardinality、intersection、same-zero、motion、temporal-update、四向量事件数/并集、",
        "TTX/H67 Q7 分数和行占用类直方图保存在 JSON；`--ordered-trace` 额外保存 Q/K/intersection/",
        "四向量并集的 stage/block 有序压缩 trace。",
    ]
    ranked_correlations = sorted(
        ((key, value) for key, value in correlations.items() if value is not None),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:16]
    lines += [
        "",
        "## 光流样本特征与硬件 workload 相关性",
        "",
        "| Pearson pair | r |",
        "|---|---:|",
    ]
    for key, value in ranked_correlations:
        lines.append(f"| {key} | {float(value):.5f} |")
    lines += [
        "",
        "相关性只用于判断是否值得做 stage/sample-aware 调度，不表示因果关系；profile100 仍需报告散点、",
        "置信区间和异常样本，不能只挑选绝对值最大的相关系数。",
    ]
    lines += [
        "",
        "## Activation / Skip 存储口径",
        "",
        "| kind | calls | elements | density | FP16 bytes | ternary packed bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in activ:
        lines.append(
            f"| {row['kind']} | {row['calls']} | {row['elements']} | {row['density']:.6f} | "
            f"{row['bytes_fp16']} | {row['bytes_ternary_packed']} |"
        )
    ternary = [row for row in atlif if row.get("output_mode") == "ternary"]
    binary = [row for row in atlif if row.get("output_mode") == "binary"]
    def avg(rows: list[dict[str, Any]], key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0
    lines += [
        "",
        "## ATLIF 活性快照",
        "",
        "| group | modules | activity | pos_rate | neg_rate |",
        "|---|---:|---:|---:|---:|",
        f"| ternary | {len(ternary)} | {avg(ternary, 'activity'):.6f} | {avg(ternary, 'pos_rate'):.6f} | {avg(ternary, 'neg_rate'):.6f} |",
        f"| binary | {len(binary)} | {avg(binary, 'activity'):.6f} | {avg(binary, 'pos_rate'):.6f} | {avg(binary, 'neg_rate'):.6f} |",
        "",
        "## 读法",
        "",
        "- `stage_skip_predownsample` 只对应 S0/S1/S2 的 downsample 前 skip。",
        "- `stage_skip_final` 对应 S3 final-stage output，硬件上要跨 bottleneck 保留给 decoder i=0。",
        "- 旧 `TTB2 empty` 按整个 window/head 的 Q 活性聚合，只保留作历史代理，不能证明完整 attention 可跳过。",
        "- `True Token-Time Bundle` 按 T=2 × contiguous spatial tokens × 32 lanes 统计 Q-or-K、K-zero 与 K-motion。",
        "- Q/K empty 仍会产生 silent/silent score并参与 Shiftmax；只有 Delta score reuse、K-zero value gating等具备单独等价证明的路径可无损跳过。",
    ]
    cross_sample = summary.get("cross_sample_by_stage", [])
    if cross_sample:
        lines += [
            "",
            "## 同序列相邻样本的stage边界变化",
            "",
            "| 边界 | 可比较样本对 | 采样值 | 精确相等 | active翻转 | 符号类变化 | 归一化绝对变化 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in cross_sample:
            lines.append(
                f"| {row['name']} | {int(row['comparable_pairs'])} | {int(row['sampled_values'])} | "
                f"{float(row['cross_sample_exact_equal_ratio']):.6f} | "
                f"{float(row['cross_sample_active_xor_ratio']):.6f} | "
                f"{float(row['cross_sample_sign_class_change_ratio']):.6f} | "
                f"{float(row['cross_sample_normalized_mean_abs_delta']):.6f} |"
            )
        lines += [
            "",
            "该表只比较验证列表中同一sequence的相邻条目，并对每个张量最多确定性采样2^20个值。",
            "它用于筛选persistent-HTT或增量更新候选，不等价于证明整帧可复用。",
        ]
    operator_scopes = summary.get("operator_by_scope", [])
    if operator_scopes:
        lines += [
            "",
            "## Linear与卷积运行时操作分账",
            "",
            "| 范围 | 模块 | 调用 | dense标量MAC | 输入活动率 | 活动率加权MAC代理 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in operator_scopes:
            lines.append(
                f"| {row['scope']} | {int(row['modules'])} | {int(row['calls'])} | "
                f"{int(row['dense_macs'])} | {float(row['input_activity']):.6f} | "
                f"{float(row['activity_weighted_macs_proxy']):.3f} |"
            )
        lines += [
            "",
            "dense标量MAC按运行时输出元素与weight fan-in计算。活动率加权MAC对Linear为连通度代理，",
            "对带padding/stride的卷积不是精确SOP；它仍优于用全网单一firing rate缩放所有层。",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--ordered-trace",
        action="store_true",
        help="store compressed stage/block-ordered Delta and TTB count traces",
    )
    parser.add_argument(
        "--dual-line-trace",
        action="store_true",
        help="emit exact per-operator/per-timestep Local versus Motion source work",
    )
    parser.add_argument(
        "--dual-line-tile-dir",
        type=Path,
        default=None,
        help="optional deterministic real 256-bit Local/Motion tile descriptors",
    )
    parser.add_argument("--dual-line-tile-bits", type=int, default=256)
    parser.add_argument("--dual-line-tile-pairs-per-call", type=int, default=4)
    parser.add_argument(
        "--dual-line-cohort-census-dir",
        type=Path,
        default=None,
        help="optional exact streaming T10 Local/Motion coefficient-cohort census",
    )
    parser.add_argument("--cohort-census-source-chunk-size", type=int, default=256)
    parser.add_argument("--cohort-census-row-block-size", type=int, default=512)
    parser.add_argument("--cohort-census-max-working-set-mib", type=int, default=256)
    parser.add_argument(
        "--shift-residual-census-dir",
        type=Path,
        default=None,
        help="optional exact T10 Conv2d shift-compensated residual opportunity census",
    )
    parser.add_argument("--shift-residual-radius", type=int, default=1)
    parser.add_argument("--shift-residual-output-tile", type=int, nargs=2, default=(16, 16))
    parser.add_argument("--shift-residual-source-chunk-size", type=int, default=256)
    parser.add_argument("--shift-residual-accumulator-bits", type=int, default=24)
    parser.add_argument("--shift-residual-expected-operator-calls", type=int, default=0)
    parser.add_argument("--shift-residual-expected-exact-calls", type=int, default=0)
    parser.add_argument(
        "--full-spatial-c4-dir",
        type=Path,
        default=None,
        help="optional exact full-spatial adjacent-C4 direct-M4 sufficient statistics",
    )
    parser.add_argument(
        "--full-spatial-c4-dependency-audit",
        type=Path,
        default=None,
        help="identity-locked dependency audit defining the direct-M4 producer allowlist",
    )
    parser.add_argument(
        "--full-spatial-c4-dependency-manifest",
        type=Path,
        default=None,
        help="v2 dependency manifest content-bound to the M17 audit",
    )
    parser.add_argument(
        "--full-spatial-c4-dependency-events",
        type=Path,
        default=None,
        help="dependency JSONL used to bind the exact sample and producer call indices",
    )
    parser.add_argument(
        "--dependency-trace-dir",
        type=Path,
        default=None,
        help="optional metadata-only storage/version and functional-op dependency DAG",
    )
    parser.add_argument("--dependency-trace-samples", type=int, default=1)
    parser.add_argument(
        "--bit-trace-dir",
        type=Path,
        default=None,
        help="可选：导出真实Q/K、Q1.7 gate和projection权重NPZ",
    )
    parser.add_argument("--bit-trace-samples", type=int, default=1)
    parser.add_argument("--bit-trace-windows", type=int, default=1)
    parser.add_argument(
        "--bit-trace-all-blocks",
        action="store_true",
        help="默认只导出四个stage的B0；启用后导出所有attention block",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config, device = load_config(config_path)
    threshold_semantics = threshold_training_semantics(config)
    dataset = DSECDatasetLite(config, file_list="valid", stereo=False, scale_factor=config.get("test", {}).get("scale_factor", 1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=False, num_workers=args.num_workers)
    transform_valid = None
    if config["loader"].get("crop") is not None:
        transform_valid = Compose([CenterCrop((config["loader"]["crop"][0], config["loader"]["crop"][1]))])
    model = build_model(config, checkpoint_path, device)
    checkpoint_load_audit = validate_h9_load_audit(model, config)
    module_counts = h9_module_counts(model)
    bn_policy = config.get("test", {}).get("bn_policy", "running")
    bn_modules_changed = configure_batch_norm_evaluation(model, bn_policy)
    eval_protocol = {
        "resolution": list(config["loader"]["resolution"]),
        "crop": config["loader"].get("crop"),
        "window_size": list(config["swin_transformer"]["window_size"]),
        "pretrained_window_size": config["swin_transformer"].get(
            "pretrained_window_size"
        ),
        "tokens_per_window": math.prod(
            int(value) for value in config["swin_transformer"]["window_size"]
        ),
        "remap": config["loader"].get("remap"),
        "bn_policy": bn_policy,
        "bn_modules_changed": bn_modules_changed,
        "eval_batch_size": 1,
        "num_workers": args.num_workers,
    }
    artifact_identity = {
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size": checkpoint_path.stat().st_size,
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    bit_trace_writer = None
    if args.bit_trace_dir is not None:
        from h67_bit_trace import AttentionBitTraceWriter

        bit_trace_writer = AttentionBitTraceWriter(
            args.bit_trace_dir,
            sample_limit=args.bit_trace_samples,
            windows_per_call=args.bit_trace_windows,
            first_block_only=not args.bit_trace_all_blocks,
        )
        trace_writer_source = Path(__file__).with_name("h67_bit_trace.py")
        bit_trace_writer.bind_run_context(
            {
                "artifact_identity": artifact_identity,
                "eval_protocol": eval_protocol,
                "module_counts": module_counts,
                "checkpoint_load_audit": checkpoint_load_audit,
                "threshold_training_semantics": threshold_semantics,
                "source_sha256": {
                    "profiler": file_sha256(Path(__file__).resolve()),
                    "trace_writer": file_sha256(trace_writer_source.resolve()),
                },
            }
        )
    dual_line_tile_writer = None
    if args.dual_line_tile_dir is not None:
        if not args.dual_line_trace:
            raise ValueError("--dual-line-tile-dir requires --dual-line-trace")
        from h67_dual_line_tile_trace import DualLineTileTraceWriter

        dual_line_tile_writer = DualLineTileTraceWriter(
            args.dual_line_tile_dir,
            tile_bits=args.dual_line_tile_bits,
            pairs_per_call=args.dual_line_tile_pairs_per_call,
        )
        tile_writer_source = Path(__file__).with_name("h67_dual_line_tile_trace.py")
        dual_line_tile_writer.bind_run_context({
            "artifact_identity": artifact_identity,
            "eval_protocol": eval_protocol,
            "checkpoint_load_audit": checkpoint_load_audit,
            "source_sha256": {
                "profiler": file_sha256(Path(__file__).resolve()),
                "tile_writer": file_sha256(tile_writer_source.resolve()),
            },
        })
    dual_line_cohort_writer = None
    if args.dual_line_cohort_census_dir is not None:
        if not args.dual_line_trace:
            raise ValueError(
                "--dual-line-cohort-census-dir requires --dual-line-trace"
            )
        from h67_dual_line_cohort_census import StreamingCohortCensusWriter

        dual_line_cohort_writer = StreamingCohortCensusWriter(
            args.dual_line_cohort_census_dir,
            temporal_steps=10,
            source_chunk_size=args.cohort_census_source_chunk_size,
            requested_row_block_size=args.cohort_census_row_block_size,
            max_working_set_mib=args.cohort_census_max_working_set_mib,
        )
        cohort_writer_source = Path(__file__).with_name(
            "h67_dual_line_cohort_census.py"
        )
        dual_line_reference_source = Path(__file__).with_name(
            "h67_dual_line_trace.py"
        )
        dual_line_cohort_writer.bind_run_context({
            "artifact_identity": artifact_identity,
            "eval_protocol": eval_protocol,
            "checkpoint_load_audit": checkpoint_load_audit,
            "source_sha256": {
                "profiler": file_sha256(Path(__file__).resolve()),
                "cohort_census_writer": file_sha256(cohort_writer_source.resolve()),
                "dual_line_reference": file_sha256(
                    dual_line_reference_source.resolve()
                ),
            },
        })
    shift_residual_writer = None
    if args.shift_residual_census_dir is not None:
        if not args.dual_line_trace:
            raise ValueError(
                "--shift-residual-census-dir requires --dual-line-trace"
            )
        from h67_shift_residual_census import StreamingShiftResidualCensusWriter

        shift_residual_writer = StreamingShiftResidualCensusWriter(
            args.shift_residual_census_dir,
            temporal_steps=10,
            shift_radius=args.shift_residual_radius,
            output_tile=tuple(args.shift_residual_output_tile),
            source_chunk_size=args.shift_residual_source_chunk_size,
            accumulator_bits=args.shift_residual_accumulator_bits,
            expected_samples=args.samples,
            expected_operator_calls=args.shift_residual_expected_operator_calls,
            expected_exact_calls=args.shift_residual_expected_exact_calls,
        )
        shift_writer_source = Path(__file__).with_name(
            "h67_shift_residual_census.py"
        )
        shift_eval_protocol = dict(eval_protocol)
        shift_eval_protocol.update({
            "temporal_steps": 10,
            "requested_profile_samples": args.samples,
            "expected_operator_calls": args.shift_residual_expected_operator_calls,
            "expected_exact_calls": args.shift_residual_expected_exact_calls,
            "temporal_axis_contract": "hook_input_dim0_is_T10_and_dim1_is_eval_batch",
        })
        shift_residual_writer.bind_run_context({
            "artifact_identity": artifact_identity,
            "eval_protocol": shift_eval_protocol,
            "checkpoint_load_audit": checkpoint_load_audit,
            "source_sha256": {
                "profiler": file_sha256(Path(__file__).resolve()),
                "shift_residual_census_writer": file_sha256(
                    shift_writer_source.resolve()
                ),
            },
        })
    full_spatial_c4_writer = None
    full_spatial_inputs = (
        args.full_spatial_c4_dir, args.full_spatial_c4_dependency_audit,
        args.full_spatial_c4_dependency_manifest, args.full_spatial_c4_dependency_events,
    )
    if any(value is not None for value in full_spatial_inputs):
        if any(value is None for value in full_spatial_inputs):
            raise ValueError(
                "full-spatial C4 output, audit, manifest, and events must be provided together"
            )
        if not args.dual_line_trace or args.samples != 1:
            raise ValueError("M17 v2 requires --dual-line-trace and exactly one sample")
        dependency_audit_path = args.full_spatial_c4_dependency_audit.resolve()
        dependency_manifest_path = args.full_spatial_c4_dependency_manifest.resolve()
        dependency_events_path = args.full_spatial_c4_dependency_events.resolve()
        from h67_full_spatial_c4_oracle import (
            H67FullSpatialC4OracleWriter, validate_dependency_contract,
        )

        allowed_calls = validate_dependency_contract(
            dependency_manifest_path, dependency_audit_path, dependency_events_path,
            artifact_identity=artifact_identity, eval_protocol=eval_protocol,
            checkpoint_load_audit=checkpoint_load_audit,
        )

        full_spatial_c4_writer = H67FullSpatialC4OracleWriter(
            args.full_spatial_c4_dir, allowed_calls=allowed_calls,
        )
        full_oracle_source = Path(__file__).with_name("h67_full_spatial_c4_oracle.py")
        full_spatial_c4_writer.bind_run_context({
            "artifact_identity": artifact_identity,
            "eval_protocol": eval_protocol,
            "checkpoint_load_audit": checkpoint_load_audit,
            "dependency_audit_path": str(dependency_audit_path),
            "dependency_audit_sha256": file_sha256(dependency_audit_path),
            "dependency_manifest_path": str(dependency_manifest_path),
            "dependency_manifest_sha256": file_sha256(dependency_manifest_path),
            "dependency_events_path": str(dependency_events_path),
            "dependency_events_sha256": file_sha256(dependency_events_path),
            "source_sha256": {
                "profiler": file_sha256(Path(__file__).resolve()),
                "full_spatial_c4_writer": file_sha256(full_oracle_source.resolve()),
            },
        })
    dependency_trace_writer = None
    if args.dependency_trace_dir is not None:
        from h67_dependency_trace import TensorDependencyTraceWriter

        dependency_trace_writer = TensorDependencyTraceWriter(
            args.dependency_trace_dir,
            sample_limit=args.dependency_trace_samples,
        )
        dependency_writer_source = Path(__file__).with_name("h67_dependency_trace.py")
        dependency_trace_writer.bind_run_context({
            "artifact_identity": artifact_identity,
            "eval_protocol": eval_protocol,
            "checkpoint_load_audit": checkpoint_load_audit,
            "source_sha256": {
                "profiler": file_sha256(Path(__file__).resolve()),
                "dependency_writer": file_sha256(dependency_writer_source.resolve()),
            },
        })
        dependency_trace_writer.attach(model)
    profiler = HardwareProfiler(
        model,
        ordered_trace=args.ordered_trace,
        dual_line_trace=args.dual_line_trace,
        bit_trace_writer=bit_trace_writer,
        dual_line_tile_writer=dual_line_tile_writer,
        full_spatial_c4_writer=full_spatial_c4_writer,
        dual_line_cohort_writer=dual_line_cohort_writer,
        shift_residual_writer=shift_residual_writer,
    )
    profiler.attach()
    processed = 0
    profile_completed = False
    try:
        with torch.no_grad():
            for chunk, mask, label in loader:
                if processed >= args.samples:
                    break
                functional.reset_net(model)
                file_row = dataset.files[processed]
                file_names = list(file_row) if isinstance(file_row, (list, tuple)) else [str(file_row)]
                sample_key = "|".join(str(item) for item in file_names)
                sequence_names = ["_".join(Path(str(item)).stem.split("_")[:-1]) for item in file_names]
                sequence_key = "|".join(sequence_names)
                profiler.begin_sample(
                    processed,
                    sample_key=sample_key,
                    sequence_key=sequence_key,
                )
                if dependency_trace_writer is not None:
                    dependency_trace_writer.begin_sample(
                        processed,
                        sample_key=sample_key,
                        sequence_key=sequence_key,
                    )
                x, transformed_label, transformed_mask = preprocess_chunk(
                    config, chunk, label, mask, transform_valid, device
                )
                dependency_capture = (
                    dependency_trace_writer.capture()
                    if dependency_trace_writer is not None
                    else contextlib.nullcontext()
                )
                with dependency_capture:
                    output = model(x)
                if dependency_trace_writer is not None:
                    dependency_trace_writer.end_sample()
                prediction = output["flow"][-1]
                profiler.record_sample(
                    chunk=x,
                    label=transformed_label,
                    mask=transformed_mask,
                    prediction=prediction,
                    flow_scaling=float(config.get("metrics", {}).get("flow_scaling", 128)),
                )
                processed += 1
                if processed % 5 == 0:
                    print(f"[profile] processed {processed}/{args.samples}", flush=True)
        if dual_line_cohort_writer is not None and processed <= 0:
            raise RuntimeError("M24C cannot admit an empty profile")
        if shift_residual_writer is not None and processed <= 0:
            raise RuntimeError("M28A cannot admit an empty profile")
        profile_completed = True
    finally:
        close_completed = False
        try:
            profiler.close()
            if dual_line_tile_writer is not None:
                dual_line_tile_writer.close()
            if full_spatial_c4_writer is not None:
                full_spatial_c4_writer.close()
            if dependency_trace_writer is not None:
                dependency_trace_writer.close()
            close_completed = True
        finally:
            try:
                if dual_line_cohort_writer is not None:
                    if profile_completed and close_completed:
                        dual_line_cohort_writer.close()
                    else:
                        dual_line_cohort_writer.abort("profiler did not complete")
            finally:
                if shift_residual_writer is not None:
                    if profile_completed and close_completed:
                        shift_residual_writer.close()
                    else:
                        shift_residual_writer.abort("profiler did not complete")

    try:
        from models.STSwinNet_SNN.atlif_ternary_psn import atlif_ternary_summary
        atlif_summary = atlif_ternary_summary(model)
    except Exception as exc:
        atlif_summary = {"error": str(exc)}
    result = {
        "experiment": config.get("experiment", args.config.stem),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "samples": processed,
        "ordered_trace": bool(args.ordered_trace),
        "dual_line_trace": bool(args.dual_line_trace),
        "dual_line_tile_manifest": (
            str(dual_line_tile_writer.manifest_path)
            if dual_line_tile_writer is not None else None
        ),
        "dual_line_cohort_census_manifest": (
            str(dual_line_cohort_writer.manifest_path)
            if dual_line_cohort_writer is not None else None
        ),
        "shift_residual_census_manifest": (
            str(shift_residual_writer.manifest_path)
            if shift_residual_writer is not None else None
        ),
        "full_spatial_c4_manifest": (
            str(full_spatial_c4_writer.manifest_path)
            if full_spatial_c4_writer is not None else None
        ),
        "dependency_trace_manifest": (
            str(dependency_trace_writer.manifest_path)
            if dependency_trace_writer is not None else None
        ),
        "bit_trace_manifest": (
            str(bit_trace_writer.manifest_path)
            if bit_trace_writer is not None
            else None
        ),
        "bit_trace_records": (
            len(bit_trace_writer.records) if bit_trace_writer is not None else 0
        ),
        "module_counts": module_counts,
        "checkpoint_load_audit": checkpoint_load_audit,
        "threshold_training_semantics": threshold_semantics,
        "eval_protocol": eval_protocol,
        "artifact_identity": artifact_identity,
        "atlif_summary": atlif_summary,
        "summary": profiler.summary(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "nts11_hardware_p0_profile.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output_dir / "h60_by_block.csv", result["summary"]["h60_by_block"])
    write_csv(args.output_dir / "h60_by_stage.csv", result["summary"]["h60_by_stage"])
    write_csv(args.output_dir / "activation_records.csv", result["summary"]["activation_records"])
    write_csv(args.output_dir / "stage_cross_sample_delta.csv", result["summary"]["cross_sample_by_stage"])
    write_csv(args.output_dir / "atlif_activity.csv", result["summary"]["atlif_rows"])
    write_csv(args.output_dir / "operator_runtime.csv", result["summary"]["operator_rows"])
    write_csv(args.output_dir / "execution_trace.csv", result["summary"]["execution_records"])
    write_csv(args.output_dir / "dual_line_operator_trace.csv", result["summary"]["dual_line_records"])
    write_csv(args.output_dir / "operator_by_scope.csv", result["summary"]["operator_by_scope"])
    write_csv(args.output_dir / "sample_workload.csv", result["summary"]["sample_records"])
    write_md(args.output_dir / "nts11_hardware_p0_profile.md", result)
    print(f"[profile] wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
