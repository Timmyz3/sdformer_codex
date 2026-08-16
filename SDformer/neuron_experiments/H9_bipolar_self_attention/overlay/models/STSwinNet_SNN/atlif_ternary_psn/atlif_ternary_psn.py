"""PSN temporal mixer with ATLIF threshold updates and binary/ternary output."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def zif_backward(x: torch.Tensor, thre: torch.Tensor) -> torch.Tensor:
    return (1.0 - (x / thre).abs()).clamp_min(0)


class TernarySurrogate(torch.autograd.Function):
    """ATLIF-style threshold update with ternary {-thre, 0, +thre} output."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, thre: torch.Tensor, sp: float, neg_scale: float):
        neg_thre = thre * float(neg_scale)
        pos_active = (input >= thre).float()
        neg_active = (input <= -neg_thre).float()
        ternary = pos_active - neg_active
        pos_updates = zif_backward(input - thre, thre) * pos_active
        neg_updates = zif_backward((-input) - neg_thre, neg_thre) * neg_active
        thre_updates = (sp * (pos_updates + neg_updates)).sum(0).mean().item()
        ctx.save_for_backward(input, thre, neg_thre)
        return ternary * thre, thre_updates

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor, _dummy):
        input, thre, neg_thre = ctx.saved_tensors
        pos_tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
        neg_tmp = (1.0 - (((-input) - neg_thre) / neg_thre).abs()).clamp(min=0)
        tmp = torch.maximum(pos_tmp, neg_tmp)
        grad_input = grad_input * tmp
        grad_thre = -(grad_input.abs() * tmp).mean()
        return grad_input, grad_thre, None, None


class BinarySurrogate(torch.autograd.Function):
    """ATLIF-style threshold update with binary {0, +thre} output."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, thre: torch.Tensor, sp: float):
        active = (input >= thre).float()
        pos_updates = zif_backward(input - thre, thre) * active
        thre_updates = (sp * pos_updates).sum(0).mean().item()
        ctx.save_for_backward(input, thre)
        return active * thre, thre_updates

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor, _dummy):
        input, thre = ctx.saved_tensors
        tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
        grad_input = grad_input * tmp
        grad_thre = -(grad_input.abs() * tmp).mean()
        return grad_input, grad_thre, None


class SymmetricBinarySurrogate(torch.autograd.Function):
    """Equal-magnitude +/- threshold detection with one-bit output."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, thre: torch.Tensor, sp: float):
        pos_active = (input >= thre).float()
        neg_active = (input <= -thre).float()
        active = torch.maximum(pos_active, neg_active)
        pos_updates = zif_backward(input - thre, thre) * pos_active
        neg_updates = zif_backward((-input) - thre, thre) * neg_active
        thre_updates = (sp * (pos_updates + neg_updates)).sum(0).mean().item()
        ctx.save_for_backward(input, thre)
        return active * thre, thre_updates

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor, _dummy):
        input, thre = ctx.saved_tensors
        pos_tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
        neg_tmp = (1.0 - (((-input) - thre) / thre).abs()).clamp(min=0)
        tmp = torch.maximum(pos_tmp, neg_tmp)
        grad_input = grad_input * tmp
        grad_thre = -(grad_input.abs() * tmp).mean()
        return grad_input, grad_thre, None


class OfficialATLIFSurrogate(torch.autograd.Function):
    """Official Activity-Pruning-SNN binary ATLIF surrogate.

    Ported from:
    optimization_sources/neuron_optimization/ATLIF_Activity-Pruning-SNN/
    models/submodules/layers.py

    The official update is binary-only: output is {0, thresh}. The manual
    threshold increment is accumulated separately in ``update_value`` and is
    applied after optimizer.step().
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, thre: torch.Tensor, sp: float):
        out = (input >= thre).float()
        thre_updates = (sp * zif_backward(input - thre, thre) * out).sum(0).mean().item()
        ctx.save_for_backward(input, thre)
        return out * thre, thre_updates

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor, _dummy):
        input, thre = ctx.saved_tensors
        normalized = (input - thre) / thre
        tmp = (1.0 - normalized.abs()).clamp(min=0)
        grad_input = grad_input * tmp
        grad_thre = -(grad_input * tmp).mean()
        return grad_input, grad_thre, None


class ATLIFTernaryPSN(nn.Module):
    """PSN-compatible neuron combining PSN, ATLIF threshold growth, and sparse output."""

    official_atlif_source = "https://github.com/putshua/Activity-Pruning-SNN"
    official_tsn_source = "https://github.com/yfguo91/Ternary-Spike"

    def __init__(
        self,
        T: int,
        base_psn: nn.Module | None = None,
        thresh: float = 1.0,
        sparsity_eta: float = 0.0,
        negative_threshold_scale: float = 5.0,
        activity_eta: float = 0.0,
        min_threshold: float | None = 1.0e-3,
        max_threshold: float | None = None,
        threshold_lr_scale: float | None = None,
        target_rate: float | None = None,
        target_rate_eta: float = 0.0,
        target_rate_mode: str = "upper_bound",
        negative_target_rate: float | None = None,
        negative_target_eta: float = 0.0,
        negative_scale_min: float | None = None,
        negative_scale_max: float | None = None,
        center_mode: str = "zero",
        output_mode: str = "ternary",
        threshold_mode: str = "asymmetric_scale",
        quantile_q: float | None = None,
        quantile_momentum: float = 0.9,
        quantile_guard_margin: float = 0.25,
        quantile_min_guard: float = 0.0,
        quantile_sample_size: int = 4096,
        importance_enabled: bool = False,
        importance_momentum: float = 0.9,
        importance_scale: float = 0.0,
        importance_min_guard: float = 0.1,
    ) -> None:
        super().__init__()
        self.T = int(T)
        if output_mode not in {"ternary", "binary"}:
            raise ValueError("output_mode must be ternary or binary")
        if center_mode not in {"zero", "bias", "calibrated"}:
            raise ValueError("center_mode must be zero, bias, or calibrated")
        if threshold_mode not in {
            "asymmetric_scale",
            "symmetric_bsa_tsn",
            "symmetric_target_rate",
            "symmetric_binary_abs",
            "official_atlif",
        }:
            raise ValueError(
                "threshold_mode must be asymmetric_scale, symmetric_bsa_tsn, symmetric_target_rate, "
                "symmetric_binary_abs, or official_atlif"
            )
        if threshold_mode == "official_atlif" and output_mode != "binary":
            raise ValueError("official_atlif follows the official binary ATLIF output {0, thresh}")
        if threshold_mode == "symmetric_binary_abs" and output_mode != "binary":
            raise ValueError("symmetric_binary_abs requires one-bit binary output")
        self.output_mode = output_mode
        self.threshold_mode = threshold_mode
        self.center_mode = center_mode
        if threshold_mode == "official_atlif":
            self.act = OfficialATLIFSurrogate.apply
        elif threshold_mode == "symmetric_binary_abs":
            self.act = SymmetricBinarySurrogate.apply
        else:
            self.act = TernarySurrogate.apply if output_mode == "ternary" else BinarySurrogate.apply
        self.thresh = nn.Parameter(torch.tensor(float(thresh)), requires_grad=True)
        self.sp = float(sparsity_eta)
        self.negative_threshold_scale = float(negative_threshold_scale)
        self.activity_eta = float(activity_eta)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.threshold_lr_scale = None if threshold_lr_scale is None else float(threshold_lr_scale)
        self.target_rate = target_rate
        self.target_rate_eta = float(target_rate_eta)
        self.target_rate_mode = str(target_rate_mode)
        self.negative_target_rate = negative_target_rate
        self.negative_target_eta = float(negative_target_eta)
        self.negative_scale_min = negative_scale_min
        self.negative_scale_max = negative_scale_max
        self.quantile_q = None if quantile_q is None else float(quantile_q)
        self.quantile_momentum = float(quantile_momentum)
        self.quantile_guard_margin = float(quantile_guard_margin)
        self.quantile_min_guard = float(quantile_min_guard)
        self.quantile_sample_size = int(quantile_sample_size)
        self.quantile_value = 0.0
        self._quantile_initialized = False
        self.importance_enabled = bool(importance_enabled)
        self.importance_momentum = float(importance_momentum)
        self.importance_scale = float(importance_scale)
        self.importance_min_guard = float(importance_min_guard)
        self.importance_ema = 0.0
        self.importance_last = 0.0
        self._importance_initialized = False
        self.r = 0.0
        self.pos_r = 0.0
        self.neg_r = 0.0
        self.positive_trigger_r = 0.0
        self.negative_trigger_r = 0.0
        self.act_value = 0.0
        self.update_value = 0.0

        if base_psn is not None and hasattr(base_psn, "weight") and hasattr(base_psn, "bias"):
            self.weight = nn.Parameter(base_psn.weight.detach().clone())
            self.bias = nn.Parameter(base_psn.bias.detach().clone())
            self.surrogate_function = getattr(base_psn, "surrogate_function", None)
        else:
            self.weight = nn.Parameter(torch.zeros([self.T, self.T]))
            self.bias = nn.Parameter(torch.zeros([self.T, 1]))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.constant_(self.bias, -1.0)
            self.surrogate_function = None
        center = self.bias.detach().clone() if center_mode == "bias" else torch.zeros_like(self.bias.detach())
        self.register_buffer("center", center)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        if self.center_mode != "zero":
            h_seq = h_seq - self.center.to(device=h_seq.device, dtype=h_seq.dtype)
        observer = getattr(self, "_h9_calibration_observer", None)
        if observer is not None:
            observer(h_seq.detach(), self.thresh.detach())
        if self.quantile_q is not None:
            with torch.no_grad():
                values = h_seq.detach().abs().float().reshape(-1)
                if self.quantile_sample_size > 0 and values.numel() > self.quantile_sample_size:
                    stride = max(1, values.numel() // self.quantile_sample_size)
                    values = values[::stride][: self.quantile_sample_size]
                q = torch.quantile(values, self.quantile_q)
                q_value = float(q.detach().cpu())
                if not self._quantile_initialized:
                    self.quantile_value = q_value
                    self._quantile_initialized = True
                else:
                    momentum = min(max(self.quantile_momentum, 0.0), 1.0)
                    self.quantile_value = momentum * self.quantile_value + (1.0 - momentum) * q_value
        if self.output_mode == "ternary":
            negative_scale = 1.0 if self.threshold_mode in {"symmetric_bsa_tsn", "symmetric_target_rate"} else self.negative_threshold_scale
            spike, thre_updates = self.act(h_seq, self.thresh, self.sp, negative_scale)
        else:
            spike, thre_updates = self.act(h_seq, self.thresh, self.sp)
            if self.threshold_mode == "official_atlif":
                thre_updates = thre_updates / max(1, self.T)
        self.update_value += thre_updates
        out = spike.view(x_seq.shape)
        with torch.no_grad():
            thresh = self.thresh.detach().clamp_min(1e-12)
            ternary = out / thresh
            active = ternary.ne(0).float()
            self.r = active.mean().item()
            self.pos_r = ternary.gt(0).float().mean().item()
            self.neg_r = ternary.lt(0).float().mean().item()
            self.positive_trigger_r = h_seq.ge(thresh).float().mean().item()
            self.negative_trigger_r = h_seq.le(-thresh).float().mean().item()
        self.act_value = out.abs().reshape(out.size(0), -1).mean(1).sum()
        if self.importance_enabled and out.requires_grad:
            activation = out.detach()

            def _capture_importance(grad: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    grad_abs_mean = grad.detach().float().abs().mean().clamp_min(1.0e-12)
                    saliency = (activation.float() * grad.detach().float()).abs().mean() / grad_abs_mean
                    value = float(saliency.detach().cpu())
                    self.importance_last = value
                    if not self._importance_initialized:
                        self.importance_ema = value
                        self._importance_initialized = True
                    else:
                        momentum = min(max(self.importance_momentum, 0.0), 1.0)
                        self.importance_ema = momentum * self.importance_ema + (1.0 - momentum) * value
                return grad

            out.register_hook(_capture_importance)
        return out

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, thresh={float(self.thresh.detach().cpu()):.4f}, "
            f"sp={self.sp}, neg_scale={self.negative_threshold_scale}, "
            f"mode={self.output_mode}, threshold_mode={self.threshold_mode}, center_mode={self.center_mode}"
        )
