"""DynamicSwin mechanism adaptation, not its original network reproduction.

Official PredictorLG/fastMlp topology: upstream/models/dyswin.py (MIT).
Our additions: TBHWC complete-T10 input, time-mean selection view, valid-only
tile pooling, dense differentiable dual branches and frozen-BN tile execution.
The returned branch excludes the outer residual u, matching native TileMLP.
"""
from dataclasses import dataclass, asdict
import math
from typing import Callable
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    channels: int
    hidden_channels: int
    time_steps: int = 10
    tile: int = 9
    keep_ratio: float = 0.5
    gumbel_temperature: float = 1.0
    slow_contract: str = "complete_T_spatially_separable_frozen_BN"

    def __post_init__(self):
        if self.channels < 4 or self.channels % 4 or self.hidden_channels < 1:
            raise ValueError("channels must be a positive multiple of 4")
        if self.time_steps != 10 or self.tile not in (1, 9):
            raise ValueError("Only T10 and pointT/tile9T contracts are supported")
        if not 0 <= self.keep_ratio <= 1 or self.gumbel_temperature <= 0:
            raise ValueError("Invalid selection ratio/temperature")
        if self.slow_contract != "complete_T_spatially_separable_frozen_BN":
            raise ValueError("Caller must explicitly supply the separable frozen-BN contract")


class LocalGlobalPredictor(nn.Module):
    """Official PredictorLG architecture, exposing scores before selection."""
    def __init__(self, channels):
        super().__init__()
        self.in_conv = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.GELU())
        self.out_conv = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.GELU(),
            nn.Linear(channels // 2, channels // 4), nn.GELU(),
            nn.Linear(channels // 4, 2), nn.LogSoftmax(dim=-1))

    def forward(self, x):
        z = self.in_conv(x)  # [B, spatial points, C]
        c = z.shape[-1]
        local = z[..., :c // 2]
        context = z[..., c // 2:].mean(1, keepdim=True).expand_as(local)
        return self.out_conv(torch.cat((local, context), dim=-1))


class FastBranch(nn.Module):
    """Official fastMlp topology: per-token LN + real-valued C-to-C linear."""
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels))
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        return self.drop(self.fc1(x))


def frozen_bn_guard(module):
    if not isinstance(module, nn.Module):
        # A plain callable cannot be introspected; its owner bears this contract.
        return
    for name, m in module.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.training or not m.track_running_stats or m.running_mean is None or m.running_var is None:
                raise ValueError("Frozen running-stat BN required: " + name)
            if not torch.isfinite(m.running_mean).all() or not torch.isfinite(m.running_var).all() or (m.running_var < 0).any():
                raise ValueError("Invalid BN statistics: " + name)


@dataclass
class Result:
    branch: torch.Tensor
    tile_gate: torch.Tensor
    tile_log_probs: torch.Tensor | None
    trace: dict

    def add_residual(self, u):
        return u + self.branch


class DynamicSwinMLPAdapter(nn.Module):
    def __init__(self, slow_branch: Callable, config: Config):
        super().__init__()
        if not callable(slow_branch):
            raise TypeError("slow_branch must return only the MLP residual branch")
        if slow_branch.__class__.__name__ == "MS_Spiking_Mlp":
            if slow_branch.norm_layer != "BN":
                raise ValueError("Pinned native adapter requires the BN MLP variant")
            if (slow_branch.fc1.in_features, slow_branch.fc1.out_features,
                slow_branch.fc2.in_features, slow_branch.fc2.out_features) != (
                    config.channels, config.hidden_channels, config.hidden_channels, config.channels):
                raise ValueError("Native FC dimensions disagree with arithmetic cost contract")
            for sn in (slow_branch.sn1.spiking_neuron, slow_branch.sn2.spiking_neuron):
                if sn.__class__.__name__ != "PSN" or sn.T != config.time_steps:
                    raise ValueError("Only stateless complete-T10 native PSN is supported")
        self.slow = slow_branch
        self.config = config
        self.predictor = LocalGlobalPredictor(config.channels)
        self.fast = FastBranch(config.channels)

    def train(self, mode=True):
        super().train(mode)
        # Recovery can update affine/FC/PSN weights while BN statistics stay fixed.
        if isinstance(self.slow, nn.Module):
            for m in self.slow.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def _validate(self, u):
        if u.ndim != 5 or u.shape[0] != 10 or u.shape[1] != 1 or u.shape[-1] != self.config.channels:
            raise ValueError("Expected complete [T10,B1,H,W,C] input u")
        if min(u.shape[2:4]) < 1 or not u.is_floating_point():
            raise ValueError("Nonempty floating-point input required")
        frozen_bn_guard(self.slow)

    def _tiles(self, h, w):
        k = self.config.tile
        for gy in range(math.ceil(h / k)):
            for gx in range(math.ceil(w / k)):
                yield gy, gx, slice(gy*k, min((gy+1)*k, h)), slice(gx*k, min((gx+1)*k, w))

    def select(self, u, forced_keep=None, stochastic=True):
        h, w = u.shape[2:4]
        ny, nx = math.ceil(h/self.config.tile), math.ceil(w/self.config.tile)
        if forced_keep is not None:
            if forced_keep.dtype != torch.bool or tuple(forced_keep.shape) != (ny, nx):
                raise ValueError("forced_keep must be a boolean [tile_y,tile_x] grid")
            return forced_keep.to(device=u.device, dtype=u.dtype), None
        point = self.predictor(u.mean(0).reshape(1, h*w, -1)).reshape(h, w, 2)
        pooled = torch.stack([point[ys, xs].mean((0, 1)) for _, _, ys, xs in self._tiles(h, w)]).reshape(ny, nx, 2)
        if self.training and stochastic:
            gate = F.gumbel_softmax(pooled, tau=self.config.gumbel_temperature, hard=True, dim=-1)[..., 0]
        else:
            n = ny*nx
            num_keep = math.floor(n*self.config.keep_ratio)
            # Stable sorting defines deterministic raster-order tie handling.
            indices = torch.argsort(pooled[..., 0].reshape(-1), descending=True, stable=True)
            gate = torch.zeros(n, device=u.device, dtype=u.dtype)
            gate[indices[:num_keep]] = 1
            gate = gate.reshape(ny, nx)
        return gate, pooled

    def forward(self, u, *, execution="dense_train", forced_keep=None, stochastic=True):
        """dense_train/reference execute both paths; sparse_eval only selected ones.

        Attention is outside this module. Input u must already equal x+Attention(x).
        A plain callable's parameters are not registered: the caller must own them.
        """
        self._validate(u)
        if execution not in ("dense_train", "dense_reference", "sparse_eval"):
            raise ValueError("Unknown execution mode")
        if execution == "sparse_eval" and (self.training or torch.is_grad_enabled()):
            raise ValueError("sparse_eval requires eval() and no_grad/inference_mode")
        gate, logits = self.select(u, forced_keep, stochastic)
        h, w = u.shape[2:4]
        expanded = gate.repeat_interleave(self.config.tile, 0).repeat_interleave(self.config.tile, 1)[:h, :w]
        selected_slow = int(expanded.detach().sum().item())
        mask = expanded[None, None, :, :, None]
        calls = {"slow": 0, "fast": 0}
        observed_points = {"slow": 0, "fast": 0}

        def run(which, x):
            calls[which] += 1
            observed_points[which] += x.shape[2]*x.shape[3]
            y = self.slow(x) if which == "slow" else self.fast(x)
            if y.shape != x.shape:
                raise ValueError("Branch callable must preserve TBHWC, excluding residual")
            return y

        if execution != "sparse_eval":
            slow, fast = run("slow", u), run("fast", u)
            branch = slow*mask + fast*(1-mask)
        elif selected_slow in (0, h*w):
            branch = run("slow" if selected_slow else "fast", u)
        else:
            branch = torch.empty_like(u)
            for gy, gx, ys, xs in self._tiles(h, w):
                which = "slow" if bool(gate[gy, gx].item()) else "fast"
                branch[:, :, ys, xs, :] = run(which, u[:, :, ys, xs, :].contiguous())
        trace = {
            "shape_TBHWC": list(u.shape), "config": asdict(self.config),
            "execution": execution, "origin": "forced_diagnostic" if forced_keep is not None else "untrained_predictor",
            "selection_rule": "forced" if forced_keep is not None else ("straight_through_Gumbel" if self.training and stochastic else "stable_tile_topk"),
            "tile_keep": gate.detach().to(torch.bool).cpu().tolist(),
            "tile_spans": [{"tile_y": gy, "tile_x": gx, "y0": ys.start, "y1": ys.stop,
                            "x0": xs.start, "x1": xs.stop, "valid_points": (ys.stop-ys.start)*(xs.stop-xs.start)}
                           for gy, gx, ys, xs in self._tiles(h, w)],
            "selected_slow_points": selected_slow, "selected_fast_points": h*w-selected_slow,
            "executed_spatial_points": observed_points, "branch_calls": calls,
            "branch_input_rows_T_times_points": {k: 10*v for k,v in observed_points.items()},
            "attention_changed": False, "complete_T_shared_decision": True,
            "global_score_barrier": forced_keep is None, "source_is_trained_policy": False,
        }
        trace["cost"] = operation_cost(trace, self.config, selector_executed=forced_keep is None)
        return Result(branch, gate, logits, trace)


def ratio_loss(result: Result, target: float, *, weight_by_valid_points=True):
    """Our explicit group-ratio adaptation; not the complete original loss."""
    if weight_by_valid_points:
        weights = result.tile_gate.new_tensor([x["valid_points"] for x in result.trace["tile_spans"]]).reshape_as(result.tile_gate)
        ratio = (weights*result.tile_gate).sum()/weights.sum()
    else:
        ratio = result.tile_gate.mean()
    return (ratio-target).square()


def operation_cost(trace, cfg, selector_executed):
    """Shape-derived scalar services, never cycle counts, FLOP speed or SRAM bytes.

    Slow PSN/FC numbers assume the pinned native MLP contract; for toy callables
    they are only template estimates. Actual branch input rows are in the trace.
    """
    t, _, h, w, c = trace["shape_TBHWC"]
    f, n = cfg.hidden_channels, h*w
    s, fast = (trace["executed_spatial_points"][k] for k in ("slow", "fast"))
    nt = len(trace["tile_spans"])
    selector = {
        "executed": selector_executed,
        "predictor_real_scalar_products": n*(c*c+c*(c//2)+(c//2)*(c//4)+(c//4)*2) if selector_executed else 0,
        "time_mean_additions": (t-1)*n*c if selector_executed else 0,
        "time_mean_rescales": n*c if selector_executed else 0,
        "global_context_additions": (n-1)*(c//2) if selector_executed else 0,
        "global_context_rescales": c//2 if selector_executed else 0,
        "tile_pool_additions": 2*(n-nt) if selector_executed else 0,
        "tile_pool_rescales": 2*nt if selector_executed else 0,
        "layernorm_elements": n*c if selector_executed else 0,
        "gelu_elements": n*(c+c//2+c//4) if selector_executed else 0,
        "logsoftmax_elements": 2*n if selector_executed else 0,
        "selection_items": nt if selector_executed else 0,
        "linear_bias_additions": n*(c+c//2+c//4+2) if selector_executed else 0,
        "Gumbel_samples": 2*nt if trace["selection_rule"] == "straight_through_Gumbel" else 0,
        "tile_softmax_elements": 2*nt if trace["selection_rule"] == "straight_through_Gumbel" else 0,
        "sorting_cycles": None, "selector_memory_transactions": None,
    }
    return {
        "evidence": "shape-derived scalar services; not cycle/traffic/energy measurements",
        "slow_contract_template": {"psn_real_scalar_products": s*t*t*(c+f),
                                   "fc_binary_weighted_accumulates": s*t*2*c*f,
                                   "bn_affine_elements": s*t*(c+f),
                                   "threshold_elements": s*t*(c+f)},
        "fast": {"real_scalar_products": fast*t*c*c, "layernorm_elements": fast*t*c,
                 "linear_bias_additions": fast*t*c},
        "outer_residual_additions_if_called": t*n*c,
        "dense_branch_mixing": {"multiplications": 2*t*n*c, "additions": t*n*c} if trace["execution"] != "sparse_eval" else {"multiplications": 0, "additions": 0},
        "selector": selector,
        "logical_output_elements": t*n*c, "actual_memory_transactions": None,
        "cycles": None, "area": None, "aee": None,
        "training_teacher_and_backward_cost": "not included; no training has been run",
        "not_a_complete_FLOP_total": "LN/GELU/threshold represented as element counts; slow bias and nonlinear implementation costs unresolved",
    }
