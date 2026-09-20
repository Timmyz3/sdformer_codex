"""Opt-in, train-only September refresh. Export retains the C12 dense graph."""
from __future__ import annotations

import copy
import json
import math

import torch
from torch import nn
from torch.nn.utils import parametrize


class SharedDelayResidual(nn.Module):
    """Channel-shared linear delay prior, not the full MD-Mixer operator."""

    def __init__(self, weight):
        super().__init__()
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError("delay prior requires square temporal weight")
        n = weight.shape[0]
        basis = torch.stack([torch.diag(weight.new_ones(n - 1), diagonal=d)
                             for d in (-1, 1)])
        self.register_buffer("basis", basis)
        self.coefficients = nn.Parameter(weight.new_zeros(2))

    def forward(self, base):
        # Materialize before addmm, so export does not reassociate large GEMMs.
        return base + (self.coefficients[:, None, None] * self.basis).sum(0)


def install_delay_prior(model, enabled):
    if not enabled:
        return []
    from .atlif_ternary_psn import iter_atlif_ternary_psn
    names = []
    for name, module in list(iter_atlif_ternary_psn(model)):
        if ".attn.sn2_q." in name or ".attn.attn_sn." in name:
            continue
        if module.temporal_factor_rank or parametrize.is_parametrized(module):
            raise ValueError("delay prior cannot be combined with existing factorization")
        before = module.weight.detach().clone()
        parametrize.register_parametrization(module, "weight", SharedDelayResidual(before))
        if not torch.equal(before, module.weight):
            raise RuntimeError("delay initialization changed effective weights")
        names.append(name)
    if len(names) != 81:
        raise RuntimeError(f"expected 81 functional C12 ATLIF sites, got {len(names)}")
    return names


def dense_export(model):
    state = model.state_dict()
    for name, module in model.named_modules():
        if not parametrize.is_parametrized(module, "weight"):
            continue
        if not isinstance(module.parametrizations.weight[0], SharedDelayResidual):
            raise ValueError("unrecognized parametrization in refresh export")
        prefix = name + "." if name else ""
        for key in list(state):
            if key.startswith(prefix + "parametrizations.weight."):
                del state[key]
        state[prefix + "weight"] = module.weight.detach().clone()
    if any("parametrizations." in key for key in state):
        raise RuntimeError("train-only keys leaked into export")
    return state


def corrupt_voxel_view(chunk, generator, drop_rate, sample_probability):
    if chunk.ndim != 5 or chunk.shape[2] != 2:
        raise ValueError("expected post-normalization voxel [B,T,2,H,W]")
    if not 0 <= drop_rate < 1 or not 0 <= sample_probability <= 1:
        raise ValueError("invalid corruption probability")
    selected = torch.rand((chunk.shape[0],), device=chunk.device,
                          generator=generator) < sample_probability
    shape = (chunk.shape[0], chunk.shape[1], 1, *chunk.shape[-2:])
    dropped = torch.rand(shape, device=chunk.device, generator=generator) < drop_rate
    dropped = dropped & selected[:, None, None, None, None]
    return chunk.masked_fill(dropped, 0), selected


def confidence_distillation(student, teacher, gt, valid, selected, scale, max_epe):
    if student.shape != teacher.shape or student.shape != gt.shape:
        raise ValueError("flow tensors must have identical endpoint/shape")
    student, teacher, gt = student.float() * scale, teacher.detach().float() * scale, gt.float()
    valid = valid[:, 0] if valid.ndim == 4 else valid
    teacher_error = torch.linalg.vector_norm(teacher - gt, dim=1)
    student_error = torch.linalg.vector_norm(student.detach() - gt, dim=1)
    eligible = ((valid > 0.5) & selected[:, None, None]
                & torch.isfinite(gt).all(1) & (torch.linalg.vector_norm(gt, dim=1) < 400)
                & (teacher_error <= max_epe) & (teacher_error <= student_error))
    delta = (student - teacher).masked_fill(~eligible[:, None], 0)
    # Zero derivative and zero value for masked/identical predictions.
    distance = (delta.square().sum(1) + 1e-6).sqrt() - 1e-3
    loss = (distance * eligible).sum() / eligible.sum().clamp_min(1)
    return loss, int(eligible.sum()), int((valid > 0.5).sum())


class RefreshTraining:
    """Teacher is held outside the student module and optimizer state."""

    def __init__(self, model, config, device):
        cfg = config.get("algorithm_refresh") or {}
        self.mode = cfg.get("mode")
        if self.mode not in {"control", "augment", "distill", "delay"}:
            raise ValueError(f"unsupported refresh mode: {self.mode}")
        if config.get("teacher_distill", {}).get("enabled") or config.get("pattern_paft", {}).get("enabled"):
            raise ValueError("refresh cannot silently combine legacy teachers/PAFT")
        self.weight = float(cfg.get("distill_weight", 0.1))
        self.drop_rate = float(cfg.get("drop_rate", 0.05))
        self.sample_probability = float(cfg.get("sample_probability", 0.5))
        self.max_epe = float(cfg.get("teacher_max_epe", 1.0))
        self.scale = float(config["metrics"]["flow_scaling"])
        self.warmup = int(cfg.get("warmup_steps", 200))
        if (not all(math.isfinite(v) for v in (self.weight, self.drop_rate,
                self.sample_probability, self.max_epe, self.scale))
                or self.weight < 0 or self.max_epe <= 0 or self.scale <= 0
                or self.warmup < 0 or not 0 <= self.drop_rate < 1
                or not 0 <= self.sample_probability <= 1):
            raise ValueError("invalid refresh hyperparameters")
        self.generator = torch.Generator(device=device).manual_seed(
            int(config["runtime"].get("seed", 0)) + 190906)
        self.teacher = None
        self.teacher_pred = None
        self.selected = None
        self.last = {}
        if self.mode == "distill":
            from spikingjelly.activation_based import functional
            functional.reset_net(model)
            self.teacher = copy.deepcopy(model)
            student_state, teacher_state = model.state_dict(), self.teacher.state_dict()
            if student_state.keys() != teacher_state.keys() or any(
                not torch.equal(value, teacher_state[key]) for key, value in student_state.items()
            ):
                raise RuntimeError("teacher/student initialization mismatch")
            self.teacher.requires_grad_(False)
            self.teacher.eval()
            count = 0
            for module in self.teacher.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.track_running_stats = False
                    module.running_mean = module.running_var = module.num_batches_tracked = None
                    count += 1
            print(f"[REFRESH] teacher exact init PASS; no_running BN={count}; eval_batch=1", flush=True)
        self.delay_names = install_delay_prior(model, self.mode == "delay")
        print(f"[REFRESH] mode={self.mode} delay_sites={len(self.delay_names)}", flush=True)

    def prepare(self, chunk, step):
        self.teacher_pred = None
        self.last = {"step": step, "mode": self.mode}
        if self.mode not in {"augment", "distill"}:
            return chunk
        corrupted, self.selected = corrupt_voxel_view(
            chunk, self.generator, self.drop_rate, self.sample_probability)
        if self.teacher is not None:
            from spikingjelly.activation_based import functional
            outputs = []
            # Keep extra teacher RNG consumption out of the student's sequence.
            devices = [chunk.device.index] if chunk.is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for item in chunk.split(1):
                    functional.reset_net(self.teacher)
                    self.teacher.eval()
                    outputs.append(self.teacher(item)["flow"][-1].detach())
                functional.reset_net(self.teacher)
            self.teacher_pred = torch.cat(outputs)
        self.last["selected_samples"] = int(self.selected.sum())
        self.last["dropped_nonzero_fraction"] = float(
            ((chunk != 0) & (corrupted == 0)).sum() / (chunk != 0).sum().clamp_min(1))
        return corrupted

    def penalty(self, pred, label, mask, step):
        if self.teacher_pred is None:
            return pred[-1].new_zeros(())
        loss, count, total = confidence_distillation(
            pred[-1], self.teacher_pred, label, mask, self.selected,
            self.scale, self.max_epe)
        self.teacher_pred = None
        weight = self.weight * min(1.0, step / max(1, self.warmup))
        self.last.update(kd_raw=float(loss.detach()), kd_weight=weight,
                         kd_pixels=count, valid_pixels=total)
        return loss * weight

    def report(self, model, step):
        if self.delay_names:
            modules = dict(model.named_modules())
            coeffs = torch.cat([modules[n].parametrizations.weight[0].coefficients.detach()
                                for n in self.delay_names])
            self.last["delay_coefficient_absmax"] = float(coeffs.abs().max())
        if step <= 4 or step % 20 == 0:
            print("[REFRESH] " + json.dumps(self.last, sort_keys=True), flush=True)
