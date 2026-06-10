"""Experiment-local supervised losses for H9 variants.

This file intentionally lives in the H9 overlay so baseline SDFormerFlow loss
code remains untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class AngularFlowLossSupervised(torch.nn.Module):
    """Supervised flow loss with the angular term that baseline leaves disabled."""

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.flow_scaling = config["metrics"]["flow_scaling"]
        self.lambda_mod = float(config["loss"].get("lambda_mod", 1.0))
        self.lambda_ang = float(config["loss"].get("lambda_ang", 0.0))

    @staticmethod
    def _valid_mask(mask: torch.Tensor, flow: torch.Tensor, max_flow_mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        mask = mask.to(dtype=flow.dtype)
        if max_flow_mask is not None:
            mask = mask * max_flow_mask.to(dtype=flow.dtype)
        return mask

    @staticmethod
    def _mod_loss(flow: torch.Tensor, gt_flow: torch.Tensor, mask: torch.Tensor, num_valid_px: torch.Tensor) -> torch.Tensor:
        error = torch.sqrt((flow - gt_flow).pow(2).sum(1) + 1.0e-8)
        error = error.reshape(flow.shape[0], -1)
        mask = mask.reshape(flow.shape[0], -1)
        return torch.sum(error * mask, dim=1) / (num_valid_px + 1.0e-9)

    @staticmethod
    def _angular_loss(
        flow: torch.Tensor,
        gt_flow: torch.Tensor,
        mask: torch.Tensor,
        num_valid_px: torch.Tensor,
        epsilon: float = 1.0e-8,
    ) -> torch.Tensor:
        flow_mag = torch.sqrt(flow.pow(2).sum(1) + epsilon)
        gt_mag = torch.sqrt(gt_flow.pow(2).sum(1) + epsilon)
        dot_product = flow[:, 0] * gt_flow[:, 0] + flow[:, 1] * gt_flow[:, 1]
        cosine = (dot_product + epsilon) / (flow_mag * gt_mag + epsilon)
        cosine = torch.clamp(cosine, min=-1.0 + epsilon, max=1.0 - epsilon)
        per_pixel = torch.acos(cosine) * mask
        return torch.sum(per_pixel) / (torch.sum(num_valid_px) + 1.0e-9)

    @staticmethod
    def _sequence_loss(flow_preds, flow_gt: torch.Tensor, valid: torch.Tensor, gamma: float, max_flow: float = 400.0):
        n_predictions = len(flow_preds)
        flow_loss = 0.0
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        return flow_loss

    def forward(self, pred_list, gt_flow: torch.Tensor, mask: torch.Tensor, gamma=None) -> torch.Tensor:
        if gamma is not None:
            return self._sequence_loss(pred_list, gt_flow, mask, gamma)

        mag = torch.sum(gt_flow**2, dim=1).sqrt()
        max_flow_mask = mag < 400.0
        curr_loss = 0.0
        for pred in pred_list:
            flow = pred * self.flow_scaling
            valid = self._valid_mask(mask, flow, max_flow_mask)
            num_valid_px = torch.sum(valid.reshape(flow.shape[0], -1), dim=1)
            mod_loss = self._mod_loss(flow, gt_flow, valid, num_valid_px)
            ang_loss = self._angular_loss(flow, gt_flow, valid, num_valid_px)
            curr_loss = curr_loss + self.lambda_mod * mod_loss + self.lambda_ang * ang_loss
        curr_loss = curr_loss / len(pred_list)
        return torch.mean(curr_loss)


def maybe_replace_flow_loss(loss_function: torch.nn.Module, config: dict, device: torch.device) -> torch.nn.Module:
    lambda_ang = float(config.get("loss", {}).get("lambda_ang", 0.0) or 0.0)
    use_angular = bool(config.get("loss", {}).get("use_angular_loss", False)) or lambda_ang != 0.0
    distill_cfg = config.get("teacher_distill") or {}
    use_distill = bool(distill_cfg.get("enabled", False))
    if use_angular:
        print(f"[H9] angular supervised loss enabled: lambda_mod={config['loss'].get('lambda_mod', 1)}, lambda_ang={lambda_ang}")
        loss_function = AngularFlowLossSupervised(config, device)
    if use_distill:
        loss_function = TeacherFlowDistillLoss(loss_function, config)
        print(f"[H55] teacher distill loss enabled: {loss_function.describe()}")
    return loss_function


def _as_flow_list(preds) -> list[torch.Tensor]:
    if isinstance(preds, dict):
        preds = preds["flow"]
    if isinstance(preds, torch.Tensor):
        return [preds]
    return list(preds)


class TeacherFlowDistillLoss(torch.nn.Module):
    """Add teacher EPE/direction regularization without touching baseline loss."""

    def __init__(self, base_loss: torch.nn.Module, config: dict):
        super().__init__()
        cfg = config.get("teacher_distill") or {}
        self.base_loss = base_loss
        self.lambda_epe = float(cfg.get("lambda_epe", 0.0) or 0.0)
        self.lambda_dir = float(cfg.get("lambda_dir", 0.0) or 0.0)
        self.min_gt_mag = float(cfg.get("min_gt_mag", 0.0) or 0.0)
        self.full_weight_gt_mag = float(cfg.get("full_weight_gt_mag", max(self.min_gt_mag, 1.0)) or 1.0)
        self.teacher_confidence_max_epe = float(cfg.get("teacher_confidence_max_epe", 0.0) or 0.0)
        self.use_all_predictions = bool(cfg.get("use_all_predictions", False))
        self.epsilon = float(cfg.get("epsilon", 1.0e-6) or 1.0e-6)
        self._teacher_preds: list[torch.Tensor] | None = None
        self._last_stats: dict[str, float] = {}

    def describe(self) -> dict[str, float | bool]:
        return {
            "lambda_epe": self.lambda_epe,
            "lambda_dir": self.lambda_dir,
            "min_gt_mag": self.min_gt_mag,
            "full_weight_gt_mag": self.full_weight_gt_mag,
            "teacher_confidence_max_epe": self.teacher_confidence_max_epe,
            "use_all_predictions": self.use_all_predictions,
        }

    def set_teacher_prediction(self, teacher_preds) -> None:
        self._teacher_preds = [item.detach() for item in _as_flow_list(teacher_preds)]

    def _mask_and_weights(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        gt_flow: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.dim() == 4 and mask.shape[1] == 1:
            valid = mask[:, 0]
        else:
            valid = mask
        valid = valid.to(dtype=student.dtype)

        gt_mag = torch.sqrt(gt_flow.pow(2).sum(1) + self.epsilon)
        if self.full_weight_gt_mag <= self.min_gt_mag:
            mag_weight = (gt_mag >= self.min_gt_mag).to(dtype=student.dtype)
        else:
            mag_weight = ((gt_mag - self.min_gt_mag) / (self.full_weight_gt_mag - self.min_gt_mag)).clamp(0.0, 1.0)
        weight = valid * mag_weight

        if self.teacher_confidence_max_epe > 0.0:
            teacher_epe = torch.sqrt((teacher - gt_flow).pow(2).sum(1) + self.epsilon)
            confidence = (teacher_epe <= self.teacher_confidence_max_epe).to(dtype=student.dtype)
            weight = weight * confidence
        return valid, weight

    def _epe_distill(self, student: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        epe = torch.sqrt((student - teacher).pow(2).sum(1) + self.epsilon)
        return torch.sum(epe * valid) / (torch.sum(valid) + self.epsilon)

    def _direction_distill(self, student: torch.Tensor, teacher: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        student_unit = F.normalize(student, p=2, dim=1, eps=self.epsilon)
        teacher_unit = F.normalize(teacher, p=2, dim=1, eps=self.epsilon)
        cosine = torch.sum(student_unit * teacher_unit, dim=1).clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        direction = 1.0 - cosine
        return torch.sum(direction * weight) / (torch.sum(weight) + self.epsilon)

    def forward(self, pred_list, gt_flow: torch.Tensor, mask: torch.Tensor, gamma=None) -> torch.Tensor:
        base = self.base_loss(pred_list, gt_flow, mask, gamma=gamma)
        teacher_preds = self._teacher_preds
        self._teacher_preds = None
        if teacher_preds is None or (self.lambda_epe == 0.0 and self.lambda_dir == 0.0):
            return base

        student_preds = _as_flow_list(pred_list)
        if not self.use_all_predictions:
            student_preds = [student_preds[-1]]
            teacher_preds = [teacher_preds[-1]]
        count = min(len(student_preds), len(teacher_preds))
        if count == 0:
            return base

        distill = torch.zeros((), dtype=base.dtype, device=base.device)
        epe_value = torch.zeros((), dtype=base.dtype, device=base.device)
        dir_value = torch.zeros((), dtype=base.dtype, device=base.device)
        for student, teacher in zip(student_preds[-count:], teacher_preds[-count:]):
            valid, direction_weight = self._mask_and_weights(student, teacher, gt_flow, mask)
            if self.lambda_epe:
                item = self._epe_distill(student, teacher, valid)
                epe_value = epe_value + item
                distill = distill + self.lambda_epe * item
            if self.lambda_dir:
                item = self._direction_distill(student, teacher, direction_weight)
                dir_value = dir_value + item
                distill = distill + self.lambda_dir * item
        distill = distill / float(count)
        self._last_stats = {
            "base": float(base.detach().item()),
            "epe": float((epe_value / float(count)).detach().item()),
            "dir": float((dir_value / float(count)).detach().item()),
            "total": float((base + distill).detach().item()),
        }
        return base + distill
