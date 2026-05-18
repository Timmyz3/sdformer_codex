"""Experiment-local supervised losses for H9 variants.

This file intentionally lives in the H9 overlay so baseline SDFormerFlow loss
code remains untouched.
"""

from __future__ import annotations

import torch


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
    if not use_angular:
        return loss_function
    print(f"[H9] angular supervised loss enabled: lambda_mod={config['loss'].get('lambda_mod', 1)}, lambda_ang={lambda_ang}")
    return AngularFlowLossSupervised(config, device)
