#!/usr/bin/env python3
"""Exact Local/Motion source-work counts for temporal Linear/Conv2d calls."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _require_zero_padding(module: torch.nn.Conv2d) -> None:
    padding_mode = str(getattr(module, "padding_mode", "zeros"))
    if padding_mode != "zeros":
        raise ValueError(
            "dual-line exact Conv2d requires padding_mode='zeros', got {!r}".format(
                padding_mode
            )
        )


def _linear_counts(mask: torch.Tensor, module: torch.nn.Linear) -> tuple[torch.Tensor, int]:
    if mask.shape[-1] != module.in_features:
        raise ValueError("Linear input feature dimension mismatch")
    return mask.reshape(-1, module.in_features).sum(dim=1, dtype=torch.int64), int(
        module.out_features
    )


def _conv2d_counts(mask: torch.Tensor, module: torch.nn.Conv2d) -> tuple[torch.Tensor, int]:
    _require_zero_padding(module)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4 or mask.shape[1] != module.in_channels:
        raise ValueError("Conv2d temporal slice must be [B,C,H,W] or [C,H,W]")
    kernel = torch.ones(
        (module.groups, module.in_channels // module.groups, *module.kernel_size),
        dtype=torch.float32,
        device=mask.device,
    )
    counts = F.conv2d(
        mask.to(torch.float32),
        kernel,
        bias=None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    ).round().to(torch.int64)
    return counts.reshape(-1), int(module.out_channels // module.groups)


def _source_counts(
    mask: torch.Tensor, module: torch.nn.Module
) -> tuple[torch.Tensor, int]:
    if isinstance(module, torch.nn.Linear):
        return _linear_counts(mask, module)
    if isinstance(module, torch.nn.Conv2d):
        return _conv2d_counts(mask, module)
    raise TypeError("dual-line trace supports only Linear and Conv2d")


def profile_operator_temporal_work(
    module: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    temporal_steps: int,
) -> list[dict[str, Any]]:
    """Return exact source-column work at one selector decision per output row.

    For Linear, one row is every vector along the last input dimension.  For
    Conv2d, one row is one receptive field and input group at one output
    position.  The output-channel fanout is included in all work totals.
    """

    if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        return [{"status": "UNSUPPORTED_OPERATOR", "temporal_step": -1}]
    if isinstance(module, torch.nn.Conv2d):
        _require_zero_padding(module)
    value = input_tensor.detach()
    if value.ndim < 2 or int(value.shape[0]) != int(temporal_steps):
        return [{
            "status": "TEMPORAL_AXIS_UNQUALIFIED",
            "temporal_step": -1,
            "input_shape": list(value.shape),
        }]
    binary = torch.logical_or(value.eq(0), value.eq(1))
    if not bool(binary.all().item()):
        return [{
            "status": "NON_BINARY_BYPASS",
            "temporal_step": -1,
            "input_shape": list(value.shape),
        }]

    rows: list[dict[str, Any]] = []
    previous = torch.zeros_like(value[0], dtype=torch.bool)
    for timestep in range(temporal_steps):
        current = value[timestep].eq(1)
        positive = current & ~previous
        negative = previous & ~current
        current_counts, fanout = _source_counts(current, module)
        positive_counts, positive_fanout = _source_counts(positive, module)
        negative_counts, negative_fanout = _source_counts(negative, module)
        if fanout != positive_fanout or fanout != negative_fanout:
            raise RuntimeError("inconsistent source fanout")
        transition_counts = positive_counts + negative_counts
        state_valid = timestep > 0
        choose_motion = transition_counts < current_counts
        if not state_valid:
            choose_motion = torch.zeros_like(choose_motion)
        selected_counts = torch.where(choose_motion, transition_counts, current_counts)
        valid_counts, valid_fanout = _source_counts(torch.ones_like(current), module)
        if fanout != valid_fanout:
            raise RuntimeError("inconsistent valid source fanout")
        rows.append({
            "status": "PASS_EXACT_SOURCE_WORK",
            "temporal_step": timestep,
            "state_valid": state_valid,
            "selector_rows": int(current_counts.numel()),
            "motion_selected_rows": int(choose_motion.sum().item()),
            "local_selected_rows": int((~choose_motion).sum().item()),
            "valid_source_work": int(valid_counts.sum().item()) * fanout,
            "current_source_count": int(current_counts.sum().item()),
            "positive_transition_source_count": int(positive_counts.sum().item()),
            "negative_transition_source_count": int(negative_counts.sum().item()),
            "local_work": int(current_counts.sum().item()) * fanout,
            "motion_work": int(transition_counts.sum().item()) * fanout,
            "selected_work": int(selected_counts.sum().item()) * fanout,
            "selector_saved_work": int(
                (current_counts - selected_counts).sum().item()
            ) * fanout,
            "output_channel_fanout": fanout,
        })
        previous = current
    return rows
