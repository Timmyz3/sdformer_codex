"""Auditable frame-, pixel-, and sequence-level optical-flow metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch


_METRICS = ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")


class FlowMetricAggregationAudit:
    """Accumulate raw metric numerators without changing legacy evaluation."""

    def __init__(self) -> None:
        self._frame_count = 0
        self._valid_pixels = 0.0
        self._numerators = {name: 0.0 for name in _METRICS}
        self._frame_means = {name: 0.0 for name in _METRICS}
        self._sequences = defaultdict(
            lambda: {
                "frame_count": 0,
                "valid_pixels": 0.0,
                "numerators": {name: 0.0 for name in _METRICS},
                "frame_means": {name: 0.0 for name in _METRICS},
            }
        )

    @staticmethod
    def _maps(
        pred: torch.Tensor,
        label: torch.Tensor,
        flow_scaling: float,
    ) -> dict[str, torch.Tensor]:
        flow = pred * flow_scaling
        aee = (flow - label).pow(2).sum(1).sqrt()
        dsec_fl = (
            (aee > 3.0)
            & (aee > 0.05 * label.pow(2).sum(1).sqrt())
        ).to(dtype=flow.dtype) * 100.0

        flow_mag = flow.pow(2).sum(1).sqrt()
        label_mag = label.pow(2).sum(1).sqrt()
        dot_2d = flow[:, 0] * label[:, 0] + flow[:, 1] * label[:, 1]
        cosine_2d = (dot_2d + 1e-7) / (flow_mag * label_mag + 1e-7)
        cosine_2d = cosine_2d.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        aae = torch.acos(cosine_2d) * (180.0 / math.pi)

        dot_3d = dot_2d + 1.0
        norm_3d = torch.sqrt(flow.pow(2).sum(1) + 1.0) * torch.sqrt(
            label.pow(2).sum(1) + 1.0
        )
        cosine_3d = (dot_3d / norm_3d.clamp_min(1e-7)).clamp(
            min=-1.0 + 1e-7, max=1.0 - 1e-7
        )
        aae_benchmark = torch.acos(cosine_3d) * (180.0 / math.pi)
        return {
            "AEE": aee,
            "AAE": aae,
            "AAE_Benchmark": aae_benchmark,
            "DSEC_Fl": dsec_fl,
        }

    def update(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor,
        flow_scaling: float,
        sequence_ids: Iterable[str],
    ) -> None:
        sequence_ids = list(sequence_ids)
        if len(sequence_ids) != pred.shape[0]:
            raise ValueError("sequence_ids must match evaluation batch size")
        maps = self._maps(pred.detach(), label.detach(), flow_scaling)
        flat_mask = mask.detach().reshape(pred.shape[0], -1).to(dtype=pred.dtype)

        for batch, sequence_id in enumerate(sequence_ids):
            valid_pixels = float(flat_mask[batch].sum().cpu())
            sequence = self._sequences[str(sequence_id)]
            self._frame_count += 1
            self._valid_pixels += valid_pixels
            sequence["frame_count"] += 1
            sequence["valid_pixels"] += valid_pixels
            for name, value_map in maps.items():
                numerator = float(
                    (value_map[batch].reshape(-1) * flat_mask[batch]).sum().cpu()
                )
                frame_mean = numerator / (valid_pixels + 1e-9)
                self._numerators[name] += numerator
                self._frame_means[name] += frame_mean
                sequence["numerators"][name] += numerator
                sequence["frame_means"][name] += frame_mean

    @staticmethod
    def _means(record: dict[str, object]) -> dict[str, dict[str, float]]:
        frame_count = int(record["frame_count"])
        valid_pixels = float(record["valid_pixels"])
        numerators = record["numerators"]
        frame_means = record["frame_means"]
        return {
            "frame_equal_mean": {
                name: float(frame_means[name]) / max(frame_count, 1)
                for name in _METRICS
            },
            "pixel_global_mean": {
                name: float(numerators[name]) / max(valid_pixels, 1e-9)
                for name in _METRICS
            },
        }

    def summary(self) -> dict[str, object]:
        aggregate = {
            "frame_count": self._frame_count,
            "valid_pixels": self._valid_pixels,
            "numerators": dict(self._numerators),
            "frame_means": dict(self._frame_means),
        }
        per_sequence = {}
        for sequence_id in sorted(self._sequences):
            raw = self._sequences[sequence_id]
            per_sequence[sequence_id] = {
                "frame_count": int(raw["frame_count"]),
                "valid_pixels": float(raw["valid_pixels"]),
                **self._means(raw),
            }
        sequence_count = len(per_sequence)
        sequence_balanced = {
            name: sum(
                sequence["pixel_global_mean"][name]
                for sequence in per_sequence.values()
            )
            / max(sequence_count, 1)
            for name in _METRICS
        }
        return {
            "schema": "flow_metric_aggregation_audit_v1",
            "definitions": {
                "frame_equal_mean": "masked mean per frame, then equal mean over frames",
                "pixel_global_mean": "sum masked error over all frames divided by all valid pixels",
                "sequence_balanced_mean": "equal mean of each sequence pixel-global metric",
            },
            "frame_count": self._frame_count,
            "valid_pixels": self._valid_pixels,
            "sequence_count": sequence_count,
            **self._means(aggregate),
            "sequence_balanced_mean": sequence_balanced,
            "per_sequence": per_sequence,
        }
