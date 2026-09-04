"""Metric helpers for optical flow evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_upstream_path(cfg) -> None:
    upstream_root = str((_repo_root() / cfg["upstream"]["repo_root"]).resolve())
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)


def compute_metrics(cfg, pred, label, mask) -> Dict[str, float]:
    _ensure_upstream_path(cfg)
    from loss.flow_supervised import AAE, AEE

    results = {}
    flow_scaling = cfg["metrics"]["flow_scaling"]
    if "AEE" in cfg["metrics"]["names"]:
        aee, pe1, pe2, pe3, outliers = AEE(pred, label, mask, flow_scaling)()
        results.update(
            {
                "AEE": float(aee.mean().item()),
                "AEE_PE1": float(pe1.mean().item()),
                "AEE_PE2": float(pe2.mean().item()),
                "AEE_PE3": float(pe3.mean().item()),
                "AEE_outliers": float(outliers.mean().item()),
            }
        )
    if "AAE" in cfg["metrics"]["names"]:
        aae = AAE(pred, label, mask, flow_scaling)()[0]
        results["AAE"] = float(aae.mean().item())
    return results

