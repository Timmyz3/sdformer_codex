"""Loss builders."""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_upstream_path(cfg) -> None:
    upstream_root = str((_repo_root() / cfg["upstream"]["repo_root"]).resolve())
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)


def build_loss(cfg, device):
    _ensure_upstream_path(cfg)
    from loss.flow_supervised import flow_loss_supervised

    upstream_cfg = {
        "metrics": {"flow_scaling": cfg["metrics"]["flow_scaling"]},
        "loss": {
            "lambda_mod": cfg["loss"]["lambda_mod"],
            "lambda_ang": cfg["loss"]["lambda_ang"],
        },
    }
    return flow_loss_supervised(upstream_cfg, device)

