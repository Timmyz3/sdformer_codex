"""MVSEC optical-flow dataset wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_upstream_path(cfg) -> None:
    upstream_root = str((_repo_root() / cfg["upstream"]["repo_root"]).resolve())
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)


class MVSECFlowDataset(Dataset):
    """
    Standardized MVSEC wrapper.

    Output fields:
        `event_voxel`: `[T, H, W]`
        `gt_flow`: `[2, H, W]`
        `valid_mask`: `[1, H, W]`
    """

    def __init__(self, cfg: Dict, split: str) -> None:
        _ensure_upstream_path(cfg)
        from MDR_dataloader.MVSEC import MvsecEventFlow

        self.cfg = cfg
        upstream_cfg = {
            "data": {
                "num_frames": cfg["model"]["num_bins"],
                "num_chunks": cfg["dataset"].get("num_chunks", 1),
                "test_sequence": cfg["dataset"].get("test_sequence", "indoor_flying3"),
            },
            "loader": {
                "resolution": cfg["dataset"].get("resolution", [260, 346]),
                "crop": cfg["dataset"].get("crop", [256, 256]),
                "polarity": cfg["dataset"].get("polarity", True),
            },
        }
        self.dataset = MvsecEventFlow(upstream_cfg, train=(split == "train"), aug=(split == "train"))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[index]
        event_voxel = sample["event_volume_new"].float()
        if self.cfg["dataset"].get("num_chunks", 1) == 2:
            event_voxel = torch.cat((sample["event_volume_old"], event_voxel), dim=0)
        return {
            "event_voxel": event_voxel,
            "gt_flow": sample["flow"].float(),
            "valid_mask": sample["valid"].unsqueeze(0).float(),
            "dataset_name": "mvsec",
        }

