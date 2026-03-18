"""DSEC optical-flow dataset wrapper."""

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


class DSECFlowDataset(Dataset):
    """
    Standardized DSEC wrapper.

    Output fields:
        `event_voxel`: `[T, H, W]`
        `gt_flow`: `[2, H, W]`
        `valid_mask`: `[1, H, W]`
    """

    def __init__(self, cfg: Dict, split: str) -> None:
        _ensure_upstream_path(cfg)
        from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite

        self.cfg = cfg
        split_name = cfg["dataset"]["train_split"] if split == "train" else cfg["dataset"]["eval_split"]
        upstream_cfg = {
            "data": {
                "path": cfg["dataset"]["root"],
                "preprocessed": cfg["dataset"]["preprocessed"],
                "num_frames": cfg["model"]["num_bins"],
                "num_chunks": cfg["dataset"].get("num_chunks", 1),
            },
            "model": {"encoding": cfg["model"]["encoding"]},
            "loader": {
                "resolution": cfg["dataset"]["resolution"],
                "crop": cfg["dataset"].get("crop"),
                "polarity": cfg["dataset"].get("polarity", True),
            },
        }
        self.dataset = DSECDatasetLite(upstream_cfg, file_list=split_name, stereo=False)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        chunk, mask, label = self.dataset[index]
        return {
            "event_voxel": chunk.float(),
            "gt_flow": label.float(),
            "valid_mask": mask.unsqueeze(0).float(),
            "dataset_name": "dsec",
        }

