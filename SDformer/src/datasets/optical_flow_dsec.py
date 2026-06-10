"""DSEC optical-flow dataset wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torchvision.transforms.functional as TvF
from torch.utils.data import Dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_upstream_path(cfg) -> None:
    upstream_root = str((_repo_root() / cfg["upstream"]["repo_root"]).resolve())
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)


def _center_crop(
    events: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor, size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    th, tw = size
    h, w = events.shape[-2], events.shape[-1]
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return events[..., i : i + th, j : j + tw], flow[..., i : i + th, j : j + tw], mask[..., i : i + th, j : j + tw]


def _random_crop(
    events: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor, size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    th, tw = size
    h, w = events.shape[-2], events.shape[-1]
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return events[..., i : i + th, j : j + tw], flow[..., i : i + th, j : j + tw], mask[..., i : i + th, j : j + tw]


def _random_hflip(
    events: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor, p: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() <= p:
        events = TvF.hflip(events)
        flow = TvF.hflip(flow)
        flow[0] *= -1
        mask = TvF.hflip(mask.float()).bool()
    return events, flow, mask


def _random_vflip(
    events: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor, p: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() <= p:
        events = TvF.vflip(events)
        flow = TvF.vflip(flow)
        flow[1] *= -1
        mask = TvF.vflip(mask.float()).bool()
    return events, flow, mask


class DSECFlowDataset(Dataset):
    """
    Standardized DSEC wrapper with paper-consistent augmentations.

    Output fields:
        `event_voxel`: `[T, H, W]`
        `gt_flow`: `[2, H, W]`
        `valid_mask`: `[1, H, W]`
    """

    def __init__(self, cfg: Dict, split: str) -> None:
        _ensure_upstream_path(cfg)
        from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite

        self.cfg = cfg
        self.split = split
        self._crop_size: Optional[Tuple[int, int]] = None
        crop_cfg = cfg["dataset"].get("crop")
        if crop_cfg is not None:
            self._crop_size = (int(crop_cfg[0]), int(crop_cfg[1]))

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
        if self._crop_size is not None:
            if self.split == "train":
                chunk, label, mask = _random_crop(chunk, label, mask, self._crop_size)
                chunk, label, mask = _random_hflip(chunk, label, mask, p=0.5)
                chunk, label, mask = _random_vflip(chunk, label, mask, p=0.5)
            else:
                chunk, label, mask = _center_crop(chunk, label, mask, self._crop_size)
        return {
            "event_voxel": chunk.float(),
            "gt_flow": label.float(),
            "valid_mask": mask.unsqueeze(0).float(),
            "dataset_name": "dsec",
        }

