"""Deterministic MVSEC split and augmentation helpers for direct training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


MVSEC_VALID_TIME_INDEX = {
    "indoor_flying1": [(314, 2197)],
    "indoor_flying2": [(314, 2199)],
    "indoor_flying3": [(314, 2199)],
    "indoor_flying4": [(196, 570)],
    "outdoor_day1": [(245, 3000)],
    "outdoor_day2": [(4375, 7002)],
}


def event_activity_mask(event_volume: torch.Tensor) -> torch.Tensor:
    """Collapse time/polarity axes to the [B, 1, H, W] flow-mask contract."""
    if event_volume.ndim not in (4, 5):
        raise ValueError(
            "event volume must be [B,T,H,W] or [B,T,P,H,W], "
            f"got shape {tuple(event_volume.shape)}"
        )
    reduce_dims = tuple(range(1, event_volume.ndim - 2))
    return torch.any(event_volume != 0, dim=reduce_dims).unsqueeze(1)


def apply_mvsec_source_valid_region(valid, sequence: str, bottom_row: int = 193):
    """Apply the MVSEC outdoor valid-FOV contract in source-frame coordinates."""
    if isinstance(valid, torch.Tensor):
        result = valid.clone()
    else:
        result = np.array(valid, copy=True)
    if sequence in {"outdoor_day1", "outdoor_day2"}:
        result[bottom_row:, :] = False
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mvsec_split_manifest(
    manifest_path: str | Path,
    role: str,
    expected_sequence: str,
) -> tuple[list[int], str]:
    path = Path(manifest_path).expanduser()
    if not path.is_absolute():
        candidates = (Path.cwd() / path, Path(__file__).resolve().parents[3] / path)
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "mvsec_cicc_split_manifest_v1":
        raise RuntimeError(f"Unsupported MVSEC split manifest schema: {path}")
    split = (manifest.get("splits") or {}).get(role)
    if not isinstance(split, dict):
        raise RuntimeError(f"MVSEC split manifest has no role {role!r}: {path}")
    sequence = str(split.get("sequence", ""))
    if sequence != expected_sequence:
        raise RuntimeError(
            f"MVSEC split {role!r} targets {sequence!r}, expected {expected_sequence!r}"
        )
    indices = [int(index) for index in split.get("indices") or []]
    if not indices or len(indices) != len(set(indices)):
        raise RuntimeError(f"MVSEC split {role!r} is empty or contains duplicates")
    if indices != sorted(indices):
        raise RuntimeError(f"MVSEC split {role!r} indices must be sorted")
    legal = {
        index
        for start, end in MVSEC_VALID_TIME_INDEX[sequence]
        for index in range(start, end)
    }
    illegal = [index for index in indices if index not in legal]
    if illegal:
        raise RuntimeError(f"MVSEC split {role!r} has out-of-range indices: {illegal[:8]}")
    return indices, file_sha256(path)


class MVSECDirectAugmentor:
    """Spike-FlowNet-style random crop plus horizontal/vertical flips."""

    def __init__(
        self,
        crop_size: list[int] | tuple[int, int],
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
    ) -> None:
        self.crop_height = int(crop_size[0])
        self.crop_width = int(crop_size[1])
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.vertical_flip_probability = float(vertical_flip_probability)

    def __call__(self, event1, event2, d_event1, d_event2, flow, valid):
        arrays = [event1, event2, d_event1, d_event2]
        if np.random.random() < self.horizontal_flip_probability:
            arrays = [array[:, ::-1] for array in arrays]
            flow = flow[:, ::-1] * np.asarray([-1.0, 1.0])
            valid = valid[:, ::-1]
        if np.random.random() < self.vertical_flip_probability:
            arrays = [array[::-1, :] for array in arrays]
            flow = flow[::-1, :] * np.asarray([1.0, -1.0])
            valid = valid[::-1, :]

        height, width = flow.shape[:2]
        if height < self.crop_height or width < self.crop_width:
            raise RuntimeError(
                f"MVSEC crop {self.crop_height}x{self.crop_width} exceeds input "
                f"{height}x{width}"
            )
        y0 = int(np.random.randint(0, height - self.crop_height + 1))
        x0 = int(np.random.randint(0, width - self.crop_width + 1))
        ys = slice(y0, y0 + self.crop_height)
        xs = slice(x0, x0 + self.crop_width)
        arrays = [np.ascontiguousarray(array[ys, xs]) for array in arrays]
        flow = np.ascontiguousarray(flow[ys, xs])
        valid = np.ascontiguousarray(valid[ys, xs])
        return (*arrays, flow, valid)
