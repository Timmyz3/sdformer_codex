from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "third_party" / "SDformerFlow"
sys.path.insert(0, str(UPSTREAM_ROOT))

from DSEC_dataloader.event_representations import EventSlicer, VoxelGrid, rectify_events


def decode_flow_png(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode DSEC 16-bit optical-flow PNG into float flow and valid mask."""
    flow_16 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if flow_16 is None:
        raise FileNotFoundError(f"Failed to read flow file: {path}")
    if flow_16.ndim != 3 or flow_16.shape[2] != 3:
        raise ValueError(f"Unexpected DSEC flow shape {flow_16.shape} for {path}")

    flow_x = (flow_16[:, :, 2].astype(np.float32) - 2**15) / 128.0
    flow_y = (flow_16[:, :, 1].astype(np.float32) - 2**15) / 128.0
    valid = flow_16[:, :, 0].astype(bool)
    flow = np.stack([flow_x, flow_y], axis=0).astype(np.float32)
    return flow, valid


def make_voxel_chunk(
    events_path: Path,
    t_beg: int,
    t_end: int,
    num_bins: int,
) -> np.ndarray:
    with h5py.File(events_path / "events.h5", "r") as h5f:
        slicer = EventSlicer(h5f)
        # Upstream stores GPS-scale offsets around 5.8e10 us.
        # On Windows, adding them back as plain Python ints overflows in numpy.
        slicer.t_offset = np.int64(slicer.t_offset)
        event_data = slicer.get_events(t_beg, t_end)
        if event_data is None or event_data["t"].size == 0:
            return np.zeros((num_bins, 480, 640), dtype=np.float32)

        with h5py.File(events_path / "rectify_map.h5", "r") as rectify_file:
            rectify_map = rectify_file["rectify_map"][()]

        x_rect, y_rect = rectify_events(event_data["x"], event_data["y"], rectify_map).T
        mask = (x_rect >= 0) & (x_rect < 640) & (y_rect >= 0) & (y_rect < 480)

        t = event_data["t"][mask].astype(np.float32)
        if t.size == 0:
            return np.zeros((num_bins, 480, 640), dtype=np.float32)
        if t[-1] == t[0]:
            t = np.linspace(0.0, 1.0, t.size, dtype=np.float32)
        else:
            t = (t - t[0]) / (t[-1] - t[0])

        event_data_torch = {
            "p": torch.from_numpy(event_data["p"][mask].astype(np.float32)),
            "t": torch.from_numpy(t),
            "x": torch.from_numpy(x_rect[mask]),
            "y": torch.from_numpy(y_rect[mask]),
        }
        voxel_grid = VoxelGrid((num_bins, 480, 640))
        return voxel_grid.convert_CHW(event_data_torch).cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/Datasets/DSEC")
    parser.add_argument("--sequence", default="zurich_city_09_a")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--valid-stride", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    sequence = args.sequence

    flow_dir = root / "train_optical_flow" / sequence / "flow" / "forward"
    timestamps_path = root / "train_optical_flow" / sequence / "flow" / "forward_timestamps.txt"
    events_dir = root / "train_events" / sequence / "events" / "left"

    if not flow_dir.exists():
        raise FileNotFoundError(f"Missing flow directory: {flow_dir}")
    if not timestamps_path.exists():
        raise FileNotFoundError(f"Missing timestamps file: {timestamps_path}")
    if not events_dir.exists():
        raise FileNotFoundError(f"Missing events directory: {events_dir}")

    saved_root = root / "saved_flow_data"
    gt_dir = saved_root / "gt_tensors"
    mask_dir = saved_root / "mask_tensors"
    voxel_seq_dir = saved_root / "event_tensors" / f"{args.num_bins:02d}bins" / "left" / sequence
    split_dir = saved_root / "sequence_lists"
    for path in (gt_dir, mask_dir, voxel_seq_dir, split_dir):
        path.mkdir(parents=True, exist_ok=True)

    timestamps = np.loadtxt(timestamps_path, delimiter=",", dtype=np.int64)
    flow_files = sorted(flow_dir.glob("*.png"))
    if len(flow_files) != len(timestamps):
        raise ValueError(
            f"Flow file count ({len(flow_files)}) != timestamp count ({len(timestamps)}) for {sequence}"
        )

    generated_names: list[str] = []
    for idx, (flow_png, (t_beg, t_end)) in enumerate(zip(flow_files, timestamps), start=1):
        filename = f"{sequence}_{idx:04d}.npy"
        flow, valid = decode_flow_png(flow_png)
        voxel = make_voxel_chunk(events_dir, int(t_beg), int(t_end), args.num_bins)

        np.save(gt_dir / filename, flow)
        np.save(mask_dir / filename, valid)
        np.save(voxel_seq_dir / filename, voxel)
        generated_names.append(filename)

    train_names = [name for i, name in enumerate(generated_names) if i % args.valid_stride != 0]
    valid_names = [name for i, name in enumerate(generated_names) if i % args.valid_stride == 0]
    if not valid_names and generated_names:
        valid_names = [generated_names[-1]]
        train_names = generated_names[:-1]

    pd.DataFrame(train_names).to_csv(split_dir / "train_split_seq.csv", header=False, index=False)
    pd.DataFrame(valid_names).to_csv(split_dir / "valid_split_seq.csv", header=False, index=False)

    print(f"Generated {len(generated_names)} samples for {sequence}")
    print(f"Train samples: {len(train_names)}")
    print(f"Valid samples: {len(valid_names)}")
    print(f"saved_flow_data root: {saved_root}")


if __name__ == "__main__":
    main()
